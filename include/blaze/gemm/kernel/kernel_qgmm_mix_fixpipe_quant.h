/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qgmm_mix_fixpipe_quant.h
 * \brief Grouped int8 Cube-input per-channel/per-group kernel using FixPipe quant and an AIV epilogue.
 *
 * The operator repository supplies int8 Cube inputs and optional asymmetric-quantization workspaces.
 * AIC writes the FixPipe result to a per-core workspace slot. AIV applies per-token scaling and,
 * when withOffset is enabled, the row-sum/offset correction before storing the final output.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/epilogue/block/block_epilogue_per_token_scale.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define GMM_FIXPIPE_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define GMM_FIXPIPE_TEM_PARAMS                                                                    \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                       \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelGroupedMmadWithScaleFixpipeQuant, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>
#define GMM_FIXPIPE_TEMPLATE_DEF \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>

GMM_FIXPIPE_CLASS_TEM_PARAMS
class GemmUniversal<GMM_FIXPIPE_TEM_PARAMS> {
private:
    static constexpr uint8_t GMM_GROUP_LIST_OFFSET = 0U;
    static constexpr uint8_t GMM_GROUP_LIST_LENGTH = 1U;
    static constexpr uint8_t GMM_GROUP_LIST_SPARSE = 2U;
    static constexpr uint32_t GMM_SPARSE_STRIDE = 2U;
    static constexpr uint32_t GMM_SPARSE_VALUE_OFFSET = 1U;
    static constexpr int64_t GMM_BLOCK_MASK = BLOCK_CUBE - 1;

public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using ScaleType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockInfo = typename BlockScheduler::BlockInfo;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    static_assert(AscendC::IsSameType<AType, int8_t>::value && AscendC::IsSameType<BType, int8_t>::value,
                  "GMM Fixpipe Quant only supports int8_t AType and BType.");
    static_assert(AscendC::IsSameType<ScaleType, uint64_t>::value,
                  "GMM Fixpipe Quant only supports uint64_t scale type.");
    static_assert(AscendC::IsSameType<CType, half>::value && AscendC::IsSameType<BiasType, int32_t>::value,
                  "GMM Fixpipe Quant only supports half CType and int32_t BiasType.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "GMM Fixpipe Quant only supports ND LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GMM Fixpipe Quant only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutBias, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GMM Fixpipe Quant does not support NZ/ZN LayoutC or LayoutBias.");
    static_assert(AscendC::IsSameType<typename BlockEpilogue::FixpipeType, CType>::value,
                  "GMM Fixpipe Quant requires BlockEpilogue FixpipeType to match CType.");

    struct GMMTiling {
        uint32_t groupNum{0};
        int64_t m{0};
        int64_t n{0};
        int64_t k{0};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        // K elements sharing one [N] scale vector; 0 falls back to baseK.
        uint32_t quantGroupSize{0};
        uint32_t quantMode{static_cast<uint32_t>(QuantMode::PERCHANNEL_MODE)};
        uint32_t kAL1{0};
        uint32_t kBL1{0};
        uint32_t nBufferNum{0};
        uint8_t dbL0C{0};
        uint8_t groupListType{GMM_GROUP_LIST_OFFSET};
    };

    struct Params {
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr{nullptr};
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    // This kernel uses direct Cube/Vector pairing (sync mode 2). In this mode one
    // flag operation synchronizes the paired Cube and Vector task; unlike mode
    // 4 it must not explicitly signal the second Vector sub-block with +16.
    static constexpr uint16_t GMM_SYNC_MODE = 2;
    // Keep the flag allocation consistent with the A5 grouped_matmul mode-2
    // implementation in grouped_matmul_utils.h.
    static constexpr uint16_t AIC_TO_AIV_FLAG = 5;
    static constexpr uint16_t AIV_TO_AIC_FLAG = 3;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline int64_t GetPerGroupBOffset(int64_t n, int64_t k) const;
    template <class GroupListTensor>
    __aicore__ inline uint32_t GetGroupIdx(const GroupListTensor& groupList, uint32_t loopIdx);
    template <class GroupListTensor>
    __aicore__ inline int64_t GetGroupM(const GroupListTensor& groupList, uint32_t loopIdx);
    template <class GroupListTensor>
    __aicore__ inline void SetGroupShape(const GroupListTensor& groupList, uint32_t loopIdx);
    __aicore__ inline bool IsValidProblem() const;
    __aicore__ inline void PrepareScheduler(BlockScheduler& scheduler);
    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler);
    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler& scheduler) const;
    __aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx);
    template <class TensorA, class TensorB, class TensorScale, class TensorBias, class TensorC>
    __aicore__ inline void ProcessPerGroup(const TensorA& gmA, const TensorB& gmB, const TensorScale& gmScale,
                                           const TensorBias& gmBias, const TensorC& gmC, const BlockShape& blockShape);

    template <class TensorA, class TensorB>
    __aicore__ inline void ProcessOneBlock(const TensorA& gmA, const TensorB& gmB, const BlockShape& blockShape,
                                           uint32_t groupIdx, int64_t groupMOffset, int64_t mPos, int64_t nPos,
                                           int64_t curM, int64_t curN, int64_t k, int64_t n);
    __aicore__ inline void NotifyVector() { AscendC::CrossCoreSetFlag<GMM_SYNC_MODE, PIPE_FIX>(AIC_TO_AIV_FLAG); }
    __aicore__ inline void WaitForVector() { AscendC::CrossCoreWaitFlag(AIV_TO_AIC_FLAG); }
    __aicore__ inline void NotifyCube() { AscendC::CrossCoreSetFlag<GMM_SYNC_MODE, PIPE_MTE2>(AIV_TO_AIC_FLAG); }
    __aicore__ inline void WaitForCube() { AscendC::CrossCoreWaitFlag(AIC_TO_AIV_FLAG); }

    BlockMmad mmOp_;
    BlockEpilogue epilogueOp_;
    ProblemShape problemShape_{};
    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ CType* workspaceBasePtr_{nullptr};
    __gm__ ScaleType* scaleBasePtr_{nullptr};
    int64_t preOffset_{0};
    int64_t perGroupBOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseM_{0};
    uint32_t baseN_{0};
    uint32_t quantGroupSize_{0};
    uint32_t quantGroupNum_{1};
    QuantMode quantMode_{QuantMode::PERCHANNEL_MODE};
    uint8_t groupListType_{GMM_GROUP_LIST_OFFSET};
    bool isPerGroup_{false};
    bool isFirstBlock_{true};
};

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    if (groupNum_ == 0) {
        return;
    }

    const auto groupListLayout = asc::te::make_layout(asc::te::make_shape(static_cast<int64_t>(groupNum_)),
                                                      asc::te::make_stride(1L));
    const auto groupList = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
        groupListLayout);
    if ASCEND_IS_AIC {
        AscendC::SetMMLayoutTransform(true);
    }
    const auto& tiling = params.gmmParams;
    BlockScheduler scheduler(baseM_, baseN_, tiling.baseK);
    SetSchedulerTailAlign(scheduler);
    const uint32_t lastLoopIdx = groupNum_ - 1;
    for (uint32_t loopIdx = 0; loopIdx < lastLoopIdx; ++loopIdx) {
        const uint32_t groupIdx = GetGroupIdx(groupList, loopIdx);
        SetGroupShape(groupList, loopIdx);
        if (!IsValidProblem()) {
            if (groupListType_ == GMM_GROUP_LIST_SPARSE && asc::te::get<MNK_M>(problemShape_) <= 0) {
                break;
            }
            continue;
        }
        PrepareScheduler(scheduler);
        ProcessSingleGroup(scheduler, groupIdx);
    }

    const uint32_t groupIdx = GetGroupIdx(groupList, lastLoopIdx);
    SetGroupShape(groupList, lastLoopIdx);
    if (IsValidProblem()) {
        PrepareScheduler(scheduler);
        if (IsLastGroupAndNeedSplit(scheduler)) {
            scheduler.UpdateTailTile();
        }
        ProcessSingleGroup(scheduler, groupIdx);
    }
    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            WaitForVector();
        }
        AscendC::SetMMLayoutTransform(false);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::Init(const Params& params)
{
    const auto& tiling = params.gmmParams;
    problemShape_ = ProblemShape{tiling.m, tiling.n, tiling.k, 0};
    groupNum_ = tiling.groupNum;
    groupListType_ = tiling.groupListType;
    baseM_ = tiling.baseM;
    baseN_ = tiling.baseN;
    curBaseM_ = baseM_;
    isPerGroup_ = static_cast<QuantMode>(tiling.quantMode) == QuantMode::PERGROUP_MODE;
    quantMode_ = isPerGroup_ ? QuantMode::PERGROUP_MODE : QuantMode::PERCHANNEL_MODE;
    const int64_t problemK = tiling.k > 0 ? tiling.k : 0;
    quantGroupSize_ = isPerGroup_ ? (tiling.quantGroupSize > 0 ?
                                         tiling.quantGroupSize :
                                         (tiling.baseK > 0 ? tiling.baseK : static_cast<uint32_t>(problemK))) :
                                    static_cast<uint32_t>(problemK);
    quantGroupNum_ = isPerGroup_ && quantGroupSize_ > 0 && problemK > 0 ?
                         static_cast<uint32_t>(CeilDiv(problemK, static_cast<int64_t>(quantGroupSize_))) :
                         1;
    workspaceBasePtr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if ASCEND_IS_AIC {
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        scaleBasePtr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
        perGroupBOffset_ = GetPerGroupBOffset(tiling.n, tiling.k);
        const int64_t mmK = isPerGroup_ ? Min(problemK, static_cast<int64_t>(quantGroupSize_)) : problemK;
        const BlockShape mmProblemShape{tiling.m, tiling.n, mmK, 0};
        const BlockShape l0Shape{static_cast<int64_t>(baseM_), static_cast<int64_t>(baseN_),
                                 Min(static_cast<int64_t>(tiling.baseK), mmK), 0};
        mmOp_.Init(mmProblemShape, l0Shape, tiling.kAL1, tiling.kBL1, tiling.nBufferNum, quantMode_, false,
                   tiling.dbL0C > 1);
    }
    if ASCEND_IS_AIV {
        epilogueOp_.Init(params.epilogueParams);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline int64_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetPerGroupBOffset(int64_t n, int64_t k) const
{
    if constexpr (!WEIGHT_NZ) {
        return n * k;
    } else if constexpr (TRANS_B) {
        return Align16(n) * Align32(k);
    } else {
        return Align32(n) * Align16(k);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class GroupListTensor>
__aicore__ inline uint32_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetGroupIdx(const GroupListTensor& groupList,
                                                                              uint32_t loopIdx)
{
    if (groupListType_ == GMM_GROUP_LIST_SPARSE) {
        return static_cast<uint32_t>(groupList[loopIdx * GMM_SPARSE_STRIDE]);
    }
    return loopIdx;
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class GroupListTensor>
__aicore__ inline int64_t GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::GetGroupM(const GroupListTensor& groupList,
                                                                           uint32_t loopIdx)
{
    int64_t groupM = 0;
    if (groupListType_ == GMM_GROUP_LIST_OFFSET) {
        const int64_t offset = groupList[loopIdx];
        groupM = offset - preOffset_;
        preOffset_ = offset;
    } else if (groupListType_ == GMM_GROUP_LIST_LENGTH) {
        groupM = groupList[loopIdx];
        preOffset_ += groupM;
    } else {
        groupM = groupList[loopIdx * GMM_SPARSE_STRIDE + GMM_SPARSE_VALUE_OFFSET];
        preOffset_ += groupM;
    }
    return groupM;
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class GroupListTensor>
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::SetGroupShape(const GroupListTensor& groupList,
                                                                            uint32_t loopIdx)
{
    const int64_t groupM = GetGroupM(groupList, loopIdx);
    problemShape_ = ProblemShape{groupM, asc::te::get<MNK_N>(problemShape_), asc::te::get<MNK_K>(problemShape_), 0};
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline bool GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::IsValidProblem() const
{
    return asc::te::get<MNK_M>(problemShape_) > 0 && asc::te::get<MNK_N>(problemShape_) > 0 &&
           asc::te::get<MNK_K>(problemShape_) > 0;
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::PrepareScheduler(BlockScheduler& scheduler)
{
    const int64_t m = asc::te::get<MNK_M>(problemShape_);
    const int64_t n = asc::te::get<MNK_N>(problemShape_);
    const int64_t k = asc::te::get<MNK_K>(problemShape_);
    const int64_t safeBaseM = baseM_ > 0 ? baseM_ : static_cast<int64_t>(BLOCK_CUBE);
    const int64_t blockCount = CeilDiv(m, safeBaseM);
    const int64_t balancedBaseM = CeilDiv(m, blockCount);
    curBaseM_ = static_cast<uint32_t>((balancedBaseM + GMM_BLOCK_MASK) & ~GMM_BLOCK_MASK);
    scheduler.UpdateBaseM(curBaseM_);
    scheduler.UpdateNextProblem(SchedulerProblemShape{m, n, k, 0});
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::SetSchedulerTailAlign(BlockScheduler& scheduler)
{
    constexpr uint32_t mTailAlign = 1;
    constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(C0_SIZE);
    scheduler.SetTailAlign(mTailAlign, nTailAlign);
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline bool GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::IsLastGroupAndNeedSplit(
    const BlockScheduler& scheduler) const
{
    return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
}

GMM_FIXPIPE_TEMPLATE_DEF
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::ProcessSingleGroup(BlockScheduler& scheduler,
                                                                                 uint32_t groupIdx)
{
    const int64_t m = asc::te::get<MNK_M>(problemShape_);
    const int64_t n = asc::te::get<MNK_N>(problemShape_);
    const int64_t k = asc::te::get<MNK_K>(problemShape_);
    const int64_t groupMOffset = preOffset_ - m;

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + groupMOffset * k),
                                    layoutA);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(bBasePtr_ + static_cast<int64_t>(groupIdx) * perGroupBOffset_),
        layoutB);
    SchedulerBlockInfo blockInfo;
    while (scheduler.GetNextBlock(blockInfo)) {
        const int64_t curM = asc::te::get<MNK_M>(blockInfo.blockShape);
        const int64_t curN = asc::te::get<MNK_N>(blockInfo.blockShape);
        if (curM <= 0 || curN <= 0) {
            continue;
        }

        const int64_t mPos = asc::te::get<MNK_M>(blockInfo.blockCoord);
        const int64_t nPos = asc::te::get<MNK_N>(blockInfo.blockCoord);
        const BlockShape blockShape{curM, curN, k, 0};
        ProcessOneBlock(gmA, gmB, blockShape, groupIdx, groupMOffset, mPos, nPos, curM, curN, k, n);
    }
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class TensorA, class TensorB, class TensorScale, class TensorBias, class TensorC>
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::ProcessPerGroup(const TensorA& gmA, const TensorB& gmB,
                                                                              const TensorScale& gmScale,
                                                                              const TensorBias& gmBias,
                                                                              const TensorC& gmC,
                                                                              const BlockShape& blockShape)
{
    const uint64_t fullK = static_cast<uint64_t>(asc::te::get<IDX_K_IDX>(blockShape));
    const uint64_t curM = static_cast<uint64_t>(asc::te::get<IDX_M_TILEIDX>(blockShape));
    const uint64_t curN = static_cast<uint64_t>(asc::te::get<IDX_N_TILEIDX>(blockShape));

    for (uint32_t groupIdx = 0; groupIdx < quantGroupNum_; ++groupIdx) {
        const uint64_t kOffset = static_cast<uint64_t>(groupIdx) * quantGroupSize_;
        const uint64_t curK = Min(fullK - kOffset, static_cast<uint64_t>(quantGroupSize_));
        auto gmGroupA = gmA.slice(asc::te::make_coord(0UL, kOffset), asc::te::make_shape(curM, curK));
        auto gmGroupB = gmB.slice(asc::te::make_coord(kOffset, 0UL), asc::te::make_shape(curK, curN));
        auto gmGroupScale = gmScale.slice(asc::te::make_coord(static_cast<uint64_t>(groupIdx), 0UL),
                                          asc::te::make_shape(1UL, curN));
        const BlockShape groupShape{static_cast<int64_t>(curM), static_cast<int64_t>(curN), static_cast<int64_t>(curK),
                                    0};

        mmOp_.UpdateParamsForKSlice(curK);
        if (groupIdx == 1U) {
            // Group 0 initializes C with a normal FixPipe write. Keep atomic-add enabled for all remaining groups.
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::SetAtomicAdd<CType>();
        }
        mmOp_(gmGroupA, gmGroupB, gmGroupScale, gmBias, gmC, groupShape);
    }
    if (quantGroupNum_ > 1U) {
        AscendC::PipeBarrier<PIPE_FIX>();
        AscendC::SetAtomicNone();
    }
    mmOp_.UpdateParamsForKSlice(fullK);
}

GMM_FIXPIPE_TEMPLATE_DEF
template <class TensorA, class TensorB>
__aicore__ inline void GemmUniversal<GMM_FIXPIPE_TEM_PARAMS>::ProcessOneBlock(const TensorA& gmA, const TensorB& gmB,
                                                                              const BlockShape& blockShape,
                                                                              uint32_t groupIdx, int64_t groupMOffset,
                                                                              int64_t mPos, int64_t nPos, int64_t curM,
                                                                              int64_t curN, int64_t k, int64_t n)
{
    const uint64_t workspaceOffset = static_cast<uint64_t>(AscendC::GetBlockIdx() / AscendC::GetTaskRation()) *
                                     static_cast<uint64_t>(baseN_) * baseM_;
    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            WaitForVector();
        }
        auto workspaceLayout = MakeLayoutC{}(curM, curN);
        auto gmWorkspace = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(workspaceBasePtr_ + workspaceOffset), workspaceLayout);
        // BlockMmad keeps a bias tensor in its call signature. Init uses isBias=false,
        // so this placeholder is never loaded or involved in the calculation.
        auto unusedBiasLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, 1L);
        auto gmUnusedBias = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(workspaceBasePtr_)),
            unusedBiasLayout);

        auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)), asc::te::make_shape(curM, k));
        auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos), asc::te::make_shape(k, curN));
        const int64_t scaleExpertOffset = static_cast<int64_t>(groupIdx) * quantGroupNum_ * n;
        auto scaleLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(quantGroupNum_),
                                                                                  n);
        auto gmScale = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleBasePtr_ + scaleExpertOffset), scaleLayout);
        auto gmBlockScale = gmScale.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                          asc::te::make_shape(static_cast<int64_t>(quantGroupNum_), curN));
        if (isPerGroup_) {
            ProcessPerGroup(gmBlockA, gmBlockB, gmBlockScale, gmUnusedBias, gmWorkspace, blockShape);
        } else {
            mmOp_(gmBlockA, gmBlockB, gmBlockScale, gmUnusedBias, gmWorkspace, blockShape);
        }
        NotifyVector();
        isFirstBlock_ = false;
    }

    if ASCEND_IS_AIV {
        WaitForCube();
        const int64_t offsetPerTokenScale = groupMOffset + mPos;
        const int64_t offsetOffset = static_cast<int64_t>(groupIdx) * n + nPos;
        const int64_t offsetRowSum = offsetPerTokenScale;
        const int64_t offsetC = (groupMOffset + mPos) * n + nPos;
        epilogueOp_(curM, curN, offsetPerTokenScale, offsetOffset, offsetRowSum, offsetC, workspaceOffset);
        NotifyCube();
    }
}

#undef GMM_FIXPIPE_TEMPLATE_DEF
#undef GMM_FIXPIPE_TEM_PARAMS
#undef GMM_FIXPIPE_CLASS_TEM_PARAMS

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
