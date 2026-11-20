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
 * \file kernel_qgmm_mx.h
 * \brief
 */

#pragma once
#include "kernel_basic_intf.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGroupedMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>> {
private:
    static constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL;
    static constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL;
    static constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL;
    static constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
    static constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
    static constexpr uint32_t TENSOR_LIST_BYTE_OFFSET_TO_INDEX_SHIFT = 3U;
    static constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
    static constexpr int64_t SCALE_CACHE_MASK = 0xff;

    template <typename T>
    __aicore__ inline static __gm__ T* GetTensorAddrFromTensorList(uint32_t index, __gm__ T* tensorPtr)
    {
        const auto tensorListAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        const auto tensorListHeader = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(tensorListAddr),
            asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_stride(1L)));
        const uint64_t tensorPtrOffset = tensorListHeader[0];
        const auto tensorList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(tensorListAddr +
                                                         (tensorPtrOffset >> TENSOR_LIST_BYTE_OFFSET_TO_INDEX_SHIFT)),
            asc::te::make_layout(asc::te::make_shape(static_cast<int64_t>(index) + 1), asc::te::make_stride(1L)));
        const uint64_t tensorAddr = tensorList[index];
        return reinterpret_cast<__gm__ T*>(tensorAddr);
    }

public:
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutBias = typename BlockMmad::LayoutBias;

    static constexpr bool ENABLE_UB_INPLACE_ADD = !AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>;
    static constexpr bool ENABLE_ATOMIC_ADD = DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr bool ENABLE_INPLACE_ADD = ENABLE_UB_INPLACE_ADD || ENABLE_ATOMIC_ADD;
    static_assert(!ENABLE_ATOMIC_ADD || AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>,
                  "QGMM MX atomic add requires an empty epilogue.");
    static constexpr bool IS_MIXED_INPUT = AscendC::Std::is_same_v<AType, fp8_e4m3fn_t> &&
                                           (AscendC::Std::is_same_v<BType, fp4x2_e2m1_t> || IsHif4<BType>());
    static constexpr bool IS_MX_A8W4 = BlockMmad::IS_MX_A8W4;
    using ScaleBType = MxScaleType<BType>;
    static_assert((IsFp4<AType>() && !IsHif4<AType>() && IsFp4<BType>() && !IsHif4<BType>()) ||
                      (IsFp8<AType>() && IsFp8<BType>()) || (ENABLE_INPLACE_ADD && IS_MIXED_INPUT) || IS_MX_A8W4,
                  "QGMM MX requires same-width inputs, ND A / ZN B / DN scaleB MXA8W4, or FP8 x FP4/HIF4 inplace add.");
    static_assert(!ENABLE_INPLACE_ADD || (IS_MIXED_INPUT && AscendC::Std::is_same_v<CType, float> &&
                                          AscendC::Std::is_same_v<LayoutA, asc::te::dn_ext_layout_ptn> &&
                                          AscendC::Std::is_same_v<LayoutB, asc::te::nd_ext_layout_ptn> &&
                                          AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn>),
                  "QGMM MX inplace add requires FP8 x FP4/HIF4, DN A, ND B/C, FP32 C.");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
                  "QGMM MX only supports half/bfloat16_t/float CType.");
    static_assert(AscendC::Std::is_same_v<BiasType, float> ||
                      (IS_MX_A8W4 && AscendC::Std::is_one_of_v<BiasType, half, bfloat16_t>),
                  "QGMM MX requires float bias; MXA8W4 also supports half/bfloat16 bias via L1-to-BT.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QGMM MX only supports ND/DN LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QGMM MX only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutBias, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QGMM MX does not support NZ/ZN LayoutC or LayoutBias.");

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool SCALE_NZ = IsScaleNz<LayoutScaleB>::value;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;

    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1; // ScaleA L1 K-axis split size; must equal scaleKBL1 for MX.
        uint32_t scaleKBL1; // ScaleB L1 K-axis split size; kept for tiling compatibility.
        uint8_t isBias;
        uint8_t dbL0C{DOUBLE_BUFFER_COUNT};
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
        // Reserved for GMM tiling compatibility. Current kernel does not read this field;
        // The split axis is selected by LayoutA: !TRANS_A means split-M, TRANS_A means split-K.
        int8_t groupType;
        uint8_t groupListType;
        uint8_t singleW;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr uint32_t C0_SIZE_B = IsFp4<BType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE_B>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, AscendC::Std::Int<GetScaleC0<BType>()>>;

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler)
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            scheduler.SetTailAlign(1, C0_SIZE_B);
        } else if constexpr (!TRANS_A) {
            constexpr uint32_t mTailAlign = 1;
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(C0_SIZE_B);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        } else {
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, TensorScaleB& gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        if constexpr (WEIGHT_NZ) {
            if (curBaseM >= mSize) {
                gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
            } else {
                gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
            }
        } else {
            if constexpr (TRANS_B) {
                if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            } else {
                if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            }
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if constexpr (!ENABLE_UB_INPLACE_ADD) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        if (params.gmmParams.groupNum == 0) {
            return;
        }
        Init(params);
        const int64_t groupListSize = static_cast<int64_t>(groupNum_) *
                                      (groupListType_ == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1UL);
        const auto groupListLayout = asc::te::make_layout(asc::te::make_shape(groupListSize), asc::te::make_stride(1L));
        const auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
            groupListLayout);
        const auto& gmmParams = params.gmmParams;
        BlockScheduler scheduler(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        if constexpr (ENABLE_UB_INPLACE_ADD) {
            epilogueOp_.Init(params.epilogueParams);
        }
        if constexpr (ENABLE_ATOMIC_ADD) {
            AscendC::SetAtomicAdd<float>();
        }
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t loopIdx = 0; loopIdx < lastGroupIdx; ++loopIdx) {
            uint32_t groupIdx = loopIdx;
            if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
                groupIdx = static_cast<uint32_t>(gmGroupList[loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
            }
            SetMNK(gmGroupList, loopIdx);
            const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
            const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
            const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && problemM <= 0) {
                    break;
                }
                continue;
            }
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            ProcessSingleGroup<false>(scheduler, groupIdx);
        }

        uint32_t groupIdx = lastGroupIdx;
        if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            groupIdx = static_cast<uint32_t>(gmGroupList[lastGroupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
        }
        SetMNK(gmGroupList, lastGroupIdx);
        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
                ProcessSingleGroup<true>(scheduler, groupIdx);
            } else {
                ProcessSingleGroup<false>(scheduler, groupIdx);
            }
        }
        if constexpr (ENABLE_ATOMIC_ADD) {
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::SetAtomicNone();
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ ScaleBType*>(params.mmadParams.scaleBGmAddr);
        singleW_ = params.gmmParams.singleW == 1;
        if (gmmParams.isBias == 1) {
            biasBasePtr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
            isBias_ = true;
        }
        const ProblemShape initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        if constexpr (!TRANS_A) {
            if constexpr (!WEIGHT_NZ) {
                perGroupBOffset_ = gmmParams.n * gmmParams.k;
            } else {
                if constexpr (TRANS_B) {
                    perGroupBOffset_ = static_cast<int64_t>(
                        Align16(gmmParams.n) * (IsFp4<BType>() ? Align64(gmmParams.k) : Align32(gmmParams.k)));
                } else {
                    perGroupBOffset_ = static_cast<int64_t>(
                        (IsFp4<BType>() ? Align64(gmmParams.n) : Align32(gmmParams.n)) * Align16(gmmParams.k));
                }
            }
        }

        if ASCEND_IS_AIV {
            return; // Both sides need group/address metadata; only AIC initializes MMAD resources.
        }
        const BlockShape l0Shape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                 static_cast<int64_t>(gmmParams.baseK), 0};
        // MX uses one shared scale K window for ScaleA/ScaleB. Tiling must provide scaleKAL1 == scaleKBL1.
        uint64_t scaleKL1 = gmmParams.scaleKAL1;
        if constexpr (ENABLE_INPLACE_ADD) {
            const uint64_t outerKL1 = Max(static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1));
            // Scale reloads must stay on complete outer K-strip boundaries.
            scaleKL1 = Min(Max(scaleKL1 / outerKL1, 1UL) * outerKL1, static_cast<uint64_t>(gmmParams.k));
        }
        const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1), scaleKL1};
        const typename BlockMmad::MmadParams mmadParams{
            l0Shape, l1Params, isBias_, gmmParams.dbL0C == DOUBLE_BUFFER_COUNT, gmmParams.l1BufferStage};
        mmadOp_.Init(initProblemShape, mmadParams);
    }

    __aicore__ inline void BaseMBalance(BlockScheduler& scheduler, int64_t m, int64_t baseM)
    {
        if constexpr (!TRANS_A) {
            if (m <= 0) {
                return;
            }
            const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
            const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
            const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
            curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
            scheduler.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler& scheduler)
    {
        if constexpr (ENABLE_INPLACE_ADD) {
            return false;
        }
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    template <typename GroupListTensor>
    __aicore__ inline void SetMNK(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(gmGroupList, groupIdx);
        const int64_t n = asc::te::get<MNK_N>(problemShape_);
        // Current MX scalar path selects the split axis from LayoutA, not from GMMTiling::groupType.
        if constexpr (!TRANS_A) {
            problemShape_ = ProblemShape{splitValue, n, asc::te::get<MNK_K>(problemShape_), 0};
        } else {
            problemShape_ = ProblemShape{asc::te::get<MNK_M>(problemShape_), n, splitValue, 0};
        }
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = gmGroupList[groupIdx];
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            splitValue = gmGroupList[groupIdx];
            preOffset_ += splitValue;
        } else {
            const uint32_t splitValueIdx = groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE +
                                           SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = gmGroupList[splitValueIdx];
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void UpdateBaseOffsets(uint32_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }

        const int64_t m = asc::te::get<MNK_M>(problemShape_);
        const int64_t n = asc::te::get<MNK_N>(problemShape_);
        const int64_t k = asc::te::get<MNK_K>(problemShape_);
        // The previous split offset follows the same current-scenario binding as SetMNK.
        const int64_t prevSplitOffset = preOffset_ - (TRANS_A ? k : m);
        if constexpr (!TRANS_A) {
            aOffset_ = IsFp4<AType>() ? ((prevSplitOffset * k) >> 1) : (prevSplitOffset * k);
            const int64_t scaleK = GetScaleK(k);
            if (singleW_) {
                bOffset_ = perGroupBOffset_ * static_cast<int64_t>(groupIdx);
                if constexpr (IsFp4<BType>()) {
                    bOffset_ >>= 1;
                }
                if constexpr (SCALE_NZ) {
                    scaleBOffset_ = static_cast<int64_t>(
                        CalScaleNZGmAddrOffset(TRANS_B, static_cast<int64_t>(groupIdx), n, scaleK));
                } else {
                    scaleBOffset_ = static_cast<int64_t>(groupIdx) * n * scaleK;
                }
            } else {
                bOffset_ = 0;
                scaleBOffset_ = 0;
            }
            scaleAOffset_ = prevSplitOffset * scaleK;
            cOffset_ = prevSplitOffset * n;
        } else {
            aOffset_ = IsFp4<AType>() ? ((m * prevSplitOffset) >> 1) : (m * prevSplitOffset);
            bOffset_ = IsFp4<BType>() ? ((n * prevSplitOffset) >> 1) : (n * prevSplitOffset);
            const int64_t transScaleGroupOffset = prevSplitOffset >> MXFP_DIVISOR_SHIFT;
            const int64_t transScaleStart = (transScaleGroupOffset + static_cast<int64_t>(groupIdx))
                                            << MXFP_MULTI_BASE_SHIFT;
            scaleAOffset_ = m * transScaleStart;
            scaleBOffset_ = n * transScaleStart;
            if constexpr (IsHif4<BType>()) {
                // HiFloat4 has one hif4_scale per K64 group; FP8 has two scale elements.
                scaleBOffset_ >>= 1;
            }
            cOffset_ = static_cast<int64_t>(groupIdx) * m * n;
        }
        biasOffset_ = static_cast<int64_t>(groupIdx) * n;
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    template <bool isLastGroupAndNeedSplit, bool UseUb = ENABLE_UB_INPLACE_ADD>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
    {
        if constexpr (UseUb) {
            // Match PR14161's per-group K policy; the host only knows total K.
            constexpr int64_t MIN_UB_GROUP_K = 2048;
            constexpr int64_t MAX_UB_GROUP_K = 16384;
            const int64_t groupK = asc::te::get<MNK_K>(problemShape_);
            if (groupK < MIN_UB_GROUP_K || groupK > MAX_UB_GROUP_K) {
                if ASCEND_IS_AIC {
                    AscendC::SetAtomicAdd<float>();
                }
                // Both processors must advance the same scheduler and group offsets.
                ProcessSingleGroup<isLastGroupAndNeedSplit, false>(scheduler, groupIdx);
                if ASCEND_IS_AIC {
                    AscendC::PipeBarrier<PIPE_FIX>();
                    AscendC::SetAtomicNone();
                }
                return;
            }
        }
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        UpdateBaseOffsets(groupIdx);

        // Different groups may carry different split-M / split-K sizes. Keep the double-buffer phase continuous
        // across groups and only refresh the problem shape.
        if ASCEND_IS_AIC {
            mmadOp_.UpdateParamsForNextProblem(problemShape_);
        }

        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        const int64_t scaleK = GetScaleK(problemK);
        const int64_t scaleKB = static_cast<int64_t>(MxScaleKLen<BType>(static_cast<uint64_t>(problemK)));
        const int64_t baseN = static_cast<int64_t>(baseN_);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleKB, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + aOffset_), layoutA);
        auto gmScaleA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleABasePtr_ + scaleAOffset_), layoutScaleA);
        auto gmB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                (singleW_ ? bBasePtr_ : GetTensorAddrFromTensorList(groupIdx, bBasePtr_)) + bOffset_),
            layoutB);
        auto scaleBPtr = (singleW_ ? scaleBBasePtr_ : GetTensorAddrFromTensorList(groupIdx, scaleBBasePtr_)) +
                         scaleBOffset_;
        auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBPtr), layoutScaleB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cBasePtr_ + cOffset_), layoutC);
        auto biasPtr = isBias_ ? (biasBasePtr_ + biasOffset_) : nullptr;
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), problemN);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_), static_cast<int64_t>(baseN_));
        }

        do {
            const SchedulerBlockShape schedulerBlockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t blockM = asc::te::get<MNK_M>(schedulerBlockShape);
            const int64_t blockN = asc::te::get<MNK_N>(schedulerBlockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t mSplitOffset = asc::te::get<MNK_K>(schedulerBlockShape);
            const int64_t nSplitOffset = asc::te::get<MNK_B>(schedulerBlockShape);
            const int64_t mBlockIdx = asc::te::get<MNK_M>(blockCoord);
            const int64_t nBlockIdx = asc::te::get<MNK_N>(blockCoord);
            const int64_t blockK = problemK;
            BlockShape blockShape{blockM, blockN, blockK, 0};
            const int64_t mPos = mBlockIdx * curBaseM_ + mSplitOffset;
            const int64_t nPos = nBlockIdx * baseN + nSplitOffset;

            auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)),
                                      asc::te::make_shape(blockM, blockK));
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)),
                                                asc::te::make_shape(blockM, scaleK));
            auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                      asc::te::make_shape(blockK, blockN));
            // HiF4 ND scales store Ea/Eb and Ec as two B16 rows, as emitted by QuantHiF4.
            const int64_t scaleNPos = IsHif4<BType>() &&
                                              AscendC::Std::is_same_v<LayoutScaleB, asc::te::scaleb_nd_layout_ptn> ?
                                          (nPos >> 1) :
                                          nPos;
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(static_cast<int64_t>(0), scaleNPos),
                                                asc::te::make_shape(scaleKB, blockN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                            asc::te::make_shape(static_cast<int64_t>(1), blockN));
            auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(blockM, blockN));
            if constexpr (UseUb) {
                auto ubPing = epilogueOp_.GetAccumulatorTensor({blockM, blockN, 0});
                auto ubPong = epilogueOp_.GetAccumulatorTensor({blockM, blockN, 1});
                const int64_t panelM = static_cast<int64_t>(asc::te::get<0, 1>(ubPing.layout().shape()));
                if (panelM <= 0) {
                    continue;
                }
                if ASCEND_IS_AIC {
                    // Compute the full tile once; alternate the two UB panel tensors on output.
                    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, ubPing, ubPong, blockShape,
                            inplacePanelIndex_);
                }
                if ASCEND_IS_AIV {
                    for (int64_t row = 0, panel = 0; row < blockM; row += panelM, ++panel) {
                        const int64_t rows = Min(panelM, blockM - row);
                        const uint64_t panelIndex = inplacePanelIndex_ + panel;
                        auto ubResult = epilogueOp_.GetAccumulatorTensor(
                            {rows, blockN, static_cast<uint32_t>(panelIndex & 1UL)});
                        auto panelC = gmBlockC.slice(asc::te::make_coord(row, 0L), asc::te::make_shape(rows, blockN));
                        epilogueOp_(ubResult, panelC, rows, blockN, panelIndex);
                    }
                }
                inplacePanelIndex_ += static_cast<uint64_t>(CeilDiv(blockM, panelM));
            } else {
                if ASCEND_IS_AIC {
                    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
                }
            }
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    uint64_t inplacePanelIndex_{0};
    ProblemShape problemShape_{};
    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ CType* cBasePtr_{nullptr};
    __gm__ BiasType* biasBasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ ScaleBType* scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};
    int64_t aOffset_{0};
    int64_t bOffset_{0};
    int64_t perGroupBOffset_{0};
    int64_t scaleAOffset_{0};
    int64_t scaleBOffset_{0};
    int64_t biasOffset_{0};
    int64_t cOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{GROUP_LIST_TYPE_OFFSET};
    bool isBias_{false};
    bool singleW_{true};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
