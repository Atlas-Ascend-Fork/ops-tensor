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
 * \file kernel_qgmm_cube.h
 * \brief Grouped quantized matmul cube kernel (A8W8 fixpipe dequant, Tensor API)
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGroupedMmadFixpipeQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using GmmArrayPtr = typename BlockMmad::DispatchPolicy::GmmArrayPtr;

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    // Fixpipe dequant scale is carried in a uint64 register / L1 layout regardless of the GM dtype.
    using X2ScaleType = uint64_t;
    // GM scale tensor's real dtype (fp32 / bf16 / uint64 / int64), resolved in Init().
    using ScaleGmType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

private:
    static constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL;
    static constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL;
    static constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL;
    static constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
    static constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
    static constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
    static constexpr uint32_t LEFT_SHIFT_16 = 16;
    // Bit mask that keeps the upper 19 bits of the fp32 scale bits (see kernel_qbmm_cube.h).
    static constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000UL;
    static constexpr int64_t GMM_CUBE_MKN_LIST_LEN = 128; // GMMArray mList/kList/nList length
    static constexpr int64_t GMM_CUBE_NZ_OUTER_SIZE = 16; // fractal NZ outer alignment
    static constexpr int8_t GMM_CUBE_NO_SPLIT = -1;
    static constexpr int8_t GMM_CUBE_SPLIT_M = 0;
    static constexpr uint32_t HALF_CORE_NUM = 2; // divisor to compute half of the participating cores
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;

    // Compile-time validation for the current QGMM Cube (AIC_ONLY) supported scenarios.
    using InputTypePair = AscendC::Std::tuple<AType, BType>;
    using OutputBiasTypePair = AscendC::Std::tuple<CType, BiasType>;
    using Int8InputTypePair = AscendC::Std::tuple<int8_t, int8_t>;
    using HiFloat8InputTypePair = AscendC::Std::tuple<hifloat8_t, hifloat8_t>;

    static constexpr bool IS_INT8_INPUT = AscendC::Std::is_same_v<InputTypePair, Int8InputTypePair>;
    static constexpr bool IS_HIFLOAT8_INPUT = AscendC::Std::is_same_v<InputTypePair, HiFloat8InputTypePair>;
    static constexpr bool IS_FP8_INPUT = AscendC::Std::is_one_of_v<AType, fp8_e4m3fn_t, fp8_e5m2_t> &&
                                         AscendC::Std::is_one_of_v<BType, fp8_e4m3fn_t, fp8_e5m2_t>;

    static_assert(IS_INT8_INPUT || IS_HIFLOAT8_INPUT || IS_FP8_INPUT,
                  "QGMM Cube only supports INT8xINT8, HIFLOAT8xHIFLOAT8 or FP8xFP8 inputs.");
    static_assert(
        !IS_INT8_INPUT ||
            AscendC::Std::is_one_of_v<OutputBiasTypePair, AscendC::Std::tuple<half, int32_t>,
                                      AscendC::Std::tuple<bfloat16_t, int32_t>, AscendC::Std::tuple<int8_t, int32_t>,
                                      AscendC::Std::tuple<int32_t, int32_t>>,
        "QGMM Cube INT8xINT8 requires CType FP16/BF16/INT8/INT32 and BiasType INT32.");
    static_assert(
        !(IS_HIFLOAT8_INPUT || IS_FP8_INPUT) ||
            AscendC::Std::is_one_of_v<OutputBiasTypePair, AscendC::Std::tuple<half, float>,
                                      AscendC::Std::tuple<bfloat16_t, float>, AscendC::Std::tuple<float, float>>,
        "QGMM Cube HIFLOAT8/FP8 requires CType FP16/BF16/FP32 and BiasType FLOAT.");
    static_assert(AscendC::Std::is_one_of_v<ScaleGmType, uint64_t, int64_t, bfloat16_t, float>,
                  "QGMM Cube only supports UINT64/INT64/BF16/FLOAT ScaleGmType.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "QGMM Cube only supports non-transposed A (NDExtLayoutPtn).");
    static_assert(AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn>, "QGMM Cube only supports ND LayoutC.");

public:
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct GmmParams {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1; // = stepKa * baseK
        uint32_t kBL1; // = stepKb * baseK
        // L1 buffer count is intentionally not part of GmmParams: the grouped fixpipe-quant
        // BlockMmad path currently only supports double buffering (see Init below).
        uint32_t x1QuantMode; // A side quant mode (QuantMode)
        uint32_t x2QuantMode; // B side quant mode (QuantMode)
        uint8_t isBias;
        uint8_t dbL0C;
        // Supported group types: NO_SPLIT (-1) and SPLIT_M (0). SPLIT_K is unsupported.
        int8_t groupType;
        uint8_t groupListType;
        uint8_t singleW;
        uint8_t singleX;
        uint8_t singleY;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GmmArrayPtr gmmArrayGmAddr; // GMMArray (mList/kList/nList), per-group independent shapes
        GmmParams gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    template <typename T>
    __aicore__ inline __gm__ T* GetTensorAddrFromTensorList(uint32_t index, __gm__ T* tensorPtr) const
    {
        AscendC::ListTensorDesc tensorList(reinterpret_cast<__gm__ void*>(tensorPtr));
        return tensorList.GetDataPtr<T>(index);
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler)
    {
        if constexpr (!TRANS_A) {
            // Tail-split M granularity must keep the cube 16-row alignment (same as cgmct
            // GmmASWKernel::CalcTailTile mMin = CUBE_BLOCK for the non-transA case); a value of 1
            // produced unaligned split shares (e.g. 43 rows) that break the Mmad hardware.
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(BLOCK_CUBE);
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(C0_SIZE);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        } else {
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                      static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    template <typename TensorB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, int64_t mSize, int64_t curBaseM, int64_t baseN)
    {
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        if constexpr (WEIGHT_NZ) {
            if (curBaseM >= mSize) {
                gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
            } else {
                gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
            }
        } else {
            if constexpr (TRANS_B) {
                if (curBaseM >= mSize) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            } else {
                constexpr uint64_t cacheLineAlignMask = 0xffUL; // 256 cache line alignment
                if (curBaseM >= mSize && (problemN & cacheLineAlignMask) == 0 && (baseN & cacheLineAlignMask) == 0) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            }
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const int64_t groupListLength = groupType_ == GMM_CUBE_NO_SPLIT ?
                                            0 :
                                            static_cast<int64_t>(groupNum_) *
                                                (groupListType_ == GROUP_LIST_TYPE_SPARSE ?
                                                     SPARSE_GROUP_LIST_ITEM_STRIDE :
                                                     1UL);
        const auto groupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
            asc::te::make_layout(asc::te::make_shape(groupListLength), asc::te::make_stride(1L)));
        const auto& gmmParams = params.gmmParams;
        BlockScheduler scheduler(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t loopIdx = 0; loopIdx < lastGroupIdx; ++loopIdx) {
            uint32_t groupIdx = loopIdx;
            if (groupType_ != GMM_CUBE_NO_SPLIT && groupListType_ == GROUP_LIST_TYPE_SPARSE) {
                groupIdx = static_cast<uint32_t>(groupList[loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
            }
            SetMNK(groupList, loopIdx, groupIdx);
            const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
            const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
            const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
            // Offsets must advance for EVERY group on EVERY core regardless of whether this core
            // owns a block in the group (same contract as cgmct UpdateGroupOffset, which runs
            // unconditionally before the skip check). Otherwise cores that skip a group fall
            // behind on the accumulated x/w/y/scale offsets and write later groups at wrong
            // addresses (multi-group output corruption).
            UpdateBaseOffsets(groupIdx);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && problemM <= 0) {
                    break;
                }
                continue;
            }
            // NOTE: BaseMBalance (per-group dynamic M re-balancing) is intentionally disabled to
            // keep behaviour aligned with cgmct (host-provided baseM). The implementation is kept
            // for future performance work; re-enable by uncommenting the line below.
            // BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            ProcessSingleGroup<false>(scheduler, groupIdx);
        }

        uint32_t groupIdx = lastGroupIdx;
        if (groupType_ != GMM_CUBE_NO_SPLIT && groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            groupIdx = static_cast<uint32_t>(groupList[lastGroupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
        }
        SetMNK(groupList, lastGroupIdx, groupIdx);
        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        UpdateBaseOffsets(groupIdx); // unconditional offset advance, see main loop comment
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            // BaseMBalance(scheduler, problemM, gmmParams.baseM);  // see note in main loop
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
                ProcessSingleGroup<true>(scheduler, groupIdx);
            } else {
                ProcessSingleGroup<false>(scheduler, groupIdx);
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        // x / weight / y / bias are ListTensorDesc. Keep the descriptor addresses here;
        // ProcessSingleGroup resolves tensor zero and adds the corresponding group offset.
        xDescAddr_ = params.mmadParams.aGmAddr;
        wDescAddr_ = params.mmadParams.bGmAddr;
        yDescAddr_ = params.mmadParams.cGmAddr;
        biasDescAddr_ = params.mmadParams.biasGmAddr;
        singleW_ = gmmParams.singleW == 1;
        if (gmmParams.isBias == 1) {
            isBias_ = true;
        }
        // Per-group independent shapes come from GMMArray (mList/kList/nList), same contract as the
        // cgmct GmmASWKernel: SPLIT_M -> mList[0]=-1 (M from groupList).
        if (params.gmmArrayGmAddr != nullptr) {
            mListGm_ = params.gmmArrayGmAddr;
            kListGm_ = params.gmmArrayGmAddr + GMM_CUBE_MKN_LIST_LEN;
            nListGm_ = params.gmmArrayGmAddr + GMM_CUBE_MKN_LIST_LEN * 2;
        }
        const ProblemShape initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        groupType_ = gmmParams.groupType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        x1QuantMode_ = gmmParams.x1QuantMode;
        x2QuantMode_ = gmmParams.x2QuantMode;

        // Resolve the dequant scale the same way as kernel_qbmm_cube.h / gqmm_cube_on_the_fly.h:
        //  - B side PERCHANNEL: keep the GM tensor pointer, per-group offset + per-tile slice at call time;
        //  - A side PERTENSOR (x1) x B side PERTENSOR (double scale): per-group A scalar from perTokenScale
        //    multiplied by per-group B scalar;
        //  - B side PERTENSOR: per-group scalar from scaleBGmAddr, packed into scaleScalar_ per group.
        if (static_cast<QuantMode>(gmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE) {
            scaleBasePtr_ = reinterpret_cast<__gm__ X2ScaleType*>(params.mmadParams.scaleBGmAddr);
            isPerChannel_ = true;
        } else {
            // PERTENSOR paths: remember the GM base addresses; the per-group scalar is resolved per group.
            scaleBBasePtr_ = reinterpret_cast<__gm__ ScaleGmType*>(params.mmadParams.scaleBGmAddr);
            pertokenScaleBasePtr_ = reinterpret_cast<__gm__ float*>(params.mmadParams.scaleAGmAddr);
        }

        const BlockShape l0Shape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                 static_cast<int64_t>(gmmParams.baseK), 0};
        const bool enableL0CPingPong = gmmParams.dbL0C > 1;
        // Flat Init signature of the fixpipe-quant BlockMmad (see block_mmad_a8w8_fixpipe_quant.h).
        mmadOp_.Init(initProblemShape, l0Shape, static_cast<uint64_t>(gmmParams.kAL1),
                     static_cast<uint64_t>(gmmParams.kBL1), Blaze::Gemm::DOUBLE_BUFFER_COUNT,
                     static_cast<QuantMode>(gmmParams.x2QuantMode), isBias_, enableL0CPingPong);
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
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() / HALF_CORE_NUM);
    }

    __aicore__ inline void UpdateScaleScalar(uint32_t groupIdx)
    {
        if constexpr (AscendC::IsSameType<CType, int32_t>::value) {
            scaleScalar_ = 0;
            return;
        }
        if (isPerChannel_) {
            return;
        }
        // Per-group B side scalar, following gqmm_cube_on_the_fly.h:UpdateMMGlobalAddr:
        // scaleB[groupIdx] is the B scalar of this group; the A side pertoken scalar (when enabled)
        // is perTokenScale[groupIdx] and multiplied in for the doubleScale path (any scale GM dtype).
        const bool isDoubleScale = static_cast<QuantMode>(x1QuantMode_) == QuantMode::PERTENSOR_MODE &&
                                   static_cast<QuantMode>(x2QuantMode_) == QuantMode::PERTENSOR_MODE;
        if (isDoubleScale && !AscendC::IsSameType<ScaleGmType, uint64_t>::value &&
            !AscendC::IsSameType<ScaleGmType, int64_t>::value) {
            // Read B using its GM dtype before multiplying by the fp32 A scalar.
            float scaleB;
            if constexpr (AscendC::IsSameType<ScaleGmType, bfloat16_t>::value) {
                auto scale = AscendC::GlobalTensor<uint16_t>();
                scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scaleBBasePtr_) + groupIdx);
                uint32_t scaleBits = static_cast<uint32_t>(scale.GetValue(0)) << LEFT_SHIFT_16;
                scaleB = *reinterpret_cast<float*>(&scaleBits);
            } else {
                auto scale = AscendC::GlobalTensor<float>();
                scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleBBasePtr_) + groupIdx);
                scaleB = scale.GetValue(0);
            }
            auto pertokenScale = AscendC::GlobalTensor<float>();
            pertokenScale.SetGlobalBuffer(pertokenScaleBasePtr_ + groupIdx);
            float deqScale = scaleB * pertokenScale.GetValue(0);
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
            return;
        }
        if constexpr (AscendC::IsSameType<ScaleGmType, uint64_t>::value ||
                      AscendC::IsSameType<ScaleGmType, int64_t>::value) {
            // Use GlobalTensor GM load (same as CMCT); do not dereference GM_ADDR directly on AICore.
            auto scale = AscendC::GlobalTensor<uint64_t>();
            scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scaleBBasePtr_) + groupIdx);
            scaleScalar_ = scale.GetValue(0);
        } else if constexpr (AscendC::IsSameType<ScaleGmType, bfloat16_t>::value) {
            auto scale = AscendC::GlobalTensor<uint16_t>();
            scale.SetGlobalBuffer((__gm__ uint16_t*)scaleBBasePtr_ + groupIdx);
            uint16_t uint16Scale = scale.GetValue(0);
            uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << LEFT_SHIFT_16);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        } else {
            auto scale = AscendC::GlobalTensor<uint32_t>();
            scale.SetGlobalBuffer((__gm__ uint32_t*)scaleBBasePtr_ + groupIdx);
            uint32_t uint32Scale = scale.GetValue(0);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        }
    }

    template <class GroupListTensor>
    __aicore__ inline void SetMNK(const GroupListTensor& groupList, uint32_t listIdx, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(groupList, listIdx);
        // Per-group independent shapes come from GMMArray, same contract as the cgmct
        // GmmASWKernel::SetMNK: SPLIT_M -> M from groupList, k/n from list;
        // NO_SPLIT -> m/k/n all from list. SPLIT_K is not supported in this kernel.
        switch (groupType_) {
            case GMM_CUBE_SPLIT_M:
                problemShape_ = ProblemShape{splitValue, static_cast<int64_t>(nListGm_[singleW_ ? 0 : groupIdx]),
                                             static_cast<int64_t>(kListGm_[singleW_ ? 0 : groupIdx]), 0};
                break;
            default: // NO_SPLIT
                problemShape_ = ProblemShape{static_cast<int64_t>(mListGm_[groupIdx]),
                                             static_cast<int64_t>(nListGm_[groupIdx]),
                                             static_cast<int64_t>(kListGm_[groupIdx]), 0};
                break;
        }
    }

    template <class GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& groupList, uint32_t listIdx)
    {
        if (groupType_ == GMM_CUBE_NO_SPLIT) {
            return 0;
        }
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = groupList[listIdx];
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            splitValue = groupList[listIdx];
        } else {
            const uint32_t splitValueIdx = listIdx * SPARSE_GROUP_LIST_ITEM_STRIDE +
                                           SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = groupList[splitValueIdx];
        }
        return splitValue;
    }

    __aicore__ inline void UpdateBaseOffsets(uint32_t groupIdx)
    {
        // X/Y follow groupList traversal order. Sparse split-M expert parameters follow
        // the actual group index, including index zero after another expert.
        const int64_t m = asc::te::get<MNK_M>(problemShape_);
        const int64_t n = asc::te::get<MNK_N>(problemShape_);
        const int64_t k = asc::te::get<MNK_K>(problemShape_);
        aOffset_ = xBaseOffset_;
        wOffset_ = wBaseOffset_;
        biasOffset_ = nAxisBaseOffset_;
        cOffset_ = yBaseOffset_;
        scaleOffset_ = nAxisBaseOffset_; // per-channel B scale: one uint64 per column, n columns per group
        xBaseOffset_ += m * k;
        int64_t weightGroupStride = n * k;
        if constexpr (WEIGHT_NZ) {
            constexpr int64_t c0 = asc::te::c0_element<BType>;
            if constexpr (TRANS_B) {
                weightGroupStride = CeilAlign(n, GMM_CUBE_NZ_OUTER_SIZE) * CeilAlign(k, c0);
            } else {
                weightGroupStride = CeilAlign(n, c0) * CeilAlign(k, GMM_CUBE_NZ_OUTER_SIZE);
            }
        }
        if (groupType_ == GMM_CUBE_SPLIT_M && groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            wOffset_ = static_cast<int64_t>(groupIdx) * weightGroupStride;
            biasOffset_ = static_cast<int64_t>(groupIdx) * n;
            scaleOffset_ = biasOffset_;
        }
        wBaseOffset_ += weightGroupStride;
        nAxisBaseOffset_ += n;
        yBaseOffset_ += m * n;
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
    {
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        UpdateScaleScalar(groupIdx);

        // Different groups may carry different M/N/K sizes (SPLIT_M or NO_SPLIT). The fixpipe-quant BlockMmad does not
        // cache a per-problem shape (it takes the tile shape from each operator() call), so nothing to refresh
        // here; double-buffer phase continuity is kept by the kernel-level scheduler loop.
        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        const int64_t baseN = static_cast<int64_t>(baseN_);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        // Match GmmASWKernel: resolve tensor zero and apply the group offset.
        auto aPtr = GetTensorAddrFromTensorList(0, reinterpret_cast<__gm__ AType*>(xDescAddr_)) + aOffset_;
        auto bPtr = GetTensorAddrFromTensorList(0, reinterpret_cast<__gm__ BType*>(wDescAddr_)) + wOffset_;
        auto cPtr = GetTensorAddrFromTensorList(0, reinterpret_cast<__gm__ CType*>(yDescAddr_)) + cOffset_;
        __gm__ BiasType* biasPtr = nullptr;
        if (isBias_) {
            biasPtr = GetTensorAddrFromTensorList(0, reinterpret_cast<__gm__ BiasType*>(biasDescAddr_)) + biasOffset_;
        }
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aPtr), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bPtr), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cPtr), layoutC);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), problemN);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);

        const bool isPerChannel = isPerChannel_;

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, problemM, static_cast<int64_t>(curBaseM_), static_cast<int64_t>(baseN_));
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
            auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                      asc::te::make_shape(blockK, blockN));
            auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(blockM, blockN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                            asc::te::make_shape(static_cast<int64_t>(1), blockN));
            if (isPerChannel && !AscendC::IsSameType<CType, int32_t>::value) {
                // Per-channel B scale: single tensor [E, n], one uint64 per column, n columns per group;
                // The offset follows the expert index for sparse split-M, traversal order otherwise.
                auto layoutScale = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                              asc::te::layout_trait_default<X2ScaleType>>(
                    static_cast<int64_t>(1), problemN);
                auto gmScale = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(scaleBasePtr_ + scaleOffset_), layoutScale);
                auto gmBlockScale = gmScale.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                                  asc::te::make_shape(static_cast<int64_t>(1), blockN));
                mmadOp_(gmBlockA, gmBlockB, gmBlockScale, gmBlockBias, gmBlockC, blockShape);
            } else {
                mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, blockShape);
            }
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

    BlockMmad mmadOp_;
    ProblemShape problemShape_{};

    GM_ADDR xDescAddr_{nullptr};
    GM_ADDR wDescAddr_{nullptr};
    GM_ADDR yDescAddr_{nullptr};
    GM_ADDR biasDescAddr_{nullptr};
    GmmArrayPtr mListGm_{nullptr};
    GmmArrayPtr kListGm_{nullptr};
    GmmArrayPtr nListGm_{nullptr};
    __gm__ X2ScaleType* scaleBasePtr_{nullptr};   // per-channel B scale (uint64 per column)
    __gm__ ScaleGmType* scaleBBasePtr_{nullptr};  // per-tensor B scalar base (GM real dtype)
    __gm__ float* pertokenScaleBasePtr_{nullptr}; // per-token/per-group A scalar base (perTokenScale)

    int64_t preOffset_{0};
    int64_t aOffset_{0};
    int64_t wOffset_{0};
    int64_t scaleOffset_{0};
    int64_t biasOffset_{0};
    int64_t cOffset_{0};
    int64_t xBaseOffset_{0};
    int64_t wBaseOffset_{0};
    int64_t nAxisBaseOffset_{0};
    int64_t yBaseOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint32_t x1QuantMode_{0};
    uint32_t x2QuantMode_{0};
    int8_t groupType_{GMM_CUBE_SPLIT_M};
    uint8_t groupListType_{GROUP_LIST_TYPE_OFFSET};
    bool isBias_{false};
    bool singleW_{true};
    bool isPerChannel_{false};
    uint64_t scaleScalar_{0UL};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
