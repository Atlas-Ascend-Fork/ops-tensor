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
 * \file kernel_qbmm_mix.h
 * \brief MIX template kernel: AIC cube compute + AIV dequant epilogue (Tensor API).
 *        Replicates scheduling of original QuantBmmPertokenRegbaseKernel/AL1FullLoad.
 */

#pragma once
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MIX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_KERNEL_TEM_PARAMS                          \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, \
        AscendC::Std::enable_if_t<                          \
            AscendC::Std::is_same_v<KernelMmadWithScaleMix, typename BlockMmad::DispatchPolicy::ScheduleType>>

#define QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_KERNEL_FUNC_TEMPLATE_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

QBMM_MIX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using OutType = typename BlockEpilogue::OutType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

private:
    static constexpr bool IS_INT8_INPUT = AscendC::Std::is_same_v<AType, int8_t> &&
                                          AscendC::Std::is_same_v<BType, int8_t>;
    static constexpr bool IS_HIFLOAT8_INPUT = AscendC::Std::is_same_v<AType, hifloat8_t> &&
                                              AscendC::Std::is_same_v<BType, hifloat8_t>;
    static constexpr bool IS_FP8_INPUT = IsFp8<AType>() && IsFp8<BType>();
    static constexpr bool IS_SUPPORTED_OUTPUT = AscendC::Std::is_one_of_v<OutType, half, bfloat16_t, float>;

    static_assert(IS_INT8_INPUT || IS_HIFLOAT8_INPUT || IS_FP8_INPUT,
                  "QBMM Mix: AType/BType must both be int8_t, both be hifloat8_t, or each be fp8_e4m3fn_t/fp8_e5m2_t.");
    static_assert(IS_SUPPORTED_OUTPUT, "QBMM Mix: BlockEpilogue::OutType must be half/bfloat16_t/float.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::L0CType, L0CType>,
                  "QBMM Mix: BlockEpilogue::L0CType must match BlockMmad::L0CType.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM Mix: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QBMM Mix: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");

public:
    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        EpilogueParams epilogueParams;
    };

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void AddBatchOffset(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint64_t restBatch,
                                              bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB)
    {
        if ASCEND_IS_AIC {
            // 0x7f: 128-element alignment for 128-byte B matrix GM streaming
            constexpr uint64_t cacheLineAlignMask = 0x7fUL;
            const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
            const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                         isCurrentNAligned;
            gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }

    // Precompute batch dimension products and A(B)->C broadcast multipliers used by ProcessWithBatch.
    struct BatchMultipliers {
        uint64_t batchC3C4;
        uint64_t batchC2C3C4;
        uint64_t batchB3B4;
        uint64_t batchB2B3B4;
        uint64_t batchA3A4;
        uint64_t batchA2A3A4;
        uint32_t multiA1C1;
        uint32_t multiA2C2;
        uint32_t multiA3C3;
        uint32_t multiA4C4;
        uint32_t multiB1C1;
        uint32_t multiB2C2;
        uint32_t multiB3C3;
        uint32_t multiB4C4;
    };
    __aicore__ inline BatchMultipliers ComputeBatchMultipliers(const Params& params)
    {
        BatchMultipliers mul;
        mul.batchC3C4 = static_cast<uint64_t>(params.qbmmParams.batchC3) * params.qbmmParams.batchC4;
        mul.batchC2C3C4 = params.qbmmParams.batchC2 * mul.batchC3C4;
        mul.batchB3B4 = static_cast<uint64_t>(params.qbmmParams.batchB3) * params.qbmmParams.batchB4;
        mul.batchB2B3B4 = params.qbmmParams.batchB2 * mul.batchB3B4;
        mul.batchA3A4 = static_cast<uint64_t>(params.qbmmParams.batchA3) * params.qbmmParams.batchA4;
        mul.batchA2A3A4 = params.qbmmParams.batchA2 * mul.batchA3A4;
        mul.multiA1C1 = params.qbmmParams.batchA1 / params.qbmmParams.batchC1;
        mul.multiA2C2 = params.qbmmParams.batchA2 / params.qbmmParams.batchC2;
        mul.multiA3C3 = params.qbmmParams.batchA3 / params.qbmmParams.batchC3;
        mul.multiA4C4 = params.qbmmParams.batchA4 / params.qbmmParams.batchC4;
        mul.multiB1C1 = params.qbmmParams.batchB1 / params.qbmmParams.batchC1;
        mul.multiB2C2 = params.qbmmParams.batchB2 / params.qbmmParams.batchC2;
        mul.multiB3C3 = params.qbmmParams.batchB3 / params.qbmmParams.batchC3;
        mul.multiB4C4 = params.qbmmParams.batchB4 / params.qbmmParams.batchC4;
        return mul;
    }

    // Process one block on AIC(cube) and AIV(dequant), keeping ProcessSingleBatch compact.
    template <class GmTensorA, class GmTensorB, class GmTensorBias>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorBias& gmBlockBias,
                                           const BlockShape& singleShape, int64_t mPos, int64_t nPos, int64_t curM,
                                           int64_t curN, int64_t k, int64_t m, int64_t n, int64_t l0cUbBaseOffset)
    {
        constexpr int64_t kPos = 0;
        if ASCEND_IS_AIC {
            if (!isFirstBlock_) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }

            auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(curM, k));
            auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, curN));

            // DATA_BLOCK=32 matches BlockEpilogueDequant::DATA_BLOCK.
            constexpr int64_t l0cAlign = BLOCK_BYTE_SIZE / sizeof(L0CType);
            const int64_t curNAligned = Blaze::Gemm::CeilAlign(curN, l0cAlign);
            const int64_t curMAligned = Blaze::Gemm::CeilAlign(curM, static_cast<int64_t>(2));
            auto layoutUbC = asc::te::make_frame_layout<asc::te::nd_layout_ptn>(curMAligned, curNAligned);
            auto ubC = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(l0cUbBaseOffset * sizeof(L0CType)), layoutUbC);
            mmadOp_(gmBlockA, gmBlockB, ubC, singleShape, gmBlockBias);
            Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
            isFirstBlock_ = false;
        }
        if ASCEND_IS_AIV {
            Sync::WaitForCube();
            int64_t offsetScale = nPos;
            int64_t offsetPtScale = mPos;
            int64_t offsetBias = nPos;
            int64_t offsetC = batchCOffset_ * m * n + mPos * n + nPos;
            if (isBiasThreeDim_) {
                offsetBias += batchCOffset_ * n;
            }
            epilogueOp_(curM, curN, offsetScale, offsetPtScale, offsetBias, offsetC, l0cUbBaseOffset);
            Sync::NotifyCube();
        }
    }

    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    __gm__ AType* aGmBase_{nullptr};
    __gm__ BType* bGmBase_{nullptr};
    bool isBiasThreeDim_{false};
    bool isFirstBlock_{true};
    bool needUpdateTail_{false};
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
};

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    BlockScheduler bs(params.problemShape, params.schParams);

    if ASCEND_IS_AIC {
        mmadOp_.Init(params.mmadParams);
    }
    if ASCEND_IS_AIV {
        epilogueOp_.Init(params.epilogueParams);
    }

    if (asc::te::get<MNK_B>(params.problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
    } else {
        ProcessWithBatch(params, bs);
    }

    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
        }
    }
}

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    if (params.qbmmParams.isBias == 1 && params.qbmmParams.biasThreeDim == 1) {
        isBiasThreeDim_ = true;
    }
    if ASCEND_IS_AIC {
        aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    }
}

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIC {
        aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    }
}

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    if ASCEND_IS_AIC {
        aGmBase_ += batchAOffset_ * asc::te::get<MNK_M>(params.problemShape) * asc::te::get<MNK_K>(params.problemShape);
        if constexpr (WEIGHT_NZ) {
            if constexpr (TRANS_B) {
                bGmBase_ += batchBOffset_ * Blaze::Gemm::CeilDiv(asc::te::get<MNK_K>(params.problemShape), C0_SIZE) *
                            Blaze::Gemm::CeilDiv(asc::te::get<MNK_N>(params.problemShape),
                                                 static_cast<int64_t>(BLOCK_CUBE)) *
                            BLOCK_CUBE * C0_SIZE;
            } else {
                bGmBase_ += batchBOffset_ * Blaze::Gemm::CeilDiv(asc::te::get<MNK_N>(params.problemShape), C0_SIZE) *
                            Blaze::Gemm::CeilDiv(asc::te::get<MNK_K>(params.problemShape),
                                                 static_cast<int64_t>(BLOCK_CUBE)) *
                            BLOCK_CUBE * C0_SIZE;
            }
        } else {
            bGmBase_ += batchBOffset_ * asc::te::get<MNK_N>(params.problemShape) *
                        asc::te::get<MNK_K>(params.problemShape);
        }
    }
}

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params& params,
                                                                                   BlockScheduler& bs)
{
    if (params.qbmmParams.batchC1 == 0 || params.qbmmParams.batchC2 == 0 || params.qbmmParams.batchC3 == 0 ||
        params.qbmmParams.batchC4 == 0) {
        return;
    }
    BatchMultipliers mul = ComputeBatchMultipliers(params);

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    const uint64_t totalCnt = bs.GetTotalCnt() * asc::te::get<MNK_B>(params.problemShape);
    const uint64_t nonTailRoundCnt = (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < params.qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < params.qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < params.qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < params.qbmmParams.batchC4; ++b4Index) {
                    const bool isTailRound = curBatchC * bs.GetTotalCnt() > nonTailRoundCnt;
                    AddBatchOffset(params);
                    ProcessSingleBatch(params, bs, asc::te::get<MNK_B>(params.problemShape) - curBatchC, isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += mul.multiA4C4;
                    batchBOffset_ += mul.multiB4C4;
                }
                batchC3Offset += params.qbmmParams.batchC4;
                batchA3Offset += params.qbmmParams.batchA4 * static_cast<uint64_t>(mul.multiA3C3);
                batchB3Offset += params.qbmmParams.batchB4 * static_cast<uint64_t>(mul.multiB3C3);
            }
            batchC2Offset += mul.batchC3C4;
            batchA2Offset += mul.batchA3A4 * mul.multiA2C2;
            batchB2Offset += mul.batchB3B4 * mul.multiB2C2;
        }
        batchC1Offset += mul.batchC2C3C4;
        batchA1Offset += mul.batchA2A3A4 * mul.multiA1C1;
        batchB1Offset += mul.batchB2B3B4 * mul.multiB1C1;
    }
}

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(const Params& params,
                                                                                     BlockScheduler& bs,
                                                                                     uint64_t restBatch,
                                                                                     bool isTailRound)
{
    const int64_t m = asc::te::get<MNK_M>(params.problemShape);
    const int64_t n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t k = asc::te::get<MNK_K>(params.problemShape);

    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmBase_), layoutA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmBase_), layoutB);
    auto biasGmBase = reinterpret_cast<__gm__ int32_t*>(params.mmadParams.biasGmAddr);
    if (params.mmadParams.isBias && isBiasThreeDim_) {
        biasGmBase += batchCOffset_ * n;
    }
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<int32_t>>(1L,
                                                                                                                     n);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmBase), layoutBias);

    if (needUpdateTail_ || (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) *
                                                   params.schParams.mTailTile * params.schParams.nTailTile <=
                                               AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    }

    BlockCoord blockCoord;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::DEFAULT, QuantMode::DEFAULT, WEIGHT_NZ>(
            blockCoord);
        if (asc::te::get<IDX_M_TILEIDX>(singleShape) <= 0 || asc::te::get<IDX_N_TILEIDX>(singleShape) <= 0) {
            break;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        const int64_t curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const int64_t curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        const int64_t l0cUbBaseOffset = 0;
        if ASCEND_IS_AIC {
            SetBL2Cache(params.problemShape, curM, curN, params.qbmmParams.bMustHitL2, gmB);
        }
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, params.mmadParams.isBias ? nPos : 0L),
                                        asc::te::make_shape(1L, curN));
        ProcessOneBlock(gmA, gmB, gmBlockBias, singleShape, mPos, nPos, curM, curN, k, m, n, l0cUbBaseOffset);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
