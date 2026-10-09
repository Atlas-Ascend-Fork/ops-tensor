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
 * \file kernel_qbmm_mx_activation_quant.h
 * \brief
 */

#pragma once

#include "kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_KERNEL_TEM_PARAMS                                                               \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                     \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleMxActivationQuant, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;

    static_assert((IsFp8<AType>() && IsFp8<BType>()) || (IsFp4<AType>() && IsFp4<BType>()),
                  "QBMM MX Activation Quant: AType/BType must each be fp8_e4m3fn_t/fp8_e5m2_t, or each be "
                  "fp4x2_e2m1_t/fp4x2_e1m2_t.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX Activation Quant: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(
        AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                  asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
        "QBMM MX Activation Quant: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");

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
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint64_t restBatch,
                                              bool isTailRound);
    __aicore__ inline void ProcessTileLoop(const Params& params, BlockScheduler& bs);
    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class GmTensorC, class UbMemPtr>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos,
                                           int64_t nPos, int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen,
                                           int64_t n, const UbMemPtr& ubmemPtr);

    struct BatchStrideInfo {
        uint64_t aBatchElementStride;
        uint64_t bBatchElementStride;
        uint64_t cBatchStride;
        uint64_t biasBatchStride;
        uint64_t scaleABatchStride;
        uint64_t scaleBBatchStride;
        uint64_t batchC2C3C4;
        uint64_t batchB2B3B4;
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
    __aicore__ inline BatchStrideInfo CalcBatchStrides(const Params& params);
    __aicore__ inline void ProcessBatchLoop(const Params& params, BlockScheduler& bs, const BatchStrideInfo& info);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void AddBatchOffset(const Params& params, uint64_t aBatchElementStride,
                                          uint64_t bBatchElementStride, uint64_t cBatchStride,
                                          uint64_t scaleABatchStride, uint64_t scaleBBatchStride,
                                          uint64_t biasBatchStride);

    template <typename TensorB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, uint64_t baseM, uint64_t baseN, TensorB& gmB,
                                      TensorC& gmC);

    __aicore__ inline void End();

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    bool isVecSetSyncCom_ = false;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isSameBatch_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    Init(params);
    BlockScheduler bs(problemShape, params.schParams);

    const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, qbmmParams.dbL0C > 1);
    epilogueOp_.Init(params.epilogueParams);
    epilogueOp_.UpdateNextProblem(problemShape);

    if (asc::te::get<MNK_B>(problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
        End();
        return;
    }

    ProcessWithBatch(params, bs);
    End();
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetL2Cache(const ProblemShape& problemShape,
                                                                            uint64_t baseM, uint64_t baseN,
                                                                            TensorB& gmB, TensorC& gmC)
{
    if constexpr (IS_ATOMIC_ADD) {
        gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
    }

    const bool fullMBlock = baseM >= asc::te::get<MNK_M>(problemShape);
    if (!(isSameBatch_ && fullMBlock)) {
        return;
    }

    if constexpr (WEIGHT_NZ) {
        gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
    } else {
        constexpr int64_t cacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
        // 0xff: 256 cache line alignment for FP4 weight GM streaming
        // 0x7f: 128 cache line alignment for FP8 weight GM streaming
        if constexpr (TRANS_B) {
            bool bAlignForL2Stream = (asc::te::get<MNK_K>(problemShape) & cacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        } else {
            bool bAlignForL2Stream = (asc::te::get<MNK_N>(problemShape) & cacheLineAlignMask) == 0 &&
                                     (baseN & cacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    const auto& qbmmParams = params.qbmmParams;
    if (qbmmParams.isBias == 1) {
        if (qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
    }
    if (qbmmParams.batchA1 == qbmmParams.batchB1 && qbmmParams.batchA2 == qbmmParams.batchB2 &&
        qbmmParams.batchA3 == qbmmParams.batchB3 && qbmmParams.batchA4 == qbmmParams.batchB4) {
        isSameBatch_ = true;
    }
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline auto GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::CalcBatchStrides(const Params& params)
    -> BatchStrideInfo
{
    const auto& qbmmParams = params.qbmmParams;
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    const auto k = asc::te::get<MNK_K>(params.problemShape);

    BatchStrideInfo info{};
    info.aBatchElementStride = m * k;
    if constexpr (WEIGHT_NZ) {
        if constexpr (TRANS_B) {
            info.bBatchElementStride = Blaze::Gemm::CeilDiv(k, C0_SIZE) *
                                       Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE)) * BLOCK_CUBE * C0_SIZE;
        } else {
            info.bBatchElementStride = Blaze::Gemm::CeilDiv(n, C0_SIZE) *
                                       Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE)) * BLOCK_CUBE * C0_SIZE;
        }
    } else {
        info.bBatchElementStride = n * k;
    }
    info.cBatchStride = m * n;
    info.biasBatchStride = isBiasThreeDim_ ? n : 0;
    const uint64_t scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    info.scaleABatchStride = m * scaleKLen;
    info.scaleBBatchStride = n * scaleKLen;
    info.batchC2C3C4 = qbmmParams.batchC2 * static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    info.batchB2B3B4 = qbmmParams.batchB2 * static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    info.batchA2A3A4 = qbmmParams.batchA2 * static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    info.multiA1C1 = qbmmParams.batchA1 / qbmmParams.batchC1;
    info.multiA2C2 = qbmmParams.batchA2 / qbmmParams.batchC2;
    info.multiA3C3 = qbmmParams.batchA3 / qbmmParams.batchC3;
    info.multiA4C4 = qbmmParams.batchA4 / qbmmParams.batchC4;
    info.multiB1C1 = qbmmParams.batchB1 / qbmmParams.batchC1;
    info.multiB2C2 = qbmmParams.batchB2 / qbmmParams.batchC2;
    info.multiB3C3 = qbmmParams.batchB3 / qbmmParams.batchC3;
    info.multiB4C4 = qbmmParams.batchB4 / qbmmParams.batchC4;
    return info;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessBatchLoop(const Params& params,
                                                                                  BlockScheduler& bs,
                                                                                  const BatchStrideInfo& info)
{
    const auto& qbmmParams = params.qbmmParams;
    const uint64_t batchC3C4 = static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    const uint64_t singleBatchBlockCnt = bs.GetTotalCnt();
    const uint64_t batchCount = asc::te::get<MNK_B>(params.problemShape);
    const uint64_t tailRoundStart = (singleBatchBlockCnt * batchCount / AscendC::GetBlockNum()) *
                                    AscendC::GetBlockNum();

    uint64_t batchC1Offset = 0, batchA1Offset = 0, batchB1Offset = 0, curBatchC = 1UL;
    for (uint64_t b1 = 0; b1 < qbmmParams.batchC1; ++b1) {
        uint64_t c2 = batchC1Offset, a2 = batchA1Offset, b2 = batchB1Offset;
        for (uint64_t b2i = 0; b2i < qbmmParams.batchC2; ++b2i) {
            uint64_t c3 = c2, a3 = a2, b3 = b2;
            for (uint64_t b3i = 0; b3i < qbmmParams.batchC3; ++b3i) {
                batchCOffset_ = c3;
                batchAOffset_ = a3;
                batchBOffset_ = b3;
                for (uint64_t b4 = 0; b4 < qbmmParams.batchC4; ++b4) {
                    bool isTailRound = curBatchC * singleBatchBlockCnt > tailRoundStart;
                    AddBatchOffset(params, info.aBatchElementStride, info.bBatchElementStride, info.cBatchStride,
                                   info.scaleABatchStride, info.scaleBBatchStride, info.biasBatchStride);
                    ProcessSingleBatch(params, bs, batchCount - curBatchC, isTailRound);
                    curBatchC++;
                    batchCOffset_++;
                    batchAOffset_ += info.multiA4C4;
                    batchBOffset_ += info.multiB4C4;
                }
                c3 += qbmmParams.batchC4;
                a3 += qbmmParams.batchA4 * info.multiA3C3;
                b3 += qbmmParams.batchB4 * info.multiB3C3;
            }
            c2 += batchC3C4;
            a2 += batchA3A4 * info.multiA2C2;
            b2 += batchB3B4 * info.multiB2C2;
        }
        batchC1Offset += info.batchC2C3C4;
        batchA1Offset += info.batchA2A3A4 * info.multiA1C1;
        batchB1Offset += info.batchB2B3B4 * info.multiB1C1;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params& params,
                                                                                  BlockScheduler& bs)
{
    BatchStrideInfo info = CalcBatchStrides(params);
    ProcessBatchLoop(params, bs, info);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::AddBatchOffset(
    const Params& params, uint64_t aBatchElementStride, uint64_t bBatchElementStride, uint64_t cBatchStride,
    uint64_t scaleABatchStride, uint64_t scaleBBatchStride, uint64_t biasBatchStride)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    aGmAddr_ += (batchAOffset_ * aBatchElementStride) >> sizeShift;
    bGmAddr_ += (batchBOffset_ * bBatchElementStride) >> sizeShift;
    cGmAddr_ += batchCOffset_ * cBatchStride;
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * biasBatchStride;
    }
    scaleAGmAddr_ += batchAOffset_ * scaleABatchStride;
    scaleBGmAddr_ += batchBOffset_ * scaleBBatchStride;
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t scaleN = CeilDiv(n, BLOCK_SIZE * ALIGN_NUM_2) * ALIGN_NUM_2;
    epilogueOp_.UpdateGlobalAddr({batchCOffset_ * m * n >> sizeShift, batchCOffset_ * m * scaleN, 0, 0, 0});
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        if (isVecSetSyncCom_) {
            Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
        }
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class GmTensorC, class UbMemPtr>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessOneBlock(
    const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
    const GmTensorBias& gmBias, const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
    int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen, int64_t n, const UbMemPtr& ubmemPtr)
{
    constexpr int64_t kPos = 0L;
    auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
    auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
    auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, baseN));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
    auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(baseM, baseN));
    auto locOutUb = asc::te::make_tensor(
        ubmemPtr, asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>((baseM + 1) & ~1, Align32(baseN)));
    if ASCEND_IS_AIC {
        if (isVecSetSyncCom_) {
            Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
        }
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, locOutUb, singleShape);
        Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
    }
    isVecSetSyncCom_ = true;
    if ASCEND_IS_AIV {
        Sync::WaitForCube();
        epilogueOp_({baseM, baseN, 0, 0},
                    {mPos * n + nPos,
                     mPos * CeilDiv(n, BLOCK_SIZE * ALIGN_NUM_2) * ALIGN_NUM_2 + CeilDiv(nPos, BLOCK_SIZE), 0, 0, 0});
        Sync::NotifyCube();
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessTileLoop(const Params& params,
                                                                                 BlockScheduler& bs)
{
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    const auto scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    auto layoutC = MakeLayoutC{}(m, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
    auto ubmemPtr = asc::te::make_mem_ptr<asc::te::location::ub, float>(0);
    SetL2Cache(problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, gmB, gmC);

    BlockCoord blockCoord;
    int64_t mPos = 0L, nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                                           WEIGHT_NZ, 32>(blockCoord);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            if ASCEND_IS_AIC {
                Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
            }
            if ASCEND_IS_AIV {
                Sync::NotifyCube();
            }
            return;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        ProcessOneBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, singleShape, mPos, nPos, baseM, baseN, k, scaleKLen,
                        n, ubmemPtr);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(const Params& params,
                                                                                    BlockScheduler& bs,
                                                                                    uint64_t restBatch,
                                                                                    bool isTailRound)
{
    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    ProcessTileLoop(params, bs);
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
