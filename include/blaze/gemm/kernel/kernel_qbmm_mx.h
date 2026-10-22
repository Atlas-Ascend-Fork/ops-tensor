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
 * \file kernel_qbmm_mx.h
 * \brief GemmUniversal specialization for QBMM MX with batch iteration and L2 cache control.
 *
 * Orchestrates BlockMmad and BlockScheduler for quantized batch matmul.
 * Iterates over batch dimension (single-batch fast path or multi-batch loop) with
 * per-batch GM address reset. Provides L2 cache hint management for the B matrix
 * and optional atomic-add for split-K accumulation.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_KERNEL_TEM_PARAMS                           \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, \
        AscendC::Std::enable_if_t<                          \
            AscendC::Std::is_same_v<KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
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

    static_assert((((AscendC::Std::is_one_of_v<AType, fp8_e4m3fn_t, fp8_e5m2_t>) &&
                    (AscendC::Std::is_one_of_v<BType, fp8_e4m3fn_t, fp8_e5m2_t>)) ||
                   ((AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>) &&
                    (AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>))) &&
                      AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float> &&
                      AscendC::Std::is_same_v<BiasType, float>,
                  "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QBMM MX: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");
    // Preserve the ND/DN C Tensor layouts supported by the L0C copy path.
    // A supplied epilogue must use the same output layout.
    static_assert(AscendC::Std::is_one_of_v<LayoutC, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX: LayoutC must be nd_ext_layout_ptn/dn_ext_layout_ptn.");

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
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
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

    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint64_t restBatch,
                                              bool isTailRound);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void ProcessBatchTraversal(const Params& params, BlockScheduler& bs, uint64_t tailRoundStart);
    __aicore__ inline void ProcessBatchC4(const Params& params, BlockScheduler& bs, uint64_t& curBatchC,
                                          uint64_t tailRoundStart, uint64_t batchC3Offset, uint64_t batchA3Offset,
                                          uint64_t batchB3Offset, uint32_t multiA4C4, uint32_t multiB4C4);
    __aicore__ inline void AddBatchOffset(const Params& params);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class GmTensorC>
    __aicore__ inline bool ProcessTiles(const Params& params, BlockScheduler& bs, const GmTensorA& gmA, GmTensorB& gmB,
                                        const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
                                        const GmTensorBias& gmBias, const GmTensorC& gmC, int64_t k, int64_t scaleKLen);
    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class GmTensorC>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos,
                                           int64_t nPos, int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen);

    BlockMmad mmadOp_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    uint64_t aBatchElementStride_{0};
    uint64_t bBatchElementStride_{0};
    uint64_t cBatchStride_{0};
    uint64_t scaleABatchStride_{0};
    uint64_t scaleBBatchStride_{0};
    uint64_t biasBatchStride_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    Init(params);
    BlockScheduler bs(problemShape, params.schParams);

    const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, qbmmParams.dbL0C > 1);

    if (asc::te::get<MNK_B>(problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetBL2Cache(const ProblemShape& problemShape,
                                                                             uint64_t currentBasicBlockM,
                                                                             uint64_t currentBasicBlockN,
                                                                             uint32_t bMustHitL2, TensorB& gmB)
{
    // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
    // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                 isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
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
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params& params,
                                                                                  BlockScheduler& bs)
{
    const auto& qbmmParams = params.qbmmParams;
    if (qbmmParams.batchC1 == 0 || qbmmParams.batchC2 == 0 || qbmmParams.batchC3 == 0 || qbmmParams.batchC4 == 0) {
        return;
    }
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    aBatchElementStride_ = static_cast<uint64_t>(m) * k;
    bBatchElementStride_ = static_cast<uint64_t>(n) * k;
    if constexpr (WEIGHT_NZ && TRANS_B) {
        const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, C0_SIZE);
        const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE));
        bBatchElementStride_ = kBlockCnt * nBlockCnt * BLOCK_CUBE * C0_SIZE;
    } else if constexpr (WEIGHT_NZ) {
        const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, C0_SIZE);
        const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE));
        bBatchElementStride_ = nBlockCnt * kBlockCnt * BLOCK_CUBE * C0_SIZE;
    }
    cBatchStride_ = static_cast<uint64_t>(m) * n;
    biasBatchStride_ = isBiasThreeDim_ ? n : 0;
    const uint64_t scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    scaleABatchStride_ = static_cast<uint64_t>(m) * scaleKLen;
    scaleBBatchStride_ = static_cast<uint64_t>(n) * scaleKLen;

    const uint64_t singleBatchBlockCnt = bs.GetTotalCnt();
    const uint64_t batchCount = asc::te::get<MNK_B>(params.problemShape);
    const uint64_t tailRoundStart = (singleBatchBlockCnt * batchCount / AscendC::GetBlockNum()) *
                                    AscendC::GetBlockNum();
    ProcessBatchTraversal(params, bs, tailRoundStart);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessBatchTraversal(const Params& params,
                                                                                       BlockScheduler& bs,
                                                                                       uint64_t tailRoundStart)
{
    const auto& qbmmParams = params.qbmmParams;
    const uint64_t batchC3C4 = static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    const uint64_t batchC2C3C4 = qbmmParams.batchC2 * batchC3C4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    const uint64_t batchB2B3B4 = qbmmParams.batchB2 * batchB3B4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    const uint64_t batchA2A3A4 = qbmmParams.batchA2 * batchA3A4;
    const uint32_t multiA1C1 = qbmmParams.batchA1 / qbmmParams.batchC1;
    const uint32_t multiA2C2 = qbmmParams.batchA2 / qbmmParams.batchC2;
    const uint32_t multiA3C3 = qbmmParams.batchA3 / qbmmParams.batchC3;
    const uint32_t multiA4C4 = qbmmParams.batchA4 / qbmmParams.batchC4;
    const uint32_t multiB1C1 = qbmmParams.batchB1 / qbmmParams.batchC1;
    const uint32_t multiB2C2 = qbmmParams.batchB2 / qbmmParams.batchC2;
    const uint32_t multiB3C3 = qbmmParams.batchB3 / qbmmParams.batchC3;
    const uint32_t multiB4C4 = qbmmParams.batchB4 / qbmmParams.batchC4;
    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    for (uint64_t b1Index = 0; b1Index < qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < qbmmParams.batchC3; ++b3Index) {
                ProcessBatchC4(params, bs, curBatchC, tailRoundStart, batchC3Offset, batchA3Offset, batchB3Offset,
                               multiA4C4, multiB4C4);
                batchC3Offset += qbmmParams.batchC4;
                batchA3Offset += qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessBatchC4(
    const Params& params, BlockScheduler& bs, uint64_t& curBatchC, uint64_t tailRoundStart, uint64_t batchC3Offset,
    uint64_t batchA3Offset, uint64_t batchB3Offset, uint32_t multiA4C4, uint32_t multiB4C4)
{
    const auto& qbmmParams = params.qbmmParams;
    batchCOffset_ = batchC3Offset;
    batchAOffset_ = batchA3Offset;
    batchBOffset_ = batchB3Offset;
    const uint64_t singleBatchBlockCnt = bs.GetTotalCnt();
    const uint64_t batchCount = asc::te::get<MNK_B>(params.problemShape);
    for (uint64_t b4Index = 0; b4Index < qbmmParams.batchC4; ++b4Index) {
        const bool isTailRound = curBatchC * singleBatchBlockCnt > tailRoundStart;
        AddBatchOffset(params);
        ProcessSingleBatch(params, bs, batchCount - curBatchC, isTailRound);
        curBatchC++;
        batchCOffset_ += 1;
        batchAOffset_ += multiA4C4;
        batchBOffset_ += multiB4C4;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    aGmAddr_ += (batchAOffset_ * aBatchElementStride_) >> sizeShift;
    bGmAddr_ += (batchBOffset_ * bBatchElementStride_) >> sizeShift;
    cGmAddr_ += batchCOffset_ * cBatchStride_;
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * biasBatchStride_;
    }
    scaleAGmAddr_ += batchAOffset_ * scaleABatchStride_;
    scaleBGmAddr_ += batchBOffset_ * scaleBBatchStride_;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(const Params& params,
                                                                                    BlockScheduler& bs,
                                                                                    uint64_t restBatch,
                                                                                    bool isTailRound)
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
    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    if constexpr (IS_ATOMIC_ADD) {
        gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
    }
    if (!ProcessTiles(params, bs, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, k, scaleKLen)) {
        return;
    }
    bs.UpdateNextBatchBlockRoundParams();
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class GmTensorC>
__aicore__ inline bool GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessTiles(
    const Params& params, BlockScheduler& bs, const GmTensorA& gmA, GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
    const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias, const GmTensorC& gmC, int64_t k, int64_t scaleKLen)
{
    const auto& problemShape = params.problemShape;
    BlockCoord blockCoord;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                                           WEIGHT_NZ>(blockCoord);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            return false;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        SetBL2Cache(problemShape, baseM, baseN, params.qbmmParams.bMustHitL2, gmB);
        ProcessOneBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, singleShape, mPos, nPos, baseM, baseN, k, scaleKLen);
    }
    return true;
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class GmTensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessOneBlock(
    const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
    const GmTensorBias& gmBias, const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
    int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen)
{
    constexpr int64_t kPos = 0L;
    auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
    auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
    auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, baseN));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
    auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(baseM, baseN));
    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
