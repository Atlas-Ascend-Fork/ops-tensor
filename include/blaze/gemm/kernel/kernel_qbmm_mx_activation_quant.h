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
 * \brief QBMM MX with GELU/SwiGLU/Swish activation and MX quantization.
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
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS                                                         \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                                 \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, \
                                                          KernelMmadWithScaleMxActivationQuant>>

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
class GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

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
        uint32_t bMustHitL2 = 1;
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
    // Concat-N selects the SwiGLU path: the mmad produces [gate|linear] concatenated N
    // columns, so the scheduler/epilogue run on an N/2-wide output. The flag comes from
    // the dispatch policy (MatmulWithScaleMx<..., ConcatN=true>).
    static constexpr bool IS_SWIGLU = BlockMmad::CONCAT_N;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr uint16_t C0_SIZE_SHIFT = IsFp4<AType>() ? MXFP_DIVISOR_SHIFT : ALIGN_32_BYTES_SHIFT;
    static constexpr uint16_t BLOCK_CUBE_SHIFT = 4;
    static constexpr uint64_t MX_QUANT_GROUP_SIZE = 32;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
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
    // Process one block on AIC(cube) and AIV(dequant), keeping ProcessSingleBatch compact.
    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class UbMemPtr>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const BlockShape& singleShape, int64_t mPos, int64_t nPos, int64_t baseM,
                                           int64_t baseN, int64_t k, int64_t scaleKLen, int64_t n,
                                           const UbMemPtr& ubmemPtr);

    template <class GmTensorBias>
    __aicore__ inline auto GetBiasTile(const GmTensorBias& gmBias, int64_t nPos, int64_t baseN, int64_t n)
    {
        if constexpr (IS_SWIGLU) {
            auto biasAddr = biasGmAddr_ == nullptr ? nullptr : biasGmAddr_ + nPos;
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasAddr),
                                        MakeNDExtLayout(2, baseN, n >> 1));
        } else {
            return gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
        }
    }

    template <class UbMemPtr>
    __aicore__ inline auto GetOutputTile(int64_t baseM, int64_t baseN, const UbMemPtr& ubmemPtr)
    {
        if constexpr (IS_SWIGLU) {
            return epilogueOp_.GetConcatL0c2UbTensor(baseM, baseN);
        } else {
            return asc::te::make_tensor(
                ubmemPtr, asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>((baseM + 1) & ~1, Align32(baseN)));
        }
    }

    struct BatchStrideInfo {
        uint64_t aBatchElementStride;
        uint64_t bBatchElementStride;
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
                                          uint64_t bBatchElementStride, uint64_t scaleABatchStride,
                                          uint64_t scaleBBatchStride, uint64_t biasBatchStride);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_{nullptr};
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_{nullptr};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isFirstBlock_{true};
    bool needUpdateTail_{false};
};

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::Run(const Params& params)
{
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    Init(params);
    ProblemShape logicalSchedulerShape = problemShape;
    BlockSchedulerParams logicalSchedulerParams = params.schParams;
    if constexpr (IS_SWIGLU) {
        // Schedule output columns H=N/2; BlockMmad still reserves the full-N tile. The host has already
        // selected M-only tail splitting and normalized all N-tail fields for this logical scheduler.
        logicalSchedulerShape = ProblemShape{asc::te::get<MNK_M>(problemShape), asc::te::get<MNK_N>(problemShape) >> 1,
                                             asc::te::get<MNK_K>(problemShape), asc::te::get<MNK_B>(problemShape)};
        logicalSchedulerParams.baseN >>= 1;
    }
    BlockScheduler bs(logicalSchedulerShape, logicalSchedulerParams);

    if ASCEND_IS_AIC {
        const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
        mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, qbmmParams.dbL0C > 1);
    }
    epilogueOp_.Init(params.epilogueParams);
    if ASCEND_IS_AIV {
        if constexpr (IS_SWIGLU) {
            epilogueOp_.UpdateNextProblem(typename BlockEpilogue::ProblemShape{
                asc::te::get<MNK_M>(logicalSchedulerShape), asc::te::get<MNK_N>(logicalSchedulerShape),
                asc::te::get<MNK_K>(logicalSchedulerShape)});
            epilogueOp_.UpdateGlobalAddr(typename BlockEpilogue::OutputOffsets{0, 0});
        } else {
            epilogueOp_.UpdateNextProblem(problemShape);
        }
    }

    if (asc::te::get<MNK_B>(problemShape) == 1) {
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

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::SetBL2Cache(
    const ProblemShape& problemShape, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN, uint32_t bMustHitL2,
    TensorB& gmB)
{
    if ASCEND_IS_AIC {
        // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
        // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
        constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
        const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
        const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                     isCurrentNAligned;
        gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    }
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::Init(const Params& params)
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

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIC {
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
        if (isBias_) {
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        }
    }
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline auto GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::CalcBatchStrides(
    const Params& params) -> BatchStrideInfo
{
    const auto& qbmmParams = params.qbmmParams;
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    const auto k = asc::te::get<MNK_K>(params.problemShape);

    BatchStrideInfo info{};
    info.aBatchElementStride = m * k;
    if constexpr (WEIGHT_NZ) {
        if constexpr (TRANS_B) {
            info.bBatchElementStride = ((k + C0_SIZE - 1) >> C0_SIZE_SHIFT) *
                                       ((n + static_cast<int64_t>(BLOCK_CUBE) - 1) >> BLOCK_CUBE_SHIFT) * BLOCK_CUBE *
                                       C0_SIZE;
        } else {
            info.bBatchElementStride = ((n + C0_SIZE - 1) >> C0_SIZE_SHIFT) *
                                       ((k + static_cast<int64_t>(BLOCK_CUBE) - 1) >> BLOCK_CUBE_SHIFT) * BLOCK_CUBE *
                                       C0_SIZE;
        }
    } else {
        info.bBatchElementStride = n * k;
    }
    info.biasBatchStride = isBiasThreeDim_ ? n : 0;
    const uint64_t scaleKLen = ((k + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                               << MXFP_MULTI_BASE_SHIFT;
    info.scaleABatchStride = m * scaleKLen;
    info.scaleBBatchStride = n * scaleKLen;
    info.batchC2C3C4 = qbmmParams.batchC2 * static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    info.batchB2B3B4 = qbmmParams.batchB2 * static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    info.batchA2A3A4 = qbmmParams.batchA2 * static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    // A valid broadcast dimension is either 1 or equal to C, so each offset multiplier is only 0 or 1.
    info.multiA1C1 = static_cast<uint32_t>(qbmmParams.batchA1 == qbmmParams.batchC1);
    info.multiA2C2 = static_cast<uint32_t>(qbmmParams.batchA2 == qbmmParams.batchC2);
    info.multiA3C3 = static_cast<uint32_t>(qbmmParams.batchA3 == qbmmParams.batchC3);
    info.multiA4C4 = static_cast<uint32_t>(qbmmParams.batchA4 == qbmmParams.batchC4);
    info.multiB1C1 = static_cast<uint32_t>(qbmmParams.batchB1 == qbmmParams.batchC1);
    info.multiB2C2 = static_cast<uint32_t>(qbmmParams.batchB2 == qbmmParams.batchC2);
    info.multiB3C3 = static_cast<uint32_t>(qbmmParams.batchB3 == qbmmParams.batchC3);
    info.multiB4C4 = static_cast<uint32_t>(qbmmParams.batchB4 == qbmmParams.batchC4);
    return info;
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::ProcessBatchLoop(
    const Params& params, BlockScheduler& bs, const BatchStrideInfo& info)
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
                    AddBatchOffset(params, info.aBatchElementStride, info.bBatchElementStride, info.scaleABatchStride,
                                   info.scaleBBatchStride, info.biasBatchStride);
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

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::ProcessWithBatch(
    const Params& params, BlockScheduler& bs)
{
    BatchStrideInfo info = CalcBatchStrides(params);
    ProcessBatchLoop(params, bs, info);
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::AddBatchOffset(
    const Params& params, uint64_t aBatchElementStride, uint64_t bBatchElementStride, uint64_t scaleABatchStride,
    uint64_t scaleBBatchStride, uint64_t biasBatchStride)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    if ASCEND_IS_AIC {
        aGmAddr_ += (batchAOffset_ * aBatchElementStride) >> sizeShift;
        bGmAddr_ += (batchBOffset_ * bBatchElementStride) >> sizeShift;
        if (isBiasThreeDim_) {
            biasGmAddr_ += batchCOffset_ * biasBatchStride;
        }
        scaleAGmAddr_ += batchAOffset_ * scaleABatchStride;
        scaleBGmAddr_ += batchBOffset_ * scaleBBatchStride;
    }
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    if ASCEND_IS_AIV {
        if constexpr (IS_SWIGLU) {
            const int64_t h = n >> 1;
            const int64_t scaleH = ((h + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                                   << MXFP_MULTI_BASE_SHIFT;
            epilogueOp_.UpdateGlobalAddr(typename BlockEpilogue::OutputOffsets{
                static_cast<int64_t>(batchCOffset_) * m * h, static_cast<int64_t>(batchCOffset_) * m * scaleH});
        } else {
            const int64_t scaleN = ((n + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                                   << MXFP_MULTI_BASE_SHIFT;
            epilogueOp_.UpdateGlobalAddr({static_cast<int64_t>(batchCOffset_ * m * n >> sizeShift),
                                          static_cast<int64_t>(batchCOffset_ * m * scaleN)});
        }
    }
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class UbMemPtr>
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::ProcessOneBlock(
    const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
    const GmTensorBias& gmBias, const BlockShape& singleShape, int64_t mPos, int64_t nPos, int64_t baseM, int64_t baseN,
    int64_t k, int64_t scaleKLen, int64_t n, const UbMemPtr& ubmemPtr)
{
    constexpr int64_t kPos = 0L;
    if ASCEND_IS_AIC {
        if (!isFirstBlock_) {
            Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
        }
        auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
        auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
        // WeightNZ SwiGLU gathers two logical N halves with separate aligned copies.
        auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos),
                                  asc::te::make_shape(k, IS_SWIGLU && WEIGHT_NZ ? n - nPos : baseN));
        // The SwiGLU scale view includes both halves for the concat copy in BlockMmad.
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos),
                                            asc::te::make_shape(scaleKLen, IS_SWIGLU ? n - nPos : baseN));
        auto gmBlockBias = GetBiasTile(gmBias, nPos, baseN, n);
        auto locOutUb = GetOutputTile(baseM, baseN, ubmemPtr);
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, locOutUb, singleShape);
        Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
        isFirstBlock_ = false;
    }
    if ASCEND_IS_AIV {
        Sync::WaitForCube();
        if constexpr (IS_SWIGLU) {
            const int64_t h = n >> 1;
            const int64_t scaleH = ((h + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                                   << MXFP_MULTI_BASE_SHIFT;
            epilogueOp_({baseM, baseN, 0, 0},
                        typename BlockEpilogue::OutputOffsets{
                            mPos * h + nPos, mPos * scaleH + ((nPos >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT)});
        } else {
            const int64_t scaleN = ((n + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                                   << MXFP_MULTI_BASE_SHIFT;
            const int64_t scaleNPos = (nPos + BLOCK_SIZE - 1) >> ALIGN_32_BYTES_SHIFT;
            epilogueOp_({baseM, baseN, 0, 0}, {mPos * n + nPos, mPos * scaleN + scaleNPos});
        }
        if constexpr (IS_SWIGLU) {
            Sync::NotifyCube();
        }
    }
}

QUANT_MATMUL_ACTIVATION_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void GemmUniversal<QUANT_MATMUL_ACTIVATION_QUANT_TEMPLATE_ARGS>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    const auto scaleKLen = ((k + static_cast<int64_t>(MXFP_DIVISOR_SIZE) - 1) >> MXFP_DIVISOR_SHIFT)
                           << MXFP_MULTI_BASE_SHIFT;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto ubmemPtr = asc::te::make_mem_ptr<asc::te::location::ub, float>(0);

    if constexpr (!IS_SWIGLU || BlockMmad::DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
        const auto mTailTile = params.schParams.mTailTile;
        const auto nTailTile = params.schParams.nTailTile;
        if (needUpdateTail_ ||
            (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                                AscendC::GetBlockNum())) {
            needUpdateTail_ = true;
            bs.UpdateTailTile(mTailTile, nTailTile);
        }
    }

    BlockCoord blockCoord;
    int64_t mPos = 0L, nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                                           WEIGHT_NZ, MX_QUANT_GROUP_SIZE>(blockCoord);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            break;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        if ASCEND_IS_AIC {
            SetBL2Cache(problemShape, baseM, baseN, params.qbmmParams.bMustHitL2, gmB);
        }
        ProcessOneBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, mPos, nPos, baseM, baseN, k, scaleKLen, n,
                        ubmemPtr);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
