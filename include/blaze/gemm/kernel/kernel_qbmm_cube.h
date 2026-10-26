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
 * \file kernel_qbmm_cube.h
 * \brief Quantized batch matmul cube kernel (A8W8 fixpipe, Tensor API)
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"
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

#define QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_TEM_PARAMS                                                        \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleFixpipeQuant, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

#define QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_FUNC_TEMPLATE_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using X2ScaleType = uint64_t;
    using ScaleGmType = typename BlockMmad::X2ScaleType;

private:
    using InputTypePair = AscendC::Std::tuple<AType, BType>;
    using OutputBiasTypePair = AscendC::Std::tuple<CType, BiasType>;
    using Int8InputTypePair = AscendC::Std::tuple<int8_t, int8_t>;
    using HiFloat8InputTypePair = AscendC::Std::tuple<hifloat8_t, hifloat8_t>;

    static constexpr bool IS_INT8_INPUT = AscendC::Std::is_same_v<InputTypePair, Int8InputTypePair>;
    static constexpr bool IS_HIFLOAT8_INPUT = AscendC::Std::is_same_v<InputTypePair, HiFloat8InputTypePair>;
    static constexpr bool IS_FP8_INPUT = AscendC::Std::is_one_of_v<AType, fp8_e4m3fn_t, fp8_e5m2_t> &&
                                         AscendC::Std::is_one_of_v<BType, fp8_e4m3fn_t, fp8_e5m2_t>;

    static_assert(IS_INT8_INPUT || IS_HIFLOAT8_INPUT || IS_FP8_INPUT,
                  "QBMM Cube only supports int8_t A/B, hifloat8_t A/B, or FP8 A/B combinations.");
    static_assert(
        !IS_INT8_INPUT ||
            AscendC::Std::is_one_of_v<OutputBiasTypePair, AscendC::Std::tuple<half, int32_t>,
                                      AscendC::Std::tuple<bfloat16_t, int32_t>, AscendC::Std::tuple<int8_t, int32_t>,
                                      AscendC::Std::tuple<int32_t, int32_t>>,
        "QBMM Cube requires half/bfloat16_t/int8_t/int32_t CType and int32_t BiasType for int8_t A/B.");
    static_assert(
        !(IS_HIFLOAT8_INPUT || IS_FP8_INPUT) ||
            AscendC::Std::is_one_of_v<OutputBiasTypePair, AscendC::Std::tuple<half, float>,
                                      AscendC::Std::tuple<bfloat16_t, float>, AscendC::Std::tuple<float, float>>,
        "QBMM Cube requires half/bfloat16_t/float CType and float BiasType for HiFloat8/FP8 A/B.");
    static_assert(AscendC::Std::is_one_of_v<ScaleGmType, uint64_t, int64_t, bfloat16_t, float>,
                  "QBMM Cube only supports uint64_t/int64_t/bfloat16_t/float ScaleGmType.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM Cube only supports ND/DN LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QBMM Cube: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");
    static_assert(AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn>, "QBMM Cube only supports ND LayoutC.");

public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;

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
    };

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);

    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    static constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
    static constexpr uint32_t LEFT_SHIFT_16 = 16;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    struct BatchBroadcastInfo {
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

    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void AddBatchOffset(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint64_t batchCnt,
                                              bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline BatchBroadcastInfo ComputeBatchBroadcastInfo(const QBMMTiling& qbmmParams) const;

    template <typename TensorA, typename TensorB, typename TensorC, typename TensorBias, typename TensorScale>
    __aicore__ inline void ProcessOneBlock(TensorA& gmA, TensorB& gmB, TensorC& gmC, TensorBias& gmBias,
                                           TensorScale& gmScale, const BlockShape& singleShape, int64_t mPos,
                                           int64_t nPos, int64_t curM, int64_t curN, int64_t k, bool isPerChannel);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

    template <typename T>
    __aicore__ inline T ReadGmScalar(GM_ADDR gmAddr)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1L, 1L);
        auto tensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ T*>(gmAddr)), layout);
        return tensor[asc::te::make_coord(0L, 0L)];
    }

    BlockMmad mmadOp_;
    __gm__ AType* aGmBase_{nullptr};
    __gm__ BType* bGmBase_{nullptr};
    __gm__ CType* cGmBase_{nullptr};
    __gm__ BiasType* biasGmBase_{nullptr};
    __gm__ X2ScaleType* scaleGmBase_{nullptr};
    bool isBias_{false};
    bool isBiasThreeDim_{false};
    uint64_t scaleScalar_{0};
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool needUpdateTail_{false};
};

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    Init(params);
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockMmadParams blockMmadParams = params.mmadParams;
    blockMmadParams.oriK = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
    blockMmadParams.kAL1 = params.qbmmParams.kAL1;
    blockMmadParams.kBL1 = params.qbmmParams.kBL1;
    blockMmadParams.l1BufNum = params.qbmmParams.nBufferNum;
    blockMmadParams.mL0 = params.qbmmParams.baseM;
    blockMmadParams.nL0 = params.qbmmParams.baseN;
    blockMmadParams.kL0 = params.qbmmParams.baseK;
    blockMmadParams.quantMode = static_cast<QuantMode>(params.qbmmParams.x2QuantMode);
    blockMmadParams.isBias = params.qbmmParams.isBias != 0U;
    blockMmadParams.enableL0cPingPong = params.qbmmParams.dbL0C > 1;
    mmadOp_.Init(blockMmadParams);

    if (asc::te::get<MNK_B>(params.problemShape) == 1) {
        AddBatchOffset(params);
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

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias != 0U) {
        isBias_ = true;
        biasGmBase_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
    }
    aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmBase_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE) {
        scaleGmBase_ = reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.scaleBGmAddr);
    } else if (static_cast<QuantMode>(params.qbmmParams.x1QuantMode) == QuantMode::PERTENSOR_MODE) {
        float deqScale = ReadGmScalar<float>(params.mmadParams.scaleAGmAddr) *
                         ReadGmScalar<float>(params.mmadParams.scaleBGmAddr);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
    } else if (static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERTENSOR_MODE) {
        if constexpr (AscendC::IsSameType<ScaleGmType, uint64_t>::value ||
                      AscendC::IsSameType<ScaleGmType, int64_t>::value) {
            scaleScalar_ = ReadGmScalar<uint64_t>(params.mmadParams.scaleBGmAddr);
        } else if constexpr (AscendC::IsSameType<ScaleGmType, bfloat16_t>::value) {
            uint16_t uint16Scale = ReadGmScalar<uint16_t>(params.mmadParams.scaleBGmAddr);
            uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << LEFT_SHIFT_16);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        } else {
            uint32_t uint32Scale = ReadGmScalar<uint32_t>(params.mmadParams.scaleBGmAddr);
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        }
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmBase_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmBase_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmBase_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (isBias_) {
        biasGmBase_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
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
        bGmBase_ += batchBOffset_ * asc::te::get<MNK_N>(params.problemShape) * asc::te::get<MNK_K>(params.problemShape);
    }
    cGmBase_ += batchCOffset_ * asc::te::get<MNK_M>(params.problemShape) * asc::te::get<MNK_N>(params.problemShape);
    if (isBiasThreeDim_) {
        biasGmBase_ += batchCOffset_ * asc::te::get<MNK_N>(params.problemShape);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline auto GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ComputeBatchBroadcastInfo(
    const QBMMTiling& qbmmParams) const -> BatchBroadcastInfo
{
    BatchBroadcastInfo info;
    info.batchC3C4 = static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    info.batchC2C3C4 = qbmmParams.batchC2 * info.batchC3C4;
    info.batchB3B4 = static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    info.batchB2B3B4 = qbmmParams.batchB2 * info.batchB3B4;
    info.batchA3A4 = static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    info.batchA2A3A4 = qbmmParams.batchA2 * info.batchA3A4;
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

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params& params,
                                                                                    BlockScheduler& bs)
{
    const QBMMTiling& qbmmParams = params.qbmmParams;
    const BatchBroadcastInfo info = ComputeBatchBroadcastInfo(qbmmParams);
    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    const uint64_t totalCnt = bs.GetTotalCnt() * asc::te::get<MNK_B>(params.problemShape);
    const uint64_t nonTailRoundCnt = (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < qbmmParams.batchC4; ++b4Index) {
                    const bool isTailRound = curBatchC * bs.GetTotalCnt() > nonTailRoundCnt;
                    AddBatchOffset(params);
                    ProcessSingleBatch(params, bs, (asc::te::get<MNK_B>(params.problemShape) - curBatchC), isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += info.multiA4C4;
                    batchBOffset_ += info.multiB4C4;
                }
                batchC3Offset += qbmmParams.batchC4;
                batchA3Offset += qbmmParams.batchA4 * static_cast<uint64_t>(info.multiA3C3);
                batchB3Offset += qbmmParams.batchB4 * static_cast<uint64_t>(info.multiB3C3);
            }
            batchC2Offset += info.batchC3C4;
            batchA2Offset += info.batchA3A4 * info.multiA2C2;
            batchB2Offset += info.batchB3B4 * info.multiB2C2;
        }
        batchC1Offset += info.batchC2C3C4;
        batchA1Offset += info.batchA2A3A4 * info.multiA1C1;
        batchB1Offset += info.batchB2B3B4 * info.multiB1C1;
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::SetBL2Cache(const ProblemShape& problemShape,
                                                                               uint64_t currentBasicBlockM,
                                                                               uint64_t currentBasicBlockN,
                                                                               uint32_t bMustHitL2, TensorB& gmB)
{
    // 0x7f: 128-element alignment for 128-byte B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                 isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
template <typename TensorA, typename TensorB, typename TensorC, typename TensorBias, typename TensorScale>
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ProcessOneBlock(
    TensorA& gmA, TensorB& gmB, TensorC& gmC, TensorBias& gmBias, TensorScale& gmScale, const BlockShape& singleShape,
    int64_t mPos, int64_t nPos, int64_t curM, int64_t curN, int64_t k, bool isPerChannel)
{
    constexpr int64_t kPos = 0L;
    auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(curM, k));
    auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, curN));
    auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(curM, curN));
    const int64_t biasNPos = isBias_ ? nPos : 0L;
    const int64_t biasNSize = isBias_ ? curN : 1L;
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, biasNPos), asc::te::make_shape(1L, biasNSize));
    if (isPerChannel) {
        auto gmBlockScale = gmScale.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, curN));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScale, gmBlockBias, gmBlockC, singleShape);
    } else {
        mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, singleShape);
    }
}

QBMM_CUBE_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_KERNEL_TEM_PARAMS>::ProcessSingleBatch(const Params& params,
                                                                                      BlockScheduler& bs,
                                                                                      uint64_t restBatch,
                                                                                      bool isTailRound)
{
    const int64_t m = asc::te::get<MNK_M>(params.problemShape);
    const int64_t n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t k = asc::te::get<MNK_K>(params.problemShape);
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutC = MakeLayoutC{}(m, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmBase_), layoutA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmBase_), layoutB);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmBase_), layoutC);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    __gm__ BiasType* biasPtr = isBias_ ? biasGmBase_ : reinterpret_cast<__gm__ BiasType*>(cGmBase_);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);
    auto layoutScale = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                  asc::te::layout_trait_default<X2ScaleType>>(1, n);
    auto gmScale = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleGmBase_), layoutScale);
    const bool isPerChannel = static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE;

    const bool canSplitTail = ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) *
                                  params.schParams.mTailTile * params.schParams.nTailTile <=
                              AscendC::GetBlockNum();
    if (needUpdateTail_ || (isTailRound && canSplitTail)) {
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
        SetBL2Cache(params.problemShape, curM, curN, params.qbmmParams.bMustHitL2, gmB);
        ProcessOneBlock(gmA, gmB, gmC, gmBias, gmScale, singleShape, mPos, nPos, curM, curN, k, isPerChannel);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
