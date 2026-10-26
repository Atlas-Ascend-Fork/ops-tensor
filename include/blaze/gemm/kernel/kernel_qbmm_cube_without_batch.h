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
 * \file kernel_qbmm_cube_without_batch.h
 * \brief Quantized matmul cube kernel without batch (A8W8 fixpipe, Tensor API)
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS                                                      \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                            \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleFixpipeQuantWithoutBatch, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS> {
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
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    static constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
    static constexpr uint32_t LEFT_SHIFT_16 = 16;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockScheduler& bs);

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
    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ X2ScaleType* scaleGmAddr_{nullptr};
    bool isBias_{false};
    uint64_t scaleScalar_{0};
};

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }

    Init(params);
    const QBMMTiling& qbmmParams = params.qbmmParams;
    BlockScheduler bs(params.problemShape, params.schParams);
    BlockMmadParams blockMmadParams = params.mmadParams;
    blockMmadParams.oriK = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
    blockMmadParams.kAL1 = qbmmParams.kAL1;
    blockMmadParams.kBL1 = qbmmParams.kBL1;
    blockMmadParams.l1BufNum = qbmmParams.nBufferNum;
    blockMmadParams.mL0 = qbmmParams.baseM;
    blockMmadParams.nL0 = qbmmParams.baseN;
    blockMmadParams.kL0 = qbmmParams.baseK;
    blockMmadParams.quantMode = static_cast<QuantMode>(qbmmParams.x2QuantMode);
    blockMmadParams.isBias = isBias_;
    blockMmadParams.enableL0cPingPong = qbmmParams.dbL0C > 1;
    mmadOp_.Init(blockMmadParams);

    Process(params, bs);
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Init(const Params& params)
{
    const QBMMTiling& qbmmParams = params.qbmmParams;
    isBias_ = qbmmParams.isBias != 0U;
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
    if (static_cast<QuantMode>(qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE) {
        scaleGmAddr_ = reinterpret_cast<__gm__ X2ScaleType*>(params.mmadParams.scaleBGmAddr);
    } else if (static_cast<QuantMode>(qbmmParams.x1QuantMode) == QuantMode::PERTENSOR_MODE) {
        float deqScale = ReadGmScalar<float>(params.mmadParams.scaleAGmAddr) *
                         ReadGmScalar<float>(params.mmadParams.scaleBGmAddr);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
    } else if (static_cast<QuantMode>(qbmmParams.x2QuantMode) == QuantMode::PERTENSOR_MODE) {
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

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::SetBL2Cache(
    const ProblemShape& problemShape, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN, uint32_t bMustHitL2,
    TensorB& gmB)
{
    // 0x7f: 128-element alignment for 128-byte B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                 isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <typename TensorA, typename TensorB, typename TensorC, typename TensorBias, typename TensorScale>
__aicore__ inline void GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::ProcessOneBlock(
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

QBMM_CUBE_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_CUBE_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Process(const Params& params,
                                                                                         BlockScheduler& bs)
{
    const int64_t m = asc::te::get<MNK_M>(params.problemShape);
    const int64_t n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t k = asc::te::get<MNK_K>(params.problemShape);
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutC = MakeLayoutC{}(m, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    __gm__ BiasType* biasPtr = isBias_ ? biasGmAddr_ : reinterpret_cast<__gm__ BiasType*>(cGmAddr_);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);
    auto layoutScale = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                  asc::te::layout_trait_default<X2ScaleType>>(1, n);
    auto gmScale = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleGmAddr_), layoutScale);
    const bool isPerChannel = static_cast<QuantMode>(params.qbmmParams.x2QuantMode) == QuantMode::PERCHANNEL_MODE;

    if ((bs.GetEndBlockIdx() + 1) * params.schParams.mTailTile * params.schParams.nTailTile <= AscendC::GetBlockNum()) {
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
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
