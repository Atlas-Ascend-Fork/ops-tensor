/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file qbmm_mix.h
 * \brief QBMM MIX A8W8 Kernel UT Wrapper（AIC int8 GEMM + AIV 反量化）。
 *        与 qbmm_cube.h 的 fixpipe wrapper 平级：装配 MatmulWithScaleMix / BlockEpilogueDequant /
 *        GemmUniversal（多 batch / 单 batch without_batch 特化），并填好各子 Params。
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

// MIX 路径的公共类型装配：AIC 侧 int8 GEMM（int32 L0C→UB），AIV 侧反量化。
template <typename AType, typename BType, typename OutType, typename X2ScaleType, typename X1ScaleType,
          typename BiasType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE,
          typename LayoutA_ = asc::te::nd_ext_layout_ptn, typename LayoutB_ = asc::te::nd_ext_layout_ptn,
          typename LayoutC_ = asc::te::nd_ext_layout_ptn, typename ScheduleType_ = Blaze::Gemm::KernelMmadWithScaleMix>
struct QBMMMixTypes {
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<FullLoadMode, false, ScheduleType_>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, OutType,
                                                    LayoutC, BiasType, LayoutC>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType, BiasType, X2ScaleType, X1ScaleType,
                                                                       typename BlockMmad::L0CType>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename OutType_, typename BiasType_, typename X2ScaleType_, typename X1ScaleType_, typename L0CType_>
class LegacyBlockEpilogueDequant {
public:
    using OutType = OutType_;
    using L0CType = L0CType_;
    using Params = typename Blaze::Epilogue::Block::BlockEpilogueDequant<OutType, BiasType_, X2ScaleType_, X1ScaleType_,
                                                                         L0CType>::Params;

    __aicore__ inline void Init(const Params&) {}
    __aicore__ inline void operator()(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
};

template <typename AType, typename BType, typename OutType, typename X2ScaleType, typename X1ScaleType,
          typename BiasType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
struct QBMMLegacyMixTypes : QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode> {
    using Base = QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode>;
    using BlockEpilogue = LegacyBlockEpilogueDequant<OutType, BiasType, X2ScaleType, X1ScaleType,
                                                     typename Base::BlockMmad::L0CType>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename Base::ProblemShape, typename Base::BlockMmad,
                                                      BlockEpilogue, typename Base::BlockScheduler>;
};

// 用 tilingData 填充 MIX BlockMmad::Params（aGm/bGm + 全套 tile/L1/L0C 配置）。
template <typename MmadParams>
__aicore__ inline void FillMixMmadParams(MmadParams& mmadParams, GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM,
                                         const QBMMV3TilingData& tilingData)
{
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    mmadParams.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    mmadParams.l0TileShape = BlockShape{static_cast<int64_t>(tilingData.baseM_qbmm),
                                        static_cast<int64_t>(tilingData.baseN_qbmm),
                                        static_cast<int64_t>(tilingData.baseK_qbmm), 0};
    mmadParams.kAL1 = tilingData.kAL1;
    mmadParams.kBL1 = tilingData.kBL1;
    mmadParams.l1BufferNum = tilingData.nBufferNum;
    mmadParams.enableL0CPingPong = (tilingData.dbL0C > 1);
    mmadParams.biasGmAddr = biasGM;
    mmadParams.isBias = tilingData.isBias != 0 && tilingData.biasDtype == DT_INT32;
}

// Fill BlockEpilogueDequant::Params from tiling and GM addresses.
template <typename EpilogueParams>
__aicore__ inline void FillEpilogueParams(EpilogueParams& epilogueParams, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                          GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    epilogueParams.x2ScaleGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);         // 权重 scale
    epilogueParams.x1ScaleGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM); // 激活 per-token scale
    epilogueParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    epilogueParams.outGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    epilogueParams.m = tilingData.m;
    epilogueParams.n = tilingData.n;
    epilogueParams.baseM = tilingData.baseM_qbmm;
    epilogueParams.baseN = tilingData.baseN_qbmm;
    epilogueParams.x1QuantMode = tilingData.x1QuantMode;
    epilogueParams.x2QuantMode = tilingData.x2QuantMode;
    epilogueParams.isBias = tilingData.isBias != 0 && tilingData.biasDtype != DT_INT32;
    epilogueParams.biasDtype = tilingData.biasDtype;
}

// 多 batch MIX：GemmUniversal（KernelMmadWithScaleMix 特化）。
template <typename AType, typename BType, typename OutType, typename X2ScaleType = float, typename X1ScaleType = float,
          typename BiasType = int32_t, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE,
          bool USE_LEGACY_EPILOGUE = false>
__aicore__ inline void QBMMMixWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                      GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using Types = AscendC::Std::conditional_t<
        USE_LEGACY_EPILOGUE,
        QBMMLegacyMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode>,
        QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode>>;
    using QBMMKernel = typename Types::Kernel;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    FillMixMmadParams(params.mmadParams, x1GM, x2GM, biasGM, tilingData);
    FillQbmmSchParams(params.schParams, tilingData);

    FillQbmmBatchParams(params.qbmmParams, tilingData);
    FillQbmmTileParams(params.qbmmParams, tilingData);

    FillEpilogueParams(params.epilogueParams, pertokenScaleGM, scaleGM, biasGM, yGM, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

// 单 batch MIX：GemmUniversal（KernelMmadWithScaleMixWithoutBatch 特化，裁剪 batch 广播路径）。
template <typename AType, typename BType, typename OutType, typename X2ScaleType = float, typename X1ScaleType = float,
          typename BiasType = int32_t, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMixWithoutBatchWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                  GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using Types = QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode,
                               asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn,
                               Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>;
    using QBMMKernel = typename Types::Kernel;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, 1L};
    FillMixMmadParams(params.mmParams, x1GM, x2GM, biasGM, tilingData);
    params.mmParams.problemShape = params.problemShape;
    FillQbmmSchParams(params.schParams, tilingData);

    FillEpilogueParams(params.epilogueParams, pertokenScaleGM, scaleGM, biasGM, yGM, tilingData);
    params.qbmmParams.bMustHitL2 = tilingData.weightMustHitL2;

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class X2_SCALE_TYPE = float,
          class X1_SCALE_TYPE = float, bool USE_LEGACY_EPILOGUE = false>
__global__ __aicore__ void qbmm_mix_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                 GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, X2_SCALE_TYPE, X1_SCALE_TYPE, DTYPE_BIAS,
                           Blaze::Gemm::NONE_FULL_LOAD_MODE, USE_LEGACY_EPILOGUE>(x1GM, x2GM, pertokenScaleGM, scaleGM,
                                                                                  biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class X2_SCALE_TYPE = float,
          class X1_SCALE_TYPE = float>
__global__ __aicore__ void qbmm_mix_without_batch_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                               GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                               GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, X2_SCALE_TYPE, X1_SCALE_TYPE, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class X2_SCALE_TYPE = float,
          class X1_SCALE_TYPE = float>
__global__ __aicore__ void qbmm_mix_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                             GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                             GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, X2_SCALE_TYPE, X1_SCALE_TYPE, DTYPE_BIAS,
                           Blaze::Gemm::A_FULL_LOAD_MODE>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM,
                                                          *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class X2_SCALE_TYPE = float,
          class X1_SCALE_TYPE = float>
__global__ __aicore__ void qbmm_mix_without_batch_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                           GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                                           GM_ADDR biasGM, GM_ADDR yGM,
                                                                           GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, X2_SCALE_TYPE, X1_SCALE_TYPE, DTYPE_BIAS,
                                       Blaze::Gemm::A_FULL_LOAD_MODE>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM,
                                                                      *tilingData);
}
