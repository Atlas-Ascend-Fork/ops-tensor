/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_mx.h
 * \brief Kernel UT wrapper for weight-only MX matmul with an AIV weight prologue.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "weight_quant_batch_matmul_mx_cpu_debug_stub.h"

#include "blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h"
#include "weight_quant_batch_matmul_mx_tiling_data.h"

namespace WeightQuantBatchMatmulMxUT {

template <bool WeightNz, bool ScaleBNz>
__aicore__ inline void Run(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm, GM_ADDR scaleBGm, GM_ADDR cGm,
                           const WeightQuantBatchMatmulMxTilingData& tiling)
{
    using AType = fp8_e4m3fn_t;
    using BType = fp4x2_e2m1_t;
    using ScaleType = AscendC::fp8_e8m0_t;
    using CType = half;
    using BiasType = half;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<WeightNz, asc::te::zn_layout_ptn, asc::te::dn_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = AscendC::Std::conditional_t<ScaleBNz, asc::te::nn_layout_ptn, asc::te::scaleb_dn_layout_ptn>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
        LayoutC>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

    typename Kernel::Params params{
        asc::te::make_shape(tiling.m, tiling.n, tiling.k),
        {aGm, scaleAGm, scaleBGm, cGm,
         asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                             static_cast<int64_t>(tiling.tileShapeKL1), static_cast<int64_t>(tiling.tileShapeScaleKL1)),
         asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                             static_cast<int64_t>(tiling.baseK)),
         tiling.l1BufferNum, tiling.hasBias != 0U},
        {bGm, biasGm, tiling.kBubSize, tiling.nBubSize},
        {tiling.baseM, tiling.baseN, tiling.mTailTile, tiling.nTailTile, tiling.mBaseTailSplitCnt,
         tiling.nBaseTailSplitCnt, tiling.mTailMain, tiling.nTailMain}};
    Kernel kernel;
    kernel(params);
}

} // namespace WeightQuantBatchMatmulMxUT

template <bool WeightNz, bool ScaleBNz = false>
__global__ __aicore__ void weight_quant_batch_matmul_mx_kernel_entry(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm,
                                                                     GM_ADDR scaleAGm, GM_ADDR scaleBGm, GM_ADDR cGm,
                                                                     GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto* tiling = reinterpret_cast<const WeightQuantBatchMatmulMxTilingData*>(tilingGm);
    WeightQuantBatchMatmulMxUT::Run<WeightNz, ScaleBNz>(aGm, bGm, biasGm, scaleAGm, scaleBGm, cGm, *tiling);
}
