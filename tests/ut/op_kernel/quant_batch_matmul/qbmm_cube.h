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
 * \file qbmm_cube.h
 * \brief QBMM Cube and per-tensor StreamK Kernel UT wrappers.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

template <typename AType, typename BType, typename X2ScaleType, typename CType, typename BiasType,
          typename LayoutA = asc::te::nd_ext_layout_ptn, typename LayoutB = asc::te::nd_ext_layout_ptn,
          typename LayoutC = asc::te::nd_ext_layout_ptn, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE,
          typename ScheduleType = Blaze::Gemm::KernelMmadWithScaleFixpipeQuant>
struct QBMMCubeTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode, false, ScheduleType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA,
                                                    AscendC::Std::tuple<BType, X2ScaleType>, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE, typename X2ScaleType = uint64_t,
          typename LayoutB = asc::te::nd_ext_layout_ptn>
__aicore__ inline void QBMMCubeWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                       GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using Types = QBMMCubeTypes<AType, BType, X2ScaleType, CType, BiasType, asc::te::nd_ext_layout_ptn, LayoutB,
                                asc::te::nd_ext_layout_ptn, FullLoadMode>;
    using QBMMKernel = typename Types::Kernel;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);

    FillQbmmSchParams(params.schParams, tilingData);
    FillQbmmBatchParams(params.qbmmParams, tilingData);
    FillQbmmTileParams(params.qbmmParams, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename X2ScaleType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMPertensorStreamKWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                                   GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR workspaceGM,
                                                   const QBMMPertensorStreamKTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA,
                                                    AscendC::Std::tuple<BType, X2ScaleType>, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using WorkspaceType = typename BlockMmad::WorkspaceType;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<WorkspaceType, CType,
                                                                                    DispatchPolicy, X2ScaleType, float>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    bool hasBias = tilingData.isBias != 0U;
    // Bias placement must follow BlockMmad's compile-time implementation path. biasDtype only describes the
    // epilogue input type after the placement has been selected; it must not override BIAS_IN_MMAD at runtime.
    GM_ADDR biasMmadGM = hasBias && BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;
    GM_ADDR biasEpilogueGM = hasBias && !BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;
    bool isBiasEpilogue = biasEpilogueGM != nullptr;

    Params params{{tilingData.m, tilingData.n, tilingData.k, tilingData.b},
                  {x1GM, x2GM, yGM, biasMmadGM, perTokenScaleGM, scaleGM},
                  {yGM, workspaceGM, scaleGM, perTokenScaleGM, biasEpilogueGM, isBiasEpilogue, tilingData.biasDtype},
                  {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.singleCoreK,
                   tilingData.kL1}};

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE, typename X2ScaleType = uint64_t>
__aicore__ inline void QBMMCubeWithoutBatchWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                   GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using Types = QBMMCubeTypes<AType, BType, X2ScaleType, CType, BiasType, asc::te::nd_ext_layout_ptn,
                                asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn, FullLoadMode,
                                Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch>;
    using QBMMKernel = typename Types::Kernel;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, 1L};
    params.mmadParams = {x1GM, x2GM, yGM, biasGM, pertokenScaleGM, scaleGM};
    FillQbmmSchParams(params.schParams, tilingData);
    params.qbmmParams.x1QuantMode = tilingData.x1QuantMode;
    params.qbmmParams.x2QuantMode = tilingData.x2QuantMode;
    FillQbmmTileParams(params.qbmmParams, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class SCALE_TYPE = uint64_t>
__global__ __aicore__ void qbmm_cube_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                  GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMCubeWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::NONE_FULL_LOAD_MODE, SCALE_TYPE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class SCALE_TYPE, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_pertensor_streamk_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR perTokenScaleGM,
                                                               GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM);
    QBMMUT::QBMMPertensorStreamKWrapper<DTYPE_X1, DTYPE_X2, SCALE_TYPE, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, perTokenScaleGM, scaleGM, biasGM, yGM, workspaceGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class SCALE_TYPE = uint64_t,
          class LayoutB = asc::te::nd_ext_layout_ptn>
__global__ __aicore__ void qbmm_cube_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                              GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                              GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMCubeWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE, SCALE_TYPE,
                            LayoutB>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class SCALE_TYPE = uint64_t>
__global__ __aicore__ void qbmm_cube_without_batch_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                                GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                                GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMCubeWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                        SCALE_TYPE>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, class SCALE_TYPE = uint64_t>
__global__ __aicore__ void qbmm_cube_without_batch_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                            GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                                            GM_ADDR biasGM, GM_ADDR yGM,
                                                                            GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMCubeWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE,
                                        SCALE_TYPE>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}
