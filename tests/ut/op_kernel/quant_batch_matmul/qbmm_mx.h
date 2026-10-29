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
 * \file qbmm_mx.h
 * \brief QBMM MX Kernel UT wrappers.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMxWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                     GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);
    params.l1Params.kL1 = tilingData.kAL1;
    params.l1Params.scaleKL1 = tilingData.kBL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;

    FillQbmmSchParams(params.schParams, tilingData);

    params.qbmmParams.batchA1 = tilingData.batchA1;
    params.qbmmParams.batchA2 = tilingData.batchA2;
    params.qbmmParams.batchA3 = tilingData.batchA3;
    params.qbmmParams.batchA4 = tilingData.batchA4;
    params.qbmmParams.batchB1 = tilingData.batchB1;
    params.qbmmParams.batchB2 = tilingData.batchB2;
    params.qbmmParams.batchB3 = tilingData.batchB3;
    params.qbmmParams.batchB4 = tilingData.batchB4;
    params.qbmmParams.batchC1 = tilingData.batchC1;
    params.qbmmParams.batchC2 = tilingData.batchC2;
    params.qbmmParams.batchC3 = tilingData.batchC3;
    params.qbmmParams.batchC4 = tilingData.batchC4;
    params.qbmmParams.biasThreeDim = tilingData.biasThreeDim;
    params.qbmmParams.baseM = tilingData.baseM_qbmm;
    params.qbmmParams.baseN = tilingData.baseN_qbmm;
    params.qbmmParams.baseK = tilingData.baseK_qbmm;
    params.qbmmParams.isBias = tilingData.isBias;
    params.qbmmParams.dbL0C = tilingData.dbL0C;
    params.qbmmParams.bMustHitL2 = tilingData.weightMustHitL2;

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType, bool ScaleNz = false>
__aicore__ inline void QBMMStreamKWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                          GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR workspaceGM,
                                          const QBMMStreamKTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpilogueDispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using ScaleLayout = AscendC::Std::conditional_t<ScaleNz, asc::te::nn_layout_ptn, asc::te::scaleb_nd_layout_ptn>;
    using LayoutPair = AscendC::Std::tuple<LayoutB, ScaleLayout>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutPair, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, CType, EpilogueDispatchPolicy>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    typename QBMMKernel::QBMMStreamKParams qbmmParams{tilingData.scaleKL1, tilingData.dbL0C,
                                                      tilingData.weightMustHitL2};
    GM_ADDR biasMmadGM = tilingData.isBias == 0U ? nullptr : biasGM;
    Params params{{tilingData.m, tilingData.n, tilingData.k, tilingData.b},
                  {x1GM, x2GM, yGM, biasMmadGM, pertokenScaleGM, scaleGM},
                  {yGM, workspaceGM},
                  {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.singleCoreK,
                   tilingData.kL1},
                  qbmmParams};

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMML0CPingpongWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                              GM_ADDR biasGM, GM_ADDR yGM, const QBMML0CPingpongTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = x1GM;
    params.mmadParams.bGmAddr = x2GM;
    params.mmadParams.cGmAddr = yGM;
    params.mmadParams.biasGmAddr = biasGM;
    params.mmadParams.scaleAGmAddr = pertokenScaleGM;
    params.mmadParams.scaleBGmAddr = scaleGM;
    params.l1Params.kL1 = tilingData.kL1;
    params.l1Params.scaleKL1 = tilingData.scaleKL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;
    params.schParams.baseM = tilingData.baseM;
    params.schParams.baseN = tilingData.baseN;
    params.schParams.mTailTile = 1;
    params.schParams.nTailTile = 1;
    params.schParams.mBaseTailSplitCnt = 1;
    params.schParams.nBaseTailSplitCnt = 1;
    params.schParams.mTailMain = 0;
    params.schParams.nTailMain = 0;
    params.qbmmParams = {1U,
                         1U,
                         1U,
                         static_cast<uint32_t>(tilingData.b),
                         1U,
                         1U,
                         1U,
                         static_cast<uint32_t>(tilingData.b),
                         1U,
                         1U,
                         1U,
                         static_cast<uint32_t>(tilingData.b),
                         0U,
                         tilingData.baseM,
                         tilingData.baseN,
                         tilingData.baseK,
                         tilingData.isBias,
                         tilingData.dbL0C,
                         tilingData.weightMustHitL2};

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMxWithoutBatchWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                 GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, 1L};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);
    params.l1Params.kL1 = tilingData.kAL1;
    params.l1Params.scaleKL1 = tilingData.kBL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;

    FillQbmmSchParams(params.schParams, tilingData);

    params.qbmmParams.baseM = tilingData.baseM_qbmm;
    params.qbmmParams.baseN = tilingData.baseN_qbmm;
    params.qbmmParams.baseK = tilingData.baseK_qbmm;
    params.qbmmParams.isBias = tilingData.isBias;
    params.qbmmParams.dbL0C = tilingData.dbL0C;
    params.qbmmParams.bMustHitL2 = tilingData.weightMustHitL2;

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMML0CPingpongWithoutBatchWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                          GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                          const QBMML0CPingpongTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false,
                                                                     Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params{{tilingData.m, tilingData.n, tilingData.k, 1L},
                  {x1GM, x2GM, yGM, biasGM, pertokenScaleGM, scaleGM},
                  {tilingData.kL1, tilingData.scaleKL1, tilingData.nBufferNum},
                  {tilingData.baseM, tilingData.baseN, 1L, 1L, 1L, 1L, 0L, 0L},
                  {tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.isBias, tilingData.dbL0C,
                   tilingData.weightMustHitL2}};

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM,
                                                                   *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS, bool ScaleNz = false>
__global__ __aicore__ void qbmm_streamk_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                     GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR workspaceGM,
                                                     GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMMStreamKTilingData*>(tilingGM);
    QBMMUT::QBMMStreamKWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, ScaleNz>(x1GM, x2GM, pertokenScaleGM, scaleGM,
                                                                                 biasGM, yGM, workspaceGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_l0c_pingpong_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                             GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                             GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMML0CPingpongTilingData*>(tilingGM);
    QBMMUT::QBMML0CPingpongWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, pertokenScaleGM, scaleGM,
                                                                            biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_l0c_pingpong_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                         GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                                         GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMML0CPingpongTilingData*>(tilingGM);
    QBMMUT::QBMML0CPingpongWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_l0c_pingpong_without_batch_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                           GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                                           GM_ADDR biasGM, GM_ADDR yGM,
                                                                           GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMML0CPingpongTilingData*>(tilingGM);
    QBMMUT::QBMML0CPingpongWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_l0c_pingpong_without_batch_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                                       GM_ADDR pertokenScaleGM,
                                                                                       GM_ADDR scaleGM, GM_ADDR biasGM,
                                                                                       GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMML0CPingpongTilingData*>(tilingGM);
    QBMMUT::QBMML0CPingpongWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_without_batch_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                              GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                              GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, pertokenScaleGM, scaleGM,
                                                                               biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM,
                                                            GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                            GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_without_batch_a_full_load_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM,
                                                                          GM_ADDR pertokenScaleGM, GM_ADDR scaleGM,
                                                                          GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}
