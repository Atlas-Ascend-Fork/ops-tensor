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
 * \file fused_mat_mul.cpp
 * \brief FusedMatMul Kernel UT统一入口
 */

#pragma once

#include <cstring>
#include "fused_mat_mul_with_scale_add.h"
#include "blaze/epilogue/block/block_epilogue_matmul_with_sigmoid.h"

template <typename ElementType>
__global__ __aicore__ void fused_mat_mul_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR x3GM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    FusedMatMulUT::FusedMatMulTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(tilingData));
    FusedMatMulUT::FusedMatMulWithScaleAddWrapper<ElementType>(x1GM, x2GM, x3GM, yGM, workspaceGM, tilingData);
}

template <typename ElementType, typename BiasType, bool TransA, bool TransB>
__global__ __aicore__ void fused_mat_mul_sigmoid_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR yGM,
                                                              GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    using AccType = float;
    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 10,
                                                                    Blaze::Gemm::KernelMmadWithEpilogue>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                         false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, AccType,
                                                    LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulWithSigmoid<ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    FusedMatMulUT::FusedMatMulTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(tilingData));
    const auto& matmulTiling = tilingData.matMulTilingData.matMulTilingData;
    Params params = {{matmulTiling.m, matmulTiling.n, matmulTiling.k, 1},
                     {x1GM, x2GM, nullptr, biasGM, nullptr, workspaceGM, matmulTiling.k, matmulTiling.mL1,
                      matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
                      matmulTiling.l1BufferNum, matmulTiling.l0cDB, true, 1U},
                     {yGM},
                     {matmulTiling.mL1, matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN,
                      matmulTiling.baseK, matmulTiling.mTailCnt, matmulTiling.nTailCnt, matmulTiling.mBaseTailSplitCnt,
                      matmulTiling.nBaseTailSplitCnt, matmulTiling.mTailMain, matmulTiling.nTailMain,
                      matmulTiling.mmadParam, static_cast<uint32_t>(matmulTiling.l2CacheDisable), matmulTiling.sliceM,
                      matmulTiling.srcNdStride, matmulTiling.innerBatch}};
    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.splitM = params.mmadParams.splitM != 0;

    MatmulKernel kernel;
    kernel(params);
}
