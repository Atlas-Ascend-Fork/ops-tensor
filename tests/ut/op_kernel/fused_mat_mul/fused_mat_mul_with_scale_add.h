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
 * \file fused_mat_mul_with_scale_add.h
 * \brief FusedMatMul ScaleAdd Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_with_scale_add.h"
#include "fused_mat_mul_tiling_data.h"

namespace FusedMatMulUT {

template <typename ElementType>
__aicore__ inline void FusedMatMulWithScaleAddWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR x3GM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, const FusedMatMulTilingData& tilingData)
{
    using AccType = float;
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 0,
                                                                    Blaze::Gemm::KernelMmadFmmWithScaleAdd>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                         false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, Layout, ElementType, Layout, AccType,
                                                    Layout, ElementType, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    const auto& batchTiling = tilingData.matMulTilingData;
    const auto& matmulTiling = batchTiling.matMulTilingData;
    static constexpr bool enable2UB = AscendC::IsSameType<AccType, float>::value;
    static constexpr uint8_t singleUbBuffer = 1U;
    Params params = {{matmulTiling.m, matmulTiling.n, matmulTiling.k, batchTiling.batchDimAll},
                     {x1GM, x2GM, nullptr, nullptr, nullptr, workspaceGM, matmulTiling.k, matmulTiling.mL1,
                      matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
                      matmulTiling.l1BufferNum, matmulTiling.l0cDB, enable2UB, singleUbBuffer},
                     {x3GM, yGM, tilingData.alpha, tilingData.beta},
                     {matmulTiling.mL1, matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN,
                      matmulTiling.baseK, matmulTiling.mTailCnt, matmulTiling.nTailCnt, matmulTiling.mBaseTailSplitCnt,
                      matmulTiling.nBaseTailSplitCnt, matmulTiling.mTailMain, matmulTiling.nTailMain,
                      matmulTiling.mmadParam, static_cast<uint32_t>(matmulTiling.l2CacheDisable), matmulTiling.sliceM,
                      matmulTiling.srcNdStride, matmulTiling.innerBatch}};

    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.splitM = params.mmadParams.splitM != 0;
    params.epilogueParams.baseM = params.schParams.baseM;
    params.epilogueParams.baseN = params.schParams.baseN;

    MatmulKernel kernel;
    kernel(params);
}

} // namespace FusedMatMulUT
