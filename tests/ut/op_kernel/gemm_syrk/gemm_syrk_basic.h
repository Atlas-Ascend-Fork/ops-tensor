/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file gemm_syrk_basic.h
 * \brief GemmSyrk Kernel UT Wrappers: single-fetch syrk assembly.
 *
 * Single-fetch (SYRK_KERNEL_SINGLE_FETCH): MatmulSyrk + KernelMmadSyrk. One nd2nz
 * (dn2nz when TRANS) CopyGM2L1 per A row-block per k-chunk; the NZ image doubles as
 * the ZN image of A^T and feeds both cube inputs. Upper-triangle slots compute the
 * {(i, j), (j, i)} tile pairs with a single Mmad chain plus a transposed store of
 * the mirrored tile.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_syrk.h"
#include "blaze/gemm/block/block_scheduler_matmul_syrk.h"
#include "blaze/gemm/kernel/kernel_matmul_syrk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "gemm_syrk_tiling_data.h"

namespace GemmSyrkUT {

template <typename ElementType, bool TRANS>
__aicore__ inline void GemmSyrkSingleFetchWrapper(GM_ADDR aGM, GM_ADDR cInGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                  const GemmSyrkTilingData& tilingData)
{
    using AccType = float;
    // TRANS: the DNExt (m, k) view over the transposed (k, m) storage; the
    // GM->L1 fetch auto-routes to dn2nz and yields the same dual-view image.
    using LayoutA = AscendC::Std::conditional_t<TRANS, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulSyrk;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerSyrkTriangular<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutA, AccType,
                                                    LayoutC, ElementType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    const auto& batchTiling = tilingData.matMulTilingData;
    const auto& matmulTiling = batchTiling.matMulTilingData;
    static constexpr bool enable2UB = AscendC::IsSameType<AccType, float>::value;
    static constexpr uint8_t singleUbBuffer = 1U;
    // The syrk contract collapses mL1/nL1/baseM/baseN into ONE symmetric
    // square block; mL1 carries it here (the production tiling keeps them
    // equal, see GemmSyrkTilingData).
    const uint32_t block = matmulTiling.mL1;
    KernelParams params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, batchTiling.batchDimAll},
        {aGM, workspaceGM, matmulTiling.k, block, matmulTiling.kL1, matmulTiling.baseK, matmulTiling.l1BufferNum},
        // x3 == output == C: in-place read of beta * C and overwrite with the result.
        {cInGM, cGM, tilingData.alpha, tilingData.beta},
        {block}};

    params.epilogueParams.problemShape = params.problemShape;

    MatmulKernel kernel;
    kernel(params);
}

} // namespace GemmSyrkUT
