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
 * \file transpose_quant_batch_mat_mul_mx.cpp
 * \brief TQBMM MX Kernel UT统一入口
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#if defined(ASCENDC_CPU_DEBUG)
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#endif
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e2m1x2
#undef __fp8e4m3
#endif
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "tqbmm_mx_tiling_data.h"

enum TqbmmMxOpType : int8_t {
    OP_TYPE_TQBMM_MX_BASIC = 0,
    OP_TYPE_TQBMM_MX_TRANS_BATCH_A = 1,
};

namespace TqbmmUT {

template <typename AType, typename BType, typename CType, typename BiasType, uint64_t NON_CONTIGUOUS_TYPE = 0>
__aicore__ inline void TqbmmMxBasicWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleAGM, GM_ADDR scaleBGM,
                                           GM_ADDR cGM, const TqbmmMxTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMM,
                                                          Blaze::Gemm::L0C2UB_MODE_NONE, NON_CONTIGUOUS_TYPE>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB,
                                                                                AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params{};
    params.problemShape = ProblemShape{static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
                                       static_cast<int64_t>(tilingData.k), static_cast<int64_t>(tilingData.batch)};
    params.mmadParams = {aGM, bGM, cGM, biasGM, scaleAGM, scaleBGM};
    params.l1Params = {static_cast<uint64_t>(tilingData.kL1), static_cast<uint64_t>(tilingData.kL1),
                       static_cast<uint64_t>(tilingData.l1BufferNum)};
    params.schParams = {
        static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN), 1, 1, 1, 1, 0, 0};
    params.tqbmmParams = {1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          0,
                          static_cast<uint32_t>(tilingData.baseM),
                          static_cast<uint32_t>(tilingData.baseN),
                          static_cast<uint32_t>(tilingData.baseK),
                          0U,
                          static_cast<uint32_t>(tilingData.l0cDB),
                          static_cast<uint32_t>(tilingData.bMustHitL2)};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace TqbmmUT

template <int8_t OP_TYPE, typename DTYPE_X1, typename DTYPE_X2, typename DTYPE_Y, typename DTYPE_BIAS,
          uint64_t NON_CONTIGUOUS_TYPE = 0>
__global__ __aicore__ void tqbmm_mx_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR scaleAGM,
                                                 GM_ADDR scaleBGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const TqbmmMxTilingData*>(tilingGM);

    if constexpr (OP_TYPE == OP_TYPE_TQBMM_MX_BASIC || OP_TYPE == OP_TYPE_TQBMM_MX_TRANS_BATCH_A) {
        TqbmmUT::TqbmmMxBasicWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, NON_CONTIGUOUS_TYPE>(
            x1GM, x2GM, biasGM, scaleAGM, scaleBGM, yGM, *tilingData);
    } else {
        static_assert((OP_TYPE == OP_TYPE_TQBMM_MX_BASIC || OP_TYPE == OP_TYPE_TQBMM_MX_TRANS_BATCH_A),
                      "Unsupported OP_TYPE for tqbmm_mx_kernel_entry");
    }
}
