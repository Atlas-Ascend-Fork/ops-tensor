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
 * \file transpose_batch_mat_mul_basic.h
 * \brief TransposeBatchMatMul Basic Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_iterbatch.h"
#include "blaze/epilogue/block/block_epilogue_iterbatch.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "transpose_batch_mat_mul_tiling_data.h"

namespace TbmmUT {

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE, uint64_t NON_CONTIGUOUS_TYPE = 0,
          class A_LAYOUT = asc::te::nd_ext_layout_ptn, class B_LAYOUT = asc::te::nd_ext_layout_ptn,
          bool SPLIT_K = false>
__aicore__ inline void TbmmBasicWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                        const TbmmBasicTilingData& tilingData, GM_ADDR scaleGM)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using BiasType = BIAS_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = LayoutC;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = AscendC::Std::conditional_t<
        SPLIT_K,
        Blaze::Gemm::MatmulMultiBlockBasicSplitK<0, true, Blaze::Gemm::KernelMmadMultiBlockBasic, NON_CONTIGUOUS_TYPE>,
        Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, NON_CONTIGUOUS_TYPE>>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
        ProblemShape, 0, AscendC::Std::is_same_v<AType, float>, !Blaze::Gemm::IsWeightNz<LayoutB>::value>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k),
         static_cast<int64_t>(tilingData.batch), static_cast<int64_t>(tilingData.batchSplitFactor)},
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM, tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM,
         tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB},
        {},
        {static_cast<uint32_t>(tilingData.mL1), static_cast<uint32_t>(tilingData.nL1),
         static_cast<uint32_t>(tilingData.kL1), static_cast<uint32_t>(tilingData.baseM),
         static_cast<uint32_t>(tilingData.baseN), static_cast<uint32_t>(tilingData.baseK),
         static_cast<uint32_t>(tilingData.mTailCnt), static_cast<uint32_t>(tilingData.nTailCnt),
         static_cast<uint32_t>(tilingData.mBaseTailSplitCnt), static_cast<uint32_t>(tilingData.nBaseTailSplitCnt),
         static_cast<uint32_t>(tilingData.mTailMain), static_cast<uint32_t>(tilingData.nTailMain),
         static_cast<uint8_t>(tilingData.isHf32), static_cast<uint32_t>(tilingData.l2CacheDisable),
         static_cast<uint32_t>(tilingData.sliceM), static_cast<uint32_t>(tilingData.srcNdStride),
         static_cast<uint32_t>(tilingData.innerBatch)}};

    if constexpr (SPLIT_K) {
        params.mmadParams.k = tilingData.k;
        params.mmadParams.splitSingleCoreKBase = tilingData.baseK / 2;
    } else {
        params.mmadParams.scaleGmAddr = scaleGM;
    }

    MatmulKernel kernel;
    kernel(params);
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE, class A_LAYOUT, class B_LAYOUT,
          uint64_t NON_CONTIGUOUS_TYPE = 0,
          Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODEL = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>
__aicore__ inline void TbmmIterBatchWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                            const TbmmBasicTilingData& tilingData)
{
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulIterBatch<L0C2OUT_MODEL, 0, NON_CONTIGUOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, A_LAYOUT, B_TYPE, B_LAYOUT, C_TYPE, LayoutC,
                                                    BIAS_TYPE, LayoutC>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulIterBatch<ProblemShape>;
    using BlockEpilogue = AscendC::Std::conditional_t<L0C2OUT_MODEL != Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY,
                                                      Blaze::Epilogue::Block::BlockEpilogueIterbatch<C_TYPE, C_TYPE>,
                                                      Blaze::Gemm::Block::BlockEpilogueEmpty>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params = {
        {tilingData.m, tilingData.n, tilingData.k, tilingData.batch},
        {aGM, bGM, cGM, biasGM, tilingData.m, tilingData.n, tilingData.k, tilingData.baseM, tilingData.baseN,
         tilingData.baseK, tilingData.iterBatchL1, tilingData.iterBatchL0},
        {},
        {tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.iterBatchL1, tilingData.iterBatchL0,
         tilingData.isHf32, static_cast<uint32_t>(tilingData.l2CacheDisable)}};
    if constexpr (L0C2OUT_MODEL != Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY) {
        params.epilogueParams = {cGM};
    }
    MatmulKernel kernel;
    kernel(params);
}

} // namespace TbmmUT
