/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file tqbmm_fixpipe_quant.cpp
 * \brief TQBMM fixpipe quant (HIFP8) Kernel UT entry
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "tqbmm_fixpipe_quant_tiling_data.h"

enum TqbmmFixpipeQuantOpType : int8_t {
    OP_TYPE_TQBMM_FIXPIPE_QUANT_BASIC = 0,
    OP_TYPE_TQBMM_FIXPIPE_QUANT_TRANS_BATCH_A = 1,
};

namespace TqbmmFixpipeQuantUT {

template <typename AType, typename BType, typename X2ScaleType, typename CType, typename BiasType,
          uint64_t NON_CONTIGUOUS_TYPE = 0>
__aicore__ inline void TqbmmFixpipeQuantWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR biasGM,
                                                GM_ADDR pertokenScaleGM, GM_ADDR cGM,
                                                const TqbmmFixpipeQuantTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMFixpipeQuant, __gm__ int32_t*, NON_CONTIGUOUS_TYPE>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB,
                                                                                AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    using BlockMmadParams = typename BlockMmad::Params;

    constexpr uint32_t x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::DEFAULT);
    constexpr uint32_t x2QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);

    Params params;
    params.problemShape = {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
                           static_cast<int64_t>(tilingData.k), static_cast<int64_t>(tilingData.batch)};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = aGM;
    mmadParams.bGmAddr = bGM;
    mmadParams.cGmAddr = cGM;
    mmadParams.biasGmAddr = biasGM;
    mmadParams.scaleAGmAddr = pertokenScaleGM;
    mmadParams.scaleBGmAddr = scaleGM;
    params.mmadParams = mmadParams;

    params.schParams = {
        static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN), 1, 1, 1, 1, 0, 0};

    params.tqbmmParams = {x1QuantMode,           x2QuantMode,      tilingData.kAL1,      tilingData.kBL1,
                          tilingData.nBufferNum, tilingData.baseM, tilingData.baseN,     tilingData.baseK,
                          tilingData.isBias,     tilingData.l0cDB, tilingData.bMustHitL2};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace TqbmmFixpipeQuantUT

template <int8_t OP_TYPE, typename DTYPE_A, typename DTYPE_B, typename DTYPE_SCALE, typename DTYPE_Y,
          typename DTYPE_BIAS, uint64_t NON_CONTIGUOUS_TYPE = 0>
__global__ __aicore__ void tqbmm_fixpipe_quant_kernel_entry(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR biasGM,
                                                            GM_ADDR pertokenScaleGM, GM_ADDR cGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const TqbmmFixpipeQuantTilingData*>(tilingGM);

    if constexpr (OP_TYPE == OP_TYPE_TQBMM_FIXPIPE_QUANT_BASIC ||
                  OP_TYPE == OP_TYPE_TQBMM_FIXPIPE_QUANT_TRANS_BATCH_A) {
        TqbmmFixpipeQuantUT::TqbmmFixpipeQuantWrapper<DTYPE_A, DTYPE_B, DTYPE_SCALE, DTYPE_Y, DTYPE_BIAS,
                                                      NON_CONTIGUOUS_TYPE>(aGM, bGM, scaleGM, biasGM, pertokenScaleGM,
                                                                           cGM, *tilingData);
    } else {
        static_assert(
            (OP_TYPE == OP_TYPE_TQBMM_FIXPIPE_QUANT_BASIC || OP_TYPE == OP_TYPE_TQBMM_FIXPIPE_QUANT_TRANS_BATCH_A),
            "Unsupported OP_TYPE for tqbmm_fixpipe_quant_kernel_entry");
    }
}
