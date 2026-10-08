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
 * \file tqbmm_mix.cpp
 * \brief TQBMM MIX (FP8 per-token dequant, AIC+AIV) Kernel UT entry
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "tqbmm_mix_tiling_data.h"

enum TqbmmMixOpType : int8_t {
    OP_TYPE_TQBMM_MIX_BASIC = 0,
    OP_TYPE_TQBMM_MIX_TRANS_BATCH_A = 1,
};

namespace TqbmmMixUT {

template <typename AType, typename BType, typename X2ScaleType, typename X1ScaleType, typename OutType,
          typename BiasType, uint64_t NON_CONTIGUOUS_TYPE = 0>
__aicore__ inline void TqbmmMixWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR biasGM,
                                       GM_ADDR pertokenScaleGM, GM_ADDR cGM, const TqbmmMixTilingData& tilingData)
{
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMMx,
                                                           NON_CONTIGUOUS_TYPE>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB,
                                                                                AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, int32_t,
                                                    LayoutC, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType, BiasType, X2ScaleType, X1ScaleType,
                                                                       float>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using MmadParams = typename BlockMmad::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    constexpr uint32_t x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERTOKEN_MODE);
    constexpr uint32_t x2QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);
    constexpr uint32_t biasDtypeVal = 0;

    Params params;
    params.problemShape = {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
                           static_cast<int64_t>(tilingData.k), static_cast<int64_t>(tilingData.batch)};

    MmadParams mmadParams;
    mmadParams.aGmAddr = aGM;
    mmadParams.bGmAddr = bGM;
    mmadParams.problemShape = params.problemShape;
    mmadParams.l0TileShape = BlockShape{static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
                                        static_cast<int64_t>(tilingData.baseK), 0};
    mmadParams.kAL1 = tilingData.kAL1;
    mmadParams.kBL1 = tilingData.kBL1;
    mmadParams.l1BufferNum = tilingData.nBufferNum;
    mmadParams.enableL0CPingPong = tilingData.l0cDB > 1;
    params.mmadParams = mmadParams;

    params.schParams = {
        static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN), 1, 1, 1, 1, 0, 0};

    params.tqbmmParams = {tilingData.kAL1,  tilingData.kBL1,  tilingData.nBufferNum, tilingData.baseM,
                          tilingData.baseN, tilingData.baseK, tilingData.l0cDB,      tilingData.bMustHitL2};

    EpilogueParams epilogueParams;
    epilogueParams.x2ScaleGmAddr = scaleGM;
    epilogueParams.x1ScaleGmAddr = pertokenScaleGM;
    epilogueParams.biasGmAddr = biasGM;
    epilogueParams.outGmAddr = cGM;
    epilogueParams.m = static_cast<int64_t>(tilingData.m);
    epilogueParams.n = static_cast<int64_t>(tilingData.n);
    epilogueParams.baseM = static_cast<int64_t>(tilingData.baseM);
    epilogueParams.baseN = static_cast<int64_t>(tilingData.baseN);
    epilogueParams.x1QuantMode = x1QuantMode;
    epilogueParams.x2QuantMode = x2QuantMode;
    epilogueParams.isBias = tilingData.isBias != 0;
    epilogueParams.biasDtype = biasDtypeVal;
    params.epilogueParams = epilogueParams;

    MatmulKernel kernel;
    kernel(params);
}

} // namespace TqbmmMixUT

template <int8_t OP_TYPE, typename DTYPE_A, typename DTYPE_B, typename DTYPE_X2_SCALE, typename DTYPE_X1_SCALE,
          typename DTYPE_Y, typename DTYPE_BIAS, uint64_t NON_CONTIGUOUS_TYPE = 0>
__global__ __aicore__ void tqbmm_mix_kernel_entry(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR biasGM,
                                                  GM_ADDR pertokenScaleGM, GM_ADDR cGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const TqbmmMixTilingData*>(tilingGM);

    if constexpr (OP_TYPE == OP_TYPE_TQBMM_MIX_BASIC || OP_TYPE == OP_TYPE_TQBMM_MIX_TRANS_BATCH_A) {
        TqbmmMixUT::TqbmmMixWrapper<DTYPE_A, DTYPE_B, DTYPE_X2_SCALE, DTYPE_X1_SCALE, DTYPE_Y, DTYPE_BIAS,
                                    NON_CONTIGUOUS_TYPE>(aGM, bGM, scaleGM, biasGM, pertokenScaleGM, cGM, *tilingData);
    } else {
        static_assert((OP_TYPE == OP_TYPE_TQBMM_MIX_BASIC || OP_TYPE == OP_TYPE_TQBMM_MIX_TRANS_BATCH_A),
                      "Unsupported OP_TYPE for tqbmm_mix_kernel_entry");
    }
}
