/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confidence_head_fused.h
 * \brief ConfidenceHeadFused mixed-kernel UT wrapper.
 */

#pragma once

#include <cstring>
#include "blaze/epilogue/block/block_epilogue_confidence_head_fused.h"
#include "blaze/gemm/block/block_mmad_confidence_head_fused.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_confidence_head_fused_mix.h"

namespace ConfidenceHeadFusedUT {

struct TestTiling {
    uint32_t n{128};
    uint32_t kEmbed{128};
    uint32_t kMarkov{0};
    uint32_t biasN{0};
    uint32_t usedCoreNum{1};
    bool hasTemperature{false};
    float temperature{1.0F};
    float lowLimit{0.0F};
    float highLimit{0.0F};
    bool hasLowLimit{false};
    bool hasHighLimit{false};
};

template <bool HasMarkov, typename ElementType>
__global__ __aicore__ void ConfidenceHeadFusedKernelEntry(GM_ADDR hidden, GM_ADDR weight, GM_ADDR markov,
                                                          GM_ADDR temperature, GM_ADDR bias, GM_ADDR out,
                                                          GM_ADDR tilingGm)
{
    TestTiling tiling;
    memcpy(&tiling, tilingGm, sizeof(tiling));
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::ConfidenceHeadFusedPolicy<Blaze::Gemm::A_FULL_LOAD_MODE, false, HasMarkov>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, float,
                                                    LayoutA, float, LayoutA>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::A_FULL_LOAD_MODE>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueConfidenceHeadFused<float, float>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    typename Kernel::Params params{};
    params.problemShape = {1L, tiling.n, tiling.kEmbed, tiling.kMarkov};
    params.weightGmAddr = weight;
    params.hiddenGmAddr = hidden;
    params.markovGmAddr = markov;
    params.mmadParams.kEmbed = tiling.kEmbed;
    params.mmadParams.kMarkov = HasMarkov ? tiling.kMarkov : 0U;
    params.mmadParams.baseN = 128;
    params.mmadParams.baseK = 128;
    params.mmadParams.kL1 = 256;
    params.mmadParams.l1Stages = 4;
    params.mmadParams.l0cStages = 2;
    params.mmadParams.biasGmAddr = bias;
    params.epilogueParams.outputGmAddr = out;
    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.baseN = 128;
    params.epilogueParams.temperatureGmAddr = temperature;
    params.epilogueParams.biasGmAddr = bias;
    params.epilogueParams.biasN = tiling.biasN;
    params.epilogueParams.hasBias = bias != nullptr;
    params.epilogueParams.lowLimit = tiling.lowLimit;
    params.epilogueParams.highLimit = tiling.highLimit;
    params.epilogueParams.hasLowLimit = tiling.hasLowLimit;
    params.epilogueParams.hasHighLimit = tiling.hasHighLimit;
    params.schParams.mL1 = 1;
    params.schParams.nL1 = params.mmadParams.baseN;
    params.schParams.kL1 = params.mmadParams.kL1;
    params.schParams.baseM = 1;
    params.schParams.baseN = params.mmadParams.baseN;
    params.schParams.baseK = params.mmadParams.baseK;
    params.schParams.mTailCnt = 1;
    params.schParams.nTailCnt = 1;
    Kernel kernel;
    kernel(params);
}

template <typename OutputType>
__global__ __aicore__ void SigmoidKernelEntry(GM_ADDR input, GM_ADDR output)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (AscendC::GetBlockIdx() % AscendC::GetTaskRation() != 0) {
        return;
    }
    constexpr int64_t N = 17;
    constexpr int64_t ROW_PITCH = 64;
    auto inputUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(0),
                                        Blaze::Gemm::MakeNDExtLayout<float>(1L, N, ROW_PITCH));
    auto outputUb = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, OutputType>(ROW_PITCH * sizeof(float)),
        Blaze::Gemm::MakeNDExtLayout<OutputType>(1L, N, ROW_PITCH));
    auto inputGm = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(input)),
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, N));
    auto outputGm = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutputType*>(output)),
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, N));
    asc::te::copy(inputUb, inputGm);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    Blaze::Epilogue::Tile::Sigmoid::Run(inputUb, outputUb);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    asc::te::copy(outputGm, outputUb);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}

} // namespace ConfidenceHeadFusedUT
