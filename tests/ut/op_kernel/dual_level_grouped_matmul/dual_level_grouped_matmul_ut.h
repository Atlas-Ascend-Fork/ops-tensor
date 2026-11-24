/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "../quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

#include "blaze/epilogue/block/block_epilogue_dual_level_scale.h"

namespace DualLevelGmmUt {
constexpr int64_t EXPERT_COUNT = 3;
constexpr uint32_t BASIC_BLOCK = 256;
constexpr uint32_t LEVEL1_SCALE_L1_WINDOW_K = 4096;

struct Case {
    const char* name;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t groupRows[EXPERT_COUNT];
    uint8_t groupListType;
    bool bias;
    bool transpose;
    bool bf16;
    uint32_t cores;
    uint32_t level0Size{512};
};

struct Buffers {
    GM_ADDR x;
    GM_ADDR a0;
    GM_ADDR a1;
    GM_ADDR weight;
    GM_ADDR b0;
    GM_ADDR b1;
    GM_ADDR groups;
    GM_ADDR bias;
    GM_ADDR y;
};

template <class Kernel>
__aicore__ inline void SetParams(typename Kernel::Params& p, const Case& c, const Buffers& b)
{
    p.problemShape = {c.m, c.n, c.k, 0};
    p.mmadParams.aGmAddr = b.x;
    p.mmadParams.bGmAddr = b.weight;
    p.mmadParams.cGmAddr = b.y;
    p.mmadParams.biasGmAddr = b.bias;
    p.mmadParams.scaleAGmAddr = b.a1;
    p.mmadParams.scaleBGmAddr = b.b1;
    p.epilogueParams.level0ScaleAGmAddr = b.a0;
    p.epilogueParams.level0ScaleBGmAddr = b.b0;
    p.groupListGmAddr = b.groups;
    auto& g = p.gmmParams;
    g.groupNum = EXPERT_COUNT;
    g.m = c.m;
    g.n = c.n;
    g.k = c.k;
    g.baseM = BASIC_BLOCK;
    g.baseN = BASIC_BLOCK;
    g.baseK = c.level0Size;
    g.kAL1 = c.level0Size;
    g.kBL1 = c.level0Size;
    g.scaleKAL1 = LEVEL1_SCALE_L1_WINDOW_K;
    g.scaleKBL1 = g.scaleKAL1;
    g.isBias = c.bias;
    g.groupType = 0;
    g.groupListType = c.groupListType;
    g.singleW = 1;
}
} // namespace DualLevelGmmUt

template <class Output, bool Transpose>
__global__ __aicore__ void DualLevelGmmKernelEntry(DualLevelGmmUt::Case c, DualLevelGmmUt::Buffers b)
{
    using Nd = asc::te::nd_ext_layout_ptn;
    using WeightLayout = AscendC::Std::conditional_t<Transpose, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithDualLevelScaleMx>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp4x2_e2m1_t, Nd, fp4x2_e2m1_t, WeightLayout, Output, Nd, float,
                                               Nd>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDualLevelScale<Output>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename Mmad::ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    typename Kernel::Params p{};
    DualLevelGmmUt::SetParams<Kernel>(p, c, b);
    Kernel kernel;
    kernel(p);
}
