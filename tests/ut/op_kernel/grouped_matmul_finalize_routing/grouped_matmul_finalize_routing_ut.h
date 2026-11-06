/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include "grouped_matmul_finalize_routing_cpu_debug_compat.h"
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e1m2x2
#undef __fp4e2m1x2
#undef __fp8e4m3
#undef Cast
#endif

namespace GroupedMatmulFinalizeRoutingUt {
struct AlignedCaseConfig {
    static constexpr int64_t GROUP_NUM = 2;
    static constexpr int64_t GROUP_M = 16;
    static constexpr int64_t TOTAL_M = GROUP_NUM * GROUP_M;
    static constexpr int64_t BATCH = 16;
    static constexpr int64_t N = 64;
    static constexpr int64_t K = 64;
    static constexpr uint32_t BASE_M = 16;
    static constexpr uint32_t BASE_N = 64;
    static constexpr uint32_t BASE_K = 64;
    static constexpr uint32_t SCALE_K = 2;
    static constexpr uint32_t SHARED_INPUT_LEN = 0;
    static constexpr uint8_t HAS_BIAS = 0;
};

struct TailCaseConfig {
    static constexpr int64_t GROUP_NUM = 2;
    static constexpr int64_t GROUP_M = 29;
    static constexpr int64_t TOTAL_M = GROUP_NUM * GROUP_M;
    static constexpr int64_t BATCH = 29;
    static constexpr int64_t N = 65;
    static constexpr int64_t K = 126;
    static constexpr uint32_t BASE_M = 32;
    static constexpr uint32_t BASE_N = 64;
    static constexpr uint32_t BASE_K = 64;
    static constexpr uint32_t SCALE_K = 4;
    static constexpr uint32_t SHARED_INPUT_LEN = BATCH;
    static constexpr uint8_t HAS_BIAS = 1;
};

template <typename Config, typename AType, typename BType, typename CType, typename LogitType, typename RowIndexType,
          typename LayoutB>
__aicore__ inline void Run(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR bias, GM_ADDR xScale,
                           GM_ADDR groupList, GM_ADDR sharedInput, GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR y,
                           GM_ADDR tiling)
{
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = bfloat16_t;
    using MmadType = float;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelQgmmMxMixFinalizeRouting>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, MmadType, LayoutC, BiasType,
                                               LayoutBias>;
    using Prologue = Blaze::Gemm::Kernel::BlockPrologueFinalizeRouting<CType, BiasType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<CType, MmadType, LogitType, RowIndexType>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, AscendC::Std::tuple<Prologue, Epilogue>,
                                                      Scheduler>;
    typename Kernel::Params params{};
    params.problemShape = {Config::TOTAL_M, Config::N, Config::K, 1};
    params.blockMmadParams = {x, weight, y, bias, xScale, weightScale};
    params.prologueParams = {sharedInput,   y,   0, Config::SHARED_INPUT_LEN, static_cast<int32_t>(Config::N),
                             Config::BATCH, 1.0F};
    params.epilogueParams = {y, weightScale, xScale, bias, logit, rowIndex};
    params.groupListGmAddr = groupList;
    params.gmmParams = {static_cast<uint32_t>(Config::GROUP_NUM),
                        static_cast<uint32_t>(Config::BATCH),
                        0,
                        Config::SHARED_INPUT_LEN,
                        1.0F,
                        Config::BASE_M,
                        Config::BASE_N,
                        Config::BASE_K,
                        Config::BASE_K,
                        Config::BASE_K,
                        static_cast<uint32_t>(Config::K),
                        static_cast<uint32_t>(Config::K),
                        Config::HAS_BIAS,
                        1,
                        1};
    Kernel kernel;
    kernel(params);
    (void)tiling;
}
} // namespace GroupedMatmulFinalizeRoutingUt

template <typename Config, typename AType, typename BType, typename CType, typename LogitType, typename RowIndexType,
          typename LayoutB>
__global__ __aicore__ void grouped_matmul_finalize_routing_kernel_entry(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                        GM_ADDR bias, GM_ADDR xScale, GM_ADDR groupList,
                                                                        GM_ADDR sharedInput, GM_ADDR logit,
                                                                        GM_ADDR rowIndex, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    GroupedMatmulFinalizeRoutingUt::Run<Config, AType, BType, CType, LogitType, RowIndexType, LayoutB>(
        x, weight, weightScale, bias, xScale, groupList, sharedInput, logit, rowIndex, y, tiling);
}
