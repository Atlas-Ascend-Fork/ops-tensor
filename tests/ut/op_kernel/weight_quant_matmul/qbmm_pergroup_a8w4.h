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
 * \file qbmm_pergroup_a8w4.h
 * \brief T-CG per-group A8W4 Kernel UT Wrapper（AIV FP4 反量化 + AIC 量化 Fixpipe）。
 *        装配 MatmulWithWeightQuantPergroup / KernelPergroupWeightPrologue /
 *        BlockSchedulerWqmmBlockSplit / GemmUniversal 特化，并填好各子 Params。
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"

#if defined(IMPL_STD_ASCENDC_STD_INT_IMPL_H) && !defined(IMPL_TENSOR_API_UTILS_INT_IMPL_H)
#define IMPL_TENSOR_API_UTILS_INT_IMPL_H
#endif

#include "tensor_api/tensor.h"

#if defined(ASCENDC_CPU_DEBUG)
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#endif
#include "pergroup_cpu_debug.h"
#include "blaze/gemm/kernel/kernel_wqmm_mix_pergroup.h"
#include "blaze/gemm/utils/layout_utils.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e2m1x2
#undef __fp8e4m3
#endif

namespace QBMMPerGroupA8W4UT {

struct TilingData {
    int64_t mSize{0};
    int64_t nSize{0};
    int64_t kSize{0};
    uint64_t groupSize{32U};
    uint64_t nBubSize{0U};
    uint64_t kBubSize{0U};
    uint32_t cubeNumBlocksM{1U};
    uint32_t cubeNumBlocksN{1U};
    uint32_t baseM{0U};
    uint32_t baseN{0U};
    uint32_t baseK{128U};
    uint32_t iterateOrder{1U};
    uint8_t vecCoreParallel{0U};
    uint16_t al1Pingpong{2U};
    uint16_t bl1Pingpong{4U};
    uint32_t dbL0C{2U};
    int64_t kAL1Tile{0};
    int64_t kBL1Tile{0};
};

template <bool IS_WEIGHT_NZ>
__aicore__ inline void Run(GM_ADDR x1Gm, GM_ADDR x2Gm, GM_ADDR x2ScaleGm, GM_ADDR yScaleGm, GM_ADDR yGm,
                           const TilingData& tiling)
{
    using AType = fp8_e4m3fn_t;
    using BType = fp4x2_e2m1_t;
    using ScaleType = bfloat16_t;
    using CType = int8_t;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<IS_WEIGHT_NZ, asc::te::nz_layout_ptn, asc::te::dn_ext_layout_ptn>;
    using BTypeTuple = AscendC::Std::tuple<BType, ScaleType>;
    using ProductionBlockMmad = Blaze::Gemm::Block::BlockMmad<Blaze::Gemm::MatmulWithWeightQuantPergroup, AType,
                                                              LayoutA, BTypeTuple, LayoutB, CType, LayoutC, void, void>;
    using BlockMmadType = CpuDebug::ObservedBlock<ProductionBlockMmad>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
    using BlockSchedulerType = Blaze::Gemm::Block::BlockSchedulerWqmmBlockSplit<ProblemShape, LayoutB, AType>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmadType, void, BlockSchedulerType>;

    typename BlockMmadType::Params mmadParams{};
    mmadParams.aGmAddr = x1Gm;
    mmadParams.cGmAddr = yGm;
    mmadParams.yScaleGmAddr = yScaleGm;
    mmadParams.l1TileShape = asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                                                 tiling.kAL1Tile ? tiling.kAL1Tile : int64_t(tiling.baseK),
                                                 tiling.kBL1Tile ? tiling.kBL1Tile : int64_t(tiling.baseK));
    mmadParams.l0TileShape = asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                                                 static_cast<int64_t>(tiling.baseK));
    mmadParams.vecCoreParallel = tiling.vecCoreParallel;
    mmadParams.AL1Pingpong = tiling.al1Pingpong;
    mmadParams.BL1Pingpong = tiling.bl1Pingpong;
    mmadParams.dbL0C = tiling.dbL0C;

    typename KernelImpl::Params params{
        asc::te::make_shape(tiling.mSize, tiling.nSize, tiling.kSize),
        mmadParams,
        {x2Gm, x2ScaleGm, tiling.groupSize, tiling.nBubSize, tiling.kBubSize},
        {tiling.cubeNumBlocksM, tiling.cubeNumBlocksN, tiling.baseM, tiling.baseN, tiling.iterateOrder}};
    KernelImpl kernel;
    kernel(params);
}

} // namespace QBMMPerGroupA8W4UT

template <bool IS_WEIGHT_NZ>
__global__ __aicore__ void QbmmPergroupA8W4KernelEntry(GM_ADDR x1Gm, GM_ADDR x2Gm, GM_ADDR x2ScaleGm, GM_ADDR yScaleGm,
                                                       GM_ADDR yGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const auto* tiling = reinterpret_cast<const QBMMPerGroupA8W4UT::TilingData*>(tilingGm);
    QBMMPerGroupA8W4UT::Run<IS_WEIGHT_NZ>(x1Gm, x2Gm, x2ScaleGm, yScaleGm, yGm, *tiling);
}
