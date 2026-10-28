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
 * \file grouped_matmul_mx_a8w4.h
 * \brief Global-kernel wrapper for grouped MX A8W4 smoke tests.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "grouped_matmul_mx_a8w4_cpu_debug_stub.h"
#include "tensor_api/tensor.h"

#if defined(ASCENDC_CPU_DEBUG)
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#define __fp4e1m2x2 fp4x2_e1m2_t
#endif
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e1m2x2
#undef __fp4e2m1x2
#undef __fp8e4m3
#endif
#include "grouped_matmul_mx_a8w4_tiling_data.h"

namespace GroupedMatmulMxA8W4UT {

template <typename WeightType_, typename OutputType_, bool IsSingleMultiSingle_>
__aicore__ inline void Run(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm, GM_ADDR scaleBGm, GM_ADDR cGm,
                           GM_ADDR groupListGm, const GroupedMatmulMxA8W4TilingData& inputTiling)
{
    using AType = fp8_e4m3fn_t;
    using BType = WeightType_;
    using ScaleType = fp8_e8m0_t;
    using CType = OutputType_;
    using BiasType = OutputType_;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithWeightQuantMx;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using ProblemShape = decltype(asc::te::make_shape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler = Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(asc::te::make_shape(0UL, 0UL,
                                                                                                         0UL))>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
        LayoutBias>;
    using BlockPrologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<AType, BType, BiasType>;
    using KernelImpl = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<ProblemShape, BlockMmad, BlockScheduler, void,
                                                                   BlockPrologue, IsSingleMultiSingle_>;

    typename BlockMmad::Params mmParams{};
    mmParams.aGmAddr = aGm;
    mmParams.scaleAGmAddr = scaleAGm;
    mmParams.scaleBGmAddr = scaleBGm;
    mmParams.biasGmAddr = biasGm;
    mmParams.cGmAddr = cGm;
    typename BlockScheduler::Params schedulerParams{inputTiling.mainBlockCount,
                                                    inputTiling.mainBlockSize,
                                                    inputTiling.firstTailBlockCount,
                                                    inputTiling.firstTailBlockSize,
                                                    inputTiling.secondTailBlockCount,
                                                    inputTiling.secondTailBlockSize,
                                                    inputTiling.coreNum,
                                                    inputTiling.cubeNumBlocksN,
                                                    inputTiling.baseM,
                                                    inputTiling.nSize};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType*>(bGm)};
    typename KernelImpl::Params params{
        asc::te::make_shape(0UL, static_cast<uint64_t>(inputTiling.kSize), static_cast<uint64_t>(inputTiling.nSize),
                            static_cast<uint64_t>(inputTiling.groupNum)),
        mmParams,
        schedulerParams,
        prologueParams,
        groupListGm,
        inputTiling.groupListType,
        inputTiling.hasBias};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace GroupedMatmulMxA8W4UT

template <typename WeightType_, typename OutputType_, bool IsSingleMultiSingle_>
__global__ __aicore__ void GroupedMatmulMxA8W4KernelEntry(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm,
                                                          GM_ADDR scaleBGm, GM_ADDR cGm, GM_ADDR groupListGm,
                                                          GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const auto* tiling = reinterpret_cast<const GroupedMatmulMxA8W4TilingData*>(tilingGm);
    GroupedMatmulMxA8W4UT::Run<WeightType_, OutputType_, IsSingleMultiSingle_>(aGm, bGm, biasGm, scaleAGm, scaleBGm,
                                                                               cGm, groupListGm, *tiling);
}
