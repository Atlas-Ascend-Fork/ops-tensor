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
 * \file rotate_quant.h
 * \brief RotateQuant Blaze kernel UT assembly and smoke-entry.
 *
 * As in the FlatQuant UT, the real epilogue participates in public assembly checks.
 * Kernel smoke runs the real scheduler, BlockMmad and kernel control flow with a no-op
 * epilogue to avoid instantiating the MicroAPI FP4/FP8 Cast bodies in the host compiler.
 * These smoke tests do not check numerical quantization results.
 */

#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_rotate_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "blaze/gemm/kernel/kernel_universal.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace RotateQuantUT {

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
constexpr int64_t BLOCK_M = 16;
constexpr int64_t MAX_BASE_K = 64;
constexpr int64_t GROUP_LIST_CUMSUM = 0;
constexpr int64_t GROUP_LIST_COUNT = 1;
constexpr int64_t ROUND_MODE_RINT = 4;

struct RotateQuantTilingData {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t b{1};
    int64_t mL1{BLOCK_M};
    int64_t baseK{MAX_BASE_K};
    int64_t groupNum{0};
    int64_t groupListType{GROUP_LIST_COUNT};
    int64_t hasAlpha{0};
    int64_t roundMode{ROUND_MODE_RINT};
    int64_t scaleAlg{0};
    float dstTypeMax{0.0f};
    float invDstTypeMax{0.0f};
};

template <typename ElementType_>
struct UtMatmulType {
    using T = ElementType_;
};

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_>
struct RotateQuantBlazeTypes {
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerRotateQuant<ProblemShape, EnableGrouped_>;
    using DispatchPolicy = Blaze::Gemm::BlockRotateQuant<Blaze::Gemm::KernelRotateQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, UtMatmulType<XType_>, Layout_, UtMatmulType<XType_>,
                                                    Layout_, UtMatmulType<bfloat16_t>, Layout_,
                                                    UtMatmulType<ScaleType_>, Layout_>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<bfloat16_t, YType_, ScaleType_>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename YType_, typename ScaleType_>
class KernelSmokeEpilogue {
public:
    using DataTypeIn = bfloat16_t;
    using DataTypeOut = YType_;
    using DataTypeScale = ScaleType_;
    using PublicEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<DataTypeIn, DataTypeOut, DataTypeScale>;
    using Params = typename PublicEpilogue::Params;
    using BlockShape = typename PublicEpilogue::BlockShape;
    using BlockCoord = typename PublicEpilogue::BlockCoord;

    __aicore__ inline void Init(const Params&) {}
    __aicore__ inline void operator()(const BlockShape&, const BlockCoord&) {}
};

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_>
struct KernelSmokeTypes {
    using PublicTypes = RotateQuantBlazeTypes<XType_, YType_, ScaleType_, Layout_, EnableGrouped_>;
    using BlockEpilogue = KernelSmokeEpilogue<YType_, ScaleType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, typename PublicTypes::BlockMmad, BlockEpilogue,
                                                      typename PublicTypes::BlockScheduler>;
};

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_>
__aicore__ inline void RunRotateQuantBlaze(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm, GM_ADDR groupListGm,
                                           GM_ADDR yGm, GM_ADDR scaleGm, const RotateQuantTilingData& tiling)
{
    using Kernel = typename KernelSmokeTypes<XType_, YType_, ScaleType_, Layout_, EnableGrouped_>::Kernel;
    const int64_t m = tiling.m;
    const int64_t n = tiling.n;
    const int64_t k = tiling.k;
    const int64_t b = tiling.b;
    const int64_t mL1 = tiling.mL1;
    const int64_t remainder = (m * (n / k)) % mL1;
    const int64_t tailM = remainder == 0 ? mL1 : remainder;
    typename Kernel::Params params = {
        {m, n, k, b},
        {xGm, rotationGm, alphaGm, {m, n, k, b}, {mL1, k, k, 1}, {mL1, k, tiling.baseK, 1}, b},
        {yGm,
         scaleGm,
         {m, n, k, b},
         {},
         false,
         -1,
         tiling.roundMode,
         tiling.scaleAlg,
         tiling.dstTypeMax,
         tiling.invDstTypeMax},
        {mL1, k, k, mL1, k, tiling.baseK, n / k, tailM, tiling.dstTypeMax, tiling.invDstTypeMax, -1, tiling.roundMode,
         tiling.scaleAlg, tiling.hasAlpha, 0}};
    if constexpr (EnableGrouped_) {
        params.groupParams = {groupListGm, tiling.groupNum, tiling.groupListType};
    }
    Kernel kernel;
    kernel(params);
}

} // namespace RotateQuantUT

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_>
__global__ __aicore__ void RotateQuantKernelEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm, GM_ADDR groupListGm,
                                                  GM_ADDR yGm, GM_ADDR scaleGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RotateQuantUT::RunRotateQuantBlaze<XType_, YType_, ScaleType_, Layout_, EnableGrouped_>(
        xGm, rotationGm, alphaGm, groupListGm, yGm, scaleGm,
        *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm));
}
