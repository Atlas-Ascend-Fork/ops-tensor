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
 * \file flat_quant.h
 * \brief FlatQuant Blaze kernel UT assembly and smoke-entry.
 *
 * The production epilogue (BlockEpilogueFlatQuant) contains __simd_vf__ MicroAPI Cast
 * instructions (fp4x2_e2m1_t) that trigger a GCC 14 ICE when expanded by the host-side
 * C++ compiler. The real epilogue is kept in FlatQuantBlazeTypes for assembly-contract
 * static_assert checks; a no-op KernelSmokeEpilogue is used for KERNEL_RUN_KF smoke
 * execution of the real scheduler, BlockMmad, and kernel control flow.
 */

#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

using AscendC::GatherMaskParams;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;

#include "blaze/attention/block/block_mmad.h"
#include "blaze/attention/block/block_scheduler_flat_quant.h"
#include "blaze/attention/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_flat_quant.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"

using Blaze::Gemm::CeilAlign;

#include "blaze/attention/kernel/kernel_universal.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

template <typename T_>
struct UtMatmulType {
    using T = T_;
};

#pragma pack(push, 1)
struct FlatQuantTilingData {
    uint8_t dataType = 0;
    uint8_t hasP2 = 1;
    int64_t K = 0;
    int64_t M = 0;
    int64_t N = 0;
    int64_t iterBatch = 1;
    float clipRatio = 1.0f;
    float dstTypeMax = 0.0f;
    float invDstTypeMax = 0.0f;
    int64_t groupNum = 0;
    int64_t groupListType = 0;
};
#pragma pack(pop)

namespace FlatQuantUT {

template <typename DataTypeOut_, typename DataTypeIn_ = float, typename DataTypeScale_ = float>
class KernelSmokeEpilogue {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeScale = DataTypeScale_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        ProblemShape problemShape{};
        float dstTypeMax{0.0f};
        float invDstTypeMax{0.0f};
    };

    __aicore__ inline void Init(const Params&) {}
    __aicore__ inline void operator()(uint64_t, uint64_t) {}
};

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT>
struct FlatQuantBlazeTypes {
    using AType = X_TYPE;
    using BType = X_TYPE;
    using BiasType = SCALE_TYPE;
    using OutType = Y_TYPE;

    using LayoutA = C_LAYOUT;
    using LayoutB = C_LAYOUT;
    using LayoutC = C_LAYOUT;

    using AMatmulType = UtMatmulType<AType>;
    using BMatmulType = UtMatmulType<BType>;
    using CMatmulType = UtMatmulType<OutType>;
    using BiasMatmulType = UtMatmulType<BiasType>;

    using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<
        asc::te::shape<int64_t, int64_t, int64_t, int64_t>>;
    using DispatchPolicy = Blaze::Attention::BlockFlatQuant<Blaze::Attention::KernelFlatQuant>;
    using BlockMmad = Blaze::Attention::Block::BlockMmad<DispatchPolicy, AMatmulType, LayoutA, BMatmulType, LayoutB,
                                                         BiasMatmulType, LayoutC, CMatmulType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<AType, OutType, BiasType>;
    using FusionOp = Blaze::Epilogue::Fusion::DefaultFusion<OutType, AType>;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MatmulKernel = Blaze::Attention::Kernel::AttentionUniversal<ProblemShape, BlockMmad, BlockEpilogue,
                                                                      BlockScheduler>;
};

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT>
struct KernelSmokeTypes {
    using PublicTypes = FlatQuantBlazeTypes<X_TYPE, Y_TYPE, SCALE_TYPE, C_LAYOUT>;
    using BlockEpilogue = KernelSmokeEpilogue<Y_TYPE, X_TYPE, SCALE_TYPE>;
    using Kernel = Blaze::Attention::Kernel::AttentionUniversal<typename PublicTypes::ProblemShape,
                                                                typename PublicTypes::BlockMmad, BlockEpilogue,
                                                                typename PublicTypes::BlockScheduler>;
};

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT>
__aicore__ inline void RunFlatQuantBlaze(GM_ADDR aGM, GM_ADDR p1GM, GM_ADDR p2GM, GM_ADDR cGM, GM_ADDR scaleGM,
                                         GM_ADDR workspaceGM, const FlatQuantTilingData& tilingData)
{
    using Types = KernelSmokeTypes<X_TYPE, Y_TYPE, SCALE_TYPE, C_LAYOUT>;
    using Kernel = typename Types::Kernel;
    using Params = typename Kernel::Params;

    typename Kernel::BlockScheduler::Params schParams;
    schParams.iterBatch = tilingData.iterBatch;
    schParams.dstTypeMax = tilingData.dstTypeMax;
    schParams.invDstTypeMax = tilingData.invDstTypeMax;

    constexpr int64_t BASE_K = 64;
    int64_t M = tilingData.M;
    int64_t N = tilingData.N;
    int64_t K = tilingData.K;
    int64_t iterBatch = tilingData.iterBatch;

    Params params = {{M, N, N, K},
                     {aGM,
                      p1GM,
                      p2GM,
                      {M, N, N, K},
                      {M * iterBatch, N, N, iterBatch},
                      {M * iterBatch, N, BASE_K, 1},
                      tilingData.hasP2 == 1},
                     {cGM, scaleGM, {M, N, N, K}, tilingData.dstTypeMax, tilingData.invDstTypeMax},
                     schParams};

    Kernel mm;
    mm(params);
}
} // namespace FlatQuantUT

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT = asc::te::nd_ext_layout_ptn>
__global__ __aicore__ void FlatQuantKernelEntry(GM_ADDR x, GM_ADDR p1, GM_ADDR p2, GM_ADDR out, GM_ADDR quant_scale,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    FlatQuantUT::RunFlatQuantBlaze<X_TYPE, Y_TYPE, SCALE_TYPE, C_LAYOUT>(
        x, p1, p2, out, quant_scale, workspace, *reinterpret_cast<const FlatQuantTilingData*>(tiling));
}
