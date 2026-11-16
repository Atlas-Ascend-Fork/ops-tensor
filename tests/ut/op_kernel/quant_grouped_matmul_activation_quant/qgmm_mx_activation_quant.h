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
 * \file qgmm_mx_activation_quant.h
 * \brief QGMM MX GeluTanh activation-quantization public template assembly.
 */

#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"

#include "blaze/epilogue/block/block_epilogue_gelu_tanh_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_ = asc::te::nz_layout_ptn>
struct QgmmMxActivationQuantTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<
        0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMxActivationQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA, BType_, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluTanhMxQuant<OutputType_, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

namespace GMMAQUT {
// The host-side kernel UT is compiled by the system C++ compiler. Expanding the production epilogue's
// device-only MicroAPI Cast instructions in that compiler is unsupported and triggers a GCC 14 ICE. Keep the
// production epilogue in QgmmMxActivationQuantTypes for assembly-contract checks, and use this no-op epilogue only
// for KERNEL_RUN_KF smoke execution of the real grouped scheduler, BlockMmad, and kernel control flow.
template <typename DataTypeOut_, typename DataTypeIn_ = float>
class KernelSmokeEpilogue {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using OutputOffsets = Blaze::Epilogue::Block::OutputOffsets;

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t scaleAlg{0};
        float dstTypeMax{0.0f};
    };

    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    __aicore__ inline void Init(const Params&) {}
    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const
    {
        const int64_t rows = (params.rows + 1) & ~static_cast<int64_t>(1);
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rows, Blaze::Gemm::Align32(params.cols)));
    }
    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets&) {}
    __aicore__ inline void UpdateNextProblem(const ProblemShape&) {}
    __aicore__ inline void operator()(const BlockShape&, const OutputOffsets&) {}
};

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_>
struct KernelSmokeTypes {
    using PublicTypes = QgmmMxActivationQuantTypes<AType_, BType_, OutputType_, LayoutB_>;
    using BlockEpilogue = KernelSmokeEpilogue<OutputType_, float>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename PublicTypes::ProblemShape,
                                                      typename PublicTypes::BlockMmad, BlockEpilogue,
                                                      typename PublicTypes::BlockScheduler>;
};

#pragma pack(push, 8)
struct GmmaqTilingData {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
    uint32_t scaleAlg;
    float dstTypeMax;
};
#pragma pack(pop)

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_>
__aicore__ inline void RunGmmaqMx(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale, GM_ADDR groupList,
                                  GM_ADDR y, GM_ADDR yScale, const GmmaqTilingData& t)
{
    using Types = KernelSmokeTypes<AType_, BType_, OutputType_, LayoutB_>;
    using Kernel = typename Types::Kernel;
    typename Kernel::Params params{};
    params.problemShape = {t.m, t.n, t.k, 0};
    params.mmadParams = {x, weight, y, nullptr, xScale, weightScale};
    params.epilogueParams = {y, yScale, t.baseM, t.baseN, t.scaleAlg, t.dstTypeMax};
    params.groupListGmAddr = groupList;
    params.gmmParams = {
        t.groupNum,  t.m,         t.n,      t.k,     t.baseM,         t.baseN,     t.baseK,         t.kAL1,   t.kBL1,
        t.scaleKAL1, t.scaleKBL1, t.isBias, t.dbL0C, t.l1BufferStage, t.groupType, t.groupListType, t.singleW};
    Kernel kernel;
    kernel(params);
}
} // namespace GMMAQUT

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_ = asc::te::nz_layout_ptn>
__global__ __aicore__ void GmmaqMxKernelEntry(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                              GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR tiling)
{
    GMMAQUT::RunGmmaqMx<AType_, BType_, OutputType_, LayoutB_>(
        x, weight, weightScale, xScale, groupList, y, yScale,
        *reinterpret_cast<const GMMAQUT::GmmaqTilingData*>(tiling));
}
