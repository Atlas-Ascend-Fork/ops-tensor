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
 * \file quant_matmul_activation_quant.h
 * \brief QuantMatmulActivationQuant Blaze assembly types and CPU-debug epilogue setup entries.
 */

#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "quant_batch_matmul/qbmm_cpu_debug_stub.h"

#include "blaze/epilogue/block/block_epilogue_gelu_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swish_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

template <typename AType_, typename BType_, typename OutputType_, typename LayoutA_ = asc::te::nd_ext_layout_ptn,
          typename LayoutB_ = asc::te::nd_ext_layout_ptn, uint64_t FullLoadMode_ = Blaze::Gemm::NONE_FULL_LOAD_MODE>
struct QuantMatmulActivationQuantTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode_, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA, BType_, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutputType_, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode_, LayoutA,
                                                                                LayoutB, AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_, uint64_t FullLoadMode_>
struct QuantMatmulSwiGluQuantTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode_, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M, 0, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA, BType_, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutputType_, float, fp8_e8m0_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode_, LayoutA,
                                                                                LayoutB, AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename AType_, typename BType_, typename OutputType_, typename LayoutA_ = asc::te::nd_ext_layout_ptn,
          typename LayoutB_ = asc::te::nd_ext_layout_ptn, uint64_t FullLoadMode_ = Blaze::Gemm::NONE_FULL_LOAD_MODE>
struct QuantMatmulActivationQuantWithoutBatchTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode_, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA, BType_, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutputType_, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode_, LayoutA,
                                                                                LayoutB, AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename AType_, typename BType_, typename OutputType_, typename LayoutB_, uint64_t FullLoadMode_>
struct QuantMatmulSwiGluQuantWithoutBatchTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode_, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M, 0, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA, BType_, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutputType_, float, fp8_e8m0_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode_, LayoutA,
                                                                                LayoutB, AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename AType_, typename BType_, typename LayoutA_, typename LayoutB_, uint64_t FullLoadMode_>
struct QuantMatmulSwishQuantTypes {
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode_, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA_, BType_, LayoutB_, float, LayoutC,
                                                    float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<AType_, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode_, LayoutA_,
                                                                                LayoutB_, AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
};

template <typename OutputType_, uint32_t ScaleAlg_>
__global__ __aicore__ void quant_matmul_swish_quant_epilogue_setup_entry(GM_ADDR y, GM_ADDR yScale, float beta)
{
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<OutputType_, float>;
    typename Epilogue::Params params{y, yScale, 16U, 64U, ScaleAlg_, beta};
    Epilogue epilogue;
    // CPU-debug checks setup only; use the CSV example to validate device computation.
    epilogue.Init(params);
    epilogue.UpdateNextProblem({16L, 64L, 64L, 1L});
    epilogue.UpdateGlobalAddr({0L, 0L});
}

template <typename OutputType_, Blaze::Epilogue::Block::GeluAlg GeluAlg_, Blaze::Epilogue::Block::QuantAlg QuantAlg_,
          Blaze::Epilogue::Block::ROUND_MODE_FP4 RoundMode_>
__global__ __aicore__ void quant_matmul_activation_quant_epilogue_setup_entry(GM_ADDR y, GM_ADDR yScale,
                                                                              float dstTypeMax)
{
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutputType_, float>;
    typename Epilogue::Params params{y, yScale, 16U, 64U, GeluAlg_, QuantAlg_, RoundMode_, dstTypeMax};
    Epilogue epilogue;
    // CPU-debug checks setup only; use the CSV example to validate device computation.
    epilogue.Init(params);
    epilogue.UpdateNextProblem({16L, 64L, 64L, 1L});
    epilogue.UpdateGlobalAddr({0L, 0L});
}
