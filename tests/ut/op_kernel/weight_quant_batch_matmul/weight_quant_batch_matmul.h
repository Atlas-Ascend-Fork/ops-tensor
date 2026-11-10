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
 * \file weight_quant_batch_matmul.h
 * \brief Kernel UT assembly for the B8 ND weight-antiquant GemmUniversal specialization.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "wqmm_antiquant_cpu_debug.h"
#include "blaze/gemm/block/block_scheduler_wqmm.h"

namespace WeightQuantBatchMatmulUT {

template <class XType, class WeightType, class ScaleType, class YType, uint32_t UbInner, uint32_t UbBufferNum,
          bool TransB, Blaze::Gemm::QuantMode AntiquantType, bool HasOffset, bool BiasFp32>
struct Components {
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightAntiquant<2, UbInner, UbBufferNum, AntiquantType, HasOffset>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>;
    using LayoutScale = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = AscendC::Std::conditional_t<BiasFp32, float, XType>;
    using BTypes = AscendC::Std::tuple<WeightType, ScaleType>;
    using BLayouts = AscendC::Std::tuple<LayoutB, LayoutScale>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, XType, LayoutA, BTypes, BLayouts, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmTailResplit<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, Scheduler>;
};

using Blaze::Gemm::QuantMode;
using Fp16Int8 = Components<half, int8_t, half, half, 512, 2, true, QuantMode::PERCHANNEL_MODE, false, false>;
using Fp16Int8Offset = Components<half, int8_t, half, half, 256, 4, false, QuantMode::PERTENSOR_MODE, true, true>;
using Bf16Int8 = Components<bfloat16_t, int8_t, bfloat16_t, bfloat16_t, 1024, 2, true, QuantMode::PERCHANNEL_MODE, true,
                            false>;
using Fp16Fp8 = Components<half, float8_e4m3_t, half, half, 512, 4, false, QuantMode::PERCHANNEL_MODE, false, false>;
using Bf16Fp8 = Components<bfloat16_t, float8_e4m3_t, bfloat16_t, bfloat16_t, 512, 2, true, QuantMode::PERTENSOR_MODE,
                           false, true>;
using Fp16HiFloat8 = Components<half, hifloat8_t, half, half, 1024, 2, true, QuantMode::PERCHANNEL_MODE, true, false>;
using Bf16HiFloat8 = Components<bfloat16_t, hifloat8_t, bfloat16_t, bfloat16_t, 256, 4, false,
                                QuantMode::PERTENSOR_MODE, false, false>;

} // namespace WeightQuantBatchMatmulUT

template <class Components>
__global__ __aicore__ void weight_quant_batch_matmul_kernel_entry(GM_ADDR xGm, GM_ADDR weightGm, GM_ADDR scaleGm,
                                                                  GM_ADDR offsetGm, GM_ADDR biasGm, GM_ADDR yGm,
                                                                  GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using Kernel = typename Components::Kernel;
    auto params = *reinterpret_cast<const typename Kernel::Params*>(tilingGm);
    params.mmadParams.aGmAddr = xGm;
    params.mmadParams.cGmAddr = yGm;
    params.mmadParams.biasGmAddr = biasGm;
    params.prologueParams.bGmAddr = weightGm;
    params.prologueParams.scaleGmAddr = scaleGm;
    params.prologueParams.offsetGmAddr = offsetGm;
    Kernel{}(params);
}
