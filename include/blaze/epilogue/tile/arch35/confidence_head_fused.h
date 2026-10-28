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
 * \brief Tile-level temperature, clamp and sigmoid processing for FP32 logits.
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Epilogue::Tile {

#ifdef __NPU_ARCH__
class ConfidenceHeadFused {
public:
    struct Params {
        float negInvTemperature{-1.0F};
        float lowLimit{0.0F};
        float highLimit{0.0F};
        bool hasLowLimit{false};
        bool hasHighLimit{false};
    };

    template <bool ZeroInput = false, typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor, const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        using SrcType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
        using DstType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
        static_assert(AscendC::Std::is_same_v<SrcType, float> && AscendC::Std::is_same_v<DstType, float>,
                      "ConfidenceHeadFused requires float input and output.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                      "ConfidenceHeadFused only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_layout_pattern<typename SrcTensor::layout_type>,
                                              asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<asc::te::get_layout_pattern<typename DstTensor::layout_type>,
                                                  asc::te::nd_ext_layout_ptn>,
                      "ConfidenceHeadFused only supports NDExt tensor layouts.");
        constexpr uint32_t sizePerRepeat = static_cast<uint32_t>(asc_get_vf_len() / sizeof(float));
        const uint32_t mSize = static_cast<uint32_t>(asc::te::get<0, 1>(srcTensor.layout().shape()));
        const uint32_t nSize = static_cast<uint32_t>(asc::te::get<1, 1>(srcTensor.layout().shape()));
        VfParams vfParams{reinterpret_cast<__ubuf__ float*>(srcTensor.data().get()),
                          reinterpret_cast<__ubuf__ float*>(dstTensor.data().get()),
                          mSize,
                          nSize,
                          static_cast<uint32_t>(asc::te::get<0, 1>(srcTensor.layout().stride())),
                          static_cast<uint32_t>(asc::te::get<0, 1>(dstTensor.layout().stride())),
                          sizePerRepeat,
                          (nSize + sizePerRepeat - 1U) / sizePerRepeat,
                          params};
        asc_vf_call<Compute<ZeroInput>>(vfParams);
    }

private:
    struct VfParams {
        __ubuf__ float* srcAddr;
        __ubuf__ float* dstAddr;
        uint32_t mSize;
        uint32_t nSize;
        uint32_t srcRowStride;
        uint32_t dstRowStride;
        uint32_t sizePerRepeat;
        uint32_t oneRowRepeatTimes;
        Params params;
    };

    template <bool ZeroInput>
    static __simd_vf__ inline void Compute(VfParams params)
    {
        AscendC::Reg::RegTensor<float> numerator;
        AscendC::Reg::Duplicate(numerator, 1.0F);
        for (uint32_t row = 0; row < params.mSize; ++row) {
            for (uint32_t i = 0; i < params.oneRowRepeatTimes; ++i) {
                const uint32_t col = i * params.sizePerRepeat;
                const uint32_t remain = params.nSize - col;
                uint32_t valid = remain < params.sizePerRepeat ? remain : params.sizePerRepeat;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(valid);
                AscendC::Reg::RegTensor<float> input;
                AscendC::Reg::RegTensor<float> scaled;
                AscendC::Reg::RegTensor<float> lowerClamped;
                AscendC::Reg::RegTensor<float> upperClamped;
                AscendC::Reg::RegTensor<float> exponential;
                AscendC::Reg::RegTensor<float> denominator;
                AscendC::Reg::RegTensor<float> result;
                if constexpr (ZeroInput) {
                    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING>(input, 0.0F, mask);
                } else {
                    AscendC::Reg::LoadAlign(input, params.srcAddr + row * params.srcRowStride + col);
                }
                AscendC::Reg::Muls(scaled, input, params.params.negInvTemperature, mask);
                if (params.params.hasLowLimit) {
                    AscendC::Reg::Maxs(lowerClamped, scaled, params.params.lowLimit, mask);
                } else {
                    lowerClamped = scaled;
                }
                if (params.params.hasHighLimit) {
                    AscendC::Reg::Mins(upperClamped, lowerClamped, params.params.highLimit, mask);
                } else {
                    upperClamped = lowerClamped;
                }
                AscendC::Reg::Exp(exponential, upperClamped, mask);
                AscendC::Reg::Adds(denominator, exponential, 1.0F, mask);
                // Use default non-precise Div after temperature scaling and clamp.
                AscendC::Reg::Div(result, numerator, denominator, mask);
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    params.dstAddr + row * params.dstRowStride + col, result, mask);
            }
        }
    }
};
#endif // __NPU_ARCH__

} // namespace Blaze::Epilogue::Tile
