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
 * \file sigmoid.h
 * \brief Tile-level sigmoid for FP32 UB input and float/bfloat16/half output.
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Epilogue::Tile {

class Sigmoid {
public:
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline static void run(const SrcTensor& srcTensor, const DstTensor& dstTensor);

private:
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    static constexpr AscendC::Reg::DivSpecificMode SIGMOID_DIV_MODE = {
        AscendC::Reg::MaskMergeMode::ZEROING,
        true,
    };

    template <typename DataTypeOut>
    struct SigmoidVfParams {
        __ubuf__ float* srcAddr;
        __ubuf__ DataTypeOut* dstAddr;
        uint32_t mSize;
        uint32_t nSize;
        uint32_t srcRowStride;
        uint32_t dstRowStride;
        uint32_t sizePerRepeat;
        uint32_t oneRowRepeatTimes;
    };

    template <typename DataTypeOut>
    static __simd_vf__ inline void SigmoidVf(SigmoidVfParams<DataTypeOut> params)
    {
        AscendC::Reg::RegTensor<float> numerator;
        AscendC::Reg::Duplicate(numerator, 1.0F);
        for (uint32_t row = 0; row < params.mSize; ++row) {
            __ubuf__ float* inputRow = params.srcAddr + row * params.srcRowStride;
            __ubuf__ DataTypeOut* outputRow = params.dstAddr + row * params.dstRowStride;
            for (uint32_t i = 0; i < params.oneRowRepeatTimes; ++i) {
                const uint32_t col = i * params.sizePerRepeat;
                const uint32_t remain = params.nSize - col;
                uint32_t valid = remain < params.sizePerRepeat ? remain : params.sizePerRepeat;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(valid);

                AscendC::Reg::RegTensor<float> input;
                AscendC::Reg::RegTensor<float> negativeInput;
                AscendC::Reg::RegTensor<float> exponential;
                AscendC::Reg::RegTensor<float> denominator;
                AscendC::Reg::RegTensor<float> result;
                AscendC::Reg::LoadAlign(input, inputRow + col);
                AscendC::Reg::Muls(negativeInput, input, -1.0F, mask);
                AscendC::Reg::Exp(exponential, negativeInput, mask);
                AscendC::Reg::Adds(denominator, exponential, 1.0F, mask);
                AscendC::Reg::Div<float, &SIGMOID_DIV_MODE>(result, numerator, denominator, mask);

                if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
                    AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_NORM_B32>(outputRow + col,
                                                                                                  result, mask);
                } else {
                    AscendC::Reg::RegTensor<DataTypeOut> output;
                    AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_16F>(output, result, mask);
                    AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(outputRow + col,
                                                                                                  output, mask);
                }
            }
        }
    }
};

template <typename SrcTensor, typename DstTensor>
__aicore__ inline void Sigmoid::run(const SrcTensor& srcTensor, const DstTensor& dstTensor)
{
    using DataTypeIn = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DataTypeOut = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float>, "Sigmoid input must be float.");
    static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeOut, half>,
                  "Sigmoid output must be float, bfloat16_t or half.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                  "Sigmoid only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "Sigmoid only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t mSize = static_cast<uint32_t>(asc::te::get<0, 1>(srcTensor.layout().shape()));
    const uint32_t nSize = static_cast<uint32_t>(asc::te::get<1, 1>(srcTensor.layout().shape()));
    constexpr uint32_t sizePerRepeat = static_cast<uint32_t>(asc_get_vf_len() / sizeof(float));
    const uint32_t oneRowRepeatTimes = (nSize + sizePerRepeat - 1U) / sizePerRepeat;
    const uint32_t srcRowStride = static_cast<uint32_t>(asc::te::get<0, 1>(srcTensor.layout().stride()));
    const uint32_t dstRowStride = static_cast<uint32_t>(asc::te::get<0, 1>(dstTensor.layout().stride()));
    SigmoidVfParams<DataTypeOut> params{reinterpret_cast<__ubuf__ float*>(srcTensor.data().get()),
                                        reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get()),
                                        mSize,
                                        nSize,
                                        srcRowStride,
                                        dstRowStride,
                                        sizePerRepeat,
                                        oneRowRepeatTimes};
    asc_vf_call<SigmoidVf<DataTypeOut>>(params);
}

} // namespace Blaze::Epilogue::Tile
