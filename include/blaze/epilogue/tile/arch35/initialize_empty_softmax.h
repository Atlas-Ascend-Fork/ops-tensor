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
 * \file initialize_empty_softmax.h
 * \brief Initialize empty softmax statistics in UB.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

namespace Blaze::Epilogue::Tile {

class InitializeEmptySoftmax {
public:
    template <typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(const MaxTensor& maxTensor, const SumTensor& sumTensor)
    {
        using MaxElementType = asc::te::get_attribute_element_type<typename MaxTensor::element_type*>;
        using SumElementType = asc::te::get_attribute_element_type<typename SumTensor::element_type*>;
        using MaxLayoutPattern = asc::te::get_layout_pattern<typename MaxTensor::layout_type>;
        using SumLayoutPattern = asc::te::get_layout_pattern<typename SumTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
                      "InitializeEmptySoftmax only supports FP32 tensors.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<MaxTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<SumTensor>, asc::te::location::ub>,
                      "InitializeEmptySoftmax only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<MaxLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<SumLayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "InitializeEmptySoftmax requires NDExt tensor layouts.");

        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.data().get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.data().get());
        asc_vf_call<InitializeEmptySoftmaxVf>(maxAddr, sumAddr);
    }

private:
    static constexpr float FP32_LOWEST_FINITE = -__FLT_MAX__;

    static __simd_vf__ inline void InitializeEmptySoftmaxVf(__ubuf__ float* maxAddr, __ubuf__ float* sumAddr)
    {
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
        AscendC::Reg::Duplicate(sumReg, 1.0F, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace Blaze::Epilogue::Tile
