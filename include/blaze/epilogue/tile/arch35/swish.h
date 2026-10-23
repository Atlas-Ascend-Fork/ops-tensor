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
 * \file swish.h
 * \brief Tile-level FP32 Swish with BF16 RINT output and MX tail padding.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Epilogue::Tile {

class Swish {
public:
    __aicore__ inline Swish() = default;

    // beta is a finite scalar shared by all elements; beta=1 gives SiLU.
    // Both UB tensors have the same logical shape, read from srcTensor's layout.
    // UB tensors use Align32(nSize) elements per row. Output padding is zeroed.
    // LoadAlign reads complete 64-float vectors, including on the last row.
    // The caller owns readable input padding, buffers and synchronization.
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline static void Run(const SrcTensor& srcTensor, const DstTensor& dstTensor, float beta = 1.0F);

private:
    static constexpr uint32_t ROW_ALIGNMENT = 32;
    static constexpr uint32_t ROW_MASK = ROW_ALIGNMENT - 1;
    static constexpr float SCALAR_ONE = 1.0F;
#ifdef __NPU_ARCH__
    static constexpr AscendC::Reg::CastTrait SWISH_FP32_TO_BF16_CAST_TRAIT = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
#endif

    static __simd_callee__ inline void SwishScalarCallee(__ubuf__ float* inputAddr,
                                                         __ubuf__ bfloat16_t* swishOutputAddr,
                                                         AscendC::Reg::MaskReg& computeMask,
                                                         AscendC::Reg::MaskReg& storeMask, float negBeta)
    {
        AscendC::Reg::RegTensor<float> input, denominator, swishResult;
        AscendC::Reg::RegTensor<bfloat16_t> swishResultBf16;
        AscendC::Reg::LoadAlign(input, inputAddr);
        AscendC::Reg::Muls(denominator, input, negBeta, computeMask);
        AscendC::Reg::Exp(denominator, denominator, computeMask);
        AscendC::Reg::Adds(denominator, denominator, SCALAR_ONE, computeMask);
        AscendC::Reg::Div(swishResult, input, denominator, computeMask);
        // Cast clears masked-out destination lanes; storeMask writes these zeros to UB padding.
        AscendC::Reg::Cast<bfloat16_t, float, SWISH_FP32_TO_BF16_CAST_TRAIT>(swishResultBf16, swishResult, computeMask);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(swishOutputAddr, swishResultBf16,
                                                                                     storeMask);
    }

    static __simd_vf__ inline void SwishScalarVf(__ubuf__ float* inputAddr, __ubuf__ bfloat16_t* swishOutputAddr,
                                                 uint16_t rowCount, uint64_t rowStride, uint16_t fullRepeats,
                                                 uint16_t tailRepeats, uint32_t tailCount, uint32_t tailAlignedCount,
                                                 float negBeta)
    {
        auto fullMask = AscendC::Reg::CreateMask<float>();
        auto tailMask = AscendC::Reg::UpdateMask<float>(tailCount);
        auto tailStoreMask = AscendC::Reg::UpdateMask<float>(tailAlignedCount);
        constexpr uint32_t fp32ElementsPerRepeat = asc_get_vf_len() / sizeof(float);
        const uint64_t tailOffset = static_cast<uint64_t>(fullRepeats) * fp32ElementsPerRepeat;
        for (uint16_t row = 0; row < rowCount; ++row) {
            for (uint16_t repeat = 0; repeat < fullRepeats; ++repeat) {
                const uint64_t offset = row * rowStride + static_cast<uint64_t>(repeat) * fp32ElementsPerRepeat;
                SwishScalarCallee(inputAddr + offset, swishOutputAddr + offset, fullMask, fullMask, negBeta);
            }
            // A 0/1 hardware loop keeps runtime branches out of the row loop.
            for (uint16_t tail = 0; tail < tailRepeats; ++tail) {
                const uint64_t offset = row * rowStride + tailOffset;
                SwishScalarCallee(inputAddr + offset, swishOutputAddr + offset, tailMask, tailStoreMask, negBeta);
            }
        }
    }
};

template <typename SrcTensor, typename DstTensor>
__aicore__ inline void Swish::Run(const SrcTensor& srcTensor, const DstTensor& dstTensor, float beta)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, float>, "Swish only supports float input.");
    static_assert(AscendC::Std::is_same_v<DstElementType, bfloat16_t>, "Swish only supports bfloat16_t output.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                  "Swish only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "Swish only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    const uint16_t mSize = static_cast<uint16_t>(asc::te::get_total_row_shape(srcTensor.layout()));
    const uint32_t nSize = static_cast<uint32_t>(asc::te::get_total_column_shape(srcTensor.layout()));
    const uint32_t nAligned = (nSize + ROW_MASK) & ~ROW_MASK;
    const uint32_t elementCount = static_cast<uint32_t>(mSize) * nAligned;
    // Without row padding, process the entire tile as one contiguous span.
    const bool contiguous = nSize == nAligned;
    const uint16_t rowCount = contiguous ? 1 : mSize;
    const uint32_t rowSize = contiguous ? elementCount : nSize;
    const uint64_t rowStride = contiguous ? rowSize : nAligned;
    constexpr uint32_t fp32ElementsPerRepeat = asc_get_vf_len() / sizeof(SrcElementType);
    const uint16_t fullRepeats = static_cast<uint16_t>(rowSize / fp32ElementsPerRepeat);
    const uint32_t tailCount = rowSize % fp32ElementsPerRepeat;
    const uint16_t tailRepeats = static_cast<uint16_t>(tailCount != 0);
    const uint32_t tailAlignedCount = (tailCount + ROW_MASK) & ~ROW_MASK;
    auto* inputAddr = reinterpret_cast<__ubuf__ SrcElementType*>(srcTensor.data().get());
    auto* swishOutputAddr = reinterpret_cast<__ubuf__ DstElementType*>(dstTensor.data().get());
    const float negBeta = -beta;
    asc_vf_call<SwishScalarVf>(inputAddr, swishOutputAddr, rowCount, rowStride, fullRepeats, tailRepeats, tailCount,
                               tailAlignedCount, negBeta);
}

} // namespace Blaze::Epilogue::Tile
