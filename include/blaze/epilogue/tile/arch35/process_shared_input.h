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
 * \file process_shared_input.h
 * \brief Weighted shared-input initialization for finalize routing.
 */
#pragma once

#include <cstdint>
#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze::Epilogue::Tile {

// Weighted shared-input initialization, including GM writeback. The caller owns
// core partitioning and clearing the rest of the output.
template <class OutType_, bool DoubleBuffered_ = true>
class ProcessSharedInput {
private:
    using OutType = OutType_;
#ifdef __NPU_ARCH__
    static constexpr AscendC::Reg::CastTrait CAST_BF16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
#endif // __NPU_ARCH__
    static constexpr uint32_t BF16_MANTISSA_SHIFT = 16;
    static constexpr uint32_t BF16_RNE_BIAS = 0x7FFFU;

public:
    static_assert(AscendC::Std::is_one_of_v<OutType, float, bfloat16_t>,
                  "ProcessSharedInput supports float/bfloat16_t output.");

    struct Buffers {
        uint64_t inputPing;
        uint64_t inputPong;
        uint64_t outputPing;
        uint64_t outputPong;
        uint64_t chunkElements;
    };

    template <class InputTensor, class OutputTensor>
    __aicore__ inline static void Run(const InputTensor& input, const OutputTensor& output, float weight,
                                      const Buffers& buffers)
    {
        using InputType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using OutputType = asc::te::get_attribute_element_type<typename OutputTensor::element_type*>;
        static_assert(AscendC::Std::is_same_v<InputType, bfloat16_t>, "Shared input tensor must use bfloat16_t.");
        static_assert(AscendC::Std::is_same_v<OutputType, OutType>,
                      "Output tensor must match ProcessSharedInput type.");
        if ASCEND_IS_AIC {
            return;
        }
        const uint64_t count = asc::te::get<0>(input.layout().shape());
        uint32_t slot = 0;
        for (uint64_t offset = 0; offset < count; offset += buffers.chunkElements) {
            const uint64_t current = Blaze::Gemm::Min(buffers.chunkElements, count - offset);
            const auto coord = asc::te::make_coord(offset);
            const auto shape = asc::te::make_shape(current);
            auto inputTile = input.slice(coord, shape);
            auto outputTile = output.slice(coord, shape);
            const auto ubLayout = asc::te::make_layout(shape, asc::te::make_stride(1L));
            const uint64_t inputUbOffset = slot == 0 ? buffers.inputPing : buffers.inputPong;
            const uint64_t outputUbOffset = slot == 0 ? buffers.outputPing : buffers.outputPong;
            auto inputUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(inputUbOffset),
                                                ubLayout);
            auto outputUb = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, OutputType>(outputUbOffset), ubLayout);
            if constexpr (DoubleBuffered_) {
                const Blaze::Gemm::BufferSlot bufferSlot{0UL, static_cast<uint8_t>(slot)};
                {
                    auto lock = bufferSlot.LockMte2();
                    asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), inputUb, inputTile);
                }
                {
                    auto lock = bufferSlot.LockV();
                    Compute(outputUb, inputUb, current, weight);
                }
                {
                    auto lock = bufferSlot.LockMte3();
                    asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), outputTile, outputUb);
                }
                slot = (slot + 1) & 1;
            } else {
                asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), inputUb, inputTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                Compute(outputUb, inputUb, current, weight);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), outputTile, outputUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            }
        }
    }

private:
    __aicore__ inline static uint16_t FloatToBf16Bits(float value)
    {
        union {
            float f;
            uint32_t u;
        } converter{value};
        const uint32_t roundingBias = BF16_RNE_BIAS + ((converter.u >> BF16_MANTISSA_SHIFT) & 1U);
        return static_cast<uint16_t>((converter.u + roundingBias) >> BF16_MANTISSA_SHIFT);
    }

    template <class OutputTensor, class InputTensor>
    __aicore__ inline static void Compute(const OutputTensor& output, const InputTensor& input, uint64_t count,
                                          float weight)
    {
        constexpr uint64_t VECTOR_LENGTH = asc_get_vf_len() / sizeof(float);
        const uint16_t loops = Blaze::Gemm::CeilDiv(count, VECTOR_LENGTH);
        asc_vf_call<VfDoSharedCastAndMuls>(reinterpret_cast<__ubuf__ OutType*>(output.data().get()),
                                           reinterpret_cast<__ubuf__ bfloat16_t*>(input.data().get()), weight,
                                           FloatToBf16Bits(weight), loops, count);
    }

    static __simd_vf__ inline void VfDoSharedCastAndMuls(__ubuf__ OutType* dstPtr,
                                                         __ubuf__ bfloat16_t* residualLocalUbAddr, float sharedWeight,
                                                         uint16_t sharedWeightBf16, uint16_t loopNum,
                                                         uint64_t curCount);
};
#ifdef __NPU_ARCH__
template <class OutType_, bool DoubleBuffered_>
__simd_vf__ inline void ProcessSharedInput<OutType_, DoubleBuffered_>::VfDoSharedCastAndMuls(
    __ubuf__ OutType* dstPtr, __ubuf__ bfloat16_t* residualLocalUbAddr, float sharedWeight, uint16_t sharedWeightBf16,
    uint16_t loopNum, uint64_t curCount)
{
    constexpr uint32_t VECTOR_REG_LEN = AscendC::IsSameType<OutType, bfloat16_t>::value ?
                                            asc_get_vf_len() / sizeof(float) :
                                            asc_get_vf_len() / sizeof(OutType);
    uint32_t elementNum = static_cast<uint32_t>(curCount);
    AscendC::Reg::MaskReg mask;
    uint16_t oneRepeatSize = VECTOR_REG_LEN;
    if constexpr (AscendC::IsSameType<OutType, bfloat16_t>::value) {
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(elementNum);
            AscendC::Reg::RegTensor<bfloat16_t> vSrcReg;
            AscendC::Reg::RegTensor<uint16_t> vSharedWeightReg;
            AscendC::Reg::RegTensor<bfloat16_t> vDstReg;
            AscendC::Reg::Duplicate<uint16_t, AscendC::Reg::MaskMergeMode::ZEROING>(vSharedWeightReg, sharedWeightBf16,
                                                                                    mask);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vSrcReg, residualLocalUbAddr + i * oneRepeatSize);
            AscendC::Reg::Mul<bfloat16_t, AscendC::Reg::MaskMergeMode::ZEROING>(
                vDstReg, vSrcReg, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(vSharedWeightReg), mask);
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstPtr + i * oneRepeatSize,
                                                                                         vDstReg, mask);
        }
    } else {
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<OutType>(elementNum);
            AscendC::Reg::RegTensor<bfloat16_t> vSrcReg;
            AscendC::Reg::RegTensor<OutType> vRegCast;
            AscendC::Reg::RegTensor<OutType> vDstReg;
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vSrcReg, residualLocalUbAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<OutType, bfloat16_t, CAST_BF16_TO_FP32>(vRegCast, vSrcReg, mask);
            AscendC::Reg::Muls(vDstReg, vRegCast, sharedWeight, mask);
            AscendC::Reg::StoreAlign<OutType, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstPtr + i * oneRepeatSize,
                                                                                      vDstReg, mask);
        }
    }
}
#endif // __NPU_ARCH__
} // namespace Blaze::Epilogue::Tile
