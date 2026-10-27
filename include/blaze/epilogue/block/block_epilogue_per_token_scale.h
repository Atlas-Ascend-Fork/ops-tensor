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
 * \file block_epilogue_per_token_scale.h
 * \brief AIV epilogue for per-token scaling with optional row-sum offset correction.
 *
 * Computes (fixpipeResult + xRowSum * offset) * perTokenScale, then converts
 * and stores the final output. The offset term is omitted when withOffset is false.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <class OutType_, class FixpipeType_ = half>
class BlockEpiloguePerTokenScale {
public:
    using OutType = OutType_;
    using FixpipeType = FixpipeType_;
    static_assert(AscendC::IsSameType<OutType, float>::value || AscendC::IsSameType<OutType, half>::value ||
                  AscendC::IsSameType<OutType, bfloat16_t>::value);
    static_assert(AscendC::IsSameType<FixpipeType, float>::value || AscendC::IsSameType<FixpipeType, half>::value);

    struct Params {
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR perTokenScaleGmAddr{nullptr};
        GM_ADDR offsetGmAddr{nullptr};
        GM_ADDR xRowSumGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        int64_t n{0};
        int64_t baseM{0};
        int64_t baseN{0};
        bool withOffset{false};
    };

    __aicore__ inline BlockEpiloguePerTokenScale() {}
    __aicore__ inline ~BlockEpiloguePerTokenScale() {}
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()(int64_t singleM, int64_t singleN, int64_t perTokenOffset, int64_t offsetOffset,
                                      int64_t rowSumOffset, int64_t outOffset, uint64_t workspaceOffset);

private:
    static constexpr uint64_t DATA_BLOCK = 32;
    static constexpr uint32_t CV_RATIO = 2;
#ifdef __NPU_ARCH__
    static constexpr AscendC::Reg::CastTrait CAST_IN_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CAST_IN_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CAST_OUT = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};
#endif // __NPU_ARCH__

    template <class T>
    __aicore__ inline static uint64_t AlignBytes(uint64_t elements)
    {
        return Blaze::Gemm::CeilAlign(elements * sizeof(T), DATA_BLOCK);
    }

    template <class T>
    __aicore__ inline static int64_t AlignElements(int64_t elements)
    {
        return static_cast<int64_t>(AlignBytes<T>(elements) / sizeof(T));
    }

    __aicore__ inline static auto MakeLayout(int64_t rows, int64_t cols, int64_t rowPitch);
    __aicore__ inline void CopyInputs(int64_t m, int64_t n, int64_t perTokenOffset, int64_t offsetOffset,
                                      int64_t rowSumOffset, uint64_t workspaceOffset);
    template <bool WithOffset>
    __aicore__ inline void VectorCompute(int64_t m, int64_t n);
    __aicore__ inline void CopyOut(int64_t m, int64_t n, int64_t outOffset);

    GM_ADDR workspaceGmAddr_{nullptr};
    GM_ADDR perTokenScaleGmAddr_{nullptr};
    GM_ADDR offsetGmAddr_{nullptr};
    GM_ADDR xRowSumGmAddr_{nullptr};
    GM_ADDR outGmAddr_{nullptr};
    int64_t n_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    uint64_t inputOffset_{0};
    uint64_t outputOffset_{0};
    uint64_t perTokenOffset_{0};
    uint64_t rowSumOffset_{0};
    uint64_t offsetOffset_{0};
    bool withOffset_{false};
};

template <class OutType, class FixpipeType>
__aicore__ inline auto BlockEpiloguePerTokenScale<OutType, FixpipeType>::MakeLayout(int64_t rows, int64_t cols,
                                                                                    int64_t rowPitch)
{
    auto shape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<1>{}, rows),
                                     asc::te::make_shape(AscendC::Std::Int<1>{}, cols));
    auto stride = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<0>{}, rowPitch),
                                       asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
    return asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<float>>(shape,
                                                                                                          stride);
}

template <class OutType, class FixpipeType>
__aicore__ inline void BlockEpiloguePerTokenScale<OutType, FixpipeType>::Init(const Params& params)
{
    workspaceGmAddr_ = params.workspaceGmAddr;
    perTokenScaleGmAddr_ = params.perTokenScaleGmAddr;
    offsetGmAddr_ = params.offsetGmAddr;
    xRowSumGmAddr_ = params.xRowSumGmAddr;
    outGmAddr_ = params.outGmAddr;
    n_ = params.n;
    baseM_ = params.baseM;
    baseN_ = params.baseN;
    withOffset_ = params.withOffset;

    const uint64_t inputPitch = static_cast<uint64_t>(AlignElements<FixpipeType>(baseN_));
    const uint64_t outputPitch = static_cast<uint64_t>(AlignElements<OutType>(baseN_));
    // One cube core is paired with two vector subblocks. Each subblock only
    // materializes half of the M tile in UB, so reserving baseM rows here both
    // wastes UB and makes a 256 x 256 tile exceed the A5 UB budget.
    const uint64_t vectorBaseM = static_cast<uint64_t>(Blaze::Gemm::CeilDiv(baseM_, static_cast<int64_t>(CV_RATIO)));
    inputOffset_ = 0;
    outputOffset_ = AlignBytes<FixpipeType>(vectorBaseM * inputPitch);
    perTokenOffset_ = outputOffset_ + AlignBytes<OutType>(vectorBaseM * outputPitch);
    rowSumOffset_ = perTokenOffset_ + AlignBytes<float>(vectorBaseM);
    offsetOffset_ = rowSumOffset_ + AlignBytes<float>(vectorBaseM);
}

template <class OutType, class FixpipeType>
__aicore__ inline void BlockEpiloguePerTokenScale<OutType, FixpipeType>::operator()(
    int64_t singleM, int64_t singleN, int64_t perTokenOffset, int64_t offsetOffset, int64_t rowSumOffset,
    int64_t outOffset, uint64_t workspaceOffset)
{
    const int64_t firstHalfM = Blaze::Gemm::CeilDiv(singleM, static_cast<int64_t>(CV_RATIO));
    const int64_t mOffset = AscendC::GetSubBlockIdx() * firstHalfM;
    const int64_t m = AscendC::GetSubBlockIdx() == 0 ? firstHalfM : singleM - firstHalfM;
    if (m <= 0) {
        return;
    }

    CopyInputs(m, singleN, perTokenOffset + mOffset, offsetOffset, rowSumOffset + mOffset,
               workspaceOffset + mOffset * singleN);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    if (withOffset_) {
        VectorCompute<true>(m, singleN);
    } else {
        VectorCompute<false>(m, singleN);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOut(m, singleN, outOffset + mOffset * n_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

template <class OutType, class FixpipeType>
__aicore__ inline void BlockEpiloguePerTokenScale<OutType, FixpipeType>::CopyInputs(
    int64_t m, int64_t n, int64_t perTokenOffset, int64_t offsetOffset, int64_t rowSumOffset, uint64_t workspaceOffset)
{
    auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    const int64_t inputPitch = AlignElements<FixpipeType>(n);
    auto inputUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, FixpipeType>(inputOffset_),
                                        MakeLayout(m, n, inputPitch));
    auto inputGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                            reinterpret_cast<__gm__ FixpipeType*>(workspaceGmAddr_) + workspaceOffset),
                                        MakeLayout(m, n, n));
    asc::te::copy(copyGM2UB, inputUb, inputGm);

    auto perTokenUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(perTokenOffset_),
                                           MakeLayout(1, m, AlignElements<float>(m)));
    auto perTokenGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                               reinterpret_cast<__gm__ float*>(perTokenScaleGmAddr_) + perTokenOffset),
                                           MakeLayout(1, m, m));
    asc::te::copy(copyGM2UB, perTokenUb, perTokenGm);

    if (withOffset_) {
        auto rowSumUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(rowSumOffset_),
                                             MakeLayout(1, m, AlignElements<float>(m)));
        auto rowSumGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ float*>(xRowSumGmAddr_) + rowSumOffset),
                                             MakeLayout(1, m, m));
        asc::te::copy(copyGM2UB, rowSumUb, rowSumGm);

        auto offsetUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(offsetOffset_),
                                             MakeLayout(1, n, AlignElements<float>(n)));
        auto offsetGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(offsetGmAddr_) + offsetOffset),
            MakeLayout(1, n, n));
        asc::te::copy(copyGM2UB, offsetUb, offsetGm);
    }
}

template <class OutType, class FixpipeType>
__aicore__ inline void BlockEpiloguePerTokenScale<OutType, FixpipeType>::CopyOut(int64_t m, int64_t n,
                                                                                 int64_t outOffset)
{
    auto outputUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(outputOffset_),
                                         MakeLayout(m, n, AlignElements<OutType>(n)));
    auto outputGm = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(outGmAddr_) + outOffset),
        MakeLayout(m, n, n_));
    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    asc::te::copy(copyUB2GM, outputGm, outputUb);
}

template <class OutType, class FixpipeType>
template <bool WithOffset>
__aicore__ inline void BlockEpiloguePerTokenScale<OutType, FixpipeType>::VectorCompute(int64_t m, int64_t n)
{
    __ubuf__ FixpipeType* input = reinterpret_cast<__ubuf__ FixpipeType*>(asc_get_phy_buf_addr(0) + inputOffset_);
    __ubuf__ OutType* output = reinterpret_cast<__ubuf__ OutType*>(asc_get_phy_buf_addr(0) + outputOffset_);
    __ubuf__ float* perToken = reinterpret_cast<__ubuf__ float*>(asc_get_phy_buf_addr(0) + perTokenOffset_);
    __ubuf__ float* rowSum = reinterpret_cast<__ubuf__ float*>(asc_get_phy_buf_addr(0) + rowSumOffset_);
    __ubuf__ float* offset = reinterpret_cast<__ubuf__ float*>(asc_get_phy_buf_addr(0) + offsetOffset_);

    const uint32_t inputPitch = static_cast<uint32_t>(AlignElements<FixpipeType>(n));
    const uint32_t outputPitch = static_cast<uint32_t>(AlignElements<OutType>(n));
    const uint32_t elementsPerVf = asc_get_vf_len() / sizeof(float);
    const uint16_t nLoops = static_cast<uint16_t>(Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(elementsPerVf)));

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < static_cast<uint16_t>(m); ++mIdx) {
            AscendC::Reg::RegTensor<float> perTokenReg;
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(perTokenReg, perToken + mIdx);
            AscendC::Reg::RegTensor<float> rowSumReg;
            if constexpr (WithOffset) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rowSumReg, rowSum + mIdx);
            }
            uint32_t remaining = static_cast<uint32_t>(n);
            for (uint16_t nIdx = 0; nIdx < nLoops; ++nIdx) {
                const uint32_t blockOffset = nIdx * elementsPerVf;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::RegTensor<FixpipeType> inputReg;
                AscendC::Reg::DataCopy(inputReg, input + mIdx * inputPitch + blockOffset);
                AscendC::Reg::RegTensor<float> valueReg;
                if constexpr (AscendC::IsSameType<FixpipeType, float>::value) {
                    valueReg = inputReg;
                } else {
                    AscendC::Reg::RegTensor<float> oneReg;
                    AscendC::Reg::Cast<float, FixpipeType, CAST_IN_ZERO>(valueReg, inputReg, mask);
                    AscendC::Reg::Cast<float, FixpipeType, CAST_IN_ONE>(oneReg, inputReg, maskB16);
                    AscendC::Reg::Interleave(valueReg, oneReg, valueReg, oneReg);
                }
                if constexpr (WithOffset) {
                    AscendC::Reg::RegTensor<float> offsetReg;
                    AscendC::Reg::RegTensor<float> correctionReg;
                    AscendC::Reg::DataCopy(offsetReg, offset + blockOffset);
                    AscendC::Reg::Mul(correctionReg, offsetReg, rowSumReg, mask);
                    AscendC::Reg::Add(valueReg, valueReg, correctionReg, mask);
                }
                AscendC::Reg::Mul(valueReg, valueReg, perTokenReg, mask);
                if constexpr (AscendC::IsSameType<OutType, float>::value) {
                    AscendC::Reg::DataCopy<OutType, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                        output + mIdx * outputPitch + blockOffset, valueReg, mask);
                } else {
                    AscendC::Reg::RegTensor<OutType> outReg;
                    AscendC::Reg::Cast<OutType, float, CAST_OUT>(outReg, valueReg, mask);
                    AscendC::Reg::DataCopy<OutType, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        output + mIdx * outputPitch + blockOffset, outReg, mask);
                }
            }
        }
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
