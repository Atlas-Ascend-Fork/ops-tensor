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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief AIV epilogue for SwiGLU activation and MXFP8 output quantization.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

namespace Constant {
constexpr float DEFAULT_CLAMP_LIMIT = 7.0F;
constexpr float DEFAULT_GLU_ALPHA = 1.702F;
constexpr float DEFAULT_GLU_BIAS = 1.0F;
constexpr float RECIPROCAL_NUMERATOR = 1.0F;
constexpr uint32_t SCALE_ALG_CUBLAS = 1;
constexpr uint16_t MX_SCALE_REDUCTION_MASK = 0x003f;
constexpr uint32_t SECOND_VECTOR_SUBBLOCK = 1;
constexpr int64_t FLAT_VECTOR_ROW_COUNT = 1;
constexpr float SIGMOID_DENOMINATOR_OFFSET = 1.0F;
constexpr uint32_t SWIGLU_HALF_COUNT = 2;
constexpr uint32_t VECTOR_SUBBLOCK_COUNT = 2;
constexpr uint64_t MX_QUANT_COMPUTE_ALIGN = 64UL;
constexpr uint32_t MAX_SINGLE_MN = 64 * 256;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;
} // namespace Constant

#ifdef __NPU_ARCH__
constexpr AscendC::Reg::CastTrait CT_FP32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::DivSpecificMode DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};
#endif // __NPU_ARCH__

/*!
 * \brief Consumes concatenated left/right MMAD results, applies SwiGLU, and writes quantized Y/YScale.
 *
 * This shared component retains mode 0 for the legacy V2 path. The V3 integration uses mode 2 only and validates
 * scaleAlg (0/1) plus dstTypeMax (0) in the upper layer. Required output pointers must be checked before launch;
 * this low-level device component has no ACLNN error return.
 *
 * \tparam DataTypeOut_ Quantized output type.
 * \tparam DataTypeIn_ MMAD result type, currently float for the fused QGMM kernel.
 * \tparam DataTypeScale_ Output scale type, normally fp8_e8m0_t.
 */
template <typename DataTypeOut_, typename DataTypeIn_ = float, typename DataTypeScale_ = fp8_e8m0_t>
class BlockEpilogueSwigluMxQuant {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeScale = DataTypeScale_;
    // Legacy APT also instantiates inactive FP4 combinations. Validate execution dtypes in the upper-layer route.
    static constexpr uint64_t INPUT_UB_TILE_ELEMENTS = Constant::MAX_SINGLE_MN;
    static constexpr uint64_t INPUT_UB_BUFFER_BYTES = INPUT_UB_TILE_ELEMENTS * sizeof(DataTypeIn);
    static constexpr uint64_t OUTPUT_C0_SIZE = asc::te::c0_element<DataTypeIn>;
    static constexpr uint64_t SPLIT_M_ALIGN = Constant::VECTOR_SUBBLOCK_COUNT;

    enum class L0c2UbTensorType : uint8_t {
        SWISH_INPUT = 0,
        GATE_INPUT = 1,
    };

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;

    /*! \brief Element offsets of the current block in Y and YScale. */
    struct OutputOffsets {
        int64_t yOffset{0};
        int64_t yScaleOffset{0};
    };

    /*! \brief Runtime output addresses, tile sizes, and fused-operation attributes. */
    struct Params {
        GM_ADDR yGmAddr{nullptr};      //!< Required Y address; validated by the upper layer.
        GM_ADDR yScaleGmAddr{nullptr}; //!< Required YScale address; validated by the upper layer.
        uint32_t baseM{0};             //!< Reserved compatibility field; not read by the current implementation.
        uint32_t baseN{0};             //!< Reserved compatibility field; not read by the current implementation.
        // Keep scaleAlg fifth for legacy five-field aggregate initialization.
        uint32_t scaleAlg{0}; //!< Quantization algorithm: 0 (OCP) or 1 (cuBLAS).
        // The shared epilogue serves legacy V2 mode 0 and V3 mode 2.
        uint32_t swigluMode{0};
        float clampLimit{Constant::DEFAULT_CLAMP_LIMIT};
        float gluAlpha{Constant::DEFAULT_GLU_ALPHA};
        float gluBias{Constant::DEFAULT_GLU_BIAS};
        float dstTypeMax{0.0F}; //!< Reserved in this component; V3 requires 0.
        Params() = default;
    };

    __aicore__ inline BlockEpilogueSwigluMxQuant()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline ~BlockEpilogueSwigluMxQuant()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols, L0c2UbTensorType tensorType)
    {
        // The Split-M Fixpipe instruction requires an even mSize. The epilogue still receives the original
        // logical row count and ignores the padding row on sub-block 1.
        const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), SPLIT_M_ALIGN);
        const auto
            layoutOutUb = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<OUTPUT_C0_SIZE>>(
                copyRows, static_cast<uint64_t>(cols));
        const uint64_t ubOffset = static_cast<uint64_t>(tensorType) * INPUT_UB_BUFFER_BYTES;
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(ubOffset), layoutOutUb);
    }

    __aicore__ inline auto GetConcatL0c2UbTensor(int64_t rows, int64_t cols)
    {
        const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), SPLIT_M_ALIGN);
        const auto
            layoutOutUb = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<OUTPUT_C0_SIZE>>(
                copyRows, static_cast<uint64_t>(cols) * Constant::SWIGLU_HALF_COUNT);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layoutOutUb);
    }

    /*! \brief Binds output addresses, attributes, and UB layout. */
    __aicore__ inline void Init(Params const& params);
    /*! \brief Processes one logical half-width block and writes its Y/YScale results. */
    __aicore__ inline void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets);
    /*! \brief Updates the current post-SwiGLU problem shape. */
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);
    /*! \brief Applies group-level output base offsets. */
    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets);

private:
    template <class T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void ComputeSwiglu(uint16_t mSize);
    __aicore__ inline void ComputeMxQuant(uint16_t mSize);

    __aicore__ inline void SetupUbLayout();

    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    // ---- Params ----
    const Params* params_{nullptr};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    // ---- UB byte offsets (set in SetupUbLayout) ----
    uint64_t quantOutputUbOffset_{0};
    uint64_t quantScaleOutputUbOffset_{0};
    uint64_t quantScaleBlockOutputUbOffset_{0};
    uint64_t gluResUbOffset_{0};
    uint64_t maxExpUbOffset_{0};
    uint64_t halfScaleUbOffset_{0};

    // ---- Dimensions ----
    int64_t n_{0};
    int64_t scaleN_{0};
    int64_t scaleBlockN_{0};
    uint32_t subBlockIdx_{0};
    uint32_t singleM_{0};
    uint32_t singleN_{0};

    uint16_t fpEmax_{0};
    float invDstTypeMax_{Constant::RECIPROCAL_NUMERATOR / Constant::FP8_E4M3_MAX_VALUE};
};

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::SetCtrlSpr<Constant::FLOAT_OVERFLOW_MODE_CTRL, Constant::FLOAT_OVERFLOW_MODE_CTRL>(0);
    params_ = &params;
    subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());

    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = Constant::FP8_E4M3_MAX_EXP;
        invDstTypeMax_ = Constant::RECIPROCAL_NUMERATOR / Constant::FP8_E4M3_MAX_VALUE;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = Constant::FP8_E5M2_MAX_EXP;
        invDstTypeMax_ = Constant::RECIPROCAL_NUMERATOR / Constant::FP8_E5M2_MAX_VALUE;
    } else {
        // Preserve initialization for legacy, inactive non-FP8 template combinations.
        fpEmax_ = Constant::FP8_E4M3_MAX_EXP;
        invDstTypeMax_ = Constant::RECIPROCAL_NUMERATOR / Constant::FP8_E4M3_MAX_VALUE;
    }
    SetupUbLayout();
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::SetupUbLayout()
{
    constexpr uint32_t afterIn = Constant::SWIGLU_HALF_COUNT * Constant::MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutputUbOffset_ = afterIn;
    quantScaleOutputUbOffset_ = afterIn + Constant::MAX_SINGLE_MN * sizeof(int8_t);
    constexpr uint32_t afterIO = afterIn + Constant::MAX_SINGLE_MN * sizeof(int8_t) +
                                 Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    gluResUbOffset_ = afterIO;
    constexpr uint32_t afterIOAndGlu = afterIO + Constant::MAX_SINGLE_MN * sizeof(bfloat16_t);
    maxExpUbOffset_ = afterIOAndGlu;
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu +
                                          Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScaleUbOffset_ = afterIOAndGluExp;
    quantScaleBlockOutputUbOffset_ = afterIOAndGluExp +
                                     Constant::MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::UpdateGlobalAddr(
    const OutputOffsets& baseOffsets)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + baseOffsets.yOffset;
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) + baseOffsets.yScaleOffset;
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = asc::te::get<Blaze::Gemm::MNK_N>(problemShape);
    scaleN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
              Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::operator()(
    const BlockShape& blockShape, const OutputOffsets& outputOffsets)
{
    if ASCEND_IS_AIC {
        return;
    }
    singleM_ = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_M>(blockShape));
    singleN_ = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_N>(blockShape));
    scaleBlockN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
                   Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    auto halfSingleM = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleM_),
                                            static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = (subBlockIdx_ == Constant::SECOND_VECTOR_SUBBLOCK) ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);

    ComputeSwiglu(static_cast<uint16_t>(singleMInVec));
    ComputeMxQuant(static_cast<uint16_t>(singleMInVec));

    int64_t yOffset = outputOffsets.yOffset + static_cast<int64_t>(mOffset) * n_;
    int64_t yScaleOffset = outputOffsets.yScaleOffset + static_cast<int64_t>(mOffset) * scaleN_;

    TransMxScaleLayout(static_cast<uint16_t>(singleMInVec), static_cast<uint16_t>(scaleBlockN_));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(static_cast<uint64_t>(singleMInVec), yOffset);
    CopyScaleFromUb2Gm(static_cast<uint64_t>(singleMInVec), yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::CopyOutputFromUb2Gm(
    uint64_t blockCount, int64_t gmOffset)
{
    int64_t nValid;
    int64_t nUbAligned;
    int64_t gmRowPitch;

    nValid = static_cast<int64_t>(singleN_);
    nUbAligned = static_cast<int64_t>(Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_)));
    gmRowPitch = n_;

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                      ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantOutputGmAddr_ + gmOffset),
                                      gmLayout);

    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    asc::te::copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::CopyScaleFromUb2Gm(
    uint64_t blockCount, int64_t gmOffset)
{
    int64_t blockScaleN = static_cast<int64_t>(
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
        Blaze::Gemm::MXFP_MULTI_BASE_SIZE);

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), blockScaleN,
                                          static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), blockScaleN, scaleN_);
    auto outUb = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleBlockOutputUbOffset_), ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantScaleGmAddr_ + gmOffset),
                                      gmLayout);

    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    asc::te::copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::TransMxScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), static_cast<int64_t>(scaleBlockN),
                                                   static_cast<int64_t>(scaleBlockN));
    auto srcTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleOutputUbOffset_), srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleBlockOutputUbOffset_), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransScaleLayout(srcTensor, dstTensor, mSize, scaleBlockN);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeSwiglu(
    uint16_t mSize)
{
    __ubuf__ DataTypeIn* swishInputUbAddr = GetUbAddr<DataTypeIn>(0);
    __ubuf__ bfloat16_t* gluResAddr = GetUbAddr<bfloat16_t>(gluResUbOffset_);

    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    const uint16_t oneRowRepeatTimes = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_),
                                                            static_cast<uint64_t>(sizePerRepeat));
    const uint32_t nSrcUbAligned = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(singleN_),
                                                          AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    const uint32_t concatNSrcUbAligned = nSrcUbAligned * Constant::SWIGLU_HALF_COUNT;
    const uint32_t nDstUbAligned64 = Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_));

    const float scalarOne = Constant::SIGMOID_DENOMINATOR_OFFSET;

    // Zero-initialize gluRes when N tail (non-64-aligned)
    if (__builtin_expect((singleN_ % Constant::MX_QUANT_COMPUTE_ALIGN) != 0, 0)) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0));
            constexpr uint32_t bf16Vl = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
            uint32_t remainingElements = static_cast<uint32_t>(mSize) * nDstUbAligned64;
            uint32_t zeroOffset = 0;
            while (remainingElements > 0) {
                AscendC::Reg::MaskReg zeroMask = AscendC::Reg::UpdateMask<bfloat16_t>(remainingElements);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(gluResAddr + zeroOffset,
                                                                                           zeroReg, zeroMask);
                zeroOffset += bf16Vl;
            }
        }
    }

    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = singleN_;
            AscendC::Reg::MaskReg mask;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; vfBlockIdx++) {
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);

                AscendC::Reg::RegTensor<bfloat16_t> verg7;
                AscendC::Reg::RegTensor<float> swishInput, gateInput;
                AscendC::Reg::RegTensor<float> verg1, verg2, verg3, verg4, verg6, swishOutput;

                const uint32_t l0cOutOffset = mIdx * concatNSrcUbAligned + vfBlockIdx * sizePerRepeat;
                const uint32_t firstOffset = l0cOutOffset;
                const uint32_t secondOffset = l0cOutOffset + nSrcUbAligned;
                // Mode 2 fixes the first N half as activation and the second N half as gate.
                AscendC::Reg::DataCopy(swishInput, swishInputUbAddr + firstOffset);
                AscendC::Reg::DataCopy(gateInput, swishInputUbAddr + secondOffset);

                if (params_->swigluMode == 0) {
                    // Legacy V2: act / (1 + exp(-act)) * gate.
                    AscendC::Reg::Muls(verg2, swishInput, -scalarOne, mask);
                    AscendC::Reg::Exp(verg3, verg2, mask);
                    AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                    AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);
                } else {
                    // V3 mode 2: min(act, clamp) / (1 + exp(-alpha * min(act, clamp)))
                    //              * (clamp(gate, -clamp, clamp) + bias).
                    AscendC::Reg::Mins(verg1, swishInput, params_->clampLimit, mask);
                    AscendC::Reg::Muls(verg2, verg1, -params_->gluAlpha, mask);
                    AscendC::Reg::Exp(verg3, verg2, mask);
                    AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                    AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, verg1, verg4, mask);
                    AscendC::Reg::Mins(gateInput, gateInput, params_->clampLimit, mask);
                    AscendC::Reg::Maxs(gateInput, gateInput, -params_->clampLimit, mask);
                    AscendC::Reg::Adds(gateInput, gateInput, params_->gluBias, mask);
                }

                // SwiGLU = Swish(act) * gate.
                AscendC::Reg::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::Reg::Cast<bfloat16_t, float, CT_FP32_TO_BF16>(verg7, verg6, mask);

                uint32_t dstUbOffset = mIdx * nDstUbAligned64 + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset,
                                                                                           verg7, mask);
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_>::ComputeMxQuant(
    uint16_t mSize)
{
    const uint32_t nDstUbAligned64 = Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_));
    const uint32_t totalDataInUb = mSize * nDstUbAligned64;
    const uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;

    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(
        Constant::FLAT_VECTOR_ROW_COUNT, static_cast<int64_t>(totalDataInUb), static_cast<int64_t>(totalDataInUb));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(
        Constant::FLAT_VECTOR_ROW_COUNT, static_cast<int64_t>(totalScaleInUb), static_cast<int64_t>(totalScaleInUb));
    auto gluResTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(gluResUbOffset_),
                                             dataLayout);
    auto maxExpTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(maxExpUbOffset_),
                                             scaleLayout);
    auto yScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleOutputUbOffset_), scaleLayout);
    auto reciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(halfScaleUbOffset_), scaleLayout);
    auto yTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                        dataLayout);

    const bool useCublas = params_->scaleAlg == Constant::SCALE_ALG_CUBLAS;
    Tile::MxQuantConfig cfg{useCublas ? Tile::MxScaleAlg::CUBLAS : Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_,
                            Constant::MX_SCALE_REDUCTION_MASK, true};
    Tile::MxQuant<DataTypeOut> mx;
    mx.GroupMaxExp(gluResTensor, maxExpTensor, totalDataInUb, useCublas);
    mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScaleInUb);
    mx.Quantize(gluResTensor, reciprocalTensor, yTensor, totalDataInUb);
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
