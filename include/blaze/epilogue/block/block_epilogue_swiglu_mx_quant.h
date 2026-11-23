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

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/epilogue/block/block_epilogue_types.h"
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
constexpr uint32_t UB_BYTES_PER_KIB = 1024U;
constexpr uint32_t A8W4_INPUT_UB_OFFSET = 64U * UB_BYTES_PER_KIB;
constexpr uint32_t A8W4_OUTPUT_EVENT_BASE = 5U;
constexpr uint32_t A8W4_QUANT_OUTPUT_UB_OFFSET = 212U * UB_BYTES_PER_KIB;
constexpr uint32_t A8W4_SCALE_OUTPUT_UB_OFFSET = 96U * UB_BYTES_PER_KIB;
constexpr uint32_t A8W4_MAX_EXP_UB_OFFSET = 192U * UB_BYTES_PER_KIB;
constexpr uint32_t A8W4_HALF_SCALE_UB_OFFSET = 97U * UB_BYTES_PER_KIB;
constexpr uint32_t A8W4_SCALE_BLOCK_UB_OFFSET = 196U * UB_BYTES_PER_KIB;
constexpr uint32_t OUTPUT_SLOT_COUNT = 2U;
constexpr uint32_t OUTPUT_DATA_SLOT_BYTES = MAX_SINGLE_MN;
constexpr uint32_t OUTPUT_SCALE_SLOT_BYTES = MAX_SINGLE_MN / OUTPUT_SLOT_COUNT;
constexpr uint64_t SMOOTH_SCALE_UB_OFFSET = 246UL * 1024UL;
constexpr uint64_t SMOOTH_SCALE_MAX_COLUMNS = 128UL;
constexpr uint64_t UB_CAPACITY_BYTES = 248UL * 1024UL;
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
constexpr AscendC::Reg::DivSpecificMode A8W4_MODE2_DIV_1ULP = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
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
 * \tparam DirectBf16Output_ Enables V3 A4W4 direct
 * BF16 output. It is opt-in
 *         so existing BF16 instantiations preserve their MX quant behavior.
 * \tparam EnableSmoothScale_ Enables expert smoothing and weight-prologue scale restoration for MX A8W4.
 * \tparam A8W4V3Fast_ Selects the opt-in V3 A8W4 UB/pipeline specialization; default false preserves legacy behavior.
 */
template <typename DataTypeOut_, typename DataTypeIn_ = float, typename DataTypeScale_ = fp8_e8m0_t,
          bool DirectBf16Output_ = false, bool EnableSmoothScale_ = false, bool A8W4V3Fast_ = false>
class BlockEpilogueSwigluMxQuant {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeScale = DataTypeScale_;
    // Legacy APT also instantiates inactive FP4 combinations. Validate execution dtypes in the upper-layer route.
    static constexpr bool DIRECT_BF16_OUTPUT = DirectBf16Output_;
    static constexpr bool ENABLE_SMOOTH_SCALE = EnableSmoothScale_;
    // Opt-in compile-time selection of the V3 A8W4 epilogue: dedicated UB layout,
    // double-buffered output, smooth-scale prefetch, and fused SwiGLU/MX quant stages.
    // "FAST" is the historical specialization name, not a per-shape speed guarantee.
    // The default false keeps legacy instantiations on their existing implementation.
    static constexpr bool A8W4_V3_FAST = A8W4V3Fast_;
    static_assert(!A8W4_V3_FAST || (ENABLE_SMOOTH_SCALE && !DIRECT_BF16_OUTPUT),
                  "The V3 A8W4 fast path requires the MX expert-smoothing specialization.");
    static_assert(!ENABLE_SMOOTH_SCALE || !DIRECT_BF16_OUTPUT,
                  "Expert smoothing is supported by the MX-quantized output path.");
    static_assert(!DIRECT_BF16_OUTPUT || AscendC::IsSameType<DataTypeOut, bfloat16_t>::value,
                  "Direct BF16 output requires bfloat16_t DataTypeOut.");
    static constexpr uint64_t INPUT_UB_TILE_ELEMENTS = Constant::MAX_SINGLE_MN;
    static constexpr uint64_t INPUT_UB_BUFFER_BYTES = INPUT_UB_TILE_ELEMENTS * sizeof(DataTypeIn);
    static constexpr uint64_t OUTPUT_C0_SIZE = asc::te::c0_element<DataTypeIn>;
    static constexpr uint64_t SPLIT_M_ALIGN = Constant::VECTOR_SUBBLOCK_COUNT;
    static constexpr uint32_t V3_INPUT_OFFSET = Constant::A8W4_INPUT_UB_OFFSET;
    static constexpr uint32_t OUTPUT_EVENT_BASE = A8W4_V3_FAST ? Constant::A8W4_OUTPUT_EVENT_BASE : 0U;

    enum class L0c2UbTensorType : uint8_t {
        SWISH_INPUT = 0,
        GATE_INPUT = 1,
        CONCATENATED_INPUT = 2,
    };

    using UbTensorKind = L0c2UbTensorType;

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;

    /*! \brief Element offsets of the current block in Y and YScale. */
    using OutputOffsets = Blaze::Epilogue::Block::OutputOffsets;

    /*! \brief Runtime output addresses, tile sizes, and fused-operation attributes. */
    struct Params {
        GM_ADDR yGmAddr{nullptr};      //!< Required Y address; validated by the upper layer.
        GM_ADDR yScaleGmAddr{nullptr}; //!< Required for MX-quantized Y; unused for direct BF16 Y.
        uint32_t baseM{0};             //!< Reserved compatibility field; not read by the current implementation.
        uint32_t baseN{0};             //!< Reserved compatibility field; not read by the current implementation.
        // Keep scaleAlg fifth for legacy five-field aggregate initialization.
        uint32_t scaleAlg{0}; //!< Quantization algorithm: 0 (OCP) or 1 (cuBLAS).
        // The shared epilogue serves legacy V2 mode 0 and V3 mode 2.
        uint32_t swigluMode{0};
        float clampLimit{Constant::DEFAULT_CLAMP_LIMIT};
        float gluAlpha{Constant::DEFAULT_GLU_ALPHA};
        float gluBias{Constant::DEFAULT_GLU_BIAS};
        float dstTypeMax{0.0F};             //!< Reserved in this component; V3 requires 0.
        GM_ADDR smoothScaleGmAddr{nullptr}; //!< Optional FP32 expert/channel scale, applied before BF16 rounding.
        float inputScale{1.0F};             //!< Restores the FP4-to-FP8 weight-prologue scale before activation.
        Params() = default;
    };

    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    __aicore__ inline BlockEpilogueSwigluMxQuant();
    __aicore__ inline ~BlockEpilogueSwigluMxQuant();
    /*! \brief Binds output addresses, attributes, and UB layout. */
    __aicore__ inline void Init(Params const& params);

    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;

    // Compatibility interfaces retained for existing callers.
    [[deprecated("GetL0c2UbTensor接口在CANN 9.2.0版本标记废弃，将在2027年11月10日之后的版本删除，"
                 "替换为GetAccumulatorTensor(const AccumulatorTensorParams&)")]]
    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols, L0c2UbTensorType tensorType);

    [[deprecated("GetConcatL0c2UbTensor接口在CANN 9.2.0版本标记废弃，将在2027年11月10日之后的版本删除，"
                 "替换为GetAccumulatorTensor(const AccumulatorTensorParams&)")]]
    __aicore__ inline auto GetConcatL0c2UbTensor(int64_t rows, int64_t cols);

    /*! \brief Processes one logical half-width block and writes its Y/YScale results. */
    __aicore__ inline void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets);
    /*! \brief Updates the current post-SwiGLU problem shape. */
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);
    /*! \brief Applies group-level output base offsets. */
    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets);
    /*! \brief Selects a smooth-scale expert row and channel tile; offsets are in FP32 elements. */
    __aicore__ inline void SetSmoothScaleOffset(int64_t offset)
    {
        if constexpr (A8W4_V3_FAST) {
            if (smoothScaleOffset_ != offset) {
                smoothScalePrefetchedColumns_ = 0;
            }
        }
        smoothScaleOffset_ = offset;
    }

    __aicore__ inline void PrefetchSmoothScale(uint32_t columns = 0)
    {
        if constexpr (A8W4_V3_FAST) {
            if (columns != 0) {
                singleN_ = columns;
            }
            if (params_->smoothScaleGmAddr != nullptr && smoothScalePrefetchedColumns_ < singleN_) {
                CopySmoothScaleToUb();
                smoothScalePrefetchedColumns_ = singleN_;
            }
        }
    }

private:
    __aicore__ inline auto MakeUbTensor(const AccumulatorTensorParams& params, UbTensorKind kind) const;

    template <class T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    struct A8w4SwigluVfParams {
        __ubuf__ float* input;
        __ubuf__ bfloat16_t* output;
        __ubuf__ float* smooth;
        uint32_t rows;
        uint32_t columns;
        uint32_t rowOffset;
        uint32_t inputPitch;
        uint32_t branchPitch;
        uint32_t outputPitch;
        float inputScale;
        float limit;
        float alpha;
        float bias;
    };
    template <bool Smooth>
    static __simd_vf__ inline void A8w4SwigluVf(A8w4SwigluVfParams p)
    {
        namespace R = AscendC::Reg;
        constexpr uint32_t LANES = asc_get_vf_len() / sizeof(float);
        const uint16_t repeats = (p.columns + LANES - 1) / LANES;
        for (uint16_t row = 0; row < p.rows; ++row) {
            uint32_t remaining = p.columns;
            for (uint16_t block = 0; block < repeats; ++block) {
                auto mask = R::UpdateMask<float>(remaining);
                R::RegTensor<float> act, gate, exponent, denominator, swish, result;
                R::RegTensor<bfloat16_t> rounded;
                const uint32_t offset = (row + p.rowOffset) * p.inputPitch + block * LANES;
                R::LoadAlign(act, p.input + offset);
                R::Muls(act, act, p.inputScale, mask);
                R::Mins(act, act, p.limit, mask);
                R::Muls(exponent, act, -p.alpha, mask);
                R::Exp(exponent, exponent, mask);
                R::LoadAlign(gate, p.input + offset + p.branchPitch);
                R::Muls(gate, gate, p.inputScale, mask);
                R::Mins(gate, gate, p.limit, mask);
                R::Maxs(gate, gate, -p.limit, mask);
                R::Adds(gate, gate, p.bias, mask);
                R::Adds(denominator, exponent, Constant::SIGMOID_DENOMINATOR_OFFSET, mask);
                R::Div<float, &A8W4_MODE2_DIV_1ULP>(swish, act, denominator, mask);
                R::Mul(result, swish, gate, mask);
                if constexpr (Smooth) {
                    R::RegTensor<float> factor;
                    R::LoadAlign(factor, p.smooth + block * LANES);
                    R::Mul(result, result, factor, mask);
                }
                R::Cast<bfloat16_t, float, CT_FP32_TO_BF16>(rounded, result, mask);
                R::StoreAlign<bfloat16_t, R::StoreDist::DIST_PACK_B32>(p.output + row * p.outputPitch + block * LANES,
                                                                       rounded, mask);
            }
        }
    }

    __aicore__ inline void ComputeSwiglu(uint16_t mSize);
    __aicore__ inline void RunA8w4V3Output(uint32_t rows, int64_t yOffset, int64_t yScaleOffset);
    __aicore__ inline void CopySmoothScaleToUb();
    __aicore__ inline void ComputeMxQuant(uint16_t mSize);

    __aicore__ inline void SetupUbLayout();

    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    // ---- Params ----
    const Params* params_{nullptr};
    uint32_t outputSlot_{0};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ DataTypeOut* outputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    // ---- UB byte offsets (set in SetupUbLayout) ----
    uint64_t quantOutputUbOffset_{0};
    uint64_t quantScaleOutputUbOffset_{0};
    uint64_t quantScaleBlockOutputUbOffset_{0};
    uint64_t gluResUbOffset_{0};
    uint64_t maxExpUbOffset_{0};
    uint64_t halfScaleUbOffset_{0};
    uint64_t smoothScaleUbOffset_{0};
    int64_t smoothScaleOffset_{0L};
    uint32_t smoothScalePrefetchedColumns_{0};

    // ---- Dimensions ----
    int64_t n_{0};
    int64_t scaleN_{0};
    bool hasScalePadding_{false};
    int64_t scaleBlockN_{0};
    uint32_t subBlockIdx_{0};
    uint32_t singleM_{0};
    uint32_t singleN_{0};
    uint32_t inputRowOffset_{0};

    uint16_t fpEmax_{0};
    float invDstTypeMax_{Constant::RECIPROCAL_NUMERATOR / Constant::FP8_E4M3_MAX_VALUE};
};

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                             EnableSmoothScale_, A8W4V3Fast_>::BlockEpilogueSwigluMxQuant()
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE);
    if constexpr (A8W4_V3_FAST) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + 1);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                             EnableSmoothScale_, A8W4V3Fast_>::~BlockEpilogueSwigluMxQuant()
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE);
    if constexpr (A8W4_V3_FAST) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + 1);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline auto
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::GetAccumulatorTensor(const AccumulatorTensorParams& params) const
{
    return MakeUbTensor(params, UbTensorKind::CONCATENATED_INPUT);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline auto
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::GetL0c2UbTensor(int64_t rows, int64_t cols, L0c2UbTensorType tensorType)
{
    return MakeUbTensor({rows, cols}, tensorType);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline auto BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::GetConcatL0c2UbTensor(int64_t rows,
                                                                                                          int64_t cols)
{
    return MakeUbTensor({rows, cols}, UbTensorKind::CONCATENATED_INPUT);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline auto
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::MakeUbTensor(const AccumulatorTensorParams& params, UbTensorKind kind) const
{
    const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(params.rows), SPLIT_M_ALIGN);
    const uint64_t copyCols = static_cast<uint64_t>(params.cols) *
                              (kind == UbTensorKind::CONCATENATED_INPUT ? Constant::SWIGLU_HALF_COUNT : 1UL);
    const auto layoutOutUb = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<OUTPUT_C0_SIZE>>(
        copyRows, copyCols);
    const uint64_t ubOffset = kind == UbTensorKind::CONCATENATED_INPUT ?
                                  (A8W4_V3_FAST ? V3_INPUT_OFFSET : 0UL) :
                                  static_cast<uint64_t>(kind) * INPUT_UB_BUFFER_BYTES;
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(ubOffset), layoutOutUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
template <class T>
__aicore__ inline __ubuf__ T*
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::GetUbAddr(uint64_t byteOffset)
{
    return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::SetCtrlSpr<Constant::FLOAT_OVERFLOW_MODE_CTRL, Constant::FLOAT_OVERFLOW_MODE_CTRL>(0);
    params_ = &params;
    SetupUbLayout();
    if constexpr (A8W4_V3_FAST) {
        smoothScalePrefetchedColumns_ = 0;
    }
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
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::SetupUbLayout()
{
    constexpr uint32_t afterIn = Constant::SWIGLU_HALF_COUNT * Constant::MAX_SINGLE_MN * sizeof(DataTypeIn);
    if constexpr (DIRECT_BF16_OUTPUT) {
        gluResUbOffset_ = afterIn;
        return;
    }

    if constexpr (A8W4_V3_FAST) {
        quantOutputUbOffset_ = Constant::A8W4_QUANT_OUTPUT_UB_OFFSET;
        quantScaleOutputUbOffset_ = Constant::A8W4_SCALE_OUTPUT_UB_OFFSET;
        maxExpUbOffset_ = Constant::A8W4_MAX_EXP_UB_OFFSET;
        halfScaleUbOffset_ = Constant::A8W4_HALF_SCALE_UB_OFFSET;
        quantScaleBlockOutputUbOffset_ = Constant::A8W4_SCALE_BLOCK_UB_OFFSET;
        gluResUbOffset_ = V3_INPUT_OFFSET;
        smoothScaleUbOffset_ = Constant::SMOOTH_SCALE_UB_OFFSET;
        return;
    }
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
    // A8W4's weight prologue uses UB through 245 KiB. Keep the smooth row outside
    // that shared staging/output region; its 128 FP32 channels occupy 512 bytes.
    static_assert(Constant::SMOOTH_SCALE_UB_OFFSET + Constant::SMOOTH_SCALE_MAX_COLUMNS * sizeof(float) <=
                      Constant::UB_CAPACITY_BYTES,
                  "The smooth scale row must fit in the reserved UB tail.");
    smoothScaleUbOffset_ = ENABLE_SMOOTH_SCALE ?
                               Constant::SMOOTH_SCALE_UB_OFFSET :
                               quantScaleBlockOutputUbOffset_ + Constant::MAX_SINGLE_MN * sizeof(int8_t);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::UpdateGlobalAddr(const OutputOffsets& baseOffsets)
{
    if ASCEND_IS_AIV {
        outputGmAddr_ = reinterpret_cast<__gm__ DataTypeOut*>(params_->yGmAddr) + baseOffsets.yOffset;
        if constexpr (!DIRECT_BF16_OUTPUT) {
            quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) + baseOffsets.yScaleOffset;
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::UpdateNextProblem(const ProblemShape& problemShape)
{
    n_ = asc::te::get<Blaze::Gemm::MNK_N>(problemShape);
    scaleN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
              Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets)
{
    if ASCEND_IS_AIC {
        return;
    }
    singleM_ = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_M>(blockShape));
    singleN_ = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_N>(blockShape));
    scaleBlockN_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
                   Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    // Only the final N tile owns the optional byte in the public scale pair.
    constexpr uint64_t scaleGroupSize = Blaze::Gemm::MXFP_DIVISOR_SIZE / Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    const uint64_t validRowScales = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_), scaleGroupSize);
    hasScalePadding_ = n_ > 0 && singleN_ > 0 && validRowScales % Blaze::Gemm::MXFP_MULTI_BASE_SIZE != 0 &&
                       outputOffsets.yOffset % n_ + singleN_ == n_;

    auto halfSingleM = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleM_),
                                            static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = (subBlockIdx_ == Constant::SECOND_VECTOR_SUBBLOCK) ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    if constexpr (A8W4_V3_FAST) {
        RunA8w4V3Output(static_cast<uint32_t>(singleMInVec), outputOffsets.yOffset + static_cast<int64_t>(mOffset) * n_,
                        outputOffsets.yScaleOffset + static_cast<int64_t>(mOffset) * scaleN_);
        return;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);

    if constexpr (ENABLE_SMOOTH_SCALE) {
        if (params_->smoothScaleGmAddr != nullptr) {
            CopySmoothScaleToUb();
        }
    }
    ComputeSwiglu(static_cast<uint16_t>(singleMInVec));

    int64_t yOffset = outputOffsets.yOffset + static_cast<int64_t>(mOffset) * n_;
    int64_t yScaleOffset{0};
    if constexpr (!DIRECT_BF16_OUTPUT) {
        ComputeMxQuant(static_cast<uint16_t>(singleMInVec));
        yScaleOffset = outputOffsets.yScaleOffset + static_cast<int64_t>(mOffset) * scaleN_;
        TransMxScaleLayout(static_cast<uint16_t>(singleMInVec), static_cast<uint16_t>(scaleBlockN_));
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(static_cast<uint64_t>(singleMInVec), yOffset);
    if constexpr (!DIRECT_BF16_OUTPUT) {
        CopyScaleFromUb2Gm(static_cast<uint64_t>(singleMInVec), yScaleOffset);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset)
{
    int64_t nValid;
    int64_t nUbAligned;
    int64_t gmRowPitch;

    nValid = static_cast<int64_t>(singleN_);
    nUbAligned = static_cast<int64_t>(Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_)));
    gmRowPitch = n_;

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    if constexpr (DIRECT_BF16_OUTPUT) {
        auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeOut>(gluResUbOffset_),
                                          ubLayout);
        auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(outputGmAddr_ + gmOffset),
                                          gmLayout);
        asc::te::copy(copyUB2GM, outGm, outUb);
    } else {
        // Keep the legacy MXFP8 copy type unchanged. Quantized output is stored as raw bytes.
        auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                          ubLayout);
        auto outGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int8_t*>(outputGmAddr_) + gmOffset),
            gmLayout);
        asc::te::copy(copyUB2GM, outGm, outUb);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset)
{
    int64_t blockScaleN = static_cast<int64_t>(
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), Blaze::Gemm::MXFP_DIVISOR_SIZE) *
        Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    if constexpr (ENABLE_SMOOTH_SCALE) {
        constexpr uint64_t scaleGroupSize = Blaze::Gemm::MXFP_DIVISOR_SIZE / Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
        blockScaleN = static_cast<int64_t>(Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize)) +
                      hasScalePadding_;
    }

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

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN)
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
    if constexpr (ENABLE_SMOOTH_SCALE) {
        constexpr uint64_t scaleGroupSize = Blaze::Gemm::MXFP_DIVISOR_SIZE / Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
        const auto validScales = static_cast<uint16_t>(
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize));
        mx.TransScaleLayout(srcTensor, dstTensor, mSize, scaleBlockN, validScales);
    } else {
        mx.TransScaleLayout(srcTensor, dstTensor, mSize, scaleBlockN);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::CopySmoothScaleToUb()
{
    auto layout = Gemm::MakeNDExtLayout(1L, static_cast<int64_t>(singleN_),
                                        static_cast<int64_t>(Blaze::Gemm::Align64(singleN_)));
    auto source = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(params_->smoothScaleGmAddr) +
                                                     smoothScaleOffset_),
        layout);
    auto destination = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(smoothScaleUbOffset_),
                                            layout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), destination, source);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::ComputeSwiglu(uint16_t mSize)
{
    if constexpr (A8W4_V3_FAST) {
        A8w4SwigluVfParams vf{reinterpret_cast<__ubuf__ float*>(asc_get_phy_buf_addr(0) + V3_INPUT_OFFSET),
                              reinterpret_cast<__ubuf__ bfloat16_t*>(asc_get_phy_buf_addr(0) + gluResUbOffset_),
                              reinterpret_cast<__ubuf__ float*>(asc_get_phy_buf_addr(0) + smoothScaleUbOffset_),
                              mSize,
                              singleN_,
                              inputRowOffset_,
                              static_cast<uint32_t>(Blaze::Gemm::Align64(singleN_)) * Constant::SWIGLU_HALF_COUNT,
                              static_cast<uint32_t>(Blaze::Gemm::Align64(singleN_)),
                              static_cast<uint32_t>(Blaze::Gemm::Align64(singleN_)),
                              params_->inputScale,
                              params_->clampLimit,
                              params_->gluAlpha,
                              params_->gluBias};
        if (params_->smoothScaleGmAddr != nullptr) {
            asc_vf_call<A8w4SwigluVf<true>>(vf);
        } else {
            asc_vf_call<A8w4SwigluVf<false>>(vf);
        }
        return;
    }

    __ubuf__ DataTypeIn* swishInputUbAddr = GetUbAddr<DataTypeIn>(0);
    __ubuf__ bfloat16_t* gluResAddr = GetUbAddr<bfloat16_t>(gluResUbOffset_);

    constexpr uint16_t sizePerRepeat = asc_get_vf_len() / sizeof(DataTypeIn);
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
            constexpr uint32_t bf16Vl = asc_get_vf_len() / sizeof(bfloat16_t);
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
                if constexpr (ENABLE_SMOOTH_SCALE) {
                    AscendC::Reg::Muls(swishInput, swishInput, params_->inputScale, mask);
                    AscendC::Reg::Muls(gateInput, gateInput, params_->inputScale, mask);
                }

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
                if constexpr (ENABLE_SMOOTH_SCALE) {
                    if (params_->smoothScaleGmAddr != nullptr) {
                        AscendC::Reg::RegTensor<float> smoothScale;
                        AscendC::Reg::LoadAlign(smoothScale,
                                                GetUbAddr<float>(smoothScaleUbOffset_) + vfBlockIdx * sizePerRepeat);
                        AscendC::Reg::Mul(verg6, verg6, smoothScale, mask);
                    }
                }

                AscendC::Reg::Cast<bfloat16_t, float, CT_FP32_TO_BF16>(verg7, verg6, mask);

                uint32_t dstUbOffset = mIdx * nDstUbAligned64 + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset,
                                                                                           verg7, mask);
            }
        }
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_,
                                                  EnableSmoothScale_, A8W4V3Fast_>::ComputeMxQuant(uint16_t mSize)
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
    // OCP clears the reciprocal when sharedExp is zero.
    Tile::MxQuantConfig cfg{useCublas ? Tile::MxScaleAlg::CUBLAS : Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_,
                            Constant::MX_SCALE_REDUCTION_MASK, false};
    Tile::MxQuant<DataTypeOut> mx;
    if constexpr (A8W4_V3_FAST) {
        mx.GroupMaxAndScale(gluResTensor, maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalDataInUb,
                            totalScaleInUb);
    } else {
        mx.GroupMaxExp(gluResTensor, maxExpTensor, totalDataInUb, useCublas);
        mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScaleInUb);
    }
    if constexpr (A8W4_V3_FAST) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputSlot_ + OUTPUT_EVENT_BASE);
    }
    mx.Quantize(gluResTensor, reciprocalTensor, yTensor, totalDataInUb);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeScale_, bool DirectBf16Output_,
          bool EnableSmoothScale_, bool A8W4V3Fast_>
__aicore__ inline void
BlockEpilogueSwigluMxQuant<DataTypeOut_, DataTypeIn_, DataTypeScale_, DirectBf16Output_, EnableSmoothScale_,
                           A8W4V3Fast_>::RunA8w4V3Output(uint32_t rows, int64_t yOffset, int64_t yScaleOffset)
{
    const uint32_t rowPitch = static_cast<uint32_t>(Blaze::Gemm::Align64(static_cast<uint64_t>(singleN_)));
    const uint32_t rowsPerSlot = Constant::OUTPUT_DATA_SLOT_BYTES / rowPitch;
    const uint64_t dataBase = quantOutputUbOffset_;
    const uint64_t scaleBase = quantScaleBlockOutputUbOffset_;
    PrefetchSmoothScale();
    uint32_t slot = outputSlot_;
    for (uint32_t row = 0; row < rows; row += rowsPerSlot) {
        const uint32_t count = rows - row < rowsPerSlot ? rows - row : rowsPerSlot;
        outputSlot_ = slot;
        inputRowOffset_ = row;
        quantOutputUbOffset_ = dataBase + slot * Constant::OUTPUT_DATA_SLOT_BYTES;
        quantScaleBlockOutputUbOffset_ = scaleBase + slot * Constant::OUTPUT_SCALE_SLOT_BYTES;
        ComputeSwiglu(static_cast<uint16_t>(count));
        ComputeMxQuant(static_cast<uint16_t>(count));
        TransMxScaleLayout(static_cast<uint16_t>(count), static_cast<uint16_t>(scaleBlockN_));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot + OUTPUT_EVENT_BASE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot + OUTPUT_EVENT_BASE);
        CopyOutputFromUb2Gm(count, yOffset + static_cast<int64_t>(row) * n_);
        CopyScaleFromUb2Gm(count, yScaleOffset + static_cast<int64_t>(row) * scaleN_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot + OUTPUT_EVENT_BASE);
        slot = (slot + 1) % Constant::OUTPUT_SLOT_COUNT;
    }
    outputSlot_ = slot;
    inputRowOffset_ = 0;
    quantOutputUbOffset_ = dataBase;
    quantScaleBlockOutputUbOffset_ = scaleBase;
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
