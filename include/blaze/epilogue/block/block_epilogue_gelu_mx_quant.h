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
 * \file block_epilogue_gelu_mx_quant.h
 * \brief AIV epilogue for GELU followed by MXFP8/MXFP4 quantization.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

enum class QuantAlg : uint32_t {
    OCP = 0,
    BLAS = 1,
    DYN_DTYPE_RANGE = 2,
};

enum class GeluAlg : uint8_t {
    TANH = 0,
    ERF = 1,
};

enum class ROUND_MODE_FP4 : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;

constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint32_t MAX_SINGLE_SCALE_NUM = MAX_SINGLE_MN / BLOCK_SIZE;

template <typename DataTypeOut_, typename DataTypeIn_>
class BlockEpilogueGeluMxQuant {
public:
    __aicore__ inline BlockEpilogueGeluMxQuant()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline ~BlockEpilogueGeluMxQuant()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        GeluAlg geluAlg{GeluAlg::TANH};
        QuantAlg quantAlg{QuantAlg::OCP};
        ROUND_MODE_FP4 fp4RoundMode{ROUND_MODE_FP4::RINT};
        float dtypeMax{SCALAR_ZERO};
        Params() = default;
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    // shape
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params);

    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    static constexpr uint32_t MX_GROUP_SHIFT = 5;
    static constexpr uint32_t MX_GROUP_MASK = BLOCK_SIZE - 1;
    static constexpr uint32_t FP4_ELEMENTS_PER_BLOCK = 64;
    // Unbiased output exponents shifted into the BF16 exponent-field position.
    static constexpr uint16_t FP8_E4M3_MAX_EXP_BITS = 0x0400;
    static constexpr uint16_t FP8_E5M2_MAX_EXP_BITS = 0x0780;
    static constexpr uint16_t FP4_E2M1_MAX_EXP_BITS = 0x0100;
    static constexpr uint16_t FP4_E1M2_MAX_EXP_BITS = 0x0000;
    static constexpr float FP8_E4M3_MAX_VALUE = 448.0F;
    static constexpr float FP8_E5M2_MAX_VALUE = 57344.0F;
    static constexpr float FP4_E2M1_MAX_VALUE = 6.0F;
    static constexpr float FP4_E1M2_MAX_VALUE = 3.5F;
    static constexpr float FP4_E2M1_EXTENDED_MAX_VALUE = 7.0F;
    // Additive bit biases for DYN_DTYPE_RANGE, with a separate bias for dtypeMax=7.
    static constexpr uint16_t MX_SCALE_ADD_VALUE_BITS = 0x003f;
    static constexpr uint16_t MX_SCALE_EXTENDED_ADD_VALUE_BITS = 0x001f;
    static constexpr float SCALAR_ZERO = 0.0F;
    static constexpr float SCALAR_ONE = 1.0F;
    static constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL_BIT = 60;

    static constexpr uint64_t QUANT_OUTPUT_UB_OFFSET = MAX_SINGLE_MN * sizeof(DataTypeIn);
    static constexpr uint64_t QUANT_SCALE_UB_OFFSET = QUANT_OUTPUT_UB_OFFSET + MAX_SINGLE_MN;
    static constexpr uint64_t GELU_OUTPUT_UB_OFFSET = QUANT_SCALE_UB_OFFSET + MAX_SINGLE_SCALE_NUM;
    static constexpr uint64_t GROUP_MAX_UB_OFFSET = GELU_OUTPUT_UB_OFFSET + MAX_SINGLE_MN * sizeof(bfloat16_t);
    static constexpr uint64_t SCALE_RECIPROCAL_UB_OFFSET = GROUP_MAX_UB_OFFSET +
                                                           MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    static constexpr uint64_t PADDED_QUANT_SCALE_UB_OFFSET = SCALE_RECIPROCAL_UB_OFFSET +
                                                             MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    static constexpr bool IS_FP4 = AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                                   AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value;

    __aicore__ inline void PadMxScaleRows(uint16_t mSize);
    __aicore__ inline void TransFp4MxOutLayout(uint16_t mSize);
    __aicore__ inline void VFDoGeluAndQuantForMX(uint16_t mSize, uint16_t nSize, uint32_t nAligned);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t rowCount, int64_t gmOffset, uint64_t ubOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t rowCount, int64_t gmOffset);

    // ---- Params ----
    const Params* params_{nullptr};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    int64_t n_;
    int64_t scaleRowStride_;
    uint32_t scalePaddingCols_;
    uint32_t scaleCols_;
    uint32_t scaleCopyCols_;
    uint32_t mxScaleAlg_;
    uint32_t subBlockIdx_;
    uint32_t singleN_;
    float invDstMaxValue_{SCALAR_ONE / FP4_E2M1_MAX_VALUE};
    uint32_t scaleAddValueBits_{MX_SCALE_ADD_VALUE_BITS}; // DYN_DTYPE_RANGE 的进位附加值

    uint16_t dstMaxExpBits_{0};
    uint8_t fp4RoundMode_;
    bool useErf_;
};

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    // 量化结果的Nan值会转变成极大值
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL_BIT, FLOAT_OVERFLOW_MODE_CTRL_BIT>(0);
    params_ = &params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr);
    quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr);
    float fp4DtypeMax = SCALAR_ZERO;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        dstMaxExpBits_ = FP8_E4M3_MAX_EXP_BITS;
        invDstMaxValue_ = SCALAR_ONE / FP8_E4M3_MAX_VALUE;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        dstMaxExpBits_ = FP8_E5M2_MAX_EXP_BITS;
        invDstMaxValue_ = SCALAR_ONE / FP8_E5M2_MAX_VALUE;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        dstMaxExpBits_ = FP4_E2M1_MAX_EXP_BITS;
        fp4DtypeMax = params_->dtypeMax;
        invDstMaxValue_ = SCALAR_ONE / FP4_E2M1_MAX_VALUE;
    } else {
        dstMaxExpBits_ = FP4_E1M2_MAX_EXP_BITS;
        fp4DtypeMax = params_->dtypeMax;
        invDstMaxValue_ = SCALAR_ONE / FP4_E1M2_MAX_VALUE;
    }
    if (params_->quantAlg == QuantAlg::DYN_DTYPE_RANGE && params_->dtypeMax != SCALAR_ZERO) {
        invDstMaxValue_ = SCALAR_ONE / params_->dtypeMax;
    }
    // 进位附加值规则统一为 GELU (tanh) 语义
    scaleAddValueBits_ = params_->dtypeMax == FP4_E2M1_EXTENDED_MAX_VALUE ? MX_SCALE_EXTENDED_ADD_VALUE_BITS :
                                                                            MX_SCALE_ADD_VALUE_BITS;
    useErf_ = params.geluAlg == GeluAlg::ERF;
    mxScaleAlg_ = static_cast<uint32_t>(params.quantAlg);
    if (params.quantAlg == QuantAlg::DYN_DTYPE_RANGE && fp4DtypeMax != SCALAR_ZERO &&
        fp4DtypeMax != FP4_E2M1_MAX_VALUE && fp4DtypeMax != FP4_E2M1_EXTENDED_MAX_VALUE) {
        mxScaleAlg_ = static_cast<uint32_t>(QuantAlg::BLAS);
    }
    fp4RoundMode_ = static_cast<uint8_t>(params.fp4RoundMode);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + asc::te::get<Y_IDX>(baseOffset);
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) +
                            asc::te::get<Y_SCALE_IDX>(baseOffset);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = asc::te::get<Gemm::MNK_N>(problemShape);
    const uint64_t scaleGroups = (static_cast<uint64_t>(n_) + MX_GROUP_MASK) >> MX_GROUP_SHIFT;
    scalePaddingCols_ = static_cast<uint32_t>(scaleGroups & 1U);
    scaleRowStride_ = static_cast<int64_t>(scaleGroups + scalePaddingCols_);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyOutputFromUb2Gm(uint64_t rowCount,
                                                                                                int64_t gmOffset,
                                                                                                uint64_t ubOffset)
{
    int64_t rowBytes = static_cast<int64_t>(singleN_);
    int64_t gmRowStride = n_;

    if constexpr (IS_FP4) {
        rowBytes = rowBytes >> 1;
        gmRowStride = gmRowStride >> 1;
        gmOffset = gmOffset >> 1;
    }
    int64_t ubRowStride = static_cast<int64_t>(Gemm::Align32(static_cast<uint64_t>(rowBytes)));

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), rowBytes, ubRowStride);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), rowBytes, gmRowStride);
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOffset), ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantOutputGmAddr_ + gmOffset),
                                      gmLayout);

    asc::te::copy(outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyScaleFromUb2Gm(uint64_t rowCount,
                                                                                               int64_t gmOffset)
{
    int64_t scaleCopyCols = static_cast<int64_t>(scaleCopyCols_);
    const bool needsPadding = scaleCopyCols_ != scaleCols_;
    int64_t ubRowStride = needsPadding ? AscendC::ONE_BLK_SIZE : scaleCols_;
    const uint64_t ubOffset = needsPadding ? PADDED_QUANT_SCALE_UB_OFFSET : QUANT_SCALE_UB_OFFSET;
    int64_t gmRowStride = scaleRowStride_;

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), scaleCopyCols, ubRowStride);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), scaleCopyCols, gmRowStride);
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOffset), ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantScaleGmAddr_ + gmOffset),
                                      gmLayout);

    asc::te::copy(outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluAndQuantForMX(uint16_t mSize,
                                                                                                  uint16_t nSize,
                                                                                                  uint32_t nAligned)
{
    const uint32_t elementCount = static_cast<uint32_t>(mSize) * nAligned;
    auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize), static_cast<int64_t>(nSize),
                                        static_cast<int64_t>(nAligned));
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layout);
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(GELU_OUTPUT_UB_OFFSET), layout);
    // ZEROING cast plus an aligned store mask initializes only the final MX group's tail.
    // Aligned rows form one contiguous span, including rows that are 32 modulo 64 elements.
    const bool contiguous = nSize == nAligned;
    const uint16_t rows = contiguous ? 1 : mSize;
    const uint16_t cols = contiguous ? static_cast<uint16_t>(elementCount) : nSize;
    if constexpr (IS_FP4) {
        // FP4 layout conversion may reuse the GELU result as the previous MTE3 source.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
    Tile::Gelu<bfloat16_t, DataTypeIn, false, true> gelu;
    if (useErf_) {
        gelu.GeluErf(srcTensor, dstTensor, rows, cols);
    } else {
        gelu.GeluTanh(srcTensor, dstTensor, rows, cols);
    }
    // Quantization only reads the BF16 result; Fixpipe can reuse the FP32 input now.
    Gemm::Sync::NotifyCube();

    const uint32_t scaleCount = elementCount >> MX_GROUP_SHIFT;

    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(elementCount),
                                                    static_cast<int64_t>(elementCount));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(scaleCount),
                                                     static_cast<int64_t>(scaleCount));
    auto geluTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(GELU_OUTPUT_UB_OFFSET), dataLayout);
    auto groupMaxTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GROUP_MAX_UB_OFFSET), scaleLayout);
    auto quantScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_SCALE_UB_OFFSET), scaleLayout);
    auto scaleReciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(SCALE_RECIPROCAL_UB_OFFSET), scaleLayout);
    auto quantOutputTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_OUTPUT_UB_OFFSET), dataLayout);

    // In OCP mode, zero maxExp clears both the scale and reciprocal.
    Tile::MxQuantConfig mxQuantConfig{static_cast<Tile::MxScaleAlg>(mxScaleAlg_), dstMaxExpBits_, invDstMaxValue_,
                                      static_cast<uint16_t>(scaleAddValueBits_), true};

    Tile::MxQuant<DataTypeOut> mxQuant;
    mxQuant.GroupMaxExp(geluTensor, groupMaxTensor, elementCount, mxScaleAlg_ != static_cast<uint32_t>(QuantAlg::OCP));
    if constexpr (!IS_FP4) {
        // Wait only before overwriting the previous tile's output and scale buffers.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
    mxQuant.GenScale(groupMaxTensor, quantScaleTensor, scaleReciprocalTensor, mxQuantConfig, scaleCount);
    const auto fp4RoundMode = static_cast<Tile::MxQuantFp4RoundMode>(fp4RoundMode_);
    mxQuant.Quantize(geluTensor, scaleReciprocalTensor, quantOutputTensor, elementCount, fp4RoundMode);
}

/**
 * @brief 转换FP4 MX量化输出的数据布局
 *
 * 委托 MxQuant::TransFp4OutLayout：将FP4量化输出从 Align16(n/2) 字节行距的线性
 * 布局转为 Align32(n/2) 字节行距的块对齐布局，满足 MTE 搬运的对齐要求。
 *
 * 仅用于FP4 E2M1输出且singleN_不满足64元素对齐的情况。
 */
template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransFp4MxOutLayout(uint16_t mSize)
{
    const int64_t packedBytes = static_cast<int64_t>(singleN_ >> 1);
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), packedBytes,
                                                   static_cast<int64_t>(Gemm::Align16(packedBytes)));
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_OUTPUT_UB_OFFSET),
                                          srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), packedBytes,
                                                   static_cast<int64_t>(Gemm::Align32(packedBytes)));
    auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GELU_OUTPUT_UB_OFFSET),
                                          dstLayout);
    Tile::MxQuant<DataTypeOut> mxQuant;
    mxQuant.TransFp4OutLayout(srcTensor, dstTensor, mSize, static_cast<uint16_t>(singleN_));
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::PadMxScaleRows(uint16_t mSize)
{
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, scaleCols_, scaleCols_);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, scaleCopyCols_, AscendC::ONE_BLK_SIZE);
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_SCALE_UB_OFFSET),
                                          srcLayout);
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(PADDED_QUANT_SCALE_UB_OFFSET), dstLayout);
    Tile::MxQuant<DataTypeOut> mxQuant;
    mxQuant.PadScaleRows(srcTensor, dstTensor, mSize, scaleCols_, scaleCopyCols_);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::operator()(const BlockShape& blockShape,
                                                                                       const BlockCoord& blockCoord)
{
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t singleM = asc::te::get<Gemm::MNK_M>(blockShape);
    const uint32_t rowsPerSubBlock = Gemm::CeilDiv(singleM, static_cast<uint32_t>(AscendC::GetTaskRation()));
    const uint16_t rowCount = static_cast<uint16_t>(subBlockIdx_ == 1 ? singleM - rowsPerSubBlock : rowsPerSubBlock);
    if (rowCount == 0) {
        Gemm::Sync::NotifyCube();
        return;
    }

    singleN_ = asc::te::get<Gemm::MNK_N>(blockShape);
    const uint32_t nAligned = (singleN_ + MX_GROUP_MASK) & ~MX_GROUP_MASK;
    scaleCols_ = nAligned >> MX_GROUP_SHIFT;
    VFDoGeluAndQuantForMX(rowCount, static_cast<uint16_t>(singleN_), nAligned);
    uint64_t outputUbOffset = QUANT_OUTPUT_UB_OFFSET;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        if ((singleN_ & (FP4_ELEMENTS_PER_BLOCK - 1)) != 0) {
            TransFp4MxOutLayout(rowCount);
            outputUbOffset = GELU_OUTPUT_UB_OFFSET;
        }
    }
    scaleCopyCols_ = scaleCols_;
    if (scalePaddingCols_ != 0 && asc::te::get<Y_IDX>(blockCoord) % n_ + singleN_ == n_) {
        // Only the final N tile writes the unused byte of the paired E8M0 layout.
        ++scaleCopyCols_;
        PadMxScaleRows(rowCount);
    }
    const int64_t mOffset = static_cast<int64_t>(subBlockIdx_ * rowsPerSubBlock);
    const int64_t yOffset = asc::te::get<Y_IDX>(blockCoord) + mOffset * n_;
    const int64_t yScaleOffset = asc::te::get<Y_SCALE_IDX>(blockCoord) + mOffset * scaleRowStride_;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(rowCount, yOffset, outputUbOffset);
    CopyScaleFromUb2Gm(rowCount, yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}
} // namespace Block
} // namespace Epilogue
} // namespace Blaze
