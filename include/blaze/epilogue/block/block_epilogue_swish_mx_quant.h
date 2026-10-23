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
 * \file block_epilogue_swish_mx_quant.h
 * \brief AIV epilogue for Swish(x, beta) followed by MXFP8 quantization.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
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

enum class SwishBetaMode {
    Scalar,
    PerChannel,
};

#define SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS template <typename DataTypeOut_, typename DataTypeIn_, SwishBetaMode Mode>
#define SWISH_MX_QUANT_TEMPLATE_ARGS DataTypeOut_, DataTypeIn_, Mode

template <typename DataTypeOut_, typename DataTypeIn_ = float, SwishBetaMode Mode = SwishBetaMode::Scalar>
class BlockEpilogueSwishMxQuant {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t scaleAlg{0};
        float beta{1.0F}; // Finite scalar shared by all output elements.
        Params() = default;
    };

    static_assert(Mode == SwishBetaMode::Scalar, "BlockEpilogueSwishMxQuant: PerChannel beta is not implemented.");
    static_assert(AscendC::IsSameType<DataTypeIn, float>::value,
                  "BlockEpilogueSwishMxQuant only supports float L0C input.");
    static_assert(AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value,
                  "BlockEpilogueSwishMxQuant only supports FP8 output.");

    __aicore__ inline BlockEpilogueSwishMxQuant()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline ~BlockEpilogueSwishMxQuant()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    static constexpr uint32_t MX_GROUP_SHIFT = 5;
    static constexpr uint32_t MX_GROUP_SIZE = 32;
    static constexpr uint32_t MX_GROUP_MASK = MX_GROUP_SIZE - 1;
    static constexpr uint32_t MAX_TILE_ELEMENTS = 128 * 256;
    // Unbiased FP8 exponents shifted into the BF16 exponent-field position.
    static constexpr uint16_t FP8_E4M3_MAX_EXP_BITS = 0x0400;
    static constexpr uint16_t FP8_E5M2_MAX_EXP_BITS = 0x0780;
    static constexpr float FP8_E4M3_MAX_VALUE = 448.0F;
    static constexpr float FP8_E5M2_MAX_VALUE = 57344.0F;
    // Additive bit bias for DYN_DTYPE_RANGE; unused by OCP/CUBLAS.
    static constexpr uint16_t MX_SCALE_ADD_VALUE_BITS = 0x003f;
    static constexpr uint32_t SCALE_ALG_CUBLAS = 1;
    static constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL_BIT = 60;

    // Per-AIV UB layout: up to 128 rows x 256 columns; byte offsets from the UB base, aligned to 32B.
    // Region start                   Byte range [begin, end)  Size     Storage
    // 0                               [     0, 131072)         128 KiB  FP32 matmul input
    // QUANT_OUTPUT_UB_OFFSET          [131072, 163840)          32 KiB  FP8 quantized output
    // QUANT_SCALE_UB_OFFSET           [163840, 164864)           1 KiB  E8M0 scale bytes
    // SWISH_OUTPUT_UB_OFFSET          [164864, 230400)          64 KiB  BF16 Swish result
    // GROUP_MAX_UB_OFFSET             [230400, 232448)           2 KiB  Group max exponent or abs bits
    // SCALE_RECIPROCAL_UB_OFFSET      [232448, 234496)           2 KiB  BF16 scale reciprocal bits
    // PADDED_QUANT_SCALE_UB_OFFSET    [234496, 238592)           4 KiB  Repacked scale bytes
    // Total: 238592 bytes (233 KiB) per AIV.
    static constexpr uint64_t QUANT_OUTPUT_UB_OFFSET = static_cast<uint64_t>(MAX_TILE_ELEMENTS) * sizeof(DataTypeIn);
    static constexpr uint64_t QUANT_SCALE_UB_OFFSET = QUANT_OUTPUT_UB_OFFSET + MAX_TILE_ELEMENTS * sizeof(int8_t);
    static constexpr uint64_t SWISH_OUTPUT_UB_OFFSET = QUANT_SCALE_UB_OFFSET +
                                                       MAX_TILE_ELEMENTS / MX_GROUP_SIZE * sizeof(int8_t);
    static constexpr uint64_t GROUP_MAX_UB_OFFSET = SWISH_OUTPUT_UB_OFFSET + MAX_TILE_ELEMENTS * sizeof(bfloat16_t);
    static constexpr uint64_t SCALE_RECIPROCAL_UB_OFFSET = GROUP_MAX_UB_OFFSET +
                                                           MAX_TILE_ELEMENTS / MX_GROUP_SIZE * sizeof(uint16_t);
    static constexpr uint64_t PADDED_QUANT_SCALE_UB_OFFSET = SCALE_RECIPROCAL_UB_OFFSET +
                                                             MAX_TILE_ELEMENTS / MX_GROUP_SIZE * sizeof(uint16_t);
    static constexpr uint16_t FP8_MAX_EXP_BITS = AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ?
                                                     FP8_E4M3_MAX_EXP_BITS :
                                                     FP8_E5M2_MAX_EXP_BITS;
    static constexpr float INV_FP8_MAX_VALUE = AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ?
                                                   1.0F / FP8_E4M3_MAX_VALUE :
                                                   1.0F / FP8_E5M2_MAX_VALUE;

    __aicore__ inline void ComputeSwish(uint16_t mSize, uint32_t nSize, uint32_t nAligned);
    __aicore__ inline void ComputeMxQuant(uint32_t elementCount);
    __aicore__ inline void PadMxScaleRows(uint16_t mSize, uint32_t scaleCols, uint32_t scaleCopyCols);
    __aicore__ inline void CopyOutputFromUb2Gm(uint16_t rowCount, uint32_t nSize, uint32_t nAligned, uint64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint16_t rowCount, uint32_t scaleCols, uint32_t scaleCopyCols,
                                              uint64_t gmOffset);

    const Params* params_{nullptr};
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    uint64_t n_{0};
    uint64_t scaleRowStride_{0};
    uint32_t scalePaddingCols_{0};
    uint32_t subBlockIdx_{0};
    float beta_{1.0F};
    bool useCublas_{false};
};

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::Init(const Params& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL_BIT, FLOAT_OVERFLOW_MODE_CTRL_BIT>(0);
    params_ = &params;
    subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
    useCublas_ = params.scaleAlg == SCALE_ALG_CUBLAS;
    beta_ = params.beta;
    UpdateGlobalAddr({0, 0});
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + asc::te::get<0>(baseOffset);
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) + asc::te::get<1>(baseOffset);
    }
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = static_cast<uint64_t>(asc::te::get<Blaze::Gemm::MNK_N>(problemShape));
    const uint64_t scaleGroups = (n_ + MX_GROUP_MASK) >> MX_GROUP_SHIFT;
    scalePaddingCols_ = static_cast<uint32_t>(scaleGroups & 1U);
    scaleRowStride_ = scaleGroups + scalePaddingCols_;
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::operator()(const BlockShape& blockShape,
                                                                                           const BlockCoord& blockCoord)
{
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t singleM = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_M>(blockShape));
    const uint32_t halfSingleM = Blaze::Gemm::CeilDiv(singleM, static_cast<uint32_t>(AscendC::GetTaskRation()));
    const uint16_t singleMInVec = static_cast<uint16_t>(subBlockIdx_ == 1 ? singleM - halfSingleM : halfSingleM);
    if (singleMInVec == 0) {
        Gemm::Sync::NotifyCube();
        return;
    }
    const uint64_t mOffset = subBlockIdx_ * halfSingleM;
    const uint32_t singleN = static_cast<uint32_t>(asc::te::get<Blaze::Gemm::MNK_N>(blockShape));
    const uint32_t nAligned = (singleN + MX_GROUP_MASK) & ~MX_GROUP_MASK;
    const uint32_t elementCount = static_cast<uint32_t>(singleMInVec) * nAligned;

    ComputeSwish(singleMInVec, singleN, nAligned);
    // MX quantization only reads the BF16 result; the FP32 input can be reused by Fixpipe.
    Gemm::Sync::NotifyCube();
    ComputeMxQuant(elementCount);

    const uint32_t scaleCols = nAligned >> MX_GROUP_SHIFT;
    uint32_t scaleCopyCols = scaleCols;
    if (scalePaddingCols_ != 0 && asc::te::get<0>(blockCoord) % n_ + singleN == n_) {
        // Only the last N tile owns the padding byte in the paired E8M0 output layout.
        ++scaleCopyCols;
        PadMxScaleRows(singleMInVec, scaleCols, scaleCopyCols);
    }
    const uint64_t yOffset = asc::te::get<0>(blockCoord) + mOffset * n_;
    const uint64_t yScaleOffset = asc::te::get<1>(blockCoord) + mOffset * scaleRowStride_;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, singleN, nAligned, yOffset);
    CopyScaleFromUb2Gm(singleMInVec, scaleCols, scaleCopyCols, yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::CopyOutputFromUb2Gm(uint16_t rowCount,
                                                                                                    uint32_t nSize,
                                                                                                    uint32_t nAligned,
                                                                                                    uint64_t gmOffset)
{
    const int64_t nValid = static_cast<int64_t>(nSize);
    const int64_t nUbAligned = static_cast<int64_t>(nAligned);
    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), nValid, nUbAligned);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), nValid, static_cast<int64_t>(n_));
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_OUTPUT_UB_OFFSET),
                                      ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantOutputGmAddr_ + gmOffset),
                                      gmLayout);
    asc::te::copy(outGm, outUb);
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::CopyScaleFromUb2Gm(
    uint16_t rowCount, uint32_t scaleCols, uint32_t scaleCopyCols, uint64_t gmOffset)
{
    const bool needsScalePadding = scaleCopyCols != scaleCols;
    const int64_t ubStride = needsScalePadding ? AscendC::ONE_BLK_SIZE : scaleCols;
    const uint64_t ubOffset = needsScalePadding ? PADDED_QUANT_SCALE_UB_OFFSET : QUANT_SCALE_UB_OFFSET;
    // Equal row stride and copy length select compact UB-to-GM transfer.
    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), static_cast<int64_t>(scaleCopyCols),
                                          ubStride);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(rowCount), static_cast<int64_t>(scaleCopyCols),
                                          static_cast<int64_t>(scaleRowStride_));
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOffset), ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantScaleGmAddr_ + gmOffset),
                                      gmLayout);
    asc::te::copy(outGm, outUb);
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::ComputeSwish(uint16_t mSize,
                                                                                             uint32_t nSize,
                                                                                             uint32_t nAligned)
{
    auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize), static_cast<int64_t>(nSize),
                                        static_cast<int64_t>(nAligned));
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layout);
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(SWISH_OUTPUT_UB_OFFSET), layout);
    Tile::Swish::Run(srcTensor, dstTensor, beta_);
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::ComputeMxQuant(uint32_t elementCount)
{
    const uint32_t scaleCount = elementCount >> MX_GROUP_SHIFT;
    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(elementCount),
                                                    static_cast<int64_t>(elementCount));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(scaleCount),
                                                     static_cast<int64_t>(scaleCount));
    auto swishTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(SWISH_OUTPUT_UB_OFFSET), dataLayout);
    // OCP stores BF16 exponent bits; CUBLAS stores absolute BF16 value bits.
    auto groupMaxTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GROUP_MAX_UB_OFFSET), scaleLayout);
    auto quantScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_SCALE_UB_OFFSET), scaleLayout);
    auto scaleReciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(SCALE_RECIPROCAL_UB_OFFSET), scaleLayout);
    auto quantOutputTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_OUTPUT_UB_OFFSET), dataLayout);
    const bool useCublas = useCublas_;
    Tile::MxQuantConfig mxQuantConfig{useCublas ? Tile::MxScaleAlg::CUBLAS : Tile::MxScaleAlg::OCP, FP8_MAX_EXP_BITS,
                                      INV_FP8_MAX_VALUE, MX_SCALE_ADD_VALUE_BITS, false};
    Tile::MxQuant<DataTypeOut> mxQuant;
    mxQuant.GroupMaxExp(swishTensor, groupMaxTensor, elementCount, useCublas);
    // Wait for the previous MTE3 copies before GenScale/Quantize reuse the scale/output buffers.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    mxQuant.GenScale(groupMaxTensor, quantScaleTensor, scaleReciprocalTensor, mxQuantConfig, scaleCount);
    // FP8 quantization uses RINT.
    mxQuant.Quantize(swishTensor, scaleReciprocalTensor, quantOutputTensor, elementCount);
}

SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
__aicore__ inline void BlockEpilogueSwishMxQuant<SWISH_MX_QUANT_TEMPLATE_ARGS>::PadMxScaleRows(uint16_t mSize,
                                                                                               uint32_t scaleCols,
                                                                                               uint32_t scaleCopyCols)
{
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, scaleCols, scaleCols);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, scaleCopyCols, AscendC::ONE_BLK_SIZE);
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(QUANT_SCALE_UB_OFFSET),
                                          srcLayout);
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(PADDED_QUANT_SCALE_UB_OFFSET), dstLayout);
    Tile::MxQuant<DataTypeOut> mxQuant;
    mxQuant.PadScaleRows(srcTensor, dstTensor, mSize, scaleCols, scaleCopyCols);
}

#undef SWISH_MX_QUANT_CLASS_TEMPLATE_PARAMS
#undef SWISH_MX_QUANT_TEMPLATE_ARGS

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
