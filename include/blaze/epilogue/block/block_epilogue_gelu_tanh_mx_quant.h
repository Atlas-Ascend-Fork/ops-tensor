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
 * \file block_epilogue_gelu_tanh_mx_quant.h
 * \brief MIX epilogue: float L0C in UB -> GeluTanh -> dynamic MX y/yScale.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "interface/reg_compute/kernel_reg_compute_utils.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

namespace {
constexpr uint32_t GELU_MX_BLOCK_SIZE = 32;
constexpr int64_t GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK = 64;
constexpr uint32_t GELU_MX_MAX_SINGLE_MN = 128 * 256;
constexpr uint16_t GELU_MX_MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t GELU_MX_MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t GELU_MX_BF16_EXP_BIAS = 0x7f00;
constexpr int16_t GELU_MX_SHR_NUM_FOR_BF16 = 7;
constexpr int16_t GELU_MX_SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t GELU_MX_NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t GELU_MX_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t GELU_MX_FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t GELU_MX_FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t GELU_MX_FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t GELU_MX_FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t GELU_MX_ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t GELU_MX_MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t GELU_MX_MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t GELU_MX_NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t GELU_MX_MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t GELU_MX_FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t GELU_MX_NUMBER_ZERO = 0x00000000;
constexpr uint32_t GELU_MX_NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t GELU_MX_NUMBER_HALF = 0x00400000;
constexpr uint32_t GELU_MX_INTERLEAVED_REG_FACTOR = 2;
constexpr uint32_t GELU_MX_HALF_REG_FACTOR = 2;
constexpr uint32_t GELU_MX_SCALE_ALG_OCP = 0;
constexpr uint32_t GELU_MX_SCALE_ALG_DYNAMIC_DTYPE_RANGE = 2;
constexpr uint16_t GELU_MX_BF16_ADD_VALUE_MAN1 = 0x003f;
constexpr uint16_t GELU_MX_BF16_ADD_VALUE_MAN2 = 0x001f;
constexpr float GELU_MX_DEFAULT_DST_TYPE_MAX = 0.0f;
constexpr float GELU_MX_FP4_E2M1_DST_TYPE_MAX = 6.0f;
constexpr float GELU_MX_FP4_E2M1_SPECIAL_DST_TYPE_MAX = 7.0f;
constexpr float GELU_MX_SCALAR_ONE = 1.0f;
} // namespace

template <typename DataTypeOut_, typename DataTypeIn_ = float>
class BlockEpilogueGeluTanhMxQuant {
public:
    static constexpr uint32_t EPILOGUE_UB_DB_COUNT = 2;
    static constexpr uint32_t EPILOGUE_BASE_N_ALIGNMENT = AscendC::ONE_BLK_SIZE;
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct OutputOffsets {
        int64_t yOffset{0};
        int64_t yScaleOffset{0};
    };

    static_assert(AscendC::IsSameType<DataTypeIn, float>::value,
                  "BlockEpilogueGeluTanhMxQuant only supports float L0C input.");
    static_assert(AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value,
                  "BlockEpilogueGeluTanhMxQuant only supports FP8 or FP4 output.");

    __aicore__ inline BlockEpilogueGeluTanhMxQuant() {}

    __aicore__ inline ~BlockEpilogueGeluTanhMxQuant()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (!isValid_) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        if (bufferCount_ == EPILOGUE_UB_DB_COUNT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        }
    }

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t scaleAlg{0};
        float dstTypeMax{0.0f};
        // Host tiling contract: ceil(baseM / GetTaskRation()) * baseN <=
        // GELU_MX_MAX_SINGLE_MN. Each AIV owns one such M partition and the
        // scratch buffers below are sized for that partition.
    };

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        isValid_ = true;
        const bool isValidBaseN = params.baseN != 0 && params.baseN % EPILOGUE_BASE_N_ALIGNMENT == 0;
        ASCENDC_ASSERT(isValidBaseN, {
            KERNEL_LOG(KERNEL_ERROR, "GMMAQ epilogue baseN must be aligned to %u, got %u.", EPILOGUE_BASE_N_ALIGNMENT,
                       params.baseN);
        });
        if (!isValidBaseN) {
            isValid_ = false;
            return;
        }
        params_ = &params;
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
            fpEmax_ = GELU_MX_FP8_E4M3_MAX_EXP;
            invDstTypeMax_ = GELU_MX_SCALAR_ONE / 448.0f;
        } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
            fpEmax_ = GELU_MX_FP8_E5M2_MAX_EXP;
            invDstTypeMax_ = GELU_MX_SCALAR_ONE / 57344.0f;
        } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            fpEmax_ = GELU_MX_FP4_E2M1_MAX_EXP;
            invDstTypeMax_ = GELU_MX_SCALAR_ONE / GELU_MX_FP4_E2M1_DST_TYPE_MAX;
        } else {
            fpEmax_ = GELU_MX_FP4_E1M2_MAX_EXP;
            invDstTypeMax_ = GELU_MX_SCALAR_ONE / 3.5f;
        }
        if (params_->scaleAlg == GELU_MX_SCALE_ALG_DYNAMIC_DTYPE_RANGE &&
            params_->dstTypeMax != GELU_MX_DEFAULT_DST_TYPE_MAX) {
            invDstTypeMax_ = GELU_MX_SCALAR_ONE / params_->dstTypeMax;
        }
        addValueBits_ = params_->dstTypeMax == GELU_MX_FP4_E2M1_SPECIAL_DST_TYPE_MAX ? GELU_MX_BF16_ADD_VALUE_MAN2 :
                                                                                       GELU_MX_BF16_ADD_VALUE_MAN1;

        const uint64_t mPerVector = Gemm::CeilDiv(static_cast<uint64_t>(params.baseM),
                                                  static_cast<uint64_t>(Gemm::DOUBLE_BUFFER_COUNT));
        const uint64_t maxBlockCount = mPerVector * params.baseN;
        const uint64_t maxScaleCount = Gemm::CeilDiv(maxBlockCount, static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        const uint64_t afterIn = Gemm::Align32(maxBlockCount * sizeof(DataTypeIn));
        const uint64_t quantOutputBytes = Gemm::Align32(maxBlockCount * sizeof(int8_t));
        const uint64_t scaleOutputBytes = maxScaleCount * sizeof(int8_t);
        const uint64_t activationBytes = maxBlockCount * sizeof(bfloat16_t);
        const uint64_t exponentBytes = maxScaleCount * sizeof(uint16_t);
        const uint64_t scaleBlockBytes = mPerVector * AscendC::ONE_BLK_SIZE * sizeof(int8_t);

        const uint64_t singleAfterOutput = Gemm::Align32(afterIn + quantOutputBytes);
        const uint64_t singleAfterIo = Gemm::Align32(singleAfterOutput + scaleOutputBytes);
        const uint64_t singleAfterActivation = Gemm::Align32(singleAfterIo + activationBytes);
        const uint64_t singleAfterMaxExp = Gemm::Align32(singleAfterActivation + exponentBytes);
        const uint64_t singleScaleBlockOffset = Gemm::Align32(singleAfterMaxExp + exponentBytes);

        const uint64_t doubleAfterOutput = Gemm::Align32(afterIn + 2 * quantOutputBytes);
        const uint64_t doubleAfterIo = Gemm::Align32(doubleAfterOutput + scaleOutputBytes);
        const uint64_t doubleAfterActivation = Gemm::Align32(doubleAfterIo + activationBytes);
        const uint64_t doubleAfterMaxExp = Gemm::Align32(doubleAfterActivation + exponentBytes);
        const uint64_t doubleScaleBlockOffset = Gemm::Align32(doubleAfterMaxExp + exponentBytes);
        const uint64_t doubleBufferBytes = doubleScaleBlockOffset + 2 * scaleBlockBytes;
        bufferCount_ = doubleBufferBytes <= AscendC::TOTAL_UB_SIZE ? EPILOGUE_UB_DB_COUNT : 1U;

        for (uint32_t slot = 0; slot < bufferCount_; ++slot) {
            quantOutput_[slot] = AscendC::LocalTensor<int8_t>(
                AscendC::TPosition::VECOUT, Gemm::Align32(afterIn + slot * quantOutputBytes), maxBlockCount);
        }
        const uint64_t afterOutput = bufferCount_ == EPILOGUE_UB_DB_COUNT ? doubleAfterOutput : singleAfterOutput;
        quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterOutput, maxScaleCount);
        const uint64_t afterIo = bufferCount_ == EPILOGUE_UB_DB_COUNT ? doubleAfterIo : singleAfterIo;
        activationResult_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIo, maxBlockCount);
        const uint64_t afterActivation = bufferCount_ == EPILOGUE_UB_DB_COUNT ? doubleAfterActivation :
                                                                                singleAfterActivation;
        maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterActivation, maxScaleCount);
        const uint64_t afterMaxExp = bufferCount_ == EPILOGUE_UB_DB_COUNT ? doubleAfterMaxExp : singleAfterMaxExp;
        halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterMaxExp, maxScaleCount);
        const uint64_t scaleBlockOffset = bufferCount_ == EPILOGUE_UB_DB_COUNT ? doubleScaleBlockOffset :
                                                                                 singleScaleBlockOffset;
        for (uint32_t slot = 0; slot < bufferCount_; ++slot) {
            quantScaleBlockOutput_[slot] = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT,
                                                                        scaleBlockOffset + slot * scaleBlockBytes,
                                                                        mPerVector * AscendC::ONE_BLK_SIZE);
        }

        quantOutputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.yGmAddr));
        quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.yScaleGmAddr));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        if (bufferCount_ == EPILOGUE_UB_DB_COUNT) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        }
    }

    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets)
    {
        if ASCEND_IS_AIV {
            int64_t yBaseOffset = baseOffsets.yOffset;
            if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                          AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
                yBaseOffset >>= 1;
            }
            quantOutputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + yBaseOffset);
            quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) +
                                              baseOffsets.yScaleOffset);
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        n_ = asc::te::get<Gemm::MNK_N>(problemShape);
        scaleN_ = Gemm::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) *
                  Gemm::MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline void operator()(const BlockShape& blockShape, const OutputOffsets& outputOffsets)
    {
        if (!isValid_) {
            return;
        }
        singleM_ = asc::te::get<Gemm::MNK_M>(blockShape);
        singleN_ = asc::te::get<Gemm::MNK_N>(blockShape);
        scaleBlockN_ = Gemm::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) *
                       Gemm::MXFP_MULTI_BASE_SIZE;
        const uint64_t halfSingleM = Gemm::CeilDiv(static_cast<uint64_t>(singleM_),
                                                   static_cast<uint64_t>(AscendC::GetTaskRation()));
        const uint64_t mOffset = subBlockIdx_ * halfSingleM;
        if (mOffset >= singleM_) {
            return;
        }
        const uint64_t singleMInVector = Gemm::Min(static_cast<uint64_t>(singleM_) - mOffset, halfSingleM);
        vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
        elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / GELU_MX_BLOCK_SIZE;

        const uint32_t slot = bufferCount_ == EPILOGUE_UB_DB_COUNT ? pingPongId_ : 0U;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        RunActivation(singleMInVector);
        RunDynamicMxQuant(singleMInVector, slot);
        const int64_t yOffset = outputOffsets.yOffset + static_cast<int64_t>(mOffset) * n_;
        const int64_t yScaleOffset = outputOffsets.yScaleOffset + static_cast<int64_t>(mOffset) * scaleN_;
        TransScaleLayout(singleMInVector, slot);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        CopyOutputToGm(singleMInVector, yOffset, slot);
        CopyScaleToGm(singleMInVector, yScaleOffset, slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        if (bufferCount_ == EPILOGUE_UB_DB_COUNT) {
            pingPongId_ ^= 1U;
        }
    }

private:
    __aicore__ inline void CopyOutputToGm(uint64_t blockCount, int64_t offset, uint32_t slot)
    {
        AscendC::DataCopyExtParams params{1, 0, 0, 0, 0};
        params.blockCount = blockCount;
        params.blockLen = singleN_ * sizeof(int8_t);
        params.dstStride = (n_ - singleN_) * sizeof(int8_t);
        if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
            params.blockLen >>= 1;
            params.dstStride >>= 1;
            offset >>= 1;
        }
        AscendC::DataCopyPad(quantOutputGlobal_[offset], quantOutput_[slot], params);
    }

    __aicore__ inline void CopyScaleToGm(uint64_t blockCount, int64_t offset, uint32_t slot)
    {
        AscendC::DataCopyExtParams params{1, 0, 0, 0, 0};
        params.blockCount = blockCount;
        params.blockLen = scaleBlockN_ * sizeof(int8_t);
        params.dstStride = (scaleN_ - scaleBlockN_) * sizeof(int8_t);
        AscendC::DataCopyPad(quantScaleGlobal_[offset], quantScaleBlockOutput_[slot], params);
    }

    __aicore__ inline void ComputeMaxExpOcp(__ubuf__ bfloat16_t* src, __ubuf__ uint16_t* maxExp, uint32_t totalCount,
                                            uint16_t loopCount)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> value0;
            AscendC::Reg::RegTensor<bfloat16_t> value1;
            AscendC::Reg::RegTensor<uint16_t> exp0;
            AscendC::Reg::RegTensor<uint16_t> exp1;
            AscendC::Reg::RegTensor<uint16_t> expMask;
            AscendC::Reg::RegTensor<uint16_t> maxValue;
            AscendC::Reg::MaskReg mask0;
            AscendC::Reg::MaskReg mask1;
            AscendC::Reg::MaskReg maskEven;
            AscendC::Reg::MaskReg maskOdd;
            AscendC::Reg::UnalignReg unalign;
            AscendC::Reg::Duplicate(expMask, GELU_MX_MAX_EXP_FOR_BF16);
            uint32_t remainingData = totalCount;
            for (uint16_t i = 0; i < loopCount; ++i) {
                mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    value0, value1, src, vlForHalfNumber_ * GELU_MX_INTERLEAVED_REG_FACTOR);
                AscendC::Reg::And(exp0, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0), expMask,
                                  maskEven);
                AscendC::Reg::And(exp1, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), expMask, maskOdd);
                AscendC::Reg::Max(maxValue, exp0, exp1, mask0);
                AscendC::Reg::ReduceMaxWithDataBlock(maxValue, maxValue, mask0);
                AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxExp, maxValue, unalign, elementAfterReduce_);
            }
            AscendC::Reg::DataCopyUnAlignPost(maxExp, unalign, 0);
        }
    }

    __aicore__ inline void ComputeScaleOcp(__ubuf__ uint16_t* maxExp, __ubuf__ uint16_t* mxScale,
                                           __ubuf__ uint16_t* reciprocalScale, uint32_t totalScale, uint16_t loopCount)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint16_t> expMask;
            AscendC::Reg::RegTensor<uint16_t> sharedExp;
            AscendC::Reg::RegTensor<uint16_t> scaleValue;
            AscendC::Reg::RegTensor<uint16_t> scaleBias;
            AscendC::Reg::RegTensor<uint16_t> halfScale;
            AscendC::Reg::RegTensor<uint16_t> fp8Nan;
            AscendC::Reg::RegTensor<uint16_t> maxValue;
            AscendC::Reg::RegTensor<uint16_t> maxExpValue;
            AscendC::Reg::RegTensor<uint16_t> zero;
            AscendC::Reg::RegTensor<uint16_t> nan;
            AscendC::Reg::RegTensor<uint16_t> special;
            AscendC::Reg::MaskReg invalidMask;
            AscendC::Reg::MaskReg infNanMask;
            AscendC::Reg::MaskReg zeroMask;
            AscendC::Reg::MaskReg specialMask;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::Duplicate(expMask, GELU_MX_MAX_EXP_FOR_BF16);
            AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
            AscendC::Reg::Duplicate(scaleBias, GELU_MX_BF16_EXP_BIAS);
            AscendC::Reg::Duplicate(fp8Nan, GELU_MX_MAX_EXP_FOR_FP8);
            AscendC::Reg::Duplicate(zero, 0);
            AscendC::Reg::Duplicate(nan, GELU_MX_NAN_CUSTOMIZATION);
            AscendC::Reg::Duplicate(special, GELU_MX_SPECIAL_EXP_THRESHOLD);
            uint32_t remainingScale = totalScale;
            for (uint16_t i = 0; i < loopCount; ++i) {
                mask = AscendC::Reg::UpdateMask<uint16_t>(remainingScale);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxValue, maxExp,
                                                                                              vlForHalfNumber_);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(infNanMask, maxValue, expMask, mask);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidMask, maxValue, maxExpValue, mask);
                AscendC::Reg::Select<uint16_t>(maxValue, maxExpValue, maxValue, invalidMask);
                AscendC::Reg::Sub(sharedExp, maxValue, maxExpValue, mask);
                AscendC::Reg::ShiftRights(scaleValue, sharedExp, GELU_MX_SHR_NUM_FOR_BF16, mask);
                AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, infNanMask);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    mxScale, scaleValue, vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR, mask);

                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zero, mask);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialMask, sharedExp, scaleBias, mask);
                AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, mask);
                AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nan, infNanMask);
                AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zero, zeroMask);
                AscendC::Reg::Select<uint16_t>(halfScale, special, halfScale, specialMask);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    reciprocalScale, halfScale, vlForHalfNumber_, mask);
            }
        }
    }

    __aicore__ inline void ComputeMaxExpCublas(__ubuf__ bfloat16_t* src, __ubuf__ uint16_t* maxExp, uint32_t totalCount,
                                               uint16_t loopCount)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> value0;
            AscendC::Reg::RegTensor<bfloat16_t> value1;
            AscendC::Reg::RegTensor<uint16_t> absMask;
            AscendC::Reg::RegTensor<uint16_t> maxValue;
            AscendC::Reg::MaskReg mask0;
            AscendC::Reg::MaskReg mask1;
            AscendC::Reg::MaskReg maskEven;
            AscendC::Reg::MaskReg maskOdd;
            AscendC::Reg::UnalignReg unalign;
            AscendC::Reg::Duplicate(absMask, GELU_MX_ABS_MASK_FOR_16BIT);
            uint32_t remainingData = totalCount;
            for (uint16_t i = 0; i < loopCount; ++i) {
                mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    value0, value1, src, vlForHalfNumber_ * GELU_MX_INTERLEAVED_REG_FACTOR);
                AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                                  reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0), absMask, maskEven);
                AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1),
                                  reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), absMask, maskOdd);
                AscendC::Reg::Max(maxValue, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                                  reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), mask0);
                AscendC::Reg::ReduceMaxWithDataBlock(maxValue, maxValue, mask0);
                AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxExp, maxValue, unalign, elementAfterReduce_);
            }
            AscendC::Reg::DataCopyUnAlignPost(maxExp, unalign, 0);
        }
    }

    __aicore__ inline void ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExp, __ubuf__ uint16_t* mxScale,
                                                         __ubuf__ uint16_t* reciprocalScale, uint32_t totalScale,
                                                         uint16_t loopCount)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint16_t> maxValue;
            AscendC::Reg::RegTensor<uint16_t> maxExpOnly;
            AscendC::Reg::RegTensor<uint16_t> roundedMaxExp;
            AscendC::Reg::RegTensor<uint16_t> sharedExp;
            AscendC::Reg::RegTensor<uint16_t> scaleValue;
            AscendC::Reg::RegTensor<uint16_t> reciprocalValue;
            AscendC::Reg::RegTensor<uint16_t> expMask;
            AscendC::Reg::RegTensor<uint16_t> addValue;
            AscendC::Reg::RegTensor<uint16_t> maxExpValue;
            AscendC::Reg::RegTensor<uint16_t> scaleBias;
            AscendC::Reg::RegTensor<uint16_t> fp8Nan;
            AscendC::Reg::RegTensor<uint16_t> zero;
            AscendC::Reg::RegTensor<uint16_t> nan;
            AscendC::Reg::RegTensor<uint16_t> special;
            AscendC::Reg::MaskReg finiteMask;
            AscendC::Reg::MaskReg zeroMask;
            AscendC::Reg::MaskReg belowRangeMask;
            AscendC::Reg::MaskReg specialMask;
            AscendC::Reg::MaskReg mask;

            AscendC::Reg::Duplicate(expMask, GELU_MX_MAX_EXP_FOR_BF16);
            AscendC::Reg::Duplicate(addValue, addValueBits_);
            AscendC::Reg::Duplicate(maxExpValue, GELU_MX_FP4_E2M1_MAX_EXP);
            AscendC::Reg::Duplicate(scaleBias, GELU_MX_BF16_EXP_BIAS);
            AscendC::Reg::Duplicate(fp8Nan, GELU_MX_MAX_EXP_FOR_FP8);
            AscendC::Reg::Duplicate(zero, 0);
            AscendC::Reg::Duplicate(nan, GELU_MX_NAN_CUSTOMIZATION);
            AscendC::Reg::Duplicate(special, GELU_MX_SPECIAL_EXP_THRESHOLD);
            uint32_t remainingScale = totalScale;

            for (uint16_t i = 0; i < loopCount; ++i) {
                mask = AscendC::Reg::UpdateMask<uint16_t>(remainingScale);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxValue, maxExp,
                                                                                              vlForHalfNumber_);
                AscendC::Reg::And(maxExpOnly, maxValue, expMask, mask);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(finiteMask, maxExpOnly, expMask, mask);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(belowRangeMask, maxExpOnly, maxExpValue, mask);
                AscendC::Reg::Add(roundedMaxExp, maxValue, addValue, mask);
                AscendC::Reg::And(roundedMaxExp, roundedMaxExp, expMask, mask);
                AscendC::Reg::Select<uint16_t>(roundedMaxExp, maxExpValue, roundedMaxExp, belowRangeMask);
                AscendC::Reg::Sub(sharedExp, roundedMaxExp, maxExpValue, mask);
                AscendC::Reg::ShiftRights(scaleValue, sharedExp, GELU_MX_SHR_NUM_FOR_BF16, mask);
                AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, finiteMask);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    mxScale, scaleValue, vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR, mask);

                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zero, mask);
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialMask, sharedExp, scaleBias, mask);
                AscendC::Reg::Sub(reciprocalValue, scaleBias, sharedExp, mask);
                AscendC::Reg::Select<uint16_t>(reciprocalValue, reciprocalValue, nan, finiteMask);
                AscendC::Reg::Select<uint16_t>(reciprocalValue, reciprocalValue, zero, zeroMask);
                AscendC::Reg::Select<uint16_t>(reciprocalValue, special, reciprocalValue, specialMask);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                    reciprocalScale, reciprocalValue, vlForHalfNumber_, mask);
            }
        }
    }

    __aicore__ inline void ComputeScaleCublas(__ubuf__ uint16_t* maxExp, __ubuf__ uint16_t* mxScale,
                                              __ubuf__ uint16_t* reciprocalScale, uint32_t totalScale,
                                              uint16_t loopCount)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint16_t> max16;
            AscendC::Reg::RegTensor<uint32_t> max32;
            AscendC::Reg::RegTensor<uint32_t> exp32;
            AscendC::Reg::RegTensor<uint32_t> mantissa32;
            AscendC::Reg::RegTensor<uint32_t> expAddOne32;
            AscendC::Reg::RegTensor<uint32_t> extractExp;
            AscendC::Reg::RegTensor<uint16_t> expOut;
            AscendC::Reg::RegTensor<uint32_t> halfScale;
            AscendC::Reg::RegTensor<uint16_t> reciprocalExpOut;
            AscendC::Reg::RegTensor<float> invMax;
            AscendC::Reg::RegTensor<uint32_t> mantissaMask;
            AscendC::Reg::RegTensor<uint32_t> expMask;
            AscendC::Reg::RegTensor<uint32_t> zero;
            AscendC::Reg::RegTensor<uint32_t> scaleBias;
            AscendC::Reg::RegTensor<uint32_t> nan;
            AscendC::Reg::RegTensor<uint32_t> fp8Nan;
            AscendC::Reg::MaskReg finiteMask;
            AscendC::Reg::MaskReg nonzeroMask;
            AscendC::Reg::MaskReg predicate0;
            AscendC::Reg::MaskReg predicate1;
            AscendC::Reg::MaskReg predicate2;
            uint32_t remainingScale = totalScale;
            AscendC::Reg::MaskReg maskB16;
            AscendC::Reg::MaskReg maskFloat;
            static constexpr AscendC::Reg::CastTrait castBf16ToFloat = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};

            AscendC::Reg::Duplicate(invMax, invDstTypeMax_);
            AscendC::Reg::Duplicate(mantissaMask, GELU_MX_MAN_MASK_FLOAT);
            AscendC::Reg::Duplicate(expMask, GELU_MX_MAX_EXP_FOR_FP32);
            AscendC::Reg::Duplicate(zero, 0);
            AscendC::Reg::Duplicate(scaleBias, GELU_MX_FP32_EXP_BIAS_CUBLAS);
            AscendC::Reg::Duplicate(nan, GELU_MX_NAN_CUSTOMIZATION_PACK);
            AscendC::Reg::Duplicate(fp8Nan, GELU_MX_MAX_EXP_FOR_FP8_IN_FP32);

            for (uint16_t i = 0; i < loopCount; ++i) {
                const uint32_t processedScale = remainingScale < vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR ?
                                                    remainingScale :
                                                    vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR;
                uint32_t b16MaskElementCount = processedScale;
                uint32_t floatMaskElementCount = processedScale;
                maskB16 = AscendC::Reg::UpdateMask<uint16_t>(b16MaskElementCount);
                maskFloat = AscendC::Reg::UpdateMask<uint32_t>(floatMaskElementCount);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    max16, maxExp, vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR);
                AscendC::Reg::Cast<float, bfloat16_t, castBf16ToFloat>(
                    reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                    reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(max16), maskFloat);
                AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(finiteMask, max32, expMask, maskFloat);
                AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(nonzeroMask, max32, zero, maskFloat);
                AscendC::Reg::Mul(reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                                  reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32), invMax, maskFloat);
                AscendC::Reg::ShiftRights(exp32, max32, GELU_MX_SHR_NUM_FOR_FP32, maskFloat);
                AscendC::Reg::And(mantissa32, max32, mantissaMask, maskFloat);

                AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate0, exp32, GELU_MX_NUMBER_ZERO,
                                                                            maskFloat);
                AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(predicate1, exp32,
                                                                            GELU_MX_NUMBER_TWO_FIVE_FOUR, maskFloat);
                AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate2, mantissa32, GELU_MX_NUMBER_ZERO,
                                                                            maskFloat);
                AscendC::Reg::MaskAnd(predicate0, predicate0, predicate1, maskFloat);
                AscendC::Reg::MaskAnd(predicate0, predicate0, predicate2, maskFloat);
                AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(predicate1, exp32, GELU_MX_NUMBER_ZERO,
                                                                            maskFloat);
                AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate2, mantissa32, GELU_MX_NUMBER_HALF,
                                                                            maskFloat);
                AscendC::Reg::MaskAnd(predicate1, predicate1, predicate2, maskFloat);
                AscendC::Reg::MaskOr(predicate0, predicate0, predicate1, maskFloat);

                AscendC::Reg::Adds(expAddOne32, exp32, 1, maskFloat);
                AscendC::Reg::Select(extractExp, expAddOne32, exp32, predicate0);
                AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan, finiteMask);
                AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zero, nonzeroMask);
                AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                    mxScale + i * vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR / GELU_MX_HALF_REG_FACTOR, expOut,
                    maskB16);

                AscendC::Reg::ShiftLefts(extractExp, extractExp, GELU_MX_SHR_NUM_FOR_BF16, maskFloat);
                AscendC::Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
                AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nan, finiteMask);
                AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zero, nonzeroMask);
                AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(reciprocalExpOut, halfScale);
                AscendC::Reg::DataCopy<uint16_t>(reciprocalScale + i * vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR,
                                                 reciprocalExpOut, maskB16);
                remainingScale = remainingScale > processedScale ? remainingScale - processedScale : 0;
            }
        }
    }

    __aicore__ inline void QuantizeFp8(__ubuf__ bfloat16_t* src, __ubuf__ uint16_t* reciprocalScale,
                                       __ubuf__ int8_t* output, uint32_t totalCount, uint16_t loopCount)
    {
        uint32_t remainingData = totalCount;
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg mask0;
            AscendC::Reg::MaskReg mask1;
            AscendC::Reg::MaskReg mask2;
            AscendC::Reg::MaskReg mask3;
            AscendC::Reg::MaskReg maskEven;
            AscendC::Reg::MaskReg maskOdd;
            AscendC::Reg::RegTensor<uint16_t> scale;
            AscendC::Reg::RegTensor<bfloat16_t> value0;
            AscendC::Reg::RegTensor<bfloat16_t> value1;
            AscendC::Reg::RegTensor<float> fp32Value00;
            AscendC::Reg::RegTensor<float> fp32Value01;
            AscendC::Reg::RegTensor<float> fp32Value10;
            AscendC::Reg::RegTensor<float> fp32Value11;
            AscendC::Reg::RegTensor<DataTypeOut> fp8Value00;
            AscendC::Reg::RegTensor<DataTypeOut> fp8Value01;
            AscendC::Reg::RegTensor<DataTypeOut> fp8Value10;
            AscendC::Reg::RegTensor<DataTypeOut> fp8Value11;
            static constexpr AscendC::Reg::CastTrait castZero = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castOne = {
                AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castFp8 = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_RINT};

            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t expandedRemaining = remainingData * GELU_MX_INTERLEAVED_REG_FACTOR;
                mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                mask2 = AscendC::Reg::UpdateMask<bfloat16_t>(expandedRemaining);
                mask3 = AscendC::Reg::UpdateMask<bfloat16_t>(expandedRemaining);
                AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    value0, value1, src, vlForHalfNumber_ * GELU_MX_INTERLEAVED_REG_FACTOR);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_E2B_B16>(scale, reciprocalScale,
                                                                             elementAfterReduce_);
                AscendC::Reg::Mul(value0, value0, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale),
                                  maskEven);
                AscendC::Reg::Mul(value1, value1, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale),
                                  maskOdd);
                AscendC::Reg::Interleave(value0, value1, value0, value1);
                AscendC::Reg::Cast<float, bfloat16_t, castZero>(fp32Value00, value0, mask0);
                AscendC::Reg::Cast<float, bfloat16_t, castOne>(fp32Value01, value0, mask0);
                AscendC::Reg::Interleave(fp32Value00, fp32Value01, fp32Value00, fp32Value01);
                AscendC::Reg::Cast<float, bfloat16_t, castZero>(fp32Value10, value1, mask1);
                AscendC::Reg::Cast<float, bfloat16_t, castOne>(fp32Value11, value1, mask1);
                AscendC::Reg::Interleave(fp32Value10, fp32Value11, fp32Value10, fp32Value11);
                AscendC::Reg::Cast<DataTypeOut, float, castFp8>(fp8Value00, fp32Value00, mask2);
                AscendC::Reg::Cast<DataTypeOut, float, castFp8>(fp8Value01, fp32Value01, mask2);
                AscendC::Reg::Cast<DataTypeOut, float, castFp8>(fp8Value10, fp32Value10, mask3);
                AscendC::Reg::Cast<DataTypeOut, float, castFp8>(fp8Value11, fp32Value11, mask3);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp8Value00),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask2);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp8Value01),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask2);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp8Value10),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask3);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp8Value11),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask3);
            }
        }
    }

    __aicore__ inline void QuantizeFp4(__ubuf__ bfloat16_t* src, __ubuf__ uint16_t* reciprocalScale,
                                       __ubuf__ int8_t* output, uint32_t totalCount, uint16_t loopCount)
    {
        uint32_t remainingData = totalCount;
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg mask0;
            AscendC::Reg::MaskReg mask1;
            AscendC::Reg::MaskReg maskEven;
            AscendC::Reg::MaskReg maskOdd;
            AscendC::Reg::RegTensor<uint16_t> scale;
            AscendC::Reg::RegTensor<bfloat16_t> value0;
            AscendC::Reg::RegTensor<bfloat16_t> value1;
            AscendC::Reg::RegTensor<DataTypeOut> fp4Value0;
            AscendC::Reg::RegTensor<DataTypeOut> fp4Value1;
            static constexpr AscendC::Reg::CastTrait castFp4 = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_RINT};

            for (uint16_t i = 0; i < loopCount; ++i) {
                mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
                AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                    value0, value1, src, vlForHalfNumber_ * GELU_MX_INTERLEAVED_REG_FACTOR);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_E2B_B16>(scale, reciprocalScale,
                                                                             elementAfterReduce_);
                AscendC::Reg::Mul(value0, value0, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale),
                                  maskEven);
                AscendC::Reg::Mul(value1, value1, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale),
                                  maskOdd);
                AscendC::Reg::Interleave(value0, value1, value0, value1);
                AscendC::Reg::Cast<DataTypeOut, bfloat16_t, castFp4>(fp4Value0, value0, mask0);
                AscendC::Reg::Cast<DataTypeOut, bfloat16_t, castFp4>(fp4Value1, value1, mask1);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp4Value0),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask0);
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    output, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp4Value1),
                    GELU_MX_OUTPUT_ELEMENTS_PER_BLOCK, mask1);
            }
        }
    }

    __aicore__ inline void RunActivation(uint16_t mSize)
    {
        const uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(singleN_));
        const uint32_t activationUbOffset = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(activationResult_.GetPhyAddr()) - asc_get_phy_buf_addr(0));
        auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize), static_cast<int64_t>(singleN_),
                                            static_cast<int64_t>(nAligned));
        auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layout);
        auto dstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(activationUbOffset), layout);
        Gelu<bfloat16_t, DataTypeIn> gelu;
        gelu.GeluTanh(srcTensor, dstTensor, mSize, static_cast<uint16_t>(singleN_));
    }

    __aicore__ inline void RunDynamicMxQuant(uint16_t mSize, uint32_t slot)
    {
        __ubuf__ int8_t* quantOutput = reinterpret_cast<__ubuf__ int8_t*>(quantOutput_[slot].GetPhyAddr());
        __ubuf__ uint16_t* quantScale = reinterpret_cast<__ubuf__ uint16_t*>(quantScaleOutput_.GetPhyAddr());
        __ubuf__ bfloat16_t* activation = reinterpret_cast<__ubuf__ bfloat16_t*>(activationResult_.GetPhyAddr());

        const uint32_t alignedN = Gemm::CeilAlign(static_cast<uint32_t>(singleN_),
                                                  static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
        const uint32_t totalData = mSize * alignedN;
        const uint32_t totalScale = totalData / AscendC::ONE_BLK_SIZE;
        const uint16_t dataLoopCount = Gemm::CeilDiv(totalData, vlForHalfNumber_ * GELU_MX_INTERLEAVED_REG_FACTOR);
        __ubuf__ uint16_t* maxExp = reinterpret_cast<__ubuf__ uint16_t*>(maxExp_.GetPhyAddr());
        __ubuf__ uint16_t* reciprocalScale = reinterpret_cast<__ubuf__ uint16_t*>(halfScale_.GetPhyAddr());
        if (params_->scaleAlg == GELU_MX_SCALE_ALG_OCP) {
            const uint16_t scaleLoopCount = Gemm::CeilDiv(totalScale, vlForHalfNumber_);
            ComputeMaxExpOcp(activation, maxExp, totalData, dataLoopCount);
            ComputeScaleOcp(maxExp, quantScale, reciprocalScale, totalScale, scaleLoopCount);
        } else if (params_->scaleAlg == GELU_MX_SCALE_ALG_DYNAMIC_DTYPE_RANGE &&
                   (params_->dstTypeMax == GELU_MX_DEFAULT_DST_TYPE_MAX ||
                    params_->dstTypeMax == GELU_MX_FP4_E2M1_DST_TYPE_MAX ||
                    params_->dstTypeMax == GELU_MX_FP4_E2M1_SPECIAL_DST_TYPE_MAX)) {
            const uint16_t scaleLoopCount = Gemm::CeilDiv(totalScale, vlForHalfNumber_);
            ComputeMaxExpCublas(activation, maxExp, totalData, dataLoopCount);
            ComputeScaleDynamicDtypeRange(maxExp, quantScale, reciprocalScale, totalScale, scaleLoopCount);
        } else {
            const uint16_t scaleLoopCount = Gemm::CeilDiv(totalScale, vlForHalfNumber_ / GELU_MX_HALF_REG_FACTOR);
            ComputeMaxExpCublas(activation, maxExp, totalData, dataLoopCount);
            ComputeScaleCublas(maxExp, quantScale, reciprocalScale, totalScale, scaleLoopCount);
        }
        if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
            QuantizeFp8(activation, reciprocalScale, quantOutput, totalData, dataLoopCount);
        } else {
            QuantizeFp4(activation, reciprocalScale, quantOutput, totalData, dataLoopCount);
        }
    }

    __aicore__ inline void TransScaleLayout(uint16_t mSize, uint32_t slot)
    {
        __ubuf__ int8_t* source = reinterpret_cast<__ubuf__ int8_t*>(quantScaleOutput_.GetPhyAddr());
        __ubuf__ int8_t* destination = reinterpret_cast<__ubuf__ int8_t*>(quantScaleBlockOutput_[slot].GetPhyAddr());
        AscendC::Duplicate<int8_t>(quantScaleBlockOutput_[slot], 0, mSize * AscendC::ONE_BLK_SIZE);
        // The source contains ceil(N / 32) valid scales, while yScale reserves ceil(N / 64) * 2 slots.
        const uint32_t validScaleBlockN = Gemm::CeilDiv(static_cast<uint64_t>(singleN_),
                                                        static_cast<uint64_t>(GELU_MX_BLOCK_SIZE));
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                uint32_t elementCount = validScaleBlockN;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int8_t>(elementCount);
                AscendC::Reg::RegTensor<int8_t> value;
                AscendC::Reg::UnalignReg unalign;
                __ubuf__ int8_t* rowSource = source + row * validScaleBlockN;
                AscendC::Reg::DataCopyUnAlignPre(unalign, rowSource);
                AscendC::Reg::DataCopyUnAlign(value, unalign, rowSource);
                __ubuf__ int8_t* rowDestination = destination + row * AscendC::ONE_BLK_SIZE;
                AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(rowDestination, value, mask);
            }
        }
    }

private:
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;

    AscendC::LocalTensor<DataTypeIn> l0cOutputUb_{AscendC::TPosition::VECIN, 0, GELU_MX_MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_[EPILOGUE_UB_DB_COUNT];
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_[EPILOGUE_UB_DB_COUNT];
    AscendC::LocalTensor<bfloat16_t> activationResult_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    const Params* params_{nullptr};
    bool isValid_{true};
    int64_t n_{0};
    int64_t scaleN_{0};
    uint32_t subBlockIdx_{0};
    uint32_t singleM_{0};
    uint32_t singleN_{0};
    uint32_t scaleBlockN_{0};
    uint32_t vlForHalfNumber_{0};
    uint16_t elementAfterReduce_{0};
    uint16_t fpEmax_{0};
    float invDstTypeMax_{GELU_MX_SCALAR_ONE / GELU_MX_FP4_E2M1_DST_TYPE_MAX};
    uint16_t addValueBits_{GELU_MX_BF16_ADD_VALUE_MAN1};
    uint32_t pingPongId_{0};
    uint32_t bufferCount_{1};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
