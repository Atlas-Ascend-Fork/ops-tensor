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
 * \brief MIX epilogue: float L0C in UB -> GELU (tanh) -> dynamic MX y/yScale.
 */

#pragma once

#include "kernel_basic_intf.h"

#include "interface/reg_compute/kernel_reg_compute_utils.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_ = float>
class BlockEpilogueGeluTanhMxQuant {
private:
    static constexpr uint32_t GELU_MX_BLOCK_SIZE = 32;
    static constexpr uint32_t GELU_MX_MAX_SINGLE_MN = 128 * 256;
    static constexpr uint16_t GELU_MX_FP8_E4M3_MAX_EXP = 0x0400;
    static constexpr uint16_t GELU_MX_FP8_E5M2_MAX_EXP = 0x0780;
    static constexpr uint16_t GELU_MX_FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t GELU_MX_FP4_E1M2_MAX_EXP = 0x0000;
    static constexpr uint32_t GELU_MX_SCALE_ALG_OCP = 0;
    static constexpr uint32_t GELU_MX_SCALE_ALG_DYNAMIC_DTYPE_RANGE = 2;
    static constexpr uint16_t GELU_MX_BF16_ADD_VALUE_MAN1 = 0x003f;
    static constexpr uint16_t GELU_MX_BF16_ADD_VALUE_MAN2 = 0x001f;
    static constexpr float GELU_MX_DEFAULT_DST_TYPE_MAX = 0.0f;
    static constexpr float GELU_MX_FP4_E2M1_DST_TYPE_MAX = 6.0f;
    static constexpr float GELU_MX_FP4_E2M1_SPECIAL_DST_TYPE_MAX = 7.0f;
    static constexpr float GELU_MX_SCALAR_ONE = 1.0f;

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

    template <typename T>
    __aicore__ inline static uint32_t GetUbByteOffset(const AscendC::LocalTensor<T>& tensor)
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.GetPhyAddr()) - asc_get_phy_buf_addr(0));
    }

    __aicore__ inline void RunActivation(uint16_t mSize)
    {
        const uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(singleN_));
        const uint32_t activationUbOffset = GetUbByteOffset(activationResult_);
        auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize), static_cast<int64_t>(singleN_),
                                            static_cast<int64_t>(nAligned));
        auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layout);
        auto dstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(activationUbOffset), layout);
        // kPreciseDiv=true 误差补偿除法
        Tile::Gelu<Tile::GeluTanhTag>::Run</*kPreciseDiv=*/true>(srcTensor, dstTensor);
    }

    __aicore__ inline void RunDynamicMxQuant(uint16_t mSize, uint32_t slot)
    {
        const uint32_t alignedN = Gemm::CeilAlign(static_cast<uint32_t>(singleN_),
                                                  static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
        const uint32_t totalData = mSize * alignedN;
        const uint32_t totalScale = totalData / AscendC::ONE_BLK_SIZE;

        auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalData),
                                                        static_cast<int64_t>(totalData));
        auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalScale),
                                                         static_cast<int64_t>(totalScale));
        auto actTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(GetUbByteOffset(activationResult_)), dataLayout);
        auto maxExpTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GetUbByteOffset(maxExp_)), scaleLayout);
        auto yScaleTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(quantScaleOutput_)), scaleLayout);
        auto reciprocalTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GetUbByteOffset(halfScale_)), scaleLayout);
        auto yTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(quantOutput_[slot])), dataLayout);

        // MxQuantConfig members are resolved at Init time (fpEmax_ / invDstTypeMax_ / addValueBits_);
        // the tile config is built at the call site (tile types must stay out of class scope).
        // In OCP mode, zero sharedExp clears the reciprocal.
        Tile::MxQuantConfig cfg{Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_, addValueBits_, false};

        Tile::MxQuant<DataTypeOut> mx;
        if (params_->scaleAlg == GELU_MX_SCALE_ALG_OCP) {
            cfg.alg = Tile::MxScaleAlg::OCP;
            mx.GroupMaxExp(actTensor, maxExpTensor, totalData, false);
        } else if (params_->scaleAlg == GELU_MX_SCALE_ALG_DYNAMIC_DTYPE_RANGE &&
                   (params_->dstTypeMax == GELU_MX_DEFAULT_DST_TYPE_MAX ||
                    params_->dstTypeMax == GELU_MX_FP4_E2M1_DST_TYPE_MAX ||
                    params_->dstTypeMax == GELU_MX_FP4_E2M1_SPECIAL_DST_TYPE_MAX)) {
            cfg.alg = Tile::MxScaleAlg::DYN_DTYPE_RANGE;
            mx.GroupMaxExp(actTensor, maxExpTensor, totalData, true);
        } else {
            cfg.alg = Tile::MxScaleAlg::CUBLAS;
            mx.GroupMaxExp(actTensor, maxExpTensor, totalData, true);
        }
        mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
        mx.Quantize(actTensor, reciprocalTensor, yTensor, totalData);
    }

    __aicore__ inline void TransScaleLayout(uint16_t mSize, uint32_t slot)
    {
        AscendC::Duplicate<int8_t>(quantScaleBlockOutput_[slot], 0, mSize * AscendC::ONE_BLK_SIZE);
        // The source contains ceil(N / 32) valid scales, while yScale reserves ceil(N / 64) * 2 slots.
        const uint32_t validScaleBlockN = Gemm::CeilDiv(static_cast<uint64_t>(singleN_),
                                                        static_cast<uint64_t>(GELU_MX_BLOCK_SIZE));
        auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
                                                       static_cast<int64_t>(validScaleBlockN),
                                                       static_cast<int64_t>(validScaleBlockN));
        auto srcTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(quantScaleOutput_)), srcLayout);
        auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
                                                       static_cast<int64_t>(AscendC::ONE_BLK_SIZE),
                                                       static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
        auto dstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(quantScaleBlockOutput_[slot])),
            dstLayout);
        Tile::MxQuant<DataTypeOut> mx;
        mx.TransScaleLayout(srcTensor, dstTensor, mSize, static_cast<uint16_t>(validScaleBlockN));
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
    uint16_t fpEmax_{0};
    float invDstTypeMax_{GELU_MX_SCALAR_ONE / GELU_MX_FP4_E2M1_DST_TYPE_MAX};
    uint16_t addValueBits_{GELU_MX_BF16_ADD_VALUE_MAN1};
    uint32_t pingPongId_{0};
    uint32_t bufferCount_{1};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
