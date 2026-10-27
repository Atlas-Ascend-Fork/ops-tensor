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
 * \file block_epilogue_dequant.h
 * \brief MIX template epilogue (AIV side): out = x1@x2(L0C in UB) * x2Scale * x1Scale + bias.
 *        The x1@x2 accumulator is read from UB (L0C->UB by AIC fixpipe, raw NoQuant copy);
 *        x2Scale / x1Scale / bias are read from GM. All scale multiplies + bias add run in the
 *        reusable Tile::Dequant chain (blaze/epilogue/tile/arch35/dequant.h); this block keeps
 *        GM->UB staging, runtime mode dispatch, dual AIV sub-block split, 4-way M partition
 *        and ping-pong output.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/epilogue/tile/compute.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

using AscendC::IsSameType;
using Blaze::Gemm::CeilAlign;
using Blaze::Gemm::CeilDiv;

template <class OutType_, class BiasType_, class X2ScaleType_, class X1ScaleType_ = float, class L0CType_ = int32_t>
class BlockEpilogueDequant {
public:
    using OutType = OutType_;
    using BiasType = BiasType_;
    using X2ScaleType = X2ScaleType_;
    using X1ScaleType = X1ScaleType_;
    using L0CType = L0CType_;

    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t FLOAT_ALIGN = DATA_BLOCK / sizeof(float);
    static constexpr uint32_t L0C_ALIGN = DATA_BLOCK / sizeof(L0CType);
    static constexpr uint32_t OUT_ALIGN = DATA_BLOCK / sizeof(OutType);
    static constexpr uint32_t CV_RATIO = Blaze::Gemm::MIX_CV_RATIO;
    static constexpr uint32_t FP32_OUTPUT_TIMES = IsSameType<OutType, float>::value ? 4 : 2;
    static constexpr uint16_t BIAS_EVENT_ID = 2U;

    // The host passes the ACTUAL bias tensor dtype (a ge::DataType code) in params.biasDtype.
    // For int8 inputs the compile-time BiasType is forced to int32_t (DTYPE_BIAS), which does
    // NOT match the real bias buffer (bf16/fp16/fp32); we therefore interpret the bias GM/UB
    // buffer by this runtime code (DT_FLOAT / DT_FLOAT16 / DT_BF16), not by BiasType.

    enum class QuantMode : uint32_t {
        DEFAULT = 0x0U,
        PERTENSOR_MODE = 0x1U,
        PERCHANNEL_MODE = 0x2U,
        PERTOKEN_MODE = 0x4U,
    };

    struct Params {
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR outGmAddr{nullptr};
        int64_t m{0};
        int64_t n{0};
        int64_t baseM{0};
        int64_t baseN{0};
        uint32_t x1QuantMode{0};
        uint32_t x2QuantMode{0};
        bool isBias{false};
        uint32_t biasDtype{0};
        int64_t batch{1};
    };

    __aicore__ inline BlockEpilogueDequant() {}
    __aicore__ inline ~BlockEpilogueDequant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = params.m;
        n_ = params.n;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        batch_ = params.batch;
        isBias_ = params.isBias;
        isPerChannel_ = (params.x2QuantMode == static_cast<uint32_t>(QuantMode::PERCHANNEL_MODE));
        isPerToken_ = (params.x1QuantMode == static_cast<uint32_t>(QuantMode::PERTOKEN_MODE));
        isX1PerTensor_ = (params.x1QuantMode == static_cast<uint32_t>(QuantMode::PERTENSOR_MODE));

        subBlockIdx_ = AscendC::GetSubBlockIdx();

        outGmAddr_ = params.outGmAddr;

        if (isPerChannel_) {
            x2ScaleGmAddr_ = params.x2ScaleGmAddr;
        } else if constexpr (IsSameType<X2ScaleType, float>::value) {
            auto x2ScaleTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(params.x2ScaleGmAddr)),
                Gemm::MakeNDExtLayout(1, 1, 1));
            ReadX2ScaleScalar(x2ScaleTensor);
        } else {
            auto x2ScaleTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ uint16_t*>(params.x2ScaleGmAddr)),
                Gemm::MakeNDExtLayout(1, 1, 1));
            ReadX2ScaleScalar(x2ScaleTensor);
        }
        if (isPerToken_) {
            x1ScaleGmAddr_ = params.x1ScaleGmAddr;
        } else if (isX1PerTensor_) {
            auto ptTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(params.x1ScaleGmAddr)),
                Gemm::MakeNDExtLayout(1, 1, 1));
            x1ScaleScalar_ = ptTensor[asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
        }
        if (isBias_) {
            biasDtype_ = params.biasDtype;
            biasGmAddr_ = params.biasGmAddr;
        }

        SetupUbLayout();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    __aicore__ inline void operator()(int64_t singleCoreM, int64_t singleCoreN, int64_t offsetScale,
                                      int64_t offsetPtScale, int64_t offsetBias, int64_t offsetC,
                                      int64_t l0cBaseOffset = 0)
    {
        int64_t halfSingleM = CeilDiv(singleCoreM, static_cast<int64_t>(CV_RATIO));
        int64_t singleMInVec = (subBlockIdx_ == 1) ? (singleCoreM - halfSingleM) : halfSingleM;
        if (singleMInVec <= 0) {
            return;
        }
        int64_t mOffset = subBlockIdx_ * halfSingleM;

        CopyScaleBiasToUb(singleCoreN, singleMInVec, offsetScale, offsetPtScale + mOffset, offsetBias);

        int64_t splitNumOfOut = (singleMInVec >= 4) ? 4 : singleMInVec;
        int64_t mSizeForOnce = CeilDiv(singleMInVec, splitNumOfOut);
        int64_t nAligned = CeilAlign(singleCoreN, static_cast<int64_t>(L0C_ALIGN));

        for (int64_t i = 0; i < splitNumOfOut; i++) {
            if (i * mSizeForOnce >= singleMInVec) {
                break;
            }
            int64_t mSize = ((singleMInVec - i * mSizeForOnce) >= mSizeForOnce) ? mSizeForOnce :
                                                                                  (singleMInVec - i * mSizeForOnce);
            int64_t l0cOffset = l0cBaseOffset + i * mSizeForOnce * nAligned;
            int64_t ptScaleRowOffset = i * mSizeForOnce;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongId_);

            DequantCompute(mSize, singleCoreN, nAligned, l0cOffset, ptScaleRowOffset);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongId_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongId_);

            int64_t gmOffset = offsetC + (mOffset + i * mSizeForOnce) * n_ * batch_;
            CopyResultToGm(mSize, singleCoreN, gmOffset);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongId_);
            pingPongId_ ^= 1;
        }

        UbSetFlag();
    }

private:
    template <class X2ScaleTensor>
    __aicore__ inline void ReadX2ScaleScalar(const X2ScaleTensor& x2ScaleTensor)
    {
        if constexpr (IsSameType<X2ScaleType, float>::value) {
            x2ScaleScalar_ = x2ScaleTensor[asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
        } else {
            uint16_t raw = x2ScaleTensor[asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
            uint32_t bits = static_cast<uint32_t>(raw) << 16;
            x2ScaleScalar_ = *reinterpret_cast<float*>(&bits);
        }
    }

    // UB layout (single contiguous arena, all regions 32-byte aligned):
    //   [0]                : L0C raw accumulator (L0CType), mForSingleVec * nAligned
    //   [x2ScaleUbOffset_] : per-channel x2Scale (baseN, raw X2ScaleType), only when PERCHANNEL
    //   [x1ScaleUbOffset_] : per-token x1Scale (mForSingleVec, fp32), only when PERTOKEN
    //   [biasUbOffset_]    : bias (baseN, raw BiasType), only when isBias
    //   [dequantPingOffset_/dequantPongOffset_] : dequant output (OutType) ping-pong
    // bf16/fp16 x2Scale and bias are widened to fp32 inline in the VF loop, so no separate
    // float staging buffers are needed.
    __aicore__ inline void SetupUbLayout()
    {
        uint64_t mForSingleVec = CeilDiv(static_cast<uint64_t>(baseM_), static_cast<uint64_t>(CV_RATIO));
        uint64_t nAligned = CeilAlign(static_cast<uint64_t>(baseN_), static_cast<uint64_t>(L0C_ALIGN));

        uint64_t l0cUbSize = mForSingleVec * nAligned * sizeof(L0CType);
        uint64_t offset = l0cUbSize;

        if (isPerChannel_) {
            x2ScaleUbOffset_ = offset;
            offset += CeilAlign(static_cast<uint64_t>(baseN_) * sizeof(X2ScaleType), static_cast<uint64_t>(DATA_BLOCK));
        }
        if (isPerToken_) {
            x1ScaleUbOffset_ = offset;
            offset += CeilAlign(mForSingleVec * sizeof(X1ScaleType), static_cast<uint64_t>(DATA_BLOCK));
        }
        if (isBias_) {
            biasUbOffset_ = offset;
            offset += CeilAlign(static_cast<uint64_t>(baseN_) * sizeof(BiasType), static_cast<uint64_t>(DATA_BLOCK));
        }
        uint64_t outOnceSize = CeilDiv(mForSingleVec, static_cast<uint64_t>(FP32_OUTPUT_TIMES)) *
                               CeilAlign(static_cast<uint64_t>(baseN_), static_cast<uint64_t>(OUT_ALIGN)) *
                               sizeof(OutType);
        dequantPingOffset_ = offset;
        offset += CeilAlign(outOnceSize, static_cast<uint64_t>(DATA_BLOCK));
        dequantPongOffset_ = offset;
    }

    // 32B-aligned UB row pitch (in elements of T) for a row of `cols` elements. CopyGmToUbufAlignV2
    // (see arch/vector/gm_to_ub/.../data_copy.h) requires the UB destination region AND its row
    // stride to be 32-byte aligned, otherwise the ND/DN load produces wrong data. sizeof(T) divides
    // DATA_BLOCK for every scale/bias dtype (fp32/int32=4, fp16/bf16=2), so the result is exact.
    template <class T>
    __aicore__ inline static int64_t AlignedUbPitch(int64_t cols)
    {
        return static_cast<int64_t>(
            CeilAlign(static_cast<uint64_t>(cols) * sizeof(T), static_cast<uint64_t>(DATA_BLOCK)) / sizeof(T));
    }

    // Copy x2Scale / x1Scale / bias from GM to UB (raw, no pre-cast). bf16/fp16 widening is
    // done inline in the VF dequant loop. The GM->UB move uses the Tensor API copy_gm_to_ub
    // operation (a single contiguous 1-row NDExt block) instead of DataCopyPad.
    __aicore__ inline void CopyScaleBiasToUb(int64_t singleCoreN, int64_t singleMInVec, int64_t offsetScale,
                                             int64_t offsetPtScale, int64_t offsetBias)
    {
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});

        if (isPerChannel_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            // UB dest pitch is 32B-aligned (matches the CeilAlign region reserved in SetupUbLayout);
            // GM src pitch stays the contiguous singleCoreN. CopyGmToUbufAlignV2 requires the UB
            // stride to be 32B-aligned or the load corrupts data.
            auto ubLayout = Gemm::MakeNDExtLayout(1, singleCoreN, AlignedUbPitch<X2ScaleType>(singleCoreN));
            auto gmLayout = Gemm::MakeNDExtLayout(1, singleCoreN, singleCoreN);
            auto x2Ub = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, X2ScaleType>(x2ScaleUbOffset_), ubLayout);
            auto x2Gm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ X2ScaleType*>(x2ScaleGmAddr_) + offsetScale),
                                             gmLayout);
            asc::te::copy(copyGM2UB, x2Ub, x2Gm);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        }

        if (isPerToken_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            auto ubLayout = Gemm::MakeNDExtLayout(1, singleMInVec, AlignedUbPitch<X1ScaleType>(singleMInVec));
            auto gmLayout = Gemm::MakeNDExtLayout(1, singleMInVec, singleMInVec);
            auto x1Ub = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, X1ScaleType>(x1ScaleUbOffset_), ubLayout);
            auto x1Gm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ X1ScaleType*>(x1ScaleGmAddr_) + offsetPtScale),
                                             gmLayout);
            asc::te::copy(copyGM2UB, x1Ub, x1Gm);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        }

        if (isBias_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
            // Interpret the bias buffer by its RUNTIME dtype, not the compile-time BiasType.
            if (biasDtype_ == DT_FLOAT) {
                CopyBiasToUbTyped<float>(singleCoreN, offsetBias);
            } else if (biasDtype_ == DT_FLOAT16) {
                CopyBiasToUbTyped<half>(singleCoreN, offsetBias);
            } else {
                CopyBiasToUbTyped<bfloat16_t>(singleCoreN, offsetBias);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BIAS_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BIAS_EVENT_ID);
        }
    }

    // GM->UB copy of the bias row, reinterpreting the raw bias GM pointer to its ACTUAL runtime
    // dtype (float/half/bfloat16_t). Typed pointer arithmetic gives the correct byte stride for
    // offsetBias, unlike a BiasType(int32)-typed access which would read wrong bytes for bf16/fp16.
    template <class ActualBiasType>
    __aicore__ inline void CopyBiasToUbTyped(int64_t singleCoreN, int64_t offsetBias)
    {
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        // UB dest pitch 32B-aligned (per-ActualBiasType), GM src pitch contiguous singleCoreN.
        auto ubLayout = Gemm::MakeNDExtLayout(1, singleCoreN, AlignedUbPitch<ActualBiasType>(singleCoreN));
        auto gmLayout = Gemm::MakeNDExtLayout(1, singleCoreN, singleCoreN);
        auto biasUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ActualBiasType>(biasUbOffset_),
                                           ubLayout);
        __gm__ ActualBiasType* biasGmPtr = reinterpret_cast<__gm__ ActualBiasType*>(biasGmAddr_) + offsetBias;
        auto biasGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmPtr), gmLayout);
        asc::te::copy(copyGM2UB, biasUb, biasGm);
    }

    // Dispatch entry for one M-chunk: compute UB tensors for the current ping-pong buffer,
    // then call the Tile::Dequant chain with the runtime x1/x2/bias flags translated into
    // compile-time template arguments.
    __aicore__ inline void DequantCompute(int64_t mSize, int64_t singleCoreN, int64_t nAligned, int64_t l0cOffset,
                                          int64_t ptScaleRowOffset)
    {
        // Choose the actual bias UB type from the RUNTIME dtype, then run the typed dequant.
        if (!isBias_ || biasDtype_ == DT_FLOAT) {
            DequantComputeTyped<float>(mSize, singleCoreN, nAligned, l0cOffset, ptScaleRowOffset);
        } else if (biasDtype_ == DT_FLOAT16) {
            DequantComputeTyped<half>(mSize, singleCoreN, nAligned, l0cOffset, ptScaleRowOffset);
        } else {
            DequantComputeTyped<bfloat16_t>(mSize, singleCoreN, nAligned, l0cOffset, ptScaleRowOffset);
        }
    }

    template <class BiasDtype>
    __aicore__ inline void DequantComputeTyped(int64_t mSize, int64_t singleCoreN, int64_t nAligned, int64_t l0cOffset,
                                               int64_t ptScaleRowOffset)
    {
        uint32_t nSrcAligned = static_cast<uint32_t>(nAligned);
        uint32_t nDstAligned = static_cast<uint32_t>(
            CeilAlign(static_cast<uint64_t>(singleCoreN), static_cast<uint64_t>(OUT_ALIGN)));
        uint64_t dequantOffset = (pingPongId_ == 0) ? dequantPingOffset_ : dequantPongOffset_;
        uint16_t mSize16 = static_cast<uint16_t>(mSize);
        uint16_t nSize16 = static_cast<uint16_t>(singleCoreN);

        // Tensors for the Dequant tile: disabled mode slots receive same-element-type
        // placeholder tensors (never dereferenced by the tile).
        auto accTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(static_cast<uint64_t>(l0cOffset) * sizeof(L0CType)),
            Gemm::MakeNDExtLayout(mSize, singleCoreN, nAligned));
        auto dstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, OutType>(dequantOffset),
            Gemm::MakeNDExtLayout(mSize, singleCoreN, static_cast<int64_t>(nDstAligned)));
        auto x2ScaleTensor = MakeX2ScaleTensor(singleCoreN);
        auto x1ScaleTensor = MakeX1ScaleTensor(ptScaleRowOffset, mSize);
        auto biasTensor = MakeBiasTensor<BiasDtype>(singleCoreN);

        using B32ToFpDequant = Tile::Dequant<Tile::B32ToFpTag>;
        using DequantMode = Tile::DequantMode;
        if (isPerToken_) {
            if (isBias_) {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::PER_TOKEN, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::PER_TOKEN, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                }
            } else {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::PER_TOKEN, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::PER_TOKEN, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                }
            }
        } else if (isX1PerTensor_) {
            if (isBias_) {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::PER_TENSOR, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, x1ScaleScalar_,
                        mSize16, nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::PER_TENSOR, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, x1ScaleScalar_,
                        mSize16, nSize16, nSrcAligned, nDstAligned);
                }
            } else {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::PER_TENSOR, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, x1ScaleScalar_,
                        mSize16, nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::PER_TENSOR, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, x1ScaleScalar_,
                        mSize16, nSize16, nSrcAligned, nDstAligned);
                }
            }
        } else {
            if (isBias_) {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::NONE, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::NONE, true>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                }
            } else {
                if (isPerChannel_) {
                    B32ToFpDequant::template Run<DequantMode::PER_CHANNEL, DequantMode::NONE, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                } else {
                    B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::NONE, false>(
                        accTensor, dstTensor, x2ScaleTensor, x1ScaleTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16,
                        nSize16, nSrcAligned, nDstAligned);
                }
            }
        }
    }

    // PER_CHANNEL: x2Scale single-row vector at x2ScaleUbOffset_; otherwise a placeholder
    // tensor (PER_TENSOR mode never dereferences the slot).
    __aicore__ inline auto MakeX2ScaleTensor(int64_t singleCoreN)
    {
        if (isPerChannel_) {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, X2ScaleType>(x2ScaleUbOffset_),
                Gemm::MakeNDExtLayout(1, singleCoreN, AlignedUbPitch<X2ScaleType>(singleCoreN)));
        }
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, X2ScaleType>(0),
                                    Gemm::MakeNDExtLayout(1, 1, 1));
    }

    // PER_TOKEN: fp32 linear mSize entries with the row offset already applied; otherwise a
    // placeholder tensor (PER_TENSOR / NONE modes never dereference the slot).
    __aicore__ inline auto MakeX1ScaleTensor(int64_t ptScaleRowOffset, int64_t mSize)
    {
        if (isPerToken_) {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, X1ScaleType>(
                    x1ScaleUbOffset_ + static_cast<uint64_t>(ptScaleRowOffset) * sizeof(X1ScaleType)),
                Gemm::MakeNDExtLayout(mSize, 1, 1));
        }
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, X1ScaleType>(0),
                                    Gemm::MakeNDExtLayout(1, 1, 1));
    }

    // Bias single-row vector (runtime dtype) at biasUbOffset_; placeholder when no bias.
    template <class BiasDtype>
    __aicore__ inline auto MakeBiasTensor(int64_t singleCoreN)
    {
        if (isBias_) {
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasDtype>(biasUbOffset_),
                                        Gemm::MakeNDExtLayout(1, singleCoreN, AlignedUbPitch<BiasDtype>(singleCoreN)));
        }
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasDtype>(0),
                                    Gemm::MakeNDExtLayout(1, 1, 1));
    }

    __aicore__ inline void CopyResultToGm(int64_t mSize, int64_t singleCoreN, int64_t gmOffset)
    {
        uint64_t dequantOffset = (pingPongId_ == 0) ? dequantPingOffset_ : dequantPongOffset_;
        uint64_t nDstAligned = CeilAlign(static_cast<uint64_t>(singleCoreN), static_cast<uint64_t>(OUT_ALIGN));

        // UB->GM via Tensor API copy_ub_to_gm. Both ends are nd_ext_layout_ptn with mSize rows and
        // singleCoreN valid columns; the row pitch differs per buffer (UB padded to nDstAligned,
        // GM strided by the full output row pitch n_ * batch_ for the (M, B, N) layout), which
        // reproduces the original DataCopyPad gap-based strided copy out.
        auto ubLayout = Gemm::MakeNDExtLayout(mSize, singleCoreN, static_cast<int64_t>(nDstAligned));
        auto gmLayout = Gemm::MakeNDExtLayout(mSize, singleCoreN, n_ * batch_);
        auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(dequantOffset),
                                          ubLayout);
        auto outGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(outGmAddr_) + gmOffset),
            gmLayout);

        auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        asc::te::copy(copyUB2GM, outGm, outUb);
    }

    __aicore__ inline void UbSetFlag()
    {
        if (isPerChannel_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        if (isPerToken_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        }
        if (isBias_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        }
    }

    int64_t m_{0};
    int64_t n_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t batch_{1};
    bool isBias_{false};
    uint32_t biasDtype_{0};
    GM_ADDR biasGmAddr_{nullptr};
    bool isPerChannel_{false};
    bool isPerToken_{false};
    bool isX1PerTensor_{false};
    uint32_t subBlockIdx_{0};
    float x2ScaleScalar_{1.0f};
    float x1ScaleScalar_{1.0f};
    uint32_t pingPongId_{0};
    uint64_t x2ScaleUbOffset_{0};
    uint64_t x1ScaleUbOffset_{0};
    uint64_t biasUbOffset_{0};
    uint64_t dequantPingOffset_{0};
    uint64_t dequantPongOffset_{0};
    GM_ADDR outGmAddr_{nullptr};
    GM_ADDR x2ScaleGmAddr_{nullptr};
    GM_ADDR x1ScaleGmAddr_{nullptr};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
