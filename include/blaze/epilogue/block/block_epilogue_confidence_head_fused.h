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
 * \file block_epilogue_confidence_head_fused.h
 * \brief UB buffers, transfers and synchronization for the ConfidenceHeadFused epilogue.
 */

#pragma once

#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Epilogue::Block {

template <class InType_, class OutType_>
class BlockEpilogueConfidenceHeadFused {
public:
    using InType = InType_;
    using OutType = OutType_;
    using L0CDataType = InType;
    using OutputType = OutType;
    static_assert(AscendC::Std::is_same_v<InType, float> && AscendC::Std::is_same_v<OutType, float>,
                  "ConfidenceHeadFused epilogue requires FP32 input and output.");
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outputGmAddr{nullptr};
        GM_ADDR temperatureGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        float lowLimit{0.0F};
        float highLimit{0.0F};
        uint32_t biasN{0};
        bool hasBias{false};
        bool hasLowLimit{false};
        bool hasHighLimit{false};
        ProblemShape problemShape{};
        uint32_t baseN{128};
    };

    __aicore__ inline BlockEpilogueConfidenceHeadFused() = default;

    __aicore__ inline ~BlockEpilogueConfidenceHeadFused()
    {
        if ASCEND_IS_AIV {
            if (initialized_) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(PING_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(PONG_EVENT);
                if (hasZeroReductionBias_) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(PING_EVENT);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(PONG_EVENT);
                }
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        baseN_ = params.baseN;
        // Fixpipe pads M=1 to two physical rows in UB.
        const uint64_t baseInSize = FIXPIPE_ROWS * baseN_ * sizeof(InType);
        const uint64_t baseOutSize = baseN_ * sizeof(OutType);
        ubInputPing_ = 0;
        ubInputPong_ = baseInSize;
        ubOutputPing_ = ubInputPong_ + baseInSize;
        ubOutputPong_ = ubOutputPing_ + baseOutSize;
        outputGmAddr_ = params.outputGmAddr;
        n_ = asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape);
        biasGmAddr_ = params.biasGmAddr;
        biasN_ = params.biasN;
        hasZeroReductionBias_ = params.hasBias && biasGmAddr_ != nullptr &&
                                asc::te::get<Gemm::CONFIDENCE_HEAD_K_EMBED>(params.problemShape) == 0 &&
                                asc::te::get<Gemm::CONFIDENCE_HEAD_K_MARKOV>(params.problemShape) == 0;
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(PING_EVENT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(PONG_EVENT);
            if (hasZeroReductionBias_) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(PING_EVENT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(PONG_EVENT);
            }
            initialized_ = true;
            InitTileParams(params);
            AscendC::ICachePreLoad(NUM_TWO);
        }
    }

    __aicore__ inline uint64_t GetUbAddr() const { return GetUbAddr(loopCnt_); }

    __aicore__ inline uint64_t GetUbAddr(uint64_t dbIdx) const
    {
        return (dbIdx & 1) == 0 ? ubInputPing_ : ubInputPong_;
    }

    template <bool ZeroInput = false, typename TensorC>
    __aicore__ inline void operator()(TensorC& ubTensor, const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        const uint16_t slot = static_cast<uint16_t>(loopCnt_ & 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        const int64_t curN = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        // The scheduler flattens [batch, N] into this N coordinate and keeps bias tiles within one batch.
        const int64_t dstOffset = asc::te::get<Blaze::Gemm::MNK_N>(blockCoord);
        const auto origin = asc::te::make_coord(0L, 0L);
        const auto validShape = asc::te::make_shape(1L, curN);
        const uint64_t outUbOffset = slot == 0 ? ubOutputPing_ : ubOutputPong_;
        auto tensorOut = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(outUbOffset),
                                              Gemm::MakeNDExtLayout<OutType>(1L, curN, baseN_));
        const Tile::ConfidenceHeadFused::Params tileParams{negInvTemperature_, lowLimit_, highLimit_, hasLowLimit_,
                                                           hasHighLimit_};
        if constexpr (ZeroInput) {
            if (hasZeroReductionBias_) {
                // K=0 bypasses Cube. Bias itself is the logit; reuse the input buffer without a vector add.
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
                auto biasGm = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ InType*>(biasGmAddr_) +
                                                                 dstOffset % biasN_),
                    asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, curN));
                asc::te::copy(ubTensor, biasGm);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
                Tile::ConfidenceHeadFused::Run(ubTensor, tensorOut, tileParams);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
            } else {
                Tile::ConfidenceHeadFused::Run<true>(ubTensor, tensorOut, tileParams);
            }
        } else {
            // The Cube accumulator already contains bias on every nonzero-K path.
            Tile::ConfidenceHeadFused::Run(ubTensor, tensorOut, tileParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
        auto outputGmStorage = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(outputGmAddr_) + dstOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n_));
        auto outputGm = outputGmStorage.slice(origin, validShape);
        asc::te::copy(outputGm, tensorOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        ++loopCnt_;
    }

private:
    __aicore__ inline void InitTileParams(const Params& params)
    {
        negInvTemperature_ = -1.0F;
        if (params.temperatureGmAddr != nullptr) {
            AscendC::GlobalTensor<float> temperatureGlobal;
            temperatureGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.temperatureGmAddr));
            const float temperature = temperatureGlobal.GetValue(0);
            negInvTemperature_ = temperature == 0.0F ? -1.0F : (-1.0F / temperature);
        }
        lowLimit_ = params.lowLimit;
        highLimit_ = params.highLimit;
        hasLowLimit_ = params.hasLowLimit;
        hasHighLimit_ = params.hasHighLimit;
    }

private:
    static constexpr uint16_t NUM_TWO = 2;
    static constexpr uint16_t PING_EVENT = 0;
    static constexpr uint16_t PONG_EVENT = 1;
    static constexpr uint16_t OUTPUT_EVENT = 0;
    static constexpr uint32_t DEFAULT_BASE_N = 128;
    static constexpr uint32_t FIXPIPE_ROWS = 2;

    uint64_t ubInputPing_{0};
    uint64_t ubInputPong_{0};
    uint64_t ubOutputPing_{0};
    uint64_t ubOutputPong_{0};
    uint64_t baseN_{DEFAULT_BASE_N};
    uint64_t loopCnt_{0};
    GM_ADDR outputGmAddr_{nullptr};
    GM_ADDR biasGmAddr_{nullptr};
    int64_t n_{0};
    uint32_t biasN_{0};
    float negInvTemperature_{-1.0F};
    float lowLimit_{0.0F};
    float highLimit_{0.0F};
    bool hasLowLimit_{false};
    bool hasHighLimit_{false};
    bool hasZeroReductionBias_{false};
    bool initialized_{false};
};

} // namespace Blaze::Epilogue::Block
