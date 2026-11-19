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
 * \file block_mmad_weight_prologue_mx.h
 * \brief AIC-side block MMAD for weight-only MX matmul.
 */
#pragma once

#include "kernel_basic_intf.h"

#include "blaze/gemm/block/block_mmad_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/tile/arch35/shift_w4_to_w8.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

namespace Block {

#if defined(__NPU_ARCH__) || defined(ASCENDC_CPU_DEBUG)
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

static constexpr uint64_t WQGMM_MX_GROUP_SIZE = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
static constexpr uint64_t WQGMM_MX_SYNC_MODE = Sync::SYNC_MODE_INTRA;
static constexpr uint64_t WQGMM_MX_FLAG_ID_MAX = Sync::MIX_FLAG_ID_MAX;

template <class ScheduleType_, WeightPrologueMxKSchedule KSchedule_, WeightPrologueMxScaleBSource ScaleBSource_,
          WeightPrologueMxOutput Output_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_,
          class LayoutBTuple_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<WeightQuantMxPolicy<ScheduleType_, KSchedule_, ScaleBSource_, Output_>, ATypeTuple_, LayoutATuple_,
                BTypeTuple_, LayoutBTuple_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    static_assert(AscendC::Std::tuple_size_v<ATypeTuple_> == 2, "A type tuple must contain A and ScaleA");
    static_assert(AscendC::Std::tuple_size_v<LayoutATuple_> == 2, "A layout tuple must contain A and ScaleA");
    static_assert(AscendC::Std::tuple_size_v<BTypeTuple_> == 2, "B type tuple must contain B and ScaleB");
    static_assert(AscendC::Std::tuple_size_v<LayoutBTuple_> == 2, "B layout tuple must contain B and ScaleB");
    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using CType = CType_;
    using ConvertedBType = AType;
    using BiasType = BiasType_;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = WeightQuantMxPolicy<ScheduleType_, KSchedule_, ScaleBSource_, Output_>;
    static constexpr bool DEFERRED_UB = Output_ == WeightPrologueMxOutput::DEFERRED_UB;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static_assert(IsFp4<BType>(), "Weight Quant MX expects packed FP4 B");
    static_assert(sizeof(AType) == 1, "Weight Quant MX expects 8-bit A");
    static_assert(IsTrans<LayoutB>::value, "Weight Quant MX expects a transposed B layout");
    static_assert(AscendC::Std::is_same_v<ScaleAType, AscendC::fp8_e8m0_t> &&
                      AscendC::Std::is_same_v<ScaleBType, AscendC::fp8_e8m0_t>,
                  "Weight Quant MX expects E8M0 scales");

    // Accept only the paired kernel/prologue scheduling and output policies.
    static_assert(AscendC::Std::is_same_v<DispatchPolicy, MatmulWithWeightQuantMx> ||
                      AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulWithWeightQuantMx> ||
                      AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulSwigluWithWeightQuantMx> ||
                      AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulFinalizeRoutingWithWeightQuantMx>,
                  "MX weight prologue requires a paired QBMM, GMM or deferred-UB scheduling policy");

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        L1TileShape l1TileShape{};
        L0TileShape l0TileShape{};
        uint64_t l1BufferNum{0};
        bool hasBias{false};
        uint64_t aBufferBytes{0};
        uint64_t bBufferBytes{0};
        GM_ADDR biasGmAddr{nullptr};
        bool explicitL1Offsets{false};
        uint64_t weightOffsets[QUADRUPLE_BUFFER_COUNT]{};
        uint64_t biasOffsets[QUADRUPLE_BUFFER_COUNT]{};
        uint64_t aOffsets[QUADRUPLE_BUFFER_COUNT]{};
        uint64_t scaleAOffsets[DOUBLE_BUFFER_COUNT]{};
        uint64_t scaleBOffsets[DOUBLE_BUFFER_COUNT]{};
        uint16_t readyFlag{6};
        uint16_t freeFlag{8};
        bool seedWeightCredits{true};
        bool disableWeightL2{false};
        uint64_t problemM{0};
    };

    __aicore__ inline explicit BlockMmad(const Params& params) { Init(params); }

    __aicore__ inline ~BlockMmad()
    {
        if constexpr (DEFERRED_UB) {
            for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
            }

        } else {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& tensorA, const TensorScaleA_& tensorScaleA,
                                      const TensorScaleB_& tensorScaleB, const TensorC_& tensorC)
    {
        static_assert(DispatchPolicy::K_SCHEDULE == WeightPrologueMxKSchedule::FIXED,
                      "Dynamic K blocking must be supplied by the GMM kernel");
        InitBlockShape(tensorA, tensorC);
        ProcessBlock(tensorA, tensorScaleA, tensorScaleB, tensorC);
    }

    // Internal GMM interface: K blocking is supplied by the paired kernel's CalcDynamicKBlock.
    // Requires nonzero ka/kb/k0, ka divisible by kb, kb divisible by k0, and k0 aligned to 64.
    // kb must divide the Scale K window; ka need not divide it. Aligned operand sizes must fit
    // the configured A/B L1 slots and each L0 ping/pong slot. See the API for capacity formulas.
    template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& tensorA, const TensorScaleA_& tensorScaleA,
                                      const TensorScaleB_& tensorScaleB, const TensorC_& tensorC, uint64_t kaL1Size,
                                      uint64_t kbL1Size, uint64_t l0KSize)
    {
        static_assert(DispatchPolicy::K_SCHEDULE == WeightPrologueMxKSchedule::DYNAMIC,
                      "Fixed K blocking comes from the QBMM tiling parameters");
        kAL1Size_ = kaL1Size;
        kL1Size_ = kbL1Size;
        kL0Size_ = l0KSize;
        l1Storage_.SetKSize(kL1Size_);
        InitBlockShape(tensorA, tensorC);
        ProcessBlock(tensorA, tensorScaleA, tensorScaleB, tensorC);
    }

    static constexpr uint64_t MX_GROUP_SIZE = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;

    struct SyncProtocol {
        static constexpr uint16_t MODE = Sync::SYNC_MODE_INTRA;
        static constexpr uint16_t AIV_READY_FLAG = 6U;
        static constexpr uint16_t AIC_FREE_FLAG = 8U;
        static constexpr uint16_t FLAG_ID_MAX = Sync::MIX_FLAG_ID_MAX;
    };

    class WeightL1Storage {
    public:
        __aicore__ inline void Init(uint64_t baseN, uint64_t kL1Size, uint64_t bufferNum, bool hasBias,
                                    uint64_t bufferBytes = 0)
        {
            bufferNum_ = bufferNum;
            buffersPerHalf_ = bufferNum / DOUBLE_BUFFER_COUNT;
            kL1SizeAligned_ = Align64(kL1Size);
            weightBufferSize_ = bufferBytes != 0 ? bufferBytes : baseN * kL1SizeAligned_ * sizeof(ConvertedBType);
            biasBufferSize_ = hasBias ? Align16(baseN) * sizeof(BiasType) : 0UL;

            constexpr uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT;
            for (uint16_t bufferId = 0; bufferId < bufferNum_; ++bufferId) {
                uint64_t halfOffset = (bufferId & 1U) * HALF_L1_SIZE;
                uint64_t slotInHalf = bufferId >> 1U;
                weightOffsets_[bufferId] = halfOffset + slotInHalf * weightBufferSize_;
                biasOffsets_[bufferId] = halfOffset + buffersPerHalf_ * weightBufferSize_ +
                                         slotInHalf * biasBufferSize_;
            }
        }

        __aicore__ inline void SetOffsets(uint64_t index, uint64_t weight, uint64_t bias)
        {
            weightOffsets_[index] = weight;
            biasOffsets_[index] = bias;
        }

        __aicore__ inline void SetKSize(uint64_t k) { kL1SizeAligned_ = Align64(k); }

        __aicore__ inline uint64_t WeightBufferSize() const { return weightBufferSize_; }

        __aicore__ inline uint64_t BiasBufferSize() const { return biasBufferSize_; }

        __aicore__ inline uint64_t BuffersPerHalf() const { return buffersPerHalf_; }

        __aicore__ inline auto MakeWeightTensor(uint64_t bufferId, uint64_t nOffset, uint64_t kOffset, uint64_t nStride,
                                                int64_t nSize, int64_t kSize) const
        {
            auto fullLayout = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                         AscendC::Std::Int<asc::te::c0_element<ConvertedBType>>>(
                static_cast<int64_t>(kL1SizeAligned_), static_cast<int64_t>(nStride));
            auto fullTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, ConvertedBType>(weightOffsets_[bufferId]), fullLayout);
            return fullTensor.slice(asc::te::make_coord(static_cast<int64_t>(kOffset), static_cast<int64_t>(nOffset)),
                                    asc::te::make_shape(kSize, nSize));
        }

        __aicore__ inline auto MakeBiasTensor(uint64_t bufferId, int64_t nSize) const
        {
            auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                static_cast<int64_t>(1), static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))));
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasOffsets_[bufferId]),
                                        layout);
        }

    private:
        uint64_t weightOffsets_[QUADRUPLE_BUFFER_COUNT] = {0UL};
        uint64_t biasOffsets_[QUADRUPLE_BUFFER_COUNT] = {0UL};
        uint64_t kL1SizeAligned_{0};
        uint64_t weightBufferSize_{0};
        uint64_t biasBufferSize_{0};
        uint64_t bufferNum_{0};
        uint64_t buffersPerHalf_{0};
    };

private:
    class L1Storage {
    public:
        __aicore__ inline void Init(uint64_t baseM, uint64_t baseN, uint64_t kL1Size, uint64_t scaleKL1Size,
                                    uint64_t bufferNum, bool hasBias, uint64_t aBytes = 0, uint64_t bBytes = 0)
        {
            weightStorage_.Init(baseN, kL1Size, bufferNum, hasBias, bBytes);
            uint64_t buffersPerHalf = weightStorage_.BuffersPerHalf();
            uint64_t aBufferSize = aBytes != 0 ? aBytes : baseM * Align64(kL1Size) * sizeof(AType);
            uint64_t scaleBBufferSize = baseN * Align64(scaleKL1Size) / MX_GROUP_SIZE * sizeof(ScaleBType);

            constexpr uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT;
            for (uint16_t bufferId = 0; bufferId < bufferNum; ++bufferId) {
                uint64_t halfOffset = (bufferId & 1U) * HALF_L1_SIZE;
                uint64_t slotInHalf = bufferId >> 1U;
                aOffsets_[bufferId] = halfOffset +
                                      buffersPerHalf *
                                          (weightStorage_.WeightBufferSize() + weightStorage_.BiasBufferSize()) +
                                      slotInHalf * aBufferSize;
            }
            for (uint16_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; ++bufferId) {
                uint64_t halfOffset = (bufferId & 1U) * HALF_L1_SIZE;
                scaleBOffsets_[bufferId] = halfOffset +
                                           buffersPerHalf * (weightStorage_.WeightBufferSize() +
                                                             weightStorage_.BiasBufferSize() + aBufferSize);
                scaleAOffsets_[bufferId] = scaleBOffsets_[bufferId] + scaleBBufferSize;
            }
        }

        __aicore__ inline auto MakeWeightTensor(uint64_t bufferId, uint64_t nOffset, uint64_t kOffset, uint64_t nStride,
                                                int64_t nSize, int64_t kSize) const
        {
            return weightStorage_.MakeWeightTensor(bufferId, nOffset, kOffset, nStride, nSize, kSize);
        }

        __aicore__ inline auto MakeATensor(uint64_t bufferId, int64_t mSize, int64_t kSize) const
        {
            auto layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                     AscendC::Std::Int<asc::te::c0_element<AType>>>(mSize, kSize);
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aOffsets_[bufferId]),
                                        layout);
        }

        __aicore__ inline auto MakeScaleATensor(uint64_t bufferId, int64_t mSize, int64_t scaleSize) const
        {
            auto layout = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                mSize, scaleSize);
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, ScaleAType>(scaleAOffsets_[bufferId]), layout);
        }

        __aicore__ inline auto MakeScaleBTensor(uint64_t bufferId, int64_t scaleSize, int64_t nSize) const
        {
            auto layout = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
                scaleSize, nSize);
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, ScaleBType>(scaleBOffsets_[bufferId]), layout);
        }

        __aicore__ inline auto MakeBiasTensor(uint64_t bufferId, int64_t nSize) const
        {
            return weightStorage_.MakeBiasTensor(bufferId, nSize);
        }

        __aicore__ inline void SetOffsets(const Params& params)
        {
            for (uint64_t i = 0; i < params.l1BufferNum; ++i) {
                weightStorage_.SetOffsets(i, params.weightOffsets[i], params.biasOffsets[i]);
                aOffsets_[i] = params.aOffsets[i];
            }
            for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
                scaleAOffsets_[i] = params.scaleAOffsets[i];
                scaleBOffsets_[i] = params.scaleBOffsets[i];
            }
        }

        __aicore__ inline void SetKSize(uint64_t k) { weightStorage_.SetKSize(k); }

        __aicore__ inline uint64_t AOffset(uint64_t bufferId) const { return aOffsets_[bufferId]; }

        __aicore__ inline uint64_t ScaleAOffset(uint64_t bufferId) const { return scaleAOffsets_[bufferId]; }

        __aicore__ inline uint64_t ScaleBOffset(uint64_t bufferId) const { return scaleBOffsets_[bufferId]; }

    private:
        WeightL1Storage weightStorage_;
        uint64_t aOffsets_[QUADRUPLE_BUFFER_COUNT] = {0UL};
        uint64_t scaleAOffsets_[DOUBLE_BUFFER_COUNT] = {0UL};
        uint64_t scaleBOffsets_[DOUBLE_BUFFER_COUNT] = {0UL};
    };

private:
    static constexpr uint64_t CACHE_LINE_BYTES = 128UL;

    __aicore__ inline static bool IsScaleCacheLineAligned(uint64_t innerSize)
    {
        return innerSize % CACHE_LINE_BYTES == 0UL;
    }

    __aicore__ inline void Init(const Params& params)
    {
        constexpr uint32_t SCALE_K_INDEX = 3;
        uint64_t l1BaseM = static_cast<uint64_t>(asc::te::get<MNK_M>(params.l1TileShape));
        uint64_t l1BaseN = static_cast<uint64_t>(asc::te::get<MNK_N>(params.l1TileShape));
        kL1Size_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.l1TileShape));
        scaleKL1Size_ = static_cast<uint64_t>(asc::te::get<SCALE_K_INDEX>(params.l1TileShape));
        kL0Size_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.l0TileShape));
        l1BufferNum_ = params.l1BufferNum;
        l1BufferMask_ = l1BufferNum_ - 1U;
        hasBias_ = params.hasBias;
        l1Storage_.Init(l1BaseM, l1BaseN, kL1Size_, scaleKL1Size_, l1BufferNum_, hasBias_, params.aBufferBytes,
                        params.bBufferBytes);
        if (params.explicitL1Offsets) {
            l1Storage_.SetOffsets(params);
        }
        readyFlag_ = params.readyFlag;
        freeFlag_ = params.freeFlag;
        disableWeightL2_ = params.disableWeightL2;
        problemM_ = params.problemM;
        kAL1Size_ = kL1Size_;
        InitBufferSlots(l1BaseN);
        if (params.seedWeightCredits) {
            for (uint16_t index = 0; index < l1BufferNum_; ++index) {
                ReleaseWeightBufferToVector();
            }
        }
        AscendC::SetMMLayoutTransform(true);
    }

    template <typename TensorA, typename TensorC>
    __aicore__ inline void InitBlockShape(const TensorA& tensorA, const TensorC& tensorC)
    {
        const auto& layoutA = tensorA.layout();
        const auto& layoutC = tensorC.layout();
        kSize_ = static_cast<uint64_t>(asc::te::get_total_column_shape(layoutA));
        mL1Len_ = static_cast<uint64_t>(asc::te::get_total_row_shape(layoutC));
        nL1Len_ = static_cast<uint64_t>(asc::te::get_total_column_shape(layoutC));
        kTileCount_ = CeilDiv(kSize_, kL1Size_);
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void ProcessBlock(const TensorA& tensorA, const TensorScaleA& tensorScaleA,
                                        const TensorScaleB& tensorScaleB, const TensorC& tensorC)
    {
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<BLOCK_CUBE>>(mL1Len_,
                                                                                                           nL1Len_);
        const BufferSlot l0cSlot{bufferManager_.GetL0CSlot(0U).Addr(), L0C_BUFFER_ID};
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot.Addr()),
                                              layoutL0C);

        uint64_t scaleWindowIter = 0;
        // GM 窗口偏移使用有符号 64 位整数，避免切片前发生无符号运算。
        int64_t aWindowStart = 0;
        int64_t aWindowLen = 0;
        uint64_t scaleWindowCols = 0;
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; ++kLoopIdx) {
            uint64_t l1BufIdx = l1BufIdx_;
            uint64_t scaleBufIdx = scaleBufIdx_;
            int64_t kL1Offset = static_cast<int64_t>(kLoopIdx) * static_cast<int64_t>(kL1Size_);
            int64_t kL1Len = Min(static_cast<int64_t>(kSize_) - kL1Offset, static_cast<int64_t>(kL1Size_));
            uint64_t kL1LenAligned = Align64(kL1Len);
            int64_t scaleKOffset = kL1Offset / static_cast<int64_t>(MX_GROUP_SIZE);
            // A and B advance independently: an A load may serve several converted-B stages.
            if (kL1Offset % static_cast<int64_t>(kAL1Size_) == 0) {
                aWindowStart = kL1Offset;
                aWindowLen = Min(static_cast<int64_t>(kSize_) - aWindowStart, static_cast<int64_t>(kAL1Size_));
            }
            auto tensorAL1 = l1Storage_.MakeATensor(aL1BufIdx_, mL1Len_, Align64(aWindowLen));
            auto tensorBL1 = l1Storage_.MakeWeightTensor(l1BufIdx, 0, 0, Align16(nL1Len_), nL1Len_, kL1LenAligned);

            if (scaleWindowIter == 0) {
                scaleWindowCols = CalcScaleWindowCols(kL1Offset);
            }
            auto tensorScaleAL1 = l1Storage_.MakeScaleATensor(scaleBufIdx, mL1Len_, scaleWindowCols);
            auto tensorScaleBL1 = l1Storage_.MakeScaleBTensor(scaleBufIdx, scaleWindowCols, nL1Len_);
            auto tensorBiasL1 = l1Storage_.MakeBiasTensor(l1BufIdx, nL1Len_);
            const auto& l1DataSlot = bufferManager_.GetL1ASlot(aL1BufIdx_);
            const auto& scaleASlot = bufferManager_.GetL1BSlot(SCALE_A_SLOT_BASE + scaleBufIdx);
            const auto& scaleBSlot = bufferManager_.GetL1BSlot(SCALE_B_SLOT_BASE + scaleBufIdx);
            if (scaleWindowIter == 0) {
                auto scaleALock = scaleASlot.LockMte2();
                auto scaleBLock = scaleBSlot.LockMte2();
                CopyMxScaleGmToL1(tensorScaleA, tensorScaleB, tensorScaleAL1, tensorScaleBL1, scaleKOffset,
                                  scaleWindowCols);
            }
            if (kL1Offset == aWindowStart) {
                auto l1DataLock = l1DataSlot.LockMte2();
                CopyAGmToL1(tensorA, tensorAL1, aWindowStart, aWindowLen);
            }
            const BufferSlot weightSlot{0, static_cast<uint8_t>(WEIGHT_BUFFER_ID_BASE + l1BufIdx)};
            {
                auto weightLock = weightSlot.LockMte2();
                ClearConvertedWeightL1Tail(tensorBL1, kL1Len);
            }
            WaitConvertedWeightReady();
            IterateMatmul(kLoopIdx, tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1,
                          kL1Len, kL1Offset - aWindowStart, scaleWindowIter * kL1Size_, l1DataSlot, scaleASlot,
                          scaleBSlot, weightSlot, l0cSlot);
            bool releaseScaleBuffer = ((scaleWindowIter + 1U) * kL1Size_ >= scaleKL1Size_) ||
                                      (kLoopIdx + 1U == kTileCount_);
            if (kL1Offset + kL1Len >= aWindowStart + aWindowLen) {
                aL1BufIdx_ = (aL1BufIdx_ + 1U) & l1BufferMask_;
            }
            PostProcess(releaseScaleBuffer, l1BufIdx);
            scaleWindowIter = releaseScaleBuffer ? 0U : scaleWindowIter + 1U;
        }
        if constexpr (AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulFinalizeRoutingWithWeightQuantMx>) {
            // The next two weight strips are staged before AIV releases its W8/output arena.
            Sync::WaitForVector<SyncProtocol::MODE, PIPE_FIX>(FINALIZE_ROUTING_OUTPUT_FREE_FLAG, true);
        }
        CopyCL0c2Gm(tensorC, tensorL0C, l0cSlot);
    }

    __aicore__ inline void InitBufferSlots(uint64_t baseN)
    {
        for (uint32_t index = 0; index < l1BufferNum_; ++index) {
            uint8_t bufferId = BufferLayout::L1ADataBufferId(index);
            // A is protected independently so it can outlive an individual converted-B stage.
            bufferManager_.InitAL1(index, l1Storage_.AOffset(index), bufferId);
        }
        for (uint32_t index = 0; index < DOUBLE_BUFFER_COUNT; ++index) {
            uint32_t scaleASlot = SCALE_A_SLOT_BASE + index;
            uint32_t scaleBSlot = SCALE_B_SLOT_BASE + index;
            bufferManager_.InitBL1(scaleASlot, l1Storage_.ScaleAOffset(index),
                                   BufferLayout::L1BDataBufferId(scaleASlot));
            bufferManager_.InitBL1(scaleBSlot, l1Storage_.ScaleBOffset(index),
                                   BufferLayout::L1BDataBufferId(scaleBSlot));
        }
        bufferManager_.InitBT(Align16(baseN) * sizeof(float));
        bufferManager_.InitL0();
        bufferManager_.InitL0C();
        // Explicit slot IDs reserve 8..11 for converted-weight tail fills.
        // BT=12, L0C=13, L0 ping/pong=14/15.
    }

    __aicore__ inline void WaitConvertedWeightReady() { WaitWeightFlag(readyFlag_); }

    __aicore__ inline void ReleaseWeightBufferToVector() { SetWeightFlag(freeFlag_); }

    __aicore__ inline void WaitWeightFlag(uint16_t flag) const
    {
        Sync::WaitForVector<SyncProtocol::MODE, PIPE_MTE1>(flag, true);
    }

    __aicore__ inline void SetWeightFlag(uint16_t flag) const
    {
        Sync::NotifyVector<SyncProtocol::MODE, PIPE_MTE1>(flag, true);
    }

    __aicore__ inline uint64_t CalcScaleWindowCols(int64_t kL1Offset) const
    {
        int64_t totalScaleCols = static_cast<int64_t>(Align64(kSize_) / MX_GROUP_SIZE);
        int64_t scaleKOffset = kL1Offset / static_cast<int64_t>(MX_GROUP_SIZE);
        if (scaleKOffset >= totalScaleCols) {
            return 0UL;
        }
        int64_t scaleWindowKLen = Min(static_cast<int64_t>(kSize_) - kL1Offset, static_cast<int64_t>(scaleKL1Size_));
        int64_t scaleWindowCols = static_cast<int64_t>(Align64(scaleWindowKLen) / MX_GROUP_SIZE);
        return Min(scaleWindowCols, totalScaleCols - scaleKOffset);
    }

// Padding primitives below are device-only; host UT only parses the surrounding metadata.
#if defined(__NPU_ARCH__)
    template <typename TensorA, typename TensorAL1>
    __aicore__ inline void CopyAGmToL1(const TensorA& tensorA, const TensorAL1& tensorAL1, int64_t kL1Offset,
                                       int64_t kL1Len)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kL1Offset),
                                      asc::te::make_shape(static_cast<int64_t>(mL1Len_), kL1Len));
        asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);
        Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
    }

    template <typename TensorBL1>
    __aicore__ inline void ClearConvertedWeightL1Tail(const TensorBL1& tensorBL1, uint64_t kL1Len)
    {
        uint64_t kPhysicalLen = Align32(kL1Len);
        uint64_t kMmadLen = Align64(kL1Len);
        if (kPhysicalLen == kMmadLen) {
            return;
        }

        // AIV writes through the final physical K32 block. Clearing starts at the next block,
        // so AIC fill and AIV UB2L1 can run concurrently before their existing barriers converge.
        auto tailTensor = tensorBL1.slice(
            asc::te::make_coord(static_cast<int64_t>(kPhysicalLen), 0),
            asc::te::make_shape(static_cast<int64_t>(kMmadLen - kPhysicalLen), static_cast<int64_t>(Align16(nL1Len_))));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(tailTensor, 1U, nL1Len_, 0U);
    }
#endif

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB,
                                             const TensorScaleAL1& tensorScaleAL1, const TensorScaleBL1& tensorScaleBL1,
                                             int64_t scaleKOffset, uint64_t scaleWindowCols)
    {
        auto copyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto gmBlockScaleA = tensorScaleA.slice(
            asc::te::make_coord(0, scaleKOffset),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(scaleWindowCols)));
        asc::te::copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

        if constexpr (DispatchPolicy::SCALE_B_SOURCE == WeightPrologueMxScaleBSource::AIC) {
            auto gmBlockScaleB = tensorScaleB.slice(
                asc::te::make_coord(scaleKOffset, 0),
                asc::te::make_shape(static_cast<int64_t>(scaleWindowCols), static_cast<int64_t>(nL1Len_)));
            const bool disableScaleBL2 = disableWeightL2_ && mL1Len_ >= problemM_ &&
                                         IsScaleCacheLineAligned(scaleWindowCols);
            gmBlockScaleB.set_l2_cache_hint(disableScaleBL2 ? asc::te::cache_mode::disable :
                                                              asc::te::cache_mode::normal);
            asc::te::copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
        }
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1, typename TensorBiasL1>
    __aicore__ inline void IterateMatmul(uint64_t kLoopIdx, const TensorL0C& tensorL0C, const TensorAL1& tensorAL1,
                                         const TensorBL1& tensorBL1, const TensorScaleAL1& tensorScaleAL1,
                                         const TensorScaleBL1& tensorScaleBL1, const TensorBiasL1& tensorBiasL1,
                                         uint64_t kL1Len, uint64_t kOffsetInAWindow, uint64_t kOffsetInScaleWindow,
                                         const BufferSlot& l1DataSlot, const BufferSlot& scaleASlot,
                                         const BufferSlot& scaleBSlot, const BufferSlot& weightSlot,
                                         const BufferSlot& l0cSlot)
    {
        uint64_t kL0Iter = CeilDiv(kL1Len, kL0Size_);
        for (uint64_t kL0Idx = 0; kL0Idx < kL0Iter; ++kL0Idx) {
            uint64_t kL0Offset = kL0Idx * kL0Size_;
            uint64_t realL0K = Min(kL1Len - kL0Offset, kL0Size_);
            uint64_t realL0KAligned = Align64(realL0K);
            uint64_t l0BufIdx = l0BufIdx_ & 1U;
            const BufferSlot l0Slot{bufferManager_.GetL0Slot(l0BufIdx).Addr(),
                                    static_cast<uint8_t>(L0_BUFFER_ID_BASE + l0BufIdx)};
            const BufferSlot btSlot{bufferManager_.GetBTSlot(0U).Addr(), BT_BUFFER_ID};
            uint64_t l0Offset = l0Slot.Addr();

            auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                        AscendC::Std::Int<asc::te::c0_element<AType>>>(mL1Len_,
                                                                                                       realL0KAligned);
            auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                                                  layoutAL0);
            auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                        AscendC::Std::Int<asc::te::c0_element<AType>>>(realL0KAligned,
                                                                                                       nL1Len_);
            auto tensorBL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0b, ConvertedBType>(l0Offset), layoutBL0);
            auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                static_cast<int64_t>(1), static_cast<int64_t>(Align16(nL1Len_)));
            auto tensorBt = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(btSlot.Addr()),
                                                 layoutBt);
            bool needBias = hasBias_ && kLoopIdx == 0U && kL0Idx == 0U;
            {
                auto l1DataLock = l1DataSlot.LockMte1();
                auto scaleALock = scaleASlot.LockMte1();
                auto scaleBLock = scaleBSlot.LockMte1();
                auto l0Lock = l0Slot.LockMte1();
                CopyAL1ToL0(tensorAL0, tensorAL1, kOffsetInAWindow + kL0Offset, realL0KAligned);
                {
                    auto weightLock = weightSlot.LockMte1();
                    CopyBL1ToL0(tensorBL0, tensorBL1, kL0Offset, realL0KAligned);
                }
                CopyScaleL1ToL0(tensorScaleAL1, tensorScaleBL1, kOffsetInScaleWindow + kL0Offset, realL0K, l0Offset);
                if (needBias) {
                    auto btLock = btSlot.LockMte1();
                    auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
                    asc::te::copy(copyL12BT, tensorBt, tensorBiasL1);
                }
            }

            auto mmadParams = MakeMmadParams(kLoopIdx, kL0Idx, kL0Iter, realL0K, needBias);
            ComputeMmad(mmadParams, tensorL0C, tensorAL0, tensorBL0, tensorBt, needBias, l0Slot, l0cSlot, btSlot);
            l0BufIdx_ ^= 1U;
        }
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1, typename TensorBiasL1>
    __aicore__ inline void IterateDeferredMatmul(uint64_t kLoopIdx, const TensorL0C& tensorL0C,
                                                 const TensorAL1& tensorAL1, const TensorBL1& tensorBL1,
                                                 const TensorScaleAL1& tensorScaleAL1,
                                                 const TensorScaleBL1& tensorScaleBL1, const TensorBiasL1& tensorBiasL1,
                                                 uint64_t kL1Len, uint64_t kOffsetInAWindow,
                                                 uint64_t kOffsetInScaleWindow, const BufferSlot& l1DataSlot,
                                                 const BufferSlot& scaleASlot, const BufferSlot& scaleBSlot,
                                                 const BufferSlot& weightSlot, const BufferSlot& l0cSlot)
    {
        const uint64_t kL0Iter = CeilDiv(kL1Len, kL0Size_);
        for (uint64_t kL0Idx = 0; kL0Idx < kL0Iter; ++kL0Idx) {
            const uint64_t kL0Offset = kL0Idx * kL0Size_;
            const uint64_t realL0K = Min(kL1Len - kL0Offset, kL0Size_);
            const uint64_t operandK = DEFERRED_UB ? realL0K : Align64(realL0K);
            const auto l0Slot = GetOperandSlot();
            const BufferSlot btSlot{0UL, BT_BUFFER_ID};
            auto tensorAL0 = Tile::MakeMxL0Operand<AType, false>(l0Slot.Addr(), mL1Len_, operandK);
            auto tensorBL0 = Tile::MakeMxL0Operand<ConvertedBType, true>(l0Slot.Addr(), operandK, nL1Len_);
            auto tensorBt = Tile::MakeMxBiasTable(btSlot.Addr(), nL1Len_);
            const bool needBias = hasBias_ && kLoopIdx == 0U && kL0Idx == 0U;
            LoadL0Operands(tensorAL0, tensorBL0, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1,
                           tensorBt, kOffsetInAWindow + kL0Offset, kL0Offset, kOffsetInScaleWindow + kL0Offset, realL0K,
                           operandK, needBias, l1DataSlot, scaleASlot, scaleBSlot, weightSlot, l0Slot, btSlot);
            auto params = DeferredMakeMmadParams(kLoopIdx, kL0Idx, kL0Iter, realL0K, needBias);
            DeferredComputeMmad(params, tensorL0C, tensorAL0, tensorBL0, tensorBt, needBias, l0Slot, l0cSlot, btSlot);
            l0BufIdx_ ^= 1U;
        }
    }

    __aicore__ inline BufferSlot GetOperandSlot()
    {
        const uint64_t index = l0BufIdx_ & 1U;
        return {index * L0_BUF_OFFSET, static_cast<uint8_t>(L0_BUFFER_ID_BASE + index)};
    }

    template <typename AL0, typename BL0, typename AL1, typename BL1, typename SA, typename SB, typename Bias,
              typename BT>
    __aicore__ inline void LoadL0Operands(const AL0& a0, const BL0& b0, const AL1& a1, const BL1& b1, const SA& sa,
                                          const SB& sb, const Bias& bias, const BT& bt, uint64_t aK, uint64_t bK,
                                          uint64_t scaleK, uint64_t realK, uint64_t operandK, bool needBias,
                                          const BufferSlot& aSlot, const BufferSlot& saSlot, const BufferSlot& sbSlot,
                                          const BufferSlot& bSlot, const BufferSlot& l0Slot, const BufferSlot& btSlot)
    {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
        CopyL0Operand<false>(a0, a1, aK, operandK);
        const uint64_t cols = CeilDiv(realK, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const uint64_t offset = scaleK / MXFP_DIVISOR_SIZE * MXFP_MULTI_BASE_SIZE;
        Tile::CopyMxScaleToL0<ScaleAType, false, uint64_t>(sa, l0Slot.Addr(), mL1Len_, cols, offset);
        CopyL0Operand<true>(b0, b1, bK, operandK);
        Tile::CopyMxScaleToL0<ScaleBType, true, uint64_t>(sb, l0Slot.Addr(), cols, nL1Len_, offset);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    }

    template <bool IsB, typename Destination, typename Source>
    __aicore__ inline void CopyL0Operand(const Destination& destination, const Source& source, uint64_t kOffset,
                                         uint64_t kSize)
    {
        if constexpr (IsB) {
            Tile::CopyMxOperandTile(COPY_L1_TO_L0B_ATOM, destination, source, static_cast<int64_t>(kOffset),
                                    int64_t{0L}, static_cast<int64_t>(kSize), static_cast<int64_t>(nL1Len_));
        } else {
            Tile::CopyMxOperandTile(COPY_L1_TO_L0A_ATOM, destination, source, int64_t{0L},
                                    static_cast<int64_t>(kOffset), static_cast<int64_t>(mL1Len_),
                                    static_cast<int64_t>(kSize));
        }
    }

    __aicore__ inline asc::te::mmad_params MakeMmadParams(uint64_t kLoopIdx, uint64_t kL0Idx, uint64_t kL0Iter,
                                                          uint64_t realL0K, bool needBias) const
    {
        asc::te::mmad_params mmadParams;
        mmadParams.m = static_cast<uint16_t>(mL1Len_);
        mmadParams.k = static_cast<uint16_t>(Align64(realL0K));
        mmadParams.n = static_cast<uint16_t>(Align16(nL1Len_));
        mmadParams.unit_flag = (kLoopIdx + 1U == kTileCount_ && kL0Idx + 1U == kL0Iter) ?
                                   asc::te::unit_flag_mode::enable_update :
                                   asc::te::unit_flag_mode::enable_keep;
        mmadParams.init_with_zero = (kLoopIdx == 0U && kL0Idx == 0U && !needBias);
        return mmadParams;
    }

    __aicore__ inline asc::te::mmad_params DeferredMakeMmadParams(uint64_t kLoopIdx, uint64_t kL0Idx, uint64_t kL0Iter,
                                                                  uint64_t realL0K, bool needBias) const
    {
        asc::te::mmad_params mmadParams;
        mmadParams.m = static_cast<uint16_t>(mL1Len_);
        mmadParams.n = static_cast<uint16_t>(nL1Len_);
        Tile::UpdateMxMmadParams(mmadParams, static_cast<uint16_t>(realL0K), kLoopIdx == 0U && kL0Idx == 0U,
                                 ((kLoopIdx + 1U) * kL1Size_ >= kSize_) && kL0Idx + 1U == kL0Iter, needBias);
        return mmadParams;
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBt>
    __aicore__ inline void ComputeMmad(const asc::te::mmad_params& mmadParams, const TensorL0C& tensorL0C,
                                       const TensorAL0& tensorAL0, const TensorBL0& tensorBL0, const TensorBt& tensorBt,
                                       bool needBias, const BufferSlot& l0Slot, const BufferSlot& l0cSlot,
                                       const BufferSlot& btSlot)
    {
        auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, Blaze::Gemm::Tile::MmadTraitMX{})
                            .with(mmadParams);
        auto l0Lock = l0Slot.LockM();
        auto l0cLock = l0cSlot.LockM();
        if (needBias) {
            auto btLock = btSlot.LockM();
            asc::te::mmad(mmadAtom, tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            asc::te::mmad(mmadAtom, tensorL0C, tensorAL0, tensorBL0);
        }
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBt>
    __aicore__ inline void DeferredComputeMmad(const asc::te::mmad_params& mmadParams, const TensorL0C& tensorL0C,
                                               const TensorAL0& tensorAL0, const TensorBL0& tensorBL0,
                                               const TensorBt& tensorBt, bool needBias, const BufferSlot& l0Slot,
                                               const BufferSlot& l0cSlot, const BufferSlot& btSlot)
    {
        Tile::IssueMxMmad(mmadParams, tensorL0C, tensorAL0, tensorBL0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
    }

    template <typename TensorAL0, typename TensorAL1>
    __aicore__ inline void CopyAL1ToL0(const TensorAL0& tensorAL0, const TensorAL1& tensorAL1, uint64_t kL0Offset,
                                       uint64_t realL0K)
    {
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto tensorBlockAL1 = tensorAL1.slice(
            asc::te::make_coord(0, static_cast<int64_t>(kL0Offset)),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(realL0K)));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);
    }

    template <typename TensorBL0, typename TensorBL1>
    __aicore__ inline void CopyBL1ToL0(const TensorBL0& tensorBL0, const TensorBL1& tensorBL1, uint64_t kL0Offset,
                                       uint64_t realL0K)
    {
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto tensorBlockBL1 = tensorBL1.slice(
            asc::te::make_coord(static_cast<int64_t>(kL0Offset), 0),
            asc::te::make_shape(static_cast<int64_t>(realL0K), static_cast<int64_t>(nL1Len_)));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void CopyScaleL1ToL0(const TensorScaleAL1& tensorScaleAL1, const TensorScaleBL1& tensorScaleBL1,
                                           uint64_t kOffsetInScaleWindow, uint64_t realL0K, uint64_t l0Offset)
    {
        uint64_t realL0ScaleCols = CeilDiv(realL0K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleAL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn,
                                                         AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(mL1Len_,
                                                                                                  realL0ScaleCols);
        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, ScaleAType>(l0Offset >> 4), layoutScaleAL0);
        auto copyScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        uint64_t scaleKOffset = (kOffsetInScaleWindow / MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(
            asc::te::make_coord(0, static_cast<int64_t>(scaleKOffset)),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(realL0ScaleCols)));
        asc::te::copy(copyScaleA, tensorScaleAL0, tensorBlockScaleAL1);

        auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn,
                                                         AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(realL0ScaleCols,
                                                                                                  nL1Len_);
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, ScaleBType>(l0Offset >> 4), layoutScaleBL0);
        auto copyScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(
            asc::te::make_coord(static_cast<int64_t>(scaleKOffset), 0),
            asc::te::make_shape(static_cast<int64_t>(realL0ScaleCols), static_cast<int64_t>(nL1Len_)));
        asc::te::copy(copyScaleB, tensorScaleBL0, tensorBlockScaleBL1);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyCL0c2Gm(const TensorC& tensorC, const TensorL0C& tensorL0C, const BufferSlot& l0cSlot)
    {
        if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
            auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, Tile::CopyL0C2UBTraitMixSplitM{});
            asc::te::copy(copyL0C2UB.with(asc::te::l0c_to_ub_params{asc::te::unit_flag_mode::enable_update}), tensorC,
                          tensorL0C);
        } else {
            constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), tensorC,
                          tensorL0C, FP32_64_AS_UINT64);
        }
    }

    __aicore__ inline void PostProcess(bool releaseScaleBuffer, uint64_t l1BufIdx)
    {
        ReleaseWeightBufferToVector();
        l1BufIdx_ = (l1BufIdx + 1U) & l1BufferMask_;
        if (releaseScaleBuffer) {
            scaleBufIdx_ ^= 1U;
        }
    }

    using BufferLayout = BufferIdLayout<QUADRUPLE_BUFFER_COUNT, QUADRUPLE_BUFFER_COUNT, DOUBLE_BUFFER_COUNT>;
    static constexpr uint16_t FINALIZE_ROUTING_OUTPUT_FREE_FLAG = 3U;
    static constexpr uint32_t SCALE_A_SLOT_BASE = 0U;
    static constexpr uint32_t SCALE_B_SLOT_BASE = DOUBLE_BUFFER_COUNT;
    // Disjoint local lock IDs: A 0..3, scales 4..7, B 8..11, BT 12, L0C 13, L0 14..15.
    static constexpr uint8_t WEIGHT_BUFFER_ID_BASE = 8;
    static constexpr uint8_t BT_BUFFER_ID = 12;
    static constexpr uint8_t L0C_BUFFER_ID = 13;
    static constexpr uint8_t L0_BUFFER_ID_BASE = 14;

    uint16_t readyFlag_{6};
    uint16_t freeFlag_{8};
    uint64_t kSize_{0};
    uint64_t kL1Size_{0};
    uint64_t kAL1Size_{0};
    uint64_t aL1BufIdx_{0};
    uint64_t scaleKL1Size_{0};
    uint64_t kL0Size_{0};
    uint64_t mL1Len_{0};
    uint64_t nL1Len_{0};
    uint64_t kTileCount_{0};
    uint64_t l1BufIdx_{0};
    uint64_t scaleBufIdx_{0};
    uint64_t l0BufIdx_{0};
    uint64_t l1BufferNum_{DOUBLE_BUFFER_COUNT};
    uint64_t l1BufferMask_{DOUBLE_BUFFER_COUNT - 1U};
    bool hasBias_{false};
    bool disableWeightL2_{false};
    uint64_t problemM_{0};
    AscendC::Std::conditional_t<DEFERRED_UB, AscendC::Std::tuple<>, L1Storage> l1Storage_;
    AscendC::Std::conditional_t<DEFERRED_UB, AscendC::Std::tuple<>,
                                BufferManager<QUADRUPLE_BUFFER_COUNT, QUADRUPLE_BUFFER_COUNT, DOUBLE_BUFFER_COUNT>>
        bufferManager_;

public:
    static_assert(!DEFERRED_UB ||
                  asc::te::is_satisfied_ptn_format_v<
                      decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((AType*)0),
                                                    asc::te::frame_layout_format<LayoutA>{}(0UL, 0UL))),
                      asc::te::nd_ext_layout_ptn>);

    constexpr static int32_t C0_SIZE = asc::te::c0_element<AType>;
    constexpr static int32_t SCALE_C0 = MXFP_MULTI_BASE_SIZE;
    constexpr static int32_t L0C_C0 = BLOCK_CUBE;
    static constexpr uint64_t SWIGLU_HALF_COUNT = 2UL;
    static constexpr uint64_t SPLIT_M_ROW_ALIGNMENT = 2UL;

    __aicore__ inline BlockMmad();
    template <typename A, typename S, typename B>
    __aicore__ inline auto ComputeContinuousToL0c(const A& a, const S& scaleA, const B& scaleB, uint64_t rows,
                                                  uint64_t halfN, uint64_t nOffset)
    {
        tensorScaleBGm_ = scaleB;
        scaleBNOffset_ = nOffset;
        scaleBufIdx_ = 0;
        mL1Len_ = rows;
        kSize_ = asc::te::get_total_column_shape(a.layout());
        nL1Len_ = halfN * SWIGLU_HALF_COUNT;
        CalcWeightQuantMxKBlock<AType, L1_N>(rows, nL1Len_, MxWeightQuantGmmL1Layout::A_L1_SINGLE_BUF_SIZE,
                                             MxWeightQuantGmmL1Layout::A_L1_SINGLE_BUF_SIZE, A_B_LOAD_BALANCE_FACTOR,
                                             kAL1Size_, kL1Size_, kL0Size_);
        const uint64_t copyRows = CeilAlign(rows, SPLIT_M_ROW_ALIGNMENT);
        const uint64_t bytes = CeilAlign(rows, static_cast<uint64_t>(BLOCK_CUBE)) * nL1Len_ * sizeof(float);
        if (continuousL0cOffset_ + bytes > AscendC::TOTAL_L0C_SIZE)
            continuousL0cOffset_ = 0;
        auto layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(copyRows, nL1Len_);
        auto accumulator = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(continuousL0cOffset_), layout);
        if (kSize_ % kL1Size_ == 0)
            ComputeTile<true>(a, scaleA, accumulator);
        else
            ComputeTile<false>(a, scaleA, accumulator);
        return accumulator;
    }
    template <typename C>
    __aicore__ inline void CopyContinuousResultToUb(const C& output, uint64_t rows, uint64_t halfN)
    {
        const uint64_t copyRows = CeilAlign(rows, SPLIT_M_ROW_ALIGNMENT);
        auto layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(
            copyRows, halfN * SWIGLU_HALF_COUNT);
        auto accumulator = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(continuousL0cOffset_), layout);
        auto copy = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, Tile::CopyL0C2UBTraitMixSplitM{});
        // MMAD and FIX both check/update the L0C unit flags, as in the native GMM pipeline.
        // The next MMAD can proceed as soon as its L0C region is released.
        asc::te::copy(copy.with(asc::te::l0c_to_ub_params{asc::te::unit_flag_mode::enable_update}), output,
                      accumulator);
        continuousL0cOffset_ += CeilAlign(rows, static_cast<uint64_t>(BLOCK_CUBE)) * halfN * SWIGLU_HALF_COUNT *
                                sizeof(float);
    }

private:
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    template <typename TensorL0C_>
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const TensorL0C_& tensorL0C);
    __aicore__ inline void SetMTE1ToMTE2();
    __aicore__ inline void SetScaleMTE1ToMTE2();
    template <typename TensorA_>
    __aicore__ inline void CopyAGmToL1(const TensorA_& tensorA, int64_t kaGmOffset);
    template <typename TensorScaleA_>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA_& tensorScaleA, uint64_t kbL1Offset);
    template <bool FullKStages, typename TensorA_, typename TensorScaleA_, typename TensorL0C_>
    __aicore__ inline void ComputeTile(const TensorA_& tensorA, const TensorScaleA_& tensorScaleA,
                                       const TensorL0C_& tensorL0C);

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 = typename asc::te::frame_layout_format<asc::te::zz_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 = typename asc::te::frame_layout_format<asc::te::nn_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;

    static constexpr uint64_t L1_N = 256UL;

    static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096UL;
    static constexpr uint64_t A_B_LOAD_BALANCE_FACTOR = 2UL;

    uint64_t continuousL0cOffset_ = 0;
    uint64_t scaleBNOffset_ = 0;
    using TensorScaleBGm = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(static_cast<__gm__ ScaleBType*>(nullptr)),
        asc::te::make_frame_layout<LayoutScaleB, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(int64_t{BLOCK_CUBE},
                                                                                          int64_t{BLOCK_CUBE})));
    AscendC::Std::conditional_t<DEFERRED_UB, TensorScaleBGm, AscendC::Std::tuple<>> tensorScaleBGm_;

    using TensorAL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(0UL),
        MakeLayoutAL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));
    using TensorScaleAL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, ScaleAType>(0UL),
        MakeLayoutScaleAL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));
    using TensorScaleBL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, ScaleBType>(0UL),
        MakeLayoutScaleBL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));

    AscendC::Std::conditional_t<DEFERRED_UB, TensorAL1, AscendC::Std::tuple<>> tensorAL1_;
    AscendC::Std::conditional_t<DEFERRED_UB, TensorScaleAL1, AscendC::Std::tuple<>> tensorScaleAL1_;
    AscendC::Std::conditional_t<DEFERRED_UB, TensorScaleBL1, AscendC::Std::tuple<>> tensorScaleBL1_;

    // --- sync ---
    // 2 buffer
    static constexpr TEventID EVENT_IDS_MTE1_TO_MTE2 = 0;
    // 2 buffer
    static constexpr TEventID EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 = 2;
    static constexpr TEventID EVENT_ID_MTE1_TO_MTE2 = 4;
    static constexpr TEventID EVENT_ID_MTE2_TO_MTE1 = 0;
    // Match the original basic API event assignment: M_MTE1 ping/pong uses
    // slots 3/4 and the shared MTE1_M fence uses slot 3.
    static constexpr TEventID EVENT_ID_M_TO_MTE1 = 3;
    static constexpr TEventID EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0UL;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1UL;

    /**
     * L1 512KB Memory Map
     * * Segment 1 [0KB - 256KB]:
     * [0k]    [64k]      [96k]    [128k]                        [256k]
     * |--B B0---|--scA0---|--scB0---|----------- A (Part 1) -------|
     *     (64KB)     (32KB)    (32KB)            (128KB)
     * * Segment 2 [256KB - 512KB]:
     * [256k]                    [384k]        [448k]   [480k]   [512k]
     * |-------- A (Part 2) -------|---- B B1 ----|--scA1--|--scB1--|
     *          (128KB)               (64KB)        (32KB)    (32KB)
     * * Note: A is a contiguous 256KB block spanning the middle of the buffer.
     */

    /**
     * L0 64KB Memory Map (double-buffered B tiles)
     * [0k]      [32KB]    [64KB]
     * |--- B0 ---|--- B1 ---|
     *    (32KB)     (32KB)
     */
    static constexpr uint64_t L0_BUF_OFFSET = 32UL * 1024UL;

    // Tensor API atoms carry no runtime state. Keeping them as class constants avoids rebuilding the descriptors
    // in every K iteration while preserving the same generated data-movement/MMAD operations.
    static constexpr auto COPY_GM_TO_L1_ATOM = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    static constexpr auto COPY_L1_TO_L0A_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
    static constexpr auto COPY_L1_TO_L0B_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
};

#define BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM                                                                       \
    template <class ScheduleType_, WeightPrologueMxKSchedule KSchedule_, WeightPrologueMxScaleBSource ScaleBSource_, \
              WeightPrologueMxOutput Output_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_,             \
              class LayoutBTuple_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>

#define BLAZE_WQGMM_MX_MIX_MMAD_CLASS                                                                             \
    BlockMmad<WeightQuantMxPolicy<ScheduleType_, KSchedule_, ScaleBSource_, Output_>, ATypeTuple_, LayoutATuple_, \
              BTypeTuple_, LayoutBTuple_, CType_, LayoutC_, BiasType_, LayoutBias_>

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorL0C_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize,
                                                                    const TensorL0C_& tensorL0C)
{
    auto tensorBL1 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufIdx_ * MxWeightQuantGmmL1Layout::L1_BUF_OFFSET),
        asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>{}(kbL1RealSize, nL1Len_));
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    const BufferSlot unused{0, 0};
    const uint64_t stage = static_cast<uint64_t>(kbOffset) / kL1Size_;
    IterateDeferredMatmul(stage, tensorL0C, tensorAL1_, tensorBL1, tensorScaleAL1_, tensorScaleBL1_, tensorBL1,
                          kbL1RealSize, static_cast<uint64_t>(kbOffset) % kAL1Size_,
                          static_cast<uint64_t>(kbOffset) % MX_SCALE_K_L1_SIZE, unused, unused, unused, unused, unused);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetScaleMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleBufIdx_);
    scaleBufIdx_ ^= 1;
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
    aL1BufIdx_ ^= 1;
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorA_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CopyAGmToL1(const TensorA_& tensorA, int64_t kaGmOffset)
{
    int64_t kaL1RealSize = (kaGmOffset + kAL1Size_) >= kSize_ ? kSize_ - kaGmOffset : kAL1Size_;
    auto layoutAL1 = MakeLayoutAL1{}(mL1Len_, kaL1RealSize);
    tensorAL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(MxWeightQuantGmmL1Layout::A_L1_OFFSET +
                                                            aL1BufIdx_ * MxWeightQuantGmmL1Layout::A_L1_BUF_OFFSET),
        layoutAL1);
    Tile::CopyMxOperandTile(COPY_GM_TO_L1_ATOM, tensorAL1_, tensorA, 0, kaGmOffset, mL1Len_, kaL1RealSize);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorScaleA_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CopyMxScaleGmToL1(const TensorScaleA_& tensorScaleA,
                                                                        uint64_t kbL1Offset)
{
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / WQGMM_MX_GROUP_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > kSize_ ?
                                    (kSize_ - kbL1Offset) / WQGMM_MX_GROUP_SIZE :
                                    scaleKL1StandardLen;
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(mL1Len_, scaleKL1RealSize);
    tensorScaleAL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(
            MxWeightQuantGmmL1Layout::SCALE_AL1_OFFSET + scaleBufIdx_ * MxWeightQuantGmmL1Layout::L1_BUF_OFFSET),
        layoutScaleAL1);
    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, nL1Len_);
    tensorScaleBL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(
            MxWeightQuantGmmL1Layout::SCALE_BL1_OFFSET + scaleBufIdx_ * MxWeightQuantGmmL1Layout::L1_BUF_OFFSET),
        layoutScaleBL1);
    Tile::CopyMxScalesToL1<uint64_t>(tensorScaleA, tensorScaleBGm_, tensorScaleAL1_, tensorScaleBL1_, mL1Len_, nL1Len_,
                                     kbL1Offset / WQGMM_MX_GROUP_SIZE, scaleKL1RealSize, scaleBNOffset_);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline BLAZE_WQGMM_MX_MIX_MMAD_CLASS::BlockMmad()
{
    static_assert(DEFERRED_UB, "Default construction requires deferred UB output");
    for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
    }
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <bool FullKStages, typename TensorA_, typename TensorScaleA_, typename TensorL0C_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::ComputeTile(const TensorA_& tensorA,
                                                                  const TensorScaleA_& tensorScaleA,
                                                                  const TensorL0C_& tensorL0C)
{
    for (uint64_t kbGmOffset = 0; kbGmOffset < kSize_; kbGmOffset += kL1Size_, l1BufIdx_ ^= 1) {
        uint64_t kbL1RealSize = FullKStages ? kL1Size_ :
                                              ((kbGmOffset + kL1Size_) >= kSize_ ? kSize_ - kbGmOffset : kL1Size_);
        if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleBufIdx_);
            CopyMxScaleGmToL1(tensorScaleA, kbGmOffset);
        }

        if (kbGmOffset % kAL1Size_ == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
            CopyAGmToL1(tensorA, kbGmOffset);
        }

        WaitAivToAic();
        ProcessTileL1(kbGmOffset, kbL1RealSize, tensorL0C);
        uint64_t nextKbGmOffset = kbGmOffset + kL1Size_;
        if (nextKbGmOffset % kAL1Size_ == 0 || nextKbGmOffset >= kSize_) {
            SetMTE1ToMTE2();
        }
        if (nextKbGmOffset % MX_SCALE_K_L1_SIZE == 0 || nextKbGmOffset >= kSize_) {
            SetScaleMTE1ToMTE2();
        }
        SetAicToAiv();
    }
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::WaitAivToAic()
{
    // Preserve second-AIV-first order; sync.h waits in base-first order.
    CrossCoreWaitFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + WQGMM_MX_FLAG_ID_MAX);
    CrossCoreWaitFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetAicToAiv()
{
    CrossCoreSetFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + WQGMM_MX_FLAG_ID_MAX);
    CrossCoreSetFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
#undef BLAZE_WQGMM_MX_MIX_MMAD_CLASS

#endif

} // namespace Block
} // namespace Gemm
} // namespace Blaze
