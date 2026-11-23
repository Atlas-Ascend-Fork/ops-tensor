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
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

/*!
 * \brief Shared physical L1 partition for the grouped MX weight-prologue producer and consumers.
 */
struct MxWeightQuantGmmL1Layout {
    static constexpr uint64_t KIB = 1024UL;
    static constexpr uint64_t SCALE_AL1_OFFSET = 64UL * KIB;
    static constexpr uint64_t SCALE_BL1_OFFSET = 96UL * KIB;
    static constexpr uint64_t A_L1_OFFSET = 128UL * KIB;
    static constexpr uint64_t L1_BUF_OFFSET = 384UL * KIB;
    static constexpr uint64_t A_L1_BUF_OFFSET = 128UL * KIB;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128UL * KIB;
    static constexpr uint64_t B_L1_SINGLE_BUF_SIZE = 64UL * KIB;
    static constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = 4UL * KIB;
    static constexpr uint64_t WEIGHT_OFFSETS[DOUBLE_BUFFER_COUNT] = {0, L1_BUF_OFFSET};
    static constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64 * KIB, 380 * KIB};
    static constexpr uint64_t SCALE_B_OFFSETS[DOUBLE_BUFFER_COUNT] = {SCALE_BL1_OFFSET,
                                                                      SCALE_BL1_OFFSET + L1_BUF_OFFSET};
    static constexpr uint64_t SCALE_K_WINDOW = 4096UL;
};

namespace Block {

#if defined(__NPU_ARCH__) || defined(ASCENDC_CPU_DEBUG)
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
    static constexpr bool IS_MATMUL = AscendC::Std::is_same_v<DispatchPolicy, MatmulWithWeightQuantMx>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;
    static constexpr uint32_t SCALE_K_INDEX = 3U;

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

    struct MatmulParams {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        L1TileShape l1TileShape{};
        L0TileShape l0TileShape{};
        uint64_t l1BufferNum{0};
        bool hasBias{false};
        uint64_t ubBufferNum{0};
    };

    struct GroupedMatmulParams {
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
    };

    using Params = AscendC::Std::conditional_t<AscendC::Std::is_same_v<DispatchPolicy, MatmulWithWeightQuantMx>,
                                               MatmulParams, GroupedMatmulParams>;

    __aicore__ inline BlockMmad(const GroupedMatmulParams& params) { Init(params); }

    __aicore__ inline BlockMmad(const MatmulParams& params)
        : params_(params),
          l1Storage_(static_cast<uint64_t>(asc::te::get<MNK_M>(params.l1TileShape)),
                     static_cast<uint64_t>(asc::te::get<MNK_N>(params.l1TileShape)),
                     static_cast<uint64_t>(asc::te::get<MNK_K>(params.l1TileShape)),
                     static_cast<uint64_t>(asc::te::get<SCALE_K_INDEX>(params.l1TileShape)), params.l1BufferNum,
                     params.hasBias)
    {
        InitBufferSlots();
        for (uint16_t index = 0; index < params.l1BufferNum; ++index) {
            ReleaseWeightBufferToVector(index);
        }
        AscendC::SetMMLayoutTransform(true);
    }

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

    __aicore__ inline static uint64_t NextBufferId(uint64_t bufferId, uint64_t bufferNum)
    {
        return bufferId + 1U == bufferNum ? 0U : bufferId + 1U;
    }

    struct SyncProtocol {
        static constexpr uint16_t MODE = Sync::SYNC_MODE_INTRA;
        static constexpr uint16_t AIV_READY_FLAG = 6U;
        static constexpr uint16_t AIC_FREE_FLAG = 8U;
        static constexpr uint16_t FLAG_ID_MAX = Sync::MIX_FLAG_ID_MAX;
    };

    class WeightL1Storage {
    public:
        __aicore__ inline WeightL1Storage() = default;

        __aicore__ inline WeightL1Storage(uint64_t baseN, uint64_t kL1Size, uint64_t bufferNum, bool hasBias,
                                          uint64_t bufferBytes = 0)
        {
            Init(baseN, kL1Size, bufferNum, hasBias, bufferBytes);
        }

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
    /*!
     * \brief Select MX weight-prologue K blocking without changing the caller's capacity model.
     */
    template <typename ElementType, uint64_t FullWidth = 0>
    __aicore__ static inline void CalcWeightQuantMxKBlock(uint64_t rows, uint64_t columns, uint64_t modelBytes,
                                                          uint64_t actualBytes, uint64_t reuseFactor, uint64_t& ka,
                                                          uint64_t& kb, uint64_t& k0)
    {
        constexpr uint64_t LARGE_K = 512UL;
        constexpr uint64_t MEDIUM_K = 256UL;
        constexpr uint64_t SMALL_K = 128UL;
        constexpr uint64_t SMALL_TILE_LIMIT = 128UL;
        if (rows == 0 || reuseFactor == 0) {
            ka = 0;
            kb = 0;
            k0 = SMALL_K;
            return;
        }
        const uint64_t mThreshold = modelBytes / sizeof(ElementType) / LARGE_K / BLOCK_CUBE * BLOCK_CUBE;
        kb = rows <= mThreshold && columns <= SMALL_TILE_LIMIT ? LARGE_K : MEDIUM_K;
        const uint64_t alignedRows = CeilAlign(rows, static_cast<uint64_t>(BLOCK_CUBE));
        if constexpr (FullWidth != 0) {
            reuseFactor = alignedRows >= FullWidth / reuseFactor && columns == FullWidth ? 1 : reuseFactor;
        }
        uint64_t depth = CeilDiv(columns, alignedRows * reuseFactor);
        depth = depth == 0 ? 1 : depth;
        const uint64_t modelDepth = (modelBytes / sizeof(ElementType)) / (alignedRows * kb);
        const uint64_t actualDepth = (actualBytes / sizeof(ElementType)) / (alignedRows * kb);
        uint64_t maxDepth = modelDepth < actualDepth ? modelDepth : actualDepth;
        maxDepth = maxDepth == 0 ? 1 : maxDepth;
        depth = depth < maxDepth ? depth : maxDepth;
        ka = depth * kb;
        k0 = rows <= SMALL_TILE_LIMIT && columns <= SMALL_TILE_LIMIT ? MEDIUM_K : SMALL_K;
    }

    /*!
     * \brief Shared MX operand operations; buffer ownership and fences belong to the caller.
     */
    template <typename T, bool OperandB>
    __aicore__ static inline auto MakeMxL0Operand(uint64_t offset, uint64_t rows, uint64_t columns)
    {
        using Layout = AscendC::Std::conditional_t<OperandB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0b, asc::te::location::l0a>;
        auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<asc::te::c0_element<T>>>(rows, columns);
        return asc::te::make_tensor(asc::te::make_mem_ptr<Location, T>(offset), layout);
    }

    template <typename Source, typename Destination, typename Copy, typename Row, typename Column, typename Rows,
              typename Columns>
    __aicore__ static inline void CopyMxOperandTile(const Copy& copy, const Destination& destination,
                                                    const Source& source, Row row, Column column, Rows rows,
                                                    Columns columns)
    {
        auto tile = source.slice(asc::te::make_coord(row, column), asc::te::make_shape(rows, columns));
        asc::te::copy(copy, destination, tile);
    }

    __aicore__ static inline auto MakeMxBiasTable(uint64_t offset, uint64_t columns)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1L),
                                                                             static_cast<int64_t>(Align16(columns)));
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(offset), layout);
    }

    template <typename Index = int64_t, typename SourceA, typename SourceB, typename DestinationA,
              typename DestinationB>
    __aicore__ static inline void CopyMxScalesToL1(const SourceA& a, const SourceB& b, const DestinationA& outA,
                                                   const DestinationB& outB, uint64_t rows, uint64_t columns,
                                                   uint64_t kOffset, uint64_t kColumns, uint64_t nOffset = 0)
    {
        constexpr uint64_t HALF_COUNT = 2UL;
        auto copy = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        CopyMxOperandTile(copy, outA, a, 0, static_cast<Index>(kOffset), static_cast<Index>(rows),
                          static_cast<Index>(kColumns));
        const uint64_t partColumns = columns / HALF_COUNT;
        const uint64_t sourceStride = asc::te::get_total_column_shape(b.layout()) / HALF_COUNT;
        for (uint64_t part = 0; part < HALF_COUNT; ++part) {
            auto destination = outB.slice(
                asc::te::make_coord(0, static_cast<Index>(part * partColumns)),
                asc::te::make_shape(static_cast<Index>(kColumns), static_cast<Index>(partColumns)));
            CopyMxOperandTile(copy, destination, b, static_cast<Index>(kOffset),
                              static_cast<Index>(nOffset + part * sourceStride), static_cast<Index>(kColumns),
                              static_cast<Index>(partColumns));
        }
    }

    template <typename C, typename A, typename B>
    __aicore__ static inline void IssueMxMmad(const asc::te::mmad_params& params, const C& c, const A& a, const B& b)
    {
        auto atom = asc::te::make_mmad(asc::te::mmad_operation{}, Tile::MmadTraitMX{}).with(params);
        asc::te::mmad(atom, c, a, b);
    }

    static constexpr uint64_t MX_L0_SCALE_ADDRESS_SHIFT = 4UL;

    __aicore__ static inline void UpdateMxMmadParams(asc::te::mmad_params& params, uint16_t k, bool firstK, bool lastK,
                                                     bool needBias)
    {
        params.k = k;
        params.unit_flag = lastK ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = firstK && !needBias;
    }

    template <typename Scale, bool OperandB, typename Index = int64_t, typename Tensor>
    __aicore__ static inline void CopyMxScaleToL0(const Tensor& source, uint64_t l0Offset, uint64_t rows,
                                                  uint64_t columns, uint64_t kOffset, uint64_t outerOffset = 0)
    {
        using Layout = AscendC::Std::conditional_t<OperandB, asc::te::nn_layout_ptn, asc::te::zz_layout_ptn>;
        using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0scaleb,
                                                     asc::te::location::l0scalea>;
        using Copy = AscendC::Std::conditional_t<OperandB, asc::te::copy_l1_to_l0scaleb, asc::te::copy_l1_to_l0scalea>;
        auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(rows, columns);
        auto destination = asc::te::make_tensor(
            asc::te::make_mem_ptr<Location, Scale>(l0Offset >> MX_L0_SCALE_ADDRESS_SHIFT), layout);
        const Index offset = static_cast<Index>(kOffset);
        if constexpr (OperandB) {
            CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, offset, static_cast<Index>(outerOffset),
                              static_cast<Index>(rows), static_cast<Index>(columns));
        } else {
            CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, static_cast<Index>(outerOffset), offset,
                              static_cast<Index>(rows), static_cast<Index>(columns));
        }
    }

    class L1Storage {
    public:
        __aicore__ inline L1Storage() = default;

        __aicore__ inline L1Storage(uint64_t baseM, uint64_t baseN, uint64_t kL1Size, uint64_t scaleKL1Size,
                                    uint64_t bufferNum, bool hasBias, uint64_t aBytes = 0, uint64_t bBytes = 0)
        {
            Init(baseM, baseN, kL1Size, scaleKL1Size, bufferNum, hasBias, aBytes, bBytes);
        }

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

        __aicore__ inline void SetOffsets(const GroupedMatmulParams& params)
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
    // Tensor API atoms carry no runtime state. Class constants avoid rebuilding descriptors in K loops.
    static constexpr auto COPY_GM_TO_L1_ATOM = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    static constexpr auto COPY_L1_TO_L0A_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
    static constexpr auto COPY_L1_TO_L0B_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
    static constexpr auto COPY_L1_TO_L0SCALE_A_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
    static constexpr auto COPY_L1_TO_L0SCALE_B_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
    static constexpr auto COPY_L1_TO_BT_ATOM = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
    static constexpr auto COPY_L0C_TO_UB_ATOM = asc::te::make_copy(asc::te::copy_l0c_to_ub{},
                                                                   Tile::CopyL0C2UBTraitMixSplitM{});
    static constexpr auto COPY_L0C_TO_GM_ATOM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});

    __aicore__ inline void Init(const GroupedMatmulParams& params)
    {
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
        kAL1Size_ = kL1Size_;
        InitBufferSlots();
        if (params.seedWeightCredits) {
            for (uint16_t index = 0; index < l1BufferNum_; ++index) {
                ReleaseWeightBufferToVector(index);
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
        nL1LenAligned_ = Align16(nL1Len_);
        if constexpr (IS_MATMUL) {
            kTileCount_ = CeilDiv(kSize_, static_cast<uint64_t>(asc::te::get<MNK_K>(params_.l1TileShape)));
        } else {
            kTileCount_ = CeilDiv(kSize_, kL1Size_);
        }
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void ProcessBlock(const TensorA& tensorA, const TensorScaleA& tensorScaleA,
                                        const TensorScaleB& tensorScaleB, const TensorC& tensorC)
    {
        const bool splitNForK256 = CanSplitNForK256();
        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<BLOCK_CUBE>>(mL1Len_,
                                                                                                           nL1Len_);
        const auto& l0cSlot = l0cSlot_;
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot.Addr()),
                                              layoutL0C);

        uint64_t scaleWindowIter = 0;
        // GM 窗口偏移使用有符号 64 位整数，避免切片前发生无符号运算。
        int64_t aWindowStart = 0;
        int64_t aWindowLen = 0;
        uint64_t scaleWindowCols = 0;
        uint64_t scaleWindowOffset = 0;
        int64_t kL1Offset = 0;
        uint64_t kL1Size = kL1Size_;
        uint64_t kAL1Size = kAL1Size_;
        uint64_t scaleKL1Size = scaleKL1Size_;
        uint64_t l1BufferNum = l1BufferNum_;
        if constexpr (IS_MATMUL) {
            kL1Size = static_cast<uint64_t>(asc::te::get<MNK_K>(params_.l1TileShape));
            kAL1Size = kL1Size;
            scaleKL1Size = static_cast<uint64_t>(asc::te::get<SCALE_K_INDEX>(params_.l1TileShape));
            l1BufferNum = params_.l1BufferNum;
        }
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; ++kLoopIdx) {
            uint64_t l1BufIdx = l1BufIdx_;
            uint64_t scaleBufIdx = scaleBufIdx_;
            int64_t kL1Len = Min(static_cast<int64_t>(kSize_) - kL1Offset, static_cast<int64_t>(kL1Size));
            uint64_t kL1LenAligned = Align64(kL1Len);
            int64_t scaleKOffset = kL1Offset / static_cast<int64_t>(MX_GROUP_SIZE);
            // A and B advance independently: an A load may serve several converted-B stages.
            if (kL1Offset % static_cast<int64_t>(kAL1Size) == 0) {
                aWindowStart = kL1Offset;
                aWindowLen = Min(static_cast<int64_t>(kSize_) - aWindowStart, static_cast<int64_t>(kAL1Size));
            }
            auto tensorAL1 = l1Storage_.MakeATensor(aL1BufIdx_, mL1Len_, Align64(aWindowLen));
            auto tensorBL1 = l1Storage_.MakeWeightTensor(l1BufIdx, 0, 0, nL1LenAligned_, nL1Len_, kL1LenAligned);

            if (scaleWindowIter == 0) {
                scaleWindowCols = CalcScaleWindowCols(kL1Offset, scaleKL1Size);
            }
            auto tensorScaleAL1 = l1Storage_.MakeScaleATensor(scaleBufIdx, mL1Len_, scaleWindowCols);
            auto tensorScaleBL1 = l1Storage_.MakeScaleBTensor(scaleBufIdx, scaleWindowCols, nL1Len_);
            auto tensorBiasL1 = l1Storage_.MakeBiasTensor(l1BufIdx, nL1Len_);
            const auto& l1DataSlot = aSlots_[aL1BufIdx_];
            const auto& scaleASlot = scaleASlots_[scaleBufIdx];
            const auto& scaleBSlot = scaleBSlots_[scaleBufIdx];
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
            const auto& weightSlot = weightSlots_[l1BufIdx];
            {
                auto weightLock = weightSlot.LockMte2();
                ClearConvertedWeightL1Tail(tensorBL1, kL1Len);
            }
            if (splitNForK256) {
                IterateNFragmentsK256(kLoopIdx, tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1,
                                      kL1Offset - aWindowStart, scaleWindowOffset, l1DataSlot, scaleASlot, scaleBSlot,
                                      weightSlot, l0cSlot);
            } else {
                WaitConvertedWeightReady(l1BufIdx);
                IterateMatmul(kLoopIdx, tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1,
                              kL1Len, kL1Offset - aWindowStart, scaleWindowOffset, l1DataSlot, scaleASlot, scaleBSlot,
                              weightSlot, l0cSlot);
            }
            bool releaseScaleBuffer = ((scaleWindowIter + 1U) * kL1Size >= scaleKL1Size) ||
                                      (kLoopIdx + 1U == kTileCount_);
            if (kL1Offset + kL1Len >= aWindowStart + aWindowLen) {
                if constexpr (IS_MATMUL) {
                    aL1BufIdx_ = NextBufferId(aL1BufIdx_, l1BufferNum);
                } else {
                    aL1BufIdx_ = (aL1BufIdx_ + 1U) & l1BufferMask_;
                }
            }
            ReleaseWeightBufferToVector(l1BufIdx);
            if constexpr (IS_MATMUL) {
                l1BufIdx_ = NextBufferId(l1BufIdx, l1BufferNum);
            } else {
                l1BufIdx_ = (l1BufIdx + 1U) & l1BufferMask_;
            }
            if (releaseScaleBuffer) {
                scaleBufIdx_ ^= 1U;
            }
            scaleWindowIter = releaseScaleBuffer ? 0U : scaleWindowIter + 1U;
            scaleWindowOffset = releaseScaleBuffer ? 0U : scaleWindowOffset + kL1Size;
            kL1Offset += static_cast<int64_t>(kL1Size);
        }
        if constexpr (AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulFinalizeRoutingWithWeightQuantMx>) {
            // The next two weight strips are staged before AIV releases its W8/output arena.
            Sync::WaitForVector<SyncProtocol::MODE, PIPE_FIX>(FINALIZE_ROUTING_OUTPUT_FREE_FLAG, true);
        }
        CopyCL0c2Gm(tensorC, tensorL0C, l0cSlot);
    }

    __aicore__ inline void InitBufferSlots()
    {
        uint64_t l1BufferNum = l1BufferNum_;
        if constexpr (IS_MATMUL) {
            l1BufferNum = params_.l1BufferNum;
        }
        for (uint32_t index = 0; index < l1BufferNum; ++index) {
            // A is protected independently so it can outlive an individual converted-B stage.
            aSlots_[index] = {l1Storage_.AOffset(index), static_cast<uint8_t>(index)};
            weightSlots_[index] = {0UL, static_cast<uint8_t>(WEIGHT_BUFFER_ID_BASE + index)};
        }
        for (uint32_t index = 0; index < DOUBLE_BUFFER_COUNT; ++index) {
            scaleASlots_[index] = {l1Storage_.ScaleAOffset(index), static_cast<uint8_t>(SCALE_A_LOCK_ID_BASE + index)};
            scaleBSlots_[index] = {l1Storage_.ScaleBOffset(index), static_cast<uint8_t>(SCALE_B_LOCK_ID_BASE + index)};
        }
        btSlot_ = {0UL, BT_BUFFER_ID};
        constexpr uint64_t L0_STAGE_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        for (uint32_t index = 0; index < DOUBLE_BUFFER_COUNT; ++index) {
            l0Slots_[index] = {L0_STAGE_SIZE * index, static_cast<uint8_t>(L0_BUFFER_ID_BASE + index)};
        }
        l0cSlot_ = {0UL, L0C_BUFFER_ID};
        // Explicit slot IDs reserve 8..11 for converted-weight tail fills.
        // BT=12, L0C=13, L0 ping/pong=14/15.
    }

    __aicore__ inline void WaitConvertedWeightReady(uint64_t l1Stage)
    {
        const uint16_t flag = IS_MATMUL ? static_cast<uint16_t>(l1Stage) : readyFlag_;
        WaitWeightFlag(flag);
    }

    __aicore__ inline void ReleaseWeightBufferToVector(uint64_t l1Stage)
    {
        const uint16_t flag = IS_MATMUL ? static_cast<uint16_t>(l1Stage) : freeFlag_;
        SetWeightFlag(flag);
    }

    __aicore__ inline void WaitWeightFlag(uint16_t flag) const
    {
        Sync::WaitForVector<SyncProtocol::MODE, PIPE_MTE1>(flag, true);
    }

    __aicore__ inline void SetWeightFlag(uint16_t flag) const
    {
        Sync::NotifyVector<SyncProtocol::MODE, PIPE_MTE1>(flag, true);
    }

    __aicore__ inline uint64_t CalcScaleWindowCols(int64_t kL1Offset, uint64_t scaleKL1Size) const
    {
        int64_t totalScaleCols = static_cast<int64_t>(Align64(kSize_) / MX_GROUP_SIZE);
        int64_t scaleKOffset = kL1Offset / static_cast<int64_t>(MX_GROUP_SIZE);
        if (scaleKOffset >= totalScaleCols) {
            return 0UL;
        }
        int64_t scaleWindowKLen = Min(static_cast<int64_t>(kSize_) - kL1Offset, static_cast<int64_t>(scaleKL1Size));
        int64_t scaleWindowCols = static_cast<int64_t>(Align64(scaleWindowKLen) / MX_GROUP_SIZE);
        return Min(scaleWindowCols, totalScaleCols - scaleKOffset);
    }

    // K256 keeps one A/scale-A operand in L0A and feeds two half-L0B N fragments.
    static constexpr uint64_t K256_N_SPLIT_BASE_K = 128UL;
    static constexpr uint64_t K256_N_SPLIT_SIZE = 256UL;
    static constexpr uint64_t K256_N_SPLIT_N = AscendC::TOTAL_L0B_SIZE / DOUBLE_BUFFER_COUNT / K256_N_SPLIT_SIZE /
                                               sizeof(ConvertedBType);
    static constexpr uint64_t K256_N_SPLIT_MAX_M = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT / K256_N_SPLIT_SIZE /
                                                   sizeof(AType);

    __aicore__ inline bool CanSplitNForK256() const
    {
        if constexpr (IS_MATMUL && IsWeightNz<LayoutB>::value) {
            return !params_.hasBias && mL1Len_ <= K256_N_SPLIT_MAX_M && kL0Size_ == K256_N_SPLIT_BASE_K &&
                   kL1Size_ == K256_N_SPLIT_SIZE && kSize_ % K256_N_SPLIT_SIZE == 0UL &&
                   scaleKL1Size_ >= K256_N_SPLIT_SIZE && scaleKL1Size_ % K256_N_SPLIT_SIZE == 0UL && mL1Len_ != 0UL &&
                   nL1Len_ == DOUBLE_BUFFER_COUNT * K256_N_SPLIT_N;
        }
        return false;
    }

    __aicore__ inline BufferSlot GetK256SplitBL0Slot(uint64_t index) const
    {
        constexpr uint8_t FIRST_K256_SPLIT_B_ID = 16U;
        static_assert(FIRST_K256_SPLIT_B_ID > L0_BUFFER_ID_BASE + 1U);
        return {(AscendC::TOTAL_L0B_SIZE / DOUBLE_BUFFER_COUNT) * index,
                static_cast<uint8_t>(FIRST_K256_SPLIT_B_ID + index)};
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1>
    __aicore__ inline void IterateNFragmentsK256(uint64_t kLoopIdx, const TensorL0C& tensorL0C,
                                                 const TensorAL1& tensorAL1, const TensorBL1& tensorBL1,
                                                 const TensorScaleAL1& tensorScaleAL1,
                                                 const TensorScaleBL1& tensorScaleBL1, uint64_t kOffsetInAWindow,
                                                 uint64_t kOffsetInScaleWindow, const BufferSlot& l1DataSlot,
                                                 const BufferSlot& scaleASlot, const BufferSlot& scaleBSlot,
                                                 const BufferSlot& weightSlot, const BufferSlot& l0cSlot)
    {
        const auto& aSlot = l0Slots_[l0BufIdx_ & 1U];
        constexpr uint64_t SCALE_COLS = K256_N_SPLIT_SIZE / MX_GROUP_SIZE;
        const uint64_t scaleKOffset = kOffsetInScaleWindow / MX_GROUP_SIZE;
        auto tensorAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0a, AType>(aSlot.Addr()),
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<asc::te::c0_element<AType>>>(
                mL1Len_, K256_N_SPLIT_SIZE));
        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, ScaleAType>(aSlot.Addr() >> 4),
            asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(mL1Len_,
                                                                                                        SCALE_COLS));
        CopyK256AOperands(tensorAL0, tensorScaleAL0, tensorAL1, tensorScaleAL1, kOffsetInAWindow, scaleKOffset,
                          l1DataSlot, scaleASlot, aSlot);
        WaitConvertedWeightReady(l1BufIdx_);
        auto weightSource = weightSlot.LockMte1();
        auto scaleBSource = scaleBSlot.LockMte1();
        auto aRead = aSlot.LockM();
        auto cRead = l0cSlot.LockM();
        asc::te::mmad_params mmadParams;
        mmadParams.m = static_cast<uint16_t>(mL1Len_);
        mmadParams.k = static_cast<uint16_t>(K256_N_SPLIT_SIZE);
        mmadParams.n = static_cast<uint16_t>(Align16(K256_N_SPLIT_N));
        mmadParams.init_with_zero = kLoopIdx == 0UL;
        mmadParams.unit_flag = kLoopIdx + 1UL == kTileCount_ ? asc::te::unit_flag_mode::enable_update :
                                                               asc::te::unit_flag_mode::enable_keep;
        const auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, Blaze::Gemm::Tile::MmadTraitMX{})
                                  .with(mmadParams);
        for (uint64_t fragment = 0UL; fragment < DOUBLE_BUFFER_COUNT; ++fragment) {
            ComputeK256NFragment(fragment, tensorL0C, tensorAL0, tensorBL1, tensorScaleBL1, scaleKOffset, mmadAtom);
        }
        l0BufIdx_ ^= 1U;
    }

    template <typename TensorAL0, typename TensorScaleAL0, typename TensorAL1, typename TensorScaleAL1>
    __aicore__ inline void CopyK256AOperands(const TensorAL0& tensorAL0, const TensorScaleAL0& tensorScaleAL0,
                                             const TensorAL1& tensorAL1, const TensorScaleAL1& tensorScaleAL1,
                                             uint64_t kOffsetInAWindow, uint64_t scaleKOffset,
                                             const BufferSlot& l1DataSlot, const BufferSlot& scaleASlot,
                                             const BufferSlot& aSlot)
    {
        constexpr uint64_t SCALE_COLS = K256_N_SPLIT_SIZE / MX_GROUP_SIZE;
        {
            auto dataSource = l1DataSlot.LockMte1();
            auto scaleSource = scaleASlot.LockMte1();
            auto aWrite = aSlot.LockMte1();
            auto tensorBlockAL1 = tensorAL1.slice(
                asc::te::make_coord(int64_t{0}, static_cast<int64_t>(kOffsetInAWindow)),
                asc::te::make_shape(static_cast<int64_t>(mL1Len_), int64_t{K256_N_SPLIT_SIZE}));
            asc::te::copy(COPY_L1_TO_L0A_ATOM, tensorAL0, tensorBlockAL1);
            auto tensorBlockScaleAL1 = tensorScaleAL1.slice(
                asc::te::make_coord(int64_t{0}, static_cast<int64_t>(scaleKOffset)),
                asc::te::make_shape(static_cast<int64_t>(mL1Len_), int64_t{SCALE_COLS}));
            asc::te::copy(COPY_L1_TO_L0SCALE_A_ATOM, tensorScaleAL0, tensorBlockScaleAL1);
        }
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL1, typename TensorScaleBL1, typename MmadAtom>
    __aicore__ inline void ComputeK256NFragment(uint64_t fragment, const TensorL0C& tensorL0C,
                                                const TensorAL0& tensorAL0, const TensorBL1& tensorBL1,
                                                const TensorScaleBL1& tensorScaleBL1, uint64_t scaleKOffset,
                                                const MmadAtom& mmadAtom)
    {
        constexpr uint64_t SCALE_COLS = K256_N_SPLIT_SIZE / MX_GROUP_SIZE;
        const uint64_t nOffset = fragment * K256_N_SPLIT_N;
        const uint64_t nLength = Min(nL1Len_ - nOffset, K256_N_SPLIT_N);
        const auto bSlot = GetK256SplitBL0Slot(fragment);
        auto tensorBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, ConvertedBType>(bSlot.Addr()),
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<asc::te::c0_element<AType>>>(
                K256_N_SPLIT_SIZE, nLength));
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, ScaleBType>(bSlot.Addr() >> 4),
            asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(SCALE_COLS,
                                                                                                        nLength));
        {
            auto bWrite = bSlot.LockMte1();
            auto tensorBlockBL1 = tensorBL1.slice(
                asc::te::make_coord(int64_t{0}, static_cast<int64_t>(nOffset)),
                asc::te::make_shape(int64_t{K256_N_SPLIT_SIZE}, static_cast<int64_t>(nLength)));
            asc::te::copy(COPY_L1_TO_L0B_ATOM, tensorBL0, tensorBlockBL1);
            auto tensorBlockScaleBL1 = tensorScaleBL1.slice(
                asc::te::make_coord(static_cast<int64_t>(scaleKOffset), static_cast<int64_t>(nOffset)),
                asc::te::make_shape(int64_t{SCALE_COLS}, static_cast<int64_t>(nLength)));
            asc::te::copy(COPY_L1_TO_L0SCALE_B_ATOM, tensorScaleBL0, tensorBlockScaleBL1);
        }
        auto cFragment = tensorL0C.slice(
            asc::te::make_coord(int64_t{0}, static_cast<int64_t>(nOffset)),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(nLength)));
        auto bRead = bSlot.LockM();
        asc::te::mmad(mmadAtom, cFragment, tensorAL0, tensorBL0);
    }

// Padding primitives below are device-only; host UT only parses the surrounding metadata.
#if defined(__NPU_ARCH__)
    template <typename TensorA, typename TensorAL1>
    __aicore__ inline void CopyAGmToL1(const TensorA& tensorA, const TensorAL1& tensorAL1, int64_t kL1Offset,
                                       int64_t kL1Len)
    {
        auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kL1Offset),
                                      asc::te::make_shape(static_cast<int64_t>(mL1Len_), kL1Len));
        asc::te::copy(COPY_GM_TO_L1_ATOM, tensorAL1, gmBlockA);
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
        auto gmBlockScaleA = tensorScaleA.slice(
            asc::te::make_coord(0, scaleKOffset),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(scaleWindowCols)));
        asc::te::copy(COPY_GM_TO_L1_ATOM, tensorScaleAL1, gmBlockScaleA);

        if constexpr (DispatchPolicy::SCALE_B_SOURCE == WeightPrologueMxScaleBSource::AIC) {
            auto gmBlockScaleB = tensorScaleB.slice(
                asc::te::make_coord(scaleKOffset, 0),
                asc::te::make_shape(static_cast<int64_t>(scaleWindowCols), static_cast<int64_t>(nL1Len_)));
            asc::te::copy(COPY_GM_TO_L1_ATOM, tensorScaleBL1, gmBlockScaleB);
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
        uint64_t kL0Size = kL0Size_;
        bool hasBias = hasBias_;
        if constexpr (IS_MATMUL) {
            kL0Size = static_cast<uint64_t>(asc::te::get<MNK_K>(params_.l0TileShape));
            hasBias = params_.hasBias;
        }
        uint64_t kL0Iter = CeilDiv(kL1Len, kL0Size);
        const auto& btSlot = btSlot_;
        auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1),
                                                                               static_cast<int64_t>(nL1LenAligned_));
        auto tensorBt = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(btSlot.Addr()),
                                             layoutBt);
        uint64_t kL0Offset = 0;
        for (uint64_t kL0Idx = 0; kL0Idx < kL0Iter; ++kL0Idx) {
            uint64_t realL0K = Min(kL1Len - kL0Offset, kL0Size);
            uint64_t realL0KAligned = Align64(realL0K);
            uint64_t l0BufIdx = l0BufIdx_ & 1U;
            const auto& l0Slot = l0Slots_[l0BufIdx];
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
            const bool needBias = hasBias && kLoopIdx == 0U && kL0Idx == 0U;
            {
                auto l1DataLock = l1DataSlot.LockMte1();
                auto scaleALock = scaleASlot.LockMte1();
                auto scaleBLock = scaleBSlot.LockMte1();
                auto l0Lock = l0Slot.LockMte1();
                auto tensorBlockAL1 = tensorAL1.slice(
                    asc::te::make_coord(0, static_cast<int64_t>(kOffsetInAWindow + kL0Offset)),
                    asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(realL0KAligned)));
                asc::te::copy(COPY_L1_TO_L0A_ATOM, tensorAL0, tensorBlockAL1);
                {
                    auto weightLock = weightSlot.LockMte1();
                    auto tensorBlockBL1 = tensorBL1.slice(
                        asc::te::make_coord(static_cast<int64_t>(kL0Offset), 0),
                        asc::te::make_shape(static_cast<int64_t>(realL0KAligned), static_cast<int64_t>(nL1Len_)));
                    asc::te::copy(COPY_L1_TO_L0B_ATOM, tensorBL0, tensorBlockBL1);
                }
                CopyScaleL1ToL0(tensorScaleAL1, tensorScaleBL1, kOffsetInScaleWindow + kL0Offset, realL0K, l0Offset);
                if (needBias) {
                    auto btLock = btSlot.LockMte1();
                    asc::te::copy(COPY_L1_TO_BT_ATOM, tensorBt, tensorBiasL1);
                }
            }

            asc::te::mmad_params mmadParams;
            mmadParams.m = static_cast<uint16_t>(mL1Len_);
            mmadParams.k = static_cast<uint16_t>(realL0KAligned);
            mmadParams.n = static_cast<uint16_t>(nL1LenAligned_);
            mmadParams.unit_flag = (kLoopIdx + 1U == kTileCount_ && kL0Idx + 1U == kL0Iter) ?
                                       asc::te::unit_flag_mode::enable_update :
                                       asc::te::unit_flag_mode::enable_keep;
            mmadParams.init_with_zero = (kLoopIdx == 0U && kL0Idx == 0U && !needBias);
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
            l0BufIdx_ ^= 1U;
            kL0Offset += kL0Size;
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
            auto tensorAL0 = MakeMxL0Operand<AType, false>(l0Slot.Addr(), mL1Len_, operandK);
            auto tensorBL0 = MakeMxL0Operand<ConvertedBType, true>(l0Slot.Addr(), operandK, nL1Len_);
            auto tensorBt = MakeMxBiasTable(btSlot.Addr(), nL1Len_);
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
        CopyMxScaleToL0<ScaleAType, false, uint64_t>(sa, l0Slot.Addr(), mL1Len_, cols, offset);
        CopyL0Operand<true>(b0, b1, bK, operandK);
        CopyMxScaleToL0<ScaleBType, true, uint64_t>(sb, l0Slot.Addr(), cols, nL1Len_, offset);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    }

    template <bool IsB, typename Destination, typename Source>
    __aicore__ inline void CopyL0Operand(const Destination& destination, const Source& source, uint64_t kOffset,
                                         uint64_t kSize)
    {
        if constexpr (IsB) {
            CopyMxOperandTile(COPY_L1_TO_L0B_ATOM, destination, source, static_cast<int64_t>(kOffset), int64_t{0L},
                              static_cast<int64_t>(kSize), static_cast<int64_t>(nL1Len_));
        } else {
            CopyMxOperandTile(COPY_L1_TO_L0A_ATOM, destination, source, int64_t{0L}, static_cast<int64_t>(kOffset),
                              static_cast<int64_t>(mL1Len_), static_cast<int64_t>(kSize));
        }
    }

    __aicore__ inline asc::te::mmad_params DeferredMakeMmadParams(uint64_t kLoopIdx, uint64_t kL0Idx, uint64_t kL0Iter,
                                                                  uint64_t realL0K, bool needBias) const
    {
        asc::te::mmad_params mmadParams;
        mmadParams.m = static_cast<uint16_t>(mL1Len_);
        mmadParams.n = static_cast<uint16_t>(nL1Len_);
        UpdateMxMmadParams(mmadParams, static_cast<uint16_t>(realL0K), kLoopIdx == 0U && kL0Idx == 0U,
                           ((kLoopIdx + 1U) * kL1Size_ >= kSize_) && kL0Idx + 1U == kL0Iter, needBias);
        return mmadParams;
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBt>
    __aicore__ inline void DeferredComputeMmad(const asc::te::mmad_params& mmadParams, const TensorL0C& tensorL0C,
                                               const TensorAL0& tensorAL0, const TensorBL0& tensorBL0,
                                               const TensorBt& tensorBt, bool needBias, const BufferSlot& l0Slot,
                                               const BufferSlot& l0cSlot, const BufferSlot& btSlot)
    {
        IssueMxMmad(mmadParams, tensorL0C, tensorAL0, tensorBL0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
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
        uint64_t scaleKOffset = (kOffsetInScaleWindow / MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(
            asc::te::make_coord(0, static_cast<int64_t>(scaleKOffset)),
            asc::te::make_shape(static_cast<int64_t>(mL1Len_), static_cast<int64_t>(realL0ScaleCols)));
        asc::te::copy(COPY_L1_TO_L0SCALE_A_ATOM, tensorScaleAL0, tensorBlockScaleAL1);

        auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn,
                                                         AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(realL0ScaleCols,
                                                                                                  nL1Len_);
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, ScaleBType>(l0Offset >> 4), layoutScaleBL0);
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(
            asc::te::make_coord(static_cast<int64_t>(scaleKOffset), 0),
            asc::te::make_shape(static_cast<int64_t>(realL0ScaleCols), static_cast<int64_t>(nL1Len_)));
        asc::te::copy(COPY_L1_TO_L0SCALE_B_ATOM, tensorScaleBL0, tensorBlockScaleBL1);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyCL0c2Gm(const TensorC& tensorC, const TensorL0C& tensorL0C, const BufferSlot& l0cSlot)
    {
        if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
            asc::te::copy(COPY_L0C_TO_UB_ATOM.with(asc::te::l0c_to_ub_params{asc::te::unit_flag_mode::enable_update}),
                          tensorC, tensorL0C);
        } else {
            constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
            asc::te::copy(COPY_L0C_TO_GM_ATOM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}),
                          tensorC, tensorL0C, FP32_64_AS_UINT64);
        }
    }

    static constexpr uint16_t FINALIZE_ROUTING_OUTPUT_FREE_FLAG = 3U;
    static constexpr uint32_t SCALE_A_LOCK_ID_BASE = QUADRUPLE_BUFFER_COUNT;
    static constexpr uint32_t SCALE_B_LOCK_ID_BASE = SCALE_A_LOCK_ID_BASE + DOUBLE_BUFFER_COUNT;
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
    uint64_t nL1LenAligned_{0};
    uint64_t kTileCount_{0};
    uint64_t l1BufIdx_{0};
    uint64_t scaleBufIdx_{0};
    uint64_t l0BufIdx_{0};
    AscendC::Std::conditional_t<IS_MATMUL, const MatmulParams&, AscendC::Std::tuple<>> params_;
    uint64_t l1BufferNum_{DOUBLE_BUFFER_COUNT};
    uint64_t l1BufferMask_{DOUBLE_BUFFER_COUNT - 1U};
    bool hasBias_{false};
    AscendC::Std::conditional_t<DEFERRED_UB, AscendC::Std::tuple<>, L1Storage> l1Storage_;
    BufferSlot aSlots_[QUADRUPLE_BUFFER_COUNT];
    BufferSlot scaleASlots_[DOUBLE_BUFFER_COUNT];
    BufferSlot scaleBSlots_[DOUBLE_BUFFER_COUNT];
    BufferSlot weightSlots_[QUADRUPLE_BUFFER_COUNT];
    BufferSlot btSlot_;
    BufferSlot l0Slots_[DOUBLE_BUFFER_COUNT];
    BufferSlot l0cSlot_;

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
        // MMAD and FIX both check/update the L0C unit flags, as in the native GMM pipeline.
        // The next MMAD can proceed as soon as its L0C region is released.
        asc::te::copy(COPY_L0C_TO_UB_ATOM.with(asc::te::l0c_to_ub_params{asc::te::unit_flag_mode::enable_update}),
                      output, accumulator);
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
    CopyMxOperandTile(COPY_GM_TO_L1_ATOM, tensorAL1_, tensorA, 0, kaGmOffset, mL1Len_, kaL1RealSize);
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
    CopyMxScalesToL1<uint64_t>(tensorScaleA, tensorScaleBGm_, tensorScaleAL1_, tensorScaleBL1_, mL1Len_, nL1Len_,
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
    // Preserve second-AIV-first ordering with the existing Sync interface.
    Sync::WaitForVector<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + WQGMM_MX_FLAG_ID_MAX);
    Sync::WaitForVector<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetAicToAiv()
{
    Sync::NotifyVector<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + WQGMM_MX_FLAG_ID_MAX);
    Sync::NotifyVector<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
#undef BLAZE_WQGMM_MX_MIX_MMAD_CLASS

#endif

} // namespace Block
} // namespace Gemm
} // namespace Blaze

namespace Blaze {
namespace Gemm {

// Shared AIC/AIV resource profile; mode zero keeps the existing parameter-driven path.
template <uint32_t Mode_, class AType, class BiasType>
struct MxWeightPipelineConfig {
    using ScheduleType = KernelWqgmmMxMix;
    static constexpr auto K_SCHEDULE = WeightPrologueMxKSchedule::DYNAMIC;
    static constexpr auto SCALE_B_SOURCE = WeightPrologueMxScaleBSource::AIV;
    static constexpr uint32_t MODE = Mode_;
    static constexpr uint64_t DEFAULT_SCALE_WINDOW_K = 4096UL;
    static constexpr uint64_t QUAD_SCALE_WINDOW_K = 1024UL;
    static constexpr uint64_t WEIGHT_STAGES = (MODE == static_cast<uint32_t>(MxPipelineMode::WeightQuad)) ?
                                                  QUADRUPLE_BUFFER_COUNT :
                                                  ((MODE == static_cast<uint32_t>(MxPipelineMode::OperandTriple)) ?
                                                       TRIPLE_BUFFER_COUNT :
                                                       DOUBLE_BUFFER_COUNT);
    static constexpr uint64_t SCALE_STAGES = (MODE == static_cast<uint32_t>(MxPipelineMode::WeightQuad)) ?
                                                 QUADRUPLE_BUFFER_COUNT :
                                                 DOUBLE_BUFFER_COUNT;
    static constexpr uint64_t SCALE_WINDOW_K = (MODE == static_cast<uint32_t>(MxPipelineMode::WeightQuad)) ?
                                                   QUAD_SCALE_WINDOW_K :
                                                   DEFAULT_SCALE_WINDOW_K;
    static constexpr uint16_t READY_FLAG = 0;
    static constexpr uint16_t FREE_FLAG = 1;
    __aicore__ static constexpr uint64_t WeightStages(bool hasBias)
    {
        return hasBias ? DOUBLE_BUFFER_COUNT : WEIGHT_STAGES;
    }
    __aicore__ static constexpr uint64_t ScaleWindow(bool hasBias)
    {
        return hasBias ? DEFAULT_SCALE_WINDOW_K : SCALE_WINDOW_K;
    }
    __aicore__ static constexpr uint64_t CreditLimit(uint64_t k, uint64_t kb, bool hasBias)
    {
        if (k == 0 || kb == 0)
            return 0;
        const uint64_t window = ScaleWindow(hasBias);
        const uint64_t tail = k % window;
        const uint64_t shortest = tail == 0 ? window : tail;
        const uint64_t scaleCredits = (hasBias ? DOUBLE_BUFFER_COUNT : SCALE_STAGES) * ((shortest + kb - 1) / kb);
        const uint64_t weightCredits = WeightStages(hasBias);
        return scaleCredits < weightCredits ? scaleCredits : weightCredits;
    }
    struct Consumer {
        static constexpr uint64_t L1_M = 256UL;
        static constexpr uint64_t L1_N = 256UL;
    };
};

namespace Block {
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

#if defined(__NPU_ARCH__)
template <uint32_t Mode_, class PolicyA_, class PolicyBias_, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_,
          class LayoutBTuple_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MxWeightPipelineConfig<Mode_, PolicyA_, PolicyBias_>, ATypeTuple_, LayoutATuple_, BTypeTuple_,
                LayoutBTuple_, CType_, LayoutC_, BiasType_, LayoutBias_> {
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
    using DispatchPolicy = MxWeightPipelineConfig<Mode_, PolicyA_, PolicyBias_>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static_assert(IsFp4<BType>(), "Weight Quant MX expects packed FP4 B");
    static_assert(sizeof(AType) == 1, "Weight Quant MX expects 8-bit A");
    static_assert(IsTrans<LayoutB>::value, "Weight Quant MX expects a transposed B layout");
    static_assert(AscendC::Std::is_same_v<ScaleAType, AscendC::fp8_e8m0_t> &&
                      AscendC::Std::is_same_v<ScaleBType, AscendC::fp8_e8m0_t>,
                  "Weight Quant MX expects E8M0 scales");

    using PipelineConfig = DispatchPolicy;
    template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& tensorA, const TensorScaleA_& scaleA, const TensorScaleB_& scaleB,
                                      const TensorC_& tensorC, bool hasBias)
    {
        if constexpr (!(MODE == MxPipelineMode::ParameterDriven)) {
            const auto plan = MakePipelinePlan(tensorA, tensorC, hasBias);
            const auto& param = plan.block;
            const uint64_t offset = plan.wholeTile ? tileL0cSlot_ * ACCUMULATOR_REGION_BYTES : 0;
            auto ping = MakePipelineAccumulator(plan.leftM, plan.leftN, offset);
            auto pong = MakePipelineAccumulator(plan.rightM, plan.rightN, ACCUMULATOR_REGION_BYTES);
            for (uint64_t k = 0UL; k < param.kSize; k += param.kbL1Size) {
                const uint64_t realK = Min(param.kbL1Size, param.kSize - k);
                LoadPipelineStage(tensorA, scaleA, scaleB, param, k);
                if constexpr (!(MODE == MxPipelineMode::ParameterDriven)) {
                    Sync::WaitForVector<Sync::SYNC_MODE_INTRA, PIPE_MTE1>(SYNC_AIC_AIV_FLAG,
                                                                          !(MODE == MxPipelineMode::Aligned));
                }
                ProcessTileL1(k, realK, param, ping);
                if (plan.splitM || plan.splitN) {
                    ComputePipelineRemainder(plan, k, realK, tensorC, ping, pong);
                }
                ReleasePipelineStage(param, k + param.kbL1Size);
            }
            if (!plan.splitM && !plan.splitN) {
                CopyMxAccumulatorOutput(tensorC, ping);
            }
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (plan.wholeTile) {
                    tileL0cSlot_ ^= 1;
                }
            }
        }
    }

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

    __aicore__ inline explicit BlockMmad(uint64_t l1BufferStage = Mode_)
    {
        static_assert(!(MODE == MxPipelineMode::ParameterDriven),
                      "The original path requires explicit block parameters");
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            l1BufferStage_ = l1BufferStage == TRIPLE_BUFFER_COUNT ? TRIPLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT;
        }
        for (uint64_t i = 0UL; i < DOUBLE_BUFFER_COUNT; i++) {
            SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
            if constexpr (!(MODE == MxPipelineMode::OperandTriple)) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
            }
        }
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            for (uint64_t i = 0UL; i < TRIPLE_BUFFER_COUNT; i++) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
            }
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            AscendC::SetMMLayoutTransform(false);
        } else if constexpr ((MODE == MxPipelineMode::Aligned) || (MODE == MxPipelineMode::OperandTriple)) {
        } else if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            for (uint64_t i = 0UL; i < DOUBLE_BUFFER_COUNT; i++) {
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
            }
        }
    }

    template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& tensorA, const TensorScaleA_& tensorScaleA,
                                      const TensorScaleB_& tensorScaleB, const TensorC_& tensorC)
    {
        RunBlock<false>(tensorA, tensorScaleA, tensorScaleB, tensorC);
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
        RunBlock<true>(tensorA, tensorScaleA, tensorScaleB, tensorC, kaL1Size, kbL1Size, l0KSize);
    }

    static constexpr uint64_t MX_GROUP_SIZE = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;

private:
    static constexpr auto MODE = static_cast<MxPipelineMode>(Mode_);
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
    static constexpr uint64_t ACCUMULATOR_REGION_BYTES = 128UL * 1024UL;
    static constexpr uint64_t CACHE_LINE_BYTES = 128UL;
    static constexpr uint64_t MX_L0_SCALE_ADDRESS_SHIFT = 4UL;

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
                Sync::NotifyVector<SyncProtocol::MODE, PIPE_MTE1>(freeFlag_, true);
            }
        }
        AscendC::SetMMLayoutTransform(true);
    }

    template <bool OverrideK, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void RunBlock(const TensorA& tensorA, const TensorScaleA& tensorScaleA,
                                    const TensorScaleB& tensorScaleB, const TensorC& tensorC, uint64_t kaL1Size = 0UL,
                                    uint64_t kbL1Size = 0UL, uint64_t l0KSize = 0UL)
    {
        if constexpr (OverrideK) {
            kAL1Size_ = kaL1Size;
            kL1Size_ = kbL1Size;
            kL0Size_ = l0KSize;
            l1Storage_.SetKSize(kL1Size_);
        }
        const auto& layoutA = tensorA.layout();
        const auto& layoutC = tensorC.layout();
        kSize_ = static_cast<uint64_t>(asc::te::get_total_column_shape(layoutA));
        mL1Len_ = static_cast<uint64_t>(asc::te::get_total_row_shape(layoutC));
        nL1Len_ = static_cast<uint64_t>(asc::te::get_total_column_shape(layoutC));
        kTileCount_ = CeilDiv(kSize_, kL1Size_);
        ProcessBlock(tensorA, tensorScaleA, tensorScaleB, tensorC);
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

    __aicore__ static inline uint64_t CalcMxScaleWindowCols(int64_t kL1Offset, uint64_t kSize, uint64_t scaleKL1Size)
    {
        int64_t totalScaleCols = static_cast<int64_t>(Align64(kSize) / (MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE));
        int64_t scaleKOffset = kL1Offset / static_cast<int64_t>((MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE));
        if (scaleKOffset >= totalScaleCols) {
            return 0UL;
        }
        int64_t scaleWindowKLen = Min(static_cast<int64_t>(kSize) - kL1Offset, static_cast<int64_t>(scaleKL1Size));
        int64_t scaleWindowCols = static_cast<int64_t>(Align64(scaleWindowKLen) /
                                                       (MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE));
        return Min(scaleWindowCols, totalScaleCols - scaleKOffset);
    }

    template <typename TensorA, typename TensorAL1>
    __aicore__ static inline void CopyMxAGmToL1(const TensorA& tensorA, const TensorAL1& tensorAL1, int64_t kL1Offset,
                                                int64_t kL1Len, uint64_t mL1Len)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kL1Offset),
                                      asc::te::make_shape(static_cast<int64_t>(mL1Len), kL1Len));
        asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);
        Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
    }

    template <typename TensorBL1>
    __aicore__ static inline void ClearMxConvertedWeightL1Tail(const TensorBL1& tensorBL1, uint64_t kL1Len,
                                                               uint64_t nL1Len)
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
            asc::te::make_shape(static_cast<int64_t>(kMmadLen - kPhysicalLen), static_cast<int64_t>(Align16(nL1Len))));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(tailTensor, 1U, nL1Len, 0U);
    }

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

    template <bool TailWeightLockOnly, typename TensorL0C, typename TensorAL1, typename TensorBL1,
              typename TensorScaleAL1, typename TensorScaleBL1, typename TensorBiasL1>
    __aicore__ inline void IterateMatmul(uint64_t kLoopIdx, const TensorL0C& tensorL0C, const TensorAL1& tensorAL1,
                                         const TensorBL1& tensorBL1, const TensorScaleAL1& tensorScaleAL1,
                                         const TensorScaleBL1& tensorScaleBL1, const TensorBiasL1& tensorBiasL1,
                                         uint64_t kL1Len, uint64_t kOffsetInAWindow, uint64_t kOffsetInScaleWindow,
                                         const BufferSlot& l1DataSlot, const BufferSlot& scaleASlot,
                                         const BufferSlot& scaleBSlot, const BufferSlot& weightSlot,
                                         const BufferSlot& l0cSlot)
    {
        uint64_t kL0Iter = CeilDiv(kL1Len, kL0Size_);
        for (uint64_t kL0Idx = 0UL; kL0Idx < kL0Iter; ++kL0Idx) {
            uint64_t kL0Offset = kL0Idx * kL0Size_;
            uint64_t realL0K = Min(kL1Len - kL0Offset, kL0Size_);
            uint64_t realL0KAligned = Align64(realL0K);
            uint64_t l0BufIdx = l0BufIdx_ & 1U;
            const BufferSlot l0Slot{bufferManager_.GetL0Slot(l0BufIdx).Addr(),
                                    static_cast<uint8_t>(L0_BUFFER_ID_BASE + l0BufIdx)};
            const BufferSlot btSlot{bufferManager_.GetBTSlot(0U).Addr(), BT_BUFFER_ID};
            uint64_t l0Offset = l0Slot.Addr();
            auto tensorAL0 = MakeMxL0Operand<AType, false>(l0Offset, mL1Len_, realL0KAligned);
            auto tensorBL0 = MakeMxL0Operand<ConvertedBType, true>(l0Offset, realL0KAligned, nL1Len_);
            auto tensorBt = MakeMxBiasTable(btSlot.Addr(), nL1Len_);
            bool needBias = hasBias_ && kLoopIdx == 0U && kL0Idx == 0U;
            {
                auto l1DataLock = l1DataSlot.LockMte1();
                auto scaleALock = scaleASlot.LockMte1();
                auto scaleBLock = scaleBSlot.LockMte1();
                auto l0Lock = l0Slot.LockMte1();
                CopyMxOperandTile(COPY_L1_TO_L0A_ATOM, tensorAL0, tensorAL1, 0,
                                  static_cast<int64_t>(kOffsetInAWindow + kL0Offset), static_cast<int64_t>(mL1Len_),
                                  static_cast<int64_t>(realL0KAligned));
                const auto copyWeight = [&]() __aicore__ {
                    CopyMxOperandTile(COPY_L1_TO_L0B_ATOM, tensorBL0, tensorBL1, static_cast<int64_t>(kL0Offset), 0,
                                      static_cast<int64_t>(realL0KAligned), static_cast<int64_t>(nL1Len_));
                };
                if (!TailWeightLockOnly || Align32(kL1Len) != Align64(kL1Len)) {
                    auto weightLock = weightSlot.LockMte1();
                    copyWeight();
                } else {
                    copyWeight();
                }
                CopyScaleL1ToL0(tensorScaleAL1, tensorScaleBL1, kOffsetInScaleWindow + kL0Offset, realL0K, l0Offset);
                if (needBias) {
                    CopyMxBiasL1ToL0(tensorBt, tensorBiasL1, btSlot);
                }
            }
            auto mmadParams = MakeMmadParams(kLoopIdx, kL0Idx, kL0Iter, realL0K, needBias);
            ComputeMxMmad(mmadParams, tensorL0C, tensorAL0, tensorBL0, tensorBt, needBias, l0Slot, l0cSlot, btSlot);
            l0BufIdx_ ^= 1U;
        }
    }

    __aicore__ inline asc::te::mmad_params MakeMmadParams(uint64_t kLoopIdx, uint64_t kL0Idx, uint64_t kL0Iter,
                                                          uint64_t realL0K, bool needBias)
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

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename TensorBt>
    __aicore__ static inline void ComputeMxMmad(const asc::te::mmad_params& mmadParams, const TensorL0C& tensorL0C,
                                                const TensorAL0& tensorAL0, const TensorBL0& tensorBL0,
                                                const TensorBt& tensorBt, bool needBias, const BufferSlot& l0Slot,
                                                const BufferSlot& l0cSlot, const BufferSlot& btSlot)
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

    template <typename T, bool OperandB>
    __aicore__ static inline auto MakeMxL0Operand(uint64_t offset, uint64_t rows, uint64_t columns)
    {
        using Layout = AscendC::Std::conditional_t<OperandB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0b, asc::te::location::l0a>;
        auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<asc::te::c0_element<T>>>(rows, columns);
        return asc::te::make_tensor(asc::te::make_mem_ptr<Location, T>(offset), layout);
    }

    template <typename Source, typename Destination, typename Copy, typename Row, typename Column, typename Rows,
              typename Columns>
    __aicore__ static inline void CopyMxOperandTile(const Copy& copy, const Destination& destination,
                                                    const Source& source, Row row, Column column, Rows rows,
                                                    Columns columns)
    {
        auto tile = source.slice(asc::te::make_coord(row, column), asc::te::make_shape(rows, columns));
        asc::te::copy(copy, destination, tile);
    }

    __aicore__ static inline auto MakeMxBiasTable(uint64_t offset, uint64_t columns)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1L),
                                                                             static_cast<int64_t>(Align16(columns)));
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(offset), layout);
    }

    template <typename C, typename A, typename B>
    __aicore__ static inline void IssueMxMmad(const asc::te::mmad_params& params, const C& c, const A& a, const B& b)
    {
        auto atom = asc::te::make_mmad(asc::te::mmad_operation{}, Tile::MmadTraitMX{}).with(params);
        asc::te::mmad(atom, c, a, b);
    }

    __aicore__ static inline void UpdateMxMmadParams(asc::te::mmad_params& params, uint16_t k, bool firstK, bool lastK,
                                                     bool needBias)
    {
        params.k = k;
        params.unit_flag = lastK ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = firstK && !needBias;
    }

    template <typename TensorBt, typename TensorBiasL1>
    __aicore__ static inline void CopyMxBiasL1ToL0(const TensorBt& tensorBt, const TensorBiasL1& tensorBiasL1,
                                                   const BufferSlot& btSlot)
    {
        auto btLock = btSlot.LockMte1();
        auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
        asc::te::copy(copyL12BT, tensorBt, tensorBiasL1);
    }

    template <typename Scale, bool OperandB, typename Index = int64_t, typename Tensor>
    __aicore__ static inline void CopyMxScaleToL0(const Tensor& source, uint64_t l0Offset, uint64_t rows,
                                                  uint64_t columns, uint64_t kOffset, uint64_t outerOffset = 0)
    {
        using Layout = AscendC::Std::conditional_t<OperandB, asc::te::nn_layout_ptn, asc::te::zz_layout_ptn>;
        using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0scaleb,
                                                     asc::te::location::l0scalea>;
        using Copy = AscendC::Std::conditional_t<OperandB, asc::te::copy_l1_to_l0scaleb, asc::te::copy_l1_to_l0scalea>;
        auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(rows, columns);
        auto destination = asc::te::make_tensor(
            asc::te::make_mem_ptr<Location, Scale>(l0Offset >> MX_L0_SCALE_ADDRESS_SHIFT), layout);
        const Index offset = static_cast<Index>(kOffset);
        if constexpr (OperandB) {
            CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, offset, static_cast<Index>(outerOffset),
                              static_cast<Index>(rows), static_cast<Index>(columns));
        } else {
            CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, static_cast<Index>(outerOffset), offset,
                              static_cast<Index>(rows), static_cast<Index>(columns));
        }
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void CopyScaleL1ToL0(const TensorScaleAL1& tensorScaleAL1, const TensorScaleBL1& tensorScaleBL1,
                                           uint64_t kOffsetInScaleWindow, uint64_t realL0K, uint64_t l0Offset)
    {
        uint64_t realL0ScaleCols = CeilDiv(realL0K, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t scaleKOffset = (kOffsetInScaleWindow / MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        CopyMxScaleToL0<ScaleAType, false>(tensorScaleAL1, l0Offset, mL1Len_, realL0ScaleCols, scaleKOffset);
        CopyMxScaleToL0<ScaleBType, true>(tensorScaleBL1, l0Offset, realL0ScaleCols, nL1Len_, scaleKOffset);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ static inline void CopyMxAccumulatorOutput(const TensorC& tensorC, const TensorL0C& tensorL0C)
    {
        if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
            auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, Tile::CopyL0C2UBTraitMixSplitM{});
            asc::te::copy(copyL0C2UB.with(asc::te::l0c_to_ub_params{asc::te::unit_flag_mode::enable_update}), tensorC,
                          tensorL0C);
        } else {
            constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000UL;
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), tensorC,
                          tensorL0C, FP32_64_AS_UINT64);
        }
    }

    __aicore__ inline void PostProcess(bool releaseScaleBuffer, uint64_t l1BufIdx)
    {
        Sync::NotifyVector<SyncProtocol::MODE, PIPE_MTE1>(freeFlag_, true);
        l1BufIdx_ = (l1BufIdx + 1U) & l1BufferMask_;
        if (releaseScaleBuffer) {
            scaleBufIdx_ ^= 1U;
        }
    }

    struct OriginalWindow {
        uint64_t scaleIteration = 0UL;
        int64_t aStart = 0L;
        int64_t aLength = 0L;
        uint64_t scaleColumns = 0UL;
    };
    struct OriginalStage {
        uint64_t weightSlot;
        uint64_t scaleSlot;
        int64_t kOffset;
        int64_t kLength;
    };
    struct OriginalSlots {
        const BufferSlot& a;
        const BufferSlot& scaleA;
        const BufferSlot& scaleB;
        const BufferSlot& weight;
    };

    __aicore__ inline OriginalStage MakeOriginalStage(uint64_t index, OriginalWindow& window)
    {
        const int64_t k = static_cast<int64_t>(index) * static_cast<int64_t>(kL1Size_);
        if (k % static_cast<int64_t>(kAL1Size_) == 0) {
            window.aStart = k;
            window.aLength = Min(static_cast<int64_t>(kSize_) - k, static_cast<int64_t>(kAL1Size_));
        }
        if (window.scaleIteration == 0) {
            window.scaleColumns = CalcMxScaleWindowCols(k, kSize_, scaleKL1Size_);
        }
        return {l1BufIdx_, scaleBufIdx_, k, Min(static_cast<int64_t>(kSize_) - k, static_cast<int64_t>(kL1Size_))};
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorAL1, typename TensorBL1,
              typename TensorScaleAL1, typename TensorScaleBL1>
    __aicore__ inline void LoadOriginalStage(const TensorA& a, const TensorScaleA& scaleA, const TensorScaleB& scaleB,
                                             const TensorAL1& aL1, const TensorBL1& bL1, const TensorScaleAL1& scaleAL1,
                                             const TensorScaleBL1& scaleBL1, const OriginalStage& stage,
                                             const OriginalWindow& window, const OriginalSlots& slots)
    {
        if (window.scaleIteration == 0) {
            auto lockA = slots.scaleA.LockMte2();
            auto lockB = slots.scaleB.LockMte2();
            CopyMxScaleGmToL1(scaleA, scaleB, scaleAL1, scaleBL1, stage.kOffset / static_cast<int64_t>(MX_GROUP_SIZE),
                              window.scaleColumns);
        }
        if (stage.kOffset == window.aStart) {
            auto lock = slots.a.LockMte2();
            CopyMxAGmToL1(a, aL1, window.aStart, window.aLength, mL1Len_);
        }
        if (Align32(stage.kLength) != Align64(stage.kLength)) {
            auto lock = slots.weight.LockMte2();
            ClearMxConvertedWeightL1Tail(bL1, stage.kLength, nL1Len_);
        }
        Sync::WaitForVector<SyncProtocol::MODE, PIPE_MTE1>(readyFlag_, true);
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorL0C>
    __aicore__ inline void ComputeOriginalStage(uint64_t index, OriginalWindow& window, const TensorA& a,
                                                const TensorScaleA& scaleA, const TensorScaleB& scaleB,
                                                const TensorL0C& c, const BufferSlot& cSlot)
    {
        const auto stage = MakeOriginalStage(index, window);
        auto aL1 = l1Storage_.MakeATensor(aL1BufIdx_, mL1Len_, Align64(window.aLength));
        auto bL1 = l1Storage_.MakeWeightTensor(stage.weightSlot, 0, 0, Align16(nL1Len_), nL1Len_,
                                               Align64(stage.kLength));
        auto scaleAL1 = l1Storage_.MakeScaleATensor(stage.scaleSlot, mL1Len_, window.scaleColumns);
        auto scaleBL1 = l1Storage_.MakeScaleBTensor(stage.scaleSlot, window.scaleColumns, nL1Len_);
        auto biasL1 = l1Storage_.MakeBiasTensor(stage.weightSlot, nL1Len_);
        const BufferSlot weightSlot{0, static_cast<uint8_t>(WEIGHT_BUFFER_ID_BASE + stage.weightSlot)};
        OriginalSlots slots{bufferManager_.GetL1ASlot(aL1BufIdx_),
                            bufferManager_.GetL1BSlot(SCALE_A_SLOT_BASE + stage.scaleSlot),
                            bufferManager_.GetL1BSlot(SCALE_B_SLOT_BASE + stage.scaleSlot), weightSlot};
        LoadOriginalStage(a, scaleA, scaleB, aL1, bL1, scaleAL1, scaleBL1, stage, window, slots);
        IterateMatmul<true>(index, c, aL1, bL1, scaleAL1, scaleBL1, biasL1, stage.kLength,
                            stage.kOffset - window.aStart, window.scaleIteration * kL1Size_, slots.a, slots.scaleA,
                            slots.scaleB, slots.weight, cSlot);
        const bool release = (window.scaleIteration + 1U) * kL1Size_ >= scaleKL1Size_ || index + 1U == kTileCount_;
        if (stage.kOffset + stage.kLength >= window.aStart + window.aLength) {
            aL1BufIdx_ = (aL1BufIdx_ + 1U) & l1BufferMask_;
        }
        PostProcess(release, stage.weightSlot);
        window.scaleIteration = release ? 0U : window.scaleIteration + 1U;
    }

    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void ProcessBlock(const TensorA& a, const TensorScaleA& scaleA, const TensorScaleB& scaleB,
                                        const TensorC& c)
    {
        const BufferSlot slot{bufferManager_.GetL0CSlot(0U).Addr(), L0C_BUFFER_ID};
        auto accumulator = MakePipelineAccumulator(mL1Len_, nL1Len_, slot.Addr());
        OriginalWindow window{};
        for (uint64_t index = 0UL; index < kTileCount_; ++index) {
            ComputeOriginalStage(index, window, a, scaleA, scaleB, accumulator, slot);
        }
        if constexpr (AscendC::Std::is_same_v<DispatchPolicy, GroupedMatmulFinalizeRoutingWithWeightQuantMx>) {
            // Wait for both AIVs before reusing the shared W8/output arena.
            Sync::WaitForVector<SyncProtocol::MODE, PIPE_FIX>(FINALIZE_ROUTING_OUTPUT_FREE_FLAG, true);
        }
        CopyMxAccumulatorOutput(c, accumulator);
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

    static constexpr uint16_t DEFAULT_READY_FLAG = 6U;
    static constexpr uint16_t DEFAULT_FREE_FLAG = 8U;
    uint16_t readyFlag_{DEFAULT_READY_FLAG};
    uint16_t freeFlag_{DEFAULT_FREE_FLAG};
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
    uint64_t problemM_{0UL};
    L1Storage l1Storage_;
    BufferManager<QUADRUPLE_BUFFER_COUNT, QUADRUPLE_BUFFER_COUNT, DOUBLE_BUFFER_COUNT> bufferManager_;

private:
    struct BlockMmadOffsetParam {
        uint64_t mL1Size;
        uint64_t kaL1Size;
        uint64_t kbL1Size;
        uint64_t l0KSize;
        uint64_t nL1Size;
        uint64_t kSize;
    };

    struct PipelinePlan {
        BlockMmadOffsetParam block{};
        uint64_t leftM;
        uint64_t rightM;
        uint64_t leftN;
        uint64_t rightN;
        bool splitM = false;
        bool splitN = false;
        bool wholeTile = false;
    };

    __aicore__ inline void ConfigureFullLoadA(BlockMmadOffsetParam& param)
    {
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            const uint64_t alignedM = CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
            fullLoadA_ = !hasBias_ && alignedM != 0 && param.kSize != 0 && param.kSize % L1_K_CONFIG_256 == 0 &&
                         param.kSize <= PipelineConfig::SCALE_WINDOW_K &&
                         param.kSize <= (QUAD_A_STRIDE * DOUBLE_BUFFER_COUNT / sizeof(AType)) / alignedM;
            if (fullLoadA_) {
                param.kaL1Size = param.kSize;
                for (uint64_t i = 0UL; i < DOUBLE_BUFFER_COUNT; ++i) {
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
                }
            }
        }
    }

    template <typename TensorA_, typename TensorC_>
    __aicore__ inline PipelinePlan MakePipelinePlan(const TensorA_& tensorA, const TensorC_& tensorC, bool hasBias)
    {
        hasBias_ = hasBias;
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            if (hasBias_) {
                l1BufferStage_ = DOUBLE_BUFFER_COUNT;
            }
        }
        PipelinePlan plan{};
        auto& p = plan.block;
        p.mL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(tensorC.layout());
        p.kSize = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorA.layout());
        p.nL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorC.layout());
        CalcDynamicKBlock(p.mL1Size, p.nL1Size, p.kaL1Size, p.kbL1Size);
        constexpr uint64_t halfK = L1_K_CONFIG_256 / DOUBLE_BUFFER_COUNT;
        p.l0KSize = p.mL1Size <= halfK && p.nL1Size <= halfK ? L1_K_CONFIG_256 : halfK;
        ConfigureFullLoadA(p);
        ConfigureOutputSplit(plan);
        plan.leftM = plan.splitM ? CeilAlign(p.mL1Size / DOUBLE_BUFFER_COUNT, static_cast<uint64_t>(BLOCK_CUBE)) :
                                   p.mL1Size;
        plan.rightM = plan.splitM ? p.mL1Size - plan.leftM : p.mL1Size;
        plan.leftN = plan.splitN ? CeilAlign(p.nL1Size / DOUBLE_BUFFER_COUNT, static_cast<uint64_t>(BLOCK_CUBE)) :
                                   p.nL1Size;
        plan.rightN = plan.splitN ? p.nL1Size - plan.leftN : p.nL1Size;
        return plan;
    }

    __aicore__ inline void ConfigureOutputSplit(PipelinePlan& plan) const
    {
        const auto& p = plan.block;
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            plan.wholeTile = !hasBias_ && CeilAlign(p.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                                  CeilAlign(p.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                                  sizeof(float) <=
                                              ACCUMULATOR_REGION_BYTES;
            plan.splitM = !plan.wholeTile && !hasBias_ && p.mL1Size > MX_GROUP_SIZE && p.mL1Size <= L1_M &&
                          p.nL1Size <= L1_N && p.kbL1Size == L1_K_CONFIG_256 && p.kSize % L1_K_CONFIG_256 == 0;
        } else if constexpr ((MODE == MxPipelineMode::Aligned)) {
            plan.splitN = !hasBias_ && p.mL1Size == L1_M && p.nL1Size == L1_N && p.kbL1Size == L1_K_CONFIG_256 &&
                          p.kSize % L1_K_CONFIG_256 == 0;
        } else {
            plan.splitN = !hasBias_ && p.mL1Size > 0 && p.mL1Size < L1_M && p.nL1Size >= MX_GROUP_SIZE &&
                          p.nL1Size <= L1_N && p.nL1Size % BLOCK_CUBE == 0;
        }
    }

    __aicore__ inline auto MakePipelineAccumulator(uint64_t rows, uint64_t columns, uint64_t offset) const
    {
        auto layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(rows, columns);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(offset), layout);
    }

    template <typename TensorA_, typename ScaleA_, typename ScaleB_>
    __aicore__ inline void LoadPipelineStage(const TensorA_& a, const ScaleA_& scaleA, const ScaleB_& scaleB,
                                             const BlockMmadOffsetParam& p, uint64_t offset)
    {
        const uint64_t window = PipelineConfig::ScaleWindow(hasBias_);
        if (offset % window == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleL1BufIdx_);
            CopyMxScaleGmToL1(scaleA, p, offset);
            if constexpr ((MODE == MxPipelineMode::Aligned)) {
                const uint64_t realScale = Min(window, p.kSize - offset) / MX_GROUP_SIZE;
                auto slice = scaleB.slice(asc::te::make_coord(offset / MX_GROUP_SIZE, 0),
                                          asc::te::make_shape(realScale, p.nL1Size));
                asc::te::copy(COPY_GM_TO_L1_ATOM, tensorScaleBL1_, slice);
            }
        }
        if (offset % p.kaL1Size == 0) {
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (!fullLoadA_) {
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
                }
            } else {
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
            }
            CopyAGmToL1(a, p, offset);
        }
    }

    template <typename TensorC_, typename TensorL0C_, typename TensorPong_>
    __aicore__ inline void ComputePipelineRemainder(const PipelinePlan& plan, uint64_t k, uint64_t realK,
                                                    const TensorC_& c, const TensorL0C_& ping, const TensorPong_& pong)
    {
        const bool last = k + realK == plan.block.kSize;
        if (last) {
            auto left = c.slice(asc::te::make_coord(0, 0), asc::te::make_shape(plan.leftM, plan.leftN));
            CopyMxAccumulatorOutput(left, ping);
        }
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            ProcessTileL1(k, realK, plan.block, pong, plan.leftM);
        } else {
            ProcessTileL1(k, realK, plan.block, pong, plan.leftN, 0);
        }
        if (last) {
            auto right = c.slice(asc::te::make_coord(plan.splitM ? plan.leftM : 0, plan.splitN ? plan.leftN : 0),
                                 asc::te::make_shape(plan.rightM, plan.rightN));
            CopyMxAccumulatorOutput(right, pong);
        }
    }

    __aicore__ inline void ReleasePipelineStage(const BlockMmadOffsetParam& p, uint64_t next)
    {
        if (next % p.kaL1Size == 0 || next >= p.kSize) {
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (fullLoadA_) {
                    for (uint64_t i = 0UL; i < DOUBLE_BUFFER_COUNT; ++i) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
                    }
                } else {
                    SetMTE1ToMTE2();
                }
            } else {
                SetMTE1ToMTE2();
            }
        }
        if (next % PipelineConfig::ScaleWindow(hasBias_) == 0 || next >= p.kSize) {
            SetScaleMTE1ToMTE2();
        }
        if constexpr (!(MODE == MxPipelineMode::ParameterDriven)) {
            Sync::NotifyVector<Sync::SYNC_MODE_INTRA, PIPE_MTE1>(SYNC_AIV_AIC_FLAG, !(MODE == MxPipelineMode::Aligned));
        }
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            bL1BufIdx_ = (l1BufferStage_ == TRIPLE_BUFFER_COUNT ? (bL1BufIdx_ + 1UL) % TRIPLE_BUFFER_COUNT :
                                                                  ((bL1BufIdx_ + 1UL) & 1UL));
        } else if constexpr ((MODE == MxPipelineMode::Aligned)) {
            bL1BufIdx_ ^= 1;
        } else {
            bL1BufIdx_ = (bL1BufIdx_ + 1) & ((PipelineConfig::WeightStages(hasBias_)) - 1);
        }
    }

    static constexpr uint64_t WQGMM_MX_GROUP_SIZE = MX_GROUP_SIZE;

    static constexpr uint64_t QUAD_B_STRIDE = 65536UL;
    static constexpr uint64_t QUAD_A_BASE = 311296UL;
    static constexpr uint64_t QUAD_A_STRIDE = 106496UL;
    static constexpr uint64_t QUAD_SCALE_A_BASE = 262144UL;
    static constexpr uint64_t QUAD_SCALE_B_BASE = 278528UL;
    static constexpr uint64_t QUAD_SCALE_STRIDE = 8192UL;
    static constexpr uint64_t KB = 1024UL;
    static constexpr uint64_t BIAS_L1_BYTES = 4UL * KB;
    static constexpr uint64_t DOUBLE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {128UL * KB, 256UL * KB};
    static constexpr uint64_t DOUBLE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {0UL, 384UL * KB};
    static constexpr uint64_t DOUBLE_SCALE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64UL * KB, 448UL * KB};
    static constexpr uint64_t DOUBLE_SCALE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {96UL * KB, 480UL * KB};
    static constexpr uint64_t DOUBLE_BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64UL * KB, 380UL * KB};
    static constexpr uint64_t TRIPLE_A_L1_OFFSETS[TRIPLE_BUFFER_COUNT] = {0UL, 256UL * KB, 192UL * KB};
    static constexpr uint64_t TRIPLE_B_L1_OFFSETS[TRIPLE_BUFFER_COUNT] = {64UL * KB, 320UL * KB, 448UL * KB};
    static constexpr uint64_t TRIPLE_SCALE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {128UL * KB, 384UL * KB};
    static constexpr uint64_t TRIPLE_SCALE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {160UL * KB, 416UL * KB};

    constexpr static int32_t C0_SIZE = asc::te::c0_element<AType>;
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;
    static constexpr uint64_t L1_M = 256UL;
    static constexpr uint64_t L1_N = 256UL;
    static constexpr uint64_t L1_K_CONFIG_512 = 512UL;
    static constexpr uint64_t L1_K_CONFIG_256 = 256UL;
    static constexpr uint64_t MX_SCALE_K_L1_SIZE = (MODE == MxPipelineMode::WeightQuad) ?
                                                       PipelineConfig::DEFAULT_SCALE_WINDOW_K :
                                                       PipelineConfig::SCALE_WINDOW_K;
    static constexpr uint64_t L1_K_DYNAMIC_CONFIG_N_THRESHOLD = L1_N >> 1UL;
    static constexpr uint64_t A_L1_MODEL_SINGLE_BUF_SIZE = 80UL * 1024UL;
    static constexpr uint64_t A_B_LOAD_BALANCE_FACTOR = 2UL;
    static constexpr uint64_t L1_K_DYNAMIC_CONFIG_M_THRESHOLD = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType) /
                                                                 L1_K_CONFIG_512 / AscendC::BLOCK_CUBE) *
                                                                AscendC::BLOCK_CUBE;
    static constexpr AscendC::TEventID EVENT_IDS_MTE1_TO_MTE2 = 0;
    static constexpr AscendC::TEventID EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 = ((MODE == MxPipelineMode::Aligned) ||
                                                                          (MODE == MxPipelineMode::WeightQuad)) ?
                                                                             (2) :
                                                                             (3);
    static constexpr AscendC::TEventID EVENT_ID_MTE1_TO_MTE2 = ((MODE == MxPipelineMode::Aligned) ||
                                                                (MODE == MxPipelineMode::WeightQuad)) ?
                                                                   (4) :
                                                                   (5);
    static constexpr AscendC::TEventID EVENT_ID_MTE2_TO_MTE1 = 0;
    static constexpr AscendC::TEventID EVENT_ID_M_TO_MTE1 = 3;
    static constexpr AscendC::TEventID EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = PipelineConfig::READY_FLAG;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = PipelineConfig::FREE_FLAG;
    static constexpr uint64_t SCALE_AL1_OFFSET = DOUBLE_SCALE_A_L1_OFFSETS[0UL];
    static constexpr uint64_t SCALE_BL1_OFFSET = DOUBLE_SCALE_B_L1_OFFSETS[0UL];
    static constexpr uint64_t A_L1_OFFSET = DOUBLE_A_L1_OFFSETS[0UL];
    static constexpr uint64_t L1_BUF_OFFSET = DOUBLE_SCALE_A_L1_OFFSETS[1UL] - SCALE_AL1_OFFSET;
    static constexpr uint64_t A_L1_BUF_OFFSET = DOUBLE_A_L1_OFFSETS[1UL] - A_L1_OFFSET;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128UL * 1024UL;
    static constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = BIAS_L1_BYTES;
    static constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {DOUBLE_BIAS_L1_OFFSETS[0UL],
                                                                      DOUBLE_BIAS_L1_OFFSETS[1UL]};
    static constexpr uint64_t BIAS_TABLE_SLOT_SIZE = 2UL * 256UL * sizeof(float);
    static constexpr uint64_t L0_BUF_OFFSET = 32UL * 1024UL;
    static constexpr auto COPY_GM_TO_L1_ATOM = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    static constexpr auto COPY_L1_TO_L0A_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
    static constexpr auto COPY_L1_TO_L0B_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
    static constexpr auto COPY_L1_TO_BT_ATOM = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
    static constexpr auto COPY_L0C_TO_GM_ATOM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 = typename asc::te::frame_layout_format<asc::te::zz_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 = typename asc::te::frame_layout_format<asc::te::nn_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;

    using TensorAL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(0UL),
        MakeLayoutAL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));
    using TensorScaleAL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, ScaleAType>(0UL),
        MakeLayoutScaleAL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));
    using TensorScaleBL1 = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, ScaleBType>(0UL),
        MakeLayoutScaleBL1{}(static_cast<uint64_t>(BLOCK_CUBE), static_cast<uint64_t>(BLOCK_CUBE))));
    uint64_t bL1BufIdx_ = 0UL;
    uint64_t scaleL1BufIdx_ = 0UL;
    TensorAL1 tensorAL1_;
    TensorScaleAL1 tensorScaleAL1_;
    TensorScaleBL1 tensorScaleBL1_;
    uint64_t scaleBL1Seq_ = 0UL;
    uint64_t tileL0cSlot_ = 0UL;
    bool fullLoadA_ = false;
    uint64_t l1BufferStage_{DOUBLE_BUFFER_COUNT};

    __aicore__ inline uint64_t GetAL1Offset(uint64_t bufferId) const
    {
        uint64_t offset = l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_A_L1_OFFSETS[bufferId] :
                                                                  DOUBLE_A_L1_OFFSETS[bufferId];
        return offset +
               (l1BufferStage_ == DOUBLE_BUFFER_COUNT && hasBias_ && bufferId == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0UL);
    }

    __aicore__ inline uint64_t GetScaleAL1Offset(uint64_t bufferId) const
    {
        uint64_t offset = l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_SCALE_A_L1_OFFSETS[bufferId] :
                                                                  DOUBLE_SCALE_A_L1_OFFSETS[bufferId];
        return offset +
               (l1BufferStage_ == DOUBLE_BUFFER_COUNT && hasBias_ && bufferId == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0UL);
    }
    __aicore__ inline uint64_t GetScaleBL1Offset(uint64_t bufferId) const
    {
        uint64_t offset = l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_SCALE_B_L1_OFFSETS[bufferId] :
                                                                  DOUBLE_SCALE_B_L1_OFFSETS[bufferId];
        return offset +
               (l1BufferStage_ == DOUBLE_BUFFER_COUNT && hasBias_ && bufferId == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0UL);
    }

    struct L0SlicePlan {
        uint64_t m;
        uint64_t n;
        uint64_t mOffset = 0UL;
        uint64_t nOffset = 0UL;
        bool reuseA = false;
        bool reuseB = false;
        bool copyA = true;
        bool copyB = true;
        bool waitBefore = true;
        bool releaseAfter = true;
    };

    __aicore__ inline void ConfigureAReuse(L0SlicePlan& plan, const BlockMmadOffsetParam& p, uint64_t realK) const
    {
        constexpr uint64_t halfK = L1_K_CONFIG_256 / DOUBLE_BUFFER_COUNT;
        if constexpr ((MODE == MxPipelineMode::Aligned)) {
            plan.reuseA = !hasBias_ && plan.m == L1_M && plan.n == halfK && p.nL1Size == L1_N && p.l0KSize == halfK;
            plan.copyA = plan.nOffset == 0;
            plan.copyB = plan.mOffset == 0;
        } else {
            plan.reuseA = !hasBias_ && p.mL1Size > 0 && p.mL1Size <= L1_M && plan.n <= halfK &&
                          p.nL1Size == DOUBLE_BUFFER_COUNT * plan.n && p.l0KSize == halfK && realK == L1_K_CONFIG_256;
            plan.copyA = !plan.reuseA || plan.nOffset == 0;
        }
        plan.waitBefore = !plan.reuseA || plan.nOffset == 0;
        plan.releaseAfter = !plan.reuseA || plan.nOffset + plan.n == p.nL1Size;
    }

    __aicore__ inline void ConfigureBReuse(L0SlicePlan& plan, const BlockMmadOffsetParam& p, uint64_t realK,
                                           uint64_t storageM) const
    {
        constexpr uint64_t halfK = L1_K_CONFIG_256 / DOUBLE_BUFFER_COUNT;
        const uint64_t alignment = BLOCK_CUBE;
        plan.m = plan.mOffset != 0 ? p.mL1Size - plan.mOffset :
                                     (storageM < CeilAlign(p.mL1Size, alignment) ?
                                          CeilAlign(p.mL1Size / DOUBLE_BUFFER_COUNT, alignment) :
                                          p.mL1Size);
        plan.n = p.nL1Size;
        plan.reuseB = !hasBias_ && plan.m < p.mL1Size && plan.m <= halfK && p.l0KSize == halfK &&
                      realK == L1_K_CONFIG_256;
        plan.copyB = !plan.reuseB || plan.mOffset == 0;
        plan.waitBefore = plan.copyB;
        plan.releaseAfter = !plan.reuseB || plan.mOffset + plan.m == p.mL1Size;
    }

    template <typename TensorL0C_>
    __aicore__ inline L0SlicePlan MakeL0SlicePlan(const BlockMmadOffsetParam& p, uint64_t realK, const TensorL0C_& l0c,
                                                  uint64_t offset0, uint64_t offset1) const
    {
        L0SlicePlan plan{};
        plan.m = (MODE == MxPipelineMode::OperandTriple) ? p.mL1Size : asc::te::get_total_row_shape(l0c.layout());
        plan.n = asc::te::get_total_column_shape(l0c.layout());
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            plan.mOffset = offset0;
            ConfigureBReuse(plan, p, realK, asc::te::get_total_row_shape(l0c.layout()));
        } else {
            plan.nOffset = offset0;
            plan.mOffset = (MODE == MxPipelineMode::Aligned) ? offset1 : 0;
            ConfigureAReuse(plan, p, realK);
        }
        return plan;
    }

    __aicore__ inline uint64_t PipelineOperandOffset(bool split, bool second) const
    {
        const uint64_t stride = split ? L0_BUF_OFFSET / DOUBLE_BUFFER_COUNT : L0_BUF_OFFSET;
        return l0BufIdx_ * stride + (split && second ? L0_BUF_OFFSET : 0);
    }

    __aicore__ inline auto LoadPipelineOperandA(const L0SlicePlan& plan, const BlockMmadOffsetParam& p, uint64_t kb,
                                                uint64_t k, uint64_t realK)
    {
        const uint64_t offset = PipelineOperandOffset(plan.reuseB, plan.mOffset != 0);
        auto a = MakeMxL0Operand<AType, false>(offset, plan.m, realK);
        const uint64_t scaleK = CeilDiv(realK, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if (plan.copyA) {
            CopyMxOperandTile(COPY_L1_TO_L0A_ATOM, a, tensorAL1_, plan.mOffset, (k + kb) % p.kaL1Size, plan.m, realK);
            CopyMxScaleToL0<ScaleAType, false, uint64_t>(
                tensorScaleAL1_, offset, plan.m, scaleK,
                (((k + kb) % PipelineConfig::ScaleWindow(hasBias_) / MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                plan.mOffset);
        }
        return a;
    }

    template <typename TensorBL1_>
    __aicore__ inline auto LoadPipelineOperandB(const L0SlicePlan& plan, const BlockMmadOffsetParam& p,
                                                const TensorBL1_& b, uint64_t kb, uint64_t k, uint64_t realK)
    {
        const uint64_t offset = PipelineOperandOffset(plan.reuseA, plan.nOffset != 0);
        auto result = MakeMxL0Operand<AType, true>(offset, realK, plan.n);
        const uint64_t scaleK = CeilDiv(realK, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if (plan.copyB) {
            CopyMxOperandTile(COPY_L1_TO_L0B_ATOM, result, b, k, plan.nOffset, realK, plan.n);
            CopyMxScaleToL0<ScaleBType, true, uint64_t>(
                tensorScaleBL1_, offset, scaleK, plan.n,
                (((k + kb) % PipelineConfig::ScaleWindow(hasBias_) / MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                plan.nOffset);
        }
        return result;
    }

    template <typename TensorL0C_, typename TensorAL0_, typename TensorBL0_>
    __aicore__ inline void IssuePipelineMmad(const L0SlicePlan& plan, const BlockMmadOffsetParam& p, uint64_t kb,
                                             uint64_t realKb, uint64_t k, uint64_t realK, const TensorL0C_& c,
                                             const TensorAL0_& a, const TensorBL0_& b)
    {
        const bool bias = hasBias_ && kb == 0 && k == 0;
        asc::te::mmad_params params;
        params.m = static_cast<uint16_t>(plan.m);
        params.n = static_cast<uint16_t>(plan.n);
        UpdateMxMmadParams(params, static_cast<uint16_t>(realK), kb == 0 && k == 0,
                           kb + realKb >= p.kSize && k + p.l0KSize >= realKb, bias);
        if (bias) {
            const uint64_t width = (MODE == MxPipelineMode::Aligned) ? plan.n : p.nL1Size;
            auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1, CeilAlign(width, static_cast<uint64_t>(BLOCK_CUBE)));
            const uint64_t offset = (MODE == MxPipelineMode::OperandTriple) ? (DOUBLE_BIAS_L1_OFFSETS[(bL1BufIdx_)]) :
                                                                              BIAS_L1_OFFSETS[bL1BufIdx_];
            auto source = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(offset), layout);
            auto table = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::bias, float>(l0BufIdx_ * BIAS_TABLE_SLOT_SIZE), layout);
            asc::te::copy(COPY_L1_TO_BT_ATOM, table, source);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            auto atom = asc::te::make_mmad(asc::te::mmad_operation{}, Tile::MmadTraitMX{}).with(params);
            asc::te::mmad(atom, c, a, b, table);
        } else {
            SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            IssueMxMmad(params, c, a, b);
        }
    }

    __aicore__ inline auto MakePipelineWeightL1(const BlockMmadOffsetParam& p, uint64_t realK) const
    {
        uint64_t offset = 0UL;
        if constexpr ((MODE == MxPipelineMode::Aligned)) {
            offset = bL1BufIdx_ * L1_BUF_OFFSET;
        } else if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            offset = bL1BufIdx_ * (hasBias_ ? L1_BUF_OFFSET : QUAD_B_STRIDE);
        } else {
            offset = (l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_B_L1_OFFSETS[(bL1BufIdx_)] :
                                                              DOUBLE_B_L1_OFFSETS[(bL1BufIdx_)]);
        }
        auto layout = asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>{}(realK,
                                                                                                         p.nL1Size);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offset), layout);
    }

    template <typename TensorL0C_>
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam& p,
                                         const TensorL0C_& c, uint64_t offset0 = 0UL, uint64_t offset1 = 0UL)
    {
        const auto plan = MakeL0SlicePlan(p, kbL1RealSize, c, offset0, offset1);
        const auto b = MakePipelineWeightL1(p, kbL1RealSize);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
        for (uint64_t k = 0UL; k < kbL1RealSize; k += p.l0KSize) {
            const uint64_t realK = Min(p.l0KSize, kbL1RealSize - k);
            if (plan.waitBefore) {
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
            }
            const auto a = LoadPipelineOperandA(plan, p, kbOffset, k, realK);
            const auto operandB = LoadPipelineOperandB(plan, p, b, kbOffset, k, realK);
            IssuePipelineMmad(plan, p, kbOffset, kbL1RealSize, k, realK, c, a, operandB);
            if (plan.releaseAfter) {
                SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
            }
            l0BufIdx_ ^= 1;
        }
    }

    __aicore__ inline void SetScaleMTE1ToMTE2()
    {
        if constexpr ((MODE == MxPipelineMode::Aligned) || (MODE == MxPipelineMode::OperandTriple)) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleL1BufIdx_);
            scaleL1BufIdx_ ^= 1;
        } else if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleL1BufIdx_);
            scaleL1BufIdx_ ^= 1;
            ++scaleBL1Seq_;
        }
    }

    __aicore__ inline void SetMTE1ToMTE2()
    {
        if constexpr ((MODE == MxPipelineMode::Aligned) || (MODE == MxPipelineMode::WeightQuad)) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
            aL1BufIdx_ ^= 1;
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
            aL1BufIdx_ = (l1BufferStage_ == TRIPLE_BUFFER_COUNT ? (aL1BufIdx_ + 1UL) % TRIPLE_BUFFER_COUNT :
                                                                  ((aL1BufIdx_ + 1UL) & 1UL));
        }
    }

    __aicore__ inline uint64_t PipelineAL1Address() const
    {
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return GetAL1Offset(aL1BufIdx_);
        } else {
            uint64_t base = A_L1_OFFSET;
            uint64_t stride = A_L1_BUF_OFFSET;
            uint64_t slot = aL1BufIdx_;
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (!hasBias_) {
                    base = QUAD_A_BASE;
                    stride = QUAD_A_STRIDE;
                }
                slot = fullLoadA_ ? 0 : slot;
            }
            const uint64_t shift = hasBias_ && aL1BufIdx_ == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0;
            return base + slot * stride + shift;
        }
    }

    template <typename TensorA_>
    __aicore__ inline void CopyAGmToL1(const TensorA_& tensorA, const BlockMmadOffsetParam& param, int64_t kaGmOffset)
    {
        const int64_t realK = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
        auto layout = MakeLayoutAL1{}(param.mL1Size, realK);
        auto source = tensorA.slice(asc::te::make_coord(0, kaGmOffset), asc::te::make_shape(param.mL1Size, realK));
        tensorAL1_ = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(PipelineAL1Address()),
                                          layout);
        asc::te::copy(COPY_GM_TO_L1_ATOM, tensorAL1_, source);
    }

    struct ScaleL1Offsets {
        uint64_t a;
        uint64_t b;
    };

    __aicore__ inline ScaleL1Offsets GetPipelineScaleAddresses() const
    {
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return {GetScaleAL1Offset(scaleL1BufIdx_), GetScaleBL1Offset(scaleL1BufIdx_)};
        } else {
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (!hasBias_) {
                    return {QUAD_SCALE_A_BASE + scaleL1BufIdx_ * QUAD_SCALE_STRIDE,
                            QUAD_SCALE_B_BASE + (scaleBL1Seq_ & (QUADRUPLE_BUFFER_COUNT - 1)) * QUAD_SCALE_STRIDE};
                }
            }
            const uint64_t shift = hasBias_ && scaleL1BufIdx_ == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0;
            return {SCALE_AL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET + shift,
                    SCALE_BL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET + shift};
        }
    }

    template <typename TensorScaleA_>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA_& a, const BlockMmadOffsetParam& p, uint64_t offset)
    {
        const uint64_t window = PipelineConfig::ScaleWindow(hasBias_);
        const uint64_t realK = Min(window, p.kSize - offset) / WQGMM_MX_GROUP_SIZE;
        const auto addresses = GetPipelineScaleAddresses();
        tensorScaleAL1_ = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(addresses.a),
                                               MakeLayoutScaleAL1{}(p.mL1Size, realK));
        auto slice = a.slice(asc::te::make_coord(0, offset / WQGMM_MX_GROUP_SIZE),
                             asc::te::make_shape(p.mL1Size, realK));
        asc::te::copy(COPY_GM_TO_L1_ATOM, tensorScaleAL1_, slice);
        tensorScaleBL1_ = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(addresses.b),
                                               MakeLayoutScaleBL1{}(realK, p.nL1Size));
    }

    __aicore__ inline void CalcDynamicKBlock(uint64_t rows, uint64_t columns, uint64_t& ka, uint64_t& kb) const
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return;
        }
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            if (l1BufferStage_ == TRIPLE_BUFFER_COUNT) {
                ka = L1_K_CONFIG_256;
                kb = L1_K_CONFIG_256;
                return;
            }
        }
        constexpr uint64_t halfK = L1_K_CONFIG_256 / DOUBLE_BUFFER_COUNT;
        kb = rows <= L1_K_DYNAMIC_CONFIG_M_THRESHOLD && columns <= L1_K_DYNAMIC_CONFIG_N_THRESHOLD ? L1_K_CONFIG_512 :
                                                                                                     L1_K_CONFIG_256;
        uint64_t actualA = A_L1_SINGLE_BUF_SIZE - (hasBias_ ? BIAS_L1_SINGLE_BUF_SIZE : 0);
        uint64_t balance = A_B_LOAD_BALANCE_FACTOR;
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            if (!hasBias_) {
                const uint64_t alignedN = CeilAlign(columns, static_cast<uint64_t>(BLOCK_CUBE));
                while (kb > halfK && alignedN * kb * sizeof(AType) > QUAD_B_STRIDE) {
                    kb >>= 1;
                }
                actualA = QUAD_A_STRIDE;
                balance = rows > halfK ? 1UL : A_B_LOAD_BALANCE_FACTOR;
            }
        }
        const uint64_t alignedM = CeilAlign(rows, static_cast<uint64_t>(BLOCK_CUBE));
        const uint64_t requested = CeilDiv(columns, alignedM * balance);
        const uint64_t modelLimit = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType)) / (alignedM * kb);
        const uint64_t actualLimit = (actualA / sizeof(AType)) / (alignedM * kb);
        const uint64_t limit = Min(modelLimit, actualLimit);
        ka = Min(requested == 0 ? 1UL : requested, limit == 0 ? 1UL : limit) * kb;
    }
};
#endif // #if defined(__NPU_ARCH__)

} // namespace Block
} // namespace Gemm
} // namespace Blaze
