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
 * \file kernel_matmul_mix_weight_prologue.h
 * \brief Kernel orchestration for weight-only MX matmul with an AIV weight-conversion prologue.
 */
#pragma once

#include "kernel_basic_intf.h"

#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/arch35/copy_gm_to_ub.h"
#include "blaze/gemm/tile/arch35/copy_ub_to_l1.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class BlockMmad_>
class KernelMatmulMixWeightPrologue {
public:
    using BlockMmad = BlockMmad_;
    using OutType = typename BlockMmad::AType;
    using InType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static_assert(AscendC::IsSameType<InType, fp4x2_e2m1_t>::value,
                  "Mix weight prologue expects float4_e2m1 input weights");
    static_assert(AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value,
                  "Mix weight prologue expects float8_e4m3 converted weights");

    struct Params {
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t kUbSize{0};
        uint64_t nUbSize{0};
        bool disableWeightL2{false};
    };

    __aicore__ inline KernelMatmulMixWeightPrologue(const Params& params, const typename BlockMmad::Params& mmadParams);

    __aicore__ inline ~KernelMatmulMixWeightPrologue();

    template <typename GMWeightTensor, typename GMBiasTensor, typename ActualBlockShape>
    __aicore__ inline void operator()(const GMWeightTensor& gmWeightTensor, const GMBiasTensor& gmBiasTensor,
                                      const ActualBlockShape& actualBlockShape, int64_t problemM);

private:
    static constexpr uint64_t CACHE_LINE_BYTES = 128UL;

    template <typename Tp>
    __aicore__ inline static bool IsCacheLineAligned(uint64_t innerSize);

    using L1Storage = typename BlockMmad::WeightL1Storage;
    using BiasUbTensor = decltype(asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(uint64_t{}),
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(int64_t{}, int64_t{})));
    using BiasL1Tensor = decltype(L1Storage{}.MakeBiasTensor(uint64_t{}, int64_t{}));
    static constexpr uint16_t FP4_PACK_SHIFT = 1U;
    static constexpr uint64_t OUTPUT_SYNC_ID_BASE = QUADRUPLE_BUFFER_COUNT;
    static constexpr uint64_t VECTOR_ELEMENT_COUNT = static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / sizeof(OutType);
    static constexpr auto COPY_GM_TO_UB_ATOM = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    static constexpr auto COPY_UB_TO_L1_ATOM = asc::te::make_copy(asc::te::copy_ub_to_l1{});
    static constexpr auto COPY_WEIGHT_GM_TO_UB_ATOM = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
    static constexpr auto COPY_WEIGHT_UB_TO_L1_ATOM = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});

    __aicore__ inline static uint64_t SingleWeightInSize(const Params& params);
    __aicore__ inline static uint64_t SingleWeightOutSize(const Params& params);
    __aicore__ inline static uint64_t SingleBiasSize(const typename BlockMmad::Params& mmadParams);

    __aicore__ inline auto MakePackedWeight(uint64_t offset, int64_t nSize, int64_t kSize) const
    {
        if constexpr (IsWeightNz<LayoutB>::value) {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, InType>(offset),
                asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<asc::te::c0_element<OutType>>>(
                    kSize, nSize));
        } else {
            const int64_t kStride = static_cast<int64_t>(Align64(static_cast<uint64_t>(kSize)));
            auto fullTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, InType>(offset),
                asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(kStride, nSize));
            return fullTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                    asc::te::make_shape(kSize, nSize));
        }
    }

    __aicore__ inline auto MakeConvertedWeight(uint64_t offset, int64_t nSize, int64_t kSize) const
    {
        if constexpr (IsWeightNz<LayoutB>::value) {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, OutType>(offset),
                Blaze::Gemm::Weight8BitZnToZnUBLayout<OutType>{}(
                    kSize, static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))), znOutputInterBufferStride_));
        } else {
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(offset),
                                        Blaze::Gemm::Weight8BitDnToZnUBLayout<OutType>{}(kSize, nSize));
        }
    }

    __aicore__ inline auto MakeBias(uint64_t offset, int64_t nSize) const
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            static_cast<int64_t>(1), static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))));
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(offset), layout);
    }

    __aicore__ inline void WaitWeightFlag(uint8_t flag) const;

    __aicore__ inline void SetWeightFlag(uint8_t flag) const;

    template <typename GMWeightTensor, typename GMBiasTensor>
    __aicore__ inline void Process(const GMWeightTensor& gmWeightTensor, const GMBiasTensor& gmBiasTensor,
                                   int64_t kGmOffset, int64_t kL1Length, int64_t nBlockLength, bool allowL2Bypass,
                                   uint8_t flag);

    template <typename GMWeightSlice, typename Weight4Tensor, typename GMBiasTensor>
    __aicore__ inline void CopyWeightGmToUb(const GMWeightSlice& gmSlice, const Weight4Tensor& weight4BitTensor,
                                            const GMBiasTensor& gmBiasTensor, const BiasUbTensor& biasInUbTensor,
                                            const BufferSlot& inputSlot, bool processBias) const;

    template <typename Weight4Tensor, typename Weight8Tensor>
    __aicore__ inline void ConvertWeight(const Weight4Tensor& weight4BitTensor, const Weight8Tensor& weight8BitTensor,
                                         const BiasUbTensor& biasInUbTensor, const BiasUbTensor& biasOutUbTensor,
                                         const BufferSlot& inputSlot, const BufferSlot& outputSlot,
                                         bool processBias) const;

    template <typename L1Tensor, typename Weight8Tensor>
    __aicore__ inline void CopyWeightUbToL1(const L1Tensor& l1Tensor, const Weight8Tensor& weight8BitTensor,
                                            const BiasUbTensor& biasOutUbTensor, const BiasL1Tensor& biasL1Tensor,
                                            const BufferSlot& outputSlot, bool processBias) const;

    __aicore__ inline void AdvanceUbBuffer();

    __aicore__ inline bool ShouldProcessBias(int64_t kGmOffset) const;

    const Params& params_;
    const typename BlockMmad::Params& mmadParams_;
    uint64_t singleWeightInSize_{0};
    uint64_t singleWeightOutSize_{0};
    uint64_t singleBiasSize_{0};
    uint64_t weightInBase_{0};
    uint64_t biasOutBase_{0};
    uint64_t biasInBase_{0};
    uint64_t weightInOffset_{0};
    uint64_t weightOutOffset_{0};
    uint64_t biasInOffset_{0};
    uint64_t biasOutOffset_{0};
    uint64_t znOutputInterBufferStride_{0};
    L1Storage l1Storage_;
    uint64_t l1NStride_{0};
    uint64_t ubBufIdx_{0};
    uint64_t l1BufIdx_{0};
};

#define BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE template <class BlockMmad_>
#define BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS KernelMatmulMixWeightPrologue<BlockMmad_>

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::KernelMatmulMixWeightPrologue(
    const Params& params, const typename BlockMmad::Params& mmadParams)
    : params_(params),
      mmadParams_(mmadParams),
      singleWeightInSize_(SingleWeightInSize(params)),
      singleWeightOutSize_(SingleWeightOutSize(params)),
      singleBiasSize_(SingleBiasSize(mmadParams)),
      weightInBase_(mmadParams.ubBufferNum * singleWeightOutSize_),
      biasOutBase_(weightInBase_ + mmadParams.ubBufferNum * singleWeightInSize_),
      biasInBase_(biasOutBase_ + mmadParams.ubBufferNum * singleBiasSize_),
      weightInOffset_(weightInBase_),
      weightOutOffset_(0UL),
      biasInOffset_(biasInBase_),
      biasOutOffset_(biasOutBase_),
      znOutputInterBufferStride_(VECTOR_ELEMENT_COUNT * mmadParams.ubBufferNum),
      l1Storage_(static_cast<uint64_t>(asc::te::get<MNK_N>(mmadParams.l1TileShape)),
                 static_cast<uint64_t>(asc::te::get<MNK_K>(mmadParams.l1TileShape)), mmadParams.l1BufferNum,
                 mmadParams.hasBias)
{}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::~KernelMatmulMixWeightPrologue()
{
    for (uint64_t index = 0; index < mmadParams_.l1BufferNum; ++index) {
        WaitWeightFlag(static_cast<uint8_t>(index));
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename GMWeightTensor, typename GMBiasTensor, typename ActualBlockShape>
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::operator()(const GMWeightTensor& gmWeightTensor,
                                                                          const GMBiasTensor& gmBiasTensor,
                                                                          const ActualBlockShape& actualBlockShape,
                                                                          int64_t problemM)
{
    const int64_t totalKLength = asc::te::get<MNK_K>(actualBlockShape);
    const int64_t nLength = asc::te::get<MNK_N>(actualBlockShape);
    const int64_t mLength = asc::te::get<MNK_M>(actualBlockShape);
    const int64_t kL1Size = static_cast<int64_t>(asc::te::get<MNK_K>(mmadParams_.l1TileShape));
    const bool allowL2Bypass = mLength >= problemM;
    l1NStride_ = Align16(static_cast<uint64_t>(nLength));
    for (int64_t kOffset = 0; kOffset < totalKLength; kOffset += kL1Size) {
        const int64_t kLength = Min(totalKLength - kOffset, kL1Size);
        const uint8_t freeFlag = static_cast<uint8_t>(l1BufIdx_);
        Process(gmWeightTensor, gmBiasTensor, kOffset, kLength, nLength, allowL2Bypass, freeFlag);
        l1BufIdx_ = BlockMmad::NextBufferId(l1BufIdx_, mmadParams_.l1BufferNum);
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename Tp>
__aicore__ inline bool BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::IsCacheLineAligned(uint64_t innerSize)
{
    constexpr uint64_t FP4_ELEMENTS_PER_BYTE = 2UL;
    if constexpr (IsFp4<Tp>()) {
        return innerSize % (CACHE_LINE_BYTES * FP4_ELEMENTS_PER_BYTE) == 0UL;
    }
    return innerSize % CACHE_LINE_BYTES == 0UL;
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline uint64_t BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::SingleWeightInSize(const Params& params)
{
    if constexpr (IsWeightNz<LayoutB>::value) {
        return (Align16(params.nUbSize) * Align32(params.kUbSize)) >> FP4_PACK_SHIFT;
    }
    return (params.nUbSize * Align64(params.kUbSize)) >> FP4_PACK_SHIFT;
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline uint64_t BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::SingleWeightOutSize(const Params& params)
{
    const uint64_t kUbStride = Align64(params.kUbSize);
    if constexpr (IsWeightNz<LayoutB>::value) {
        return Align16(params.nUbSize) * kUbStride * sizeof(OutType);
    }
    return (Align16(params.nUbSize) + 1UL) * kUbStride * sizeof(OutType);
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline uint64_t BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::SingleBiasSize(
    const typename BlockMmad::Params& mmadParams)
{
    if (!mmadParams.hasBias) {
        return 0UL;
    }
    constexpr uint64_t VECTOR_ELEM = static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / sizeof(BiasType);
    const uint64_t baseN = static_cast<uint64_t>(asc::te::get<MNK_N>(mmadParams.l1TileShape));
    return CeilAlign(baseN, VECTOR_ELEM) * sizeof(BiasType);
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::WaitWeightFlag(uint8_t flag) const
{
    Sync::WaitForCube<BlockMmad::SyncProtocol::MODE, PIPE_MTE3>(flag);
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::SetWeightFlag(uint8_t flag) const
{
    Sync::NotifyCube<BlockMmad::SyncProtocol::MODE, PIPE_MTE3>(flag);
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename GMWeightTensor, typename GMBiasTensor>
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::Process(const GMWeightTensor& gmWeightTensor,
                                                                       const GMBiasTensor& gmBiasTensor,
                                                                       int64_t kGmOffset, int64_t kL1Length,
                                                                       int64_t nBlockLength, bool allowL2Bypass,
                                                                       uint8_t flag)
{
    WaitWeightFlag(flag);
    const bool processBias = ShouldProcessBias(kGmOffset);
    const bool isSecondAiv = AscendC::GetSubBlockIdx() == 1U;
    int64_t kOffset = 0;
    int64_t nOffset = 0;
    int64_t kLength = Min(kL1Length, static_cast<int64_t>(params_.kUbSize));
    int64_t nLength = Min(nBlockLength, static_cast<int64_t>(params_.nUbSize));
    // One UB partition is owned by one AIV: NZ splits K and ND splits N.
    if constexpr (IsWeightNz<LayoutB>::value) {
        kOffset = isSecondAiv ? static_cast<int64_t>(params_.kUbSize) : 0;
        kLength = Min(kL1Length - kOffset, static_cast<int64_t>(params_.kUbSize));
    } else {
        nOffset = isSecondAiv ? static_cast<int64_t>(params_.nUbSize) : 0;
        nLength = Min(nBlockLength - nOffset, static_cast<int64_t>(params_.nUbSize));
    }

    const uint64_t ubBufIdx = ubBufIdx_;
    if (kLength > 0 && nLength > 0) {
        const BufferSlot inputSlot{weightInOffset_, static_cast<uint8_t>(ubBufIdx)};
        const BufferSlot outputSlot{weightOutOffset_, static_cast<uint8_t>(OUTPUT_SYNC_ID_BASE + ubBufIdx)};
        auto gmSlice = gmWeightTensor.slice(asc::te::make_coord(kGmOffset + kOffset, nOffset),
                                            asc::te::make_shape(kLength, nLength));
        auto weight4BitTensor = MakePackedWeight(weightInOffset_, nLength, kLength);
        auto weight8BitTensor = MakeConvertedWeight(weightOutOffset_, nLength, kLength);
        auto l1Tensor = l1Storage_.MakeWeightTensor(l1BufIdx_, nOffset, kOffset, l1NStride_, nLength, kLength);
        BiasUbTensor biasInUbTensor;
        BiasUbTensor biasOutUbTensor;
        BiasL1Tensor biasL1Tensor;
        if (processBias) {
            biasInUbTensor = MakeBias(biasInOffset_, nBlockLength);
            biasOutUbTensor = MakeBias(biasOutOffset_, nBlockLength);
            biasL1Tensor = l1Storage_.MakeBiasTensor(l1BufIdx_, nBlockLength);
        }
        const bool disableWeightL2 = params_.disableWeightL2 && allowL2Bypass &&
                                     (IsWeightNz<LayoutB>::value || IsCacheLineAligned<InType>(kLength));
        gmSlice.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        CopyWeightGmToUb(gmSlice, weight4BitTensor, gmBiasTensor, biasInUbTensor, inputSlot, processBias);
        ConvertWeight(weight4BitTensor, weight8BitTensor, biasInUbTensor, biasOutUbTensor, inputSlot, outputSlot,
                      processBias);
        CopyWeightUbToL1(l1Tensor, weight8BitTensor, biasOutUbTensor, biasL1Tensor, outputSlot, processBias);
    }
    AdvanceUbBuffer();
    SetWeightFlag(flag);
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename GMWeightSlice, typename Weight4Tensor, typename GMBiasTensor>
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::CopyWeightGmToUb(
    const GMWeightSlice& gmSlice, const Weight4Tensor& weight4BitTensor, const GMBiasTensor& gmBiasTensor,
    const BiasUbTensor& biasInUbTensor, const BufferSlot& inputSlot, bool processBias) const
{
    auto mte2Lock = inputSlot.LockMte2();
    asc::te::copy(COPY_WEIGHT_GM_TO_UB_ATOM, weight4BitTensor, gmSlice);
    if (processBias) {
        asc::te::copy(COPY_GM_TO_UB_ATOM, biasInUbTensor, gmBiasTensor);
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename Weight4Tensor, typename Weight8Tensor>
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::ConvertWeight(
    const Weight4Tensor& weight4BitTensor, const Weight8Tensor& weight8BitTensor, const BiasUbTensor& biasInUbTensor,
    const BiasUbTensor& biasOutUbTensor, const BufferSlot& inputSlot, const BufferSlot& outputSlot,
    bool processBias) const
{
    auto inputVectorLock = inputSlot.LockV();
    auto outputVectorLock = outputSlot.LockV();
    Blaze::Gemm::Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor);
    if (processBias) {
        Blaze::Gemm::Tile::ScaleMxBias<BiasType>(biasInUbTensor, biasOutUbTensor);
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
template <typename L1Tensor, typename Weight8Tensor>
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::CopyWeightUbToL1(
    const L1Tensor& l1Tensor, const Weight8Tensor& weight8BitTensor, const BiasUbTensor& biasOutUbTensor,
    const BiasL1Tensor& biasL1Tensor, const BufferSlot& outputSlot, bool processBias) const
{
    auto mte3Lock = outputSlot.LockMte3();
    asc::te::copy(COPY_WEIGHT_UB_TO_L1_ATOM, l1Tensor, weight8BitTensor);
    if (processBias) {
        asc::te::copy(COPY_UB_TO_L1_ATOM, biasL1Tensor, biasOutUbTensor);
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::AdvanceUbBuffer()
{
    weightInOffset_ += singleWeightInSize_;
    if constexpr (IsWeightNz<LayoutB>::value) {
        weightOutOffset_ += VECTOR_ELEMENT_COUNT;
    } else {
        weightOutOffset_ += singleWeightOutSize_;
    }
    biasInOffset_ += singleBiasSize_;
    biasOutOffset_ += singleBiasSize_;
    ubBufIdx_ = BlockMmad::NextBufferId(ubBufIdx_, mmadParams_.ubBufferNum);
    if (ubBufIdx_ == 0UL) {
        weightInOffset_ = weightInBase_;
        weightOutOffset_ = 0UL;
        biasInOffset_ = biasInBase_;
        biasOutOffset_ = biasOutBase_;
    }
}

BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
__aicore__ inline bool BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS::ShouldProcessBias(int64_t kGmOffset) const
{
    if (!mmadParams_.hasBias || kGmOffset != 0) {
        return false;
    }
    return AscendC::GetSubBlockIdx() == 0;
}

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<BlockEpilogue_, void> &&
        AscendC::Std::is_same_v<KernelMixWithWeightPrologue, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleA = typename BlockMmad::LayoutScaleA;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockPrologue = KernelMatmulMixWeightPrologue<BlockMmad>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<LayoutScaleA, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, AscendC::Std::Int<SCALE_C0>>;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using PrologueParams = typename BlockPrologue::Params;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        PrologueParams prologueParams;
        BlockSchedulerParams schedulerParams;
    };

    __aicore__ inline GemmUniversal() = default;

    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline static auto MakeGmBiasTensor(const Params& params, int64_t n)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                        reinterpret_cast<__gm__ BiasType*>(params.prologueParams.biasGmAddr)),
                                    asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), n));
    }

    __aicore__ inline static auto MakeGmWeightTensor(const Params& params, int64_t k, int64_t n)
    {
        if constexpr (IsWeightNz<LayoutB>::value) {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
                asc::te::make_frame_layout<LayoutB, AscendC::Std::Int<asc::te::c0_element<AType>>>(k, n));
        } else {
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                            reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
                                        asc::te::frame_layout_format<LayoutB>{}(k, n));
        }
    }

    __aicore__ inline static void RunAiv(const Params& params, const BlockScheduler& scheduler);
    __aicore__ inline static void RunAic(const Params& params, const BlockScheduler& scheduler);
};

#undef BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_TEMPLATE
#undef BLAZE_MATMUL_MIX_WEIGHT_PROLOGUE_CLASS

#define BLAZE_MATMUL_MIX_GEMM_TEMPLATE \
    template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
#define BLAZE_MATMUL_MIX_GEMM_CLASS                                 \
    GemmUniversal<                                                  \
        ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_, \
        AscendC::Std::enable_if_t<                                  \
            AscendC::Std::is_same_v<BlockEpilogue_, void> &&        \
            AscendC::Std::is_same_v<KernelMixWithWeightPrologue, typename BlockMmad_::DispatchPolicy::ScheduleType>>>

BLAZE_MATMUL_MIX_GEMM_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_GEMM_CLASS::operator()(const Params& params)
{
    BlockScheduler scheduler(params.problemShape, params.schedulerParams);
    if ASCEND_IS_AIV {
        RunAiv(params, scheduler);
    }
    if ASCEND_IS_AIC {
        RunAic(params, scheduler);
    }
}

BLAZE_MATMUL_MIX_GEMM_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_GEMM_CLASS::RunAiv(const Params& params, const BlockScheduler& scheduler)
{
    const int64_t k = asc::te::get<MNK_K>(params.problemShape);
    const int64_t n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t problemM = asc::te::get<MNK_M>(params.problemShape);
    const uint64_t tileNum = scheduler.GetTileCount();
    const uint64_t curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    if (curBlockIdx >= tileNum) {
        return;
    }

    BlockPrologue blockPrologue(params.prologueParams, params.mmadParams);
    auto gmBias = MakeGmBiasTensor(params, n);
    auto gmWeight = MakeGmWeightTensor(params, k, n);
    const uint64_t blockNum = AscendC::GetBlockNum();
    for (uint64_t loopIdx = curBlockIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto blockShape = scheduler.GetBlockShape(blockCoord);
        const int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
        const int64_t kSize = asc::te::get<MNK_K>(blockShape);
        const int64_t nL1Size = asc::te::get<MNK_N>(blockShape);
        auto gmBlockWeight = gmWeight.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(kSize, nL1Size));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0, nOffset),
                                        asc::te::make_shape(static_cast<int64_t>(1), nL1Size));
        blockPrologue(gmBlockWeight, gmBlockBias, blockShape, problemM);
    }
}

BLAZE_MATMUL_MIX_GEMM_TEMPLATE
__aicore__ inline void BLAZE_MATMUL_MIX_GEMM_CLASS::RunAic(const Params& params, const BlockScheduler& scheduler)
{
    const uint64_t tileNum = scheduler.GetTileCount();
    const uint64_t curBlockIdx = AscendC::GetBlockIdx();
    if (curBlockIdx >= tileNum) {
        return;
    }
    const int64_t m = asc::te::get<MNK_M>(params.problemShape);
    const int64_t n = asc::te::get<MNK_N>(params.problemShape);
    const int64_t k = asc::te::get<MNK_K>(params.problemShape);
    const int64_t scaleKSize = static_cast<int64_t>(
        CeilDiv(Align64(static_cast<uint64_t>(k)), BlockMmad::MX_GROUP_SIZE));
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)),
        MakeLayoutA{}(m, k));
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ScaleAType*>(params.mmadParams.scaleAGmAddr)),
                                         MakeLayoutScaleA{}(m, scaleKSize));
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ScaleBType*>(params.mmadParams.scaleBGmAddr)),
                                         MakeLayoutScaleB{}(scaleKSize, n));
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)),
        MakeLayoutC{}(m, n));
    BlockMmad blockMmad(params.mmadParams);
    const uint64_t blockNum = AscendC::GetBlockNum();
    for (uint64_t loopIdx = curBlockIdx; loopIdx < tileNum; loopIdx += blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto blockShape = scheduler.GetBlockShape(blockCoord);
        const int64_t mOffset = asc::te::get<MNK_M>(blockCoord);
        const int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
        const int64_t mL1Size = asc::te::get<MNK_M>(blockShape);
        const int64_t nL1Size = asc::te::get<MNK_N>(blockShape);
        const int64_t kSize = asc::te::get<MNK_K>(blockShape);
        auto gmBlockA = gmA.slice(asc::te::make_coord(mOffset, 0), asc::te::make_shape(mL1Size, kSize));
        auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mOffset, 0), asc::te::make_shape(mL1Size, scaleKSize));
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(scaleKSize, nL1Size));
        auto gmBlockC = gmC.slice(asc::te::make_coord(mOffset, nOffset), asc::te::make_shape(mL1Size, nL1Size));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, gmBlockC);
    }
}

#undef BLAZE_MATMUL_MIX_GEMM_TEMPLATE
#undef BLAZE_MATMUL_MIX_GEMM_CLASS

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
