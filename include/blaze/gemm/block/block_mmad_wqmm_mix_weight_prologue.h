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
 * \file block_mmad_wqmm_mix_weight_prologue.h
 * \brief AIC-side BlockMmad specialization for multiplication with dequantized 8-bit weights.
 *
 * The AIV prologue dequantizes weight into shared L1 buffers. This block loads x and optional bias,
 * copies inputs from L1 to L0, computes MMAD, and writes y through Fixpipe. Cross-core flags coordinate
 * weight buffer reuse; BufferSlot entries manage local buffer addresses and pipeline synchronization.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Gemm::Block {

template <uint64_t AivNum_, uint32_t UbMte2InnerSize_, uint32_t UbMte2BufNum_, QuantMode AntiquantType_,
          bool HasAntiquantOffset_, class AType_, class LayoutA_, class BTypeTuple_, class LayoutBTuple_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulWithWeightAntiquant<AivNum_, UbMte2InnerSize_, UbMte2BufNum_, AntiquantType_, HasAntiquantOffset_>, AType_,
    LayoutA_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    static_assert(AscendC::Std::tuple_size_v<BTypeTuple_> == 2, "B type tuple must contain weight and scale");
    static_assert(AscendC::Std::tuple_size_v<LayoutBTuple_> == 2, "B layout tuple must contain weight and scale");
    using DispatchPolicy = MatmulWithWeightAntiquant<AivNum_, UbMte2InnerSize_, UbMte2BufNum_, AntiquantType_,
                                                     HasAntiquantOffset_>;
    using AType = AType_;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using ScaleType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScale = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static_assert(AscendC::Std::is_same_v<AType, half> || AscendC::Std::is_same_v<AType, bfloat16_t>,
                  "WQMM Blaze requires FP16/BF16 A");
    static_assert(AscendC::Std::is_same_v<CType, half> || AscendC::Std::is_same_v<CType, bfloat16_t>,
                  "WQMM Blaze requires FP16/BF16 C");
    static_assert(AscendC::Std::is_same_v<BType, int8_t> || AscendC::Std::is_same_v<BType, float8_e4m3_t> ||
                      AscendC::Std::is_same_v<BType, hifloat8_t>,
                  "WQMM Blaze requires int8/FP8 E4M3/HiFloat8 B");
    static_assert(AscendC::Std::is_same_v<ScaleType, AType>,
                  "WQMM Blaze requires antiquant scale/offset type to match A");

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        L1TileShape l1TileShape;
        L0TileShape l0TileShape;
        bool hasBias{false};
        uint64_t kSize{0};
    };

    __aicore__ inline BlockMmad() = default;

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        kSize_ = params.kSize;
        hasBias_ = params.hasBias;
        nBase_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.l1TileShape));
        kL1Size_ = static_cast<uint64_t>(asc::te::get<3>(params.l1TileShape));
        kL0Size_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.l0TileShape));
        nL0Size_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.l0TileShape));
        kTileCount_ = CeilDiv(kSize_, kL1Size_);
        aTilesPerBank_ = CeilDiv(kTileCount_, DOUBLE_BUFFER_COUNT);
        // Load the complete x tile when KaL1 exceeds KbL1, including K tails.
        aFullLoad_ = static_cast<uint64_t>(asc::te::get<2>(params.l1TileShape)) > kL1Size_;
        InitBufferSlots();

        if ASCEND_IS_AIV {
            return;
        }
        AscendC::SetMMLayoutTransform(true);
    }

    template <typename MemoryLocation, typename DType>
    __aicore__ inline auto GetShareMemPtr(uint32_t bufferId) const
    {
        static_assert(AscendC::Std::is_same_v<MemoryLocation, asc::te::location::l1>,
                      "WQMM shared weight memory must be in L1");
        static_assert(AscendC::Std::is_same_v<DType, AType>,
                      "WQMM shared weight type must match the antiquant output type");
        return asc::te::make_mem_ptr<MemoryLocation, DType>(bL1Slots_[bufferId].Addr());
    }

    template <class TensorA, class TensorBias, class TensorC, class Shape>
    __aicore__ inline void operator()(const TensorA& tensorA, const TensorBias& tensorBias, TensorC& tensorC,
                                      const Shape& actualShape)
    {
        mL1Size_ = static_cast<uint64_t>(asc::te::get<MNK_M>(actualShape));
        nL1Size_ = static_cast<uint64_t>(asc::te::get<MNK_N>(actualShape));
        if (aFullLoad_) {
            const bool interleaved = CanInterleaveFullA();
            const auto& aSlot0 = aL1Slots_[0U];
            const auto& aSlot1 = aL1Slots_[1U];
            {
                auto aWrite0 = aSlot0.LockMte2();
                auto aWrite1 = aSlot1.LockMte2();
                CopyFullAToL1(tensorA, interleaved);
                CopyBiasToL1(tensorBias, cvLoopIdx_ & 1U);
            }
            {
                // Full x loading reserves both buffers until all K/L0 reads of this M/N tile finish.
                auto aRead0 = aSlot0.LockMte1();
                auto aRead1 = aSlot1.LockMte1();
                RunKLoop<true>(tensorA, tensorBias, interleaved);
            }
        } else {
            RunKLoop<false>(tensorA, tensorBias, false);
        }
        CopyCToGm(tensorC);
    }

private:
    using SyncProtocol = typename DispatchPolicy::SyncProtocol;
    using MakeAL1Layout = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>;
    // LayoutB uses (N,K) coordinates for weight: ND stores N x K, DN stores K x N.
    // The L1 layout describes the logical (K,N) matrix used by MMAD.
    using MakeBL1Layout = AscendC::Std::conditional_t<
        !IsTrans<LayoutB>::value,
        asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;

    __aicore__ inline void InitBufferSlots()
    {
        // Keep the L1 bias reservation consistent with Kernel::PreloadA.
        constexpr uint64_t BIAS_L1_BYTES = 4096UL;
        const uint64_t weightBytes = nBase_ * kL1Size_ * sizeof(AType);
        const uint64_t biasBytes = hasBias_ ? BIAS_L1_BYTES : 0UL;
        const uint64_t aL1Base = weightBytes + biasBytes;
        aHalfBytes_ = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT - aL1Base;
        // L1 regions: weight0 | bias0 | x0 | x1 | bias1 | weight1. All offsets are bytes.
        for (uint32_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
            uint8_t bufferId = BufferLayout::L1ADataBufferId(i);
            aL1Slots_[i] = {aL1Base + i * aHalfBytes_, bufferId};
            bL1Slots_[i] = {i * (AscendC::TOTAL_L1_SIZE - weightBytes), bufferId};
            if (hasBias_) {
                biasL1Slots_[i] = {i == 0 ? weightBytes : AscendC::TOTAL_L1_SIZE - weightBytes - biasBytes, bufferId};
            }
            btSlots_[i] = {nL0Size_ * sizeof(float) * i, BufferLayout::BTBufferId(i)};
            l0Slots_[i] = {(AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT) * i, BufferLayout::L0BufferId(i)};
        }
        l0cSlot_ = {0UL, BufferLayout::L0CBufferId(0U)};
    }

    __aicore__ inline bool CanInterleaveFullA() const
    {
        const uint64_t maxHalfBytes = aTilesPerBank_ * kL1Size_ * CeilAlign(mL1Size_, BLOCK_CUBE) * sizeof(AType);
        return !IsTrans<LayoutA>::value && kSize_ % kL1Size_ == 0 && maxHalfBytes <= aHalfBytes_;
    }

    template <typename TensorA>
    __aicore__ inline void CopyFullAToL1(const TensorA& tensorA, bool interleaved)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if (!interleaved) {
            auto l1A = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Slots_[0U].Addr()),
                                            MakeAL1Layout{}(mL1Size_, kSize_));
            asc::te::copy(copyGM2L1, l1A, tensorA);
            return;
        }

        for (uint32_t parity = 0; parity < DOUBLE_BUFFER_COUNT; ++parity) {
            const uint64_t count = parity == 0 ? aTilesPerBank_ : kTileCount_ - aTilesPerBank_;
            if (count == 0) {
                continue;
            }
            auto gmTile = tensorA.slice(asc::te::make_coord(0UL, parity * kL1Size_),
                                        asc::te::make_shape(mL1Size_, kL1Size_));
            auto gmLayout = gmTile.layout();
            using GmPattern = asc::te::get_layout_pattern<typename TensorA::layout_type>;
            auto batchGmLayout = asc::te::make_pattern_layout<GmPattern, asc::te::layout_trait_default<AType>>(
                asc::te::make_shape(count, gmLayout.shape()),
                asc::te::make_stride(DOUBLE_BUFFER_COUNT * kL1Size_, gmLayout.stride()));
            auto gmBatch = asc::te::make_tensor(gmTile.data(), batchGmLayout);
            auto l1Layout = MakeAL1Layout{}(mL1Size_, kL1Size_);
            auto batchL1Layout = asc::te::make_pattern_layout<asc::te::nz_layout_ptn,
                                                              asc::te::layout_trait_default<AType>>(
                asc::te::make_shape(count, l1Layout.shape()),
                asc::te::make_stride(CeilAlign(mL1Size_, BLOCK_CUBE) * kL1Size_, l1Layout.stride()));
            const uint64_t bank = (cvLoopIdx_ + parity) & 1U;
            auto l1Batch = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Slots_[bank].Addr()), batchL1Layout);
            asc::te::copy(copyGM2L1, l1Batch, gmBatch);
        }
    }

    template <typename TensorBias>
    __aicore__ inline void CopyBiasToL1(const TensorBias& tensorBias, uint64_t bank)
    {
        if (hasBias_) {
            auto biasLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nL1Size_);
            auto l1Bias = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Slots_[bank].Addr()), biasLayout);
            asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_l1{}), l1Bias, tensorBias);
        }
    }

    template <bool FullLoad, typename TensorA, typename TensorBias>
    __aicore__ inline void RunKLoop(const TensorA& tensorA, const TensorBias& tensorBias, bool interleaved)
    {
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount_; ++kLoopIdx, ++cvLoopIdx_) {
            const uint64_t kOffset = kLoopIdx * kL1Size_;
            const uint64_t kL1Size = Min(kL1Size_, kSize_ - kOffset);
            if constexpr (!FullLoad) {
                auto tensorATile = tensorA.slice(asc::te::make_coord(0UL, kOffset),
                                                 asc::te::make_shape(mL1Size_, kL1Size));
                CopyAAndBiasToL1(tensorATile, tensorBias, kOffset, kL1Size, cvLoopIdx_);
            }
            WaitConvertedWeight();
            if (FullLoad && !interleaved) {
                auto fullAL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Slots_[0U].Addr()),
                    MakeAL1Layout{}(mL1Size_, kSize_));
                auto tileAL1 = fullAL1.slice(asc::te::make_coord(0UL, kOffset), asc::te::make_shape(mL1Size_, kL1Size));
                IterateMmad<FullLoad>(tileAL1, kOffset, kL1Size);
            } else {
                auto tileAL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, AType>(CurrentAOffset(kOffset, interleaved)),
                    MakeAL1Layout{}(mL1Size_, kL1Size));
                IterateMmad<FullLoad>(tileAL1, kOffset, kL1Size);
            }
            ReleaseConvertedWeight();
        }
    }

    template <typename TensorA, typename TensorBias>
    __aicore__ inline void CopyAAndBiasToL1(const TensorA& tensorA, const TensorBias& tensorBias, uint64_t kOffset,
                                            uint64_t kL1Size, uint64_t loopIdx)
    {
        uint64_t bank = loopIdx & 1U;
        const auto& l1DataSlot = aL1Slots_[bank];
        auto l1A = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1DataSlot.Addr()),
                                        MakeAL1Layout{}(mL1Size_, kL1Size));
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto l1DataLock = l1DataSlot.LockMte2();
        asc::te::copy(copyGM2L1, l1A, tensorA);

        if (kOffset == 0) {
            CopyBiasToL1(tensorBias, bank);
        }
    }

    __aicore__ inline uint64_t CurrentAOffset(uint64_t kOffset, bool interleaved) const
    {
        uint64_t bank = cvLoopIdx_ & 1U;
        uint64_t offset = aL1Slots_[bank].Addr();
        if (interleaved) {
            offset += (kOffset / (kL1Size_ * 2U)) * CeilAlign(mL1Size_, BLOCK_CUBE) * kL1Size_ * sizeof(AType);
        }
        return offset;
    }

    template <bool AL1AlreadyLocked, typename TensorAL1>
    __aicore__ inline void IterateMmad(const TensorAL1& tensorAL1, uint64_t kOffset, uint64_t kL1Size)
    {
        uint64_t bank = cvLoopIdx_ & 1U;
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(bL1Slots_[bank].Addr()),
            MakeBL1Layout{}(kL1Size, nL1Size_));
        auto tensorL0C = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot_.Addr()),
            asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(mL1Size_, nL1Size_));

        uint64_t kL0Count = CeilDiv(kL1Size, kL0Size_);
        for (uint64_t kL0Idx = 0; kL0Idx < kL0Count; ++kL0Idx) {
            uint64_t kL0Offset = kL0Idx * kL0Size_;
            uint64_t kL0Size = Min(kL0Size_, kL1Size - kL0Offset);
            uint64_t l0Buffer = l0LoopIdx_ & 1U;
            const auto& l0Slot = l0Slots_[l0Buffer];
            const auto& btSlot = btSlots_[l0Buffer];
            uint64_t l0Offset = l0Slot.Addr();
            auto tensorAL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(mL1Size_,
                                                                                                         kL0Size));
            auto tensorBL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0Offset),
                asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>(kL0Size,
                                                                                                         nL1Size_));
            auto aSlice = tensorAL1.slice(asc::te::make_coord(0UL, kL0Offset), asc::te::make_shape(mL1Size_, kL0Size));
            auto bSlice = tensorBL1.slice(asc::te::make_coord(kL0Offset, 0UL), asc::te::make_shape(kL0Size, nL1Size_));

            auto tensorBiasL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::bias, float>(btSlot.Addr()),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, CeilAlign(nL1Size_, BLOCK_CUBE)));
            bool needBias = hasBias_ && kOffset == 0 && kL0Idx == 0;
            auto copyL1ToL0 = [&]() __aicore__ {
                auto l0Lock = l0Slot.LockMte1();
                asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0a{}), tensorAL0, aSlice);
                asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0b{}), tensorBL0, bSlice);
                if (needBias) {
                    auto tensorBiasL1 = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Slots_[bank].Addr()),
                        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nL1Size_));
                    auto btLock = btSlot.LockMte1();
                    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_biastable{}), tensorBiasL0, tensorBiasL1);
                }
            };
            if constexpr (AL1AlreadyLocked) {
                copyL1ToL0();
            } else {
                auto l1DataLock = aL1Slots_[bank].LockMte1();
                copyL1ToL0();
            }

            uint8_t unitFlag = kOffset + kL0Offset + kL0Size == kSize_ ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            bool initC = kOffset == 0 && kL0Idx == 0 && !hasBias_;
            asc::te::mmad_params mmParams{static_cast<uint16_t>(mL1Size_), static_cast<uint16_t>(nL1Size_),
                                          static_cast<uint16_t>(kL0Size),
                                          static_cast<asc::te::unit_flag_mode>(unitFlag), initC};
            constexpr auto mmAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
            auto l0Lock = l0Slot.LockM();
            auto l0cLock = l0cSlot_.LockM();
            if (needBias) {
                auto btLock = btSlot.LockM();
                asc::te::mmad(mmAtom.with(mmParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
            } else {
                asc::te::mmad(mmAtom.with(mmParams), tensorL0C, tensorAL0, tensorBL0);
            }
            ++l0LoopIdx_;
        }
    }

    template <typename TensorC>
    __aicore__ inline void CopyCToGm(TensorC& tensorC)
    {
        auto l0cLock = l0cSlot_.LockFix();
        auto tensorL0C = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot_.Addr()),
            asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(mL1Size_, nL1Size_));
        asc::te::l0c_to_gm_params fixParams{static_cast<asc::te::unit_flag_mode>(FINAL_ACCUMULATION)};
        asc::te::copy(asc::te::make_copy(asc::te::copy_l0c_to_gm{}).with(fixParams), tensorC, tensorL0C);
    }

    __aicore__ inline void WaitConvertedWeight() { WaitWeightFlag<SyncProtocol::AIV_READY_FLAG>(); }

    __aicore__ inline void ReleaseConvertedWeight() { SetWeightFlag<SyncProtocol::AIC_FREE_FLAG>(); }

    template <uint16_t FLAG>
    __aicore__ inline void WaitWeightFlag() const
    {
        if constexpr (AivNum_ == 2) {
            AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
        }
        AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
    }

    template <uint16_t FLAG>
    __aicore__ inline void SetWeightFlag() const
    {
        if constexpr (AivNum_ == 2) {
            AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
        }
        AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
    }

    static constexpr uint64_t DOUBLE_BUFFER_COUNT = 2UL;
    using BufferLayout = BufferIdLayout<DOUBLE_BUFFER_COUNT, DOUBLE_BUFFER_COUNT, DOUBLE_BUFFER_COUNT>;

    uint64_t kSize_{0};
    uint64_t nBase_{0};
    uint64_t kL1Size_{0};
    uint64_t kL0Size_{0};
    uint64_t nL0Size_{0};
    uint64_t kTileCount_{0};
    uint64_t aTilesPerBank_{0};
    uint64_t aHalfBytes_{0};
    uint64_t mL1Size_{0};
    uint64_t nL1Size_{0};
    uint64_t l0LoopIdx_{0};
    uint64_t cvLoopIdx_{0};
    bool hasBias_{false};
    bool aFullLoad_{false};
    BufferSlot aL1Slots_[DOUBLE_BUFFER_COUNT];
    BufferSlot bL1Slots_[DOUBLE_BUFFER_COUNT];
    BufferSlot biasL1Slots_[DOUBLE_BUFFER_COUNT];
    BufferSlot btSlots_[DOUBLE_BUFFER_COUNT];
    BufferSlot l0Slots_[DOUBLE_BUFFER_COUNT]; // L0A/L0B share offsets and synchronization IDs.
    BufferSlot l0cSlot_;                      // Only one accumulator buffer is used.
};

} // namespace Blaze::Gemm::Block
