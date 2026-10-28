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
 * \file block_mmad_confidence_head_fused.h
 * \brief AIC BlockMmad specialization for ConfidenceHeadFused GEMV.
 */

#ifndef CONFIDENCE_HEAD_FUSED_BLOCK_H
#define CONFIDENCE_HEAD_FUSED_BLOCK_H

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/block/block_mmad_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <bool HasMarkov_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_>
class BlockMmad<ConfidenceHeadFusedPolicy<A_FULL_LOAD_MODE, false, HasMarkov_>, AType_, LayoutA_, BType_, LayoutB_,
                CType_, LayoutC_, BiasType_, LayoutBias_> {
private:
    static constexpr uint32_t DEFAULT_BASE_N = 128;
    static constexpr uint32_t DEFAULT_BASE_K = 128;
    static constexpr uint32_t DEFAULT_K_L1 = 256;

public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    static_assert(AscendC::Std::is_same_v<AType, BType> &&
                      (AscendC::Std::is_same_v<AType, half> || AscendC::Std::is_same_v<AType, bfloat16_t>),
                  "ConfidenceHeadFused requires matching FP16 or BF16 inputs.");
    static_assert(AscendC::Std::is_same_v<CType, float> && AscendC::Std::is_same_v<BiasType, float>,
                  "ConfidenceHeadFused requires FP32 accumulation and bias.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<LayoutB, asc::te::dn_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<LayoutBias, asc::te::nd_ext_layout_ptn>,
                  "ConfidenceHeadFused requires NDExt weights/output/bias and DNExt features.");
    using DispatchPolicy = Blaze::Gemm::ConfidenceHeadFusedPolicy<Blaze::Gemm::A_FULL_LOAD_MODE, false, HasMarkov_>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static constexpr bool HAS_MARKOV = HasMarkov_;

    struct Params {
        uint32_t kEmbed{0};
        uint32_t kMarkov{0};
        uint32_t baseN{DEFAULT_BASE_N};
        uint32_t baseK{DEFAULT_BASE_K};
        uint32_t kL1{DEFAULT_K_L1};
        uint32_t l1Stages{static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
        uint16_t l0cStages{static_cast<uint16_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
        GM_ADDR biasGmAddr{nullptr};
    };

    __aicore__ inline BlockMmad() {}
    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        kEmbed_ = params.kEmbed;
        kMarkov_ = HAS_MARKOV ? params.kMarkov : 0U;

        alignedKEmbed_ = HAS_MARKOV ? Blaze::Gemm::CeilAlign(static_cast<int64_t>(kEmbed_),
                                                             static_cast<int64_t>(AscendC::BLOCK_CUBE)) :
                                      static_cast<int64_t>(kEmbed_);
        const int64_t k = HAS_MARKOV ? alignedKEmbed_ + static_cast<int64_t>(kMarkov_) : static_cast<int64_t>(kEmbed_);
        alignedK_ = Blaze::Gemm::CeilAlign(k, static_cast<int64_t>(AscendC::BLOCK_CUBE));

        baseN_ = params.baseN;
        baseK_ = params.baseK;
        // Use the tile sizes and buffer counts supplied by tiling.
        kL1_ = params.kL1;
        l1Stages_ = params.l1Stages;
        enableL0cPingPong_ = params.l0cStages >= Blaze::Gemm::DOUBLE_BUFFER_COUNT;
        isBias_ = params.biasGmAddr != nullptr;
        isAL1Loaded_ = false;

        const int64_t alignedKL1 = Blaze::Gemm::CeilAlign(static_cast<int64_t>(kL1_),
                                                          static_cast<int64_t>(AscendC::BLOCK_CUBE));
        const int64_t alignedBaseN = Blaze::Gemm::CeilAlign(static_cast<int64_t>(baseN_),
                                                            static_cast<int64_t>(AscendC::BLOCK_CUBE));
        const uint64_t aL1Size = alignedK_ * sizeof(AType);
        const uint64_t bL1OneBuffer = alignedKL1 * alignedBaseN * sizeof(BType);
        const uint64_t biasL1OneBuffer = isBias_ ? alignedBaseN * sizeof(BiasType) : 0;
        for (uint32_t i = 0; i < L1_BUFFER_CAPACITY; ++i) {
            bL1Buffer_[i] = aL1Size + i * (bL1OneBuffer + biasL1OneBuffer);
            biasL1Buffer_[i] = bL1Buffer_[i] + bL1OneBuffer;
        }

        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Blaze::Gemm::FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Blaze::Gemm::SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(true);
        }
    }

    template <typename TensorA, typename TensorB1, typename TensorB2, typename TensorBias, typename TensorOut>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB1& gmB1, const TensorB2& gmB2,
                                      const TensorBias& gmBias, TensorOut& ubOut, const BlockShape& blockShape)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (!isAL1Loaded_) {
            CopyAL1FromGM(gmA);
            isAL1Loaded_ = true;
        }
        int64_t curN = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        uint16_t l0cFlag = static_cast<uint16_t>(l0cPingPong_ & 1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFlag);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * (AscendC::TOTAL_L0C_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT);
        int64_t alignedCurN = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn,
                                                      AscendC::Std::Int<Blaze::Gemm::C0_SIZE_L0C>>{}(1L, alignedCurN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                              layoutL0C);

        bool initC = true;
        bool b1IsLast = true;
        if constexpr (HAS_MARKOV) {
            b1IsLast = kMarkov_ == 0;
        }
        initC = ProcessBMatrixKLoop(gmB1, gmBias, curN, static_cast<int64_t>(kEmbed_), 0L, tensorL0C, initC, b1IsLast);

        if constexpr (HAS_MARKOV) {
            if (kMarkov_ != 0) {
                ProcessBMatrixKLoop(gmB2, gmBias, curN, static_cast<int64_t>(kMarkov_), alignedKEmbed_, tensorL0C,
                                    initC, true);
            }
        }

        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::enable_update};
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        // Keep the physical UB row pitch aligned for the N tail.
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(FIXPIPE_ROWS),
                                                                               alignedCurN);
        auto physicalOut = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(ubOut.data().get()),
                                                layoutUB);
        asc::te::copy(copyL0C2UB.with(fixpParams), physicalOut, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cFlag);

        if (enableL0cPingPong_) {
            ++l0cPingPong_;
        }
    }

private:
    template <typename TensorA>
    __aicore__ inline void CopyAL1FromGM(const TensorA& gmA)
    {
        if ASCEND_IS_AIV {
            return;
        }
        auto layoutAL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, alignedK_);
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(0), layoutAL1);

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if (kEmbed_ != 0) {
            auto hiddenSrc = gmA.slice(asc::te::make_coord(0L, 0L),
                                       asc::te::make_shape(1L, static_cast<int64_t>(kEmbed_)));
            auto hiddenDst = tensorAL1.slice(asc::te::make_coord(0L, 0L),
                                             asc::te::make_shape(1L, static_cast<int64_t>(kEmbed_)));
            asc::te::copy(copyGM2L1, hiddenDst, hiddenSrc);
        }

        if constexpr (HAS_MARKOV) {
            if (kMarkov_ != 0) {
                auto markovSrc = gmA.slice(asc::te::make_coord(0L, static_cast<int64_t>(kEmbed_)),
                                           asc::te::make_shape(1L, static_cast<int64_t>(kMarkov_)));
                auto markovDst = tensorAL1.slice(asc::te::make_coord(0L, alignedKEmbed_),
                                                 asc::te::make_shape(1L, static_cast<int64_t>(kMarkov_)));
                asc::te::copy(copyGM2L1, markovDst, markovSrc);
            }
        }
    }

    template <typename TensorB, typename TensorBias, typename TensorL0C>
    __aicore__ inline bool ProcessBMatrixKLoop(const TensorB& gmB, const TensorBias& gmBias, int64_t curN,
                                               int64_t totalK, int64_t aL1KBaseOffset, TensorL0C& tensorL0C, bool initC,
                                               bool finalOnLastK)
    {
        int64_t kL1BlockNum = Blaze::Gemm::CeilDiv(totalK, static_cast<int64_t>(kL1_));
        for (int64_t kL1BlockIdx = 0; kL1BlockIdx < kL1BlockNum; ++kL1BlockIdx) {
            int64_t kL1Offset = kL1BlockIdx * kL1_;
            int64_t curKL1 = Blaze::Gemm::Min((totalK - kL1Offset), static_cast<int64_t>(kL1_));
            bool isLastKL1 = kL1BlockIdx + 1 == kL1BlockNum;
            ProcessKL1Block(gmB, gmBias, curN, kL1Offset, curKL1, aL1KBaseOffset + kL1Offset, tensorL0C, initC,
                            finalOnLastK && isLastKL1);
            initC = false;
        }
        return initC;
    }

    template <typename TensorB, typename TensorBias, typename TensorL0C>
    __aicore__ inline void ProcessKL1Block(const TensorB& gmB, const TensorBias& gmBias, int64_t curN, int64_t bKOffset,
                                           int64_t curKL1, int64_t aL1KBaseOffset, TensorL0C& tensorL0C, bool initC,
                                           bool isLastK)
    {
        uint16_t l1BufId = static_cast<uint16_t>(abL1LoopCnt_ % l1Stages_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

        auto tensorTuple = CopyL1FromGM(gmB, gmBias, curN, bKOffset, curKL1, l1BufId, initC);
        auto tensorBL1 = asc::te::get<0>(tensorTuple);
        auto tensorBiasL1 = asc::te::get<1>(tensorTuple);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

        int64_t kBlockNum = Blaze::Gemm::CeilDiv(curKL1, static_cast<int64_t>(baseK_));
        for (int64_t kBlockIdx = 0; kBlockIdx < kBlockNum; ++kBlockIdx) {
            int64_t kL1Offset = kBlockIdx * baseK_;
            int64_t curK = Blaze::Gemm::Min((curKL1 - kL1Offset), static_cast<int64_t>(baseK_));
            bool curInitC = initC && kBlockIdx == 0;
            bool isLastL0K = isLastK && (kBlockIdx + 1 == kBlockNum);
            ProcessKBlock(tensorBL1, tensorBiasL1, curN, kL1Offset, curK, aL1KBaseOffset + kL1Offset, tensorL0C,
                          curInitC, isLastL0K);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        ++abL1LoopCnt_;
    }

    template <typename TensorBL1, typename TensorBias, typename TensorL0C>
    __aicore__ inline void ProcessKBlock(const TensorBL1& tensorBL1, const TensorBias& tensorBiasL1, int64_t curN,
                                         int64_t bL1KOffset, int64_t curK, int64_t aL1KOffset, TensorL0C& tensorL0C,
                                         bool initC, bool isLastK)
    {
        uint64_t l0Offset = (l0PingPong_ & 1) * (AscendC::TOTAL_L0A_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT);
        uint16_t mte1Flag = static_cast<uint16_t>(Blaze::Gemm::SIXTH_FLAG + (l0PingPong_ & 1));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);

        TileShape l0Shape{1L, curN, curK};
        // Share the first-K state across embedding and markov, including markov-only inputs.
        const bool needBias = isBias_ && initC;
        auto tensorTuple = CopyL0FromL1(tensorBL1, tensorBiasL1, l0Shape, l0Offset, aL1KOffset, bL1KOffset, needBias);
        auto tensorAL0 = asc::te::get<0>(tensorTuple);
        auto tensorBL0 = asc::te::get<1>(tensorTuple);
        auto tensorBiasL0 = asc::te::get<2>(tensorTuple);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1Flag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1Flag);

        uint8_t unitFlag = isLastK ? Blaze::Gemm::FINAL_ACCUMULATION : Blaze::Gemm::NON_FINAL_ACCUMULATION;
        Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag, initC && !needBias);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);
        ++l0PingPong_;
    }

    template <typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(const TensorB& gmB, const TensorBias& gmBias, int64_t curN, int64_t kOffset,
                                        int64_t curK, uint16_t l1BufId, bool firstK)
    {
        int64_t alignedCurK = Blaze::Gemm::CeilAlign(curK, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        int64_t alignedCurN = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBL1 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            alignedCurK, alignedCurN);
        auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Buffer_[l1BufId]),
                                              layoutBL1);

        auto gmSlice = gmB.slice(asc::te::make_coord(kOffset, 0L), asc::te::make_shape(curK, curN));
        auto l1Slice = tensorBL1.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(curK, curN));
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        asc::te::copy(copyGM2L1, l1Slice, gmSlice);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Buffer_[l1BufId]),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, curN));
        if (isBias_ && firstK) {
            asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
        }
        return AscendC::Std::make_tuple(tensorBL1, tensorBiasL1);
    }

    template <typename TensorBL1, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorBL1& tensorBL1, const TensorBias& tensorBiasL1,
                                        const TileShape& l0Shape, uint64_t l0Offset, int64_t aL1KOffset,
                                        int64_t bL1KOffset, bool needBias)
    {
        int64_t curN = asc::te::get<Blaze::Gemm::MNK_N>(l0Shape);
        int64_t curK = asc::te::get<Blaze::Gemm::MNK_K>(l0Shape);
        int64_t alignedCurK = Blaze::Gemm::CeilAlign(curK, static_cast<int64_t>(AscendC::BLOCK_CUBE));

        auto layoutAL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, alignedK_);
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(0), layoutAL1);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0L, aL1KOffset),
                                              asc::te::make_shape(1L, alignedCurK));
        auto tensorAL1Nz = asc::te::make_tensor(
            tensorBlockAL1.data(),
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
                alignedCurK / AscendC::BLOCK_CUBE, AscendC::BLOCK_CUBE));

        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            alignedCurK / AscendC::BLOCK_CUBE, AscendC::BLOCK_CUBE);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                                              layoutAL0);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        asc::te::copy(copyL12L0A, tensorAL0, tensorAL1Nz);

        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            alignedCurK, curN);
        auto tensorBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, BType>(
                (l0PingPong_ & 1) * (AscendC::TOTAL_L0B_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT)),
            layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bL1KOffset, 0L),
                                              asc::te::make_shape(alignedCurK, curN));
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        auto tensorBiasL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::bias, BiasType>((l0PingPong_ & 1) * baseN_ * sizeof(BiasType)),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1L, Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE))));
        if (needBias) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }
        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TileShape& l0Shape, bool needBias, uint8_t unitFlag,
                                   bool initC)
    {
        int64_t curN = asc::te::get<Blaze::Gemm::MNK_N>(l0Shape);
        int64_t curK = asc::te::get<Blaze::Gemm::MNK_K>(l0Shape);
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{},
                                                     Blaze::Gemm::Tile::MmadTraitConfidenceHeadGemv{});
        asc::te::mmad_params mmadParams{1U, static_cast<uint16_t>(curN), static_cast<uint16_t>(curK),
                                        static_cast<asc::te::unit_flag_mode>(unitFlag), initC};
        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint32_t L1_BUFFER_CAPACITY = static_cast<uint32_t>(Blaze::Gemm::QUADRUPLE_BUFFER_COUNT);
    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = static_cast<uint16_t>(L1_BUFFER_CAPACITY);
    static constexpr uint32_t FIXPIPE_ROWS = 2; // Physical UB rows written by Fixpipe for GEMV.

    int64_t alignedKEmbed_{0};
    uint32_t kEmbed_{0};
    uint32_t kMarkov_{0};
    uint32_t baseN_{0};
    uint32_t baseK_{0};
    uint32_t kL1_{0};
    uint32_t l1Stages_{0};

    int64_t alignedK_{0};
    uint64_t bL1Buffer_[L1_BUFFER_CAPACITY] = {0};
    uint64_t biasL1Buffer_[L1_BUFFER_CAPACITY] = {0};

    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    bool isBias_{false};
    bool isAL1Loaded_{false};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze

#endif // CONFIDENCE_HEAD_FUSED_BLOCK_H
