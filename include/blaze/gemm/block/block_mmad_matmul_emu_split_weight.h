/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

constexpr uint64_t L1_STAGES = 2UL;
constexpr uint64_t L1_BUFFER_MASK = L1_STAGES - 1UL;

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class L0CType_, class LayoutC_, class BiasType_,
          class LayoutBias_>
class BlockMmad<MatmulEmuSplitWeightPolicy, AType_, LayoutA_, BType_, LayoutB_, L0CType_, LayoutC_, BiasType_,
                LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = L0CType_;
    using L0CType = L0CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using DispatchPolicy = MatmulEmuSplitWeightPolicy;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = Blaze::Gemm::IsTrans<LayoutA_>::value;
    static constexpr bool TRANS_B = Blaze::Gemm::IsTrans<LayoutB_>::value;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType_>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType_>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType_>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType_>>>;

    struct Params {
        GM_ADDR xGmAddr{nullptr};
        GM_ADDR wHighGmAddr{nullptr};
        GM_ADDR wLowGmAddr{nullptr};
        uint64_t k{0UL};
        uint64_t kL1{0UL};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t usedCoreNum{0};
    };

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = params.k;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        kL1_ = params.kL1;
        l1LoopCnt_ = 0;
        l0PingPong_ = 0;

        uint64_t aL1OneSize = baseM_ * kL1_ * sizeof(AType_);
        uint64_t bL1OneSize = kL1_ * baseN_ * sizeof(BType_);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT;

        // 2buffer时：|APing,BLowPing,BHighPing---|APong,BLowPong,BHighPong---|
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            uint64_t base = slotSize * i;
            aL1Buffer_[i] = base;
            bLowL1Buffer_[i] = base + aL1OneSize;
            bHighL1Buffer_[i] = base + aL1OneSize + bL1OneSize;
        }
    }

    template <typename TensorA, typename TensorBHigh, typename TensorBLow, typename TensorCHigh, typename TensorCLow>
    __aicore__ inline void operator()(TensorA gmA, TensorBHigh gmBHigh, TensorBLow gmBLow, TensorCHigh ubBlockCHigh,
                                      TensorCLow ubBlockCLow, BlockShape singleShape, bool targetSubBlockId = false)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t kL1Iter = Blaze::Gemm::CeilDiv(k_, kL1_);

        auto gmTensors = AscendC::Std::make_tuple(gmA, gmBHigh, gmBLow);
        auto ubTensors = AscendC::Std::make_tuple(ubBlockCHigh, ubBlockCLow);

        auto curM = asc::te::get<MNK_M>(singleShape);
        auto curN = asc::te::get<MNK_N>(singleShape);
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(curM,
                                                                                                                curN);
        auto tensorL0C1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType_>(0), layoutL0C);
        auto tensorL0C2 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType_>(HALF_L0C_SIZE),
                                               layoutL0C);
        auto l0CTensors = AscendC::Std::make_tuple(tensorL0C1, tensorL0C2);

        for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
            uint16_t l1BufId = l1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curKL1 = (iter0 + 1 == kL1Iter) ? (k_ - kL1Offset) : kL1_;

            auto l1Shape = BlockShape{curM, curN, static_cast<int64_t>(curKL1), static_cast<int64_t>(iter0)};
            auto l1Tensors = CopyL1FromGM(gmTensors, l1Shape, l1BufId);

            uint64_t kL0Iter = Blaze::Gemm::CeilDiv(curKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curKL1) ? (curKL1 - kL0Offset) : baseK_;
                uint16_t l0BufferId = l0PingPong_ & 0x1;

                auto l0Shape = BlockShape{curM, curN, static_cast<int64_t>(curKL0), static_cast<int64_t>(kL0Offset)};
                auto l0Tensors = CopyL0FromL1(l1Tensors, l0Shape, l0BufferId, l1BufId);

                bool isFirstKBeat = (iter0 == 0 && iter1 == 0);
                bool isLastKBeat = (iter0 + 1 == kL1Iter && iter1 + 1 == kL0Iter);
                {
                    Gemm::LockM(l0BufferId + L0_BUFFER_ID_BASE);
                    Compute(l0CTensors, l0Tensors, l0Shape, isFirstKBeat, isLastKBeat);
                    Gemm::UnLockM(l0BufferId + L0_BUFFER_ID_BASE);
                }
                l0PingPong_++;
            }
            l1LoopCnt_++;
        }

        CopyL0CToUB(l0CTensors, ubTensors, targetSubBlockId);
    }

private:
    uint64_t k_{0UL};
    uint64_t kL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l1LoopCnt_{0UL};

    // Weight(B)切分成wHigh和wLow两部分: A与BLow共享L1 stage锁, BHigh使用独立锁
    uint64_t aL1Buffer_[L1_STAGES] = {0};
    uint64_t bLowL1Buffer_[L1_STAGES] = {0};
    uint64_t bHighL1Buffer_[L1_STAGES] = {0};

    template <typename GmTensors>
    __aicore__ inline auto CopyL1FromGM(const GmTensors& gmTensors, const BlockShape& blockShape, uint16_t l1BufferId)
    {
        const auto& gmA = asc::te::get<0>(gmTensors);
        const auto& gmBHigh = asc::te::get<1>(gmTensors);
        const auto& gmBLow = asc::te::get<2>(gmTensors);

        auto curM = asc::te::get<MNK_M>(blockShape);
        auto curN = asc::te::get<MNK_N>(blockShape);
        auto curKL1 = asc::te::get<MNK_K>(blockShape);
        auto kIdx = static_cast<uint64_t>(asc::te::get<3>(blockShape));

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

        Gemm::LockMte2(l1BufferId + L1_BUFFER_ID_BASE);

        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType_>(aL1Buffer_[l1BufferId]), layoutAL1);
        auto gmBlockA = gmA.slice(asc::te::make_coord(0, kIdx * kL1_), asc::te::make_shape(curM, curKL1));
        asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);

        auto layoutWL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBLowL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType_>(bLowL1Buffer_[l1BufferId]), layoutWL1);
        auto gmBlockBLow = gmBLow.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curN));
        asc::te::copy(copyGM2L1, tensorBLowL1, gmBlockBLow);

        auto tensorBHighL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType_>(bHighL1Buffer_[l1BufferId]), layoutWL1);
        auto gmBlockBHigh = gmBHigh.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curN));
        asc::te::copy(copyGM2L1, tensorBHighL1, gmBlockBHigh);

        Gemm::UnLockMte2(l1BufferId + L1_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorAL1, tensorBLowL1, tensorBHighL1);
    }

    template <typename L1Tensors>
    __aicore__ inline auto CopyL0FromL1(const L1Tensors& l1Tensors, const BlockShape& blockShape, uint16_t l0BufferId,
                                        uint16_t l1BufferId)
    {
        const auto& tensorAL1 = asc::te::get<0>(l1Tensors);
        const auto& tensorBLowL1 = asc::te::get<1>(l1Tensors);
        const auto& tensorBHighL1 = asc::te::get<2>(l1Tensors);

        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0BaseOffset = HALF_L0_SIZE * l0BufferId;
        auto curM = asc::te::get<MNK_M>(blockShape);
        auto curN = asc::te::get<MNK_N>(blockShape);
        auto curKL0 = asc::te::get<MNK_K>(blockShape);
        auto kL0Offset = static_cast<uint64_t>(asc::te::get<3>(blockShape));

        Gemm::LockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});

        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType_>>(
            curM, curKL0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType_>(l0BaseOffset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kL0Offset), asc::te::make_shape(curM, curKL0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType_>>(
            curKL0, curN);
        auto tensorBL01 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType_>(l0BaseOffset),
                                               layoutBL0);
        auto tensorBlockBL1Low = tensorBLowL1.slice(asc::te::make_coord(kL0Offset, 0),
                                                    asc::te::make_shape(curKL0, curN));
        asc::te::copy(copyL12L0B, tensorBL01, tensorBlockBL1Low);

        uint64_t l0bOneBufBytes = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(curKL0),
                                                         static_cast<uint64_t>(C0_SIZE_L0C)) *
                                  Blaze::Gemm::CeilAlign(static_cast<uint64_t>(curN),
                                                         static_cast<uint64_t>(C0_SIZE_L0C)) *
                                  sizeof(BType_);
        auto tensorBL02 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, BType_>(l0BaseOffset + l0bOneBufBytes), layoutBL0);
        auto tensorBlockBL1High = tensorBHighL1.slice(asc::te::make_coord(kL0Offset, 0),
                                                      asc::te::make_shape(curKL0, curN));
        asc::te::copy(copyL12L0B, tensorBL02, tensorBlockBL1High);

        Gemm::UnLockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorAL0, tensorBL01, tensorBL02);
    }

    template <typename L0CTensors, typename L0Tensors>
    __aicore__ inline void Compute(const L0CTensors& l0CTensors, const L0Tensors& l0Tensors,
                                   const BlockShape& blockShape, bool isFirstKBeat, bool isLastKBeat)
    {
        const auto& tensorL0C1 = asc::te::get<0>(l0CTensors);
        const auto& tensorL0C2 = asc::te::get<1>(l0CTensors);
        const auto& tensorAL0 = asc::te::get<0>(l0Tensors);
        const auto& tensorBL01 = asc::te::get<1>(l0Tensors);
        const auto& tensorBL02 = asc::te::get<2>(l0Tensors);

        auto curM = asc::te::get<MNK_M>(blockShape);
        auto curN = asc::te::get<MNK_N>(blockShape);
        auto curKL0 = asc::te::get<MNK_K>(blockShape);

        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        asc::te::unit_flag_mode mmadUnitFlag = isLastKBeat ? asc::te::unit_flag_mode::enable_update :
                                                             asc::te::unit_flag_mode::enable_keep;
        bool mmadCmatrixInitVal = isFirstKBeat;

        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curKL0), mmadUnitFlag, mmadCmatrixInitVal};
        asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C1, tensorAL0, tensorBL01);
        asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C2, tensorAL0, tensorBL02);
    }

    template <typename L0CTensors, typename UbTensors>
    __aicore__ inline void CopyL0CToUB(const L0CTensors& l0CTensors, const UbTensors& ubTensors, bool targetSubBlockId)
    {
        const auto& tensorL0C1 = asc::te::get<0>(l0CTensors);
        const auto& tensorL0C2 = asc::te::get<1>(l0CTensors);
        const auto& ubBlockCHigh = asc::te::get<0>(ubTensors);
        const auto& ubBlockCLow = asc::te::get<1>(ubTensors);

        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::enable_update,
                                             static_cast<uint8_t>(targetSubBlockId)};
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        asc::te::copy(copyL0C2UB.with(fixpParams), ubBlockCLow, tensorL0C1);
        asc::te::copy(copyL0C2UB.with(fixpParams), ubBlockCHigh, tensorL0C2);
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
