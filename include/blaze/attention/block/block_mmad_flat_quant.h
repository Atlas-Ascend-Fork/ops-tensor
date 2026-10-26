/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_flat_quant.h
 * \brief
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/attention/policy/dispatch_policy.h"
#include "blaze/attention/block/block_mmad_decl.h"

namespace Blaze {
namespace Attention {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class LayoutC_, class CType_, class OutType_,
          class LayoutOut_>
class BlockMmad<Attention::BlockFlatQuant<>, AType_, LayoutA_, BType_, LayoutB_, OutType_, LayoutC_, CType_,
                LayoutOut_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using OutType = OutType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutOut = LayoutOut_;
    using DispatchPolicy = Attention::BlockFlatQuant<>;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Out_T = typename OutType::T;
    using L0cType = float;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        TupleShape problemShape{};
        TupleShape tileL1{};
        TupleShape tileL0{};
        bool hasP2{true};
    };

    __aicore__ inline BlockMmad() {}

    __aicore__ inline ~BlockMmad() {}

    __aicore__ inline void Init(const Params& params)
    {
        m_ = asc::te::get<Gemm::MNK_M>(params.problemShape);
        n_ = asc::te::get<Gemm::MNK_N>(params.problemShape);
        k_ = asc::te::get<Gemm::MNK_K>(params.problemShape);
        mL1_ = asc::te::get<Gemm::MNK_M>(params.tileL1);
        nL1_ = asc::te::get<Gemm::MNK_N>(params.tileL1);
        kL1_ = asc::te::get<Gemm::MNK_K>(params.tileL1);
        iterBatch_ = asc::te::get<Gemm::MNK_B>(params.tileL1);
        baseK_ = asc::te::get<Gemm::MNK_K>(params.tileL0);
        aL1OneBuffer_ = Gemm::CeilAlign(iterBatch_ * m_, Gemm::BLOCK_CUBE) * Gemm::CeilAlign(kL1_, Gemm::BLOCK_CUBE);
        aL1OneBufferBytes_ = aL1OneBuffer_ * sizeof(A_T);
        uint64_t nl1Align = Gemm::CeilAlign(static_cast<uint64_t>(nL1_), Gemm::BLOCK_CUBE);
        // 第一次矩阵乘为 (K, N) @ (N, N)，B 操作数 K/N 两轴长度均为 N，故 kl1Align 与 nl1Align 同取 nL1_
        uint64_t kl1Align = Gemm::CeilAlign(static_cast<uint64_t>(nL1_), Gemm::BLOCK_CUBE);
        bl1OffsetP2_ = nl1Align * kl1Align;
        l0PingPong_ = 0;
        l0cPingPong_ = 0;
        subBlockId_ = 0;
        hasP2_ = params.hasP2;
        bufMgr_.InitL0();
        bufMgr_.InitL0C();
    }

    template <typename TensorA, typename TensorP1, typename TensorP2>
    __aicore__ inline void operator()(TensorA gmA, TensorP1 gmP1, TensorP2 gmP2, TupleShape tileShape,
                                      bool isFirstRound)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / Gemm::DOUBLE_BUFFER_COUNT / sizeof(L0cType);
        mL1_ = asc::te::get<Gemm::MNK_M>(tileShape);
        nL1_ = asc::te::get<Gemm::MNK_N>(tileShape);
        kL1_ = asc::te::get<Gemm::MNK_K>(tileShape);
        iterBatch_ = asc::te::get<Gemm::MNK_B>(tileShape);

        uint64_t curML1 = mL1_;
        uint64_t curNL1 = nL1_;
        uint64_t curKL1 = kL1_;
        uint64_t curML0 = mL1_;
        uint64_t curNL0 = nL1_;

        bufMgr_.InitAL1(0, 0, 0);
        bufMgr_.InitAL1(1, aL1OneBufferBytes_, 1);
        bufMgr_.InitBL1(0, aL1OneBufferBytes_ * Gemm::DOUBLE_BUFFER_COUNT, 2);

        const auto& aL1Slot = bufMgr_.GetL1ASlot(l0cPingPong_ & 0x1);
        const auto& p1p2L1Slot = bufMgr_.GetL1BSlot(0);
        const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cPingPong_ & 0x1);

        uint64_t l0cOffsetBytes = l0cSlot.Addr();
        uint64_t alignedML0 = Gemm::CeilAlign(curML0, Gemm::BLOCK_CUBE);
        uint64_t alignedNL0 = Gemm::CeilAlign(curNL0, Gemm::BLOCK_CUBE);
        if (alignedML0 * alignedNL0 > HALF_L0C_SIZE) {
            l0cOffsetBytes = 0;
        }
        uint64_t al1OffsetBytes = aL1Slot.Addr();

        auto layoutAL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<A_T>>(curML1,
                                                                                                                curKL1);
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, A_T>(al1OffsetBytes),
                                              layoutAL1);
        CopyGM2L1(tensorAL1, gmA, aL1Slot);

        if (hasP2_) {
            if (isFirstRound) {
                auto layoutP2L1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                             asc::te::layout_trait_default<B_T>>(curKL1, curNL1);
                auto tensorP2L1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, B_T>(aL1OneBufferBytes_ * Gemm::DOUBLE_BUFFER_COUNT),
                    layoutP2L1);
                CopyGM2L1(tensorP2L1, gmP2, p1p2L1Slot);
            }

            uint64_t kL0Iter = Gemm::CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

                auto layoutP2L1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                             asc::te::layout_trait_default<B_T>>(curKL1, curNL0);
                auto tensorP2L1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, B_T>(aL1OneBufferBytes_ * Gemm::DOUBLE_BUFFER_COUNT),
                    layoutP2L1);
                auto l1Tensors = AscendC::Std::make_tuple(tensorAL1, tensorP2L1);
                auto l1Slots = AscendC::Std::make_tuple(aL1Slot, p1p2L1Slot);
                auto l0Shape = AscendC::Std::make_tuple(curML0, curNL0, curK0, iter1 * baseK_, 0UL);
                auto l0Tensors = CopyL1ToL0(l1Tensors, l0Shape, l0Slot, l1Slots);

                auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                            AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(curML0, curNL0);
                auto tensorL0C = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l0c, L0cType>(l0cOffsetBytes), layoutL0C);
                auto mmadSlots = AscendC::Std::make_tuple(l0Slot, l0cSlot);
                auto mnkShape = AscendC::Std::make_tuple(curML0, curNL0, curK0);
                Mmad(tensorL0C, l0Tensors, mnkShape, asc::te::unit_flag_mode::disable, iter1 == 0, mmadSlots);
                l0PingPong_++;
            }

            auto layoutL1Dst = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<A_T>>(
                curML0, curNL0);
            auto tensorL1Dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, A_T>(al1OffsetBytes),
                                                    layoutL1Dst);
            auto layoutL0CSrc = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                           AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(curML0, curNL0);
            auto tensorL0CSrc = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0c, L0cType>(l0cOffsetBytes), layoutL0CSrc);
            auto fixTensors = AscendC::Std::make_tuple(tensorL1Dst, tensorL0CSrc);
            auto fixSlots = AscendC::Std::make_tuple(l0cSlot, aL1Slot);
            FixpipeL0CToL1(fixTensors, fixSlots);
        }

        mL1_ = asc::te::get<Gemm::MNK_M>(tileShape);
        curKL1 = mL1_ / iterBatch_;
        curML1 /= iterBatch_;
        curML0 /= iterBatch_;
        if (isFirstRound) {
            auto layoutP1L1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<A_T>>(
                curML1, curKL1);
            auto tensorP1L1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, A_T>(aL1OneBufferBytes_ * Gemm::DOUBLE_BUFFER_COUNT +
                                                                  bl1OffsetP2_ * sizeof(A_T)),
                layoutP1L1);
            CopyGM2L1(tensorP1L1, gmP1, p1p2L1Slot);
        }

        uint64_t kL0Iter = Gemm::CeilDiv(curKL1, baseK_);
        for (uint64_t batchIdx = 0; batchIdx < iterBatch_; batchIdx++) {
            uint64_t l0cBatchOffset = batchIdx * curML0 * curNL0;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

                auto layoutP1L1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                             asc::te::layout_trait_default<A_T>>(curML1, curKL1);
                auto tensorP1L1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, A_T>(aL1OneBufferBytes_ * Gemm::DOUBLE_BUFFER_COUNT +
                                                                      bl1OffsetP2_ * sizeof(A_T)),
                    layoutP1L1);
                auto layoutTempL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                               asc::te::layout_trait_default<A_T>>(curKL1 * iterBatch_,
                                                                                                   curNL0);
                auto tensorTempL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, A_T>(al1OffsetBytes), layoutTempL1);
                auto l1Tensors = AscendC::Std::make_tuple(tensorP1L1, tensorTempL1);
                auto l1Slots = AscendC::Std::make_tuple(p1p2L1Slot, aL1Slot);
                auto l0Shape = AscendC::Std::make_tuple(curML0, curNL0, curK0, iter1 * baseK_, batchIdx * curKL1);
                auto l0Tensors = CopyL1ToL0(l1Tensors, l0Shape, l0Slot, l1Slots);

                auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                            AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(curML0, curNL0);
                auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0cType>(
                                                          l0cOffsetBytes + l0cBatchOffset * sizeof(L0cType)),
                                                      layoutL0C);
                auto mmadSlots = AscendC::Std::make_tuple(l0Slot, l0cSlot);
                auto mnkShape = AscendC::Std::make_tuple(curML0, curNL0, curK0);
                Mmad(tensorL0C, l0Tensors, mnkShape, asc::te::unit_flag_mode::disable, iter1 == 0, mmadSlots);
                l0PingPong_++;
            }
        }

        auto layoutL0COut = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(
            static_cast<uint64_t>(iterBatch_), curML0, curNL0);
        auto tensorL0COut = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0cType>(l0cOffsetBytes),
                                                 layoutL0COut);
        uint64_t nAlign = Gemm::CeilAlign(curNL0, Gemm::BLOCK_CUBE);
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<uint64_t>(iterBatch_),
                                                                               curML0, nAlign);
        auto tensorUB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(0), layoutUB);
        auto ubTensors = AscendC::Std::make_tuple(tensorUB, tensorL0COut);
        FixpipeL0CToUB(ubTensors, l0cSlot);
        if (alignedML0 * alignedNL0 <= HALF_L0C_SIZE) {
            l0cPingPong_++;
        }
    }

private:
    template <typename TensorDst, typename TensorSrc>
    __aicore__ inline void CopyGM2L1(TensorDst& dst, const TensorSrc& src, const Gemm::BufferSlot& slot)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto lock = slot.LockMte2();
        asc::te::copy(copyGM2L1, dst, src);
    }

    template <typename L1Tensors, typename SlotsTuple, typename L0Shape>
    __aicore__ inline auto CopyL1ToL0(const L1Tensors& l1Tensors, const L0Shape& l0Shape,
                                      const Gemm::BufferSlot& l0Slot, const SlotsTuple& slotsTuple)
    {
        const auto& tensorAL1 = asc::te::get<0>(l1Tensors);
        const auto& tensorBL1 = asc::te::get<1>(l1Tensors);
        const auto& aL1Slot = asc::te::get<0>(slotsTuple);
        const auto& bL1Slot = asc::te::get<1>(slotsTuple);
        auto curML0 = asc::te::get<0>(l0Shape);
        auto curNL0 = asc::te::get<1>(l0Shape);
        auto curK0 = asc::te::get<2>(l0Shape);
        auto kOffset = asc::te::get<3>(l0Shape);
        auto bRowOffset = asc::te::get<4>(l0Shape);

        auto layoutL0A = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<A_T>>(curML0,
                                                                                                                curK0);
        auto tensorL0A = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, A_T>(l0Slot.Addr()),
                                              layoutL0A);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kOffset), asc::te::make_shape(curML0, curK0));
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        {
            auto l1LockA = aL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0A, tensorL0A, tensorBlockAL1);
        }

        auto layoutL0B = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<B_T>>(curK0,
                                                                                                                curNL0);
        auto tensorL0B = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, B_T>(l0Slot.Addr()),
                                              layoutL0B);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bRowOffset + kOffset, 0),
                                              asc::te::make_shape(curK0, curNL0));
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        {
            auto l1LockB = bL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0B, tensorL0B, tensorBlockBL1);
        }

        return AscendC::Std::make_tuple(tensorL0A, tensorL0B);
    }

    template <typename TensorL0C, typename L0Tensors, typename SlotsTuple, typename MnkShape>
    __aicore__ inline void Mmad(TensorL0C& tensorL0C, const L0Tensors& l0Tensors, const MnkShape& mnkShape,
                                asc::te::unit_flag_mode unitFlag, bool cmatrixInitVal, const SlotsTuple& slots)
    {
        const auto& tensorL0A = asc::te::get<0>(l0Tensors);
        const auto& tensorL0B = asc::te::get<1>(l0Tensors);
        const auto& l0Slot = asc::te::get<0>(slots);
        const auto& l0cSlot = asc::te::get<1>(slots);
        auto curM = asc::te::get<0>(mnkShape);
        auto curN = asc::te::get<1>(mnkShape);
        auto curK = asc::te::get<2>(mnkShape);

        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK), unitFlag, cmatrixInitVal};
        auto l0Lock = l0Slot.LockM();
        auto l0cLock = l0cSlot.LockM();
        asc::te::mmad(asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation>>{}.with(mmadParams), tensorL0C,
                      tensorL0A, tensorL0B);
    }

    template <typename TensorsTuple, typename SlotsTuple>
    __aicore__ inline void FixpipeL0CToL1(const TensorsTuple& tensors, const SlotsTuple& slots)
    {
        const auto& dst = asc::te::get<0>(tensors);
        const auto& src = asc::te::get<1>(tensors);
        const auto& l0cSlot = asc::te::get<0>(slots);
        const auto& l1Slot = asc::te::get<1>(slots);

        auto copyL0C2L1 = asc::te::make_copy(asc::te::copy_l0c_to_l1{});
        auto l0cLock = l0cSlot.LockFix();
        auto l1Lock = l1Slot.LockFix();
        asc::te::copy(copyL0C2L1, dst, src);
    }

    template <typename TensorsTuple>
    __aicore__ inline void FixpipeL0CToUB(const TensorsTuple& tensors, const Gemm::BufferSlot& l0cSlot)
    {
        const auto& dst = asc::te::get<0>(tensors);
        const auto& src = asc::te::get<1>(tensors);

        asc::te::l0c_to_ub_params fixpipeParams{asc::te::unit_flag_mode::disable,
                                                static_cast<uint8_t>((subBlockId_++) & 0x1)};
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        auto l0cLock = l0cSlot.LockFix();
        asc::te::copy(copyL0C2UB.with(fixpipeParams), dst, src);
    }

    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseK_{16};
    uint64_t l0PingPong_{1};
    uint64_t l0cPingPong_{1};
    uint64_t subBlockId_{0};
    bool hasP2_{true};
    uint64_t aL1OneBuffer_ = 0;
    uint64_t aL1OneBufferBytes_ = 0;
    uint64_t iterBatch_ = 0;
    uint64_t bl1OffsetP2_ = 0;

    Gemm::BufferManager<2, 2, 2> bufMgr_;
};

} // namespace Block
} // namespace Attention
} // namespace Blaze
