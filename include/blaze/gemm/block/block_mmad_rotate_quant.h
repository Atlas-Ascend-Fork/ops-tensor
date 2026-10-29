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
 * \file block_mmad_rotate_quant.h
 * \brief
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"

namespace Blaze {
namespace Gemm {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_>
class BlockMmad<Gemm::BlockRotateQuant<>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
                LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = Gemm::BlockRotateQuant<>;
    using AElementType = typename AType::T;
    using BElementType = typename BType::T;
    using CElementType = typename CType::T;
    using BiasElementType = typename BiasType::T;
    using L0CType = float;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::Std::is_same_v<AElementType, half> || AscendC::Std::is_same_v<AElementType, bfloat16_t>,
                  "RotateQuant BlockMmad requires FP16 or BF16 inputs.");
    static_assert(AscendC::Std::is_same_v<AElementType, BElementType>,
                  "RotateQuant X and rotation must have the same element type.");
    static_assert(AscendC::Std::is_same_v<CElementType, bfloat16_t>, "RotateQuant Fixpipe output must be BF16.");

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR alphaGMAddr{nullptr};
        TupleShape problemShape{};
        TupleShape blockL1{};
        TupleShape blockL0{};
        int64_t b{1};
    };

    __aicore__ inline BlockMmad() {}

    __aicore__ inline ~BlockMmad() {}

    __aicore__ inline void Init(const Params& params)
    {
        b_ = asc::te::get<Gemm::MNK_B>(params.problemShape);
        baseK_ = asc::te::get<Gemm::MNK_K>(params.blockL0);
        uint64_t mL1 = asc::te::get<Gemm::MNK_M>(params.blockL1);
        uint64_t nL1 = asc::te::get<Gemm::MNK_N>(params.blockL1);
        uint64_t kL1 = asc::te::get<Gemm::MNK_K>(params.blockL1);
        uint64_t aL1OneBuffer = Gemm::CeilAlign(mL1, Gemm::BLOCK_CUBE) * Gemm::CeilAlign(kL1, Gemm::BLOCK_CUBE);
        uint64_t aL1OneBufferBytes = aL1OneBuffer * sizeof(AElementType);
        uint64_t rotL1OneBufferBytes = Gemm::CeilAlign(kL1, Gemm::BLOCK_CUBE) * Gemm::CeilAlign(nL1, Gemm::BLOCK_CUBE) *
                                       sizeof(BElementType);
        uint64_t rotL1Base = aL1OneBufferBytes * Gemm::DOUBLE_BUFFER_COUNT;
        for (uint32_t i = 0; i < Gemm::DOUBLE_BUFFER_COUNT; ++i) {
            aL1Buffer_[i] = aL1OneBufferBytes * i;
            rotL1Buffer_[i] = rotL1Base + rotL1OneBufferBytes * i;
        }
        l0PingPong_ = 0;
        blockPingPong_ = 0;
        subBlockId_ = 0;
    }

    // Single matmul per block: A(blkM x K) @ rot(K x K) -> L0C(float) -> Fixpipe(F322BF16) -> UB(bfloat16).
    // gmA carries the row pitch in its GM layout (B > 1: b_ * k_, otherwise k_), CopyGM2L1 derives Nd2Nz from it.
    template <typename TensorA, typename TensorRot>
    __aicore__ inline void operator()(TensorA gmA, TensorRot gmRot, TupleL1L0Shape blockShape, bool isFirstRound)
    {
        uint64_t curML1 = asc::te::get<Gemm::MNK_M>(blockShape);
        uint64_t curNL1 = asc::te::get<Gemm::MNK_N>(blockShape);
        uint64_t curKL1 = asc::te::get<Gemm::MNK_K>(blockShape);

        // A and per-batch rotation ping-pong by block. Shared rotation remains in slot 0.
        // A and rotation use independent L1 mutex IDs so a cached rotation can outlive an A block.
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / Gemm::DOUBLE_BUFFER_COUNT;
        const uint16_t blockSlot = blockPingPong_ & 0x1;
        const uint16_t rotL1BufferId = b_ > 1 ? blockSlot : 0;
        const uint64_t l0cOffset = HALF_L0C_SIZE * blockSlot;

        uint64_t al1OffsetBytes = aL1Buffer_[blockSlot];

        auto layoutAL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                    asc::te::layout_trait_default<AElementType>>(curML1, curKL1);
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AElementType>(al1OffsetBytes), layoutAL1);
        CopyGM2L1(tensorAL1, gmA, blockSlot + Gemm::L1_BUFFER_ID_BASE);

        // B == 1: retain rot in L1 until the caller requests a reload, including group changes.
        // B > 1: reload every block to populate its selected ping-pong slot.
        if (isFirstRound || b_ > 1) {
            auto layoutRotL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                          asc::te::layout_trait_default<BElementType>>(curKL1, curNL1);
            auto tensorRotL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BElementType>(rotL1Buffer_[rotL1BufferId]), layoutRotL1);
            CopyGM2L1(tensorRotL1, gmRot, rotL1BufferId + ROT_L1_BUFFER_ID_BASE);
        }

        uint64_t kL0Iter = Gemm::CeilDiv(curKL1, baseK_);
        uint64_t curNL0 = asc::te::get<5>(blockShape); // baseN
        for (uint64_t iter = 0; iter < kL0Iter; ++iter) {
            uint64_t curK0 = (iter + 1 == kL0Iter) ? (curKL1 - iter * baseK_) : baseK_;
            const uint16_t l0BufferId = l0PingPong_ & 0x1;

            auto layoutRotL1 = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                          asc::te::layout_trait_default<BElementType>>(curKL1, curNL0);
            auto tensorRotL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BElementType>(rotL1Buffer_[rotL1BufferId]), layoutRotL1);
            auto l1Tensors = AscendC::Std::make_tuple(tensorAL1, tensorRotL1);
            auto l0Shape = AscendC::Std::make_tuple(curML1, curNL0, curK0, iter * baseK_, 0UL);
            auto l0Tensors = CopyL1ToL0(l1Tensors, l0Shape, l0BufferId, blockSlot, rotL1BufferId);

            auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(
                curML1, curNL0);
            auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType>(l0cOffset),
                                                  layoutL0C);
            auto mnkShape = AscendC::Std::make_tuple(curML1, curNL0, curK0);
            Mmad(tensorL0C, l0Tensors, mnkShape, asc::te::unit_flag_mode::disable, iter == 0, l0BufferId, blockSlot);
            l0PingPong_++;
        }

        // Fixpipe reads the same L0C buffer that Mmad wrote. Explicit M/FIX locks order
        // producer/consumer access and buffer reuse; the other L0C buffer permits overlap.
        // Cross-core PIPE_FIX waits protect the destination AIV UB, not the M pipeline.
        uint64_t nAlign = Gemm::CeilAlign(curNL0, Gemm::BLOCK_CUBE);
        auto layoutL0COut = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<Gemm::C0_SIZE_L0C>>(
            curML1, curNL0);
        auto tensorL0COut = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType>(l0cOffset),
                                                 layoutL0COut);
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(curML1, nAlign);
        auto tensorUB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(0), layoutUB);
        auto ubTensors = AscendC::Std::make_tuple(tensorUB, tensorL0COut);
        FixpipeL0CToUB(ubTensors, blockSlot);
        ++blockPingPong_;
    }

private:
    static constexpr uint16_t ROT_L1_BUFFER_ID_BASE = Gemm::L1_BUFFER_ID_BASE + Gemm::DOUBLE_BUFFER_COUNT;

    template <typename TensorDst, typename TensorSrc>
    __aicore__ inline void CopyGM2L1(TensorDst& dst, const TensorSrc& src, uint16_t bufferId)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        Gemm::LockMte2(bufferId);
        asc::te::copy(copyGM2L1, dst, src);
        Gemm::UnLockMte2(bufferId);
    }

    template <typename L1Tensors, typename L0Shape>
    __aicore__ inline auto CopyL1ToL0(const L1Tensors& l1Tensors, const L0Shape& l0Shape, uint16_t l0BufferId,
                                      uint16_t aL1BufferId, uint16_t rotL1BufferId)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / Gemm::DOUBLE_BUFFER_COUNT;
        uint64_t l0Offset = HALF_L0_SIZE * l0BufferId;
        const auto& tensorAL1 = asc::te::get<0>(l1Tensors);
        const auto& tensorBL1 = asc::te::get<1>(l1Tensors);
        auto curML0 = asc::te::get<0>(l0Shape);
        auto curNL0 = asc::te::get<1>(l0Shape);
        auto curK0 = asc::te::get<2>(l0Shape);
        auto kOffset = asc::te::get<3>(l0Shape);
        auto bRowOffset = asc::te::get<4>(l0Shape);

        auto layoutL0A = asc::te::make_frame_layout<asc::te::nz_layout_ptn,
                                                    asc::te::layout_trait_default<AElementType>>(curML0, curK0);
        auto tensorL0A = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AElementType>(l0Offset),
                                              layoutL0A);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kOffset), asc::te::make_shape(curML0, curK0));
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        Gemm::LockMte1(aL1BufferId + Gemm::L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
        asc::te::copy(copyL12L0A, tensorL0A, tensorBlockAL1);
        Gemm::UnLockMte1(aL1BufferId + Gemm::L1_BUFFER_ID_BASE);

        auto layoutL0B = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                    asc::te::layout_trait_default<BElementType>>(curK0, curNL0);
        auto tensorL0B = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BElementType>(l0Offset),
                                              layoutL0B);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bRowOffset + kOffset, 0),
                                              asc::te::make_shape(curK0, curNL0));
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        Gemm::LockMte1(rotL1BufferId + ROT_L1_BUFFER_ID_BASE);
        asc::te::copy(copyL12L0B, tensorL0B, tensorBlockBL1);
        Gemm::UnLockMte1(rotL1BufferId + ROT_L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + Gemm::L0_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorL0A, tensorL0B);
    }

    template <typename TensorL0C, typename L0Tensors, typename MnkShape>
    __aicore__ inline void Mmad(TensorL0C& tensorL0C, const L0Tensors& l0Tensors, const MnkShape& mnkShape,
                                asc::te::unit_flag_mode unitFlag, bool cmatrixInitVal, uint16_t l0BufferId,
                                uint16_t l0cBufferId)
    {
        const auto& tensorL0A = asc::te::get<0>(l0Tensors);
        const auto& tensorL0B = asc::te::get<1>(l0Tensors);
        auto curM = asc::te::get<0>(mnkShape);
        auto curN = asc::te::get<1>(mnkShape);
        auto curK = asc::te::get<2>(mnkShape);

        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK), unitFlag, cmatrixInitVal};
        Gemm::LockM(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
        Gemm::LockM(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
        asc::te::mmad(asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation>>{}.with(mmadParams), tensorL0C,
                      tensorL0A, tensorL0B);
        Gemm::UnLockM(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
        Gemm::UnLockM(l0BufferId + Gemm::L0_BUFFER_ID_BASE);
    }

    template <typename TensorsTuple>
    __aicore__ inline void FixpipeL0CToUB(const TensorsTuple& tensors, uint16_t l0cBufferId)
    {
        const auto& dst = asc::te::get<0>(tensors);
        const auto& src = asc::te::get<1>(tensors);

        asc::te::l0c_to_ub_params fixpipeParams{asc::te::unit_flag_mode::disable,
                                                static_cast<uint8_t>((subBlockId_++) & 0x1)};
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        Gemm::LockFix(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
        asc::te::copy(copyL0C2UB.with(fixpipeParams), dst, src);
        Gemm::UnLockFix(l0cBufferId + Gemm::L0C_BUFFER_ID_BASE);
    }

    uint64_t b_{1};
    uint64_t baseK_{16};
    uint64_t l0PingPong_{1};
    uint64_t blockPingPong_{0};
    uint64_t subBlockId_{0};

    uint64_t aL1Buffer_[Gemm::DOUBLE_BUFFER_COUNT] = {0};
    uint64_t rotL1Buffer_[Gemm::DOUBLE_BUFFER_COUNT] = {0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
