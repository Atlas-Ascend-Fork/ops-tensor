/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_matmul_syrk.h
 * \brief BlockMmad specialization for symmetric rank-k update (syrk).
 *
 * C[i, j] = A[i-rows] @ A[j-rows]^T computed for one upper-triangle slot.
 * Each A row-block is fetched from GM exactly once per k-chunk with a single
 * nd2nz CopyGM2L1: the NZ arrangement of X(m, k) is byte-identical to the ZN
 * arrangement of X^T(k, m), so one L1 image feeds both cube inputs:
 *   - L0A <- aL1Buffer NZ view (curM, k0)
 *   - L0B <- bL1Buffer ZN view (k0, curN)   [aL1Buffer itself on the diagonal]
 * The mirrored tile (j, i) is NOT computed by a second Mmad chain: the
 * single L0C accumulator is fixpiped out twice -- once nz2nd into the (i, j)
 * ND image consumed by AIV sub-block 0, once nz2dn (hardware transposed,
 * DN-layout destination) into the (j, i) image consumed by sub-block 1 --
 * so each pair still costs a single Mmad chain (halved cube work).
 *
 * In-core sync follows the block_mmad_matmul_basic.h pattern: flat L1/L0
 * offset arrays and free-function Gemm::Lock and UnLock pairs on explicit
 * buffer IDs. Both A and B copies share ONE lock scope per pipeline stage
 * (one MTE2 domain for GM->L1, one MTE1 domain for L1->L0A+L0B, one M
 * domain for Mmad) so the DMA engines pipeline the pair transfers without
 * intermediate unlock/relock boundaries.
 *
 * Kernel-side usage contract (enforced by the host tiling):
 *   - mL1 <= min(baseM, baseN) and nL1 <= min(baseM, baseN): the epilogue's
 *     single-N-chunk rule and baseM row clamp hold for the (i, j) tile.
 *   - baseM * baseN * sizeof(float) <= L0C_SIZE / 4: UB hosts both fp32
 *     accumulator images plus both AIVs' staging regions.
 *   - no per-core tail splitting (a single tail block per axis).
 *   - single UB accumulator buffer (no ping-pong): the (i, j) accumulator
 *     fixpipe lands at UB offset 0, the (j, i) DN image right after it; both
 *     AIVs' ready/free handshakes serialize the reuse across slots.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/block/block_mmad_decl.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class AType_, class LayoutA_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulSyrk, AType_, LayoutA_, AType_, LayoutA_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = MatmulSyrk;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR aGmAddr{nullptr}; // single GM source: B is A re-read via the ZN L1 view
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t oriK{0};
        // The syrk contract collapses the block geometry into ONE symmetric
        // square size: mL1 == nL1 == mL0 == nL0 (enforced by the host tiling).
        uint64_t block{0};
        uint64_t kL1{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
    };

public:
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
        k_ = params.oriK;
        mL1_ = params.block;
        nL1_ = params.block;
        kL1_ = params.kL1;
        baseM_ = params.block;
        baseN_ = params.block;
        baseK_ = params.kL0;
        l1Stages_ = params.l1Stages;
        // Single L0C accumulator slot for the (i, j) tile; the mirrored (j, i)
        // tile comes from the kernel's transposed store, so no L0C ping-pong.
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;

        uint64_t aL1OneSize = mL1_ * kL1_ * sizeof(AType);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        uint64_t stride = QUADRUPLE_BUFFER_COUNT / l1Stages_;
        // 2buffer: |APing,BPing------|APong,BPong---|   (B image follows A in
        // the same stage region; on the diagonal slot the B region stays
        // unused because both cube inputs read the A image's dual views).
        // 4buffer: |A0,B0 |A1,B1 |A2,B2 |A3,B3 |
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            uint64_t base = slotSize * stride * i;
            aL1Buffer_[i] = base;
            bL1Buffer_[i] = base + aL1OneSize;
        }
    }

    template <typename TensorA, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorC& ubC, TupleShape& blockShape, TupleShape& blockCoord,
                                      bool isDiagonal)
    {
        const uint64_t curM = asc::te::get<MNK_M>(blockShape);
        const uint64_t curN = asc::te::get<MNK_N>(blockShape);
        const uint64_t curK = asc::te::get<MNK_K>(blockShape);
        const uint64_t coordM = asc::te::get<MNK_M>(blockCoord);
        const uint64_t coordN = asc::te::get<MNK_N>(blockCoord);

        // Single L0C accumulator for the (i, j) tile (slot 0 of the L0C
        // ping-pong pair, per the singleL0cSlot contract); the (j, i) tile is
        // produced by the kernel's transposed store.
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(curM, curN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);

        kL1_ = Min(k_, kL1_);
        kL1Iter_ = CeilDiv(k_, kL1_);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1_ * iter0) : kL1_;
            // Rotate over the L1 stages: & (stages - 1) is the double-buffer
            // slot index (l1Stages_ is a power of two).
            uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);

            // Single nd2nz fetch per row-block per k-chunk, both row blocks
            // under ONE MTE2 lock domain. The NZ arrangement of X(m, k)
            // doubles as the ZN arrangement of X^T(k, m).
            auto l1TensorTuple = CopyL1FromGM(gmA, coordM, curM, coordN, curN, curKL1, l1BufId, iter0, isDiagonal);
            auto tensorAL1 = asc::te::get<0>(l1TensorTuple);
            auto tensorBL1 = asc::te::get<1>(l1TensorTuple);

            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // L0A/L0B ping-pong: alternate between the two L0 slots per
                // k step so the next Mmad overlaps the previous L1->L0 load.
                uint16_t l0BufferId = l0PingPong_ & 0x1;
                uint64_t kOffset = iter1 * baseK_;

                auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, curM, curN, curK0, curKL1, kOffset, l0BufferId,
                                                  l1BufId, isDiagonal);
                auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                auto tensorBL0 = asc::te::get<1>(l0TensorTuple);

                bool initCmatrix = iter0 == 0 && iter1 == 0;
                bool isFinal = iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter;
                {
                    Gemm::LockM(l0BufferId + L0_BUFFER_ID_BASE);
                    Compute(tensorAL0, tensorBL0, tensorL0C, curM, curN, curK0, initCmatrix, isFinal);
                    Gemm::UnLockM(l0BufferId + L0_BUFFER_ID_BASE);
                }
                l0PingPong_++;
            }
            abL1LoopCnt_++;
        }

        CopyOutFromL0C2UB(ubC, tensorL0C, curN, curM, isDiagonal);
    }

private:
    // GM -> L1 for both row blocks under one MTE2 lock domain.
    // The A row block lands at aL1Buffer_[l1BufId] (NZ view); the B row block
    // at bL1Buffer_[l1BufId] (reinterpreted as ZN for the L0B path). On the
    // diagonal slot only the A image is fetched -- its NZ arrangement already
    // doubles as the ZN arrangement of A^T, so B reads the same bytes.
    template <typename TensorRows>
    __aicore__ inline auto CopyL1FromGM(const TensorRows& tensorRows, uint64_t rowOffM, uint64_t rowsM,
                                        uint64_t rowOffN, uint64_t rowsN, uint64_t curKL1, uint16_t l1BufferId,
                                        uint64_t kIdx, bool isDiagonal)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        Gemm::LockMte2(l1BufferId + L1_BUFFER_ID_BASE);

        auto layoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>{}(
            rowsM, curKL1);
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Buffer_[l1BufferId]), layoutAL1);
        auto gmTileA = tensorRows.slice(asc::te::make_coord(rowOffM, kIdx * kL1_), asc::te::make_shape(rowsM, curKL1));
        asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

        // B image: on the diagonal it IS the A image (dual view, no second
        // fetch); off the diagonal fetch the N-row block into bL1Buffer_.
        auto layoutBL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>{}(
            rowsN, curKL1);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(bL1Buffer_[l1BufferId]), layoutBL1);
        if (!isDiagonal) {
            auto gmTileB = tensorRows.slice(asc::te::make_coord(rowOffN, kIdx * kL1_),
                                            asc::te::make_shape(rowsN, curKL1));
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);
        }

        Gemm::UnLockMte2(l1BufferId + L1_BUFFER_ID_BASE);
        return AscendC::Std::make_tuple(tensorAL1, tensorBL1);
    }

    // L1 -> L0A + L0B under one MTE1 lock domain (L1 stage + L0 slot pair).
    // A reads the L1 NZ view (rowsM, k0); B reinterprets the same L1 image
    // through a ZN layout (kL1Chunk, rowsN) -- the fractal duality makes the
    // two views byte-identical on the diagonal, and on off-diagonal slots the
    // B image is a genuine second fetch that shares the lock domain.
    template <typename TensorAL1, typename TensorBL1>
    __aicore__ inline auto CopyL0FromL1(const TensorAL1& tensorAL1, const TensorBL1& tensorBL1, uint64_t curM,
                                        uint64_t curN, uint64_t curK0, uint64_t curKL1, uint64_t kOffset,
                                        uint16_t l0BufferId, uint16_t l1BufferId, bool isDiagonal)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0BaseOffset = HALF_L0_SIZE * l0BufferId;
        Gemm::LockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        // A L1 -> L0A (NZ view)
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BaseOffset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kOffset), asc::te::make_shape(curM, curK0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        // B L1 -> L0B (ZN view of the same / second L1 image)
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>(curK0,
                                                                                                                  curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0BaseOffset),
                                              layoutBL0);
        // Off-diagonal: reinterpret the B L1 image (rowsN, curKL1 NZ) as ZN
        // (curKL1, rowsN) -- the duality NZ(X)(m,k) == ZN(X^T)(k,m) makes the
        // bytes identical. On the diagonal the A image serves both views.
        const auto& tensorBL1Zn = isDiagonal ? tensorAL1 : tensorBL1;
        auto layoutBL1Zn = asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>{}(
            curKL1, curN);
        auto tensorL1Zn = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(
                                                   isDiagonal ? aL1Buffer_[l1BufferId] : bL1Buffer_[l1BufferId]),
                                               layoutBL1Zn);
        auto tensorBlockBL1 = tensorL1Zn.slice(asc::te::make_coord(kOffset, 0), asc::te::make_shape(curK0, curN));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        Gemm::UnLockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + L0_BUFFER_ID_BASE);
        return AscendC::Std::make_tuple(tensorAL0, tensorBL0);
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, TensorC& tensorL0C,
                                   uint64_t curM, uint64_t curN, uint64_t curK0, bool initCmatrix, bool isFinal)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        // FINAL_ACCUMULATION == unit_flag_mode::enable_update, NON_FINAL == enable_keep.
        asc::te::mmad_params mmadParams{
            static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curK0),
            isFinal ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep, initCmatrix};
        asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
    }

    template <typename TensorUB, typename TensorL0C>
    __aicore__ inline void CopyOutFromL0C2UB(TensorUB& tensorC, TensorL0C& tensorL0C, uint64_t tileN, uint64_t curM,
                                             bool isDiagonal)
    {
        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::enable_keep, 0};
        uint64_t tileNAlign = Blaze::Gemm::CeilAlign(tileN, static_cast<uint64_t>(asc::te::c0_element<CType>));
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(curM, tileNAlign);
        auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(tensorC.data().get()),
                                             layoutUB);
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        Gemm::LockFix(L0C_BUFFER_ID_BASE);
        asc::te::copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);
        // L0C2UB NZ2DN: the same L0C accumulator fixpiped a second time with
        // the hardware transposed (DN-layout) destination for the (j, i)
        // mirror tile consumed by AIV sub-block 1.
        uint64_t curMAlign = Blaze::Gemm::CeilAlign(curM, static_cast<uint64_t>(asc::te::c0_element<CType>));
        auto layoutUBT = asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(curMAlign, tileN);
        auto ubTensorT = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(0), layoutUBT);
        asc::te::l0c_to_ub_params fixpParamsT{asc::te::unit_flag_mode::enable_update, 1};
        asc::te::copy(copyL0C2UB.with(fixpParamsT), ubTensorT, tensorL0C);
        Gemm::UnLockFix(L0C_BUFFER_ID_BASE);
    }

private:
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t kL1Iter_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};

    uint64_t aL1Buffer_[4] = {0};
    uint64_t bL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
