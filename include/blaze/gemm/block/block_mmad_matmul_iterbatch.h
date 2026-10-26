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
 * \file block_mmad_matmul_iterbatch.h
 * \brief MMAD block for the IterBatch path (ON_THE_FLY and ND_FIXPIPE_1_2).
 *        Tensor API pipeline:
 *        - 3D frame layouts, A/B multiple batches loaded into L1 per loop iteration
 *        - MNK tiling within each batch step (baseM/baseN/baseK may be smaller than M/N/K)
 *        - L0A/L0B ping-pong on K axis, L0C ping-pong on MN axis (independent double buffering)
 *        - Single bias row [N] loaded once per MN tile, applied to every batch (SING_BIAS)
 *        - Batched fixpipe for multiple batches per MNK tile
 *        GM→L1: kernel layer slices gmA/gmB at the batch group start before calling MMAD
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_decl.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, MatMulL0C2Out FixpOpt_>
class BlockMmad<MatmulIterBatch<FixpOpt_>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
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
    using DispatchPolicy = MatmulIterBatch<FixpOpt_>;
    static constexpr bool IS_MIX = (FixpOpt_ != MatMulL0C2Out::ON_THE_FLY);
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    // 1:2 flag scheme: ready = slot * FLAG_ID_MAX, release = SYNC_OFFSET.
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint16_t SYNC_OFFSET = 2;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t m{0};
        uint64_t n{0};
        uint64_t k{0};
        uint64_t baseM{0};
        uint64_t baseN{0};
        uint64_t baseK{0};
        uint64_t iterBatchL1{1};
        uint64_t iterBatchL0{1};
    };

private:
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr uint64_t BUFFER_NUM = 2;

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
        m_ = params.m;
        n_ = params.n;
        k_ = params.k;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        isBias_ = params.biasGmAddr != nullptr;
        mainIterBatchL1_ = params.iterBatchL1;
        mainIterBatchL0_ = params.iterBatchL0;
        loadBufId_ = 0;
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(AType);
        if constexpr (!TRANS_A) {
            aL1BatchStrideElems_ = CeilAlign(m_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(k_, c0Size);
        } else {
            aL1BatchStrideElems_ = CeilAlign(m_, c0Size) * CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        if constexpr (!TRANS_B) {
            bL1BatchStrideElems_ = CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(n_, c0Size);
        } else {
            bL1BatchStrideElems_ = CeilAlign(k_, c0Size) * CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        aL1OneBuffer_ = aL1BatchStrideElems_ * sizeof(AType) * mainIterBatchL1_;
        bL1OneBuffer_ = bL1BatchStrideElems_ * sizeof(BType) * mainIterBatchL1_;
        // L1 layout: [bias0 | bias1 | A0 | A1 | B0 | B1]; A/bias share one event id per half, B rides its own.
        uint64_t biasL1OneSlot = CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(BiasType);
        uint64_t biasL1Total = isBias_ ? biasL1OneSlot * BUFFER_NUM : 0;
        using L1IdLayout = BufferIdLayout<BUFFER_NUM, BUFFER_NUM, BUFFER_NUM>;
        for (uint32_t i = 0; i < BUFFER_NUM; ++i) {
            bufMgr_.InitBias(i, biasL1OneSlot * i, L1IdLayout::L1ADataBufferId(i));
            bufMgr_.InitAL1(i, biasL1Total + aL1OneBuffer_ * i, L1IdLayout::L1ADataBufferId(i));
            bufMgr_.InitBL1(i, biasL1Total + aL1OneBuffer_ * BUFFER_NUM + bL1OneBuffer_ * i,
                            L1IdLayout::L1BDataBufferId(i));
        }
        // BT ping-pong slots, one per N block parity
        bufMgr_.InitBT(sizeof(float) * baseN_);
        // L0 ping-pong slots (event ids 6/7)
        bufMgr_.InitL0();
        // L0C ping-pong slots (event ids 8/9)
        bufMgr_.InitL0C();
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmCurA, const TensorB& gmCurB, const TensorA& gmNextA,
                                      const TensorB& gmNextB, const TensorBias& gmBias, const TensorC& gmC,
                                      uint64_t curIterBatchL1, uint64_t nextIterBatchL1, bool isPreLoadRound,
                                      bool isFinalRound)
    {
        uint64_t l1BufId = loadBufId_ & 0x1;
        const auto& aL1Slot = bufMgr_.GetL1ASlot(l1BufId);
        const auto& bL1Slot = bufMgr_.GetL1BSlot(l1BufId);
        const auto& biasL1Slot = bufMgr_.GetL1BiasSlot(l1BufId);
        if (isPreLoadRound) {
            CopyL1FromGM(aL1Slot, bL1Slot, biasL1Slot, gmCurA, gmCurB, gmBias, curIterBatchL1);
        }
        if (!isFinalRound) {
            CopyL1FromGM(bufMgr_.GetL1ASlot(l1BufId ^ 0x1), bufMgr_.GetL1BSlot(l1BufId ^ 0x1),
                         bufMgr_.GetL1BiasSlot(l1BufId ^ 0x1), gmNextA, gmNextB, gmBias, nextIterBatchL1);
        }
        {
            auto al1Tensor = MakeL1TensorA(aL1Slot.Addr(), curIterBatchL1);
            auto bl1Tensor = MakeL1TensorB(bL1Slot.Addr(), curIterBatchL1);
            auto biasL1Tensor = MakeL1TensorBias(biasL1Slot.Addr());

            {
                // The whole L1 half stays claimed through the loop nest (the next preload targets the other half).
                auto l1LockA = aL1Slot.LockMte1();
                auto l1LockB = bL1Slot.LockMte1();
                uint64_t batchStepCnt = CeilDiv(curIterBatchL1, mainIterBatchL0_);
                uint64_t ml0Cnt = CeilDiv(m_, baseM_);
                uint64_t nl0Cnt = CeilDiv(n_, baseN_);
                uint64_t kl0Cnt = CeilDiv(k_, baseK_);
                for (uint64_t iter1 = 0; iter1 < batchStepCnt; ++iter1) {
                    uint64_t curIterBatchL0 = (iter1 + 1 == batchStepCnt) ?
                                                  (curIterBatchL1 - mainIterBatchL0_ * iter1) :
                                                  mainIterBatchL0_;
                    uint64_t l0BatchOffset = iter1 * mainIterBatchL0_;
                    for (uint64_t iterNl0 = 0; iterNl0 < nl0Cnt; ++iterNl0) {
                        uint64_t curN = (iterNl0 == nl0Cnt - 1) ? (n_ - iterNl0 * baseN_) : baseN_;
                        // BT slot ping-pongs per N block
                        const auto& btSlot = bufMgr_.GetBTSlot(biasEventId_ & 0x1);
                        for (uint64_t iterMl0 = 0; iterMl0 < ml0Cnt; ++iterMl0) {
                            uint64_t curM = (iterMl0 == ml0Cnt - 1) ? (m_ - iterMl0 * baseM_) : baseM_;
                            const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cEventId_ % BUFFER_NUM);
                            L0TileInfo tileInfo{curIterBatchL0, curM, curN, 0, l0BatchOffset, 0, iterNl0, iterMl0};
                            {
                                // M holds the L0C half for the whole K-loop; the release signals the fixpipe below
                                auto l0cLock = l0cSlot.LockM();
                                for (uint64_t iterKl0 = 0; iterKl0 < kl0Cnt; ++iterKl0) {
                                    uint64_t curK = (iterKl0 == kl0Cnt - 1) ? (k_ - iterKl0 * baseK_) : baseK_;
                                    // L0 ping-pong runs across tiles; a per-tile parity would serialize kl0Cnt == 1
                                    // tiles.
                                    const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ % BUFFER_NUM);
                                    tileInfo.curK = curK;
                                    tileInfo.iterK = iterKl0;
                                    CopyL0FromL1(al1Tensor, bl1Tensor, biasL1Tensor, l0Slot, btSlot, tileInfo);
                                    Compute(l0Slot, l0cSlot, btSlot, tileInfo);
                                    l0PingPong_++;
                                }
                            }
                            {
                                auto l0cLock = l0cSlot.LockFix();
                                FixL0CToGM(gmC, l0cSlot, tileInfo);
                            }
                            l0cEventId_++;
                        }
                        biasEventId_++;
                    }
                }
            }
        }
        loadBufId_ ^= 0x1;
    }

private:
    struct L0TileInfo {
        uint64_t curIterBatchL0;
        uint64_t curM;
        uint64_t curN;
        uint64_t curK;
        uint64_t l0BatchOffset;
        uint64_t iterK;
        uint64_t iterN;
        uint64_t iterM;
    };
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t baseM_{1};
    uint64_t baseN_{1};
    uint64_t baseK_{1};
    uint64_t mainIterBatchL1_{1};
    uint64_t mainIterBatchL0_{1};
    bool isBias_{false};
    uint64_t loadBufId_{0};
    uint64_t l0cEventId_{0};
    uint64_t biasEventId_{0};
    uint64_t l0PingPong_{0};
    BufferManager<BUFFER_NUM, BUFFER_NUM, BUFFER_NUM> bufMgr_;
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t aL1BatchStrideElems_{1};
    uint64_t bL1BatchStrideElems_{1};

    // One group's GM->L1 load into the given half's slots.
    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline void CopyL1FromGM(const BufferSlot& aL1Slot, const BufferSlot& bL1Slot,
                                        const BufferSlot& biasL1Slot, const TensorA& gmA, const TensorB& gmB,
                                        const TensorBias& gmBias, uint64_t iterBatchL1)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        {
            auto lock = aL1Slot.LockMte2();
            asc::te::copy(copyGM2L1, MakeL1TensorA(aL1Slot.Addr(), iterBatchL1), gmA);
            if (isBias_) {
                asc::te::copy(copyGM2L1, MakeL1TensorBias(biasL1Slot.Addr()), gmBias);
            }
        }
        {
            auto lock = bL1Slot.LockMte2();
            asc::te::copy(copyGM2L1, MakeL1TensorB(bL1Slot.Addr(), iterBatchL1), gmB);
        }
    }

    __aicore__ inline auto MakeL1TensorA(uint64_t offsetAl1, uint64_t al1Count)
    {
        using LayoutAl1Ptn = AscendC::Std::conditional_t<TRANS_A, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        auto al1Layout = asc::te::make_frame_layout<LayoutAl1Ptn, asc::te::layout_trait_default<AType>>(al1Count, m_,
                                                                                                        k_);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAl1), al1Layout);
    }

    __aicore__ inline auto MakeL1TensorB(uint64_t offsetBl1, uint64_t bl1Count)
    {
        using LayoutBl1Ptn = AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        auto bl1Layout = asc::te::make_frame_layout<LayoutBl1Ptn, asc::te::layout_trait_default<BType>>(bl1Count, k_,
                                                                                                        n_);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBl1), bl1Layout);
    }

    __aicore__ inline auto MakeL1TensorBias(uint64_t offsetBiasL1)
    {
        uint64_t nAlign = CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        auto biasL1Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nAlign);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(offsetBiasL1), biasL1Layout);
    }

    // The batch-axis fold requires the K slice to span the whole matrix: multi-batch L0 implies the
    // base block covers the aligned m/n/k, and M/N/K-blocked tiles come with iterBatchL0 == 1 (host contract).
    template <typename TensorAl1, typename TensorBl1, typename TensorBiasL1>
    __aicore__ inline void CopyL0FromL1(const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
                                        const TensorBiasL1& biasL1Tensor, const BufferSlot& l0Slot,
                                        const BufferSlot& btSlot, const L0TileInfo& tileInfo)
    {
        auto l0Lock = l0Slot.LockMte1();
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{}, asc::te::l1_to_l0a_trait_default{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{}, asc::te::l1_to_l0b_trait_default{});
        // A L1->L0A (batched multi-frame)
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            tileInfo.curIterBatchL0, tileInfo.curM, tileInfo.curK);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Slot.Addr()),
                                              layoutAL0);
        auto tensorBlockAL1 = al1Tensor.slice(
            asc::te::make_coord(tileInfo.l0BatchOffset,
                                asc::te::make_coord(tileInfo.iterM * baseM_, tileInfo.iterK * baseK_)),
            asc::te::make_shape(tileInfo.curIterBatchL0, asc::te::make_shape(tileInfo.curM, tileInfo.curK)));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        // B L1->L0B (batched multi-frame)
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            tileInfo.curIterBatchL0, tileInfo.curK, tileInfo.curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Slot.Addr()),
                                              layoutBL0);
        auto tensorBlockBL1 = bl1Tensor.slice(
            asc::te::make_coord(tileInfo.l0BatchOffset,
                                asc::te::make_coord(tileInfo.iterK * baseK_, tileInfo.iterN * baseN_)),
            asc::te::make_shape(tileInfo.curIterBatchL0, asc::te::make_shape(tileInfo.curK, tileInfo.curN)));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        // Bias loads only at the first M block's first K step; later M blocks reuse the BT value.
        if (isBias_ && tileInfo.iterK == 0 && tileInfo.iterM == 0) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{},
                                                asc::te::l1_to_biastable_trait_default{});
            uint64_t nl0Align = CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl0Align);
            auto biasL0Tensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::bias, float>(btSlot.Addr()), biasL0Layout);
            auto biasL1Slice = biasL1Tensor.slice(asc::te::make_coord(0, tileInfo.iterN * baseN_),
                                                  asc::te::make_shape(1UL, tileInfo.curN));
            auto btLock = btSlot.LockMte1();
            asc::te::copy(copyL12BT, biasL0Tensor, biasL1Slice);
        }
    }

    __aicore__ inline void Compute(const BufferSlot& l0Slot, const BufferSlot& l0cSlot, const BufferSlot& btSlot,
                                   const L0TileInfo& tileInfo)
    {
        auto l0Lock = l0Slot.LockM();
        auto btLock = btSlot.LockM();
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(AType);
        uint64_t l0aPerBatchBytes = CeilAlign(tileInfo.curM, static_cast<uint64_t>(BLOCK_CUBE)) *
                                    CeilAlign(tileInfo.curK, c0Size) * sizeof(AType);
        uint64_t l0bPerBatchBytes = CeilAlign(tileInfo.curK, c0Size) *
                                    CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(BType);
        uint64_t l0cPerBatchBytes = CeilAlign(tileInfo.curM, static_cast<uint64_t>(BLOCK_CUBE)) *
                                    CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(float);
        for (uint64_t batchL0Idx = 0; batchL0Idx < tileInfo.curIterBatchL0; ++batchL0Idx) {
            ComputeMmad(tileInfo.curM, tileInfo.curN, tileInfo.curK, tileInfo.iterK,
                        l0Slot.Addr() + batchL0Idx * l0aPerBatchBytes, l0Slot.Addr() + batchL0Idx * l0bPerBatchBytes,
                        l0cSlot.Addr() + batchL0Idx * l0cPerBatchBytes, btSlot.Addr());
        }
    }

    __aicore__ inline auto MakeL0CTensor(uint64_t offset, uint64_t batchCnt, uint64_t m, uint64_t n)
    {
        auto l0cLayout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::_16>(batchCnt, m, n);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(offset), l0cLayout);
    }

    __aicore__ inline void ComputeMmad(uint64_t curM, uint64_t curN, uint64_t curK, uint64_t iterK, uint64_t al0ByteOff,
                                       uint64_t bl0ByteOff, uint64_t l0cByteOff, uint64_t biasBtAddr)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        constexpr auto mmadUnitFlag = asc::te::unit_flag_mode::disable;
        auto al0Layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(curM,
                                                                                                                  curK);
        auto al0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(al0ByteOff),
                                              al0Layout);
        auto bl0Layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK,
                                                                                                                  curN);
        auto bl0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(bl0ByteOff),
                                              bl0Layout);
        auto l0cTensor = MakeL0CTensor(l0cByteOff, 1UL, curM, curN);
        bool cmatrixInitVal = (iterK == 0 && !isBias_);
        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK), mmadUnitFlag, cmatrixInitVal};
        if (isBias_ && iterK == 0) {
            uint64_t nl0Align = CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl0Align);
            auto biasL0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(biasBtAddr),
                                                     biasL0Layout);
            asc::te::mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor, biasL0Tensor);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor);
        }
    }

    template <typename TensorC>
    __aicore__ inline void FixL0CToGM(const TensorC& gmC, const BufferSlot& l0cSlot, const L0TileInfo& tileInfo)
    {
        auto l0cOutTensor = MakeL0CTensor(l0cSlot.Addr(), tileInfo.curIterBatchL0, tileInfo.curM, tileInfo.curN);
        auto gmCSlice = gmC.slice(
            asc::te::make_coord(tileInfo.l0BatchOffset,
                                asc::te::make_coord(tileInfo.iterM * baseM_, tileInfo.iterN * baseN_)),
            asc::te::make_shape(tileInfo.curIterBatchL0, asc::te::make_shape(tileInfo.curM, tileInfo.curN)));

        if constexpr (IS_MIX) {
            // The first tile per sub-block needs no release wait; later tiles wait before overwriting its UB.
            uint16_t slot = static_cast<uint16_t>(l0cEventId_ & 0x1);
            if (l0cEventId_ > 1) {
                Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    static_cast<uint16_t>(slot * FLAG_ID_MAX + SYNC_OFFSET));
            }
            FixL0CToUB(slot, l0cSlot.Addr(), tileInfo.curIterBatchL0, tileInfo.curM, tileInfo.curN);
            Sync::NotifyVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(static_cast<uint16_t>(slot * FLAG_ID_MAX));
        } else {
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{}, asc::te::l0c_to_gm_trait_default{});
            asc::te::l0c_to_gm_params fixpParams(asc::te::unit_flag_mode::disable);
            asc::te::copy(copyL0C2GM.with(fixpParams), gmCSlice, l0cOutTensor);
        }
    }

    __aicore__ inline void FixL0CToUB(uint16_t slot, uint64_t l0cAddr, uint64_t curIterBatchL0, uint64_t curM,
                                      uint64_t curN)
    {
        uint64_t alignM = CeilAlign(curM, static_cast<uint64_t>(BLOCK_CUBE));
        uint64_t alignN = CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE));
        // Batched NZ L0C -> NDExt UB via one multi-frame fixpipe: the compact frame stride (curM * alignN)
        // packs each batch's valid rows densely, and sub_block_id = slot routes to the paired AIV's UB window.
        auto ubShape = asc::te::make_shape(curIterBatchL0,
                                           asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, alignM),
                                                               asc::te::make_shape(asc::te::_1{}, alignN)));
        auto ubStride = asc::te::make_stride(curM * alignN,
                                             asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, alignN),
                                                                  asc::te::make_stride(asc::te::_0{}, asc::te::_1{})));
        auto ubLayout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<CType>>(
            ubShape, ubStride);
        auto l0cBatched = MakeL0CTensor(l0cAddr, curIterBatchL0, curM, curN);
        auto ubBatched = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), ubLayout);
        auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{}, asc::te::l0c_to_ub_trait_default{});
        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::disable, static_cast<uint8_t>(slot)};
        asc::te::copy(copyL0C2UB.with(fixpParams), ubBatched, l0cBatched);
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
