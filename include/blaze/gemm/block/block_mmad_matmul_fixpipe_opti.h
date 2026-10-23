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
 * \file block_mmad_matmul_fixpipe_opti.h
 * \brief BlockMmad specialization for fixpipe-optimized matmul without B full-load.
 *
 * A and B both use pipeline buffering with l1Stages_ stages. Output uses fixpipe (L0C -> UB)
 * with AIC/AIV cross-core synchronization. Uses offset arrays and tile lock primitives,
 * mirroring the block_mmad_matmul_basic.h style.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t L0COutModel_, uint64_t FusedOpType_, class KernelSchedule_, class AType_, class LayoutA_,
          class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockFixpipeOpti<L0COutModel_, FusedOpType_, KernelSchedule_>, AType_, LayoutA_, BType_,
                LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockFixpipeOpti<L0COutModel_, FusedOpType_, KernelSchedule_>;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;

    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t oriK{0};
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        uint64_t splitM{0};
        uint8_t ubDB{1};
        uint64_t rowStride{0};
        uint32_t ubPitchGran{0};
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
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        splitM_ = params.splitM;
        ubDB_ = params.ubDB;
        ubPitchGran_ = params.ubPitchGran;
        enableL0cPingPong_ = params.l0cStages > 1;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        cvPingPong_ = 0;

        uint64_t aL1OneSize = mL1_ * kL1_ * sizeof(AType);
        uint64_t bL1OneSize = nL1_ * kL1_ * sizeof(BType);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        uint64_t stride = QUADRUPLE_BUFFER_COUNT / l1Stages_;
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            uint64_t base = slotSize * stride * i;
            aL1Buffer_[i] = base;
            bL1Buffer_[i] = base + aL1OneSize;
            biasL1Buffer_[i] = bL1Buffer_[i] + bL1OneSize;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& tensorC,
                                      TupleShape& tileShape)
    {
        uint64_t curM = asc::te::get<MNK_M>(tileShape);
        uint64_t curN = asc::te::get<MNK_N>(tileShape);
        uint64_t curK = asc::te::get<MNK_K>(tileShape);

        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        curBaseN_ = Min(curN, baseN_);
        nL1Iter_ = CeilDiv(curN, curBaseN_);
        const uint64_t curKL1Base = Min(curK, kL1_);
        const uint64_t kL1Iter = CeilDiv(curK, curKL1Base);
        for (uint64_t iterN = 0; iterN < nL1Iter_; ++iterN) {
            auto tileN = (iterN + 1 == nL1Iter_) ? (curN - curBaseN_ * iterN) : curBaseN_;
            uint64_t l0cOffset = HALF_L0C_SIZE * (l0cPingPong_ & 0x1);
            // L0C always describes the valid MMAD tile. Destination alignment and row pitch are physical UB
            // properties and must not enlarge the logical source shape used by the MMAD/Fixpipe unit-flag pair.
            auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<16>>{}(curM, tileN);
            auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                                  layoutL0C);

            for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
                const uint64_t curKL1 = (iter0 + 1 == kL1Iter) ? (curK - curKL1Base * iter0) : curKL1Base;
                uint16_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
                uint16_t btBufId = abL1LoopCnt_ & 0x1;

                TileShape l1Shape{curM, tileN, curKL1};
                const uint64_t kOffset = iter0 * curKL1Base;
                auto l1TensorTuple = CopyL1FromGM(gmA, gmB, gmBias, l1Shape, l1BufId, kOffset);
                auto tensorAL1 = asc::te::get<0>(l1TensorTuple);
                auto tensorBL1 = asc::te::get<1>(l1TensorTuple);
                auto tensorBiasL1 = asc::te::get<2>(l1TensorTuple);

                uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    uint16_t l0BufferId = l0PingPong_ & 0x1;

                    uint64_t aL1KOffset = iter1 * baseK_;
                    uint64_t bL1KOffset = aL1KOffset;
                    uint64_t bL1NOffset = iterN * curBaseN_;

                    TileShape l0Shape{curM, tileN, curK0};
                    bool needBias = NeedProcessBias(iter0, iter1);
                    auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0BufferId, l1BufId,
                                                      aL1KOffset, bL1KOffset, bL1NOffset, needBias, btBufId);
                    auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                    auto tensorBL0 = asc::te::get<1>(l0TensorTuple);
                    auto tensorBiasL0 = asc::te::get<2>(l0TensorTuple);

                    {
                        Gemm::LockM(l0BufferId + L0_BUFFER_ID_BASE);
                        bool initCmatrix = iter0 == 0 && iter1 == 0 && !isBias_;
                        bool isFinalK = iter0 + 1 == kL1Iter && iter1 + 1 == kL0Iter;
                        uint8_t unitFlag = isFinalK ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
                        Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag,
                                initCmatrix);
                        Gemm::UnLockM(l0BufferId + L0_BUFFER_ID_BASE);
                    }
                    l0PingPong_++;
                }
                abL1LoopCnt_++;
            }

            uint16_t slot = (ubDB_ > 1) ? static_cast<uint16_t>(cvPingPong_ & 0x1) : 0U;
            Sync::WaitForVector<Sync::SYNC_MODE_INTRA, PIPE_FIX>(Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot, splitM_);
            CopyOutFromL0C2UB(tensorC, tensorL0C, tileN, curM, slot);
            Sync::NotifyVector<Sync::SYNC_MODE_INTRA, PIPE_FIX>(Sync::FIXPIPE_AIC_READY_FLAG_BASE + slot, splitM_);
            cvPingPong_++;

            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(const TensorA& tensorA, const TensorB& tensorB, const TensorBias& tensorBias,
                                        const TileShape& l1Shape, uint16_t l1BufferId, uint64_t kOffset)
    {
        uint64_t curM = asc::te::get<MNK_M>(l1Shape);
        uint64_t curN = asc::te::get<MNK_N>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{}, Blaze::Gemm::Tile::CopyGM2L1TraitPreAllocation{});

        Gemm::LockMte2(l1BufferId + L1_BUFFER_ID_BASE);

        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Buffer_[l1BufferId]), layoutAL1);
        auto gmTileA = tensorA.slice(asc::te::make_coord(0, kOffset), asc::te::make_shape(curM, curKL1));
        asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Buffer_[l1BufferId]), layoutBL1);
        auto gmTileB = tensorB.slice(asc::te::make_coord(kOffset, 0), asc::te::make_shape(curKL1, curN));
        asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curN);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Buffer_[l1BufferId]), layoutBiasL1);
        if (isBias_ && kOffset == 0) {
            asc::te::copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        Gemm::UnLockMte2(l1BufferId + L1_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorAL1, tensorBL1, tensorBiasL1);
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorA& tensorAL1, const TensorB& tensorBL1,
                                        const TensorBias& tensorBiasL1, const TileShape& l0Shape, uint16_t l0BufferId,
                                        uint16_t l1BufferId, uint64_t aL1KOffset, uint64_t bL1KOffset,
                                        uint64_t bL1NOffset, bool needBias, uint16_t btBufferId)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0BaseOffset = HALF_L0_SIZE * l0BufferId;
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);

        Gemm::LockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BaseOffset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, aL1KOffset), asc::te::make_shape(curM, curK0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK0,
                                                                                                                  curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0BaseOffset),
                                              layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bL1KOffset, bL1NOffset),
                                              asc::te::make_shape(curK0, curN));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        uint64_t nL1Align = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nL1Align);
        auto tensorBiasL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::bias, float>(baseN_ * btBufferId * sizeof(float)), layoutBiasL0);
        if (needBias) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }

        Gemm::UnLockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorUB, typename TensorL0C>
    __aicore__ inline void CopyOutFromL0C2UB(TensorUB& tensorC, TensorL0C& tensorL0C, uint64_t tileN, uint64_t curM,
                                             uint16_t slotIdx)
    {
        using TensorUbType = asc::te::get_attribute_element_type<typename TensorUB::element_type*>;
        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::enable_update};
        uint64_t tileNAlign = GetFixpipeRowPitch<TensorUB>(tileN);
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN), tileNAlign);
        constexpr int64_t totalUbElems = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(TensorUbType));
        const int64_t ubSlotElems = ubDB_ > 1U ? totalUbElems / DOUBLE_BUFFER_COUNT : totalUbElems;
        auto ubTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub>(tensorC.data().get() + slotIdx * ubSlotElems), layoutUB);
        if (splitM_) {
            auto copyL0C2UBSplitM = asc::te::make_copy(asc::te::copy_l0c_to_ub{},
                                                       Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
            asc::te::copy(copyL0C2UBSplitM.with(fixpParams), ubTensor, tensorL0C);
        } else {
            auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
            asc::te::copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TileShape& l0Shape, bool needBias, uint8_t unitFlag,
                                   bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK0), static_cast<asc::te::unit_flag_mode>(unitFlag),
                                        initCmatrix};
        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint64_t SPLIT_M_ALIGN = 2;

    template <typename TensorUB>
    __aicore__ inline uint64_t GetFixpipeRowPitch(uint64_t tileN) const
    {
        using TensorUbType = asc::te::get_attribute_element_type<typename TensorUB::element_type*>;
        const uint64_t pitchGran = ubPitchGran_ == 0 ? static_cast<uint64_t>(asc::te::c0_element<TensorUbType>) :
                                                       static_cast<uint64_t>(ubPitchGran_);
        return Blaze::Gemm::CeilAlign(tileN, pitchGran);
    }

    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t curBaseN_{16};
    uint64_t nL1Iter_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t ubDB_{0};
    uint32_t ubPitchGran_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool splitM_{false};
    uint64_t cvPingPong_{0};

    uint64_t aL1Buffer_[4] = {0};
    uint64_t bL1Buffer_[4] = {0};
    uint64_t biasL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
