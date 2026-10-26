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
 * \file block_mmad_matmul_al1_full_load.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/block/block_mmad_decl.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t FullLoadMode_, uint64_t FusedOpType_, class KernelSchedule_, class AType_, class LayoutA_,
          class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockAFullLoad<FullLoadMode_, FusedOpType_, KernelSchedule_>, AType_, LayoutA_, BType_,
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
    using DispatchPolicy = MatmulMultiBlockAFullLoad<FullLoadMode_, FusedOpType_, KernelSchedule_>;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = DispatchPolicy::NON_CONTIGUOUS_TYPE;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TripleShape = asc::te::shape<int64_t, int64_t, int64_t>;

    // TRANS_A and TRANS_B
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;
    // AL1 Layout
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;
    // BL1 Layout
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    // kernel params
    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t oriK{0};
        uint64_t mL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        uint64_t rowStride{0};
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
        oriK_ = params.oriK;
        mL1_ = params.mL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        enableL0cPingPong_ = params.l0cStages > 1;
        kAlign_ = Blaze::Gemm::CeilAlign(oriK_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        // A全载
        uint64_t aL1OneSize = mL1_ * kAlign_ * sizeof(AType);
        uint64_t bL1OneSize = baseN_ * kL1_ * sizeof(BType);
        uint64_t biasL1OneSize = baseN_ * sizeof(BiasType);
        l0PingPong_ = 0;
        bL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        isAL1Loaded_ = false;
        // 2 buffer: A | B0 B1 | Bias0 Bias1
        // 4 buffer: A | B0 B1 B2 B3 | Bias0 Bias1 Bias2 Bias3
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            bL1Buffer_[i] = aL1OneSize + bL1OneSize * i;
            biasL1Buffer_[i] = aL1OneSize + bL1OneSize * l1Stages_ + biasL1OneSize * i;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& gmC,
                                      TupleShape& blockShape)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t curM = asc::te::get<MNK_M>(blockShape);
        uint64_t curN = asc::te::get<MNK_N>(blockShape);

        // A GM->L1 A全载模式
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{}, Blaze::Gemm::Tile::CopyGM2L1TraitPreAllocation{});
        auto layoutAL1 = MakeLayoutAL1{}(curM, oriK_);
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Buffer_),
                                              layoutAL1);
        if (!isAL1Loaded_) {
            Gemm::LockMte2(L1_BUFFER_ID_BASE);
            asc::te::copy(copyGM2L1, tensorAL1, gmA);
            Gemm::UnLockMte2(L1_BUFFER_ID_BASE);
            isAL1Loaded_ = true;
        }

        baseM_ = Min(curM, baseM_);
        mL1Iter_ = CeilDiv(curM, baseM_);
        kL1_ = Min(oriK_, kL1_);
        kL1Iter_ = CeilDiv(oriK_, kL1_);
        for (uint64_t iterM = 0; iterM < mL1Iter_; ++iterM) {
            auto tileM = (iterM + 1 == mL1Iter_) ? (curM - baseM_ * iterM) : baseM_;
            uint64_t l0cOffset = HALF_L0C_SIZE * (l0cPingPong_ & 0x1);
            // LoC搬出
            auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<16>>{}(tileM, curN);
            auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                                  layoutL0C);

            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                auto curKL1 = (iter0 + 1 == kL1Iter_) ? (oriK_ - kL1_ * iter0) : kL1_;
                // A常驻L1，B/Bias按2或4个stage轮转
                uint16_t bL1BufId = bL1LoopCnt_ & (l1Stages_ - 1);
                uint16_t btBufId = bL1LoopCnt_ & 0x1;

                // GM->L1
                TripleShape l1Shape{tileM, curN, curKL1};
                auto l1TensorTuple = CopyL1FromGM(gmB, gmBias, l1Shape, bL1BufId, iter0);
                auto tensorBL1 = asc::te::get<0>(l1TensorTuple);
                auto tensorBiasL1 = asc::te::get<1>(l1TensorTuple);

                uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    uint16_t l0BufferId = l0PingPong_ & 0x1;

                    uint64_t aL1MOffset = iterM * baseM_;
                    uint64_t aL1KOffset = iter0 * kL1_ + iter1 * baseK_;
                    uint64_t bL1KOffset = iter1 * baseK_;
                    uint64_t bL1NOffset = 0UL;

                    // A L1->L0
                    TripleShape l0Shape{tileM, curN, curK0};
                    bool needBias = NeedProcessBias(iter0, iter1);
                    auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0BufferId, needBias,
                                                      bL1BufId, btBufId, aL1MOffset, aL1KOffset, bL1KOffset,
                                                      bL1NOffset);
                    auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                    auto tensorBL0 = asc::te::get<1>(l0TensorTuple);
                    auto tensorBiasL0 = asc::te::get<2>(l0TensorTuple);

                    bool initCmatrix = iter0 == 0 && iter1 == 0 && !isBias_;
                    asc::te::unit_flag_mode unitFlag = ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ?
                                                            asc::te::unit_flag_mode::enable_update :
                                                            asc::te::unit_flag_mode::enable_keep);

                    {
                        Gemm::LockM(l0BufferId + L0_BUFFER_ID_BASE);
                        Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag,
                                initCmatrix);
                        Gemm::UnLockM(l0BufferId + L0_BUFFER_ID_BASE);
                    }
                    l0PingPong_++;
                }
                bL1LoopCnt_++;
            }
            // 数据搬出到GM
            asc::te::l0c_to_gm_params fixpParams{asc::te::unit_flag_mode::enable_update};
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            auto tensorgmC = gmC.slice(asc::te::make_coord(iterM * baseM_, 0), asc::te::make_shape(tileM, curN));
            asc::te::copy(copyL0C2GM.with(fixpParams), tensorgmC, tensorL0C);

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

    template <typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(const TensorB& tensorB, const TensorBias& tensorBias,
                                        const TripleShape& l1Shape, uint16_t bL1BufferId, uint64_t kIdx)
    {
        uint64_t curN = asc::te::get<MNK_N>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{}, Blaze::Gemm::Tile::CopyGM2L1TraitPreAllocation{});

        Gemm::LockMte2(bL1BufferId + L1_BUFFER_ID_BASE);

        // B GM->L1
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Buffer_[bL1BufferId]), layoutBL1);
        auto gmTileB = tensorB.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curN));
        asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

        // Bias GM->L1
        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curN);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Buffer_[bL1BufferId]), layoutBiasL1);
        if (isBias_ && kIdx == 0) {
            asc::te::copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        Gemm::UnLockMte2(bL1BufferId + L1_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorBL1, tensorBiasL1);
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorA& tensorAL1, const TensorB& tensorBL1,
                                        const TensorBias& tensorBiasL1, const TripleShape& l0Shape, uint16_t l0BufferId,
                                        bool needBias, uint16_t bL1BufferId, uint16_t btBufferId, uint64_t aL1MOffset,
                                        uint64_t aL1KOffset, uint64_t bL1KOffset, uint64_t bL1NOffset)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0BaseOffset = HALF_L0_SIZE * l0BufferId;
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);

        Gemm::LockMte1(bL1BufferId + L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        // A L1->L0A
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BaseOffset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(aL1MOffset, aL1KOffset),
                                              asc::te::make_shape(curM, curK0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        // B L1->L0B
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK0,
                                                                                                                  curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0BaseOffset),
                                              layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bL1KOffset, bL1NOffset),
                                              asc::te::make_shape(curK0, curN));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        // Bias L1->L0
        uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl1Align);
        auto tensorBiasL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::bias, float>(baseN_ * btBufferId * sizeof(float)), layoutBiasL0);
        if (needBias) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }

        Gemm::UnLockMte1(bL1BufferId + L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + L0_BUFFER_ID_BASE);

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TripleShape& l0Shape, bool needBias,
                                   asc::te::unit_flag_mode unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        // Mmad参数
        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK0), unitFlag, initCmatrix};
        // 传入自定义Trait类型
        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    uint64_t oriK_{1};
    uint64_t mL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    uint64_t kL1Iter_{0};
    uint64_t mL1Iter_{0};
    uint64_t kAlign_{0};
    uint32_t l1Stages_{1};
    uint64_t bL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool isAL1Loaded_{false};

    uint64_t aL1Buffer_{0};
    uint64_t bL1Buffer_[4] = {0};
    uint64_t biasL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
