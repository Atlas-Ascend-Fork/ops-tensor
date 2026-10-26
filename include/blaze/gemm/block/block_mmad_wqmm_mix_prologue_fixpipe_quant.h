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
 * \file block_mmad_wqmm_mix_prologue_fixpipe_quant.h
 * \brief AIC-side block MMAD for T-CG per-group weight dequant matmul.
 *        AIV converts FP4 weights + per-group scale to FP8, writes to L1.
 *        AIC performs FP8 x FP8 matmul with per-channel output quant via yScale in fixpipe.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/block/block_mmad_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class AType_, class LayoutA_, class BTypeTuple_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_>
class BlockMmad<MatmulWithWeightQuantPergroup, AType_, LayoutA_, BTypeTuple_, LayoutB_, CType_, LayoutC_, BiasType_,
                LayoutBias_> {
public:
    static_assert(AscendC::Std::tuple_size_v<BTypeTuple_> == 2, "B type tuple must contain B and ScaleB");
    using AType = AType_;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using ScaleType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = MatmulWithWeightQuantPergroup;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using xType = AType;
    using wType = BType;
    using yType = CType;
    using scaleType = ScaleType;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;

    static_assert(AscendC::IsSameType<AType, fp8_e4m3fn_t>::value, "T-CG pergroup expects FP8 E4M3 A type");
    static_assert(IsFp4<BType>(), "T-CG pergroup expects packed FP4 B type");
    static_assert(!TRANS_A, "T-CG pergroup expects a non-transposed ND A layout (x1)");
    // The trailing BiasType_/LayoutBias_ template slots are signature placeholders only:
    // T-CG takes no bias input; the B-side second tensor is the per-group scale carried in
    // BTypeTuple and consumed by the AIV dequant in UB.
    static_assert(AscendC::Std::is_same_v<BiasType_, void> && AscendC::Std::is_same_v<LayoutBias_, void>,
                  "T-CG pergroup takes no bias; pass void for the BiasType/LayoutBias slots");

    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    using SyncProtocol = typename DispatchPolicy::SyncProtocol;

    static constexpr int32_t IDX_1 = 1;
    static constexpr int32_t IDX_2 = 2;
    static constexpr int32_t IDX_3 = 3;

    using L1TileShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using L0TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        L1TileShape l1TileShape; // (mAL1Size, nBL1Size, kAL1Size, kBL1Size)
        L0TileShape l0TileShape; // (baseM, baseN, baseK)
        uint8_t vecCoreParallel{0};
        uint16_t AL1Pingpong{0};
        uint16_t BL1Pingpong{0};
        uint32_t dbL0C{0};
    };

    __aicore__ inline BlockMmad() {}

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(true);
        }
        params_ = params;

        mAL1Size_ = asc::te::get<MNK_M>(params_.l1TileShape);
        nBL1Size_ = asc::te::get<MNK_N>(params_.l1TileShape);
        kAL1Size_ = asc::te::get<MNK_K>(params_.l1TileShape);
        kBL1Size_ = asc::te::get<IDX_3>(params_.l1TileShape);
        const int64_t maxKL1 = Max(kAL1Size_, kBL1Size_);
        kAl1Factor_ = CeilDiv(maxKL1, kBL1Size_);
        kBl1Factor_ = CeilDiv(maxKL1, kAL1Size_);

        aL1DataSize_ = mAL1Size_ * kAL1Size_;
        bL1DataSize_ = nBL1Size_ * kBL1Size_;

        baseN_ = asc::te::get<MNK_N>(params_.l0TileShape);
        baseK_ = asc::te::get<MNK_K>(params_.l0TileShape);

        if (params_.BL1Pingpong != QUADRUPLE_BUFFER) {
            if constexpr (TRANS_B) {
                vecPingpong_ = static_cast<uint8_t>(Min<int64_t>(params_.BL1Pingpong, DOUBLE_BUFFER));
            } else {
                vecPingpong_ = params_.BL1Pingpong;
            }
        }
        // Pure layout computation shared by AIC and AIV: the paired AIV prologue queries
        // the converted-weight BL1 slot addresses through GetSharedMemPtr, so the storage
        // must be initialized here in the constructor (which runs on both core types),
        // never inside the AIC-only InitAIC.
        InitL1Storage(bL1DataSize_, aL1DataSize_, params_.BL1Pingpong, params_.AL1Pingpong, vecPingpong_);
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    // Cross-core shared L1 operand tags (review Appendix A, plan 1). GetSharedMemPtr is a
    // pure address query over the L1 slot layout (no AIC-only instructions), so the AIC
    // and its paired AIVs observe the same slot addresses; slotId selects the BL1
    // ping/pong slot and must be smaller than the BL1Pingpong buffer count.
    struct WeightOperand {};

    template <class Operand>
    __aicore__ inline uint64_t GetSharedMemPtr(uint32_t slotId) const
    {
        static_assert(AscendC::Std::is_same_v<Operand, WeightOperand>, "unsupported shared operand");
        return bL1Offsets_[slotId];
    }

    __aicore__ inline void InitSync()
    {
        for (int32_t idx = 0; idx < params_.BL1Pingpong; idx++) {
            NotifyVector(idx);
        }
    }

    // Processes the single GM tensor block: stores the per-tile shape, then runs the A L1
    // generation loop and the quantized fixpipe output. The tensors are per-tile GM slices
    // (A: (validM, k), YScale: (1, validN), C: (validM, validN)) prepared by the kernel.
    // mAL1Len_/nBL1Len_ equal the tile sizes under the stepM == stepN == 1 tiling invariant.
    template <typename TensorA_, typename TensorYScale_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& gmBlockA, const TensorYScale_& gmBlockYScale,
                                      const TensorC_& gmBlockC, const BlockShape& blockShape)
    {
        baseUseM_ = asc::te::get<MNK_M>(blockShape);
        baseUseN_ = asc::te::get<MNK_N>(blockShape);
        mAL1Len_ = baseUseM_;
        nBL1Len_ = baseUseN_;
        kSize_ = asc::te::get_total_column_shape(gmBlockA.layout());
        kSingleCoreIterNum_ = CeilDiv(kSize_, Min(kAL1Size_, kBL1Size_));
        for (int32_t kGenIdx = 0; kGenIdx < kSingleCoreIterNum_; kGenIdx += static_cast<int32_t>(kAl1Factor_)) {
            const int64_t kAL1Offset = kGenIdx / kAl1Factor_ * kAL1Size_;
            const int64_t kAL1Len = Min(static_cast<int32_t>(kSize_ - kAL1Offset), static_cast<int32_t>(kAL1Size_));
            CopyAGmToL1(kAL1Offset, kAL1Len, gmBlockA);
            {
                auto aL1Lock = bufMgr_.GetL1ASlot(curAL1BufIdx_).LockMte1();
                int32_t genEnd = Min(kGenIdx + static_cast<int32_t>(kAl1Factor_),
                                     static_cast<int32_t>(kSingleCoreIterNum_));
                for (int32_t kFactorIdx = kGenIdx; kFactorIdx < genEnd; kFactorIdx++) {
                    const int64_t kBL1Offset = kFactorIdx / kBl1Factor_ * kBL1Size_;
                    const int64_t kBL1Len = Min(static_cast<int32_t>(kSize_ - kBL1Offset),
                                                static_cast<int32_t>(kBL1Size_));
                    WaitBL1(kFactorIdx);
                    IterateMatmul(kFactorIdx, kAL1Len, kBL1Len);
                    PostProcess(kFactorIdx);
                }
            }
            curAL1BufIdx_ = (curAL1BufIdx_ + 1) % params_.AL1Pingpong;
        }
        FixpipeOutput(gmBlockYScale, gmBlockC);
    }

    __aicore__ inline void InitAIC()
    {
        static constexpr uint64_t L1_BUFFER_SIZE = AscendC::TOTAL_L1_SIZE;
        for (uint32_t i = 0; i < QUADRUPLE_BUFFER; i++) {
            bufMgr_.InitAL1(i, aL1Offsets_[i], BufferLayout::L1ADataBufferId(i));
            bufMgr_.InitBL1(i, bL1Offsets_[i], BufferLayout::L1BDataBufferId(i));
        }
        dbL0C_ = (params_.dbL0C > 0) ? params_.dbL0C : 2;
        bufMgr_.InitL0();
        bufMgr_.InitL0C();
        bufMgr_.InitScaleL1(L1_BUFFER_SIZE - baseN_ * sizeof(uint64_t));
    }

private:
    __aicore__ inline void InitL1Storage(int64_t bL1DataSize, int64_t aL1DataSize, uint16_t bl1Pingpong,
                                         uint16_t al1Pingpong, uint8_t vecPingpong)
    {
        static constexpr uint64_t L1_BUFFER_HALF_SIZE = AscendC::TOTAL_L1_SIZE >> 1;
        if (bl1Pingpong == QUADRUPLE_BUFFER) {
            int32_t bL1DataSizeTotal = DOUBLE_BUFFER * bL1DataSize * sizeof(AType);
            if (al1Pingpong == QUADRUPLE_BUFFER) {
                bL1Offsets_[0] = 0;
                bL1Offsets_[2] = bL1DataSize * sizeof(AType);
                bL1Offsets_[1] = L1_BUFFER_HALF_SIZE;
                bL1Offsets_[IDX_3] = L1_BUFFER_HALF_SIZE + bL1DataSize * sizeof(AType);
                aL1Offsets_[0] = L1_BUFFER_HALF_SIZE + bL1DataSizeTotal;
                aL1Offsets_[2] = L1_BUFFER_HALF_SIZE + bL1DataSizeTotal + aL1DataSize * sizeof(AType);
                aL1Offsets_[1] = bL1DataSizeTotal;
                aL1Offsets_[IDX_3] = bL1DataSizeTotal + aL1DataSize * sizeof(AType);
            } else if (al1Pingpong == DOUBLE_BUFFER) {
                bL1Offsets_[0] = 0;
                bL1Offsets_[2] = bL1DataSize * sizeof(AType);
                bL1Offsets_[1] = L1_BUFFER_HALF_SIZE;
                bL1Offsets_[IDX_3] = L1_BUFFER_HALF_SIZE + bL1DataSize * sizeof(AType);
                aL1Offsets_[0] = L1_BUFFER_HALF_SIZE + bL1DataSizeTotal;
                aL1Offsets_[1] = bL1DataSizeTotal;
            } else {
                bL1Offsets_[0] = 0;
                bL1Offsets_[2] = bL1DataSize * sizeof(AType);
                bL1Offsets_[IDX_3] = bL1Offsets_[1] + bL1DataSize * sizeof(AType);
                aL1Offsets_[0] = bL1DataSizeTotal;
                aL1Offsets_[1] = Blaze::Gemm::Max(L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + aL1DataSize * sizeof(AType));
            }
        } else {
            bL1Offsets_[0] = 0;
            if (bl1Pingpong == DOUBLE_BUFFER) {
                bL1Offsets_[1] = bL1DataSize * sizeof(AType);
            }
            aL1Offsets_[0] = vecPingpong * bL1DataSize * sizeof(AType);
            if (al1Pingpong == QUADRUPLE_BUFFER) {
                aL1Offsets_[1] = (aL1DataSize + bl1Pingpong * bL1DataSize) * sizeof(AType);
                aL1Offsets_[2] = (aL1DataSize * IDX_2 + bl1Pingpong * bL1DataSize) * sizeof(AType);
                aL1Offsets_[IDX_3] = (aL1DataSize * IDX_3 + bl1Pingpong * bL1DataSize) * sizeof(AType);
            } else if (al1Pingpong == DOUBLE_BUFFER) {
                aL1Offsets_[1] = (aL1DataSize + bl1Pingpong * bL1DataSize) * sizeof(AType);
            }
        }
    }

    template <typename GmTensor_>
    __aicore__ inline void CopyGm2L1Nd2Nz(__cbuf__ uint8_t* dst, const GmTensor_& gmTensor)
    {
        const auto& gmLayout = gmTensor.layout();
        const int64_t nValue = asc::te::get_total_row_shape(gmLayout);
        const int64_t dValue = asc::te::get_total_column_shape(gmLayout);
        if (nValue == 0 || dValue == 0) {
            return;
        }
        auto l1Layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<xType>>(
            nValue, dValue);
        auto l1Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, xType>((uint64_t)dst),
                                             l1Layout);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        asc::te::copy(copyGM2L1, l1Tensor, gmTensor);
    }

    // Which id(s) of the paired weight-flag group (FLAG, FLAG + FLAG_ID_MAX) a wait/set
    // operates on: BASE_ONLY = the base id (single AIV, or slot 0 of the BL1 double
    // buffer), OFFSET_ONLY = the shadow id (slot 1), BOTH = both ids, one per paired AIV
    // in the quad / K-split modes (signalled shadow first, legacy order).
    enum class WeightFlagSel : uint8_t { BASE_ONLY, OFFSET_ONLY, BOTH };

    template <uint64_t FLAG, WeightFlagSel SEL = WeightFlagSel::BOTH>
    __aicore__ inline void WaitWeightFlag() const
    {
        if constexpr (SEL == WeightFlagSel::BASE_ONLY) {
            AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
        } else if constexpr (SEL == WeightFlagSel::OFFSET_ONLY) {
            AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
        } else {
            AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
            AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
        }
    }

    template <uint64_t FLAG, WeightFlagSel SEL = WeightFlagSel::BOTH>
    __aicore__ inline void SetWeightFlag() const
    {
        if constexpr (SEL == WeightFlagSel::BASE_ONLY) {
            AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
        } else if constexpr (SEL == WeightFlagSel::OFFSET_ONLY) {
            AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
        } else {
            AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG + SyncProtocol::FLAG_ID_MAX);
            AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE1>(FLAG);
        }
    }

    __aicore__ inline void WaitForVector(uint64_t bL1BufIdx)
    {
        if (likely(params_.BL1Pingpong == QUADRUPLE_BUFFER)) {
            WaitWeightFlag<SyncProtocol::AIV_READY_FLAG>();
            return;
        }

        if (unlikely((params_.BL1Pingpong == 1) && (params_.vecCoreParallel == 0))) {
            WaitWeightFlag<SyncProtocol::AIV_READY_FLAG, WeightFlagSel::BASE_ONLY>();
            return;
        }
        if (unlikely((params_.BL1Pingpong == 1) && (params_.vecCoreParallel == 1))) {
            WaitWeightFlag<SyncProtocol::AIV_READY_FLAG>();
            return;
        }

        if (likely(params_.BL1Pingpong == DOUBLE_BUFFER)) {
            if (bL1BufIdx == 1) {
                WaitWeightFlag<SyncProtocol::AIV_READY_FLAG, WeightFlagSel::OFFSET_ONLY>();
            } else {
                WaitWeightFlag<SyncProtocol::AIV_READY_FLAG, WeightFlagSel::BASE_ONLY>();
            }
        }
    }

    __aicore__ inline void NotifyVector(uint64_t bL1BufIdx)
    {
        if (likely(params_.BL1Pingpong == QUADRUPLE_BUFFER)) {
            SetWeightFlag<SyncProtocol::AIC_FREE_FLAG>();
            return;
        }

        if (unlikely((params_.BL1Pingpong == 1) && (params_.vecCoreParallel == 0))) {
            SetWeightFlag<SyncProtocol::AIC_FREE_FLAG, WeightFlagSel::BASE_ONLY>();
            return;
        }

        if (unlikely((params_.BL1Pingpong == 1) && (params_.vecCoreParallel == 1))) {
            SetWeightFlag<SyncProtocol::AIC_FREE_FLAG>();
            return;
        }

        if (likely(params_.BL1Pingpong == DOUBLE_BUFFER)) {
            if (bL1BufIdx == IDX_1) {
                SetWeightFlag<SyncProtocol::AIC_FREE_FLAG, WeightFlagSel::OFFSET_ONLY>();
            } else {
                SetWeightFlag<SyncProtocol::AIC_FREE_FLAG, WeightFlagSel::BASE_ONLY>();
            }
        }
    }

    Params params_;

    int64_t mAL1Size_;
    int64_t kAL1Size_;
    int64_t nBL1Size_;
    int64_t kBL1Size_;
    int64_t aL1DataSize_;
    int64_t bL1DataSize_;

    int64_t kSingleCoreIterNum_;

    int64_t kSize_;
    int64_t baseN_;
    int64_t baseK_;

    int64_t baseUseM_;
    int64_t baseUseN_;
    int64_t mAL1Len_;
    int64_t nBL1Len_;

    int64_t kAl1Factor_;
    int64_t kBl1Factor_;

    uint8_t curAL1BufIdx_ = 0;
    uint64_t curBL1BufIdx_ = 0;
    uint8_t vecPingpong_ = SINGLE_BUFFER;

    uint64_t bL1Offsets_[QUADRUPLE_BUFFER] = {0, 0, 0, 0};
    uint64_t aL1Offsets_[QUADRUPLE_BUFFER] = {0, 0, 0, 0};
    using BufferLayout = BufferIdLayout<QUADRUPLE_BUFFER, QUADRUPLE_BUFFER, DOUBLE_BUFFER>;
    Blaze::Gemm::BufferManager<4, 4, 2> bufMgr_; // <MaxL1ASlots, MaxL1BSlots, MaxL0Slots>: A/B pipeline, L0 ping-pong

    int64_t madLoopIdx_ = 0;
    int64_t totalKLoopIdx_ = 0;
    int32_t dbL0C_ = 2;

    template <typename TensorYScale_, typename TensorC_>
    __aicore__ inline void FixpipeOutput(const TensorYScale_& gmBlockYScale, const TensorC_& gmBlockC)
    {
        auto layoutScaleL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                        asc::te::layout_trait_default<uint64_t>>(
            1UL, static_cast<int64_t>(baseUseN_));
        auto tensorScaleL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, uint64_t>(bufMgr_.GetScaleL1Slot().Addr()), layoutScaleL1);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        {
            auto scaleL1Lock = bufMgr_.GetScaleL1Slot().LockMte2();
            asc::te::copy(copyGM2L1, tensorScaleL1, gmBlockYScale);
        }
        int32_t l0cIdx = static_cast<int32_t>(madLoopIdx_ % dbL0C_);
        const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cIdx);
        uint64_t l0cAddr = l0cSlot.Addr();
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(baseUseM_, baseUseN_);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cAddr), layoutL0C);
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        {
            auto scaleL1Lock = bufMgr_.GetScaleL1Slot().LockFix();
            auto l0cLock = l0cSlot.LockFix();
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params{}), gmBlockC, tensorL0C, tensorScaleL1);
        }
        madLoopIdx_ += 1;
        totalKLoopIdx_ = 0;
    }

    __aicore__ inline void IterateMatmul(int64_t kFactorIdx, int64_t kAL1Len, int64_t kBL1Len)
    {
        const int64_t aL1Offset = (kFactorIdx % kAl1Factor_) * kBL1Size_ *
                                  CeilAlign(mAL1Len_, static_cast<int64_t>(BLOCK_CUBE));
        const int64_t l0MAL1Size = mAL1Len_;
        const int64_t l0KAL1Size = CeilAlign(kAL1Len, static_cast<int64_t>(BLOCK_CUBE));
        const int64_t l0KBL1Size = CeilAlign(kBL1Len, static_cast<int64_t>(BLOCK_CUBE));
        int64_t l0NBL1Size = 0;
        int64_t bL1Offset = 0;
        if constexpr (TRANS_B) {
            l0NBL1Size = CeilAlign(nBL1Len_, static_cast<int64_t>(BLOCK_CUBE));
            bL1Offset = (kFactorIdx % kBl1Factor_) * kAL1Size_ * CeilAlign(nBL1Len_, static_cast<int64_t>(BLOCK_CUBE));
        } else {
            l0NBL1Size = nBL1Len_;
            bL1Offset = (kFactorIdx % kBl1Factor_) * kAL1Size_ * BLOCK_CUBE;
        }

        const uint64_t aL1Base = bufMgr_.GetL1ASlot(curAL1BufIdx_).Addr();
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Base +
                                                                static_cast<uint64_t>(aL1Offset) * sizeof(AType)),
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(l0MAL1Size,
                                                                                                     l0KAL1Size));
        using LayoutBL1 = AscendC::Std::conditional_t<
            TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
            asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;
        const uint64_t bL1Base = bufMgr_.GetL1BSlot(curBL1BufIdx_).Addr();
        auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(
                                                  bL1Base + static_cast<uint64_t>(bL1Offset) * sizeof(AType)),
                                              LayoutBL1{}(l0KBL1Size, l0NBL1Size));

        const int64_t alignedM = CeilAlign(baseUseM_, static_cast<int64_t>(BLOCK_CUBE));
        const int64_t alignedN = CeilAlign(baseUseN_, static_cast<int64_t>(BLOCK_CUBE));
        const int64_t l0EffectiveK = (l0KAL1Size < l0KBL1Size) ? l0KAL1Size : l0KBL1Size;
        const int32_t l0cIdx = static_cast<int32_t>(madLoopIdx_ % dbL0C_);
        IterateL0Mmad(kFactorIdx, alignedM, alignedN, l0cIdx, l0EffectiveK, tensorAL1, tensorBL1);
    }

    template <typename TensorAL1_, typename TensorBL1_>
    __aicore__ inline void IterateL0Mmad(int64_t kFactorIdx, int64_t alignedM, int64_t alignedN, int32_t l0cIdx,
                                         int64_t l0EffectiveK, const TensorAL1_& tensorAL1, const TensorBL1_& tensorBL1)
    {
        const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cIdx);
        uint64_t l0cAddr = l0cSlot.Addr();
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(baseUseM_, baseUseN_);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cAddr), layoutL0C);

        int32_t stepK = CeilDiv(l0EffectiveK, static_cast<int64_t>(baseK_));
        int32_t kFractalIdx = static_cast<int32_t>(totalKLoopIdx_) * stepK;
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        for (int32_t kL0Idx = 0; kL0Idx < stepK; kL0Idx++) {
            int32_t baseK = (kL0Idx == stepK - 1) ? l0EffectiveK - kL0Idx * baseK_ : baseK_;
            const auto& l0Slot = bufMgr_.GetL0Slot(kFractalIdx & 1);

            uint64_t l0aAddr = l0Slot.Addr();
            uint64_t l0bAddr = l0Slot.Addr();
            auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
                alignedM, baseK);
            auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0aAddr),
                                                  layoutAL0);
            auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>(
                baseK, alignedN);
            auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0bAddr),
                                                  layoutBL0);

            {
                auto l0Lock = l0Slot.LockMte1();
                auto l1BlockA = tensorAL1.slice(asc::te::make_coord(0, static_cast<int64_t>(kL0Idx) * baseK_),
                                                asc::te::make_shape(alignedM, static_cast<int64_t>(baseK)));
                asc::te::copy(copyL12L0A, tensorAL0, l1BlockA);
                auto l1BlockB = tensorBL1.slice(asc::te::make_coord(static_cast<int64_t>(kL0Idx) * baseK_, 0),
                                                asc::te::make_shape(static_cast<int64_t>(baseK), alignedN));
                asc::te::copy(copyL12L0B, tensorBL0, l1BlockB);
            }
            {
                auto l0Lock = l0Slot.LockM();
                auto l0cLock = l0cSlot.LockM();
                asc::te::mmad_params mmadParams{static_cast<uint16_t>(alignedM), static_cast<uint16_t>(alignedN),
                                                static_cast<uint16_t>(baseK), asc::te::unit_flag_mode::disable,
                                                (kFactorIdx == 0 && kL0Idx == 0)};
                asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
            }
            kFractalIdx++;
        }
        totalKLoopIdx_ += 1;
    }

    template <typename TensorA_>
    __aicore__ inline void CopyAGmToL1(int64_t kAL1Offset, int64_t kAL1Len, const TensorA_& gmBlockA)
    {
        auto gmBlockAK = gmBlockA.slice(asc::te::make_coord(static_cast<int64_t>(0), kAL1Offset),
                                        asc::te::make_shape(mAL1Len_, kAL1Len));
        CopyGm2L1(curAL1BufIdx_, gmBlockAK);
    }

    template <typename TensorA_>
    __aicore__ inline void CopyGm2L1(int aL1Idx, const TensorA_& gmTensor)
    {
        __cbuf__ uint8_t* dstPtr = (__cbuf__ uint8_t*)bufMgr_.GetL1ASlot(aL1Idx).Addr();
        {
            auto aL1Lock = bufMgr_.GetL1ASlot(aL1Idx).LockMte2();
            CopyGm2L1Nd2Nz(dstPtr, gmTensor);
        }
    }

    __aicore__ inline void WaitBL1(int64_t kFactorIdx)
    {
        if (kFactorIdx % kBl1Factor_ != 0) {
            return;
        }
        WaitForVector(curBL1BufIdx_);
    }

    __aicore__ inline void PostProcess(int32_t kFactorIdx)
    {
        if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
            NotifyVector(curBL1BufIdx_);
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % params_.BL1Pingpong;
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
