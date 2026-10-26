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
 * \file kernel_wqmm_mix_pergroup.h
 * \brief Kernel orchestration for T-CG per-group weight dequant matmul.
 *        AIV: FP4 weight + per-group scale -> FP8, write to L1 (KernelPergroupWeightPrologue).
 *        AIC: FP8 x FP8 matmul, fixpipe with per-channel yScale quant (BlockMmad).
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/block/block_mmad_wqmm_mix_prologue_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_wqmm_block_split.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/epilogue/tile/compute.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class BlockMmad_>
class KernelPergroupWeightPrologue;

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMixWithWeightPergroupPrologue, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using ScaleType = typename BlockMmad::ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC>;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct PrologueParams {
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        uint64_t groupSize{0};
        uint64_t nBubSize{0};
        uint64_t kBubSize{0};
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        PrologueParams prologueParams;
        BlockSchedulerParams schedulerParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Execute(params); }

private:
    static constexpr int32_t IDX_3 = 3;

    __aicore__ inline void Execute(const Params& params)
    {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        // Single BlockMmad instance shared by the AIC and AIV paths: its initialization from
        // mmadParams is pure layout computation (SetMMLayoutTransform only fires on the AIC),
        // so the AIV prologue queries the shared converted-weight L1 slot addresses through
        // GetSharedMemPtr with the paired AIC (review Appendix A).
        mmadOp_.Init(params.mmadParams);
        if ASCEND_IS_AIV {
            RunAiv(params, scheduler);
        }
        if ASCEND_IS_AIC {
            RunAic(params, scheduler);
        }
    }

    __aicore__ inline void RunAic(const Params& params, const BlockScheduler& scheduler)
    {
        mmadOp_.InitAIC();
        const uint64_t usedCoreNum = static_cast<uint64_t>(params.schedulerParams.cubeNumBlocksM) *
                                     static_cast<uint64_t>(params.schedulerParams.cubeNumBlocksN);
        if (GetCurrentBlockIdx() >= usedCoreNum) {
            return;
        }
        mmadOp_.InitSync();
        const int64_t m = asc::te::get<MNK_M>(params.problemShape);
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        auto gmA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)),
            MakeLayoutA{}(m, k));
        auto gmYScale = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.yScaleGmAddr)),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<uint64_t>>(
                static_cast<int64_t>(1), n));
        auto gmC = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)),
            MakeLayoutC{}(m, n));
        const uint64_t tileCount = scheduler.GetBlockNums();
        for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            const auto blockCoord = scheduler.GetBlockCoord(tileIdx);
            const auto blockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t mOffset = asc::te::get<MNK_M>(blockCoord);
            const int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
            const int64_t validM = asc::te::get<MNK_M>(blockShape);
            const int64_t validN = asc::te::get<MNK_N>(blockShape);
            auto blockA = gmA.slice(asc::te::make_coord(mOffset, 0), asc::te::make_shape(validM, k));
            auto blockYScale = gmYScale.slice(asc::te::make_coord(0, nOffset),
                                              asc::te::make_shape(static_cast<int64_t>(1), validN));
            auto blockC = gmC.slice(asc::te::make_coord(mOffset, nOffset), asc::te::make_shape(validM, validN));
            mmadOp_(blockA, blockYScale, blockC, blockShape);
        }
    }

    __aicore__ inline void RunAiv(const Params& params, const BlockScheduler& scheduler)
    {
        using BlockPrologue = KernelPergroupWeightPrologue<BlockMmad>;
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        const int64_t kAL1Size = asc::te::get<MNK_K>(params.mmadParams.l1TileShape);
        const int64_t kBL1Size = asc::te::get<IDX_3>(params.mmadParams.l1TileShape);
        const int64_t maxKL1 = Max(kAL1Size, kBL1Size);
        uint8_t vecPingpong = 1;
        if (params.mmadParams.BL1Pingpong != BlockMmad::QUADRUPLE_BUFFER) {
            if constexpr (IsTrans<LayoutB>::value) {
                vecPingpong = static_cast<uint8_t>(
                    Min(static_cast<int64_t>(params.mmadParams.BL1Pingpong), static_cast<int64_t>(2)));
            } else {
                vecPingpong = static_cast<uint8_t>(params.mmadParams.BL1Pingpong);
            }
        }
        typename BlockPrologue::Params prologueParams{params.mmadParams.vecCoreParallel,
                                                      params.mmadParams.BL1Pingpong,
                                                      static_cast<uint64_t>(k),
                                                      params.prologueParams.groupSize,
                                                      params.prologueParams.nBubSize,
                                                      params.prologueParams.kBubSize,
                                                      vecPingpong,
                                                      CeilDiv(maxKL1, kAL1Size),
                                                      CeilDiv(k, Min(kAL1Size, kBL1Size)),
                                                      kBL1Size};
        BlockPrologue blockPrologue(prologueParams, mmadOp_);
        const uint64_t usedCoreNum = static_cast<uint64_t>(params.schedulerParams.cubeNumBlocksM) *
                                     static_cast<uint64_t>(params.schedulerParams.cubeNumBlocksN);
        if (GetCurrentBlockIdx() >= usedCoreNum) {
            return;
        }
        ProcessAivTiles(params, blockPrologue, scheduler);
        blockPrologue.EndSync();
    }

    // Builds the tile's GM weight tensor view according to the B layout. ND (bTrans):
    // logical (k, n) over physical (n, k) rows. NZ: logical (k, n) NZ fractals
    // (C0 = 32 packed-FP4 elements).
    template <bool IsNdWeight>
    __aicore__ inline static auto MakeGmWeightTensor(const Params& params, int64_t k, int64_t n)
    {
        if constexpr (IsNdWeight) {
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                            reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
                                        MakeLayoutB{}(k, n));
        } else {
            return asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
                asc::te::make_frame_layout<LayoutB, AscendC::Std::Int<asc::te::c0_element<AType>>>(k, n));
        }
    }

    __aicore__ inline static auto MakeGmScaleTensor(const Params& params, int64_t kGroupTotal, int64_t n)
    {
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.scaleBGmAddr)),
            asc::te::make_frame_layout<AscendC::Std::conditional_t<IsTrans<LayoutB>::value, asc::te::dn_ext_layout_ptn,
                                                                   asc::te::nd_ext_layout_ptn>>(kGroupTotal, n));
    }

    // Drives the prologue over the current core's tiles. The GM weight/scale tensor views
    // are built here according to the B layout (ND: gmScale logical (kGroup, n) over
    // physical (n, kGroup) rows; NZ: gmScale logical (kGroup, n) over physical
    // (kGroup, n) rows); each tile passes its full-K x validN slices to the prologue.
    template <typename BlockPrologue, typename BlockScheduler>
    __aicore__ inline static void ProcessAivTiles(const Params& params, BlockPrologue& blockPrologue,
                                                  const BlockScheduler& scheduler)
    {
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        const int64_t groupSize = static_cast<int64_t>(params.prologueParams.groupSize);
        const int64_t kGroupTotal = CeilDiv(k, groupSize);
        auto gmWeight = MakeGmWeightTensor<IsTrans<LayoutB>::value>(params, k, n);
        auto gmScale = MakeGmScaleTensor(params, kGroupTotal, n);
        const uint64_t tileCount = scheduler.GetBlockNums();
        for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            const auto blockCoord = scheduler.GetBlockCoord(tileIdx);
            const auto blockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
            const int64_t validN = asc::te::get<MNK_N>(blockShape);
            const int64_t kSize = asc::te::get<MNK_K>(blockShape);
            auto gmBlockWeight = gmWeight.slice(asc::te::make_coord(static_cast<int64_t>(0), nOffset),
                                                asc::te::make_shape(kSize, validN));
            auto blockScale = gmScale.slice(asc::te::make_coord(static_cast<int64_t>(0), nOffset),
                                            asc::te::make_shape(CeilDiv(kSize, groupSize), validN));
            blockPrologue(gmBlockWeight, blockScale, blockShape);
        }
    }

    BlockMmad mmadOp_;
};

template <class BlockMmad_>
class KernelPergroupWeightPrologue {
public:
    using BlockMmad = BlockMmad_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleType = typename BlockMmad::ScaleType;
    using CType = typename BlockMmad::CType;
    using xType = AType;
    using wType = BType;
    using scaleType = ScaleType;
    using BlockCoord = typename BlockMmad::BlockCoord;
    using BlockShape = typename BlockMmad::BlockShape;
    using LayoutB = typename BlockMmad::LayoutB;

    struct Params {
        uint8_t vecCoreParallel{0};
        uint16_t BL1Pingpong{0};
        uint64_t kSize{0};
        uint64_t groupSize{0};
        uint64_t nBubSize{0};
        uint64_t kBubSize{0};
        uint8_t vecPingpong{0};
        int64_t kBl1Factor{0};
        int64_t kSingleCoreIterNum{0};
        int64_t kBL1Size{0};
    };
    // The BlockMmad reference is the cross-core shared L1 allocation owner: the converted-weight
    // BL1 slot addresses are snapshotted once here through GetSharedMemPtr (fixed after
    // BlockMmad::Init), so the prologue keeps plain addresses instead of the owner reference.
    __aicore__ inline explicit KernelPergroupWeightPrologue(const Params& prologueParams, const BlockMmad& sharedMmad)
        : prologueParams_(prologueParams)
    {
        for (uint32_t slotId = 0; slotId < QUADRUPLE_BUFFER; ++slotId) {
            sharedL1SlotAddrs_[slotId] = sharedMmad.template GetSharedMemPtr<typename BlockMmad::WeightOperand>(slotId);
        }
        uint64_t bl1Pingpong = prologueParams_.BL1Pingpong;
        uint64_t nBubSize = prologueParams_.nBubSize;
        uint64_t kBubSize = prologueParams_.kBubSize;
        uint64_t scaleOffsetLen = GetVecScaleLen();
        InitUbStorage(bl1Pingpong, prologueParams_.vecPingpong, nBubSize, kBubSize, scaleOffsetLen);
    }

    __aicore__ inline ~KernelPergroupWeightPrologue() {}

    // Processes one M/N tile: gmBlockB/gmBlockScale are the tile's GM slices (full-K x validN,
    // already positioned at the tile's absolute N offset by the kernel). Runs the K-factor loop
    // filling the converted-weight L1 buffers; internal N offsets are tile-relative.
    // nBL1Len_ equals the tile N size under the stepM == stepN == 1 tiling invariant.
    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void operator()(const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale,
                                      const BlockShape& blockShape)
    {
        baseUseN_ = asc::te::get<MNK_N>(blockShape);
        nBL1Len_ = baseUseN_;
        for (int32_t kFactorIdx = 0; kFactorIdx < prologueParams_.kSingleCoreIterNum; kFactorIdx++) {
            kBL1Offset_ = kFactorIdx / prologueParams_.kBl1Factor * prologueParams_.kBL1Size;
            kBL1Len_ = Min<int32_t>(static_cast<int32_t>(prologueParams_.kSize - kBL1Offset_),
                                    static_cast<int32_t>(prologueParams_.kBL1Size));
            nBL1Offset_ = 0;
            if (kFactorIdx % prologueParams_.kBl1Factor == 0) {
                ProduceBL1(gmBlockB, gmBlockScale);
            }
            AdvanceBL1Slot(kFactorIdx);
        }
    }

    __aicore__ inline void EndSync()
    {
        if (prologueParams_.BL1Pingpong == QUADRUPLE_BUFFER) {
            WaitForCube();
            WaitForCube();
            WaitForCube();
            WaitForCube();
            return;
        }
        if ((prologueParams_.BL1Pingpong == 1) && (prologueParams_.vecCoreParallel == 0)) {
            if (AscendC::GetSubBlockIdx() == 0) {
                WaitForCube();
            }
            return;
        }
        if ((prologueParams_.BL1Pingpong == 1) && (prologueParams_.vecCoreParallel == 1)) {
            WaitForCube();
            return;
        }
        if (prologueParams_.BL1Pingpong == DOUBLE_BUFFER) {
            WaitForCube();
            return;
        }
    }

private:
    using SyncProtocol = typename BlockMmad::SyncProtocol;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t INT4_PACK_SHIFT = 1;
    static constexpr int32_t ALIGNED_32_SIZE = 32;
    static constexpr int32_t BLOCK_NUM_REG = AscendC::VECTOR_REG_WIDTH / AscendC::ONE_BLK_SIZE;
    static constexpr int32_t AIV_NUM = 2;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4_BITS = 64;
    static constexpr int32_t GROUP_SIZE_32 = 32;

    // Returns the converted-weight BL1 slot address snapshotted at construction (the
    // cross-core shared L1 allocation is fixed after BlockMmad::Init).
    __aicore__ inline uint64_t SharedWeightL1Addr(uint64_t slotId) const { return sharedL1SlotAddrs_[slotId]; }

    // UB ping-pong storage for the per-bub conversion pipeline, slot-based (MX pattern):
    // one input slot covers the weight and scale regions of one phase; ND output slots are
    // contiguous phases, NZ output slots are 256-byte NzToNz interleave phases. Region
    // sizes carry an exact BL1Pingpong factor, so slot addresses match the former formulas.
    __aicore__ inline void InitUbStorage(uint64_t bl1Pingpong, uint64_t vecPingpong, uint64_t nBubSize,
                                         uint64_t kBubSize, uint64_t scaleOffsetLen)
    {
        uint64_t vecWeightInLen;
        uint64_t vecWeightOutLen;
        if (bl1Pingpong == QUADRUPLE_BUFFER) {
            if constexpr (IsTrans<LayoutB>::value) {
                vecWeightInLen = bl1Pingpong *
                                     (nBubSize * Blaze::Gemm::CeilAlign(kBubSize, static_cast<uint64_t>(OFFSET_64))) >>
                                 INT4_PACK_SHIFT;
                vecWeightOutLen = bl1Pingpong * (Blaze::Gemm::CeilAlign(nBubSize, Blaze::Gemm::BLOCK_CUBE) + 1) *
                                  Blaze::Gemm::CeilAlign(kBubSize, static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
            } else {
                vecWeightInLen = (bl1Pingpong * nBubSize * kBubSize) >> INT4_PACK_SHIFT;
                vecWeightOutLen = bl1Pingpong * nBubSize * kBubSize * sizeof(xType);
            }
        } else {
            if constexpr (IsTrans<LayoutB>::value) {
                vecWeightInLen = vecPingpong *
                                     (nBubSize * Blaze::Gemm::CeilAlign(kBubSize, static_cast<uint64_t>(OFFSET_64))) >>
                                 INT4_PACK_SHIFT;
                vecWeightOutLen = vecPingpong * (Blaze::Gemm::CeilAlign(nBubSize, Blaze::Gemm::BLOCK_CUBE) + 1) *
                                  Blaze::Gemm::CeilAlign(kBubSize, static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
            } else {
                vecWeightInLen = (vecPingpong * nBubSize * kBubSize) >> INT4_PACK_SHIFT;
                vecWeightOutLen = vecPingpong * nBubSize * kBubSize * sizeof(xType);
            }
        }

        const uint64_t scaleInBase = vecWeightInLen;
        const uint64_t weightOutBase = vecWeightInLen + scaleOffsetLen;
        scaleInBase_ = scaleInBase;
        singleScaleSize_ = scaleOffsetLen / bl1Pingpong;
        scaleMaskOffset_ = weightOutBase + vecWeightOutLen;
        const uint64_t singleWeightInSize = vecWeightInLen / bl1Pingpong;
        const uint64_t singleWeightOutSize = vecWeightOutLen / bl1Pingpong;
        for (uint64_t index = 0; index < QUADRUPLE_BUFFER; ++index) {
            inputSlots_[index] = {index * singleWeightInSize, static_cast<uint8_t>(index)};
            uint64_t outputOffset = weightOutBase + index * singleWeightOutSize;
            if constexpr (!IsTrans<LayoutB>::value) {
                outputOffset = weightOutBase + index * static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH);
            }
            // The output mutex ids are offset from the input ids: the V pipe locks
            // both slots per phase and the ids must stay distinct (MX convention).
            outputSlots_[index] = {outputOffset, static_cast<uint8_t>(OUTPUT_SYNC_ID_BASE + index)};
        }

        __ubuf__ uint8_t* scaleMaskUb = (__ubuf__ uint8_t*)0 + scaleMaskOffset_;
        for (uint32_t wordIdx = 0; wordIdx < SCALE_MASK_WORD_NUM; ++wordIdx) {
            *((__ubuf__ uint64_t*)scaleMaskUb + wordIdx) = SCALE_MASK_WORD;
        }
    }

    __aicore__ inline const BufferSlot& GetInputSlot(uint64_t slotId) const { return inputSlots_[slotId]; }

    __aicore__ inline const BufferSlot& GetOutputSlot(uint64_t slotId) const { return outputSlots_[slotId]; }

    __aicore__ inline __ubuf__ uint8_t* WeightIn(uint64_t slotId) const
    {
        return (__ubuf__ uint8_t*)0 + inputSlots_[slotId].Addr();
    }

    __aicore__ inline __ubuf__ uint8_t* ScaleIn(uint64_t slotId) const
    {
        return (__ubuf__ uint8_t*)0 + scaleInBase_ + slotId * singleScaleSize_;
    }

    __aicore__ inline __ubuf__ uint8_t* WeightOut(uint64_t slotId) const
    {
        return (__ubuf__ uint8_t*)0 + outputSlots_[slotId].Addr();
    }

    __aicore__ inline __ubuf__ uint8_t* ScaleMaskAddr() const { return (__ubuf__ uint8_t*)0 + scaleMaskOffset_; }

    static constexpr uint64_t OUTPUT_SYNC_ID_BASE = QUADRUPLE_BUFFER;
    // The VF scale-select mask blob: one full vector mask register, every uint64 word
    // repeating the 32-bit-on / 32-bit-off pattern. The NZ vector path loads the two
    // scale banks (BLOCK_CUBE elements apart) with the same block-distributed address
    // and Select()s the lanes with this mask, reading from alternating banks.
    static constexpr uint64_t SCALE_MASK_WORD = 0x00000000ffffffff;
    // '/ 8' converts VECTOR_REG_WIDTH (in bits) to bytes; the result is the uint64 word count of one mask register.
    static constexpr uint32_t SCALE_MASK_WORD_NUM = AscendC::VECTOR_REG_WIDTH / 8 / sizeof(uint64_t);

    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void ProduceBL1(const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale)
    {
        vecKBL1Len_ = kBL1Len_;
        vecNBL1Len_ = nBL1Len_;
        if (likely(prologueParams_.BL1Pingpong == QUADRUPLE_BUFFER)) {
            TwoVectorCoreSplit();
            VectorProcess(gmBlockB, gmBlockScale);
        } else if (prologueParams_.BL1Pingpong == DOUBLE_BUFFER) {
            if (curBL1BufIdx_ == AscendC::GetSubBlockIdx()) {
                VectorProcess(gmBlockB, gmBlockScale);
            }
        } else if (prologueParams_.vecCoreParallel == 1) {
            if (AscendC::GetSubBlockIdx() == 1) {
                kBL1Offset_ += prologueParams_.kBubSize;
                vecKBL1Len_ = Min(prologueParams_.kSize - kBL1Offset_, prologueParams_.kBubSize);
            } else {
                vecKBL1Len_ = Min(prologueParams_.kSize - kBL1Offset_, prologueParams_.kBubSize);
            }
            VectorProcess(gmBlockB, gmBlockScale);
        } else if (AscendC::GetSubBlockIdx() == 0) {
            VectorProcess(gmBlockB, gmBlockScale);
        }
    }

    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void VectorProcess(const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale)
    {
        WaitForCube();
        BL1Process(curBL1BufIdx_, nBL1Offset_, kBL1Offset_, kBL1Len_, baseUseN_, gmBlockB, gmBlockScale);
        NotifyCube();
    }

    __aicore__ inline void AdvanceBL1Slot(int32_t kFactorIdx)
    {
        if ((kFactorIdx + 1) % prologueParams_.kBl1Factor == 0 ||
            (kFactorIdx + 1) == prologueParams_.kSingleCoreIterNum) {
            curBL1BufIdx_ = (curBL1BufIdx_ + 1) % prologueParams_.BL1Pingpong;
        }
    }

    __aicore__ inline void TwoVectorCoreSplit()
    {
        uint64_t nBubFactor = CeilDiv<int64_t>(nBL1Len_, prologueParams_.nBubSize);
        uint64_t kBubFactor = CeilDiv<int64_t>(kBL1Len_, prologueParams_.kBubSize);
        if (nBubFactor > 1) {
            twoVectorCoreSplitN_ = true;
            vecNBL1Len_ = CeilDiv<int64_t>(nBubFactor, AIV_NUM) * prologueParams_.nBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                nBL1Offset_ += vecNBL1Len_;
                vecNBL1Len_ = Min<int64_t>(baseUseN_ - nBL1Offset_, nBL1Len_ - vecNBL1Len_);
            } else {
                vecNBL1Len_ = Min<int64_t>(baseUseN_ - nBL1Offset_, vecNBL1Len_);
            }
        } else if (kBubFactor > 1) {
            twoVectorCoreSplitK_ = true;
            vecKBL1Len_ = CeilDiv<int64_t>(kBubFactor, AIV_NUM) * prologueParams_.kBubSize;
            if (AscendC::GetSubBlockIdx() == 1) {
                kBL1Offset_ += vecKBL1Len_;
                vecKBL1Len_ = Min<int64_t>(prologueParams_.kSize - kBL1Offset_, kBL1Len_ - vecKBL1Len_);
            } else {
                vecKBL1Len_ = Min<int64_t>(prologueParams_.kSize - kBL1Offset_, vecKBL1Len_);
            }
        }
    }

    __aicore__ inline int64_t GetVecGroupNum(int32_t bubKLen)
    {
        return CeilDiv<int64_t>(bubKLen, prologueParams_.groupSize);
    }

    __aicore__ inline uint64_t GetVecScaleLen()
    {
        uint64_t vecScaleLen;
        if constexpr (IsTrans<LayoutB>::value) {
            vecScaleLen = prologueParams_.BL1Pingpong * prologueParams_.nBubSize *
                          CeilAlign<int64_t>(CeilDiv(prologueParams_.kBubSize, prologueParams_.groupSize),
                                             AscendC::ONE_BLK_SIZE) *
                          sizeof(scaleType);
        } else {
            vecScaleLen = prologueParams_.BL1Pingpong * CeilDiv(prologueParams_.kBubSize, prologueParams_.groupSize) *
                          prologueParams_.nBubSize * sizeof(scaleType);
        }
        return vecScaleLen;
    }

    __aicore__ inline void NotifyCube() { SetWeightFlag<SyncProtocol::AIV_READY_FLAG>(); }

    __aicore__ inline void WaitForCube() { WaitWeightFlag<SyncProtocol::AIC_FREE_FLAG>(); }

    template <uint64_t FLAG>
    __aicore__ inline void WaitWeightFlag() const
    {
        AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE3>(FLAG);
    }

    template <uint64_t FLAG>
    __aicore__ inline void SetWeightFlag() const
    {
        AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE3>(FLAG);
    }

    // Streams the packed-FP4 weight block from the tile's GM slice into the padded UB slot:
    // ND through the CopyGM2UBWeight tile (DNExt frame + Slice), NZ through the standard
    // copy_gm_to_ub trait (compact NZ pattern). The UB destinations are the Dequant input
    // contracts themselves, so the copy chain and the VF stage share one layout per pattern.
    template <typename GmBTensor_>
    __aicore__ inline void CopyInTensorWeight(const GmBTensor_& gmBlockB, int64_t bubKOffset, int64_t bubNOffset,
                                              int32_t bubKLen, int32_t bubNLen)
    {
        auto gmSlice = gmBlockB.slice(
            asc::te::make_coord(bubKOffset, bubNOffset),
            asc::te::make_shape(static_cast<int64_t>(bubKLen), static_cast<int64_t>(bubNLen)));
        __ubuf__ uint8_t* ubWeightDst = WeightIn(ubBufIdx_);
        if (bubKLen <= 0 || bubNLen <= 0) {
            return;
        }
        const int64_t kLen = static_cast<int64_t>(bubKLen);
        const int64_t nLen = static_cast<int64_t>(bubNLen);
        if constexpr (IsTrans<LayoutB>::value) {
            const int64_t kPitch = CeilAlign<int64_t>(kLen, UB_ALIGN_SIZE_FOR_4_BITS);
            auto weightInFull = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, wType>((uint64_t)ubWeightDst),
                asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(kPitch, nLen));
            auto weightIn = weightInFull.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                               asc::te::make_shape(kLen, nLen));
            auto copyGM2UB = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
            asc::te::copy(copyGM2UB, weightIn, gmSlice);
        } else {
            constexpr int64_t K0_FRACTAL = 16;
            constexpr int64_t C0_ELE = static_cast<int64_t>(Blaze::Gemm::C0_SIZE_B8);
            const int64_t k1LoopNum = CeilDiv<int64_t>(kLen, K0_FRACTAL);
            const int64_t n1LoopNum = CeilDiv<int64_t>(nLen, C0_ELE);
            auto inShape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<K0_FRACTAL>{}, k1LoopNum),
                                               asc::te::make_shape(AscendC::Std::Int<C0_ELE>{}, n1LoopNum));
            auto inStride = asc::te::make_stride(
                asc::te::make_stride(AscendC::Std::Int<C0_ELE>{}, AscendC::Std::Int<K0_FRACTAL * C0_ELE>{}),
                asc::te::make_stride(AscendC::Std::Int<1>{}, C0_ELE * kLen));
            auto weightIn = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, wType>((uint64_t)ubWeightDst),
                asc::te::make_pattern_layout<asc::te::nz_layout_ptn,
                                             asc::te::layout_trait<AscendC::Std::ignore_t, AscendC::Std::Int<C0_ELE>>>(
                    inShape, inStride));
            auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
            asc::te::copy(copyGM2UB, weightIn, gmSlice);
        }
    }

    // Streams the per-group scale block into the padded UB slot through the standard
    // copy_gm_to_ub trait's dn2dn/nd2nd routing. The UB destinations are the Dequant scale
    // input contracts (ND: DNExt frame, NZ: compact NDExt frame); NZ clamps the copied N
    // to the valid N - the stale padded tail only feeds masked padding lanes.
    template <typename GmScaleTensor_>
    __aicore__ inline void CopyInScale(const GmScaleTensor_& gmBlockScale, int64_t bubNOffset, int32_t bubNLen,
                                       int64_t bubKOffset)
    {
        const int64_t groupOffset = bubKOffset / prologueParams_.groupSize;
        const int64_t groupNum = static_cast<int64_t>(groupNumBub_);
        __ubuf__ uint8_t* ubScaleDst = ScaleIn(ubBufIdx_);
        if constexpr (IsTrans<LayoutB>::value) {
            if (bubNLen <= 0 || groupNum <= 0) {
                return;
            }
            auto gmSlice = gmBlockScale.slice(asc::te::make_coord(groupOffset, bubNOffset),
                                              asc::te::make_shape(groupNum, static_cast<int64_t>(bubNLen)));
            const int64_t scaleBytes = groupNum * static_cast<int64_t>(sizeof(scaleType));
            const int64_t groupPitch = CeilDiv<int64_t>(CeilAlign<int64_t>(scaleBytes, ALIGNED_32_SIZE),
                                                        static_cast<int64_t>(sizeof(scaleType)));
            auto scaleFull = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, scaleType>((uint64_t)ubScaleDst),
                asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(groupPitch, static_cast<int64_t>(bubNLen)));
            auto scaleTensor = scaleFull.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                               asc::te::make_shape(groupNum, static_cast<int64_t>(bubNLen)));
            auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
            asc::te::copy(copyGM2UB, scaleTensor, gmSlice);
        } else {
            const int32_t bubNLenReal = (bubNOffset + bubNLen) > baseUseN_ ?
                                            static_cast<int32_t>(baseUseN_ - bubNOffset) :
                                            bubNLen;
            if (bubNLenReal <= 0 || groupNum <= 0) {
                return;
            }
            auto gmSlice = gmBlockScale.slice(asc::te::make_coord(groupOffset, bubNOffset),
                                              asc::te::make_shape(groupNum, static_cast<int64_t>(bubNLenReal)));
            auto scaleTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, scaleType>((uint64_t)ubScaleDst),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(groupNum, static_cast<int64_t>(bubNLen)));
            auto scaleSlice = scaleTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                                asc::te::make_shape(groupNum, static_cast<int64_t>(bubNLenReal)));
            auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
            asc::te::copy(copyGM2UB, scaleSlice, gmSlice);
        }
    }

    // NZ path: builds UB tensor views over the existing byte-level patterns (NZ-fractal
    // FP4 input, per-group scale, NzToNz FP8 output) and delegates to the Epilogue::Tile
    // Dequant component. Strides keep the copy-chain padding, so the VF parameters stay
    // identical to the former hand-assembled ones.
    __aicore__ inline void DequantComputeNz(int32_t bubNLen, int32_t bubKLen, __ubuf__ int8_t* weightInBase,
                                            __ubuf__ scaleType* scaleBase, __ubuf__ xType* weightOutBase)
    {
        if (prologueParams_.groupSize != GROUP_SIZE_32) {
            return;
        }
        const int64_t kLen = static_cast<int64_t>(bubKLen);
        const int64_t nLen = static_cast<int64_t>(bubNLen);
        // NZLayoutPtn logical (k, n): n1 fractals (32 N) outer; within a fractal the
        // K rows are tight at 32 packed-FP4 elements each, so the n1 pitch is 32 * kLen
        // elements (the GM-side fractal padding is not reproduced in UB).
        constexpr int64_t K0_FRACTAL = 16;
        constexpr int64_t C0_ELE = static_cast<int64_t>(Blaze::Gemm::C0_SIZE_B8);
        const int64_t k1LoopNum = CeilDiv<int64_t>(kLen, K0_FRACTAL);
        const int64_t n1LoopNum = CeilDiv<int64_t>(nLen, C0_ELE);
        auto inShape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<K0_FRACTAL>{}, k1LoopNum),
                                           asc::te::make_shape(AscendC::Std::Int<C0_ELE>{}, n1LoopNum));
        auto inStride = asc::te::make_stride(
            asc::te::make_stride(AscendC::Std::Int<C0_ELE>{}, AscendC::Std::Int<K0_FRACTAL * C0_ELE>{}),
            asc::te::make_stride(AscendC::Std::Int<1>{}, C0_ELE * kLen));
        auto weightIn = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, wType>((uint64_t)weightInBase),
            asc::te::make_pattern_layout<asc::te::nz_layout_ptn,
                                         asc::te::layout_trait<AscendC::Std::ignore_t, AscendC::Std::Int<C0_ELE>>>(
                inShape, inStride));

        // NDExt (kGroup, n): physical kGroup rows whose pitch is the 32-aligned N size.
        const int64_t groupNum = static_cast<int64_t>(groupNumBub_);
        auto scaleTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, scaleType>((uint64_t)scaleBase),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(groupNum, nLen));

        const int64_t weightOutInterleave = static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH) *
                                            prologueParams_.BL1Pingpong;
        auto weightOut = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, xType>((uint64_t)weightOutBase),
            Blaze::Gemm::NzRowPaddingLayout<xType>{}(kLen, nLen, static_cast<uint64_t>(weightOutInterleave)));

        using Fp4ToFp8Dequant = Blaze::Epilogue::Tile::Dequant<Blaze::Epilogue::Tile::Fp4ToFp8Tag>;
        using Gran = Blaze::Epilogue::Tile::DequantMode;
        Blaze::Epilogue::Tile::A8W4TcgDequantParams dequantParams;
        dequantParams.validK = static_cast<uint32_t>(bubKLen);
        dequantParams.scaleMaskAddr = ScaleMaskAddr();
        Fp4ToFp8Dequant::Run<Gran::PER_GROUP, GROUP_SIZE_32>(weightIn, scaleTensor, weightOut, dequantParams);
    }

    // ND-trans path: builds UB tensor views over the existing byte-level patterns (DNExt
    // FP4 input, per-group DNExt scale, DnToZn FP8 output) and delegates to the
    // Epilogue::Tile Dequant component. Strides keep the copy-chain padding, so the VF
    // parameters stay identical to the former hand-assembled ones.
    __aicore__ inline void DequantComputeNdTrans(int32_t bubNLen, int32_t bubKLen, __ubuf__ int8_t* weightInBase,
                                                 __ubuf__ scaleType* scaleBase, __ubuf__ xType* weightOutBase)
    {
        if (prologueParams_.groupSize != GROUP_SIZE_32) {
            return;
        }
        const int64_t kLen = static_cast<int64_t>(bubKLen);
        const int64_t nLen = static_cast<int64_t>(bubNLen);
        const int64_t kPitch = CeilAlign<int64_t>(kLen, UB_ALIGN_SIZE_FOR_4_BITS);
        auto weightInFull = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, wType>((uint64_t)weightInBase),
            asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(kPitch, nLen));
        auto weightIn = weightInFull.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                           asc::te::make_shape(kLen, nLen));

        const int64_t groupNum = static_cast<int64_t>(groupNumBub_);
        const int64_t scaleBytes = groupNum * static_cast<int64_t>(sizeof(scaleType));
        const int64_t groupPitch = CeilDiv<int64_t>(CeilAlign<int64_t>(scaleBytes, ALIGNED_32_SIZE),
                                                    static_cast<int64_t>(sizeof(scaleType)));
        auto scaleFull = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, scaleType>((uint64_t)scaleBase),
            asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(groupPitch, nLen));
        auto scaleTensor = scaleFull.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                           asc::te::make_shape(groupNum, nLen));

        auto weightOut = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, xType>((uint64_t)weightOutBase),
            Blaze::Gemm::Weight8BitDnToZnUBLayout<xType>{}(kLen, nLen));

        using Fp4ToFp8Dequant = Blaze::Epilogue::Tile::Dequant<Blaze::Epilogue::Tile::Fp4ToFp8Tag>;
        using Gran = Blaze::Epilogue::Tile::DequantMode;
        Blaze::Epilogue::Tile::A8W4TcgDequantParams dequantParams;
        dequantParams.validK = static_cast<uint32_t>(bubKLen);
        dequantParams.scaleMaskAddr = ScaleMaskAddr();
        Fp4ToFp8Dequant::Run<Gran::PER_GROUP, GROUP_SIZE_32>(weightIn, scaleTensor, weightOut, dequantParams);
    }

    __aicore__ inline void DequantCompute(int32_t bubNLen, int32_t bubKLen)
    {
        __ubuf__ int8_t* weightInBase = (__ubuf__ int8_t*)(WeightIn(ubBufIdx_));
        __ubuf__ scaleType* scaleBase = (__ubuf__ scaleType*)(ScaleIn(ubBufIdx_));
        __ubuf__ xType* weightOutBase = (__ubuf__ xType*)(WeightOut(ubBufIdx_));

        if constexpr (IsTrans<LayoutB>::value) {
            if constexpr (!IsWeightNz<LayoutB>::value) {
                DequantComputeNdTrans(bubNLen, bubKLen, weightInBase, scaleBase, weightOutBase);
            }
        } else {
            DequantComputeNz(bubNLen, bubKLen, weightInBase, scaleBase, weightOutBase);
        }
    }

    // ND-trans path of the converted-weight UB -> L1 copy, fully tensor-driven: the L1
    // destination is a ZN-frame view of the shared slot (mirroring the AIC read side's
    // tensorBL1), sliced at the bub window's (kOffset, nOffset); kOffset must be 32-aligned
    // (ZN k1 slab boundary). The UB source follows the Dequant output layout contract.
    __aicore__ inline void CopyVecOut2L1Nd(uint64_t l1SlotAddr, int64_t kOffset, int64_t nOffset,
                                           __ubuf__ uint8_t* ubLocal, int32_t bubKLen, int32_t bubNLen)
    {
        const int64_t kLen = static_cast<int64_t>(bubKLen);
        const int64_t nLen = static_cast<int64_t>(bubNLen);
        auto l1Full = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, xType>(l1SlotAddr),
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<xType>>(
                CeilAlign<int64_t>(kBL1Len_, Blaze::Gemm::BLOCK_CUBE),
                CeilAlign<int64_t>(nBL1Len_, Blaze::Gemm::BLOCK_CUBE)));
        auto l1Tensor = l1Full.slice(asc::te::make_coord(kOffset, nOffset), asc::te::make_shape(kLen, nLen));
        auto weightOutUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, xType>((uint64_t)ubLocal),
                                                Blaze::Gemm::Weight8BitDnToZnUBLayout<xType>{}(kLen, nLen));
        auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
        asc::te::copy(copyUB2L1, l1Tensor, weightOutUb);
    }

    // NZ path of the converted-weight UB -> L1 copy, fully tensor-driven: the L1
    // destination is an NZ-frame view mirroring the AIC read side's tensorBL1, sliced at
    // the 32-aligned bub N offset (NZ n1 fractal boundary); the UB source follows the
    // Dequant output contract (NzRowPaddingLayout).
    __aicore__ inline void CopyVecOut2L1Nz(uint64_t l1SlotAddr, int64_t nOffset, __ubuf__ uint8_t* ubLocal,
                                           int32_t bubKLen, int32_t bubNLen)
    {
        const int64_t kLen = static_cast<int64_t>(bubKLen);
        const int64_t nLen = static_cast<int64_t>(bubNLen);
        auto l1Full = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, xType>(l1SlotAddr),
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<xType>>(
                CeilAlign<int64_t>(kBL1Len_, Blaze::Gemm::BLOCK_CUBE), nBL1Len_));
        auto l1Tensor = l1Full.slice(asc::te::make_coord(static_cast<int64_t>(0), nOffset),
                                     asc::te::make_shape(kLen, nLen));
        const int64_t weightOutInterleave = static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH) *
                                            prologueParams_.BL1Pingpong;
        auto weightOutUb = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, xType>((uint64_t)ubLocal),
            Blaze::Gemm::NzRowPaddingLayout<xType>{}(kLen, nLen, static_cast<uint64_t>(weightOutInterleave)));
        auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
        asc::te::copy(copyUB2L1, l1Tensor, weightOutUb);
    }

    // Shared scaffolding executor of the CopyVecOut2L1NzKSplit run patterns: builds the
    // (blockCount x 256B) NDExt byte-pattern tensors with the caller's src/dst row strides
    // and repeats the standard copy_ub_to_l1 trait runCount times, advancing l1Offset by
    // l1OffsetStep per run; the pattern quantities stay verbatim in the callers.
    __aicore__ inline void CopyNzKSplitRuns(uint64_t l1BaseBytes, int64_t l1Offset, __ubuf__ uint8_t* ubLocal,
                                            uint16_t blockCount, int64_t srcStride, int64_t dstStride, int64_t runCount,
                                            int64_t l1OffsetStep)
    {
        uint32_t blockLen = BLOCK_NUM_REG;
        if (blockCount == 0 || blockLen == 0) {
            return;
        }
        uint32_t blockLenBytes = blockLen * AscendC::ONE_BLK_SIZE;
        int64_t srcRowStrideBytes = (static_cast<int64_t>(blockLen) + srcStride) * AscendC::ONE_BLK_SIZE;
        int64_t dstRowStrideBytes = (static_cast<int64_t>(blockLen) + dstStride) * AscendC::ONE_BLK_SIZE;
        auto shape = asc::te::make_shape(
            asc::te::make_shape(AscendC::Std::Int<1>{}, static_cast<int64_t>(blockCount)),
            asc::te::make_shape(AscendC::Std::Int<1>{}, static_cast<int64_t>(blockLenBytes)));
        auto strideSrc = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<0>{}, srcRowStrideBytes),
                                              asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        auto strideDst = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<0>{}, dstRowStrideBytes),
                                              asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        auto l1Layout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn,
                                                     asc::te::layout_trait_default<uint8_t>>(shape, strideDst);
        auto ubLayout = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn,
                                                     asc::te::layout_trait_default<uint8_t>>(shape, strideSrc);
        auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>((uint64_t)ubLocal),
                                             ubLayout);
        for (int32_t runIdx = 0; runIdx < runCount; runIdx++) {
            auto l1Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, uint8_t>(
                                                     l1BaseBytes + static_cast<uint64_t>(l1Offset) * sizeof(xType)),
                                                 l1Layout);
            auto copyUB2L1 = asc::te::make_copy(asc::te::copy_ub_to_l1{});
            asc::te::copy(copyUB2L1, l1Tensor, ubTensor);
            l1Offset += l1OffsetStep;
        }
    }

    // NZ two-AIV K-split path of the converted-weight UB -> L1 copy (QUAD, nBubFactor == 1
    // && kBubFactor > 1; the bubKLen < bubNLen sub-branch is the second AIV's trailing
    // partial group). Kept as the original hand-assembled pattern: the per-run l1Offset
    // advance couples with the AIC read side - never rewrite without an equivalence proof.
    __aicore__ inline void CopyVecOut2L1NzKSplit(uint64_t l1BaseBytes, int64_t l1Offset, __ubuf__ uint8_t* ubLocal,
                                                 int32_t bubKLen, int32_t bubNLen)
    {
        if (bubKLen < bubNLen) {
            uint16_t blockCount = static_cast<uint16_t>((Blaze::Gemm::BLOCK_CUBE >> 1) * bubNLen * sizeof(xType) /
                                                        AscendC::VECTOR_REG_WIDTH);
            int64_t srcStride = (CeilDiv<int64_t>(vecKBL1Len_, Blaze::Gemm::BLOCK_CUBE) * 2 - 1) * BLOCK_NUM_REG;
            int64_t dstStride = (kBL1Len_ * Blaze::Gemm::BLOCK_CUBE - AscendC::VECTOR_REG_WIDTH) /
                                AscendC::ONE_BLK_SIZE;
            CopyNzKSplitRuns(l1BaseBytes, l1Offset, ubLocal, blockCount, srcStride, dstStride,
                             CeilDiv<int64_t>(bubKLen, Blaze::Gemm::BLOCK_CUBE >> 1), AscendC::VECTOR_REG_WIDTH);
        } else {
            uint16_t blockCount = static_cast<uint16_t>(bubKLen * Blaze::Gemm::BLOCK_CUBE * sizeof(xType) /
                                                        AscendC::VECTOR_REG_WIDTH);
            int64_t srcStride = (prologueParams_.BL1Pingpong - 1) * BLOCK_NUM_REG;
            int64_t dstStride = 0;
            CopyNzKSplitRuns(l1BaseBytes, l1Offset, ubLocal, blockCount, srcStride, dstStride,
                             CeilDiv<int64_t>(bubNLen, Blaze::Gemm::BLOCK_CUBE),
                             (kBL1Len_ - bubKLen) * Blaze::Gemm::BLOCK_CUBE / AscendC::ONE_BLK_SIZE);
        }
    }

    // ND-trans path, double/single vector buffer; bub offsets are tile-relative on N and
    // full-K on K. Intra-AIV pipelining is managed by the slot mutexes: MTE2 locks the
    // input slot, V locks both slots, MTE3 locks the output slot; a fresh mutex is free,
    // so the first phase acquires every lock without blocking.
    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void BL1ProcessNd(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len, const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale)
    {
        int64_t bubNFactor = CeilDiv<int64_t>(vecNBL1Len_, prologueParams_.nBubSize);
        int64_t bubKFactor = CeilDiv<int64_t>(vecKBL1Len_, prologueParams_.kBubSize);
        for (int32_t bubNLoopIdx = 0; bubNLoopIdx < bubNFactor; bubNLoopIdx++) {
            int64_t vecNBL1Offset = bubNLoopIdx * prologueParams_.nBubSize;
            int64_t bubNOffset = nBL1Offset + vecNBL1Offset;
            int32_t bubNLen = Min<int64_t>(vecNBL1Len_ - vecNBL1Offset, prologueParams_.nBubSize);
            for (int32_t bubKLoopIdx = 0; bubKLoopIdx < bubKFactor; bubKLoopIdx++) {
                int64_t vecKBL1Offset = bubKLoopIdx * prologueParams_.kBubSize;
                int64_t bubKOffset = kBL1Offset + vecKBL1Offset;

                idx_ += 1;
                ubBufIdx_ = idx_ % prologueParams_.vecPingpong;
                int32_t bubKLen = Min<int64_t>(vecKBL1Len_ - vecKBL1Offset, prologueParams_.kBubSize);
                groupNumBub_ = GetVecGroupNum(bubKLen);
                const auto& inputSlot = GetInputSlot(ubBufIdx_);
                const auto& outputSlot = GetOutputSlot(ubBufIdx_);
                {
                    auto mte2Lock = inputSlot.LockMte2();
                    CopyInScale(gmBlockScale, bubNOffset, bubNLen, bubKOffset);
                    CopyInTensorWeight(gmBlockB, bubKOffset, bubNOffset, bubKLen, bubNLen);
                }
                {
                    auto inputVectorLock = inputSlot.LockV();
                    auto outputVectorLock = outputSlot.LockV();
                    DequantCompute(bubNLen, bubKLen);
                }
                int64_t kl1Offset = vecKBL1Offset;
                if (AscendC::GetSubBlockIdx() == 1 && prologueParams_.vecCoreParallel == 1) {
                    kl1Offset += prologueParams_.kBubSize;
                }
                const uint64_t weightL1Addr = SharedWeightL1Addr(curBL1BufIdx);
                {
                    auto mte3Lock = outputSlot.LockMte3();
                    CopyVecOut2L1Nd(weightL1Addr, kl1Offset, vecNBL1Offset, WeightOut(ubBufIdx_), bubKLen, bubNLen);
                }
            }
        }
    }

    // ND-trans path, quadruple vector buffer (one bub per BL1 generation, two AIVs split it).
    // Intra-AIV pipelining is managed by the input/output slot mutexes like BL1ProcessNd;
    // both AIVs run this independently on their core-private UB and mutexes, each filling
    // its half of the shared BL1 slot selected by kl1Offset/nl1Offset below.
    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void BL1ProcessNd4Buffer(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset,
                                               int32_t kL1Len, int32_t nL0Len, const GmBTensor_& gmBlockB,
                                               const GmScaleTensor_& gmBlockScale)
    {
        const uint64_t weightL1Addr = SharedWeightL1Addr(curBL1BufIdx);

        idx_ += 1;
        ubBufIdx_ = idx_ & (prologueParams_.BL1Pingpong - 1);
        int32_t bubKLen = Min<int64_t>(vecKBL1Len_, prologueParams_.kBubSize);
        int32_t bubNLen = Min<int64_t>(vecNBL1Len_, prologueParams_.nBubSize);
        groupNumBub_ = GetVecGroupNum(bubKLen);
        const auto& inputSlot = GetInputSlot(ubBufIdx_);
        const auto& outputSlot = GetOutputSlot(ubBufIdx_);
        {
            auto mte2Lock = inputSlot.LockMte2();
            CopyInScale(gmBlockScale, nBL1Offset, bubNLen, kBL1Offset);
            CopyInTensorWeight(gmBlockB, kBL1Offset, nBL1Offset, bubKLen, bubNLen);
        }
        {
            auto inputVectorLock = inputSlot.LockV();
            auto outputVectorLock = outputSlot.LockV();
            DequantCompute(bubNLen, bubKLen);
        }
        int64_t nl1Offset = 0;
        int64_t kl1Offset = 0;
        if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitK_ && kBL1Len_ > bubKLen) {
            kl1Offset += CeilDiv<int64_t>(CeilDiv<int64_t>(kBL1Len_, prologueParams_.kBubSize), AIV_NUM) *
                         prologueParams_.kBubSize;
        } else if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitN_ && nBL1Len_ > bubNLen) {
            nl1Offset += CeilDiv<int64_t>(CeilDiv<int64_t>(nBL1Len_, prologueParams_.nBubSize), AIV_NUM) *
                         prologueParams_.nBubSize;
        }
        {
            auto mte3Lock = outputSlot.LockMte3();
            CopyVecOut2L1Nd(weightL1Addr, kl1Offset, nl1Offset, WeightOut(ubBufIdx_), bubKLen, bubNLen);
        }
    }

    // NZ path. The N copy length is CeilAlign'd to 32 for the fractal pattern; the tile GM
    // slice covers the fractal-padded extent, so the padded read stays in bounds.
    // Intra-AIV pipelining is managed by the input/output slot mutexes like the ND paths;
    // the two-AIV QUAD split runs independently on each core's private UB and mutexes.
    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void BL1ProcessNz(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                        int32_t nL0Len, const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale)
    {
        const uint64_t weightL1Addr = SharedWeightL1Addr(curBL1BufIdx);
        int32_t bubKLen = Min<int64_t>(vecKBL1Len_, prologueParams_.kBubSize);
        groupNumBub_ = GetVecGroupNum(bubKLen);
        idx_ += 1;
        ubBufIdx_ = idx_ & (prologueParams_.BL1Pingpong - 1);
        int32_t bubNLen = CeilAlign<int64_t>(Min<int64_t>(vecNBL1Len_, prologueParams_.nBubSize), ALIGNED_32_SIZE);
        const auto& inputSlot = GetInputSlot(ubBufIdx_);
        const auto& outputSlot = GetOutputSlot(ubBufIdx_);
        {
            auto mte2Lock = inputSlot.LockMte2();
            CopyInScale(gmBlockScale, nBL1Offset, bubNLen, kBL1Offset);
            CopyInTensorWeight(gmBlockB, kBL1Offset, nBL1Offset, bubKLen, bubNLen);
        }
        {
            auto inputVectorLock = inputSlot.LockV();
            auto outputVectorLock = outputSlot.LockV();
            DequantCompute(bubNLen, bubKLen);
        }
        int64_t nl1Offset = 0;
        int64_t kl1Offset = 0;
        if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitK_ && kBL1Len_ > bubKLen) {
            kl1Offset += CeilDiv<int64_t>(CeilDiv<int64_t>(kBL1Len_, prologueParams_.kBubSize), AIV_NUM) *
                         prologueParams_.kBubSize;
        } else if (AscendC::GetSubBlockIdx() == 1 && twoVectorCoreSplitN_ && nBL1Len_ > bubNLen) {
            nl1Offset += CeilDiv<int64_t>(CeilDiv<int64_t>(nBL1Len_, prologueParams_.nBubSize), AIV_NUM) *
                         prologueParams_.nBubSize;
        }
        {
            auto mte3Lock = outputSlot.LockMte3();
            if (twoVectorCoreSplitK_) {
                const int64_t l1Offset = nl1Offset * CeilAlign<int64_t>(kBL1Len_, Blaze::Gemm::BLOCK_CUBE) +
                                         kl1Offset * Blaze::Gemm::BLOCK_CUBE;
                CopyVecOut2L1NzKSplit(weightL1Addr, l1Offset, WeightOut(ubBufIdx_), bubKLen, bubNLen);
            } else {
                CopyVecOut2L1Nz(weightL1Addr, nl1Offset, WeightOut(ubBufIdx_), bubKLen, bubNLen);
            }
        }
    }

    template <typename GmBTensor_, typename GmScaleTensor_>
    __aicore__ inline void BL1Process(uint64_t curBL1BufIdx, int64_t nBL1Offset, int64_t kBL1Offset, int32_t kL1Len,
                                      int32_t nL0Len, const GmBTensor_& gmBlockB, const GmScaleTensor_& gmBlockScale)
    {
        if constexpr (IsTrans<LayoutB>::value) {
            if (prologueParams_.BL1Pingpong == QUADRUPLE_BUFFER) {
                BL1ProcessNd4Buffer(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len, gmBlockB, gmBlockScale);
            } else {
                BL1ProcessNd(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len, gmBlockB, gmBlockScale);
            }
        } else {
            BL1ProcessNz(curBL1BufIdx, nBL1Offset, kBL1Offset, kL1Len, nL0Len, gmBlockB, gmBlockScale);
        }
    }

    uint64_t ubBufIdx_;
    int64_t kBL1Offset_{0};
    int64_t kBL1Len_{0};
    int64_t nBL1Offset_{0};
    int64_t baseUseN_{0};
    int64_t nBL1Len_{0};
    uint64_t curBL1BufIdx_{0};
    int64_t idx_ = -1;
    uint32_t groupNumBub_;
    int64_t vecKBL1Len_;
    int64_t vecNBL1Len_;
    bool twoVectorCoreSplitN_ = false;
    bool twoVectorCoreSplitK_ = false;

    uint64_t scaleInBase_{0};
    uint64_t singleScaleSize_{0};
    uint64_t scaleMaskOffset_{0};
    BufferSlot inputSlots_[QUADRUPLE_BUFFER];
    BufferSlot outputSlots_[QUADRUPLE_BUFFER];
    Params prologueParams_;
    uint64_t sharedL1SlotAddrs_[QUADRUPLE_BUFFER];
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
