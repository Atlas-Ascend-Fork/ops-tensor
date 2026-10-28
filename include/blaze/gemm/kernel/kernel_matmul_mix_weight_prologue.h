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
 * \file kernel_matmul_mix_weight_prologue.h
 * \brief Kernel orchestration for weight-only MX matmul with an AIV weight-conversion prologue.
 */
#pragma once

#include "kernel_basic_intf.h"

#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class BlockMmad_>
class KernelMatmulMixWeightPrologue {
public:
    using BlockMmad = BlockMmad_;
    using OutType = typename BlockMmad::AType;
    using InType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static_assert(sizeof(OutType) == 1, "Weight Quant MX expects 8-bit converted weights");
    static_assert(IsFp4<InType>(), "Weight Quant MX expects packed FP4 input weights");

    struct Params {
        uint64_t baseN{0};
        uint64_t kL1Size{0};
        uint64_t kUbSize{0};
        uint64_t nUbSize{0};
        uint64_t l1BufferNum{0};
        bool hasBias{false};
    };

    __aicore__ inline explicit KernelMatmulMixWeightPrologue(const Params& params) { Init(params); }

    __aicore__ inline ~KernelMatmulMixWeightPrologue()
    {
        for (int8_t index = 0; index < static_cast<int8_t>(l1BufferNum_); ++index) {
            WaitWeightFlag<SyncProtocol::AIC_FREE_FLAG>();
        }
    }

    template <typename GMWeightTensor, typename GMBiasTensor, typename ActualBlockShape>
    __aicore__ inline void operator()(const GMWeightTensor& gmWeightTensor, const GMBiasTensor& gmBiasTensor,
                                      const ActualBlockShape& actualBlockShape)
    {
        BlockContext block{static_cast<int64_t>(asc::te::get<MNK_K>(actualBlockShape)),
                           static_cast<int64_t>(asc::te::get<MNK_N>(actualBlockShape)),
                           Align16(static_cast<uint64_t>(asc::te::get<MNK_N>(actualBlockShape)))};
        const int64_t kL1Size = static_cast<int64_t>(kL1Size_);
        const uint64_t kTileCount = CeilDiv(static_cast<uint64_t>(block.kLength), kL1Size_);
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            const int64_t kOffset = static_cast<int64_t>(kLoopIdx) * kL1Size;
            const int64_t kLength = Min(block.kLength - kOffset, kL1Size);
            SliceContext slice{kOffset, 0, 0, block.nLength, kLength, kLength};
            if (UpdateUbSliceForCurrentSubBlock(slice, block)) {
                ConvertCurrentWeightSlice(gmWeightTensor, gmBiasTensor, block, slice, kLoopIdx == 0U);
            } else {
                NotifyCurrentWeightSliceReady();
            }
            l1BufIdx_ = (l1BufIdx_ + 1) & l1BufferMask_;
        }
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;

    struct BlockContext {
        int64_t kLength;
        int64_t nLength;
        uint64_t nStride;
    };

    struct SliceContext {
        int64_t kGmOffset{0};
        int64_t nOffset{0};
        int64_t kOffset{0};
        int64_t nLength{0};
        int64_t kLength{0};
        int64_t kL1Length{0};
    };

    using L1Storage = typename BlockMmad::WeightL1Storage;
    using SyncProtocol = typename BlockMmad::SyncProtocol;

    class UbStorage {
    public:
        __aicore__ inline void Init(uint64_t baseN, uint64_t nUbSize, uint64_t kUbSize, uint64_t bufferNum,
                                    bool hasBias)
        {
            bufferNum_ = bufferNum;
            uint64_t nUbSizeAligned = Align16(nUbSize);
            uint64_t kUbSizeAligned = Align64(kUbSize);
            if constexpr (WEIGHT_NZ) {
                uint64_t kUbSizeBlockAligned = Align32(kUbSize);
                singleWeightInSize_ = (nUbSizeAligned * kUbSizeBlockAligned) >> FP4_PACK_SHIFT;
                singleWeightOutSize_ = nUbSizeAligned * kUbSizeAligned * sizeof(OutType);
            } else {
                singleWeightInSize_ = (nUbSize * kUbSizeAligned) >> FP4_PACK_SHIFT;
                singleWeightOutSize_ = PhysicalNStride(nUbSize) * kUbSizeAligned * sizeof(OutType);
            }
            weightInSize_ = bufferNum * singleWeightInSize_;
            weightOutSize_ = bufferNum * singleWeightOutSize_;
            weightInBase_ = weightOutSize_;

            if (hasBias) {
                constexpr uint64_t VECTOR_ELEM = static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / sizeof(BiasType);
                singleBiasSize_ = CeilAlign(baseN, VECTOR_ELEM) * sizeof(BiasType);
                biasOutBase_ = weightInBase_ + weightInSize_;
                biasInBase_ = biasOutBase_ + bufferNum * singleBiasSize_;
            }

            for (uint64_t index = 0; index < bufferNum_; ++index) {
                inputSlots_[index] = {weightInBase_ + index * singleWeightInSize_, static_cast<uint8_t>(index)};
                uint64_t outputOffset = index * singleWeightOutSize_;
                if constexpr (WEIGHT_NZ) {
                    outputOffset = index * static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / sizeof(OutType);
                }
                outputSlots_[index] = {outputOffset, static_cast<uint8_t>(OUTPUT_SYNC_ID_BASE + index)};
            }
        }

        __aicore__ inline auto MakePackedWeight(uint64_t bufferId, int64_t nSize, int64_t kSize) const
        {
            uint64_t offset = inputSlots_[bufferId].Addr();
            if constexpr (WEIGHT_NZ) {
                return asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::ub, InType>(offset),
                    asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<asc::te::c0_element<OutType>>>(
                        kSize, nSize));
            } else {
                int64_t kStride = static_cast<int64_t>(Align64(static_cast<uint64_t>(kSize)));
                auto fullTensor = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::ub, InType>(offset),
                    asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(kStride, nSize));
                return fullTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                        asc::te::make_shape(kSize, nSize));
            }
        }

        __aicore__ inline auto MakeConvertedWeight(uint64_t bufferId, int64_t nSize, int64_t kSize) const
        {
            if constexpr (WEIGHT_NZ) {
                constexpr uint64_t VECTOR_ELEMENTS = static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / sizeof(OutType);
                return asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::ub, OutType>(outputSlots_[bufferId].Addr()),
                    Blaze::Gemm::Weight8BitZnToZnUBLayout<OutType>{}(
                        kSize, static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))),
                        VECTOR_ELEMENTS * bufferNum_));
            } else {
                return asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::ub, OutType>(outputSlots_[bufferId].Addr()),
                    Blaze::Gemm::Weight8BitDnToZnUBLayout<OutType>{}(kSize, nSize));
            }
        }

        __aicore__ inline auto MakeBiasIn(uint64_t bufferId, int64_t nSize) const
        {
            return MakeBias(biasInBase_, bufferId, nSize);
        }

        __aicore__ inline auto MakeBiasOut(uint64_t bufferId, int64_t nSize) const
        {
            return MakeBias(biasOutBase_, bufferId, nSize);
        }

        __aicore__ inline const BufferSlot& GetInputSlot(uint64_t bufferId) const { return inputSlots_[bufferId]; }

        __aicore__ inline const BufferSlot& GetOutputSlot(uint64_t bufferId) const { return outputSlots_[bufferId]; }

    private:
        static constexpr uint16_t FP4_PACK_SHIFT = 1U;
        static constexpr uint64_t OUTPUT_SYNC_ID_BASE = QUADRUPLE_BUFFER_COUNT;

        __aicore__ inline static uint64_t PhysicalNStride(uint64_t nSize) { return Align16(nSize) + 1UL; }

        __aicore__ inline auto MakeBias(uint64_t baseOffset, uint64_t bufferId, int64_t nSize) const
        {
            uint64_t offset = baseOffset + bufferId * singleBiasSize_;
            auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                static_cast<int64_t>(1), static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))));
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(offset), layout);
        }

        uint64_t weightOutSize_{0};
        uint64_t weightInSize_{0};
        uint64_t singleWeightInSize_{0};
        uint64_t singleWeightOutSize_{0};
        uint64_t weightInBase_{0};
        uint64_t biasInBase_{0};
        uint64_t biasOutBase_{0};
        uint64_t singleBiasSize_{0};
        uint64_t bufferNum_{0};
        BufferSlot inputSlots_[QUADRUPLE_BUFFER_COUNT];
        BufferSlot outputSlots_[QUADRUPLE_BUFFER_COUNT];
    };

    __aicore__ inline void Init(const Params& params)
    {
        kL1Size_ = params.kL1Size;
        nUbSize_ = params.nUbSize;
        kUbSize_ = params.kUbSize;
        l1BufferNum_ = params.l1BufferNum;
        l1BufferMask_ = l1BufferNum_ - 1U;
        hasBias_ = params.hasBias;
        ubStorage_.Init(params.baseN, nUbSize_, kUbSize_, l1BufferNum_, hasBias_);
        l1Storage_.Init(params.baseN, kL1Size_, l1BufferNum_, hasBias_);
    }

    template <uint64_t FLAG>
    __aicore__ inline void WaitWeightFlag() const
    {
        Sync::WaitForCube<SyncProtocol::MODE, PIPE_MTE3>(FLAG);
    }

    template <uint64_t FLAG>
    __aicore__ inline void SetWeightFlag() const
    {
        Sync::NotifyCube<SyncProtocol::MODE, PIPE_MTE3>(FLAG);
    }

    __aicore__ inline bool UpdateUbSliceForCurrentSubBlock(SliceContext& slice, const BlockContext& block)
    {
        if constexpr (WEIGHT_NZ) {
            if (slice.kL1Length <= static_cast<int64_t>(kUbSize_)) {
                slice.kLength = slice.kL1Length;
                if (AscendC::GetSubBlockIdx() == 0) {
                    return true;
                }
                slice.kLength = 0;
                return false;
            }
            slice.kLength = static_cast<int64_t>(kUbSize_);
            if (AscendC::GetSubBlockIdx() == 1) {
                slice.kOffset = slice.kLength;
                slice.kLength = slice.kL1Length - slice.kLength;
            }
            return slice.kLength > 0;
        }

        if (l1BufferNum_ == QUADRUPLE_BUFFER_COUNT) {
            slice.nLength = Min(slice.nLength, static_cast<int64_t>(nUbSize_));
            if (AscendC::GetSubBlockIdx() == 1) {
                slice.nOffset = static_cast<int64_t>(nUbSize_);
                slice.nLength = Max(block.nLength - static_cast<int64_t>(nUbSize_), static_cast<int64_t>(0));
            }
            return slice.nLength > 0;
        }
        // With two L1 buffers, the two AIVs take turns processing one complete ND tile.
        return l1BufIdx_ == static_cast<uint64_t>(AscendC::GetSubBlockIdx());
    }

    template <typename GMWeightTensor, typename GMBiasTensor>
    __aicore__ inline void ConvertCurrentWeightSlice(const GMWeightTensor& gmWeightTensor,
                                                     const GMBiasTensor& gmBiasTensor, const BlockContext& block,
                                                     const SliceContext& slice, bool firstKTile)
    {
        WaitWeightFlag<SyncProtocol::AIC_FREE_FLAG>();
        bool processBias = ShouldProcessBias(firstKTile);
        CopyConvertStoreWeight(gmWeightTensor, gmBiasTensor, block, slice, processBias);
        SetWeightFlag<SyncProtocol::AIV_READY_FLAG>();
    }

    __aicore__ inline bool ShouldProcessBias(bool firstKTile) const
    {
        if (!hasBias_ || !firstKTile) {
            return false;
        }
        if constexpr (WEIGHT_NZ) {
            return AscendC::GetSubBlockIdx() == 0;
        }
        if (l1BufferNum_ == QUADRUPLE_BUFFER_COUNT) {
            return AscendC::GetSubBlockIdx() == 0;
        }
        return l1BufIdx_ == static_cast<uint64_t>(AscendC::GetSubBlockIdx());
    }

    __aicore__ inline void NotifyCurrentWeightSliceReady()
    {
        WaitWeightFlag<SyncProtocol::AIC_FREE_FLAG>();
        SetWeightFlag<SyncProtocol::AIV_READY_FLAG>();
    }

    template <typename GMWeightTensor, typename GMBiasTensor>
    __aicore__ inline void CopyConvertStoreWeight(const GMWeightTensor& gmWeightTensor,
                                                  const GMBiasTensor& gmBiasTensor, const BlockContext& block,
                                                  const SliceContext& slice, bool processBias)
    {
        idx_ += 1;
        uint64_t ubBufIdx = static_cast<uint64_t>(idx_) & l1BufferMask_;
        const auto& inputSlot = ubStorage_.GetInputSlot(ubBufIdx);
        const auto& outputSlot = ubStorage_.GetOutputSlot(ubBufIdx);
        auto gmSlice = MakeGmWeightSlice(gmWeightTensor, slice);
        auto weight4BitTensor = ubStorage_.MakePackedWeight(ubBufIdx, slice.nLength, slice.kLength);
        {
            auto mte2Lock = inputSlot.LockMte2();
            CopyPackedWeightGmToUb(gmSlice, weight4BitTensor, slice);
            if (processBias) {
                auto biasInUbTensor = ubStorage_.MakeBiasIn(ubBufIdx, block.nLength);
                auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
                asc::te::copy(copyGM2UB, biasInUbTensor, gmBiasTensor);
            }
        }
        auto weight8BitTensor = ubStorage_.MakeConvertedWeight(ubBufIdx, slice.nLength, slice.kLength);
        {
            auto inputVectorLock = inputSlot.LockV();
            auto outputVectorLock = outputSlot.LockV();
            Blaze::Gemm::Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor);
            if (processBias) {
                auto biasInUbTensor = ubStorage_.MakeBiasIn(ubBufIdx, block.nLength);
                auto biasOutUbTensor = ubStorage_.MakeBiasOut(ubBufIdx, block.nLength);
                Blaze::Gemm::Tile::ScaleMxBias<BiasType>(biasInUbTensor, biasOutUbTensor);
            }
        }
        auto l1Tensor = l1Storage_.MakeWeightTensor(l1BufIdx_, slice.nOffset, slice.kOffset, block.nStride,
                                                    slice.nLength, slice.kLength);
        {
            auto mte3Lock = outputSlot.LockMte3();
            auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
            asc::te::copy(copyUB2L1, l1Tensor, weight8BitTensor);
            if (processBias) {
                auto biasOutUbTensor = ubStorage_.MakeBiasOut(ubBufIdx, block.nLength);
                auto biasL1Tensor = l1Storage_.MakeBiasTensor(l1BufIdx_, block.nLength);
                auto copyUB2L1 = asc::te::make_copy(asc::te::copy_ub_to_l1{});
                asc::te::copy(copyUB2L1, biasL1Tensor, biasOutUbTensor);
            }
        }
    }

    template <typename GMWeightTensor>
    __aicore__ inline auto MakeGmWeightSlice(const GMWeightTensor& gmWeightTensor, const SliceContext& slice)
    {
        return gmWeightTensor.slice(asc::te::make_coord(slice.kGmOffset + slice.kOffset, slice.nOffset),
                                    asc::te::make_shape(slice.kLength, slice.nLength));
    }

    template <typename GMWeightSlice, typename Weight4BitTensor>
    __aicore__ inline void CopyPackedWeightGmToUb(const GMWeightSlice& gmSlice,
                                                  const Weight4BitTensor& weight4BitTensor, const SliceContext& slice)
    {
        if (slice.kLength <= 0 || slice.nLength <= 0) {
            return;
        }
        auto copyGM2UB = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
        asc::te::copy(copyGM2UB, weight4BitTensor, gmSlice);
    }

    uint64_t nUbSize_{0};
    uint64_t kUbSize_{0};
    uint64_t kL1Size_{0};
    uint64_t l1BufIdx_{0};
    int64_t idx_{-1};
    uint64_t l1BufferNum_{DOUBLE_BUFFER_COUNT};
    uint64_t l1BufferMask_{DOUBLE_BUFFER_COUNT - 1U};
    bool hasBias_{false};
    UbStorage ubStorage_;
    L1Storage l1Storage_;
};

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<BlockEpilogue_, void> &&
        AscendC::Std::is_same_v<KernelMixWithWeightPrologue, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleA = typename BlockMmad::LayoutScaleA;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockPrologue = KernelMatmulMixWeightPrologue<BlockMmad>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<LayoutScaleA, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, AscendC::Std::Int<SCALE_C0>>;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct PrologueParams {
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t kBubSize{0};
        uint64_t nBubSize{0};
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
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;

    __aicore__ inline static void Execute(const Params& params)
    {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        if ASCEND_IS_AIV {
            RunAiv(params, scheduler);
        }
        if ASCEND_IS_AIC {
            RunAic(params, scheduler);
        }
    }

    __aicore__ inline static void RunAiv(const Params& params, const BlockScheduler& scheduler)
    {
        int64_t k = asc::te::get<MNK_K>(params.problemShape);
        int64_t n = asc::te::get<MNK_N>(params.problemShape);
        uint64_t tileNum = scheduler.GetTileCount();
        uint64_t curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        if (curBlockIdx >= tileNum) {
            return;
        }
        typename BlockPrologue::Params prologueParams{
            static_cast<uint64_t>(asc::te::get<MNK_N>(params.mmadParams.l1TileShape)),
            static_cast<uint64_t>(asc::te::get<MNK_K>(params.mmadParams.l1TileShape)),
            params.prologueParams.kBubSize,
            params.prologueParams.nBubSize,
            params.mmadParams.l1BufferNum,
            params.mmadParams.hasBias};
        BlockPrologue blockPrologue(prologueParams);
        auto gmBias = MakeGmBiasTensor(params, n);
        if constexpr (WEIGHT_NZ) {
            auto gmWeight = MakeGmNzWeightTensor(params, k, n);
            ProcessAivTiles(gmWeight, gmBias, blockPrologue, scheduler, curBlockIdx, tileNum);
        } else {
            auto gmWeight = MakeGmNdWeightTensor(params, k, n);
            ProcessAivTiles(gmWeight, gmBias, blockPrologue, scheduler, curBlockIdx, tileNum);
        }
    }

    __aicore__ inline static auto MakeGmBiasTensor(const Params& params, int64_t n)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                        reinterpret_cast<__gm__ BiasType*>(params.prologueParams.biasGmAddr)),
                                    asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), n));
    }

    __aicore__ inline static auto MakeGmNzWeightTensor(const Params& params, int64_t k, int64_t n)
    {
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
            asc::te::make_frame_layout<LayoutB, AscendC::Std::Int<asc::te::c0_element<AType>>>(k, n));
    }

    __aicore__ inline static auto MakeGmNdWeightTensor(const Params& params, int64_t k, int64_t n)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                        reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr)),
                                    asc::te::frame_layout_format<LayoutB>{}(k, n));
    }

    template <typename GMWeightTensor, typename GMBiasTensor>
    __aicore__ inline static void ProcessAivTiles(const GMWeightTensor& gmWeight, const GMBiasTensor& gmBias,
                                                  BlockPrologue& blockPrologue, const BlockScheduler& scheduler,
                                                  uint64_t curBlockIdx, uint64_t tileNum)
    {
        for (uint64_t loopIdx = curBlockIdx; loopIdx < tileNum; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto blockShape = scheduler.GetBlockShape(blockCoord);
            int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
            int64_t kSize = asc::te::get<MNK_K>(blockShape);
            int64_t nL1Size = asc::te::get<MNK_N>(blockShape);
            auto gmBlockWeight = gmWeight.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(kSize, nL1Size));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0, nOffset),
                                            asc::te::make_shape(static_cast<int64_t>(1), nL1Size));
            blockPrologue(gmBlockWeight, gmBlockBias, blockShape);
        }
    }

    __aicore__ inline static void RunAic(const Params& params, const BlockScheduler& scheduler)
    {
        uint64_t tileNum = scheduler.GetTileCount();
        uint64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx >= tileNum) {
            return;
        }
        int64_t m = asc::te::get<MNK_M>(params.problemShape);
        int64_t n = asc::te::get<MNK_N>(params.problemShape);
        int64_t k = asc::te::get<MNK_K>(params.problemShape);
        int64_t scaleKSize = static_cast<int64_t>(CeilDiv(Align64(static_cast<uint64_t>(k)), BlockMmad::MX_GROUP_SIZE));
        auto gmA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)),
            MakeLayoutA{}(m, k));
        auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ ScaleAType*>(params.mmadParams.scaleAGmAddr)),
                                             MakeLayoutScaleA{}(m, scaleKSize));
        auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ ScaleBType*>(params.mmadParams.scaleBGmAddr)),
                                             MakeLayoutScaleB{}(scaleKSize, n));
        auto gmC = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)),
            MakeLayoutC{}(m, n));
        BlockMmad blockMmad(params.mmadParams);
        for (uint64_t loopIdx = curBlockIdx; loopIdx < tileNum; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto blockShape = scheduler.GetBlockShape(blockCoord);
            int64_t mOffset = asc::te::get<MNK_M>(blockCoord);
            int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
            int64_t mL1Size = asc::te::get<MNK_M>(blockShape);
            int64_t nL1Size = asc::te::get<MNK_N>(blockShape);
            int64_t kSize = asc::te::get<MNK_K>(blockShape);

            auto gmBlockA = gmA.slice(asc::te::make_coord(mOffset, 0), asc::te::make_shape(mL1Size, kSize));
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mOffset, 0),
                                                asc::te::make_shape(mL1Size, scaleKSize));
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nOffset),
                                                asc::te::make_shape(scaleKSize, nL1Size));
            auto gmBlockC = gmC.slice(asc::te::make_coord(mOffset, nOffset), asc::te::make_shape(mL1Size, nL1Size));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, gmBlockC);
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
