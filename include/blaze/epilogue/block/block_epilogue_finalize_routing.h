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
 * \file block_epilogue_finalize_routing.h
 * \brief Routing epilogues for grouped matmul.
 */

#ifndef BLAZE_BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#define BLAZE_BLOCK_EPILOGUE_FINALIZE_ROUTING_H

#include <cstdint>

#if defined(__NPU_ARCH__)
#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze::Epilogue::Block {

enum class FinalizeRoutingMode : uint8_t { MX, PERTOKEN, A8W4 };

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeLogit_, typename RowIndexType_,
          typename DataTypeX2Scale_ = float, bool IsDeter_ = false, FinalizeRoutingMode Mode_ = FinalizeRoutingMode::MX>
class BlockEpilogueFinalizeRouting {
private:
    using DataTypeX2Scale = DataTypeX2Scale_;
    using DataTypeX1Scale = float;
    using BiasDtype = bfloat16_t;
    using DataTypeRowIndex = RowIndexType_;

    static constexpr bool IS_MX = Mode_ == FinalizeRoutingMode::MX;
    static constexpr bool IS_PERTOKEN = Mode_ == FinalizeRoutingMode::PERTOKEN;
    static constexpr bool IS_A8W4 = Mode_ == FinalizeRoutingMode::A8W4;
    static constexpr uint64_t UB_KIB = 1024;
    static constexpr uint64_t OUTPUT_ROW_ALIGN_BYTES = 32;
    static constexpr uint64_t MX_ACC_ROW_ALIGN_ELEMENTS = 32;
    static constexpr uint64_t FIXPIPE_ROW_ALIGN = 2;
    static constexpr uint64_t A8W4_OUTPUT_ALIGN_ELEMENTS = 32;
    static constexpr uint64_t A8W4_ACC_UB_OFFSET = 80 * UB_KIB;
    static constexpr uint32_t METADATA_MAX_ELEMENTS = 256;
    static constexpr float A8W4_WEIGHT_SCALE = 64.0F;
    static constexpr uint16_t A8W4_OUTPUT_EVENT = 4;
    static constexpr uint16_t A8W4_METADATA_EVENT_BASE = 4;
    static constexpr uint64_t A8W4_ROW_INDEX_UB_OFFSET = 242 * UB_KIB;
    static constexpr uint64_t A8W4_LOGIT_UB_OFFSET = 244 * UB_KIB;
    static constexpr uint64_t A8W4_ROW_INDEX_SLOT_BYTES = UB_KIB;
    static constexpr uint64_t A8W4_LOGIT_SLOT_BYTES = 512;
    static constexpr int64_t GMMFR_DQ_OUT_ELE_NUM_ONE_BLK = 64L;
    static constexpr uint32_t GMMFR_DQ_DOUBLE_BUFFER_COUNT = 2;
    static constexpr uint32_t GMMFR_DQ_L0C_MAX_M = 128;
    static constexpr uint32_t GMMFR_DQ_L0C_MAX_N = 256;
    static constexpr uint32_t GMMFR_DQ_BLOCK_BYTES = 256;
    static constexpr uint32_t GMMFR_DQ_VEC_MAX_M = 32;
    static constexpr uint32_t GMMFR_DQ_MAX_SINGLE_MN = GMMFR_DQ_L0C_MAX_M * GMMFR_DQ_L0C_MAX_N;
    static constexpr uint32_t GMMFR_DQ_HALF_DB_MAX_SINGLE_MN = GMMFR_DQ_VEC_MAX_M * GMMFR_DQ_L0C_MAX_N;
    static constexpr uint32_t GMMFR_DQ_Y_IDX = 0;
    static constexpr uint32_t GMMFR_DQ_BIAS_IDX = 1;
    static constexpr uint32_t GMMFR_DQ_X2SCALE_IDX = 2;
    static constexpr uint32_t GMMFR_DQ_X1SCALE_IDX = 3;
    static constexpr uint32_t GMMFR_DQ_LOGIT_INDEX = 4;
    static constexpr uint32_t GMMFR_DQ_ROW_INDEX = 5;
    static constexpr uint64_t GMMFR_DQ_MAX_OUTPUT_M_UB = GMMFR_DQ_VEC_MAX_M;
    static constexpr uint16_t OUTPUT_BUFFER_COUNT = IS_A8W4 ? 1 : GMMFR_DQ_DOUBLE_BUFFER_COUNT;
    static constexpr uint16_t OUTPUT_EVENT_BASE = IS_A8W4 ? A8W4_OUTPUT_EVENT : 0;
    // Row pitch of the AIC fixpipe destination in UB (elements). Matches the ubC tensor the
    // GMM per-token-dequant kernel builds for BlockMmad(MatmulWithScaleMix): 256B alignment
    // of the L0C accumulator row (int32/float are both 4 bytes).
    static constexpr uint64_t GMMFR_DQ_L0C_UB_ROW_ALIGN = GMMFR_DQ_BLOCK_BYTES / sizeof(float);

    static constexpr AscendC::Reg::CastTrait CAST_GMM_FR_BF16_TO_FP32_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    static constexpr AscendC::Reg::CastTrait CAST_GMM_FR_FP32_TO_BF16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using RowIndexType = RowIndexType_;
    using DataTypeLogit = DataTypeLogit_;
    static constexpr bool IS_DETER = IsDeter_;
    static_assert(AscendC::Std::is_one_of_v<DataTypeOut, float, bfloat16_t>,
                  "GMM FR epilogue only supports float/bfloat16_t output.");
    static_assert(AscendC::Std::is_one_of_v<DataTypeIn, int32_t, float>,
                  "GMM FR epilogue only supports int32_t/float MMAD output.");
    static_assert(AscendC::Std::is_one_of_v<DataTypeX2Scale, float, bfloat16_t>,
                  "GMM FR epilogue only supports float/bfloat16_t x2 scale.");
    static_assert(AscendC::Std::is_same_v<DataTypeX1Scale, float>,
                  "GMM FR epilogue only supports float per-token scale.");
    static_assert(AscendC::Std::is_same_v<BiasDtype, bfloat16_t>, "GMM FR epilogue only supports bfloat16_t bias.");
    static_assert(AscendC::Std::is_one_of_v<DataTypeRowIndex, int32_t, int64_t>,
                  "GMM FR epilogue only supports int32_t/int64_t row index.");
    static_assert(AscendC::Std::is_one_of_v<DataTypeLogit, float, bfloat16_t>,
                  "GMM FR epilogue only supports float/bfloat16_t logit.");

    static_assert(IS_PERTOKEN || AscendC::Std::is_same_v<DataTypeIn, float>,
                  "MX finalize routing consumes float accumulators.");
    static_assert(!IsDeter_ || IS_PERTOKEN, "Deterministic workspace is supported by the pertoken schedule.");
    static_assert(!IS_A8W4 ||
                      (AscendC::Std::is_same_v<DataTypeOut, float> && AscendC::Std::is_same_v<DataTypeLogit, float> &&
                       AscendC::Std::is_same_v<DataTypeRowIndex, int64_t>),
                  "A8W4 finalize routing requires FP32 output/logit and INT64 row indices.");

    static_assert(!IS_MX || ((AscendC::Std::is_same_v<DataTypeOut, float> &&
                              AscendC::Std::is_same_v<RowIndexType, int64_t>) ||
                             (AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> &&
                              AscendC::Std::is_same_v<RowIndexType, int32_t>)),
                  "MX FP32 output requires INT64 row indices and BF16 output requires INT32 row indices.");

private:
    struct MxParams {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        int32_t baseM{256};
        int32_t baseN{256};
    };

    struct QuantParams {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        int64_t nSize{0};
    };

public:
    // Keep the aggregate layout used by the original four-argument MX template.
    using Params = AscendC::Std::conditional_t<IS_MX, MxParams, QuantParams>;

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // blk_m, blk_n, _, _
    // nOffset, bias, x2Scale, x1Scale, logit, rowIndex
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>; // m, n, k

    __aicore__ inline BlockEpilogueFinalizeRouting() {}

    __aicore__ inline ~BlockEpilogueFinalizeRouting()
    {
        if ASCEND_IS_AIV {
            if (IS_MX || !initialized_) {
                return;
            }
            if constexpr (IS_A8W4) {
                for (uint16_t slot = 0; slot < GMMFR_DQ_DOUBLE_BUFFER_COUNT; ++slot) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(A8W4_METADATA_EVENT_BASE + slot);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(A8W4_METADATA_EVENT_BASE + slot);
                }
            }
            for (uint16_t slot = 0; slot < OUTPUT_BUFFER_COUNT; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + slot);
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (!IS_MX) {
            for (uint16_t slot = 0; slot < OUTPUT_BUFFER_COUNT; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + slot);
            }
            initialized_ = true;
            n_ = params.nSize;
        }
        params_ = &params;
        subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        yGm_ = reinterpret_cast<__gm__ DataTypeOut*>(params_->yGmAddr);
        rowIndexGm_ = reinterpret_cast<__gm__ DataTypeRowIndex*>(params_->rowIndexGmAddr);
        rowIndexBaseGm_ = reinterpret_cast<__gm__ DataTypeRowIndex*>(params_->rowIndexGmAddr);
        if constexpr (IS_DETER) {
            mmQuantOutGm_ = reinterpret_cast<__gm__ DataTypeOut*>(params_->workspaceGmAddr);
        }
        InitBufferLayout();
    }

    __aicore__ inline auto GetL0c2UbTensor()
    {
        return AscendC::LocalTensor<DataTypeIn>(AscendC::TPosition::VECIN, 0, GMMFR_DQ_MAX_SINGLE_MN);
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        n_ = asc::te::get<Blaze::Gemm::MNK_N>(problemShape);
    }

    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset)
    {
        if ASCEND_IS_AIV {
            const auto offset = asc::te::get<GMMFR_DQ_LOGIT_INDEX>(baseOffset);
            logitGm_ = reinterpret_cast<__gm__ DataTypeLogit*>(params_->logitGmAddr) + offset;
            rowIndexGm_ = reinterpret_cast<__gm__ DataTypeRowIndex*>(params_->rowIndexGmAddr) + offset;
        }
    }

    // AIC and AIV must use the same row pitch and physical UB base.
    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols)
    {
        const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), FIXPIPE_ROW_ALIGN);
        const uint64_t copyCols = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(cols), ACC_ROW_ALIGN);
        if constexpr (IS_PERTOKEN) {
            const auto layout = asc::te::make_frame_layout<asc::te::nd_layout_ptn>(static_cast<int64_t>(copyRows),
                                                                                   static_cast<int64_t>(copyCols));
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(L0C_UB_OFFSET),
                                        layout);
        } else {
            const auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                           AscendC::Std::Int<asc::te::c0_element<DataTypeIn>>>(
                copyRows, copyCols);
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(L0C_UB_OFFSET),
                                        layout);
        }
    }

    // Deterministic flush indexes rows globally (window-relative), while rowIndexGm_
    // carries the active group offset; read through the unshifted base instead.
    __aicore__ inline void PrefetchRowIndicesFromBase(uint64_t offset, uint64_t count)
    {
        CopyGmToUb(rowIndexBaseGm_ + offset, rowIndexWindowUb_, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }

    __aicore__ inline uint64_t GetPrefetchedRowIndex(uint64_t offset)
    {
        AscendC::LocalTensor<DataTypeRowIndex> rowUb(AscendC::TPosition::VECIN, rowIndexWindowUb_,
                                                     GMMFR_DQ_BLOCK_BYTES);
        return static_cast<uint64_t>(rowUb.GetValue(offset));
    }

    // Enqueue once per A8W4 tile after its first two weight strips, before consuming
    // the previous tile. Two slots retain that tile's logit and row indices across groups.
    // Each slot is reusable only after both vector and scalar readers have finished.
    __aicore__ inline void PrefetchRoutingMetadata(uint64_t rows, uint64_t mOffset, uint64_t groupMOffset)
    {
        if constexpr (IS_A8W4) {
            const uint64_t half = Blaze::Gemm::CeilDiv(rows, static_cast<uint64_t>(AscendC::GetTaskRation()));
            const uint64_t count = subBlockIdx_ == 1 ? rows - half : half;
            const uint64_t slot = metadataWriteIdx_++ % GMMFR_DQ_DOUBLE_BUFFER_COUNT;
            const uint16_t event = A8W4_METADATA_EVENT_BASE + slot;
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            const uint64_t offset = groupMOffset + mOffset + subBlockIdx_ * half;
            if (count != 0) {
                CopyGmToUb(reinterpret_cast<__gm__ DataTypeLogit*>(params_->logitGmAddr) + offset,
                           A8W4_LOGIT_UB_OFFSET + slot * A8W4_LOGIT_SLOT_BYTES, count);
                CopyGmToUb(reinterpret_cast<__gm__ DataTypeRowIndex*>(params_->rowIndexGmAddr) + offset,
                           A8W4_ROW_INDEX_UB_OFFSET + slot * A8W4_ROW_INDEX_SLOT_BYTES, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    }

    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        RouteTile(blockShape, blockCoord);
    }

    __aicore__ inline void operator()(const ProblemShape& problemShape, const BlockShape& blockShape,
                                      const BlockCoord& blockCoord, const BlockCoord& baseOffset,
                                      uint64_t groupAccumM = 0)
    {
        n_ = asc::te::get<Blaze::Gemm::MNK_N>(problemShape);
        groupAccumM_ = groupAccumM;
        BindGroupInputs(baseOffset);
        RouteTile(blockShape, blockCoord);
    }

    // Deterministic mode: flush one workspace row into y with atomic add. The caller owns
    // the workspace window and assigns rows to vector cores by row index modulo.
    __aicore__ inline void DeterministicFlushRow(uint64_t mOffset, uint64_t outRow, uint64_t nSize)
    {
        constexpr uint64_t CHUNK_SIZE = GMMFR_DQ_L0C_MAX_N;
        constexpr uint64_t BUFFER_BYTES = CHUNK_SIZE * sizeof(DataTypeOut);
        for (uint16_t slot = 0; slot < GMMFR_DQ_DOUBLE_BUFFER_COUNT; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        }
        uint16_t slot = 0;
        for (uint64_t c = 0; c < nSize; c += CHUNK_SIZE) {
            const uint64_t count = Blaze::Gemm::Min(CHUNK_SIZE, nSize - c);
            const uint64_t ubOffset = l0cUbOffset_ + slot * BUFFER_BYTES;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            CopyGmToUb(mmQuantOutGm_ + mOffset * nSize + c, ubOffset, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(slot);
            AscendC::SetAtomicAdd<DataTypeOut>();
            CopyUbToGm(yGm_ + outRow * nSize + c, ubOffset, count);
            AscendC::SetAtomicNone();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
            slot = (slot + 1) % GMMFR_DQ_DOUBLE_BUFFER_COUNT;
        }
        // Complete this row before the next contribution to the same output row.
        for (uint16_t slot = 0; slot < GMMFR_DQ_DOUBLE_BUFFER_COUNT; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        }
    }

private:
    __aicore__ inline void RouteTile(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        const uint64_t singleM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
        const uint64_t singleN = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        singleN_ = singleN;
        const uint32_t halfSingleM = Blaze::Gemm::CeilDiv(singleM, static_cast<uint64_t>(AscendC::GetTaskRation()));
        // Fixpipe split-M row pitch of the accumulator in UB.
        const uint64_t l0cAlignN = Blaze::Gemm::CeilAlign(singleN, ACC_ROW_ALIGN);
        // Every UB row is copied to GM separately, so each row start must be 32-byte aligned.
        constexpr uint64_t OUTPUT_ALIGN_ELEMENTS = IS_A8W4 ?
                                                       A8W4_OUTPUT_ALIGN_ELEMENTS :
                                                       (IS_PERTOKEN ? GMMFR_DQ_OUT_ELE_NUM_ONE_BLK /
                                                                          (sizeof(DataTypeOut) / sizeof(bfloat16_t)) :
                                                                      OUTPUT_ROW_ALIGN_BYTES / sizeof(DataTypeOut));
        const uint64_t alignN = Blaze::Gemm::CeilAlign(singleN, OUTPUT_ALIGN_ELEMENTS);
        const uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM - halfSingleM : halfSingleM;
        if constexpr (IS_A8W4) {
            logitPingPongId_ = A8W4_METADATA_EVENT_BASE + metadataReadIdx_ % GMMFR_DQ_DOUBLE_BUFFER_COUNT;
        }
        if (singleMInVec == 0) {
            if constexpr (IS_A8W4) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(logitPingPongId_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(logitPingPongId_);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(logitPingPongId_);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitPingPongId_);
                metadataReadIdx_++;
            }
            return;
        }
        const uint32_t mOffset = subBlockIdx_ * halfSingleM;
        constexpr uint64_t VECTOR_LENGTH = GMMFR_DQ_BLOCK_BYTES / sizeof(float);
        const auto repeatTimesRe = Blaze::Gemm::CeilDiv(singleN, VECTOR_LENGTH);
        const uint64_t logitOffset = asc::te::get<GMMFR_DQ_LOGIT_INDEX>(blockCoord) + mOffset;
        const uint64_t yOffset = asc::te::get<GMMFR_DQ_Y_IDX>(blockCoord);
        // A8W4 updates its entire in-place accumulator before routing, as in the original VCV flow.
        const uint64_t outputRowsPerIteration = IS_A8W4 ? singleMInVec : GMMFR_DQ_MAX_OUTPUT_M_UB;
        const uint32_t loopNumY = IS_A8W4 ? 1 : Blaze::Gemm::CeilDiv(singleMInVec, outputRowsPerIteration);
        const uint16_t remainRepeatTimesLogit = singleMInVec - (loopNumY - 1) * outputRowsPerIteration;
        blockCoord_ = blockCoord;

        const uint64_t logitUbOffset = PrepareTileInputs(singleMInVec, mOffset, logitOffset, l0cAlignN, singleN);

        if constexpr (IS_MX) {
            // The legacy caller owns no outstanding output events between tiles.
            logitPingPongId_ = (logitPingPongId_ + 1) & 1;
            for (uint16_t slot = 0; slot < OUTPUT_BUFFER_COUNT; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + slot);
            }
        }
        for (uint32_t i = 0; i < loopNumY; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yPingPongId_);
            const uint16_t repeatTimesLine = (i == loopNumY - 1) ? remainRepeatTimesLogit : outputRowsPerIteration;
            const uint64_t outUbOffset = IS_A8W4 ? outUbPingOffset_ :
                                                   (yPingPongId_ == 0 ? outUbPingOffset_ : outUbPongOffset_);
            ComputeLogit(i * outputRowsPerIteration * l0cAlignN, i * outputRowsPerIteration, repeatTimesLine,
                         repeatTimesRe, singleN, l0cAlignN, alignN, VECTOR_LENGTH, GetUbAddr<DataTypeOut>(outUbOffset),
                         GetUbAddr<float>(l0cUbOffset_), GetUbAddr<DataTypeLogit>(logitUbOffset));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yPingPongId_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yPingPongId_);
            if constexpr (IS_DETER) {
                LinearWriteToWorkspace(singleN, repeatTimesLine, logitOffset + i * outputRowsPerIteration, yOffset,
                                       outUbOffset, alignN);
            } else {
                VectorAtomicProcess(singleN, repeatTimesLine, logitOffset + i * outputRowsPerIteration, yOffset, alignN,
                                    outUbOffset);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yPingPongId_);
            if constexpr (!IS_A8W4) {
                yPingPongId_ = (yPingPongId_ + 1) & 1;
            }
        }
        if constexpr (IS_A8W4) {
            // The following weight conversion reuses this output arena.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(A8W4_OUTPUT_EVENT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(A8W4_OUTPUT_EVENT);
        }
        if constexpr (IS_MX) {
            for (uint16_t slot = 0; slot < OUTPUT_BUFFER_COUNT; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT_BASE + slot);
            }
        }
        // Release metadata only after the current vector reads and scalar row-index reads.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitPingPongId_);
        if constexpr (IS_A8W4) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(logitPingPongId_);
            metadataReadIdx_++;
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(logitPingPongId_);
            if constexpr (!IS_MX) {
                logitPingPongId_ = (logitPingPongId_ + 1) & 1;
            }
        }
    }

    __aicore__ inline void BindGroupInputs(const BlockCoord& baseOffset)
    {
        if ASCEND_IS_AIV {
            logitGm_ = reinterpret_cast<__gm__ DataTypeLogit*>(params_->logitGmAddr) +
                       asc::te::get<GMMFR_DQ_LOGIT_INDEX>(baseOffset);
            rowIndexGm_ = reinterpret_cast<__gm__ DataTypeRowIndex*>(params_->rowIndexGmAddr) +
                          asc::te::get<GMMFR_DQ_ROW_INDEX>(baseOffset);
            if constexpr (IS_PERTOKEN) {
                if (params_->x1ScaleGmAddr != nullptr) {
                    x1ScaleGm_ = reinterpret_cast<__gm__ DataTypeX1Scale*>(params_->x1ScaleGmAddr) +
                                 asc::te::get<GMMFR_DQ_X1SCALE_IDX>(baseOffset);
                }
                x2ScaleGm_ = reinterpret_cast<__gm__ DataTypeX2Scale*>(params_->x2ScaleGmAddr) +
                             asc::te::get<GMMFR_DQ_X2SCALE_IDX>(baseOffset);
                if (params_->biasGmAddr != nullptr) {
                    biasGm_ = reinterpret_cast<__gm__ BiasDtype*>(params_->biasGmAddr) +
                              asc::te::get<GMMFR_DQ_BIAS_IDX>(baseOffset);
                }
            }
        }
    }

    __aicore__ inline void InitBufferLayout()
    {
        // Preserve each producer's UB allocation and event ownership.
        if constexpr (IS_A8W4) {
            logitUbPingOffset_ = A8W4_LOGIT_UB_OFFSET;
            logitUbPongOffset_ = logitUbPingOffset_;
            outUbPingOffset_ = A8W4_ACC_UB_OFFSET;
            outUbPongOffset_ = outUbPingOffset_;
            logitPingPongId_ = A8W4_METADATA_EVENT_BASE;
            yPingPongId_ = A8W4_OUTPUT_EVENT;
            for (uint16_t slot = 0; slot < GMMFR_DQ_DOUBLE_BUFFER_COUNT; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(A8W4_METADATA_EVENT_BASE + slot);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(A8W4_METADATA_EVENT_BASE + slot);
            }
        } else {
            uint64_t offset = static_cast<uint64_t>(GMMFR_DQ_MAX_SINGLE_MN) * sizeof(DataTypeIn);
            logitUbPingOffset_ = offset;
            offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeLogit);
            logitUbPongOffset_ = offset;
            offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeLogit);
            if constexpr (IS_PERTOKEN) {
                x2ScaleUbPingOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeX2Scale);
                x2ScaleUbPongOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeX2Scale);
                x1ScaleUbPingOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeX1Scale);
                x1ScaleUbPongOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeX1Scale);
                biasUbPingOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(BiasDtype);
                biasUbPongOffset_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(BiasDtype);
            }
            outUbPingOffset_ = offset;
            offset += static_cast<uint64_t>(GMMFR_DQ_HALF_DB_MAX_SINGLE_MN) * sizeof(DataTypeOut);
            outUbPongOffset_ = offset;
            offset += static_cast<uint64_t>(GMMFR_DQ_HALF_DB_MAX_SINGLE_MN) * sizeof(DataTypeOut);
            if constexpr (IS_DETER) {
                rowIndexWindowUb_ = offset;
                offset += METADATA_MAX_ELEMENTS * sizeof(DataTypeRowIndex);
            }
            ASCENDC_ASSERT(offset <= AscendC::TOTAL_UB_SIZE,
                           { KERNEL_LOG(KERNEL_ERROR, "GMM FR epilogue UB usage %lu exceeds limit", offset); });
        }
    }

    __aicore__ inline uint64_t PrepareTileInputs(uint64_t singleMInVec, uint32_t mOffset, uint64_t logitOffset,
                                                 uint64_t l0cAlignN, uint64_t singleN)
    {
        const uint64_t logitUbOffset = IS_A8W4 ?
                                           A8W4_LOGIT_UB_OFFSET + (metadataReadIdx_ % GMMFR_DQ_DOUBLE_BUFFER_COUNT) *
                                                                      A8W4_LOGIT_SLOT_BYTES :
                                           (logitPingPongId_ == 0 ? logitUbPingOffset_ : logitUbPongOffset_);
        const uint64_t x2ScaleUbOffset = logitPingPongId_ == 0 ? x2ScaleUbPingOffset_ : x2ScaleUbPongOffset_;
        const uint64_t x1ScaleUbOffset = logitPingPongId_ == 0 ? x1ScaleUbPingOffset_ : x1ScaleUbPongOffset_;
        const uint64_t biasUbOffset = logitPingPongId_ == 0 ? biasUbPingOffset_ : biasUbPongOffset_;
        if constexpr (IS_A8W4) {
            rowIndexWindowOffset_ = logitOffset;
            rowIndexWindowUb_ = A8W4_ROW_INDEX_UB_OFFSET +
                                (metadataReadIdx_ % GMMFR_DQ_DOUBLE_BUFFER_COUNT) * A8W4_ROW_INDEX_SLOT_BYTES;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(logitPingPongId_);
        } else if constexpr (IS_MX) {
            AscendC::GlobalTensor<DataTypeLogit> logitGlobal;
            logitGlobal.SetGlobalBuffer(logitGm_);
            AscendC::LocalTensor<DataTypeLogit> logitLocal(AscendC::TPosition::VECIN, logitUbOffset,
                                                           METADATA_MAX_ELEMENTS);
            const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(singleMInVec * sizeof(DataTypeLogit)),
                                                        0, 0, 0};
            AscendC::DataCopyPadExtParams<DataTypeLogit> padParams;
            AscendC::DataCopyPad(logitLocal, logitGlobal[logitOffset], copyParams, padParams);
        } else {
            CopyGmToUb(logitGm_ + logitOffset, logitUbOffset, singleMInVec);
        }
        if constexpr (IS_PERTOKEN) {
            CopyX2ScaleFromGm2Ub(x2ScaleUbOffset);
            if (params_->x1ScaleGmAddr != nullptr) {
                CopyX1ScaleFromGm2Ub(x1ScaleUbOffset, singleMInVec * sizeof(DataTypeX1Scale), mOffset);
            }
            if (params_->biasGmAddr != nullptr) {
                CopyBiasFromGm2Ub(biasUbOffset);
            }
        }
        if constexpr (!IS_A8W4) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(logitPingPongId_);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(logitPingPongId_);

        if constexpr (IS_PERTOKEN) {
            DequantAcc(l0cAlignN, x2ScaleUbOffset, x1ScaleUbOffset, biasUbOffset, static_cast<uint16_t>(singleMInVec),
                       static_cast<uint16_t>(singleN));
        }
        return logitUbOffset;
    }

    template <typename DataType_>
    __aicore__ inline void CopyGmToUb(__gm__ DataType_* gm, uint64_t ubOffset, uint64_t count)
    {
        const auto layout = asc::te::make_layout(asc::te::make_shape(count), asc::te::make_stride(1L));
        const auto gmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gm), layout);
        const auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataType_>(ubOffset),
                                                   layout);
        asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), ubTensor, gmTensor);
    }

    template <typename DataType_>
    __aicore__ inline void CopyUbToGm(__gm__ DataType_* gm, uint64_t ubOffset, uint64_t count)
    {
        const auto layout = asc::te::make_layout(asc::te::make_shape(count), asc::te::make_stride(1L));
        const auto gmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gm), layout);
        const auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataType_>(ubOffset),
                                                   layout);
        asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), gmTensor, ubTensor);
    }

    template <class DataType_>
    __aicore__ inline static __ubuf__ DataType_* GetUbAddr(uint64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ DataType_*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    __aicore__ inline void CopyX1ScaleFromGm2Ub(uint64_t x1ScaleUbOffset, uint64_t blockLen, uint64_t offset)
    {
        CopyGmToUb(x1ScaleGm_ + asc::te::get<GMMFR_DQ_X1SCALE_IDX>(blockCoord_) + offset, x1ScaleUbOffset,
                   blockLen / sizeof(DataTypeX1Scale));
    }

    __aicore__ inline void CopyX2ScaleFromGm2Ub(uint64_t x2ScaleUbOffset)
    {
        CopyGmToUb(x2ScaleGm_ + asc::te::get<GMMFR_DQ_X2SCALE_IDX>(blockCoord_), x2ScaleUbOffset, singleN_);
    }

    __aicore__ inline void CopyBiasFromGm2Ub(uint64_t biasUbOffset)
    {
        CopyGmToUb(biasGm_ + asc::te::get<GMMFR_DQ_BIAS_IDX>(blockCoord_), biasUbOffset, singleN_);
    }

    __aicore__ inline uint64_t GetRoutingRowIndex(uint64_t offset)
    {
        if constexpr (IS_A8W4) {
            return GetPrefetchedRowIndex(offset - rowIndexWindowOffset_);
        } else if constexpr (IS_MX) {
            AscendC::GlobalTensor<DataTypeRowIndex> rowIndexGlobal;
            rowIndexGlobal.SetGlobalBuffer(rowIndexGm_);
            return static_cast<uint64_t>(rowIndexGlobal.GetValue(offset));
        } else {
            const auto layout = asc::te::make_layout(asc::te::make_shape(offset + 1), asc::te::make_stride(1L));
            const auto indices = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(rowIndexGm_),
                                                      layout);
            return static_cast<uint64_t>(indices[offset]);
        }
    }

    __aicore__ inline void VectorAtomicProcess(uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM,
                                               uint64_t yOffset, uint64_t alignN, uint64_t outUbOffset)
    {
        AscendC::SetAtomicAdd<DataTypeOut>();
        for (uint32_t i = 0; i < curVecBaseM; ++i) {
            const uint64_t outRow = GetRoutingRowIndex(offsetM + i);
            if constexpr (IS_A8W4) {
                if (static_cast<int64_t>(outRow) < 0) {
                    continue;
                }
            }
            const uint64_t outputOffset = outRow * n_ + yOffset;
            if constexpr (IS_MX) {
                AscendC::GlobalTensor<DataTypeOut> yGlobal;
                yGlobal.SetGlobalBuffer(yGm_);
                AscendC::LocalTensor<DataTypeOut> outLocal(AscendC::TPosition::VECOUT, outUbOffset,
                                                           GMMFR_DQ_HALF_DB_MAX_SINGLE_MN);
                const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0,
                                                            0, 0};
                AscendC::DataCopyPad(yGlobal[outputOffset], outLocal[i * alignN], copyParams);
            } else {
                CopyUbToGm(yGm_ + outputOffset, outUbOffset + i * alignN * sizeof(DataTypeOut), curBaseN);
            }
        }
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void LinearWriteToWorkspace(uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM,
                                                  uint64_t yOffset, uint64_t outUbOffset, uint64_t alignN)
    {
        for (uint32_t i = 0; i < curVecBaseM; ++i) {
            const uint64_t wsRow = groupAccumM_ + offsetM + i;
            const uint64_t wsIdx = wsRow * n_ + yOffset;
            CopyUbToGm(mmQuantOutGm_ + wsIdx, outUbOffset + i * alignN * sizeof(DataTypeOut), curBaseN);
        }
    }

    template <class Type>
    __aicore__ inline auto MakeUbTensor(uint64_t offset, uint64_t rows, uint64_t cols)
    {
        const auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(rows),
                                                                                   static_cast<int64_t>(cols));
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, Type>(offset), layout);
    }

    template <Tile::DequantMode X1Mode, bool HasBias>
    __aicore__ inline void RunDequant(uint64_t pitch, uint64_t x2Offset, uint64_t x1Offset, uint64_t biasOffset,
                                      uint16_t rows, uint16_t cols)
    {
        auto acc = MakeUbTensor<DataTypeIn>(l0cUbOffset_, rows, pitch);
        auto dst = MakeUbTensor<float>(l0cUbOffset_, rows, pitch);
        auto scaleB = MakeUbTensor<DataTypeX2Scale>(x2Offset, 1, cols);
        auto scaleA = MakeUbTensor<float>(x1Offset, 1, rows);
        auto bias = MakeUbTensor<BiasDtype>(biasOffset, 1, cols);
        Tile::Dequant<Tile::B32ToFpTag>::template Run<Tile::DequantMode::PER_CHANNEL, X1Mode, HasBias>(
            acc, dst, scaleB, scaleA, bias, 1.0F, 1.0F, rows, cols, static_cast<uint32_t>(pitch),
            static_cast<uint32_t>(pitch));
    }

    __aicore__ inline void DequantAcc(uint64_t pitch, uint64_t x2Offset, uint64_t x1Offset, uint64_t biasOffset,
                                      uint16_t rows, uint16_t cols)
    {
        if (params_->x1ScaleGmAddr != nullptr) {
            if (params_->biasGmAddr != nullptr) {
                RunDequant<Tile::DequantMode::PER_TOKEN, true>(pitch, x2Offset, x1Offset, biasOffset, rows, cols);
            } else {
                RunDequant<Tile::DequantMode::PER_TOKEN, false>(pitch, x2Offset, x1Offset, biasOffset, rows, cols);
            }
        } else if (params_->biasGmAddr != nullptr) {
            RunDequant<Tile::DequantMode::NONE, true>(pitch, x2Offset, x1Offset, biasOffset, rows, cols);
        } else {
            RunDequant<Tile::DequantMode::NONE, false>(pitch, x2Offset, x1Offset, biasOffset, rows, cols);
        }
    }

    static __simd_vf__ inline void VfLogitBf16(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                               uint16_t repeatTimesRe, uint64_t singleN, uint64_t l0cAlignN,
                                               uint64_t alignN, uint64_t vectorLength, __ubuf__ DataTypeOut* outUbAddr,
                                               __ubuf__ float* l0cOutUbAddr, __ubuf__ DataTypeLogit* logitUbAddr)
    {
        l0cOutUbAddr += offsetRe;
        logitUbAddr += offsetLogit;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            uint32_t elementNum = static_cast<uint32_t>(singleN);
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::Reg::UpdateMask<float>(elementNum);
                AscendC::Reg::RegTensor<bfloat16_t> vRegLogitBf16, vRegResultBf16, vRegDstBf16;
                if constexpr (AscendC::IsSameType<DataTypeLogit, bfloat16_t>::value) {
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B16>(vRegLogitBf16,
                                                                                                 logitUbAddr + i);
                } else {
                    AscendC::Reg::RegTensor<float> vRegLogitFp32;
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B32>(vRegLogitFp32,
                                                                                                 logitUbAddr + i);
                    AscendC::Reg::Cast<bfloat16_t, float, CAST_GMM_FR_FP32_TO_BF16>(vRegLogitBf16, vRegLogitFp32, mask);
                }
                AscendC::Reg::RegTensor<float> vRegResultFp32;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vRegResultFp32, l0cOutUbAddr + i * l0cAlignN + j * vectorLength);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_GMM_FR_FP32_TO_BF16>(vRegResultBf16, vRegResultFp32, mask);
                AscendC::Reg::Mul(vRegDstBf16, vRegLogitBf16, vRegResultBf16, mask);
                AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    outUbAddr + i * alignN + j * vectorLength, vRegDstBf16, mask);
            }
        }
    }

    static __simd_vf__ inline void VfLogitFp32(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                               uint16_t repeatTimesRe, uint64_t singleN, uint64_t l0cAlignN,
                                               uint64_t alignN, uint64_t vectorLength, __ubuf__ DataTypeOut* outUbAddr,
                                               __ubuf__ float* l0cOutUbAddr, __ubuf__ DataTypeLogit* logitUbAddr)
    {
        l0cOutUbAddr += offsetRe;
        logitUbAddr += offsetLogit;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            AscendC::Reg::RegTensor<float> vRegLogit;
            uint32_t elementNum = static_cast<uint32_t>(singleN);
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::Reg::UpdateMask<float>(elementNum);
                if constexpr (AscendC::IsSameType<DataTypeLogit, bfloat16_t>::value) {
                    AscendC::Reg::RegTensor<bfloat16_t> vRegLogitBf16;
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B16>(vRegLogitBf16,
                                                                                                 logitUbAddr + i);
                    AscendC::Reg::Cast<float, bfloat16_t, CAST_GMM_FR_BF16_TO_FP32_ZERO>(vRegLogit, vRegLogitBf16,
                                                                                         mask);
                } else {
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B32>(vRegLogit,
                                                                                                 logitUbAddr + i);
                }
                AscendC::Reg::RegTensor<float> vRegResult;
                AscendC::Reg::RegTensor<DataTypeOut> vRegDst;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vRegResult, l0cOutUbAddr + i * l0cAlignN + j * vectorLength);
                if constexpr (IS_A8W4) {
                    AscendC::Reg::RegTensor<float> scaled;
                    AscendC::Reg::Muls(scaled, vRegResult, A8W4_WEIGHT_SCALE, mask);
                    AscendC::Reg::Mul(vRegDst, scaled, vRegLogit, mask);
                } else {
                    AscendC::Reg::Mul(vRegDst, vRegLogit, vRegResult, mask);
                }
                AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    outUbAddr + i * alignN + j * vectorLength, vRegDst, mask);
            }
        }
    }

    __aicore__ inline static void ComputeLogit(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                               uint16_t repeatTimesRe, uint64_t singleN, uint64_t l0cAlignN,
                                               uint64_t alignN, uint64_t vectorLength, __ubuf__ DataTypeOut* outUbAddr,
                                               __ubuf__ float* l0cOutUbAddr, __ubuf__ DataTypeLogit* logitUbAddr)
    {
        if constexpr (AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
            asc_vf_call<VfLogitBf16>(offsetRe, offsetLogit, repeatTimesLogit, repeatTimesRe, singleN, l0cAlignN, alignN,
                                     vectorLength, outUbAddr, l0cOutUbAddr, logitUbAddr);
        } else {
            asc_vf_call<VfLogitFp32>(offsetRe, offsetLogit, repeatTimesLogit, repeatTimesRe, singleN, l0cAlignN, alignN,
                                     vectorLength, outUbAddr, l0cOutUbAddr, logitUbAddr);
        }
    }

    // GM ADDR
    __gm__ DataTypeLogit* logitGm_{nullptr};
    __gm__ DataTypeRowIndex* rowIndexGm_{nullptr};
    __gm__ DataTypeX1Scale* x1ScaleGm_{nullptr};
    __gm__ DataTypeX2Scale* x2ScaleGm_{nullptr};
    __gm__ BiasDtype* biasGm_{nullptr};
    __gm__ DataTypeOut* yGm_{nullptr};
    __gm__ DataTypeRowIndex* rowIndexBaseGm_{nullptr};
    __gm__ DataTypeOut* mmQuantOutGm_{nullptr};

    const Params* params_{nullptr};
    bool initialized_{false};
    int64_t n_{0};
    uint32_t subBlockIdx_{0};
    uint64_t singleN_{0};
    uint16_t yPingPongId_{0};
    uint16_t logitPingPongId_{0};
    uint64_t groupAccumM_{0};
    uint64_t rowIndexWindowOffset_{0};
    uint64_t rowIndexWindowUb_{A8W4_ROW_INDEX_UB_OFFSET};
    uint64_t metadataReadIdx_{0};
    uint64_t metadataWriteIdx_{0};
    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};

    // UB byte offsets (per AIV sub-block arena).
    static constexpr uint64_t L0C_UB_OFFSET = IS_A8W4 ? A8W4_ACC_UB_OFFSET : 0;
    uint64_t l0cUbOffset_{L0C_UB_OFFSET};
    // Per-token MatmulWithScaleMix uses a 256B row pitch; MX/A8W4 use Align32.
    static constexpr uint64_t ACC_ROW_ALIGN = IS_PERTOKEN ? GMMFR_DQ_L0C_UB_ROW_ALIGN : MX_ACC_ROW_ALIGN_ELEMENTS;
    uint64_t logitUbPingOffset_{0};
    uint64_t logitUbPongOffset_{0};
    uint64_t x2ScaleUbPingOffset_{0};
    uint64_t x2ScaleUbPongOffset_{0};
    uint64_t x1ScaleUbPingOffset_{0};
    uint64_t x1ScaleUbPongOffset_{0};
    uint64_t biasUbPingOffset_{0};
    uint64_t biasUbPongOffset_{0};
    uint64_t outUbPingOffset_{0};
    uint64_t outUbPongOffset_{0};
};

} // namespace Blaze::Epilogue::Block

namespace Blaze::Gemm::Block {
using Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting;
}
#endif
#endif
