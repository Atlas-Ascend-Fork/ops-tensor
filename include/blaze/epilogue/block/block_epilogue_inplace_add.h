/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file block_epilogue_inplace_add.h
 * \brief FP32 Fixpipe UB result + existing GM output -> UB add -> GM.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/epilogue/block/block_epilogue_types.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze::Epilogue::Block {

// Cube result slots and old-C chunks occupy disjoint UB regions. The result
// slots can be released on PIPE_V; outstanding GM stores only read old-C chunks.
class BlockEpilogueInplaceAdd {
public:
    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    struct Params {
        int64_t baseM{128};
        int64_t baseN{128};
        int64_t ubPanelM{0};
    };

    __aicore__ inline void Init(const Params& params)
    {
        config_ = MakeBufferConfig(params.baseM, params.baseN, static_cast<int64_t>(AscendC::TOTAL_UB_SIZE),
                                   params.ubPanelM);
        if (config_.stageM <= 0) {
            return;
        }
        if ASCEND_IS_AIV {
            for (uint16_t slot = 0; slot < BUFFER_COUNT; ++slot) {
                Gemm::Sync::NotifyCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE +
                                                                            slot);
            }
        }
    }

    // Both processors use the same two result slots, alternating by panel index.
    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const
    {
        const int64_t panelRows = Gemm::Min(params.rows, config_.stageM);
        const int64_t rowPitch = Gemm::CeilAlign(params.cols, static_cast<int64_t>(ROW_PITCH_GRANULARITY));
        const uint16_t slot = static_cast<uint16_t>(params.slot & 1U);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(slot * config_.resultBytes),
                                    Gemm::MakeNDExtLayout<float>(panelRows, params.cols, rowPitch));
    }

    __aicore__ inline ~BlockEpilogueInplaceAdd()
    {
        if (config_.stageM <= 0) {
            return;
        }
        if ASCEND_IS_AIC {
            for (uint16_t slot = 0; slot < BUFFER_COUNT; ++slot) {
                Gemm::Sync::WaitForVector<Gemm::Sync::SYNC_MODE_INTRA, PIPE_FIX>(Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE +
                                                                                 slot);
            }
        }
    }

    template <typename TensorUbC, typename TensorC>
    __aicore__ inline void operator()(const TensorUbC& ubResult, const TensorC& gmC, int64_t m, int64_t n,
                                      uint64_t panelIndex = 0);

private:
    static constexpr uint32_t ROW_PITCH_GRANULARITY = 64;
    static constexpr uint16_t BUFFER_COUNT = 2;
    static constexpr int64_t MAX_CHUNK_ROWS = 64;

    struct BufferConfig {
        int64_t chunkRows{0};
        int64_t stageM{0};
        int64_t resultBytes{0};
        int64_t chunkBytes{0};
    };

    // UB layout: [result 0][result 1][old C 0][old C 1].
    // Plan in padded FP32 rows so the byte budget becomes:
    // 2 * panelRows + 2 * inputChunkRows <= totalRows.
    __aicore__ inline static constexpr BufferConfig MakeBufferConfig(int64_t baseM, int64_t baseN, int64_t ubBytes,
                                                                     int64_t ubPanelM = 0)
    {
        constexpr int64_t PANEL_ROW_ALIGNMENT = 16;
        constexpr int64_t INPUT_RESERVE_ROWS = 16;
        if (baseM <= 0 || baseN <= 0 || ubBytes <= 0 || ubPanelM < 0 || ubPanelM > baseM) {
            return {};
        }

        const int64_t rowPitch = (baseN + ROW_PITCH_GRANULARITY - 1) / ROW_PITCH_GRANULARITY * ROW_PITCH_GRANULARITY;
        const int64_t rowBytes = rowPitch * sizeof(float);
        const int64_t totalRows = ubBytes / rowBytes;
        // Ping/pong requires two result slots and two old-C chunks, each at least one row.
        if (totalRows < BUFFER_COUNT * 2) {
            return {};
        }

        // Reserve old-C space first. Each result/input pair gets half of UB;
        // the initial input reservation is at most half of that pair's budget.
        const int64_t rowsPerBufferPair = totalRows / BUFFER_COUNT;
        const int64_t reserveLimit = rowsPerBufferPair / 2;
        const int64_t reservedInputRows = reserveLimit < INPUT_RESERVE_ROWS ? reserveLimit : INPUT_RESERVE_ROWS;
        const int64_t initialInputRows = baseM < reservedInputRows ? baseM : reservedInputRows;
        const int64_t maxPanelRows = rowsPerBufferPair - initialInputRows;

        // Prefer a full tile. Otherwise aim for two balanced, aligned panels;
        // if even half a tile cannot fit, use the aligned capacity instead.
        int64_t panelRows = baseM;
        if (baseM > maxPanelRows) {
            const int64_t halfTileRows = (baseM + 1) / 2;
            const int64_t alignedHalfTileRows = (halfTileRows + PANEL_ROW_ALIGNMENT - 1) / PANEL_ROW_ALIGNMENT *
                                                PANEL_ROW_ALIGNMENT;
            const int64_t alignedCapacity = maxPanelRows >= PANEL_ROW_ALIGNMENT ?
                                                maxPanelRows / PANEL_ROW_ALIGNMENT * PANEL_ROW_ALIGNMENT :
                                                maxPanelRows;
            panelRows = alignedHalfTileRows < alignedCapacity ? alignedHalfTileRows : alignedCapacity;
        }

        // An explicit tiling value must leave at least one old-C row per buffer.
        if (ubPanelM > 0) {
            if (ubPanelM >= rowsPerBufferPair) {
                return {};
            }
            panelRows = ubPanelM;
        }

        // Give the remaining space to old C, capped at one panel and 64 rows.
        const int64_t remainingInputRows = rowsPerBufferPair - panelRows;
        const int64_t cappedInputRows = remainingInputRows < MAX_CHUNK_ROWS ? remainingInputRows : MAX_CHUNK_ROWS;
        const int64_t inputChunkRows = cappedInputRows < panelRows ? cappedInputRows : panelRows;
        BufferConfig config{};
        config.stageM = panelRows;
        config.chunkRows = inputChunkRows;
        config.resultBytes = panelRows * rowBytes;
        config.chunkBytes = inputChunkRows * rowBytes;
        return config;
    }

    __aicore__ inline Gemm::BufferSlot GetInputSlot(uint16_t buffer) const
    {
        const int64_t offset = BUFFER_COUNT * config_.resultBytes + buffer * config_.chunkBytes;
        return {static_cast<uint64_t>(offset), static_cast<uint8_t>(buffer)};
    }

    __aicore__ inline auto MakeInputTensor(uint16_t buffer, int64_t rows, int64_t n, int64_t rowPitch) const
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(GetInputSlot(buffer).Addr()),
                                    Gemm::MakeNDExtLayout<float>(rows, n, rowPitch));
    }

    template <typename TensorC>
    __aicore__ inline void Prefetch(const TensorC& gmC, int64_t row, int64_t m, int64_t n, int64_t rowPitch,
                                    uint16_t buffer)
    {
        const int64_t rows = Gemm::Min(config_.chunkRows, m - row);
        auto oldChunk = MakeInputTensor(buffer, rows, n, rowPitch);
        auto gmChunk = gmC.slice(asc::te::make_coord(row, 0L), asc::te::make_shape(rows, n));
        const auto inputSlot = GetInputSlot(buffer);
        auto lock = inputSlot.LockMte2();
        asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), oldChunk, gmChunk);
    }

    static __simd_vf__ inline void AddChunk(__ubuf__ float* old, __ubuf__ float* delta, uint32_t rows, uint32_t n,
                                            uint32_t rowPitch)
    {
        constexpr uint32_t FLOATS_PER_VECTOR = AscendC::GetVecLen() / sizeof(float);
        // A contiguous multirow chunk needs only one vector loop. Keep the row
        // loop for padded N tails so padding never participates in the output.
        const uint32_t rowCount = n == rowPitch ? 1U : rows;
        const uint32_t elements = n == rowPitch ? rows * n : n;
        const uint16_t repeats = static_cast<uint16_t>((elements + FLOATS_PER_VECTOR - 1) / FLOATS_PER_VECTOR);
        for (uint32_t row = 0; row < rowCount; ++row) {
            uint32_t remaining = elements;
            for (uint16_t i = 0; i < repeats; ++i) {
                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                const uint32_t offset = row * rowPitch + i * FLOATS_PER_VECTOR;
                AscendC::Reg::RegTensor<float> oldReg;
                AscendC::Reg::RegTensor<float> deltaReg;
                AscendC::Reg::DataCopy(oldReg, old + offset);
                AscendC::Reg::DataCopy(deltaReg, delta + offset);
                AscendC::Reg::Add(oldReg, oldReg, deltaReg, mask);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(old + offset, oldReg, mask);
            }
        }
    }

    BufferConfig config_{};
    uint16_t inputBuffer_{0};
};

template <typename TensorUbC, typename TensorC>
__aicore__ inline void BlockEpilogueInplaceAdd::operator()(const TensorUbC& ubResult, const TensorC& gmC, int64_t m,
                                                           int64_t n, uint64_t panelIndex)
{
    const uint16_t slot = static_cast<uint16_t>(panelIndex & 1UL);
    using ElementType = asc::te::get_attribute_element_type<typename TensorC::element_type*>;
    static_assert(AscendC::Std::is_same_v<ElementType, float>, "InplaceAdd requires FP32 output.");
    const int64_t rowPitch = Gemm::CeilAlign(n, static_cast<int64_t>(ROW_PITCH_GRANULARITY));
    const int64_t chunkCount = Gemm::CeilDiv(m, config_.chunkRows);
    // MTE2 can fetch old C while the paired Cube is still computing. Two
    // chunks can be issued before waiting for the first Fixpipe result.
    for (uint16_t buffer = 0; buffer < BUFFER_COUNT && buffer < chunkCount; ++buffer) {
        const uint16_t input = static_cast<uint16_t>(inputBuffer_ ^ buffer);
        Prefetch(gmC, buffer * config_.chunkRows, m, n, rowPitch, input);
    }
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<TensorUbC>, asc::te::location::ub>,
                  "InplaceAdd matmul intermediate tensor must be located in UB.");
    auto result = ubResult.data().get();
    Gemm::Sync::WaitForCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(Gemm::Sync::FIXPIPE_AIC_READY_FLAG_BASE + slot);
    for (int64_t chunk = 0; chunk < chunkCount; ++chunk) {
        const uint16_t buffer = inputBuffer_;
        const int64_t row = chunk * config_.chunkRows;
        const int64_t rows = Gemm::Min(config_.chunkRows, m - row);
        auto oldChunk = MakeInputTensor(buffer, rows, n, rowPitch);
        auto gmChunk = gmC.slice(asc::te::make_coord(row, 0L), asc::te::make_shape(rows, n));
        const auto inputSlot = GetInputSlot(buffer);
        {
            auto lock = inputSlot.LockV();
            asc_vf_call<AddChunk>(oldChunk.data().get(), result + row * rowPitch, static_cast<uint32_t>(rows),
                                  static_cast<uint32_t>(n), static_cast<uint32_t>(rowPitch));
        }
        {
            auto lock = inputSlot.LockMte3();
            asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), gmChunk, oldChunk);
        }
        // Reuse this input buffer only after its store. The other buffer's
        // load/add/store can proceed independently in the meantime.
        const int64_t nextRow = (chunk + BUFFER_COUNT) * config_.chunkRows;
        if (nextRow < m) {
            Prefetch(gmC, nextRow, m, n, rowPitch, buffer);
        }
        // Keep the input-buffer phase across panels, including odd chunk counts.
        inputBuffer_ ^= 1;
    }
    // Addition stores into oldChunk, so Fixpipe may reuse the result slot
    // as soon as Vector has finished reading it, without draining MTE3.
    Gemm::Sync::NotifyCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot);
}

} // namespace Blaze::Epilogue::Block
