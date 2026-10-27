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
 * \file block_epilogue_gelu_fixpipe.h
 * \brief AIV GELU-tanh epilogue for grouped matmul: shared UB fixpipe output -> GELU -> GM.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_ = float>
class BlockEpilogueGeluFixpipe {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    struct Params {
        Params() = default;
        uint64_t splitM{1};
    };

    // A 256 x 256 FP32 tile is split across two AIVs into 128 x 256 elements (128 KiB per AIV).
    // Together with a 64 KiB 16-bit output staging area it cannot fit in half of the 248 KiB usable UB,
    // so the FP32 GELU path uses one UB slot.
    static constexpr uint8_t UB_BUFFER_DEPTH = 1;
    static constexpr uint32_t ROW_PITCH_GRANULARITY = static_cast<uint32_t>(asc_get_vf_len() / sizeof(float));
    // A single AIV reuses the same UB slot and consumes a full Cube tile in M-axis stages.
    // Keep the stage equal to the per-AIV height of the established two-AIV path so the
    // worst-case FP32 source and narrowed output staging remain within the usable UB.
    static constexpr uint32_t SINGLE_AIV_STAGE_M = 128;

    __aicore__ inline BlockEpilogueGeluFixpipe()
    {
        static_assert(AscendC::Std::is_same_v<DataTypeIn, float>, "GMM GELU Fixpipe input must be FP32.");
        static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                          AscendC::Std::is_same_v<DataTypeOut, half>,
                      "GMM GELU Fixpipe output must be float, bfloat16_t or half.");
    }

    __aicore__ inline void Init(const Params& params) { splitM_ = params.splitM != 0; }

    __aicore__ inline void operator()(__gm__ DataTypeOut* blockCPtr, int64_t blockM, int64_t blockN, int64_t n,
                                      int64_t baseN)
    {
        if (!splitM_ && AscendC::GetSubBlockIdx() != 0) {
            return;
        }

        const int64_t taskRatio = static_cast<int64_t>(AscendC::GetTaskRation());
        const int64_t subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx()) & 1L;
        const int64_t halfM = splitM_ ? Gemm::CeilDiv(blockM, taskRatio) : 0L;
        const int64_t curBaseN = Gemm::Min(blockN, baseN);
        const int64_t nIter = Gemm::CeilDiv(blockN, curBaseN);
        constexpr int64_t ubSlotBytes = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / UB_BUFFER_DEPTH);
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});

        for (int64_t nIdx = 0; nIdx < nIter; ++nIdx) {
            const int64_t tileN = nIdx + 1 == nIter ? blockN - curBaseN * nIdx : curBaseN;
            const int64_t rowPitch = Gemm::CeilAlign(tileN, static_cast<int64_t>(ROW_PITCH_GRANULARITY));
            const int64_t stageCount = splitM_ ? 1L : Gemm::CeilDiv(blockM, static_cast<int64_t>(SINGLE_AIV_STAGE_M));
            for (int64_t stageIdx = 0; stageIdx < stageCount; ++stageIdx) {
                const int64_t localMOffset = splitM_ ? subBlockIdx * halfM :
                                                       stageIdx * static_cast<int64_t>(SINGLE_AIV_STAGE_M);
                const int64_t localRows = localMOffset < blockM ?
                                              Gemm::Min(splitM_ ? halfM : static_cast<int64_t>(SINGLE_AIV_STAGE_M),
                                                        blockM - localMOffset) :
                                              0L;
                const uint16_t slot = UB_BUFFER_DEPTH > 1U ? static_cast<uint16_t>(blockCnt_ & 1UL) : 0U;

                // Consume each slot's READY on PIPE_V, including an empty split-M half.
                // Later groups can give this AIV rows, so the counter must not switch pipelines.
                Gemm::Sync::WaitForCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(Gemm::Sync::FIXPIPE_AIC_READY_FLAG_BASE +
                                                                             slot);
                if (localRows > 0) {
                    const int64_t slotBaseBytes = static_cast<int64_t>(slot) * ubSlotBytes;
                    // Fixpipe pads an odd M before DUAL_DST_SPLIT_M, so reserve all physical rows in
                    // two-AIV mode. A single AIV receives an exact M-stage and needs no extra row.
                    const int64_t physicalRows = splitM_ ? halfM : localRows;
                    const int64_t srcBytes = physicalRows * rowPitch * static_cast<int64_t>(sizeof(DataTypeIn));
                    int64_t dstOffsetBytes = slotBaseBytes;
                    if constexpr (!AscendC::Std::is_same_v<DataTypeOut, DataTypeIn>) {
                        dstOffsetBytes += Gemm::CeilAlign(srcBytes, static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
                    }

                    auto ubGeluSrcLayout = Gemm::MakeNDExtLayout<DataTypeIn>(localRows, rowPitch, rowPitch);
                    auto ubGeluSrc = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(slotBaseBytes), ubGeluSrcLayout);
                    auto ubGeluDstLayout = Gemm::MakeNDExtLayout<DataTypeOut>(localRows, rowPitch, rowPitch);
                    auto ubGeluDst = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::ub, DataTypeOut>(dstOffsetBytes), ubGeluDstLayout);
                    Tile::Gelu<Tile::GeluTanhTag>::Run(ubGeluSrc, ubGeluDst);

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GELU_V_MTE3_EVENT_ID);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GELU_V_MTE3_EVENT_ID);
                    const int64_t gmOffset = localMOffset * n + nIdx * curBaseN;
                    auto gmLayout = Gemm::MakeNDExtLayout<DataTypeOut>(localRows, tileN, n);
                    auto gmTile = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(blockCPtr + gmOffset), gmLayout);
                    auto ubCopyLayout = Gemm::MakeNDExtLayout<DataTypeOut>(localRows, tileN, rowPitch);
                    auto ubCopy = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::ub, DataTypeOut>(dstOffsetBytes), ubCopyLayout);
                    asc::te::copy(copyUbToGm, gmTile, ubCopy);
                } else {
                    // Order the MTE3 ACK after the V-side READY wait even when no data is consumed.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GELU_V_MTE3_EVENT_ID);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GELU_V_MTE3_EVENT_ID);
                }
                Gemm::Sync::NotifyCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_MTE3>(Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE +
                                                                               slot);
                blockCnt_++;
            }
        }
    }

private:
    static constexpr uint8_t GELU_V_MTE3_EVENT_ID = 1;
    bool splitM_{true};
    uint64_t blockCnt_{0};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
