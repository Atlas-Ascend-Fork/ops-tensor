/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_matmul_syrk.h
 * \brief Compact upper-triangle scheduler for syrk.
 *
 * Unlike BlockSchedulerMatmulBasic, whose linear index space covers the full
 * M x N grid and leaves the triangle filter to the consumer, this scheduler
 * enumerates ONLY the upper-triangle slots (coordN >= coordM, the mirrored
 * (j, i) tile is produced by the transposed epilogue). With stride polling
 * over a compact index space every core receives slots/cores +- 1 tiles
 * instead of locking onto triangle-density columns, which removes the
 * last-core makespan (~2x on square grids sized to the core count).
 *
 * Index map (per batch): slot t in [0, T) decodes to grid (rowIdx, colIdx)
 * with row i holding columns [i, nBlockNums) in order; odd rows scan the
 * row right-to-left (snake) to preserve the A-row-block L2 locality of the
 * basic scheduler. Row lookup is an exact binary search over the row-start
 * offsets -- no float sqrt, no per-tile O(R) scan.
 *
 * Host contract: no per-core tail splitting -- each axis has a single tail
 * block that carries the full remainder (the syrk tiling enforces this), so
 * the tail-split machinery (tail count / split count / tail main) of the
 * basic scheduler is absent.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerSyrkTriangular {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        // The syrk contract collapses the block geometry into ONE symmetric
        // square size: mL1 == nL1 == baseM == baseN (enforced by the host
        // tiling). The kernel also consumes it as the epilogue row clamp /
        // N chunk bound.
        uint32_t block = 0;
    };

public:
    __aicore__ inline BlockSchedulerSyrkTriangular(const ProblemShape& shape, const Params& params)
    {
        k_ = asc::te::get<MNK_K>(shape);
        batch_ = AscendC::Std::max(asc::te::get<MNK_B>(shape), 1L);
        mL1_ = params.block;
        nL1_ = params.block;
        blockNum_ = AscendC::GetBlockNum();
        const int64_t m = asc::te::get<MNK_M>(shape);
        const int64_t n = asc::te::get<MNK_N>(shape);
        mBlockNums_ = CeilDiv(static_cast<uint32_t>(m), params.block);
        nBlockNums_ = CeilDiv(static_cast<uint32_t>(n), params.block);
        if (blockNum_ <= 0 || mBlockNums_ <= 0 || nBlockNums_ <= 0) {
            return;
        }
        // Rows >= nBlockNums_ hold no upper-triangle slot (coordN <= nBlockNums_-1 < coordM).
        effRows_ = AscendC::Std::min(mBlockNums_, nBlockNums_);
        slotsPerBatch_ = effRows_ * nBlockNums_ - effRows_ * (effRows_ - 1) / 2;
        // Single tail block per axis: the last block carries the full remainder.
        tailM_ = m - (mBlockNums_ - 1) * mL1_;
        tailN_ = n - (nBlockNums_ - 1) * nL1_;
    }

    __aicore__ inline int64_t GetBlockNums() { return slotsPerBatch_ * batch_; }

    __aicore__ inline int64_t GetCoreNums()
    {
        const int64_t total = GetBlockNums();
        return total < blockNum_ ? total : blockNum_;
    }

    template <bool TransB_ = false, class BType_>
    __aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
    {
        UpdateBlockIdx(blockIdx);
        const int64_t blkM = mBlockIdx_ == (mBlockNums_ - 1) ? tailM_ : mL1_;
        const int64_t blkN = nBlockIdx_ == (nBlockNums_ - 1) ? tailN_ : nL1_;
        return {blkM, blkN, k_, batch_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx)
    {
        UpdateBlockIdx(blockIdx);
        const int64_t batchIdx = batch_ > 1 ? blockIdx / slotsPerBatch_ : 0;
        return {mBlockIdx_ * mL1_, nBlockIdx_ * nL1_, 0, batchIdx};
    }

private:
    // slot -> (mBlockIdx_, nBlockIdx_): binary search on row starts, then
    // snake-scan within the row. offset(i) = i * nBlockNums_ - i*(i-1)/2.
    __aicore__ inline void UpdateBlockIdx(int64_t blockIdx)
    {
        if (lastSlot_ == blockIdx) {
            return;
        }
        lastSlot_ = blockIdx;
        // Anti-orbit-collapse: rotate the within-batch slot order by the batch
        // index. When blockDim is a multiple of slotsPerBatch_ the plain
        // "slot = blockIdx % slotsPerBatch_" locks every core onto ONE fixed
        // (i, j) position for all batches (observed with blockDim ==
        // slotsPerBatch_ == 28): each core then repeats an identical
        // lock/flag operation sequence for the whole kernel, which wedges the
        // on-core sync semaphores (MTE1 L1->L0 copy stops retiring on the 7
        // diagonal-position cores, AIV vector op stalls 7 slots later). The
        // per-batch rotation keeps the slot map a bijection, preserves the
        // snake adjacency inside a batch, and spreads the diagonal slots
        // across all cores; a constant orbit now requires slotsPerBatch_^2 to
        // divide blockDim, which cannot happen for realistic core counts.
        const int64_t batchIdx = batch_ > 1 ? blockIdx / slotsPerBatch_ : 0;
        const int64_t slot = batch_ > 1 ? (blockIdx + batchIdx) % slotsPerBatch_ : blockIdx;
        int64_t low = 0;
        int64_t high = effRows_ - 1;
        while (low < high) {
            const int64_t mid = (low + high + 1) / 2;
            const int64_t off = mid * nBlockNums_ - mid * (mid - 1) / 2;
            if (off <= slot) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        mBlockIdx_ = low;
        const int64_t rowStart = low * nBlockNums_ - low * (low - 1) / 2;
        const int64_t d = slot - rowStart; // in [0, nBlockNums_-1-low]
        // Snake: even rows scan left-to-right, odd rows right-to-left.
        nBlockIdx_ = (low % 2 == 0) ? (low + d) : (nBlockNums_ - 1 - d);
    }

private:
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t effRows_{0};
    int64_t slotsPerBatch_{0};
    int64_t blockNum_{0};
    int64_t batch_{1};
    int64_t k_{0};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t tailM_{0};
    int64_t tailN_{0};
    int64_t mBlockIdx_{0};
    int64_t nBlockIdx_{0};
    int64_t lastSlot_{-1};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
