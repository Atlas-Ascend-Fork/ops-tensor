/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file block_scheduler_matmul_streamk.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerMatmulStreamK {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t usedCoreNum{0};
        int64_t baseM{0};
        int64_t baseN{0};
        int64_t baseK{0};
        int64_t singleCoreK{0};
        int64_t kL1{0};
        uint8_t isHf32{0};
        uint32_t l2CacheMode = L2_CACHE_DEFAULT;
        uint32_t windowLen = 4; // ASW滑窗长度，由tiling自适应下发
    };

public:
    __aicore__ inline BlockSchedulerMatmulStreamK(const ProblemShape& shape, const Params& params)
    {
        usedCoreNums_ = params.usedCoreNum;
        windowLen_ = params.windowLen == 0 ? 4 : params.windowLen; // 0视为未下发，回退默认4防除零
        if (usedCoreNums_ <= 0) {
            return;
        }
        m_ = asc::te::get<MNK_M>(shape);
        n_ = asc::te::get<MNK_N>(shape);
        k_ = asc::te::get<MNK_K>(shape);
        batch_ = AscendC::Std::max(asc::te::get<MNK_B>(shape), 1L);

        mL1_ = params.baseM;                 // size of m in L1 & L0 & singlecore, per core use L1 once in stream k
        nL1_ = params.baseN;                 // size of n in L1 & L0 & singlecore, per core use L1 once in stream k
        skSingleCoreK_ = params.singleCoreK; // size of k in singlecore

        mBlockNums_ = CeilDiv(m_, mL1_);
        nBlockNums_ = CeilDiv(n_, nL1_);
        skBlockNums_ = CeilDiv(k_, skSingleCoreK_);
        int64_t tailMNBlockNums = (mBlockNums_ * nBlockNums_) % usedCoreNums_; // tail mCnt * nCnt num of SK
        // core num of DP (m*n) + tail core num of SK (m*n*k)
        blockNums_ = (mBlockNums_ * nBlockNums_ - tailMNBlockNums) + tailMNBlockNums * skBlockNums_;
        totalMNBlockNumsInDP_ = mBlockNums_ * nBlockNums_ - tailMNBlockNums;
    }

    /**
       获取总的分块数
    */
    __aicore__ inline int64_t GetBlockNums() { return blockNums_ * batch_; }

    /**
       获取需要的核数
    */
    __aicore__ inline int64_t GetCoreNums()
    {
        int64_t tilingBlockNum = 0;
        if (blockNums_ * batch_ < AscendC::GetBlockNum()) {
            tilingBlockNum = blockNums_ * batch_;
        } else {
            tilingBlockNum = AscendC::GetBlockNum();
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
    {
        UpdateMNBlockIdx(blockIdx);
        int64_t tailL1M = m_ - (mBlockNums_ - 1) * mL1_;
        int64_t tailL1N = n_ - (nBlockNums_ - 1) * nL1_;
        int64_t tailSingleCoreK = k_ - (curKBlockNums_ - 1) * skSingleCoreK_;
        int64_t blkM = (mBlockIdx_ == (mBlockNums_ - 1)) ? tailL1M : mL1_;
        int64_t blkN = (nBlockIdx_ == (nBlockNums_ - 1)) ? tailL1N : nL1_;
        int64_t blkK = (kBlockIdx_ == (curKBlockNums_ - 1)) ? tailSingleCoreK : skSingleCoreK_;
        return {blkM, blkN, blkK, 0};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx)
    {
        UpdateMNBlockIdx(blockIdx);
        return {mBlockIdx_, nBlockIdx_, kBlockIdx_, bBlockIdx_};
    }

    __aicore__ inline bool CheckIsSkScene(int64_t blockIdx)
    {
        return CeilDiv((blockIdx + 1), usedCoreNums_) == CeilDiv(blockNums_, usedCoreNums_); // true is sk, false is dp
    }

private:
    __aicore__ inline void UpdateMNBlockIdx(int64_t blockIdx)
    {
        if (mBlockNums_ <= 0 || nBlockNums_ <= 0 || skBlockNums_ <= 0) {
            return;
        }
        // judge now in dp loop (kTileNum = 1) or in sk loop
        curKBlockNums_ = CheckIsSkScene(blockIdx) ? skBlockNums_ : 1;
        bBlockIdx_ = blockIdx / (mBlockNums_ * nBlockNums_ * skBlockNums_);
        int64_t mnkBlockIdx = blockIdx % (mBlockNums_ * nBlockNums_ * skBlockNums_);
        int64_t mnIdxInCurLoop = 0;
        if (CheckIsSkScene(blockIdx)) { // SK scene
            kBlockIdx_ = (mnkBlockIdx % usedCoreNums_) % curKBlockNums_;
            mnIdxInCurLoop = (mnkBlockIdx % usedCoreNums_) / curKBlockNums_ + totalMNBlockNumsInDP_;
        } else { // DP scene
            kBlockIdx_ = 0;
            mnIdxInCurLoop = mnkBlockIdx / curKBlockNums_;
        }
        int64_t mainWindow = AscendC::Std::min(windowLen_, mBlockNums_);
        int64_t mainRow = mBlockNums_ / mainWindow - 1UL;
        int64_t tailWindow = mBlockNums_ - mainRow * mainWindow;
        int64_t rowIdx = mnIdxInCurLoop / nBlockNums_ / mainWindow;
        if (rowIdx < mainRow) {
            mBlockIdx_ = rowIdx * mainWindow + mnIdxInCurLoop % mainWindow;
            nBlockIdx_ = (mnIdxInCurLoop / mainWindow) % nBlockNums_;
        } else {
            rowIdx = mainRow;
            int64_t tailIndex = mnIdxInCurLoop - mainRow * mainWindow * nBlockNums_;
            mBlockIdx_ = mainRow * mainWindow + tailIndex % tailWindow;
            nBlockIdx_ = (tailIndex / tailWindow) % nBlockNums_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % 2 != 0UL) {
            nBlockIdx_ = nBlockNums_ - 1UL - nBlockIdx_;
        }
    }

private:
    int64_t usedCoreNums_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
    int64_t blockNums_{1};
    int64_t totalMNBlockNumsInDP_{0};
    int64_t windowLen_{4};

    int64_t batch_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};

    int64_t mBlockIdx_{1};
    int64_t nBlockIdx_{1};
    int64_t kBlockIdx_{1};
    int64_t bBlockIdx_{1};
    int64_t curKBlockNums_{1};

    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t skSingleCoreK_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
