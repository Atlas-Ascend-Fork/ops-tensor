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
 * \file block_scheduler_qbmm.h
 * \brief Block scheduler for QBMM with SWAT windowing and tail tile splitting.
 *
 * Maps logical M/N block coordinates to physical cores using an adaptive sliding
 * window traversal for L2 cache locality. Splits the tail M/N block across idle cores
 * via UpdateTailTile, and supports A_FULL_LOAD_MODE scheduling variant.
 */

#pragma once

#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_, uint64_t FullLoadMode_, class LayoutA_, class LayoutB_, class AType_>
class BlockSchedulerQuantBatchMatmulV3 {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    using AType = AType_;

    struct Params {
        int64_t baseM;
        int64_t baseN;
        int64_t mTailTile;
        int64_t nTailTile;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
        int64_t windowLen = WINDOW_LEN;
    };

    __aicore__ inline BlockSchedulerQuantBatchMatmulV3(const ProblemShape& shape, const Params& params)
    {
        const int64_t m = asc::te::get<MNK_M>(shape);
        const int64_t n = asc::te::get<MNK_N>(shape);
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mBlockNums_ = Blaze::Gemm::CeilDiv(m, baseM_);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n, baseN_);
        blockNums_ = mBlockNums_ * nBlockNums_;
        blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        coreNums_ = AscendC::GetBlockNum();
        if (coreNums_ <= 0) {
            return;
        }
        mainWindow_ = Blaze::Gemm::Min(params.windowLen, mBlockNums_);
        if (mainWindow_ != 0) {
            mainRow_ = mBlockNums_ / mainWindow_ - 1;
        }
        tailWindow_ = mBlockNums_ - mainWindow_ * mainRow_;
        endBlockIdx_ = (blockNums_ - 1) % coreNums_;
        roundNums_ = Blaze::Gemm::CeilDiv(blockNums_, coreNums_);
        if (blockIdx_ > endBlockIdx_) {
            roundNums_ -= 1;
        }
        if constexpr (!TRANS_A) {
            mBaseNormCnt_ = mBlockNums_ - params.mBaseTailSplitCnt;
            int64_t mMergeSize = m - mBaseNormCnt_ * baseM_;
            mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
            mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;
        } else {
            mBaseTailMain_ = m - (mBlockNums_ - 1) * baseM_;
        }
        if constexpr (TRANS_B) {
            nBaseNormCnt_ = nBlockNums_ - params.nBaseTailSplitCnt;
            int64_t nMergeSize = n - nBaseNormCnt_ * baseN_;
            nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nMergeSize : params.nTailMain;
            nBaseTailLast_ = nMergeSize - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
        } else {
            nBaseTailMain_ = n - (nBlockNums_ - 1) * baseN_;
        }
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
    {
        mTailTile_ = mTailTile;
        nTailTile_ = nTailTile;
        totalTailTile_ = mTailTile * nTailTile;
        uint64_t tailOriginalBlockNums = AscendC::Std::min(blockNums_, endBlockIdx_ + 1);
        int64_t newEndBlockIdx = endBlockIdx_ + tailOriginalBlockNums * (totalTailTile_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            roundNums_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailTile_ = 1;
            nTailTile_ = 1;
            totalTailTile_ = 1;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline int64_t GetTotalCnt() { return blockNums_; }

    __aicore__ inline int64_t GetEndBlockIdx() { return endBlockIdx_; }

    template <QuantMode AQuantMode_, QuantMode BQuantMode_, bool WeightNz_ = false, uint64_t NAlignValue_ = 1>
    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t singleCoreM = baseM_;
        int64_t singleCoreN = baseN_;
        CalSingleCoreShapeByCoord(singleCoreM, singleCoreN, blockCoord);

        if (totalTailTile_ == 1 || roundIdx_ < roundNums_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = Blaze::Gemm::CeilDiv(singleCoreM, mTailTile_);
        int64_t singleCoreNSplit = Blaze::Gemm::CeilDiv(singleCoreN, nTailTile_);
        if constexpr (IsFp4<AType>() && TRANS_A) {
            singleCoreMSplit = (singleCoreMSplit + 1) & ~1;
        }
        if constexpr (IsFp4<AType>() && !TRANS_B) {
            singleCoreNSplit = (singleCoreNSplit + 1) & ~1;
        }
        if constexpr ((AQuantMode_ == QuantMode::PERGROUP_MODE || AQuantMode_ == QuantMode::PERBLOCK_MODE) && TRANS_A) {
            singleCoreMSplit = PER_BLOCK_SIZE << (singleCoreMSplit > PER_BLOCK_SIZE);
        } else if constexpr (AQuantMode_ == QuantMode::PERBLOCK_MODE) {
            singleCoreMSplit = CeilPowerOfTwo(singleCoreMSplit);
        }
        if constexpr (BQuantMode_ == QuantMode::PERBLOCK_MODE) {
            if constexpr (!TRANS_B) { // (k, n)
                singleCoreNSplit = PER_BLOCK_SIZE << (singleCoreNSplit > PER_BLOCK_SIZE);
            } else {
                singleCoreNSplit = CeilPowerOfTwo(singleCoreNSplit);
            }
        }

        if constexpr (WeightNz_) {
            if constexpr (!TRANS_B) {
                if constexpr (IsFp4<AType>()) {
                    singleCoreNSplit = Align64(singleCoreNSplit);
                } else {
                    singleCoreNSplit = Align32(singleCoreNSplit);
                }
            } else {
                singleCoreNSplit = Align16(singleCoreNSplit);
            }
        }
        if constexpr (NAlignValue_ > 1) {
            singleCoreNSplit = static_cast<int64_t>((static_cast<uint64_t>(singleCoreNSplit) + NAlignValue_ - 1) &
                                                    ~(NAlignValue_ - 1));
        }

        const int64_t tailSplitIdx = blockIdx_ % totalTailTile_;
        int64_t mSplitIdx = tailSplitIdx % mTailTile_;
        int64_t nSplitIdx = 0;
        if constexpr (FullLoadMode_ == A_FULL_LOAD_MODE) {
            nSplitIdx = blockIdx_ / mBlockNums_ % nTailTile_;
        } else {
            nSplitIdx = tailSplitIdx / mTailTile_;
        }
        mSplitAddrOffset_ = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset_ = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset_ >= singleCoreM || nSplitAddrOffset_ >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Blaze::Gemm::Min(singleCoreM - mSplitAddrOffset_, singleCoreMSplit);
        singleCoreN = Blaze::Gemm::Min(singleCoreN - nSplitAddrOffset_, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
    }

    __aicore__ inline void UpdateNextBatchBlockRoundParams()
    {
        startBlockIdx_ = endBlockIdx_ + 1 == coreNums_ ? 0 : (endBlockIdx_ + 1);
        endBlockIdx_ = (blockNums_ + startBlockIdx_ - 1) % coreNums_;

        roundIdx_ = 0;
        roundNums_ = Blaze::Gemm::CeilDiv(blockNums_, coreNums_);
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            roundNums_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            roundNums_ -= 1;
        }
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (roundIdx_ >= roundNums_) {
            return false;
        }

        int64_t blockCoordM = 0;
        int64_t blockCoordN = 0;

        int64_t currentBlockIdx = (roundIdx_ == roundNums_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
        int64_t logicalBlockIdx = currentBlockIdx + roundIdx_ * coreNums_;
        if constexpr (FullLoadMode_ == A_FULL_LOAD_MODE) {
            blockCoordM = blockIdx_ % mBlockNums_;
            int64_t curNTailTile = (roundIdx_ == roundNums_ - 1) ? nTailTile_ : 1;
            blockCoordN = roundIdx_ * coreNums_ / mBlockNums_ % nBlockNums_ + blockIdx_ / mBlockNums_ / curNTailTile;
            roundIdx_++;
            blockCoord = BlockCoord{blockCoordM, blockCoordN, 0, 0};
            return true;
        }
        if (blockIdx_ < startBlockIdx_) {
            logicalBlockIdx += coreNums_ - startBlockIdx_;
        } else if (endBlockIdx_ + 1 >= totalTailTile_ * blockNums_) {
            logicalBlockIdx -= startBlockIdx_ / totalTailTile_;
        } else {
            logicalBlockIdx -= startBlockIdx_;
        }
        int64_t rowIdx = logicalBlockIdx / nBlockNums_ / mainWindow_;
        int64_t nIdx = 0;
        if (rowIdx < mainRow_) {
            blockCoordM = rowIdx * mainWindow_ + logicalBlockIdx % mainWindow_;
            nIdx = (logicalBlockIdx / mainWindow_) % nBlockNums_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIdx = logicalBlockIdx - mainRow_ * mainWindow_ * nBlockNums_;
            blockCoordM = mainRow_ * mainWindow_ + tailIdx % tailWindow_;
            nIdx = (tailIdx / tailWindow_) % nBlockNums_;
        }
        if (rowIdx & 1) {
            nIdx = nBlockNums_ - 1 - nIdx;
        }
        blockCoordN = nIdx;
        roundIdx_++;
        blockCoord = BlockCoord{blockCoordM, blockCoordN, 0, 0};
        return true;
    }

    __aicore__ inline void GetTileCoord(BlockCoord blockCoord, int64_t& mPos, int64_t& nPos)
    {
        auto mBlockIdx = asc::te::get<MNK_M>(blockCoord);
        auto nBlockIdx = asc::te::get<MNK_N>(blockCoord);
        mPos = mBlockIdx * baseM_ + mSplitAddrOffset_;
        nPos = nBlockIdx * baseN_ + nSplitAddrOffset_;
        if constexpr (!TRANS_A) {
            if (mBlockIdx > mBaseNormCnt_) {
                mPos -= (mBlockIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
            }
        }
        if constexpr (TRANS_B) {
            if (nBlockIdx > nBaseNormCnt_) {
                nPos -= (nBlockIdx - nBaseNormCnt_) * (baseN_ - nBaseTailMain_);
            }
        }
    }

private:
    /**
     * @brief Round the input value up to the smallest power of two.
     *
     * Returns the smallest power of two greater than or equal to the input value.
     * This implementation uses a bit-smearing technique and assumes
     * the input value is in the range [1, 256].
     *
     * @param inputValue  Input value to be rounded up.
     */
    __aicore__ inline int64_t CeilPowerOfTwo(int64_t inputValue)
    {
        inputValue--;
        inputValue |= inputValue >> 1; // Propagate the highest set bit to the right by 1 position,ensuring the most
                                       // significant bit and its adjacent lower bit are set.
        inputValue |= inputValue >> 2; // Continue propagating the highest set bit by 2 positions, expanding the
                                       // contiguous range of set bits below the MSB to 3 bits.
        inputValue |= inputValue >> 4; // Further propagate the highest set bit by 4 positions, resulting in all bits
                                       // below the MSB (up to 7 positions) being set.
        inputValue++;
        return inputValue;
    }

    __aicore__ inline void CalSingleCoreShapeByCoord(int64_t& singleCoreM, int64_t& singleCoreN, BlockCoord blockCoord)
    {
        const int64_t mIdx = asc::te::get<MNK_M>(blockCoord);
        const int64_t nIdx = asc::te::get<MNK_N>(blockCoord);
        if constexpr (!TRANS_A) {
            if (mIdx >= mBaseNormCnt_) {
                singleCoreM = mIdx < mBlockNums_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
            }
        } else {
            if (mIdx == mBlockNums_ - 1) {
                singleCoreM = mBaseTailMain_;
            }
        }
        if constexpr (TRANS_B) {
            if (nIdx >= nBaseNormCnt_) {
                singleCoreN = nIdx < nBlockNums_ - 1 ? nBaseTailMain_ : nBaseTailLast_;
            }
        } else {
            if (nIdx == nBlockNums_ - 1) {
                singleCoreN = nBaseTailMain_;
            }
        }
    }

    static constexpr bool TRANS_A = IsTrans<LayoutA_>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB_>::value;

    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t blockNums_{0};
    int64_t mBaseNormCnt_{0};
    int64_t nBaseNormCnt_{0};
    int64_t mBaseTailMain_{0};
    int64_t nBaseTailMain_{0};
    int64_t mBaseTailLast_{0};
    int64_t nBaseTailLast_{0};
    int64_t mainWindow_{0};
    int64_t tailWindow_{0};
    int64_t blockIdx_{0};
    int64_t coreNums_{0};
    int64_t startBlockIdx_{0};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t roundNums_{0};
    int64_t mTailTile_{1};     // init value must be 1
    int64_t nTailTile_{1};     // init value must be 1
    int64_t totalTailTile_{1}; // init value must be 1
    int64_t mainRow_{0};
    int64_t mSplitAddrOffset_{0};
    int64_t nSplitAddrOffset_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
