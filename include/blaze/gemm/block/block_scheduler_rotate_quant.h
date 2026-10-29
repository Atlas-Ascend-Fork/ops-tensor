/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_rotate_quant.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor/layout.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_, bool EnableGrouped_ = false>
class BlockSchedulerRotateQuant {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static_assert(AscendC::Std::is_same_v<ProblemShape, BlockShape>,
                  "RotateQuant scheduler requires an int64 M/N/K/B problem shape.");

    struct Params {
        int64_t mL1 = 0;
        int64_t nL1 = 0;
        int64_t kL1 = 0;
        int64_t baseM = 0;
        int64_t baseN = 0;
        int64_t baseK = 0;
        int64_t nKRatio = 0;
        int64_t tailML1 = 0;
        float dstTypeMax = 0.0f;
        float invDstTypeMax = 0.0f;
        int64_t axis = 0;
        int64_t roundMode = 0;
        int64_t scaleAlg = 0;
        int64_t hasAlpha = 0;
        int64_t trans = 0;
    };

    struct GroupParams {
        GM_ADDR groupListGmAddr = nullptr;
        int64_t groupNum = 0;
        int64_t groupListType = 0; // 0: cumulative row ends; 1: row counts
    };

    struct BlockInfo {
        BlockL1L0Shape blockShape;
        BlockCoord blockCoord;
        int64_t rotationOffset; // Element offset of this block's K-by-K rotation in GM.
    };

    __aicore__ inline BlockSchedulerRotateQuant(const ProblemShape& shape, int64_t blockNum, const Params& params)
        : BlockSchedulerRotateQuant(shape, blockNum, params, GroupParams{})
    {}

    __aicore__ inline BlockSchedulerRotateQuant(const ProblemShape& shape, int64_t blockNum, const Params& params,
                                                const GroupParams& groups)
        : params_(params), groups_(groups)
    {
        m_ = asc::te::get<Gemm::MNK_M>(shape);
        int64_t n = asc::te::get<Gemm::MNK_N>(shape);
        k_ = asc::te::get<Gemm::MNK_K>(shape);
        b_ = asc::te::get<Gemm::MNK_B>(shape);
        if (m_ < 0 || n <= 0 || k_ <= 0 || n % k_ != 0 || params_.mL1 <= 0) {
            return;
        }
        ratio_ = EnableGrouped_ ? n / k_ : params_.nKRatio;
        if (ratio_ <= 0 || (b_ != 1 && b_ != n / k_) || m_ > MAX_OFFSET / ratio_ || k_ > MAX_OFFSET / k_) {
            return;
        }
        if constexpr (EnableGrouped_) {
            InitGroups(blockNum);
        } else {
            totalTasks_ = GetTaskCount(m_);
            valid_ = b_ <= MAX_OFFSET / (k_ * k_);
            if (!valid_) {
                totalTasks_ = 0;
            }
        }
    }

    __aicore__ inline bool IsValid() const { return valid_; }
    __aicore__ inline int64_t GetBlockNums() const { return totalTasks_; }
    __aicore__ inline int64_t GetCoreNums(int64_t blockNum) const
    {
        return blockNum < totalTasks_ ? blockNum : totalTasks_;
    }

    // Coordinates are global: flattened K-wide rows for B=1, original M rows for B>1.
    // Repeated and backwards queries do not consume a block or change its result.
    __aicore__ inline BlockInfo GetBlockInfo(int64_t blockIdx)
    {
        if (!valid_ || blockIdx < 0 || blockIdx >= totalTasks_) {
            return {{0, k_, k_, params_.mL1, params_.baseM, params_.baseN}, {0, 0, 0, 0}, 0};
        }
        if constexpr (EnableGrouped_) {
            if (blockIdx < taskStart_) {
                groupIdx_ = 0;
                rowStart_ = 0;
                taskStart_ = 0;
            }
            while (groupIdx_ < groups_.groupNum) {
                int64_t value = groupList_[groupIdx_];
                int64_t rowEnd = groups_.groupListType == 0 ? value : rowStart_ + value;
                int64_t groupRows = rowEnd - rowStart_;
                int64_t tasks = GetTaskCount(groupRows);
                if (blockIdx < taskStart_ + tasks) {
                    return MakeBlockInfo(blockIdx - taskStart_, rowStart_, groupRows, groupIdx_);
                }
                rowStart_ = rowEnd;
                taskStart_ += tasks;
                ++groupIdx_;
            }
        } else {
            return MakeBlockInfo(blockIdx, 0, m_, 0);
        }
        return {{0, k_, k_, params_.mL1, params_.baseM, params_.baseN}, {0, 0, 0, 0}, 0};
    }

    __aicore__ inline BlockL1L0Shape GetBlockShape(int64_t blockIdx) { return GetBlockInfo(blockIdx).blockShape; }
    __aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx) { return GetBlockInfo(blockIdx).blockCoord; }

private:
    static constexpr int64_t MAX_OFFSET = 0x7fffffffffffffffLL;

    __aicore__ inline void InitGroups(int64_t blockNum)
    {
        if (blockNum <= 0 || groups_.groupListGmAddr == nullptr || groups_.groupNum <= 0 || groups_.groupNum > 1024 ||
            (groups_.groupListType != 0 && groups_.groupListType != 1) || b_ > MAX_OFFSET / groups_.groupNum ||
            groups_.groupNum * b_ > MAX_OFFSET / (k_ * k_)) {
            return;
        }
        groupList_ = reinterpret_cast<__gm__ int64_t*>(groups_.groupListGmAddr);
        int64_t end = 0;
        int64_t tasks = 0;
        for (int64_t i = 0; i < groups_.groupNum; ++i) {
            int64_t value = groupList_[i];
            if (value < 0 || (groups_.groupListType == 0 && (value < end || value > m_)) ||
                (groups_.groupListType == 1 && value > m_ - end)) {
                return;
            }
            int64_t next = groups_.groupListType == 0 ? value : end + value;
            tasks += GetTaskCount(next - end);
            end = next;
        }
        totalTasks_ = tasks;
        valid_ = true;
    }

    __aicore__ inline BlockInfo MakeBlockInfo(int64_t localTask, int64_t rowStart, int64_t groupRows,
                                              int64_t groupIdx) const
    {
        int64_t localRow = (b_ > 1 ? localTask / b_ : localTask) * params_.mL1;
        int64_t rows = b_ > 1 ? groupRows : groupRows * ratio_;
        int64_t blkM = rows - localRow;
        blkM = blkM < params_.mL1 ? blkM : params_.mL1;
        int64_t blkN = params_.nL1;
        int64_t blkK = params_.kL1;
        if constexpr (EnableGrouped_) {
            blkN = k_;
            blkK = k_;
        } else {
            if (b_ == 1 && localTask == totalTasks_ - 1) {
                blkM = params_.tailML1;
            }
        }
        int64_t coordM = (b_ > 1 ? rowStart : rowStart * ratio_) + localRow;
        int64_t coordB = b_ > 1 ? localTask % b_ : 0;
        int64_t rotationOffset = (groupIdx * b_ + coordB) * k_ * k_;
        return {{blkM, blkN, blkK, params_.mL1, params_.baseM, params_.baseN}, {coordM, 0, 0, coordB}, rotationOffset};
    }

    __aicore__ inline int64_t GetTaskCount(int64_t groupRows) const
    {
        int64_t rows = b_ > 1 ? groupRows : groupRows * ratio_;
        int64_t blockCount = rows / params_.mL1 + (rows % params_.mL1 != 0);
        return b_ > 1 ? blockCount * b_ : blockCount;
    }

    Params params_;
    GroupParams groups_;
    __gm__ int64_t* groupList_{nullptr};
    int64_t m_{0};
    int64_t k_{0};
    int64_t b_{1};
    int64_t ratio_{0};
    int64_t totalTasks_{0};
    int64_t groupIdx_{0};
    int64_t rowStart_{0};
    int64_t taskStart_{0};
    bool valid_{false};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
