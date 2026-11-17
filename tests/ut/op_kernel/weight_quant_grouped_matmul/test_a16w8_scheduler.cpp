/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"

namespace {
using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
constexpr uint32_t CORES = 8;
constexpr uint32_t GROUPS = 6;
constexpr uint32_t LANES = 2;
constexpr uint32_t FIELDS = 4; // m, n, rows, columns
constexpr int64_t ROWS[GROUPS] = {0, 1, 129, 0, 255, 1025};
constexpr int64_t N = 273;
constexpr size_t STREAMS = CORES * GROUPS * LANES;
constexpr uint32_t BASE_M_NO_BIAS = 256;
constexpr uint32_t BASE_M_WITH_BIAS = 240;
constexpr uint32_t BASE_N = 256;
constexpr uint32_t MIN_BASE_M = BASE_M_NO_BIAS < BASE_M_WITH_BIAS ? BASE_M_NO_BIAS : BASE_M_WITH_BIAS;

constexpr int64_t MaxGroupRows()
{
    int64_t maxRows = 0;
    for (const auto rows : ROWS) {
        maxRows = rows > maxRows ? rows : maxRows;
    }
    return maxRows;
}

constexpr int64_t MAX_BLOCKS = ((MaxGroupRows() + MIN_BASE_M - 1) / MIN_BASE_M) * ((N + BASE_N - 1) / BASE_N);
// Resetting to (1, 1) adds no rounds; the single automatic tail split adds at most one per core.
constexpr uint32_t MAX_TAIL_EXTRA_ROUNDS = 1;
constexpr uint32_t CAPACITY = static_cast<uint32_t>((MAX_BLOCKS + CORES - 1) / CORES) + MAX_TAIL_EXTRA_ROUNDS;

// Rows follow expert order; columns are physical core IDs. Zero-sized split blocks are excluded.
constexpr uint64_t EXPECTED_COUNTS_NO_BIAS[GROUPS][CORES] = {
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 1, 1, 1, 1},
};
constexpr uint64_t EXPECTED_COUNTS_WITH_BIAS[GROUPS][CORES] = {
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2, 2},
};

// GM_ADDR adaptation boundary, shared by the CPU-debug kernel and its host reader.
template <class T_, class Layout_>
__aicore__ inline auto MakeGmView(GM_ADDR address, const Layout_& layout)
{
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ T_*>(address)),
                                layout);
}

__global__ __aicore__ void Schedule(GM_ADDR records, GM_ADDR lengths, uint32_t baseM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    auto out = MakeGmView<int64_t>(records,
                                   asc::te::make_layout(asc::te::make_shape(LANES, CORES, GROUPS, CAPACITY, FIELDS)));
    auto counts = MakeGmView<uint64_t>(lengths, asc::te::make_layout(asc::te::make_shape(LANES, CORES, GROUPS)));
    uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    uint32_t lane = 0;
    if ASCEND_IS_AIV {
        lane = 1; // The only AIV paired with this AIC.
    }
    Scheduler scheduler(Scheduler::Params{static_cast<int32_t>(baseM), static_cast<int32_t>(BASE_N)});
    scheduler.SetTailAlign(1, 16);
    for (uint32_t group = 0; group < GROUPS; ++group) {
        uint64_t count = 0;
        if (ROWS[group] != 0) {
            scheduler.UpdateNextProblem({ROWS[group], N, 1041, 1});
            scheduler.UpdateTailTile(1, 1);
            scheduler.UpdateTailTile();
            Scheduler::BlockInfo block;
            while (scheduler.GetNextBlock(block)) {
                auto m = asc::te::get<0>(block.blockShape);
                auto n = asc::te::get<1>(block.blockShape);
                if (m == 0 || n == 0) {
                    continue;
                }
                if (count < CAPACITY) {
                    out[asc::te::make_coord(lane, core, group, count, 0U)] = asc::te::get<0>(block.blockCoord);
                    out[asc::te::make_coord(lane, core, group, count, 1U)] = asc::te::get<1>(block.blockCoord);
                    out[asc::te::make_coord(lane, core, group, count, 2U)] = m;
                    out[asc::te::make_coord(lane, core, group, count, 3U)] = n;
                }
                ++count;
            }
        }
        counts[asc::te::make_coord(lane, core, group)] = count;
    }
}

class GmStorage {
public:
    explicit GmStorage(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes)))
    {
        if (ptr_) {
            std::memset(ptr_, 0, bytes);
        }
    }
    ~GmStorage()
    {
        if (ptr_)
            AscendC::GmFree(ptr_);
    }
    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_;
};

void CheckSchedule(uint32_t baseM, const uint64_t (&expectedCounts)[GROUPS][CORES])
{
    GmStorage records(STREAMS * CAPACITY * FIELDS * sizeof(int64_t));
    GmStorage lengths(STREAMS * sizeof(uint64_t));
    ASSERT_NE(records.Get(), nullptr);
    ASSERT_NE(lengths.Get(), nullptr);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(Schedule, CORES, records.Get(), lengths.Get(), baseM));
    auto out = MakeGmView<int64_t>(records.Get(),
                                   asc::te::make_layout(asc::te::make_shape(LANES, CORES, GROUPS, CAPACITY, FIELDS)));
    auto counts = MakeGmView<uint64_t>(lengths.Get(), asc::te::make_layout(asc::te::make_shape(LANES, CORES, GROUPS)));
    for (uint32_t group = 0; group < GROUPS; ++group) {
        std::vector<uint8_t> coverage(ROWS[group] * N, 0);
        for (uint32_t core = 0; core < CORES; ++core) {
            SCOPED_TRACE(::testing::Message() << "baseM=" << baseM << ", group=" << group << ", core=" << core);
            const uint64_t count = counts[asc::te::make_coord(0U, core, group)];
            ASSERT_LE(count, CAPACITY);
            EXPECT_EQ(count, expectedCounts[group][core]);
            for (uint32_t lane = 1; lane < LANES; ++lane) {
                ASSERT_EQ(count, counts[asc::te::make_coord(lane, core, group)]);
                for (uint64_t i = 0; i < count; ++i) {
                    for (uint32_t field = 0; field < FIELDS; ++field) {
                        EXPECT_EQ(out[asc::te::make_coord(0U, core, group, i, field)],
                                  out[asc::te::make_coord(lane, core, group, i, field)]);
                    }
                }
            }
            for (uint64_t i = 0; i < count; ++i) {
                const int64_t mStart = out[asc::te::make_coord(0U, core, group, i, 0U)];
                const int64_t nStart = out[asc::te::make_coord(0U, core, group, i, 1U)];
                const int64_t mSize = out[asc::te::make_coord(0U, core, group, i, 2U)];
                const int64_t nSize = out[asc::te::make_coord(0U, core, group, i, 3U)];
                ASSERT_GE(mStart, 0);
                ASSERT_GE(nStart, 0);
                ASSERT_GT(mSize, 0);
                ASSERT_GT(nSize, 0);
                ASSERT_LE(mStart + mSize, ROWS[group]);
                ASSERT_LE(nStart + nSize, N);
                EXPECT_EQ(nStart % 16, 0);
                for (int64_t m = mStart; m < mStart + mSize; ++m) {
                    for (int64_t n = nStart; n < nStart + nSize; ++n) {
                        ++coverage[m * N + n];
                    }
                }
            }
        }
        for (auto writers : coverage) {
            EXPECT_EQ(writers, 1);
        }
    }
}

TEST(GroupedA16W8Schedule, NoBiasEmptyExpertsRollingStartAndTailSplitsHaveUniqueWriters)
{
    CheckSchedule(BASE_M_NO_BIAS, EXPECTED_COUNTS_NO_BIAS);
}

TEST(GroupedA16W8Schedule, BiasEmptyExpertsRollingStartAndTailSplitsHaveUniqueWriters)
{
    CheckSchedule(BASE_M_WITH_BIAS, EXPECTED_COUNTS_WITH_BIAS);
}
} // namespace
