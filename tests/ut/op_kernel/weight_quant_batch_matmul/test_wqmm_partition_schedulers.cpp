/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <numeric>
#include "gtest/gtest.h"
#include "wqmm_test_utils.h"
#include "blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h"
#include "blaze/gemm/block/block_scheduler_wqmm_block_split.h"

namespace {
using namespace WeightQuantBatchMatmulUT;
using Shape = asc::te::shape<int64_t, int64_t, int64_t>;
using Swat = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<Shape>;
using Record = std::array<int64_t, 8>; // complete coordinate and shape
constexpr uint64_t CORES = 8, CAPACITY = 512;
struct SwatCase {
    int64_t m, n, k;
    Swat::Params params;
};
template <class Scheduler>
void RecordTiles(const Scheduler& scheduler, Record* records, uint64_t count)
{
    for (uint64_t i = 0; i < count && i < CAPACITY; ++i) {
        auto coord = scheduler.GetBlockCoord(i);
        auto shape = scheduler.GetBlockShape(coord);
        records[i] = {asc::te::get<0>(coord), asc::te::get<1>(coord), asc::te::get<2>(coord), asc::te::get<3>(coord),
                      asc::te::get<0>(shape), asc::te::get<1>(shape), asc::te::get<2>(shape), asc::te::get<3>(shape)};
    }
}
__global__ __aicore__ void SwatKernel(GM_ADDR cases, GM_ADDR output, GM_ADDR totals, uint64_t caseCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint64_t lane = 0;
    if ASCEND_IS_AIV {
        lane = 1 + AscendC::GetSubBlockIdx();
    }
    // SWAT is a global schedule: every core must observe the same stream.
    uint64_t core = Blaze::Gemm::GetCurrentBlockIdx();
    for (uint64_t trial = 0; trial < caseCount; ++trial) {
        const auto& c = reinterpret_cast<SwatCase*>(cases)[trial];
        Swat scheduler(asc::te::make_shape(c.m, c.n, c.k), c.params);
        uint64_t slot = (trial * 3 + lane) * CORES + core;
        reinterpret_cast<uint64_t*>(totals)[slot] = scheduler.GetTileCount();
        RecordTiles(scheduler, reinterpret_cast<Record*>(output) + slot * CAPACITY, scheduler.GetTileCount());
    }
}
int64_t RoundUp(int64_t n, int64_t unit) { return (n + unit - 1) / unit * unit; }
std::vector<std::pair<int64_t, int64_t>> Axis(int64_t size, uint64_t base, uint64_t merged, uint64_t main)
{
    int64_t count = (size + base - 1) / base;
    int64_t tailCount = std::min<int64_t>(std::max<uint64_t>(1, merged), count);
    int64_t normal = count - tailCount;
    int64_t tailMain = tailCount == 1 ? size - normal * base : main;
    std::vector<std::pair<int64_t, int64_t>> out;
    int64_t origin = 0;
    for (int64_t i = 0; i < count; ++i) {
        int64_t extent = i < normal ? base : (i == count - 1 ? size - origin : tailMain);
        out.emplace_back(origin, extent);
        origin += extent;
    }
    return out;
}
std::vector<Record> SwatOracle(const SwatCase& c)
{
    if (!c.params.baseM || !c.params.baseN)
        return {};
    auto ms = Axis(c.m, c.params.baseM, c.params.mBaseTailSplitCnt, c.params.mTailMain);
    auto ns = Axis(c.n, c.params.baseN, c.params.nBaseTailSplitCnt, c.params.nTailMain);
    std::vector<Record> logical;
    // Four-row windows, with the last short window merged into its predecessor.
    for (size_t start = 0, window = 0; start < ms.size(); ++window) {
        size_t end = ms.size() - start < 8 ? ms.size() : start + 4;
        for (size_t col = 0; col < ns.size(); ++col) {
            size_t ni = window % 2 ? ns.size() - 1 - col : col;
            for (size_t mi = start; mi < end; ++mi)
                logical.push_back(
                    {ms[mi].first, ns[ni].first, int64_t(logical.size()), -1, ms[mi].second, ns[ni].second, c.k, 1});
        }
        start = end;
    }
    size_t tail = logical.size() % CORES;
    auto split = [&](uint64_t mt, uint64_t nt) {
        std::vector<Record> out;
        for (size_t i = logical.size() - tail; i < logical.size(); ++i) {
            const auto& r = logical[i];
            int64_t stepM = RoundUp((r[4] + mt - 1) / mt, 16);
            int64_t stepN = RoundUp((r[5] + nt - 1) / nt, 16);
            for (uint64_t ni = 0; ni < nt; ++ni)
                for (uint64_t mi = 0; mi < mt; ++mi) {
                    int64_t dm = mi * stepM, dn = ni * stepN;
                    if (dm < r[4] && dn < r[5])
                        out.push_back({r[0] + dm, r[1] + dn, r[2], int64_t(ni * mt + mi), std::min(stepM, r[4] - dm),
                                       std::min(stepN, r[5] - dn), c.k, 1});
                }
        }
        return out;
    };
    auto compact = split(std::max<uint64_t>(1, c.params.mTailTile), std::max<uint64_t>(1, c.params.nTailTile));
    if (compact.size() > CORES)
        compact = split(1, 1);
    logical.resize(logical.size() - tail);
    logical.insert(logical.end(), compact.begin(), compact.end());
    return logical;
}
void CheckPartition(const std::vector<Record>& records, int64_t m, int64_t n)
{
    std::vector<uint8_t> visits(m * n, 0);
    for (const auto& r : records) {
        ASSERT_GE(r[0], 0);
        ASSERT_GE(r[1], 0);
        ASSERT_GT(r[4], 0);
        ASSERT_GT(r[5], 0);
        ASSERT_LE(r[0] + r[4], m);
        ASSERT_LE(r[1] + r[5], n);
        for (int64_t i = r[0]; i < r[0] + r[4]; ++i)
            for (int64_t j = r[1]; j < r[1] + r[5]; ++j)
                ASSERT_EQ(visits[i * n + j]++, 0) << i << ',' << j;
    }
    ASSERT_TRUE(std::all_of(visits.begin(), visits.end(), [](uint8_t v) { return v == 1; }));
}
TEST(WqmmPartitionSchedulerTest, SwatOrderMergingAndCompactTails)
{
    std::vector<SwatCase> cases{{1, 1, 32, {16, 16}},
                                {64, 64, 32, {32, 16}},                       // complete core round
                                {931, 67, 65, {32, 32}},                      // odd/even windows, short final window
                                {145, 81, 128, {32, 32, 2, 2}},               // split both axes
                                {65, 33, 128, {32, 32, 4, 4}},                // compact away empty 16-aligned splits
                                {97, 81, 32, {32, 32, 4, 4}},                 // split exceeds cores: fallback
                                {145, 81, 128, {32, 32, 2, 1, 2, 2, 24, 24}}, // merged tails on both axes
                                {81, 145, 128, {32, 32, 1, 2, 2, 2, 24, 24}},
                                {17, 33, 32, {32, 32, 0, 0, 0, 0}}, // clamp split settings
                                {32, 32, 32, {0, 32}},
                                {32, 32, 32, {32, 0}}};
    GmBuffer configs(cases.size() * sizeof(SwatCase));
    configs.Set(cases);
    const auto slots = cases.size() * 3 * CORES;
    GmBuffer output(slots * CAPACITY * sizeof(Record)), totals(slots * sizeof(uint64_t));
    std::memset(output.Get(), 0xA5, slots * CAPACITY * sizeof(Record));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(SwatKernel, CORES, configs.Get(), output.Get(), totals.Get(), cases.size()));
    for (size_t trial = 0; trial < cases.size(); ++trial) {
        SCOPED_TRACE(trial);
        auto expected = SwatOracle(cases[trial]);
        ASSERT_LT(expected.size(), CAPACITY);
        if (!expected.empty())
            ASSERT_NO_FATAL_FAILURE(CheckPartition(expected, cases[trial].m, cases[trial].n));
        for (size_t slot = trial * 3 * CORES; slot < (trial + 1) * 3 * CORES; ++slot) {
            ASSERT_EQ(reinterpret_cast<uint64_t*>(totals.Get())[slot], expected.size());
            auto* actual = reinterpret_cast<Record*>(output.Get()) + slot * CAPACITY;
            for (size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(actual[i], expected[i]);
            for (auto value : actual[expected.size()])
                ASSERT_EQ(uint64_t(value), 0xA5A5A5A5A5A5A5A5ULL);
        }
    }
}
struct SplitCase {
    int64_t m, n, k;
    uint32_t cm, cn, bm, bn, order;
};
template <class Layout, class Element>
__global__ __aicore__ void SplitKernel(GM_ADDR configs, GM_ADDR output, GM_ADDR totals, uint64_t caseCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmBlockSplit<Shape, Layout, Element>;
    uint64_t lane = 0;
    if ASCEND_IS_AIV {
        lane = 1 + AscendC::GetSubBlockIdx();
    }
    uint64_t core = Blaze::Gemm::GetCurrentBlockIdx();
    for (uint64_t trial = 0; trial < caseCount; ++trial) {
        const auto& c = reinterpret_cast<SplitCase*>(configs)[trial];
        Scheduler scheduler(asc::te::make_shape(c.m, c.n, c.k), {c.cm, c.cn, c.bm, c.bn, c.order});
        uint64_t slot = (trial * 3 + lane) * CORES + core;
        reinterpret_cast<uint64_t*>(totals)[slot] = scheduler.GetBlockNums();
        RecordTiles(scheduler, reinterpret_cast<Record*>(output) + slot * CAPACITY, scheduler.GetBlockNums());
    }
}
template <class Layout, class Element>
void CheckBlockSplit(int64_t nAlignment)
{
    std::vector<SplitCase> cases{
        {1, 1, 1, 1, 1, 16, 16, 0},    {67, 99, 129, 2, 2, 16, 16, 0}, {67, 99, 129, 2, 2, 16, 16, 1},
        {17, 17, 32, 4, 2, 16, 16, 0}, {65, 129, 64, 1, 2, 32, 32, 1}, {32, 32, 32, 1, 1, 0, 16, 0},
        {32, 32, 32, 1, 1, 16, 0, 1},  {32, 32, 32, 0, 1, 16, 16, 0},  {32, 32, 32, 1, 0, 16, 16, 1}};
    auto slots = cases.size() * 3 * CORES;
    GmBuffer configs(cases.size() * sizeof(SplitCase)), output(slots * CAPACITY * sizeof(Record)),
        totals(slots * sizeof(uint64_t));
    configs.Set(cases);
    std::memset(output.Get(), 0xA5, slots * CAPACITY * sizeof(Record));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = SplitKernel<Layout, Element>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, CORES, configs.Get(), output.Get(), totals.Get(), cases.size()));
    for (size_t trial = 0; trial < cases.size(); ++trial) {
        SCOPED_TRACE(trial);
        const auto& c = cases[trial];
        std::vector<Record> all;
        for (uint64_t core = 0; core < CORES; ++core) {
            std::vector<Record> expected;
            if (c.bm && c.bn && core < uint64_t(c.cm) * c.cn) {
                int64_t spanM = RoundUp((c.m + c.cm - 1) / c.cm, 16);
                int64_t spanN = RoundUp((c.n + c.cn - 1) / c.cn, nAlignment);
                int64_t startM = (core % c.cm) * spanM, startN = (core / c.cm) * spanN;
                for (int64_t row = startM; row < std::min(c.m, startM + spanM); row += c.bm)
                    for (int64_t col = startN; col < std::min(c.n, startN + spanN); col += c.bn)
                        expected.push_back({row, col, 0, 0,
                                            std::min<int64_t>(c.bm, std::min(c.m, startM + spanM) - row),
                                            std::min<int64_t>(c.bn, std::min(c.n, startN + spanN) - col), c.k, 1});
                if (c.order == 1)
                    std::stable_sort(expected.begin(), expected.end(), [](const Record& a, const Record& b) {
                        return std::make_pair(a[1], a[0]) < std::make_pair(b[1], b[0]);
                    });
            }
            all.insert(all.end(), expected.begin(), expected.end());
            ASSERT_LT(expected.size(), CAPACITY);
            for (uint64_t lane = 0; lane < 3; ++lane) {
                auto slot = (trial * 3 + lane) * CORES + core;
                ASSERT_EQ(reinterpret_cast<uint64_t*>(totals.Get())[slot], expected.size());
                auto* actual = reinterpret_cast<Record*>(output.Get()) + slot * CAPACITY;
                for (size_t i = 0; i < expected.size(); ++i)
                    ASSERT_EQ(actual[i], expected[i]);
                for (auto value : actual[expected.size()])
                    ASSERT_EQ(uint64_t(value), 0xA5A5A5A5A5A5A5A5ULL);
            }
        }
        if (c.bm && c.bn && c.cm && c.cn)
            ASSERT_NO_FATAL_FAILURE(CheckPartition(all, c.m, c.n));
    }
}
TEST(WqmmPartitionSchedulerTest, BlockSplitNd) { CheckBlockSplit<asc::te::dn_ext_layout_ptn, fp8_e4m3fn_t>(16); }
TEST(WqmmPartitionSchedulerTest, BlockSplitNz) { CheckBlockSplit<asc::te::nz_layout_ptn, fp8_e4m3fn_t>(32); }
TEST(WqmmPartitionSchedulerTest, BlockSplitTransposedNd) { CheckBlockSplit<asc::te::nd_ext_layout_ptn, half>(16); }
TEST(WqmmPartitionSchedulerTest, BlockSplitTransposedNz) { CheckBlockSplit<asc::te::zn_layout_ptn, half>(16); }
} // namespace
