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
 * \file test_rotate_quant.cpp
 * \brief RotateQuant public assembly, kernel smoke and scheduler boundary tests.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "rotate_quant.h"
#include "tikicpulib.h"

namespace {

using ProblemShape = RotateQuantUT::ProblemShape;
using GroupedScheduler = Blaze::Gemm::Block::BlockSchedulerRotateQuant<ProblemShape, true>;
using OrdinaryScheduler = Blaze::Gemm::Block::BlockSchedulerRotateQuant<ProblemShape>;
using TilingData = RotateQuantUT::RotateQuantTilingData;
using RotateQuantUT::GROUP_LIST_COUNT;
using RotateQuantUT::GROUP_LIST_CUMSUM;
using RotateQuantUT::MAX_BASE_K;
constexpr uint32_t SMOKE_BLOCK_NUM = 2;
constexpr int64_t SCHEDULER_BLOCK_NUM = 4;
constexpr int64_t CORE_ROLE_NUM = 3; // One AIC and two AIV subcores.
constexpr int64_t RECORD_CAPACITY = 4096;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr int64_t MX_SCALE_PAIR_SIZE = 64;
constexpr int64_t MX_SCALES_PER_PAIR = 2;
constexpr uint8_t SENTINEL_BYTE = 0xA5;
constexpr uint64_t SENTINEL_WORD = 0xA5A5A5A5A5A5A5A5ULL;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_{};
};

template <typename XType_, typename YType_, bool EnableGrouped_>
void CheckPublicAssembly()
{
    using Types = RotateQuantUT::RotateQuantBlazeTypes<XType_, YType_, int8_t, asc::te::nd_ext_layout_ptn,
                                                       EnableGrouped_>;
    using Kernel = typename Types::MatmulKernel;
    using BlockMmad = typename Types::BlockMmad;
    using BlockEpilogue = typename Types::BlockEpilogue;
    using BlockScheduler = typename Types::BlockScheduler;
    using SmokeTypes = RotateQuantUT::KernelSmokeTypes<XType_, YType_, int8_t, asc::te::nd_ext_layout_ptn,
                                                       EnableGrouped_>;
    static_assert(std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, Blaze::Gemm::KernelRotateQuant>);
    static_assert(std::is_same_v<typename BlockMmad::AElementType, XType_>);
    static_assert(std::is_same_v<typename BlockMmad::BElementType, XType_>);
    static_assert(std::is_same_v<typename BlockMmad::CElementType, bfloat16_t>);
    static_assert(std::is_same_v<typename BlockMmad::L0CType, float>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeIn, bfloat16_t>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeOut, YType_>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeScale, int8_t>);
    static_assert(std::is_same_v<typename Kernel::BlockMmad, BlockMmad>);
    static_assert(std::is_same_v<typename Kernel::BlockEpilogue, BlockEpilogue>);
    static_assert(std::is_same_v<typename Kernel::BlockScheduler, BlockScheduler>);
    static_assert(std::is_same_v<typename SmokeTypes::BlockEpilogue::Params, typename BlockEpilogue::Params>);

    typename Kernel::Params params{};
    params.problemShape = {17, 96, 32, 1};
    params.mmadParams.alphaGMAddr = nullptr;
    params.schParams.mL1 = 16;
    params.groupParams = {nullptr, EnableGrouped_ ? 3 : 0, GROUP_LIST_COUNT};
    params.epilogueParams = {nullptr, nullptr, params.problemShape, {}, false, -1, RotateQuantUT::ROUND_MODE_RINT,
                             2,       6.0f,    1.0f / 6.0f};
    EXPECT_EQ(params.mmadParams.alphaGMAddr, nullptr);
    EXPECT_EQ(params.schParams.mL1, 16);
    EXPECT_EQ(params.groupParams.groupNum, EnableGrouped_ ? 3 : 0);
    EXPECT_FLOAT_EQ(params.epilogueParams.dstTypeMax, 6.0f);
}

std::vector<int64_t> MakeGroupList(const std::vector<int64_t>& counts, int64_t groupListType)
{
    std::vector<int64_t> groupList = counts;
    if (groupListType == GROUP_LIST_CUMSUM) {
        for (size_t groupIdx = 1; groupIdx < groupList.size(); ++groupIdx) {
            groupList[groupIdx] += groupList[groupIdx - 1];
        }
    }
    return groupList;
}

template <typename XType_, typename YType_, bool EnableGrouped_>
void RunKernelSmoke(TilingData tiling, const std::vector<int64_t>& counts = {})
{
    tiling.baseK = std::min(tiling.k, MAX_BASE_K);
    tiling.groupNum = EnableGrouped_ ? static_cast<int64_t>(counts.size()) : 0;
    if constexpr (EnableGrouped_) {
        ASSERT_TRUE(!counts.empty());
    }
    if constexpr (std::is_same_v<YType_, fp4x2_e2m1_t>) {
        tiling.scaleAlg = 2;
        tiling.dstTypeMax = 6.0f;
        tiling.invDstTypeMax = 1.0f / 6.0f;
    }
    const auto groupList = MakeGroupList(counts, tiling.groupListType);
    const size_t elements = static_cast<size_t>(tiling.m) * tiling.n;
    const size_t matrixNum = static_cast<size_t>(std::max<int64_t>(tiling.groupNum, 1)) * tiling.b;
    const size_t xBytes = elements * sizeof(XType_);
    const size_t rotationBytes = matrixNum * tiling.k * tiling.k * sizeof(XType_);
    const size_t yBytes = std::is_same_v<YType_, fp4x2_e2m1_t> ? elements / 2 : elements;
    const size_t scaleBytes = tiling.m * ((tiling.n + MX_SCALE_PAIR_SIZE - 1) / MX_SCALE_PAIR_SIZE) *
                              MX_SCALES_PER_PAIR;
    const size_t groupBytes = std::max<size_t>(groupList.size(), 1) * sizeof(int64_t);
    GmBuffer x(xBytes);
    GmBuffer rotation(rotationBytes);
    GmBuffer alpha(sizeof(bfloat16_t));
    GmBuffer groups(groupBytes);
    GmBuffer y(yBytes);
    GmBuffer scale(scaleBytes);
    GmBuffer tilingGm(sizeof(tiling));
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(rotation.Get(), nullptr);
    ASSERT_NE(alpha.Get(), nullptr);
    ASSERT_NE(groups.Get(), nullptr);
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(scale.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);
    std::fill_n(reinterpret_cast<uint8_t*>(x.Get()), xBytes, 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(rotation.Get()), rotationBytes, 0x38U);
    *reinterpret_cast<bfloat16_t*>(alpha.Get()) = bfloat16_t(0.75f);
    std::memset(groups.Get(), 0, groupBytes);
    if (!groupList.empty()) {
        std::memcpy(groups.Get(), groupList.data(), groupList.size() * sizeof(int64_t));
    }
    std::memset(y.Get(), SENTINEL_BYTE, yBytes);
    std::memset(scale.Get(), SENTINEL_BYTE, scaleBytes);
    std::memcpy(tilingGm.Get(), &tiling, sizeof(tiling));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = RotateQuantKernelEntry<XType_, YType_, int8_t, asc::te::nd_ext_layout_ptn, EnableGrouped_>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, SMOKE_BLOCK_NUM, x.Get(), rotation.Get(),
                              tiling.hasAlpha ? alpha.Get() : nullptr, EnableGrouped_ ? groups.Get() : nullptr, y.Get(),
                              scale.Get(), tilingGm.Get()))
        << "RotateQuant Blaze kernel smoke failed";
}

struct SchedulerCaseConfig {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t blockSize{16};
    int64_t groupNum{0};
    int64_t groupListType{GROUP_LIST_COUNT};
    int64_t b{1};
};

struct SchedulerRecord {
    int64_t rotationOffset{0};
    std::array<int64_t, 4> coord{};
    std::array<int64_t, 6> shape{};
    int64_t consistent{1};
};

struct SchedulerTotals {
    int64_t valid{0};
    int64_t blockNum{0};
    int64_t coreNum{0};
    int64_t visited{0};
};

__aicore__ inline SchedulerRecord MakeSchedulerRecord(const GroupedScheduler::BlockInfo& info)
{
    return {info.rotationOffset,
            {asc::te::get<0>(info.blockCoord), asc::te::get<1>(info.blockCoord), asc::te::get<2>(info.blockCoord),
             asc::te::get<3>(info.blockCoord)},
            {asc::te::get<0>(info.blockShape), asc::te::get<1>(info.blockShape), asc::te::get<2>(info.blockShape),
             asc::te::get<3>(info.blockShape), asc::te::get<4>(info.blockShape), asc::te::get<5>(info.blockShape)},
            1};
}

__global__ __aicore__ void ScheduleKernel(GM_ADDR configGm, GM_ADDR groupsGm, GM_ADDR recordsGm, GM_ADDR totalsGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    int64_t coreIdx = AscendC::GetBlockIdx();
    int64_t roleIdx = 0;
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetSubBlockNum();
        roleIdx = 1 + AscendC::GetSubBlockIdx();
    }
    const auto& config = *reinterpret_cast<const SchedulerCaseConfig*>(configGm);
    GroupedScheduler::Params params;
    params.mL1 = params.baseM = config.blockSize;
    params.nL1 = params.kL1 = params.baseN = config.k;
    params.baseK = std::min(config.k, MAX_BASE_K);
    params.nKRatio = config.n / config.k;
    GroupedScheduler scheduler({config.m, config.n, config.k, config.b}, SCHEDULER_BLOCK_NUM, params,
                               {groupsGm, config.groupNum, config.groupListType});
    const int64_t slot = roleIdx * SCHEDULER_BLOCK_NUM + coreIdx;
    auto* records = reinterpret_cast<SchedulerRecord*>(recordsGm) + slot * RECORD_CAPACITY;
    auto* totals = reinterpret_cast<SchedulerTotals*>(totalsGm) + slot;
    *totals = {scheduler.IsValid(), scheduler.GetBlockNums(), scheduler.GetCoreNums(SCHEDULER_BLOCK_NUM), 0};
    for (int64_t blockIdx = coreIdx; blockIdx < scheduler.GetBlockNums(); blockIdx += SCHEDULER_BLOCK_NUM) {
        auto info = scheduler.GetBlockInfo(blockIdx);
        auto record = MakeSchedulerRecord(info);
        auto getters = MakeSchedulerRecord(
            {scheduler.GetBlockShape(blockIdx), scheduler.GetBlockCoord(blockIdx), info.rotationOffset});
        scheduler.GetBlockInfo(0);
        auto repeat = MakeSchedulerRecord(scheduler.GetBlockInfo(blockIdx));
        record.consistent = record.coord == getters.coord && record.shape == getters.shape &&
                            record.rotationOffset == repeat.rotationOffset && record.coord == repeat.coord &&
                            record.shape == repeat.shape;
        if (totals->visited < RECORD_CAPACITY) {
            records[totals->visited] = record;
        }
        ++totals->visited;
    }
}

void CheckScheduler(const std::vector<int64_t>& counts, int64_t m, int64_t n, int64_t k, int64_t blockSize,
                    int64_t groupListType, bool multiRotation = false)
{
    const auto groupList = MakeGroupList(counts, groupListType);
    std::vector<SchedulerRecord> expected;
    int64_t row = 0;
    const int64_t b = multiRotation ? n / k : 1;
    for (size_t groupIdx = 0; groupIdx < counts.size(); ++groupIdx) {
        const int64_t stop = row + counts[groupIdx] * (b > 1 ? 1 : n / k);
        while (row < stop) {
            const int64_t rows = std::min(blockSize, stop - row);
            for (int64_t slice = 0; slice < b; ++slice) {
                expected.push_back({(static_cast<int64_t>(groupIdx) * b + slice) * k * k,
                                    {row, 0, 0, slice},
                                    {rows, k, k, blockSize, blockSize, k},
                                    1});
            }
            row += rows;
        }
    }
    SchedulerCaseConfig config{m, n, k, blockSize, static_cast<int64_t>(groupList.size()), groupListType, b};
    const size_t slots = CORE_ROLE_NUM * SCHEDULER_BLOCK_NUM;
    const size_t recordBytes = slots * RECORD_CAPACITY * sizeof(SchedulerRecord);
    const size_t totalsBytes = slots * sizeof(SchedulerTotals);
    GmBuffer configGm(sizeof(config));
    GmBuffer groups(groupList.size() * sizeof(int64_t));
    GmBuffer records(recordBytes);
    GmBuffer totals(totalsBytes);
    ASSERT_NE(configGm.Get(), nullptr);
    ASSERT_NE(groups.Get(), nullptr);
    ASSERT_NE(records.Get(), nullptr);
    ASSERT_NE(totals.Get(), nullptr);
    std::memcpy(configGm.Get(), &config, sizeof(config));
    std::memcpy(groups.Get(), groupList.data(), groupList.size() * sizeof(int64_t));
    std::memset(records.Get(), SENTINEL_BYTE, recordBytes);
    std::memset(totals.Get(), SENTINEL_BYTE, totalsBytes);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(
        KERNEL_RUN_KF(ScheduleKernel, SCHEDULER_BLOCK_NUM, configGm.Get(), groups.Get(), records.Get(), totals.Get()));
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto& actualTotals = reinterpret_cast<SchedulerTotals*>(totals.Get())[slot];
        const auto* actual = reinterpret_cast<SchedulerRecord*>(records.Get()) + slot * RECORD_CAPACITY;
        ASSERT_EQ(actualTotals.valid, 1);
        ASSERT_EQ(actualTotals.blockNum, expected.size());
        ASSERT_EQ(actualTotals.coreNum, std::min<int64_t>(expected.size(), SCHEDULER_BLOCK_NUM));
        int64_t count = 0;
        for (size_t blockIdx = slot % SCHEDULER_BLOCK_NUM; blockIdx < expected.size();
             blockIdx += SCHEDULER_BLOCK_NUM) {
            ASSERT_LT(count, RECORD_CAPACITY);
            EXPECT_EQ(actual[count].rotationOffset, expected[blockIdx].rotationOffset);
            EXPECT_EQ(actual[count].coord, expected[blockIdx].coord);
            EXPECT_EQ(actual[count].shape, expected[blockIdx].shape);
            EXPECT_EQ(actual[count].consistent, 1);
            ++count;
        }
        ASSERT_EQ(actualTotals.visited, count);
        ASSERT_LT(count, RECORD_CAPACITY);
        EXPECT_EQ(static_cast<uint64_t>(actual[count].rotationOffset), SENTINEL_WORD);
        EXPECT_EQ(static_cast<uint64_t>(actual[count].consistent), SENTINEL_WORD);
        for (auto value : actual[count].coord) {
            EXPECT_EQ(static_cast<uint64_t>(value), SENTINEL_WORD);
        }
        for (auto value : actual[count].shape) {
            EXPECT_EQ(static_cast<uint64_t>(value), SENTINEL_WORD);
        }
    }
}

void CheckInvalidScheduler(const std::vector<int64_t>& groupList, int64_t groupListType, int64_t groupNum = 2,
                           bool nullGroups = false)
{
    SchedulerCaseConfig config{6, 96, 32, 16, groupNum, groupListType};
    const size_t slots = CORE_ROLE_NUM * SCHEDULER_BLOCK_NUM;
    const size_t totalsBytes = slots * sizeof(SchedulerTotals);
    GmBuffer configGm(sizeof(config));
    GmBuffer groups(groupList.size() * sizeof(int64_t));
    GmBuffer records(slots * RECORD_CAPACITY * sizeof(SchedulerRecord));
    GmBuffer totals(totalsBytes);
    ASSERT_NE(configGm.Get(), nullptr);
    ASSERT_NE(groups.Get(), nullptr);
    ASSERT_NE(records.Get(), nullptr);
    ASSERT_NE(totals.Get(), nullptr);
    std::memcpy(configGm.Get(), &config, sizeof(config));
    std::memcpy(groups.Get(), groupList.data(), groupList.size() * sizeof(int64_t));
    std::memset(totals.Get(), SENTINEL_BYTE, totalsBytes);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(ScheduleKernel, SCHEDULER_BLOCK_NUM, configGm.Get(), nullGroups ? nullptr : groups.Get(),
                              records.Get(), totals.Get()));
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto& actual = reinterpret_cast<SchedulerTotals*>(totals.Get())[slot];
        EXPECT_EQ(actual.valid, 0);
        EXPECT_EQ(actual.blockNum, 0);
        EXPECT_EQ(actual.coreNum, 0);
        EXPECT_EQ(actual.visited, 0);
    }
}

} // namespace

TEST(RotateQuantBlazeTest, AssembleFp16Outputs)
{
    CheckPublicAssembly<half, fp4x2_e2m1_t, false>();
    CheckPublicAssembly<half, fp8_e4m3fn_t, false>();
    CheckPublicAssembly<half, fp8_e5m2_t, false>();
    CheckPublicAssembly<half, fp4x2_e2m1_t, true>();
    CheckPublicAssembly<half, fp8_e4m3fn_t, true>();
    CheckPublicAssembly<half, fp8_e5m2_t, true>();
}

TEST(RotateQuantBlazeTest, AssembleBf16Outputs)
{
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, false>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, false>();
    CheckPublicAssembly<bfloat16_t, fp8_e5m2_t, false>();
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, true>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, true>();
    CheckPublicAssembly<bfloat16_t, fp8_e5m2_t, true>();
}

TEST(RotateQuantBlazeTest, KernelSmokeSharedRotation)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false>({17, 96, 32, 1});
    RunKernelSmoke<half, fp8_e4m3fn_t, false>({3, 32, 32, 1});
}

TEST(RotateQuantBlazeTest, KernelSmokePerNRotation) { RunKernelSmoke<half, fp8_e5m2_t, false>({17, 96, 32, 3}); }

TEST(RotateQuantBlazeTest, KernelSmokeGroupedSharedRotation)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData tiling{20, 96, 32, 1};
        tiling.groupListType = groupListType;
        tiling.hasAlpha = 1;
        RunKernelSmoke<half, fp4x2_e2m1_t, true>(tiling, {1, 0, 17});
    }
}

TEST(RotateQuantBlazeTest, KernelSmokeGroupedPerNRotation)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData tiling{20, 96, 32, 3};
        tiling.groupListType = groupListType;
        RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, true>(tiling, {1, 0, 17});
    }
}

TEST(RotateQuantBlazeTest, KernelSmokeK256)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false>({33, 512, 256, 1});
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, true>({33, 512, 256, 1}, {7, 0, 26});
    RunKernelSmoke<half, fp4x2_e2m1_t, true>({33, 512, 256, 2}, {7, 0, 26});
}

TEST(RotateQuantBlazeTest, KernelSmokeAllEmptyGroups)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, true>({7, 64, 32, 1}, {0, 0, 0});
}

TEST(RotateQuantBlazeTest, GroupBoundariesAndBothRepresentationsOnAllLanes)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        for (int64_t k : {32, 64, 128, 256}) {
            CheckScheduler({0, 1, 17, 0, 2, 33, 0}, 64, 3 * k, k, 16, groupListType);
            CheckScheduler({1}, 8, k, k, 16, groupListType);
            CheckScheduler({0, 0, 0}, 64, 3 * k, k, 16, groupListType);
        }
        CheckScheduler(std::vector<int64_t>(MAX_GROUP_NUM, 1), 1030, 96, 32, 16, groupListType);
    }
}

TEST(RotateQuantBlazeTest, PerNRotationGroupsOnAllLanes)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        for (int64_t k : {32, 64, 128, 256}) {
            CheckScheduler({0, 1, 17, 0, 2, 33, 0}, 64, 3 * k, k, 16, groupListType, true);
            CheckScheduler({1}, 8, k, k, 16, groupListType, true);
            CheckScheduler({0, 0, 0}, 64, 3 * k, k, 16, groupListType, true);
        }
        CheckScheduler(std::vector<int64_t>(MAX_GROUP_NUM, 1), 1030, 96, 32, 16, groupListType, true);
    }
}

TEST(RotateQuantBlazeTest, InvalidDeviceListsProduceNoTasks)
{
    for (const auto& groupList : std::vector<std::vector<int64_t>>{{2, -1}, {5, 1}, {1, 9}}) {
        for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
            if (groupList == std::vector<int64_t>{5, 1} && groupListType == GROUP_LIST_COUNT) {
                continue; // This list is valid as counts.
            }
            CheckInvalidScheduler(groupList, groupListType);
        }
    }
    CheckInvalidScheduler({1, 1}, GROUP_LIST_COUNT, 0);
    CheckInvalidScheduler({1, 1}, GROUP_LIST_COUNT, MAX_GROUP_NUM + 1);
    CheckInvalidScheduler({1, 1}, 2);
    CheckInvalidScheduler({1, 1}, GROUP_LIST_COUNT, 2, true);
}

TEST(RotateQuantBlazeTest, DefaultModePreservesSingleAndMultiRotationCoordinates)
{
    OrdinaryScheduler::Params params;
    params.mL1 = params.baseM = 16;
    params.nL1 = params.kL1 = params.baseN = params.baseK = 32;
    params.nKRatio = 3;
    params.tailML1 = 7;
    OrdinaryScheduler single({13, 96, 32, 1}, SCHEDULER_BLOCK_NUM, params);
    ASSERT_EQ(single.GetBlockNums(), 3);
    EXPECT_EQ(asc::te::get<0>(single.GetBlockCoord(2)), 32);
    EXPECT_EQ(asc::te::get<0>(single.GetBlockShape(2)), 7);
    EXPECT_EQ(single.GetBlockInfo(2).rotationOffset, 0);
    OrdinaryScheduler multi({17, 96, 32, 3}, SCHEDULER_BLOCK_NUM, params);
    ASSERT_EQ(multi.GetBlockNums(), 6);
    EXPECT_EQ(asc::te::get<0>(multi.GetBlockCoord(5)), 16);
    EXPECT_EQ(asc::te::get<3>(multi.GetBlockCoord(5)), 2);
    EXPECT_EQ(asc::te::get<0>(multi.GetBlockShape(5)), 1);
    EXPECT_EQ(multi.GetBlockInfo(5).rotationOffset, 2 * 32 * 32);
}

TEST(RotateQuantBlazeTest, RotationOffsetsPreserve64BitsAndEmptyGroupIndices)
{
    constexpr int64_t k = 256;
    constexpr int64_t slices = 65536;
    const std::array<int64_t, 4> counts{0, 1, 0, 1};
    GmBuffer groups(sizeof(counts));
    ASSERT_NE(groups.Get(), nullptr);
    std::memcpy(groups.Get(), counts.data(), sizeof(counts));
    GroupedScheduler::Params params;
    params.mL1 = params.baseM = 16;
    params.nL1 = params.kL1 = params.baseN = k;
    params.baseK = MAX_BASE_K;
    params.nKRatio = slices;
    GroupedScheduler scheduler({2, slices * k, k, slices}, SCHEDULER_BLOCK_NUM, params,
                               {groups.Get(), static_cast<int64_t>(counts.size()), GROUP_LIST_COUNT});
    ASSERT_TRUE(scheduler.IsValid());
    ASSERT_EQ(scheduler.GetBlockNums(), 2 * slices);
    EXPECT_EQ(scheduler.GetBlockInfo(0).rotationOffset, slices * k * k);
    const auto info = scheduler.GetBlockInfo(slices + 3);
    EXPECT_EQ(asc::te::get<0>(info.blockCoord), 1);
    EXPECT_EQ(asc::te::get<3>(info.blockCoord), 3);
    EXPECT_EQ(asc::te::get<0>(info.blockShape), 1);
    EXPECT_EQ(info.rotationOffset, (3 * slices + 3) * k * k);
}

TEST(RotateQuantBlazeTest, OrdinaryModeIgnoresGroupMetadata)
{
    OrdinaryScheduler::Params params;
    params.mL1 = params.baseM = 16;
    params.nL1 = params.kL1 = params.baseN = params.baseK = 32;
    params.nKRatio = 3;
    params.tailML1 = 7;
    OrdinaryScheduler scheduler({13, 96, 32, 1}, SCHEDULER_BLOCK_NUM, params, {nullptr, MAX_GROUP_NUM + 1, -1});
    ASSERT_TRUE(scheduler.IsValid());
    ASSERT_EQ(scheduler.GetBlockNums(), 3);
    const auto tail = scheduler.GetBlockInfo(2);
    EXPECT_EQ(asc::te::get<0>(tail.blockShape), 7);
    EXPECT_EQ(asc::te::get<0>(tail.blockCoord), 32);
    EXPECT_EQ(tail.rotationOffset, 0);
}

TEST(RotateQuantBlazeTest, RotationExtentOverflowProducesNoTasks)
{
    OrdinaryScheduler::Params params;
    params.mL1 = 16;
    params.nKRatio = 1;
    const int64_t k = std::numeric_limits<int64_t>::max();
    OrdinaryScheduler scheduler({1, k, k, 1}, SCHEDULER_BLOCK_NUM, params);
    EXPECT_EQ(scheduler.IsValid(), false);
    EXPECT_EQ(scheduler.GetBlockNums(), 0);
    EXPECT_EQ(scheduler.GetCoreNums(SCHEDULER_BLOCK_NUM), 0);
}
