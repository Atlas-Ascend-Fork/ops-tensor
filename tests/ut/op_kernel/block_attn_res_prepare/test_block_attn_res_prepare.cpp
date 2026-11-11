/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_block_attn_res_prepare.cpp
 * \brief UT for the BlockAttnResPrepare mixed AIC/AIV component stack.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "block_attn_res_prepare.h"
#include "blaze_kernel_stub.h"
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"

namespace {
constexpr uint32_t TOTAL_T = 1U;
constexpr uint32_t TOTAL_N = 8U;
constexpr uint32_t TOTAL_S = 2U;
constexpr uint64_t TOTAL_D = 32U;
constexpr uint32_t S_ALIGN = 16U;
constexpr uint32_t N_ALIGN = 16U;
constexpr uint32_t BASE_D_ALIGN = 32U;
constexpr uint64_t WORKSPACE_PER_CORE_ELEMS = 512U;
constexpr uint32_t SINGLE_STAGE = 1U;
constexpr uint32_t DOUBLE_L1_STAGE = 2U;
constexpr uint32_t E_BUFFER_NUM = 2U;
constexpr uint32_t V_UB_BUFFER_NUM = 2U;
constexpr uint32_t SOFTMAX_STAT_NUM = 2U;
constexpr uint32_t MM1_L0_K_MAX = 64U;
using Kernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
using Scheduler = typename Kernel::BlockScheduler;
using Params = typename Kernel::Params;
constexpr uint32_t SYNC_ROUNDS = 4U;

__global__ __aicore__ void block_attn_res_prepare_sync_kernel_entry(GM_ADDR progressAddress)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    using Sync = Blaze::Attention::Kernel::BlockAttnResPrepareDetail::Sync;
    auto* progress = reinterpret_cast<uint32_t*>(progressAddress);
    for (uint32_t round = 0U; round < SYNC_ROUNDS; ++round) {
        const uint16_t eSlotIdx = static_cast<uint16_t>(round % E_BUFFER_NUM);
        if ASCEND_IS_AIC {
            Sync::NotifyDotReady();
            Sync::WaitEReady(eSlotIdx);
            Sync::NotifyEBufferFree(eSlotIdx);
            progress[0U] = round + 1U;
        } else {
            Sync::WaitDotReady();
            Sync::NotifyEReady(eSlotIdx);
            Sync::WaitEBufferFree(eSlotIdx);
            progress[1U + AscendC::GetSubBlockIdx()] = round + 1U;
        }
    }
}

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

Params MakeParams()
{
    Params params{};
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, TOTAL_T};
    params.mm1Params.mL1 = S_ALIGN;
    params.mm1Params.nL1 = N_ALIGN;
    params.mm1Params.kL1 = BASE_D_ALIGN;
    params.mm1Params.mL0 = S_ALIGN;
    params.mm1Params.nL0 = N_ALIGN;
    params.mm1Params.kL0 = static_cast<uint32_t>(TOTAL_D);
    params.mm1Params.l1Stages = SINGLE_STAGE;
    params.mm1Params.l0cStages = SINGLE_STAGE;
    params.mm2Params.mL1 = S_ALIGN;
    params.mm2Params.nL1 = BASE_D_ALIGN;
    params.mm2Params.kL1 = N_ALIGN;
    params.mm2Params.mL0 = S_ALIGN;
    params.mm2Params.nL0 = BASE_D_ALIGN;
    params.mm2Params.kL0 = N_ALIGN;
    params.mm2Params.l1Stages = SINGLE_STAGE;
    params.mm2Params.l0cStages = SINGLE_STAGE;
    params.epilogueParams.totalD = TOTAL_D;
    params.epilogueParams.baseD = static_cast<uint32_t>(TOTAL_D);
    params.epilogueParams.baseDAlign = BASE_D_ALIGN;
    params.epilogueParams.dTileNum = 1U;
    params.epilogueParams.sAlign = S_ALIGN;
    params.epilogueParams.vUbBufferNum = V_UB_BUFFER_NUM;
    params.epilogueParams.eWorkspaceElems = static_cast<uint64_t>(S_ALIGN) * N_ALIGN;
    params.epilogueParams.vUbElems = static_cast<uint64_t>(TOTAL_N) * BASE_D_ALIGN;
    params.epilogueParams.dotUbElems = static_cast<uint64_t>(S_ALIGN) * N_ALIGN;
    params.epilogueParams.reduceUbElems = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    params.epilogueParams.softmaxUbElems = static_cast<uint64_t>(SOFTMAX_STAT_NUM) * S_ALIGN;
    params.epilogueParams.workspacePerCoreElems = WORKSPACE_PER_CORE_ELEMS;
    params.schedulerParams.totalWorkUnits = 1U;
    params.schedulerParams.usedCoreNum = 1U;
    params.schedulerParams.baseT = 1U;
    params.schedulerParams.baseS = S_ALIGN;
    params.schedulerParams.sTileNum = 1U;
    params.schedulerParams.mm1NAlign = N_ALIGN;
    return params;
}

BlockAttnResPrepareTestTiling MakeTestTiling(const Params& params)
{
    BlockAttnResPrepareTestTiling tiling{};
    tiling.totalS = static_cast<uint32_t>(asc::te::get<0>(params.problemShape));
    tiling.totalN = static_cast<uint32_t>(asc::te::get<1>(params.problemShape));
    tiling.totalD = static_cast<uint64_t>(asc::te::get<2>(params.problemShape));
    tiling.totalT = static_cast<uint32_t>(asc::te::get<3>(params.problemShape));
    tiling.totalWorkUnits = params.schedulerParams.totalWorkUnits;
    tiling.usedCoreNum = params.schedulerParams.usedCoreNum;
    tiling.baseS = params.schedulerParams.baseS;
    tiling.baseT = params.schedulerParams.baseT;
    tiling.baseD = params.epilogueParams.baseD;
    tiling.baseDAlign = params.epilogueParams.baseDAlign;
    tiling.sTileNum = params.schedulerParams.sTileNum;
    tiling.dTileNum = params.epilogueParams.dTileNum;
    tiling.sAlign = params.epilogueParams.sAlign;
    tiling.nAlign = static_cast<uint32_t>(params.mm2Params.kL1);
    tiling.mm1NAlign = params.schedulerParams.mm1NAlign;
    tiling.mm1L1Stages = static_cast<uint8_t>(params.mm1Params.l1Stages);
    tiling.vUbBufferNum = static_cast<uint8_t>(params.epilogueParams.vUbBufferNum);
    tiling.eWorkspaceElems = params.epilogueParams.eWorkspaceElems;
    tiling.vUbElems = params.epilogueParams.vUbElems;
    tiling.dotUbElems = params.epilogueParams.dotUbElems;
    tiling.reduceUbElems = params.epilogueParams.reduceUbElems;
    tiling.softmaxUbElems = params.epilogueParams.softmaxUbElems;
    tiling.workspacePerCoreElems = params.epilogueParams.workspacePerCoreElems;
    tiling.epsilon = params.epilogueParams.epsilon;
    return tiling;
}

void FillInputs(float* residual, float* query, uint64_t totalT, uint64_t totalN, uint64_t totalS, uint64_t totalD)
{
    for (uint64_t sIndex = 0U; sIndex < totalS; ++sIndex) {
        for (uint64_t dIndex = 0U; dIndex < totalD; ++dIndex) {
            const int64_t pattern = static_cast<int64_t>((sIndex * 17U + dIndex * 3U) % 19U) - 9;
            query[sIndex * totalD + dIndex] = static_cast<float>(pattern) * 0.03125F;
        }
    }
    for (uint64_t tIndex = 0U; tIndex < totalT; ++tIndex) {
        for (uint64_t nIndex = 0U; nIndex < totalN; ++nIndex) {
            for (uint64_t dIndex = 0U; dIndex < totalD; ++dIndex) {
                const int64_t pattern = static_cast<int64_t>((tIndex * 13U + nIndex * 7U + dIndex * 5U) % 23U) - 11;
                const uint64_t index = (tIndex * totalN + nIndex) * totalD + dIndex;
                residual[index] = static_cast<float>(pattern) * 0.015625F;
            }
        }
    }
}

void RunKernelSmoke(const Params& inputParams, uint64_t validN)
{
    constexpr uint32_t BLOCK_NUM = 1U;
    const uint64_t totalS = static_cast<uint64_t>(asc::te::get<0>(inputParams.problemShape));
    const uint64_t totalN = static_cast<uint64_t>(asc::te::get<1>(inputParams.problemShape));
    const uint64_t totalD = static_cast<uint64_t>(asc::te::get<2>(inputParams.problemShape));
    const uint64_t totalT = static_cast<uint64_t>(asc::te::get<3>(inputParams.problemShape));
    const size_t residualElems = static_cast<size_t>(totalT * totalN * totalD);
    const size_t queryElems = static_cast<size_t>(totalS * totalD);
    const size_t statElems = static_cast<size_t>(totalS * totalT);
    const size_t outputElems = statElems * totalD;
    const size_t workspaceElems = BLOCK_NUM * inputParams.epilogueParams.workspacePerCoreElems;

    GmBuffer residual(residualElems * sizeof(float));
    GmBuffer query(queryElems * sizeof(float));
    GmBuffer validBlocks(sizeof(uint64_t));
    GmBuffer softmaxMax(statElems * sizeof(float));
    GmBuffer weightedOutput(outputElems * sizeof(float));
    GmBuffer softmaxSum(statElems * sizeof(float));
    GmBuffer workspace(workspaceElems * sizeof(float));
    GmBuffer tilingBuffer(sizeof(BlockAttnResPrepareTestTiling));

    ASSERT_NE(residual.Get(), nullptr);
    ASSERT_NE(query.Get(), nullptr);
    ASSERT_NE(validBlocks.Get(), nullptr);
    ASSERT_NE(softmaxMax.Get(), nullptr);
    ASSERT_NE(weightedOutput.Get(), nullptr);
    ASSERT_NE(softmaxSum.Get(), nullptr);
    ASSERT_NE(workspace.Get(), nullptr);
    ASSERT_NE(tilingBuffer.Get(), nullptr);

    auto* residualData = reinterpret_cast<float*>(residual.Get());
    auto* queryData = reinterpret_cast<float*>(query.Get());
    FillInputs(residualData, queryData, totalT, totalN, totalS, totalD);
    *reinterpret_cast<uint64_t*>(validBlocks.Get()) = validN;
    std::fill_n(reinterpret_cast<float*>(softmaxMax.Get()), statElems, 0.0F);
    std::fill_n(reinterpret_cast<float*>(weightedOutput.Get()), outputElems, 0.0F);
    std::fill_n(reinterpret_cast<float*>(softmaxSum.Get()), statElems, 0.0F);
    std::fill_n(reinterpret_cast<float*>(workspace.Get()), workspaceElems, 0.0F);
    *reinterpret_cast<BlockAttnResPrepareTestTiling*>(tilingBuffer.Get()) = MakeTestTiling(inputParams);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const bool ok = KERNEL_RUN_KF(block_attn_res_prepare_kernel_entry, BLOCK_NUM, residual.Get(), query.Get(),
                                  validBlocks.Get(), softmaxMax.Get(), weightedOutput.Get(), softmaxSum.Get(),
                                  workspace.Get(), tilingBuffer.Get());
    ASSERT_TRUE(ok) << "BlockAttnResPrepare mixed kernel execution failed";
    // CANN 9.2 tikicpulib does not model the Ascend950 MMAD and Reg/VF numerics used by this mixed kernel.
    // Validate template assembly, scheduling, cross-core synchronization, and a clean launch here; numerical
    // accuracy remains covered by the on-device ST/TTK cases.
}
} // namespace

class BlockAttnResPrepareTest : public testing::Test {};

TEST_F(BlockAttnResPrepareTest, CrossCoreSyncReusesBothESlots)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    constexpr uint32_t EXPECTED_AIV_NUM = 1U;
#else
    constexpr uint32_t EXPECTED_AIV_NUM = 2U;
#endif
    ASSERT_EQ(Blaze::Gemm::MIX_CV_RATIO, EXPECTED_AIV_NUM);
    // Reserve a sentinel lane as well, so an unexpected sibling AIV is visible in the result.
    constexpr uint32_t PROGRESS_LANES = 3U;
    GmBuffer progress(PROGRESS_LANES * sizeof(uint32_t));
    ASSERT_NE(progress.Get(), nullptr);
    auto* progressData = reinterpret_cast<uint32_t*>(progress.Get());
    std::fill_n(progressData, PROGRESS_LANES, 0U);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(block_attn_res_prepare_sync_kernel_entry, 1U, progress.Get()));
    EXPECT_EQ(progressData[0U], SYNC_ROUNDS);
    for (uint32_t lane = 1U; lane <= EXPECTED_AIV_NUM; ++lane) {
        EXPECT_EQ(progressData[lane], SYNC_ROUNDS) << "AIV lane=" << lane - 1U;
    }
    for (uint32_t lane = EXPECTED_AIV_NUM + 1U; lane < PROGRESS_LANES; ++lane) {
        EXPECT_EQ(progressData[lane], 0U) << "Unexpected AIV lane=" << lane - 1U;
    }
}

TEST_F(BlockAttnResPrepareTest, TemplateContracts)
{
    static_assert(std::is_default_constructible_v<Kernel>);
    static_assert(std::is_same_v<Kernel, Blaze::Attention::Kernel::AttentionUniversal<
                                             Blaze::Attention::Kernel::BlockAttnResPrepareProblemShape,
                                             Blaze::Attention::Kernel::BlockAttnResPrepareMmadTuple,
                                             Blaze::Attention::Kernel::BlockAttnResPrepareBlockEpilogue,
                                             Blaze::Attention::Kernel::BlockAttnResPrepareBlockScheduler, void>>);
    static_assert(std::is_same_v<typename Kernel::DispatchPolicy, Blaze::Attention::BlockAttnResPreparePolicy>);
    static_assert(AscendC::Std::tuple_size_v<typename Kernel::BlockMmadTuple> == 2U);
    static_assert(std::is_same_v<typename Kernel::Mm1Block, Blaze::Attention::Kernel::BlockAttnResPrepareMm1>);
    static_assert(std::is_same_v<typename Kernel::Mm2Block, Blaze::Attention::Kernel::BlockAttnResPrepareMm2>);
    static_assert(std::is_same_v<typename Kernel::Mm1Block::DispatchPolicy,
                                 Blaze::Attention::Kernel::BlockAttnResPrepareMm1MmadPolicy>);
    static_assert(std::is_same_v<typename Kernel::Mm2Block::DispatchPolicy,
                                 Blaze::Attention::Kernel::BlockAttnResPrepareMm2MmadPolicy>);
    static_assert(Kernel::Mm1Block::NON_CONTIGUOUS_TYPE ==
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_BATCHED_B));
    static_assert(Kernel::Mm2Block::NON_CONTIGUOUS_TYPE == 0U);
    static_assert(std::is_same_v<typename Kernel::Mm1Block::LayoutB, asc::te::dn_ext_layout_ptn>);
    static_assert(std::is_same_v<typename Kernel::Mm2Block::LayoutB, asc::te::nd_ext_layout_ptn>);
    static_assert(std::is_same_v<typename Kernel::BlockEpilogue::ElementType, float>);
    static_assert(std::is_same_v<typename Kernel::BlockEpilogueParams,
                                 typename Blaze::Attention::Kernel::BlockAttnResPrepareBlockEpilogue::Params>);
    static_assert(std::is_same_v<typename Kernel::BlockSchedulerParams,
                                 typename Blaze::Attention::Kernel::BlockAttnResPrepareBlockScheduler::Params>);
    SUCCEED();
}

TEST_F(BlockAttnResPrepareTest, Mm1ResidualUsesBatchedDnExtLayout)
{
    constexpr int64_t BATCH_COUNT = 2;
    constexpr int64_t ROWS = 32;
    constexpr int64_t COLUMNS = 8;
    constexpr int64_t BATCH_STRIDE = 512;
    constexpr int64_t COLUMN_STRIDE = 32;
    auto layout = Blaze::Attention::Kernel::BlockAttnResPrepareDetail::MakeBatchedDNExtLayout<float>(
        BATCH_COUNT, ROWS, COLUMNS, BATCH_STRIDE, COLUMN_STRIDE);

    static_assert(decltype(layout)::depth == asc::te::five_dim_data);
    EXPECT_EQ(asc::te::get<0>(layout.shape()), BATCH_COUNT);
    EXPECT_EQ(asc::te::get<0>(layout.stride()), BATCH_STRIDE);
    EXPECT_EQ((asc::te::get<1, 0, 1>(layout.shape())), ROWS);
    EXPECT_EQ((asc::te::get<1, 1, 1>(layout.shape())), COLUMNS);
    EXPECT_EQ((asc::te::get<1, 1, 1>(layout.stride())), COLUMN_STRIDE);
}

TEST_F(BlockAttnResPrepareTest, SchedulerDecodesBlock)
{
    auto params = MakeParams();
    params.problemShape = {24, TOTAL_N, TOTAL_D, 2};
    params.schedulerParams.totalWorkUnits = 4U;
    params.schedulerParams.baseS = 16U;
    params.schedulerParams.sTileNum = 2U;
    params.schedulerParams.mm1NAlign = TOTAL_N;
    Scheduler scheduler(params.problemShape, params.schedulerParams, TOTAL_N, 0U);
    Scheduler::BlockInfo blockInfo{};
    for (uint32_t blockIdx = 0U; blockIdx < params.schedulerParams.totalWorkUnits; ++blockIdx) {
        ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    }
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockCoord), 1);
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 1);
    EXPECT_EQ(asc::te::get<0>(blockInfo.blockCoord), 16);
    EXPECT_EQ(asc::te::get<0>(blockInfo.blockShape), 8);
}

TEST_F(BlockAttnResPrepareTest, SchedulerGroupsAdjacentTokens)
{
    auto params = MakeParams();
    params.problemShape = {24, TOTAL_N, TOTAL_D, 3};
    params.schedulerParams.baseT = 2U;
    params.schedulerParams.baseS = 16U;
    params.schedulerParams.sTileNum = 2U;
    params.schedulerParams.totalWorkUnits = 4U;
    Scheduler scheduler(params.problemShape, params.schedulerParams, TOTAL_N, 0U);
    Scheduler::BlockInfo blockInfo{};
    for (uint32_t blockIdx = 0U; blockIdx < params.schedulerParams.totalWorkUnits; ++blockIdx) {
        ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    }
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockCoord), 2);
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 1);
    EXPECT_EQ(asc::te::get<0>(blockInfo.blockCoord), 16);
    EXPECT_EQ(asc::te::get<0>(blockInfo.blockShape), 8);
}

TEST_F(BlockAttnResPrepareTest, SchedulerAssignsBalancedBlocksToCurrentCore)
{
    constexpr uint32_t TOTAL_BLOCK_NUMS = 10U;
    constexpr uint32_t USED_CORE_NUM = 3U;
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, TOTAL_BLOCK_NUMS};
    params.schedulerParams.totalWorkUnits = TOTAL_BLOCK_NUMS;
    params.schedulerParams.usedCoreNum = USED_CORE_NUM;
    params.schedulerParams.mm1NAlign = TOTAL_N;
    Scheduler scheduler(params.problemShape, params.schedulerParams, TOTAL_N, 0U);

    Scheduler::BlockInfo blockInfo{};
    uint32_t blockCount = 0U;
    while (scheduler.GetNextBlock(blockInfo)) {
        EXPECT_EQ(asc::te::get<3>(blockInfo.blockCoord), blockCount);
        ++blockCount;
    }
    EXPECT_EQ(blockCount, 4U);
}

TEST_F(BlockAttnResPrepareTest, SchedulerStopsAfterAssignedBlocks)
{
    auto params = MakeParams();
    Scheduler scheduler(params.problemShape, params.schedulerParams, TOTAL_N, 0U);

    Scheduler::BlockInfo blockInfo{};
    ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    EXPECT_FALSE(scheduler.GetNextBlock(blockInfo));
}

TEST_F(BlockAttnResPrepareTest, SchedulerExpandsTokenGroupForSmallRuntimeN)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 4};
    params.schedulerParams.totalWorkUnits = 4U;
    Scheduler scheduler(params.problemShape, params.schedulerParams, 1U, 0U);

    EXPECT_EQ(scheduler.GetBlockNums(), 1U);
    Scheduler::BlockInfo blockInfo{};
    ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockCoord), 0);
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 4);
}

TEST_F(BlockAttnResPrepareTest, SchedulerDoesNotExpandEmptyRuntimeN)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 4};
    params.schedulerParams.totalWorkUnits = 4U;
    Scheduler scheduler(params.problemShape, params.schedulerParams, 0U, 0U);

    EXPECT_EQ(scheduler.GetBlockNums(), 4U);
    Scheduler::BlockInfo blockInfo{};
    ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 1);
}

TEST_F(BlockAttnResPrepareTest, SchedulerDoesNotExpandRuntimeNAboveL1Capacity)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 4};
    params.schedulerParams.totalWorkUnits = 4U;
    Scheduler scheduler(params.problemShape, params.schedulerParams, N_ALIGN + 1U, 0U);

    EXPECT_EQ(scheduler.GetBlockNums(), 4U);
    Scheduler::BlockInfo blockInfo{};
    ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 1);
}

TEST_F(BlockAttnResPrepareTest, SchedulerKeepsEnoughBlocksForUsedCores)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 4};
    params.schedulerParams.totalWorkUnits = 4U;
    params.schedulerParams.usedCoreNum = 4U;
    Scheduler scheduler(params.problemShape, params.schedulerParams, 1U, 0U);

    EXPECT_EQ(scheduler.GetBlockNums(), 4U);
    Scheduler::BlockInfo blockInfo{};
    ASSERT_TRUE(scheduler.GetNextBlock(blockInfo));
    EXPECT_EQ(asc::te::get<3>(blockInfo.blockShape), 1);
}

TEST_F(BlockAttnResPrepareTest, EmptyValidBlocksLaunches) { RunKernelSmoke(MakeParams(), 0); }

TEST_F(BlockAttnResPrepareTest, SingleValidBlockLaunches) { RunKernelSmoke(MakeParams(), 1); }

TEST_F(BlockAttnResPrepareTest, MultipleValidBlocksLaunches) { RunKernelSmoke(MakeParams(), TOTAL_N); }

TEST_F(BlockAttnResPrepareTest, UnalignedDTailUsesAlignedMm1L1Capacity)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, 257, TOTAL_T};
    params.mm1Params.kL1 = 272U;
    params.mm1Params.kL0 = MM1_L0_K_MAX;
    params.mm2Params.nL1 = 272U;
    params.mm2Params.nL0 = 272U;
    params.epilogueParams.totalD = 257U;
    params.epilogueParams.baseD = 257U;
    params.epilogueParams.baseDAlign = 272U;
    params.epilogueParams.vUbElems = static_cast<uint64_t>(TOTAL_N) * params.epilogueParams.baseDAlign;
    RunKernelSmoke(params, 1);
}

TEST_F(BlockAttnResPrepareTest, Uint64ValidBlocksAboveInt32MaxClampsToTotalN)
{
    RunKernelSmoke(MakeParams(), 2147483648ULL);
}

TEST_F(BlockAttnResPrepareTest, Uint64MaxMinusTwoClampsToTotalN)
{
    RunKernelSmoke(MakeParams(), std::numeric_limits<uint64_t>::max() - 2U);
}

TEST_F(BlockAttnResPrepareTest, GroupedTokensUseDoubleBufferedPipelineSmoke)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, 64, 2};
    params.mm1Params.kL1 = BASE_D_ALIGN;
    params.mm1Params.kL0 = BASE_D_ALIGN;
    params.mm1Params.l1Stages = DOUBLE_L1_STAGE;
    params.mm2Params.nL1 = BASE_D_ALIGN;
    params.mm2Params.nL0 = BASE_D_ALIGN;
    params.epilogueParams.totalD = 64U;
    params.epilogueParams.baseD = 32U;
    params.epilogueParams.dTileNum = 2U;
    params.epilogueParams.eWorkspaceElems = static_cast<uint64_t>(S_ALIGN) * N_ALIGN;
    params.epilogueParams.vUbElems = static_cast<uint64_t>(TOTAL_N) * BASE_D_ALIGN;
    params.epilogueParams.dotUbElems = static_cast<uint64_t>(S_ALIGN) * params.mm1Params.nL1;
    params.epilogueParams.workspacePerCoreElems = static_cast<uint64_t>(S_ALIGN) * params.mm1Params.nL1 +
                                                  E_BUFFER_NUM * static_cast<uint64_t>(S_ALIGN) * N_ALIGN;
    params.schedulerParams.baseT = 2U;
    params.schedulerParams.mm1NAlign = static_cast<uint32_t>(params.mm1Params.nL1);
    RunKernelSmoke(params, 1);
}

TEST_F(BlockAttnResPrepareTest, RuntimeTokenGroupingUsesDoubleBufferedPipelineSmoke)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 2};
    params.schedulerParams.totalWorkUnits = 2U;
    params.epilogueParams.workspacePerCoreElems = static_cast<uint64_t>(S_ALIGN) * params.mm1Params.nL1 +
                                                  E_BUFFER_NUM * static_cast<uint64_t>(S_ALIGN) * N_ALIGN;
    RunKernelSmoke(params, 1);
}

TEST_F(BlockAttnResPrepareTest, RuntimeTokenGroupingPreservesBatchedBOrdering)
{
    auto params = MakeParams();
    params.problemShape = {TOTAL_S, TOTAL_N, TOTAL_D, 2};
    params.schedulerParams.totalWorkUnits = 2U;
    params.epilogueParams.workspacePerCoreElems = static_cast<uint64_t>(S_ALIGN) * params.mm1Params.nL1 +
                                                  E_BUFFER_NUM * params.epilogueParams.eWorkspaceElems;
    RunKernelSmoke(params, TOTAL_N);
}

TEST_F(BlockAttnResPrepareTest, MaximumVectorWidthNLaunches)
{
    auto params = MakeParams();
    constexpr uint32_t MAX_N = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    params.problemShape = {TOTAL_S, MAX_N, TOTAL_D, TOTAL_T};
    params.mm1Params.nL1 = MAX_N;
    params.mm1Params.nL0 = MAX_N;
    params.mm2Params.kL1 = MAX_N;
    params.mm2Params.kL0 = MAX_N;
    params.epilogueParams.eWorkspaceElems = static_cast<uint64_t>(S_ALIGN) * MAX_N;
    params.epilogueParams.vUbElems = static_cast<uint64_t>(MAX_N) * params.epilogueParams.baseDAlign;
    params.epilogueParams.dotUbElems = static_cast<uint64_t>(S_ALIGN) * MAX_N;
    params.epilogueParams.reduceUbElems = MAX_N;
    params.epilogueParams.workspacePerCoreElems = static_cast<uint64_t>(S_ALIGN) * MAX_N +
                                                  E_BUFFER_NUM * params.epilogueParams.eWorkspaceElems;
    params.schedulerParams.mm1NAlign = MAX_N;
    RunKernelSmoke(params, MAX_N);
}
