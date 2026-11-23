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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "rotate_quant.h"
#include "tikicpulib.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

namespace {

class RotateQuantBlazeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
        GTEST_SKIP()
            << "This retained Ascend950 suite assumes a 1:2 launch; single-AIV coverage uses RotateQuantSingleAivTest";
#endif
    }
};

using TilingData = RotateQuantUT::RotateQuantTilingData;
using RotateQuantUT::GROUP_LIST_COUNT;
using RotateQuantUT::GROUP_LIST_CUMSUM;
using RotateQuantUT::MAX_BASE_K;
constexpr uint32_t SMOKE_BLOCK_NUM = 2;
constexpr int64_t CORE_ROLE_NUM = 3; // One AIC and two AIV subcores.
constexpr int64_t RECORD_CAPACITY = 4096;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr int64_t MX_SCALE_PAIR_SIZE = 64;
constexpr int64_t MX_SCALES_PER_PAIR = 2;
constexpr uint8_t SENTINEL_BYTE = 0xA5;

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

template <typename XType_, typename YType_, bool EnableGrouped_, bool SplitM_ = true, uint32_t PipelineStages_ = 1,
          typename ScaleType_ = int8_t>
void CheckPublicAssembly()
{
    using Types = RotateQuantUT::RotateQuantBlazeTypes<XType_, YType_, ScaleType_, asc::te::nd_ext_layout_ptn,
                                                       EnableGrouped_, SplitM_, PipelineStages_>;
    using Kernel = typename Types::MatmulKernel;
    using BlockMmad = typename Types::BlockMmad;
    using BlockEpilogue = typename Types::BlockEpilogue;
    using BlockScheduler = typename Types::BlockScheduler;
    using SmokeTypes = RotateQuantUT::KernelSmokeTypes<XType_, YType_, ScaleType_, asc::te::nd_ext_layout_ptn,
                                                       EnableGrouped_, SplitM_, PipelineStages_>;
    if constexpr (EnableGrouped_) {
        static_assert(
            std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, Blaze::Gemm::KernelGroupedMmadNoQuant>);
        static_assert(std::is_same_v<typename BlockMmad::AType, XType_>);
        static_assert(std::is_same_v<typename BlockMmad::BType, XType_>);
        static_assert(std::is_same_v<typename BlockMmad::CType, bfloat16_t>);
        static_assert(std::is_same_v<BlockScheduler, Blaze::Gemm::Block::BlockSchedulerGmmNoQuant>);
    } else {
        static_assert(
            std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, Blaze::Gemm::KernelMmadMultiBlockTBMM>);
        static_assert(std::is_same_v<typename BlockMmad::AType, XType_>);
        static_assert(std::is_same_v<typename BlockMmad::BType, XType_>);
        static_assert(std::is_same_v<typename BlockMmad::CType, bfloat16_t>);
        static_assert(Kernel::NEED_EPILOGUE);
        static_assert(BlockMmad::DispatchPolicy::L0C2OUT_MODEL ==
                      (SplitM_ ? Blaze::Gemm::ND_ALIG_1V2_FIXPIPE : Blaze::Gemm::ND_ALIG_1V1_FIXPIPE));
        static_assert(BlockMmad::NON_CONTIGUOUS_TYPE ==
                      static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1));
        static_assert(std::is_same_v<typename Kernel::ProblemShape, RotateQuantUT::TbmmProblemShape>);
        using ExpectedScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<RotateQuantUT::TbmmProblemShape, 0,
                                                                                false, true, true>;
        static_assert(std::is_same_v<BlockScheduler, ExpectedScheduler>);
        static_assert(std::is_same_v<typename BlockScheduler::ProblemShape, RotateQuantUT::TbmmProblemShape>);
        static_assert(std::is_same_v<typename Kernel::BlockShape, typename BlockEpilogue::BlockShape>);
        static_assert(std::is_same_v<typename BlockMmad::LayoutA, asc::te::nd_ext_layout_ptn>);
        static_assert(!BlockMmad::WEIGHT_NZ_FORMAT);
    }
    static_assert(BlockMmad::DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_QUANT);
    static_assert(Kernel::ENABLE_QUANT);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeIn, bfloat16_t>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeOut, YType_>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeScale, ScaleType_>);
    static_assert(std::is_same_v<typename Kernel::BlockMmad, BlockMmad>);
    static_assert(std::is_same_v<typename Kernel::BlockEpilogue, BlockEpilogue>);
    static_assert(std::is_same_v<typename Kernel::BlockScheduler, BlockScheduler>);
    static_assert(std::is_same_v<typename BlockScheduler::BlockShape, typename BlockEpilogue::BlockShape>);
    // A single canonical entry also prevents a compatibility overload being reintroduced.
    static_assert(
        std::is_invocable_r_v<void, decltype(&BlockEpilogue::operator()), BlockEpilogue&,
                              const typename BlockEpilogue::BlockShape&, const typename BlockEpilogue::BlockCoord&>);
    static_assert(std::is_same_v<typename SmokeTypes::BlockEpilogue::Params, typename BlockEpilogue::Params>);

    typename Kernel::Params params{};
    constexpr float dstTypeMax = 6.0f;
    constexpr int64_t rotationSize = 32;
    constexpr int64_t originalN = 3 * rotationSize;
    params.epilogueParams = {nullptr,
                             nullptr,
                             {17, originalN, rotationSize, 1},
                             {},
                             false,
                             -1,
                             RotateQuantUT::ROUND_MODE_RINT,
                             2,
                             dstTypeMax,
                             1.0f / 6.0f,
                             PipelineStages_,
                             SplitM_};
    if constexpr (EnableGrouped_) {
        params.schedulerParams.baseM = 16;
        params.schedulerParams.baseN = rotationSize;
        params.schedulerParams.matrixView = BlockScheduler::MatrixView::BatchTransposedA;
        params.schedulerParams.splitM = SplitM_;
        params.problemShape = {17, originalN, rotationSize, 1};
        params.mmParams.groupListGmAddr = nullptr;
        params.gmmParams = {3, 0, GROUP_LIST_COUNT, 1, 1, 1, 0, 0};
        EXPECT_EQ(params.mmParams.groupListGmAddr, nullptr);
        EXPECT_EQ(params.gmmParams.groupNum, 3);
        params.mmParams.ubDB = PipelineStages_;
        EXPECT_EQ(params.mmParams.ubDB, PipelineStages_);
        EXPECT_EQ(params.schedulerParams.baseM, 16);
    } else {
        params.schParams.mL1 = 16;
        params.problemShape = {51, rotationSize, rotationSize, 1, 1};
        using BasicPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockTBMM>;
        using BasicBlock = Blaze::Gemm::Block::BlockMmad<BasicPolicy, XType_, asc::te::nd_ext_layout_ptn, XType_,
                                                         asc::te::nd_ext_layout_ptn, XType_, asc::te::nd_ext_layout_ptn,
                                                         XType_, asc::te::nd_ext_layout_ptn>;
        using BasicScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<RotateQuantUT::TbmmProblemShape>;
        static_assert(std::is_same_v<BasicScheduler, Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
                                                         RotateQuantUT::TbmmProblemShape, 0, false, true, false>>);
        using BasicKernel = Blaze::Gemm::Kernel::GemmUniversal<
            RotateQuantUT::TbmmProblemShape, BasicBlock, Blaze::Epilogue::Block::BlockEpilogueEmpty, BasicScheduler>;
        static_assert(!BasicKernel::NEED_EPILOGUE);
        // Existing callers still initialize exactly the original four Params members.
        typename BasicKernel::Params basicParams = {{17, 32, 32, 1, 1}, {}, {}, {}};
        EXPECT_EQ(asc::te::get<0>(basicParams.problemShape), 17);
        EXPECT_FALSE(BasicKernel::ENABLE_QUANT);
        params.mmadParams.ubDB = PipelineStages_;
        EXPECT_EQ(params.mmadParams.ubDB, PipelineStages_);
        const auto slotBytes = AscendC::TOTAL_UB_SIZE / PipelineStages_;
        EXPECT_LE(RotateQuantUT::EpilogueWorkspaceBytes(params.schParams.mL1), slotBytes);
        EXPECT_EQ(asc::te::get<0>(params.problemShape), 51);
        EXPECT_EQ(asc::te::get<1>(params.problemShape), rotationSize);
        EXPECT_EQ(asc::te::get<4>(params.problemShape), 1);
        EXPECT_EQ(asc::te::get<0>(params.epilogueParams.problemShape), 17);
        EXPECT_EQ(asc::te::get<1>(params.epilogueParams.problemShape), originalN);
    }
    if constexpr (!EnableGrouped_) {
        EXPECT_EQ(params.schParams.mL1, 16);
    }
    EXPECT_FLOAT_EQ(params.epilogueParams.dstTypeMax, dstTypeMax);
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

template <typename XType_, typename YType_, bool EnableGrouped_, bool SplitM_ = true, uint32_t PipelineStages_ = 1,
          bool SingleAivLaunch_ = false>
void RunKernelSmoke(TilingData tiling, const std::vector<int64_t>& counts = {})
{
    static_assert(!SingleAivLaunch_ || !SplitM_);
    ASSERT_GT(tiling.mL1, 0);
    ASSERT_EQ(tiling.mL1 % RotateQuantUT::BLOCK_M, 0);
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
    auto kernelFunc = [] {
        if constexpr (SingleAivLaunch_) {
            return RotateQuantSingleAivKernelEntry<XType_, YType_, int8_t, asc::te::nd_ext_layout_ptn, EnableGrouped_,
                                                   PipelineStages_>;
        } else {
            return RotateQuantKernelEntry<XType_, YType_, int8_t, asc::te::nd_ext_layout_ptn, EnableGrouped_, SplitM_,
                                          PipelineStages_>;
        }
    }();
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, SMOKE_BLOCK_NUM, x.Get(), rotation.Get(),
                              tiling.hasAlpha ? alpha.Get() : nullptr, EnableGrouped_ ? groups.Get() : nullptr, y.Get(),
                              scale.Get(), tilingGm.Get()))
        << "RotateQuant Blaze kernel smoke failed";
}

float CaptureInputValue(int64_t row, int64_t segment, int64_t k)
{
    return static_cast<float>(128 + (row * 11 + segment * 17 + k * 7) % 128) / 128.0f;
}

float CaptureRotationValue(int64_t batch, int64_t k, int64_t column)
{
    return static_cast<float>(1 + (batch * 7 + k * 3 + column * 5) % 17) / 256.0f;
}

uint16_t RoundToBf16Bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

bool VerifyBf16IntermediateEnabled()
{
    const char* enabled = std::getenv("ROTATE_QUANT_UT_VERIFY_INTERMEDIATE");
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

// Some CPU runners always launch MIX as 1:2 even for a 1:1 entry. Probe the
// runner without synchronization before testing the actual one-AIV contract.
::testing::AssertionResult ProbeSingleAivLaunch(bool& supported)
{
    std::array<int64_t, SMOKE_BLOCK_NUM * 2> ratios;
    ratios.fill(-1);
    GmBuffer probe(sizeof(ratios));
    if (probe.Get() == nullptr) {
        return ::testing::AssertionFailure() << "Cannot allocate task-ratio probe";
    }
    std::memcpy(probe.Get(), ratios.data(), sizeof(ratios));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    if (!KERNEL_RUN_KF(RotateQuantSingleAivLaunchProbe, SMOKE_BLOCK_NUM, probe.Get())) {
        return ::testing::AssertionFailure() << "One-AIV launch probe failed";
    }
    std::memcpy(ratios.data(), probe.Get(), sizeof(ratios));
    size_t aivTasks = 0;
    supported = true;
    for (const auto ratio : ratios) {
        if (ratio != -1) {
            ++aivTasks;
            supported = supported && ratio == 1;
        }
    }
    supported = supported && aivTasks == SMOKE_BLOCK_NUM;
    return ::testing::AssertionSuccess();
}

template <typename XType_, bool SplitM_ = true, bool SingleAivLaunch_ = false, uint32_t PipelineStages_ = 1,
          bool EnableGrouped_ = false>
void CheckBf16Intermediate(TilingData tiling, const std::vector<int64_t>& groupList = std::vector<int64_t>(),
                           uint32_t blockNum = SMOKE_BLOCK_NUM)
{
    static_assert(!SingleAivLaunch_ || !SplitM_);
    ASSERT_GT(blockNum, 0U);
    ASSERT_LE(blockNum, SMOKE_BLOCK_NUM);
    ASSERT_GT(tiling.mL1, 0);
    ASSERT_EQ(tiling.mL1 % RotateQuantUT::BLOCK_M, 0);
    tiling.baseK = std::min(tiling.k, MAX_BASE_K);
    if constexpr (EnableGrouped_) {
        tiling.groupNum = static_cast<int64_t>(groupList.size());
    }
    const int64_t segments = tiling.n / tiling.k;
    const size_t elements = static_cast<size_t>(tiling.m) * tiling.n;
    const size_t rotationElements = std::max<size_t>(EnableGrouped_ ? groupList.size() : 1, 1) * tiling.b * tiling.k *
                                    tiling.k;
    constexpr size_t GUARD_BYTES = RotateQuantUT::CAPTURE_GUARD_BYTES;
    constexpr size_t VISIT_WORDS = SMOKE_BLOCK_NUM * 2 * 2; // AIVs times workspace slots.
    constexpr size_t VISIT_BYTES = PipelineStages_ == 2 ? VISIT_WORDS * sizeof(uint64_t) : 0;
    GmBuffer x(elements * sizeof(XType_));
    GmBuffer rotation(rotationElements * sizeof(XType_));
    GmBuffer groups(std::max<size_t>(groupList.size(), 1) * sizeof(int64_t));
    GmBuffer output(elements * sizeof(bfloat16_t) + GUARD_BYTES + VISIT_BYTES);
    GmBuffer tilingGm(sizeof(tiling));
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(rotation.Get(), nullptr);
    ASSERT_NE(groups.Get(), nullptr);
    ASSERT_NE(output.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);
    auto* xValues = reinterpret_cast<XType_*>(x.Get());
    auto* rotationValues = reinterpret_cast<XType_*>(rotation.Get());
    for (int64_t row = 0; row < tiling.m; ++row) {
        for (int64_t segment = 0; segment < segments; ++segment) {
            for (int64_t k = 0; k < tiling.k; ++k) {
                xValues[row * tiling.n + segment * tiling.k + k] = XType_(CaptureInputValue(row, segment, k));
            }
        }
    }
    const int64_t matrixCount = static_cast<int64_t>(rotationElements / tiling.k / tiling.k);
    for (int64_t batch = 0; batch < matrixCount; ++batch) {
        for (int64_t k = 0; k < tiling.k; ++k) {
            for (int64_t column = 0; column < tiling.k; ++column) {
                rotationValues[(batch * tiling.k + k) * tiling.k + column] = XType_(
                    CaptureRotationValue(batch, k, column));
            }
        }
    }
    std::memset(output.Get(), SENTINEL_BYTE, elements * sizeof(bfloat16_t) + GUARD_BYTES);
    if constexpr (PipelineStages_ == 2) {
        std::memset(output.Get() + elements * sizeof(bfloat16_t) + GUARD_BYTES, 0, VISIT_BYTES);
    }
    std::memcpy(tilingGm.Get(), &tiling, sizeof(tiling));
    if (!groupList.empty()) {
        std::memcpy(groups.Get(), groupList.data(), groupList.size() * sizeof(int64_t));
    }
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    if constexpr (EnableGrouped_) {
        auto kernel = SingleAivLaunch_ ? RotateQuantGroupedBf16CaptureSingleAivEntry<XType_, PipelineStages_> :
                                         RotateQuantGroupedBf16CaptureEntry<XType_, SplitM_, PipelineStages_>;
        ASSERT_TRUE(
            KERNEL_RUN_KF(kernel, blockNum, x.Get(), rotation.Get(), groups.Get(), output.Get(), tilingGm.Get()));
    } else {
        auto kernel = SingleAivLaunch_ ? RotateQuantBf16CaptureSingleAivEntry<XType_, PipelineStages_> :
                                         RotateQuantBf16CaptureEntry<XType_, SplitM_, PipelineStages_>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernel, blockNum, x.Get(), rotation.Get(), output.Get(), tilingGm.Get()));
    }

    // This mathematical oracle deliberately does not use either scheduler. A
    // malformed later group leaves the completed valid prefix, then stops.
    std::vector<int64_t> rowGroups(static_cast<size_t>(tiling.m), EnableGrouped_ ? -1 : 0);
    int64_t expectedTasks = 0;
    if constexpr (EnableGrouped_) {
        int64_t end = 0;
        if (groupList.size() <= MAX_GROUP_NUM && (tiling.groupListType == 0 || tiling.groupListType == 1)) {
            for (size_t group = 0; group < groupList.size(); ++group) {
                const auto value = groupList[group];
                if (value < 0 || (tiling.groupListType == 0 && (value < end || value > tiling.m)) ||
                    (tiling.groupListType == 1 && value > tiling.m - end)) {
                    break;
                }
                const int64_t next = tiling.groupListType == 0 ? value : end + value;
                std::fill(rowGroups.begin() + end, rowGroups.begin() + next, static_cast<int64_t>(group));
                const int64_t rows = tiling.b == 1 ? (next - end) * segments : next - end;
                expectedTasks += (rows + tiling.mL1 - 1) / tiling.mL1 * tiling.b;
                end = next;
            }
        }
    } else {
        const int64_t problemM = tiling.b == 1 ? tiling.m * segments : tiling.m;
        expectedTasks = (problemM + tiling.mL1 - 1) / tiling.mL1 * tiling.b;
    }

    // Inputs and products are binary fractions. All K<=256 sums are exact FP32,
    // making this reference independent of the MMAD reduction order. These dense
    // cases also distinguish direct BF16 conversion from an intermediate FP16 cast.
    for (int64_t row = 0; row < tiling.m; ++row) {
        for (int64_t segment = 0; segment < segments; ++segment) {
            const int64_t group = rowGroups[row];
            const int64_t batch = group * tiling.b + (tiling.b == 1 ? 0 : segment);
            for (int64_t column = 0; column < tiling.k; ++column) {
                float sum = 0.0f;
                if (group >= 0) {
                    for (int64_t k = 0; k < tiling.k; ++k) {
                        sum += CaptureInputValue(row, segment, k) * CaptureRotationValue(batch, k, column);
                    }
                }
                const size_t index = row * tiling.n + segment * tiling.k + column;
                uint16_t actual;
                std::memcpy(&actual, output.Get() + index * sizeof(bfloat16_t), sizeof(actual));
                const uint16_t expected = group < 0 ? static_cast<uint16_t>(0xa5a5) : RoundToBf16Bits(sum);
                ASSERT_EQ(actual, expected) << "BF16 intermediate index " << index;
            }
        }
    }
    const auto* guard = reinterpret_cast<const uint8_t*>(output.Get()) + elements * sizeof(bfloat16_t);
    for (size_t i = 0; i < GUARD_BYTES; ++i) {
        EXPECT_EQ(guard[i], SENTINEL_BYTE) << "Output guard byte " << i;
    }
    if constexpr (PipelineStages_ == 2) {
        const auto* visits = reinterpret_cast<const uint64_t*>(guard + GUARD_BYTES);
        bool usedSecondSlot = false;
        for (size_t aiv = 0; aiv < SMOKE_BLOCK_NUM * 2; ++aiv) {
            usedSecondSlot = usedSecondSlot || visits[aiv * 2 + 1] == 1;
        }
        EXPECT_EQ(usedSecondSlot, expectedTasks > blockNum)
            << "Requested stage 2 must execute its second workspace on multi-tile cores";
    }
}

#if defined(ROTATE_QUANT_UT_REAL_EPILOGUE)
template <bool EnableGrouped_, bool SplitM_>
void CheckMxFp4Output()
{
    constexpr int64_t rows = 65;
    constexpr int64_t k = 64;
    constexpr int64_t packedBytesPerRow = k / 2;
    constexpr int64_t scalesPerRow = k / 32;
    constexpr size_t yBytes = rows * packedBytesPerRow;
    constexpr size_t scaleBytes = rows * scalesPerRow;
    constexpr size_t guardBytes = RotateQuantUT::CAPTURE_GUARD_BYTES;
    constexpr std::array<int64_t, 4> counts{17, 0, 16, 32};
    constexpr size_t rotationCount = EnableGrouped_ ? counts.size() : 1;
    GmBuffer x(rows * k * sizeof(bfloat16_t));
    GmBuffer rotation(rotationCount * k * k * sizeof(bfloat16_t));
    GmBuffer groupList(sizeof(counts));
    GmBuffer y(yBytes + guardBytes);
    GmBuffer scale(scaleBytes + guardBytes);
    GmBuffer tilingGm(sizeof(TilingData));
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(rotation.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(scale.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);

    auto* xValues = reinterpret_cast<bfloat16_t*>(x.Get());
    for (int64_t row = 0; row < rows; ++row) {
        const float magnitude = (row % 2 == 0) ? 1.0f : 2.0f;
        for (int64_t col = 0; col < k; ++col) {
            xValues[row * k + col] = bfloat16_t((row + col / 2) % 2 == 0 ? magnitude : -magnitude);
        }
    }
    auto* rotValues = reinterpret_cast<bfloat16_t*>(rotation.Get());
    std::fill_n(rotValues, rotationCount * k * k, bfloat16_t(0.0f));
    for (size_t index = 0; index < rotationCount; ++index) {
        for (int64_t col = 0; col < k; ++col) {
            rotValues[index * k * k + col * k + col] = bfloat16_t(1.0f);
        }
    }
    TilingData tiling{rows, k, k, 1};
    if constexpr (EnableGrouped_) {
        tiling.groupNum = counts.size();
        std::memcpy(groupList.Get(), counts.data(), sizeof(counts));
    }
    std::memcpy(tilingGm.Get(), &tiling, sizeof(tiling));
    std::memset(y.Get(), SENTINEL_BYTE, yBytes + guardBytes);
    std::memset(scale.Get(), SENTINEL_BYTE, scaleBytes + guardBytes);
    auto kernel = [] {
        if constexpr (SplitM_) {
            return RotateQuantKernelEntry<bfloat16_t, fp4x2_e2m1_t, int8_t, asc::te::nd_ext_layout_ptn, EnableGrouped_,
                                          true, 2, true>;
        } else {
            return RotateQuantSingleAivKernelEntry<bfloat16_t, fp4x2_e2m1_t, int8_t, asc::te::nd_ext_layout_ptn,
                                                   EnableGrouped_, 2, true>;
        }
    }();
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1, x.Get(), rotation.Get(), nullptr, EnableGrouped_ ? groupList.Get() : nullptr,
                              y.Get(), scale.Get(), tilingGm.Get()));

    // Identity rotation preserves the signed powers of two. OCP MXFP4 subtracts
    // Emax=2, so each mantissa sees +4 or -4, packed as two 0x6 or 0xE nibbles.
    const auto* yValues = reinterpret_cast<const uint8_t*>(y.Get());
    const auto* scaleValues = reinterpret_cast<const uint8_t*>(scale.Get());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < packedBytesPerRow; ++col) {
            ASSERT_EQ(yValues[row * packedBytesPerRow + col], (row + col) % 2 == 0 ? 0x66 : 0xEE)
                << "row=" << row << ", packed column=" << col;
        }
        for (int64_t col = 0; col < scalesPerRow; ++col) {
            ASSERT_EQ(scaleValues[row * scalesPerRow + col], 125 + row % 2)
                << "row=" << row << ", scale column=" << col;
        }
    }
    for (size_t i = 0; i < guardBytes; ++i) {
        EXPECT_EQ(yValues[yBytes + i], SENTINEL_BYTE);
        EXPECT_EQ(scaleValues[scaleBytes + i], SENTINEL_BYTE);
    }
}
#endif

struct SchedulerCaseConfig {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t blockSize{16};
    int64_t groupNum{0};
    int64_t groupListType{GROUP_LIST_COUNT};
    int64_t b{1};
    bool splitM{true};
};

struct SchedulerTotals {
    int64_t valid{0};
    int64_t blockNum{0};
    int64_t coreNum{0};
    int64_t visited{0};
};

struct OrdinarySchedulerComparison {
    int64_t expectedTaskNums{0};
    int64_t interleavedTaskNums{0};
    int64_t defaultTaskNums{0};
    int64_t expectedCoreNums{0};
    int64_t interleavedCoreNums{0};
    int64_t defaultCoreNums{0};
    int64_t compared{0};
    int64_t interleavedMismatches{0};
    int64_t defaultMismatches{0};
};

struct GmmIndexProbe {
    int64_t largeM{0};
    int64_t unsignedM{0};
    int64_t tasks{0};
};

struct GmmBatchedRecord {
    int64_t group{0};
    int64_t task{0};
    int64_t rotationOffset{0};
    int64_t inputOffset{0};
    std::array<int64_t, 4> shape{};
    std::array<int64_t, 4> coord{};
    std::array<int64_t, 4> outputCoord{};
    int64_t consistent{0};
};

__global__ __aicore__ void ProbeGmmBatchedView(GM_ADDR configGm, GM_ADDR countsGm, GM_ADDR recordsGm, GM_ADDR totalsGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto& config = *reinterpret_cast<const SchedulerCaseConfig*>(configGm);
    const auto* counts = reinterpret_cast<const int64_t*>(countsGm);
    int64_t core = AscendC::GetBlockIdx();
    int64_t role = 0;
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
        role = 1 + AscendC::GetSubBlockIdx();
    }
    const int64_t lane = role * AscendC::GetBlockNum() + core;
    auto* records = reinterpret_cast<GmmBatchedRecord*>(recordsGm) + lane * RECORD_CAPACITY;
    auto& totals = reinterpret_cast<SchedulerTotals*>(totalsGm)[lane];
    totals = {1, 0, 0, 0};
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    Scheduler::Params params{};
    params.baseM = config.blockSize;
    params.baseN = config.k;
    params.groupType = 0;
    params.groupNum = config.groupNum;
    params.matrixView = Scheduler::MatrixView::BatchTransposedA;
    params.splitM = config.splitM;
    Scheduler scheduler(params);
    for (int64_t group = 0; group < config.groupNum; ++group) {
        const auto offsets = scheduler.UpdateNextGroup({counts[group], config.n, config.k, config.b});
        for (int64_t task = scheduler.GetFirstBlockIdx(core); task < scheduler.GetBlockNums();
             task += AscendC::GetBlockNum()) {
            // Query coord first, then repeat both queries to check the shared role-local cache.
            const auto coord = scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetBlockShape<false, half>(task);
            const auto outputCoord = scheduler.GetOutputBlockCoord(task);
            const auto repeatedCoord = scheduler.GetBlockCoord(task);
            const auto repeatedShape = scheduler.GetBlockShape<false, half>(task);
            const bool consistent = asc::te::get<0>(coord) == asc::te::get<0>(repeatedCoord) &&
                                    asc::te::get<1>(coord) == asc::te::get<1>(repeatedCoord) &&
                                    asc::te::get<2>(coord) == asc::te::get<2>(repeatedCoord) &&
                                    asc::te::get<3>(coord) == asc::te::get<3>(repeatedCoord) &&
                                    asc::te::get<0>(shape) == asc::te::get<0>(repeatedShape) &&
                                    asc::te::get<1>(shape) == asc::te::get<1>(repeatedShape) &&
                                    asc::te::get<2>(shape) == asc::te::get<2>(repeatedShape) &&
                                    asc::te::get<3>(shape) == asc::te::get<3>(repeatedShape);
            if (totals.visited < RECORD_CAPACITY) {
                records[totals.visited] = {
                    group,
                    task,
                    asc::te::get<1>(offsets),
                    asc::te::get<0>(offsets),
                    {asc::te::get<0>(shape), asc::te::get<1>(shape), asc::te::get<2>(shape), asc::te::get<3>(shape)},
                    {asc::te::get<0>(coord), asc::te::get<1>(coord), asc::te::get<2>(coord), asc::te::get<3>(coord)},
                    {asc::te::get<0>(outputCoord), asc::te::get<1>(outputCoord), asc::te::get<2>(outputCoord),
                     asc::te::get<3>(outputCoord)},
                    consistent};
            }
            ++totals.visited;
        }
    }
}

void CheckGmmBatchedView(const std::vector<int64_t>& counts, int64_t n, int64_t k, int64_t batch, int64_t blockSize,
                         bool splitM)
{
    int64_t m = 5; // Leave an uncovered suffix to catch an accidental extra tile.
    for (const auto count : counts) {
        m += count;
    }
    const SchedulerCaseConfig config{m,     n,     k, blockSize, static_cast<int64_t>(counts.size()), GROUP_LIST_COUNT,
                                     batch, splitM};
    constexpr size_t slots = CORE_ROLE_NUM * SMOKE_BLOCK_NUM;
    GmBuffer configGm(sizeof(config));
    GmBuffer countsGm(counts.size() * sizeof(int64_t));
    GmBuffer recordsGm(slots * RECORD_CAPACITY * sizeof(GmmBatchedRecord));
    GmBuffer totalsGm(slots * sizeof(SchedulerTotals));
    ASSERT_NE(configGm.Get(), nullptr);
    ASSERT_NE(countsGm.Get(), nullptr);
    ASSERT_NE(recordsGm.Get(), nullptr);
    ASSERT_NE(totalsGm.Get(), nullptr);
    std::memcpy(configGm.Get(), &config, sizeof(config));
    std::memcpy(countsGm.Get(), counts.data(), counts.size() * sizeof(int64_t));
    std::memset(recordsGm.Get(), SENTINEL_BYTE, slots * RECORD_CAPACITY * sizeof(GmmBatchedRecord));
    std::memset(totalsGm.Get(), 0, slots * sizeof(SchedulerTotals));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(ProbeGmmBatchedView, SMOKE_BLOCK_NUM, configGm.Get(), countsGm.Get(), recordsGm.Get(),
                              totalsGm.Get()));

    std::vector<int64_t> rowGroups(m, -1);
    std::vector<int64_t> groupStarts(counts.size());
    int64_t end = 0;
    int64_t expectedTasks = 0;
    for (size_t group = 0; group < counts.size(); ++group) {
        groupStarts[group] = end;
        std::fill(rowGroups.begin() + end, rowGroups.begin() + end + counts[group], static_cast<int64_t>(group));
        const int64_t groupM = batch == 1 ? counts[group] * (n / k) : counts[group];
        expectedTasks += (groupM + blockSize - 1) / blockSize * batch;
        end += counts[group];
    }
    std::array<std::vector<uint32_t>, 2> coverage;
    for (auto& covered : coverage) {
        covered.resize(m * n, 0);
    }
    const auto* records = reinterpret_cast<const GmmBatchedRecord*>(recordsGm.Get());
    const auto* totals = reinterpret_cast<const SchedulerTotals*>(totalsGm.Get());
    int64_t cubeTasks = 0;
    for (size_t lane = 0; lane < slots; ++lane) {
        const size_t role = lane / SMOKE_BLOCK_NUM;
        const size_t core = lane % SMOKE_BLOCK_NUM;
        ASSERT_EQ(totals[lane].valid, 1);
        ASSERT_LT(totals[lane].visited, RECORD_CAPACITY);
        ASSERT_EQ(totals[lane].visited, totals[core].visited);
        if (role == 0) {
            cubeTasks += totals[lane].visited;
        }
        for (int64_t index = 0; index < totals[lane].visited; ++index) {
            const auto& record = records[lane * RECORD_CAPACITY + index];
            const auto& cube = records[core * RECORD_CAPACITY + index];
            EXPECT_EQ(record.group, cube.group);
            EXPECT_EQ(record.task, cube.task);
            EXPECT_EQ(record.consistent, 1);
            ASSERT_GE(record.group, 0);
            ASSERT_LT(static_cast<size_t>(record.group), counts.size());
            EXPECT_EQ(record.inputOffset, groupStarts[record.group] * n);
            EXPECT_EQ(record.rotationOffset, record.group * batch * k * k);
            EXPECT_EQ(record.shape[1], k);
            EXPECT_EQ(record.shape[2], k);
            EXPECT_EQ(record.shape[3], batch);
            EXPECT_EQ(record.coord[1], 0);
            EXPECT_EQ(record.coord[2], 0);
            EXPECT_GE(record.coord[3], 0);
            ASSERT_LT(record.coord[3], batch);
            EXPECT_EQ(record.shape, cube.shape);
            EXPECT_EQ(record.coord, cube.coord);
            const int64_t groupOrigin = record.inputOffset / (batch * k);
            EXPECT_EQ(record.outputCoord[0], groupOrigin + record.coord[0]);
            EXPECT_EQ(record.outputCoord[3], record.coord[3]);
            const int64_t perAiv = splitM ? (record.shape[0] + 1) / 2 : record.shape[0];
            const int64_t start = role == 0 ? 0 : (role - 1) * perAiv;
            const int64_t rows = role == 0 ? record.shape[0] :
                                             (splitM || role == 1 ?
                                                  std::max<int64_t>(0, std::min(perAiv, record.shape[0] - start)) :
                                                  0);
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t col = 0; col < k; ++col) {
                    const int64_t base = role == 0 ? record.inputOffset : 0;
                    const int64_t origin = role == 0 ? record.coord[0] : record.outputCoord[0];
                    const int64_t flat = base + ((origin + start + row) * batch + record.coord[3]) * k + col;
                    ASSERT_GE(flat, 0);
                    ASSERT_LT(flat, m * n);
                    EXPECT_EQ(rowGroups[flat / n], record.group);
                    ++coverage[role == 0 ? 0 : 1][flat];
                }
            }
        }
    }
    EXPECT_EQ(cubeTasks, expectedTasks);
    for (int64_t flat = 0; flat < m * n; ++flat) {
        const uint32_t expected = rowGroups[flat / n] >= 0 ? 1 : 0;
        EXPECT_EQ(coverage[0][flat], expected);
        EXPECT_EQ(coverage[1][flat], expected);
    }
}

__global__ __aicore__ void ProbeGmmCoordinateWidth(GM_ADDR resultsGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    Scheduler::Params params{};
    params.baseM = 1;
    params.baseN = 32;
    params.groupType = 0;
    params.groupNum = 1;
    params.singleX = params.singleWeight = params.singleY = true;
    Scheduler scheduler(params);
    constexpr int64_t large = (int64_t{1} << 31) + 7;
    scheduler.UpdateNextGroup({large + 3, 32, 32, 1});
    const auto largeCoord = scheduler.GetBlockCoord(large);
    const auto unsignedCoord = scheduler.GetBlockCoord(7U); // Widens to the same int64_t entry.
    int64_t core = AscendC::GetBlockIdx();
    int64_t role = 0;
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
        role = 1 + AscendC::GetSubBlockIdx();
    }
    auto& result = reinterpret_cast<GmmIndexProbe*>(resultsGm)[role * AscendC::GetBlockNum() + core];
    result = {asc::te::get<0>(largeCoord), asc::te::get<0>(unsignedCoord), scheduler.GetBlockNums()};
}

template <typename Scheduler>
__aicore__ inline typename Scheduler::Params MakeBasicSchedulerParams(const SchedulerCaseConfig& config)
{
    typename Scheduler::Params params;
    params.mL1 = config.blockSize;
    params.nL1 = params.kL1 = config.k;
    params.mBaseTailSplitCnt = params.nBaseTailSplitCnt = 1;
    params.mTailCnt = params.nTailCnt = 1;
    return params;
}

__global__ __aicore__ void CompareOrdinarySchedulersKernel(GM_ADDR configGm, GM_ADDR totalsGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    int64_t coreIdx = AscendC::GetBlockIdx();
    int64_t roleIdx = 0;
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetSubBlockNum();
        roleIdx = 1 + AscendC::GetSubBlockIdx();
    }
    const int64_t cores = AscendC::GetBlockNum();
    const auto& config = *reinterpret_cast<const SchedulerCaseConfig*>(configGm);
    const int64_t effectiveM = config.b == 1 ? config.m * (config.n / config.k) : config.m;
    const int64_t tilesPerBatch = (effectiveM + config.blockSize - 1) / config.blockSize;
    const int64_t expectedTaskNums = tilesPerBatch * config.b;
    const int64_t expectedCoreNums = AscendC::Std::min(expectedTaskNums, cores);
    using InterleavedScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<RotateQuantUT::TbmmProblemShape, 0,
                                                                               false, true, true>;
    using DefaultScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<RotateQuantUT::TbmmProblemShape>;
    const RotateQuantUT::TbmmProblemShape problem{effectiveM, config.k, config.k, config.b, 1};
    InterleavedScheduler interleaved(problem, MakeBasicSchedulerParams<InterleavedScheduler>(config));
    DefaultScheduler defaultScheduler(problem, MakeBasicSchedulerParams<DefaultScheduler>(config));
    auto* totals = reinterpret_cast<OrdinarySchedulerComparison*>(totalsGm) + roleIdx * cores + coreIdx;
    *totals = {expectedTaskNums,
               interleaved.GetBlockNums(),
               defaultScheduler.GetBlockNums(),
               expectedCoreNums,
               interleaved.GetCoreNums(),
               defaultScheduler.GetCoreNums(),
               0,
               0,
               0};
    for (int64_t p = coreIdx; p < expectedTaskNums; p += cores) {
        const auto shape = interleaved.GetBlockShape<false, half>(p);
        const auto coord = interleaved.GetBlockCoord(p);
        const int64_t interleavedBatch = p % config.b;
        const int64_t interleavedM = (p / config.b) * config.blockSize;
        const int64_t interleavedRows = AscendC::Std::min(config.blockSize, effectiveM - interleavedM);
        totals->interleavedMismatches += asc::te::get<0>(coord) != interleavedM || asc::te::get<1>(coord) != 0 ||
                                         asc::te::get<2>(coord) != 0 || asc::te::get<3>(coord) != interleavedBatch ||
                                         asc::te::get<0>(shape) != interleavedRows ||
                                         asc::te::get<1>(shape) != config.k || asc::te::get<2>(shape) != config.k ||
                                         asc::te::get<3>(shape) != config.b;
        const auto defaultShape = defaultScheduler.GetBlockShape<false, half>(p);
        const auto defaultCoord = defaultScheduler.GetBlockCoord(p);
        const int64_t expectedM = (p % tilesPerBatch) * config.blockSize;
        const int64_t expectedRows = AscendC::Std::min(config.blockSize, effectiveM - expectedM);
        totals->defaultMismatches += asc::te::get<0>(defaultCoord) != expectedM || asc::te::get<1>(defaultCoord) != 0 ||
                                     asc::te::get<2>(defaultCoord) != 0 ||
                                     asc::te::get<3>(defaultCoord) != p / tilesPerBatch ||
                                     asc::te::get<0>(defaultShape) != expectedRows ||
                                     asc::te::get<1>(defaultShape) != config.k ||
                                     asc::te::get<2>(defaultShape) != config.k ||
                                     asc::te::get<3>(defaultShape) != config.b;
        ++totals->compared;
    }
}

void CheckOrdinarySchedulers(int64_t m, int64_t n, int64_t k, int64_t b, int64_t blockSize, uint32_t cores)
{
    const SchedulerCaseConfig config{m, n, k, blockSize, 0, GROUP_LIST_COUNT, b};
    const int64_t effectiveM = b == 1 ? m * (n / k) : m;
    const int64_t tasks = (effectiveM + blockSize - 1) / blockSize * b;
    const size_t slots = CORE_ROLE_NUM * cores;
    GmBuffer configGm(sizeof(config));
    GmBuffer totalsGm(slots * sizeof(OrdinarySchedulerComparison));
    ASSERT_NE(configGm.Get(), nullptr);
    ASSERT_NE(totalsGm.Get(), nullptr);
    std::memcpy(configGm.Get(), &config, sizeof(config));
    std::memset(totalsGm.Get(), SENTINEL_BYTE, slots * sizeof(OrdinarySchedulerComparison));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(CompareOrdinarySchedulersKernel, cores, configGm.Get(), totalsGm.Get()));
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto& actual = reinterpret_cast<const OrdinarySchedulerComparison*>(totalsGm.Get())[slot];
        const int64_t core = slot % cores;
        EXPECT_EQ(actual.expectedTaskNums, tasks);
        EXPECT_EQ(actual.interleavedTaskNums, tasks);
        EXPECT_EQ(actual.defaultTaskNums, tasks);
        EXPECT_EQ(actual.expectedCoreNums, std::min<int64_t>(tasks, cores));
        EXPECT_EQ(actual.interleavedCoreNums, actual.expectedCoreNums);
        EXPECT_EQ(actual.defaultCoreNums, actual.expectedCoreNums);
        EXPECT_EQ(actual.compared, core < tasks ? (tasks - core + cores - 1) / cores : 0);
        EXPECT_EQ(actual.interleavedMismatches, 0);
        EXPECT_EQ(actual.defaultMismatches, 0);
    }
}

} // namespace

TEST_F(RotateQuantBlazeTest, AssembleFp16Outputs)
{
    CheckPublicAssembly<half, fp4x2_e2m1_t, false>();
    CheckPublicAssembly<half, fp8_e4m3fn_t, false>();
    CheckPublicAssembly<half, fp8_e5m2_t, false>();
    CheckPublicAssembly<half, fp4x2_e2m1_t, true>();
    CheckPublicAssembly<half, fp8_e4m3fn_t, true>();
    CheckPublicAssembly<half, fp8_e5m2_t, true>();
}

TEST_F(RotateQuantBlazeTest, AssembleBf16Outputs)
{
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, false>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, false>();
    CheckPublicAssembly<bfloat16_t, fp8_e5m2_t, false>();
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, true>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, true>();
    CheckPublicAssembly<bfloat16_t, fp8_e5m2_t, true>();
}

TEST(RotateQuantSingleAivTest, AssembleSingleDestinationOutputs)
{
    CheckPublicAssembly<half, fp4x2_e2m1_t, false, false>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, false, false>();
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, true, false, 2>();
}

TEST(RotateQuantSingleAivTest, KernelSmokeOrdinaryAndGrouped)
{
    bool singleAiv = false;
    ASSERT_TRUE(ProbeSingleAivLaunch(singleAiv));
    if (!singleAiv) {
        GTEST_SKIP() << "CPU runner does not honor MIX_AIC_1_1";
    }
    RunKernelSmoke<half, fp4x2_e2m1_t, false, false, 1, true>({33, 96, 32, 3});
    RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, true, false, 2, true>({20, 96, 32, 1}, {1, 0, 19});
}

#if defined(ROTATE_QUANT_UT_REAL_EPILOGUE)
TEST_F(RotateQuantBlazeTest, MxFp4WritesOutputAndScaleWithSplitM)
{
    CheckMxFp4Output<false, true>();
    CheckMxFp4Output<true, true>();
}

TEST(RotateQuantSingleAivTest, MxFp4WritesOutputAndScale)
{
    bool singleAiv = false;
    ASSERT_TRUE(ProbeSingleAivLaunch(singleAiv));
    if (!singleAiv) {
        GTEST_SKIP() << "CPU runner does not honor MIX_AIC_1_1";
    }
    CheckMxFp4Output<false, false>();
    CheckMxFp4Output<true, false>();
}
#endif

TEST_F(RotateQuantBlazeTest, AssembleTwoStageEpilogue)
{
    CheckPublicAssembly<half, fp4x2_e2m1_t, false, true, 2>();
    CheckPublicAssembly<bfloat16_t, fp8_e4m3fn_t, false, false, 2>();
    CheckPublicAssembly<half, fp4x2_e2m1_t, true, true, 2>();
    CheckPublicAssembly<bfloat16_t, fp8_e5m2_t, true, false, 2>();
}

TEST_F(RotateQuantBlazeTest, KernelSmokeSharedRotation)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false>({17, 96, 32, 1});
    RunKernelSmoke<half, fp8_e4m3fn_t, false>({3, 32, 32, 1});
}

TEST_F(RotateQuantBlazeTest, KernelSmokePerNRotation) { RunKernelSmoke<half, fp8_e5m2_t, false>({17, 96, 32, 3}); }

TEST_F(RotateQuantBlazeTest, KernelSmokeSplitMTailsAndIdleCores)
{
    // One global task leaves the second AIC group idle. The sole row must use
    // only AIV0, even though this entry has a 1:2 launch contract.
    RunKernelSmoke<half, fp8_e4m3fn_t, false>({1, 32, 32, 1});
    // Both AIVs participate in the handshake even when the tail has only one row.
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false>({33, 96, 32, 3});
    // A 1:2 launch is also allowed to request one output consumer explicitly.
    RunKernelSmoke<half, fp8_e5m2_t, false, false>({33, 96, 32, 3});
}

TEST_F(RotateQuantBlazeTest, KernelSmokeTwoStageEpilogueDrainAndWraparound)
{
    // Force two slots even with one tile to exercise shutdown after only slot zero is used.
    RunKernelSmoke<half, fp4x2_e2m1_t, false, true, 2>({1, 32, 32, 1});
    RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, false, true, 2>({3, 32, 32, 1});
    // Full aligned tiles wrap both slots before the final one-row tiles.
    RunKernelSmoke<half, fp8_e5m2_t, false, true, 2>({91, 96, 32, 1});
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false, true, 2>({81, 96, 32, 3});
    RunKernelSmoke<half, fp8_e4m3fn_t, false, false, 2>({81, 96, 32, 3});
}

TEST_F(RotateQuantBlazeTest, SharedRotationPreservesBf16IntermediateAcrossTiles)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half>({33, 96, 32, 1});
    CheckBf16Intermediate<half>({33, 192, 64, 1});
    CheckBf16Intermediate<bfloat16_t>({33, 384, 128, 1});
    CheckBf16Intermediate<bfloat16_t>({33, 512, 256, 1});
}

TEST_F(RotateQuantBlazeTest, PerSegmentRotationPreservesBf16IntermediateAndTail)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half>({33, 96, 32, 3});
    CheckBf16Intermediate<half>({33, 192, 64, 3});
    CheckBf16Intermediate<bfloat16_t>({33, 384, 128, 3});
    CheckBf16Intermediate<bfloat16_t>({33, 768, 256, 3});
}

TEST_F(RotateQuantBlazeTest, SplitMPreservesOddAndSingleRowBf16Results)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half>({1, 32, 32, 1});
    for (const int64_t rows : {2, 3, 15, 16, 17}) {
        CheckBf16Intermediate<half>({rows, 96, 32, 3});
    }
    CheckBf16Intermediate<bfloat16_t>({33, 768, 256, 3});
    CheckBf16Intermediate<half, false>({33, 96, 32, 3});
}

TEST(RotateQuantSingleAivTest, SingleAivLaunchPreservesBf16Intermediate)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    bool supportsSingleAiv = false;
    ASSERT_TRUE(ProbeSingleAivLaunch(supportsSingleAiv));
    if (!supportsSingleAiv) {
        GTEST_SKIP() << "CPU runner does not honor MIX_AIC_1_1; one-AIV execution was not verified";
    }
    CheckBf16Intermediate<half, false, true>({1, 32, 32, 1});
    CheckBf16Intermediate<half, false, true>({33, 96, 32, 3});
    CheckBf16Intermediate<bfloat16_t, false, true>({33, 768, 256, 3});
}

TEST_F(RotateQuantBlazeTest, TwoStagePipelinePreservesBf16IntermediateAndTail)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half, true, false, 2>({1, 32, 32, 1});
    CheckBf16Intermediate<half, true, false, 2>({3, 32, 32, 1});
    CheckBf16Intermediate<half, true, false, 2>({91, 96, 32, 1});
    CheckBf16Intermediate<bfloat16_t, true, false, 2>({81, 768, 256, 3});
    CheckBf16Intermediate<half, false, false, 2>({81, 96, 32, 3});
}

TEST(RotateQuantSingleAivTest, TwoStageSingleAivLaunchPreservesBf16Intermediate)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    bool supportsSingleAiv = false;
    ASSERT_TRUE(ProbeSingleAivLaunch(supportsSingleAiv));
    if (!supportsSingleAiv) {
        GTEST_SKIP() << "CPU runner does not honor MIX_AIC_1_1; two-stage one-AIV execution was not verified";
    }
    CheckBf16Intermediate<half, false, true, 2>({1, 32, 32, 1});
    CheckBf16Intermediate<half, false, true, 2>({91, 96, 32, 1});
    CheckBf16Intermediate<bfloat16_t, false, true, 2>({81, 768, 256, 3});
}

TEST_F(RotateQuantBlazeTest, SingleEpiloguePreservesWorkspaceAfterInactiveAivTasks)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half, true, false, 2>({33, 96, 32, 3}, {}, 1);
    for (const auto type : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        // The first tile gives AIV1 no rows. It still must advance the handshake
        // so its first tile with rows uses workspace 1.
        const auto groups = MakeGroupList({1, 0, 2, 1, 3, 2}, type);
        TilingData shared{9, 32, 32, 1};
        shared.groupListType = type;
        CheckBf16Intermediate<half, true, false, 2, true>(shared, groups, 1);
        TilingData segmented{9, 96, 32, 3};
        segmented.groupListType = type;
        CheckBf16Intermediate<bfloat16_t, true, false, 2, true>(segmented, groups, 1);
    }
}

TEST_F(RotateQuantBlazeTest, BasicSchedulerPreservesOrdinaryTasksOnAllLanes)
{
    // Representative ordinary Host tilings, including reduced launches and the K=256 tail-wave case.
    CheckOrdinarySchedulers(1, 96, 32, 1, 16, 1);
    CheckOrdinarySchedulers(17, 384, 128, 3, 16, 6);
    CheckOrdinarySchedulers(17, 8448, 256, 33, 64, 32);
    CheckOrdinarySchedulers(65, 8192, 256, 32, 64, 32);
    CheckOrdinarySchedulers(65537, 2080, 32, 1, 512, 32);
}

TEST_F(RotateQuantBlazeTest, GmmSchedulerPreservesInt64AndUnsignedCoordinateCalls)
{
    constexpr size_t slots = CORE_ROLE_NUM * SMOKE_BLOCK_NUM;
    GmBuffer results(sizeof(GmmIndexProbe) * slots);
    ASSERT_NE(results.Get(), nullptr);
    std::memset(results.Get(), SENTINEL_BYTE, sizeof(GmmIndexProbe) * slots);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(ProbeGmmCoordinateWidth, SMOKE_BLOCK_NUM, results.Get()));
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto& result = reinterpret_cast<const GmmIndexProbe*>(results.Get())[slot];
        EXPECT_EQ(result.largeM, (int64_t{1} << 31) + 7);
        EXPECT_EQ(result.unsignedM, 7);
        EXPECT_EQ(result.tasks, (int64_t{1} << 31) + 10);
    }
}

TEST_F(RotateQuantBlazeTest, GmmBatchedViewCoversEachElementOnceAcrossRoles)
{
    for (const bool splitM : {false, true}) {
        CheckGmmBatchedView({0, 17, 0, 2, 1, 18, 0}, 96, 32, 1, 17, splitM);
        CheckGmmBatchedView({0, 17, 0, 2, 1, 18, 0}, 96, 32, 3, 17, splitM);
        CheckGmmBatchedView({0, 0, 0}, 64, 32, 2, 17, splitM);
    }
}

TEST_F(RotateQuantBlazeTest, KernelSmokeGroupedSharedRotation)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData tiling{20, 96, 32, 1};
        tiling.groupListType = groupListType;
        tiling.hasAlpha = 1;
        RunKernelSmoke<half, fp4x2_e2m1_t, true>(tiling, {1, 0, 17});
    }
}

TEST_F(RotateQuantBlazeTest, KernelSmokeGroupedPerNRotation)
{
    for (int64_t groupListType : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData tiling{20, 96, 32, 3};
        tiling.groupListType = groupListType;
        RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, true>(tiling, {1, 0, 17});
    }
}

TEST_F(RotateQuantBlazeTest, KernelSmokeK256)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, false>({33, 512, 256, 1});
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, true>({33, 512, 256, 1}, {7, 0, 26});
    RunKernelSmoke<half, fp4x2_e2m1_t, true>({33, 512, 256, 2}, {7, 0, 26});
}

TEST_F(RotateQuantBlazeTest, KernelSmokeAllEmptyGroups)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, true>({7, 64, 32, 1}, {0, 0, 0});
}

TEST_F(RotateQuantBlazeTest, GmmGroupedPipelineSmokeDrainsAcrossGroups)
{
    for (const auto type : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData single{1, 32, 32, 1};
        single.groupListType = type;
        RunKernelSmoke<half, fp4x2_e2m1_t, true, true, 1>(single, {0, 1, 0});
        RunKernelSmoke<half, fp4x2_e2m1_t, true, true, 2>(single, {0, 1, 0});
        TilingData twoRows{3, 32, 32, 1};
        twoRows.groupListType = type;
        RunKernelSmoke<half, fp8_e5m2_t, true, true, 2>(twoRows, {0, 2, 1, 0});
        TilingData tails{80, 32, 32, 1, 32};
        tails.groupListType = type;
        RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, true, true, 1>(tails, {0, 17, 17, 0, 1, 17, 17, 0});
        RunKernelSmoke<bfloat16_t, fp8_e4m3fn_t, true, true, 2>(tails, {0, 17, 17, 0, 1, 17, 17, 0});
    }
    RunKernelSmoke<half, fp8_e5m2_t, true, true, 2>({64, 96, 32, 3}, {6, 0, 1, 11});
    RunKernelSmoke<half, fp8_e5m2_t, true, false, 2>({64, 96, 32, 3}, {6, 0, 1, 11});
    RunKernelSmoke<half, fp4x2_e2m1_t, true, true, 2>({7, 64, 32, 1}, {0, 0, 0});
    // A malformed later count must break the group loop and drain its valid prefix.
    RunKernelSmoke<half, fp8_e5m2_t, true, true, 2>({64, 96, 32, 3, 32}, {17, 0, -1, 17});
}

TEST_F(RotateQuantBlazeTest, GmmGroupedBf16MatchesIndependentGroupReference)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    for (const auto type : {GROUP_LIST_CUMSUM, GROUP_LIST_COUNT}) {
        TilingData shared{80, 32, 32, 1, 32};
        shared.groupListType = type;
        const auto groups = MakeGroupList({0, 17, 17, 0, 1, 17, 17, 0}, type);
        CheckBf16Intermediate<half, true, false, 1, true>(shared, groups, 1);
        CheckBf16Intermediate<bfloat16_t, true, false, 2, true>(shared, groups, 1);
        TilingData segmented{64, 96, 32, 3, 32};
        segmented.groupListType = type;
        CheckBf16Intermediate<half, true, false, 2, true>(segmented, MakeGroupList({0, 1, 17, 0, 2, 33, 0}, type));
    }
    CheckBf16Intermediate<half, true, false, 2, true>({24, 96, 32, 1}, {6, 0, 1, 11});
    CheckBf16Intermediate<bfloat16_t, true, false, 2, true>({80, 768, 256, 3, 32}, {1, 17, 0, 33, 1});
    CheckBf16Intermediate<half, false, false, 2, true>({64, 96, 32, 3, 32}, {0, 1, 17, 0, 2, 33, 0});
    CheckBf16Intermediate<half, true, false, 2, true>({8, 64, 32, 1}, {0, 0, 0});
}

TEST_F(RotateQuantBlazeTest, GmmGroupedInvalidValuesFinishOnlyTheValidPrefix)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    CheckBf16Intermediate<half, true, false, 1, true>({80, 32, 32, 1, 32}, {17, 0, -1, 17}, 1);
    CheckBf16Intermediate<half, true, false, 2, true>({80, 96, 32, 3, 32}, {17, 0, -1, 17}, 1);
    CheckBf16Intermediate<half, true, false, 2, true>({80, 32, 32, 1, 32}, {-1, 17});
    CheckBf16Intermediate<half, true, false, 2, true>({80, 32, 32, 1, 32}, {17, std::numeric_limits<int64_t>::max()});
    TilingData cumulative{80, 32, 32, 1, 32};
    cumulative.groupListType = GROUP_LIST_CUMSUM;
    CheckBf16Intermediate<half, true, false, 2, true>(cumulative, {17, 17, 16, 60});
    CheckBf16Intermediate<half, true, false, 2, true>(cumulative, {80, 81});
    CheckBf16Intermediate<half, true, false, 2, true>(cumulative, {17, std::numeric_limits<int64_t>::min()});
}

TEST(RotateQuantSingleAivTest, GmmGroupedActualSingleAivUsesTheSameBf16Contract)
{
    if (!VerifyBf16IntermediateEnabled()) {
        GTEST_SKIP() << "Set ROTATE_QUANT_UT_VERIFY_INTERMEDIATE=1 on a simulator that commits Cube/Fixpipe results";
    }
    bool singleAiv = false;
    ASSERT_TRUE(ProbeSingleAivLaunch(singleAiv));
    if (!singleAiv) {
        GTEST_SKIP() << "CPU runner does not honor MIX_AIC_1_1; grouped one-AIV execution was not verified";
    }
    CheckBf16Intermediate<half, false, true, 1, true>({24, 96, 32, 1}, {6, 0, 1, 11});
    CheckBf16Intermediate<bfloat16_t, false, true, 2, true>({64, 96, 32, 3, 32}, {0, 1, 17, 0, 2, 33, 0});
    CheckBf16Intermediate<half, false, true, 2, true>({129, 192, 64, 1, 32}, {0, 1, 65, 0, 62});
    CheckBf16Intermediate<bfloat16_t, false, true, 1, true>({67, 384, 128, 3, 32}, {1, 0, 17, 33, 0, 16});
    CheckBf16Intermediate<half, false, true, 2, true>({131, 768, 256, 3, 32}, {0, 65, 1, 0, 64});
}
