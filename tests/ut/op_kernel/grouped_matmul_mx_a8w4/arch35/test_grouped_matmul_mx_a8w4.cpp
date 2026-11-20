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
 * \file test_grouped_matmul_mx_a8w4.cpp
 * \brief tikicpulib smoke tests for grouped MX A8W4 public kernel dispatch.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "grouped_matmul_mx_a8w4.h"

namespace {

constexpr uint64_t FP4_PACK_FACTOR = 2U;
constexpr uint8_t MX_IDENTITY_SCALE = 0x7FU;
constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2U;
constexpr size_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2U;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0U ? value : (value + alignment - 1U) / alignment * alignment;
}

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : size_(std::max<size_t>(size, 1U))
    {
        address_ = reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size_));
    }

    ~GmBuffer()
    {
        if (address_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(address_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return address_; }

    void Fill(uint8_t value) const { std::fill_n(reinterpret_cast<uint8_t*>(address_), size_, value); }

    template <typename T_>
    void CopyFrom(const std::vector<T_>& values) const
    {
        ASSERT_LE(values.size() * sizeof(T_), size_);
        std::memcpy(address_, values.data(), values.size() * sizeof(T_));
    }

private:
    GM_ADDR address_{nullptr};
    size_t size_{0U};
};

struct CaseConfig {
    std::vector<int64_t> groupList;
    int64_t totalM;
    uint64_t n;
    uint64_t k;
    uint16_t baseM;
    uint32_t groupListType;
    uint8_t singleW;
    uint32_t hasBias;
    uint32_t mainBlockSize;
    uint64_t mainBlockCount;
    uint16_t firstTailBlockSize;
    uint16_t firstTailBlockCount;
    uint16_t secondTailBlockSize;
    uint16_t secondTailBlockCount;
};

size_t GetGroupNum(const CaseConfig& config)
{
    return config.groupList.size() /
           (config.groupListType == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1U);
}

class GroupedCaseBuffers {
public:
    GroupedCaseBuffers(const CaseConfig& config, size_t outputElementSize)
        : groupNum_(GetGroupNum(config)),
          scaleK_(AlignUp(config.k, 64U) / 32U),
          weightGroupSize_(static_cast<size_t>(AlignUp(config.k, 32U) * AlignUp(config.n, 16U) / FP4_PACK_FACTOR)),
          scaleBGroupSize_(static_cast<size_t>(config.n) * scaleK_),
          biasGroupSize_(static_cast<size_t>(config.n) * outputElementSize),
          a_(static_cast<size_t>(config.totalM) * config.k),
          scaleA_(static_cast<size_t>(config.totalM) * scaleK_),
          c_(static_cast<size_t>(config.totalM) * config.n * outputElementSize),
          groupList_(config.groupList.size() * sizeof(int64_t)),
          tiling_(sizeof(GroupedMatmulMxA8W4TilingData)),
          b_(config.singleW == 1U ? groupNum_ * weightGroupSize_ : (groupNum_ + 3U) * sizeof(uint64_t)),
          scaleB_(config.singleW == 1U ? groupNum_ * scaleBGroupSize_ : (groupNum_ + 3U) * sizeof(uint64_t)),
          bias_(config.singleW == 1U ? groupNum_ * biasGroupSize_ : (groupNum_ + 3U) * sizeof(uint64_t))
    {
        a_.Fill(0U);
        scaleA_.Fill(MX_IDENTITY_SCALE);
        c_.Fill(0xA5U);
        groupList_.CopyFrom(config.groupList);
        if (config.singleW == 1U) {
            b_.Fill(0U);
            scaleB_.Fill(MX_IDENTITY_SCALE);
            bias_.Fill(0U);
        } else {
            PrepareTensorLists();
        }
    }

    GM_ADDR A() const { return a_.Get(); }
    GM_ADDR B() const { return b_.Get(); }
    GM_ADDR Bias() const { return bias_.Get(); }
    GM_ADDR ScaleA() const { return scaleA_.Get(); }
    GM_ADDR ScaleB() const { return scaleB_.Get(); }
    GM_ADDR C() const { return c_.Get(); }
    GM_ADDR GroupList() const { return groupList_.Get(); }
    GM_ADDR Tiling() const { return tiling_.Get(); }

private:
    void PrepareTensorLists()
    {
        constexpr uint64_t DATA_POINTER_OFFSET = 3U * sizeof(uint64_t);
        constexpr uint64_t SHAPE_SENTINEL = 0xFFFFFFFFULL;
        const uint64_t countAndDimension = static_cast<uint64_t>(groupNum_) << 32U;
        std::vector<uint64_t> weightList(groupNum_ + 3U, 0U);
        std::vector<uint64_t> scaleList(groupNum_ + 3U, 0U);
        std::vector<uint64_t> biasList(groupNum_ + 3U, 0U);
        for (auto* tensorList : {&weightList, &scaleList, &biasList}) {
            (*tensorList)[0] = DATA_POINTER_OFFSET;
            (*tensorList)[1] = countAndDimension;
            (*tensorList)[2] = SHAPE_SENTINEL;
        }
        for (size_t group = 0U; group < groupNum_; ++group) {
            weights_.emplace_back(std::make_unique<GmBuffer>(weightGroupSize_));
            scales_.emplace_back(std::make_unique<GmBuffer>(scaleBGroupSize_));
            biases_.emplace_back(std::make_unique<GmBuffer>(biasGroupSize_));
            weights_.back()->Fill(0U);
            scales_.back()->Fill(MX_IDENTITY_SCALE);
            biases_.back()->Fill(0U);
            weightList[group + 3U] = reinterpret_cast<uint64_t>(weights_.back()->Get());
            scaleList[group + 3U] = reinterpret_cast<uint64_t>(scales_.back()->Get());
            biasList[group + 3U] = reinterpret_cast<uint64_t>(biases_.back()->Get());
        }
        b_.CopyFrom(weightList);
        scaleB_.CopyFrom(scaleList);
        bias_.CopyFrom(biasList);
    }

    size_t groupNum_{0U};
    size_t scaleK_{0U};
    size_t weightGroupSize_{0U};
    size_t scaleBGroupSize_{0U};
    size_t biasGroupSize_{0U};
    GmBuffer a_;
    GmBuffer scaleA_;
    GmBuffer c_;
    GmBuffer groupList_;
    GmBuffer tiling_;
    GmBuffer b_;
    GmBuffer scaleB_;
    GmBuffer bias_;
    std::vector<std::unique_ptr<GmBuffer>> weights_;
    std::vector<std::unique_ptr<GmBuffer>> scales_;
    std::vector<std::unique_ptr<GmBuffer>> biases_;
};

GroupedMatmulMxA8W4TilingData BuildTiling(const CaseConfig& config)
{
    return GroupedMatmulMxA8W4TilingData{
        static_cast<uint32_t>(GetGroupNum(config)),
        1U,
        config.k,
        config.n,
        static_cast<uint8_t>(config.mainBlockCount + config.firstTailBlockCount + config.secondTailBlockCount),
        config.mainBlockSize,
        config.mainBlockCount,
        config.firstTailBlockSize,
        config.secondTailBlockSize,
        config.firstTailBlockCount,
        config.secondTailBlockCount,
        config.baseM,
        config.groupListType,
        config.hasBias};
}

template <typename WeightType_, typename OutputType_, bool IsSingleMultiSingle_>
void RunCase(const CaseConfig& config)
{
    ASSERT_EQ(IsSingleMultiSingle_, config.singleW == 0U);
    GroupedCaseBuffers buffers(config, sizeof(OutputType_));
    ASSERT_NE(buffers.A(), nullptr);
    ASSERT_NE(buffers.B(), nullptr);
    ASSERT_NE(buffers.ScaleA(), nullptr);
    ASSERT_NE(buffers.ScaleB(), nullptr);
    ASSERT_NE(buffers.C(), nullptr);
    ASSERT_NE(buffers.GroupList(), nullptr);

    *reinterpret_cast<GroupedMatmulMxA8W4TilingData*>(buffers.Tiling()) = BuildTiling(config);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // tikicpulib does not model this kernel's FP8 x FP4 MMAD/Fixpipe values. These tests only exercise the public
    // template instantiations, tiling mapping, continuous/TensorList addressing, and mixed-kernel control flow.
    auto kernel = GroupedMatmulMxA8W4KernelEntry<WeightType_, OutputType_, IsSingleMultiSingle_>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1U, buffers.A(), buffers.B(), config.hasBias == 1U ? buffers.Bias() : nullptr,
                              buffers.ScaleA(), buffers.ScaleB(), buffers.C(), buffers.GroupList(), buffers.Tiling()))
        << "Grouped MX A8W4 mixed-kernel smoke execution failed";
}

TEST(GroupedMatmulMxA8W4KernelTest, Weight8BitZnToZnUbLayoutKeepsPhysicalFractalTail)
{
    constexpr uint64_t INNER_STRIDE = 1024U;
    const auto layoutN8 = Blaze::Gemm::Weight8BitZnToZnUBLayout<fp8_e4m3fn_t>{}(32, 8, INNER_STRIDE);
    const auto layoutN9 = Blaze::Gemm::Weight8BitZnToZnUBLayout<fp8_e4m3fn_t>{}(32, 9, INNER_STRIDE);
    const auto layoutN16 = Blaze::Gemm::Weight8BitZnToZnUBLayout<fp8_e4m3fn_t>{}(32, 16, INNER_STRIDE);
    const auto groupedLayoutN8 = Blaze::Gemm::Weight8BitZnToZnUBLayout<fp8_e4m3fn_t>{}(
        32, static_cast<int64_t>(Blaze::Gemm::Align16(8U)), INNER_STRIDE);
    const auto groupedLayoutN9 = Blaze::Gemm::Weight8BitZnToZnUBLayout<fp8_e4m3fn_t>{}(
        32, static_cast<int64_t>(Blaze::Gemm::Align16(9U)), INNER_STRIDE);

    EXPECT_EQ(static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(layoutN8.shape()))), 1);
    EXPECT_EQ(static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(layoutN9.shape()))), 2);
    EXPECT_EQ(static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(layoutN16.shape()))), 2);
    EXPECT_EQ(static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(groupedLayoutN8.shape()))), 2);
    EXPECT_EQ(static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(groupedLayoutN9.shape()))), 2);
}

TEST(GroupedMatmulMxA8W4KernelTest, E2M1OffsetContiguousBiasBf16)
{
    const CaseConfig config{{16, 40}, 40, 64, 128, 32, 0, 1, 1, 64, 1, 0, 0, 0, 0};
    RunCase<fp4x2_e2m1_t, bfloat16_t, false>(config);
}

TEST(GroupedMatmulMxA8W4KernelTest, E2M1CountTensorListNoBiasFp16)
{
    const CaseConfig config{{0, 24, 0}, 24, 64, 128, 32, 1, 0, 0, 64, 1, 0, 0, 0, 0};
    RunCase<fp4x2_e2m1_t, half, true>(config);
}

TEST(GroupedMatmulMxA8W4KernelTest, E1M2CountTensorListBiasFp16)
{
    const CaseConfig config{{8, 0, 24}, 32, 64, 128, 32, 1, 0, 1, 64, 1, 0, 0, 0, 0};
    RunCase<fp4x2_e1m2_t, half, true>(config);
}

TEST(GroupedMatmulMxA8W4KernelTest, E1M2OffsetContiguousNoBiasBf16WithThreeNSegments)
{
    const CaseConfig config{{16, 32}, 32, 448, 128, 32, 0, 1, 0, 256, 1, 128, 1, 64, 1};
    RunCase<fp4x2_e1m2_t, bfloat16_t, false>(config);
}

TEST(GroupedMatmulMxA8W4KernelTest, E2M1SparseContiguousBiasBf16)
{
    const CaseConfig config{
        {1, 7, 3, 10, 0, 0, 2, 0}, 17, 64, 128, 32, GROUP_LIST_TYPE_SPARSE, 1, 1, 64, 1, 0, 0, 0, 0};
    RunCase<fp4x2_e2m1_t, bfloat16_t, false>(config);
}

} // namespace
