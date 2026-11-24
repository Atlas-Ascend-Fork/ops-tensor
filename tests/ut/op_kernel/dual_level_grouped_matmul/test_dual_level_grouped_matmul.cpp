/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "dual_level_grouped_matmul_ut.h"

namespace {
constexpr int64_t EXPERTS = DualLevelGmmUt::EXPERT_COUNT;
constexpr int64_t MX_SCALE_K = 64;
constexpr int64_t SCALE_PAIR = 2;
constexpr int64_t NZ_ROWS = 16;
constexpr size_t GUARD_BYTES = 64;
constexpr uint8_t GUARD_VALUE = 0xA5;
constexpr uint8_t FP4_ONE_PAIR = 0x22;
constexpr uint8_t MX_SCALE_ONE = 0x7F;

int64_t CeilDiv(int64_t value, int64_t alignment) { return (value + alignment - 1) / alignment; }

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
    GM_ADDR ptr_{nullptr};
};

struct CaseBuffers {
    explicit CaseBuffers(const DualLevelGmmUt::Case& c)
        : levels(CeilDiv(c.k, c.level0Size)),
          scaleK(CeilDiv(c.k, MX_SCALE_K) * SCALE_PAIR),
          xBytes(c.m * c.k / SCALE_PAIR),
          weightBytes(EXPERTS * CeilDiv(c.transpose ? c.n : c.k, NZ_ROWS) * NZ_ROWS *
                      CeilDiv(c.transpose ? c.k : c.n, MX_SCALE_K) * MX_SCALE_K / SCALE_PAIR),
          yBytes(c.m * c.n * sizeof(half)),
          x(xBytes),
          a0(c.m * levels * sizeof(float)),
          a1(c.m * scaleK),
          weight(weightBytes),
          b0(EXPERTS * levels * c.n * sizeof(float)),
          b1(EXPERTS * c.n * scaleK),
          groups(EXPERTS * (c.groupListType == 2 ? 2 : 1) * sizeof(int64_t)),
          bias(EXPERTS * c.n * sizeof(float)),
          y(yBytes + GUARD_BYTES * 2)
    {}

    int64_t levels;
    int64_t scaleK;
    size_t xBytes;
    size_t weightBytes;
    size_t yBytes;
    GmBuffer x, a0, a1, weight, b0, b1, groups, bias, y;

    DualLevelGmmUt::Buffers Args(bool withBias) const
    {
        return {x.Get(),
                a0.Get(),
                a1.Get(),
                weight.Get(),
                b0.Get(),
                b1.Get(),
                groups.Get(),
                withBias ? bias.Get() : nullptr,
                y.Get() == nullptr ? nullptr : y.Get() + GUARD_BYTES};
    }
};

void InitBuffers(const DualLevelGmmUt::Case& c, CaseBuffers& b)
{
    const auto args = b.Args(true);
    for (auto ptr : {args.x, args.a0, args.a1, args.weight, args.b0, args.b1, args.groups, args.bias, b.y.Get()}) {
        ASSERT_NE(ptr, nullptr);
    }
    std::memset(args.x, FP4_ONE_PAIR, b.xBytes);
    std::memset(args.weight, FP4_ONE_PAIR, b.weightBytes);
    std::memset(args.a1, MX_SCALE_ONE, c.m * b.scaleK);
    std::memset(args.b1, MX_SCALE_ONE, EXPERTS * c.n * b.scaleK);
    std::fill_n(reinterpret_cast<float*>(args.a0), c.m * b.levels, 0.5F);
    std::fill_n(reinterpret_cast<float*>(args.b0), EXPERTS * b.levels * c.n, 2.0F);
    std::fill_n(reinterpret_cast<float*>(args.bias), EXPERTS * c.n, 0.25F);
    auto* groups = reinterpret_cast<int64_t*>(args.groups);
    int64_t prefix = 0;
    for (int64_t e = 0; e < EXPERTS; ++e) {
        prefix += c.groupRows[e];
        if (c.groupListType != 2) {
            groups[e] = c.groupListType == 0 ? prefix : c.groupRows[e];
        }
    }
    if (c.groupListType == 2) {
        int64_t index = 0;
        for (bool empty : {false, true}) {
            for (int64_t e = 0; e < EXPERTS; ++e) {
                if ((c.groupRows[e] == 0) == empty) {
                    groups[index++] = e;
                    groups[index++] = c.groupRows[e];
                }
            }
        }
    }
    ASSERT_EQ(prefix, c.m);
    std::memset(b.y.Get(), GUARD_VALUE, b.yBytes + GUARD_BYTES * 2);
    std::memset(args.y, 0, b.yBytes);
}

template <class Output, bool Transpose>
void RunKernel(const DualLevelGmmUt::Case& c, CaseBuffers& b)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = DualLevelGmmKernelEntry<Output, Transpose>;
    ASSERT_TRUE(KERNEL_RUN_KF(entry, c.cores, c, b.Args(c.bias))) << c.name;
    for (size_t i = 0; i < GUARD_BYTES; ++i) {
        EXPECT_EQ(b.y.Get()[i], GUARD_VALUE) << c.name << ": output prefix byte=" << i;
        EXPECT_EQ(b.y.Get()[GUARD_BYTES + b.yBytes + i], GUARD_VALUE) << c.name << ": output suffix byte=" << i;
    }
}

class DualLevelGroupedMatmulKernelTest : public testing::TestWithParam<DualLevelGmmUt::Case> {};

TEST_P(DualLevelGroupedMatmulKernelTest, ExecutesWithoutCrashOrOutputOverrun)
{
    const auto& c = GetParam();
    CaseBuffers buffers(c);
    ASSERT_NO_FATAL_FAILURE(InitBuffers(c, buffers));
    if (c.bf16 && c.transpose) {
        RunKernel<bfloat16_t, true>(c, buffers);
    } else if (c.bf16) {
        RunKernel<bfloat16_t, false>(c, buffers);
    } else if (c.transpose) {
        RunKernel<half, true>(c, buffers);
    } else {
        RunKernel<half, false>(c, buffers);
    }
}

const DualLevelGmmUt::Case CASES[] = {
    {"Fp16NzNormal", 48, 64, 512, {16, 16, 16}, 0, false, false, false, 1},
    {"Fp16ZnBias", 48, 64, 512, {16, 16, 16}, 1, true, true, false, 1},
    {"Bf16NzBias", 48, 64, 512, {16, 16, 16}, 1, true, false, true, 1},
    {"Bf16ZnNormal", 48, 64, 512, {16, 16, 16}, 0, false, true, true, 1},
    {"NzMnKTails", 133, 130, 514, {65, 1, 67}, 1, true, false, false, 1},
    {"ZnMnKTails", 133, 130, 514, {65, 1, 67}, 0, true, true, true, 1},
    {"SingleRowEmptyExperts", 1, 2, 4, {0, 1, 0}, 0, false, false, true, 1},
    {"EmptyMiddleExpert", 35, 66, 1024, {17, 0, 18}, 1, true, true, false, 1},
    {"PartialSlotsAcrossExperts", 387, 130, 1536, {129, 129, 129}, 0, false, false, false, 1},
    {"Scale1WindowTail", 3, 16, 4098, {1, 1, 1}, 1, true, false, true, 1},
    {"Level0Segment256", 3, 16, 4098, {1, 1, 1}, 1, true, false, true, 1, 256},
    {"Scale0WindowTail", 3, 16, 16386, {1, 1, 1}, 0, true, true, false, 1},
    {"MultipleCoresAndTiles", 771, 258, 1024, {257, 257, 257}, 1, false, false, true, 2},
    {"SparseOrderedNz", 35, 66, 514, {17, 0, 18}, 2, true, false, false, 1},
    {"SparseOrderedZn", 35, 17, 514, {0, 17, 18}, 2, false, true, true, 2},
    {"SparseMultipleTrailingZeros", 1, 16, 514, {0, 1, 0}, 2, true, false, false, 1},
};

INSTANTIATE_TEST_SUITE_P(DualLevelMx, DualLevelGroupedMatmulKernelTest, testing::ValuesIn(CASES),
                         [](const testing::TestParamInfo<DualLevelGmmUt::Case>& info) { return info.param.name; });
} // namespace
