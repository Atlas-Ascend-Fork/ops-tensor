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
#include <array>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <numeric>
#include <vector>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "quant_grouped_matmul/quant_grouped_matmul_mx.h"

namespace QgmmMxA8W4Test {
constexpr uint32_t BLOCK_NUM = 1U;
constexpr uint32_t BASE_M = 16U;
constexpr uint32_t BASE_N = 64U;
constexpr uint32_t BASE_K = 128U;
constexpr uint32_t ASW_K_L1_WINDOW = 4U * BASE_K;
constexpr uint32_t SCALE_K_L1_WINDOW = 4096U;
constexpr uint32_t SCALE_K_PAIR = 64U;
constexpr uint32_t SCALE_PAIR_ELEMENTS = 2U;
constexpr uint32_t FP4_ELEMENTS_PER_BYTE = 2U;
constexpr uint8_t FP8_ONE_BITS = 0x38U;
constexpr uint8_t E8M0_ONE_BITS = 0x7fU;
constexpr std::array<uint8_t, 3> PACKED_WEIGHTS = {0x22U, 0x44U, 0x55U};
constexpr std::array<uint16_t, 3> HALF_BIASES = {0x3c00U, 0x4000U, 0x4200U};
constexpr std::array<uint16_t, 3> BFLOAT16_BIASES = {0x3f80U, 0x4000U, 0x4040U};

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : data_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (data_ != nullptr) {
            AscendC::GmFree(data_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return data_; }

private:
    GM_ADDR data_;
};

template <class BiasType_>
void FillBias(GM_ADDR address, size_t groupCount, int64_t n)
{
    if constexpr (std::is_same_v<BiasType_, float>) {
        for (size_t group = 0; group < groupCount; ++group) {
            std::fill_n(reinterpret_cast<float*>(address) + group * n, n, static_cast<float>(group + 1U));
        }
    } else {
        static_assert(sizeof(BiasType_) == sizeof(uint16_t), "Half bias storage must be 16 bits.");
        for (size_t group = 0; group < groupCount; ++group) {
            const uint16_t bits = std::is_same_v<BiasType_, half> ? HALF_BIASES[group] : BFLOAT16_BIASES[group];
            std::fill_n(reinterpret_cast<uint16_t*>(address) + group * n, n, bits);
        }
    }
}

QGMMUT::QgmmTilingData MakeTiling(const std::vector<int64_t>& groupM, int64_t n, int64_t k, bool hasBias,
                                  uint8_t groupListType, uint8_t dbL0C)
{
    const uint64_t alignedK = Blaze::Gemm::CeilAlign<uint64_t>(k, Blaze::Gemm::C0_SIZE_B4);
    const uint32_t baseK = static_cast<uint32_t>(std::min<uint64_t>(BASE_K, alignedK));
    const uint32_t kL1 = k % SCALE_K_PAIR == 0 ? static_cast<uint32_t>(k) : ASW_K_L1_WINDOW;
    return {static_cast<uint32_t>(groupM.size()),
            std::accumulate(groupM.begin(), groupM.end(), int64_t{0}),
            n,
            k,
            BASE_M,
            BASE_N,
            baseK,
            kL1,
            kL1,
            SCALE_K_L1_WINDOW,
            SCALE_K_L1_WINDOW,
            static_cast<uint8_t>(hasBias),
            dbL0C,
            static_cast<uint8_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT),
            0,
            groupListType,
            1};
}

struct CaseBuffers {
    explicit CaseBuffers(const QGMMUT::QgmmTilingData& t, size_t biasElementBytes)
        : scaleK(Blaze::Gemm::CeilDiv<uint64_t>(t.k, SCALE_K_PAIR) * SCALE_PAIR_ELEMENTS),
          bGroupBytes(Blaze::Gemm::CeilAlign<uint64_t>(t.n, Blaze::Gemm::BLOCK_CUBE) *
                      Blaze::Gemm::CeilAlign<uint64_t>(t.k, Blaze::Gemm::C0_SIZE_B4) / FP4_ELEMENTS_PER_BYTE),
          a(t.m * t.k),
          b(t.groupNum * bGroupBytes),
          scaleA(t.m * scaleK),
          scaleB(t.groupNum * t.n * scaleK),
          bias(t.groupNum * t.n * biasElementBytes),
          c(t.m * t.n * sizeof(float)),
          groupList(t.groupNum * sizeof(int64_t)),
          tiling(sizeof(t))
    {}

    size_t scaleK;
    size_t bGroupBytes;
    GmBuffer a, b, scaleA, scaleB, bias, c, groupList, tiling;
};

template <class BiasType_>
void PrepareInputs(const CaseBuffers& buffers, const QGMMUT::QgmmTilingData& t, const std::vector<int64_t>& groupM)
{
    for (const GmBuffer* buffer : {&buffers.a, &buffers.b, &buffers.scaleA, &buffers.scaleB, &buffers.bias, &buffers.c,
                                   &buffers.groupList, &buffers.tiling}) {
        ASSERT_NE(buffer->Get(), nullptr);
    }
    std::fill_n(buffers.a.Get(), t.m * t.k, FP8_ONE_BITS);
    std::fill_n(buffers.scaleA.Get(), t.m * buffers.scaleK, E8M0_ONE_BITS);
    std::fill_n(buffers.c.Get(), t.m * t.n * sizeof(float), 0xffU);
    const size_t scaleBGroupBytes = t.n * buffers.scaleK;
    int64_t offset = 0;
    for (size_t group = 0; group < groupM.size(); ++group) {
        // Keep ZN K padding nonzero so extra MMAD elements can affect the result.
        std::fill_n(buffers.b.Get() + group * buffers.bGroupBytes, buffers.bGroupBytes, PACKED_WEIGHTS[group]);
        std::fill_n(buffers.scaleB.Get() + group * scaleBGroupBytes, scaleBGroupBytes,
                    static_cast<uint8_t>(E8M0_ONE_BITS + group % SCALE_PAIR_ELEMENTS));
        offset += groupM[group];
        reinterpret_cast<int64_t*>(buffers.groupList.Get())[group] = t.groupListType == 0U ? offset : groupM[group];
    }
    FillBias<BiasType_>(buffers.bias.Get(), t.groupNum, t.n);
    *reinterpret_cast<QGMMUT::QgmmTilingData*>(buffers.tiling.Get()) = t;
}

void CheckOutput(const float* actual, const QGMMUT::QgmmTilingData& t, const std::vector<int64_t>& groupM)
{
    size_t outputOffset = 0;
    for (size_t group = 0; group < groupM.size(); ++group) {
        const int64_t weight = static_cast<int64_t>(group) + 1;
        const int64_t scale = group % SCALE_PAIR_ELEMENTS == 0U ? 1 : 2;
        const int64_t expected = t.k * weight * scale + (t.isBias ? weight : 0);
        for (int64_t index = 0; index < groupM[group] * t.n; ++index) {
            ASSERT_FLOAT_EQ(actual[outputOffset++], static_cast<float>(expected))
                << "group=" << group << " index=" << index;
        }
    }
}

template <class BiasType_>
void RunCase(const std::vector<int64_t>& groupM, int64_t n, int64_t k, bool hasBias, uint8_t groupListType,
             uint8_t dbL0C)
{
    ASSERT_LE(groupM.size(), PACKED_WEIGHTS.size());
    const auto t = MakeTiling(groupM, n, k, hasBias, groupListType, dbL0C);
    CaseBuffers buffers(t, sizeof(BiasType_));
    ASSERT_NO_FATAL_FAILURE(PrepareInputs<BiasType_>(buffers, t, groupM));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = qgmm_mx_kernel_entry<fp8_e4m3fn_t, fp4x2_e2m1_t, float, BiasType_, asc::te::nd_ext_layout_ptn,
                                      asc::te::zn_layout_ptn>;
    ASSERT_TRUE(KERNEL_RUN_KF(entry, BLOCK_NUM, buffers.a.Get(), buffers.b.Get(), buffers.scaleA.Get(),
                              buffers.scaleB.Get(), hasBias ? buffers.bias.Get() : nullptr, buffers.c.Get(),
                              buffers.groupList.Get(), buffers.tiling.Get()));
    CheckOutput(reinterpret_cast<const float*>(buffers.c.Get()), t, groupM);
}
} // namespace QgmmMxA8W4Test

TEST(QgmmMxA8W4KernelTest, LengthGroupsWithTailAndNoBias)
{
    QgmmMxA8W4Test::RunCase<float>({16, 9}, 32, 64, false, 1, 1);
}

TEST(QgmmMxA8W4KernelTest, OffsetGroupsEmptyMiddleAndHalfBias)
{
    QgmmMxA8W4Test::RunCase<half>({3, 0, 9}, 80, 128, true, 0, 1);
}

TEST(QgmmMxA8W4KernelTest, LengthGroupsBfloat16BiasAndL0cDoubleBuffer)
{
    QgmmMxA8W4Test::RunCase<bfloat16_t>({7, 10}, 16, 512, true, 1, 2);
}

TEST(QgmmMxA8W4KernelTest, OffsetGroupsFloatBias) { QgmmMxA8W4Test::RunCase<float>({1, 15}, 80, 128, true, 0, 2); }

TEST(QgmmMxA8W4KernelTest, ShortKTailReusesL0A) { QgmmMxA8W4Test::RunCase<float>({16}, 32, 258, false, 1, 1); }
