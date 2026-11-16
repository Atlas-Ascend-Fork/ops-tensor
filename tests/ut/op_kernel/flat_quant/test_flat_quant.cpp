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
 * \file test_flat_quant.cpp
 * \brief FlatQuant Blaze kernel UT: assembly-contract checks + KERNEL_RUN_KF smoke.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "flat_quant.h"
#include "tikicpulib.h"

namespace {
constexpr uint32_t BLOCK_NUM = 1;

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

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT>
void CheckPublicAssembly()
{
    using Types = FlatQuantUT::FlatQuantBlazeTypes<X_TYPE, Y_TYPE, SCALE_TYPE, C_LAYOUT>;
    using Kernel = typename Types::MatmulKernel;
    using BlockMmad = typename Types::BlockMmad;
    using BlockEpilogue = typename Types::BlockEpilogue;
    using BlockScheduler = typename Types::BlockScheduler;

    static_assert(std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, Blaze::Attention::KernelFlatQuant>);
    static_assert(std::is_same_v<typename BlockMmad::L0cType, float>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeIn, X_TYPE>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeOut, Y_TYPE>);
    static_assert(std::is_same_v<typename BlockEpilogue::DataTypeScale, SCALE_TYPE>);
    static_assert(std::is_same_v<typename Kernel::BlockMmad, BlockMmad>);
    static_assert(std::is_same_v<typename Kernel::BlockEpilogue, BlockEpilogue>);
    static_assert(std::is_same_v<typename Kernel::BlockScheduler, BlockScheduler>);

    typename BlockScheduler::Params schParams{2, 6.0f, 1.0f / 6.0f};
    EXPECT_EQ(schParams.iterBatch, 2);
    EXPECT_FLOAT_EQ(schParams.dstTypeMax, 6.0f);
    EXPECT_FLOAT_EQ(schParams.invDstTypeMax, 1.0f / 6.0f);

    typename BlockEpilogue::Params epiParams{nullptr, nullptr, {16, 16, 16, 4}, 6.0f, 1.0f / 6.0f};
    EXPECT_EQ(epiParams.outGmAddr, nullptr);
    EXPECT_FLOAT_EQ(epiParams.dstTypeMax, 6.0f);
}

template <typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, typename C_LAYOUT, bool USE_LEGACY_EPILOGUE = false>
void RunKernelSmoke(int64_t M, int64_t N, int64_t K, int64_t iterBatch, float dstTypeMax, float invDstTypeMax)
{
    const size_t xSize = static_cast<size_t>(K) * M * N * sizeof(X_TYPE);
    const size_t p1Size = static_cast<size_t>(M) * M * sizeof(X_TYPE);
    const size_t p2Size = static_cast<size_t>(N) * N * sizeof(X_TYPE);
    const size_t outSize = static_cast<size_t>(K) * M * N / 2;
    const size_t scaleSize = static_cast<size_t>(K) * ((M * N + 31) / 32) * 2;
    const size_t workspaceSize = static_cast<size_t>(K) * M * N * sizeof(float) + 16 * 1024 * 1024;

    GmBuffer x(xSize);
    GmBuffer p1(p1Size);
    GmBuffer p2(p2Size);
    GmBuffer out(outSize);
    GmBuffer scale(scaleSize);
    GmBuffer workspace(workspaceSize);
    GmBuffer tiling(sizeof(FlatQuantTilingData));

    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(p1.Get(), nullptr);
    ASSERT_NE(p2.Get(), nullptr);
    ASSERT_NE(out.Get(), nullptr);
    ASSERT_NE(scale.Get(), nullptr);
    ASSERT_NE(workspace.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    std::fill_n(reinterpret_cast<uint8_t*>(x.Get()), xSize, 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(p1.Get()), p1Size, 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(p2.Get()), p2Size, 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(out.Get()), outSize, 0U);
    std::fill_n(reinterpret_cast<uint8_t*>(scale.Get()), scaleSize, 0U);

    FlatQuantTilingData* tilingData = reinterpret_cast<FlatQuantTilingData*>(tiling.Get());
    tilingData->hasP2 = 1;
    tilingData->K = K;
    tilingData->M = M;
    tilingData->N = N;
    tilingData->iterBatch = iterBatch;
    tilingData->dstTypeMax = dstTypeMax;
    tilingData->invDstTypeMax = invDstTypeMax;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = FlatQuantKernelEntry<X_TYPE, Y_TYPE, SCALE_TYPE, C_LAYOUT, USE_LEGACY_EPILOGUE>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, x.Get(), p1.Get(), p2.Get(), out.Get(), scale.Get(), workspace.Get(),
                              tiling.Get()))
        << "FlatQuant Blaze kernel execution failed";
}
} // namespace

TEST(FlatQuantBlazeTest, AssembleBf16Fp4)
{
    CheckPublicAssembly<bfloat16_t, fp4x2_e2m1_t, float, asc::te::nd_ext_layout_ptn>();
}

TEST(FlatQuantBlazeTest, KernelSmokeBasic)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, float, asc::te::nd_ext_layout_ptn>(16, 16, 1, 1, 6.0f, 1.0f / 6.0f);
}

TEST(FlatQuantBlazeTest, KernelSmokeLegacyEpilogue)
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr float FP4_E2M1_MAX_VALUE = 6.0f;
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, float, asc::te::nd_ext_layout_ptn, true>(M, N, 1, 1, FP4_E2M1_MAX_VALUE,
                                                                                      1.0f / FP4_E2M1_MAX_VALUE);
}

TEST(FlatQuantBlazeTest, KernelSmokeIterBatch)
{
    RunKernelSmoke<bfloat16_t, fp4x2_e2m1_t, float, asc::te::nd_ext_layout_ptn>(16, 16, 4, 2, 6.0f, 1.0f / 6.0f);
}

TEST(FlatQuantBlazeTest, KernelSmokeNoP2)
{
    GmBuffer x(16 * 16 * sizeof(bfloat16_t));
    GmBuffer p1(16 * 16 * sizeof(bfloat16_t));
    GmBuffer p2(16 * 16 * sizeof(bfloat16_t));
    GmBuffer out(16 * 16 / 2);
    GmBuffer scale(((16 * 16 + 31) / 32) * 2);
    GmBuffer workspace(16 * 16 * sizeof(float) + 16 * 1024 * 1024);
    GmBuffer tiling(sizeof(FlatQuantTilingData));

    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    std::fill_n(reinterpret_cast<uint8_t*>(x.Get()), 16 * 16 * sizeof(bfloat16_t), 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(p1.Get()), 16 * 16 * sizeof(bfloat16_t), 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(p2.Get()), 16 * 16 * sizeof(bfloat16_t), 0x38U);

    FlatQuantTilingData* tilingData = reinterpret_cast<FlatQuantTilingData*>(tiling.Get());
    tilingData->hasP2 = 0;
    tilingData->K = 1;
    tilingData->M = 16;
    tilingData->N = 16;
    tilingData->iterBatch = 1;
    tilingData->dstTypeMax = 6.0f;
    tilingData->invDstTypeMax = 1.0f / 6.0f;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = FlatQuantKernelEntry<bfloat16_t, fp4x2_e2m1_t, float, asc::te::nd_ext_layout_ptn>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, x.Get(), p1.Get(), p2.Get(), out.Get(), scale.Get(), workspace.Get(),
                              tiling.Get()))
        << "FlatQuant Blaze kernel (noP2) execution failed";
}
