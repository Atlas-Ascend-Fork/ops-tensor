/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "grouped_matmul_finalize_routing_ut.h"

namespace {
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

template <typename T>
constexpr size_t MxBytes(size_t elements)
{
    if constexpr (std::is_same_v<T, fp4x2_e2m1_t>) {
        return (elements + 1U) / 2U;
    }
    return elements * sizeof(T);
}

template <typename Config, typename AType, typename BType, typename CType, typename LogitType, typename RowIndexType,
          typename LayoutB>
void RunCase()
{
    constexpr size_t weightBytes = Config::GROUP_NUM * MxBytes<BType>(Config::K * Config::N);
    constexpr size_t xBytes = MxBytes<AType>(Config::TOTAL_M * Config::K);
    constexpr size_t xScaleBytes = Config::TOTAL_M * Config::SCALE_K * sizeof(fp8_e8m0_t);
    constexpr size_t weightScaleBytes = Config::GROUP_NUM * Config::N * Config::SCALE_K * sizeof(fp8_e8m0_t);
    GmBuffer x(xBytes);
    GmBuffer weight(weightBytes);
    GmBuffer weightScale(weightScaleBytes);
    GmBuffer bias(Config::GROUP_NUM * Config::N * sizeof(bfloat16_t));
    GmBuffer xScale(xScaleBytes);
    GmBuffer groupList(Config::GROUP_NUM * sizeof(int64_t));
    GmBuffer sharedInput(std::max<size_t>(1U, Config::SHARED_INPUT_LEN * Config::N * sizeof(bfloat16_t)));
    GmBuffer logit(Config::TOTAL_M * sizeof(LogitType));
    GmBuffer rowIndex(Config::TOTAL_M * sizeof(RowIndexType));
    GmBuffer y(Config::BATCH * Config::N * sizeof(CType));
    GmBuffer tiling(8U);
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(weight.Get(), nullptr);
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);
    std::fill_n(reinterpret_cast<uint8_t*>(x.Get()), xBytes, std::is_same_v<AType, fp4x2_e2m1_t> ? 0x22U : 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(weight.Get()), weightBytes,
                std::is_same_v<BType, fp4x2_e2m1_t> ? 0x22U : 0x38U);
    std::fill_n(reinterpret_cast<uint8_t*>(weightScale.Get()), weightScaleBytes, 0x7FU);
    std::fill_n(reinterpret_cast<uint8_t*>(xScale.Get()), xScaleBytes, 0x7FU);
    std::fill_n(reinterpret_cast<uint8_t*>(bias.Get()), Config::GROUP_NUM * Config::N * sizeof(bfloat16_t), 0U);
    auto* groupListData = reinterpret_cast<int64_t*>(groupList.Get());
    std::fill_n(groupListData, Config::GROUP_NUM, Config::GROUP_M);
    std::fill_n(reinterpret_cast<uint8_t*>(sharedInput.Get()),
                std::max<size_t>(1U, Config::SHARED_INPUT_LEN * Config::N * sizeof(bfloat16_t)), 0U);
    std::fill_n(reinterpret_cast<LogitType*>(logit.Get()), Config::TOTAL_M, static_cast<LogitType>(1));
    auto* rowIndexData = reinterpret_cast<RowIndexType*>(rowIndex.Get());
    for (int64_t i = 0; i < Config::TOTAL_M; ++i) {
        rowIndexData[i] = static_cast<RowIndexType>(i % Config::BATCH);
    }
    std::fill_n(reinterpret_cast<uint8_t*>(y.Get()), Config::BATCH * Config::N * sizeof(CType), 0U);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = grouped_matmul_finalize_routing_kernel_entry<Config, AType, BType, CType, LogitType, RowIndexType,
                                                           LayoutB>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, 1U, x.Get(), weight.Get(), weightScale.Get(), bias.Get(), xScale.Get(),
                              groupList.Get(), sharedInput.Get(), logit.Get(), rowIndex.Get(), y.Get(), tiling.Get()));
}
} // namespace

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp4WeightNzBf16OutputBf16Logit)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, bfloat16_t,
            int32_t, asc::te::nz_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8E4M3WeightNzBf16OutputFp32Logit)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float, int32_t,
            asc::te::nz_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8E4M3WeightNzFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, int64_t,
            asc::te::nz_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8E5M2WeightNdFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e5m2_t, fp8_e5m2_t, float, float, int64_t,
            asc::te::nd_ext_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8MixedWeightNdFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e4m3fn_t, fp8_e5m2_t, float, float, int64_t,
            asc::te::nd_ext_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8E5M2WeightDnFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e5m2_t, fp8_e5m2_t, float, float, int64_t,
            asc::te::dn_ext_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp8E4M3WeightZnFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, int64_t,
            asc::te::zn_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp4WeightDnFp32OutputTailWithBiasAndSharedInput)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::TailCaseConfig, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, int64_t,
            asc::te::dn_ext_layout_ptn>();
}

TEST(GroupedMatmulFinalizeRoutingKernelTest, MxFp4WeightNdFp32Output)
{
    RunCase<GroupedMatmulFinalizeRoutingUt::AlignedCaseConfig, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, int64_t,
            asc::te::nd_ext_layout_ptn>();
}
