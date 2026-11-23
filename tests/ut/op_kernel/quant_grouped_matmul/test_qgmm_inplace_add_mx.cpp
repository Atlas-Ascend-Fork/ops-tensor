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
#include <cstring>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "quant_grouped_matmul_mx.h"

namespace {
constexpr uint32_t GROUP_NUM = 3;
constexpr uint32_t BLOCK_NUM = 1;
constexpr uint32_t MULTI_CORE_BLOCK_NUM = 4;
constexpr uint32_t REPEAT_COUNT = 2;
constexpr uint32_t BASE_M = 16;
constexpr uint32_t BASE_N = 32;
constexpr uint32_t BASE_K = 64;
constexpr uint32_t K_BL1 = 128;
constexpr uint32_t SCALE_WINDOW = 128;
constexpr int8_t SPLIT_K = 2;
constexpr uint8_t SINGLE_WEIGHT = 1;
constexpr uint8_t NO_BIAS = 0;
constexpr uint8_t GROUP_LIST_OFFSET = 0;
constexpr uint8_t GROUP_LIST_LENGTH = 1;
constexpr int64_t SHORT_GROUP_K = 33;
constexpr int64_t M_TAIL = 33;
constexpr int64_t FP4_M = 64;
constexpr int64_t N_TAIL = 48;
constexpr int64_t TOTAL_K = 98;
constexpr int64_t LONG_TOTAL_K = 386;
constexpr int64_t MX_SCALE_PACK_K = 64;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr uint8_t E4M3_ONE = 0x38;
constexpr uint8_t E5M2_ONE = 0x3c;
constexpr uint8_t E2M1_TWO_ONES = 0x22;
constexpr uint8_t E8M0_ONE = 127;
constexpr uint8_t E8M0_TWO = 128;
constexpr float INITIAL_Y = 3.0f;

template <typename AType, typename BType>
using InplaceMxMmad = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::GroupedMatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, true>, AType, asc::te::dn_ext_layout_ptn,
    BType, asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn>;

template <typename Kernel>
__global__ __aicore__ void qgmm_inplace_add_mx_kernel_entry(GM_ADDR config)
{
    const auto params = *reinterpret_cast<const typename Kernel::Params*>(config);
    Kernel{}(params);
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (ptr_ != nullptr)
            AscendC::GmFree(ptr_);
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_;
};

template <typename AType, typename BType>
void RunInplaceCase(uint8_t groupListType, int64_t totalK = TOTAL_K, bool allEmpty = false,
                    uint32_t blockNum = BLOCK_NUM)
{
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Epilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, InplaceMxMmad<AType, BType>, Epilogue, Scheduler>;
    constexpr bool IS_FP4 = Blaze::Gemm::IsFp4<AType>();
    constexpr int64_t VALUES_PER_BYTE = IS_FP4 ? 2 : 1;
    constexpr int64_t M = IS_FP4 ? FP4_M : M_TAIL;
    constexpr int64_t SCALE_SPARE_ROWS = 1;
    const int64_t scaleRows = totalK / MX_SCALE_PACK_K + GROUP_NUM + SCALE_SPARE_ROWS;
    const size_t aBytes = totalK * M / VALUES_PER_BYTE;
    const size_t bBytes = totalK * N_TAIL / VALUES_PER_BYTE;
    constexpr size_t GROUP_OUTPUT_ELEMENTS = M * N_TAIL;
    constexpr size_t OUTPUT_ELEMENTS = GROUP_NUM * GROUP_OUTPUT_ELEMENTS;
    GmBuffer a(aBytes);
    GmBuffer b(bBytes);
    GmBuffer scaleA(scaleRows * M * MX_SCALE_PAIR);
    GmBuffer scaleB(scaleRows * N_TAIL * MX_SCALE_PAIR);
    GmBuffer groups(GROUP_NUM * sizeof(int64_t));
    GmBuffer output(OUTPUT_ELEMENTS * sizeof(float));
    GmBuffer config(sizeof(typename Kernel::Params));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(scaleA.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(groups.Get(), nullptr);
    ASSERT_NE(output.Get(), nullptr);
    ASSERT_NE(config.Get(), nullptr);
    constexpr uint8_t A_ONE = IS_FP4 ? E2M1_TWO_ONES : (std::is_same_v<AType, fp8_e4m3fn_t> ? E4M3_ONE : E5M2_ONE);
    constexpr uint8_t B_ONE = IS_FP4 ? E2M1_TWO_ONES : (std::is_same_v<BType, fp8_e4m3fn_t> ? E4M3_ONE : E5M2_ONE);
    std::memset(a.Get(), A_ONE, aBytes);
    std::memset(b.Get(), B_ONE, bBytes);
    std::memset(scaleA.Get(), E8M0_ONE, scaleRows * M * MX_SCALE_PAIR);
    std::memset(scaleB.Get(), E8M0_ONE, scaleRows * N_TAIL * MX_SCALE_PAIR);
    // The empty middle group still consumes a scale padding row. Give only
    // the last group's ScaleB a different value to detect compressed indices.
    constexpr int64_t LAST_GROUP = GROUP_NUM - 1;
    constexpr int64_t LAST_SCALE_ROW = SHORT_GROUP_K / MX_SCALE_PACK_K + LAST_GROUP;
    std::memset(scaleB.Get() + LAST_SCALE_ROW * N_TAIL * MX_SCALE_PAIR, E8M0_TWO,
                (scaleRows - LAST_SCALE_ROW) * N_TAIL * MX_SCALE_PAIR);
    const std::array<int64_t, GROUP_NUM> lengths = allEmpty ? std::array<int64_t, GROUP_NUM>{0, 0, 0} :
                                                              std::array<int64_t, GROUP_NUM>{SHORT_GROUP_K, 0,
                                                                                             totalK - SHORT_GROUP_K};
    auto* groupValues = reinterpret_cast<int64_t*>(groups.Get());
    int64_t prefix = 0;
    for (uint32_t group = 0; group < GROUP_NUM; ++group) {
        prefix += lengths[group];
        groupValues[group] = groupListType == GROUP_LIST_OFFSET ? prefix : lengths[group];
    }
    typename Kernel::Params params{};
    params.problemShape = {M, N_TAIL, totalK, 0};
    params.mmadParams.aGmAddr = a.Get();
    params.mmadParams.bGmAddr = b.Get();
    params.mmadParams.cGmAddr = output.Get();
    params.mmadParams.biasGmAddr = nullptr;
    params.mmadParams.scaleAGmAddr = scaleA.Get();
    params.mmadParams.scaleBGmAddr = scaleB.Get();
    params.groupListGmAddr = groups.Get();
    params.gmmParams = {GROUP_NUM,
                        M,
                        N_TAIL,
                        totalK,
                        BASE_M,
                        BASE_N,
                        BASE_K,
                        BASE_K,
                        K_BL1,
                        SCALE_WINDOW,
                        SCALE_WINDOW,
                        NO_BIAS,
                        Blaze::Gemm::DOUBLE_BUFFER_COUNT,
                        Blaze::Gemm::DOUBLE_BUFFER_COUNT,
                        SPLIT_K,
                        groupListType,
                        SINGLE_WEIGHT};
    std::memcpy(config.Get(), &params, sizeof(params));
    auto* values = reinterpret_cast<float*>(output.Get());
    std::fill_n(values, OUTPUT_ELEMENTS, INITIAL_Y);
    // Keep the simulator in the same mode as the existing QGMM tests. The kernel itself is AIC-only.
    // Validate launch and empty-group writes here. MX MMAD and Fixpipe numerics are checked on device.
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = qgmm_inplace_add_mx_kernel_entry<Kernel>;
    for (uint32_t iteration = 1; iteration <= REPEAT_COUNT; ++iteration) {
        ASSERT_TRUE(KERNEL_RUN_KF(fn, blockNum, config.Get()));
        for (uint32_t group = 0; group < GROUP_NUM; ++group) {
            if (lengths[group] != 0) {
                continue;
            }
            for (size_t element = 0; element < GROUP_OUTPUT_ELEMENTS; ++element) {
                ASSERT_FLOAT_EQ(values[group * GROUP_OUTPUT_ELEMENTS + element], INITIAL_Y)
                    << "group=" << group << " element=" << element << " iteration=" << iteration;
            }
        }
    }
}

TEST(QgmmInplaceAddMx, E4M3E4M3Offset) { RunInplaceCase<fp8_e4m3fn_t, fp8_e4m3fn_t>(GROUP_LIST_OFFSET); }
TEST(QgmmInplaceAddMx, E4M3E5M2Length) { RunInplaceCase<fp8_e4m3fn_t, fp8_e5m2_t>(GROUP_LIST_LENGTH); }
TEST(QgmmInplaceAddMx, E5M2E4M3Offset) { RunInplaceCase<fp8_e5m2_t, fp8_e4m3fn_t>(GROUP_LIST_OFFSET); }
TEST(QgmmInplaceAddMx, E5M2E5M2Length) { RunInplaceCase<fp8_e5m2_t, fp8_e5m2_t>(GROUP_LIST_LENGTH); }
TEST(QgmmInplaceAddMx, PackedFp4Length) { RunInplaceCase<fp4x2_e2m1_t, fp4x2_e2m1_t>(GROUP_LIST_LENGTH); }
TEST(QgmmInplaceAddMx, CrossScaleWindow)
{
    RunInplaceCase<fp8_e4m3fn_t, fp8_e4m3fn_t>(GROUP_LIST_LENGTH, LONG_TOTAL_K);
}
TEST(QgmmInplaceAddMx, MultiCoreTail)
{
    RunInplaceCase<fp8_e4m3fn_t, fp8_e4m3fn_t>(GROUP_LIST_LENGTH, TOTAL_K, false, MULTI_CORE_BLOCK_NUM);
}
TEST(QgmmInplaceAddMx, AllEmpty) { RunInplaceCase<fp8_e4m3fn_t, fp8_e4m3fn_t>(GROUP_LIST_OFFSET, TOTAL_K, true); }
} // namespace
