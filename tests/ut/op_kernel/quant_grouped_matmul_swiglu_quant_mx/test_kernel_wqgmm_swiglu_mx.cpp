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
 * \file test_kernel_wqgmm_swiglu_mx.cpp
 * \brief Type/Params assembly and real epilogue metadata checks, not numerical or synchronization tests.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "blaze_kernel_stub.h"
#include "grouped_matmul_mx_a8w4/grouped_matmul_mx_a8w4_cpu_debug_stub.h"

// Reuse the existing A8W4 suite's builtin aliases without extending the CPU simulator.
#if defined(ASCENDC_CPU_DEBUG)
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#define __fp4e1m2x2 fp4x2_e1m2_t
#endif
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e1m2x2
#undef __fp4e2m1x2
#undef __fp8e4m3
#endif

#include "quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace {

using Nd = asc::te::nd_ext_layout_ptn;
using Scale = fp8_e8m0_t;
using Mmad = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::GroupedMatmulSwigluWithWeightQuantMx, AscendC::Std::tuple<fp8_e4m3fn_t, Scale>,
    AscendC::Std::tuple<Nd, asc::te::scalea_nd_layout_ptn>, AscendC::Std::tuple<fp4x2_e2m1_t, Scale>,
    AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, float, Nd, bfloat16_t, Nd>;
using Prologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, true>;
using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, Scale, false, true, true>;
using LegacyEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, Scale>;
using DirectBf16 = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<bfloat16_t, float, Scale, true>;
using Kernel = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<asc::te::shape<int64_t, int64_t, int64_t, int64_t>, Mmad,
                                                           Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit,
                                                           Epilogue, Prologue, true>;
constexpr uint64_t UB_BYTES = 248UL * 1024UL;
constexpr uint64_t PROLOGUE_UB_BYTES = 229UL * 1024UL;
constexpr uint64_t SMOOTH_UB_OFFSET = 246UL * 1024UL;

// Match the Transformer V3 wrapper, including its ND GM scale-A layout and BF16 bias type.
static_assert(std::is_same_v<Kernel::AType, fp8_e4m3fn_t>);
static_assert(std::is_same_v<Kernel::BType, fp4x2_e2m1_t>);
static_assert(std::is_same_v<Kernel::ScaleType, Scale>);
static_assert(std::is_same_v<Kernel::BiasType, bfloat16_t>);
static_assert(std::is_same_v<Mmad::CType, Epilogue::DataTypeIn>);
static_assert(std::is_same_v<Mmad::LayoutScaleA, asc::te::scalea_nd_layout_ptn>);
static_assert(std::is_same_v<Mmad::LayoutB, asc::te::zn_layout_ptn>);
static_assert(std::is_same_v<Prologue::OutType, Mmad::AType>);
static_assert(std::is_same_v<Prologue::InType, Mmad::BType>);
static_assert(std::is_same_v<decltype(Kernel::Params{}.epilogue), Epilogue::Params>);
static_assert(std::is_same_v<decltype(Kernel::Params{}.scheduler), Kernel::Scheduler::Params>);
static_assert(std::is_invocable_r_v<void, Kernel&, const Kernel::Params&>);
static_assert(Epilogue::A8W4_V3_FAST && Epilogue::ENABLE_SMOOTH_SCALE && !Epilogue::DIRECT_BF16_OUTPUT);
static_assert(!LegacyEpilogue::ENABLE_SMOOTH_SCALE && !LegacyEpilogue::DIRECT_BF16_OUTPUT);
static_assert(
    std::is_same_v<LegacyEpilogue,
                   Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, Scale, false, false>>);
static_assert(DirectBf16::DIRECT_BF16_OUTPUT && !DirectBf16::ENABLE_SMOOTH_SCALE);
static_assert(std::is_aggregate_v<Epilogue::Params>, "Keep C++17 legacy aggregate initialization compatible");

// Explicit-instantiation access permits inspecting real private layout metadata without changing header access
// specifiers or adding a production test hook. No compute, copy, or synchronization operation is replaced.
template <typename Tag, typename Tag::Type Member>
struct MetadataAccess {
    friend constexpr typename Tag::Type ReadMember(Tag) { return Member; }
};

struct SmoothOffset {
    using Type = int64_t Epilogue::*;
    friend constexpr Type ReadMember(SmoothOffset);
};
template struct MetadataAccess<SmoothOffset, &Epilogue::smoothScaleOffset_>;

struct SmoothUbOffset {
    using Type = uint64_t Epilogue::*;
    friend constexpr Type ReadMember(SmoothUbOffset);
};
template struct MetadataAccess<SmoothUbOffset, &Epilogue::smoothScaleUbOffset_>;

struct ScaleBlockUbOffset {
    using Type = uint64_t Epilogue::*;
    friend constexpr Type ReadMember(ScaleBlockUbOffset);
};
template struct MetadataAccess<ScaleBlockUbOffset, &Epilogue::quantScaleBlockOutputUbOffset_>;

struct SetupUb {
    using Type = void (Epilogue::*)();
    friend constexpr Type ReadMember(SetupUb);
};
template struct MetadataAccess<SetupUb, &Epilogue::SetupUbLayout>;

struct KBlockPolicy {
    static constexpr uint64_t COLUMNS = 128UL;
    using Type = void (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t&, uint64_t&, uint64_t&);
    friend constexpr Type ReadMember(KBlockPolicy);
};

template struct MetadataAccess<KBlockPolicy, &Mmad::CalcWeightQuantMxKBlock<int8_t, KBlockPolicy::COLUMNS>>;

struct PrologueScaleOffset {
    using Type = const uint64_t*;
    friend constexpr Type ReadMember(PrologueScaleOffset);
};
template struct MetadataAccess<PrologueScaleOffset, &Prologue::MX_SCALE_OUTPUT_OFFSET>;

struct PrologueScaleBytes {
    using Type = const uint64_t*;
    friend constexpr Type ReadMember(PrologueScaleBytes);
};
template struct MetadataAccess<PrologueScaleBytes, &Prologue::MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE>;

struct PrologueScaleBuffers {
    using Type = const uint64_t*;
    friend constexpr Type ReadMember(PrologueScaleBuffers);
};
template struct MetadataAccess<PrologueScaleBuffers, &Prologue::MX_SCALE_OUTPUT_BUFFER_NUM>;

constexpr uint64_t PROLOGUE_UB_END = *ReadMember(PrologueScaleOffset{}) +
                                     *ReadMember(PrologueScaleBytes{}) * *ReadMember(PrologueScaleBuffers{});
static_assert(PROLOGUE_UB_END == PROLOGUE_UB_BYTES, "The fast prologue layout reserves 229 KiB");

struct Metadata {
    int64_t smoothOffset;
    uint64_t smoothUbOffset;
    uint64_t scaleBlockUbOffset;
};

// Run only metadata methods on AIC, where the epilogue constructor/destructor have no event side effects.
// This intentionally does not execute Kernel::operator(), RunConcatN, FP4 conversion, or MX quantization.
__global__ __aicore__ void InspectEpilogueMetadata(GM_ADDR result, int64_t expert, int64_t halfN, int64_t tileNOffset,
                                                   int64_t previousOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        Epilogue epilogue;
        (epilogue.*ReadMember(SetupUb{}))();
        epilogue.SetSmoothScaleOffset(previousOffset);
        epilogue.UpdateNextProblem(asc::te::make_shape(17L, halfN, 128L));
        epilogue.SetSmoothScaleOffset(expert * halfN + tileNOffset);
        *reinterpret_cast<__gm__ Metadata*>(result) = {epilogue.*ReadMember(SmoothOffset{}),
                                                       epilogue.*ReadMember(SmoothUbOffset{}),
                                                       epilogue.*ReadMember(ScaleBlockUbOffset{})};
    }
}

class MetadataBuffer {
public:
    MetadataBuffer() : address_(static_cast<GM_ADDR>(AscendC::GmAlloc(sizeof(Metadata)))) {}
    ~MetadataBuffer()
    {
        if (address_ != nullptr) {
            AscendC::GmFree(address_);
        }
    }
    MetadataBuffer(const MetadataBuffer&) = delete;
    MetadataBuffer& operator=(const MetadataBuffer&) = delete;
    GM_ADDR Get() const { return address_; }

private:
    GM_ADDR address_;
};

TEST(WqgmmSwigluMxAssemblyTest, DefaultParamsKeepOptionalSmoothingDisabled)
{
    Kernel::Params params;
    EXPECT_EQ(params.x, nullptr);
    EXPECT_EQ(params.weight, nullptr);
    EXPECT_EQ(params.xScale, nullptr);
    EXPECT_EQ(params.weightScale, nullptr);
    EXPECT_EQ(params.groupList, nullptr);
    EXPECT_EQ(params.m, 0);
    EXPECT_EQ(params.n, 0);
    EXPECT_EQ(params.k, 0);
    EXPECT_EQ(params.groupNum, 0U);
    EXPECT_EQ(params.groupListType, 0U);
    EXPECT_EQ(params.scheduler.baseM, 0);
    EXPECT_EQ(params.scheduler.baseN, 0);
    EXPECT_EQ(params.epilogue.yGmAddr, nullptr);
    EXPECT_EQ(params.epilogue.yScaleGmAddr, nullptr);
    EXPECT_EQ(params.epilogue.smoothScaleGmAddr, nullptr);
    EXPECT_FLOAT_EQ(params.epilogue.inputScale, 1.0F);
    EXPECT_EQ(params.epilogue.swigluMode, 0U);
    EXPECT_EQ(params.epilogue.scaleAlg, 0U);
}

template <typename Epi>
void CheckLegacyParams()
{
    for (uint32_t scaleAlg : {0U, 1U}) {
        typename Epi::Params params{nullptr, nullptr, 128U, 64U, scaleAlg};
        EXPECT_EQ(params.baseM, 128U);
        EXPECT_EQ(params.baseN, 64U);
        EXPECT_EQ(params.scaleAlg, scaleAlg);
        EXPECT_EQ(params.swigluMode, 0U);
        EXPECT_FLOAT_EQ(params.clampLimit, Blaze::Epilogue::Block::Constant::DEFAULT_CLAMP_LIMIT);
        EXPECT_FLOAT_EQ(params.gluAlpha, Blaze::Epilogue::Block::Constant::DEFAULT_GLU_ALPHA);
        EXPECT_FLOAT_EQ(params.gluBias, Blaze::Epilogue::Block::Constant::DEFAULT_GLU_BIAS);
        EXPECT_FLOAT_EQ(params.dstTypeMax, 0.0F);
        EXPECT_EQ(params.smoothScaleGmAddr, nullptr);
        EXPECT_FLOAT_EQ(params.inputScale, 1.0F);
    }
}

TEST(WqgmmSwigluMxAssemblyTest, FiveFieldLegacyParamsKeepNewFieldDefaults)
{
    CheckLegacyParams<LegacyEpilogue>();
    CheckLegacyParams<DirectBf16>();
    CheckLegacyParams<Epilogue>();
}

TEST(WqgmmSwigluMxAssemblyTest, TenFieldLegacyParamsKeepConfiguredAttributes)
{
    Epilogue::Params params{nullptr, nullptr, 128U, 64U, 1U, 2U, 5.5F, 1.2F, -0.5F, 0.0F};
    EXPECT_EQ(params.scaleAlg, 1U);
    EXPECT_EQ(params.swigluMode, 2U);
    EXPECT_FLOAT_EQ(params.clampLimit, 5.5F);
    EXPECT_FLOAT_EQ(params.gluAlpha, 1.2F);
    EXPECT_FLOAT_EQ(params.gluBias, -0.5F);
    EXPECT_FLOAT_EQ(params.dstTypeMax, 0.0F);
    EXPECT_EQ(params.smoothScaleGmAddr, nullptr);
    EXPECT_FLOAT_EQ(params.inputScale, 1.0F);
}

TEST(WqgmmSwigluMxAssemblyTest, WrapperFieldsPreserveAttributesAndOptionalSmoothPointer)
{
    MetadataBuffer buffer;
    ASSERT_NE(buffer.Get(), nullptr);
    for (GM_ADDR smooth : {static_cast<GM_ADDR>(nullptr), buffer.Get()}) {
        for (uint32_t scaleAlg : {0U, 1U}) {
            Kernel::Params params{};
            params.m = 129;
            params.n = 256;
            params.k = 128;
            params.groupNum = 3;
            params.groupListType = 1;
            params.scheduler = {128, 128};
            params.epilogue = {buffer.Get(), buffer.Get(), 128,  128,  scaleAlg, 2,
                               7.0F,         1.702F,       1.0F, 0.0F, smooth,   64.0F};
            const Kernel::Params copied = params;
            EXPECT_EQ(copied.m, 129);
            EXPECT_EQ(copied.n, 256);
            EXPECT_EQ(copied.k, 128);
            EXPECT_EQ(copied.groupNum, 3U);
            EXPECT_EQ(copied.groupListType, 1U);
            EXPECT_EQ(copied.scheduler.baseM, 128);
            EXPECT_EQ(copied.scheduler.baseN, 128);
            EXPECT_EQ(copied.epilogue.yGmAddr, buffer.Get());
            EXPECT_EQ(copied.epilogue.yScaleGmAddr, buffer.Get());
            EXPECT_EQ(copied.epilogue.scaleAlg, scaleAlg);
            EXPECT_EQ(copied.epilogue.swigluMode, 2U);
            EXPECT_EQ(copied.epilogue.smoothScaleGmAddr, smooth);
            EXPECT_FLOAT_EQ(copied.epilogue.inputScale, 64.0F);
        }
    }
}

struct OffsetCase {
    int64_t expert;
    int64_t halfN;
    int64_t tileNOffset;
    int64_t previousOffset;
    int64_t expected;
};

class WqgmmSwigluMxMetadataTest : public testing::TestWithParam<OffsetCase> {};

TEST_P(WqgmmSwigluMxMetadataTest, ExpertChannelOffsetAndTimeSharedUbBudget)
{
    const auto& item = GetParam();
    MetadataBuffer buffer;
    ASSERT_NE(buffer.Get(), nullptr);
    *reinterpret_cast<Metadata*>(buffer.Get()) = {-1, UB_BYTES + 1, UB_BYTES + 1};
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(InspectEpilogueMetadata, 1, buffer.Get(), item.expert, item.halfN, item.tileNOffset,
                              item.previousOffset));
    const auto actual = *reinterpret_cast<const Metadata*>(buffer.Get());
    // Offsets count FP32 elements. This checks the epilogue contract, not assembly group-list traversal.
    EXPECT_EQ(actual.smoothOffset, item.expected);
    EXPECT_LT(actual.smoothOffset, (item.expert + 1) * item.halfN);
    EXPECT_EQ(actual.scaleBlockUbOffset, 196UL * 1024UL);
    EXPECT_EQ(actual.smoothUbOffset, SMOOTH_UB_OFFSET);
    EXPECT_GE(actual.smoothUbOffset, PROLOGUE_UB_END);
    EXPECT_GE(actual.smoothUbOffset, actual.scaleBlockUbOffset + Epilogue::INPUT_UB_TILE_ELEMENTS);
    const uint64_t epilogueEnd = actual.smoothUbOffset + item.halfN * sizeof(float);
    EXPECT_EQ(epilogueEnd, item.halfN == 64 ? 252160UL : 252416UL);
    EXPECT_LE(epilogueEnd, UB_BYTES);
    EXPECT_LE(PROLOGUE_UB_END + sizeof(float), actual.smoothUbOffset);
}

TEST(WqgmmSwigluMxMetadataTest, InvalidKBlockInputsDoNotDivideByZero)
{
    constexpr uint64_t TEST_CAPACITY = 32768UL;
    constexpr uint64_t TEST_COLUMNS = KBlockPolicy::COLUMNS;
    constexpr uint64_t TEST_ROWS = 16UL;
    constexpr uint64_t TEST_REUSE = 2UL;
    constexpr uint64_t EXPECTED_SMALL_K = 128UL;
    for (bool zeroRows : {false, true}) {
        uint64_t ka = 1, kb = 1, k0 = 1;
        ReadMember(KBlockPolicy{})(zeroRows ? 0UL : TEST_ROWS, TEST_COLUMNS, TEST_CAPACITY, TEST_CAPACITY,
                                   zeroRows ? TEST_REUSE : 0UL, ka, kb, k0);
        EXPECT_EQ(ka, 0UL);
        EXPECT_EQ(kb, 0UL);
        EXPECT_EQ(k0, EXPECTED_SMALL_K);
    }
}

INSTANTIATE_TEST_SUITE_P(HalfN64And128, WqgmmSwigluMxMetadataTest,
                         testing::Values(OffsetCase{0, 64, 0, 128, 0}, OffsetCase{1, 64, 0, 0, 64},
                                         OffsetCase{2, 64, 0, 64, 128}, OffsetCase{0, 128, 0, 256, 0},
                                         OffsetCase{1, 128, 0, 0, 128}, OffsetCase{2, 128, 0, 128, 256},
                                         OffsetCase{2, 128, 64, 256, 320}, OffsetCase{1023, 64, 0, 0, 65472},
                                         OffsetCase{1023, 128, 64, 0, 131008}));

} // namespace
