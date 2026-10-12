/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file test_quant_grouped_matmul_swiglu_quant_mx.cpp
 *  \brief Host simulator checks for the fused QGMM SwiGLU MX template and weight-copy routing.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"

#include "blaze_kernel_stub.h"
#include "quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/arch35/copy_gm_to_l1.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace {
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using NdLayout = asc::te::nd_ext_layout_ptn;
constexpr uint32_t K_ALIGN = 128;
constexpr uint32_t N = 128;

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

template <typename LayoutB>
struct FusedTypes {
    using ScaleBLayout = AscendC::Std::conditional_t<Blaze::Gemm::IsTrans<LayoutB>::value,
                                                     asc::te::scaleb_dn_layout_ptn, asc::te::scaleb_nd_layout_ptn>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGmmSwiGluMixMx>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp8_e4m3fn_t, NdLayout, fp8_e4m3fn_t,
                                               AscendC::Std::tuple<LayoutB, ScaleBLayout>, float, NdLayout, float,
                                               NdLayout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, fp8_e8m0_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
};

// The simulator's PV memory operations are stubs: these checks verify the real copy helper's return contract,
// including alignment and source-slice layout, but do not claim to verify transferred bytes.
template <typename LayoutB>
__global__ __aicore__ void CheckWeightCopy(GM_ADDR weight, GM_ADDR result, uint32_t k, uint32_t n, uint32_t windowK,
                                           uint32_t width, uint32_t kOffset, uint32_t nOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        auto sourceLayout = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<32>>{}(k, n);
        auto destinationLayout = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<32>>{}(
            windowK, width * 2U);
        auto source = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(weight)), sourceLayout);
        auto destination = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e4m3fn_t>(0),
                                                destinationLayout);
        auto left = source.slice(asc::te::make_coord(kOffset, nOffset), asc::te::make_shape(windowK, width));
        const bool copied = Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(
            destination, left, Blaze::Gemm::Tile::CopyConcatGM2L1Params{n, k}, kOffset % K_ALIGN == 0);
        *reinterpret_cast<__gm__ uint32_t*>(result) = copied ? 1U : 0U;
    }
}

template <typename LayoutB>
void RunWeightCopyCase(uint32_t k, uint32_t n, uint32_t windowK, uint32_t width, uint32_t kOffset, uint32_t nOffset,
                       bool expectedFastPath)
{
    GmBuffer weight(static_cast<size_t>(k) * n);
    GmBuffer result(sizeof(uint32_t));
    ASSERT_NE(weight.Get(), nullptr);
    ASSERT_NE(result.Get(), nullptr);
    *reinterpret_cast<uint32_t*>(result.Get()) = 2U;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(
        KERNEL_RUN_KF(CheckWeightCopy<LayoutB>, 1, weight.Get(), result.Get(), k, n, windowK, width, kOffset, nOffset));
    EXPECT_EQ(*reinterpret_cast<uint32_t*>(result.Get()), expectedFastPath ? 1U : 0U);
}

template <typename LayoutB>
void CheckFusedAssembly(uint8_t scaleAlg)
{
    using Types = FusedTypes<LayoutB>;
    using Kernel = typename Types::Kernel;
    using Epilogue = typename Types::Epilogue;
    static_assert(std::is_same_v<typename Types::Policy::ScheduleType, Blaze::Gemm::KernelGmmSwiGluMixMx>);
    static_assert(std::is_same_v<typename Types::Mmad::CType, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeIn, float>);
    typename Kernel::GMMTiling tiling{2, 129, N, K_ALIGN, 16, 64, 64, 128, 128, 2, 2, 2, 0, 1, 1};
    EXPECT_EQ(tiling.groupNum, 2U);
    EXPECT_EQ(tiling.singleW, 1U);
    typename Epilogue::Params params{};
    params.swigluMode = 2;
    params.scaleAlg = scaleAlg;
    params.gluAlpha = 1.5F;
    EXPECT_EQ(params.swigluMode, 2);
    EXPECT_EQ(params.scaleAlg, scaleAlg);
    EXPECT_FLOAT_EQ(params.gluAlpha, 1.5F);
}

} // namespace

TEST(QgmmSwigluMxTest, AssembleNzScaleAlg0) { CheckFusedAssembly<asc::te::nz_layout_ptn>(0); }

TEST(QgmmSwigluMxTest, LegacyFiveFieldParamsPreserveScaleAlg)
{
    using Epilogue = FusedTypes<asc::te::nz_layout_ptn>::Epilogue;
    for (uint8_t scaleAlg : {uint8_t{0}, uint8_t{1}}) {
        Epilogue::Params params{nullptr, nullptr, 16U, 64U, scaleAlg};
        EXPECT_EQ(params.baseM, 16U);
        EXPECT_EQ(params.baseN, 64U);
        EXPECT_EQ(params.scaleAlg, scaleAlg);
        EXPECT_EQ(params.swigluMode, 0);
        EXPECT_FLOAT_EQ(params.clampLimit, Blaze::Epilogue::Block::Constant::DEFAULT_CLAMP_LIMIT);
        EXPECT_FLOAT_EQ(params.gluAlpha, Blaze::Epilogue::Block::Constant::DEFAULT_GLU_ALPHA);
        EXPECT_FLOAT_EQ(params.gluBias, Blaze::Epilogue::Block::Constant::DEFAULT_GLU_BIAS);
        EXPECT_FLOAT_EQ(params.dstTypeMax, 0.0F);
    }
}

TEST(QgmmSwigluMxTest, LegacyInactiveFp4TemplateCanBeInstantiated)
{
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp4x2_e2m1_t, float, fp8_e8m0_t>;
    EXPECT_GT(sizeof(Epilogue), 0U);
    Epilogue::Params params{nullptr, nullptr, 16U, 64U, uint8_t{1}};
    EXPECT_EQ(params.scaleAlg, 1U);
    EXPECT_EQ(params.swigluMode, 0);
}
TEST(QgmmSwigluMxTest, AssembleNzScaleAlg1) { CheckFusedAssembly<asc::te::nz_layout_ptn>(1); }
TEST(QgmmSwigluMxTest, AssembleZnScaleAlg0) { CheckFusedAssembly<asc::te::zn_layout_ptn>(0); }
TEST(QgmmSwigluMxTest, AssembleZnScaleAlg1) { CheckFusedAssembly<asc::te::zn_layout_ptn>(1); }

TEST(QgmmSwigluMxTest, AlignedNzCopy)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(K_ALIGN, N, K_ALIGN, N / 2U, 0, 0, true);
}

TEST(QgmmSwigluMxTest, AlignedZnCopy)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(K_ALIGN, N, K_ALIGN, N / 2U, 0, 0, true);
}

TEST(QgmmSwigluMxTest, OffsetNzCopy)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(256, 256, K_ALIGN, N / 2U, K_ALIGN, 32, true);
}

TEST(QgmmSwigluMxTest, OffsetZnCopy)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(256, 256, K_ALIGN, N / 2U, K_ALIGN, 16, true);
}

TEST(QgmmSwigluMxTest, TailK68NzFallsBack)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(68, N, 68, N / 2U, 0, 0, false);
}

TEST(QgmmSwigluMxTest, TailK130ZnFallsBack)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(130, N, 130, N / 2U, 0, 0, false);
}

TEST(QgmmSwigluMxTest, ShortKWindowNzFallsBack)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(256, N, 64, N / 2U, 0, 0, false);
}

TEST(QgmmSwigluMxTest, ShortKWindowZnFallsBack)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(256, N, 64, N / 2U, 0, 0, false);
}

TEST(QgmmSwigluMxTest, UnalignedKOffsetNzFallsBack)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(256, N, K_ALIGN, N / 2U, 64, 0, false);
}

TEST(QgmmSwigluMxTest, UnalignedKOffsetZnFallsBack)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(256, N, K_ALIGN, N / 2U, 64, 0, false);
}

TEST(QgmmSwigluMxTest, UnalignedNnzFallsBack)
{
    RunWeightCopyCase<asc::te::nz_layout_ptn>(K_ALIGN, 96, K_ALIGN, 32, 0, 0, false);
}

TEST(QgmmSwigluMxTest, UnalignedNznFallsBack)
{
    RunWeightCopyCase<asc::te::zn_layout_ptn>(K_ALIGN, 96, K_ALIGN, 32, 0, 0, false);
}

TEST(QgmmSwigluMxTest, OldNdCopyNotRoutedToNzFastPath)
{
    RunWeightCopyCase<asc::te::nd_ext_layout_ptn>(K_ALIGN, N, K_ALIGN, N / 2U, 0, 0, false);
}

TEST(QgmmSwigluMxTest, OldDnCopyNotRoutedToNzFastPath)
{
    RunWeightCopyCase<asc::te::dn_ext_layout_ptn>(K_ALIGN, N, K_ALIGN, N / 2U, 0, 0, false);
}
