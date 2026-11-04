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
 *  \brief Host checks for the fused QGMM SwiGLU MX template.
 */

#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"

#include "blaze_kernel_stub.h"
#include "quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/gemm/policy/dispatch_policy.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace {
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using NdLayout = asc::te::nd_ext_layout_ptn;
constexpr uint32_t K_ALIGN = 128;
constexpr uint32_t N = 128;

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
