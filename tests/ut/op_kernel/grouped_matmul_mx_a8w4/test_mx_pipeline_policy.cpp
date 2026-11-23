/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "grouped_matmul_mx_a8w4.h"

namespace {
using Blaze::Gemm::MxPipelineMode;
template <MxPipelineMode Mode>
using Config = Blaze::Gemm::MxWeightPipelineConfig<static_cast<uint32_t>(Mode), fp8_e4m3fn_t, bfloat16_t>;

TEST(GroupedMatmulMxPipelinePolicyTest, OriginalRetainsDynamicAivContract)
{
    using Policy = Config<MxPipelineMode::ParameterDriven>;
    EXPECT_EQ(Policy::MODE, static_cast<uint32_t>(MxPipelineMode::ParameterDriven));
    EXPECT_EQ(Policy::K_SCHEDULE, Blaze::Gemm::WeightPrologueMxKSchedule::DYNAMIC);
    EXPECT_EQ(Policy::SCALE_B_SOURCE, Blaze::Gemm::WeightPrologueMxScaleBSource::AIV);
    EXPECT_EQ(Policy::WeightStages(false), 2UL);
    EXPECT_EQ(Policy::ScaleWindow(false), 4096UL);
}

TEST(GroupedMatmulMxPipelinePolicyTest, AlignedRetainsDedicatedInputLayout)
{
    using Policy = Config<MxPipelineMode::Aligned>;
    EXPECT_EQ(Policy::MODE, static_cast<uint32_t>(MxPipelineMode::Aligned));
    EXPECT_EQ(Policy::Consumer::L1_M, 256UL);
    EXPECT_EQ(Policy::Consumer::L1_N, 256UL);
    EXPECT_EQ(Policy::WeightStages(false), 2UL);
}

TEST(GroupedMatmulMxPipelinePolicyTest, TripleRetainsStagesAndBiasFallback)
{
    using Policy = Config<MxPipelineMode::OperandTriple>;
    EXPECT_EQ(Policy::MODE, static_cast<uint32_t>(MxPipelineMode::OperandTriple));
    EXPECT_EQ(Policy::WeightStages(false), 3UL);
    EXPECT_EQ(Policy::WeightStages(true), 2UL);
    EXPECT_EQ(Policy::ScaleWindow(false), 4096UL);
    EXPECT_EQ(Policy::CreditLimit(1024UL, 256UL, false), 3UL);
}

TEST(GroupedMatmulMxPipelinePolicyTest, QuadRetainsWindowAndBiasFallback)
{
    using Policy = Config<MxPipelineMode::WeightQuad>;
    EXPECT_EQ(Policy::MODE, static_cast<uint32_t>(MxPipelineMode::WeightQuad));
    EXPECT_EQ(Policy::WeightStages(false), 4UL);
    EXPECT_EQ(Policy::ScaleWindow(false), 1024UL);
    EXPECT_EQ(Policy::WeightStages(true), 2UL);
    EXPECT_EQ(Policy::ScaleWindow(true), 4096UL);
    EXPECT_EQ(Policy::CreditLimit(1280UL, 256UL, false), 4UL);
    EXPECT_EQ(Policy::CreditLimit(0UL, 256UL, false), 0UL);
    EXPECT_EQ(Policy::CreditLimit(1024UL, 0UL, false), 0UL);
}
} // namespace
