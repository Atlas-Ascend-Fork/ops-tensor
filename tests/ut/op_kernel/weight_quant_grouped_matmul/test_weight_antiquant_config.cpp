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
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "blaze/gemm/kernel/kernel_universal.h"

namespace {
template <class AType_, class BType_, bool TransB_, bool HasOffset_>
void CheckAntiquantConfig()
{
    using Nd = asc::te::nd_ext_layout_ptn;
    using WeightLayout = AscendC::Std::conditional_t<TransB_, Nd, asc::te::dn_ext_layout_ptn>;
    constexpr uint64_t AIV_COUNT = 1;
    constexpr uint32_t UB_BUFFER_COUNT = 2;
    // Shared Policy compatibility argument; GMM UB geometry comes from GroupedWeightAntiquantConfig.
    constexpr uint32_t POLICY_UB_INNER_BYTES = 256;
    using Policy = Blaze::Gemm::MatmulWithWeightAntiquant<AIV_COUNT, POLICY_UB_INNER_BYTES, UB_BUFFER_COUNT,
                                                          Blaze::Gemm::QuantMode::PERCHANNEL_MODE, HasOffset_,
                                                          Blaze::Gemm::KernelWqgmmMixAntiquant>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType_, Nd, AscendC::Std::tuple<BType_, AType_>,
                                               AscendC::Std::tuple<WeightLayout, Nd>, AType_, Nd, AType_, Nd>;
    using Config = Blaze::Gemm::Kernel::GroupedWeightAntiquantConfig<Mmad>;
    using Views = typename Config::ParameterViews;
    EXPECT_EQ(Policy::AIV_NUM, 1U);
    EXPECT_EQ(Policy::SyncProtocol::MODE, 4U);
    EXPECT_EQ(Policy::SyncProtocol::AIC_FREE_FLAG, 8U);
    EXPECT_EQ(Policy::SyncProtocol::AIV_READY_FLAG, 9U);
    EXPECT_EQ(Config::INPUT_BYTES, 65536U);
    EXPECT_EQ(Config::SCALE_BYTES, 512U);
    EXPECT_EQ(Config::OFFSET_BYTES, 512U);
    EXPECT_EQ(Config::INPUT_SLOT_BYTES, 66560U);
    EXPECT_EQ(Config::OUTPUT_SLOT_BYTES, 66048U);
    EXPECT_EQ(Config::OUTPUT_BASE, 133120U);
    const uint64_t ubBytes = Config::OUTPUT_BASE + Policy::UB_MTE2_BUFFER_NUM * Config::OUTPUT_SLOT_BYTES;
    EXPECT_EQ(ubBytes, 265216U);
    // SDK capacity for the UT compilation target; CPU debug does not query an attached device.
    const uint64_t availableUbBytes = AscendC::GetUBSizeInBytes();
    ASSERT_GT(availableUbBytes, 0U) << "UB capacity is unavailable for " << UT_SOC_VERSION;
    EXPECT_LE(ubBytes, availableUbBytes) << "Weight antiquant buffers exceed UB capacity for " << UT_SOC_VERSION;
    EXPECT_EQ(Config::INPUT_INNER_SIZE, 256U);
    EXPECT_EQ(Config::VF_K, TransB_ ? 256 : 128);
    EXPECT_EQ(Config::VF_N, TransB_ ? 128 : 256);

    auto weight = Config::MakeWeightLayout(4L, 529L, 273L);
    EXPECT_EQ(int64_t(weight(asc::te::make_coord(3L, asc::te::make_coord(528L, 272L)))),
              3L * 529 * 273 + (TransB_ ? 272L * 529 + 528 : 528L * 273 + 272));
    auto input = Config::MakeInputLayout();
    EXPECT_EQ(int64_t(input(asc::te::make_coord(1L, 2L))), TransB_ ? 513 : 258);
    auto output = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AType_>(uint64_t{0}),
                                       Config::MakeOutputLayout());
    auto tail = output.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(17L, 19L));
    EXPECT_EQ(int64_t(tail.layout()(asc::te::make_coord(16L, 16L))), 129 * 16 + 16 * 16);
    EXPECT_EQ(asc::te::size<0>(tail.layout()), TransB_ ? 32 : 17);
    EXPECT_EQ(asc::te::size<1>(tail.layout()), TransB_ ? 19 : 32);

    AType_ paramStorage[4 * 273]{};
    auto params = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType_*>(paramStorage)),
        Views::MakeGmLayout(4L, 529L, 273L));
    auto task = Views::SelectTask(params, 3L, 256L, 529L, 17L);
    auto segment = Views::SelectInput(task, 512L, 17L, 17L);
    EXPECT_EQ(asc::te::size<0>(segment.layout()), 1);
    EXPECT_EQ(asc::te::size<1>(segment.layout()), 17);
    EXPECT_EQ(int64_t(segment.layout()(asc::te::make_coord(0L, 16L))), 16);
    auto paramUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AType_>(uint64_t{0}),
                                        Views::MakeUbLayout(Config::INPUT_N));
    auto paramTile = Views::SelectTile(paramUb, TransB_ ? 0L : 128L, TransB_ ? 128L : 0L, Config::VF_K, Config::VF_N);
    EXPECT_EQ(asc::te::size<0>(paramTile.layout()), 1);
    EXPECT_EQ(asc::te::size<1>(paramTile.layout()), Config::VF_N);
}

template <class AType_, class BType_, bool HasOffset_ = false>
void CheckBothOrientations()
{
    CheckAntiquantConfig<AType_, BType_, false, HasOffset_>();
    CheckAntiquantConfig<AType_, BType_, true, HasOffset_>();
}

TEST(GroupedWeightAntiquantConfig, StorageLayoutsParametersAndBudget)
{
    CheckBothOrientations<half, int8_t>();
    CheckBothOrientations<half, int8_t, true>();
    CheckBothOrientations<bfloat16_t, int8_t>();
    CheckBothOrientations<bfloat16_t, int8_t, true>();
    CheckBothOrientations<half, float8_e4m3_t>();
    CheckBothOrientations<bfloat16_t, float8_e4m3_t>();
    CheckBothOrientations<half, float8_e5m2_t>();
    CheckBothOrientations<bfloat16_t, float8_e5m2_t>();
    CheckBothOrientations<half, hifloat8_t>();
    CheckBothOrientations<bfloat16_t, hifloat8_t>();
}
} // namespace
