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
 * \file test_confidence_head_fused.cpp
 * \brief ConfidenceHeadFused template assembly, scheduling and mixed-kernel UT.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "blaze_kernel_stub.h"
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "confidence_head_fused.h"

namespace {

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(std::max<size_t>(bytes, 32)))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

template <typename ElementType>
void FillInput(GM_ADDR addr, size_t count)
{
    auto* data = reinterpret_cast<ElementType*>(addr);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<ElementType>((static_cast<int32_t>(i % 17) - 8) * 0.03125F);
    }
}

template <bool HasMarkov, typename ElementType>
void RunKernelSmoke(const ConfidenceHeadFusedUT::TestTiling& tiling)
{
    const size_t hiddenElems = static_cast<size_t>(tiling.n) * tiling.kEmbed;
    const size_t markovElems = static_cast<size_t>(tiling.n) * tiling.kMarkov;
    const size_t weightElems = static_cast<size_t>(tiling.kEmbed) + tiling.kMarkov;
    GmBuffer hidden(hiddenElems * sizeof(ElementType));
    GmBuffer weight(weightElems * sizeof(ElementType));
    GmBuffer markov(markovElems * sizeof(ElementType));
    GmBuffer temperature(sizeof(float));
    GmBuffer bias(tiling.biasN * sizeof(float));
    GmBuffer output(tiling.n * sizeof(float));
    GmBuffer tilingBuffer(sizeof(tiling));
    ASSERT_NE(hidden.Get(), nullptr);
    ASSERT_NE(weight.Get(), nullptr);
    ASSERT_NE(markov.Get(), nullptr);
    ASSERT_NE(temperature.Get(), nullptr);
    ASSERT_NE(bias.Get(), nullptr);
    ASSERT_NE(output.Get(), nullptr);
    ASSERT_NE(tilingBuffer.Get(), nullptr);
    FillInput<ElementType>(hidden.Get(), hiddenElems);
    FillInput<ElementType>(weight.Get(), weightElems);
    FillInput<ElementType>(markov.Get(), markovElems);
    FillInput<float>(bias.Get(), tiling.biasN);
    *reinterpret_cast<float*>(temperature.Get()) = tiling.temperature;
    std::memset(output.Get(), 0, tiling.n * sizeof(float));
    std::memcpy(tilingBuffer.Get(), &tiling, sizeof(tiling));

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = ConfidenceHeadFusedUT::ConfidenceHeadFusedKernelEntry<HasMarkov, ElementType>;
    GM_ADDR markovAddr = HasMarkov ? markov.Get() : nullptr;
    GM_ADDR temperatureAddr = tiling.hasTemperature ? temperature.Get() : nullptr;
    GM_ADDR biasAddr = tiling.biasN != 0 ? bias.Get() : nullptr;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, tiling.usedCoreNum, hidden.Get(), weight.Get(), markovAddr, temperatureAddr,
                              biasAddr, output.Get(), tilingBuffer.Get()));
    // The CPU simulator does not model all Ascend950 MMAD/Reg numerics. Check launch and synchronization here;
    // numerical accuracy is verified by the on-device ST cases.
}

template <typename OutputType>
void RunSigmoidSmoke()
{
    constexpr size_t N = 17;
    GmBuffer input(N * sizeof(float));
    GmBuffer output(N * sizeof(OutputType));
    ASSERT_NE(input.Get(), nullptr);
    ASSERT_NE(output.Get(), nullptr);
    FillInput<float>(input.Get(), N);
    std::memset(output.Get(), 0, N * sizeof(OutputType));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = ConfidenceHeadFusedUT::SigmoidKernelEntry<OutputType>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1, input.Get(), output.Get()));
}

} // namespace

class ConfidenceHeadFusedTest : public testing::Test {};

TEST_F(ConfidenceHeadFusedTest, FP16NTailPingPong)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 257;
    tiling.kEmbed = 17;
    RunKernelSmoke<false, half>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, FP16MultiKLoopBias)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.kEmbed = 512;
    tiling.biasN = 128;
    RunKernelSmoke<false, half>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, BF16MarkovMultiKLoopBias)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 258;
    tiling.kEmbed = 257;
    tiling.kMarkov = 273;
    tiling.biasN = 129;
    tiling.usedCoreNum = 2;
    RunKernelSmoke<true, bfloat16_t>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, FP16MarkovOnlyBiasZeroTemperatureLowClamp)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 33;
    tiling.kEmbed = 0;
    tiling.kMarkov = 17;
    tiling.biasN = 33;
    tiling.hasTemperature = true;
    tiling.temperature = 0.0F;
    tiling.hasLowLimit = true;
    tiling.lowLimit = -0.25F;
    RunKernelSmoke<true, half>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, BF16MarkovKTailBatchBias)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 258;
    tiling.kEmbed = 17;
    tiling.kMarkov = 19;
    tiling.biasN = 129;
    tiling.usedCoreNum = 2;
    tiling.hasTemperature = true;
    tiling.temperature = 2.0F;
    tiling.hasLowLimit = true;
    tiling.hasHighLimit = true;
    tiling.lowLimit = -0.25F;
    tiling.highLimit = 0.25F;
    RunKernelSmoke<true, bfloat16_t>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, BF16ZeroKBiasZeroTemperatureHighClamp)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 258;
    tiling.kEmbed = 0;
    tiling.biasN = 129;
    tiling.hasTemperature = true;
    tiling.temperature = 0.0F;
    tiling.hasHighLimit = true;
    tiling.highLimit = -0.25F;
    RunKernelSmoke<false, bfloat16_t>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, FP16ZeroKDefaultEpilogue)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 17;
    tiling.kEmbed = 0;
    RunKernelSmoke<false, half>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, FP16EmptyN)
{
    ConfidenceHeadFusedUT::TestTiling tiling;
    tiling.n = 0;
    RunKernelSmoke<false, half>(tiling);
}

TEST_F(ConfidenceHeadFusedTest, SigmoidOutputTypes)
{
    ASSERT_NO_FATAL_FAILURE(RunSigmoidSmoke<float>());
    ASSERT_NO_FATAL_FAILURE(RunSigmoidSmoke<half>());
    ASSERT_NO_FATAL_FAILURE(RunSigmoidSmoke<bfloat16_t>());
}
