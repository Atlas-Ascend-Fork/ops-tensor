/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_weight_quant_batch_matmul_mx.cpp
 * \brief CPU-simulation UT for the Weight Quant MX Blaze mixed kernel.
 */

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "weight_quant_batch_matmul_mx.h"

namespace {

constexpr uint64_t FP4_PACK_FACTOR = 2U;

void FillGmBuffer(GM_ADDR address, size_t size, uint8_t value)
{
    std::fill_n(reinterpret_cast<uint8_t*>(address), size, value);
}

uint64_t Align(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) / alignment * alignment;
}

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : address_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}

    ~GmBuffer()
    {
        if (address_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(address_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return address_; }

private:
    GM_ADDR address_{nullptr};
};

struct CaseConfig {
    int64_t m;
    int64_t n;
    int64_t k;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    uint64_t tileShapeKL1;
    uint64_t tileShapeScaleKL1;
    uint64_t kBubSize;
    uint64_t nBubSize;
    uint64_t l1BufferNum;
    bool weightNz;
    bool hasBias;
    bool scaleBNz = false;
};

struct BufferSizes {
    size_t aSize;
    size_t bSize;
    size_t biasSize;
    size_t scaleASize;
    size_t scaleBSize;
    size_t cSize;
};

struct CaseBuffers {
    explicit CaseBuffers(const BufferSizes& sizes)
        : aGm(sizes.aSize),
          bGm(sizes.bSize),
          biasGm(sizes.biasSize),
          scaleAGm(sizes.scaleASize),
          scaleBGm(sizes.scaleBSize),
          cGm(sizes.cSize),
          tilingGm(sizeof(WeightQuantBatchMatmulMxTilingData))
    {}

    GmBuffer aGm;
    GmBuffer bGm;
    GmBuffer biasGm;
    GmBuffer scaleAGm;
    GmBuffer scaleBGm;
    GmBuffer cGm;
    GmBuffer tilingGm;
};

void ReadBinary(const std::string& path, GM_ADDR destination, size_t size)
{
    std::ifstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << "Failed to open " << path;
    stream.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(size));
    ASSERT_EQ(stream.gcount(), static_cast<std::streamsize>(size)) << "Unexpected size for " << path;
}

void GenerateData(const std::string& dataDir, const CaseConfig& config)
{
    std::string command = "cd " + dataDir + " && rm -f *.bin && python3 gen_data.py --m " + std::to_string(config.m) +
                          " --n " + std::to_string(config.n) + " --k " + std::to_string(config.k) +
                          " --weight-layout " + (config.weightNz ? "nz" : "nd") + " --scale-layout " +
                          (config.scaleBNz ? "nz" : "nd") + (config.hasBias ? " --bias" : "");
    ASSERT_EQ(system(command.c_str()), 0) << "Failed to generate Weight Quant MX test data";
}

template <bool WeightNz, bool ScaleBNz>
BufferSizes CalculateBufferSizes(const CaseConfig& config)
{
    const size_t aSize = static_cast<size_t>(config.m * config.k) * sizeof(fp8_e4m3fn_t);
    const uint64_t weightElements = WeightNz ? Align(config.k, 32U) * Align(config.n, 16U) : config.n * config.k;
    const size_t bSize = static_cast<size_t>(weightElements / FP4_PACK_FACTOR) * sizeof(fp4x2_e2m1_t);
    const size_t scaleK = static_cast<size_t>(Align(config.k, 64U) / 32U);
    const size_t scaleASize = static_cast<size_t>(config.m) * scaleK * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleBN = ScaleBNz ? Align(static_cast<uint64_t>(config.n), 16U) : static_cast<uint64_t>(config.n);
    const size_t scaleBSize = scaleK * static_cast<size_t>(scaleBN) * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(config.n) * sizeof(half);
    const size_t cSize = static_cast<size_t>(config.m * config.n) * sizeof(half);
    return BufferSizes{aSize, bSize, biasSize, scaleASize, scaleBSize, cSize};
}

void ValidateBuffers(const CaseBuffers& buffers)
{
    ASSERT_NE(buffers.aGm.Get(), nullptr);
    ASSERT_NE(buffers.bGm.Get(), nullptr);
    ASSERT_NE(buffers.biasGm.Get(), nullptr);
    ASSERT_NE(buffers.scaleAGm.Get(), nullptr);
    ASSERT_NE(buffers.scaleBGm.Get(), nullptr);
    ASSERT_NE(buffers.cGm.Get(), nullptr);
    ASSERT_NE(buffers.tilingGm.Get(), nullptr);
}

void LoadInputData(const std::string& dataDir, const CaseConfig& config, const BufferSizes& sizes,
                   const CaseBuffers& buffers)
{
    ASSERT_NO_FATAL_FAILURE(GenerateData(dataDir, config));
    ASSERT_NO_FATAL_FAILURE(ReadBinary(dataDir + "/input_a.bin", buffers.aGm.Get(), sizes.aSize));
    ASSERT_NO_FATAL_FAILURE(ReadBinary(dataDir + "/input_b.bin", buffers.bGm.Get(), sizes.bSize));
    ASSERT_NO_FATAL_FAILURE(ReadBinary(dataDir + "/bias.bin", buffers.biasGm.Get(), sizes.biasSize));
    ASSERT_NO_FATAL_FAILURE(ReadBinary(dataDir + "/scale_a.bin", buffers.scaleAGm.Get(), sizes.scaleASize));
    ASSERT_NO_FATAL_FAILURE(ReadBinary(dataDir + "/scale_b.bin", buffers.scaleBGm.Get(), sizes.scaleBSize));
}

void InitializeOutput(const CaseBuffers& buffers, const BufferSizes& sizes)
{
    // Keep the output observable if the CPU simulator provides shared GM pages. Numeric MMAD validation is still
    // reserved for the CSV-driven NPU example because tikicpulib does not model FP8 x FP4 arithmetic.
    FillGmBuffer(buffers.cGm.Get(), sizes.cSize, 0xA5U);
}

WeightQuantBatchMatmulMxTilingData BuildTiling(const CaseConfig& config)
{
    return WeightQuantBatchMatmulMxTilingData{config.m,
                                              config.n,
                                              config.k,
                                              config.baseM,
                                              config.baseN,
                                              config.baseK,
                                              config.tileShapeKL1,
                                              config.tileShapeScaleKL1,
                                              config.kBubSize,
                                              config.nBubSize,
                                              config.l1BufferNum,
                                              config.hasBias ? 1U : 0U,
                                              1U,
                                              1U,
                                              1U,
                                              1U,
                                              0U,
                                              0U};
}

template <bool WeightNz, bool ScaleBNz>
void RunKernel(const CaseBuffers& buffers)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = weight_quant_batch_matmul_mx_kernel_entry<WeightNz, ScaleBNz>;
    // tikicpulib does not model FP8 x FP4 MMAD numerics; validate the mixed-kernel pipeline and synchronization.
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1U, buffers.aGm.Get(), buffers.bGm.Get(), buffers.biasGm.Get(),
                              buffers.scaleAGm.Get(), buffers.scaleBGm.Get(), buffers.cGm.Get(),
                              buffers.tilingGm.Get()))
        << "Weight Quant MX mixed kernel execution failed";
}

template <bool WeightNz, bool ScaleBNz>
void RunCase(const CaseConfig& config)
{
    const BufferSizes sizes = CalculateBufferSizes<WeightNz, ScaleBNz>(config);
    const CaseBuffers buffers(sizes);
    ASSERT_NO_FATAL_FAILURE(ValidateBuffers(buffers));

    const std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/weight_quant_batch_matmul_mx/data";
    ASSERT_NO_FATAL_FAILURE(LoadInputData(dataDir, config, sizes, buffers));
    InitializeOutput(buffers, sizes);

    auto* tiling = reinterpret_cast<WeightQuantBatchMatmulMxTilingData*>(buffers.tilingGm.Get());
    *tiling = BuildTiling(config);
    ASSERT_NO_FATAL_FAILURE((RunKernel<WeightNz, ScaleBNz>(buffers)));
}

class WeightQuantBatchMatmulMxTest : public testing::Test {
protected:
    static void TearDownTestSuite()
    {
        const std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/weight_quant_batch_matmul_mx/data";
        const std::string command = "cd " + dataDir + " && rm -f *.bin";
        static_cast<void>(system(command.c_str()));
    }
};

TEST_F(WeightQuantBatchMatmulMxTest, NdWithoutBias)
{
    const CaseConfig config{32, 40, 128, 32, 32, 64, 64, 64, 64, 64, 2, false, false};
    RunCase<false, false>(config);
}

TEST_F(WeightQuantBatchMatmulMxTest, NzMultiKWithFp16Bias)
{
    const CaseConfig config{32, 40, 128, 32, 32, 64, 64, 64, 64, 64, 2, true, true};
    RunCase<true, false>(config);
}

TEST_F(WeightQuantBatchMatmulMxTest, NdTwoBufferWraparound)
{
    const CaseConfig config{32, 40, 192, 32, 32, 64, 64, 64, 64, 64, 2, false, false};
    RunCase<false, false>(config);
}

TEST_F(WeightQuantBatchMatmulMxTest, NzFourBufferWraparoundWithFp16Bias)
{
    const CaseConfig config{32, 40, 320, 32, 32, 64, 64, 64, 64, 64, 4, true, true};
    RunCase<true, false>(config);
}

TEST_F(WeightQuantBatchMatmulMxTest, NzScaleNzMultiK)
{
    const CaseConfig config{32, 40, 128, 32, 32, 64, 64, 64, 64, 64, 2, true, false, true};
    RunCase<true, true>(config);
}

TEST_F(WeightQuantBatchMatmulMxTest, NzScaleNzWithFp16Bias)
{
    const CaseConfig config{32, 40, 128, 32, 32, 64, 64, 64, 64, 64, 2, true, true, true};
    RunCase<true, true>(config);
}

} // namespace
