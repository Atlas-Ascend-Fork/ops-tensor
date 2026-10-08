/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_tqbmm_fixpipe_quant.cpp
 * \brief TQBMM fixpipe quant (HIFP8) Kernel UT test cases
 */

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "tqbmm_fixpipe_quant.cpp"

class TqbmmFixpipeQuantTest : public testing::Test {
protected:
    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR +
                               "/tqbmm_fixpipe_quant/tqbmm_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        aGM = nullptr;
        bGM = nullptr;
        scaleGM = nullptr;
        biasGM = nullptr;
        pertokenScaleGM = nullptr;
        cGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (aGM)
            AscendC::GmFree((void*)aGM);
        if (bGM)
            AscendC::GmFree((void*)bGM);
        if (scaleGM)
            AscendC::GmFree((void*)scaleGM);
        if (biasGM)
            AscendC::GmFree((void*)biasGM);
        if (pertokenScaleGM)
            AscendC::GmFree((void*)pertokenScaleGM);
        if (cGM)
            AscendC::GmFree((void*)cGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR scaleGM;
    GM_ADDR biasGM;
    GM_ADDR pertokenScaleGM;
    GM_ADDR cGM;
    GM_ADDR tilingGM;
};

static std::string GetDataDir() { return std::string(UT_KERNEL_SRC_DIR) + "/tqbmm_fixpipe_quant/tqbmm_data"; }

static void GenData(uint32_t m, uint32_t n, uint32_t k, uint32_t batch, bool transBatchA = false)
{
    std::string dir = GetDataDir();
    std::string cmd = "cd " + dir + " && rm -rf *.bin && python3 gen_data.py --m " + std::to_string(m) + " --n " +
                      std::to_string(n) + " --k " + std::to_string(k) + " --batch " + std::to_string(batch);
    if (transBatchA) {
        cmd += " --trans_batch_a";
    }
    ASSERT_EQ(system(cmd.c_str()), 0);
}

template <typename T>
static void ReadBinFile(const std::string& path, void* dst, size_t size)
{
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.good()) << "Failed to open " << path;
    f.read(static_cast<char*>(dst), size);
    ASSERT_TRUE(f.good()) << "Failed to read " << path;
}

// === HIFP8, batch=1, AIC-only ===
TEST_F(TqbmmFixpipeQuantTest, Test_HIFP8_Batch1)
{
    const uint32_t M = 64, N = 128, K = 128, BATCH = 1, blockNum = 1;

    size_t aSize = BATCH * M * K * sizeof(uint8_t);
    size_t bSize = BATCH * K * N * sizeof(uint8_t);
    size_t scaleSize = sizeof(uint64_t);
    size_t pertokenScaleSize = sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    pertokenScaleGM = (GM_ADDR)AscendC::GmAlloc(pertokenScaleSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmFixpipeQuantTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData(M, N, K, BATCH);
    std::string dir = GetDataDir();
    ReadBinFile<uint8_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<uint8_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<uint8_t>(dir + "/scale.bin", scaleGM, scaleSize);
    ReadBinFile<uint8_t>(dir + "/pertoken_scale.bin", pertokenScaleGM, pertokenScaleSize);
    ReadBinFile<uint8_t>(dir + "/bias.bin", biasGM, biasSize);

    auto* td = reinterpret_cast<TqbmmFixpipeQuantTilingData*>(tilingGM);
    TqbmmFixpipeQuantUT::FillTqbmmFixpipeQuantTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_fixpipe_quant_kernel_entry<OP_TYPE_TQBMM_FIXPIPE_QUANT_BASIC, hifloat8_t, hifloat8_t, uint64_t,
                                               half, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, scaleGM, biasGM, pertokenScaleGM, cGM, tilingGM))
        << "TQBMM fixpipe quant HIFP8 batch=1 kernel execution failed";
}

// === HIFP8, batch=2, trans batch A (PERM_X1) ===
TEST_F(TqbmmFixpipeQuantTest, Test_HIFP8_Batch2_TransBatchA)
{
    const uint32_t M = 64, N = 128, K = 128, BATCH = 2, blockNum = 1;

    size_t aSize = M * BATCH * K * sizeof(uint8_t);
    size_t bSize = BATCH * K * N * sizeof(uint8_t);
    size_t scaleSize = sizeof(uint64_t);
    size_t pertokenScaleSize = sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    pertokenScaleGM = (GM_ADDR)AscendC::GmAlloc(pertokenScaleSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmFixpipeQuantTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData(M, N, K, BATCH, true);
    std::string dir = GetDataDir();
    ReadBinFile<uint8_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<uint8_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<uint8_t>(dir + "/scale.bin", scaleGM, scaleSize);
    ReadBinFile<uint8_t>(dir + "/pertoken_scale.bin", pertokenScaleGM, pertokenScaleSize);
    ReadBinFile<uint8_t>(dir + "/bias.bin", biasGM, biasSize);

    auto* td = reinterpret_cast<TqbmmFixpipeQuantTilingData*>(tilingGM);
    TqbmmFixpipeQuantUT::FillTqbmmFixpipeQuantTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_fixpipe_quant_kernel_entry<
        OP_TYPE_TQBMM_FIXPIPE_QUANT_TRANS_BATCH_A, hifloat8_t, hifloat8_t, uint64_t, half, float,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, scaleGM, biasGM, pertokenScaleGM, cGM, tilingGM))
        << "TQBMM fixpipe quant HIFP8 batch=2 trans-batch-A kernel execution failed";
}

// === HIFP8, batch=4, multi-core ===
TEST_F(TqbmmFixpipeQuantTest, Test_HIFP8_Batch4_MultiCore)
{
    const uint32_t M = 32, N = 32, K = 64, BATCH = 4, blockNum = 4;

    size_t aSize = M * BATCH * K * sizeof(uint8_t);
    size_t bSize = BATCH * K * N * sizeof(uint8_t);
    size_t scaleSize = sizeof(uint64_t);
    size_t pertokenScaleSize = sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    pertokenScaleGM = (GM_ADDR)AscendC::GmAlloc(pertokenScaleSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TqbmmFixpipeQuantTilingData));
    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(cGM, nullptr);

    GenData(M, N, K, BATCH, true);
    std::string dir = GetDataDir();
    ReadBinFile<uint8_t>(dir + "/input_a.bin", aGM, aSize);
    ReadBinFile<uint8_t>(dir + "/input_b.bin", bGM, bSize);
    ReadBinFile<uint8_t>(dir + "/scale.bin", scaleGM, scaleSize);
    ReadBinFile<uint8_t>(dir + "/pertoken_scale.bin", pertokenScaleGM, pertokenScaleSize);
    ReadBinFile<uint8_t>(dir + "/bias.bin", biasGM, biasSize);

    auto* td = reinterpret_cast<TqbmmFixpipeQuantTilingData*>(tilingGM);
    TqbmmFixpipeQuantUT::FillTqbmmFixpipeQuantTilingDataDefault(*td, M, N, K, BATCH, blockNum);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kf = tqbmm_fixpipe_quant_kernel_entry<
        OP_TYPE_TQBMM_FIXPIPE_QUANT_TRANS_BATCH_A, hifloat8_t, hifloat8_t, uint64_t, half, float,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kf, blockNum, aGM, bGM, scaleGM, biasGM, pertokenScaleGM, cGM, tilingGM))
        << "TQBMM fixpipe quant HIFP8 batch=4 multi-core kernel execution failed";
}
