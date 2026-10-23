/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_fused_mat_mul.cpp
 * \brief FusedMatMul Kernel UT测试用例
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "fused_mat_mul.cpp"

namespace {

struct ScaleAddCase {
    uint32_t batch;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t nL1;
    float alpha;
    float beta;
};

struct SigmoidCase {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t nL1;
    bool hasBias;
};

void ReadBinToGm(const std::string& path, GM_ADDR addr, size_t size)
{
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << path;
    file.read(reinterpret_cast<char*>(addr), static_cast<std::streamsize>(size));
    ASSERT_EQ(file.gcount(), static_cast<std::streamsize>(size)) << "Unexpected file size: " << path;
}

} // namespace

class FusedMatMulTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR +
                               "/fused_mat_mul/fused_mat_mul_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        x1GM = nullptr;
        x2GM = nullptr;
        x3GM = nullptr;
        yGM = nullptr;
        workspaceGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (x1GM)
            AscendC::GmFree((void*)x1GM);
        if (x2GM)
            AscendC::GmFree((void*)x2GM);
        if (x3GM)
            AscendC::GmFree((void*)x3GM);
        if (yGM)
            AscendC::GmFree((void*)yGM);
        if (workspaceGM)
            AscendC::GmFree((void*)workspaceGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    template <typename ElementType>
    void RunScaleAddCase(const ScaleAddCase& testCase)
    {
        static_assert(std::is_same_v<ElementType, half> || std::is_same_v<ElementType, bfloat16_t>);
        const uint32_t blockNum = 1;
        const size_t x1Size = static_cast<size_t>(testCase.batch) * testCase.m * testCase.k * sizeof(ElementType);
        const size_t x2Size = static_cast<size_t>(testCase.batch) * testCase.k * testCase.n * sizeof(ElementType);
        const size_t outputSize = static_cast<size_t>(testCase.batch) * testCase.m * testCase.n * sizeof(ElementType);
        const size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

        x1GM = (GM_ADDR)AscendC::GmAlloc(x1Size);
        x2GM = (GM_ADDR)AscendC::GmAlloc(x2Size);
        x3GM = (GM_ADDR)AscendC::GmAlloc(outputSize);
        yGM = (GM_ADDR)AscendC::GmAlloc(outputSize);
        workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
        tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(FusedMatMulUT::FusedMatMulTilingData));

        ASSERT_NE(x1GM, nullptr);
        ASSERT_NE(x2GM, nullptr);
        ASSERT_NE(x3GM, nullptr);
        ASSERT_NE(yGM, nullptr);
        ASSERT_NE(workspaceGM, nullptr);
        ASSERT_NE(tilingGM, nullptr);

        std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/fused_mat_mul/fused_mat_mul_data";
        std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
        std::string dtype = std::is_same_v<ElementType, half> ? "float16" : "bfloat16";
        std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " +
                                 std::to_string(testCase.m) + " --n " + std::to_string(testCase.n) + " --k " +
                                 std::to_string(testCase.k) + " --batch " + std::to_string(testCase.batch) +
                                 " --dtype " + dtype + " --alpha " + std::to_string(testCase.alpha) + " --beta " +
                                 std::to_string(testCase.beta);
        int genRet = system(genCmd.c_str());
        ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in fused_mat_mul_data";
        genRet = system(genDataCmd.c_str());
        ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM, x1Size));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM, x2Size));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_x3.bin", x3GM, outputSize));
        memset(yGM, 0, outputSize);
        memset(workspaceGM, 0, workspaceSize);

        auto* tilingData = reinterpret_cast<FusedMatMulUT::FusedMatMulTilingData*>(tilingGM);
        memset(tilingData, 0, sizeof(FusedMatMulUT::FusedMatMulTilingData));
        auto& batchTiling = tilingData->matMulTilingData;
        auto& matmulTiling = batchTiling.matMulTilingData;
        matmulTiling.usedCoreNum = blockNum;
        matmulTiling.m = testCase.m;
        matmulTiling.n = testCase.n;
        matmulTiling.k = testCase.k;
        matmulTiling.mL1 = 16;
        matmulTiling.nL1 = testCase.nL1;
        matmulTiling.kL1 = 16;
        matmulTiling.baseM = 16;
        matmulTiling.baseN = 16;
        matmulTiling.baseK = 16;
        matmulTiling.skSingleCoreK = testCase.k;
        matmulTiling.mTailCnt = 1;
        matmulTiling.nTailCnt = 1;
        matmulTiling.mBaseTailSplitCnt = 1;
        matmulTiling.nBaseTailSplitCnt = 1;
        matmulTiling.mTailMain = testCase.m;
        matmulTiling.nTailMain = testCase.n;
        matmulTiling.mmadParam = 0;
        matmulTiling.l1BufferNum = 1;
        matmulTiling.l0cDB = 1;
        matmulTiling.ubDB = 1;
        matmulTiling.l2CacheDisable = FusedMatMulUT::L2CacheMode::L2_CACHE_DEFAULT;
        matmulTiling.sliceM = testCase.m;
        matmulTiling.srcNdStride = 1;
        matmulTiling.rowStride = 1;
        matmulTiling.innerBatch = 1;
        batchTiling.batchDimAll = testCase.batch;
        batchTiling.batchX3 = testCase.batch;
        tilingData->alpha = testCase.alpha;
        tilingData->beta = testCase.beta;

        AscendC::SetKernelMode(KernelMode::MIX_MODE);
        auto kernelFunc = fused_mat_mul_kernel_entry<ElementType>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, x1GM, x2GM, x3GM, yGM, workspaceGM, tilingGM))
            << "Kernel execution failed: one or more cores exited with non-zero status";
    }

    template <typename ElementType, typename BiasType, bool TransA, bool TransB>
    void RunSigmoidCase(const SigmoidCase& testCase)
    {
        const uint32_t blockNum = 1;
        const size_t x1Size = static_cast<size_t>(testCase.m) * testCase.k * sizeof(ElementType);
        const size_t x2Size = static_cast<size_t>(testCase.k) * testCase.n * sizeof(ElementType);
        const size_t outputCount = static_cast<size_t>(testCase.m) * testCase.n;
        const size_t outputSize = outputCount * sizeof(ElementType);
        const size_t biasSize = testCase.n * sizeof(BiasType);
        const size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;
        x1GM = (GM_ADDR)AscendC::GmAlloc(x1Size);
        x2GM = (GM_ADDR)AscendC::GmAlloc(x2Size);
        yGM = (GM_ADDR)AscendC::GmAlloc(outputSize);
        workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
        tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(FusedMatMulUT::FusedMatMulTilingData));
        ASSERT_NE(x1GM, nullptr);
        ASSERT_NE(x2GM, nullptr);
        ASSERT_NE(yGM, nullptr);
        ASSERT_NE(workspaceGM, nullptr);
        ASSERT_NE(tilingGM, nullptr);
        if (testCase.hasBias) {
            x3GM = (GM_ADDR)AscendC::GmAlloc(biasSize);
            ASSERT_NE(x3GM, nullptr);
        }

        const std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/fused_mat_mul/fused_mat_mul_data";
        const std::string dtype = std::is_same_v<ElementType, half> ? "float16" : "bfloat16";
        const std::string biasDtype = !testCase.hasBias ? "none" : std::is_same_v<BiasType, float> ? "float32" : "same";
        const std::string genDataCmd = "cd " + dataDir + " && python3 gen_data.py --optype sigmoid --m " +
                                       std::to_string(testCase.m) + " --n " + std::to_string(testCase.n) + " --k " +
                                       std::to_string(testCase.k) + " --dtype " + dtype + " --bias_dtype " + biasDtype +
                                       (TransA ? " --trans_a" : "") + (TransB ? " --trans_b" : "");
        ASSERT_EQ(system(genDataCmd.c_str()), 0) << "Failed to generate sigmoid data";
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM, x1Size));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM, x2Size));
        if (testCase.hasBias) {
            ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_bias.bin", x3GM, biasSize));
        }
        std::vector<ElementType> golden(outputCount);
        ASSERT_NO_FATAL_FAILURE(
            ReadBinToGm(dataDir + "/golden_c.bin", reinterpret_cast<GM_ADDR>(golden.data()), outputSize));
        memset(yGM, 0, outputSize);
        memset(workspaceGM, 0, workspaceSize);

        auto* tilingData = reinterpret_cast<FusedMatMulUT::FusedMatMulTilingData*>(tilingGM);
        memset(tilingData, 0, sizeof(FusedMatMulUT::FusedMatMulTilingData));
        auto& tiling = tilingData->matMulTilingData.matMulTilingData;
        tiling.usedCoreNum = blockNum;
        tiling.m = testCase.m;
        tiling.n = testCase.n;
        tiling.k = testCase.k;
        tiling.mL1 = 16;
        tiling.nL1 = testCase.nL1;
        tiling.kL1 = 16;
        tiling.baseM = 16;
        tiling.baseN = 16;
        tiling.baseK = 16;
        tiling.skSingleCoreK = testCase.k;
        tiling.mTailCnt = 1;
        tiling.nTailCnt = 1;
        tiling.mBaseTailSplitCnt = 1;
        tiling.nBaseTailSplitCnt = 1;
        tiling.mTailMain = testCase.m;
        tiling.nTailMain = testCase.n;
        tiling.l1BufferNum = 1;
        tiling.l0cDB = 1;
        tiling.ubDB = 1;
        tiling.sliceM = testCase.m;
        tiling.srcNdStride = 1;
        tiling.rowStride = 1;
        tiling.innerBatch = 1;
        tilingData->matMulTilingData.batchDimAll = 1;

        AscendC::SetKernelMode(KernelMode::MIX_MODE);
        auto kernelFunc = fused_mat_mul_sigmoid_kernel_entry<ElementType, BiasType, TransA, TransB>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, x1GM, x2GM, x3GM, yGM, workspaceGM, tilingGM));

        // Enable on simulators that commit Cube/Fixpipe results, following the GemmSyrk UT convention.
        const char* verifyOutput = getenv("FUSED_MATMUL_UT_VERIFY_OUTPUT");
        if (verifyOutput == nullptr || verifyOutput[0] != '1') {
            return;
        }
        const auto* output = reinterpret_cast<const ElementType*>(yGM);
        const float tolerance = std::is_same_v<ElementType, half> ? 2e-3F : 1e-2F;
        for (size_t i = 0; i < outputCount; ++i) {
            const float actual = static_cast<float>(output[i]);
            ASSERT_TRUE(std::isfinite(actual)) << "Non-finite output at " << i;
            EXPECT_GE(actual, 0.0F) << "Output index " << i;
            EXPECT_LE(actual, 1.0F) << "Output index " << i;
            ASSERT_NEAR(actual, static_cast<float>(golden[i]), tolerance) << "Output index " << i;
        }
    }

    static constexpr size_t WORKSPACE_TILE_SIZE = 256UL * 256 * 4;
    static constexpr size_t WORKSPACE_OVERHEAD = 20UL * 1024 * 1024;
    GM_ADDR x1GM;
    GM_ADDR x2GM;
    GM_ADDR x3GM;
    GM_ADDR yGM;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
};

TEST_F(FusedMatMulTest, Test_FP16_ScaleAdd_BothScales_NTail) { RunScaleAddCase<half>({112, 3, 1, 10, 16, 3.0F, 2.0F}); }

TEST_F(FusedMatMulTest, Test_FP16_ScaleAdd_BothScales_M1Sync)
{
    // 使用较小batch覆盖[8176, 1, 3] * [8176, 3, 16]场景中第二个AIV无有效行时的同步流程。
    RunScaleAddCase<half>({8, 1, 16, 3, 16, 3.687209F, 2.067589F});
}

TEST_F(FusedMatMulTest, Test_FP16_ScaleAdd_AlphaOnly_MultiNTile)
{
    RunScaleAddCase<half>({2, 3, 17, 16, 32, 2.0F, 1.0F});
}

TEST_F(FusedMatMulTest, Test_FP16_ScaleAdd_BetaOnly) { RunScaleAddCase<half>({2, 2, 8, 4, 16, 1.0F, 2.0F}); }

TEST_F(FusedMatMulTest, Test_BF16_ScaleAdd_DefaultScales) { RunScaleAddCase<bfloat16_t>({2, 2, 3, 4, 16, 1.0F, 1.0F}); }

TEST_F(FusedMatMulTest, Test_FP16_Sigmoid_NN_NoBias_MNKTail)
{
    RunSigmoidCase<half, half, false, false>({17, 19, 23, 16, false});
}

TEST_F(FusedMatMulTest, Test_FP16_Sigmoid_TN_FP32Bias_M1Sync)
{
    RunSigmoidCase<half, float, true, false>({1, 17, 19, 16, true});
}

TEST_F(FusedMatMulTest, Test_BF16_Sigmoid_NT_BF16Bias_OddM)
{
    RunSigmoidCase<bfloat16_t, bfloat16_t, false, true>({3, 17, 19, 16, true});
}

TEST_F(FusedMatMulTest, Test_BF16_Sigmoid_TT_FP32Bias_MultiTile)
{
    RunSigmoidCase<bfloat16_t, float, true, true>({17, 33, 35, 32, true});
}
