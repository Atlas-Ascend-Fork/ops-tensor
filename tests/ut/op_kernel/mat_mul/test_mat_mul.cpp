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
 * \file test_mat_mul.cpp
 * \brief MatMulV3 Kernel UT测试用例
 */

#include <cstring>
#include <fstream>
#include <string>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "mat_mul.cpp"

class MatMulV3Test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/mat_mul/matmul_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        aGM = nullptr;
        bGM = nullptr;
        biasGM = nullptr;
        cGM = nullptr;
        workspaceGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (aGM)
            AscendC::GmFree((void*)aGM);
        if (bGM)
            AscendC::GmFree((void*)bGM);
        if (biasGM)
            AscendC::GmFree((void*)biasGM);
        if (cGM)
            AscendC::GmFree((void*)cGM);
        if (workspaceGM)
            AscendC::GmFree((void*)workspaceGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR cGM;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
};

constexpr size_t WORKSPACE_TILE_SIZE = 256UL * 256 * 4;
constexpr size_t WORKSPACE_OVERHEAD = 20UL * 1024 * 1024;

TEST_F(MatMulV3Test, Test_FP16_StreamK)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->skSingleCoreK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 2;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_STREAMK, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_StreamK_L1Buffer4)
{
    const int64_t M = 16;
    const int64_t N = 128;
    const int64_t K = 64;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(half);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 128 --k 64 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 128;
    tilingData->k = 64;
    tilingData->mL1 = 16;
    tilingData->nL1 = 128;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 128;
    tilingData->baseK = 16;
    tilingData->skSingleCoreK = 64;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 4;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_STREAMK, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_Basic)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->skSingleCoreK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 1;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_BASIC, half, half, half, half>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP32_StreamK_MultiCore)
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;
    const uint32_t blockNum = 4;

    size_t aSize = M * K * sizeof(float);
    size_t bSize = K * N * sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(float);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 32 --n 32 --k 32 --dtype float32";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 32;
    tilingData->n = 32;
    tilingData->k = 32;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->skSingleCoreK = 16;
    tilingData->mTailCnt = 0;
    tilingData->nTailCnt = 0;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 2;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_STREAMK, float, float, float, float,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP32_StreamK_SingleBatch)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 32;
    const uint32_t blockNum = 4;

    size_t aSize = M * K * sizeof(float);
    size_t bSize = K * N * sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(float);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    float* aPtr = reinterpret_cast<float*>(aGM);
    float* bPtr = reinterpret_cast<float*>(bGM);
    float* cPtr = reinterpret_cast<float*>(cGM);
    for (int64_t i = 0; i < M * K; ++i) {
        aPtr[i] = 1.0f;
    }
    for (int64_t i = 0; i < K * N; ++i) {
        bPtr[i] = 1.0f;
    }
    for (int64_t i = 0; i < M * N; ++i) {
        cPtr[i] = -1.0f;
    }
    std::memset(biasGM, 0, biasSize);

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 8;
    tilingData->skSingleCoreK = 8;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 2;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;
    tilingData->batch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_STREAMK, float, float, float, float,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";

    tilingData->batch = 1;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_Basic_Slice)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const int64_t SLICE_M = 2;
    const int64_t ORI_M = 4;
    const uint32_t blockNum = 1;
    const int64_t srcNdStride = ORI_M * K;
    const int64_t sliceBatch = M / SLICE_M;
    const int64_t aStorageM = sliceBatch * ORI_M;
    const int64_t aStorageElements = aStorageM * K;

    size_t aSize = aStorageElements * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " + std::to_string(aStorageM) +
                             " --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->skSingleCoreK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 1;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = SLICE_M;
    tilingData->srcNdStride = srcNdStride;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<
        OP_TYPE_MATMUL_BASIC, half, half, half, half, Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_AFullLoad)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 64;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 4;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_AFULLLOAD, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0, 0, MatMulV3BasicTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_BFullLoad)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 64;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 4;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_BFULLLOAD, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0, 0, MatMulV3BasicTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_FixpipeOpt)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 64;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 4;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_FIXPIPE_OPT, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2, 0, 0, MatMulV3BasicTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_FixpipeOpt_UBDB)
{
    const int64_t M = 32;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t aSize = M * K * sizeof(half);
    size_t bSize = K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 32 --n 16 --k 16 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BasicTilingData* tilingData = reinterpret_cast<MatMulV3BasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 32;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 64;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 4;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 2;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_FIXPIPE_OPT, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2, 0, 0, MatMulV3BasicTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_BmmBroadCast)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t batchA = 2;
    const uint32_t batchB = 2;
    const uint32_t batchC = 2;
    const uint32_t blockNum = 1;

    size_t aSize = batchA * M * K * sizeof(half);
    size_t bSize = batchB * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = batchC * M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3BmmBroadcastTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m 16 --n 16 --k 16 --batch " +
                             std::to_string(batchC) + " --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3BmmBroadcastTilingData* tilingData = reinterpret_cast<MatMulV3BmmBroadcastTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->mTailCnt = 1;
    tilingData->nTailCnt = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 1;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 16;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 0;

    tilingData->aBatchDim0 = batchA;
    tilingData->aBatchDim1 = 1;
    tilingData->aBatchDim2 = 1;
    tilingData->aBatchDim3 = 1;
    tilingData->bBatchDim0 = batchB;
    tilingData->bBatchDim1 = 1;
    tilingData->bBatchDim2 = 1;
    tilingData->bBatchDim3 = 1;
    tilingData->cBatchDim0 = batchC;
    tilingData->cBatchDim1 = 1;
    tilingData->cBatchDim2 = 1;
    tilingData->cBatchDim3 = 1;
    tilingData->biasBatchDimAll = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto
        kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_BMM_BROADCAST, half, half, half, half,
                                          Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0, 0, MatMulV3BmmBroadcastTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_IterateBatch)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t batchA = 2;
    const uint32_t batchB = 2;
    const uint32_t batchC = 2;
    const uint32_t blockNum = 1;

    size_t aSize = batchA * M * K * sizeof(half);
    size_t bSize = batchB * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = batchC * M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3IterBatchTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m 16 --n 16 --k 16 --batch " +
                             std::to_string(batchC) + " --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";

    MatMulV3IterBatchTilingData* tilingData = reinterpret_cast<MatMulV3IterBatchTilingData*>(tilingGM);
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 1;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->iterBatchL1 = 1;
    tilingData->iterBatchL0 = 1;
    tilingData->broadcastAxisA = 1;
    tilingData->broadcastAxisB = 1;
    tilingData->aBatchDim0 = batchA;
    tilingData->aBatchDim1 = 1;
    tilingData->aBatchDim2 = 1;
    tilingData->aBatchDim3 = 1;
    tilingData->bBatchDim0 = batchB;
    tilingData->bBatchDim1 = 1;
    tilingData->bBatchDim2 = 1;
    tilingData->bBatchDim3 = 1;
    tilingData->cBatchDim0 = batchC;
    tilingData->cBatchDim1 = 1;
    tilingData->cBatchDim2 = 1;
    tilingData->cBatchDim3 = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_BMM_ITERBATCH_BROADCAST, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0, 0, MatMulV3IterBatchTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(MatMulV3Test, Test_FP16_IterBatch)
{
    const int64_t M = 64;
    const int64_t N = 160;
    const int64_t K = 64;
    const uint32_t batch = 8;
    const uint32_t blockNum = 1;

    size_t aSize = batch * M * K * sizeof(half);
    size_t bSize = batch * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = batch * M * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(MatMulV3IterBatchTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/mat_mul/matmul_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m 64 --n 160 --k 64 --batch " +
                             std::to_string(batch) + " --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in matmul_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream aFile(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(aFile.is_open()) << "Failed to open input_a.bin";
    std::ifstream bFile(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(bFile.is_open()) << "Failed to open input_b.bin";
    aFile.read(reinterpret_cast<char*>(aGM), aSize);
    ASSERT_TRUE(aFile.good()) << "Failed to read input_a.bin (expected " << aSize << " bytes)";
    bFile.read(reinterpret_cast<char*>(bGM), bSize);
    ASSERT_TRUE(bFile.good()) << "Failed to read input_b.bin (expected " << bSize << " bytes)";
    std::memset(biasGM, 0, biasSize);

    MatMulV3IterBatchTilingData* tilingData = reinterpret_cast<MatMulV3IterBatchTilingData*>(tilingGM);
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->mL1 = M;
    tilingData->nL1 = N;
    tilingData->kL1 = K;
    tilingData->baseM = M;
    tilingData->baseN = N;
    tilingData->baseK = K;
    tilingData->isHf32 = 0;
    tilingData->l1BufferNum = 1;
    tilingData->l0cDB = 1;
    tilingData->ubDB = 1;
    tilingData->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData->iterBatchL1 = 2;
    tilingData->iterBatchL0 = 1;
    tilingData->broadcastAxisA = 1;
    tilingData->broadcastAxisB = 1;
    tilingData->aBatchDim0 = 1;
    tilingData->aBatchDim1 = 1;
    tilingData->aBatchDim2 = 1;
    tilingData->aBatchDim3 = batch;
    tilingData->bBatchDim0 = 1;
    tilingData->bBatchDim1 = 1;
    tilingData->bBatchDim2 = 1;
    tilingData->bBatchDim3 = batch;
    tilingData->cBatchDim0 = 1;
    tilingData->cBatchDim1 = 1;
    tilingData->cBatchDim2 = 1;
    tilingData->cBatchDim3 = batch;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = mat_mul_kernel_entry<OP_TYPE_MATMUL_BMM_ITERBATCH, half, half, half, half,
                                           Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, 0, 0, MatMulV3IterBatchTilingData>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}
