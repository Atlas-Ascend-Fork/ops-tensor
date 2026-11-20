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
 * \file test_transpose_batch_mat_mul.cpp
 * \brief TransposeBatchMatMul Kernel UT测试用例
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "transpose_batch_mat_mul.cpp"

class TbmmTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR +
                               "/transpose_batch_mat_mul/tbmm_data && rm -rf *.bin";
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
        scaleGM = nullptr;
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
        if (scaleGM)
            AscendC::GmFree((void*)scaleGM);
    }

    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR cGM;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
    GM_ADDR scaleGM;

    template <typename T>
    T* Allocate(size_t count, GM_ADDR& buffer)
    {
        if (buffer != nullptr) {
            AscendC::GmFree(buffer);
        }
        buffer = reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(count * sizeof(T)));
        if (buffer != nullptr) {
            std::memset(buffer, 0, count * sizeof(T));
        }
        return reinterpret_cast<T*>(buffer);
    }

    template <typename T, typename BiasT, bool IterBatch, uint64_t NonContiguousType, bool TransA, bool TransB,
              bool WeightNz = false, bool SplitK = false,
              Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODEL = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>
    void RunLayoutCase(bool hasBias = true, bool splitL0 = false, bool hf32 = false, uint64_t batchSplitFactor = 1)
    {
        TbmmBasicTilingData cfg;
        cfg.m = 17;
        cfg.n = 16;
        cfg.k = 32;
        cfg.batch = 7;
        cfg.baseM = 32;
        cfg.baseN = 16;
        cfg.baseK = 32;
        cfg.iterBatchL1 = 3;
        cfg.iterBatchL0 = 2;
        cfg.l1BufferNum = 2;
        cfg.isHf32 = hf32;
        cfg.batchSplitFactor = batchSplitFactor;
        if constexpr (L0C2OUT_MODEL == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2) {
            cfg.n = 129;
            cfg.baseN = 144;
        }
        if (batchSplitFactor > 1) {
            cfg.batch = 6;
            const auto innerBatch = cfg.batch / batchSplitFactor;
            while (innerBatch % cfg.iterBatchL1 != 0) {
                --cfg.iterBatchL1;
            }
            cfg.iterBatchL0 = std::min(cfg.iterBatchL0, cfg.iterBatchL1);
        }
        if (splitL0) {
            cfg.baseM = 16;
            cfg.baseK = 16;
            cfg.iterBatchL0 = 1;
        }
        const auto batch = cfg.batch;
        const auto m = cfg.m;
        const auto n = cfg.n;
        const auto k = cfg.k;
        auto* a = Allocate<T>(batch * m * k, aGM);
        auto* b = Allocate<T>(batch * k * n, bGM);
        auto* c = Allocate<T>(batch * m * n, cGM);
        auto* bias = hasBias ? Allocate<BiasT>(n, biasGM) : nullptr;
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        ASSERT_NE(c, nullptr);
        if (hasBias) {
            ASSERT_NE(bias, nullptr);
        }
        cfg.mL1 = cfg.baseM;
        cfg.nL1 = cfg.baseN;
        cfg.kL1 = cfg.baseK;
        cfg.innerBatch = cfg.batch;
        auto* tiling = Allocate<TbmmBasicTilingData>(1, tilingGM);
        ASSERT_NE(tiling, nullptr);
        std::memcpy(tiling, &cfg, sizeof(cfg));

        constexpr auto opType = IterBatch ? OP_TYPE_TBMM_ITER_BATCH :
                                            (SplitK ? OP_TYPE_TBMM_BASIC_SPLIT_K : OP_TYPE_TBMM_BASIC);
        AscendC::SetKernelMode(KernelMode::MIX_MODE);
        auto kernelFunc = transpose_batch_mat_mul_kernel_entry<opType, T, T, T, BiasT, NonContiguousType, TransA,
                                                               TransB, WeightNz, L0C2OUT_MODEL>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, 1, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, scaleGM))
            << "Kernel execution failed: one or more cores exited with non-zero status";
    }
};

constexpr size_t WORKSPACE_TILE_SIZE = 256UL * 256 * 4;
constexpr size_t WORKSPACE_OVERHEAD = 20UL * 1024 * 1024;

TEST_F(TbmmTest, Test_FP16_Batch2)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const int64_t BATCH = 2;
    const uint32_t blockNum = 1;

    size_t aSize = BATCH * M * K * sizeof(half);
    size_t bSize = BATCH * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --batch 2 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->batch = 2;
    tilingData->batchSplitFactor = 1;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<OP_TYPE_TBMM_BASIC, half, half, half, half>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, nullptr))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_FP16_TransBatchA)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const int64_t BATCH = 2;
    const uint32_t blockNum = 1;

    size_t aSize = M * BATCH * K * sizeof(half);
    size_t bSize = BATCH * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --batch 2 --dtype float16 --trans_batch_a";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->batch = 2;
    tilingData->batchSplitFactor = 1;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<
        OP_TYPE_TBMM_TRANS_BATCH_A, half, half, half, half,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AC_TRANSPOSE)>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, nullptr))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_FP32_Batch4_MultiCore)
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;
    const int64_t BATCH = 4;
    const uint32_t blockNum = 4;

    size_t aSize = BATCH * M * K * sizeof(float);
    size_t bSize = BATCH * K * N * sizeof(float);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(float);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 32 --n 32 --k 32 --batch 4 --dtype float32";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 32;
    tilingData->n = 32;
    tilingData->k = 32;
    tilingData->batch = 4;
    tilingData->batchSplitFactor = 1;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<OP_TYPE_TBMM_BASIC, float, float, float, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, nullptr))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_INT8_Batch2)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const int64_t BATCH = 2;
    const uint32_t blockNum = 1;

    size_t aSize = BATCH * M * K * sizeof(half);
    size_t bSize = BATCH * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(int8_t);
    size_t scaleSize = BATCH * N * sizeof(uint64_t);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(scaleGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --batch 2 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    std::vector<uint64_t> scaleVec(BATCH * N, 0x3F80000000000000ULL);

    std::copy(scaleVec.begin(), scaleVec.end(), reinterpret_cast<uint64_t*>(scaleGM));

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->batch = 2;
    tilingData->batchSplitFactor = 1;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<OP_TYPE_TBMM_BASIC, half, half, int8_t, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, scaleGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_INT8_Batch4_MultiCore)
{
    const int64_t M = 32;
    const int64_t N = 32;
    const int64_t K = 32;
    const int64_t BATCH = 4;
    const uint32_t blockNum = 4;

    size_t aSize = BATCH * M * K * sizeof(half);
    size_t bSize = BATCH * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(int8_t);
    size_t scaleSize = BATCH * N * sizeof(uint64_t);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(scaleGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 32 --n 32 --k 32 --batch 4 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    std::vector<uint64_t> scaleVec(BATCH * N, 0x3F80000000000000ULL);

    std::copy(scaleVec.begin(), scaleVec.end(), reinterpret_cast<uint64_t*>(scaleGM));

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 32;
    tilingData->n = 32;
    tilingData->k = 32;
    tilingData->batch = 4;
    tilingData->batchSplitFactor = 1;
    tilingData->mL1 = 16;
    tilingData->nL1 = 16;
    tilingData->kL1 = 16;
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->baseK = 16;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<OP_TYPE_TBMM_BASIC, half, half, int8_t, float>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, scaleGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_FP16_BatchSplitFactor)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const int64_t BATCH = 4;
    const int64_t BATCH_SPLIT_FACTOR = 2;
    const uint32_t blockNum = 1;

    size_t aSize = BATCH * M * K * sizeof(half);
    size_t bSize = BATCH * K * N * sizeof(half);
    size_t biasSize = N * sizeof(float);
    size_t cSize = M * BATCH * N * sizeof(half);
    size_t workspaceSize = blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

    aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
    bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(TbmmBasicTilingData));

    ASSERT_NE(aGM, nullptr);
    ASSERT_NE(bGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(cGM, nullptr);
    ASSERT_NE(workspaceGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/transpose_batch_mat_mul/tbmm_data";
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir +
                             " && python3 gen_data.py --m 16 --n 16 --k 16 --batch 4 --dtype float16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in tbmm_data";
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

    TbmmBasicTilingData* tilingData = reinterpret_cast<TbmmBasicTilingData*>(tilingGM);
    tilingData->usedCoreNum = blockNum;
    tilingData->m = 16;
    tilingData->n = 16;
    tilingData->k = 16;
    tilingData->batch = 4;
    tilingData->batchSplitFactor = 2;
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
    tilingData->l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    tilingData->sliceM = 1;
    tilingData->srcNdStride = 1;
    tilingData->innerBatch = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = transpose_batch_mat_mul_kernel_entry<OP_TYPE_TBMM_BASIC, half, half, half, half>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, biasGM, cGM, workspaceGM, tilingGM, nullptr))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

TEST_F(TbmmTest, Test_MbkBknBasic)
{
    RunLayoutCase<half, float, false, static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1),
                  false, false, false>();
}

TEST_F(TbmmTest, Test_MbkNbkBasic)
{
    RunLayoutCase<bfloat16_t, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AB_TRANSPOSE), false, true,
                  false>();
}

TEST_F(TbmmTest, Test_BkmNbkBasic)
{
    RunLayoutCase<half, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, true,
                  false>();
}

TEST_F(TbmmTest, Test_BkmNbkIterBatch)
{
    RunLayoutCase<half, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, true,
                  false>();
}

TEST_F(TbmmTest, Test_BkmKbnBasic)
{
    RunLayoutCase<half, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, false,
                  false>();
}

TEST_F(TbmmTest, Test_BkmKbnIterBatch)
{
    RunLayoutCase<half, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, false,
                  false>();
}

TEST_F(TbmmTest, Test_BkmZnBasic)
{
    RunLayoutCase<bfloat16_t, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_NONE), true, true, true>();
}

TEST_F(TbmmTest, Test_MbkNzBasic)
{
    RunLayoutCase<bfloat16_t, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1), false, false,
                  true>();
}

TEST_F(TbmmTest, Test_IterBatchNoBiasAndL0Split)
{
    RunLayoutCase<half, half, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), false, true>(
        false, true);
}

TEST_F(TbmmTest, Test_Fp32KeepDtype)
{
    RunLayoutCase<float, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, false>();
}

TEST_F(TbmmTest, Test_Fp32Hf32)
{
    RunLayoutCase<float, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, true>(
        true, false, true);
}

TEST_F(TbmmTest, Test_MbnBf16WithFp32Bias)
{
    RunLayoutCase<bfloat16_t, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AC_TRANSPOSE), false,
                  false>();
}

TEST_F(TbmmTest, Test_BatchSplitWithoutBias)
{
    for (uint64_t split : {2UL, 3UL}) {
        RunLayoutCase<bfloat16_t, float, false,
                      static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AC_TRANSPOSE), false,
                      false>(false, false, false, split);
    }
}

TEST_F(TbmmTest, Test_MbnSplitKWithFp32Bias)
{
    RunLayoutCase<bfloat16_t, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AC_TRANSPOSE), false, false,
                  false, true>();
}

TEST_F(TbmmTest, Test_BmnBkmNbkSplitKWithFp32Bias)
{
    RunLayoutCase<float, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, true,
                  false, true>();
}

TEST_F(TbmmTest, Test_BmnMbkKbnSplitKWithFp32Bias)
{
    RunLayoutCase<float, float, false,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AB_TRANSPOSE), false, false,
                  false, true>();
}

TEST_F(TbmmTest, Test_BkmNbkIterBatchFixpipe)
{
    RunLayoutCase<half, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, true,
                  false, false, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>(false, true);
}

TEST_F(TbmmTest, Test_BkmKbnIterBatchFixpipe)
{
    RunLayoutCase<bfloat16_t, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_B_TRANSPOSE), true, false,
                  false, false, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>();
}

TEST_F(TbmmTest, Test_MbnIterBatchNoBias)
{
    RunLayoutCase<half, half, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_C_TRANSPOSE), false, false>(
        false, true);
}

TEST_F(TbmmTest, Test_MbnIterBatchNdFixpipe)
{
    RunLayoutCase<bfloat16_t, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_C_TRANSPOSE), false, true,
                  false, false, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>();
}

TEST_F(TbmmTest, Test_MbkBknIterBatch)
{
    RunLayoutCase<half, float, true, static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1),
                  false, false>();
}

TEST_F(TbmmTest, Test_MbkNbkIterBatch)
{
    RunLayoutCase<bfloat16_t, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AB_TRANSPOSE), false,
                  true>();
}

TEST_F(TbmmTest, Test_MbkKbnIterBatchFixpipe)
{
    RunLayoutCase<bfloat16_t, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AB_TRANSPOSE), false, false,
                  false, false, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>();
}

TEST_F(TbmmTest, Test_MbkMbnIterBatchFixpipe)
{
    RunLayoutCase<half, float, true,
                  static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_AC_TRANSPOSE), false, true,
                  false, false, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>();
}
