/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_gemm_syrk.cpp
 * \brief GemmSyrk Kernel UT测试用例
 *
 * GemmSyrk: C = alpha * (A @ A^T) + beta * C, computed fully in place (the same GM
 * buffer is bound as the c input and the c output). The single-fetch syrk assembly
 * (SYRK_KERNEL_SINGLE_FETCH, one nd2nz fetch per row-block per k-chunk, upper-triangle
 * pairs with a single Mmad chain and a transposed store of the mirrored tile) covers
 * all cases.
 *
 * Default mode is a smoke test (kernel exit status only), matching the repo-wide
 * kernel UT convention: on CPU debug hosts whose simulator does not commit the
 * cube/epilogue data path end to end, output buffers stay untouched and golden
 * comparison cannot pass regardless of kernel correctness. Set
 * GEMM_SYRK_UT_VERIFY_OUTPUT=1 to additionally verify the fp32-domain golden and the
 * symmetry of the complete output on hosts where the simulator commits results
 * (also the configuration used for on-board debugging).
 */

#include <algorithm>
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

#include "gemm_syrk.cpp"

namespace {

struct SyrkCase {
    uint32_t batch;
    uint32_t m;
    uint32_t k;
    uint32_t blockNum;
    float alpha;
    float beta;
    bool trans = false;
    uint32_t baseM = 16;
    uint32_t baseN = 16;
    uint32_t mL1 = 16;
    uint32_t nL1 = 16;
};

void ReadBinToVec(const std::string& path, void* data, size_t size)
{
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << path;
    file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
    ASSERT_EQ(file.gcount(), static_cast<std::streamsize>(size)) << "Unexpected file size: " << path;
}

template <typename ElementType>
bool CompareWithGolden(const ElementType* output, const std::vector<ElementType>& golden, size_t count, float atol,
                       float rtol, size_t& mismatchIdx)
{
    mismatchIdx = 0;
    for (size_t i = 0; i < count; ++i) {
        float got = static_cast<float>(output[i]);
        float exp = static_cast<float>(golden[i]);
        float diff = std::fabs(got - exp);
        float tol = atol + rtol * std::fabs(exp);
        if (diff > tol) {
            mismatchIdx = i;
            return false;
        }
    }
    return true;
}

template <typename ElementType>
float MaxSymmetryDiff(const ElementType* output, uint32_t batch, uint32_t m)
{
    float symMaxDiff = 0.0F;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = i + 1; j < m; ++j) {
                const size_t lij = (static_cast<size_t>(b) * m + i) * m + j;
                const size_t lji = (static_cast<size_t>(b) * m + j) * m + i;
                symMaxDiff = std::max(symMaxDiff,
                                      std::fabs(static_cast<float>(output[lij]) - static_cast<float>(output[lji])));
            }
        }
    }
    return symMaxDiff;
}

bool OutputVerificationEnabled()
{
    const char* env = getenv("GEMM_SYRK_UT_VERIFY_OUTPUT");
    return env != nullptr && env[0] == '1';
}

} // namespace

class GemmSyrkTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/gemm_syrk/gemm_syrk_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        aGM = nullptr;
        cGM = nullptr;
        workspaceGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (aGM)
            AscendC::GmFree((void*)aGM);
        if (cGM)
            AscendC::GmFree((void*)cGM);
        if (workspaceGM)
            AscendC::GmFree((void*)workspaceGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    template <SyrkKernelType KERNEL_TYPE, typename ElementType>
    void RunSyrkCase(const SyrkCase& testCase)
    {
        return testCase.trans ? RunSyrkCaseImpl<KERNEL_TYPE, true, ElementType>(testCase) :
                                RunSyrkCaseImpl<KERNEL_TYPE, false, ElementType>(testCase);
    }

    template <SyrkKernelType KERNEL_TYPE, bool TRANS, typename ElementType>
    void RunSyrkCaseImpl(const SyrkCase& testCase)
    {
        static_assert(std::is_same_v<ElementType, half> || std::is_same_v<ElementType, bfloat16_t>);
        const uint32_t batch = testCase.batch;
        const uint32_t m = testCase.m;
        const uint32_t k = testCase.k;
        const uint32_t blockNum = testCase.blockNum;
        const size_t aSize = static_cast<size_t>(batch) * m * k * sizeof(ElementType);
        const size_t cSize = static_cast<size_t>(batch) * m * m * sizeof(ElementType);
        const size_t cCount = static_cast<size_t>(batch) * m * m;
        const size_t workspaceSize = static_cast<size_t>(blockNum) * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD;

        aGM = (GM_ADDR)AscendC::GmAlloc(aSize);
        cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
        workspaceGM = (GM_ADDR)AscendC::GmAlloc(workspaceSize);
        tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(GemmSyrkUT::GemmSyrkTilingData));

        ASSERT_NE(aGM, nullptr);
        ASSERT_NE(cGM, nullptr);
        ASSERT_NE(workspaceGM, nullptr);
        ASSERT_NE(tilingGM, nullptr);
        memset(workspaceGM, 0, workspaceSize);

        std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/gemm_syrk/gemm_syrk_data";
        std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
        const bool isFp16 = std::is_same_v<ElementType, half>;
        const std::string dtype = isFp16 ? "float16" : "bfloat16";
        std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " + std::to_string(m) +
                                 " --k " + std::to_string(k) + " --batch " + std::to_string(batch) + " --dtype " +
                                 dtype + " --alpha " + std::to_string(testCase.alpha) + " --beta " +
                                 std::to_string(testCase.beta) + (testCase.trans ? " --trans" : "");
        int genRet = system(genCmd.c_str());
        ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in gemm_syrk_data";
        genRet = system(genDataCmd.c_str());
        ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

        ASSERT_NO_FATAL_FAILURE(ReadBinToVec(dataDir + "/input_a.bin", aGM, aSize));
        ASSERT_NO_FATAL_FAILURE(ReadBinToVec(dataDir + "/input_c.bin", cGM, cSize));
        std::vector<ElementType> golden(cCount);
        ASSERT_NO_FATAL_FAILURE(ReadBinToVec(dataDir + "/golden_c.bin", golden.data(), cCount * sizeof(ElementType)));

        auto* tilingData = reinterpret_cast<GemmSyrkUT::GemmSyrkTilingData*>(tilingGM);
        memset(tilingData, 0, sizeof(GemmSyrkUT::GemmSyrkTilingData));
        auto& batchTiling = tilingData->matMulTilingData;
        auto& matmulTiling = batchTiling.matMulTilingData;
        matmulTiling.usedCoreNum = blockNum;
        matmulTiling.m = m;
        matmulTiling.n = m; // syrk: N == M
        matmulTiling.k = k;
        matmulTiling.mL1 = testCase.mL1;
        matmulTiling.nL1 = testCase.nL1;
        matmulTiling.kL1 = 16;
        matmulTiling.baseM = testCase.baseM;
        matmulTiling.baseN = testCase.baseN;
        matmulTiling.baseK = 16;
        matmulTiling.skSingleCoreK = k;
        matmulTiling.mTailCnt = 1;
        matmulTiling.nTailCnt = 1;
        matmulTiling.mBaseTailSplitCnt = 1;
        matmulTiling.nBaseTailSplitCnt = 1;
        matmulTiling.mTailMain = m;
        matmulTiling.nTailMain = m;
        matmulTiling.mmadParam = 0;
        matmulTiling.l1BufferNum = 1;
        matmulTiling.l0cDB = 1;
        matmulTiling.ubDB = 1;
        matmulTiling.l2CacheDisable = GemmSyrkUT::L2CacheMode::L2_CACHE_DEFAULT;
        matmulTiling.sliceM = m;
        matmulTiling.srcNdStride = 1;
        matmulTiling.rowStride = 1;
        matmulTiling.innerBatch = 1;
        batchTiling.batchDimAll = batch;
        batchTiling.batchX3 = batch;
        tilingData->alpha = testCase.alpha;
        tilingData->beta = testCase.beta;

        AscendC::SetKernelMode(KernelMode::MIX_MODE);
        auto kernelFunc = gemm_syrk_kernel_entry<KERNEL_TYPE, TRANS, ElementType>;
        // cIn and c bind the same buffer: in-place update.
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, cGM, cGM, workspaceGM, tilingGM))
            << "Kernel execution failed: one or more cores exited with non-zero status";

        if (!OutputVerificationEnabled()) {
            return;
        }
        const float atol = isFp16 ? 4e-3F : 2e-2F;
        const float rtol = isFp16 ? 4e-3F : 2e-2F;
        const auto* output = reinterpret_cast<const ElementType*>(cGM);
        size_t mismatchIdx = 0;
        EXPECT_TRUE(CompareWithGolden(output, golden, cCount, atol, rtol, mismatchIdx))
            << "golden mismatch at flat index " << mismatchIdx;

        const float symMaxDiff = MaxSymmetryDiff(output, batch, m);
        EXPECT_LE(symMaxDiff, atol) << "output is not symmetric, maxDiff=" << symMaxDiff;
    }

    static constexpr size_t WORKSPACE_TILE_SIZE = 256UL * 256 * 4;
    static constexpr size_t WORKSPACE_OVERHEAD = 20UL * 1024 * 1024;
    GM_ADDR aGM;
    GM_ADDR cGM;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
};

// Single-fetch assembly: one nd2nz fetch per row-block per k-chunk, upper-triangle
// pairs, single Mmad chain + transposed store of the mirrored tile.
TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Basic)
{
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 16, 16, 1, 3.0F, 2.0F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Pair_Tail)
{
    // m/n tails (17 not aligned to 16) and multiple blocks exercising the
    // grid-stride loop over the upper-triangle slots (complete symmetric output).
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({2, 17, 10, 2, 3.687209F, 2.067589F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_KTail)
{
    // k = 40 with kL1 = 16: three k-chunks, the last one a tail of 8.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 32, 40, 1, 1.0F, 1.0F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_AlphaOnly)
{
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 8, 4, 1, 2.0F, 1.0F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_BetaOnly)
{
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({2, 3, 8, 1, 1.0F, 2.0F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_BetaZero)
{
    // cuBLAS contract: beta == 0 -> the C addend is not read and not added;
    // output = alpha * (A @ A^T) regardless of the (possibly NaN/Inf) C input.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 32, 16, 1, 1.5F, 0.0F});
}

TEST_F(GemmSyrkTest, Test_BF16_SyrkSingleFetch_BetaZero)
{
    // bf16 beta == 0 skip-x3 path (b16 cast of alpha * acc only).
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, bfloat16_t>({2, 32, 16, 1, 2.0F, 0.0F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_M1Sync)
{
    // m == 1: the second AIV of the MIX pair has no valid rows and only keeps
    // the ready/free handshake alive.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({8, 1, 3, 1, 3.687209F, 2.067589F});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_AsymBase)
{
    // Asymmetric base (baseM=32 > baseN=16) with mL1=nL1=16=min(baseM, baseN):
    // kernel-level robustness for the pair-mode cross constraints (the host
    // tiling now forces the symmetric contract baseM=baseN=mL1=nL1).
    // m=32 -> 2x2 grid: (0,0) diagonal, (0,1) pair, (1,1) diagonal.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 32, 16, 1, 2.0F, 1.0F, false, 32, 16, 16, 16});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_MultiCore)
{
    // m=64 -> 4x4 grid (10 upper-triangle slots) over two cores.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 64, 32, 2, 1.5F, 0.5F});
}

TEST_F(GemmSyrkTest, Test_BF16_SyrkSingleFetch_Basic)
{
    // bf16 pair grid: m=32 -> 2x2 grid, exercises the b16 transposed store path.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, bfloat16_t>({1, 32, 16, 1, 1.0F, 1.0F});
}

// Transposed storage (transpose_x): a is (batch, k, m); the DNExt view routes
// the same single fetch through dn2nz and computes C = alpha * (A^T @ A) + beta * C.
TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Trans_Basic)
{
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 16, 16, 1, 3.0F, 2.0F, true});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Trans_Pair_Tail)
{
    // m tail (17 not aligned to 16) with the transposed storage over two blocks:
    // grid-stride loop over the upper-triangle slots (complete symmetric output).
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({2, 17, 10, 2, 3.687209F, 2.067589F, true});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Trans_KTail)
{
    // k = 40 with kL1 = 16: three k-chunks, the last one a tail of 8.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 32, 40, 1, 1.0F, 1.0F, true});
}

TEST_F(GemmSyrkTest, Test_FP16_SyrkSingleFetch_Trans_MultiCore)
{
    // m=64 -> 4x4 grid (10 upper-triangle slots) over two cores.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, half>({1, 64, 32, 2, 1.5F, 0.5F, true});
}

TEST_F(GemmSyrkTest, Test_BF16_SyrkSingleFetch_Trans_Basic)
{
    // bf16 transposed pair grid: m=32 -> 2x2 grid.
    RunSyrkCase<SYRK_KERNEL_SINGLE_FETCH, bfloat16_t>({1, 32, 16, 1, 1.0F, 1.0F, true});
}
