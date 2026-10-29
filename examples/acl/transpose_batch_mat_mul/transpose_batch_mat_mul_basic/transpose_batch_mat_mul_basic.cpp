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
 * @file transpose_batch_mat_mul_basic.cpp
 * @brief TransposeBatchMatMul Basic example supporting multiple dtypes.
 *
 * Supported dtypes: float16, bfloat16, float32
 * Supported layouts: standard batch [batch,m,k] and transposed-batch [m,batch,k]
 *
 * Data layout:
 *   A (standard):     [batch, m, k]  (row-major)
 *   A (trans_batch_a): [m, batch, k]  (transposed batch dimension)
 *   B:                 [batch, k, n]  (row-major)
 *   C:                 [m, batch, n]  (always transposed batch)
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_tbmm_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "data_utils.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Tiling configuration                                                        */
/* ========================================================================== */

struct TilingConfig {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    int64_t dtypeSize;
    static constexpr int L1_BUFFER_NUM = 2;
    static constexpr int L0C_DB = 1;
};

static TilingConfig GetTilingConfig(const std::string& dtype)
{
    if (dtype == "float32") {
        return {128, 128, 128, 128, 128, 64, 4};
    } else { // float16, bfloat16
        return {128, 128, 128, 128, 128, 64, 2};
    }
}

static constexpr int64_t BLOCK_16 = 16L;
static constexpr size_t RPC_WORKSPACE_PADDING = 20UL * 1024UL * 1024UL;

/* ========================================================================== */
/* CLI argument parsing                                                        */
/* ========================================================================== */

struct CliArgs {
    int64_t m, k, n, batch;
    bool transBatchA = false;
    std::string dtype = "float16";
    bool isHf32 = false;
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 5) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n> <batch> [transBatchA] [dtype] [isHf32]\n";
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);

    if (argc >= 6) {
        args.transBatchA = ParseBool(argv[5]);
    }
    if (argc >= 7) {
        args.dtype = argv[6];
    }
    if (argc >= 8) {
        args.isHf32 = ParseBool(argv[7]);
    }

    if (args.m <= 0 || args.k <= 0 || args.n <= 0 || args.batch <= 0) {
        std::cerr << "Error: M, K, N, batch must be positive integers.\n";
        return false;
    }

    if (args.isHf32 && args.dtype != "float32") {
        std::cerr << "Error: isHf32 only valid with float32\n";
        return false;
    }

    if (args.dtype != "float16" && args.dtype != "bfloat16" && args.dtype != "float32") {
        std::cerr << "Error: dtype must be float16, bfloat16, or float32 (got '" << args.dtype << "')\n";
        return false;
    }

    return true;
}

/* ========================================================================== */
/* Device-side kernel                                                          */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, uint64_t NON_CONTIGUOUS_TYPE>
__global__ __aicore__ void tbmm_basic_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM,
                                             int64_t m, int64_t n, int64_t k, int64_t batch, int64_t batchSplitFactor,
                                             int64_t mL1, int64_t nL1, int64_t kL1, int64_t baseM, int64_t baseN,
                                             int64_t baseK, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockTBMM,
                                                              NON_CONTIGUOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, LayoutA, B_TYPE, LayoutB, C_TYPE, LayoutC,
                                                    BIAS_TYPE, LayoutBias>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {m, n, k, batch, batchSplitFactor},
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM, static_cast<uint32_t>(mL1), static_cast<uint32_t>(nL1),
         static_cast<uint32_t>(kL1), static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN),
         static_cast<uint32_t>(baseK), TilingConfig::L1_BUFFER_NUM, TilingConfig::L0C_DB},
        {},
        {static_cast<uint32_t>(mL1), static_cast<uint32_t>(nL1), static_cast<uint32_t>(kL1),
         static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), 0, 0, 1, 1, 1, 1,
         static_cast<uint8_t>(isHf32 ? 1 : 0), Blaze::Gemm::L2_CACHE_DEFAULT, 1, 1, 1}};

    MatmulKernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side kernel launcher                                                    */
/* ========================================================================== */

namespace {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool TransBatchA>
void LaunchKernel(int64_t m, int64_t n, int64_t k, int64_t batch, uint8_t* dA, uint8_t* dB, uint8_t* dC, uint8_t* dBias,
                  uint8_t* dWorkSpace, const TilingConfig& cfg, int64_t blockNum, aclrtStream stream, bool isHf32)
{
    constexpr uint64_t NON_CONTIGUOUS_TYPE = TransBatchA ?
                                                 static_cast<uint64_t>(
                                                     Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1) :
                                                 0ULL;

    tbmm_basic_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, NON_CONTIGUOUS_TYPE>
        <<<blockNum, 0, stream>>>(dA, dB, dC, dBias, dWorkSpace, m, n, k, batch, 1, cfg.mL1, cfg.nL1, cfg.kL1,
                                  cfg.baseM, cfg.baseN, cfg.baseK, isHf32);
}

} // namespace

/* ========================================================================== */
/* Host-side runner                                                            */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));

    TilingConfig tilingCfg = GetTilingConfig(args.dtype);
    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0) {
        std::cout << "blockNum cannot less than 0, but current: " << blockNum << std::endl;
        return;
    }

    size_t dtypeSize = (args.dtype == "float32") ? sizeof(float) : sizeof(half);

    // A size depends on layout (total elements are the same: batch * m * k)
    size_t sizeA = static_cast<size_t>(args.batch) * args.m * args.k * dtypeSize;
    size_t sizeB = static_cast<size_t>(args.batch) * args.k * args.n * dtypeSize;
    // C is stored as [m, batch, n]
    size_t sizeC = static_cast<size_t>(args.m) * args.batch * args.n * dtypeSize;

    size_t workspaceSize = blockNum * tilingCfg.baseM * tilingCfg.baseN * sizeof(float) + RPC_WORKSPACE_PADDING;

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    struct stat st;
    std::string pathA = inputDir + "/input_a.bin";
    if (stat(pathA.c_str(), &st) != 0) {
        std::cerr << "Input files not found: " << pathA << std::endl;
        return;
    }

    std::vector<uint8_t> hostA(sizeA);
    std::vector<uint8_t> hostB(sizeB);
    std::vector<uint8_t> hostC(sizeC, 0);

    if (!ReadFile(pathA, hostA.data(), sizeA)) {
        std::cerr << "Failed to read input A" << std::endl;
        return;
    }

    std::string pathB = inputDir + "/input_b.bin";
    if (!ReadFile(pathB, hostB.data(), sizeB)) {
        std::cerr << "Failed to read input B" << std::endl;
        return;
    }

    // Bias is not supported (BlockEpilogueEmpty — bias is never applied)
    uint8_t* deviceBias = nullptr;

    // Allocate device buffers
    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Copy H2D
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // Print execution summary
    std::cout << "============================================================" << std::endl;
    std::cout << "  TransposeBatchMatMul Basic — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape      : M=" << args.m << ", K=" << args.k << ", N=" << args.n << ", Batch=" << args.batch
              << std::endl;
    std::cout << "  Dtype      : " << args.dtype << std::endl;
    std::cout << "  transBatchA: " << (args.transBatchA ? "true" : "false") << std::endl;
    std::cout << "  isHf32     : " << (args.isHf32 ? "true" : "false") << std::endl;
    std::cout << "  L1 Tile    : [" << tilingCfg.mL1 << ", " << tilingCfg.nL1 << ", " << tilingCfg.kL1 << "]"
              << std::endl;
    std::cout << "  L0 Tile    : [" << tilingCfg.baseM << ", " << tilingCfg.baseN << ", " << tilingCfg.baseK << "]"
              << std::endl;
    std::cout << "  BlockNum   : " << blockNum << std::endl;
    std::cout << "  Workspace  : " << (workspaceSize / 1024) << " KB" << std::endl;
    std::cout << "  C layout   : [m, batch, n] (transposed batch)" << std::endl;
    std::cout << "  A layout   : " << (args.transBatchA ? "[m, batch, k] (transposed batch)" : "[batch, m, k]")
              << std::endl;
    std::cout << "============================================================" << std::endl;

    if (args.dtype == "float32") {
        if (args.transBatchA) {
            LaunchKernel<float, float, float, float, true>(args.m, args.n, args.k, args.batch, deviceA, deviceB,
                                                           deviceC, deviceBias, deviceWorkspace, tilingCfg, blockNum,
                                                           stream, args.isHf32);
        } else {
            LaunchKernel<float, float, float, float, false>(args.m, args.n, args.k, args.batch, deviceA, deviceB,
                                                            deviceC, deviceBias, deviceWorkspace, tilingCfg, blockNum,
                                                            stream, args.isHf32);
        }
    } else if (args.dtype == "bfloat16") {
        if (args.transBatchA) {
            LaunchKernel<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true>(
                args.m, args.n, args.k, args.batch, deviceA, deviceB, deviceC, deviceBias, deviceWorkspace, tilingCfg,
                blockNum, stream, args.isHf32);
        } else {
            LaunchKernel<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false>(
                args.m, args.n, args.k, args.batch, deviceA, deviceB, deviceC, deviceBias, deviceWorkspace, tilingCfg,
                blockNum, stream, args.isHf32);
        }
    } else {
        if (args.transBatchA) {
            LaunchKernel<half, half, half, half, true>(args.m, args.n, args.k, args.batch, deviceA, deviceB, deviceC,
                                                       deviceBias, deviceWorkspace, tilingCfg, blockNum, stream,
                                                       args.isHf32);
        } else {
            LaunchKernel<half, half, half, half, false>(args.m, args.n, args.k, args.batch, deviceA, deviceB, deviceC,
                                                        deviceBias, deviceWorkspace, tilingCfg, blockNum, stream,
                                                        args.isHf32);
        }
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Copy D2H
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // Write output
    std::string outPath = outputDir + "/npu_out.bin";
    if (!WriteFile(outPath, hostC.data(), sizeC)) {
        std::cerr << "Failed to write output" << std::endl;
    }

    // Cleanup
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceWorkspace));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
}

/* ========================================================================== */
/* Entry point                                                                 */
/* ========================================================================== */

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseCliArgs(argc, argv, args)) {
        return 1;
    }

    Run(args);
    return 0;
}
