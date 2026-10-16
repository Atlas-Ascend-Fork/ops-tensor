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
 * @file mat_mul_bmm_broadcast.cpp
 * @brief BatchMatmul Broadcast example supporting multiple dtypes.
 *
 * Per-tile batch broadcast: A and B can have different batch dimensions,
 * broadcast to C batch dimension via modulo arithmetic.
 *
 * Supported dtypes: float16, bfloat16, float32
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
#include "blaze/gemm/kernel/kernel_batch_matmul_broadcast.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL()                                                                                       \
    bmm_broadcast_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B><<<p.blockNum, 0, p.stream>>>(      \
        p.dA, p.dB, p.dC, p.dBias, p.m, p.n, p.k, p.batch, p.batchA, p.batchB, p.cfg->mL1, p.cfg->nL1, p.cfg->kL1, \
        p.cfg->baseM, p.cfg->baseN, p.cfg->baseK, p.tiling->mTailCnt, p.tiling->nTailCnt, p.isHf32)

#define DISPATCH(TYPE, BIAS_TYPE, TRANS_A, TRANS_B, PARAMS)                      \
    do {                                                                         \
        if (TRANS_A) {                                                           \
            if (TRANS_B) {                                                       \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, true>(PARAMS);   \
            } else {                                                             \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, false>(PARAMS);  \
            }                                                                    \
        } else {                                                                 \
            if (TRANS_B) {                                                       \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, true>(PARAMS);  \
            } else {                                                             \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, false>(PARAMS); \
            }                                                                    \
        }                                                                        \
    } while (0)

/* ========================================================================== */
/* Tiling configuration                                                        */
/* ========================================================================== */

struct TilingConfig {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    int64_t dtypeSize;
    static constexpr int L1_STAGES = 2;
    static constexpr int L0C_DB = 1;
};

static TilingConfig GetTilingConfig(const std::string& dtype)
{
    if (dtype == "float32") {
        return {64, 64, 64, 64, 64, 32, 4};
    } else {
        return {64, 128, 64, 64, 128, 32, 2};
    }
}

/* ========================================================================== */
/* Tiling computation                                                         */
/* ========================================================================== */

struct BmmTiling {
    uint32_t mTailCnt = 1;
    uint32_t nTailCnt = 1;
};

static constexpr int64_t BLOCK_16 = 16L;

static BmmTiling ComputeTiling(TilingConfig& cfg, int64_t m, int64_t k, int64_t n)
{
    BmmTiling tiling;

    cfg.baseM = CeilAlign(m, BLOCK_16);
    cfg.mL1 = cfg.baseM;

    int64_t nCnt = CeilDiv(n, cfg.baseN);
    int64_t tailN = n - (nCnt - 1) * cfg.baseN;
    if (tailN > 0 && tailN < cfg.baseN) {
        int64_t splitN = CeilDiv(tailN, 2L);
        tiling.nTailCnt = static_cast<uint32_t>(CeilDiv(tailN, splitN));
    }

    cfg.nL1 = cfg.baseN;
    cfg.kL1 = cfg.baseK * TilingConfig::L1_STAGES;

    return tiling;
}

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m, k, n;
    int64_t batch = 1;
    int64_t batchA = 1;
    int64_t batchB = 1;
    bool transA = false;
    bool transB = false;
    std::string dtype = "float16";
    bool isHf32 = false;
    int64_t bias = 0;
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 7) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: " << argv[0]
                  << " <m> <k> <n> <batch> <batchA> <batchB> [transA] [transB] [dtype] [isHf32] [bias]\n";
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);
    args.batchA = std::atoll(argv[5]);
    args.batchB = std::atoll(argv[6]);

    if (argc >= 8)
        args.transA = ParseBool(argv[7]);
    if (argc >= 9)
        args.transB = ParseBool(argv[8]);
    if (argc >= 10)
        args.dtype = argv[9];
    if (argc >= 11)
        args.isHf32 = ParseBool(argv[10]);
    if (argc >= 12)
        args.bias = std::atoll(argv[11]);

    if (args.m <= 0 || args.k <= 0 || args.n <= 0 || args.batch <= 0 || args.batchA <= 0 || args.batchB <= 0) {
        std::cerr << "Error: M, K, N, batch, batchA, batchB must be positive integers.\n";
        return false;
    }
    if (args.dtype != "float16" && args.dtype != "bfloat16" && args.dtype != "float32") {
        std::cerr << "Error: dtype must be float16, bfloat16, or float32\n";
        return false;
    }

    return true;
}

/* ========================================================================== */
/* Device-side kernel wrapper                                                 */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, 0>;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B>
__global__ __aicore__ void bmm_broadcast_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m,
                                                int64_t n, int64_t k, int64_t batch, int64_t batchA, int64_t batchB,
                                                int64_t mL1, int64_t nL1, int64_t kL1, int64_t baseM, int64_t baseN,
                                                int64_t baseK, uint32_t mTailCnt, uint32_t nTailCnt, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using LAYOUT_C = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBmmBroadcast>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, LAYOUT_A, B_TYPE, LAYOUT_B, C_TYPE,
                                                    LAYOUT_C, BIAS_TYPE, LAYOUT_C>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    typename MatmulKernel::BatchInfo batchInfo;
    batchInfo.aBatchDim3 = static_cast<uint32_t>(batchA);
    batchInfo.bBatchDim3 = static_cast<uint32_t>(batchB);
    batchInfo.cBatchDim3 = static_cast<uint32_t>(batch);

    Params params = {
        {m, n, k, batch},
        {aGM, bGM, cGM, biasGM, nullptr, nullptr, static_cast<uint64_t>(mL1), static_cast<uint64_t>(nL1),
         static_cast<uint64_t>(kL1), static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN),
         static_cast<uint32_t>(baseK), TilingConfig::L1_STAGES, TilingConfig::L0C_DB},
        {},
        {static_cast<uint32_t>(mL1), static_cast<uint32_t>(nL1), static_cast<uint32_t>(kL1),
         static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), mTailCnt, nTailCnt,
         1, 1, 1, 1, static_cast<uint8_t>(isHf32 ? 1 : 0), Blaze::Gemm::L2_CACHE_DEFAULT},
        batchInfo};

    MatmulKernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side kernel launcher                                                  */
/* ========================================================================== */

namespace {

struct LaunchParams {
    uint8_t* dA;
    uint8_t* dB;
    uint8_t* dC;
    uint8_t* dBias;
    int64_t m, n, k;
    int64_t batch, batchA, batchB;
    int64_t blockNum;
    const BmmTiling* tiling;
    const TilingConfig* cfg;
    aclrtStream stream;
    bool transA, transB, isHf32;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool TransA, bool TransB>
void LaunchKernel(const LaunchParams& p)
{
    using LAYOUT_A = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LAYOUT_B = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    LAUNCH_KERNEL_IMPL();
}

} // namespace

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};

    ACLDeviceGuard guard(stream);

    TilingConfig tilingCfg = GetTilingConfig(args.dtype);
    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0) {
        std::cout << "blockNum cannot less than 0, but current: " << blockNum << std::endl;
        return;
    }
    BmmTiling tiling = ComputeTiling(tilingCfg, args.m, args.k, args.n);

    size_t dtypeSize = (args.dtype == "float32") ? sizeof(float) : sizeof(half);
    size_t sizeA = static_cast<size_t>(args.batchA) * args.m * args.k * dtypeSize;
    size_t sizeB = static_cast<size_t>(args.batchB) * args.k * args.n * dtypeSize;
    size_t sizeC = static_cast<size_t>(args.batch) * args.m * args.n * dtypeSize;
    size_t sizeBias = (args.bias > 0) ? static_cast<size_t>(args.batch) * args.bias * dtypeSize : 0;

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    struct stat st;
    std::string pathA = inputDir + "/input_a.bin";
    std::string pathB = inputDir + "/input_b.bin";
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

    if (!ReadFile(pathB, hostB.data(), sizeB)) {
        std::cerr << "Failed to read input B" << std::endl;
        return;
    }

    std::vector<uint8_t> hostBias(sizeBias, 0);
    uint8_t* deviceBias = nullptr;

    if (args.bias > 0) {
        std::string biasPath = inputDir + "/bias.bin";
        if (stat(biasPath.c_str(), &st) == 0) {
            if (!ReadFile(biasPath, hostBias.data(), sizeBias)) {
                std::cerr << "Failed to read bias from " << biasPath << std::endl;
                return;
            }
        } else {
            std::cout << "[INFO] Bias file not found, using zero-initialized bias" << std::endl;
        }
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "============================================================" << std::endl;
    std::cout << "  BatchMatmul Broadcast — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << std::endl;
    std::cout << "  Batch    : C=" << args.batch << ", A=" << args.batchA << ", B=" << args.batchB << std::endl;
    std::cout << "  Dtype    : " << args.dtype << std::endl;
    std::cout << "  transA   : " << (args.transA ? "true" : "false") << std::endl;
    std::cout << "  transB   : " << (args.transB ? "true" : "false") << std::endl;
    std::cout << "  L1 Tile  : [" << tilingCfg.mL1 << ", " << tilingCfg.nL1 << ", " << tilingCfg.kL1 << "]"
              << std::endl;
    std::cout << "  L0 Tile  : [" << tilingCfg.baseM << ", " << tilingCfg.baseN << ", " << tilingCfg.baseK << "]"
              << std::endl;
    std::cout << "  BlockNum : " << blockNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams launchParams = {deviceA,    deviceB,    deviceC,     deviceBias,  args.m,     args.n,
                                 args.k,     args.batch, args.batchA, args.batchB, blockNum,   &tiling,
                                 &tilingCfg, stream,     args.transA, args.transB, args.isHf32};

    if (args.dtype == "float32") {
        DISPATCH(float, float, args.transA, args.transB, launchParams);
    } else if (args.dtype == "bfloat16") {
        DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, launchParams);
    } else {
        DISPATCH(half, half, args.transA, args.transB, launchParams);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outPath = outputDir + "/npu_out.bin";
    if (!WriteFile(outPath, hostC.data(), sizeC)) {
        std::cerr << "Failed to write output" << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (deviceBias != nullptr) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
}

/* ========================================================================== */
/* Entry point                                                                */
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
