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
 * @file mat_mul_iterbatch_broadcast.cpp
 * @brief IterBatch-Broadcast MatMul example supporting multiple dtypes.
 *
 * Combines batch broadcast with iterbatch L1/L0 pipelining.
 * Multiple batches are loaded into L1 and processed in L0 pipeline.
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
#include "blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_scheduler_iterbatch_broadcast.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL()                                                                                         \
    iterbatch_broadcast_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B, A_BC, B_BC>                    \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dC, p.dBias, p.m, p.n, p.k, p.batch, p.cfg->baseM, p.cfg->baseN, \
                                      p.cfg->baseK, p.iterBatchL1, p.iterBatchL0, p.batchA, p.batchB, p.isHf32)

#define DISPATCH(TYPE, BIAS_TYPE, TRANS_A, TRANS_B, A_BC, B_BC, PARAMS)                      \
    do {                                                                                     \
        if (TRANS_A) {                                                                       \
            if (TRANS_B) {                                                                   \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, true, A_BC, B_BC>(PARAMS);   \
            } else {                                                                         \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, false, A_BC, B_BC>(PARAMS);  \
            }                                                                                \
        } else {                                                                             \
            if (TRANS_B) {                                                                   \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, true, A_BC, B_BC>(PARAMS);  \
            } else {                                                                         \
                LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, false, A_BC, B_BC>(PARAMS); \
            }                                                                                \
        }                                                                                    \
    } while (0)

/* ========================================================================== */
/* Tiling configuration                                                        */
/* ========================================================================== */

struct TilingConfig {
    int64_t baseM, baseN, baseK;
    int64_t dtypeSize;
};

static TilingConfig GetTilingConfig(const std::string& dtype)
{
    if (dtype == "float32") {
        return {32, 32, 32, 4};
    } else {
        return {32, 32, 32, 2};
    }
}

static void ComputeTiling(TilingConfig& cfg, int64_t m, int64_t k, int64_t n)
{
    cfg.baseM = m;
    cfg.baseN = n;
    cfg.baseK = k;
}

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m, k, n;
    int64_t batch = 1;
    int64_t batchA = 1;
    int64_t batchB = 1;
    int64_t iterBatchL1 = 1;
    int64_t iterBatchL0 = 1;
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
    if (argc < 9) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: " << argv[0]
                  << " <m> <k> <n> <batch> <batchA> <batchB> <iterBatchL1> <iterBatchL0> "
                     "[transA] [transB] [dtype] [isHf32] [bias]\n";
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);
    args.batchA = std::atoll(argv[5]);
    args.batchB = std::atoll(argv[6]);
    args.iterBatchL1 = std::atoll(argv[7]);
    args.iterBatchL0 = std::atoll(argv[8]);

    if (argc >= 10)
        args.transA = ParseBool(argv[9]);
    if (argc >= 11)
        args.transB = ParseBool(argv[10]);
    if (argc >= 12)
        args.dtype = argv[11];
    if (argc >= 13)
        args.isHf32 = ParseBool(argv[12]);
    if (argc >= 14)
        args.bias = std::atoll(argv[13]);

    if (args.m <= 0 || args.k <= 0 || args.n <= 0 || args.batch <= 0) {
        std::cerr << "Error: M, K, N, batch must be positive integers.\n";
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B, bool A_BC,
          bool B_BC>
__global__ __aicore__ void iterbatch_broadcast_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m,
                                                      int64_t n, int64_t k, int64_t batch, int64_t baseM, int64_t baseN,
                                                      int64_t baseK, int64_t iterBatchL1, int64_t iterBatchL0,
                                                      int64_t batchA, int64_t batchB, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using LAYOUT_C = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulIterBatchBroadcast<A_BC, B_BC>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, LAYOUT_A, B_TYPE, LAYOUT_B, C_TYPE,
                                                    LAYOUT_C, BIAS_TYPE, LAYOUT_C>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerIterBatchBroadcast<ProblemShape>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    // broadcastAxisA and broadcastAxisB need init
    uint32_t aDims[4] = {1, 1, 1, static_cast<uint32_t>(batchA)};
    uint32_t bDims[4] = {1, 1, 1, static_cast<uint32_t>(batchB)};
    uint32_t cDims[4] = {1, 1, 1, static_cast<uint32_t>(batch)};
    uint32_t bcAxisA = 4;
    uint32_t bcAxisB = 4;
    for (uint32_t i = 0; i < 4; ++i) {
        if (aDims[i] != cDims[i]) {
            bcAxisA = i;
            break;
        }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (bDims[i] != cDims[i]) {
            bcAxisB = i;
            break;
        }
    }

    typename BlockScheduler::Params schParams;
    schParams.baseM = static_cast<uint32_t>(baseM);
    schParams.baseN = static_cast<uint32_t>(baseN);
    schParams.baseK = static_cast<uint32_t>(baseK);
    schParams.iterBatchL1 = static_cast<uint32_t>(iterBatchL1);
    schParams.iterBatchL0 = static_cast<uint32_t>(iterBatchL0);
    schParams.aBatchDim0 = aDims[0];
    schParams.aBatchDim1 = aDims[1];
    schParams.aBatchDim2 = aDims[2];
    schParams.aBatchDim3 = aDims[3];
    schParams.bBatchDim0 = bDims[0];
    schParams.bBatchDim1 = bDims[1];
    schParams.bBatchDim2 = bDims[2];
    schParams.bBatchDim3 = bDims[3];
    schParams.cBatchDim0 = cDims[0];
    schParams.cBatchDim1 = cDims[1];
    schParams.cBatchDim2 = cDims[2];
    schParams.cBatchDim3 = cDims[3];
    schParams.broadcastAxisA = bcAxisA;
    schParams.broadcastAxisB = bcAxisB;
    schParams.isHf32 = isHf32 ? 1 : 0;

    typename BlockMmad::Params mmadParams;
    mmadParams.aGmAddr = aGM;
    mmadParams.bGmAddr = bGM;
    mmadParams.cGmAddr = cGM;
    mmadParams.biasGmAddr = biasGM;
    mmadParams.m = static_cast<uint64_t>(m);
    mmadParams.n = static_cast<uint64_t>(n);
    mmadParams.k = static_cast<uint64_t>(k);
    mmadParams.baseM = static_cast<uint64_t>(baseM);
    mmadParams.baseN = static_cast<uint64_t>(baseN);
    mmadParams.baseK = static_cast<uint64_t>(baseK);
    mmadParams.iterBatchL1 = static_cast<uint64_t>(iterBatchL1);
    mmadParams.iterBatchL0 = static_cast<uint64_t>(iterBatchL0);

    Params params;
    params.problemShape = {m, n, k, batch};
    params.mmadParams = mmadParams;
    params.schedulerParams = schParams;

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
    int64_t iterBatchL1, iterBatchL0;
    int64_t blockNum;
    const TilingConfig* cfg;
    aclrtStream stream;
    bool transA, transB, isHf32;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool TransA, bool TransB, bool A_BC, bool B_BC>
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
    ComputeTiling(tilingCfg, args.m, args.k, args.n);
    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0) {
        std::cout << "blockNum cannot less than 0, but current: " << blockNum << std::endl;
        return;
    }

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

    bool aBroadcast = (args.batchA == 1 && args.batch > 1);
    bool bBroadcast = (args.batchB == 1 && args.batch > 1);

    std::cout << "============================================================" << std::endl;
    std::cout << "  IterBatch-Broadcast — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << std::endl;
    std::cout << "  Batch    : C=" << args.batch << ", A=" << args.batchA << ", B=" << args.batchB << std::endl;
    std::cout << "  IterBatch: L1=" << args.iterBatchL1 << ", L0=" << args.iterBatchL0 << std::endl;
    std::cout << "  Broadcast: A=" << (aBroadcast ? "true" : "false") << ", B=" << (bBroadcast ? "true" : "false")
              << std::endl;
    std::cout << "  Dtype    : " << args.dtype << std::endl;
    std::cout << "  L0 Tile  : [" << tilingCfg.baseM << ", " << tilingCfg.baseN << ", " << tilingCfg.baseK << "]"
              << std::endl;
    std::cout << "  BlockNum : " << blockNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams launchParams = {deviceA,  deviceB,    deviceC,     deviceBias,  args.m,           args.n,
                                 args.k,   args.batch, args.batchA, args.batchB, args.iterBatchL1, args.iterBatchL0,
                                 blockNum, &tilingCfg, stream,      args.transA, args.transB,      args.isHf32};

    if (args.dtype == "float32") {
        if (aBroadcast && !bBroadcast) {
            DISPATCH(float, float, args.transA, args.transB, true, false, launchParams);
        } else if (!aBroadcast && bBroadcast) {
            DISPATCH(float, float, args.transA, args.transB, false, true, launchParams);
        } else if (aBroadcast && bBroadcast) {
            DISPATCH(float, float, args.transA, args.transB, true, true, launchParams);
        } else {
            DISPATCH(float, float, args.transA, args.transB, false, false, launchParams);
        }
    } else if (args.dtype == "bfloat16") {
        if (aBroadcast && !bBroadcast) {
            DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, true, false, launchParams);
        } else if (!aBroadcast && bBroadcast) {
            DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, false, true, launchParams);
        } else if (aBroadcast && bBroadcast) {
            DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, true, true, launchParams);
        } else {
            DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, false, false, launchParams);
        }
    } else {
        if (aBroadcast && !bBroadcast) {
            DISPATCH(half, half, args.transA, args.transB, true, false, launchParams);
        } else if (!aBroadcast && bBroadcast) {
            DISPATCH(half, half, args.transA, args.transB, false, true, launchParams);
        } else if (aBroadcast && bBroadcast) {
            DISPATCH(half, half, args.transA, args.transB, true, true, launchParams);
        } else {
            DISPATCH(half, half, args.transA, args.transB, false, false, launchParams);
        }
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
