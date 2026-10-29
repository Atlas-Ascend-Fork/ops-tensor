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
 * @file mat_mul_iterbatch.cpp
 * @brief IterBatch MatMul example: ON_THE_FLY (AIC_ONLY) and
 *        ND_FIXPIPE_1_2 (MIX_AIC_1_2, AIV ND epilogue) variants.
 *
 * CLI: m k n batch [mode] [transA] [transB] [dtype] [isHf32] [bias]
 *   mode: 0 = ON_THE_FLY (default), 2 = ND_FIXPIPE_1_2
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
#include "blaze/epilogue/block/block_epilogue_iterbatch.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_iterbatch.h"
#include "blaze/gemm/block/block_scheduler_matmul_iterbatch.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_matmul_iterbatch.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"

/* ========================================================================== */
/* Tiling configuration (host tiling semantics)                              */
/* ========================================================================== */

struct TilingConfig {
    int64_t baseM, baseN, baseK;
    int64_t iterBatchL1, iterBatchL0;
};

static TilingConfig ComputeTiling(int64_t m, int64_t k, int64_t n, int64_t batch, int64_t aicNum, int64_t dtypeSize,
                                  bool hasBias)
{
    TilingConfig cfg;
    // Cube-fractal alignment granularity in elements (mirrors kernel-side BLOCK_CUBE).
    constexpr int64_t CUBE_ALIGN = 16;
    // Cap the MNK tile so a single tile fits the L0 ping-pong half-buffers; the kernel
    // walks any remainder in M/N/K steps.
    constexpr int64_t BASE_TILE_CAP = 256;
    auto ceilAlign = [](int64_t v) { return (v + CUBE_ALIGN - 1) / CUBE_ALIGN * CUBE_ALIGN; };
    int64_t alignM = ceilAlign(m);
    int64_t alignN = ceilAlign(n);
    int64_t alignK = ceilAlign(k);
    constexpr int64_t l0ASize = 65536;
    constexpr int64_t l0BSize = 65536;
    constexpr int64_t l0CSize = 262144;
    constexpr int64_t l1Size = 524288;
    constexpr int64_t db = 2;
    auto floorAlign = [CUBE_ALIGN](int64_t v) { return std::max<int64_t>(v / CUBE_ALIGN * CUBE_ALIGN, CUBE_ALIGN); };
    int64_t baseM = std::min<int64_t>(alignM, BASE_TILE_CAP);
    int64_t baseN = std::min<int64_t>(alignN, BASE_TILE_CAP);
    int64_t baseK = std::min<int64_t>(alignK, BASE_TILE_CAP);
    while ((baseM * baseK * dtypeSize > l0ASize / db || baseK * baseN * dtypeSize > l0BSize / db ||
            baseM * baseN * sizeof(float) > l0CSize / db) &&
           (baseM > CUBE_ALIGN || baseN > CUBE_ALIGN || baseK > CUBE_ALIGN)) {
        if (baseM * baseK * dtypeSize > l0ASize / db && baseM >= baseK) {
            baseM = floorAlign(baseM / 2);
        } else if (baseK * baseN * dtypeSize > l0BSize / db && baseN >= baseK) {
            baseN = floorAlign(baseN / 2);
        } else if (baseM * baseN * sizeof(float) > l0CSize / db) {
            baseN = floorAlign(baseN / 2);
        } else if (baseM * baseK * dtypeSize > l0ASize / db) {
            baseK = floorAlign(baseK / 2);
        } else if (baseK * baseN * dtypeSize > l0BSize / db) {
            baseK = floorAlign(baseK / 2);
        } else {
            baseM = floorAlign(baseM / 2);
        }
    }
    int64_t iterBatchL0A = (l0ASize / db) / (baseM * baseK * dtypeSize);
    int64_t iterBatchL0B = (l0BSize / db) / (baseK * baseN * dtypeSize);
    int64_t iterBatchL0C = (l0CSize / db) / (baseM * baseN * sizeof(float));
    int64_t biasBytes = hasBias ? alignN * dtypeSize : 0;
    int64_t iterBatchL1Cap = (l1Size / db - biasBytes) / ((alignM * alignK + alignK * alignN) * dtypeSize);
    bool l0CanLoadBatch = (iterBatchL0A >= 1 && iterBatchL0B >= 1 && iterBatchL0C >= 1);
    cfg.baseM = baseM;
    cfg.baseN = baseN;
    cfg.baseK = baseK;
    cfg.iterBatchL0 = 1;
    cfg.iterBatchL1 = 1;
    if (l0CanLoadBatch) {
        cfg.iterBatchL0 = std::min(std::min(iterBatchL0A, iterBatchL0B), iterBatchL0C);
        cfg.iterBatchL1 = std::max(std::min(iterBatchL1Cap, (batch + aicNum - 1) / aicNum), int64_t(1));
        if (cfg.iterBatchL1 == cfg.iterBatchL0) {
            cfg.iterBatchL0 = std::max(cfg.iterBatchL0 / 2, int64_t(1));
        }
        cfg.iterBatchL0 = std::max(std::min(cfg.iterBatchL0, cfg.iterBatchL1), int64_t(1));
        // Host tiling contract (l0CanLoadBatch): multi-batch L0 implies the base block
        // covers the whole aligned m/n/k; if the L0-fit loop shrank any base below its
        // aligned dim the tile is M/N/K-blocked and must load one batch per L0 step.
        if (baseM < alignM || baseN < alignN || baseK < alignK) {
            cfg.iterBatchL0 = 1;
        }
    }
    return cfg;
}

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m, k, n;
    int64_t batch = 1;
    int64_t mode = 0;
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
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n> <batch> [mode] [transA] [transB] [dtype] [isHf32] [bias]\n";
        return false;
    }
    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);
    if (argc >= 6) {
        args.mode = std::atoll(argv[5]);
    }
    int64_t shift = (argc >= 6) ? 1 : 0;
    if (argc >= 6 + shift) {
        args.transA = ParseBool(argv[5 + shift]);
    }
    if (argc >= 7 + shift) {
        args.transB = ParseBool(argv[6 + shift]);
    }
    if (argc >= 8 + shift) {
        args.dtype = argv[7 + shift];
    }
    if (argc >= 9 + shift) {
        args.isHf32 = ParseBool(argv[8 + shift]);
    }
    if (argc >= 10 + shift) {
        args.bias = std::atoll(argv[9 + shift]);
    }
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

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B,
          Blaze::Gemm::MatMulL0C2Out FIXP_OPT>
__aicore__ inline void RunIterBatch(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m, int64_t n,
                                    int64_t k, int64_t batch, int64_t baseM, int64_t baseN, int64_t baseK,
                                    int64_t iterBatchL1, int64_t iterBatchL0, bool isHf32)
{
    using LAYOUT_C = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulIterBatch<FIXP_OPT>;
    using BlockMmadT = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, LAYOUT_A, B_TYPE, LAYOUT_B, C_TYPE,
                                                     LAYOUT_C, BIAS_TYPE, LAYOUT_C>;
    using BlockSchedulerT = Blaze::Gemm::Block::BlockSchedulerMatmulIterBatch<ProblemShape>;
    using BlockEpilogueT = AscendC::Std::conditional_t<FIXP_OPT == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2,
                                                       Blaze::Epilogue::Block::BlockEpilogueIterbatch<C_TYPE, C_TYPE>,
                                                       Blaze::Gemm::Block::BlockEpilogueEmpty>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmadT, BlockEpilogueT, BlockSchedulerT>;

    typename BlockSchedulerT::Params schParams;
    schParams.baseM = static_cast<uint32_t>(baseM);
    schParams.baseN = static_cast<uint32_t>(baseN);
    schParams.baseK = static_cast<uint32_t>(baseK);
    schParams.iterBatchL1 = static_cast<uint32_t>(iterBatchL1);
    schParams.iterBatchL0 = static_cast<uint32_t>(iterBatchL0);
    schParams.isHf32 = isHf32 ? 1 : 0;

    typename BlockMmadT::Params mmadParams;
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

    typename MatmulKernel::Params params;
    params.problemShape = {m, n, k, batch};
    params.mmadParams = mmadParams;
    params.schedulerParams = schParams;
    if constexpr (FIXP_OPT == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2) {
        params.epilogueParams.cGmAddr = cGM;
    }

    MatmulKernel kernel;
    kernel(params);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B>
__global__ __aicore__ void iterbatch_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m, int64_t n,
                                            int64_t k, int64_t batch, int64_t baseM, int64_t baseN, int64_t baseK,
                                            int64_t iterBatchL1, int64_t iterBatchL0, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    RunIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B, Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>(
        aGM, bGM, cGM, biasGM, m, n, k, batch, baseM, baseN, baseK, iterBatchL1, iterBatchL0, isHf32);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B>
__global__ __aicore__ void iterbatch_mix_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m,
                                                int64_t n, int64_t k, int64_t batch, int64_t baseM, int64_t baseN,
                                                int64_t baseK, int64_t iterBatchL1, int64_t iterBatchL0, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    RunIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>(
        aGM, bGM, cGM, biasGM, m, n, k, batch, baseM, baseN, baseK, iterBatchL1, iterBatchL0, isHf32);
}

/* ========================================================================== */
/* Host-side launcher                                                         */
/* ========================================================================== */

namespace {

struct LaunchParams {
    uint8_t* dA;
    uint8_t* dB;
    uint8_t* dC;
    uint8_t* dBias;
    int64_t m, n, k;
    int64_t batch;
    int64_t baseM, baseN, baseK;
    int64_t iterBatchL1, iterBatchL0;
    int64_t blockNum;
    aclrtStream stream;
    bool transA, transB, isHf32;
    int64_t mode;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool TransA, bool TransB>
void LaunchIterBatch(const LaunchParams& p)
{
    using LAYOUT_A = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LAYOUT_B = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    if (p.mode == 2) {
        iterbatch_mix_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B>
            <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dC, p.dBias, p.m, p.n, p.k, p.batch, p.baseM, p.baseN, p.baseK,
                                          p.iterBatchL1, p.iterBatchL0, p.isHf32);
    } else {
        iterbatch_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B>
            <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dC, p.dBias, p.m, p.n, p.k, p.batch, p.baseM, p.baseN, p.baseK,
                                          p.iterBatchL1, p.iterBatchL0, p.isHf32);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
void LaunchIterBatchWithTrans(const LaunchParams& p)
{
    if (!p.transA && !p.transB) {
        LaunchIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, false, false>(p);
    } else if (p.transA && !p.transB) {
        LaunchIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, false>(p);
    } else if (!p.transA && p.transB) {
        LaunchIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, false, true>(p);
    } else {
        LaunchIterBatch<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, true>(p);
    }
}

void LaunchIterBatchDispatch(const LaunchParams& p, const std::string& dtype)
{
    if (dtype == "float32") {
        LaunchIterBatchWithTrans<float, float, float, float>(p);
    } else if (dtype == "bfloat16") {
        LaunchIterBatchWithTrans<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>(p);
    } else {
        LaunchIterBatchWithTrans<half, half, half, half>(p);
    }
}

} // namespace

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);

    int64_t aicNum = GetAicCoreNum();
    if (aicNum <= 0) {
        std::cout << "blockNum cannot less than 0, but current: " << aicNum << std::endl;
        return;
    }
    int64_t dtypeSize = static_cast<int64_t>((args.dtype == "float32") ? sizeof(float) : sizeof(half));
    TilingConfig cfg = ComputeTiling(args.m, args.k, args.n, args.batch, aicNum, dtypeSize, args.bias > 0);

    size_t sizeA = static_cast<size_t>(args.batch) * args.m * args.k * dtypeSize;
    size_t sizeB = static_cast<size_t>(args.batch) * args.k * args.n * dtypeSize;
    size_t sizeC = static_cast<size_t>(args.batch) * args.m * args.n * dtypeSize;
    size_t sizeBias = (args.bias > 0) ? static_cast<size_t>(args.n) * dtypeSize : 0;

    std::string inputDir = "./input";
    std::string outputDir = "./output";
    struct stat st;
    std::vector<uint8_t> hostA(sizeA);
    std::vector<uint8_t> hostB(sizeB);
    std::vector<uint8_t> hostC(sizeC, 0);
    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA)) {
        std::cerr << "Failed to read input A" << std::endl;
        return;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
        std::cerr << "Failed to read input B" << std::endl;
        return;
    }

    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<uint8_t> hostBias(sizeBias, 0);
    if (args.bias > 0) {
        std::string biasPath = inputDir + "/bias.bin";
        if (stat(biasPath.c_str(), &st) == 0) {
            if (!ReadFile(biasPath, hostBias.data(), sizeBias)) {
                std::cerr << "Failed to read bias" << std::endl;
                return;
            }
        }
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), sizeBias, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "  IterBatch-Basic — Execution Summary" << std::endl;
    std::cout << "  Mode     : " << (args.mode == 2 ? "ND_FIXPIPE_1_2 (MIX)" : "ON_THE_FLY (AIC_ONLY)") << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << ", Batch=" << args.batch
              << std::endl;
    std::cout << "  Base     : [" << cfg.baseM << ", " << cfg.baseN << ", " << cfg.baseK << "]" << std::endl;
    std::cout << "  IterBatch: L1=" << cfg.iterBatchL1 << ", L0=" << cfg.iterBatchL0 << std::endl;
    std::cout << "  BlockNum : " << aicNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams p{deviceA,    deviceB,     deviceC,     deviceBias,  args.m,          args.n,          args.k,
                   args.batch, cfg.baseM,   cfg.baseN,   cfg.baseK,   cfg.iterBatchL1, cfg.iterBatchL0, aicNum,
                   stream,     args.transA, args.transB, args.isHf32, args.mode};
    LaunchIterBatchDispatch(p, args.dtype);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(outputDir + "/npu_out.bin", hostC.data(), sizeC)) {
        std::cerr << "Failed to write output" << std::endl;
    }
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (deviceBias != nullptr) {
        ACL_CHECK(aclrtFree(deviceBias));
    }
}

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseCliArgs(argc, argv, args)) {
        return 1;
    }
    Run(args);
    return 0;
}
