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
 * @file mat_mul_b_fullload.cpp
 * @brief Unified BFullLoad MatMul example supporting multiple dtypes and formats.
 *
 * B matrix is fully loaded into L1 buffer, A matrix is streamed in K-tiles.
 * Suitable for scenarios where N is small enough that B fits entirely in L1.
 *
 * Supported dtypes: float16, bfloat16, float32
 * Supported formats: (ND,ND), (ND,NZ)
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
#include "blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL()                                                                                        \
    matmul_b_fullload_kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C>                       \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dC, p.dBias, p.m, p.n, p.k, p.cfg->mL1, p.cfg->nL1, p.cfg->kL1, \
                                      p.cfg->baseM, p.cfg->baseN, p.cfg->baseK, p.tiling->mTailCnt,                 \
                                      p.tiling->nTailCnt, p.isHf32)

#define DISPATCH(TYPE, BIAS_TYPE, TRANS_A, TRANS_B, IS_NZ, PARAMS)                          \
    do {                                                                                    \
        if (TRANS_A) {                                                                      \
            if (TRANS_B) {                                                                  \
                if (IS_NZ) {                                                                \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, true, true>(PARAMS);    \
                } else {                                                                    \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, true, false>(PARAMS);   \
                }                                                                           \
            } else {                                                                        \
                if (IS_NZ) {                                                                \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, false, true>(PARAMS);   \
                } else {                                                                    \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, true, false, false>(PARAMS);  \
                }                                                                           \
            }                                                                               \
        } else {                                                                            \
            if (TRANS_B) {                                                                  \
                if (IS_NZ) {                                                                \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, true, true>(PARAMS);   \
                } else {                                                                    \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, true, false>(PARAMS);  \
                }                                                                           \
            } else {                                                                        \
                if (IS_NZ) {                                                                \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, false, true>(PARAMS);  \
                } else {                                                                    \
                    LaunchKernel<TYPE, TYPE, TYPE, BIAS_TYPE, false, false, false>(PARAMS); \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
    } while (0)

/* ========================================================================== */
/* Tiling configuration (dtype-driven)                                        */
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
        return {128, 256, 128, 128, 256, 32, 4};
    } else {
        return {128, 256, 128, 128, 256, 32, 2};
    }
}

/* ========================================================================== */
/* Tiling computation                                                         */
/* ========================================================================== */

struct BFullLoadTiling {
    uint32_t mTailCnt = 1;
    uint32_t nTailCnt = 1;
};

static constexpr int64_t BLOCK_16 = 16L;

static BFullLoadTiling ComputeTiling(TilingConfig& cfg, int64_t m, int64_t k, int64_t n)
{
    BFullLoadTiling tiling;

    cfg.baseN = n;
    cfg.nL1 = n;

    int64_t mCnt = CeilDiv(m, cfg.baseM);

    int64_t tailM = m - (mCnt - 1) * cfg.baseM;
    if (tailM > 0 && tailM < cfg.baseM) {
        int64_t splitM = CeilDiv(tailM, 2L);
        tiling.mTailCnt = static_cast<uint32_t>(CeilDiv(tailM, splitM));
    }

    cfg.mL1 = cfg.baseM;
    cfg.kL1 = cfg.baseK * 2;

    return tiling;
}

/* ========================================================================== */
/* NZ format utilities (for (ND,NZ) format)                                   */
/* ========================================================================== */

inline int64_t CalcNZSize(int64_t k, int64_t n)
{
    int64_t kCeil = (k + BLOCK_16 - 1) / BLOCK_16 * BLOCK_16;
    int64_t nCeil = (n + BLOCK_16 - 1) / BLOCK_16 * BLOCK_16;
    return kCeil * nCeil;
}

inline void ConvertToNZ(const half* rowMajor, half* nzBuffer, int64_t k, int64_t n, bool transB)
{
    int64_t kCeil = CeilAlign(k, BLOCK_16);
    int64_t nCeil = CeilAlign(n, BLOCK_16);
    int64_t numKTiles = kCeil / BLOCK_16;
    int64_t numNTiles = nCeil / BLOCK_16;

    int64_t outIdx = 0;
    auto writeTile = [&](int64_t ki, int64_t ni, bool kInner) {
        for (int64_t a = 0; a < BLOCK_16; a++) {
            for (int64_t b = 0; b < BLOCK_16; b++) {
                int64_t kIdx = ki * BLOCK_16 + (kInner ? a : b);
                int64_t nIdx = ni * BLOCK_16 + (kInner ? b : a);
                nzBuffer[outIdx++] = (kIdx < k && nIdx < n) ? rowMajor[kIdx * n + nIdx] : static_cast<half>(0.0f);
            }
        }
    };

    if (transB) {
        for (int64_t ki = 0; ki < numKTiles; ki++)
            for (int64_t ni = 0; ni < numNTiles; ni++)
                writeTile(ki, ni, false);
    } else {
        for (int64_t ni = 0; ni < numNTiles; ni++)
            for (int64_t ki = 0; ki < numKTiles; ki++)
                writeTile(ki, ni, true);
    }
}

inline void TransposeMatrix(const half* src, half* dst, int64_t rows, int64_t cols)
{
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m, k, n;
    bool transA = false;
    bool transB = false;
    std::string dtype = "float16";
    bool isHf32 = false;
    int64_t bias = 0;
    std::string format = "ND";
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 4) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n> [transA] [transB] [dtype] [isHf32] [bias] [format]\n";
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);

    if (argc >= 5)
        args.transA = ParseBool(argv[4]);
    if (argc >= 6)
        args.transB = ParseBool(argv[5]);
    if (argc >= 7)
        args.dtype = argv[6];
    if (argc >= 8)
        args.isHf32 = ParseBool(argv[7]);
    if (argc >= 9)
        args.bias = std::atoll(argv[8]);
    if (argc >= 10)
        args.format = argv[9];

    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "Error: M, K, N must be positive integers.\n";
        return false;
    }
    if (args.bias != 0 && args.bias != args.n) {
        std::cerr << "Error: bias (" << args.bias << ") must equal n (" << args.n << ") or be 0\n";
        return false;
    }
    if (args.isHf32 && args.dtype != "float32") {
        std::cerr << "Error: isHf32 only valid with float32\n";
        return false;
    }
    if (args.format == "NZ" && args.dtype != "float16" && args.dtype != "bfloat16") {
        std::cerr << "Error: (ND,NZ) format only valid with float16/bfloat16\n";
        return false;
    }
    if (args.dtype != "float16" && args.dtype != "bfloat16" && args.dtype != "float32") {
        std::cerr << "Error: dtype must be float16, bfloat16, or float32 (got '" << args.dtype << "')\n";
        return false;
    }
    if (args.format != "ND" && args.format != "NZ") {
        std::cerr << "Error: format must be (ND,ND) or (ND,NZ) (got '" << args.format << "')\n";
        return false;
    }

    return true;
}

/* ========================================================================== */
/* Device-side kernel wrapper (template-based)                                */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::B_FULL_LOAD_MODE>;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class LAYOUT_A, class LAYOUT_B, class LAYOUT_C>
__global__ __aicore__ void matmul_b_fullload_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int64_t m,
                                                    int64_t n, int64_t k, int64_t mL1, int64_t nL1, int64_t kL1,
                                                    int64_t baseM, int64_t baseN, int64_t baseK, uint32_t mTailCnt,
                                                    uint32_t nTailCnt, bool isHf32)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBFullLoad<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, LAYOUT_A, B_TYPE, LAYOUT_B, C_TYPE,
                                                    LAYOUT_C, BIAS_TYPE, LAYOUT_C>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {m, n, k, 1},
        {aGM, bGM, cGM, biasGM, nullptr, nullptr, static_cast<uint64_t>(k), static_cast<uint64_t>(mL1),
         static_cast<uint64_t>(nL1), static_cast<uint64_t>(kL1), static_cast<uint32_t>(baseM),
         static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), TilingConfig::L1_BUFFER_NUM, TilingConfig::L0C_DB,
         0, 1},
        {},
        {static_cast<uint32_t>(baseM), static_cast<uint32_t>(nL1), static_cast<uint32_t>(kL1),
         static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), mTailCnt, nTailCnt,
         1, 1, 1, 1, static_cast<uint8_t>(isHf32 ? 1 : 0), Blaze::Gemm::L2_CACHE_DEFAULT}};

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
    int64_t blockNum;
    const BFullLoadTiling* tiling;
    const TilingConfig* cfg;
    aclrtStream stream;
    bool transA, transB, isHf32;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool TransA, bool TransB, bool IsNzFormat>
void LaunchKernel(const LaunchParams& p)
{
    using LAYOUT_A = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;

    using LAYOUT_B = std::conditional_t<
        IsNzFormat, std::conditional_t<TransB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>>;

    using LAYOUT_C = asc::te::nd_ext_layout_ptn;

    LAUNCH_KERNEL_IMPL();
}

} // namespace

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};

    TilingConfig tilingCfg = GetTilingConfig(args.dtype);
    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0) {
        std::cout << "blockNum cannot less than 0, but current: " << blockNum << std::endl;
        return;
    }

    ACLDeviceGuard guard(stream);
    BFullLoadTiling tiling = ComputeTiling(tilingCfg, args.m, args.k, args.n);

    size_t dtypeSize = (args.dtype == "float32") ? sizeof(float) : sizeof(half);
    size_t sizeA = static_cast<size_t>(args.m) * args.k * dtypeSize;
    size_t sizeC = static_cast<size_t>(args.m) * args.n * dtypeSize;
    size_t sizeB = static_cast<size_t>(args.k) * args.n * dtypeSize;
    size_t sizeBias = (args.bias > 0) ? static_cast<size_t>(args.bias) * dtypeSize : 0;

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

    std::vector<uint8_t> hostBNz;
    if (args.format == "NZ") {
        int64_t bRows = args.transB ? args.n : args.k;
        int64_t bCols = args.transB ? args.k : args.n;
        int64_t lenBND = bRows * bCols;
        size_t sizeBND = static_cast<size_t>(lenBND) * dtypeSize;

        std::vector<uint8_t> hostBND(sizeBND);
        std::memcpy(hostBND.data(), hostB.data(), sizeBND);

        int64_t lenBNz = CalcNZSize(args.k, args.n);
        sizeB = static_cast<size_t>(lenBNz) * dtypeSize;
        hostBNz.resize(sizeB);

        if (args.transB) {
            std::vector<half> hostBForNZ(args.k * args.n);
            TransposeMatrix(reinterpret_cast<half*>(hostBND.data()), hostBForNZ.data(), bRows, bCols);
            ConvertToNZ(hostBForNZ.data(), reinterpret_cast<half*>(hostBNz.data()), args.k, args.n, args.transB);
        } else {
            ConvertToNZ(reinterpret_cast<half*>(hostBND.data()), reinterpret_cast<half*>(hostBNz.data()), args.k,
                        args.n, args.transB);
        }
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
    if (args.format == "NZ") {
        ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostBNz.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::string layoutA = args.transA ? "DN" : "ND";
    std::string layoutB = (args.format == "NZ") ? (args.transB ? "ZN" : "NZ") : (args.transB ? "DN" : "ND");

    std::cout << "============================================================" << std::endl;
    std::cout << "  MatMul BFullLoad — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << std::endl;
    std::cout << "  Dtype    : " << args.dtype << std::endl;
    std::cout << "  Layout   : " << layoutA << " x " << layoutB << " -> ND" << std::endl;
    std::cout << "  transA   : " << (args.transA ? "true" : "false") << std::endl;
    std::cout << "  transB   : " << (args.transB ? "true" : "false") << std::endl;
    std::cout << "  isHf32   : " << (args.isHf32 ? "true" : "false") << std::endl;
    std::cout << "  bias     : " << args.bias << std::endl;
    std::cout << "  Format   : " << args.format << std::endl;
    std::cout << "  L1 Tile  : [" << tilingCfg.mL1 << ", " << tilingCfg.nL1 << ", " << tilingCfg.kL1 << "]"
              << std::endl;
    std::cout << "  L0 Tile  : [" << tilingCfg.baseM << ", " << tilingCfg.baseN << ", " << tilingCfg.baseK << "]"
              << std::endl;
    std::cout << "  oriK     : " << args.k << " (B fully loaded)" << std::endl;
    std::cout << "  BlockNum : " << blockNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams launchParams = {deviceA,  deviceB, deviceC,    deviceBias, args.m,      args.n,      args.k,
                                 blockNum, &tiling, &tilingCfg, stream,     args.transA, args.transB, args.isHf32};

    bool isNzFormat = (args.format == "NZ");

    if (args.dtype == "float32") {
        DISPATCH(float, float, args.transA, args.transB, isNzFormat, launchParams);
    } else if (args.dtype == "bfloat16") {
        DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, isNzFormat, launchParams);
    } else {
        DISPATCH(half, half, args.transA, args.transB, isNzFormat, launchParams);
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
