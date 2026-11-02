/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/**
 * @file fused_mat_mul_elemwise.cpp
 * @brief FusedMatMul elementwise add/mul example: y = op(A) @ op(B) + x3 or y = op(A) @ op(B) * x3.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_matmul_elemwise.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL()                                                                                      \
    fused_mat_mul_elemwise_kernel<ElementType, AccType, FusedOpType, LayoutA, LayoutB>                            \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dX3, p.dC, p.m, p.n, p.k, p.cfg->mL1, p.cfg->nL1, p.cfg->kL1, \
                                      p.cfg->baseM, p.cfg->baseN, p.cfg->baseK)

#define LAUNCH_STREAMK_IMPL()                                                                                \
    fused_mat_mul_elemwise_streamk_kernel<ElementType, AccType, FusedOpType, LayoutA, LayoutB>               \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dX3, p.dC, p.dWS, p.m, p.n, p.k, p.cfg->mL1, p.cfg->nL1, \
                                      p.cfg->kL1, p.cfg->baseM, p.cfg->baseN, p.cfg->baseK, p.usedCoreNum,   \
                                      p.singleCoreK)

#define DISPATCH_ELEMWISE(TYPE, ACC_TYPE, OP_TYPE, TRANS_A, TRANS_B, PARAMS, STREAMK) \
    do {                                                                              \
        if (TRANS_A) {                                                                \
            if (TRANS_B) {                                                            \
                LaunchKernel<TYPE, ACC_TYPE, OP_TYPE, true, true, STREAMK>(PARAMS);   \
            } else {                                                                  \
                LaunchKernel<TYPE, ACC_TYPE, OP_TYPE, true, false, STREAMK>(PARAMS);  \
            }                                                                         \
        } else {                                                                      \
            if (TRANS_B) {                                                            \
                LaunchKernel<TYPE, ACC_TYPE, OP_TYPE, false, true, STREAMK>(PARAMS);  \
            } else {                                                                  \
                LaunchKernel<TYPE, ACC_TYPE, OP_TYPE, false, false, STREAMK>(PARAMS); \
            }                                                                         \
        }                                                                             \
    } while (0)

// Op-type must be a compile-time template parameter; branch at the top level.
#define DISPATCH_PRECISION(TYPE, OP_TYPE, TRANS_A, TRANS_B, PARAMS, INNER_PRECISE, STREAMK) \
    do {                                                                                    \
        if (INNER_PRECISE == 0) {                                                           \
            DISPATCH_ELEMWISE(TYPE, float, OP_TYPE, TRANS_A, TRANS_B, PARAMS, STREAMK);     \
        } else {                                                                            \
            DISPATCH_ELEMWISE(TYPE, TYPE, OP_TYPE, TRANS_A, TRANS_B, PARAMS, STREAMK);      \
        }                                                                                   \
    } while (0)

/* ========================================================================== */
/* Tiling configuration                                                       */
/* ========================================================================== */

struct TilingConfig {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    static constexpr uint32_t L1_BUFFER_NUM = 2;
    static constexpr uint16_t L0C_DB = 2;
};

static TilingConfig GetTilingConfig() { return {64, 64, 64, 64, 64, 32}; }

// StreamK tiling: small M-tiles with the full N per tile, K split across cores.
static TilingConfig GetStreamKTilingConfig(int64_t n) { return {32, n, 256, 32, n, 32}; }

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    bool transA = false;
    bool transB = false;
    std::string dtype = "float16";
    std::string opType = "add";
    int64_t innerPrecise = 1;
    std::string mode = "basic";
};

static bool ParseBool(const char* value)
{
    const std::string str(value);
    return str == "true" || str == "True" || str == "1";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 4 || argc > 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <m> <k> <n> [transA] [transB] [dtype] [op_type] [inner_precise] [mode]\n";
        return false;
    }
    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    if (argc >= 5) {
        args.transA = ParseBool(argv[4]);
    }
    if (argc >= 6) {
        args.transB = ParseBool(argv[5]);
    }
    if (argc >= 7) {
        args.dtype = argv[6];
    }
    if (argc >= 8) {
        args.opType = argv[7];
    }
    if (argc >= 9) {
        args.innerPrecise = std::atoll(argv[8]);
    }
    if (argc >= 10) {
        args.mode = argv[9];
    }
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "Error: M, K, N must be positive integers.\n";
        return false;
    }
    if (args.dtype != "float16" && args.dtype != "bfloat16") {
        std::cerr << "Error: dtype must be float16 or bfloat16.\n";
        return false;
    }
    if (args.opType != "add" && args.opType != "mul") {
        std::cerr << "Error: op_type must be add or mul.\n";
        return false;
    }
    if (args.innerPrecise != 0 && args.innerPrecise != 1) {
        std::cerr << "Error: inner_precise must be 0 (high precision) or 1 (high performance).\n";
        return false;
    }
    if (args.mode != "basic" && args.mode != "streamk") {
        std::cerr << "Error: mode must be basic or streamk.\n";
        return false;
    }
    return true;
}

/* ========================================================================== */
/* Device-side kernel wrapper (template-based)                                */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                     false, true>;

template <typename ElementType, typename AccType, uint64_t FusedOpType, typename LayoutA, typename LayoutB>
__global__ __aicore__ void fused_mat_mul_elemwise_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR x3GM, GM_ADDR yGM, int64_t m,
                                                         int64_t n, int64_t k, int64_t mL1, int64_t nL1, int64_t kL1,
                                                         int64_t baseM, int64_t baseN, int64_t baseK)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, FusedOpType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, AccType,
                                                    LayoutC, ElementType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulElemwise<DispatchPolicy, ElementType, AccType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    // High precision (AccType=float) splits the FP32 accumulator between two AIVs.
    const bool splitM = std::is_same<AccType, float>::value;
    // ubPitchGran = 0: pitch is naturally expressed by c0_element<AccType> on the mmad side
    // (matching ACC_ALIGN on the epilogue side); no 16-element special case needed (b28d84d).
    constexpr uint32_t ubPitchGran = 0U;

    Params params = {{m, n, k, 1},
                     {aGM, bGM, nullptr, nullptr, nullptr, nullptr, static_cast<uint64_t>(k),
                      static_cast<uint64_t>(mL1), static_cast<uint64_t>(nL1), static_cast<uint64_t>(kL1),
                      static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK),
                      TilingConfig::L1_BUFFER_NUM, TilingConfig::L0C_DB, splitM ? 1U : 0U, 1U, 0U, ubPitchGran},
                     {},
                     {static_cast<uint32_t>(mL1), static_cast<uint32_t>(nL1), static_cast<uint32_t>(kL1),
                      static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), 1, 1, 1,
                      1, 1, 1, 0, Blaze::Gemm::L2_CACHE_DEFAULT, static_cast<uint32_t>(m), 1, 1}};
    // Epilogue params: x3 fusion input and output address.
    params.epilogueParams.cGmAddr = yGM;
    params.epilogueParams.x3GmAddr = x3GM;
    params.epilogueParams.x3BatchBroadcast = false;
    params.epilogueParams.x3M = m;
    params.epilogueParams.problemShape = params.problemShape;

    MatmulKernel kernel;
    kernel(params);
}

template <typename ElementType, typename AccType, uint64_t FusedOpType, typename LayoutA, typename LayoutB>
__global__ __aicore__ void fused_mat_mul_elemwise_streamk_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR x3GM, GM_ADDR yGM,
                                                                 GM_ADDR wsGM, int64_t m, int64_t n, int64_t k,
                                                                 int64_t mL1, int64_t nL1, int64_t kL1, int64_t baseM,
                                                                 int64_t baseN, int64_t baseK, int64_t usedCoreNum,
                                                                 int64_t singleCoreK)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<
        static_cast<Blaze::Gemm::MatMulL0C2Out>(Blaze::Gemm::ON_THE_FLY), FusedOpType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, float,
                                                    LayoutC, ElementType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<
        float, ElementType, DispatchPolicy, AscendC::IsSameType<AccType, ElementType>::value>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    // BlockMmad::Params: {a, b, c, bias, groupList, ws, mL1, nL1, kL1, baseM, baseN, baseK, l1Stages, l0cStages}
    Params params = {{m, n, k, 1},
                     {aGM, bGM, nullptr, nullptr, nullptr, wsGM, static_cast<uint64_t>(mL1), static_cast<uint64_t>(nL1),
                      static_cast<uint64_t>(kL1), static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN),
                      static_cast<uint32_t>(baseK), 2, 1},
                     {yGM, wsGM, 4, x3GM, false, m},
                     {usedCoreNum, baseM, baseN, baseK, static_cast<int64_t>(singleCoreK), static_cast<int64_t>(kL1), 0,
                      Blaze::Gemm::L2_CACHE_DEFAULT, 4}};
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
    uint8_t* dX3;
    uint8_t* dC;
    uint8_t* dWS{nullptr};
    int64_t m, n, k;
    int64_t blockNum;
    int64_t usedCoreNum{0};
    int64_t singleCoreK{0};
    const TilingConfig* cfg;
    aclrtStream stream;
};

template <typename ElementType, typename AccType, uint64_t FusedOpType, bool TransA, bool TransB, bool StreamK>
void LaunchKernel(const LaunchParams& p)
{
    using LayoutA = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    if constexpr (StreamK) {
        LAUNCH_STREAMK_IMPL();
    } else {
        LAUNCH_KERNEL_IMPL();
    }
}

} // namespace

// Op type must be a compile-time constant for the template parameter; mode selects
// the basic (fixpipe_opti) or streamk kernel instantiation.
template <bool StreamK>
void DispatchElemwise(const CliArgs& args, const LaunchParams& params)
{
    if (args.opType == "add") {
        constexpr uint64_t opType = 1; // OP_TYPE_ADD
        if (args.dtype == "bfloat16") {
            DISPATCH_PRECISION(bfloat16_t, opType, args.transA, args.transB, params, args.innerPrecise, StreamK);
        } else {
            DISPATCH_PRECISION(half, opType, args.transA, args.transB, params, args.innerPrecise, StreamK);
        }
    } else {
        constexpr uint64_t opType = 2; // OP_TYPE_MUL
        if (args.dtype == "bfloat16") {
            DISPATCH_PRECISION(bfloat16_t, opType, args.transA, args.transB, params, args.innerPrecise, StreamK);
        } else {
            DISPATCH_PRECISION(half, opType, args.transA, args.transB, params, args.innerPrecise, StreamK);
        }
    }
}

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static uint8_t* CopyToDevice(const std::vector<uint8_t>& host)
{
    uint8_t* device = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device), host.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    return device;
}

static bool Run(const CliArgs& args)
{
    const size_t elemSize = args.dtype == "float16" ? sizeof(half) : sizeof(bfloat16_t);
    const size_t sizeA = static_cast<size_t>(args.m) * args.k * elemSize;
    const size_t sizeB = static_cast<size_t>(args.k) * args.n * elemSize;
    const size_t sizeX3 = static_cast<size_t>(args.m) * args.n * elemSize;
    const size_t sizeY = sizeX3;
    std::vector<uint8_t> hostA(sizeA);
    std::vector<uint8_t> hostB(sizeB);
    std::vector<uint8_t> hostX3(sizeX3);
    std::vector<uint8_t> hostY(sizeY);
    if (!ReadFile("./input/input_a.bin", hostA.data(), sizeA) ||
        !ReadFile("./input/input_b.bin", hostB.data(), sizeB) ||
        !ReadFile("./input/input_x3.bin", hostX3.data(), sizeX3)) {
        return false;
    }

    aclrtStream stream = nullptr;
    ACLDeviceGuard guard(stream);
    const bool streamk = args.mode == "streamk";
    const TilingConfig basicTiling = GetTilingConfig();
    const TilingConfig streamkTiling = GetStreamKTilingConfig(args.n);
    const TilingConfig& tilingCfg = streamk ? streamkTiling : basicTiling;
    int64_t blockNum = 0;
    int64_t usedCoreNum = 0;
    int64_t singleCoreK = 0;
    if (streamk) {
        usedCoreNum = GetAicCoreNum();
        singleCoreK = std::max(tilingCfg.kL1, CeilDiv(args.k, usedCoreNum / 2));
        blockNum = usedCoreNum;
    } else {
        const int64_t tileNum = CeilDiv(args.m, tilingCfg.baseM) * CeilDiv(args.n, tilingCfg.baseN);
        blockNum = std::min(tileNum, GetAicCoreNum());
    }
    uint8_t* deviceA = CopyToDevice(hostA);
    uint8_t* deviceB = CopyToDevice(hostB);
    uint8_t* deviceX3 = CopyToDevice(hostX3);
    uint8_t* deviceY = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t* deviceWS = nullptr;
    if (streamk) {
        const int64_t skBlocks = CeilDiv(args.k, singleCoreK);
        const int64_t alignN = CeilAlign(tilingCfg.baseN, static_cast<int64_t>(32));
        const size_t sizeWS = static_cast<size_t>(usedCoreNum) * skBlocks * tilingCfg.baseM * alignN * sizeof(float);
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWS), sizeWS, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    std::cout << "FusedMatMul elemwise: M=" << args.m << ", K=" << args.k << ", N=" << args.n
              << ", dtype=" << args.dtype << ", op_type=" << args.opType << ", inner_precise=" << args.innerPrecise
              << ", mode=" << args.mode << ", blockNum=" << blockNum << std::endl;
    const LaunchParams params = {deviceA, deviceB,  deviceX3,    deviceY,     deviceWS,   args.m, args.n,
                                 args.k,  blockNum, usedCoreNum, singleCoreK, &tilingCfg, stream};
    if (streamk) {
        DispatchElemwise<true>(args, params);
    } else {
        DispatchElemwise<false>(args, params);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostY.data(), sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));
    const bool success = WriteFile("./output/npu_out.bin", hostY.data(), sizeY);
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX3));
    ACL_CHECK(aclrtFree(deviceY));
    if (deviceWS != nullptr) {
        ACL_CHECK(aclrtFree(deviceWS));
    }
    return success;
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
    return Run(args) ? 0 : 1;
}
