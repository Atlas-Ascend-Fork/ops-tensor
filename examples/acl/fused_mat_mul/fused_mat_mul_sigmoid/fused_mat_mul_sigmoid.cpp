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
 * @file fused_mat_mul_sigmoid.cpp
 * @brief FusedMatMul sigmoid example with optional bias and FP32 accumulation.
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
#include "blaze/epilogue/block/block_epilogue_matmul_with_sigmoid.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL()                                                                                        \
    fused_mat_mul_sigmoid_kernel<ElementType, BiasType, LayoutA, LayoutB>                                           \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dBias, p.dC, p.m, p.n, p.k, p.cfg->mL1, p.cfg->nL1, p.cfg->kL1, \
                                      p.cfg->baseM, p.cfg->baseN, p.cfg->baseK)

#define DISPATCH(TYPE, BIAS_TYPE, TRANS_A, TRANS_B, PARAMS)          \
    do {                                                             \
        if (TRANS_A) {                                               \
            if (TRANS_B) {                                           \
                LaunchKernel<TYPE, BIAS_TYPE, true, true>(PARAMS);   \
            } else {                                                 \
                LaunchKernel<TYPE, BIAS_TYPE, true, false>(PARAMS);  \
            }                                                        \
        } else {                                                     \
            if (TRANS_B) {                                           \
                LaunchKernel<TYPE, BIAS_TYPE, false, true>(PARAMS);  \
            } else {                                                 \
                LaunchKernel<TYPE, BIAS_TYPE, false, false>(PARAMS); \
            }                                                        \
        }                                                            \
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
    std::string biasDtype = "none";
};

static bool ParseBool(const char* value)
{
    const std::string str(value);
    return str == "true" || str == "True" || str == "1";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 4 || argc > 8) {
        std::cerr << "Usage: " << argv[0] << " <m> <k> <n> [transA] [transB] [dtype] [bias_dtype]\n";
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
        args.biasDtype = argv[7];
    }
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "Error: M, K, N must be positive integers.\n";
        return false;
    }
    if (args.dtype != "float16" && args.dtype != "bfloat16") {
        std::cerr << "Error: dtype must be float16 or bfloat16.\n";
        return false;
    }
    if (args.biasDtype != "none" && args.biasDtype != "same" && args.biasDtype != "float32") {
        std::cerr << "Error: bias_dtype must be none, same, or float32.\n";
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

template <typename ElementType, typename BiasType, typename LayoutA, typename LayoutB>
__global__ __aicore__ void fused_mat_mul_sigmoid_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                        int64_t m, int64_t n, int64_t k, int64_t mL1, int64_t nL1,
                                                        int64_t kL1, int64_t baseM, int64_t baseN, int64_t baseK)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using LayoutC = asc::te::nd_ext_layout_ptn;
    // FusedMatMul uses optype 10 for sigmoid.
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 10,
                                                                    Blaze::Gemm::KernelMmadWithEpilogue>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, float,
                                                    LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulWithSigmoid<ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    // Split the FP32 accumulator between two AIVs and reserve one UB buffer for it.
    Params params = {{m, n, k, 1},
                     {aGM, bGM, nullptr, biasGM, nullptr, nullptr, static_cast<uint64_t>(k), static_cast<uint64_t>(mL1),
                      static_cast<uint64_t>(nL1), static_cast<uint64_t>(kL1), static_cast<uint32_t>(baseM),
                      static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), TilingConfig::L1_BUFFER_NUM,
                      TilingConfig::L0C_DB, true, 1U},
                     {yGM},
                     {static_cast<uint32_t>(mL1), static_cast<uint32_t>(nL1), static_cast<uint32_t>(kL1),
                      static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN), static_cast<uint32_t>(baseK), 1, 1, 1,
                      1, 1, 1, 0, Blaze::Gemm::L2_CACHE_DEFAULT, static_cast<uint32_t>(m), 1, 1}};
    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.splitM = params.mmadParams.splitM != 0;

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
    const TilingConfig* cfg;
    aclrtStream stream;
};

template <typename ElementType, typename BiasType, bool TransA, bool TransB>
void LaunchKernel(const LaunchParams& p)
{
    using LayoutA = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    LAUNCH_KERNEL_IMPL();
}

} // namespace

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
    const size_t sizeA = static_cast<size_t>(args.m) * args.k * sizeof(half);
    const size_t sizeB = static_cast<size_t>(args.k) * args.n * sizeof(half);
    const size_t sizeY = static_cast<size_t>(args.m) * args.n * sizeof(half);
    const size_t biasTypeSize = args.biasDtype == "float32" ? sizeof(float) : sizeof(half);
    const size_t sizeBias = args.biasDtype == "none" ? 0 : static_cast<size_t>(args.n) * biasTypeSize;
    std::vector<uint8_t> hostA(sizeA);
    std::vector<uint8_t> hostB(sizeB);
    std::vector<uint8_t> hostY(sizeY);
    std::vector<uint8_t> hostBias(sizeBias);
    if (!ReadFile("./input/input_a.bin", hostA.data(), sizeA) ||
        !ReadFile("./input/input_b.bin", hostB.data(), sizeB) ||
        (sizeBias != 0 && !ReadFile("./input/bias.bin", hostBias.data(), sizeBias))) {
        return false;
    }

    aclrtStream stream = nullptr;
    ACLDeviceGuard guard(stream);
    const TilingConfig tilingCfg = GetTilingConfig();
    const int64_t tileNum = CeilDiv(args.m, tilingCfg.baseM) * CeilDiv(args.n, tilingCfg.baseN);
    const int64_t blockNum = std::min(tileNum, GetAicCoreNum());
    uint8_t* deviceA = CopyToDevice(hostA);
    uint8_t* deviceB = CopyToDevice(hostB);
    uint8_t* deviceBias = sizeBias == 0 ? nullptr : CopyToDevice(hostBias);
    uint8_t* deviceY = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), sizeY, ACL_MEM_MALLOC_HUGE_FIRST));

    std::cout << "FusedMatMul sigmoid: M=" << args.m << ", K=" << args.k << ", N=" << args.n << ", dtype=" << args.dtype
              << ", bias_dtype=" << args.biasDtype << ", blockNum=" << blockNum << std::endl;
    const LaunchParams params = {deviceA, deviceB, deviceY,  deviceBias, args.m,
                                 args.n,  args.k,  blockNum, &tilingCfg, stream};
    if (args.dtype == "bfloat16") {
        if (args.biasDtype == "float32") {
            DISPATCH(bfloat16_t, float, args.transA, args.transB, params);
        } else {
            DISPATCH(bfloat16_t, bfloat16_t, args.transA, args.transB, params);
        }
    } else {
        if (args.biasDtype == "float32") {
            DISPATCH(half, float, args.transA, args.transB, params);
        } else {
            DISPATCH(half, half, args.transA, args.transB, params);
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostY.data(), sizeY, deviceY, sizeY, ACL_MEMCPY_DEVICE_TO_HOST));
    const bool success = WriteFile("./output/npu_out.bin", hostY.data(), sizeY);
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceY));
    if (deviceBias != nullptr) {
        ACL_CHECK(aclrtFree(deviceBias));
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
