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
 * @file transpose_quant_batch_matmul_mx.cpp
 * @brief TQBMM MX Transpose Quant Batch MatMul example.
 *
 * Exercises Blaze::Gemm::Kernel::GemmUniversal (kernel_tqbmm.h)
 * with Blaze::Gemm::MatmulWithScaleMx dispatch policy and
 * Blaze::Gemm::Block::BlockMmad (block_mmad_qbmm_mx.h).
 *
 * This example handles the transpose scenario:
 *   x1: perm [1,0,2], effective shape [m, batch, k], layout nd_ext_layout_ptn
 *   x2: [batch, k, n] or [batch, n, k]
 *   x1Scale is expected to be pre-transposed to [batch, m, G, 2] layout.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#undef __fp4e2m1x2
#undef __fp8e4m3

static constexpr uint64_t GROUP_SIZE = 32UL;
static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;

struct CliArgs {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t batch = 1;
    std::string aDtype;
    std::string bDtype;
    std::string cDtype;
    bool transA = false;
    bool transB = false;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    int64_t kL1 = 0;
    int64_t scaleKL1 = 0;
    int64_t l1Buffers = 2;
    int64_t dbL0C = 1;
    bool aFullLoad = false;
    std::string format = "(ND,ND)";
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 19) {
        std::fprintf(stderr,
                     "Usage: %s <m> <k> <n> <batch> <a_dtype> <b_dtype> <c_dtype>"
                     " <transA> <transB> <format> <base_m> <base_n> <base_k> <tile_k_l1> <scale_k_l1>"
                     " <l1_buffers> <db_l0c> <a_full_load>\n",
                     argv[0]);
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);
    args.aDtype = argv[5];
    args.bDtype = argv[6];
    args.cDtype = argv[7];
    args.transA = ParseBool(argv[8]);
    args.transB = ParseBool(argv[9]);
    args.format = argv[10];

    args.baseM = std::atoll(argv[11]);
    args.baseN = std::atoll(argv[12]);
    args.baseK = std::atoll(argv[13]);
    args.kL1 = std::atoll(argv[14]);
    args.scaleKL1 = std::atoll(argv[15]);
    args.l1Buffers = std::atoll(argv[16]);
    args.dbL0C = std::atoll(argv[17]);
    args.aFullLoad = ParseBool(argv[18]);

    if (args.m <= 0 || args.k <= 0 || args.n <= 0 || args.batch <= 0) {
        std::fprintf(stderr, "Error: M, K, N, batch must be positive integers.\n");
        return false;
    }
    if (args.aDtype != "fp8_e4m3" && args.aDtype != "fp4_e2m1") {
        std::fprintf(stderr, "Error: A dtype must be fp8_e4m3 or fp4_e2m1.\n");
        return false;
    }
    if (args.bDtype != "fp8_e4m3" && args.bDtype != "fp4_e2m1") {
        std::fprintf(stderr, "Error: B dtype must be fp8_e4m3 or fp4_e2m1.\n");
        return false;
    }
    if (args.cDtype != "float16" && args.cDtype != "bfloat16") {
        std::fprintf(stderr, "Error: C dtype must be float16 or bfloat16.\n");
        return false;
    }
    return true;
}

static bool IsFp4Dtype(const std::string& dtype) { return dtype == "fp4_e2m1"; }

static size_t DtypeSize(const std::string& dtype)
{
    if (dtype == "fp4_e2m1" || dtype == "fp8_e4m3")
        return 1;
    if (dtype == "float16" || dtype == "bfloat16")
        return 2;
    return 0;
}

static size_t ElementCountToBytes(int64_t count, const std::string& dtype)
{
    if (IsFp4Dtype(dtype))
        return static_cast<size_t>((count + 1) / 2);
    return static_cast<size_t>(count) * DtypeSize(dtype);
}

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

template <class AType, class BType, class CType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
__global__ __aicore__ void tqbmm_mx_example_kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm,
                                                   GM_ADDR scaleBGm, GM_ADDR cGm, int64_t m, int64_t k, int64_t n,
                                                   int64_t batch, uint64_t baseM, uint64_t baseN, uint64_t baseK,
                                                   uint64_t kL1, uint64_t scaleKL1, uint64_t l1BufferNum,
                                                   uint64_t dbL0C, uint64_t biasElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BiasType = float;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, Blaze::Gemm::KernelMmadMultiBlockTQBMM>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      BlockScheduler>;
    typename Kernel::Params params{};
    params.problemShape = ProblemShape{m, n, k, batch};
    params.mmadParams = {aGm, bGm, cGm, biasGm, scaleAGm, scaleBGm};
    params.l1Params = {kL1, scaleKL1, l1BufferNum};
    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    params.tqbmmParams = {1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          0,
                          static_cast<uint32_t>(baseM),
                          static_cast<uint32_t>(baseN),
                          static_cast<uint32_t>(baseK),
                          0U,
                          static_cast<uint32_t>(dbL0C)};
    Kernel kernel;
    kernel(params);
}

#define LAUNCH_TQBMM_KERNEL(FULL_LOAD_MODE)                                                                    \
    tqbmm_mx_example_kernel<A_TYPE, B_TYPE, C_TYPE, LAYOUT_A, LAYOUT_B, FULL_LOAD_MODE>                        \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, nullptr, p.dScaleA, p.dScaleB, p.dC, p.m, p.k, p.n, p.batch, \
                                      p.baseM, p.baseN, p.baseK, p.kL1, p.scaleKL1, p.l1BufferNum, p.dbL0C, 0)

#define DISPATCH_TQBMM(A_TYPE, B_TYPE, C_TYPE, TRANS_B, FULL_LOAD_MODE)                \
    do {                                                                               \
        if (TRANS_B) {                                                                 \
            LaunchKernel<A_TYPE, B_TYPE, C_TYPE, true, FULL_LOAD_MODE>(launchParams);  \
        } else {                                                                       \
            LaunchKernel<A_TYPE, B_TYPE, C_TYPE, false, FULL_LOAD_MODE>(launchParams); \
        }                                                                              \
    } while (0)

struct LaunchParams {
    uint8_t* dA;
    uint8_t* dB;
    uint8_t* dScaleA;
    uint8_t* dScaleB;
    uint8_t* dC;
    int64_t m, n, k, batch;
    int64_t blockNum;
    uint64_t baseM, baseN, baseK, kL1, scaleKL1, l1BufferNum, dbL0C;
    aclrtStream stream;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, bool TransB, uint64_t FullLoadMode>
void LaunchKernel(const LaunchParams& p)
{
    using LAYOUT_A = asc::te::nd_ext_layout_ptn;
    using LAYOUT_B = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    LAUNCH_TQBMM_KERNEL(FullLoadMode);
}

template <class A_TYPE, class B_TYPE, uint64_t FullLoadMode>
void LaunchByC(const CliArgs& args, const LaunchParams& launchParams)
{
    if (args.cDtype == "float16") {
        DISPATCH_TQBMM(A_TYPE, B_TYPE, half, args.transB, FullLoadMode);
    } else if (args.cDtype == "bfloat16") {
        DISPATCH_TQBMM(A_TYPE, B_TYPE, bfloat16_t, args.transB, FullLoadMode);
    }
}

template <class A_TYPE, class B_TYPE>
void LaunchByFullLoad(const CliArgs& args, const LaunchParams& launchParams)
{
    if (args.aFullLoad) {
        LaunchByC<A_TYPE, B_TYPE, Blaze::Gemm::A_FULL_LOAD_MODE>(args, launchParams);
    } else {
        LaunchByC<A_TYPE, B_TYPE, Blaze::Gemm::NONE_FULL_LOAD_MODE>(args, launchParams);
    }
}

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);

    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0)
        return;

    const uint64_t scaleK = static_cast<uint64_t>(CeilDiv(args.k, static_cast<int64_t>(MXFP_DIVISOR_SIZE))) *
                            (MXFP_DIVISOR_SIZE / GROUP_SIZE);
    const size_t aSize = ElementCountToBytes(args.m * args.batch * args.k, args.aDtype);
    const size_t bSize = ElementCountToBytes(args.k * args.batch * args.n, args.bDtype);
    const size_t scaleASize = static_cast<size_t>(args.m * args.batch) * scaleK;
    const size_t scaleBSize = static_cast<size_t>(args.n * args.batch) * scaleK;
    const size_t cSize = static_cast<size_t>(args.m * args.batch * args.n) * DtypeSize(args.cDtype);

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    std::vector<uint8_t> hostA(aSize), hostB(bSize), hostC(cSize, 0);
    std::vector<uint8_t> hostScaleA(scaleASize), hostScaleB(scaleBSize);

    ReadFile(inputDir + "/input_a.bin", hostA.data(), aSize);
    ReadFile(inputDir + "/input_b.bin", hostB.data(), bSize);
    ReadFile(inputDir + "/scale_a.bin", hostScaleA.data(), scaleASize);
    ReadFile(inputDir + "/scale_b.bin", hostScaleB.data(), scaleBSize);
    ReadFile(inputDir + "/initial_c.bin", hostC.data(), cSize);

    uint8_t *deviceA{nullptr}, *deviceB{nullptr}, *deviceC{nullptr};
    uint8_t *deviceScaleA{nullptr}, *deviceScaleB{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleA), scaleASize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleB), scaleBSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, aSize, hostA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, bSize, hostB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, cSize, hostC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, scaleASize, hostScaleA.data(), scaleASize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, scaleBSize, hostScaleB.data(), scaleBSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchParams launchParams = {deviceA,
                                 deviceB,
                                 deviceScaleA,
                                 deviceScaleB,
                                 deviceC,
                                 args.m,
                                 args.n,
                                 args.k,
                                 args.batch,
                                 blockNum,
                                 static_cast<uint64_t>(args.baseM),
                                 static_cast<uint64_t>(args.baseN),
                                 static_cast<uint64_t>(args.baseK),
                                 static_cast<uint64_t>(args.kL1),
                                 static_cast<uint64_t>(args.scaleKL1),
                                 static_cast<uint64_t>(args.l1Buffers),
                                 static_cast<uint64_t>(args.dbL0C),
                                 stream};

    if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3") {
        LaunchByFullLoad<fp8_e4m3fn_t, fp8_e4m3fn_t>(args, launchParams);
    } else if (args.aDtype == "fp4_e2m1" && args.bDtype == "fp4_e2m1") {
        LaunchByFullLoad<fp4x2_e2m1_t, fp4x2_e2m1_t>(args, launchParams);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostC.data(), cSize, deviceC, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(outputDir + "/npu_out.bin", hostC.data(), cSize);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceScaleA));
    ACL_CHECK(aclrtFree(deviceScaleB));
}

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseCliArgs(argc, argv, args))
        return 1;
    Run(args);
    return 0;
}
