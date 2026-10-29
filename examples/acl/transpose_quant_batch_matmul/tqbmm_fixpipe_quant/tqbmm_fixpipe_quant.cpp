/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file tqbmm_fixpipe_quant.cpp
 * @brief TQBMM fixpipe quant (HIFP8) example.
 *
 * Exercises Blaze::Gemm::Kernel::GemmUniversal (kernel_tqbmm.h)
 * with Blaze::Gemm::MatmulWithScaleFixpipeQuant dispatch policy and
 * Blaze::Gemm::Block::BlockMmad (block_mmad_a8w8_fixpipe_quant.h).
 *
 * HIFP8 scenario: A/B are hifloat8, x2Scale is uint64 per-channel.
 * AIC-only: fixpipe dequant on cube side.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

struct CliArgs {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t batch = 1;
    std::string cDtype;
    bool transB = false;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    int64_t kAL1 = 0;
    int64_t kBL1 = 0;
    int64_t l1Buffers = 2;
    int64_t dbL0C = 1;
};

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 14) {
        std::fprintf(stderr,
                     "Usage: %s <m> <k> <n> <batch> <c_dtype> <transB> <base_m> <base_n> <base_k>"
                     " <tile_k_al1> <tile_k_bl1> <l1_buffers> <db_l0c>\n",
                     argv[0]);
        return false;
    }
    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.batch = std::atoll(argv[4]);
    args.cDtype = argv[5];
    args.transB = std::string(argv[6]) == "true" || std::string(argv[6]) == "1";
    args.baseM = std::atoll(argv[7]);
    args.baseN = std::atoll(argv[8]);
    args.baseK = std::atoll(argv[9]);
    args.kAL1 = std::atoll(argv[10]);
    args.kBL1 = std::atoll(argv[11]);
    args.l1Buffers = std::atoll(argv[12]);
    args.dbL0C = std::atoll(argv[13]);
    return args.m > 0 && args.k > 0 && args.n > 0 && args.batch > 0;
}

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

constexpr uint64_t NON_CONTIGUOUS_TYPE_PERM_X1 = static_cast<uint64_t>(
    Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1);

template <class CType, class LayoutB>
__global__ __aicore__ void tqbmm_fixpipe_quant_example_kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleGm,
                                                              GM_ADDR perTokenScaleGm, GM_ADDR cGm, int64_t m,
                                                              int64_t k, int64_t n, int64_t batch, uint64_t baseM,
                                                              uint64_t baseN, uint64_t baseK, uint64_t kAL1,
                                                              uint64_t kBL1, uint64_t l1BufferNum, uint64_t dbL0C)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using AType = hifloat8_t;
    using BType = hifloat8_t;
    using BiasType = float;
    using X2ScaleType = uint64_t;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMMFixpipeQuant, __gm__ int32_t*, NON_CONTIGUOUS_TYPE_PERM_X1>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB,
                                                                                AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      BlockScheduler>;

    constexpr uint32_t x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::DEFAULT);
    constexpr uint32_t x2QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);

    typename Kernel::Params params{};
    params.problemShape = ProblemShape{m, n, k, batch};

    typename BlockMmad::Params mmadParams;
    mmadParams.aGmAddr = aGm;
    mmadParams.bGmAddr = bGm;
    mmadParams.cGmAddr = cGm;
    mmadParams.biasGmAddr = biasGm;
    mmadParams.scaleAGmAddr = perTokenScaleGm;
    mmadParams.scaleBGmAddr = scaleGm;
    params.mmadParams = mmadParams;

    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    params.tqbmmParams = {x1QuantMode,
                          x2QuantMode,
                          static_cast<uint32_t>(kAL1),
                          static_cast<uint32_t>(kBL1),
                          static_cast<uint32_t>(l1BufferNum),
                          static_cast<uint32_t>(baseM),
                          static_cast<uint32_t>(baseN),
                          static_cast<uint32_t>(baseK),
                          0U,
                          static_cast<uint32_t>(dbL0C),
                          1U};
    Kernel kernel;
    kernel(params);
}

struct LaunchParams {
    uint8_t* dA;
    uint8_t* dB;
    uint8_t* dScale;
    uint8_t* dC;
    int64_t m, n, k, batch;
    int64_t blockNum;
    uint64_t baseM, baseN, baseK, kAL1, kBL1, l1BufferNum, dbL0C;
    aclrtStream stream;
};

template <class CType, bool TransB>
void LaunchKernel(const LaunchParams& p)
{
    using LAYOUT_B = std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    tqbmm_fixpipe_quant_example_kernel<CType, LAYOUT_B>
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, nullptr, p.dScale, nullptr, p.dC, p.m, p.k, p.n, p.batch, p.baseM,
                                      p.baseN, p.baseK, p.kAL1, p.kBL1, p.l1BufferNum, p.dbL0C);
}

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);

    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0)
        return;

    const size_t aSize = static_cast<size_t>(args.m * args.batch * args.k);
    const size_t bSize = static_cast<size_t>(args.k * args.batch * args.n);
    const size_t scaleSize = static_cast<size_t>(args.n) * sizeof(uint64_t);
    const size_t cElementSize = (args.cDtype == "float16" || args.cDtype == "bfloat16") ? 2 : 0;
    const size_t cSize = static_cast<size_t>(args.m * args.batch * args.n) * cElementSize;

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    std::vector<uint8_t> hostA(aSize), hostB(bSize), hostC(cSize, 0), hostScale(scaleSize);
    ReadFile(inputDir + "/input_a.bin", hostA.data(), aSize);
    ReadFile(inputDir + "/input_b.bin", hostB.data(), bSize);
    ReadFile(inputDir + "/scale.bin", hostScale.data(), scaleSize);
    ReadFile(inputDir + "/initial_c.bin", hostC.data(), cSize);

    uint8_t *deviceA{nullptr}, *deviceB{nullptr}, *deviceC{nullptr}, *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScale), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, aSize, hostA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, bSize, hostB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, cSize, hostC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScale, scaleSize, hostScale.data(), scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchParams p = {deviceA,
                      deviceB,
                      deviceScale,
                      deviceC,
                      args.m,
                      args.n,
                      args.k,
                      args.batch,
                      blockNum,
                      static_cast<uint64_t>(args.baseM),
                      static_cast<uint64_t>(args.baseN),
                      static_cast<uint64_t>(args.baseK),
                      static_cast<uint64_t>(args.kAL1),
                      static_cast<uint64_t>(args.kBL1),
                      static_cast<uint64_t>(args.l1Buffers),
                      static_cast<uint64_t>(args.dbL0C),
                      stream};

    if (args.cDtype == "float16") {
        if (args.transB) {
            LaunchKernel<half, true>(p);
        } else {
            LaunchKernel<half, false>(p);
        }
    } else if (args.cDtype == "bfloat16") {
        if (args.transB) {
            LaunchKernel<bfloat16_t, true>(p);
        } else {
            LaunchKernel<bfloat16_t, false>(p);
        }
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostC.data(), cSize, deviceC, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(outputDir + "/npu_out.bin", hostC.data(), cSize);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceScale));
}

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseCliArgs(argc, argv, args))
        return 1;
    Run(args);
    return 0;
}
