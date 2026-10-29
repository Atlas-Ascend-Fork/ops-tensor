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
 * @file quant_batch_matmul_mix.cpp
 * @brief Executable example for the QBMM MIX batch and without-batch kernels.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

using AscendC::DT_BF16;
using AscendC::DT_FLOAT;
using AscendC::DT_FLOAT16;

#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"

namespace {

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using Layout = asc::te::nd_ext_layout_ptn;

constexpr uint32_t QUANT_MODE_PERCHANNEL = 2U;
constexpr uint32_t QUANT_MODE_PERTOKEN = 4U;
constexpr uint32_t GE_DTYPE_FLOAT = 0U;

struct CliArgs {
    bool withoutBatch{false};
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
};

bool ParseArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <batch|without_batch> <m> <k> <n>" << std::endl;
        return false;
    }
    const std::string variant = argv[1];
    if (variant != "batch" && variant != "without_batch") {
        std::cerr << "Variant must be batch or without_batch." << std::endl;
        return false;
    }
    args.withoutBatch = variant == "without_batch";
    args.m = std::atoll(argv[2]);
    args.k = std::atoll(argv[3]);
    args.n = std::atoll(argv[4]);
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "M, K, and N must be positive." << std::endl;
        return false;
    }
    return true;
}

template <typename Params>
__aicore__ inline void FillMixMmadParams(Params& params, GM_ADDR x1, GM_ADDR x2, int64_t m, int64_t n, int64_t k)
{
    params.aGmAddr = x1;
    params.bGmAddr = x2;
    params.problemShape = {m, n, k, 1};
    params.l0TileShape = {m, n, k, 0};
    params.kAL1 = static_cast<uint32_t>(k);
    params.kBL1 = static_cast<uint32_t>(k);
    params.l1BufferNum = 2U;
    params.enableL0CPingPong = false;
}

template <typename Params>
__aicore__ inline void FillSchedulerParams(Params& params, int64_t m, int64_t n)
{
    params = {m, n, 1, 1, 1, 1, 0, 0};
}

template <typename Params>
__aicore__ inline void FillDequantParams(Params& params, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, int64_t m,
                                         int64_t n)
{
    params.x2ScaleGmAddr = x2Scale;
    params.x1ScaleGmAddr = x1Scale;
    params.biasGmAddr = nullptr;
    params.outGmAddr = y;
    params.m = m;
    params.n = n;
    params.baseM = m;
    params.baseN = n;
    params.x1QuantMode = QUANT_MODE_PERTOKEN;
    params.x2QuantMode = QUANT_MODE_PERCHANNEL;
    params.isBias = false;
    params.biasDtype = GE_DTYPE_FLOAT;
}

__global__ __aicore__ void quant_batch_matmul_mix_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                         GM_ADDR y, int64_t m, int64_t k, int64_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, BTypeTuple, Layout, int32_t, Layout,
                                                    int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<half, int32_t, float, float, int32_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, Layout, Layout, int8_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    FillMixMmadParams(params.mmadParams, x1, x2, m, n, k);
    FillSchedulerParams(params.schParams, m, n);
    params.qbmmParams.batchA1 = 1U;
    params.qbmmParams.batchA2 = 1U;
    params.qbmmParams.batchA3 = 1U;
    params.qbmmParams.batchA4 = 1U;
    params.qbmmParams.batchB1 = 1U;
    params.qbmmParams.batchB2 = 1U;
    params.qbmmParams.batchB3 = 1U;
    params.qbmmParams.batchB4 = 1U;
    params.qbmmParams.batchC1 = 1U;
    params.qbmmParams.batchC2 = 1U;
    params.qbmmParams.batchC3 = 1U;
    params.qbmmParams.batchC4 = 1U;
    params.qbmmParams.x1QuantMode = QUANT_MODE_PERTOKEN;
    params.qbmmParams.x2QuantMode = QUANT_MODE_PERCHANNEL;
    params.qbmmParams.kAL1 = static_cast<uint32_t>(k);
    params.qbmmParams.kBL1 = static_cast<uint32_t>(k);
    params.qbmmParams.nBufferNum = 2U;
    params.qbmmParams.baseM = static_cast<uint32_t>(m);
    params.qbmmParams.baseN = static_cast<uint32_t>(n);
    params.qbmmParams.baseK = static_cast<uint32_t>(k);
    params.qbmmParams.dbL0C = 1U;
    FillDequantParams(params.epilogueParams, x1Scale, x2Scale, y, m, n);
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void quant_batch_matmul_mix_without_batch_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                                       GM_ADDR x2Scale, GM_ADDR y, int64_t m, int64_t k,
                                                                       int64_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                           Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, BTypeTuple, Layout, int32_t, Layout,
                                                    int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<half, int32_t, float, float, int32_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, Layout, Layout, int8_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    FillMixMmadParams(params.mmParams, x1, x2, m, n, k);
    FillSchedulerParams(params.schParams, m, n);
    FillDequantParams(params.epilogueParams, x1Scale, x2Scale, y, m, n);
    Kernel kernel;
    kernel(params);
}

struct DeviceBuffers {
    uint8_t* x1{nullptr};
    uint8_t* x2{nullptr};
    uint8_t* x1Scale{nullptr};
    uint8_t* x2Scale{nullptr};
    uint8_t* y{nullptr};
};

void FreeBuffers(DeviceBuffers& buffers)
{
    for (uint8_t* ptr : {buffers.x1, buffers.x2, buffers.x1Scale, buffers.x2Scale, buffers.y}) {
        if (ptr != nullptr) {
            ACL_CHECK(aclrtFree(ptr));
        }
    }
}

void AllocateAndCopy(uint8_t** device, const std::vector<uint8_t>& host)
{
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), host.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(*device, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
}

bool LoadBytes(const std::string& path, std::vector<uint8_t>& data) { return ReadFile(path, data.data(), data.size()); }

int Run(const CliArgs& args)
{
    const size_t x1Bytes = static_cast<size_t>(args.m * args.k);
    const size_t x2Bytes = static_cast<size_t>(args.k * args.n);
    const size_t x1ScaleBytes = static_cast<size_t>(args.m) * sizeof(float);
    const size_t x2ScaleBytes = static_cast<size_t>(args.n) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(args.m * args.n) * sizeof(half);

    std::vector<uint8_t> hostX1(x1Bytes);
    std::vector<uint8_t> hostX2(x2Bytes);
    std::vector<uint8_t> hostX1Scale(x1ScaleBytes);
    std::vector<uint8_t> hostX2Scale(x2ScaleBytes);
    std::vector<uint8_t> hostY(yBytes, 0U);
    if (!LoadBytes("./input/input_a.bin", hostX1) || !LoadBytes("./input/input_b.bin", hostX2) ||
        !LoadBytes("./input/scale_a.bin", hostX1Scale) || !LoadBytes("./input/scale_b.bin", hostX2Scale)) {
        return 1;
    }

    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);
    DeviceBuffers device;
    AllocateAndCopy(&device.x1, hostX1);
    AllocateAndCopy(&device.x2, hostX2);
    AllocateAndCopy(&device.x1Scale, hostX1Scale);
    AllocateAndCopy(&device.x2Scale, hostX2Scale);
    AllocateAndCopy(&device.y, hostY);

    const uint32_t launchBlocks = static_cast<uint32_t>(GetAicCoreNum());
    if (args.withoutBatch) {
        quant_batch_matmul_mix_without_batch_kernel<<<launchBlocks, 0, stream>>>(
            device.x1, device.x2, device.x1Scale, device.x2Scale, device.y, args.m, args.k, args.n);
    } else {
        quant_batch_matmul_mix_kernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x1Scale, device.x2Scale,
                                                                   device.y, args.m, args.k, args.n);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostY.data(), yBytes, device.y, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    const bool written = WriteFile("./output/npu_out.bin", hostY.data(), hostY.size());
    FreeBuffers(device);
    if (!written) {
        return 1;
    }
    std::cout << "PASS: launched QBMM MIX " << (args.withoutBatch ? "without_batch" : "batch") << " with shape ["
              << args.m << ", " << args.k << ", " << args.n << "]" << std::endl;
    return 0;
}

} // namespace

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        return 1;
    }
    return Run(args);
}
