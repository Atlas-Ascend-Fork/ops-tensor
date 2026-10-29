/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file confidence_head_fused_mix.cpp
 * \brief ConfidenceHeadFused GEMV example with CSV-driven FP32 verification.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "data_utils.h"
#include "blaze/gemm/block/block_mmad_confidence_head_fused.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_confidence_head_fused_mix.h"
#include "blaze/epilogue/block/block_epilogue_confidence_head_fused.h"

namespace {
constexpr uint32_t BASE_N = 128;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t K_L1 = 256;
constexpr uint32_t L1_STAGES = 4;
constexpr float CLAMP_LIMIT = 2147483648.0F;

struct Args {
    int64_t n{0};
    int64_t kEmbed{0};
    int64_t kMarkov{0};
    int64_t batch{1};
    std::string dtype;
    bool hasMarkov{false};
    bool hasTemperature{false};
    bool hasBias{false};
    bool hasLow{false};
    bool hasHigh{false};
    float low{-CLAMP_LIMIT};
    float high{CLAMP_LIMIT};
};

int64_t ParseDimension(const char* value)
{
    size_t consumed = 0;
    const std::string text(value);
    const int64_t result = std::stoll(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Dimensions must be integers.");
    }
    return result;
}

bool Parse(int argc, const char** argv, Args& args)
{
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " n k_embed k_markov batch dtype has_markov temperature bias clamp_low clamp_high\n";
        return false;
    }
    args.n = ParseDimension(argv[1]);
    args.kEmbed = ParseDimension(argv[2]);
    args.kMarkov = ParseDimension(argv[3]);
    args.batch = ParseDimension(argv[4]);
    args.dtype = argv[5];
    const std::string markov(argv[6]);
    const std::string bias(argv[8]);
    if ((markov != "true" && markov != "false") || (bias != "true" && bias != "false")) {
        return false;
    }
    args.hasMarkov = markov == "true";
    args.hasTemperature = std::string(argv[7]) != "none";
    args.hasBias = bias == "true";
    args.hasLow = std::string(argv[9]) != "none";
    args.hasHigh = std::string(argv[10]) != "none";
    if (args.hasLow) {
        args.low = std::stof(argv[9]);
        args.hasLow = args.low != -CLAMP_LIMIT;
    }
    if (args.hasHigh) {
        args.high = std::stof(argv[10]);
        args.hasHigh = args.high != CLAMP_LIMIT;
    }
    const int64_t limit = std::numeric_limits<uint32_t>::max();
    for (int64_t dim : {args.n, args.kEmbed, args.kMarkov, args.batch}) {
        if (dim < 0 || dim > limit) {
            return false;
        }
    }
    if ((args.dtype != "float16" && args.dtype != "bfloat16") || (!args.hasMarkov && args.kMarkov != 0) ||
        args.kEmbed + args.kMarkov > limit || (args.n != 0 && args.batch > limit / args.n)) {
        return false;
    }
    const uint64_t rows = static_cast<uint64_t>(args.n) * args.batch;
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(half);
    return (args.kEmbed == 0 || rows <= maxElements / args.kEmbed) &&
           (args.kMarkov == 0 || rows <= maxElements / args.kMarkov);
}

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        if (addr_ != nullptr) {
            aclrtFree(addr_);
        }
    }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void Allocate(size_t bytes)
    {
        if (bytes != 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&addr_), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }
    bool Load(const char* path, size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }
        std::vector<uint8_t> host(bytes);
        if (!ReadFile(path, host.data(), bytes)) {
            return false;
        }
        Allocate(bytes);
        ACL_CHECK(aclrtMemcpy(addr_, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        return true;
    }
    uint8_t* Get() const { return addr_; }

private:
    uint8_t* addr_{nullptr};
};
} // namespace

template <bool HasMarkov, typename ElementType>
__global__ __aicore__ void confidence_head_fused_mix_kernel(GM_ADDR hidden, GM_ADDR weight, GM_ADDR markov,
                                                            GM_ADDR temperature, GM_ADDR bias, GM_ADDR output,
                                                            int64_t rows, int64_t nPerBatch, int64_t kEmbed,
                                                            int64_t kMarkov, float low, float high, bool hasLow,
                                                            bool hasHigh)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ND = asc::te::nd_ext_layout_ptn;
    using DN = asc::te::dn_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::ConfidenceHeadFusedPolicy<Blaze::Gemm::A_FULL_LOAD_MODE, false, HasMarkov>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, ND, ElementType, DN, float, ND, float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<Shape, Blaze::Gemm::A_FULL_LOAD_MODE>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueConfidenceHeadFused<float, float>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;
    typename Kernel::Params params{};
    params.problemShape = {1L, rows, kEmbed, HasMarkov ? kMarkov : 0L};
    params.weightGmAddr = weight;
    params.hiddenGmAddr = hidden;
    params.markovGmAddr = markov;
    params.mmadParams.kEmbed = static_cast<uint32_t>(kEmbed);
    params.mmadParams.kMarkov = HasMarkov ? static_cast<uint32_t>(kMarkov) : 0U;
    params.mmadParams.baseN = BASE_N;
    params.mmadParams.baseK = BASE_K;
    params.mmadParams.kL1 = K_L1;
    params.mmadParams.l1Stages = L1_STAGES;
    params.mmadParams.l0cStages = 2;
    params.mmadParams.biasGmAddr = bias;
    params.epilogueParams.outputGmAddr = output;
    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.baseN = BASE_N;
    params.epilogueParams.temperatureGmAddr = temperature;
    params.epilogueParams.biasGmAddr = bias;
    params.epilogueParams.hasBias = bias != nullptr;
    params.epilogueParams.biasN = static_cast<uint32_t>(nPerBatch);
    params.epilogueParams.lowLimit = low;
    params.epilogueParams.highLimit = high;
    params.epilogueParams.hasLowLimit = hasLow;
    params.epilogueParams.hasHighLimit = hasHigh;
    params.schParams.mL1 = 1;
    params.schParams.nL1 = BASE_N;
    params.schParams.kL1 = K_L1;
    params.schParams.baseM = 1;
    params.schParams.baseN = BASE_N;
    params.schParams.baseK = BASE_K;
    params.schParams.mTailCnt = 1;
    params.schParams.nTailCnt = 1;
    Kernel kernel;
    kernel(params);
}

namespace {
bool Run(const Args& args)
{
    const uint64_t rows = static_cast<uint64_t>(args.n) * args.batch;
    if (rows == 0) {
        const char empty = 0;
        return WriteFile("./output/npu_out.bin", &empty, 0);
    }
    aclrtStream stream = nullptr;
    ACLDeviceGuard guard(stream);
    PlatformInfo platform;
    if (!InitPlatformInfo(platform) || platform.aicNum == 0 || platform.aivNum != 2 * platform.aicNum ||
        (platform.socVersion != platform_ascendc::SocVersion::ASCEND950 &&
         platform.socVersion != platform_ascendc::SocVersion::ASCEND350)) {
        std::cerr << "The example requires arch35 and one AIC paired with two AIVs.\n";
        return false;
    }
    const uint64_t paddedK = CeilDiv(static_cast<uint64_t>(args.kEmbed), uint64_t{16}) * 16 +
                             CeilDiv(static_cast<uint64_t>(args.kMarkov), uint64_t{16}) * 16;
    const uint64_t l1Bytes = paddedK * sizeof(half) + L1_STAGES * (uint64_t{K_L1} * BASE_N * sizeof(half) +
                                                                   (args.hasBias ? BASE_N * sizeof(float) : 0));
    if (platform.ubSize < 2 * 3 * BASE_N * sizeof(float) ||
        (paddedK != 0 &&
         (l1Bytes >= platform.l1Size || platform.l0aSize < 2 * 16 * BASE_K * sizeof(half) ||
          platform.l0bSize < 2 * BASE_K * BASE_N * sizeof(half) || platform.l0cSize < 2 * 16 * BASE_N * sizeof(float) ||
          (args.hasBias && platform.btSize < 2 * BASE_N * sizeof(float))))) {
        std::cerr << "The example tile exceeds on-chip memory capacity.\n";
        return false;
    }
    DeviceBuffer hidden, weight, markov, temperature, bias, output;
    if (!hidden.Load("./input/hidden.bin", rows * args.kEmbed * sizeof(half)) ||
        !weight.Load("./input/weight.bin", (args.kEmbed + args.kMarkov) * sizeof(half)) ||
        !markov.Load("./input/markov.bin", args.hasMarkov ? rows * args.kMarkov * sizeof(half) : 0) ||
        !temperature.Load("./input/temperature.bin", args.hasTemperature ? sizeof(float) : 0) ||
        !bias.Load("./input/bias.bin", args.hasBias ? args.n * sizeof(float) : 0)) {
        return false;
    }
    const size_t outputBytes = rows * sizeof(float);
    output.Allocate(outputBytes);
    const uint64_t tiles = args.hasBias ? args.batch * CeilDiv(static_cast<uint64_t>(args.n), uint64_t{BASE_N}) :
                                          CeilDiv(rows, uint64_t{BASE_N});
    const uint32_t cores = static_cast<uint32_t>(std::min<uint64_t>(tiles, platform.aicNum));
#define LAUNCH(HAS_MARKOV, TYPE)                                                               \
    confidence_head_fused_mix_kernel<HAS_MARKOV, TYPE><<<cores, 0, stream>>>(                  \
        hidden.Get(), weight.Get(), markov.Get(), temperature.Get(), bias.Get(), output.Get(), \
        static_cast<int64_t>(rows), args.n, args.kEmbed, args.kMarkov, args.low, args.high, args.hasLow, args.hasHigh)
    if (args.dtype == "bfloat16") {
        if (args.hasMarkov) {
            LAUNCH(true, bfloat16_t);
        } else {
            LAUNCH(false, bfloat16_t);
        }
    } else {
        if (args.hasMarkov) {
            LAUNCH(true, half);
        } else {
            LAUNCH(false, half);
        }
    }
#undef LAUNCH
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<float> hostOutput(rows);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), outputBytes, output.Get(), outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    return WriteFile("./output/npu_out.bin", hostOutput.data(), outputBytes);
}
} // namespace

int main(int argc, const char** argv)
{
    try {
        Args args;
        if (!Parse(argc, argv, args)) {
            std::cerr << "Invalid ConfidenceHeadFused example arguments.\n";
            return 1;
        }
        return Run(args) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
