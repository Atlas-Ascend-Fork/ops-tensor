/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flat_quant.cpp
 * \brief CSV-driven FlatQuant example: y[b] = P1 @ x[b] @ P2 followed by MX-FP4 (E2M1) quantization.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_operator.h"

using AscendC::GatherMaskParams;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;

#include "blaze/attention/block/block_mmad.h"
#include "blaze/attention/block/block_scheduler_flat_quant.h"
#include "blaze/attention/kernel/kernel_universal.h"
#include "blaze/attention/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_flat_quant.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"

namespace {

constexpr uint32_t BLOCK_NUM = 8U;
constexpr int64_t BASE_K = 64;
constexpr int64_t MAX_MN_PER_TILE = 4096;
constexpr int64_t MAX_L0C_ELEMENTS = 32768;
constexpr float FP4_E2M1_MAX = 6.0f;

template <typename T_>
struct MatmulType {
    using T = T_;
};

#pragma pack(push, 1)
struct FlatQuantTilingData {
    int64_t K{0};
    int64_t M{0};
    int64_t N{0};
    int64_t iterBatch{1};
    uint8_t hasP2{1};
    float dstTypeMax{FP4_E2M1_MAX};
    float invDstTypeMax{1.0f / FP4_E2M1_MAX};
};
#pragma pack(pop)

__global__ __aicore__ void FlatQuantKernel(GM_ADDR xGm, GM_ADDR p1Gm, GM_ADDR p2Gm, GM_ADDR yGm, GM_ADDR scaleGm,
                                           GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const auto& tiling = *reinterpret_cast<const __gm__ FlatQuantTilingData*>(tilingGm);

    using XType = bfloat16_t;
    using YType = fp4x2_e2m1_t;
    using ScaleType = float;
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Attention::Block::BlockSchedulerFlatQuant<ProblemShape>;
    using DispatchPolicy = Blaze::Attention::BlockFlatQuant<Blaze::Attention::KernelFlatQuant>;
    using BlockMmad = Blaze::Attention::Block::BlockMmad<DispatchPolicy, MatmulType<XType>, Layout, MatmulType<XType>,
                                                         Layout, MatmulType<ScaleType>, Layout, MatmulType<YType>,
                                                         Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<XType, YType, ScaleType>;
    using Kernel = Blaze::Attention::Kernel::AttentionUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    const int64_t m = tiling.M;
    const int64_t n = tiling.N;
    const int64_t k = tiling.K;
    const int64_t iterBatch = tiling.iterBatch;

    typename Kernel::Params params{};
    params.problemShape = TupleShape{m, n, n, k};
    params.mmadParams.aGmAddr = xGm;
    params.mmadParams.bGmAddr = p1Gm;
    params.mmadParams.cGmAddr = p2Gm;
    params.mmadParams.problemShape = TupleShape{m, n, n, k};
    params.mmadParams.tileL1 = TupleShape{m * iterBatch, n, n, iterBatch};
    params.mmadParams.tileL0 = TupleShape{m * iterBatch, n, BASE_K, 1};
    params.mmadParams.hasP2 = tiling.hasP2 == 1;
    params.epilogueParams.outGmAddr = yGm;
    params.epilogueParams.scaleGmAddr = scaleGm;
    params.epilogueParams.problemShape = TupleShape{m, n, n, k};
    params.epilogueParams.dstTypeMax = tiling.dstTypeMax;
    params.epilogueParams.invDstTypeMax = tiling.invDstTypeMax;
    params.schParams.iterBatch = iterBatch;
    params.schParams.dstTypeMax = tiling.dstTypeMax;
    params.schParams.invDstTypeMax = tiling.invDstTypeMax;

    Kernel kernel;
    kernel(params);
}

struct ExampleConfig {
    int64_t batch;
    int64_t m;
    int64_t n;
    int64_t iterBatch;
    bool hasP2;
    std::string dataDir;
};

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool ParseBool(const char* text, bool& value)
{
    const std::string normalized = ToLower(text);
    if (normalized == "true" || normalized == "1") {
        value = true;
        return true;
    }
    if (normalized == "false" || normalized == "0") {
        value = false;
        return true;
    }
    return false;
}

bool ParseArgs(int argc, const char** argv, ExampleConfig& config)
{
    if (argc != 7) {
        std::fprintf(stderr, "Usage: %s <K> <M> <N> <iterBatch> <hasP2> <data_dir>\n", argv[0]);
        return false;
    }

    config.batch = std::atoll(argv[1]);
    config.m = std::atoll(argv[2]);
    config.n = std::atoll(argv[3]);
    config.iterBatch = std::atoll(argv[4]);
    config.dataDir = argv[6];
    if (!ParseBool(argv[5], config.hasP2)) {
        std::fprintf(stderr, "hasP2 must be true, false, 1, or 0\n");
        return false;
    }

    const bool validShape = config.m > 0 && config.n > 0 && config.m % 16 == 0 && config.n % 16 == 0 &&
                            (config.m * config.n) % 64 == 0 && config.m * config.n <= MAX_MN_PER_TILE;
    const bool validBatch = config.batch > 0 && config.iterBatch > 0 && config.iterBatch <= config.batch &&
                            config.iterBatch * config.m * config.n <= MAX_L0C_ELEMENTS;
    if (!validShape || !validBatch) {
        std::fprintf(stderr, "Invalid shape/batch: M and N must be multiples of 16, M*N a multiple of 64 and <= 4096,"
                             " K >= iterBatch >= 1, and iterBatch*M*N <= 32768\n");
        return false;
    }
    return true;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(size)
    {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&data_), size_, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            static_cast<void>(aclrtFree(data_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    uint8_t* Get() const { return data_; }

    void CopyFromHost(const void* host, size_t size) const
    {
        if (size != size_) {
            std::fprintf(stderr, "Host/device copy size mismatch: %zu vs %zu\n", size, size_);
            std::exit(1);
        }
        ACL_CHECK(aclrtMemcpy(data_, size_, host, size_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyFromFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        if (!ReadFile(path, host.data(), size_)) {
            std::exit(1);
        }
        CopyFromHost(host.data(), size_);
    }

    void CopyToFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        ACL_CHECK(aclrtMemcpy(host.data(), size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), size_)) {
            std::fprintf(stderr, "Failed to write %s\n", path.c_str());
            std::exit(1);
        }
    }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

void RunCase(const ExampleConfig& config, aclrtStream stream)
{
    const int64_t mn = config.m * config.n;
    const size_t xSize = static_cast<size_t>(config.batch * mn) * sizeof(uint16_t);
    const size_t p1Size = static_cast<size_t>(config.m * config.m) * sizeof(uint16_t);
    const size_t p2Size = static_cast<size_t>(config.n * config.n) * sizeof(uint16_t);
    const size_t ySize = static_cast<size_t>(config.batch * mn) / 2U;
    const size_t scaleSize = static_cast<size_t>(config.batch) * ((static_cast<size_t>(mn) + 63U) / 64U) * 2U;

    DeviceBuffer x(xSize);
    DeviceBuffer p1(p1Size);
    DeviceBuffer p2(p2Size);
    DeviceBuffer y(ySize);
    DeviceBuffer scale(scaleSize);
    DeviceBuffer tiling(sizeof(FlatQuantTilingData));

    x.CopyFromFile(config.dataDir + "/input_x.bin");
    p1.CopyFromFile(config.dataDir + "/input_p1.bin");
    p2.CopyFromFile(config.dataDir + "/input_p2.bin");

    FlatQuantTilingData tilingData{};
    tilingData.K = config.batch;
    tilingData.M = config.m;
    tilingData.N = config.n;
    tilingData.iterBatch = config.iterBatch;
    tilingData.hasP2 = config.hasP2 ? 1 : 0;
    tilingData.dstTypeMax = FP4_E2M1_MAX;
    tilingData.invDstTypeMax = 1.0f / FP4_E2M1_MAX;
    tiling.CopyFromHost(&tilingData, sizeof(tilingData));

    ACL_CHECK(aclrtMemset(y.Get(), ySize, 0, ySize));
    ACL_CHECK(aclrtMemset(scale.Get(), scaleSize, 0, scaleSize));

    FlatQuantKernel<<<BLOCK_NUM, 0, stream>>>(x.Get(), p1.Get(), p2.Get(), y.Get(), scale.Get(), tiling.Get());
    ACL_CHECK(aclrtSynchronizeStream(stream));
    y.CopyToFile(config.dataDir + "/npu_y.bin");
    scale.CopyToFile(config.dataDir + "/npu_scale.bin");
}

} // namespace

int main(int argc, const char** argv)
{
    ExampleConfig config{};
    if (!ParseArgs(argc, argv, config)) {
        return 1;
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));
    RunCase(config, stream);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
