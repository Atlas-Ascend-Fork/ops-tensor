/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_quant.cpp
 * \brief Ordinary and M-grouped RotateQuant with FP16/BF16 input and MX-FP4 output.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_operator.h"

// The ASC host pass only needs the kernel entry; parse Blaze with device BF16 types.
#if defined(__NPU_ARCH__)
#include "blaze/gemm/block/block_mmad_rotate_quant.h"
#include "blaze/gemm/block/block_scheduler_rotate_quant.h"
#include "blaze/gemm/kernel/kernel_rotate_quant.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#endif

namespace {

constexpr uint32_t MAX_BLOCK_NUM = 8;
constexpr int64_t BLOCK_M = 16;
constexpr int64_t CUBE_ALIGNMENT = 16;
constexpr int64_t MX_GROUP_SIZE = 32;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr int64_t ROUND_MODE_RINT = 4; // BlockEpilogueRotateQuant's internal round-mode encoding.
constexpr int64_t SCALE_ALG_DYN = 2;
constexpr std::array<int64_t, 4> SUPPORTED_K_SIZES{32, 64, 128, 256};
constexpr uint32_t OUTPUT_SENTINEL = 0xa5;
constexpr uint64_t REQUIRED_UB_BYTES = 64 * 1024;
constexpr float FP4_E2M1_MAX = 6.0f;

template <typename ElementType_>
struct MatmulType {
    using T = ElementType_;
};

struct RotateQuantTilingData {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t b{1};
    int64_t baseK{0};
    int64_t groupNum{0};
    int64_t groupListType{0};
    int64_t hasAlpha{0};
};

template <typename XType_, bool EnableGrouped_>
__global__ __aicore__ void RotateQuantKernel(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm, GM_ADDR groupListGm,
                                             GM_ADDR yGm, GM_ADDR scaleGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__)
    AscendC::InitSocState();
    const auto& tiling = *reinterpret_cast<const __gm__ RotateQuantTilingData*>(tilingGm);

    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerRotateQuant<ProblemShape, EnableGrouped_>;
    using DispatchPolicy = Blaze::Gemm::BlockRotateQuant<Blaze::Gemm::KernelRotateQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, MatmulType<XType_>, Layout, MatmulType<XType_>,
                                                    Layout, MatmulType<bfloat16_t>, Layout, MatmulType<int8_t>, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<bfloat16_t, fp4x2_e2m1_t, int8_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    const int64_t m = tiling.m;
    const int64_t n = tiling.n;
    const int64_t k = tiling.k;
    const int64_t b = tiling.b;
    // Tuple constructors bind local references; load GM fields into local scalars first.
    const int64_t baseK = tiling.baseK;
    const int64_t groupNum = tiling.groupNum;
    const int64_t groupListType = tiling.groupListType;
    const int64_t hasAlpha = tiling.hasAlpha;
    const int64_t remainder = (m * (n / k)) % BLOCK_M;
    const int64_t tailM = remainder == 0 ? BLOCK_M : remainder;
    typename Kernel::Params params = {
        {m, n, k, b},
        {xGm, rotationGm, alphaGm, {m, n, k, b}, {BLOCK_M, k, k, 1}, {BLOCK_M, k, baseK, 1}, b},
        {yGm, scaleGm, {m, n, k, b}, {}, false, -1, ROUND_MODE_RINT, SCALE_ALG_DYN, FP4_E2M1_MAX, 1.0f / FP4_E2M1_MAX},
        {BLOCK_M, k, k, BLOCK_M, k, baseK, n / k, tailM, FP4_E2M1_MAX, 1.0f / FP4_E2M1_MAX, -1, ROUND_MODE_RINT,
         SCALE_ALG_DYN, hasAlpha, 0},
        {groupListGm, groupNum, groupListType}};
    Kernel kernel;
    kernel(params);
#endif
}

struct ExampleConfig {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t b{1};
    int64_t groupListType{0};
    bool hasAlpha{false};
    std::vector<int64_t> groupList;
    std::string dtype;
    std::string dataDir;
};

int64_t ParseInteger(const std::string& text)
{
    size_t consumed = 0;
    int64_t value = std::stoll(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Invalid integer: " + text);
    }
    return value;
}

void ParseGroups(const std::string& text, ExampleConfig& config)
{
    if (text == "none") {
        return;
    }
    size_t start = 0;
    int64_t rowEnd = 0;
    do {
        size_t separator = text.find(';', start);
        std::string item = text.substr(start, separator == std::string::npos ? separator : separator - start);
        int64_t value = ParseInteger(item);
        if (value < 0 || (config.groupListType == 0 && (value < rowEnd || value > config.m)) ||
            (config.groupListType == 1 && value > config.m - rowEnd)) {
            throw std::invalid_argument("groupList must describe a nonnegative prefix of M");
        }
        rowEnd = config.groupListType == 0 ? value : rowEnd + value;
        config.groupList.push_back(value);
        if (config.groupList.size() > static_cast<size_t>(MAX_GROUP_NUM)) {
            throw std::invalid_argument("groupList may contain at most 1024 groups");
        }
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    } while (true);
}

bool ParseArgs(int argc, const char** argv, ExampleConfig& config)
{
    if (argc != 10) {
        std::fprintf(stderr, "Usage: %s <M> <N> <K> <B> <dtype> <groupListType> <groupList> <alpha> <data_dir>\n",
                     argv[0]);
        return false;
    }
    try {
        config.m = ParseInteger(argv[1]);
        config.n = ParseInteger(argv[2]);
        config.k = ParseInteger(argv[3]);
        config.b = ParseInteger(argv[4]);
        config.dtype = argv[5];
        config.groupListType = ParseInteger(argv[6]);
        config.dataDir = argv[9];
        const int64_t maxElements = std::numeric_limits<int32_t>::max();
        if (config.m <= 0 || config.n <= 0 || config.n > maxElements || config.m > maxElements / config.n ||
            std::find(SUPPORTED_K_SIZES.begin(), SUPPORTED_K_SIZES.end(), config.k) == SUPPORTED_K_SIZES.end()) {
            throw std::invalid_argument("Require M,N > 0, M*N <= INT32_MAX, and K in {32,64,128,256}");
        }
        if (config.n % config.k != 0 || (config.b != 1 && config.b != config.n / config.k)) {
            throw std::invalid_argument("N must be divisible by K; B must be 1 or N/K");
        }
        if ((config.dtype != "float16" && config.dtype != "bfloat16") ||
            (config.groupListType != 0 && config.groupListType != 1)) {
            throw std::invalid_argument("Require float16/bfloat16 input and groupListType 0/1");
        }
        ParseGroups(argv[7], config);
        config.hasAlpha = std::string(argv[8]) != "none";
        if (config.hasAlpha) {
            const std::string text = argv[8];
            size_t consumed = 0;
            float alpha = std::stof(text, &consumed);
            if (consumed != text.size() || !std::isfinite(alpha) || alpha < 0.0f || alpha > 1.0f) {
                throw std::invalid_argument("This example accepts alpha=none or a finite value in [0,1]");
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
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

    void CopyFromHost(const void* host) const
    {
        ACL_CHECK(aclrtMemcpy(data_, size_, host, size_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyFromFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        if (!ReadFile(path, host.data(), size_)) {
            throw std::runtime_error("Failed to read " + path);
        }
        CopyFromHost(host.data());
    }

    void CopyToFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        ACL_CHECK(aclrtMemcpy(host.data(), size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), size_)) {
            throw std::runtime_error("Failed to write " + path);
        }
    }

    void Fill(uint32_t value) const { ACL_CHECK(aclrtMemset(data_, size_, value, size_)); }

private:
    uint8_t* data_{nullptr};
    size_t size_{0};
};

RotateQuantTilingData MakeTiling(const ExampleConfig& config, const PlatformInfo& platform)
{
    const uint64_t k = static_cast<uint64_t>(config.k);
    // L0B is double buffered. K=256 therefore uses several K slices before Fixpipe.
    const uint64_t l0SlotBytes = std::min(platform.l0aSize, platform.l0bSize) / 2;
    uint64_t baseK = std::min(k, l0SlotBytes / (k * sizeof(uint16_t)));
    baseK = baseK / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    const uint64_t l1Bytes = 2 * (BLOCK_M * k + k * k) * sizeof(uint16_t);
    const uint64_t l0aBytes = 2 * BLOCK_M * baseK * sizeof(uint16_t);
    const uint64_t l0cBytes = 2 * BLOCK_M * k * sizeof(float);
    if (baseK == 0 || l1Bytes > platform.l1Size || l0aBytes > platform.l0aSize || l0cBytes > platform.l0cSize ||
        REQUIRED_UB_BYTES > platform.ubSize) {
        throw std::runtime_error("The example block does not fit the device buffers");
    }
    return {config.m,
            config.n,
            config.k,
            config.b,
            static_cast<int64_t>(baseK),
            static_cast<int64_t>(config.groupList.size()),
            config.groupListType,
            config.hasAlpha ? 1L : 0L};
}

template <typename XType_, bool EnableGrouped_>
void LaunchKernel(uint32_t blockNum, aclrtStream stream, const DeviceBuffer& x, const DeviceBuffer& rotation,
                  const DeviceBuffer& alpha, const DeviceBuffer& groups, const DeviceBuffer& y,
                  const DeviceBuffer& scale, const DeviceBuffer& tiling, bool hasAlpha)
{
    RotateQuantKernel<XType_, EnableGrouped_>
        <<<blockNum, 0, stream>>>(x.Get(), rotation.Get(), hasAlpha ? alpha.Get() : nullptr,
                                  EnableGrouped_ ? groups.Get() : nullptr, y.Get(), scale.Get(), tiling.Get());
}

template <typename XType_>
void RunCase(const ExampleConfig& config, const PlatformInfo& platform, aclrtStream stream)
{
    const auto tilingData = MakeTiling(config, platform);
    const uint32_t blockNum = std::min(MAX_BLOCK_NUM, platform.aicNum);
    const size_t elements = static_cast<size_t>(config.m) * config.n;
    const size_t matrices = std::max<size_t>(config.groupList.size(), 1) * config.b;
    const size_t scaleColumns = CeilDiv<size_t>(config.n, 2 * MX_GROUP_SIZE) * 2;
    DeviceBuffer x(elements * sizeof(uint16_t));
    DeviceBuffer rotation(matrices * config.k * config.k * sizeof(uint16_t));
    DeviceBuffer alpha(sizeof(uint16_t));
    DeviceBuffer groups(std::max<size_t>(config.groupList.size(), 1) * sizeof(int64_t));
    DeviceBuffer y(elements / 2);
    DeviceBuffer scale(static_cast<size_t>(config.m) * scaleColumns);
    DeviceBuffer tiling(sizeof(RotateQuantTilingData));

    x.CopyFromFile(config.dataDir + "/input_x.bin");
    rotation.CopyFromFile(config.dataDir + "/input_rotation.bin");
    alpha.CopyFromFile(config.dataDir + "/input_alpha.bin");
    groups.CopyFromFile(config.dataDir + "/input_group_list.bin");
    tiling.CopyFromHost(&tilingData);
    y.Fill(OUTPUT_SENTINEL);
    scale.Fill(OUTPUT_SENTINEL);

    if (config.groupList.empty()) {
        LaunchKernel<XType_, false>(blockNum, stream, x, rotation, alpha, groups, y, scale, tiling, config.hasAlpha);
    } else {
        LaunchKernel<XType_, true>(blockNum, stream, x, rotation, alpha, groups, y, scale, tiling, config.hasAlpha);
    }
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
    try {
        aclrtStream stream{nullptr};
        ACLDeviceGuard guard(stream);
        PlatformInfo platform{};
        if (!InitPlatformInfo(platform) || platform.aicNum == 0 || platform.aivNum < 2 * platform.aicNum) {
            throw std::runtime_error("This example requires Ascend950 with two AIVs per AIC");
        }
        if (config.dtype == "float16") {
            RunCase<half>(config, platform, stream);
        } else {
            RunCase<bfloat16_t>(config, platform, stream);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
