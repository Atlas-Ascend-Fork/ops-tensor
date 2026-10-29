/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "data_utils.h"
#include "blaze/gemm/kernel/kernel_wqmm_mix_antiquant.h"
#include "blaze/gemm/block/block_scheduler_wqmm.h"

namespace {
constexpr int64_t BASE_M = 32;
constexpr int64_t BASE_N = 64;
constexpr int64_t BASE_K = 64;
constexpr int64_t L1_K = 128;
constexpr uint32_t BLOCK_NUM = 2;

struct Config {
    int64_t m, k, n;
    bool bf16, transB, perTensor, offset, bias;
    std::string directory;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(data_, bytes_, 0, bytes_));
    }
    ~DeviceBuffer() { aclrtFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint8_t* Get() const { return data_; }
    void Load(const std::string& path)
    {
        std::vector<uint8_t> host(bytes_);
        if (!ReadFile(path, host.data(), bytes_)) {
            throw std::runtime_error("Failed to read input: " + path);
        }
        ACL_CHECK(aclrtMemcpy(data_, bytes_, host.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void Save(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        ACL_CHECK(aclrtMemcpy(host.data(), bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), bytes_)) {
            throw std::runtime_error("Failed to write output: " + path);
        }
    }

private:
    uint8_t* data_{nullptr};
    size_t bytes_;
};

template <typename T, bool TransB, bool PerTensor, bool HasOffset>
__global__ __aicore__ void WqmmAntiquantKernel(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias,
                                               GM_ADDR c, int64_t m, int64_t k, int64_t n, bool hasBias)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using Policy = Blaze::Gemm::MatmulWithWeightAntiquant<
        2, 512, 2, PerTensor ? Blaze::Gemm::QuantMode::PERTENSOR_MODE : Blaze::Gemm::QuantMode::PERCHANNEL_MODE,
        HasOffset>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using BLayout = AscendC::Std::conditional_t<TransB, Layout, asc::te::dn_ext_layout_ptn>;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t>;
    using Block = Blaze::Gemm::Block::BlockMmad<Policy, T, Layout, AscendC::Std::tuple<int8_t, T>,
                                                AscendC::Std::tuple<BLayout, Layout>, T, Layout, T, Layout>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmTailResplit<Shape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Block, void, Scheduler>;
    typename Kernel::Params params{};
    params.problemShape = asc::te::make_shape(m, n, k);
    params.mmadParams.aGmAddr = a;
    params.mmadParams.cGmAddr = c;
    params.mmadParams.biasGmAddr = bias;
    params.mmadParams.l1TileShape = asc::te::make_shape(BASE_M, BASE_N, L1_K, L1_K);
    params.mmadParams.l0TileShape = asc::te::make_shape(BASE_M, BASE_N, BASE_K);
    params.mmadParams.hasBias = hasBias;
    params.mmadParams.kSize = k;
    params.prologueParams = {b, scale, offset, 1U};
    // Full N rounds use both AICs. Split the remainder between two tail ranges.
    const uint64_t mainCount = static_cast<uint64_t>(n / (BASE_N * BLOCK_NUM)) * BLOCK_NUM;
    const uint64_t remainder = static_cast<uint64_t>(n) - mainCount * BASE_N;
    const uint64_t first = remainder / BLOCK_NUM;
    const uint64_t second = remainder - first;
    params.schedulerParams = {BASE_M, mainCount, first != 0 ? 1UL : 0UL, second != 0 ? 1UL : 0UL, BASE_N, first, second,
                              1UL,    BLOCK_NUM};
    Kernel{}(params);
}

template <typename T, bool TransB, bool PerTensor, bool HasOffset>
void Run(const Config& config, aclrtStream stream)
{
    const size_t quantCount = PerTensor ? 1 : config.n;
    DeviceBuffer a(config.m * config.k * sizeof(T));
    DeviceBuffer b(config.k * config.n);
    DeviceBuffer scale(quantCount * sizeof(T));
    DeviceBuffer offset(quantCount * sizeof(T));
    DeviceBuffer bias(config.n * sizeof(T));
    DeviceBuffer c(config.m * config.n * sizeof(T));
    a.Load(config.directory + "/a.bin");
    b.Load(config.directory + "/b.bin");
    scale.Load(config.directory + "/scale.bin");
    if constexpr (HasOffset) {
        offset.Load(config.directory + "/offset.bin");
    }
    if (config.bias) {
        bias.Load(config.directory + "/bias.bin");
    }
    WqmmAntiquantKernel<T, TransB, PerTensor, HasOffset><<<BLOCK_NUM, 0, stream>>>(
        a.Get(), b.Get(), scale.Get(), HasOffset ? offset.Get() : nullptr, config.bias ? bias.Get() : nullptr, c.Get(),
        config.m, config.k, config.n, config.bias);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    c.Save(config.directory + "/npu_out.bin");
}

template <typename T, bool TransB, bool PerTensor>
void SelectOffset(const Config& config, aclrtStream stream)
{
    if (config.offset) {
        Run<T, TransB, PerTensor, true>(config, stream);
    } else {
        Run<T, TransB, PerTensor, false>(config, stream);
    }
}

template <typename T, bool TransB>
void SelectQuant(const Config& config, aclrtStream stream)
{
    if (config.perTensor) {
        SelectOffset<T, TransB, true>(config, stream);
    } else {
        SelectOffset<T, TransB, false>(config, stream);
    }
}

template <typename T>
void SelectLayout(const Config& config, aclrtStream stream)
{
    if (config.transB) {
        SelectQuant<T, true>(config, stream);
    } else {
        SelectQuant<T, false>(config, stream);
    }
}

bool ParseBool(const char* arg)
{
    const std::string value(arg);
    if (value != "0" && value != "1") {
        throw std::invalid_argument("Boolean arguments must be 0 or 1");
    }
    return value == "1";
}
} // namespace

int main(int argc, const char** argv)
{
    try {
        if (argc != 10) {
            throw std::invalid_argument(
                "Usage: <m> <k> <n> <fp16|bf16> <trans_b> <per_tensor> <offset> <bias> <data_dir>");
        }
        const std::string dtype(argv[4]);
        if (dtype != "fp16" && dtype != "bf16") {
            throw std::invalid_argument("dtype must be fp16 or bf16");
        }
        Config config{std::stoll(argv[1]), std::stoll(argv[2]), std::stoll(argv[3]),
                      dtype == "bf16",     ParseBool(argv[5]),  ParseBool(argv[6]),
                      ParseBool(argv[7]),  ParseBool(argv[8]),  argv[9]};
        if (config.m <= 0 || config.m > 256 || config.k <= 0 || config.k > 4096 || config.n <= 0 || config.n > 1024) {
            throw std::invalid_argument("Example supports 1<=M<=256, 1<=K<=4096, 1<=N<=1024");
        }
        aclrtStream stream{nullptr};
        ACLDeviceGuard guard(stream);
        if (config.bf16) {
            SelectLayout<bfloat16_t>(config, stream);
        } else {
            SelectLayout<half>(config, stream);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
