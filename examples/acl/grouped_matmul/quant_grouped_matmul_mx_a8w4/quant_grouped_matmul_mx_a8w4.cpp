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
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "data_utils.h"

namespace {
constexpr uint32_t BASE_M = 16;
constexpr uint32_t BASE_N = 64;
constexpr uint32_t MAX_BASE_K = 128;
constexpr uint32_t K_L1_WINDOW = 4 * MAX_BASE_K;
constexpr uint32_t SCALE_K_L1_WINDOW = 4096;
constexpr size_t SCALE_K_PAIR = 64;
constexpr size_t SCALE_PAIR_ELEMENTS = 2;
constexpr size_t FP4_PER_BYTE = 2;

template <typename BiasType>
__global__ __aicore__ void QgmmMxA8w4Kernel(GM_ADDR a, GM_ADDR b, GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR bias,
                                            GM_ADDR c, GM_ADDR groupList, int64_t m, int64_t n, int64_t k,
                                            uint32_t groups, uint32_t listType, uint32_t dbL0C)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using Nd = asc::te::nd_ext_layout_ptn;
    using Zn = asc::te::zn_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, fp8_e4m3fn_t, Nd, fp4x2_e2m1_t, Zn, float, Nd, BiasType, Nd>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    const uint32_t alignedK = static_cast<uint32_t>(Blaze::Gemm::CeilAlign<uint64_t>(k, Blaze::Gemm::C0_SIZE_B4));
    const uint32_t baseK = Blaze::Gemm::Min(MAX_BASE_K, alignedK);
    // Use a full-K window when it divides the scale window, otherwise use fixed K strips.
    const uint32_t kL1 = k <= K_L1_WINDOW && k % SCALE_K_PAIR == 0 && SCALE_K_L1_WINDOW % k == 0 ?
                             static_cast<uint32_t>(k) :
                             K_L1_WINDOW;

    typename Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams.aGmAddr = a;
    params.mmadParams.bGmAddr = b;
    params.mmadParams.scaleAGmAddr = scaleA;
    params.mmadParams.scaleBGmAddr = scaleB;
    params.mmadParams.biasGmAddr = bias;
    params.mmadParams.cGmAddr = c;
    params.groupListGmAddr = groupList;
    auto& tiling = params.gmmParams;
    tiling.groupNum = groups;
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.baseM = BASE_M;
    tiling.baseN = BASE_N;
    tiling.baseK = baseK;
    tiling.kAL1 = kL1;
    tiling.kBL1 = kL1;
    tiling.scaleKAL1 = SCALE_K_L1_WINDOW;
    tiling.scaleKBL1 = SCALE_K_L1_WINDOW;
    tiling.isBias = bias != nullptr;
    tiling.dbL0C = static_cast<uint8_t>(dbL0C);
    tiling.groupType = 0; // Split M.
    tiling.groupListType = static_cast<uint8_t>(listType);
    tiling.singleW = 1; // B and scaleB each concatenate all experts in GM.
    Kernel{}(params);
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        if (bytes_ != 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint8_t* Get() const { return data_; }

    void Load(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        if (!ReadFile(path, host.data(), host.size())) {
            throw std::runtime_error("failed to read input: " + path);
        }
        CopyFromHost(host.data());
    }

    void CopyFromHost(const void* host) const
    {
        ACL_CHECK(aclrtMemcpy(data_, bytes_, host, bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Save(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        ACL_CHECK(aclrtMemcpy(host.data(), host.size(), data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), host.size())) {
            throw std::runtime_error("failed to write output: " + path);
        }
    }

private:
    size_t bytes_;
    uint8_t* data_{nullptr};
};

struct CaseConfig {
    int64_t groups;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t listType;
    int64_t dbL0C;
    uint32_t cores;
    std::string biasDtype;
    std::string dataDir;
    std::string output;
};

CaseConfig ParseConfig(char** argv)
{
    const auto cores = std::stoll(argv[8]);
    if (cores <= 0 || cores > UINT32_MAX) {
        throw std::invalid_argument("invalid AIC core count");
    }
    CaseConfig c{std::stoll(argv[1]),
                 std::stoll(argv[2]),
                 std::stoll(argv[3]),
                 std::stoll(argv[4]),
                 std::stoll(argv[6]),
                 std::stoll(argv[7]),
                 static_cast<uint32_t>(cores),
                 argv[5],
                 argv[9],
                 argv[10]};
    const bool validBias = c.biasDtype == "none" || c.biasDtype == "float16" || c.biasDtype == "bfloat16" ||
                           c.biasDtype == "float32";
    if (c.groups <= 0 || c.groups > 1024 || c.m <= 0 || c.m > 1048576 || c.n <= 0 || c.n > 4096 || c.k <= 0 ||
        c.k > 4096 || c.k % FP4_PER_BYTE != 0 || !validBias || (c.listType != 0 && c.listType != 1) ||
        (c.dbL0C != 1 && c.dbL0C != 2)) {
        throw std::invalid_argument("invalid MXA8W4 example configuration");
    }
    return c;
}

std::vector<int64_t> ReadGroups(const CaseConfig& c)
{
    std::vector<int64_t> values(c.groups);
    if (!ReadFile(c.dataDir + "/group_list.bin", values.data(), values.size() * sizeof(int64_t))) {
        throw std::runtime_error("failed to read group list");
    }
    int64_t total = 0;
    for (const int64_t value : values) {
        if (value < 0 || value > c.m || (c.listType == 0 && value < total)) {
            throw std::invalid_argument("invalid group list entry");
        }
        total = c.listType == 0 ? value : total + value;
    }
    if (total != c.m) {
        throw std::invalid_argument("group list must cover M rows");
    }
    return values;
}

template <typename BiasType>
void RunCase(const CaseConfig& config, aclrtStream stream)
{
    const size_t m = config.m, n = config.n, k = config.k, e = config.groups;
    const size_t scaleK = CeilDiv(k, SCALE_K_PAIR) * SCALE_PAIR_ELEMENTS;
    const size_t bGroup = CeilAlign<size_t>(k, Blaze::Gemm::C0_SIZE_B4) *
                          CeilAlign<size_t>(n, Blaze::Gemm::BLOCK_CUBE) / FP4_PER_BYTE;
    const size_t outputBytes = m * n * sizeof(float);
    DeviceBuffer a(m * k), b(e * bGroup), scaleA(m * scaleK), scaleB(e * n * scaleK);
    DeviceBuffer bias(config.biasDtype == "none" ? 0 : e * n * sizeof(BiasType));
    DeviceBuffer groupList(e * sizeof(int64_t)), c(outputBytes);
    a.Load(config.dataDir + "/input_a.bin");
    b.Load(config.dataDir + "/input_b.bin");
    scaleA.Load(config.dataDir + "/scale_a.bin");
    scaleB.Load(config.dataDir + "/scale_b.bin");
    if (bias.Get() != nullptr) {
        bias.Load(config.dataDir + "/bias.bin");
    }
    const auto groups = ReadGroups(config);
    groupList.CopyFromHost(groups.data());

    ACL_CHECK(aclrtMemset(c.Get(), outputBytes, 0xFF, outputBytes));
    QgmmMxA8w4Kernel<BiasType>
        <<<config.cores, 0, stream>>>(a.Get(), b.Get(), scaleA.Get(), scaleB.Get(), bias.Get(), c.Get(),
                                      groupList.Get(), config.m, config.n, config.k, static_cast<uint32_t>(e),
                                      static_cast<uint32_t>(config.listType), static_cast<uint32_t>(config.dbL0C));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    c.Save(config.output);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] << " E M N K biasDtype listType dbL0C cores dataDir output\n";
        return 2;
    }
    try {
        const auto config = ParseConfig(argv);
        aclrtStream stream{};
        ACLDeviceGuard guard(stream);
        if (config.cores > GetAicCoreNum()) {
            throw std::invalid_argument("cores exceeds the number of available AIC cores");
        }
        if (config.biasDtype == "float16") {
            RunCase<half>(config, stream);
        } else if (config.biasDtype == "bfloat16") {
            RunCase<bfloat16_t>(config, stream);
        } else {
            RunCase<float>(config, stream);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
