/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError result = (expr);                                                          \
        if (result != ACL_SUCCESS) {                                                             \
            throw std::runtime_error(std::string(#expr) + " failed: " + std::to_string(result)); \
        }                                                                                        \
    } while (0)

namespace {
constexpr uint64_t TENSOR_LIST_HEADER_WORDS = 3;
constexpr uint64_t TENSOR_LIST_SHAPE_SENTINEL = 0xFFFFFFFFULL;
constexpr int64_t CUBE_ALIGN = 16;
constexpr int64_t MAX_BASE_N = 256;
constexpr int64_t MAX_BASE_K = 64;
constexpr int64_t STEP_K = 4;
constexpr int64_t BASE_M_NO_BIAS = 256;
constexpr int64_t BASE_M_WITH_BIAS = 240;

struct ExampleTiling {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    uint32_t groups{0};
    uint32_t listType{0};
    bool hasBias{false};
};

struct CaseConfig {
    ExampleTiling tiling;
    std::string xDtype;
    std::string weightDtype;
    std::string biasDtype;
    bool trans{false};
    bool hasOffset{false};
    uint32_t cores{0};
    std::string dataDir;
    std::string output;
};

std::vector<uint8_t> ReadBinary(const std::string& path, size_t expectedSize)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open() || static_cast<size_t>(stream.tellg()) != expectedSize) {
        throw std::runtime_error("unexpected input file size: " + path);
    }
    std::vector<uint8_t> data(expectedSize);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return data;
}

void WriteBinary(const std::string& path, const std::vector<uint8_t>& data)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

// Cleanup must preserve an exception already propagating from the example.
aclError ReportCleanupError(const char* operation, aclError result) noexcept
{
    if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "%s failed during cleanup: %d\n", operation, static_cast<int>(result));
    }
    return result;
}

class AclRuntime {
public:
    explicit AclRuntime(int32_t deviceId) : deviceId_(deviceId)
    {
        ACL_CHECK(aclInit(nullptr));
        try {
            ACL_CHECK(aclrtSetDevice(deviceId_));
        } catch (...) {
            ReportCleanupError("aclFinalize", aclFinalize());
            throw;
        }
        active_ = true;
    }

    ~AclRuntime() noexcept { static_cast<void>(Close()); }
    AclRuntime(const AclRuntime&) = delete;
    AclRuntime& operator=(const AclRuntime&) = delete;

    aclError Close() noexcept
    {
        if (!active_) {
            return ACL_SUCCESS;
        }
        active_ = false;
        const aclError resetResult = ReportCleanupError("aclrtResetDevice", aclrtResetDevice(deviceId_));
        const aclError finalizeResult = ReportCleanupError("aclFinalize", aclFinalize());
        return resetResult != ACL_SUCCESS ? resetResult : finalizeResult;
    }

private:
    int32_t deviceId_;
    bool active_{false};
};

class AclStream {
public:
    AclStream() { ACL_CHECK(aclrtCreateStream(&stream_)); }

    ~AclStream() noexcept
    {
        if (stream_ != nullptr) {
            // On exceptional exits, finish submitted work before the buffers are released.
            ReportCleanupError("aclrtSynchronizeStream", aclrtSynchronizeStream(stream_));
            static_cast<void>(Close());
        }
    }

    AclStream(const AclStream&) = delete;
    AclStream& operator=(const AclStream&) = delete;
    aclrtStream Get() const { return stream_; }

    // The normal path has already synchronized the stream before calling Close.
    aclError Close() noexcept
    {
        if (stream_ == nullptr) {
            return ACL_SUCCESS;
        }
        const aclrtStream stream = stream_;
        stream_ = nullptr;
        return ReportCleanupError("aclrtDestroyStream", aclrtDestroyStream(stream));
    }

private:
    aclrtStream stream_{nullptr};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(std::max<size_t>(size, 1U))
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
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    uint8_t* Get() const { return data_; }

    void CopyFromHost(const void* source, size_t size) const
    {
        if (size > size_) {
            throw std::runtime_error("host-to-device copy exceeds allocation");
        }
        ACL_CHECK(aclrtMemcpy(data_, size_, source, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyFromFile(const std::string& path, size_t expectedSize) const
    {
        const auto data = ReadBinary(path, expectedSize);
        CopyFromHost(data.data(), data.size());
    }

    std::vector<uint8_t> CopyToHost(size_t size) const
    {
        if (size > size_) {
            throw std::runtime_error("device-to-host copy exceeds allocation");
        }
        std::vector<uint8_t> data(size);
        ACL_CHECK(aclrtMemcpy(data.data(), size, data_, size, ACL_MEMCPY_DEVICE_TO_HOST));
        return data;
    }

    void Poison() const { ACL_CHECK(aclrtMemset(data_, size_, 0xFF, size_)); }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

DeviceBuffer MakeSingleTensorList(const DeviceBuffer& payload)
{
    const std::array<uint64_t, TENSOR_LIST_HEADER_WORDS + 1> descriptor{TENSOR_LIST_HEADER_WORDS * sizeof(uint64_t),
                                                                        uint64_t{1} << 32, TENSOR_LIST_SHAPE_SENTINEL,
                                                                        reinterpret_cast<uint64_t>(payload.Get())};
    DeviceBuffer list(sizeof(descriptor));
    list.CopyFromHost(descriptor.data(), sizeof(descriptor));
    return list;
}

CaseConfig ParseConfig(char** argv)
{
    CaseConfig c{};
    const int64_t groups = std::stoll(argv[1]);
    c.tiling.m = std::stoll(argv[2]);
    c.tiling.n = std::stoll(argv[3]);
    c.tiling.k = std::stoll(argv[4]);
    c.xDtype = argv[5];
    c.weightDtype = argv[6];
    const int trans = std::stoi(argv[7]);
    const int offset = std::stoi(argv[8]);
    c.biasDtype = argv[9];
    const int listType = std::stoi(argv[10]);
    const int cores = std::stoi(argv[11]);
    c.dataDir = argv[12];
    c.output = argv[13];
    const bool validX = c.xDtype == "float16" || c.xDtype == "bfloat16";
    const bool validW = c.weightDtype == "int8" || c.weightDtype == "float8_e4m3fn" || c.weightDtype == "float8_e5m2" ||
                        c.weightDtype == "hifloat8";
    const bool validBias = c.biasDtype == "none" || c.biasDtype == c.xDtype ||
                           (c.xDtype == "bfloat16" && c.biasDtype == "float32");
    if (groups <= 0 || groups > 1024 || c.tiling.m <= 0 || c.tiling.n <= 0 || c.tiling.k <= 0 || c.tiling.m > 1048576 ||
        c.tiling.n > 4096 || c.tiling.k > 4096 || !validX || !validW || !validBias || trans < 0 || trans > 1 ||
        offset < 0 || offset > 1 || (offset && c.weightDtype != "int8") || listType < 0 || listType > 1 || cores <= 0 ||
        cores > 32) {
        throw std::invalid_argument("invalid grouped weight antiquant example configuration");
    }
    c.tiling.groups = static_cast<uint32_t>(groups);
    c.tiling.listType = static_cast<uint32_t>(listType);
    c.tiling.hasBias = c.biasDtype != "none";
    c.trans = trans != 0;
    c.hasOffset = offset != 0;
    c.cores = static_cast<uint32_t>(cores);
    return c;
}

void ValidateGroups(const std::vector<uint8_t>& bytes, const ExampleTiling& tiling)
{
    std::vector<int64_t> values(tiling.groups);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    int64_t total = 0;
    for (const auto value : values) {
        if (value < 0 || value > tiling.m || (tiling.listType == 0 && value < total)) {
            throw std::invalid_argument("invalid group list contents");
        }
        total = tiling.listType == 0 ? value : total + value;
    }
    if (total != tiling.m) {
        throw std::invalid_argument("group list total must equal M");
    }
}

template <class XType_, class WType_, class BiasType_, bool TransB_, bool HasOffset_>
__global__ __aicore__ void GroupedWeightAntiquantKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                                        GM_ADDR bias, GM_ADDR groups, GM_ADDR y, ExampleTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::InitSocState();
    using Nd = asc::te::nd_ext_layout_ptn;
    using WeightLayout = AscendC::Std::conditional_t<TransB_, Nd, asc::te::dn_ext_layout_ptn>;
    constexpr uint64_t AIV_COUNT = 1;
    constexpr uint32_t UB_BUFFER_COUNT = 2;
    // Shared Policy compatibility argument; GMM UB geometry comes from GroupedWeightAntiquantConfig.
    constexpr uint32_t POLICY_UB_INNER_BYTES = 256;
    using Policy = Blaze::Gemm::MatmulWithWeightAntiquant<AIV_COUNT, POLICY_UB_INNER_BYTES, UB_BUFFER_COUNT,
                                                          Blaze::Gemm::QuantMode::PERCHANNEL_MODE, HasOffset_,
                                                          Blaze::Gemm::KernelWqgmmMixAntiquant>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, XType_, Nd, AscendC::Std::tuple<WType_, XType_>,
                                               AscendC::Std::tuple<WeightLayout, Nd>, XType_, Nd, BiasType_, Nd>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Scheduler::ProblemShape, Mmad,
                                                      Blaze::Epilogue::Block::BlockEpilogueEmpty, Scheduler>;
    const int64_t baseM = tiling.hasBias ? BASE_M_WITH_BIAS : BASE_M_NO_BIAS;
    const int64_t baseN = Blaze::Gemm::Min(MAX_BASE_N, Blaze::Gemm::CeilAlign(tiling.n, CUBE_ALIGN));
    const int64_t baseK = Blaze::Gemm::Min(MAX_BASE_K, Blaze::Gemm::CeilAlign(tiling.k, CUBE_ALIGN));
    typename Kernel::Params params{};
    params.problemShape = {tiling.m, tiling.n, tiling.k, 1};
    params.mmParams.kSize = tiling.k;
    params.mmParams.hasBias = tiling.hasBias;
    params.mmParams.l1TileShape = {baseM, baseN, baseK * STEP_K, baseK * STEP_K};
    params.mmParams.l0TileShape = {baseM, baseN, baseK};
    params.schedulerParams = {static_cast<int32_t>(baseM), static_cast<int32_t>(baseN)};
    // These are single-entry TensorList descriptors, except groups (a plain INT64 Tensor).
    // The public Kernel owns the one-time typed pointer -> Tensor boundary adaptation.
    params.x = x;
    params.weight = weight;
    params.scale = scale;
    params.offset = offset;
    params.bias = bias;
    params.y = y;
    params.groupList = groups;
    params.groupNum = tiling.groups;
    params.groupListType = tiling.listType;
    Kernel{}(params);
}

template <class XType_, class WType_, class BiasType_, bool TransB_, bool HasOffset_>
void RunTyped(const CaseConfig& c)
{
    const size_t m = c.tiling.m, n = c.tiling.n, k = c.tiling.k, e = c.tiling.groups;
    DeviceBuffer x(m * k * sizeof(XType_)), weight(e * k * n * sizeof(WType_));
    DeviceBuffer scale(e * n * sizeof(XType_)), offset(e * n * sizeof(XType_));
    DeviceBuffer bias(e * n * sizeof(BiasType_)), y(m * n * sizeof(XType_));
    DeviceBuffer groups(e * sizeof(int64_t));
    x.CopyFromFile(c.dataDir + "/x.bin", m * k * sizeof(XType_));
    weight.CopyFromFile(c.dataDir + "/weight.bin", e * k * n * sizeof(WType_));
    scale.CopyFromFile(c.dataDir + "/scale.bin", e * n * sizeof(XType_));
    if constexpr (HasOffset_) {
        offset.CopyFromFile(c.dataDir + "/offset.bin", e * n * sizeof(XType_));
    }
    if (c.tiling.hasBias) {
        bias.CopyFromFile(c.dataDir + "/bias.bin", e * n * sizeof(BiasType_));
    }
    const auto groupBytes = ReadBinary(c.dataDir + "/group_list.bin", e * sizeof(int64_t));
    ValidateGroups(groupBytes, c.tiling);
    groups.CopyFromHost(groupBytes.data(), groupBytes.size());
    auto xList = MakeSingleTensorList(x), wList = MakeSingleTensorList(weight);
    auto scaleList = MakeSingleTensorList(scale), offsetList = MakeSingleTensorList(offset);
    auto biasList = MakeSingleTensorList(bias), yList = MakeSingleTensorList(y);
    AclStream stream;
    std::vector<uint8_t> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        y.Poison();
        GroupedWeightAntiquantKernel<XType_, WType_, BiasType_, TransB_, HasOffset_><<<c.cores, 0, stream.Get()>>>(
            xList.Get(), wList.Get(), scaleList.Get(), HasOffset_ ? offsetList.Get() : nullptr,
            c.tiling.hasBias ? biasList.Get() : nullptr, groups.Get(), yList.Get(), c.tiling);
        ACL_CHECK(aclrtSynchronizeStream(stream.Get()));
        auto result = y.CopyToHost(m * n * sizeof(XType_));
        if (repeat == 0) {
            first = result;
        } else if (result != first) {
            throw std::runtime_error("consecutive launches produced different outputs");
        }
    }
    ACL_CHECK(stream.Close());
    WriteBinary(c.output, first);
}

template <class XType_, class WType_, class BiasType_, bool HasOffset_>
void RunTranspose(const CaseConfig& c)
{
    if (c.trans) {
        RunTyped<XType_, WType_, BiasType_, true, HasOffset_>(c);
    } else {
        RunTyped<XType_, WType_, BiasType_, false, HasOffset_>(c);
    }
}

template <class XType_, class BiasType_>
void RunWeight(const CaseConfig& c)
{
    if (c.weightDtype == "int8") {
        if (c.hasOffset) {
            RunTranspose<XType_, int8_t, BiasType_, true>(c);
        } else {
            RunTranspose<XType_, int8_t, BiasType_, false>(c);
        }
    } else if (c.weightDtype == "float8_e4m3fn") {
        RunTranspose<XType_, float8_e4m3_t, BiasType_, false>(c);
    } else if (c.weightDtype == "float8_e5m2") {
        RunTranspose<XType_, float8_e5m2_t, BiasType_, false>(c);
    } else {
        RunTranspose<XType_, hifloat8_t, BiasType_, false>(c);
    }
}

void RunCase(const CaseConfig& c)
{
    if (c.xDtype == "float16") {
        RunWeight<half, half>(c);
    } else if (c.biasDtype == "float32") {
        RunWeight<bfloat16_t, float>(c);
    } else {
        RunWeight<bfloat16_t, bfloat16_t>(c);
    }
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 14) {
        std::cerr << "Usage: " << argv[0]
                  << " E M N K xDtype weightDtype trans hasOffset biasDtype listType cores dataDir output\n";
        return 2;
    }
    try {
        const auto config = ParseConfig(argv);
        AclRuntime runtime(0);
        RunCase(config);
        ACL_CHECK(runtime.Close());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
