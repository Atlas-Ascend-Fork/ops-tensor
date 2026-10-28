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
 * \file grouped_matmul_mx_a8w4.cpp
 * \brief CSV-driven grouped MX A8W4 example using the Blaze Tensor API kernel.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

#if defined(IMPL_STD_ASCENDC_STD_INT_IMPL_H) && !defined(IMPL_TENSOR_API_UTILS_INT_IMPL_H)
#define IMPL_TENSOR_API_UTILS_INT_IMPL_H
#endif

#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"

#define ACL_CHECK(expr)                                                                                               \
    do {                                                                                                              \
        const aclError aclCheckResult = (expr);                                                                       \
        if (aclCheckResult != ACL_SUCCESS) {                                                                          \
            std::cerr << "ACL call failed: " << #expr << ", error " << static_cast<int>(aclCheckResult) << std::endl; \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace {

constexpr uint64_t FP4_PACK_FACTOR = 2U;
constexpr uint64_t TENSOR_LIST_ADDRESS_OFFSET = 3U * sizeof(uint64_t);
constexpr uint64_t TENSOR_LIST_SHAPE_SENTINEL = 0xFFFFFFFFULL;
constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2U;
constexpr size_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2U;
constexpr int64_t INVALID_EXPERT_INDEX = -1;

struct GroupedMatmulMxTilingData {
    uint32_t groupNum{0U};
    uint32_t coreNum{0U};
    uint64_t kSize{0U};
    uint64_t nSize{0U};
    uint8_t cubeNumBlocksN{0U};
    uint32_t mainBlockSize{0U};
    uint64_t mainBlockCount{0U};
    uint16_t firstTailBlockSize{0U};
    uint16_t secondTailBlockSize{0U};
    uint16_t firstTailBlockCount{0U};
    uint16_t secondTailBlockCount{0U};
    uint16_t baseM{0U};
    uint32_t groupListType{0U};
    uint32_t hasBias{0U};
};

struct CaseConfig {
    GroupedMatmulMxTilingData tiling;
    int64_t totalM{0};
    uint8_t singleW{0U};
    std::string weightDtype;
    std::string cDtype;
    std::vector<int64_t> groupList;
    std::string dataDir;
    std::string outputPath;
};

struct BufferSizes {
    size_t a{0U};
    size_t weightPerGroup{0U};
    size_t scaleA{0U};
    size_t scaleBPerGroup{0U};
    size_t biasPerGroup{0U};
    size_t c{0U};
};

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0U ? value : (value + alignment - 1U) / alignment * alignment;
}

std::vector<int64_t> ParseGroupList(const std::string& text)
{
    std::vector<int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (item.empty()) {
            throw std::invalid_argument("groupList contains an empty item");
        }
        values.push_back(std::stoll(item));
    }
    return values;
}

void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " <groupNum> <totalM> <n> <k> <weightDtype> <cDtype> <baseM> <isBias> <groupListType>"
                 " <singleW> <groupList> <mainBlockSize> <mainBlockCount> <firstTailBlockSize>"
                 " <firstTailBlockCount> <secondTailBlockSize> <secondTailBlockCount> <coreNum>"
                 " <cubeNumBlocksN> <dataDir> <outputPath>"
              << std::endl;
}

CaseConfig ParseConfig(char** argv)
{
    CaseConfig config{};
    auto& tiling = config.tiling;
    tiling.groupNum = static_cast<uint32_t>(std::stoul(argv[1]));
    config.totalM = std::stoll(argv[2]);
    tiling.nSize = std::stoull(argv[3]);
    tiling.kSize = std::stoull(argv[4]);
    config.weightDtype = argv[5];
    config.cDtype = argv[6];
    tiling.baseM = static_cast<uint16_t>(std::stoul(argv[7]));
    tiling.hasBias = static_cast<uint32_t>(std::stoul(argv[8]));
    tiling.groupListType = static_cast<uint32_t>(std::stoul(argv[9]));
    config.singleW = static_cast<uint8_t>(std::stoul(argv[10]));
    config.groupList = ParseGroupList(argv[11]);
    tiling.mainBlockSize = static_cast<uint32_t>(std::stoul(argv[12]));
    tiling.mainBlockCount = std::stoull(argv[13]);
    tiling.firstTailBlockSize = static_cast<uint16_t>(std::stoul(argv[14]));
    tiling.firstTailBlockCount = static_cast<uint16_t>(std::stoul(argv[15]));
    tiling.secondTailBlockSize = static_cast<uint16_t>(std::stoul(argv[16]));
    tiling.secondTailBlockCount = static_cast<uint16_t>(std::stoul(argv[17]));
    tiling.coreNum = static_cast<uint32_t>(std::stoul(argv[18]));
    tiling.cubeNumBlocksN = static_cast<uint8_t>(std::stoul(argv[19]));
    config.dataDir = argv[20];
    config.outputPath = argv[21];
    return config;
}

bool IsValidGroupList(const CaseConfig& config)
{
    const auto& tiling = config.tiling;
    const size_t entryCount = static_cast<size_t>(tiling.groupNum) *
                              (tiling.groupListType == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1U);
    if (config.groupList.size() != entryCount ||
        std::any_of(config.groupList.begin(), config.groupList.end(), [](int64_t value) { return value < 0; })) {
        return false;
    }
    if (tiling.groupListType == GROUP_LIST_TYPE_SPARSE) {
        std::vector<bool> seen(tiling.groupNum, false);
        int64_t previousActive = INVALID_EXPERT_INDEX;
        int64_t previousZero = INVALID_EXPERT_INDEX;
        int64_t totalM = 0;
        bool hasZeroGroup = false;
        for (size_t index = 0U; index < entryCount; index += SPARSE_GROUP_LIST_ITEM_STRIDE) {
            const int64_t expert = config.groupList[index];
            const int64_t count = config.groupList[index + 1U];
            if (expert >= tiling.groupNum || seen[expert]) {
                return false;
            }
            seen[expert] = true;
            if (count == 0) {
                if (expert <= previousZero) {
                    return false;
                }
                previousZero = expert;
                hasZeroGroup = true;
            } else {
                if (hasZeroGroup || expert <= previousActive || count > config.totalM - totalM) {
                    return false;
                }
                previousActive = expert;
                totalM += count;
            }
        }
        return totalM == config.totalM;
    }
    if (tiling.groupListType == 0U) {
        return std::is_sorted(config.groupList.begin(), config.groupList.end()) &&
               config.groupList.back() == config.totalM;
    }
    return std::accumulate(config.groupList.begin(), config.groupList.end(), int64_t{0}) == config.totalM;
}

bool IsValidConfig(const CaseConfig& config)
{
    const auto& tiling = config.tiling;
    const bool validShape = tiling.groupNum > 0U && config.totalM > 0 && tiling.nSize > 0U && tiling.kSize > 0U &&
                            tiling.kSize % 64U == 0U && tiling.nSize % 64U == 0U;
    const bool validLimits = (config.singleW == 1U && tiling.groupNum <= 1024U) ||
                             (config.singleW == 0U && tiling.groupNum <= 128U);
    const bool validTypes = (config.weightDtype == "float4_e2m1" || config.weightDtype == "float4_e1m2") &&
                            (config.cDtype == "float16" || config.cDtype == "bfloat16");
    const bool validFlags = tiling.hasBias <= 1U && tiling.groupListType <= GROUP_LIST_TYPE_SPARSE &&
                            config.singleW <= 1U;
    const uint64_t nBlockCount = tiling.mainBlockCount + tiling.firstTailBlockCount + tiling.secondTailBlockCount;
    const uint64_t coveredN = static_cast<uint64_t>(tiling.mainBlockSize) * tiling.mainBlockCount +
                              static_cast<uint64_t>(tiling.firstTailBlockSize) * tiling.firstTailBlockCount +
                              static_cast<uint64_t>(tiling.secondTailBlockSize) * tiling.secondTailBlockCount;
    const bool validSchedule = tiling.baseM > 0U && tiling.coreNum > 0U && nBlockCount > 0U &&
                               nBlockCount == tiling.cubeNumBlocksN && coveredN == tiling.nSize &&
                               (tiling.mainBlockCount == 0U || tiling.mainBlockSize > 0U) &&
                               (tiling.firstTailBlockCount == 0U || tiling.firstTailBlockSize > 0U) &&
                               (tiling.secondTailBlockCount == 0U || tiling.secondTailBlockSize > 0U);
    return validShape && validLimits && validTypes && validFlags && validSchedule && IsValidGroupList(config);
}

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

    void Clear() const { ACL_CHECK(aclrtMemset(data_, size_, 0, size_)); }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

struct DeviceInputs {
    std::unique_ptr<DeviceBuffer> a;
    std::unique_ptr<DeviceBuffer> b;
    std::unique_ptr<DeviceBuffer> bias;
    std::unique_ptr<DeviceBuffer> scaleA;
    std::unique_ptr<DeviceBuffer> scaleB;
    std::unique_ptr<DeviceBuffer> c;
    std::unique_ptr<DeviceBuffer> groupList;
    std::unique_ptr<DeviceBuffer> tiling;
    std::vector<std::unique_ptr<DeviceBuffer>> weights;
    std::vector<std::unique_ptr<DeviceBuffer>> biases;
    std::vector<std::unique_ptr<DeviceBuffer>> scales;
};

BufferSizes CalculateBufferSizes(const CaseConfig& config, size_t outputElementSize)
{
    const auto& tiling = config.tiling;
    const size_t scaleK = static_cast<size_t>(AlignUp(tiling.kSize, 64U) / 32U);
    const size_t weightElements = static_cast<size_t>(AlignUp(tiling.kSize, 32U) * AlignUp(tiling.nSize, 16U));
    return BufferSizes{static_cast<size_t>(config.totalM) * tiling.kSize,
                       weightElements / FP4_PACK_FACTOR,
                       static_cast<size_t>(config.totalM) * scaleK,
                       static_cast<size_t>(tiling.nSize) * scaleK,
                       static_cast<size_t>(tiling.nSize) * outputElementSize,
                       static_cast<size_t>(config.totalM) * tiling.nSize * outputElementSize};
}

std::unique_ptr<DeviceBuffer> MakeTensorList(const std::vector<std::unique_ptr<DeviceBuffer>>& tensors)
{
    std::vector<uint64_t> addresses(tensors.size() + 3U, 0U);
    addresses[0] = TENSOR_LIST_ADDRESS_OFFSET;
    addresses[1] = static_cast<uint64_t>(tensors.size()) << 32U;
    addresses[2] = TENSOR_LIST_SHAPE_SENTINEL;
    for (size_t index = 0U; index < tensors.size(); ++index) {
        addresses[index + 3U] = reinterpret_cast<uint64_t>(tensors[index]->Get());
    }
    auto tensorList = std::make_unique<DeviceBuffer>(addresses.size() * sizeof(uint64_t));
    tensorList->CopyFromHost(addresses.data(), addresses.size() * sizeof(uint64_t));
    return tensorList;
}

void PrepareContiguousWeights(DeviceInputs& device, const CaseConfig& config, const BufferSizes& sizes)
{
    const size_t groupNum = config.tiling.groupNum;
    device.b = std::make_unique<DeviceBuffer>(groupNum * sizes.weightPerGroup);
    device.scaleB = std::make_unique<DeviceBuffer>(groupNum * sizes.scaleBPerGroup);
    device.b->CopyFromFile(config.dataDir + "/input_b.bin", groupNum * sizes.weightPerGroup);
    device.scaleB->CopyFromFile(config.dataDir + "/scale_b.bin", groupNum * sizes.scaleBPerGroup);
    if (config.tiling.hasBias == 1U) {
        device.bias = std::make_unique<DeviceBuffer>(groupNum * sizes.biasPerGroup);
        device.bias->CopyFromFile(config.dataDir + "/bias.bin", groupNum * sizes.biasPerGroup);
    }
}

void PrepareTensorListWeights(DeviceInputs& device, const CaseConfig& config, const BufferSizes& sizes)
{
    for (uint32_t group = 0U; group < config.tiling.groupNum; ++group) {
        auto weight = std::make_unique<DeviceBuffer>(sizes.weightPerGroup);
        auto scale = std::make_unique<DeviceBuffer>(sizes.scaleBPerGroup);
        weight->CopyFromFile(config.dataDir + "/input_b_" + std::to_string(group) + ".bin", sizes.weightPerGroup);
        scale->CopyFromFile(config.dataDir + "/scale_b_" + std::to_string(group) + ".bin", sizes.scaleBPerGroup);
        device.weights.emplace_back(std::move(weight));
        device.scales.emplace_back(std::move(scale));
        if (config.tiling.hasBias == 1U) {
            auto bias = std::make_unique<DeviceBuffer>(sizes.biasPerGroup);
            bias->CopyFromFile(config.dataDir + "/bias_" + std::to_string(group) + ".bin", sizes.biasPerGroup);
            device.biases.emplace_back(std::move(bias));
        }
    }
    device.b = MakeTensorList(device.weights);
    device.scaleB = MakeTensorList(device.scales);
    if (config.tiling.hasBias == 1U) {
        device.bias = MakeTensorList(device.biases);
    }
}

DeviceInputs PrepareInputs(const CaseConfig& config, const BufferSizes& sizes)
{
    DeviceInputs device{};
    device.a = std::make_unique<DeviceBuffer>(sizes.a);
    device.scaleA = std::make_unique<DeviceBuffer>(sizes.scaleA);
    device.c = std::make_unique<DeviceBuffer>(sizes.c);
    device.groupList = std::make_unique<DeviceBuffer>(config.groupList.size() * sizeof(int64_t));
    device.tiling = std::make_unique<DeviceBuffer>(sizeof(GroupedMatmulMxTilingData));
    device.a->CopyFromFile(config.dataDir + "/input_a.bin", sizes.a);
    device.scaleA->CopyFromFile(config.dataDir + "/scale_a.bin", sizes.scaleA);
    device.groupList->CopyFromFile(config.dataDir + "/group_list.bin", config.groupList.size() * sizeof(int64_t));
    device.tiling->CopyFromHost(&config.tiling, sizeof(config.tiling));
    device.c->Clear();
    if (config.singleW == 1U) {
        PrepareContiguousWeights(device, config, sizes);
    } else {
        PrepareTensorListWeights(device, config, sizes);
    }
    return device;
}

template <typename WeightType_, typename OutputType_, bool IsSingleMultiSingle_>
__global__ __aicore__ void GroupedMatmulMxA8W4Kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm,
                                                     GM_ADDR scaleBGm, GM_ADDR cGm, GM_ADDR groupListGm,
                                                     GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const auto* inputTiling = reinterpret_cast<__gm__ const GroupedMatmulMxTilingData*>(tilingGm);
    using AType = fp8_e4m3fn_t;
    using BType = WeightType_;
    using ScaleType = fp8_e8m0_t;
    using CType = OutputType_;
    using BiasType = OutputType_;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithWeightQuantMx;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using ProblemShape = decltype(asc::te::make_shape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler = Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(asc::te::make_shape(0UL, 0UL,
                                                                                                         0UL))>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
        LayoutBias>;
    using BlockPrologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<AType, BType, BiasType>;
    using KernelImpl = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<ProblemShape, BlockMmad, BlockScheduler, void,
                                                                   BlockPrologue, IsSingleMultiSingle_>;

    typename BlockMmad::Params mmParams{};
    mmParams.aGmAddr = aGm;
    mmParams.scaleAGmAddr = scaleAGm;
    mmParams.scaleBGmAddr = scaleBGm;
    mmParams.biasGmAddr = biasGm;
    mmParams.cGmAddr = cGm;
    typename BlockScheduler::Params schedulerParams{inputTiling->mainBlockCount,
                                                    inputTiling->mainBlockSize,
                                                    inputTiling->firstTailBlockCount,
                                                    inputTiling->firstTailBlockSize,
                                                    inputTiling->secondTailBlockCount,
                                                    inputTiling->secondTailBlockSize,
                                                    inputTiling->coreNum,
                                                    inputTiling->cubeNumBlocksN,
                                                    inputTiling->baseM,
                                                    inputTiling->nSize};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType*>(bGm)};
    typename KernelImpl::Params params{
        asc::te::make_shape(0UL, static_cast<uint64_t>(inputTiling->kSize), static_cast<uint64_t>(inputTiling->nSize),
                            static_cast<uint64_t>(inputTiling->groupNum)),
        mmParams,
        schedulerParams,
        prologueParams,
        groupListGm,
        inputTiling->groupListType,
        inputTiling->hasBias};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

template <typename WeightType_, typename OutputType_, bool IsSingleMultiSingle_>
void RunTypedCase(const CaseConfig& config)
{
    const BufferSizes sizes = CalculateBufferSizes(config, sizeof(OutputType_));
    DeviceInputs device = PrepareInputs(config, sizes);
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    GroupedMatmulMxA8W4Kernel<WeightType_, OutputType_, IsSingleMultiSingle_><<<config.tiling.coreNum, 0, stream>>>(
        device.a->Get(), device.b->Get(), device.bias == nullptr ? nullptr : device.bias->Get(), device.scaleA->Get(),
        device.scaleB->Get(), device.c->Get(), device.groupList->Get(), device.tiling->Get());
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtDestroyStream(stream));
    WriteBinary(config.outputPath, device.c->CopyToHost(sizes.c));
}

template <typename WeightType_, typename OutputType_>
void RunStorageCase(const CaseConfig& config)
{
    if (config.singleW == 0U) {
        RunTypedCase<WeightType_, OutputType_, true>(config);
    } else {
        RunTypedCase<WeightType_, OutputType_, false>(config);
    }
}

template <typename OutputType_>
void RunWeightCase(const CaseConfig& config)
{
    if (config.weightDtype == "float4_e2m1") {
        RunStorageCase<fp4x2_e2m1_t, OutputType_>(config);
    } else if (config.weightDtype == "float4_e1m2") {
        RunStorageCase<fp4x2_e1m2_t, OutputType_>(config);
    } else {
        throw std::invalid_argument("unsupported weight dtype: " + config.weightDtype);
    }
}

void RunCase(const CaseConfig& config)
{
    if (config.cDtype == "float16") {
        RunWeightCase<half>(config);
    } else if (config.cDtype == "bfloat16") {
        RunWeightCase<bfloat16_t>(config);
    } else {
        throw std::invalid_argument("unsupported output dtype: " + config.cDtype);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 22) {
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        const CaseConfig config = ParseConfig(argv);
        if (!IsValidConfig(config)) {
            std::cerr << "invalid grouped MX A8W4 case configuration" << std::endl;
            return 2;
        }
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        RunCase(config);
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::cout << "Grouped MX A8W4 kernel completed, weight=" << config.weightDtype << ", output=" << config.cDtype
                  << ", result=" << config.outputPath << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Grouped MX A8W4 example failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
