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
 * \file grouped_matmul_no_quant.cpp
 * \brief CSV-driven non-quant grouped matmul example using the Blaze Tensor API kernel.
 *
 * Covers the aclnnGroupedMatmulV5 (Ascend 950) non-quant scenario matrix:
 *   - groupType: -1 (no split, m-m-m), 0 (M-axis split: s-s-s / s-m-s / m-m-s), 2 (K-axis split:
 *     s-s-s with [G, M, N] output / s-m-m with multi output)
 *   - groupListType: 0 (cumsum), 1 (count), 2 (sparse [E, 2] pairs, M-axis only)
 *   - weight format: ND and FRACTAL_NZ
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

#include "blaze/gemm/kernel/kernel_universal.h"

#define ACL_CHECK(expr)                                                                                               \
    do {                                                                                                              \
        const aclError aclCheckResult = (expr);                                                                       \
        if (aclCheckResult != ACL_SUCCESS) {                                                                          \
            std::cerr << "ACL call failed: " << #expr << ", error " << static_cast<int>(aclCheckResult) << std::endl; \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace {

constexpr uint64_t NZ_K_INNER = 16U; // NZ frame k-axis inner block
constexpr uint64_t BLOCK_BYTES = 32U;

#pragma pack(push, 8)
struct GmmNoQuantTilingData {
    uint32_t groupNum;
    int32_t groupType;
    uint32_t groupListType;
    uint64_t singleX;
    uint64_t singleWeight;
    uint64_t singleY;
    uint32_t hasBias;
    uint32_t weightNoL2Cache;
    uint64_t mTailCnt;
    uint64_t nTailCnt;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t stepKa;
    uint32_t stepKb;
    uint16_t dbL0C;
};
#pragma pack(pop)

struct CaseConfig {
    uint32_t groupNum{0U};
    int32_t groupType{0};
    uint32_t groupListType{0U};
    uint32_t singleX{1U};
    uint32_t singleWeight{1U};
    uint32_t singleY{1U};
    std::string weightFormat;
    std::string dtype;
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    uint32_t isBias{0U};
    uint32_t hasGroupList{0U};
    uint32_t baseM{16U};
    uint32_t baseN{64U};
    uint32_t baseK{64U};
    uint32_t coreNum{2U};
    std::string dataDir;
    std::string outputPath;
    size_t elemSize{2U};
    uint32_t c0Size{16U};
    std::vector<int64_t> groupList;  // parsed from group_list.bin when present
    std::vector<int64_t> splitSizes; // per-group split-axis sizes (M for groupType 0, K for groupType 2)
    std::vector<int64_t> xMs;        // per-tensor m of multi-x inputs, inferred from file sizes
};

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0U ? value : (value + alignment - 1U) / alignment * alignment;
}

uint64_t NzWeightStorage(uint64_t kLen, uint64_t nLen, const CaseConfig& config)
{
    return AlignUp(nLen, config.c0Size) * AlignUp(kLen, NZ_K_INNER);
}

std::vector<int64_t> ParseSemicolonInts(const std::string& text)
{
    std::vector<int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (!item.empty()) {
            values.push_back(std::stoll(item));
        }
    }
    return values;
}

// Resolve per-group split sizes the same way the kernel does.
std::vector<int64_t> ResolveSplitSizes(const CaseConfig& config)
{
    const auto& values = config.groupList;
    const auto groupNum = static_cast<size_t>(config.groupNum);
    if (config.groupListType == 0U) { // cumsum
        std::vector<int64_t> sizes;
        int64_t previous = 0;
        for (size_t index = 0U; index < groupNum; ++index) {
            sizes.push_back(values[index] - previous);
            previous = values[index];
        }
        return sizes;
    }
    if (config.groupListType == 1U) { // count
        return std::vector<int64_t>(values.begin(), values.begin() + static_cast<int64_t>(groupNum));
    }
    std::vector<int64_t> sizes; // sparse: second column of the [E, 2] entries
    for (size_t index = 0U; index < groupNum; ++index) {
        sizes.push_back(values[2U * index + 1U]);
    }
    return sizes;
}

void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " <groupNum> <groupType> <groupListType> <singleX> <singleWeight> <singleY> <weightFormat>"
                 " <dtype> <m> <n> <k> <isBias> <hasGroupList> <baseM> <baseN> <baseK> <coreNum> <dataDir>"
                 " <outputPath>"
              << std::endl;
}

CaseConfig ParseConfig(char** argv)
{
    CaseConfig config{};
    config.groupNum = static_cast<uint32_t>(std::stoul(argv[1]));
    config.groupType = static_cast<int32_t>(std::stol(argv[2]));
    config.groupListType = static_cast<uint32_t>(std::stoul(argv[3]));
    config.singleX = static_cast<uint32_t>(std::stoul(argv[4]));
    config.singleWeight = static_cast<uint32_t>(std::stoul(argv[5]));
    config.singleY = static_cast<uint32_t>(std::stoul(argv[6]));
    config.weightFormat = argv[7];
    config.dtype = argv[8];
    config.m = std::stoll(argv[9]);
    config.n = std::stoll(argv[10]);
    config.k = std::stoll(argv[11]);
    config.isBias = static_cast<uint32_t>(std::stoul(argv[12]));
    config.hasGroupList = static_cast<uint32_t>(std::stoul(argv[13]));
    config.baseM = static_cast<uint32_t>(std::stoul(argv[14]));
    config.baseN = static_cast<uint32_t>(std::stoul(argv[15]));
    config.baseK = static_cast<uint32_t>(std::stoul(argv[16]));
    config.coreNum = static_cast<uint32_t>(std::stoul(argv[17]));
    config.dataDir = argv[18];
    config.outputPath = argv[19];
    config.elemSize = config.dtype == "float32" ? 4U : 2U;
    config.c0Size = static_cast<uint32_t>(BLOCK_BYTES / config.elemSize);
    return config;
}

bool IsValidConfig(const CaseConfig& config)
{
    const bool validTypes = config.dtype == "float16" || config.dtype == "bfloat16" || config.dtype == "float32";
    const bool validFormat = config.weightFormat == "nd" || config.weightFormat == "nz";
    const bool validShape = config.groupNum > 0U && config.m > 0 && config.n > 0 && config.k > 0 && config.baseM > 0U &&
                            config.baseN > 0U && config.baseK > 0U && config.coreNum > 0U;
    if (!validTypes || !validFormat || !validShape) {
        return false;
    }
    if (config.groupListType == 2U && config.groupType != 0) {
        return false; // sparse groupList is M-axis only
    }
    if (config.groupType == 2 && (config.isBias != 0U || config.groupListType == 2U || config.weightFormat == "nz")) {
        return false; // K-axis grouping: no bias, dense groupList only, ND weight only
    }
    if (config.groupType == -1 &&
        (config.hasGroupList != 0U || config.singleX != 0U || config.singleWeight != 0U || config.singleY != 0U)) {
        return false; // no-split: m-m-m without groupList
    }
    if (config.groupType == 0) {
        if (config.singleY != 1U) {
            return false; // M-axis grouping keeps a single output tensor
        }
        const bool validPair = (config.singleX == 1U && config.singleWeight == 1U) ||
                               (config.singleX == 1U && config.singleWeight == 0U) ||
                               (config.singleX == 0U && config.singleWeight == 0U);
        if (!validPair) {
            return false; // s-s-s / s-m-s / m-m-s only
        }
    }
    if (config.groupType == 2 && (config.singleX != 1U || config.singleWeight != config.singleY)) {
        return false; // K-axis grouping: single transposed x with s-s-s or s-m-m
    }
    if (config.weightFormat == "nz" && (config.k % NZ_K_INNER != 0 || config.n % config.c0Size != 0)) {
        return false; // NZ weight frame constraints
    }
    if (config.hasGroupList != 0U) {
        const size_t expect = config.groupListType == 2U ? 2U * config.groupNum : config.groupNum;
        if (config.groupList.size() != expect) {
            return false;
        }
        if (config.groupListType == 0U) {
            for (size_t index = 1U; index < config.groupList.size(); ++index) {
                if (config.groupList[index] < config.groupList[index - 1U]) {
                    return false; // cumsum must be non-decreasing
                }
            }
        }
        if (config.groupType == 2 &&
            std::accumulate(config.splitSizes.begin(), config.splitSizes.end(), int64_t{0}) != config.k) {
            return false; // K-axis groupList must partition k
        }
        if (config.groupType == 0 && config.singleX == 1U &&
            std::accumulate(config.splitSizes.begin(), config.splitSizes.end(), int64_t{0}) != config.m) {
            return false; // M-axis groupList must partition m for single-x scenarios
        }
        if (config.groupListType == 2U) {
            bool seenEmpty = false;
            for (int64_t size : config.splitSizes) {
                if (size < 0) {
                    return false;
                }
                if (size == 0) {
                    seenEmpty = true;
                } else if (seenEmpty) {
                    return false; // sparse groupList must front-load non-zero groups
                }
            }
            for (size_t index = 0U; index < config.groupNum; ++index) {
                if (config.groupList[2U * index] < 0 || config.groupList[2U * index] >= config.groupNum) {
                    return false; // sparse actual group index out of range
                }
            }
        }
    }
    if (config.singleX == 0U) {
        if (config.xMs.size() != config.groupNum ||
            std::accumulate(config.xMs.begin(), config.xMs.end(), int64_t{0}) != config.m) {
            return false; // multi-x inputs must describe one m per group summing to m
        }
    }
    return true;
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

void WriteBinary(const std::string& path, const uint8_t* data, size_t size)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
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

// Build a ListTensorDesc-compatible tensor list: [dataPtrOffset, (dim | count<<32), shape descs,
// tensor pointers]. Every entry carries a shape descriptor so the kernel can resolve per-group
// shapes from the list.
std::unique_ptr<DeviceBuffer> MakeTensorList(const std::vector<std::vector<uint64_t>>& shapes,
                                             const std::vector<const DeviceBuffer*>& tensors)
{
    const auto count = shapes.size();
    if (count == 0U || tensors.size() != count) {
        throw std::runtime_error("tensor list shape/pointer count mismatch");
    }
    const auto dim = shapes.front().size();
    const uint64_t descStructSize = 1U + dim;
    const uint64_t dataPtrOffset = sizeof(uint64_t) + count * descStructSize * sizeof(uint64_t);
    std::vector<uint64_t> words(1U + count * descStructSize + count, 0U);
    words[0] = dataPtrOffset;
    for (size_t index = 0U; index < count; ++index) {
        if (shapes[index].size() != dim) {
            throw std::runtime_error("tensor list descriptor dims differ");
        }
        uint64_t* desc = words.data() + 1U + index * descStructSize;
        desc[0] = index == 0U ? (dim | (static_cast<uint64_t>(count) << 32U)) : dim;
        for (size_t axis = 0U; axis < dim; ++axis) {
            desc[1U + axis] = shapes[index][axis];
        }
    }
    uint64_t* pointers = words.data() + 1U + count * descStructSize;
    for (size_t index = 0U; index < count; ++index) {
        pointers[index] = reinterpret_cast<uint64_t>(tensors[index]->Get());
    }
    auto tensorList = std::make_unique<DeviceBuffer>(words.size() * sizeof(uint64_t));
    tensorList->CopyFromHost(words.data(), words.size() * sizeof(uint64_t));
    return tensorList;
}

template <typename AType_, typename LayoutA_, typename LayoutB_>
__global__ __aicore__ void GmmNoQuantKernel(GM_ADDR xGm, GM_ADDR weightGm, GM_ADDR biasGm, GM_ADDR groupListGm,
                                            GM_ADDR yGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const auto* t = reinterpret_cast<__gm__ const GmmNoQuantTilingData*>(tilingGm);
    using BType = AType_;
    using CType = AType_;
    using BiasType = AType_;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelGroupedMmadNoQuant, 0,
                                                              Blaze::Gemm::MatmulOutputMode::OVERWRITE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType_, LayoutA_, BType, LayoutB_, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    const uint64_t baseM = static_cast<uint64_t>(t->baseM);
    const uint64_t baseN = static_cast<uint64_t>(t->baseN);
    const uint64_t baseK = static_cast<uint64_t>(t->baseK);
    const uint64_t sharedKStep = Blaze::Gemm::Min(static_cast<uint64_t>(t->stepKa), static_cast<uint64_t>(t->stepKb));
    const uint64_t kL1 = baseK * Blaze::Gemm::Max(sharedKStep, static_cast<uint64_t>(1));

    typename Kernel::GMMTiling gmmParams{t->groupNum,     t->groupType, t->groupListType, t->singleX,
                                         t->singleWeight, t->singleY,   t->hasBias,       t->weightNoL2Cache};
    constexpr uint32_t nTailAlign = BlockMmad::WEIGHT_NZ_FORMAT ?
                                        static_cast<uint32_t>(AscendC::Te::C0_ELEMENT<BType>) :
                                        1U;
    typename BlockScheduler::Params schedulerParams{static_cast<int32_t>(baseM),
                                                    static_cast<int32_t>(baseN),
                                                    t->mTailCnt,
                                                    t->nTailCnt,
                                                    1U, // mTailAlign
                                                    nTailAlign,
                                                    t->groupType,
                                                    t->groupNum,
                                                    t->m,
                                                    t->singleX == 1U,
                                                    t->singleWeight == 1U,
                                                    t->singleY == 1U,
                                                    BlockMmad::TRANS_B,
                                                    BlockMmad::WEIGHT_NZ_FORMAT,
                                                    static_cast<uint32_t>(sizeof(BType)),
                                                    t->groupListType};
    typename BlockMmad::Params mmParams{xGm,         weightGm, yGm, t->hasBias == 0U ? nullptr : biasGm,
                                        groupListGm,
                                        nullptr, // workspaceGmAddr
                                        baseM,       baseN,    kL1, t->baseM,
                                        t->baseN,    t->baseK,
                                        2U,                    // l1Stages
                                        t->dbL0C,    nullptr}; // scaleGmAddr
    // Copy the tiling fields into locals first: shape construction requires local-memory values,
    // while direct reads from the __gm__ tiling buffer keep the global address space qualifier.
    const int64_t problemM = t->m;
    const int64_t problemN = t->n;
    const int64_t problemK = t->k;
    ProblemShape problemShape{problemM, problemN, problemK, 1};
    typename Kernel::Params params{problemShape, mmParams, {}, schedulerParams, gmmParams};
    Kernel kernel;
    kernel(params);
}

struct DeviceInputs {
    std::unique_ptr<DeviceBuffer> xList;
    std::unique_ptr<DeviceBuffer> weightList;
    std::unique_ptr<DeviceBuffer> biasList;
    std::unique_ptr<DeviceBuffer> groupList;
    std::unique_ptr<DeviceBuffer> tiling;
    std::unique_ptr<DeviceBuffer> yList;
    std::vector<std::unique_ptr<DeviceBuffer>> xTensors;
    std::vector<std::unique_ptr<DeviceBuffer>> weightTensors;
    std::vector<std::unique_ptr<DeviceBuffer>> biasTensors;
    std::vector<std::unique_ptr<DeviceBuffer>> yTensors;
    size_t yTotalBytes{0U};
    std::vector<size_t> yTensorBytes;
};

void LoadGroupList(CaseConfig& config)
{
    if (config.hasGroupList == 0U) {
        return;
    }
    const size_t expectEntries = config.groupListType == 2U ? 2U * config.groupNum : config.groupNum;
    const auto data = ReadBinary(config.dataDir + "/group_list.bin", expectEntries * sizeof(int64_t));
    config.groupList.resize(expectEntries);
    std::copy(reinterpret_cast<const int64_t*>(data.data()),
              reinterpret_cast<const int64_t*>(data.data()) + expectEntries, config.groupList.begin());
    config.splitSizes = ResolveSplitSizes(config);
}

void InferMultiXShapes(CaseConfig& config)
{
    if (config.singleX != 0U) {
        return;
    }
    config.xMs.clear();
    const size_t xBytes = static_cast<size_t>(config.k) * config.elemSize;
    for (uint32_t index = 0U; index < config.groupNum; ++index) {
        const std::string path = config.dataDir + "/input_a_" + std::to_string(index) + ".bin";
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open()) {
            throw std::runtime_error("missing multi-x input file: " + path);
        }
        const auto size = static_cast<int64_t>(stream.tellg());
        if (size <= 0 || static_cast<uint64_t>(size) % xBytes != 0U) {
            throw std::runtime_error("multi-x input size not a multiple of k*elem: " + path);
        }
        config.xMs.push_back(size / static_cast<int64_t>(xBytes));
    }
}

DeviceInputs PrepareInputs(CaseConfig& config)
{
    const bool isNz = config.weightFormat == "nz";
    const bool isKSplit = config.groupType == 2;
    DeviceInputs device{};

    // x tensors: single [M, K] (or [K, M] under K-axis grouping); multi x per-tensor [m_i, K].
    if (config.singleX != 0U) {
        const size_t bytes = static_cast<size_t>(config.m) * config.k * config.elemSize;
        auto x = std::make_unique<DeviceBuffer>(bytes);
        x->CopyFromFile(config.dataDir + "/input_a.bin", bytes);
        device.xTensors.emplace_back(std::move(x));
        const std::vector<uint64_t> shape = isKSplit ? std::vector<uint64_t>{static_cast<uint64_t>(config.k),
                                                                             static_cast<uint64_t>(config.m)} :
                                                       std::vector<uint64_t>{static_cast<uint64_t>(config.m),
                                                                             static_cast<uint64_t>(config.k)};
        device.xList = MakeTensorList({shape}, {device.xTensors.front().get()});
    } else {
        std::vector<std::vector<uint64_t>> shapes;
        std::vector<const DeviceBuffer*> buffers;
        for (uint32_t index = 0U; index < config.groupNum; ++index) {
            const size_t bytes = static_cast<size_t>(config.xMs[index]) * config.k * config.elemSize;
            auto x = std::make_unique<DeviceBuffer>(bytes);
            x->CopyFromFile(config.dataDir + "/input_a_" + std::to_string(index) + ".bin", bytes);
            device.xTensors.emplace_back(std::move(x));
            shapes.push_back({static_cast<uint64_t>(config.xMs[index]), static_cast<uint64_t>(config.k)});
            buffers.push_back(device.xTensors.back().get());
        }
        device.xList = MakeTensorList(shapes, buffers);
    }

    // weight tensors: single M-axis [G, K, N] / K-axis [K, N]; multi per-group [K_g, N].
    // K-axis grouping only supports ND weight, so NZ storage below covers the M-axis cases.
    if (config.singleWeight != 0U) {
        size_t bytes = 0U;
        if (isNz) {
            // G matrices [K, N] framed and concatenated, one per group.
            bytes = static_cast<size_t>(NzWeightStorage(config.k, config.n, config)) * config.groupNum *
                    config.elemSize;
        } else if (isKSplit) {
            // One [K, N] matrix shared by all groups; each group reads its k-row segment.
            bytes = static_cast<size_t>(config.k) * config.n * config.elemSize;
        } else {
            // G matrices [K, N] concatenated, one per group.
            bytes = static_cast<size_t>(config.k) * config.n * config.groupNum * config.elemSize;
        }
        auto weight = std::make_unique<DeviceBuffer>(bytes);
        weight->CopyFromFile(config.dataDir + "/input_b.bin", bytes);
        device.weightTensors.emplace_back(std::move(weight));
        std::vector<uint64_t> shape;
        if (isNz) {
            shape = std::vector<uint64_t>{static_cast<uint64_t>(config.groupNum),
                                          AlignUp(config.n, config.c0Size) / config.c0Size,
                                          AlignUp(config.k, NZ_K_INNER) / NZ_K_INNER, NZ_K_INNER, config.c0Size};
        } else {
            shape = isKSplit ? std::vector<uint64_t>{static_cast<uint64_t>(config.k), static_cast<uint64_t>(config.n)} :
                               std::vector<uint64_t>{static_cast<uint64_t>(config.groupNum),
                                                     static_cast<uint64_t>(config.k), static_cast<uint64_t>(config.n)};
        }
        device.weightList = MakeTensorList({shape}, {device.weightTensors.front().get()});
    } else {
        std::vector<std::vector<uint64_t>> shapes;
        std::vector<const DeviceBuffer*> buffers;
        for (uint32_t index = 0U; index < config.groupNum; ++index) {
            const int64_t groupK = isKSplit ? config.splitSizes[index] : config.k;
            const size_t bytes = isNz ?
                                     static_cast<size_t>(NzWeightStorage(groupK, config.n, config)) * config.elemSize :
                                     static_cast<size_t>(groupK) * config.n * config.elemSize;
            auto weight = std::make_unique<DeviceBuffer>(bytes);
            weight->CopyFromFile(config.dataDir + "/input_b_" + std::to_string(index) + ".bin", bytes);
            device.weightTensors.emplace_back(std::move(weight));
            if (isNz) {
                shapes.push_back({AlignUp(config.n, config.c0Size) / config.c0Size,
                                  AlignUp(static_cast<uint64_t>(groupK), NZ_K_INNER) / NZ_K_INNER, NZ_K_INNER,
                                  config.c0Size});
            } else {
                shapes.push_back({static_cast<uint64_t>(groupK), static_cast<uint64_t>(config.n)});
            }
            buffers.push_back(device.weightTensors.back().get());
        }
        device.weightList = MakeTensorList(shapes, buffers);
    }

    // bias: single [G, N]; multi per-group [N].
    if (config.isBias != 0U) {
        if (config.singleWeight != 0U) {
            const size_t bytes = static_cast<size_t>(config.groupNum) * config.n * config.elemSize;
            auto bias = std::make_unique<DeviceBuffer>(bytes);
            bias->CopyFromFile(config.dataDir + "/bias.bin", bytes);
            device.biasTensors.emplace_back(std::move(bias));
            device.biasList = MakeTensorList(
                {{static_cast<uint64_t>(config.groupNum), static_cast<uint64_t>(config.n)}},
                {device.biasTensors.front().get()});
        } else {
            std::vector<std::vector<uint64_t>> shapes;
            std::vector<const DeviceBuffer*> buffers;
            for (uint32_t index = 0U; index < config.groupNum; ++index) {
                const size_t bytes = static_cast<size_t>(config.n) * config.elemSize;
                auto bias = std::make_unique<DeviceBuffer>(bytes);
                bias->CopyFromFile(config.dataDir + "/bias_" + std::to_string(index) + ".bin", bytes);
                device.biasTensors.emplace_back(std::move(bias));
                shapes.push_back({static_cast<uint64_t>(config.n)});
                buffers.push_back(device.biasTensors.back().get());
            }
            device.biasList = MakeTensorList(shapes, buffers);
        }
    }

    // y tensors: single [totalM, N] (M-axis) or [G, M, N] (K-axis); multi per-group [m_i, N] / [M, N].
    if (config.singleY != 0U) {
        const int64_t totalM = config.groupType == 2 ?
                                   config.m :
                                   (config.singleX != 0U ?
                                        config.m :
                                        std::accumulate(config.xMs.begin(), config.xMs.end(), int64_t{0}));
        const size_t bytes = config.groupType == 2 ?
                                 static_cast<size_t>(config.groupNum) * config.m * config.n * config.elemSize :
                                 static_cast<size_t>(totalM) * config.n * config.elemSize;
        auto y = std::make_unique<DeviceBuffer>(bytes);
        y->Clear();
        device.yTensors.emplace_back(std::move(y));
        device.yTotalBytes = bytes;
        const std::vector<uint64_t> shape = config.groupType == 2 ?
                                                std::vector<uint64_t>{static_cast<uint64_t>(config.groupNum),
                                                                      static_cast<uint64_t>(config.m),
                                                                      static_cast<uint64_t>(config.n)} :
                                                std::vector<uint64_t>{static_cast<uint64_t>(totalM),
                                                                      static_cast<uint64_t>(config.n)};
        device.yList = MakeTensorList({shape}, {device.yTensors.front().get()});
    } else {
        std::vector<std::vector<uint64_t>> shapes;
        std::vector<const DeviceBuffer*> buffers;
        for (uint32_t index = 0U; index < config.groupNum; ++index) {
            const int64_t yM = config.groupType == 2 ?
                                   config.m :
                                   (config.singleX != 0U ? config.splitSizes[index] : config.xMs[index]);
            const size_t bytes = static_cast<size_t>(yM) * config.n * config.elemSize;
            auto y = std::make_unique<DeviceBuffer>(bytes);
            y->Clear();
            device.yTensors.emplace_back(std::move(y));
            device.yTensorBytes.push_back(bytes);
            device.yTotalBytes += bytes;
            shapes.push_back({static_cast<uint64_t>(yM), static_cast<uint64_t>(config.n)});
            buffers.push_back(device.yTensors.back().get());
        }
        device.yList = MakeTensorList(shapes, buffers);
    }

    // groupList: real buffer when present, otherwise a dummy the kernel never reads (no-split).
    const size_t groupListEntries = config.groupListType == 2U ? 2U * config.groupNum : config.groupNum;
    device.groupList = std::make_unique<DeviceBuffer>(std::max<size_t>(groupListEntries, 1U) * sizeof(int64_t));
    if (config.hasGroupList != 0U) {
        device.groupList->CopyFromHost(config.groupList.data(), config.groupList.size() * sizeof(int64_t));
    } else {
        device.groupList->Clear();
    }

    GmmNoQuantTilingData tiling{};
    tiling.groupNum = config.groupNum;
    tiling.groupType = config.groupType;
    tiling.groupListType = config.groupListType;
    tiling.singleX = config.singleX;
    tiling.singleWeight = config.singleWeight;
    tiling.singleY = config.singleY;
    tiling.hasBias = config.isBias;
    tiling.weightNoL2Cache = 0U;
    tiling.mTailCnt = 1U;
    tiling.nTailCnt = 1U;
    tiling.m = config.groupType == 2 ?
                   config.m :
                   (config.singleX != 0U ? config.m :
                                           std::accumulate(config.xMs.begin(), config.xMs.end(), int64_t{0}));
    tiling.n = config.n;
    tiling.k = config.k;
    tiling.baseM = config.baseM;
    tiling.baseN = config.baseN;
    tiling.baseK = config.baseK;
    tiling.stepKa = 1U;
    tiling.stepKb = 1U;
    tiling.dbL0C = 1U;
    device.tiling = std::make_unique<DeviceBuffer>(sizeof(tiling));
    device.tiling->CopyFromHost(&tiling, sizeof(tiling));
    return device;
}

template <typename AType_, typename LayoutA_, typename LayoutB_>
void RunTypedCase(const CaseConfig& config, DeviceInputs& device)
{
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    GmmNoQuantKernel<AType_, LayoutA_, LayoutB_><<<config.coreNum, 0, stream>>>(
        device.xList->Get(), device.weightList->Get(), device.biasList == nullptr ? nullptr : device.biasList->Get(),
        device.groupList->Get(), device.yList->Get(), device.tiling->Get());
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtDestroyStream(stream));

    if (config.singleY != 0U) {
        const auto output = device.yTensors.front()->CopyToHost(device.yTotalBytes);
        WriteBinary(config.outputPath, output.data(), output.size());
        return;
    }
    std::vector<uint8_t> output;
    output.reserve(device.yTotalBytes);
    for (size_t index = 0U; index < device.yTensors.size(); ++index) {
        const auto chunk = device.yTensors[index]->CopyToHost(device.yTensorBytes[index]);
        output.insert(output.end(), chunk.begin(), chunk.end());
    }
    WriteBinary(config.outputPath, output.data(), output.size());
}

template <typename AType_>
void RunLayoutCase(const CaseConfig& config, DeviceInputs& device)
{
    if (config.groupType == 2) { // K-axis grouping: transposed x [K, M] with ND weight only
        RunTypedCase<AType_, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(config, device);
        return;
    }
    if (config.weightFormat == "nz") {
        RunTypedCase<AType_, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>(config, device);
    } else {
        RunTypedCase<AType_, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(config, device);
    }
}

void RunCase(CaseConfig& config)
{
    DeviceInputs device = PrepareInputs(config);
    if (config.dtype == "float16") {
        RunLayoutCase<half>(config, device);
    } else if (config.dtype == "bfloat16") {
        RunLayoutCase<bfloat16_t>(config, device);
    } else {
        RunLayoutCase<float>(config, device);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 20) {
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        CaseConfig config = ParseConfig(argv);
        LoadGroupList(config);
        InferMultiXShapes(config);
        if (!IsValidConfig(config)) {
            std::cerr << "invalid no-quant grouped matmul case configuration" << std::endl;
            return 2;
        }
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        RunCase(config);
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::cout << "No-quant grouped matmul kernel completed, groupType=" << config.groupType
                  << ", groupListType=" << config.groupListType << ", weightFormat=" << config.weightFormat
                  << ", dtype=" << config.dtype << ", result=" << config.outputPath << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "No-quant grouped matmul example failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
