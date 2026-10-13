/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * @file quant_grouped_matmul_mx.cpp
 * @brief Quant Grouped MatMul MX example covering MXFP4/MXFP8 and ND/NZ weight.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

using ScaleType = fp8_e8m0_t;
using NdLayout = asc::te::nd_ext_layout_ptn;

template <typename T>
inline constexpr bool IS_FP4_TYPE = std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;

template <typename AType, typename BType, typename CType, typename LayoutA, typename LayoutB, uint64_t FullLoadMode>
// Keep 64-bit arguments before 32-bit arguments so host launch stubs and device entry points agree on alignment.
// Pass small flags in 32-bit slots, then narrow them when populating the internal tiling structure.
__global__ __aicore__ void qgmm_mx_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR c, GM_ADDR bias,
                                          GM_ADDR groupList, int64_t m, int64_t n, int64_t k, uint32_t groupNum,
                                          uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t kAL1, uint32_t kBL1,
                                          uint32_t scaleKAL1, uint32_t scaleKBL1, uint32_t isBias, uint32_t dbL0C,
                                          uint32_t l1BufferStage, int32_t groupType, uint32_t groupListType,
                                          uint32_t singleW)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<FullLoadMode>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, CType, NdLayout, float,
                                               NdLayout>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    typename Kernel::Params p{};
    p.problemShape = {m, n, k, 0};
    p.mmadParams.aGmAddr = a;
    p.mmadParams.bGmAddr = b;
    p.mmadParams.cGmAddr = c;
    p.mmadParams.scaleAGmAddr = scaleA;
    p.mmadParams.scaleBGmAddr = scaleB;
    p.mmadParams.biasGmAddr = bias;
    p.groupListGmAddr = groupList;
    p.gmmParams = {groupNum,
                   m,
                   n,
                   k,
                   baseM,
                   baseN,
                   baseK,
                   kAL1,
                   kBL1,
                   scaleKAL1,
                   scaleKBL1,
                   static_cast<uint8_t>(isBias),
                   static_cast<uint8_t>(dbL0C),
                   static_cast<uint8_t>(l1BufferStage),
                   static_cast<int8_t>(groupType),
                   static_cast<uint8_t>(groupListType),
                   static_cast<uint8_t>(singleW)};
    Kernel kernel;
    kernel(p);
}

template <typename T>
constexpr size_t MxBytes(size_t count)
{
    return IS_FP4_TYPE<T> ? (count + 1U) / 2U : count * sizeof(T);
}

struct CaseBytes {
    size_t a;
    size_t bGroup;
    size_t scaleA;
    size_t scaleBGroup;
    size_t c;
};

struct DeviceBuffers {
    uint8_t *a = nullptr, *b = nullptr, *scaleA = nullptr, *scaleB = nullptr, *c = nullptr, *bias = nullptr;
    uint8_t* groupList = nullptr;
    std::vector<uint8_t*> weights;
    std::vector<uint8_t*> scales;

    explicit DeviceBuffers(uint32_t groupNum) : weights(groupNum, nullptr), scales(groupNum, nullptr) {}

    ~DeviceBuffers()
    {
        if (groupList != nullptr)
            aclrtFree(groupList);
        if (c != nullptr)
            aclrtFree(c);
        if (bias != nullptr)
            aclrtFree(bias);
        if (scaleB != nullptr)
            aclrtFree(scaleB);
        if (b != nullptr)
            aclrtFree(b);
        if (scaleA != nullptr)
            aclrtFree(scaleA);
        if (a != nullptr)
            aclrtFree(a);
        for (size_t i = 0; i < weights.size(); ++i) {
            if (scales[i] != nullptr)
                aclrtFree(scales[i]);
            if (weights[i] != nullptr)
                aclrtFree(weights[i]);
        }
    }
};

struct QgmmTilingData {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
};

void AllocateCommonBuffers(DeviceBuffers& device, const CaseBytes& bytes, size_t biasBytes, size_t groupListBytes)
{
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.a), bytes.a, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleA), bytes.scaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.c), bytes.c, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.bias), biasBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.groupList), groupListBytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

std::vector<uint8_t> ReadBinary(const std::string& path, size_t size)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open() || static_cast<size_t>(stream.tellg()) != size) {
        throw std::runtime_error("invalid input file size: " + path);
    }
    std::vector<uint8_t> data(size);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return data;
}

template <bool MultiTensor>
void PrepareWeightBuffers(DeviceBuffers& device, const CaseBytes& bytes, uint32_t groupNum, bool kGrouped,
                          const std::string& dataDir)
{
    if constexpr (!MultiTensor) {
        const size_t tensorCount = kGrouped ? 1U : groupNum;
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void**>(&device.b), tensorCount * bytes.bGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleB), tensorCount * bytes.scaleBGroup,
                              ACL_MEM_MALLOC_HUGE_FIRST));
        const auto allWeights = ReadBinary(dataDir + "/input_b.bin", tensorCount * bytes.bGroup);
        const auto allScales = ReadBinary(dataDir + "/scale_b.bin", tensorCount * bytes.scaleBGroup);
        ACL_CHECK(
            aclrtMemcpy(device.b, allWeights.size(), allWeights.data(), allWeights.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(device.scaleB, allScales.size(), allScales.data(), allScales.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        return;
    }
    std::vector<uint64_t> weightList(groupNum + 1U, sizeof(uint64_t));
    std::vector<uint64_t> scaleList(groupNum + 1U, sizeof(uint64_t));
    for (uint32_t i = 0; i < groupNum; ++i) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.weights[i]), bytes.bGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void**>(&device.scales[i]), bytes.scaleBGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        const auto values = ReadBinary(dataDir + "/input_b_" + std::to_string(i) + ".bin", bytes.bGroup);
        const auto scales = ReadBinary(dataDir + "/scale_b_" + std::to_string(i) + ".bin", bytes.scaleBGroup);
        ACL_CHECK(aclrtMemcpy(device.weights[i], bytes.bGroup, values.data(), bytes.bGroup, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(device.scales[i], bytes.scaleBGroup, scales.data(), bytes.scaleBGroup,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        weightList[i + 1U] = reinterpret_cast<uint64_t>(device.weights[i]);
        scaleList[i + 1U] = reinterpret_cast<uint64_t>(device.scales[i]);
    }
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.b), weightList.size() * sizeof(uint64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleB), scaleList.size() * sizeof(uint64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device.b, weightList.size() * sizeof(uint64_t), weightList.data(),
                          weightList.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.scaleB, scaleList.size() * sizeof(uint64_t), scaleList.data(),
                          scaleList.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
}

void CopyCommonInputs(DeviceBuffers& device, const CaseBytes& bytes, size_t biasBytes, size_t groupListBytes,
                      const std::string& dataDir)
{
    const auto values = ReadBinary(dataDir + "/input_a.bin", bytes.a);
    const auto scales = ReadBinary(dataDir + "/scale_a.bin", bytes.scaleA);
    const auto bias = ReadBinary(dataDir + "/bias.bin", biasBytes);
    const auto groupList = ReadBinary(dataDir + "/group_list.bin", groupListBytes);
    ACL_CHECK(aclrtMemcpy(device.a, bytes.a, values.data(), bytes.a, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.scaleA, bytes.scaleA, scales.data(), bytes.scaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.bias, biasBytes, bias.data(), biasBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(
        aclrtMemcpy(device.groupList, groupListBytes, groupList.data(), groupListBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device.c, bytes.c, 0, bytes.c));
}

void WriteOutput(const std::string& outputPath, const std::vector<uint8_t>& output)
{
    std::ofstream stream(outputPath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output file: " + outputPath);
    }
    stream.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
    if (!stream) {
        throw std::runtime_error("failed to write output file: " + outputPath);
    }
}

template <typename AType, typename BType, typename CType, typename LayoutA, typename LayoutB, bool MultiTensor,
          uint64_t FullLoadMode>
void LaunchKernel(DeviceBuffers& device, const CaseBytes& bytes, const QgmmTilingData& tiling,
                  const std::string& outputPath)
{
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    qgmm_mx_kernel<AType, BType, CType, LayoutA, LayoutB, FullLoadMode>
        <<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream>>>(
            device.a, device.b, device.scaleA, device.scaleB, device.c, device.bias, device.groupList, tiling.m,
            tiling.n, tiling.k, tiling.groupNum, tiling.baseM, tiling.baseN, tiling.baseK, tiling.kAL1, tiling.kBL1,
            tiling.scaleKAL1, tiling.scaleKBL1, static_cast<uint32_t>(tiling.isBias),
            static_cast<uint32_t>(tiling.dbL0C), static_cast<uint32_t>(tiling.l1BufferStage),
            static_cast<int32_t>(tiling.groupType), static_cast<uint32_t>(tiling.groupListType),
            static_cast<uint32_t>(tiling.singleW));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint8_t> output(bytes.c);
    ACL_CHECK(aclrtMemcpy(output.data(), bytes.c, device.c, bytes.c, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtDestroyStream(stream));
    WriteOutput(outputPath, output);
}

std::vector<int64_t> MakeGroupList(uint32_t groupNum, int64_t splitValue, uint8_t groupListType)
{
    std::vector<int64_t> groupList;
    groupList.reserve(groupListType == 2 ? groupNum * 2U : groupNum);
    for (uint32_t i = 0; i < groupNum; ++i) {
        if (groupListType == 2) {
            groupList.push_back(static_cast<int64_t>(i));
        }
        groupList.push_back(groupListType == 0 ? static_cast<int64_t>(i + 1U) * splitValue : splitValue);
    }
    return groupList;
}

template <typename AType, typename BType, typename CType, typename LayoutA, typename LayoutB, bool MultiTensor,
          uint64_t FullLoadMode>
int RunCase(const QgmmTilingData& tiling, const std::string& dataDir, const std::string& outputPath)
{
    const bool kGrouped = tiling.groupType == 2;
    constexpr bool weightNz = std::is_same_v<LayoutB, asc::te::nz_layout_ptn> ||
                              std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    constexpr bool transB = std::is_same_v<LayoutB, asc::te::dn_ext_layout_ptn> ||
                            std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    const size_t c0 = IS_FP4_TYPE<BType> ? 64U : 32U;
    const size_t storedK = weightNz ? (transB ? ((tiling.k + c0 - 1U) / c0 * c0) : ((tiling.k + 15U) / 16U * 16U)) :
                                      tiling.k;
    const size_t storedN = weightNz ? (transB ? ((tiling.n + 15U) / 16U * 16U) : ((tiling.n + c0 - 1U) / c0 * c0)) :
                                      tiling.n;
    const size_t scaleK = static_cast<size_t>((tiling.k + 63) / 64) * 2U;
    const size_t aElements = kGrouped ? static_cast<size_t>(tiling.m * tiling.k) :
                                        static_cast<size_t>(tiling.groupNum) * tiling.m * tiling.k;
    const size_t bElements = kGrouped ? static_cast<size_t>(tiling.k * tiling.n) : storedK * storedN;
    const size_t scaleAElements = kGrouped ? (static_cast<size_t>(tiling.k / 64) + tiling.groupNum) * tiling.m * 2U :
                                             static_cast<size_t>(tiling.groupNum) * tiling.m * scaleK;
    const size_t scaleBElements = kGrouped ? (static_cast<size_t>(tiling.k / 64) + tiling.groupNum) * tiling.n * 2U :
                                             static_cast<size_t>(tiling.n) * scaleK;
    const CaseBytes bytes = {MxBytes<AType>(aElements), MxBytes<BType>(bElements), scaleAElements * sizeof(ScaleType),
                             scaleBElements * sizeof(ScaleType),
                             static_cast<size_t>(tiling.groupNum) * tiling.m * tiling.n * sizeof(CType)};
    const std::vector<int64_t> groupList = MakeGroupList(tiling.groupNum, kGrouped ? tiling.k : tiling.m,
                                                         tiling.groupListType);
    DeviceBuffers device(tiling.groupNum);
    AllocateCommonBuffers(device, bytes, static_cast<size_t>(tiling.groupNum) * tiling.n * sizeof(float),
                          groupList.size() * sizeof(int64_t));
    PrepareWeightBuffers<MultiTensor>(device, bytes, tiling.groupNum, kGrouped, dataDir);
    CopyCommonInputs(device, bytes, static_cast<size_t>(tiling.groupNum) * tiling.n * sizeof(float),
                     groupList.size() * sizeof(int64_t), dataDir);
    LaunchKernel<AType, BType, CType, LayoutA, LayoutB, MultiTensor, FullLoadMode>(device, bytes, tiling, outputPath);
    return 0;
}

template <typename T, typename CType, typename LayoutA, typename LayoutB, uint64_t FullLoadMode>
int DispatchSingleW(const QgmmTilingData& tiling, const std::string& dataDir, const std::string& outputPath)
{
    return tiling.singleW == 0 ?
               RunCase<T, T, CType, LayoutA, LayoutB, true, FullLoadMode>(tiling, dataDir, outputPath) :
               RunCase<T, T, CType, LayoutA, LayoutB, false, FullLoadMode>(tiling, dataDir, outputPath);
}

template <typename T, typename CType, typename LayoutA, uint64_t FullLoadMode>
int DispatchLayoutB(const std::string& layoutB, const QgmmTilingData& tiling, const std::string& dataDir,
                    const std::string& outputPath)
{
    if (layoutB == "nd")
        return DispatchSingleW<T, CType, LayoutA, NdLayout, FullLoadMode>(tiling, dataDir, outputPath);
    if (layoutB == "dn")
        return DispatchSingleW<T, CType, LayoutA, asc::te::dn_ext_layout_ptn, FullLoadMode>(tiling, dataDir,
                                                                                            outputPath);
    if (layoutB == "nz")
        return DispatchSingleW<T, CType, LayoutA, asc::te::nz_layout_ptn, FullLoadMode>(tiling, dataDir, outputPath);
    if (layoutB == "zn")
        return DispatchSingleW<T, CType, LayoutA, asc::te::zn_layout_ptn, FullLoadMode>(tiling, dataDir, outputPath);
    return 2;
}

template <typename T, typename CType, uint64_t FullLoadMode>
int DispatchLayoutA(const std::string& layoutA, const std::string& layoutB, const QgmmTilingData& tiling,
                    const std::string& dataDir, const std::string& outputPath)
{
    return layoutA == "dn" ? DispatchLayoutB<T, CType, asc::te::dn_ext_layout_ptn, FullLoadMode>(layoutB, tiling,
                                                                                                 dataDir, outputPath) :
                             DispatchLayoutB<T, CType, NdLayout, FullLoadMode>(layoutB, tiling, dataDir, outputPath);
}

struct QgmmCaseConfig {
    QgmmTilingData tiling;
    std::string dtype;
    std::string outputDtype;
    std::string layoutA;
    std::string layoutB;
    bool aFullLoad;
    std::string dataDir;
    std::string outputPath;
};

void PrintUsage()
{
    std::cerr << "Usage: quant_grouped_matmul_mx <groupNum> <m> <n> <k> <baseM> <baseN> <baseK> "
                 "<kAL1> <kBL1> <scaleKAL1> <scaleKBL1> <isBias> <dbL0C> <l1BufferStage> <groupType> "
                 "<groupListType> <singleW> <mxfp4_e2m1|mxfp4_e1m2|mxfp8_e4m3|mxfp8_e5m2> "
                 "<output dtype:float16|bfloat16|float32> "
                 "<layoutA:nd|dn> <weight layoutB:nd|dn|nz|zn> <aFullLoad> "
                 "<dataDir> <outputPath>"
              << std::endl;
}

bool ParseFlagArg(const char* arg, const std::string& name)
{
    const auto value = std::stoul(arg);
    if (value > 1U) {
        throw std::invalid_argument(name + " must be 0 or 1");
    }
    return value == 1U;
}

QgmmCaseConfig ParseConfig(int argc, char** argv)
{
    if (argc != 25) {
        PrintUsage();
        throw std::invalid_argument("argument count must be 24");
    }
    QgmmCaseConfig config{};
    config.tiling.groupNum = static_cast<uint32_t>(std::stoul(argv[1]));
    config.tiling.m = std::stoll(argv[2]);
    config.tiling.n = std::stoll(argv[3]);
    config.tiling.k = std::stoll(argv[4]);
    config.tiling.baseM = static_cast<uint32_t>(std::stoul(argv[5]));
    config.tiling.baseN = static_cast<uint32_t>(std::stoul(argv[6]));
    config.tiling.baseK = static_cast<uint32_t>(std::stoul(argv[7]));
    config.tiling.kAL1 = static_cast<uint32_t>(std::stoul(argv[8]));
    config.tiling.kBL1 = static_cast<uint32_t>(std::stoul(argv[9]));
    config.tiling.scaleKAL1 = static_cast<uint32_t>(std::stoul(argv[10]));
    config.tiling.scaleKBL1 = static_cast<uint32_t>(std::stoul(argv[11]));
    config.tiling.isBias = static_cast<uint8_t>(std::stoul(argv[12]));
    config.tiling.dbL0C = static_cast<uint8_t>(std::stoul(argv[13]));
    config.tiling.l1BufferStage = static_cast<uint8_t>(std::stoul(argv[14]));
    config.tiling.groupType = static_cast<int8_t>(std::stoi(argv[15]));
    config.tiling.groupListType = static_cast<uint8_t>(std::stoul(argv[16]));
    config.tiling.singleW = static_cast<uint8_t>(std::stoul(argv[17]));
    config.dtype = argv[18];
    config.outputDtype = argv[19];
    config.layoutA = argv[20];
    config.layoutB = argv[21];
    config.aFullLoad = ParseFlagArg(argv[22], "aFullLoad");
    config.dataDir = argv[23];
    config.outputPath = argv[24];
    return config;
}

bool IsWeightNz(const std::string& layoutB) { return layoutB == "nz" || layoutB == "zn"; }

bool IsFp4Type(const std::string& dtype) { return dtype == "mxfp4_e2m1" || dtype == "mxfp4_e1m2"; }

bool CheckBasicConfig(const QgmmCaseConfig& config, std::string& reason)
{
    if (config.tiling.groupNum == 0 || config.tiling.m <= 0 || config.tiling.n <= 0 || config.tiling.k <= 0) {
        reason = "groupNum, m, n and k must be positive";
        return false;
    }
    if (config.tiling.baseM == 0 || config.tiling.baseN == 0 || config.tiling.baseK == 0 || config.tiling.kAL1 == 0 ||
        config.tiling.kBL1 == 0 || config.tiling.scaleKAL1 == 0 || config.tiling.scaleKBL1 == 0) {
        reason = "tiling block parameters must be positive";
        return false;
    }
    if (config.tiling.isBias > 1 || config.tiling.singleW > 1) {
        reason = "isBias and singleW must be 0 or 1";
        return false;
    }
    if (config.tiling.dbL0C < 1 || config.tiling.dbL0C > 2) {
        reason = "dbL0C must be 1 or 2";
        return false;
    }
    if (config.tiling.l1BufferStage != 2 && config.tiling.l1BufferStage != 3) {
        reason = "l1BufferStage must be 2 or 3";
        return false;
    }
    if (config.tiling.groupType != 0 && config.tiling.groupType != 2) {
        reason = "groupType must be 0 or 2";
        return false;
    }
    if (config.tiling.groupListType > 2) {
        reason = "groupListType must be 0, 1 or 2";
        return false;
    }
    return true;
}

bool CheckLayoutConfig(const QgmmCaseConfig& config, std::string& reason)
{
    if ((config.layoutA != "nd" && config.layoutA != "dn") ||
        (config.layoutB != "nd" && config.layoutB != "dn" && config.layoutB != "nz" && config.layoutB != "zn")) {
        reason = "layoutA must be nd or dn, weight layoutB must be nd, dn, nz or zn";
        return false;
    }
    if ((config.layoutA == "dn") != (config.tiling.groupType == 2)) {
        reason = "current example uses layoutA=dn for K-axis grouping and layoutA=nd for M-axis grouping";
        return false;
    }
    if (IsWeightNz(config.layoutB) && (config.tiling.k == 1 || config.tiling.n == 1)) {
        reason = "weight NZ/ZN requires k and n to be greater than 1";
        return false;
    }
    return true;
}

bool CheckDtypeConfig(const QgmmCaseConfig& config, std::string& reason)
{
    const bool validDtype = config.dtype == "mxfp8_e4m3" || config.dtype == "mxfp8_e5m2" ||
                            config.dtype == "mxfp4_e2m1" || config.dtype == "mxfp4_e1m2";
    if (!validDtype) {
        reason = "dtype must be mxfp8_e4m3, mxfp8_e5m2, mxfp4_e2m1 or mxfp4_e1m2";
        return false;
    }
    if (config.outputDtype != "float16" && config.outputDtype != "bfloat16" && config.outputDtype != "float32") {
        reason = "output dtype must be float16, bfloat16 or float32";
        return false;
    }
    if (config.dtype == "mxfp4_e1m2" && !IsWeightNz(config.layoutB)) {
        reason = "mxfp4_e1m2 is supported only with weight NZ/ZN in this example";
        return false;
    }
    if (IsFp4Type(config.dtype)) {
        if (config.tiling.k % 2 != 0 || config.tiling.k == 2) {
            reason = "mxfp4 requires k to be even and not equal to 2";
            return false;
        }
        if ((config.layoutB == "nd" || config.layoutB == "nz") && config.tiling.n % 2 != 0) {
            reason = "mxfp4 with non-transposed weight ND/NZ requires n to be even";
            return false;
        }
    }
    return true;
}

bool CheckKGroupedConfig(const QgmmCaseConfig& config, std::string& reason)
{
    if (config.tiling.groupType != 2) {
        return true;
    }
    if (IsFp4Type(config.dtype) || config.layoutB != "nd" || config.tiling.singleW != 1 || config.tiling.isBias != 0 ||
        config.tiling.groupListType == 2) {
        reason = "K-axis grouping supports MXFP8, single weight ND, no bias and Length/Offset Group List";
        return false;
    }
    return true;
}

bool IsValidConfig(const QgmmCaseConfig& config, std::string& reason)
{
    return CheckBasicConfig(config, reason) && CheckLayoutConfig(config, reason) && CheckDtypeConfig(config, reason) &&
           CheckKGroupedConfig(config, reason);
}

template <typename T, typename CType, uint64_t FullLoadMode>
int DispatchConfig(const QgmmCaseConfig& config)
{
    return DispatchLayoutA<T, CType, FullLoadMode>(config.layoutA, config.layoutB, config.tiling, config.dataDir,
                                                   config.outputPath);
}

template <typename T, typename CType>
int DispatchFullLoad(const QgmmCaseConfig& config)
{
    return config.aFullLoad ? DispatchConfig<T, CType, 1>(config) : DispatchConfig<T, CType, 0>(config);
}

template <typename T>
int DispatchOutputDtype(const QgmmCaseConfig& config)
{
    if (config.outputDtype == "float32")
        return DispatchFullLoad<T, float>(config);
    if (config.outputDtype == "float16")
        return DispatchFullLoad<T, half>(config);
    if (config.outputDtype == "bfloat16")
        return DispatchFullLoad<T, bfloat16_t>(config);
    return 2;
}

int RunConfiguredCase(const QgmmCaseConfig& config)
{
    if (config.dtype == "mxfp8_e4m3")
        return DispatchOutputDtype<fp8_e4m3fn_t>(config);
    if (config.dtype == "mxfp8_e5m2")
        return DispatchOutputDtype<fp8_e5m2_t>(config);
    if (config.dtype == "mxfp4_e2m1")
        return DispatchOutputDtype<fp4x2_e2m1_t>(config);
    if (config.dtype == "mxfp4_e1m2")
        return DispatchOutputDtype<fp4x2_e1m2_t>(config);
    return 2;
}

void PrintResult(int ret, const QgmmCaseConfig& config)
{
    if (ret == 0) {
        std::cout << "QGMM MX kernel execution completed, output=" << config.outputPath << std::endl;
    } else if (ret == 1) {
        std::cerr << "QGMM MX example FAILED: kernel execution failed" << std::endl;
    } else {
        std::cerr << "QGMM MX example FAILED: unsupported argument combination, dtype=" << config.dtype
                  << ", outputDtype=" << config.outputDtype << ", layoutA=" << config.layoutA
                  << ", weight layoutB=" << config.layoutB << std::endl;
    }
}

int main(int argc, char** argv)
{
    QgmmCaseConfig config{};
    try {
        config = ParseConfig(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "invalid QGMM MX case argument: " << error.what() << std::endl;
        return 2;
    }
    std::string reason;
    if (!IsValidConfig(config, reason)) {
        std::cerr << "invalid QGMM MX case configuration: " << reason << std::endl;
        return 2;
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    const int ret = RunConfiguredCase(config);
    aclrtResetDevice(0);
    aclFinalize();
    PrintResult(ret, config);
    return ret;
}
