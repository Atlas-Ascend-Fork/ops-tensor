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
 * @file quant_grouped_matmul_cubeonly.cpp
 * @brief QGMM Cube (kernel_qgmm_cube.h) NPU example for INT8/FP8 quant grouped matmul.
 *
 * This example is intentionally kept simple, following the style of
 * quant_grouped_matmul_mx: it reads a CSV row through run_case.py, prepares
 * input files, launches the Blaze cube kernel and writes npu_out.bin for
 * verification.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "lib/matmul_intf.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"

using NdLayout = asc::te::nd_ext_layout_ptn;

namespace {

constexpr uint32_t QM_DEFAULT = 0U;
constexpr uint32_t QM_PERTENSOR = 1U;
constexpr uint32_t QM_PERCHANNEL = 2U;
constexpr uint32_t GMM_ARRAY_LEN = 128U;
constexpr uint64_t TENSOR_LIST_HEADER_COUNT_SHIFT = 32UL;

struct CubeConfig {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    uint8_t isBias;
    uint8_t dbL0C;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
    std::string aType;
    std::string bType;
    std::string cType;
    std::string biasType;
    std::string x2ScaleType;
    std::string layoutA;
    std::string layoutB;
    std::string dataDir;
    std::string outputPath;
};

struct DeviceBuffers {
    uint8_t* aData = nullptr;
    uint8_t* bData = nullptr;
    uint8_t* cData = nullptr;
    uint8_t* biasData = nullptr;
    uint8_t* scaleAData = nullptr;
    uint8_t* scaleBData = nullptr;
    uint8_t* groupList = nullptr;
    uint8_t* gmmArray = nullptr;
    uint8_t* aDesc = nullptr;
    uint8_t* bDesc = nullptr;
    uint8_t* cDesc = nullptr;
    uint8_t* biasDesc = nullptr;

    ~DeviceBuffers()
    {
        auto freePtr = [](uint8_t*& ptr) {
            if (ptr != nullptr) {
                aclrtFree(ptr);
                ptr = nullptr;
            }
        };
        freePtr(aDesc);
        freePtr(bDesc);
        freePtr(cDesc);
        freePtr(biasDesc);
        freePtr(gmmArray);
        freePtr(groupList);
        freePtr(scaleBData);
        freePtr(scaleAData);
        freePtr(biasData);
        freePtr(cData);
        freePtr(bData);
        freePtr(aData);
    }
};

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

void Alloc(uint8_t*& ptr, size_t bytes)
{
    if (bytes == 0U) {
        ptr = nullptr;
        return;
    }
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void CopyToDevice(uint8_t* dst, const std::vector<uint8_t>& src)
{
    if (dst == nullptr) {
        return;
    }
    ACL_CHECK(aclrtMemcpy(dst, src.size(), src.data(), src.size(), ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

template <typename LayoutB, typename BType>
size_t BGroupBytes(int64_t n, int64_t k)
{
    constexpr size_t c0 = asc::te::c0_element<BType>;
    if constexpr (std::is_same_v<LayoutB, asc::te::nz_layout_ptn>) {
        return AlignUp<size_t>(k, 16U) * AlignUp<size_t>(n, c0) * sizeof(BType);
    } else if constexpr (std::is_same_v<LayoutB, asc::te::zn_layout_ptn>) {
        return AlignUp<size_t>(k, c0) * AlignUp<size_t>(n, 16U) * sizeof(BType);
    }
    return static_cast<size_t>(k) * n * sizeof(BType);
}

void FillListTensorDesc(std::vector<uint8_t>& desc, uint64_t dataPtr)
{
    std::vector<uint64_t> list(4U);
    list[0] = 24U;
    list[1] = 1ULL << TENSOR_LIST_HEADER_COUNT_SHIFT;
    list[2] = 0xffffffffULL;
    list[3] = dataPtr;
    desc.resize(list.size() * sizeof(uint64_t));
    std::memcpy(desc.data(), list.data(), desc.size());
}

std::vector<int64_t> MakeGroupList(uint32_t groupNum, int64_t m, uint8_t groupListType)
{
    std::vector<int64_t> groupList;
    groupList.reserve(groupListType == 2U ? groupNum * 2U : groupNum);
    for (uint32_t i = 0; i < groupNum; ++i) {
        if (groupListType == 2U) {
            groupList.push_back(static_cast<int64_t>(i));
        }
        groupList.push_back(groupListType == 0U ? static_cast<int64_t>(i + 1U) * m : m);
    }
    return groupList;
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, typename LayoutB>
// NOTE(实参顺序): 8 字节参数(int64_t)必须排在 4 字节参数之前。
// 若写成 `... gmmArray, uint32_t groupNum, int64_t totalM, ...`（uint32 紧跟 int64），host 侧启动桩
// 与 device 侧入口对该处 4 字节 padding 的规则不一致，会导致从 totalM 起所有实参整体错位 4 字节：
// 实测 groupType 收到 groupListType 的值、singleW 收到越界垃圾值，kernel 因此落到 NO_SPLIT 分支
// 读到 mList=-1，所有 group 都被 `problemM <= 0` 跳过，表现为 kernel 正常返回但输出恒为 0。
__global__ __aicore__ void QgmmCubeKernel(GM_ADDR aDesc, GM_ADDR bDesc, GM_ADDR cDesc, GM_ADDR biasDesc, GM_ADDR scaleA,
                                          GM_ADDR scaleB, GM_ADDR groupList, GM_ADDR gmmArray, int64_t totalM,
                                          int64_t n, int64_t k, uint32_t groupNum, uint32_t baseM, uint32_t baseN,
                                          uint32_t baseK, uint32_t kAL1, uint32_t kBL1, uint32_t x1QuantMode,
                                          uint32_t x2QuantMode, uint32_t isBias, uint32_t dbL0C, int32_t groupType,
                                          uint32_t groupListType, uint32_t singleW)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using LayoutA = NdLayout;
    using LayoutC = NdLayout;
    using LayoutBias = NdLayout;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType,
                                               LayoutBias>;
    using Epilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;

    typename Kernel::Params p{};
    p.problemShape = {totalM, n, k, 0};
    p.mmadParams.aGmAddr = aDesc;
    p.mmadParams.bGmAddr = bDesc;
    p.mmadParams.cGmAddr = cDesc;
    p.mmadParams.biasGmAddr = biasDesc;
    p.mmadParams.scaleAGmAddr = scaleA;
    p.mmadParams.scaleBGmAddr = scaleB;
    p.groupListGmAddr = groupList;
    p.gmmArrayGmAddr = reinterpret_cast<__gm__ int32_t*>(gmmArray);
    p.gmmParams.groupNum = groupNum;
    p.gmmParams.m = totalM;
    p.gmmParams.n = n;
    p.gmmParams.k = k;
    p.gmmParams.baseM = baseM;
    p.gmmParams.baseN = baseN;
    p.gmmParams.baseK = baseK;
    p.gmmParams.kAL1 = kAL1;
    p.gmmParams.kBL1 = kBL1;
    p.gmmParams.x1QuantMode = x1QuantMode;
    p.gmmParams.x2QuantMode = x2QuantMode;
    p.gmmParams.isBias = static_cast<uint8_t>(isBias);
    p.gmmParams.dbL0C = static_cast<uint8_t>(dbL0C);
    p.gmmParams.groupType = static_cast<int8_t>(groupType);
    p.gmmParams.groupListType = static_cast<uint8_t>(groupListType);
    p.gmmParams.singleW = static_cast<uint8_t>(singleW);
    p.gmmParams.singleX = 1;
    p.gmmParams.singleY = 1;

    Kernel kernel;
    kernel(p);
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, typename LayoutB>
void Launch(const CubeConfig& cfg, DeviceBuffers& dev, aclrtStream stream)
{
    const int64_t totalM = static_cast<int64_t>(cfg.groupNum) * cfg.m;
    QgmmCubeKernel<AType, BType, CType, BiasType, X2ScaleType, LayoutB>
        <<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream>>>(
            dev.aDesc, dev.bDesc, dev.cDesc, dev.biasDesc, dev.scaleAData, dev.scaleBData, dev.groupList, dev.gmmArray,
            totalM, cfg.n, cfg.k, cfg.groupNum, cfg.baseM, cfg.baseN, cfg.baseK, cfg.kAL1, cfg.kBL1, cfg.x1QuantMode,
            cfg.x2QuantMode, static_cast<uint32_t>(cfg.isBias), static_cast<uint32_t>(cfg.dbL0C),
            static_cast<int32_t>(cfg.groupType), static_cast<uint32_t>(cfg.groupListType),
            static_cast<uint32_t>(cfg.singleW));
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, typename LayoutB>
void RunTypedCase(const CubeConfig& cfg)
{
    const int64_t totalM = static_cast<int64_t>(cfg.groupNum) * cfg.m;
    const size_t aBytes = static_cast<size_t>(totalM) * cfg.k * sizeof(AType);
    const size_t bBytes = static_cast<size_t>(cfg.groupNum) * BGroupBytes<LayoutB, BType>(cfg.n, cfg.k);
    const size_t cBytes = static_cast<size_t>(totalM) * cfg.n * sizeof(CType);
    const size_t biasBytes = static_cast<size_t>(cfg.groupNum) * cfg.n * sizeof(BiasType);
    const bool perChannel = cfg.x2QuantMode == QM_PERCHANNEL;
    const size_t scaleAElements = cfg.x1QuantMode == QM_PERTENSOR ? cfg.groupNum : 0U;
    const size_t scaleBElements = perChannel ? static_cast<size_t>(cfg.groupNum) * cfg.n * sizeof(uint64_t) /
                                                   sizeof(X2ScaleType) :
                                               cfg.groupNum;
    const size_t scaleABytes = scaleAElements * sizeof(float);
    const size_t scaleBBytes = scaleBElements * sizeof(X2ScaleType);
    const size_t groupListItems = cfg.groupListType == 2U ? cfg.groupNum * 2U : cfg.groupNum;
    const size_t gmmArrayBytes = 3U * GMM_ARRAY_LEN * sizeof(int32_t);

    DeviceBuffers dev;
    Alloc(dev.aData, aBytes);
    Alloc(dev.bData, bBytes);
    Alloc(dev.cData, cBytes);
    Alloc(dev.biasData, biasBytes);
    Alloc(dev.scaleAData, scaleABytes);
    Alloc(dev.scaleBData, scaleBBytes);
    Alloc(dev.groupList, groupListItems * sizeof(int64_t));
    Alloc(dev.gmmArray, gmmArrayBytes);
    Alloc(dev.aDesc, 4U * sizeof(uint64_t));
    Alloc(dev.bDesc, 4U * sizeof(uint64_t));
    Alloc(dev.cDesc, 4U * sizeof(uint64_t));
    Alloc(dev.biasDesc, 4U * sizeof(uint64_t));

    CopyToDevice(dev.aData, ReadBinary(cfg.dataDir + "/input_a.bin", aBytes));
    CopyToDevice(dev.bData, ReadBinary(cfg.dataDir + "/input_b.bin", bBytes));
    CopyToDevice(dev.cData, std::vector<uint8_t>(cBytes, 0U));
    CopyToDevice(dev.biasData, ReadBinary(cfg.dataDir + "/bias.bin", biasBytes));
    CopyToDevice(dev.scaleAData,
                 scaleABytes == 0U ? std::vector<uint8_t>{} : ReadBinary(cfg.dataDir + "/scale_a.bin", scaleABytes));
    CopyToDevice(dev.scaleBData, ReadBinary(cfg.dataDir + "/scale_b.bin", scaleBBytes));
    CopyToDevice(dev.groupList, ReadBinary(cfg.dataDir + "/group_list.bin", groupListItems * sizeof(int64_t)));

    std::vector<int32_t> gmmArray(3U * GMM_ARRAY_LEN, 0);
    // 按 kernel_qgmm_cube.h 的实际读取顺序填充 GMMArray：
    //   mListGm_ = gmmArray;  kListGm_ = gmmArray + 128;  nListGm_ = gmmArray + 256
    //   SetMNK: ProblemShape{splitValue, nListGm_[idx], kListGm_[idx]}  => N 在 +256、K 在 +128
    for (size_t i = 0; i < GMM_ARRAY_LEN; ++i) {
        gmmArray[i] = -1; // mList: -1 => M 取自 groupList (SPLIT_M)
        gmmArray[GMM_ARRAY_LEN + i] = static_cast<int32_t>(cfg.k);
        gmmArray[2U * GMM_ARRAY_LEN + i] = static_cast<int32_t>(cfg.n);
    }
    std::vector<uint8_t> gmmArrayBytesVec(gmmArrayBytes);
    std::memcpy(gmmArrayBytesVec.data(), gmmArray.data(), gmmArrayBytes);
    CopyToDevice(dev.gmmArray, gmmArrayBytesVec);

    std::vector<uint8_t> aDesc;
    FillListTensorDesc(aDesc, reinterpret_cast<uint64_t>(dev.aData));
    std::vector<uint8_t> bDesc;
    FillListTensorDesc(bDesc, reinterpret_cast<uint64_t>(dev.bData));
    std::vector<uint8_t> cDesc;
    FillListTensorDesc(cDesc, reinterpret_cast<uint64_t>(dev.cData));
    std::vector<uint8_t> biasDesc;
    FillListTensorDesc(biasDesc, reinterpret_cast<uint64_t>(dev.biasData));
    CopyToDevice(dev.aDesc, aDesc);
    CopyToDevice(dev.bDesc, bDesc);
    CopyToDevice(dev.cDesc, cDesc);
    CopyToDevice(dev.biasDesc, biasDesc);

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    Launch<AType, BType, CType, BiasType, X2ScaleType, LayoutB>(cfg, dev, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint8_t> output(cBytes);
    ACL_CHECK(aclrtMemcpy(output.data(), cBytes, dev.cData, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtDestroyStream(stream));

    std::ofstream out(cfg.outputPath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output file: " + cfg.outputPath);
    }
    out.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + cfg.outputPath);
    }
}

template <typename AType, typename CType, typename BiasType, typename ScaleType>
void DispatchTypedLayout(const CubeConfig& cfg)
{
    if (cfg.layoutB == "nd") {
        RunTypedCase<AType, AType, CType, BiasType, ScaleType, NdLayout>(cfg);
    } else if (cfg.layoutB == "dn") {
        RunTypedCase<AType, AType, CType, BiasType, ScaleType, asc::te::dn_ext_layout_ptn>(cfg);
    } else if (cfg.layoutB == "nz") {
        RunTypedCase<AType, AType, CType, BiasType, ScaleType, asc::te::nz_layout_ptn>(cfg);
    } else {
        RunTypedCase<AType, AType, CType, BiasType, ScaleType, asc::te::zn_layout_ptn>(cfg);
    }
}

template <typename AType, typename CType, typename BiasType>
void DispatchScale(const CubeConfig& cfg)
{
    if (cfg.x2QuantMode == QM_PERCHANNEL) {
        DispatchTypedLayout<AType, CType, BiasType, uint64_t>(cfg);
    } else {
        DispatchTypedLayout<AType, CType, BiasType, float>(cfg);
    }
}

int DispatchLayoutB(const CubeConfig& cfg)
{
    if (cfg.layoutB != "nd" && cfg.layoutB != "dn" && cfg.layoutB != "nz" && cfg.layoutB != "zn") {
        return 2;
    }
    const bool bf16 = cfg.cType == "bf16" || cfg.cType == "bfloat16";
    const bool int8Input = (cfg.aType == "int8" || cfg.aType == "int8_t") &&
                           (cfg.bType == "int8" || cfg.bType == "int8_t");
    if (int8Input && (cfg.biasType == "int32" || cfg.biasType == "int32_t")) {
        if (bf16) {
            DispatchScale<int8_t, bfloat16_t, int32_t>(cfg);
        } else if (cfg.cType == "fp16" || cfg.cType == "float16" || cfg.cType == "half") {
            DispatchScale<int8_t, half, int32_t>(cfg);
        } else {
            return 2;
        }
        return 0;
    }
    const bool fp8Input = (cfg.aType == "fp8_e4m3" || cfg.aType == "fp8_e4m3fn" || cfg.aType == "fp8_e4m3fn_t") &&
                          (cfg.bType == "fp8_e4m3" || cfg.bType == "fp8_e4m3fn" || cfg.bType == "fp8_e4m3fn_t");
    if (fp8Input && (cfg.biasType == "float" || cfg.biasType == "float32")) {
        if (bf16) {
            DispatchScale<fp8_e4m3fn_t, bfloat16_t, float>(cfg);
        } else if (cfg.cType == "fp32" || cfg.cType == "float" || cfg.cType == "float32") {
            DispatchScale<fp8_e4m3fn_t, float, float>(cfg);
        } else {
            return 2;
        }
        return 0;
    }
    return 2;
}

void PrintUsage()
{
    std::cerr << "Usage: quant_grouped_matmul_cubeonly <groupNum> <m> <n> <k> <baseM> <baseN> <baseK> "
                 "<kAL1> <kBL1> <x1QuantMode> <x2QuantMode> <isBias> <dbL0C> <groupType> <groupListType> "
                 "<singleW> <aType> <bType> <cType> <biasType> <x2ScaleType> <layoutA> <layoutB> "
                 "<dataDir> <outputPath>"
              << std::endl;
}

CubeConfig ParseConfig(char** argv)
{
    CubeConfig cfg{};
    cfg.groupNum = static_cast<uint32_t>(std::stoul(argv[1]));
    cfg.m = std::stoll(argv[2]);
    cfg.n = std::stoll(argv[3]);
    cfg.k = std::stoll(argv[4]);
    cfg.baseM = static_cast<uint32_t>(std::stoul(argv[5]));
    cfg.baseN = static_cast<uint32_t>(std::stoul(argv[6]));
    cfg.baseK = static_cast<uint32_t>(std::stoul(argv[7]));
    cfg.kAL1 = static_cast<uint32_t>(std::stoul(argv[8]));
    cfg.kBL1 = static_cast<uint32_t>(std::stoul(argv[9]));
    cfg.x1QuantMode = static_cast<uint32_t>(std::stoul(argv[10]));
    cfg.x2QuantMode = static_cast<uint32_t>(std::stoul(argv[11]));
    cfg.isBias = static_cast<uint8_t>(std::stoul(argv[12]));
    cfg.dbL0C = static_cast<uint8_t>(std::stoul(argv[13]));
    cfg.groupType = static_cast<int8_t>(std::stoi(argv[14]));
    cfg.groupListType = static_cast<uint8_t>(std::stoul(argv[15]));
    cfg.singleW = static_cast<uint8_t>(std::stoul(argv[16]));
    cfg.aType = argv[17];
    cfg.bType = argv[18];
    cfg.cType = argv[19];
    cfg.biasType = argv[20];
    cfg.x2ScaleType = argv[21];
    cfg.layoutA = argv[22];
    cfg.layoutB = argv[23];
    cfg.dataDir = argv[24];
    cfg.outputPath = argv[25];
    return cfg;
}

bool IsValidConfig(const CubeConfig& cfg)
{
    if (cfg.groupNum == 0U || cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
        return false;
    }
    if (cfg.baseM == 0U || cfg.baseN == 0U || cfg.baseK == 0U || cfg.kAL1 == 0U || cfg.kBL1 == 0U) {
        return false;
    }
    if (cfg.x1QuantMode > QM_PERTENSOR || cfg.x2QuantMode > QM_PERCHANNEL) {
        return false;
    }
    if (cfg.isBias > 1U || cfg.dbL0C < 1U || cfg.dbL0C > 2U || cfg.groupListType > 2U || cfg.singleW != 1U) {
        return false;
    }
    if (cfg.groupType != 0 || cfg.layoutA != "nd") {
        return false;
    }
    return cfg.layoutB == "nd" || cfg.layoutB == "dn" || cfg.layoutB == "nz" || cfg.layoutB == "zn";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 26) {
        PrintUsage();
        return 2;
    }
    const CubeConfig cfg = ParseConfig(argv);
    if (!IsValidConfig(cfg)) {
        std::cerr << "invalid QGMM Cube case configuration" << std::endl;
        return 2;
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    const int ret = DispatchLayoutB(cfg);
    aclrtResetDevice(0);
    aclFinalize();
    if (ret == 0) {
        std::cout << "QGMM Cube kernel execution completed, output=" << cfg.outputPath << std::endl;
    } else {
        std::cerr << "QGMM Cube example FAILED: unsupported argument combination" << std::endl;
    }
    return ret;
}
