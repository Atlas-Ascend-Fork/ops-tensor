/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file quant_grouped_matmul_activation_quant_mx.cpp
 * @brief Minimal CSV-driven GMMAQ MX example (grouped GEMM + GeluTanh + MX quant).
 *
 * The example deliberately keeps one fixed, easy-to-read tiling configuration.  The
 * only host-side calculations are storage sizes and byte offsets required to load
 * the input files; the device kernel receives the tiling values from the CSV file.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_gelu_tanh_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "platform/platform_ascendc.h"

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using NdLayout = asc::te::nd_ext_layout_ptn;
using NzLayout = asc::te::nz_layout_ptn;

static constexpr uint32_t GROUP_LIST_TYPE_LENGTH = 1;
static constexpr uint32_t GROUP_TYPE_M = 0;
static constexpr uint32_t SINGLE_WEIGHT = 1;
static constexpr size_t MX_SCALE_GROUP_SIZE = 64;
static constexpr size_t MX_SCALE_VALUES_PER_GROUP = 2;

template <typename T>
inline constexpr bool IS_FP4 = std::is_same_v<T, fp4x2_e2m1_t>;

template <typename AType, typename BType, typename OutType>
// Keep 64-bit arguments before 32-bit arguments so host launch stubs and device entry points agree on alignment.
// Pass small flags in 32-bit slots, then narrow them when populating the internal tiling structure.
__global__ __aicore__ void GmmaqMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                         GM_ADDR groupList, GM_ADDR c, GM_ADDR y, GM_ADDR yScale, int64_t m, int64_t n,
                                         int64_t k, uint32_t groupNum, uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                         uint32_t kAL1, uint32_t kBL1, uint32_t scaleKAL1, uint32_t scaleKBL1,
                                         uint32_t dbL0C, uint32_t l1BufferStage, uint32_t scaleAlg, float dstTypeMax)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false,
                                                         Blaze::Gemm::KernelGroupedMmadWithScaleMxActivationQuant>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, NdLayout, BType, NzLayout, float, NdLayout, float,
                                               NdLayout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueGeluTanhMxQuant<OutType, float>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    typename Kernel::Params params{};
    params.problemShape = {m, n, k, 0};
    params.mmadParams = {x, weight, c, nullptr, xScale, weightScale};
    params.epilogueParams = {y, yScale, baseM, baseN, scaleAlg, dstTypeMax};
    params.groupListGmAddr = groupList;
    params.gmmParams = {groupNum,
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
                        0,
                        static_cast<uint8_t>(dbL0C),
                        static_cast<uint8_t>(l1BufferStage),
                        GROUP_TYPE_M,
                        GROUP_LIST_TYPE_LENGTH,
                        SINGLE_WEIGHT};
    Kernel kernel;
    kernel(params);
}

struct Config {
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
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    std::string dtype;
    uint32_t scaleAlg;
    float dstTypeMax;
    std::string inputDir;
    std::string outputY;
    std::string outputScale;
};

static size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

static size_t MxScaleElements(size_t dimension)
{
    return (dimension + MX_SCALE_GROUP_SIZE - 1) / MX_SCALE_GROUP_SIZE * MX_SCALE_VALUES_PER_GROUP;
}

static size_t ValueBytes(const std::string& dtype, size_t elements)
{
    if (dtype == "mxfp4_e2m1") {
        return (elements + 1) / 2;
    }
    if (dtype == "mxfp8_e4m3") {
        return elements;
    }
    throw std::invalid_argument("unsupported dtype: " + dtype);
}

static std::vector<uint8_t> ReadFile(const std::string& path, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != bytes) {
        throw std::runtime_error("invalid input file size: " + path);
    }
    std::vector<uint8_t> data(bytes);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

static void WriteFile(const std::string& path, const uint8_t* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

template <typename T>
static void CopyToDevice(uint8_t*& device, const std::string& path, size_t bytes)
{
    const auto host = ReadFile(path, bytes);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename AType, typename BType, typename OutType>
static void Run(const Config& config)
{
    const size_t c0 = IS_FP4<BType> ? 64 : 32;
    const size_t groupABytes = ValueBytes(config.dtype, static_cast<size_t>(config.m) * config.k);
    const size_t groupBBytes = ValueBytes(config.dtype, AlignUp(config.k, 16) * AlignUp(config.n, c0));
    const size_t scaleK = MxScaleElements(static_cast<size_t>(config.k));
    const size_t scaleN = MxScaleElements(static_cast<size_t>(config.n));
    const size_t groupScaleABytes = static_cast<size_t>(config.m) * scaleK;
    const size_t groupScaleBBytes = static_cast<size_t>(config.n) * scaleK;
    const size_t xBytes = config.groupNum * groupABytes;
    const size_t weightBytes = config.groupNum * groupBBytes;
    const size_t xScaleBytes = config.groupNum * groupScaleABytes;
    const size_t weightScaleBytes = config.groupNum * groupScaleBBytes;
    const size_t cBytes = config.groupNum * static_cast<size_t>(config.m) * config.n * sizeof(float);
    const size_t yBytes = ValueBytes(config.dtype, config.groupNum * static_cast<size_t>(config.m) * config.n);
    const size_t yScaleBytes = config.groupNum * static_cast<size_t>(config.m) * scaleN;
    const size_t groupListBytes = config.groupNum * sizeof(int64_t);

    uint8_t *x = nullptr, *weight = nullptr, *weightScale = nullptr, *xScale = nullptr;
    uint8_t *groupList = nullptr, *c = nullptr, *y = nullptr, *yScale = nullptr;
    CopyToDevice<uint8_t>(x, config.inputDir + "/input_x.bin", xBytes);
    CopyToDevice<uint8_t>(weight, config.inputDir + "/input_weight.bin", weightBytes);
    CopyToDevice<uint8_t>(weightScale, config.inputDir + "/weight_scale.bin", weightScaleBytes);
    CopyToDevice<uint8_t>(xScale, config.inputDir + "/x_scale.bin", xScaleBytes);
    CopyToDevice<uint8_t>(groupList, config.inputDir + "/group_list.bin", groupListBytes);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&c), cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&y), yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&yScale), yScaleBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(c, cBytes, 0, cBytes));
    ACL_CHECK(aclrtMemset(y, yBytes, 0, yBytes));
    ACL_CHECK(aclrtMemset(yScale, yScaleBytes, 0, yScaleBytes));

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    GmmaqMxKernel<AType, BType, OutType><<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream>>>(
        x, weight, weightScale, xScale, groupList, c, y, yScale, config.m, config.n, config.k, config.groupNum,
        config.baseM, config.baseN, config.baseK, config.kAL1, config.kBL1, config.scaleKAL1, config.scaleKBL1,
        static_cast<uint32_t>(config.dbL0C), static_cast<uint32_t>(config.l1BufferStage), config.scaleAlg,
        config.dstTypeMax);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> hostY(yBytes), hostScale(yScaleBytes);
    ACL_CHECK(aclrtMemcpy(hostY.data(), yBytes, y, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostScale.data(), yScaleBytes, yScale, yScaleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(config.outputY, hostY.data(), yBytes);
    WriteFile(config.outputScale, hostScale.data(), yScaleBytes);

    ACL_CHECK(aclrtDestroyStream(stream));
    for (uint8_t* ptr : {x, weight, weightScale, xScale, groupList, c, y, yScale}) {
        ACL_CHECK(aclrtFree(ptr));
    }
}

static Config ParseConfig(int argc, char** argv)
{
    if (argc != 20) {
        throw std::invalid_argument("usage: group_num m n k base_m base_n base_k k_a_l1 k_b_l1 scale_a_l1 scale_b_l1 "
                                    "db_l0c l1_stage dtype scale_alg dst_type_max input_dir output_y output_scale");
    }
    Config config{static_cast<uint32_t>(std::stoul(argv[1])),
                  std::stoll(argv[2]),
                  std::stoll(argv[3]),
                  std::stoll(argv[4]),
                  static_cast<uint32_t>(std::stoul(argv[5])),
                  static_cast<uint32_t>(std::stoul(argv[6])),
                  static_cast<uint32_t>(std::stoul(argv[7])),
                  static_cast<uint32_t>(std::stoul(argv[8])),
                  static_cast<uint32_t>(std::stoul(argv[9])),
                  static_cast<uint32_t>(std::stoul(argv[10])),
                  static_cast<uint32_t>(std::stoul(argv[11])),
                  static_cast<uint8_t>(std::stoul(argv[12])),
                  static_cast<uint8_t>(std::stoul(argv[13])),
                  argv[14],
                  static_cast<uint32_t>(std::stoul(argv[15])),
                  std::stof(argv[16]),
                  argv[17],
                  argv[18],
                  argv[19]};
    if (config.groupNum == 0 || config.m <= 0 || config.n <= 0 || config.k <= 0 || config.groupNum > 8 ||
        config.n % (config.dtype == "mxfp4_e2m1" ? 64 : 32) != 0 || config.k % 16 != 0 ||
        (config.l1BufferStage != 2 && config.l1BufferStage != 3) || config.dbL0C == 0 || config.dbL0C > 2 ||
        (config.dtype != "mxfp8_e4m3" && config.dtype != "mxfp4_e2m1")) {
        throw std::invalid_argument("invalid fixed GMMAQ example configuration");
    }
    return config;
}

int main(int argc, char** argv)
{
    try {
        const Config config = ParseConfig(argc, argv);
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        if (config.dtype == "mxfp8_e4m3") {
            Run<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t>(config);
        } else {
            Run<fp4x2_e2m1_t, fp4x2_e2m1_t, fp4x2_e2m1_t>(config);
        }
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::cout << "[PASS] GMMAQ MX example completed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << std::endl;
        return 1;
    }
}
