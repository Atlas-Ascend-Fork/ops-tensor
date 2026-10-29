/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file quant_grouped_matmul_swiglu_quant_mx.cpp
 * \brief CSV-driven QGMM + SwiGLU mode 2 + MX quant example for NZ/ZN weights.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "platform/platform_ascendc.h"

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using NdLayout = asc::te::nd_ext_layout_ptn;
using NzLayout = asc::te::nz_layout_ptn;
using ZnLayout = asc::te::zn_layout_ptn;

namespace {
constexpr int64_t SINGLE_BATCH = 1;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr int ARGUMENT_COUNT = 22;
constexpr uint32_t MAX_EXAMPLE_GROUPS = 8;
constexpr int64_t MIN_EXAMPLE_K = 1;
constexpr uint32_t EXAMPLE_BASE_M = 16;
constexpr uint32_t EXAMPLE_BASE_N = 64;
constexpr uint32_t EXAMPLE_BASE_K = 64;
constexpr uint32_t EXAMPLE_L1_K = 64;
constexpr uint8_t EXAMPLE_L0C_BUFFERS = 2;
constexpr uint32_t SCALE_ALG_CUBLAS = 1;
constexpr int FAILURE_EXIT_CODE = 1;
enum ArgumentIndex {
    PROGRAM_NAME,
    ARG_GROUP_NUM,
    ARG_M,
    ARG_N,
    ARG_K,
    ARG_BASE_M,
    ARG_BASE_N,
    ARG_BASE_K,
    ARG_KAL1,
    ARG_KBL1,
    ARG_SCALE_KAL1,
    ARG_SCALE_KBL1,
    ARG_DB_L0C,
    ARG_LAYOUT_B,
    ARG_SCALE_ALG,
    ARG_CLAMP_LIMIT,
    ARG_GLU_ALPHA,
    ARG_GLU_BIAS,
    ARG_DST_TYPE_MAX,
    ARG_INPUT_DIR,
    ARG_OUTPUT_Y,
    ARG_OUTPUT_SCALE
};
} // namespace
static constexpr uint32_t GROUP_TYPE_M = 0;
static constexpr uint32_t GROUP_LIST_TYPE_OFFSET = 0;
static constexpr uint32_t SINGLE_WEIGHT = 1;
static constexpr int64_t SWIGLU_MODE = 2;
static constexpr size_t MX_SCALE_GROUP_SIZE = 64;
static constexpr size_t MX_SCALE_VALUES_PER_GROUP = 2;
static constexpr size_t FP8_WEIGHT_K_C0 = 16;
static constexpr size_t FP8_WEIGHT_N_C0 = 32;

template <typename LayoutB>
__global__ __aicore__ void GmmsqMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                         GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k, uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                         uint32_t kAL1, uint32_t kBL1, uint32_t scaleKAL1, uint32_t scaleKBL1,
                                         uint8_t dbL0C, uint32_t scaleAlg, float clampLimit, float gluAlpha,
                                         float gluBias, float dstTypeMax)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using OutType = fp8_e4m3fn_t;
    using ScaleType = AscendC::fp8_e8m0_t;
    using LayoutScaleB = AscendC::Std::conditional_t<Blaze::Gemm::IsTrans<LayoutB>::value,
                                                     asc::te::scaleb_dn_layout_ptn, asc::te::scaleb_nd_layout_ptn>;
    using BLayoutPair = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGmmSwiGluMixMx>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, NdLayout, BType, BLayoutPair, float, NdLayout, float,
                                               NdLayout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutType, float, ScaleType>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    typename Kernel::GMMTiling gmmTiling{
        groupNum,     m,    n,         k,         baseM, baseN,        baseK,
        kAL1,         kBL1, scaleKAL1, scaleKBL1, dbL0C, GROUP_TYPE_M, GROUP_LIST_TYPE_OFFSET,
        SINGLE_WEIGHT};
    typename Mmad::Params mmAddressParams{x, weight, nullptr, nullptr, xScale, weightScale};
    typename Epilogue::Params epilogueParams{y, yScale, baseM, baseN};
    epilogueParams.swigluMode = SWIGLU_MODE;
    epilogueParams.clampLimit = clampLimit;
    epilogueParams.gluAlpha = gluAlpha;
    epilogueParams.gluBias = gluBias;
    epilogueParams.scaleAlg = scaleAlg;
    epilogueParams.dstTypeMax = dstTypeMax;

    typename Kernel::Params params{{m, n, k, SINGLE_BATCH}, mmAddressParams, epilogueParams, groupList, gmmTiling};
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
    std::string layoutB;
    uint32_t scaleAlg;
    float clampLimit;
    float gluAlpha;
    float gluBias;
    float dstTypeMax;
    std::string inputDir;
    std::string outputY;
    std::string outputScale;
};

static size_t CheckedMultiply(size_t left, size_t right)
{
    const auto limit = static_cast<size_t>(std::numeric_limits<std::streamsize>::max());
    if (right != 0 && left > limit / right) {
        throw std::invalid_argument("example tensor size exceeds the supported byte range");
    }
    return left * right;
}

static size_t AlignUp(size_t value, size_t alignment)
{
    return CheckedMultiply(value / alignment + (value % alignment != 0), alignment);
}

static size_t MxScaleElements(size_t dimension)
{
    return CheckedMultiply(dimension / MX_SCALE_GROUP_SIZE + (dimension % MX_SCALE_GROUP_SIZE != 0),
                           MX_SCALE_VALUES_PER_GROUP);
}

static void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

static void CheckCleanup(aclError status, const char* operation) noexcept
{
    if (status != ACL_SUCCESS) {
        std::cerr << "[WARN] " << operation << " failed during cleanup: " << status << std::endl;
    }
}

class DeviceRuntime {
public:
    DeviceRuntime() = default;
    DeviceRuntime(const DeviceRuntime&) = delete;
    DeviceRuntime& operator=(const DeviceRuntime&) = delete;

    void Init()
    {
        CheckAcl(aclInit(nullptr), "aclInit");
        initialized_ = true;
        CheckAcl(aclrtSetDevice(0), "aclrtSetDevice");
        deviceSet_ = true;
    }

    ~DeviceRuntime() noexcept
    {
        if (deviceSet_) {
            CheckCleanup(aclrtResetDevice(0), "aclrtResetDevice");
        }
        if (initialized_) {
            CheckCleanup(aclFinalize(), "aclFinalize");
        }
    }

private:
    bool initialized_{false};
    bool deviceSet_{false};
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Allocate(size_t bytes)
    {
        CheckAcl(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }

    uint8_t* Get() const { return data_; }

    ~DeviceBuffer() noexcept
    {
        if (data_ != nullptr) {
            CheckCleanup(aclrtFree(data_), "aclrtFree");
        }
    }

private:
    uint8_t* data_{nullptr};
};

class DeviceStream {
public:
    DeviceStream() { CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream"); }
    DeviceStream(const DeviceStream&) = delete;
    DeviceStream& operator=(const DeviceStream&) = delete;
    aclrtStream Get() const { return stream_; }

    ~DeviceStream() noexcept { CheckCleanup(aclrtDestroyStream(stream_), "aclrtDestroyStream"); }

private:
    aclrtStream stream_{nullptr};
};

static std::vector<uint8_t> ReadFile(const std::string& path, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != bytes) {
        throw std::runtime_error("invalid input file size: " + path);
    }
    std::vector<uint8_t> data(bytes);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes)) ||
        file.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("cannot read complete input file: " + path);
    }
    return data;
}

static void WriteFile(const std::string& path, const uint8_t* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    file.flush();
    if (!file) {
        throw std::runtime_error("cannot write complete output file: " + path);
    }
}

static void CopyToDevice(DeviceBuffer& device, const std::vector<uint8_t>& host)
{
    device.Allocate(host.size());
    CheckAcl(aclrtMemcpy(device.Get(), host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy");
}

static void CopyToDevice(DeviceBuffer& device, const std::string& path, size_t bytes)
{
    const auto host = ReadFile(path, bytes);
    CopyToDevice(device, host);
}

static void ValidateGroupList(const std::vector<uint8_t>& data, const Config& config)
{
    int64_t previous = 0;
    for (size_t index = 0; index < config.groupNum; ++index) {
        int64_t offset = 0;
        std::copy_n(data.data() + index * sizeof(offset), sizeof(offset), reinterpret_cast<uint8_t*>(&offset));
        if (offset < previous || offset > config.m) {
            throw std::invalid_argument("group list must be nondecreasing and within [0, m]");
        }
        previous = offset;
    }
    if (previous != config.m) {
        throw std::invalid_argument("group list must end at m");
    }
}

template <typename LayoutB>
static void Run(const Config& config)
{
    const size_t totalM = static_cast<size_t>(config.m);
    const size_t fullN = static_cast<size_t>(config.n);
    const size_t outputN = fullN / SWIGLU_SPLIT_FACTOR;
    const size_t k = static_cast<size_t>(config.k);
    const size_t scaleK = MxScaleElements(k);
    const size_t outputScaleN = MxScaleElements(outputN);
    const bool transB = config.layoutB == "zn";
    const size_t weightGroupElements = transB ? CheckedMultiply(AlignUp(fullN, FP8_WEIGHT_K_C0),
                                                                AlignUp(k, FP8_WEIGHT_N_C0)) :
                                                CheckedMultiply(AlignUp(k, FP8_WEIGHT_K_C0),
                                                                AlignUp(fullN, FP8_WEIGHT_N_C0));

    const size_t xBytes = CheckedMultiply(totalM, k);
    const size_t weightBytes = CheckedMultiply(config.groupNum, weightGroupElements);
    const size_t xScaleBytes = CheckedMultiply(totalM, scaleK);
    const size_t weightScaleBytes = CheckedMultiply(CheckedMultiply(config.groupNum, fullN), scaleK);
    const size_t groupListBytes = CheckedMultiply(config.groupNum, sizeof(int64_t));
    const size_t yBytes = CheckedMultiply(totalM, outputN);
    const size_t yScaleBytes = CheckedMultiply(totalM, outputScaleN);
    const auto groupData = ReadFile(config.inputDir + "/group_list.bin", groupListBytes);
    ValidateGroupList(groupData, config);

    DeviceBuffer x, weight, weightScale, xScale, groupList, y, yScale;
    CopyToDevice(x, config.inputDir + "/input_x.bin", xBytes);
    CopyToDevice(weight, config.inputDir + "/input_weight.bin", weightBytes);
    CopyToDevice(weightScale, config.inputDir + "/weight_scale.bin", weightScaleBytes);
    CopyToDevice(xScale, config.inputDir + "/x_scale.bin", xScaleBytes);
    CopyToDevice(groupList, groupData);
    y.Allocate(yBytes);
    yScale.Allocate(yScaleBytes);
    CheckAcl(aclrtMemset(y.Get(), yBytes, 0, yBytes), "aclrtMemset");
    CheckAcl(aclrtMemset(yScale.Get(), yScaleBytes, 0, yScaleBytes), "aclrtMemset");

    DeviceStream stream;
    GmmsqMxKernel<LayoutB><<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream.Get()>>>(
        x.Get(), weight.Get(), weightScale.Get(), xScale.Get(), groupList.Get(), y.Get(), yScale.Get(), config.groupNum,
        config.m, config.n, config.k, config.baseM, config.baseN, config.baseK, config.kAL1, config.kBL1,
        config.scaleKAL1, config.scaleKBL1, config.dbL0C, config.scaleAlg, config.clampLimit, config.gluAlpha,
        config.gluBias, config.dstTypeMax);
    CheckAcl(aclrtSynchronizeStream(stream.Get()), "aclrtSynchronizeStream");

    std::vector<uint8_t> hostY(yBytes), hostScale(yScaleBytes);
    CheckAcl(aclrtMemcpy(hostY.data(), yBytes, y.Get(), yBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy");
    CheckAcl(aclrtMemcpy(hostScale.data(), yScaleBytes, yScale.Get(), yScaleBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy");
    WriteFile(config.outputY, hostY.data(), yBytes);
    WriteFile(config.outputScale, hostScale.data(), yScaleBytes);
}

static uint64_t ParseUnsigned(const char* text, uint64_t limit)
{
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("integer arguments must contain decimal digits only");
    }
    const auto parsed = std::stoull(value);
    if (parsed > limit) {
        throw std::invalid_argument("integer argument is out of range");
    }
    return parsed;
}

static float ParseFloat(const char* text)
{
    size_t end = 0;
    const std::string value(text);
    const float result = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("floating-point arguments must be finite numbers");
    }
    return result;
}

static Config ParseConfig(int argc, char** argv)
{
    if (argc != ARGUMENT_COUNT) {
        throw std::invalid_argument(
            "usage: group_num m n k base_m base_n base_k k_a_l1 k_b_l1 scale_a_l1 scale_b_l1 db_l0c layout_b "
            "scale_alg clamp_limit glu_alpha glu_bias dst_type_max input_dir output_y output_scale");
    }
    constexpr auto maxU32 = std::numeric_limits<uint32_t>::max();
    constexpr auto maxI64 = std::numeric_limits<int64_t>::max();
    Config config{static_cast<uint32_t>(ParseUnsigned(argv[ARG_GROUP_NUM], maxU32)),
                  static_cast<int64_t>(ParseUnsigned(argv[ARG_M], maxI64)),
                  static_cast<int64_t>(ParseUnsigned(argv[ARG_N], maxI64)),
                  static_cast<int64_t>(ParseUnsigned(argv[ARG_K], maxI64)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_BASE_M], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_BASE_N], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_BASE_K], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_KAL1], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_KBL1], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_SCALE_KAL1], maxU32)),
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_SCALE_KBL1], maxU32)),
                  static_cast<uint8_t>(ParseUnsigned(argv[ARG_DB_L0C], std::numeric_limits<uint8_t>::max())),
                  argv[ARG_LAYOUT_B],
                  static_cast<uint32_t>(ParseUnsigned(argv[ARG_SCALE_ALG], maxU32)),
                  ParseFloat(argv[ARG_CLAMP_LIMIT]),
                  ParseFloat(argv[ARG_GLU_ALPHA]),
                  ParseFloat(argv[ARG_GLU_BIAS]),
                  ParseFloat(argv[ARG_DST_TYPE_MAX]),
                  argv[ARG_INPUT_DIR],
                  argv[ARG_OUTPUT_Y],
                  argv[ARG_OUTPUT_SCALE]};
    const int64_t outputN = config.n / SWIGLU_SPLIT_FACTOR;
    // This teaching example uses the accompanying fixtures' fixed tiling, not a general-purpose Host tiler.
    if (config.groupNum == 0 || config.groupNum > MAX_EXAMPLE_GROUPS || config.m <= 0 || config.n <= 0 ||
        config.k <= MIN_EXAMPLE_K || config.n % SWIGLU_SPLIT_FACTOR != 0 || outputN % FP8_WEIGHT_N_C0 != 0 ||
        config.baseM != EXAMPLE_BASE_M || config.baseN != EXAMPLE_BASE_N || config.baseK != EXAMPLE_BASE_K ||
        config.kAL1 != EXAMPLE_L1_K || config.kBL1 != EXAMPLE_L1_K || config.scaleKAL1 != EXAMPLE_L1_K ||
        config.scaleKBL1 != EXAMPLE_L1_K || config.dbL0C != EXAMPLE_L0C_BUFFERS ||
        (config.layoutB != "nz" && config.layoutB != "zn") || config.scaleAlg > SCALE_ALG_CUBLAS ||
        config.clampLimit <= 0.0F || config.dstTypeMax != 0.0F) {
        throw std::invalid_argument("invalid fixed QGMM SwiGLU MX example configuration");
    }
    return config;
}

int main(int argc, char** argv)
{
    try {
        const Config config = ParseConfig(argc, argv);
        DeviceRuntime runtime;
        runtime.Init();
        if (config.layoutB == "nz") {
            Run<NzLayout>(config);
        } else {
            Run<ZnLayout>(config);
        }
        std::cout << "[PASS] QGMM SwiGLU MX example completed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << std::endl;
        return FAILURE_EXIT_CODE;
    }
}
