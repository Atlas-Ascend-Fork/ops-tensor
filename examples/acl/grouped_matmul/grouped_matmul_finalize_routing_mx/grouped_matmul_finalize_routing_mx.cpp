/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/**
 * @file grouped_matmul_finalize_routing_mx.cpp
 * @brief CSV-driven GroupedMatmulFinalizeRouting MX WeightNZ example.
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

namespace {
struct CaseConfig {
    uint32_t groupNum;
    int64_t totalM;
    int64_t batch;
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
    uint8_t groupListType;
    std::string dtype;
    std::string outputDtype;
    std::string logitDtype;
    std::string rowIndexDtype;
    std::string dataDir;
    std::string outputPath;
};

template <typename T>
inline constexpr bool IS_FP4 = std::is_same_v<T, fp4x2_e2m1_t>;

template <typename T>
size_t MxBytes(size_t elements)
{
    return IS_FP4<T> ? (elements + 1U) / 2U : elements * sizeof(T);
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint8_t* Get() const { return ptr_; }
    size_t Size() const { return bytes_; }

private:
    uint8_t* ptr_{nullptr};
    size_t bytes_{0};
};

std::vector<uint8_t> ReadFile(const std::string& path, size_t bytes)
{
    std::vector<uint8_t> data(bytes);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("input file size mismatch: " + path);
    }
    return data;
}

void CopyToDevice(DeviceBuffer& dst, const std::vector<uint8_t>& host)
{
    ACL_CHECK(aclrtMemcpy(dst.Get(), dst.Size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
}

void WriteFile(const std::string& path, const std::vector<uint8_t>& data)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

template <typename AType, typename CType, typename LogitType, typename RowIndexType>
// Keep 64-bit arguments before 32-bit slots to match host and device launch layouts.
__global__ __aicore__ void GmmFinalizeRoutingKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR bias,
                                                    GM_ADDR xScale, GM_ADDR groupList, GM_ADDR sharedInput,
                                                    GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR y, int64_t totalM,
                                                    int64_t batch, int64_t n, int64_t k, uint32_t groupNum,
                                                    uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t kAL1,
                                                    uint32_t kBL1, uint32_t scaleKAL1, uint32_t scaleKBL1,
                                                    uint32_t dbL0C, uint32_t groupListType)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nz_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = bfloat16_t;
    using MmadType = float;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelQgmmMxMixFinalizeRouting>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, AType, LayoutB, MmadType, LayoutC, BiasType,
                                               LayoutBias>;
    using Prologue = Blaze::Gemm::Kernel::BlockPrologueFinalizeRouting<CType, BiasType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<CType, MmadType, LogitType, RowIndexType>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, AscendC::Std::tuple<Prologue, Epilogue>,
                                                      Scheduler>;

    typename Kernel::Params params{};
    params.problemShape = {totalM, n, k, 1};
    params.blockMmadParams = {x, weight, y, bias, xScale, weightScale};
    params.prologueParams = {sharedInput, y, 0, 0, static_cast<int32_t>(n), static_cast<uint32_t>(batch), 1.0F};
    params.epilogueParams = {
        y, weightScale, xScale, bias, logit, rowIndex, static_cast<int32_t>(baseM), static_cast<int32_t>(baseN)};
    params.groupListGmAddr = groupList;
    params.gmmParams = {groupNum,
                        static_cast<uint32_t>(batch),
                        0,
                        0,
                        1.0F,
                        baseM,
                        baseN,
                        baseK,
                        kAL1,
                        kBL1,
                        scaleKAL1,
                        scaleKBL1,
                        0,
                        static_cast<uint8_t>(dbL0C),
                        static_cast<uint8_t>(groupListType)};
    Kernel kernel;
    kernel(params);
}

template <typename AType, typename CType, typename LogitType, typename RowIndexType>
int RunCase(const CaseConfig& config)
{
    constexpr size_t scaleElementBytes = sizeof(fp8_e8m0_t);
    const size_t xBytes = MxBytes<AType>(static_cast<size_t>(config.totalM * config.k));
    const size_t weightBytes = static_cast<size_t>(config.groupNum) *
                               MxBytes<AType>(static_cast<size_t>(config.k * config.n));
    const size_t scaleK = static_cast<size_t>((config.k + 63) / 64 * 2);
    const size_t xScaleBytes = static_cast<size_t>(config.totalM) * scaleK * scaleElementBytes;
    const size_t weightScaleBytes = static_cast<size_t>(config.groupNum) * static_cast<size_t>(config.n) * scaleK *
                                    scaleElementBytes;
    const size_t biasBytes = static_cast<size_t>(config.groupNum) * static_cast<size_t>(config.n) * sizeof(bfloat16_t);
    const size_t sharedBytes = static_cast<size_t>(config.batch) * static_cast<size_t>(config.n) * sizeof(bfloat16_t);
    const size_t groupListBytes = static_cast<size_t>(config.groupNum) * sizeof(int64_t);
    const size_t logitBytes = static_cast<size_t>(config.totalM) * sizeof(LogitType);
    const size_t rowIndexBytes = static_cast<size_t>(config.totalM) * sizeof(RowIndexType);
    const size_t yBytes = static_cast<size_t>(config.batch) * static_cast<size_t>(config.n) * sizeof(CType);

    DeviceBuffer x(xBytes);
    DeviceBuffer weight(weightBytes);
    DeviceBuffer weightScale(weightScaleBytes);
    DeviceBuffer bias(biasBytes);
    DeviceBuffer xScale(xScaleBytes);
    DeviceBuffer groupList(groupListBytes);
    DeviceBuffer sharedInput(sharedBytes);
    DeviceBuffer logit(logitBytes);
    DeviceBuffer rowIndex(rowIndexBytes);
    DeviceBuffer y(yBytes);

    const std::string prefix = config.dataDir + "/";
    CopyToDevice(x, ReadFile(prefix + "input_x.bin", xBytes));
    CopyToDevice(weight, ReadFile(prefix + "input_weight.bin", weightBytes));
    CopyToDevice(weightScale, ReadFile(prefix + "input_scale_weight.bin", weightScaleBytes));
    CopyToDevice(bias, ReadFile(prefix + "input_bias.bin", biasBytes));
    CopyToDevice(xScale, ReadFile(prefix + "input_scale_x.bin", xScaleBytes));
    CopyToDevice(groupList, ReadFile(prefix + "input_group_list.bin", groupListBytes));
    CopyToDevice(sharedInput, ReadFile(prefix + "input_shared.bin", sharedBytes));
    CopyToDevice(logit, ReadFile(prefix + "input_logit.bin", logitBytes));
    CopyToDevice(rowIndex, ReadFile(prefix + "input_row_index.bin", rowIndexBytes));
    CopyToDevice(y, std::vector<uint8_t>(yBytes, 0U));

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    GmmFinalizeRoutingKernel<AType, CType, LogitType, RowIndexType>
        <<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream>>>(
            x.Get(), weight.Get(), weightScale.Get(), bias.Get(), xScale.Get(), groupList.Get(), sharedInput.Get(),
            logit.Get(), rowIndex.Get(), y.Get(), config.totalM, config.batch, config.n, config.k, config.groupNum,
            config.baseM, config.baseN, config.baseK, config.kAL1, config.kBL1, config.scaleKAL1, config.scaleKBL1,
            static_cast<uint32_t>(config.dbL0C), static_cast<uint32_t>(config.groupListType));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtDestroyStream(stream));

    std::vector<uint8_t> output(yBytes);
    ACL_CHECK(aclrtMemcpy(output.data(), yBytes, y.Get(), yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(config.outputPath, output);
    std::cout << "GMMFR case completed, output=" << config.outputPath << std::endl;
    return 0;
}

template <typename AType>
int DispatchOutput(const CaseConfig& config)
{
    if (config.outputDtype == "bfloat16") {
        if (config.logitDtype == "float32" && config.rowIndexDtype == "int32") {
            return RunCase<AType, bfloat16_t, float, int32_t>(config);
        }
        if (config.logitDtype == "bfloat16" && config.rowIndexDtype == "int32") {
            return RunCase<AType, bfloat16_t, bfloat16_t, int32_t>(config);
        }
    } else if (config.outputDtype == "float32" && config.logitDtype == "float32" && config.rowIndexDtype == "int64") {
        return RunCase<AType, float, float, int64_t>(config);
    }
    throw std::invalid_argument("unsupported output/logit/row index dtype combination");
}

int RunConfiguredCase(const CaseConfig& config)
{
    if (config.dtype == "mxfp4_e2m1") {
        return DispatchOutput<fp4x2_e2m1_t>(config);
    }
    if (config.dtype == "mxfp8_e4m3") {
        return DispatchOutput<fp8_e4m3fn_t>(config);
    }
    throw std::invalid_argument("unsupported input dtype: " + config.dtype);
}

CaseConfig ParseConfig(int argc, char** argv)
{
    if (argc != 21) {
        throw std::invalid_argument("expected 20 arguments from the example .conf file");
    }
    CaseConfig config{};
    config.groupNum = static_cast<uint32_t>(std::stoul(argv[1]));
    config.totalM = std::stoll(argv[2]);
    config.batch = std::stoll(argv[3]);
    config.n = std::stoll(argv[4]);
    config.k = std::stoll(argv[5]);
    config.baseM = static_cast<uint32_t>(std::stoul(argv[6]));
    config.baseN = static_cast<uint32_t>(std::stoul(argv[7]));
    config.baseK = static_cast<uint32_t>(std::stoul(argv[8]));
    config.kAL1 = static_cast<uint32_t>(std::stoul(argv[9]));
    config.kBL1 = static_cast<uint32_t>(std::stoul(argv[10]));
    config.scaleKAL1 = static_cast<uint32_t>(std::stoul(argv[11]));
    config.scaleKBL1 = static_cast<uint32_t>(std::stoul(argv[12]));
    config.dbL0C = static_cast<uint8_t>(std::stoul(argv[13]));
    config.groupListType = static_cast<uint8_t>(std::stoul(argv[14]));
    config.dtype = argv[15];
    config.outputDtype = argv[16];
    config.logitDtype = argv[17];
    config.rowIndexDtype = argv[18];
    config.dataDir = argv[19];
    config.outputPath = argv[20];
    return config;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const CaseConfig config = ParseConfig(argc, argv);
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        const int ret = RunConfiguredCase(config);
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        return ret;
    } catch (const std::exception& error) {
        std::cerr << "GMMFR example failed: " << error.what() << std::endl;
        return 2;
    }
}
