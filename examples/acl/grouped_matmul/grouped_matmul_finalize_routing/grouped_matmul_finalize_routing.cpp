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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "data_utils.h"
#include "platform/platform_ascendc.h"
#include "gmmfr_kernel.h"

using GmmfrExample::Arguments;
using GmmfrExample::Config;
using GmmfrExample::Flow;
namespace {
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

template <Flow Mode, typename A, typename B = A, typename Out = float, typename Logit = float,
          typename RowIndex = int64_t, bool Deterministic = false>
__global__ __aicore__ void QuantKernel(GM_ADDR x, GM_ADDR w, GM_ADDR sb, GM_ADDR bias, GM_ADDR sa, GM_ADDR groups,
                                       GM_ADDR shared, GM_ADDR logit, GM_ADDR rows, GM_ADDR y, GM_ADDR workspace,
                                       int64_t m, int64_t n, int64_t k, uint32_t e, uint32_t batch, uint32_t cores,
                                       uint32_t hasBias, uint32_t tokenScale, uint32_t sharedLen, uint32_t sharedOffset,
                                       uint32_t groupType, uint32_t workspaceRows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    const Config config{m,
                        n,
                        k,
                        e,
                        batch,
                        cores,
                        hasBias,
                        tokenScale,
                        sharedLen,
                        sharedOffset,
                        groupType,
                        GmmfrExample::DEFAULT_SHARED_INPUT_WEIGHT,
                        workspaceRows};
    GmmfrExample::Run<Mode, A, B, Deterministic, Out, Logit, RowIndex>(
        {x, w, sb, bias, sa, groups, shared, logit, rows, y, workspace}, config);
}

template <Flow Mode, typename A, typename B = A, typename Out = float, typename Logit = float,
          typename RowIndex = int64_t, bool Deterministic = false>
void Execute(const Arguments& a, const Config& c, aclrtStream stream)
{
    QuantKernel<Mode, A, B, Out, Logit, RowIndex, Deterministic>
        <<<c.cores, 0, stream>>>(a.x, a.weight, a.scaleB, a.bias, a.scaleA, a.groupList, a.sharedInput, a.logit,
                                 a.rowIndex, a.y, a.workspace, c.m, c.n, c.k, c.groups, c.batch, c.cores, c.hasBias,
                                 c.tokenScale, c.sharedLen, c.sharedOffset, c.groupListType, c.workspaceRows);
}

size_t Bytes(std::initializer_list<uint64_t> dimensions, size_t elementBytes = 1)
{
    size_t size = elementBytes;
    for (uint64_t dimension : dimensions) {
        if (dimension != 0 && size > std::numeric_limits<size_t>::max() / dimension) {
            throw std::overflow_error("tensor byte count overflows size_t");
        }
        size *= dimension;
    }
    return size;
}

uint32_t ReadU32(const char* text)
{
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("integer exceeds uint32 range");
    }
    return static_cast<uint32_t>(value);
}

int RunCase(int argc, char** argv)
{
    if (argc < 18 || argc > 20) {
        throw std::invalid_argument("expected quantMode,weightFormat,dtype,outputDtype,logitDtype,"
                                    "E,M,K,N,batch,bias,tokenScale,sharedLen,sharedOffset,groupType,dataDir,outputPath"
                                    "[,deterministic[,workspaceRows]]");
    }
    const std::string quantMode = argv[1], weightFormat = argv[2], dtype = argv[3];
    const std::string outputDtype = argv[4], logitDtype = argv[5];
    if ((quantMode != "pertoken" && quantMode != "mx" && quantMode != "a8w4") ||
        (weightFormat != "ND" && weightFormat != "NZ") || (quantMode != "mx" && weightFormat != "NZ")) {
        throw std::invalid_argument("unsupported quantization mode/weight format combination");
    }
    const std::string mode = quantMode == "mx" ? (weightFormat == "ND" ? "mx_nd" : "mx_nz") : quantMode;
    const bool bf16Output = outputDtype == "bfloat16";
    const bool deterministic = argc >= 19 && ReadU32(argv[18]) != 0;
    if (deterministic && (mode != "pertoken" || dtype != "int8")) {
        throw std::invalid_argument("deterministic example requires pertoken INT8");
    }
    const bool bf16Logit = logitDtype == "bfloat16";
    const bool nzTypes = mode == "mx_nz" &&
                         ((dtype == "fp4_e2m1" && bf16Output && bf16Logit) || (dtype == "fp8_e4m3fn" && !bf16Logit));
    if ((outputDtype != "float32" && !bf16Output) || (logitDtype != "float32" && !bf16Logit) ||
        (mode == "mx_nz" ? !nzTypes : (bf16Output || bf16Logit))) {
        throw std::invalid_argument("unsupported output/logit type for selected quantization mode");
    }
    Config c{};
    c.groups = ReadU32(argv[6]);
    c.m = std::stoll(argv[7]);
    c.k = std::stoll(argv[8]);
    c.n = std::stoll(argv[9]);
    c.batch = ReadU32(argv[10]);
    c.hasBias = ReadU32(argv[11]);
    c.tokenScale = ReadU32(argv[12]);
    c.sharedLen = ReadU32(argv[13]);
    c.sharedOffset = ReadU32(argv[14]);
    c.groupListType = ReadU32(argv[15]);
    c.workspaceRows = argc == 20 ? ReadU32(argv[19]) : 0;
    if (c.workspaceRows != 0 &&
        (!deterministic || c.workspaceRows < GmmfrExample::BASE_M || c.workspaceRows % GmmfrExample::BASE_M != 0)) {
        throw std::invalid_argument("workspaceRows requires deterministic mode and a positive multiple of baseM");
    }
    if (c.m <= 0 || c.n <= 0 || c.k <= 0 || c.groups == 0 || c.batch == 0 || c.k % 64 != 0 || c.n % 64 != 0 ||
        c.n > std::numeric_limits<int32_t>::max() || c.hasBias > 1 || c.tokenScale > 1 || c.groupListType > 1 ||
        c.sharedOffset > c.batch || c.sharedLen > c.batch - c.sharedOffset ||
        (mode != "pertoken" && mode != "mx_nd" && mode != "mx_nz" && mode != "a8w4")) {
        throw std::invalid_argument("invalid example configuration; K and N must be multiples of 64");
    }
    const bool supported = (mode == "pertoken" && (dtype == "int8" || dtype == "fp8_e4m3fn" || dtype == "hifloat8")) ||
                           (mode == "mx_nd" && (dtype == "fp8_e4m3fn" || dtype == "fp8_e5m2" || dtype == "fp4_e2m1")) ||
                           (mode == "a8w4" && dtype == "fp8_e4m3fn") || nzTypes;
    if (!supported) {
        throw std::invalid_argument("unsupported mode/dtype combination");
    }
    c.cores = static_cast<uint32_t>(GetAicCoreNum());
    const std::string dir = argv[16];
    const bool mx = mode != "pertoken";
    const bool fp4a = dtype == "fp4_e2m1";
    const bool fp4b = fp4a || mode == "a8w4";
    const size_t scaleWidth = mx ? static_cast<size_t>(c.k / 32) : sizeof(float);
    const std::vector<size_t> expected{
        Bytes({static_cast<uint64_t>(c.m), static_cast<uint64_t>(c.k)}) / (fp4a ? 2U : 1U),
        Bytes({c.groups, static_cast<uint64_t>(c.k), static_cast<uint64_t>(c.n)}) / (fp4b ? 2U : 1U),
        Bytes({c.groups, static_cast<uint64_t>(c.n)}, scaleWidth),
        Bytes({c.groups, static_cast<uint64_t>(c.n)}, sizeof(bfloat16_t)),
        Bytes({static_cast<uint64_t>(c.m)}, scaleWidth),
        c.groups * sizeof(int64_t),
        Bytes({std::max(1U, c.sharedLen), static_cast<uint64_t>(c.n)}, sizeof(bfloat16_t)),
        Bytes({static_cast<uint64_t>(c.m)}, bf16Logit ? sizeof(bfloat16_t) : sizeof(float)),
        Bytes({static_cast<uint64_t>(c.m)}, bf16Output ? sizeof(int32_t) : sizeof(int64_t))};
    std::vector<std::vector<uint8_t>> inputs;
    for (const auto* name : {"x", "weight", "scale_b", "bias", "scale_a", "groups", "shared", "logit", "rows"}) {
        const std::string path = dir + "/" + name + ".bin";
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() <= 0) {
            throw std::runtime_error("missing or empty input: " + path);
        }
        if (static_cast<size_t>(f.tellg()) != expected.at(inputs.size())) {
            throw std::runtime_error("input size does not match mode/shape: " + path);
        }
        inputs.emplace_back(ReadFile(path, expected.at(inputs.size())));
    }
    std::vector<DeviceBuffer*> buffers;
    // Allocation lifetime is owned by unique_ptr; the pointer list supplies the launch arguments.
    std::vector<std::unique_ptr<DeviceBuffer>> owners;
    for (const auto& host : inputs) {
        owners.emplace_back(std::make_unique<DeviceBuffer>(host.size()));
        CopyToDevice(*owners.back(), host);
        buffers.push_back(owners.back().get());
    }
    DeviceBuffer y(Bytes({c.batch, static_cast<uint64_t>(c.n)}, bf16Output ? sizeof(bfloat16_t) : sizeof(float)));
    CopyToDevice(y, std::vector<uint8_t>(y.Size(), 0));
    const uint64_t workspaceRows = c.workspaceRows != 0 ? c.workspaceRows :
                                                          ((static_cast<uint64_t>(c.m) + GmmfrExample::BASE_M - 1) /
                                                           GmmfrExample::BASE_M) *
                                                              GmmfrExample::BASE_M;
    const size_t workspaceBytes = deterministic ? Bytes({workspaceRows, static_cast<uint64_t>(c.n)}, sizeof(float)) : 1;
    if (deterministic && workspaceBytes > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("deterministic workspace exceeds uint32 byte capacity");
    }
    DeviceBuffer workspace(workspaceBytes);
    const Arguments a{buffers[0]->Get(), buffers[1]->Get(), buffers[2]->Get(), buffers[3]->Get(),
                      buffers[4]->Get(), buffers[5]->Get(), buffers[6]->Get(), buffers[7]->Get(),
                      buffers[8]->Get(), y.Get(),           workspace.Get()};
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    if (deterministic)
        Execute<Flow::PERTOKEN, int8_t, int8_t, float, float, int64_t, true>(a, c, stream);
    else if (mode == "pertoken" && dtype == "int8")
        Execute<Flow::PERTOKEN, int8_t>(a, c, stream);
    else if (mode == "pertoken" && dtype == "fp8_e4m3fn")
        Execute<Flow::PERTOKEN, fp8_e4m3fn_t>(a, c, stream);
    else if (mode == "pertoken" && dtype == "hifloat8")
        Execute<Flow::PERTOKEN, hifloat8_t>(a, c, stream);
    else if (mode == "mx_nd" && dtype == "fp8_e4m3fn")
        Execute<Flow::MX_ND, fp8_e4m3fn_t>(a, c, stream);
    else if (mode == "mx_nd" && dtype == "fp8_e5m2")
        Execute<Flow::MX_ND, fp8_e5m2_t>(a, c, stream);
    else if (mode == "mx_nd" && dtype == "fp4_e2m1")
        Execute<Flow::MX_ND, fp4x2_e2m1_t>(a, c, stream);
    else if (mode == "mx_nz" && dtype == "fp4_e2m1")
        Execute<Flow::MX_NZ, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, bfloat16_t, int32_t>(a, c, stream);
    else if (mode == "mx_nz" && bf16Output)
        Execute<Flow::MX_NZ, fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float, int32_t>(a, c, stream);
    else if (mode == "mx_nz")
        Execute<Flow::MX_NZ, fp8_e4m3fn_t>(a, c, stream);
    else if (mode == "a8w4" && dtype == "fp8_e4m3fn")
        Execute<Flow::A8W4, fp8_e4m3fn_t, fp4x2_e2m1_t>(a, c, stream);
    else {
        throw std::invalid_argument("unsupported mode/dtype combination");
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtDestroyStream(stream));
    std::vector<uint8_t> result(y.Size());
    ACL_CHECK(aclrtMemcpy(result.data(), result.size(), y.Get(), y.Size(), ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(argv[17], result);
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        const int result = RunCase(argc, argv);
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        return result;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
