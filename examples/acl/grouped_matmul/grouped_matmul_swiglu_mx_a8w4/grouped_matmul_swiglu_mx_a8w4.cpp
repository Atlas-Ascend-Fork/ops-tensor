/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include <array>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "kernel_basic_intf.h"
#if defined(IMPL_STD_ASCENDC_STD_INT_IMPL_H) && !defined(IMPL_TENSOR_API_UTILS_INT_IMPL_H)
#define IMPL_TENSOR_API_UTILS_INT_IMPL_H
#endif
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "data_utils.h"

namespace {
constexpr int64_t PAIR = 2L;
constexpr int64_t ALIGNMENT = 128L;
constexpr int64_t MIN_EXAMPLE_K = 256L;
constexpr int64_t MIN_EXAMPLE_N = 256L;
constexpr int64_t SCALE_GROUP = 32L;
constexpr int64_t BASE_M = 256L;
constexpr int64_t BASE_N = 128L;
constexpr int64_t MAX_DIMENSION = 4096L;
constexpr int64_t MAX_EXPERTS = 8L;
constexpr uint32_t MODE_SPLIT = 2;
constexpr float CLAMP_LIMIT = 7.0F;
constexpr float ALPHA = 1.702F;
constexpr float BIAS = 1.0F;
constexpr int ARGUMENT_COUNT = 11;
enum Argument { PROGRAM, EXPERTS, M, N, K, SCALE_ALG, SMOOTH, INPUT, OUTPUT, OUTPUT_SCALE, GROUP_TYPE };
enum BufferIndex { X, WEIGHT, X_SCALE, WEIGHT_SCALE, GROUPS, SMOOTH_SCALE, Y, Y_SCALE, BUFFER_COUNT };
using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using Nd = asc::te::nd_ext_layout_ptn;
using Scale = fp8_e8m0_t;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using Mmad = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::GroupedMatmulSwigluWithWeightQuantMx, AscendC::Std::tuple<fp8_e4m3fn_t, Scale>,
    AscendC::Std::tuple<Nd, asc::te::scalea_nd_layout_ptn>, AscendC::Std::tuple<fp4x2_e2m1_t, Scale>,
    AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, float, Nd, bfloat16_t, Nd>;
using Prologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, true>;
using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, Scale, false, true, true>;
using Kernel = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<Shape, Mmad, Scheduler, Epilogue, Prologue, true>;

struct Config {
    int64_t experts, m, n, k;
    uint32_t algorithm, smooth, groupType;
    std::string input, output, outputScale;
};

Config ParseConfig(int argc, char** argv)
{
    if (argc != ARGUMENT_COUNT) {
        throw std::invalid_argument("expected: experts m n k scaleAlg smooth input outputY outputScale groupListType");
    }
    Config c{std::stoll(argv[EXPERTS]),
             std::stoll(argv[M]),
             std::stoll(argv[N]),
             std::stoll(argv[K]),
             static_cast<uint32_t>(std::stoul(argv[SCALE_ALG])),
             static_cast<uint32_t>(std::stoul(argv[SMOOTH])),
             static_cast<uint32_t>(std::stoul(argv[GROUP_TYPE])),
             argv[INPUT],
             argv[OUTPUT],
             argv[OUTPUT_SCALE]};
    if (c.experts < 1 || c.experts > MAX_EXPERTS || c.m < c.experts || c.m > MAX_DIMENSION || c.n < MIN_EXAMPLE_N ||
        c.n > MAX_DIMENSION || c.n % ALIGNMENT != 0 || c.k < MIN_EXAMPLE_K || c.k > MAX_DIMENSION ||
        c.k % ALIGNMENT != 0 || c.algorithm > 1 || c.smooth > 1 || c.groupType > 1) {
        throw std::invalid_argument("invalid shape, quantization algorithm, smoothing or groupListType");
    }
    return c;
}

__global__ __aicore__ void GmmsqA8w4(GM_ADDR x, GM_ADDR weight, GM_ADDR xs, GM_ADDR ws, GM_ADDR groups, GM_ADDR smooth,
                                     GM_ADDR y, GM_ADDR ys, int64_t m, int64_t n, int64_t k, uint32_t experts,
                                     uint32_t algorithm, uint32_t groupType)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    Kernel::Params p{};
    p.x = x;
    p.weight = weight;
    p.xScale = xs;
    p.weightScale = ws;
    p.groupList = groups;
    p.m = m;
    p.n = n;
    p.k = k;
    p.groupNum = experts;
    p.groupListType = groupType;
    p.scheduler.baseM = BASE_M;
    p.scheduler.baseN = BASE_N;
    p.epilogue.yGmAddr = y;
    p.epilogue.yScaleGmAddr = ys;
    p.epilogue.swigluMode = MODE_SPLIT;
    p.epilogue.clampLimit = CLAMP_LIMIT;
    p.epilogue.gluAlpha = ALPHA;
    p.epilogue.gluBias = BIAS;
    p.epilogue.scaleAlg = algorithm;
    p.epilogue.smoothScaleGmAddr = smooth;
    Kernel kernel;
    kernel(p);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    void Allocate(size_t bytes)
    {
        size = bytes;
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    void Load(const std::string& file, size_t bytes)
    {
        std::vector<uint8_t> host(bytes);
        if (!ReadFile(file, host.data(), bytes)) {
            throw std::runtime_error("cannot read " + file);
        }
        Allocate(bytes);
        ACL_CHECK(aclrtMemcpy(ptr, size, host.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void Save(const std::string& file)
    {
        std::vector<uint8_t> host(size);
        ACL_CHECK(aclrtMemcpy(host.data(), size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(file, host.data(), size)) {
            throw std::runtime_error("cannot write " + file);
        }
    }
};

void Run(const Config& c, aclrtStream stream)
{
    std::array<DeviceBuffer, BUFFER_COUNT> b;
    b[X].Load(c.input + "/x.bin", c.m * c.k);
    b[WEIGHT].Load(c.input + "/weight.bin", c.experts * c.k * c.n / PAIR);
    b[X_SCALE].Load(c.input + "/x_scale.bin", c.m * c.k / SCALE_GROUP);
    b[WEIGHT_SCALE].Load(c.input + "/weight_scale.bin", c.experts * c.n * c.k / SCALE_GROUP);
    b[GROUPS].Load(c.input + "/group_list.bin", c.experts * sizeof(int64_t));
    if (c.smooth != 0) {
        b[SMOOTH_SCALE].Load(c.input + "/smooth.bin", c.experts * c.n / PAIR * sizeof(float));
    }
    b[Y].Allocate(c.m * c.n / PAIR);
    b[Y_SCALE].Allocate(c.m * c.n / PAIR / SCALE_GROUP);
    auto address = [&](BufferIndex i) { return static_cast<uint8_t*>(b[i].ptr); };
    GmmsqA8w4<<<GetAicCoreNum(), 0, stream>>>(address(X), address(WEIGHT), address(X_SCALE), address(WEIGHT_SCALE),
                                              address(GROUPS), address(SMOOTH_SCALE), address(Y), address(Y_SCALE), c.m,
                                              c.n, c.k, c.experts, c.algorithm, c.groupType);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    b[Y].Save(c.output);
    b[Y_SCALE].Save(c.outputScale);
}
} // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = ParseConfig(argc, argv);
        aclrtStream stream = nullptr;
        ACLDeviceGuard guard(stream);
        Run(config, stream);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
