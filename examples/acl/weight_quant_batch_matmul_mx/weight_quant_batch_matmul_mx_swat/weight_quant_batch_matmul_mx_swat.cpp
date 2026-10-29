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
 * \file weight_quant_batch_matmul_mx_swat.cpp
 * \brief CSV-driven weight-only MX quantized matmul example for ND and NZ weights.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h"
#include "kernel_basic_intf.h"

#define ACL_CHECK(expr)                                                                                       \
    do {                                                                                                      \
        const aclError aclCheckResult = (expr);                                                               \
        if (aclCheckResult != ACL_SUCCESS) {                                                                  \
            std::fprintf(stderr, "ACL call failed: %s, error %d\n", #expr, static_cast<int>(aclCheckResult)); \
            std::exit(1);                                                                                     \
        }                                                                                                     \
    } while (0)

namespace {

constexpr uint64_t FP4_PACK_FACTOR = 2U;
constexpr uint8_t MX_IDENTITY_SCALE = 0x7FU;

struct ExampleConfig {
    int64_t m;
    int64_t k;
    int64_t n;
    uint64_t biasElements;
    bool weightNz;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    uint64_t tileShapeKL1;
    uint64_t tileShapeScaleKL1;
    uint64_t kBubSize;
    uint64_t nBubSize;
    uint64_t l1BufferNum;
    uint32_t blockNum;
    std::string dataDir;
};

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) / alignment * alignment;
}

bool ParseArgs(int argc, const char** argv, ExampleConfig& config)
{
    if (argc != 16) {
        std::fprintf(stderr,
                     "Usage: %s <m> <k> <n> <bias_elements> <layout> <base_m> <base_n> <base_k>"
                     " <tile_k_l1> <scale_k_l1> <k_bub> <n_bub> <l1_buffers> <block_num> <data_dir>\n",
                     argv[0]);
        return false;
    }

    config.m = std::atoll(argv[1]);
    config.k = std::atoll(argv[2]);
    config.n = std::atoll(argv[3]);
    config.biasElements = static_cast<uint64_t>(std::atoll(argv[4]));
    const std::string layout = argv[5];
    if (layout != "nd" && layout != "ND" && layout != "nz" && layout != "NZ") {
        std::fprintf(stderr, "Unsupported weight layout: %s\n", argv[5]);
        return false;
    }
    config.weightNz = layout == "nz" || layout == "NZ";
    config.baseM = static_cast<uint64_t>(std::atoll(argv[6]));
    config.baseN = static_cast<uint64_t>(std::atoll(argv[7]));
    config.baseK = static_cast<uint64_t>(std::atoll(argv[8]));
    config.tileShapeKL1 = static_cast<uint64_t>(std::atoll(argv[9]));
    config.tileShapeScaleKL1 = static_cast<uint64_t>(std::atoll(argv[10]));
    config.kBubSize = static_cast<uint64_t>(std::atoll(argv[11]));
    config.nBubSize = static_cast<uint64_t>(std::atoll(argv[12]));
    config.l1BufferNum = static_cast<uint64_t>(std::atoll(argv[13]));
    config.blockNum = static_cast<uint32_t>(std::atoll(argv[14]));
    config.dataDir = argv[15];

    if (config.m <= 0 || config.k <= 0 || config.n <= 0 || config.blockNum == 0U ||
        (config.biasElements != 0U && config.biasElements != static_cast<uint64_t>(config.n))) {
        std::fprintf(stderr, "Invalid shape, block, or bias configuration\n");
        return false;
    }
    return true;
}

bool ReadFile(const std::string& path, void* buffer, size_t size)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "Failed to open %s\n", path.c_str());
        return false;
    }
    const size_t readSize = std::fread(buffer, 1U, size, file);
    const int closeResult = std::fclose(file);
    const bool ok = readSize == size && closeResult == 0;
    if (!ok) {
        std::fprintf(stderr, "Failed to read %zu bytes from %s\n", size, path.c_str());
    }
    return ok;
}

bool WriteFile(const std::string& path, const void* buffer, size_t size)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
        return false;
    }
    const size_t writeSize = std::fwrite(buffer, 1U, size, file);
    const int closeResult = std::fclose(file);
    const bool ok = writeSize == size && closeResult == 0;
    if (!ok) {
        std::fprintf(stderr, "Failed to write %zu bytes to %s\n", size, path.c_str());
    }
    return ok;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(size)
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

    void CopyFromFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        if (!ReadFile(path, host.data(), size_)) {
            std::exit(1);
        }
        ACL_CHECK(aclrtMemcpy(data_, size_, host.data(), size_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyToFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        ACL_CHECK(aclrtMemcpy(host.data(), size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), size_)) {
            std::exit(1);
        }
    }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

template <bool WeightNz>
__global__ __aicore__ void QuantBatchMatmulMxKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm,
                                                    GM_ADDR scaleBGm, GM_ADDR cGm, int64_t m, int64_t k, int64_t n,
                                                    uint64_t baseM, uint64_t baseN, uint64_t baseK,
                                                    uint64_t tileShapeKL1, uint64_t tileShapeScaleKL1,
                                                    uint64_t kBubSize, uint64_t nBubSize, uint64_t l1BufferNum,
                                                    uint64_t biasElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using AType = fp8_e4m3fn_t;
    using BType = fp4x2_e2m1_t;
    using ScaleType = AscendC::fp8_e8m0_t;
    using CType = half;
    using BiasType = half;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<WeightNz, asc::te::zn_layout_ptn, asc::te::dn_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
        LayoutC>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

    typename Kernel::Params params{
        asc::te::make_shape(m, n, k),
        {aGm, scaleAGm, scaleBGm, cGm,
         asc::te::make_shape(static_cast<int64_t>(baseM), static_cast<int64_t>(baseN),
                             static_cast<int64_t>(tileShapeKL1), static_cast<int64_t>(tileShapeScaleKL1)),
         asc::te::make_shape(static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), static_cast<int64_t>(baseK)),
         l1BufferNum, biasElements != 0U},
        {bGm, biasGm, kBubSize, nBubSize},
        {baseM, baseN, 1U, 1U, 1U, 1U, 0U, 0U}};
    Kernel kernel;
    kernel(params);
}

template <bool WeightNz>
void RunCase(const ExampleConfig& config, aclrtStream stream)
{
    const uint64_t weightElements = WeightNz ? AlignUp(static_cast<uint64_t>(config.k), 32U) *
                                                   AlignUp(static_cast<uint64_t>(config.n), 16U) :
                                               static_cast<uint64_t>(config.k * config.n);
    const uint64_t scaleK = AlignUp(static_cast<uint64_t>(config.k), 64U) / 32U;
    const size_t aSize = static_cast<size_t>(config.m * config.k);
    const size_t bSize = static_cast<size_t>(weightElements / FP4_PACK_FACTOR);
    const size_t biasSize = static_cast<size_t>(config.n) * sizeof(half);
    const size_t scaleASize = static_cast<size_t>(config.m) * scaleK;
    const size_t scaleBSize = static_cast<size_t>(config.n) * scaleK;
    const size_t cSize = static_cast<size_t>(config.m * config.n) * sizeof(half);

    DeviceBuffer a(aSize);
    DeviceBuffer b(bSize);
    DeviceBuffer bias(biasSize);
    DeviceBuffer scaleA(scaleASize);
    DeviceBuffer scaleB(scaleBSize);
    DeviceBuffer c(cSize);
    a.CopyFromFile(config.dataDir + "/input_a.bin");
    b.CopyFromFile(config.dataDir + "/input_b.bin");
    bias.CopyFromFile(config.dataDir + "/bias.bin");
    scaleA.CopyFromFile(config.dataDir + "/scale_a.bin");
    scaleB.CopyFromFile(config.dataDir + "/scale_b.bin");
    c.CopyFromFile(config.dataDir + "/initial_c.bin");

    QuantBatchMatmulMxKernel<WeightNz><<<config.blockNum, 0, stream>>>(
        a.Get(), b.Get(), bias.Get(), scaleA.Get(), scaleB.Get(), c.Get(), config.m, config.k, config.n, config.baseM,
        config.baseN, config.baseK, config.tileShapeKL1, config.tileShapeScaleKL1, config.kBubSize, config.nBubSize,
        config.l1BufferNum, config.biasElements);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    c.CopyToFile(config.dataDir + "/npu_out.bin");
}

} // namespace

int main(int argc, const char** argv)
{
    ExampleConfig config{};
    if (!ParseArgs(argc, argv, config)) {
        return 1;
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));
    if (config.weightNz) {
        RunCase<true>(config, stream);
    } else {
        RunCase<false>(config, stream);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
