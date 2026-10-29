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
 * \file quant_batch_matmul_cube.cpp
 * \brief CSV-driven Cube and per-tensor StreamK quant batch matmul example.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"

using AscendC::DT_BF16;
using AscendC::DT_FLOAT;
using AscendC::DT_FLOAT16;

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace {

constexpr uint32_t QUANT_MODE_DEFAULT = 0U;
constexpr uint32_t QUANT_MODE_PERTENSOR = 1U;
constexpr uint32_t QUANT_MODE_PERCHANNEL = 2U;
constexpr uint64_t MIN_L1_BUFFER_NUM = 2U;
constexpr uint64_t MAX_L1_BUFFER_NUM = 4U;
constexpr uint32_t BLOCK_NUM = 32U;
constexpr uint64_t MAX_BASE_M = 256U;
constexpr uint64_t MAX_BASE_N = 256U;
constexpr uint64_t MAX_BASE_K = 128U;
constexpr uint64_t DEQ_SCALE_MASK = 0xFFFFE000ULL;
constexpr uint64_t DEQ_SCALE_FLAG = 1ULL << 46;
constexpr uint32_t GE_DTYPE_FLOAT = 0U;
constexpr uint64_t STREAMK_WORKSPACE_TILE_BYTES = 256UL * 256UL * sizeof(float);
constexpr uint64_t STREAMK_WORKSPACE_OVERHEAD_BYTES = 20UL * 1024UL * 1024UL;
constexpr uint32_t L2_CACHE_DEFAULT_VALUE = 0U;

struct ExampleConfig {
    int64_t batch;
    int64_t m;
    int64_t k;
    int64_t n;
    std::string aType;
    std::string bType;
    std::string cType;
    uint64_t biasElements;
    std::string biasType;
    bool transA;
    bool transB;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    std::string x2ScaleType;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    uint64_t kL1;
    uint64_t l1BufferNum;
    bool aFullLoad;
    bool withoutBatch;
    bool pertensorStreamK;
    std::string dataDir;
};

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool ParseBool(const char* text, bool& value)
{
    const std::string normalized = ToLower(text);
    if (normalized == "true" || normalized == "1") {
        value = true;
        return true;
    }
    if (normalized == "false" || normalized == "0") {
        value = false;
        return true;
    }
    return false;
}

bool ParseQuantMode(const char* text, uint32_t& value)
{
    const std::string normalized = ToLower(text);
    if (normalized == "default" || normalized == "0") {
        value = QUANT_MODE_DEFAULT;
        return true;
    }
    if (normalized == "pertensor" || normalized == "per_tensor" || normalized == "1") {
        value = QUANT_MODE_PERTENSOR;
        return true;
    }
    if (normalized == "perchannel" || normalized == "per_channel" || normalized == "2") {
        value = QUANT_MODE_PERCHANNEL;
        return true;
    }
    return false;
}

bool ParseKernelVariant(const char* text, bool& withoutBatch, bool& pertensorStreamK)
{
    const std::string normalized = ToLower(text);
    withoutBatch = false;
    pertensorStreamK = false;
    if (normalized == "batch") {
        return true;
    }
    if (normalized == "without_batch") {
        withoutBatch = true;
        return true;
    }
    if (normalized == "pertensor_streamk") {
        pertensorStreamK = true;
        return true;
    }
    return false;
}

bool ParseArgs(int argc, const char** argv, ExampleConfig& config)
{
    if (argc != 23) {
        std::fprintf(stderr,
                     "Usage: %s <batch> <M> <K> <N> <AType> <BType> <CType> <bias> <biasType> <transA> <transB>"
                     " <x1quantmode> <x2quantmode> <x2ScaleType> <baseM> <baseN> <baseK> <kL1> <l1BufferNum>"
                     " <aFullLoad> <kernel_variant> <data_dir>\n",
                     argv[0]);
        return false;
    }

    config.batch = std::atoll(argv[1]);
    config.m = std::atoll(argv[2]);
    config.k = std::atoll(argv[3]);
    config.n = std::atoll(argv[4]);
    config.aType = ToLower(argv[5]);
    config.bType = ToLower(argv[6]);
    config.cType = ToLower(argv[7]);
    config.biasElements = static_cast<uint64_t>(std::atoll(argv[8]));
    config.biasType = ToLower(argv[9]);
    config.x2ScaleType = ToLower(argv[14]);
    config.baseM = static_cast<uint64_t>(std::atoll(argv[15]));
    config.baseN = static_cast<uint64_t>(std::atoll(argv[16]));
    config.baseK = static_cast<uint64_t>(std::atoll(argv[17]));
    config.kL1 = static_cast<uint64_t>(std::atoll(argv[18]));
    config.l1BufferNum = static_cast<uint64_t>(std::atoll(argv[19]));
    config.dataDir = argv[22];

    if (!ParseBool(argv[10], config.transA) || !ParseBool(argv[11], config.transB) ||
        !ParseBool(argv[20], config.aFullLoad)) {
        std::fprintf(stderr, "transA/transB/aFullLoad must be true, false, 1, or 0\n");
        return false;
    }
    if (!ParseKernelVariant(argv[21], config.withoutBatch, config.pertensorStreamK)) {
        std::fprintf(stderr, "kernel_variant must be batch, without_batch, or pertensor_streamk\n");
        return false;
    }
    if (!ParseQuantMode(argv[12], config.x1QuantMode) || !ParseQuantMode(argv[13], config.x2QuantMode)) {
        std::fprintf(stderr, "quant mode must be default, pertensor, or perchannel\n");
        return false;
    }
    const bool validBias = config.biasElements == 0U || config.biasElements == static_cast<uint64_t>(config.n);
    const bool validBase = config.baseM > 0U && config.baseM <= MAX_BASE_M && config.baseN > 0U &&
                           config.baseN <= MAX_BASE_N && config.baseK > 0U && config.baseK <= MAX_BASE_K;
    const bool validL1BufferNum = config.l1BufferNum >= MIN_L1_BUFFER_NUM && config.l1BufferNum <= MAX_L1_BUFFER_NUM;
    const bool isHiFloat8ToBf16 = (config.aType == "hifloat8" || config.aType == "hifloat8_t") &&
                                  (config.bType == "hifloat8" || config.bType == "hifloat8_t") &&
                                  (config.cType == "bfloat16" || config.cType == "bfloat16_t" ||
                                   config.cType == "bf16") &&
                                  (config.biasType == "float" || config.biasType == "float32");
    const bool isInt8Input = (config.aType == "int8" || config.aType == "int8_t") &&
                             (config.bType == "int8" || config.bType == "int8_t") &&
                             (config.biasType == "int32" || config.biasType == "int32_t");
    const bool isInt8ToInt32 = isInt8Input && (config.cType == "int32" || config.cType == "int32_t");
    const bool isInt8ToFp16StreamK = config.pertensorStreamK && isInt8Input &&
                                     (config.cType == "float16" || config.cType == "float16_t" ||
                                      config.cType == "half");
    const bool validHiFloat8QuantMode = (config.x2QuantMode == QUANT_MODE_PERTENSOR &&
                                         (config.x1QuantMode == QUANT_MODE_DEFAULT ||
                                          config.x1QuantMode == QUANT_MODE_PERTENSOR)) ||
                                        (config.x1QuantMode == QUANT_MODE_DEFAULT &&
                                         config.x2QuantMode == QUANT_MODE_PERCHANNEL);
    const bool validInt8QuantMode = config.x1QuantMode == QUANT_MODE_DEFAULT &&
                                    config.x2QuantMode == QUANT_MODE_DEFAULT;
    const bool validPertensorStreamKMode = config.x1QuantMode == QUANT_MODE_DEFAULT &&
                                           config.x2QuantMode == QUANT_MODE_PERTENSOR;
    const bool isFloatScale = config.x2ScaleType == "float" || config.x2ScaleType == "float32";
    const bool isUint64Scale = config.x2ScaleType == "uint64" || config.x2ScaleType == "uint64_t";
    const bool validTypeAndMode = (!config.pertensorStreamK &&
                                   ((isHiFloat8ToBf16 && validHiFloat8QuantMode &&
                                     (config.x2QuantMode == QUANT_MODE_PERCHANNEL ? isUint64Scale : isFloatScale)) ||
                                    (isInt8ToInt32 && validInt8QuantMode && isFloatScale))) ||
                                  (isInt8ToFp16StreamK && validPertensorStreamKMode && isFloatScale);
    if (config.batch <= 0 || config.m <= 0 || config.k <= 0 || config.n <= 0 || config.kL1 == 0U || !validBias ||
        !validBase || !validL1BufferNum) {
        std::fprintf(stderr, "Invalid shape/bias/tiling: bias must be 0 or N, maximum baseM*baseK*baseN is 256*128*256,"
                             " and l1BufferNum must be 2, 3, or 4\n");
        return false;
    }
    if ((config.withoutBatch || config.pertensorStreamK) && config.batch != 1) {
        std::fprintf(stderr, "without_batch and pertensor_streamk require batch=1\n");
        return false;
    }
    if (config.pertensorStreamK && (config.biasElements != 0U || config.transA || config.transB || config.aFullLoad)) {
        std::fprintf(stderr, "pertensor_streamk requires no bias, no transpose, and aFullLoad=false\n");
        return false;
    }
    if (!validTypeAndMode) {
        std::fprintf(stderr, "Supported configurations are hifloat8_t/hifloat8_t -> bfloat16_t with float bias and"
                             " TT/TC scale; int8_t/int8_t -> int32_t with default/default quant mode; or the"
                             " int8_t/int8_t -> float16 pertensor_streamk variant\n");
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
    const int extra = std::fgetc(file);
    const int closeResult = std::fclose(file);
    if (readSize != size || extra != EOF || closeResult != 0) {
        std::fprintf(stderr, "Unexpected file size for %s (expected %zu bytes)\n", path.c_str(), size);
        return false;
    }
    return true;
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
    return writeSize == size && closeResult == 0;
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

    void CopyFromHost(const void* host, size_t size) const
    {
        if (size != size_) {
            std::fprintf(stderr, "Host/device copy size mismatch: %zu vs %zu\n", size, size_);
            std::exit(1);
        }
        ACL_CHECK(aclrtMemcpy(data_, size_, host, size_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void CopyFromFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        if (!ReadFile(path, host.data(), size_)) {
            std::exit(1);
        }
        CopyFromHost(host.data(), size_);
    }

    void CopyToFile(const std::string& path) const
    {
        std::vector<uint8_t> host(size_);
        ACL_CHECK(aclrtMemcpy(host.data(), size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), size_)) {
            std::fprintf(stderr, "Failed to write %s\n", path.c_str());
            std::exit(1);
        }
    }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

std::vector<uint64_t> EncodePerChannelScale(const std::string& path, int64_t n)
{
    std::vector<float> fp32Scale(static_cast<size_t>(n));
    if (!ReadFile(path, fp32Scale.data(), fp32Scale.size() * sizeof(float))) {
        std::exit(1);
    }
    std::vector<uint64_t> encoded(fp32Scale.size());
    for (size_t i = 0; i < fp32Scale.size(); ++i) {
        uint32_t bits = 0U;
        std::memcpy(&bits, &fp32Scale[i], sizeof(bits));
        encoded[i] = (static_cast<uint64_t>(bits) & DEQ_SCALE_MASK) | DEQ_SCALE_FLAG;
    }
    return encoded;
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, bool TransA,
          bool TransB, uint64_t FullLoadMode>
__global__ __aicore__ void QuantBatchMatmulCubeKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleAGm, GM_ADDR scaleBGm,
                                                      GM_ADDR biasGm, GM_ADDR cGm, int64_t batch, int64_t m, int64_t k,
                                                      int64_t n, uint64_t baseM, uint64_t baseN, uint64_t baseK,
                                                      uint64_t kL1, uint64_t l1BufferNum, uint64_t biasElements,
                                                      uint32_t x1QuantMode, uint32_t x2QuantMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    typename Kernel::Params params{};
    params.problemShape = {m, n, k, batch};
    params.mmadParams.aGmAddr = aGm;
    params.mmadParams.bGmAddr = bGm;
    params.mmadParams.cGmAddr = cGm;
    params.mmadParams.biasGmAddr = biasGm;
    params.mmadParams.scaleAGmAddr = scaleAGm;
    params.mmadParams.scaleBGmAddr = scaleBGm;

    params.schParams.baseM = static_cast<int64_t>(baseM);
    params.schParams.baseN = static_cast<int64_t>(baseN);
    params.schParams.mTailTile = 1;
    params.schParams.nTailTile = 1;
    params.schParams.mBaseTailSplitCnt = 1;
    params.schParams.nBaseTailSplitCnt = 1;
    params.schParams.mTailMain = 0;
    params.schParams.nTailMain = 0;

    params.qbmmParams.batchA1 = 1U;
    params.qbmmParams.batchA2 = 1U;
    params.qbmmParams.batchA3 = 1U;
    params.qbmmParams.batchA4 = static_cast<uint32_t>(batch);
    params.qbmmParams.batchB1 = 1U;
    params.qbmmParams.batchB2 = 1U;
    params.qbmmParams.batchB3 = 1U;
    params.qbmmParams.batchB4 = static_cast<uint32_t>(batch);
    params.qbmmParams.batchC1 = 1U;
    params.qbmmParams.batchC2 = 1U;
    params.qbmmParams.batchC3 = 1U;
    params.qbmmParams.batchC4 = static_cast<uint32_t>(batch);
    params.qbmmParams.biasThreeDim = 0U;
    params.qbmmParams.x1QuantMode = x1QuantMode;
    params.qbmmParams.x2QuantMode = x2QuantMode;
    params.qbmmParams.kAL1 = static_cast<uint32_t>(kL1);
    params.qbmmParams.kBL1 = static_cast<uint32_t>(kL1);
    params.qbmmParams.nBufferNum = static_cast<uint32_t>(l1BufferNum);
    params.qbmmParams.baseM = static_cast<uint32_t>(baseM);
    params.qbmmParams.baseN = static_cast<uint32_t>(baseN);
    params.qbmmParams.baseK = static_cast<uint32_t>(baseK);
    params.qbmmParams.isBias = biasElements == 0U ? 0U : 1U;
    params.qbmmParams.dbL0C = 1U;
    params.qbmmParams.bMustHitL2 = 1U;

    Kernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, bool TransA,
          bool TransB, uint64_t FullLoadMode>
__global__ __aicore__ void QuantBatchMatmulCubeWithoutBatchKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleAGm,
                                                                  GM_ADDR scaleBGm, GM_ADDR biasGm, GM_ADDR cGm,
                                                                  int64_t m, int64_t k, int64_t n, uint64_t baseM,
                                                                  uint64_t baseN, uint64_t baseK, uint64_t kL1,
                                                                  uint64_t l1BufferNum, uint64_t biasElements,
                                                                  uint32_t x1QuantMode, uint32_t x2QuantMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        FullLoadMode, false, Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    typename Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams = {aGm, bGm, cGm, biasGm, scaleAGm, scaleBGm};
    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    params.qbmmParams.x1QuantMode = x1QuantMode;
    params.qbmmParams.x2QuantMode = x2QuantMode;
    params.qbmmParams.kAL1 = static_cast<uint32_t>(kL1);
    params.qbmmParams.kBL1 = static_cast<uint32_t>(kL1);
    params.qbmmParams.nBufferNum = static_cast<uint32_t>(l1BufferNum);
    params.qbmmParams.baseM = static_cast<uint32_t>(baseM);
    params.qbmmParams.baseN = static_cast<uint32_t>(baseN);
    params.qbmmParams.baseK = static_cast<uint32_t>(baseK);
    params.qbmmParams.isBias = biasElements == 0U ? 0U : 1U;
    params.qbmmParams.dbL0C = 1U;
    params.qbmmParams.bMustHitL2 = 1U;

    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void QuantBatchMatmulPertensorStreamKKernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleBGm,
                                                                  GM_ADDR cGm, GM_ADDR workspaceGm, int64_t m,
                                                                  int64_t k, int64_t n, uint32_t usedCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                                    Layout, half, Layout, int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename BlockMmad::WorkspaceType,
                                                                                    half, DispatchPolicy, float, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.blockMmadParams = {aGm, bGm, cGm, nullptr, nullptr, scaleBGm};
    params.epilogueParams = {cGm, workspaceGm, scaleBGm, nullptr, nullptr, false, GE_DTYPE_FLOAT};
    params.schParams = {usedCoreNum, m, n, 64, 64, 64, 0U, L2_CACHE_DEFAULT_VALUE};
    Kernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType, bool TransA,
          bool TransB, uint64_t FullLoadMode>
void Launch(const ExampleConfig& config, aclrtStream stream, const DeviceBuffer& a, const DeviceBuffer& b,
            const DeviceBuffer& scaleA, const DeviceBuffer& scaleB, const DeviceBuffer& bias, const DeviceBuffer& c)
{
    if (config.withoutBatch) {
        QuantBatchMatmulCubeWithoutBatchKernel<AType, BType, CType, BiasType, X2ScaleType, TransA, TransB, FullLoadMode>
            <<<BLOCK_NUM, 0, stream>>>(a.Get(), b.Get(), scaleA.Get(), scaleB.Get(), bias.Get(), c.Get(), config.m,
                                       config.k, config.n, config.baseM, config.baseN, config.baseK, config.kL1,
                                       config.l1BufferNum, config.biasElements, config.x1QuantMode, config.x2QuantMode);
        return;
    }
    QuantBatchMatmulCubeKernel<AType, BType, CType, BiasType, X2ScaleType, TransA, TransB, FullLoadMode>
        <<<BLOCK_NUM, 0, stream>>>(a.Get(), b.Get(), scaleA.Get(), scaleB.Get(), bias.Get(), c.Get(), config.batch,
                                   config.m, config.k, config.n, config.baseM, config.baseN, config.baseK, config.kL1,
                                   config.l1BufferNum, config.biasElements, config.x1QuantMode, config.x2QuantMode);
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType,
          uint64_t FullLoadMode>
void DispatchTranspose(const ExampleConfig& config, aclrtStream stream, const DeviceBuffer& a, const DeviceBuffer& b,
                       const DeviceBuffer& scaleA, const DeviceBuffer& scaleB, const DeviceBuffer& bias,
                       const DeviceBuffer& c)
{
    if (config.transA && config.transB) {
        Launch<AType, BType, CType, BiasType, X2ScaleType, true, true, FullLoadMode>(config, stream, a, b, scaleA,
                                                                                     scaleB, bias, c);
    } else if (config.transA) {
        Launch<AType, BType, CType, BiasType, X2ScaleType, true, false, FullLoadMode>(config, stream, a, b, scaleA,
                                                                                      scaleB, bias, c);
    } else if (config.transB) {
        Launch<AType, BType, CType, BiasType, X2ScaleType, false, true, FullLoadMode>(config, stream, a, b, scaleA,
                                                                                      scaleB, bias, c);
    } else {
        Launch<AType, BType, CType, BiasType, X2ScaleType, false, false, FullLoadMode>(config, stream, a, b, scaleA,
                                                                                       scaleB, bias, c);
    }
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType>
void RunTypedCase(const ExampleConfig& config, aclrtStream stream)
{
    const size_t aSize = static_cast<size_t>(config.batch * config.m * config.k) * sizeof(AType);
    const size_t bSize = static_cast<size_t>(config.batch * config.k * config.n) * sizeof(BType);
    const bool perChannel = config.x2QuantMode == QUANT_MODE_PERCHANNEL;
    const size_t scaleBSize = perChannel ? static_cast<size_t>(config.n) * sizeof(X2ScaleType) : sizeof(X2ScaleType);
    const size_t biasSize = static_cast<size_t>(config.n) * sizeof(BiasType);
    const size_t cSize = static_cast<size_t>(config.batch * config.m * config.n) * sizeof(CType);

    DeviceBuffer a(aSize);
    DeviceBuffer b(bSize);
    DeviceBuffer scaleA(sizeof(float));
    DeviceBuffer scaleB(scaleBSize);
    DeviceBuffer bias(biasSize);
    DeviceBuffer c(cSize);
    a.CopyFromFile(config.dataDir + "/input_a.bin");
    b.CopyFromFile(config.dataDir + "/input_b.bin");
    scaleA.CopyFromFile(config.dataDir + "/scale_a.bin");
    if (perChannel) {
        const std::vector<uint64_t> encoded = EncodePerChannelScale(config.dataDir + "/scale_b.bin", config.n);
        scaleB.CopyFromHost(encoded.data(), encoded.size() * sizeof(uint64_t));
    } else {
        scaleB.CopyFromFile(config.dataDir + "/scale_b.bin");
    }
    bias.CopyFromFile(config.dataDir + "/bias.bin");

    if (config.pertensorStreamK) {
        const int64_t aicCoreNum = GetAicCoreNum();
        if (aicCoreNum <= 0) {
            std::fprintf(stderr, "No AIC cores are available for pertensor_streamk\n");
            std::exit(1);
        }
        const uint32_t launchBlocks = static_cast<uint32_t>(aicCoreNum);
        const size_t workspaceSize = static_cast<size_t>(launchBlocks) * STREAMK_WORKSPACE_TILE_BYTES +
                                     STREAMK_WORKSPACE_OVERHEAD_BYTES;
        DeviceBuffer workspace(workspaceSize);
        ACL_CHECK(aclrtMemset(c.Get(), cSize, 0, cSize));
        ACL_CHECK(aclrtMemset(workspace.Get(), workspaceSize, 0, workspaceSize));
        QuantBatchMatmulPertensorStreamKKernel<<<launchBlocks, 0, stream>>>(
            a.Get(), b.Get(), scaleB.Get(), c.Get(), workspace.Get(), config.m, config.k, config.n, launchBlocks);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        c.CopyToFile(config.dataDir + "/npu_out.bin");
        return;
    }

    if (config.aFullLoad) {
        DispatchTranspose<AType, BType, CType, BiasType, X2ScaleType, Blaze::Gemm::A_FULL_LOAD_MODE>(
            config, stream, a, b, scaleA, scaleB, bias, c);
    } else {
        DispatchTranspose<AType, BType, CType, BiasType, X2ScaleType, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
            config, stream, a, b, scaleA, scaleB, bias, c);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    c.CopyToFile(config.dataDir + "/npu_out.bin");
}

void RunCase(const ExampleConfig& config, aclrtStream stream)
{
    const bool isHiFloat8ToBf16 = (config.aType == "hifloat8" || config.aType == "hifloat8_t") &&
                                  (config.bType == "hifloat8" || config.bType == "hifloat8_t") &&
                                  (config.cType == "bfloat16" || config.cType == "bfloat16_t" ||
                                   config.cType == "bf16") &&
                                  (config.biasType == "float" || config.biasType == "float32");
    const bool isInt8ToInt32 = (config.aType == "int8" || config.aType == "int8_t") &&
                               (config.bType == "int8" || config.bType == "int8_t") &&
                               (config.cType == "int32" || config.cType == "int32_t") &&
                               (config.biasType == "int32" || config.biasType == "int32_t");
    const bool isInt8ToFp16 = (config.aType == "int8" || config.aType == "int8_t") &&
                              (config.bType == "int8" || config.bType == "int8_t") &&
                              (config.cType == "float16" || config.cType == "float16_t" || config.cType == "half") &&
                              (config.biasType == "int32" || config.biasType == "int32_t");
    if (config.pertensorStreamK && isInt8ToFp16 && (config.x2ScaleType == "float" || config.x2ScaleType == "float32")) {
        RunTypedCase<int8_t, int8_t, half, int32_t, float>(config, stream);
        return;
    }
    if (isInt8ToInt32 && (config.x2ScaleType == "float" || config.x2ScaleType == "float32")) {
        RunTypedCase<int8_t, int8_t, int32_t, int32_t, float>(config, stream);
        return;
    }
    if (isHiFloat8ToBf16 && (config.x2ScaleType == "uint64" || config.x2ScaleType == "uint64_t")) {
        RunTypedCase<hifloat8_t, hifloat8_t, bfloat16_t, float, uint64_t>(config, stream);
        return;
    }
    if (isHiFloat8ToBf16 && (config.x2ScaleType == "float" || config.x2ScaleType == "float32")) {
        RunTypedCase<hifloat8_t, hifloat8_t, bfloat16_t, float, float>(config, stream);
        return;
    }
    std::fprintf(stderr, "No compiled kernel for the requested dtype combination\n");
    std::exit(1);
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
    RunCase(config, stream);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
