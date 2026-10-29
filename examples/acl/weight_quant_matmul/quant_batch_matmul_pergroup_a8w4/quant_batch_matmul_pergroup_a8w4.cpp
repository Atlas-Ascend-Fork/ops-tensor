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
 * @file quant_batch_matmul_pergroup_a8w4.cpp
 * @brief Executable example for the T-CG per-group A8W4 kernel (FP8 A, packed FP4 weight +
 *        per-group scale dequant on AIV, per-channel yScale quant output on AIC fixpipe).
 *
 *        y = (x1 @ (x2 * x2Scale)) * yScale, x2 layout is ND (transposed) or NZ fractal.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

#if defined(IMPL_STD_ASCENDC_STD_INT_IMPL_H) && !defined(IMPL_TENSOR_API_UTILS_INT_IMPL_H)
#define IMPL_TENSOR_API_UTILS_INT_IMPL_H
#endif

#include "blaze/gemm/kernel/kernel_wqmm_mix_pergroup.h"
#include "blaze/gemm/utils/layout_utils.h"

#define ACL_CHECK(expr)                                                                                               \
    do {                                                                                                              \
        const aclError aclCheckResult = (expr);                                                                       \
        if (aclCheckResult != ACL_SUCCESS) {                                                                          \
            std::cerr << "ACL call failed: " << #expr << ", error " << static_cast<int>(aclCheckResult) << std::endl; \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace {

constexpr int64_t SINGLE_CORE_TILE_M = 32;
constexpr int64_t SINGLE_CORE_TILE_N = 64;
constexpr int64_t BASE_K = 128;
constexpr uint16_t AL1_PINGPONG = 2U;
constexpr uint16_t BL1_PINGPONG = 4U;
constexpr uint32_t DB_L0C = 2U;
constexpr int64_t N_BUB_SIZE = 32; // half of the BL1 N size: QUAD two-AIV N split
constexpr int64_t K_BUB_SIZE = BASE_K;
constexpr uint64_t GROUP_SIZE = 32U;
constexpr uint32_t ITERATE_ORDER_N = 1U;

struct PergroupA8W4Tiling {
    int64_t mSize{0};
    int64_t nSize{0};
    int64_t kSize{0};
    uint64_t groupSize{GROUP_SIZE};
    uint64_t nBubSize{static_cast<uint64_t>(N_BUB_SIZE)};
    uint64_t kBubSize{static_cast<uint64_t>(K_BUB_SIZE)};
    uint32_t cubeNumBlocksM{1U};
    uint32_t cubeNumBlocksN{1U};
    uint32_t baseM{0U};
    uint32_t baseN{0U};
    uint32_t baseK{static_cast<uint32_t>(BASE_K)};
    uint32_t iterateOrder{ITERATE_ORDER_N};
    uint8_t vecCoreParallel{0U};
    uint16_t al1Pingpong{AL1_PINGPONG};
    uint16_t bl1Pingpong{BL1_PINGPONG};
    uint32_t dbL0C{DB_L0C};
};

struct CliArgs {
    bool weightNz{false};
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
    std::string dataDir;
    std::string outputPath;
};

bool ParseArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <nd|nz> <m> <k> <n> <dataDir> <outputPath>" << std::endl;
        return false;
    }
    const std::string variant = argv[1];
    if (variant != "nd" && variant != "nz") {
        std::cerr << "Variant must be nd or nz." << std::endl;
        return false;
    }
    args.weightNz = variant == "nz";
    args.m = std::atoll(argv[2]);
    args.k = std::atoll(argv[3]);
    args.n = std::atoll(argv[4]);
    args.dataDir = argv[5];
    args.outputPath = argv[6];
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "M, K, and N must be positive." << std::endl;
        return false;
    }
    if (args.k % 32 != 0) {
        std::cerr << "K must be a multiple of 32 (per-group CustomCheck constraint)." << std::endl;
        return false;
    }
    if (args.m > SINGLE_CORE_TILE_M || args.n > SINGLE_CORE_TILE_N) {
        std::cerr << "This single-core demo requires m <= 32 and n <= 64." << std::endl;
        return false;
    }
    if (args.weightNz && args.n % 32 != 0) {
        std::cerr << "The NZ variant requires n to be a multiple of 32 (fractal alignment)." << std::endl;
        return false;
    }
    return true;
}

// Fixed single-core tiling: one scheduler tile per core (stepM == stepN == 1 invariant),
// kAL1 == kBL1 == BASE_K so each A generation covers exactly one B block.
PergroupA8W4Tiling BuildTiling(const CliArgs& args)
{
    PergroupA8W4Tiling tiling;
    tiling.mSize = args.m;
    tiling.nSize = args.n;
    tiling.kSize = args.k;
    tiling.baseM = static_cast<uint32_t>(SINGLE_CORE_TILE_M);
    tiling.baseN = static_cast<uint32_t>(SINGLE_CORE_TILE_N);
    return tiling;
}

template <bool IS_WEIGHT_NZ>
__global__ __aicore__ void QuantBatchMatmulPergroupA8W4Kernel(GM_ADDR x1Gm, GM_ADDR x2Gm, GM_ADDR x2ScaleGm,
                                                              GM_ADDR yScaleGm, GM_ADDR yGm,
                                                              const PergroupA8W4Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using AType = fp8_e4m3fn_t;
    using BType = fp4x2_e2m1_t;
    using ScaleType = bfloat16_t;
    using CType = int8_t;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<IS_WEIGHT_NZ, asc::te::nz_layout_ptn, asc::te::dn_ext_layout_ptn>;
    // The B-side tuple carries the packed-FP4 weight plus the per-group scale; TCG has no
    // bias, the trailing slots only satisfy the BlockMmad template signature.
    using BTypeTuple = AscendC::Std::tuple<BType, ScaleType>;
    using BlockMmadType = Blaze::Gemm::Block::BlockMmad<Blaze::Gemm::MatmulWithWeightQuantPergroup, AType, LayoutA,
                                                        BTypeTuple, LayoutB, CType, LayoutC, void, void>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t>;
    using BlockSchedulerType = Blaze::Gemm::Block::BlockSchedulerWqmmBlockSplit<ProblemShape, LayoutB, AType>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmadType, void, BlockSchedulerType>;

    typename BlockMmadType::Params mmadParams{};
    mmadParams.aGmAddr = x1Gm;
    mmadParams.cGmAddr = yGm;
    mmadParams.yScaleGmAddr = yScaleGm;
    mmadParams.l1TileShape = asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                                                 static_cast<int64_t>(BASE_K), static_cast<int64_t>(BASE_K));
    mmadParams.l0TileShape = asc::te::make_shape(static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                                                 static_cast<int64_t>(BASE_K));
    mmadParams.vecCoreParallel = tiling.vecCoreParallel;
    mmadParams.AL1Pingpong = tiling.al1Pingpong;
    mmadParams.BL1Pingpong = tiling.bl1Pingpong;
    mmadParams.dbL0C = tiling.dbL0C;

    typename KernelImpl::Params params{
        asc::te::make_shape(tiling.mSize, tiling.nSize, tiling.kSize),
        mmadParams,
        {x2Gm, x2ScaleGm, tiling.groupSize, tiling.nBubSize, tiling.kBubSize},
        {tiling.cubeNumBlocksM, tiling.cubeNumBlocksN, tiling.baseM, tiling.baseN, tiling.iterateOrder}};
    KernelImpl kernel;
    kernel(params);
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

    void CopyFromFile(const std::string& path, size_t expectedSize) const
    {
        std::vector<uint8_t> host(expectedSize);
        FILE* file = std::fopen(path.c_str(), "rb");
        if (file == nullptr || std::fread(host.data(), 1U, expectedSize, file) != expectedSize) {
            std::cerr << "unexpected input file size: " << path << std::endl;
            std::exit(1);
        }
        std::fclose(file);
        ACL_CHECK(aclrtMemcpy(data_, size_, host.data(), expectedSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::vector<uint8_t> CopyToHost() const
    {
        std::vector<uint8_t> host(size_);
        ACL_CHECK(aclrtMemcpy(host.data(), size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
        return host;
    }

    void Clear() const { ACL_CHECK(aclrtMemset(data_, size_, 0, size_)); }

private:
    uint8_t* data_{nullptr};
    size_t size_{0U};
};

int64_t AlignUp(int64_t value, int64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

int Run(const CliArgs& args)
{
    const int64_t kGroupNum = (args.k + static_cast<int64_t>(GROUP_SIZE) - 1) / static_cast<int64_t>(GROUP_SIZE);
    size_t x1Bytes = static_cast<size_t>(args.m * args.k);
    size_t x2Bytes = 0U;
    if (args.weightNz) {
        // NZ fractal footprint: (n1 * k1) blocks of 16K x 32N, packed two nibbles per byte.
        const int64_t k1 = AlignUp(args.k, 16) / 16;
        const int64_t n1 = AlignUp(args.n, 32) / 32;
        x2Bytes = static_cast<size_t>(k1 * n1 * 16 * 32 / 2);
    } else {
        // ND transposed weight: physical (n, k) row-major packed FP4.
        x2Bytes = static_cast<size_t>(args.n * args.k / 2);
    }
    const size_t x2ScaleBytes = args.weightNz ? static_cast<size_t>(kGroupNum * args.n * sizeof(uint16_t)) :
                                                static_cast<size_t>(args.n * kGroupNum * sizeof(uint16_t));
    const size_t yScaleBytes = static_cast<size_t>(args.n * sizeof(uint64_t));
    const size_t yBytes = static_cast<size_t>(args.m * args.n);

    const PergroupA8W4Tiling tiling = BuildTiling(args);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream{nullptr};
    ACL_CHECK(aclrtCreateStream(&stream));

    DeviceBuffer x1(x1Bytes);
    DeviceBuffer x2(x2Bytes);
    DeviceBuffer x2Scale(x2ScaleBytes);
    DeviceBuffer yScale(yScaleBytes);
    DeviceBuffer y(yBytes);
    x1.CopyFromFile(args.dataDir + "/input_a.bin", x1Bytes);
    x2.CopyFromFile(args.dataDir + "/input_b.bin", x2Bytes);
    x2Scale.CopyFromFile(args.dataDir + "/scale_b.bin", x2ScaleBytes);
    yScale.CopyFromFile(args.dataDir + "/y_scale.bin", yScaleBytes);
    y.Clear();

    if (args.weightNz) {
        QuantBatchMatmulPergroupA8W4Kernel<true>
            <<<1U, 0, stream>>>(x1.Get(), x2.Get(), x2Scale.Get(), yScale.Get(), y.Get(), tiling);
    } else {
        QuantBatchMatmulPergroupA8W4Kernel<false>
            <<<1U, 0, stream>>>(x1.Get(), x2.Get(), x2Scale.Get(), yScale.Get(), y.Get(), tiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtDestroyStream(stream));

    const std::vector<uint8_t> hostY = y.CopyToHost();
    FILE* out = std::fopen(args.outputPath.c_str(), "wb");
    if (out == nullptr || std::fwrite(hostY.data(), 1U, hostY.size(), out) != hostY.size()) {
        std::cerr << "failed to write output: " << args.outputPath << std::endl;
        return 1;
    }
    std::fclose(out);
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    std::cout << "PASS: launched per-group A8W4 " << (args.weightNz ? "nz" : "nd") << " with shape [" << args.m << ", "
              << args.k << ", " << args.n << "]" << std::endl;
    return 0;
}

} // namespace

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        return 1;
    }
    return Run(args);
}
