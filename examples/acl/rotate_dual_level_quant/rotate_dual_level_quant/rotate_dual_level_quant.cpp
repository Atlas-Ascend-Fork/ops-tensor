/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_operator.h"

// The ASC host pass needs the launch signature. Blaze components are device-side.
#if defined(__NPU_ARCH__)
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#endif

namespace {

constexpr int64_t M = 35;
constexpr int64_t N = 544;
constexpr uint32_t ROT_K = 32;
constexpr uint32_t BLOCK_M = 16;
constexpr uint32_t LEVEL0_COLS = 512;
constexpr uint32_t LEVEL1_COLS = 32;
constexpr uint32_t LEVEL0_SCALES = (N + LEVEL0_COLS - 1) / LEVEL0_COLS;
constexpr uint32_t LEVEL1_BYTES = ((N + 2 * LEVEL1_COLS - 1) / (2 * LEVEL1_COLS)) * 2;
constexpr uint32_t GROUPS = 3;
constexpr uint32_t AIV_PER_AIC = 2;
constexpr uint32_t ROWS_PER_AIV = BLOCK_M / AIV_PER_AIC;
constexpr uint32_t C0_BF16 = 32 / sizeof(uint16_t);
constexpr uint32_t READ_GUARD_BYTES = 512;
constexpr uint32_t FIXPIPE_BYTES = ROWS_PER_AIV * ROT_K * sizeof(uint16_t) + READ_GUARD_BYTES;
constexpr uint32_t MX_BYTES = 54 * 1024;
constexpr uint32_t LEVEL0_SCALE_OFFSET = MX_BYTES + ROWS_PER_AIV * LEVEL0_COLS * sizeof(uint16_t) + READ_GUARD_BYTES;
constexpr uint32_t PRIVATE_END = FIXPIPE_BYTES + LEVEL0_SCALE_OFFSET + ROWS_PER_AIV * 32;
constexpr uint8_t OUTPUT_SENTINEL = 0xa5;

struct TilingData {
    int64_t rotationsPerGroup;
    uint32_t groupListType;
    uint8_t ubDB;
    uint8_t clamp;
};

template <class XType, class YType, bool Grouped>
__global__ __aicore__ void RotateDualLevelQuantKernel(GM_ADDR x, GM_ADDR rotation, GM_ADDR groups, GM_ADDR y,
                                                      GM_ADDR level0, GM_ADDR level1, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__)
    AscendC::InitSocState();
    const auto& tiling = *reinterpret_cast<const __gm__ TilingData*>(tilingGm);
    using namespace Blaze::Gemm;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using SchedulerShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using Scheduler = AscendC::Std::conditional_t<Grouped, Block::BlockSchedulerGmmNoQuant,
                                                  Block::BlockSchedulerMatmulBasic<SchedulerShape>>;
    using Policy = MatmulMultiBlockFixpipeOpti<ND_ALIG_1V2_FIXPIPE, OP_TYPE_QUANT, KernelMmadRotateDualLevelQuant, 0>;
    using Mmad = Block::BlockMmad<Policy, XType, Layout, XType, Layout, bfloat16_t, Layout, XType, Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<
        bfloat16_t, YType, fp8_e8m0_t, Blaze::Epilogue::Fusion::DefaultFusion<YType, bfloat16_t>, true>;
    using KernelType = Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(MX_BYTES == Epilogue::NORMALIZED_WORKSPACE_BYTES);
    static_assert(LEVEL0_COLS == Epilogue::NORMALIZED_COLS && ROWS_PER_AIV <= Epilogue::MAX_NORMALIZED_ROWS);

    typename KernelType::Params params{};
    params.problemShape = ProblemShape{M, N, ROT_K, static_cast<int64_t>(tiling.rotationsPerGroup)};
    params.mmadParams = {
        x,       rotation, nullptr, nullptr,     nullptr, nullptr, ROT_K, BLOCK_M, ROT_K, ROT_K, // oriK, mL1, nL1, kL1
        BLOCK_M, ROT_K,    ROT_K,                                                                // mL0, nL0, kL0
        2,       2,        1,       tiling.ubDB, 0,       C0_BF16, // l1Stages, l0cStages, splitM, ubDB, rowStride,
                                                                   // ubPitchGran
    };
    if constexpr (Grouped) {
        params.schedulerParams = {
            BLOCK_M,       LEVEL0_COLS,          1, 1, 1, 1, 0, GROUPS, M, true, true, true, false, false,
            sizeof(XType), tiling.groupListType,
        };
        params.groupListGmAddr = groups;
    } else {
        params.schedulerParams = {BLOCK_M, LEVEL0_COLS, ROT_K, BLOCK_M, LEVEL0_COLS, ROT_K, 1, 1};
    }

    constexpr int64_t RINT = 4; // Internal round-mode encoding used by the epilogue.
    constexpr int64_t SCALE_ALG = AscendC::IsSameType<YType, fp4x2_e2m1_t>::value ? 0 : 1;
    params.epilogueParams.mx = {
        y, level1, params.problemShape, bfloat16_t{0.0f}, false, -1, RINT, SCALE_ALG, 0.0f, 0.0f, tiling.ubDB, true,
    };
    params.epilogueParams.level0 = {
        level0,           LEVEL0_SCALES,     ROWS_PER_AIV, ROT_K, FIXPIPE_BYTES, ROT_K, MX_BYTES, LEVEL0_SCALE_OFFSET,
        bfloat16_t{0.5f}, tiling.clamp != 0,
    };
    KernelType kernel;
    kernel(params);
#endif
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ~DeviceBuffer() { static_cast<void>(aclrtFree(data_)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    uint8_t* Get() const { return data_; }
    void Fill() const { ACL_CHECK(aclrtMemset(data_, bytes_, OUTPUT_SENTINEL, bytes_)); }
    void Upload(const void* data) const
    {
        ACL_CHECK(aclrtMemcpy(data_, bytes_, data, bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void Load(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        if (!ReadFile(path, host.data(), bytes_)) {
            throw std::runtime_error("Failed to read " + path);
        }
        Upload(host.data());
    }

    void Save(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        ACL_CHECK(aclrtMemcpy(host.data(), bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        if (!WriteFile(path, host.data(), bytes_)) {
            throw std::runtime_error("Failed to write " + path);
        }
    }

private:
    uint8_t* data_{nullptr};
    size_t bytes_;
};

struct Options {
    bool grouped;
    const char* outputType;
    TilingData tiling;
    std::string dataDir;
};

template <class XType, class YType>
void RunCase(const Options& options, uint32_t blockNum, aclrtStream stream)
{
    const size_t rotations = (options.grouped ? GROUPS : 1) * options.tiling.rotationsPerGroup;
    const size_t yBytes = M * N / (std::is_same_v<YType, fp4x2_e2m1_t> ? 2 : 1);
    DeviceBuffer x(M * N * sizeof(uint16_t));
    DeviceBuffer rotation(rotations * ROT_K * ROT_K * sizeof(uint16_t));
    DeviceBuffer groups((options.grouped ? GROUPS : 1) * sizeof(int64_t));
    DeviceBuffer y(yBytes);
    DeviceBuffer level0(M * LEVEL0_SCALES * sizeof(float));
    DeviceBuffer level1(M * LEVEL1_BYTES);
    DeviceBuffer tiling(sizeof(TilingData));

    x.Load(options.dataDir + "/input_x.bin");
    rotation.Load(options.dataDir + "/input_rotation.bin");
    if (options.grouped) {
        groups.Load(options.dataDir + "/input_group_list.bin");
    }
    tiling.Upload(&options.tiling);
    y.Fill();
    level0.Fill();
    level1.Fill();

    if (options.grouped) {
        RotateDualLevelQuantKernel<XType, YType, true><<<blockNum, 0, stream>>>(
            x.Get(), rotation.Get(), groups.Get(), y.Get(), level0.Get(), level1.Get(), tiling.Get());
    } else {
        RotateDualLevelQuantKernel<XType, YType, false><<<blockNum, 0, stream>>>(
            x.Get(), rotation.Get(), nullptr, y.Get(), level0.Get(), level1.Get(), tiling.Get());
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    y.Save(options.dataDir + "/npu_y.bin");
    level0.Save(options.dataDir + "/npu_level0_scale.bin");
    level1.Save(options.dataDir + "/npu_level1_scale.bin");
}

template <class XType>
void DispatchOutput(const Options& options, uint32_t blockNum, aclrtStream stream)
{
    const std::string output = options.outputType;
    if (output == "fp4") {
        RunCase<XType, fp4x2_e2m1_t>(options, blockNum, stream);
    } else if (output == "e4m3") {
        RunCase<XType, fp8_e4m3fn_t>(options, blockNum, stream);
    } else if (output == "e5m2") {
        RunCase<XType, fp8_e5m2_t>(options, blockNum, stream);
    } else {
        throw std::invalid_argument("output dtype must be fp4, e4m3 or e5m2");
    }
}

} // namespace

int main(int argc, const char** argv)
{
    if (argc != 7) {
        std::fprintf(stderr, "Usage: %s <mode> <input_dtype> <output_dtype> <rotations> <clamp> <data_dir>\n", argv[0]);
        return 1;
    }
    try {
        const std::string mode = argv[1];
        const std::string input = argv[2];
        const std::string rotations = argv[4];
        const std::string clamp = argv[5];
        if ((mode != "ordinary" && mode != "count" && mode != "cumsum" && mode != "empty_groups") ||
            (input != "float16" && input != "bfloat16") || (rotations != "shared" && rotations != "per_block") ||
            (clamp != "0" && clamp != "1")) {
            throw std::invalid_argument("Invalid mode, input dtype, rotation layout or clamp flag");
        }

        aclrtStream stream{nullptr};
        ACLDeviceGuard guard(stream);
        PlatformInfo platform{};
        if (!InitPlatformInfo(platform) || platform.socVersion != platform_ascendc::SocVersion::ASCEND950 ||
            platform.aicNum == 0 || platform.aivNum != AIV_PER_AIC * platform.aicNum) {
            throw std::runtime_error("This example requires Ascend 950 with two AIVs per AIC");
        }
        if (platform.l1Size < 2 * (BLOCK_M * ROT_K + ROT_K * ROT_K) * sizeof(uint16_t) ||
            platform.l0aSize < 2 * BLOCK_M * ROT_K * sizeof(uint16_t) ||
            platform.l0bSize < 2 * ROT_K * ROT_K * sizeof(uint16_t) ||
            platform.l0cSize < 2 * BLOCK_M * ROT_K * sizeof(float) || platform.ubSize < PRIVATE_END) {
            throw std::runtime_error("The example tile does not fit the device buffers");
        }
        const uint8_t ubDB = platform.ubSize / 2 >= PRIVATE_END ? 2 : 1;
        const Options options{
            mode != "ordinary",
            argv[3],
            {rotations == "shared" ? 1 : N / ROT_K, mode == "cumsum" ? 0U : 1U, ubDB,
             static_cast<uint8_t>(clamp == "1")},
            argv[6],
        };
        const uint32_t blockNum = std::min<uint32_t>(platform.aicNum, CeilDiv<int64_t>(M, BLOCK_M) * LEVEL0_SCALES);
        if (input == "float16") {
            DispatchOutput<half>(options, blockNum, stream);
        } else {
            DispatchOutput<bfloat16_t>(options, blockNum, stream);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
