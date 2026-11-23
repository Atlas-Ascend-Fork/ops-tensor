/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "data_utils.h"

namespace {
constexpr int64_t BASE_M = 32;
constexpr int64_t BASE_N = 32;
constexpr int64_t BASE_K = 64;
constexpr uint64_t MX_SCALE_PACK_K = 64UL;
constexpr uint64_t MX_SCALE_PAIR = 2UL;
constexpr uint64_t FP4_VALUES_PER_BYTE = 2UL;
constexpr int8_t SPLIT_K = 2;
constexpr uint8_t SINGLE_TENSOR = 1;

enum class Scenario { HIFLOAT8_TC, HIFLOAT8_TT, MXFP8, MXFP4 };

using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using LayoutA = asc::te::dn_ext_layout_ptn;
using LayoutND = asc::te::nd_ext_layout_ptn;
using Hifloat8BTypes = AscendC::Std::tuple<hifloat8_t, float>;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

struct Config {
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t groupNum;
    uint8_t groupListType;
};

template <class Tiling>
__aicore__ inline void SetCommonTiling(Tiling& tiling, const Config& config)
{
    tiling.groupNum = config.groupNum;
    tiling.groupListType = config.groupListType;
    tiling.m = config.m;
    tiling.n = config.n;
    tiling.k = config.k;
    tiling.baseM = BASE_M;
    tiling.baseN = BASE_N;
    tiling.baseK = BASE_K;
    tiling.kAL1 = BASE_K;
    tiling.kBL1 = BASE_K;
}

__global__ __aicore__ void InplaceAddTc(GM_ADDR a, GM_ADDR b, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR groupList,
                                        GM_ADDR yRef, Config config)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using Policy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, true,
                                                   Blaze::Gemm::KernelGroupedMmadWithScaleMixInplaceAdd>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, hifloat8_t, LayoutA, Hifloat8BTypes, LayoutND, float, LayoutND,
                                               float, LayoutND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<float, float, float, float, float>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;

    Kernel::Params params{};
    params.groupListGmAddr = groupList;
    auto& tiling = params.gmmParams;
    SetCommonTiling(tiling, config);

    auto& mmadParams = params.mmadParams;
    mmadParams.aGmAddr = a;
    mmadParams.bGmAddr = b;
    mmadParams.problemShape = Shape{tiling.m, tiling.n, tiling.k, 0};
    mmadParams.l0TileShape = Shape{tiling.baseM, tiling.baseN, tiling.baseK, 0};
    mmadParams.kAL1 = tiling.kAL1;
    mmadParams.kBL1 = tiling.kBL1;
    mmadParams.l1BufferNum = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    mmadParams.enableL0CPingPong = false;

    auto& epilogueParams = params.epilogueParams;
    epilogueParams.x1ScaleGmAddr = scale1;
    epilogueParams.x2ScaleGmAddr = scale2;
    epilogueParams.outGmAddr = yRef;
    epilogueParams.m = tiling.m;
    epilogueParams.n = tiling.n;
    epilogueParams.baseM = tiling.baseM;
    epilogueParams.baseN = tiling.baseN;
    epilogueParams.x1QuantMode = static_cast<uint32_t>(Epilogue::QuantMode::PERTENSOR_MODE);
    epilogueParams.x2QuantMode = static_cast<uint32_t>(Epilogue::QuantMode::PERCHANNEL_MODE);
    epilogueParams.disableOutL2Cache = true;

    Kernel kernel;
    kernel(params);
}

template <class InputType>
__global__ __aicore__ void InplaceAddCube(GM_ADDR a, GM_ADDR b, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR groupList,
                                          GM_ADDR yRef, Config config)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    constexpr bool IS_HIFLOAT8 = std::is_same_v<InputType, hifloat8_t>;
    using TtPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        Blaze::Gemm::NONE_FULL_LOAD_MODE, true, Blaze::Gemm::KernelGroupedMmadFixpipeQuant, __gm__ int32_t*,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_NONE), false>;
    using MxPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, true>;
    using Policy = std::conditional_t<IS_HIFLOAT8, TtPolicy, MxPolicy>;
    using BTypes = std::conditional_t<IS_HIFLOAT8, Hifloat8BTypes, InputType>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, InputType, LayoutA, BTypes, LayoutND, float, LayoutND, float,
                                               LayoutND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;

    typename Kernel::Params params{};
    params.problemShape = {config.m, config.n, config.k, 0};
    params.mmadParams.aGmAddr = a;
    params.mmadParams.bGmAddr = b;
    params.mmadParams.cGmAddr = yRef;
    params.mmadParams.scaleAGmAddr = scale1;
    params.mmadParams.scaleBGmAddr = scale2;
    params.groupListGmAddr = groupList;
    auto& tiling = params.gmmParams;
    SetCommonTiling(tiling, config);
    tiling.groupType = SPLIT_K;
    tiling.singleW = SINGLE_TENSOR;
    tiling.dbL0C = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    if constexpr (IS_HIFLOAT8) {
        tiling.singleX = SINGLE_TENSOR;
        tiling.singleY = SINGLE_TENSOR;
        tiling.x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERTENSOR_MODE);
        tiling.x2QuantMode = tiling.x1QuantMode;
    } else {
        tiling.scaleKAL1 = BASE_K;
        tiling.scaleKBL1 = BASE_K;
        tiling.l1BufferStage = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    }
    Kernel kernel;
    kernel(params);
}

class DeviceBuffer {
public:
    DeviceBuffer(const std::string& path, size_t bytes) : bytes_(bytes)
    {
        std::vector<uint8_t> host(bytes);
        if (!ReadFile(path, host.data(), bytes)) {
            throw std::runtime_error("Cannot load " + path);
        }
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&data_), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(data_, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    ~DeviceBuffer() { aclrtFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    uint8_t* Get() const { return data_; }

    bool Save(const std::string& path) const
    {
        std::vector<uint8_t> host(bytes_);
        ACL_CHECK(aclrtMemcpy(host.data(), bytes_, data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        return WriteFile(path, host.data(), bytes_);
    }

private:
    uint8_t* data_{nullptr};
    size_t bytes_;
};

size_t Bytes(std::initializer_list<uint64_t> extents)
{
    size_t bytes = 1;
    for (const uint64_t extent : extents) {
        if (bytes > std::numeric_limits<size_t>::max() / extent) {
            throw std::overflow_error("Input size exceeds size_t");
        }
        bytes *= extent;
    }
    return bytes;
}

int Run(Scenario scenario, const Config& config, const std::string& dir)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);
    const auto m = static_cast<uint64_t>(config.m);
    const auto n = static_cast<uint64_t>(config.n);
    const auto k = static_cast<uint64_t>(config.k);
    const uint64_t groups = config.groupNum;
    const bool isMx = scenario == Scenario::MXFP8 || scenario == Scenario::MXFP4;
    const uint64_t valuesPerByte = scenario == Scenario::MXFP4 ? FP4_VALUES_PER_BYTE : 1UL;
    const uint64_t scaleRows = k / MX_SCALE_PACK_K + groups;
    const uint64_t scale2Columns = scenario == Scenario::HIFLOAT8_TC ? n : 1UL;
    DeviceBuffer a(dir + "/a.bin", Bytes({k, m / valuesPerByte}));
    DeviceBuffer b(dir + "/b.bin", Bytes({k, n / valuesPerByte}));
    DeviceBuffer scale1(dir + "/scale1.bin",
                        isMx ? Bytes({scaleRows, m, MX_SCALE_PAIR}) : Bytes({groups, sizeof(float)}));
    DeviceBuffer scale2(dir + "/scale2.bin",
                        isMx ? Bytes({scaleRows, n, MX_SCALE_PAIR}) : Bytes({groups, scale2Columns, sizeof(float)}));
    DeviceBuffer groupList(dir + "/group_list.bin", Bytes({groups, sizeof(int64_t)}));
    DeviceBuffer yRef(dir + "/y_ref.bin", Bytes({groups, m, n, sizeof(float)}));

    const auto cores = GetAicCoreNum();
    for (const char* output : {"/npu_out_1.bin", "/npu_out_2.bin"}) {
        switch (scenario) {
            case Scenario::HIFLOAT8_TC:
                InplaceAddTc<<<cores, 0, stream>>>(a.Get(), b.Get(), scale1.Get(), scale2.Get(), groupList.Get(),
                                                   yRef.Get(), config);
                break;
            case Scenario::HIFLOAT8_TT:
                InplaceAddCube<hifloat8_t><<<cores, 0, stream>>>(a.Get(), b.Get(), scale1.Get(), scale2.Get(),
                                                                 groupList.Get(), yRef.Get(), config);
                break;
            case Scenario::MXFP8:
                InplaceAddCube<fp8_e4m3fn_t><<<cores, 0, stream>>>(a.Get(), b.Get(), scale1.Get(), scale2.Get(),
                                                                   groupList.Get(), yRef.Get(), config);
                break;
            case Scenario::MXFP4:
                InplaceAddCube<fp4x2_e2m1_t><<<cores, 0, stream>>>(a.Get(), b.Get(), scale1.Get(), scale2.Get(),
                                                                   groupList.Get(), yRef.Get(), config);
                break;
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (!yRef.Save(dir + output)) {
            return 1;
        }
    }
    return 0;
}

Scenario ParseScenario(std::string_view value)
{
    if (value == "hif8_tc") {
        return Scenario::HIFLOAT8_TC;
    }
    if (value == "hif8_tt") {
        return Scenario::HIFLOAT8_TT;
    }
    if (value == "mxfp8") {
        return Scenario::MXFP8;
    }
    if (value == "mxfp4") {
        return Scenario::MXFP4;
    }
    throw std::invalid_argument("Expected scenario hif8_tc, hif8_tt, mxfp8 or mxfp4");
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 8) {
            throw std::invalid_argument(
                "Usage: quant_grouped_matmul_inplace_add scenario M N K groupNum groupListType dataDir");
        }
        const auto scenario = ParseScenario(argv[1]);
        const auto m = std::stoll(argv[2]);
        const auto n = std::stoll(argv[3]);
        const auto k = std::stoll(argv[4]);
        const auto groups = std::stoll(argv[5]);
        const auto listType = std::stoll(argv[6]);
        if (m <= 0 || n <= 0 || k <= 0 || groups <= 0 || groups > std::numeric_limits<uint32_t>::max() ||
            (listType != 0 && listType != 1)) {
            throw std::invalid_argument("Expected positive dimensions and groupListType 0 (offset) or 1 (length)");
        }
        if (scenario == Scenario::MXFP4 && (m % FP4_VALUES_PER_BYTE != 0 || n % FP4_VALUES_PER_BYTE != 0)) {
            throw std::invalid_argument("MXFP4 requires even M and N for packed input");
        }
        return Run(scenario, {m, n, k, static_cast<uint32_t>(groups), static_cast<uint8_t>(listType)}, argv[7]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
