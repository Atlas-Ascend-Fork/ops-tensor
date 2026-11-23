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
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "qgmm_cube.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/kernel/kernel_universal.h"

namespace {
constexpr int64_t M = 32;
constexpr int64_t N = 64;
constexpr int64_t K = 128;
constexpr uint32_t GROUP_NUM = 3;
constexpr uint32_t EMPTY_GROUP = 1;
constexpr uint32_t BLOCK_NUM = 2;
constexpr uint32_t LOOP_COUNT = 2;
constexpr uint32_t BASE_K = 64;
constexpr uint8_t OFFSET_LIST = 0;
constexpr uint8_t LENGTH_LIST = 1;
constexpr int8_t SPLIT_K = 2;
constexpr uint8_t SINGLE_TENSOR = 1;
constexpr uint8_t HIFLOAT8_ONE = 0x08;
constexpr float INITIAL_Y = 3.0F;
constexpr size_t OUTPUT_TILE_ELEMENTS = M * N;

using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using LayoutA = asc::te::dn_ext_layout_ptn;
using LayoutND = asc::te::nd_ext_layout_ptn;
using BTypes = AscendC::Std::tuple<hifloat8_t, float>;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using TtPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
    Blaze::Gemm::NONE_FULL_LOAD_MODE, true, Blaze::Gemm::KernelGroupedMmadFixpipeQuant, __gm__ int32_t*,
    static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_NONE), false>;
using TcPolicy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, true,
                                                 Blaze::Gemm::KernelGroupedMmadWithScaleMixInplaceAdd>;
using TtEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
using TcEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<float, float, float, float, float>;

template <bool PerChannel>
using Hifloat8Kernel = Blaze::Gemm::Kernel::GemmUniversal<
    Shape,
    Blaze::Gemm::Block::BlockMmad<std::conditional_t<PerChannel, TcPolicy, TtPolicy>, hifloat8_t, LayoutA, BTypes,
                                  LayoutND, float, LayoutND, float, LayoutND>,
    std::conditional_t<PerChannel, TcEpilogue, TtEpilogue>, Scheduler>;

template <class Kernel>
__global__ __aicore__ void QgmmInplaceAddHifloat8KernelEntry(GM_ADDR params)
{
    Kernel kernel;
    kernel(*reinterpret_cast<const typename Kernel::Params*>(params));
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : data_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (data_ != nullptr) {
            AscendC::GmFree(data_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return data_; }

private:
    GM_ADDR data_{nullptr};
};

template <bool PerChannel>
void RunHifloat8Case(uint8_t groupListType)
{
    using Kernel = Hifloat8Kernel<PerChannel>;
    constexpr size_t SCALE_COLUMNS = PerChannel ? N : 1;
    GmBuffer a(M * K);
    GmBuffer b(N * K);
    GmBuffer scaleA(GROUP_NUM * sizeof(float));
    GmBuffer scaleB(GROUP_NUM * SCALE_COLUMNS * sizeof(float));
    GmBuffer groupList(GROUP_NUM * sizeof(int64_t));
    GmBuffer yRef(GROUP_NUM * OUTPUT_TILE_ELEMENTS * sizeof(float));
    GmBuffer config(sizeof(typename Kernel::Params));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(scaleA.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(yRef.Get(), nullptr);
    ASSERT_NE(config.Get(), nullptr);

    std::memset(a.Get(), HIFLOAT8_ONE, M * K);
    std::memset(b.Get(), HIFLOAT8_ONE, N * K);
    std::fill_n(reinterpret_cast<float*>(scaleA.Get()), GROUP_NUM, 0.5F);
    std::fill_n(reinterpret_cast<float*>(scaleB.Get()), GROUP_NUM * SCALE_COLUMNS, 0.25F);
    auto* y = reinterpret_cast<float*>(yRef.Get());
    std::fill_n(y, GROUP_NUM * OUTPUT_TILE_ELEMENTS, INITIAL_Y);
    const std::array<int64_t, GROUP_NUM> groups = groupListType == OFFSET_LIST ?
                                                      std::array<int64_t, GROUP_NUM>{BASE_K, BASE_K, K} :
                                                      std::array<int64_t, GROUP_NUM>{BASE_K, 0, K - BASE_K};
    std::memcpy(groupList.Get(), groups.data(), sizeof(groups));

    typename Kernel::Params params{};
    params.mmadParams.aGmAddr = a.Get();
    params.mmadParams.bGmAddr = b.Get();
    params.groupListGmAddr = groupList.Get();
    auto& tiling = params.gmmParams;
    tiling.groupNum = GROUP_NUM;
    tiling.groupListType = groupListType;
    tiling.m = M;
    tiling.n = N;
    tiling.k = K;
    tiling.baseM = M;
    tiling.baseN = N;
    tiling.baseK = BASE_K;
    tiling.kAL1 = BASE_K;
    tiling.kBL1 = BASE_K;
    tiling.dbL0C = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    if constexpr (PerChannel) {
        auto& mmadParams = params.mmadParams;
        mmadParams.problemShape = Shape{tiling.m, tiling.n, tiling.k, 0};
        mmadParams.l0TileShape = Shape{tiling.baseM, tiling.baseN, tiling.baseK, 0};
        mmadParams.kAL1 = tiling.kAL1;
        mmadParams.kBL1 = tiling.kBL1;
        mmadParams.l1BufferNum = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
        mmadParams.enableL0CPingPong = tiling.dbL0C == Blaze::Gemm::DOUBLE_BUFFER_COUNT;

        auto& epilogue = params.epilogueParams;
        epilogue.x1ScaleGmAddr = scaleA.Get();
        epilogue.x2ScaleGmAddr = scaleB.Get();
        epilogue.outGmAddr = yRef.Get();
        epilogue.m = tiling.m;
        epilogue.n = tiling.n;
        epilogue.baseM = tiling.baseM;
        epilogue.baseN = tiling.baseN;
        epilogue.x1QuantMode = static_cast<uint32_t>(TcEpilogue::QuantMode::PERTENSOR_MODE);
        epilogue.x2QuantMode = static_cast<uint32_t>(TcEpilogue::QuantMode::PERCHANNEL_MODE);
        epilogue.disableOutL2Cache = true;
    } else {
        params.mmadParams.cGmAddr = yRef.Get();
        params.mmadParams.scaleAGmAddr = scaleA.Get();
        params.mmadParams.scaleBGmAddr = scaleB.Get();
        tiling.groupType = SPLIT_K;
        tiling.singleW = SINGLE_TENSOR;
        tiling.singleX = SINGLE_TENSOR;
        tiling.singleY = SINGLE_TENSOR;
        tiling.x1QuantMode = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERTENSOR_MODE);
        tiling.x2QuantMode = tiling.x1QuantMode;
    }
    std::memcpy(config.Get(), &params, sizeof(params));

    // The CPU-debug runner exercises the actual kernel and synchronization. Numerical MMAD and Fixpipe
    // verification uses device runs. An empty group must preserve every element of its initial output.
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = QgmmInplaceAddHifloat8KernelEntry<Kernel>;
    for (uint32_t launch = 0; launch < LOOP_COUNT; ++launch) {
        ASSERT_TRUE(KERNEL_RUN_KF(entry, BLOCK_NUM, config.Get()));
        for (size_t element = 0; element < OUTPUT_TILE_ELEMENTS; ++element) {
            ASSERT_FLOAT_EQ(y[EMPTY_GROUP * OUTPUT_TILE_ELEMENTS + element], INITIAL_Y);
        }
    }
}

TEST(QgmmInplaceAddHifloat8, PerTensorOffset) { RunHifloat8Case<false>(OFFSET_LIST); }
TEST(QgmmInplaceAddHifloat8, PerTensorLength) { RunHifloat8Case<false>(LENGTH_LIST); }
TEST(QgmmInplaceAddHifloat8, PerChannelOffset) { RunHifloat8Case<true>(OFFSET_LIST); }
TEST(QgmmInplaceAddHifloat8, PerChannelLength) { RunHifloat8Case<true>(LENGTH_LIST); }

TEST(QgmmInplaceAddHifloat8, PerChannelZeroGroups)
{
    using Kernel = Hifloat8Kernel<true>;
    const Kernel::Params params{};
    GmBuffer config(sizeof(params));
    ASSERT_NE(config.Get(), nullptr);
    std::memcpy(config.Get(), &params, sizeof(params));

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = QgmmInplaceAddHifloat8KernelEntry<Kernel>;
    ASSERT_TRUE(KERNEL_RUN_KF(entry, BLOCK_NUM, config.Get()));
}
} // namespace
