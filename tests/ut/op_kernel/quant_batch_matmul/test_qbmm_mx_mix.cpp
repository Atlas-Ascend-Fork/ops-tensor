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
#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "qbmm_cpu_debug_stub.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_mix.h"

namespace QbmmMxMixUT {

using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
constexpr uint32_t MAX_CALLS = 128;
constexpr uint32_t AIV_PER_AIC = 2;

// Record MMAD/epilogue arguments to check kernel scheduling, tensor slices and addresses.
// The recording components do not perform MMAD or VF arithmetic.
struct MmadCall {
    uint64_t a, b, scaleA, scaleB, bias, c;
    int64_t m, n;
    uint32_t bCache, cCache;
};

struct EpilogueCall {
    int64_t mPos, nPos, m, n;
};

struct CubeTrace {
    uint32_t initCount, callCount, isBias, pingpong;
    int64_t baseM, baseN, baseK;
    MmadCall calls[MAX_CALLS];
};

struct VectorTrace {
    uint32_t initCount, callCount;
    EpilogueCall calls[MAX_CALLS];
};

struct Trace {
    CubeTrace cube;
    VectorTrace vec[AIV_PER_AIC];
};

template <class AType_, class BType_, class CType_, class LayoutA_, class LayoutB_, class LayoutC_, bool Atomic>
class RecordingMmad {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = float;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, Atomic,
                                                          Blaze::Gemm::KernelMmadWithScaleMxMix>;
    struct Params {
        GM_ADDR aGmAddr;
        GM_ADDR bGmAddr;
        GM_ADDR cGmAddr;
        GM_ADDR biasGmAddr;
        GM_ADDR scaleAGmAddr;
        GM_ADDR scaleBGmAddr;
    };
    struct L1Params {
        GM_ADDR trace;
    };

    __aicore__ inline void Init(const Shape&, const Shape& block, const L1Params& params, bool isBias, bool pingpong)
    {
        trace_ = &reinterpret_cast<Trace*>(params.trace)[AscendC::GetBlockIdx()].cube;
        ++trace_->initCount;
        trace_->isBias = isBias;
        trace_->pingpong = pingpong;
        trace_->baseM = asc::te::get<0>(block);
        trace_->baseN = asc::te::get<1>(block);
        trace_->baseK = asc::te::get<2>(block);
    }

    template <class TA, class TB, class TSA, class TSB, class TBias, class TC>
    __aicore__ inline void operator()(const TA& a, const TB& b, const TSA& sa, const TSB& sb, const TBias& bias,
                                      const TC& c, const Shape& block)
    {
        const uint32_t index = trace_->callCount++;
        if (index < MAX_CALLS) {
            trace_->calls[index] = {reinterpret_cast<uint64_t>(a.data().get()),
                                    reinterpret_cast<uint64_t>(b.data().get()),
                                    reinterpret_cast<uint64_t>(sa.data().get()),
                                    reinterpret_cast<uint64_t>(sb.data().get()),
                                    reinterpret_cast<uint64_t>(bias.data().get()),
                                    reinterpret_cast<uint64_t>(c.data().get()),
                                    asc::te::get<0>(block),
                                    asc::te::get<1>(block),
                                    static_cast<uint32_t>(b.get_cache_mode()),
                                    static_cast<uint32_t>(c.get_cache_mode())};
        }
    }

private:
    CubeTrace* trace_{nullptr};
};

class RecordingEpilogue {
public:
    struct Params {
        GM_ADDR trace;
    };

    __aicore__ inline void Init(const Params& params)
    {
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        trace_ = &reinterpret_cast<Trace*>(params.trace)[core].vec[AscendC::GetSubBlockIdx()];
        ++trace_->initCount;
    }

    __aicore__ inline void operator()(int64_t mPos, int64_t nPos, const Shape& block)
    {
        const uint32_t index = trace_->callCount++;
        if (index < MAX_CALLS) {
            trace_->calls[index] = {mPos, nPos, asc::te::get<0>(block), asc::te::get<1>(block)};
        }
    }

private:
    VectorTrace* trace_{nullptr};
};

template <class A = fp8_e4m3fn_t, class B = fp8_e5m2_t, class C = float, bool TransA = false, bool TransB = false,
          bool WeightNz = false, bool TransC = false, bool Atomic = false>
struct Types {
    static constexpr bool FP4 = Blaze::Gemm::IsFp4<A>();
    static constexpr bool TRANS_A = TransA, TRANS_B = TransB, WEIGHT_NZ = WeightNz, TRANS_C = TransC;
    static constexpr bool ATOMIC = Atomic;
    using AType = A;
    using BType = B;
    using CType = C;
    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NdB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NzB = AscendC::Std::conditional_t<TransB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<WeightNz, NzB, NdB>;
    using LayoutC = AscendC::Std::conditional_t<TransC, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using Mmad = RecordingMmad<A, B, C, LayoutA, LayoutB, LayoutC, Atomic>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<Shape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                           LayoutA, LayoutB, A>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, RecordingEpilogue, Scheduler>;
};

struct Case {
    int64_t m{33}, n{65}, k{70};
    uint32_t cores{2}, baseM{32}, baseN{64}, baseK{64}, dbL0C{1}, bMustHitL2{1};
    std::array<uint32_t, 4> batchA{1, 1, 1, 1};
    std::array<uint32_t, 4> batchB{1, 1, 1, 1};
    std::array<uint32_t, 4> batchC{1, 1, 1, 1};
    uint32_t isBias{0}, biasThreeDim{0}, mTailTile{1}, nTailTile{1};
};

__aicore__ inline uint64_t BatchCount(const std::array<uint32_t, 4>& dims)
{
    return static_cast<uint64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
}

template <class T>
__global__ __aicore__ void RunKernel(GM_ADDR a, GM_ADDR b, GM_ADDR sa, GM_ADDR sb, GM_ADDR bias, GM_ADDR c,
                                     GM_ADDR config, GM_ADDR trace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto& cfg = *reinterpret_cast<const Case*>(config);
    const auto& ba = cfg.batchA;
    const auto& bb = cfg.batchB;
    const auto& bc = cfg.batchC;
    typename T::Kernel::Params params{};
    params.problemShape = {cfg.m, cfg.n, cfg.k, static_cast<int64_t>(BatchCount(bc))};
    params.mmadParams = {a, b, c, bias, sa, sb};
    params.l1Params = {trace};
    params.epilogueParams = {trace};
    params.schParams = {cfg.baseM, cfg.baseN, cfg.mTailTile, cfg.nTailTile, 1, 1, 0, 0};
    params.qbmmParams = {ba[0],
                         ba[1],
                         ba[2],
                         ba[3],
                         bb[0],
                         bb[1],
                         bb[2],
                         bb[3],
                         bc[0],
                         bc[1],
                         bc[2],
                         bc[3],
                         cfg.biasThreeDim,
                         cfg.baseM,
                         cfg.baseN,
                         cfg.baseK,
                         cfg.isBias,
                         cfg.dbL0C,
                         cfg.bMustHitL2};
    typename T::Kernel kernel;
    kernel(params);
}

class GmBuffer {
public:
    explicit GmBuffer(uint64_t bytes) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(std::max<uint64_t>(bytes, 1))))
    {
        if (addr_ != nullptr) {
            std::memset(addr_, 0, std::max<uint64_t>(bytes, 1));
        }
    }
    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(addr_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return addr_; }
    uint64_t Address() const { return reinterpret_cast<uint64_t>(addr_); }

private:
    GM_ADDR addr_;
};

uint64_t Align(uint64_t size, uint64_t alignment) { return (size + alignment - 1) / alignment * alignment; }

// Independent broadcast oracle: decode the output batch coordinates, then flatten each
// input's coordinates with singleton dimensions removed. Do not reuse the kernel's steps.
uint64_t InputBatch(uint64_t output, const std::array<uint32_t, 4>& input, const std::array<uint32_t, 4>& shape)
{
    std::array<uint64_t, 4> coord{};
    for (int i = 3; i >= 0; --i) {
        coord[i] = output % shape[i];
        output /= shape[i];
    }
    uint64_t index = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
        index = index * input[i] + (input[i] == 1 ? 0 : coord[i]);
    }
    return index;
}

template <class T>
void RunAndCheck(const Case& cfg, uint64_t expectedTiles, uint32_t expectedIdleCores = 0,
                 bool expectCacheTransition = false)
{
    const uint64_t m = cfg.m, n = cfg.n, k = cfg.k;
    const uint64_t batch = BatchCount(cfg.batchC);
    const uint64_t packed = T::FP4 ? 2 : 1;
    const uint64_t c0 = T::FP4 ? 64 : 32;
    const uint64_t scaleK = Align(k, 64) / 32;
    const uint64_t aBytes = sizeof(typename T::AType), bBytes = sizeof(typename T::BType);
    const uint64_t scaleBytes = sizeof(AscendC::fp8_e8m0_t);
    const uint64_t aStride = m * k / packed * aBytes;
    const uint64_t bStride = (T::WEIGHT_NZ ? Align(k, T::TRANS_B ? c0 : 16) * Align(n, T::TRANS_B ? 16 : c0) : k * n) /
                             packed * bBytes;
    GmBuffer a(aStride * BatchCount(cfg.batchA)), b(bStride * BatchCount(cfg.batchB));
    GmBuffer sa(m * scaleK * BatchCount(cfg.batchA) * scaleBytes);
    GmBuffer sb(n * scaleK * BatchCount(cfg.batchB) * scaleBytes);
    GmBuffer bias(n * sizeof(float) * (cfg.biasThreeDim ? batch : 1));
    GmBuffer c(batch * m * n * sizeof(typename T::CType));
    GmBuffer config(sizeof(Case)), captured(cfg.cores * sizeof(Trace));
    for (GM_ADDR addr : {a.Get(), b.Get(), sa.Get(), sb.Get(), bias.Get(), c.Get(), config.Get(), captured.Get()}) {
        ASSERT_NE(addr, nullptr);
    }
    std::memcpy(config.Get(), &cfg, sizeof(cfg));
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = RunKernel<T>;
    ASSERT_TRUE(KERNEL_RUN_KF(entry, cfg.cores, a.Get(), b.Get(), sa.Get(), sb.Get(), cfg.isBias ? bias.Get() : nullptr,
                              c.Get(), config.Get(), captured.Get()));

    const auto* traces = reinterpret_cast<const Trace*>(captured.Get());
    std::vector<uint32_t> visits(batch * m * n, 0);
    uint64_t total = 0;
    uint32_t idle = 0;
    bool cacheDisabled = false, cacheEnabled = false;
    for (uint32_t core = 0; core < cfg.cores; ++core) {
        SCOPED_TRACE(core);
        const auto& cube = traces[core].cube;
        ASSERT_EQ(cube.initCount, 1U);
        EXPECT_EQ(cube.isBias, cfg.isBias);
        EXPECT_EQ(cube.pingpong, cfg.dbL0C > 1);
        EXPECT_EQ(cube.baseM, cfg.baseM);
        EXPECT_EQ(cube.baseN, cfg.baseN);
        EXPECT_EQ(cube.baseK, cfg.baseK);
        ASSERT_LE(cube.callCount, MAX_CALLS);
        total += cube.callCount;
        idle += cube.callCount == 0;
        for (const auto& vec : traces[core].vec) {
            ASSERT_EQ(vec.initCount, 1U);
            ASSERT_EQ(vec.callCount, cube.callCount);
        }
        for (uint32_t i = 0; i < cube.callCount; ++i) {
            SCOPED_TRACE(i);
            const auto& call = cube.calls[i];
            ASSERT_GE(call.c, c.Address());
            const uint64_t cBytes = call.c - c.Address();
            ASSERT_EQ(cBytes % sizeof(typename T::CType), 0U);
            const uint64_t cElement = cBytes / sizeof(typename T::CType);
            ASSERT_LT(cElement, batch * m * n);
            const uint64_t batchC = cElement / (m * n);
            const uint64_t position = cElement % (m * n);
            const uint64_t row = T::TRANS_C ? position % m : position / n;
            const uint64_t col = T::TRANS_C ? position / m : position % n;
            ASSERT_GT(call.m, 0);
            ASSERT_GT(call.n, 0);
            ASSERT_LE(row + call.m, m);
            ASSERT_LE(col + call.n, n);
            for (const auto& vec : traces[core].vec) {
                EXPECT_EQ(vec.calls[i].mPos, row);
                EXPECT_EQ(vec.calls[i].nPos, col);
                EXPECT_EQ(vec.calls[i].m, call.m);
                EXPECT_EQ(vec.calls[i].n, call.n);
            }
            const uint64_t batchA = InputBatch(batchC, cfg.batchA, cfg.batchC);
            const uint64_t batchB = InputBatch(batchC, cfg.batchB, cfg.batchC);
            const uint64_t aElement = T::TRANS_A ? row : row * k;
            uint64_t bElement = T::TRANS_B ? col * k : col;
            if (T::WEIGHT_NZ) {
                bElement = T::TRANS_B ? col * c0 : col / c0 * Align(k, 16) * c0 + col % c0;
            }
            EXPECT_EQ(call.a, a.Address() + batchA * aStride + aElement / packed * aBytes);
            EXPECT_EQ(call.b, b.Address() + batchB * bStride + bElement / packed * bBytes);
            EXPECT_EQ(call.scaleA, sa.Address() + (batchA * m * scaleK + row * (T::TRANS_A ? 2 : scaleK)) * scaleBytes);
            EXPECT_EQ(call.scaleB, sb.Address() + (batchB * n * scaleK + col * (T::TRANS_B ? scaleK : 2)) * scaleBytes);
            if (cfg.isBias) {
                EXPECT_EQ(call.bias, bias.Address() + ((cfg.biasThreeDim ? batchC * n : 0) + col) * sizeof(float));
            }
            const bool disableB = cfg.bMustHitL2 == 0 && static_cast<uint64_t>(call.m) == m &&
                                  (T::TRANS_B || call.n % (T::FP4 ? 256 : 128) == 0);
            EXPECT_EQ(call.bCache,
                      static_cast<uint32_t>(disableB ? asc::te::cache_mode::disable : asc::te::cache_mode::normal));
            EXPECT_EQ(call.cCache,
                      static_cast<uint32_t>(T::ATOMIC ? asc::te::cache_mode::disable : asc::te::cache_mode::normal));
            cacheDisabled |= call.bCache == static_cast<uint32_t>(asc::te::cache_mode::disable);
            cacheEnabled |= call.bCache == static_cast<uint32_t>(asc::te::cache_mode::normal);
            for (uint64_t mi = row; mi < row + call.m; ++mi) {
                for (uint64_t ni = col; ni < col + call.n; ++ni) {
                    ++visits[(batchC * m + mi) * n + ni];
                }
            }
        }
    }
    EXPECT_EQ(total, expectedTiles);
    EXPECT_EQ(idle, expectedIdleCores);
    EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](uint32_t count) { return count == 1; }))
        << "Every output element must be assigned exactly once across cores, tails and batches";
    if (expectCacheTransition) {
        EXPECT_TRUE(cacheDisabled);
        EXPECT_TRUE(cacheEnabled);
    }
}

TEST(QbmmMxMix, SingleBatchPairsEachCubeTileWithBothAivSubcores)
{
    Case cfg;
    cfg.isBias = 1;
    cfg.dbL0C = 2;
    RunAndCheck<Types<>>(cfg, 4);
}

TEST(QbmmMxMix, FourDimensionalBroadcastAdvancesInputsScalesAndBatchedBias)
{
    Case cfg;
    cfg.batchA = {2, 1, 2, 1};
    cfg.batchB = {1, 2, 1, 3};
    cfg.batchC = {2, 2, 2, 3};
    cfg.isBias = cfg.biasThreeDim = 1;
    cfg.cores = 3;
    // Four tiles per batch on three cores also exercises wrapping batch start indices.
    RunAndCheck<Types<fp8_e5m2_t, fp8_e4m3fn_t, half, true, true>>(cfg, 96);
}

TEST(QbmmMxMix, SharedBiasAndTransposedOutputKeepIndependentBatchStrides)
{
    Case cfg;
    cfg.batchA = cfg.batchC = {1, 1, 1, 3};
    cfg.isBias = 1;
    RunAndCheck<Types<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true, false, true>>(cfg, 12);
}

TEST(QbmmMxMix, Fp8NzAndZnUsePaddedWeightBatchStrides)
{
    Case cfg;
    cfg.batchB = cfg.batchC = {1, 1, 1, 3};
    RunAndCheck<Types<fp8_e5m2_t, fp8_e5m2_t, float, true, false, true>>(cfg, 12);
    RunAndCheck<Types<fp8_e4m3fn_t, fp8_e5m2_t, half, false, true, true>>(cfg, 12);
}

TEST(QbmmMxMix, Fp4NdNzAndZnConvertElementOffsetsToPackedBytes)
{
    Case cfg;
    cfg.m = 18;
    cfg.n = 70;
    cfg.k = 130;
    cfg.baseM = cfg.baseN = 64;
    cfg.batchA = cfg.batchC = {1, 2, 1, 2};
    cfg.batchB = {1, 1, 1, 2};
    RunAndCheck<Types<fp4x2_e2m1_t, fp4x2_e1m2_t, float>>(cfg, 8);
    RunAndCheck<Types<fp4x2_e1m2_t, fp4x2_e2m1_t, half, true, false, true>>(cfg, 8);
    RunAndCheck<Types<fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, false, true, true>>(cfg, 8);
}

TEST(QbmmMxMix, WeightCacheHintIsRestoredForTheUnalignedNTail)
{
    Case cfg;
    cfg.m = cfg.baseM = 16;
    cfg.n = 144;
    cfg.baseN = 128;
    cfg.cores = 1;
    cfg.bMustHitL2 = 0;
    RunAndCheck<Types<>>(cfg, 2, 0, true);
    cfg.n = 272;
    cfg.baseN = 256;
    RunAndCheck<Types<fp4x2_e2m1_t, fp4x2_e2m1_t>>(cfg, 2, 0, true);
}

TEST(QbmmMxMix, WeightCacheHonorsTransposeFullMAndMustHitL2)
{
    Case cfg;
    cfg.n = cfg.baseN = 128;
    cfg.bMustHitL2 = 0;
    RunAndCheck<Types<>>(cfg, 2); // An aligned N tile still retains L2 when M is split.
    cfg.m = cfg.baseM = 16;
    cfg.n = 65;
    cfg.baseN = 64;
    RunAndCheck<Types<fp8_e4m3fn_t, fp8_e5m2_t, float, false, true>>(cfg, 2);
    cfg.bMustHitL2 = 1;
    RunAndCheck<Types<fp8_e4m3fn_t, fp8_e5m2_t, float, false, true>>(cfg, 2);
}

TEST(QbmmMxMix, AtomicOutputDisablesL2InSingleAndMultipleBatches)
{
    Case cfg;
    using Atomic = Types<fp8_e4m3fn_t, fp8_e5m2_t, float, false, false, false, false, true>;
    RunAndCheck<Atomic>(cfg, 4);
    cfg.batchA = cfg.batchB = cfg.batchC = {1, 1, 1, 2};
    RunAndCheck<Atomic>(cfg, 8);
}

TEST(QbmmMxMix, TailResplitCoversBothAxesWithoutOverlap)
{
    Case cfg;
    cfg.n = 35;
    cfg.baseM = cfg.baseN = 64;
    cfg.cores = 4;
    cfg.mTailTile = cfg.nTailTile = 2;
    RunAndCheck<Types<>>(cfg, 4);
}

TEST(QbmmMxMix, TailResplitPersistsAcrossRemainingBatchesAndLeavesIdleCores)
{
    Case cfg;
    cfg.n = 35;
    cfg.baseM = cfg.baseN = 64;
    cfg.cores = 8;
    cfg.mTailTile = 2;
    cfg.batchB = cfg.batchC = {1, 1, 1, 3};
    RunAndCheck<Types<>>(cfg, 6, 2);
}

TEST(QbmmMxMix, EmptySplitTilesDoNotCallMmadOrEpilogue)
{
    Case cfg;
    cfg.m = cfg.n = 1;
    cfg.cores = 8;
    cfg.mTailTile = 4;
    cfg.nTailTile = 2;
    // The other seven cores receive a zero M or zero N split from the real scheduler.
    RunAndCheck<Types<>>(cfg, 1, 7);
}

TEST(QbmmMxMix, EveryZeroBatchDimensionReturnsBeforeScheduling)
{
    for (size_t dim = 0; dim < 4; ++dim) {
        SCOPED_TRACE(dim);
        Case cfg;
        cfg.batchA[dim] = cfg.batchB[dim] = cfg.batchC[dim] = 0;
        RunAndCheck<Types<>>(cfg, 0, cfg.cores);
    }
}

} // namespace QbmmMxMixUT
