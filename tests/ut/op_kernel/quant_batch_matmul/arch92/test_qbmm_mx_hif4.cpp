/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_qbmm_mx_hif4.cpp
 * \brief Ascend960 HiF4 QBMM MX smoke cases: exercise the batched and without-batch scale addressing paths.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "qbmm_mx.h"

namespace {

using Hif4Type = hifloat4x2_t;
constexpr int64_t HIF4_SCALE_K_ALIGN = 64;
constexpr size_t HIF4_SCALE_BYTES = sizeof(AscendC::hif4_scale);

// HiF4 stores one hif4_scale per 64-element K group (no x2 E8M0 expansion).
static_assert(sizeof(AscendC::hif4_scale) == 4);
static_assert(Blaze::Gemm::MxScaleKLen<Hif4Type>(64) == 1);
static_assert(Blaze::Gemm::MxScaleKLen<Hif4Type>(65) == 2);
static_assert(Blaze::Gemm::MxScaleKLen<Hif4Type>(128) == 2);
static_assert(Blaze::Gemm::GetScaleC0<Hif4Type>() == 1);
// HiF4 packs two adjacent positions per scale element only for ScaleB ND.
static_assert(Blaze::Gemm::GetScaleGmPos<true, asc::te::scaleb_nd_layout_ptn>(64) == 32);
static_assert(Blaze::Gemm::GetScaleGmPos<true, asc::te::nn_layout_ptn>(64) == 64);

size_t Hif4InputBytes(int64_t elementCount)
{
    return (static_cast<size_t>(elementCount) + 1UL) / 2UL * sizeof(Hif4Type);
}

size_t Hif4ScaleKLen(int64_t k) { return static_cast<size_t>((k + HIF4_SCALE_K_ALIGN - 1) / HIF4_SCALE_K_ALIGN); }

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(std::max<size_t>(size, 1UL)))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

struct Hif4MxCase {
    int64_t m{64};
    int64_t n{64};
    int64_t k{128};
    uint32_t blockNum{1U};
    uint32_t baseM{64U};
    uint32_t baseN{64U};
    uint32_t baseK{64U};
    uint32_t kL1{128U};
    uint32_t scaleKL1{128U};
    uint32_t nBufferNum{2U};
    int64_t batch{1};
};

void FillHif4MxTiling(QBMMV3TilingData* tilingData, const Hif4MxCase& cfg, bool batched)
{
    const uint32_t batch = batched ? static_cast<uint32_t>(cfg.batch) : 1U;
    *tilingData = {};
    tilingData->m = cfg.m;
    tilingData->n = cfg.n;
    tilingData->k = cfg.k;
    tilingData->b = static_cast<int64_t>(batch);
    tilingData->baseM = cfg.baseM;
    tilingData->baseN = cfg.baseN;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;
    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = batch;
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = batch;
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = batch;
    tilingData->biasThreeDim = 0;
    tilingData->x1QuantMode = 0;
    tilingData->x2QuantMode = 0;
    tilingData->kAL1 = cfg.kL1;
    tilingData->kBL1 = cfg.scaleKL1;
    tilingData->nBufferNum = cfg.nBufferNum;
    tilingData->baseM_qbmm = cfg.baseM;
    tilingData->baseN_qbmm = cfg.baseN;
    tilingData->baseK_qbmm = cfg.baseK;
    tilingData->isBias = 0;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1;
    tilingData->biasDtype = 0;
}

template <bool Batched>
void RunHif4MxSmoke(const Hif4MxCase& cfg)
{
    const size_t batch = Batched ? static_cast<size_t>(cfg.batch) : 1UL;
    const size_t scaleKLen = Hif4ScaleKLen(cfg.k);
    const size_t x1Size = Hif4InputBytes(cfg.m * cfg.k) * batch;
    const size_t x2Size = Hif4InputBytes(cfg.k * cfg.n) * batch;
    const size_t scaleASize = static_cast<size_t>(cfg.m) * scaleKLen * HIF4_SCALE_BYTES * batch;
    const size_t scaleBSize = static_cast<size_t>(cfg.n) * scaleKLen * HIF4_SCALE_BYTES * batch;
    const size_t ySize = static_cast<size_t>(cfg.m) * cfg.n * sizeof(half) * batch;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer scaleAGM(scaleASize);
    GmBuffer scaleBGM(scaleBSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleAGM.Get(), nullptr);
    ASSERT_NE(scaleBGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    std::memset(x1GM.Get(), 0, x1Size);
    std::memset(x2GM.Get(), 0, x2Size);
    std::memset(scaleAGM.Get(), 0x7f, scaleASize);
    std::memset(scaleBGM.Get(), 0x7f, scaleBSize);
    std::memset(yGM.Get(), 0, ySize);

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillHif4MxTiling(tilingData, cfg, Batched);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    bool ok = false;
    if constexpr (Batched) {
        auto kernelFunc = qbmm_mx_kernel_entry<Hif4Type, Hif4Type, half, float>;
        ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), scaleAGM.Get(), scaleBGM.Get(), nullptr,
                           yGM.Get(), tilingGM.Get());
    } else {
        auto kernelFunc = qbmm_mx_without_batch_kernel_entry<Hif4Type, Hif4Type, half, float>;
        ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), scaleAGM.Get(), scaleBGM.Get(), nullptr,
                           yGM.Get(), tilingGM.Get());
    }
    ASSERT_TRUE(ok) << "QBMM MX HiF4 kernel execution failed";
}

TEST(QbmmMxHif4Test, BatchedScaleAddressing)
{
    Hif4MxCase cfg;
    cfg.m = 64;
    cfg.n = 128;
    cfg.k = 128;
    cfg.baseN = 64;
    cfg.batch = 2;
    RunHif4MxSmoke<true>(cfg);
}

TEST(QbmmMxHif4Test, WithoutBatchScaleAddressing)
{
    Hif4MxCase cfg;
    cfg.m = 64;
    cfg.n = 128;
    cfg.k = 128;
    cfg.baseN = 64;
    RunHif4MxSmoke<false>(cfg);
}

} // namespace
