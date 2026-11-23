/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "qbmm_mx.h"

namespace {

static_assert(Blaze::Gemm::Tile::MxDn2NzTrait::SUPPORTED);

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(static_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(addr_);
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_;
};

template <bool TransA>
void RunMxDn2NzSmoke(uint32_t m, uint32_t n, uint32_t kL1)
{
    using MxType = fp8_e4m3fn_t;
    constexpr uint32_t baseK = 128;
    const size_t scaleKLen = Blaze::Gemm::MxScaleKLen<MxType>(kL1);
    const size_t x1Size = static_cast<size_t>(m) * kL1 * sizeof(MxType);
    const size_t x2Size = static_cast<size_t>(n) * kL1 * sizeof(MxType);
    const size_t scaleASize = m * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleBSize = n * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t ySize = static_cast<size_t>(m) * n * sizeof(float);

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
    *tilingData = {};
    tilingData->m = m;
    tilingData->n = n;
    tilingData->k = kL1;
    // Size the L1 slots for either layout. The scheduler still uses m and n for the tail tile.
    tilingData->baseM = static_cast<uint32_t>(Blaze::Gemm::Align32(m));
    tilingData->baseN = static_cast<uint32_t>(Blaze::Gemm::Align32(n));
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->kAL1 = kL1;
    tilingData->kBL1 = kL1;
    tilingData->nBufferNum = 2;
    tilingData->baseM_qbmm = tilingData->baseM;
    tilingData->baseN_qbmm = tilingData->baseN;
    tilingData->baseK_qbmm = baseK;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_mx_without_batch_kernel_entry<MxType, MxType, float, float, TransA>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, 1U, x1GM.Get(), x2GM.Get(), scaleAGM.Get(), scaleBGM.Get(), nullptr,
                                  yGM.Get(), tilingGM.Get());
    ASSERT_TRUE(ok) << "QBMM MX DN2NZ kernel execution failed";
}

// The UT PV stubs leave GM output unchanged. These cases check kernel startup.
// One split operand, one unsplit operand, and both scales cost seven MTE1 copies per L0 tile.
// With baseK=128, the 3, 4 and 5 L0 tiles exercise 21, 28 and 35 copies, respectively.
TEST(QbmmMxDn2NzTest, KeepsOriginalLayoutBelowMte1Budget) { RunMxDn2NzSmoke<true>(16, 32, 384); }

TEST(QbmmMxDn2NzTest, KeepsOriginalLayoutAtMte1Budget) { RunMxDn2NzSmoke<true>(16, 32, 512); }

TEST(QbmmMxDn2NzTest, ConvertsOnlyAAboveMte1Budget) { RunMxDn2NzSmoke<true>(16, 32, 640); }

TEST(QbmmMxDn2NzTest, ConvertsOnlyBAboveMte1Budget) { RunMxDn2NzSmoke<false>(32, 16, 640); }

// Both operands would split. Converting just one still costs 35, so both convert.
TEST(QbmmMxDn2NzTest, ConvertsBothAboveMte1Budget) { RunMxDn2NzSmoke<true>(16, 16, 640); }

} // namespace
