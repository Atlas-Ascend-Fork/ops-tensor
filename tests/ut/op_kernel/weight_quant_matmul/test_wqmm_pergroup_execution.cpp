/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full license.
 */
#include <cerrno>
#include <array>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "qbmm_pergroup_a8w4.h"

namespace {
using namespace QBMMPerGroupA8W4UT;
class GmBuffer {
public:
    explicit GmBuffer(size_t size) : address_(static_cast<GM_ADDR>(AscendC::GmAlloc(size)))
    {
        if (!address_)
            throw std::bad_alloc();
        std::memset(address_, 0, size);
    }
    ~GmBuffer() { AscendC::GmFree(address_); }
    GM_ADDR Get() const { return address_; }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

private:
    GM_ADDR address_;
};
// Linux host-only scratch mapping for the *direct* mask store in
// KernelPergroupWeightPrologue::InitUbStorage. Never maps UB address zero,
// never replaces an existing mapping, and never relaxes mmap_min_addr.
class MaskPage {
public:
    explicit MaskPage(uintptr_t mask) : pageSize_(sysconf(_SC_PAGESIZE))
    {
        const auto page = mask / pageSize_ * pageSize_;
        if (page < 65536 || mask + AscendC::VECTOR_REG_WIDTH / 8 > AscendC::TOTAL_UB_SIZE)
            return;
        address_ = mmap(reinterpret_cast<void*>(page), pageSize_, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (address_ != MAP_FAILED && address_ != reinterpret_cast<void*>(page)) {
            munmap(address_, pageSize_);
            address_ = MAP_FAILED;
        }
        if (address_ != MAP_FAILED)
            CpuDebug::maskAddress = mask;
    }
    ~MaskPage()
    {
        if (address_ != MAP_FAILED)
            munmap(address_, pageSize_);
        CpuDebug::maskAddress = 0;
    }
    bool Valid() const { return address_ != MAP_FAILED; }
    MaskPage(const MaskPage&) = delete;
    MaskPage& operator=(const MaskPage&) = delete;

private:
    void* address_ = MAP_FAILED;
    size_t pageSize_;
};
uint64_t Align(uint64_t n, uint64_t unit) { return (n + unit - 1) / unit * unit; }
uintptr_t MaskOffset(const TilingData& t, bool nz)
{
    uint64_t phases = t.bl1Pingpong == 4 ? 4 : (nz ? t.bl1Pingpong : std::min<uint16_t>(2, t.bl1Pingpong));
    uint64_t input = phases * t.nBubSize * (nz ? t.kBubSize : Align(t.kBubSize, 64)) / 2;
    uint64_t scaleGroups = (t.kBubSize + t.groupSize - 1) / t.groupSize;
    uint64_t scales = t.bl1Pingpong * t.nBubSize * (nz ? scaleGroups : Align(scaleGroups, 32)) * 2;
    uint64_t output = phases * (nz ? t.nBubSize * t.kBubSize : (Align(t.nBubSize, 16) + 1) * Align(t.kBubSize, 32));
    return input + scales + output;
}
template <bool Nz>
__global__ __aicore__ void PergroupObservedKernel(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR yscale, GM_ADDR out,
                                                  GM_ADDR tiling, GM_ADDR traces)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    uint64_t lane = 0;
    if ASCEND_IS_AIV {
        lane = 1 + AscendC::GetSubBlockIdx();
    }
    uint64_t slot = Blaze::Gemm::GetCurrentBlockIdx() * 3 + lane;
    CpuDebug::Trace trace;
    CpuDebug::trace = &trace;
    Run<Nz>(a, b, scale, yscale, out, *reinterpret_cast<const TilingData*>(tiling));
    CpuDebug::trace = nullptr;
    reinterpret_cast<CpuDebug::Trace*>(traces)[slot] = trace;
}
struct Case {
    bool nz;
    int64_t m, n, k, ka, kb;
    uint32_t baseN;
    uint16_t al, bl;
    uint8_t parallel;
    uint64_t un, uk;
};
void Execute(const Case& c)
{
    TilingData t{};
    t.mSize = c.m;
    t.nSize = c.n;
    t.kSize = c.k;
    t.baseM = 16;
    t.baseN = c.baseN;
    t.baseK = 64;
    t.al1Pingpong = c.al;
    t.bl1Pingpong = c.bl;
    t.vecCoreParallel = c.parallel;
    t.dbL0C = c.al == 1 ? 1 : 2;
    t.kAL1Tile = c.ka;
    t.kBL1Tile = c.kb;
    t.nBubSize = c.un;
    t.kBubSize = c.uk;
    t.iterateOrder = c.parallel;
    MaskPage page(MaskOffset(t, c.nz));
    ASSERT_TRUE(page.Valid()) << "Cannot map isolated mask scratch page: " << std::strerror(errno);
    size_t aBytes = c.m * c.k, bBytes = Align(c.k, 32) * Align(c.n, 32) / 2;
    size_t scaleBytes = Align((c.k + 31) / 32, 32) * Align(c.n, 32) * 2;
    GmBuffer a(aBytes), b(bBytes), scale(scaleBytes), yscale(Align(c.n, 32) * 8), output(c.m * c.n + 64),
        tiling(sizeof(t)), traces(3 * sizeof(CpuDebug::Trace));
    std::memset(a.Get(), 0x38, aBytes);
    std::memset(b.Get(), 0x22, bBytes);
    std::memset(output.Get(), 0xA5, c.m * c.n + 64);
    *reinterpret_cast<TilingData*>(tiling.Get()) = t;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    if (c.nz) {
        auto kernel = PergroupObservedKernel<true>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1, a.Get(), b.Get(), scale.Get(), yscale.Get(), output.Get(), tiling.Get(),
                                  traces.Get()));
    } else {
        auto kernel = PergroupObservedKernel<false>;
        ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1, a.Get(), b.Get(), scale.Get(), yscale.Get(), output.Get(), tiling.Get(),
                                  traces.Get()));
    }
    auto* actual = reinterpret_cast<CpuDebug::Trace*>(traces.Get());
    std::vector<CpuDebug::TileTrace> expected;
    for (int64_t row = 0; row < c.m; row += t.baseM) {
        for (int64_t col = 0; col < c.n; col += t.baseN) {
            expected.push_back({reinterpret_cast<uintptr_t>(a.Get()) + row * c.k,
                                reinterpret_cast<uintptr_t>(output.Get()) + row * c.n + col,
                                reinterpret_cast<uintptr_t>(yscale.Get()) + col * 8,
                                std::min<int64_t>(t.baseM, c.m - row), std::min<int64_t>(t.baseN, c.n - col), c.k});
        }
    }
    if (t.iterateOrder == 1)
        std::stable_sort(expected.begin(), expected.end(), [](const auto& left, const auto& right) {
            return std::make_pair(left.scale, left.a) < std::make_pair(right.scale, right.a);
        });
    ASSERT_EQ(actual[0].tiles, expected.size());
    ASSERT_LE(actual[0].tiles, 64U);
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& got = actual[0].blocks[i];
        const auto& want = expected[i];
        EXPECT_EQ(got.a, want.a);
        EXPECT_EQ(got.c, want.c);
        EXPECT_EQ(got.scale, want.scale);
        EXPECT_EQ(got.m, want.m);
        EXPECT_EQ(got.n, want.n);
        EXPECT_EQ(got.k, want.k);
    }
    EXPECT_EQ(actual[0].loads, 0U);
    EXPECT_EQ(actual[0].stores, 0U);
    uint64_t loads = 0, stores = 0;
    for (size_t lane = 1; lane < 3; ++lane) {
        loads += actual[lane].loads;
        stores += actual[lane].stores;
        if (c.nz)
            EXPECT_EQ(actual[lane].maskWords, actual[lane].maskLoads * (AscendC::VECTOR_REG_WIDTH / 8 / 8));
    }
    EXPECT_GT(loads, 0U);
    EXPECT_GT(stores, 0U);
    if (c.nz)
        EXPECT_EQ(loads * 4, stores * 5);
    else
        EXPECT_EQ(loads, stores * 3);
    if (c.bl == 1 && !c.parallel)
        EXPECT_EQ(actual[2].stores, 0U);
    for (int64_t i = c.m * c.n; i < c.m * c.n + 64; ++i)
        ASSERT_EQ(output.Get()[i], 0xA5);
}
class WqmmPergroupExecutionTest : public testing::TestWithParam<Case> {};
TEST_P(WqmmPergroupExecutionTest, ExecutesRealKernelWithIsolatedMaskPage) { Execute(GetParam()); }
INSTANTIATE_TEST_SUITE_P(BufferModes, WqmmPergroupExecutionTest,
                         testing::Values(Case{false, 16, 32, 256, 128, 128, 32, 2, 4, 0, 128, 128},
                                         Case{true, 16, 32, 256, 128, 128, 32, 2, 4, 0, 128, 128},
                                         Case{false, 17, 32, 256, 128, 128, 32, 4, 2, 0, 128, 128},
                                         Case{true, 17, 32, 256, 128, 128, 32, 4, 2, 0, 256, 128},
                                         Case{false, 16, 32, 256, 128, 128, 32, 1, 1, 0, 256, 256},
                                         Case{true, 16, 32, 256, 128, 128, 32, 1, 1, 0, 256, 256},
                                         Case{false, 16, 32, 512, 128, 512, 32, 2, 1, 1, 256, 256},
                                         Case{true, 16, 32, 512, 128, 512, 32, 2, 1, 1, 256, 256},
                                         Case{false, 16, 64, 512, 128, 512, 64, 4, 4, 0, 32, 512},
                                         Case{true, 16, 192, 128, 128, 128, 256, 2, 4, 0, 128, 128},
                                         Case{false, 16, 32, 224, 128, 256, 32, 2, 4, 0, 128, 128},
                                         Case{true, 16, 128, 224, 128, 256, 128, 4, 4, 0, 128, 128},
                                         Case{false, 33, 96, 256, 128, 128, 32, 2, 2, 0, 128, 128},
                                         Case{true, 33, 96, 256, 128, 128, 32, 2, 2, 0, 256, 128}));
} // namespace
