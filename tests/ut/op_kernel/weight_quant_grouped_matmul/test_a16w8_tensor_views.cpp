/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/layout_struct.h"

namespace {
template <bool Trans>
void CheckWeightViews()
{
    using Pattern = AscendC::Std::conditional_t<Trans, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    // Non-aligned K/N and a later expert expose element/byte and batch-stride mistakes.
    constexpr int64_t E = 4, K = 529, N = 273;
    auto layout = asc::te::make_frame_layout<Pattern>(E, K, N);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t k : {0L, 256L, 528L}) {
            for (int64_t n : {0L, 128L, 272L}) {
                EXPECT_EQ(int64_t(layout(asc::te::make_coord(e, asc::te::make_coord(k, n)))),
                          e * K * N + (Trans ? n * K + k : k * N + n));
            }
        }
    }
    auto l1 = asc::te::make_frame_layout<
        AscendC::Std::conditional_t<Trans, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        asc::te::layout_trait_default<half>>(17L, 33L);
    for (int64_t k = 0; k < 17; ++k) {
        for (int64_t n = 0; n < 33; ++n) {
            const int64_t expected = Trans ? k / 16 * 48 * 16 + n * 16 + k % 16 : n / 16 * 32 * 16 + k * 16 + n % 16;
            EXPECT_EQ(int64_t(l1(asc::te::make_coord(k, n))), expected);
        }
    }
}
TEST(GroupedA16W8Views, BatchedNdNnAndNtTails)
{
    CheckWeightViews<false>();
    CheckWeightViews<true>();
}

TEST(GroupedA16W8Views, PaddedVfPitchDoesNotShrinkOnTail)
{
    auto nz = Blaze::Gemm::NzColPaddingUBLayout<half>{}(17, 33, 129 * 16);
    auto zn = Blaze::Gemm::ZnRowPaddingUBLayout<bfloat16_t>{}(17, 33, 129 * 16);
    for (int64_t k = 0; k < 17; ++k) {
        for (int64_t n = 0; n < 33; ++n) {
            EXPECT_EQ(int64_t(nz(asc::te::make_coord(k, n))), n / 16 * 129 * 16 + k * 16 + n % 16);
            EXPECT_EQ(int64_t(zn(asc::te::make_coord(k, n))), k / 16 * 129 * 16 + n * 16 + k % 16);
        }
    }
}

// Validate the physical footprints at the VF/copy boundary, including the second UB half of BL1.
template <bool Trans_>
void CheckUbToBl1Coverage()
{
    SCOPED_TRACE(::testing::Message() << "orientation=" << (Trans_ ? "NT" : "NN"));
    using L1Pattern = AscendC::Std::conditional_t<Trans_, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using UbLayout = AscendC::Std::conditional_t<Trans_, Blaze::Gemm::ZnRowPaddingUBLayout<half>,
                                                 Blaze::Gemm::NzColPaddingUBLayout<half>>;
    constexpr int64_t UB_K = Trans_ ? 256 : 128;
    constexpr int64_t UB_N = Trans_ ? 128 : 256;
    constexpr int64_t PITCH = (Trans_ ? UB_N + 1 : UB_K + 1) * 16;
    constexpr int64_t OUTPUT_SLOT_ELEMENTS = 129 * 256;
    for (int64_t kL1 : {1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256}) {
        for (int64_t n : {1, 15, 16, 17, 127, 128, 129, 255, 256}) {
            SCOPED_TRACE(::testing::Message() << "kL1=" << kL1 << ", n=" << n);
            const int64_t alignedK = (kL1 + 15) / 16 * 16;
            const int64_t alignedN = (n + 15) / 16 * 16;
            auto l1 = asc::te::make_frame_layout<L1Pattern, asc::te::layout_trait_default<half>>(kL1, n);
            std::vector<uint8_t> coverage(alignedK * alignedN, 0);
            for (int64_t start = 0; start < kL1; start += UB_K) {
                const int64_t tileK = kL1 - start < UB_K ? kL1 - start : UB_K;
                for (int64_t nStart = 0; nStart < n; nStart += UB_N) {
                    const int64_t tileN = n - nStart < UB_N ? n - nStart : UB_N;
                    SCOPED_TRACE(::testing::Message() << "kStart=" << start << ", nStart=" << nStart
                                                      << ", tileK=" << tileK << ", tileN=" << tileN);
                    // Use the same full Tensor / effective slice contract as the prologue.
                    auto output = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, half>(uint64_t{0}),
                                                       UbLayout{}(UB_K, UB_N, PITCH));
                    auto outputTile = output.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(tileK, tileN));
                    const auto& fullLayout = output.layout();
                    const auto& copyLayout = outputTile.layout();
                    using Pattern = asc::te::get_layout_pattern<typename decltype(outputTile)::layout_type>;
                    using ExpectedPattern = AscendC::Std::conditional_t<Trans_, Blaze::Gemm::zn_row_padding_layout_ptn,
                                                                        Blaze::Gemm::nz_col_padding_layout_ptn>;
                    static_assert(AscendC::Std::is_same_v<Pattern, ExpectedPattern>);
                    ASSERT_EQ(asc::te::size<0>(fullLayout), UB_K);
                    ASSERT_EQ(asc::te::size<1>(fullLayout), UB_N);
                    const int64_t copyK = Trans_ ? (tileK + 15) / 16 * 16 : tileK;
                    const int64_t copyN = Trans_ ? tileN : (tileN + 15) / 16 * 16;
                    ASSERT_EQ(asc::te::size<0>(copyLayout), copyK);
                    ASSERT_EQ(asc::te::size<1>(copyLayout), copyN);
                    // VF writes a full inner register width even on an inner-dimension tail.
                    const int64_t writtenK = Trans_ ? UB_K : tileK;
                    const int64_t writtenN = Trans_ ? tileN : UB_N;
                    for (int64_t k = 0; k < writtenK; ++k) {
                        for (int64_t col = 0; col < writtenN; ++col) {
                            const int64_t physical = Trans_ ? k / 16 * PITCH + col * 16 + k % 16 :
                                                              col / 16 * PITCH + k * 16 + col % 16;
                            ASSERT_LT(k, asc::te::size<0>(fullLayout));
                            ASSERT_LT(col, asc::te::size<1>(fullLayout));
                            ASSERT_EQ(int64_t(fullLayout(asc::te::make_coord(k, col))), physical);
                            ASSERT_LT(physical, OUTPUT_SLOT_ELEMENTS);
                        }
                    }
                    // Include the rounded C0 lanes actually transferred by MTE3, not just valid elements.
                    for (int64_t k = 0; k < copyK; ++k) {
                        for (int64_t col = 0; col < copyN; ++col) {
                            ASSERT_LT(k, writtenK);
                            ASSERT_LT(col, writtenN);
                            const int64_t source = copyLayout(asc::te::make_coord(k, col));
                            const int64_t expectedSource = Trans_ ? k / 16 * PITCH + col * 16 + k % 16 :
                                                                    col / 16 * PITCH + k * 16 + col % 16;
                            ASSERT_EQ(source, expectedSource);
                            const int64_t dest = l1(asc::te::make_coord(start + k, nStart + col));
                            ASSERT_GE(dest, 0);
                            ASSERT_LT(dest, static_cast<int64_t>(coverage.size()));
                            ASSERT_EQ(coverage[dest]++, 0);
                        }
                    }
                }
            }
            for (int64_t k = 0; k < kL1; ++k) {
                for (int64_t col = 0; col < n; ++col) {
                    EXPECT_EQ(coverage[l1(asc::te::make_coord(k, col))], 1);
                }
            }
        }
    }
}

TEST(GroupedA16W8Views, TwoOrientationSpecificVfTilesCoverBl1WithoutOverlap)
{
    CheckUbToBl1Coverage<false>();
    CheckUbToBl1Coverage<true>();
}
TEST(GroupedA16W8Views, InputStageSlicesRetainFullRowPitch)
{
    auto nn = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(uint64_t{0}),
                                   asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(256L, 256L));
    auto nt = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(uint64_t{0}),
                                   asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(256L, 256L));
    auto nnSecond = nn.slice(asc::te::make_coord(128L, 0L), asc::te::make_shape(128L, 256L));
    auto ntSecond = nt.slice(asc::te::make_coord(0L, 128L), asc::te::make_shape(256L, 128L));
    EXPECT_EQ(int64_t(nnSecond.layout()(asc::te::make_coord(1L, 1L))), 257);
    EXPECT_EQ(int64_t(ntSecond.layout()(asc::te::make_coord(1L, 1L))), 257);
    EXPECT_EQ(int64_t(nn.layout()(asc::te::make_coord(128L, 0L))), 32768);
    EXPECT_EQ(int64_t(nt.layout()(asc::te::make_coord(0L, 128L))), 32768);
}
} // namespace
