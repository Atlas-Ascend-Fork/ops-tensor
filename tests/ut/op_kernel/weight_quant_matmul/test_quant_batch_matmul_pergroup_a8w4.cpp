/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_batch_matmul_pergroup_a8w4.cpp
 * \brief Compile-instantiation and layout-contract tests for the T-CG per-group A8W4 kernel.
 *
 * Runtime coverage is provided separately by test_wqmm_pergroup_execution.cpp.
 * Its fixture maps only the scale-mask scratch page for direct device-address
 * stores and translates register accesses into tikicpulib's physical UB. These
 * tests retain the public template-instantiation and layout contracts.
 */

#include <cstdint>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"

#include "qbmm_pergroup_a8w4.h"

namespace {

// Taking the entry address forces full template instantiation: BlockMmad<MatmulWithWeightQuantPergroup>,
// KernelPergroupWeightPrologue, BlockSchedulerWqmmBlockSplit and the GemmUniversal
// KernelMixWithWeightPergroupPrologue specialization are all compiled (ND and NZ variants).
TEST(QbmmPergroupA8W4KernelTest, EntryInstantiation)
{
    static_assert(!std::is_same_v<decltype(QbmmPergroupA8W4KernelEntry<false>), std::nullptr_t>,
                  "ND entry must instantiate");
    static_assert(!std::is_same_v<decltype(QbmmPergroupA8W4KernelEntry<true>), std::nullptr_t>,
                  "NZ entry must instantiate");
    ASSERT_NE(&QbmmPergroupA8W4KernelEntry<false>, nullptr);
    ASSERT_NE(&QbmmPergroupA8W4KernelEntry<true>, nullptr);
}

// ZN-family converted-weight UB layout (ZnColPaddingLayoutPtn): k1 K-slabs, each holding nSize adjacent
// 32B blocks, with the slab pitch carrying the padding gap (Align16(N) + 1 blocks per slab).
TEST(QbmmPergroupA8W4KernelTest, ZnColPaddingLayoutKeepsSlabGeometry)
{
    const auto layout = Blaze::Gemm::ZnColPaddingLayout<int8_t>{}(32, 8);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.shape())), 1); // k1 = CeilDiv(32, 32)
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layout.shape())), 8); // nSize
    // Slab pitch (stride<1><0>) = (Align16(8) + 1) * C0 = 17 * 32.
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.stride())), 17 * 32);
}

TEST(QbmmPergroupA8W4KernelTest, ZnColPaddingLayoutTailK)
{
    const auto layout = Blaze::Gemm::ZnColPaddingLayout<int8_t>{}(33, 17);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.shape())), 2); // k1 = CeilDiv(33, 32)
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layout.shape())), 17);
    // Slab pitch = (Align16(17) + 1) * 32 = 33 * 32.
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.stride())), 33 * 32);
}

// The legacy Weight8BitDnToZnUbLayoutPtn tag and the new-style ZnColPaddingLayoutPtn must stay
// geometrically identical: both route to one CopyPaddedUBToL1 converted-weight branch.
TEST(QbmmPergroupA8W4KernelTest, ZnColPaddingLegacyTagSharesGeometry)
{
    const auto legacy = Blaze::Gemm::Weight8BitDnToZnUBLayout<int8_t>{}(96, 48);
    const auto modern = Blaze::Gemm::ZnColPaddingLayout<int8_t>{}(96, 48);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(legacy.shape())),
              AscendC::Std::get<1>(AscendC::Std::get<0>(modern.shape())));
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(legacy.shape())),
              AscendC::Std::get<1>(AscendC::Std::get<1>(modern.shape())));
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(legacy.stride())),
              AscendC::Std::get<1>(AscendC::Std::get<0>(modern.stride())));
}

// NZ-family converted-weight UB layout (NzRowPaddingLayoutPtn): 256-element vector chunks, K groups of
// 32, four vector loops per group, N1 fractals of 32; InnerStride carries the ping-pong interleave.
TEST(QbmmPergroupA8W4KernelTest, NzRowPaddingLayoutKeepsChunkGeometry)
{
    constexpr int64_t INTERLEAVE = 1024;
    const auto layout = Blaze::Gemm::NzRowPaddingLayout<int8_t>{}(96, 64, INTERLEAVE);
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<0>(layout.shape())), 256); // vector chunk
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.shape())), 3);   // kGroupNum = 96 / 32
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(layout.shape())), 4);   // VL loops per group
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layout.shape())), 2);   // n1 = CeilDiv(64, 32)
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(layout.stride())), INTERLEAVE);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.stride())), INTERLEAVE * 4);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layout.stride())), INTERLEAVE * 4 * 3);
}

// kGroupNum floors validK / 32: a trailing partial K group is not converted (the Dequant VF loop and
// the related copy chunk counts all take the same floor).
TEST(QbmmPergroupA8W4KernelTest, NzRowPaddingLayoutFloorsPartialKGroup)
{
    const auto layout = Blaze::Gemm::NzRowPaddingLayout<int8_t>{}(97, 32, 512);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layout.shape())), 3); // floor(97 / 32)
}

} // namespace
