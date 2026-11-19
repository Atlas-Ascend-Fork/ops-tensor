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
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "grouped_matmul_mx_a8w4/grouped_matmul_mx_a8w4_cpu_debug_stub.h"
#if defined(ASCENDC_CPU_DEBUG)
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#define __fp4e1m2x2 fp4x2_e1m2_t
#endif
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e1m2x2
#undef __fp4e2m1x2
#undef __fp8e4m3
#endif
#include "quant_grouped_matmul/quant_grouped_matmul_cpu_debug_stub.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace {
using Nd = asc::te::nd_ext_layout_ptn;
using Scale = fp8_e8m0_t;
using Mmad = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::GroupedMatmulSwigluWithWeightQuantMx, AscendC::Std::tuple<fp8_e4m3fn_t, Scale>,
    AscendC::Std::tuple<Nd, asc::te::scalea_nd_layout_ptn>, AscendC::Std::tuple<fp4x2_e2m1_t, Scale>,
    AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, float, Nd, bfloat16_t, Nd>;
using Prologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, true>;
using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<fp8_e4m3fn_t, float, Scale, false, true, true>;
using Kernel = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<asc::te::shape<int64_t, int64_t, int64_t, int64_t>, Mmad,
                                                           Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit,
                                                           Epilogue, Prologue, true>;

struct ScaleACase {
    int64_t k;
    int64_t expertRowOffset;
    int64_t tileRowOffset;
    int64_t rows;
};

class WqgmmSwigluMxScaleANdTest : public testing::TestWithParam<ScaleACase> {};

TEST_P(WqgmmSwigluMxScaleANdTest, ProductionTensorReadsRowMajorBytesAtExpertAndTileOffsets)
{
    const auto& item = GetParam();
    // Reference storage is the ACLNN [M, K/64, 2] input, with distinct adjacent rows and scale pairs.
    const int64_t rowPitch = item.k / 32L;
    const int64_t totalRows = item.expertRowOffset + item.tileRowOffset + item.rows;
    std::vector<uint8_t> storage(static_cast<size_t>(totalRows * rowPitch));
    for (int64_t row = 0L; row < totalRows; ++row) {
        for (int64_t col = 0L; col < rowPitch; ++col) {
            storage[static_cast<size_t>(row * rowPitch + col)] = static_cast<uint8_t>((row * 37 + col * 11) % 251);
        }
    }
    Kernel::Params params{};
    params.xScale = reinterpret_cast<GM_ADDR>(storage.data());
    params.k = item.k;
    auto tensor = Kernel::MakeScaleATensor(params, item.expertRowOffset, item.tileRowOffset, item.rows);
    static_assert(std::is_same_v<asc::te::get_layout_pattern<typename decltype(tensor)::layout_type>,
                                 asc::te::scalea_nd_layout_ptn>);
    for (int64_t row = 0L; row < item.rows; ++row) {
        for (int64_t col = 0L; col < rowPitch; ++col) {
            SCOPED_TRACE(testing::Message() << "K=" << item.k << ", row=" << row << ", scale=" << col);
            auto element = tensor.slice(asc::te::make_coord(row, col), asc::te::make_shape(1L, 1L));
            const auto* address = reinterpret_cast<const uint8_t*>(element.data().get());
            const int64_t expectedOffset = (item.expertRowOffset + item.tileRowOffset + row) * rowPitch + col;
            // Check the real TensorAPI address before dereferencing, so a ZZ regression cannot read out of bounds.
            ASSERT_EQ(reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(storage.data()),
                      static_cast<uintptr_t>(expectedOffset));
            EXPECT_EQ(*address, storage[static_cast<size_t>(expectedOffset)]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GmInputLayout, WqgmmSwigluMxScaleANdTest,
                         testing::Values(ScaleACase{128L, 0L, 0L, 1L}, ScaleACase{128L, 0L, 0L, 3L},
                                         ScaleACase{256L, 3L, 0L, 17L}, ScaleACase{384L, 3L, 128L, 17L},
                                         ScaleACase{4096L, 19L, 128L, 3L}));
} // namespace
