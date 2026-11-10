/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_weight_quant_batch_matmul.cpp
 * \brief Production template instantiation and public L1 bank address checks.
 */

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "weight_quant_batch_matmul.h"

namespace {

using Blaze::Gemm::QuantMode;
using namespace WeightQuantBatchMatmulUT;

TEST(WeightQuantBatchMatmulTest, InstantiatesSupportedB8Matrix)
{
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Fp16Int8>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Fp16Int8Offset>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Bf16Int8>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Fp16Fp8>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Bf16Fp8>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Fp16HiFloat8>, nullptr);
    ASSERT_NE(&weight_quant_batch_matmul_kernel_entry<Bf16HiFloat8>, nullptr);
}

template <typename Component>
void CheckConvertedWeightL1BankAddresses(int64_t baseN, int64_t kbL1, bool hasBias)
{
    using BlockMmad = typename Component::BlockMmad;
    using Element = typename BlockMmad::AType;
    typename BlockMmad::Params params{};
    params.l1TileShape = asc::te::make_shape(int64_t{128}, baseN, kbL1, kbL1);
    params.l0TileShape = asc::te::make_shape(int64_t{128}, baseN, int64_t{64});
    params.kSize = static_cast<uint64_t>(kbL1);
    params.hasBias = hasBias;
    BlockMmad blockMmad;
    blockMmad.Init(params);
    auto ping = blockMmad.template GetShareMemPtr<asc::te::location::l1, Element>(0U);
    auto pong = blockMmad.template GetShareMemPtr<asc::te::location::l1, Element>(1U);
    auto pingAddress = reinterpret_cast<uintptr_t>(ping.get());
    auto pongAddress = reinterpret_cast<uintptr_t>(pong.get());
    auto weightBytes = static_cast<uint64_t>(baseN * kbL1) * sizeof(Element);

    // Converted B occupies the two ends of L1. The second base depends on W,
    // not on the A half-buffer span or on the optional Bias reservation.
    EXPECT_EQ(pongAddress - pingAddress, AscendC::TOTAL_L1_SIZE - weightBytes);
    EXPECT_EQ(reinterpret_cast<uintptr_t>((pong + 1U).get()) - pongAddress, sizeof(Element));
}

TEST(WeightQuantBatchMatmulTest, SharesConvertedWeightL1BankAddresses)
{
    for (bool hasBias : {false, true}) {
        SCOPED_TRACE(hasBias);
        CheckConvertedWeightL1BankAddresses<Fp16Int8>(128, 128, hasBias);       // W = 32 KiB
        CheckConvertedWeightL1BankAddresses<Fp16Int8Offset>(128, 256, hasBias); // W = 64 KiB, FP32 Bias
        CheckConvertedWeightL1BankAddresses<Bf16Int8>(256, 256, hasBias);       // W = 128 KiB
        CheckConvertedWeightL1BankAddresses<Bf16Fp8>(128, 128, hasBias);        // BF16 with FP32 Bias
    }
}

} // namespace
