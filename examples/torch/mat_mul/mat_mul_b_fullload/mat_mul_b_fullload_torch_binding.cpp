/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file mat_mul_b_fullload_torch_binding.cpp
 * @brief Torch operator registration for the Blaze MatMul BFullLoad kernel.
 *
 * Registers `torch.ops.ops_tensor_examples.blaze_mat_mul_b_fullload` on the
 * PrivateUse1 (NPU) backend. BFullLoad keeps the whole B matrix resident in L1,
 * so N must be small (n <= kBFullLoadMaxN).
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../mat_mul_torch_common.h"

// ASC-compiled launchers (mat_mul_b_fullload_torch_kernel.cpp); workspace is unused.
extern "C" void launch_mat_mul_b_fullload_f16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                              uint8_t* workspace, int64_t m, int64_t n, int64_t k, int64_t blockNum);
extern "C" void launch_mat_mul_b_fullload_bf16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                               uint8_t* workspace, int64_t m, int64_t n, int64_t k, int64_t blockNum);
extern "C" void launch_mat_mul_b_fullload_f32(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                              uint8_t* workspace, int64_t m, int64_t n, int64_t k, int64_t blockNum);

namespace {

// BFullLoad keeps the whole B matrix in L1 with nL1 = n.
constexpr int64_t kBFullLoadMaxN = 256;

at::Tensor MatMulBFullLoadNpu(const at::Tensor& a, const at::Tensor& b)
{
    mat_mul_torch::CheckMatMulInputs(a, b, "blaze_mat_mul_b_fullload");
    const int64_t n = b.size(1);
    TORCH_CHECK(n <= kBFullLoadMaxN, "blaze_mat_mul_b_fullload requires n <= ", kBFullLoadMaxN,
                " (B must fit in L1), got n=", n);
    return mat_mul_torch::LaunchMatMulNpu(
        a, b, {launch_mat_mul_b_fullload_f16, launch_mat_mul_b_fullload_bf16, launch_mat_mul_b_fullload_f32});
}

} // namespace

TORCH_LIBRARY_FRAGMENT(ops_tensor_examples, m) { m.def("blaze_mat_mul_b_fullload(Tensor a, Tensor b) -> Tensor"); }

TORCH_LIBRARY_IMPL(ops_tensor_examples, PrivateUse1, m) { m.impl("blaze_mat_mul_b_fullload", &MatMulBFullLoadNpu); }
