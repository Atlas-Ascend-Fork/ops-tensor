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
 * @file mat_mul_streamk_torch_binding.cpp
 * @brief Torch operator registration for the Blaze MatMul StreamK kernel.
 *
 * Registers `torch.ops.ops_tensor_examples.blaze_mat_mul_streamk` on the
 * PrivateUse1 (NPU) backend. StreamK needs a GM workspace for cross-core
 * partial accumulation; it is allocated here as a torch byte tensor sized with
 * the largest possible L0 tile (256x256 fp32 per core) plus RPC padding.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../mat_mul_torch_common.h"

// ASC-compiled launchers (mat_mul_streamk_torch_kernel.cpp)
extern "C" void launch_mat_mul_streamk_f16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                           int64_t m, int64_t n, int64_t k, int64_t blockNum);
extern "C" void launch_mat_mul_streamk_bf16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                            int64_t m, int64_t n, int64_t k, int64_t blockNum);
extern "C" void launch_mat_mul_streamk_f32(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                           int64_t m, int64_t n, int64_t k, int64_t blockNum);

namespace {

constexpr int64_t MAX_L0_TILE = 256; // largest baseM/baseN in the StreamK tiling policy
constexpr int64_t RPC_WORKSPACE_PADDING = 20L * 1024L * 1024L;

at::Tensor MatMulStreamKNpu(const at::Tensor& a, const at::Tensor& b)
{
    mat_mul_torch::CheckMatMulInputs(a, b, "blaze_mat_mul_streamk");
    const int64_t m = a.size(0);
    const int64_t n = b.size(1);
    if (m == 0 || n == 0) {
        // Zero-sized problems return an empty output without launching the
        // kernel: the StreamK tiling would divide by a zero tile count.
        const c10_npu::OptionalNPUGuard guard(a.device());
        return at::empty({m, n}, a.options());
    }

    const int64_t blockNum = mat_mul_torch::GetAicCoreNum();
    // Same sizing policy as acl/mat_mul/mat_mul_streamk: per-core L0C tile in fp32 + RPC padding.
    const int64_t workspaceSize = blockNum * MAX_L0_TILE * MAX_L0_TILE * static_cast<int64_t>(sizeof(float)) +
                                  RPC_WORKSPACE_PADDING;
    at::Tensor workspace = at::empty({workspaceSize}, a.options().dtype(at::kByte));

    return mat_mul_torch::LaunchMatMulNpu(
        a, b, {launch_mat_mul_streamk_f16, launch_mat_mul_streamk_bf16, launch_mat_mul_streamk_f32},
        static_cast<uint8_t*>(workspace.data_ptr()));
}

} // namespace

TORCH_LIBRARY_FRAGMENT(ops_tensor_examples, m) { m.def("blaze_mat_mul_streamk(Tensor a, Tensor b) -> Tensor"); }

TORCH_LIBRARY_IMPL(ops_tensor_examples, PrivateUse1, m) { m.impl("blaze_mat_mul_streamk", &MatMulStreamKNpu); }
