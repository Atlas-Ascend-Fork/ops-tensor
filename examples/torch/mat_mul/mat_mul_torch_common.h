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
 * @file mat_mul_torch_common.h
 * @brief Shared host-side helpers for the torch mat_mul variant bindings.
 *
 * The four variants (basic / a_fullload / b_fullload / streamk) share the same
 * input validation, output allocation, stream handling and dtype dispatch.
 * Each binding only keeps its variant-specific checks (e.g. the fullload shape
 * limits) and resources (e.g. the StreamK workspace), then forwards its
 * extern "C" launchers to LaunchMatMulNpu.
 */

#ifndef MAT_MUL_TORCH_COMMON_H
#define MAT_MUL_TORCH_COMMON_H

#include <ATen/ATen.h>

#include "acl/acl.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace mat_mul_torch {

// Uniform launcher signature; variants without a workspace ignore that argument.
using LaunchFn = void (*)(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, int64_t m,
                          int64_t n, int64_t k, int64_t blockNum);

struct DTypeLaunchers {
    LaunchFn f16;
    LaunchFn bf16;
    LaunchFn f32;
};

inline int64_t GetAicCoreNum()
{
    static int64_t aicNum = [] {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        return platform == nullptr ? 0 : static_cast<int64_t>(platform->GetCoreNumAic());
    }();
    TORCH_CHECK(aicNum > 0, "GetCoreNumAic() failed, cannot determine AIC core num");
    return aicNum;
}

inline void CheckMatMulInputs(const at::Tensor& a, const at::Tensor& b, const char* opName)
{
    TORCH_CHECK(a.device().type() == c10::kPrivateUse1, opName, ": a must be on NPU");
    TORCH_CHECK(b.device().type() == c10::kPrivateUse1, opName, ": b must be on NPU");
    TORCH_CHECK(a.device() == b.device(), opName, ": a and b must live on the same NPU device, got ", a.device(),
                " vs ", b.device());
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, opName, ": a and b must be 2-D, got ", a.dim(), "-D and ", b.dim(), "-D");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), opName, ": a and b must have the same dtype");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), opName, ": a and b must be contiguous (ND layout)");
    TORCH_CHECK(b.size(0) == a.size(1), opName, ": shape mismatch: a is [", a.size(0), ", ", a.size(1), "] but b is [",
                b.size(0), ", ", b.size(1), "]");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat,
                opName, " only supports float16/bfloat16/float32, got ", a.scalar_type());
}

// Common execution path after validation: zero-sized problems are resolved
// without a kernel launch (m/n=0 gives an empty output, k=0 is an empty
// reduction and must return zeros, not uninitialized memory); otherwise
// allocates the output, takes the current torch_npu stream and dispatches on
// dtype.
inline at::Tensor LaunchMatMulNpu(const at::Tensor& a, const at::Tensor& b, const DTypeLaunchers& launchers,
                                  uint8_t* workspace = nullptr)
{
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);

    // Dispatch on the device the inputs live on (multi-card safe), then take
    // that device's current stream.
    const c10_npu::OptionalNPUGuard guard(a.device());
    if (k == 0) {
        return at::zeros({m, n}, a.options());
    }
    at::Tensor c = at::empty({m, n}, a.options());
    if (m == 0 || n == 0) {
        return c;
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream(a.device().index()).stream();
    auto* aPtr = static_cast<uint8_t*>(a.data_ptr());
    auto* bPtr = static_cast<uint8_t*>(b.data_ptr());
    auto* cPtr = static_cast<uint8_t*>(c.data_ptr());
    const int64_t blockNum = GetAicCoreNum();

    switch (a.scalar_type()) {
        case at::kHalf:
            launchers.f16(stream, aPtr, bPtr, cPtr, workspace, m, n, k, blockNum);
            break;
        case at::kBFloat16:
            launchers.bf16(stream, aPtr, bPtr, cPtr, workspace, m, n, k, blockNum);
            break;
        case at::kFloat:
            launchers.f32(stream, aPtr, bPtr, cPtr, workspace, m, n, k, blockNum);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype ", a.scalar_type());
    }
    return c;
}

} // namespace mat_mul_torch

#endif // MAT_MUL_TORCH_COMMON_H
