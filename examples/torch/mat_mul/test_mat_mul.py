# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
Torch frontend tests for the Blaze MatMul examples.

Loads the extension built by CMake (build/ops_tensor_examples.so) and checks the
registered torch.ops.ops_tensor_examples.* ops against torch.matmul for
fp16/bf16/fp32. Variants mirror the C++ examples under examples/acl/mat_mul/:

    blaze_mat_mul            basic multi-block tiling (acl/mat_mul/mat_mul_basic)
    blaze_mat_mul_a_fullload A fully resident in L1, small M (mat_mul_a_fullload)
    blaze_mat_mul_b_fullload B fully resident in L1, small N (mat_mul_b_fullload)
    blaze_mat_mul_streamk    StreamK K-split scheduling   (mat_mul_streamk)

Usage:
    source <cann>/set_env.sh
    python3 test_mat_mul.py                       # all variants, default shapes
    python3 test_mat_mul.py blaze_mat_mul 512 512 512
"""

import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "build", "ops_tensor_examples.so")

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# fp16/bf16 accumulation differences vs the fp32 CPU-side reference
TOLERANCES = {
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (1e-4, 1e-4),
}
# StreamK splits K across cores and accumulates fp32 partials on GM in a
# different order than the reference, so fp32 needs a looser bound.
VARIANT_TOLERANCES = {
    ("blaze_mat_mul_streamk", torch.float32): (1e-3, 1e-3),
}

# Default shapes follow the C++ examples' .csv cases (acl/mat_mul/*/mat_mul_*.csv)
VARIANTS = {
    "blaze_mat_mul": (1024, 512, 768),
    "blaze_mat_mul_a_fullload": (32, 512, 256),  # AFullLoad: m <= 256
    "blaze_mat_mul_b_fullload": (256, 512, 32),  # BFullLoad: n <= 256
    "blaze_mat_mul_streamk": (100, 8192, 100),  # StreamK: few tiles, large K
}

# Zero-sized problems must not launch the kernel: m=0/n=0 return an empty
# output (StreamK previously died with SIGFPE from a zero tile count in
# GetTiling); k=0 is an empty reduction and must return zeros, not
# uninitialized memory.
ZERO_SIZE_CASES = [(0, 512, 128), (128, 512, 0), (64, 0, 128)]


def run_variant(name, m, k, n):
    op = getattr(torch.ops.ops_tensor_examples, name)
    all_ok = True
    for dtype in DTYPES:
        torch.manual_seed(0)  # reproducible inputs
        a = torch.randn(m, k, dtype=dtype).npu()
        b = torch.randn(k, n, dtype=dtype).npu()

        c = op(a, b)

        # reference in fp32 on CPU for a stable golden baseline
        ref = torch.matmul(a.cpu().float(), b.cpu().float())
        rtol, atol = VARIANT_TOLERANCES.get((name, dtype), TOLERANCES[dtype])
        ok = torch.allclose(c.cpu().float(), ref, rtol=rtol, atol=atol)
        max_diff = (c.cpu().float() - ref).abs().max().item()
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(
            f"[{status}] {name:28s} dtype={str(dtype):15s} out={tuple(c.shape)} max_abs_diff={max_diff:.6f}"
        )
    return all_ok


def run_zero_size(name):
    op = getattr(torch.ops.ops_tensor_examples, name)
    all_ok = True
    for m, k, n in ZERO_SIZE_CASES:
        a = torch.randn(m, k, dtype=torch.float16).npu()
        b = torch.randn(k, n, dtype=torch.float16).npu()
        c = op(a, b)
        ok = tuple(c.shape) == (m, n)
        if m == 0 or n == 0:
            ok &= c.numel() == 0
        else:  # k == 0: empty reduction, output must be all zeros
            ok &= bool((c == 0).all().item())
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name:28s} zero-size M={m}, K={k}, N={n} out={tuple(c.shape)}"
        )
    return all_ok


def main():
    args = sys.argv[1:]
    if args:
        name = args[0]
        if name not in VARIANTS:
            print(f"[ERROR] unknown variant '{name}', choose from: {list(VARIANTS)}")
            return 1
        m, k, n = (int(x) for x in args[1:4]) if len(args) >= 4 else VARIANTS[name]
        cases = {name: (m, k, n)}
    else:
        cases = VARIANTS

    if not os.path.exists(LIB_PATH):
        print(f"[ERROR] extension not found: {LIB_PATH}")
        print("        build first: cmake -S . -B build && cmake --build build -j")
        return 1

    torch.ops.load_library(LIB_PATH)
    print(f"[INFO] loaded extension: {LIB_PATH}")

    all_ok = True
    for name, (m, k, n) in cases.items():
        print(f"[INFO] {name}: M={m}, K={k}, N={n}")
        all_ok &= run_variant(name, m, k, n)
        all_ok &= run_zero_size(name)

    print("=== ALL PASS ===" if all_ok else "=== FAILED ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
