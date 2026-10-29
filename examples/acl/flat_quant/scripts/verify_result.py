#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Verify flat_quant NPU outputs against the float32 golden reference.

Outputs of the kernel per batch b (row-major MxN matrix, flattened to mn elements):
  - npu_y.bin: packed MX-FP4 (E2M1) values, two nibbles per byte (low = even element,
    high = odd element), batch b at byte offset b * mn / 2.
  - npu_scale.bin: one E8M0 scale byte per 32-element group, batch b at byte offset
    b * ceil(mn / 64) * 2 (== b * mn / 32 when mn % 64 == 0).

Checks:
  - scale: the NPU scale byte must match the golden DYN_DTYPE_RANGE exponent derived
    from the BF16-rounded golden reference within +/-1 (a 1-ulp BF16 difference of the
    group amax may flip the rounded exponent by one).
  - y: dequantized values (nibble value * 2^(scale-127)) must match the golden float
    reference within 1.1x the group scale, which bounds the FP4 round-to-nearest error
    plus BF16 rounding of the matmul result.
"""

import argparse
import os
import sys

import ml_dtypes
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json


GROUP_SIZE = 32
FP4_LUT = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
SCALE_EXP_TOL = 1
DEQUANT_TOL_FACTOR = 1.1


def golden_scale_bytes(ref_bf16_flat):
    """Replicate the DYN_DTYPE_RANGE scale generation for FP4-E2M1 (dstTypeMax=6)."""
    bits = ref_bf16_flat.view(np.uint16) & 0x7FFF
    amax_bits = bits.reshape(-1, GROUP_SIZE).max(axis=1)
    exp_field = (amax_bits >> 7) & 0xFF
    mantissa = amax_bits & 0x7F
    rounded = exp_field + (mantissa >= 65)
    # exponent fields below 2 (tiny amax) are clamped to a rounded exponent of 2
    byte = np.where(exp_field < 2, 0, rounded.astype(np.int32) - 2)
    return byte.astype(np.uint8)


def verify(golden_dir, actual_dir, k, m, n):
    mn = m * n
    groups = mn // GROUP_SIZE
    golden = np.fromfile(
        os.path.join(golden_dir, "golden_ref.bin"), dtype=np.float32
    ).reshape(k, m, n)
    npu_y = np.fromfile(os.path.join(actual_dir, "npu_y.bin"), dtype=np.uint8)
    npu_scale = np.fromfile(os.path.join(actual_dir, "npu_scale.bin"), dtype=np.uint8)

    if npu_y.size != k * mn // 2:
        raise ValueError(f"NPU y size ({npu_y.size}) != expected {k * mn // 2}")
    if npu_scale.size != k * ((mn + 63) // 64) * 2:
        raise ValueError(
            f"NPU scale size ({npu_scale.size}) != expected {k * ((mn + 63) // 64) * 2}"
        )

    y_errors = 0
    scale_errors = 0
    y_max_norm_diff = 0.0
    for b in range(k):
        ref = golden[b].reshape(-1)
        ref_bf16 = golden[b].astype(ml_dtypes.bfloat16).reshape(-1)

        scale_golden = golden_scale_bytes(ref_bf16)
        scale_npu = npu_scale[b * groups : (b + 1) * groups]
        scale_diff = np.abs(scale_npu.astype(np.int32) - scale_golden.astype(np.int32))
        scale_errors += int(np.count_nonzero(scale_diff > SCALE_EXP_TOL))

        packed = npu_y[b * mn // 2 : (b + 1) * mn // 2]
        codes = np.empty(mn, dtype=np.uint8)
        codes[0::2] = np.bitwise_and(packed, 0x0F)
        codes[1::2] = np.right_shift(packed, 4)
        values = FP4_LUT[codes]

        scale_elem = np.exp2(scale_npu.astype(np.float32) - 127.0)
        scale_per_elem = np.repeat(scale_elem, GROUP_SIZE)
        dequant = values * scale_per_elem

        abs_diff = np.abs(dequant - ref)
        tolerance = DEQUANT_TOL_FACTOR * scale_per_elem + 1.0e-9
        y_errors += int(np.count_nonzero(abs_diff > tolerance))
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_diff = np.where(
                scale_per_elem > 0, abs_diff / scale_per_elem, abs_diff
            )
        y_max_norm_diff = max(y_max_norm_diff, float(np.max(norm_diff)))

    y_error_ratio = y_errors / (k * mn)
    scale_error_ratio = scale_errors / (k * groups)
    print(
        f"[verify] y: max abs diff: {y_max_norm_diff:.6e} (unit: group scale), "
        f"error ratio: {y_error_ratio:.6f}"
    )
    print(
        f"[verify] scale: count(|byte diff| > {SCALE_EXP_TOL}): {scale_errors}/{k * groups}, "
        f"error ratio: {scale_error_ratio:.6f}"
    )

    metrics = [
        {
            "name": "y",
            "max_abs_diff": y_max_norm_diff,
            "error_ratio": y_error_ratio,
            "status": "pass" if y_error_ratio == 0.0 else "fail",
        },
        {
            "name": "scale",
            "max_abs_diff": float(scale_errors),
            "error_ratio": scale_error_ratio,
            "status": "pass" if scale_error_ratio == 0.0 else "fail",
        },
    ]
    passed = all(o["status"] == "pass" for o in metrics)
    write_metrics_json(metrics, "pass" if passed else "fail", actual_dir)
    print(
        "[PASS] NPU results are consistent with CPU."
        if passed
        else "[FAIL] NPU results differ from CPU."
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify flat_quant NPU outputs.")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--actual-dir", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--iter-batch", type=int, required=True)
    parser.add_argument("--has-p2", type=lambda v: v.lower() == "true", required=True)
    args = parser.parse_args()
    return 0 if verify(args.golden_dir, args.actual_dir, args.k, args.m, args.n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
