#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compare the NPU MX quant output (Y + Y_scale) with the golden result.

Y dtype follows A dtype: fp8_e4m3, fp8_e5m2, or fp4_e2m1.
Y_scale is always fp8_e8m0.
"""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json

import ml_dtypes
import numpy as np

_COLLECTED_METRICS = []


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2
FP4_E2M1_TO_FP32 = (
    np.arange(16, dtype=np.int8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
)

_DTYPE_MAP = {"fp8_e4m3": FP8_E4M3FN, "fp8_e5m2": FP8_E5M2}


def unpack_fp4(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = np.bitwise_and(packed, 0x0F)
    codes[1::2] = np.right_shift(packed, 4)
    return codes


def load_y_as_fp32(path, dtype):
    raw = np.fromfile(path, dtype=np.uint8)
    if dtype == "fp4_e2m1":
        codes = unpack_fp4(raw)
        return FP4_E2M1_TO_FP32[codes]
    return raw.view(_DTYPE_MAP[dtype]).astype(np.float32)


def compare(name, golden_path, actual_path, dtype, is_scale=False):
    if is_scale:
        # Compare E8M0 bytes exactly, including paired-scale padding.
        golden = np.fromfile(golden_path, dtype=np.uint8)
        actual = np.fromfile(actual_path, dtype=np.uint8)
        atol = 0.0
        rtol = 0.0
    else:
        golden = load_y_as_fp32(golden_path, dtype)
        actual = load_y_as_fp32(actual_path, dtype)
        atol = 1.0
        rtol = 1e-3

    if golden.size != actual.size:
        raise ValueError(
            f"{name}: size mismatch: expected {golden.size}, got {actual.size}"
        )
    n = golden.size
    if n == 0:
        raise ValueError(f"{name}: empty output")
    if is_scale:
        close = (actual == golden) & (actual != 0xFF) & (golden != 0xFF)
    else:
        close = (
            np.isfinite(actual)
            & np.isfinite(golden)
            & np.isclose(actual, golden, rtol=rtol, atol=atol)
        )
    error_count = int(np.count_nonzero(~close))
    error_ratio = error_count / n
    max_abs_error = float(
        np.max(np.abs(golden.astype(np.float64) - actual.astype(np.float64)))
    )

    print(f"[verify] {name}: dtype={dtype}, elements={n}")
    print(f"  max abs diff: {max_abs_error:.6e}")
    print(f"  error count (atol>{atol}): {error_count}/{n}, ratio={error_ratio:.6f}")

    _status = "pass" if np.all(close) else "fail"
    _COLLECTED_METRICS.append(
        {
            "name": name,
            "max_abs_diff": float(max_abs_error),
            "error_ratio": float(error_ratio),
            "ratio_tol": float(rtol),
            "status": _status,
        }
    )

    if not np.all(close):
        index = int(np.flatnonzero(~close)[0])
        raise ValueError(
            f"{name} mismatch at {index}: expected {golden[index]}, "
            f"got {actual[index]}, abs_error={abs(float(golden[index]) - float(actual[index]))}"
        )
    print(f"[PASS] {name}")


def _write_metrics_json(failed=False):
    overall = (
        "pass"
        if not failed and all(o["status"] == "pass" for o in _COLLECTED_METRICS)
        else "fail"
    )
    write_metrics_json(_COLLECTED_METRICS, overall, "./output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify NPU output (Y + Y_scale) against golden data."
    )
    parser.add_argument("golden_y", help="Path to golden Y binary")
    parser.add_argument("actual_y", help="Path to NPU Y binary")
    parser.add_argument("golden_y_scale", help="Path to golden Y_scale binary")
    parser.add_argument("actual_y_scale", help="Path to NPU Y_scale binary")
    parser.add_argument(
        "--dtype",
        required=True,
        choices=("fp8_e4m3", "fp8_e5m2", "fp4_e2m1"),
        help="Y output dtype (follows A dtype)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        compare("y", args.golden_y, args.actual_y, args.dtype, is_scale=False)
        compare(
            "y_scale",
            args.golden_y_scale,
            args.actual_y_scale,
            args.dtype,
            is_scale=True,
        )
    except (OSError, ValueError) as error:
        print(error)
        _write_metrics_json(failed=True)
        return 1
    _write_metrics_json()
    print("[PASS] NPU results are consistent with golden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
