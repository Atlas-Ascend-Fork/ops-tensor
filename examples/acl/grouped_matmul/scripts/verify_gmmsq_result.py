#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Numerical verification for QGMM SwiGLU MX E4M3FN Y and E8M0 YScale."""

import argparse

import ml_dtypes
import numpy as np


Y_RELATIVE_TOLERANCE = 1e-3
Y_MAX_MISMATCH_RATIO = 1e-3


def compare_values(expected, actual, name, is_scale=False):
    """Check FP8 Y with dual 0.1% limits; require discrete scales to match."""
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(
            f"{name} size mismatch: expected={expected.shape}, actual={actual.shape}"
        )
    if expected.size == 0:
        raise ValueError(f"{name}: empty output")
    finite = np.isfinite(expected) & np.isfinite(actual)
    if not np.all(finite):
        first = int(np.flatnonzero(~finite)[0])
        raise ValueError(
            f"{name} non-finite value at {first}: "
            f"expected={expected.flat[first]}, actual={actual.flat[first]}"
        )

    # Inputs are decoded to FP32; FP64 arithmetic avoids overflow or underflow
    # in the diagnostic calculation without changing those decoded values.
    expected_f64 = expected.astype(np.float64)
    actual_f64 = actual.astype(np.float64)
    difference = np.abs(actual_f64 - expected_f64)
    denominator = np.maximum(np.abs(actual_f64), np.abs(expected_f64))
    relative_error = np.divide(
        difference,
        denominator,
        out=np.zeros_like(difference),
        where=denominator != 0,
    )
    # E8M0 represents discrete powers of two, including 2**-127 at code zero.
    # Never allow a wrong scale to cancel an opposite error in Y.
    tolerance = 0.0 if is_scale else Y_RELATIVE_TOLERANCE
    mismatch = relative_error > tolerance
    mismatch_count = int(np.count_nonzero(mismatch))
    max_mismatch_ratio = 0.0 if is_scale else Y_MAX_MISMATCH_RATIO
    mismatch_ratio = mismatch_count / expected.size
    if mismatch_ratio > max_mismatch_ratio:
        first = int(np.flatnonzero(mismatch)[0])
        raise ValueError(
            f"{name} mismatch at {first}: expected={expected.flat[first]}, "
            f"actual={actual.flat[first]}, relative_error={relative_error.flat[first]}, "
            f"tolerance={tolerance}, errors={mismatch_count}/{expected.size}, "
            f"max_mismatch_ratio={max_mismatch_ratio}"
        )
    print(
        f"[PASS] {name}: {actual.size} decoded values, "
        f"max_relative_error={np.max(relative_error)}, tolerance={tolerance}, "
        f"errors={mismatch_count}/{expected.size}"
    )


def compare(expected_path, actual_path, name, is_scale=False):
    dtype = ml_dtypes.float8_e8m0fnu if is_scale else ml_dtypes.float8_e4m3fn
    expected = np.fromfile(expected_path, dtype=np.uint8).view(dtype)
    actual = np.fromfile(actual_path, dtype=np.uint8).view(dtype)
    compare_values(expected, actual, name, is_scale)


def main():
    parser = argparse.ArgumentParser(description="Verify QGMM SwiGLU MX outputs")
    parser.add_argument("golden_y")
    parser.add_argument("actual_y")
    parser.add_argument("golden_scale")
    parser.add_argument("actual_scale")
    args = parser.parse_args()
    compare(args.golden_y, args.actual_y, "Y")
    compare(args.golden_scale, args.actual_scale, "YScale", is_scale=True)


if __name__ == "__main__":
    main()
