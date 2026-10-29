#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compare BF16/FP16 outputs numerically or compare Int32 exactly."""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np


POINT_ERROR_TOL = 1e-1
RATIO_POINT_ERROR_TOL = 1e-3
ERROR_RATIO_TOL = 1e-3


def _write_metrics_json(status, max_abs_diff, error_ratio, ratio_tol, elements):
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(max_abs_diff),
                "error_ratio": float(error_ratio),
                "ratio_tol": float(ratio_tol),
                "status": status,
            }
        ],
        status,
        "./output",
    )


def _bfloat16_to_float32(values):
    return np.left_shift(values.astype(np.uint32), np.uint32(16)).view(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()

    dtype = args.dtype.lower()
    is_int32 = dtype in ("int32", "int32_t")
    is_float16 = dtype in ("float16", "float16_t", "half")
    if (
        not is_int32
        and not is_float16
        and dtype
        not in (
            "bfloat16",
            "bfloat16_t",
            "bf16",
        )
    ):
        print(f"unsupported output dtype: {args.dtype}")
        return 1

    file_dtype = np.int32 if is_int32 else (np.float16 if is_float16 else np.uint16)
    golden = np.fromfile(args.golden, dtype=file_dtype)
    actual = np.fromfile(args.actual, dtype=file_dtype)
    if golden.shape != actual.shape:
        print(f"shape mismatch: golden={golden.shape}, actual={actual.shape}")
        return 1

    expected_size = args.batch * args.m * args.n
    if golden.size != expected_size:
        print(f"element count mismatch: expected={expected_size}, actual={golden.size}")
        return 1

    if is_int32:
        mismatch_count = int(np.count_nonzero(golden != actual))
        max_abs_diff = (
            int(np.max(np.abs(actual.astype(np.int64) - golden.astype(np.int64))))
            if expected_size
            else 0
        )
        print(f"max abs diff: {max_abs_diff}")
        print(f"mismatch count: {mismatch_count}/{expected_size}")
        if mismatch_count != 0:
            _write_metrics_json(
                "fail",
                float(max_abs_diff),
                float(mismatch_count) / expected_size if expected_size else 0.0,
                0.0,
                expected_size,
            )
            return 1
        print(f"PASS: verified {expected_size} Int32 elements")
        _write_metrics_json("pass", float(max_abs_diff), 0.0, 0.0, expected_size)
        return 0

    shape = (args.batch, args.m, args.n)
    if is_float16:
        golden_values = golden.reshape(shape).astype(np.float32)
        actual_values = actual.reshape(shape).astype(np.float32)
    else:
        golden_values = _bfloat16_to_float32(golden).reshape(shape)
        actual_values = _bfloat16_to_float32(actual).reshape(shape)
    abs_diff = np.abs(actual_values - golden_values)
    finite_mask = (
        np.isfinite(golden_values) & np.isfinite(actual_values) & np.isfinite(abs_diff)
    )
    abs_golden = np.abs(golden_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(
            abs_golden > 0,
            abs_diff / abs_golden,
            np.where(abs_diff == 0, np.zeros_like(abs_diff), np.inf),
        )
    point_error_count = int(
        np.count_nonzero((rel_diff > POINT_ERROR_TOL) | ~finite_mask)
    )
    ratio_error_count = int(
        np.count_nonzero((abs_diff > RATIO_POINT_ERROR_TOL) | ~finite_mask)
    )
    error_ratio = ratio_error_count / expected_size if expected_size else 0.0

    max_abs_diff = float(abs_diff.max()) if expected_size else 0.0
    print(f"max abs diff: {max_abs_diff}")
    print(f"point error count(>{POINT_ERROR_TOL}): {point_error_count}/{expected_size}")
    print(
        f"ratio error count(>{RATIO_POINT_ERROR_TOL}): {ratio_error_count}/{expected_size}, "
        f"error ratio: {error_ratio:.6f}"
    )
    if point_error_count != 0 or error_ratio > ERROR_RATIO_TOL:
        _write_metrics_json(
            "fail",
            max_abs_diff,
            float(error_ratio),
            float(ERROR_RATIO_TOL),
            expected_size,
        )
        return 1
    output_label = "FP16" if is_float16 else "BF16"
    print(f"PASS: verified {expected_size} {output_label} elements")
    _write_metrics_json(
        "pass",
        max_abs_diff,
        float(error_ratio),
        float(ERROR_RATIO_TOL),
        expected_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
