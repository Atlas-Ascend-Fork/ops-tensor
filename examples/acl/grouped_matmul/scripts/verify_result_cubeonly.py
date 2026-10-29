#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Compare QGMM Cube NPU output with NumPy golden output."""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json

import numpy as np
import ml_dtypes


def main():
    parser = argparse.ArgumentParser(description="Verify QGMM Cube output")
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()

    dtype = args.dtype.lower()
    if dtype in ("fp16", "float16", "half"):
        np_dtype = np.float16
        error_ratio_tol = 1e-3
        abs_tol = 1e-3
    elif dtype in ("bf16", "bfloat16"):
        np_dtype = ml_dtypes.bfloat16
        error_ratio_tol = abs_tol = 1e-3
    elif dtype in ("fp32", "float", "float32"):
        np_dtype = np.float32
        error_ratio_tol = 1e-4
        abs_tol = 1e-4
    else:
        print(f"unsupported output dtype: {args.dtype}")
        return 1

    golden = np.fromfile(args.golden, dtype=np_dtype)
    actual = np.fromfile(args.actual, dtype=np_dtype)
    if golden.shape != actual.shape:
        raise ValueError(
            f"output size mismatch: actual={actual.size}, golden={golden.size}"
        )
    expected = args.groups * args.m * args.n
    if golden.size != expected:
        raise ValueError(
            f"element count mismatch: expected={expected}, actual={golden.size}"
        )

    actual_f32, golden_f32 = actual.astype(np.float32), golden.astype(np.float32)
    abs_diff = np.abs(actual_f32 - golden_f32)
    error_mask = (abs_diff > abs_tol) | ~(
        np.isfinite(actual_f32) & np.isfinite(golden_f32) & np.isfinite(abs_diff)
    )
    error_count = int(np.count_nonzero(error_mask))
    error_ratio = error_count / actual.size if actual.size else 0.0
    max_error = float(np.max(abs_diff)) if actual.size else 0.0
    status = "fail" if error_ratio > error_ratio_tol else "pass"

    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": max_error,
                "error_ratio": error_ratio,
                "ratio_tol": float(error_ratio_tol),
                "status": status,
            }
        ],
        status,
        "./output",
    )
    if status == "fail":
        index = int(np.flatnonzero(error_mask)[0])
        raise ValueError(
            f"mismatch at {index}: expected={golden[index]}, actual={actual[index]}, "
            f"errors={error_count}/{actual.size}, abs_tol={abs_tol}, "
            f"error_ratio={error_ratio}, error_ratio_tol={error_ratio_tol}"
        )
    print(f"[PASS] {actual.size} {dtype} outputs, max_abs_error={max_error}")


if __name__ == "__main__":
    sys.exit(main())
