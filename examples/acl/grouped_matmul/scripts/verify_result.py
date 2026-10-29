#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Compare QGMM MX NPU output with the generated NumPy golden result."""

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


DTYPE_CONFIG = {
    "float16": (np.float16, 1e-3, 1e-3),
    "bfloat16": (ml_dtypes.bfloat16, 1e-3, 1e-3),
    "float32": (np.float32, 1e-4, 1e-4),
}


def main():
    parser = argparse.ArgumentParser(description="Verify QGMM MX output")
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--dtype", choices=tuple(DTYPE_CONFIG), required=True)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()
    np_dtype, abs_tol, error_ratio_tol = DTYPE_CONFIG[args.dtype]
    golden = np.fromfile(args.golden, dtype=np_dtype)
    actual = np.fromfile(args.actual, dtype=np_dtype)
    if golden.shape != actual.shape:
        raise ValueError(
            f"output size mismatch: actual={actual.size}, golden={golden.size}"
        )
    golden_f32 = golden.astype(np.float32)
    actual_f32 = actual.astype(np.float32)
    abs_diff = np.abs(actual_f32 - golden_f32)
    error_mask = (abs_diff > abs_tol) | ~(
        np.isfinite(actual_f32) & np.isfinite(golden_f32) & np.isfinite(abs_diff)
    )
    if args.m and args.n and actual.size == args.groups * args.m * args.n:
        grouped_actual = actual_f32.reshape(args.groups, args.m, args.n)
        grouped_golden = golden_f32.reshape(args.groups, args.m, args.n)
        for group_index in range(args.groups):
            group_error = float(
                np.max(
                    np.abs(grouped_actual[group_index] - grouped_golden[group_index])
                )
            )
            print(f"[INFO] group {group_index}: max_abs_error={group_error}")
    error_count = int(np.count_nonzero(error_mask))
    error_ratio = error_count / actual.size if actual.size else 0.0
    max_error = float(np.max(abs_diff)) if actual.size else 0.0
    overall_status = "fail" if error_ratio > error_ratio_tol else "pass"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": max_error,
                "error_ratio": error_ratio,
                "ratio_tol": error_ratio_tol,
                "status": overall_status,
            }
        ],
        overall_status,
        "./output",
    )
    if error_ratio > error_ratio_tol:
        mismatch = np.flatnonzero(error_mask)
        index = int(mismatch[0])
        raise ValueError(
            f"mismatch at {index}: expected={golden[index]}, actual={actual[index]}, "
            f"errors={error_count}/{actual.size}, dtype={args.dtype}, abs_tol={abs_tol}, "
            f"error_ratio={error_ratio}, error_ratio_tol={error_ratio_tol}, max_abs_error={max_error}"
        )
    print(
        f"[PASS] {actual.size} {args.dtype} outputs, abs_tol={abs_tol}, "
        f"error_ratio={error_ratio}, error_ratio_tol={error_ratio_tol}, max_abs_error={max_error}"
    )


if __name__ == "__main__":
    main()
