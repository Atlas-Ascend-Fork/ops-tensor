#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare grouped MX A8W4 FP16/BF16 output with the generated golden."""

import argparse

import ml_dtypes
import numpy as np


OUTPUT_DTYPES = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
}


def main():
    parser = argparse.ArgumentParser(description="Verify grouped MX A8W4 output")
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(OUTPUT_DTYPES), required=True)
    parser.add_argument("--point-tol", type=float, default=1e-3)
    parser.add_argument("--max-error-ratio", type=float, default=1e-3)
    args = parser.parse_args()

    expected_size = args.m * args.n
    output_dtype = OUTPUT_DTYPES[args.dtype]
    golden_output = np.fromfile(args.golden, dtype=output_dtype)
    actual_output = np.fromfile(args.actual, dtype=output_dtype)
    if golden_output.size != expected_size or actual_output.size != expected_size:
        raise ValueError(
            f"output size mismatch: expected={expected_size}, "
            f"golden={golden_output.size}, actual={actual_output.size}"
        )

    golden = golden_output.astype(np.float32)
    actual = actual_output.astype(np.float32)
    abs_error = np.abs(actual - golden)
    finite = np.isfinite(actual) & np.isfinite(golden) & np.isfinite(abs_error)
    error_mask = (abs_error > args.point_tol) | ~finite
    mismatch = np.flatnonzero(error_mask)
    max_error = float(np.max(abs_error)) if np.all(finite) else float("inf")
    error_ratio = mismatch.size / expected_size
    print(
        f"[verify] dtype={args.dtype}, elements={expected_size}, max_abs_error={max_error}, "
        f"errors={mismatch.size}, error_ratio={error_ratio:.6f}"
    )
    if error_ratio > args.max_error_ratio:
        index = int(mismatch[0])
        row, column = divmod(index, args.n)
        raise ValueError(
            f"mismatch at [{row}, {column}]: expected={golden[index]}, actual={actual[index]}, "
            f"error_ratio={error_ratio:.6f} exceeds {args.max_error_ratio:g}"
        )
    print("[PASS] grouped MX A8W4 output meets the repository accuracy baseline")


if __name__ == "__main__":
    main()
