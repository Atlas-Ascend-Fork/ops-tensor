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

"""Compare the per-group A8W4 INT8 output against the golden with a +/-1 tolerance.

The kernel accumulates in FP32 with its own summation order, so the RINT boundary can
flip by one ULP near half-integers; a strict bitwise compare is not meaningful here.
"""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json

import numpy as np


POINT_TOL = 1
ERROR_RATIO_TOL = 2e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()

    expected_size = args.m * args.n
    golden = np.fromfile(args.golden, dtype=np.int8)
    actual = np.fromfile(args.actual, dtype=np.int8)
    if golden.size != expected_size or actual.size != expected_size:
        print(
            f"element count mismatch: golden={golden.size}, actual={actual.size}, "
            f"expected={expected_size}"
        )
        return 1

    abs_diff = np.abs(actual.astype(np.int32) - golden.astype(np.int32))
    max_abs_diff = int(abs_diff.max()) if expected_size else 0
    mismatch_count = int(np.count_nonzero(abs_diff > POINT_TOL))
    error_ratio = mismatch_count / expected_size if expected_size else 0.0

    print(f"max abs diff: {max_abs_diff}")
    print(f"mismatch count(>{POINT_TOL}): {mismatch_count}/{expected_size}")

    status = "pass" if error_ratio <= ERROR_RATIO_TOL else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(max_abs_diff),
                "error_ratio": float(error_ratio),
                "ratio_tol": float(ERROR_RATIO_TOL),
                "status": status,
            }
        ],
        status,
        "./output",
    )
    if status == "fail":
        return 1
    print(f"PASS: verified {expected_size} INT8 elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
