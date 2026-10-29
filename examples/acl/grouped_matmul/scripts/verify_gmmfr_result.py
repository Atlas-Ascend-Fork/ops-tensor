#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""Compare GMMFR NPU output with the generated golden result."""

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
    # The two limits are deliberately the same: element-wise atol/rtol and the
    # allowed mismatch ratio both follow the requested accuracy baseline.
    "float32": (np.float32, 4e-3, 4e-3),
    "bfloat16": (ml_dtypes.bfloat16, 2e-2, 2e-2),
}


def main():
    parser = argparse.ArgumentParser(
        description="Verify GMMFR output against CPU golden"
    )
    parser.add_argument("golden")
    parser.add_argument("actual")
    parser.add_argument("--dtype", choices=tuple(DTYPE_CONFIG), required=True)
    args = parser.parse_args()

    np_dtype, value_tol, ratio_tol = DTYPE_CONFIG[args.dtype]
    golden = np.fromfile(args.golden, dtype=np_dtype).astype(np.float32)
    actual = np.fromfile(args.actual, dtype=np_dtype).astype(np.float32)
    if golden.size != actual.size:
        raise ValueError(
            f"output size mismatch: golden={golden.size}, actual={actual.size}"
        )

    finite = np.isfinite(golden) & np.isfinite(actual)
    close = np.isclose(actual, golden, rtol=value_tol, atol=value_tol, equal_nan=False)
    error_mask = ~(finite & close)
    error_count = int(np.count_nonzero(error_mask))
    error_ratio = error_count / actual.size if actual.size else 0.0
    abs_diff = np.abs(actual - golden)
    max_abs_diff = float(np.max(abs_diff)) if actual.size else 0.0
    status = "pass" if error_ratio <= ratio_tol else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": max_abs_diff,
                "error_ratio": error_ratio,
                "ratio_tol": ratio_tol,
                "status": status,
            }
        ],
        status,
        os.path.dirname(os.path.abspath(args.actual)),
    )
    print(
        f"[verify] dtype={args.dtype}, max_abs_diff={max_abs_diff:.6e}, "
        f"error_ratio={error_ratio:.6e}, value_tol={value_tol:.6e}, ratio_tol={ratio_tol:.6e}"
    )
    if status != "pass":
        index = int(np.flatnonzero(error_mask)[0])
        raise ValueError(
            f"mismatch at {index}: expected={golden[index]}, actual={actual[index]}, "
            f"errors={error_count}/{actual.size}"
        )
    print("[PASS] GMMFR output meets the configured accuracy baseline")


if __name__ == "__main__":
    main()
