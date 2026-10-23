#!/usr/bin/python3
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

import argparse
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common")
)
from metrics import write_metrics_json


def read_output(path, dtype):
    if dtype == "float16":
        return np.fromfile(path, dtype=np.float16).astype(np.float32)
    storage = np.fromfile(path, dtype=np.uint16)
    return (storage.astype(np.uint32) << 16).view(np.float32)


def verify_result(m, n, dtype):
    output = read_output("./output/npu_out.bin", dtype)
    golden = read_output("./output/cpu_output.bin", dtype)
    if m <= 0 or n <= 0 or output.size != m * n or golden.size != m * n:
        raise ValueError("Output length must equal M * N")
    tol = 1e-3 if dtype == "float16" else 8e-3
    finite = np.isfinite(output) & np.isfinite(golden)
    abs_diff = np.abs(output - golden)
    matches = finite & (abs_diff <= tol + tol * np.abs(golden))
    error_ratio = float(np.count_nonzero(~matches)) / output.size
    success = bool(np.all(matches))
    max_abs_diff = float(abs_diff.max()) if np.all(finite) else None
    status = "pass" if success else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": max_abs_diff,
                "error_ratio": error_ratio,
                "ratio_tol": 0.0,
                "atol": tol,
                "rtol": tol,
                "status": status,
            }
        ],
        status,
        "./output",
    )
    print(f"max abs diff: {max_abs_diff}")
    print(f"error ratio: {error_ratio:.6f}, atol={tol}, rtol={tol}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Verify FusedMatMul sigmoid output.")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("dtype", choices=["float16", "bfloat16"])
    args = parser.parse_args()
    if not verify_result(args.m, args.n, args.dtype):
        print("[FAIL] NPU results differ from CPU.")
        return 1
    print("[PASS] NPU results are consistent with CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
