#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare every output element; reject empty, truncated or non-finite outputs."""

import argparse
from pathlib import Path
import sys

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import write_metrics_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    args = parser.parse_args()
    dtype = np.float16 if args.dtype == "fp16" else ml_dtypes.bfloat16
    if (
        args.golden.stat().st_size == 0
        or args.golden.stat().st_size != args.actual.stat().st_size
    ):
        raise ValueError("Empty output or mismatched binary sizes")
    if args.actual.stat().st_size % np.dtype(dtype).itemsize:
        raise ValueError("Output contains an incomplete element")
    golden = np.fromfile(args.golden, dtype=dtype).astype(np.float32)
    actual = np.fromfile(args.actual, dtype=dtype).astype(np.float32)
    rtol, atol = (0.001, 0.001) if args.dtype == "fp16" else (0.008, 0.008)
    finite = np.isfinite(golden) & np.isfinite(actual)
    diff = np.abs(actual - golden)
    good = finite & (diff <= atol + rtol * np.abs(golden))
    passed = bool(np.all(good))
    maximum = float(diff.max()) if np.all(finite) else float("inf")
    status = "pass" if passed else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": maximum,
                "error_ratio": float(np.mean(~good)),
                "status": status,
            }
        ],
        status,
        str(args.actual.parent),
    )
    print(
        f"[{status.upper()}] elements={golden.size}, max_abs_diff={maximum}, mismatches={np.count_nonzero(~good)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
