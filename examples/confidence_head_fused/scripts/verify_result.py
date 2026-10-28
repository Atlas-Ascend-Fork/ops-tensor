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
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import write_metrics_json


def verify(batch, n):
    if batch < 0 or n < 0:
        raise ValueError("batch and n must be nonnegative")
    expected_bytes = batch * n * np.dtype(np.float32).itemsize
    for name in ("npu_out.bin", "cpu_output.bin"):
        if (Path("output") / name).stat().st_size != expected_bytes:
            raise ValueError("FP32 output byte length must equal batch*n*4")
    actual = np.fromfile("output/npu_out.bin", dtype=np.float32)
    golden = np.fromfile("output/cpu_output.bin", dtype=np.float32)
    if batch < 0 or n < 0 or actual.size != batch * n or golden.size != batch * n:
        raise ValueError("FP32 output length must equal batch*n")
    finite = np.isfinite(actual) & np.isfinite(golden)
    difference = np.abs(actual - golden)
    # Preserve the existing tolerance for FP32 projection and the default non-precise Div.
    matches = finite & (difference <= 1e-4 + 1e-3 * np.abs(golden))
    success = bool(matches.all())
    status = "pass" if success else "fail"
    max_abs = (
        float(difference.max())
        if difference.size and finite.all()
        else (0.0 if not difference.size else None)
    )
    ratio = float(np.count_nonzero(~matches) / actual.size) if actual.size else 0.0
    write_metrics_json(
        [
            {
                "name": "output",
                "status": status,
                "atol": 1e-4,
                "rtol": 1e-3,
                "ratio_tol": 0.0,
                "max_abs_diff": max_abs,
                "error_ratio": ratio,
            }
        ],
        status,
        "./output",
    )
    print(f"{status}: max_abs_diff={max_abs}, error_ratio={ratio}")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify ConfidenceHeadFused FP32 output."
    )
    parser.add_argument("batch", type=int)
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    sys.exit(0 if verify(args.batch, args.n) else 1)
