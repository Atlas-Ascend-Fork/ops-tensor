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

"""Verify all BlockAttnResPrepare outputs against generated golden files."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json


OUTPUTS = ("max", "output", "sum")
ABS_TOLERANCE = 2.0e-3
REL_TOLERANCE = 2.0e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--actual-dir", required=True)
    args = parser.parse_args()
    metrics = []
    passed = True
    for name in OUTPUTS:
        golden = np.fromfile(
            os.path.join(args.golden_dir, f"golden_{name}.bin"), dtype=np.float32
        )
        actual = np.fromfile(
            os.path.join(args.actual_dir, f"npu_{name}.bin"), dtype=np.float32
        )
        if golden.shape != actual.shape:
            raise ValueError(f"{name}: shape mismatch {golden.shape} != {actual.shape}")
        difference = np.abs(golden - actual)
        finite = np.isfinite(golden) & np.isfinite(actual)
        close = (
            np.isclose(golden, actual, rtol=REL_TOLERANCE, atol=ABS_TOLERANCE) & finite
        )
        error_ratio = (
            float(np.count_nonzero(~close) / golden.size) if golden.size else 1.0
        )
        max_difference = (
            float(np.max(difference))
            if difference.size and np.all(np.isfinite(difference))
            else float("inf")
        )
        status = "pass" if error_ratio == 0.0 else "fail"
        metrics.append(
            {
                "name": name,
                "max_abs_diff": max_difference,
                "error_ratio": error_ratio,
                "ratio_tol": 0.0,
                "status": status,
            }
        )
        passed = passed and status == "pass"
        print(
            f"{name}: max_abs_diff={max_difference:.6e}, error_ratio={error_ratio:.6f}"
        )
    write_metrics_json(metrics, "pass" if passed else "fail", args.actual_dir)
    print(
        "[PASS] NPU results are consistent with CPU."
        if passed
        else "[FAIL] NPU results differ from CPU."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
