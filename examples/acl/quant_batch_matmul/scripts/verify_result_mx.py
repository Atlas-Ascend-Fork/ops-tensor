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

"""Verify the NPU output against the generated golden result for quant_batch_matmul_mx."""

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json

import ml_dtypes
import numpy as np


FULL_TENSOR_PRINT_MAX_ELEMENTS = 128

DTYPE_CONFIG = {
    "float16": {"np_dtype": np.float16, "ratio_tol": 1e-3},
    "bfloat16": {"np_dtype": ml_dtypes.bfloat16, "ratio_tol": 1e-3},
    "float32": {"np_dtype": np.float32, "ratio_tol": 1e-4},
}


@dataclass(frozen=True)
class VerifySummaryCfg:
    num_elements: int
    ratio_tol: float
    dtype: str


def _compute_ratio_error_mask(abs_diff, non_finite_mask, ratio_tol):
    return (abs_diff > ratio_tol) | non_finite_mask


def _ratio_label(ratio_tol):
    return f"abs_diff > {ratio_tol:g}"


def _max_abs_diff(abs_diff):
    if not np.all(np.isfinite(abs_diff)):
        return float("inf")
    return float(np.max(abs_diff))


def _print_summary(golden, actual, abs_diff, error_mask, config):
    if config.num_elements <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print("\ncpu golden:\n", golden)
        print("npu output:\n", actual)

    error_count = int(np.count_nonzero(error_mask))
    print(f"\n[verify] dtype={config.dtype}, elements={config.num_elements}")
    print(f"  max abs diff: {_max_abs_diff(abs_diff):.6e}")
    print(
        f"  count({_ratio_label(config.ratio_tol)}): {error_count} / {config.num_elements}"
    )


def verify_result(golden_path, actual_path, dtype):
    config = DTYPE_CONFIG[dtype]
    ratio_tol = config["ratio_tol"]
    golden = np.fromfile(golden_path, dtype=config["np_dtype"])
    actual = np.fromfile(actual_path, dtype=config["np_dtype"])

    if actual.size != golden.size:
        raise ValueError(
            f"NPU output size ({actual.size}) != CPU output size ({golden.size})"
        )
    if golden.size == 0:
        raise ValueError("output tensor is empty")

    golden_f32 = golden.astype(np.float32)
    actual_f32 = actual.astype(np.float32)
    abs_diff = np.abs(golden_f32 - actual_f32)
    non_finite_mask = ~(
        np.isfinite(golden_f32) & np.isfinite(actual_f32) & np.isfinite(abs_diff)
    )
    ratio_error_mask = _compute_ratio_error_mask(abs_diff, non_finite_mask, ratio_tol)

    summary_cfg = VerifySummaryCfg(
        num_elements=golden.size, ratio_tol=ratio_tol, dtype=dtype
    )
    _print_summary(golden, actual, abs_diff, ratio_error_mask, summary_cfg)

    error_count = int(np.count_nonzero(ratio_error_mask))
    error_ratio = error_count / golden.size
    print(
        f"ratio error count({_ratio_label(ratio_tol)}): {error_count}/{golden.size}, "
        f"error ratio: {error_ratio:.6f}"
    )

    if error_ratio > ratio_tol:
        first_error = int(np.flatnonzero(ratio_error_mask)[0])
        print(
            f"first error at {first_error}: expected={golden[first_error]}, "
            f"actual={actual[first_error]}, abs_diff={abs_diff[first_error]}"
        )

    _status = "pass" if error_ratio <= ratio_tol else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(_max_abs_diff(abs_diff)),
                "error_ratio": float(error_ratio),
                "ratio_tol": float(ratio_tol),
                "status": _status,
            }
        ],
        _status,
        "./output",
    )

    return error_ratio <= ratio_tol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify NPU output against CPU golden data."
    )
    parser.add_argument("golden", help="Path to the CPU golden binary")
    parser.add_argument("actual", help="Path to the NPU output binary")
    parser.add_argument(
        "--dtype",
        required=True,
        choices=("float16", "bfloat16", "float32"),
        help="Output dtype",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if not verify_result(args.golden, args.actual, args.dtype):
            ratio_tol = DTYPE_CONFIG[args.dtype]["ratio_tol"]
            raise ValueError(
                f"[ERROR] NPU results differ from CPU. The ratio of points with "
                f"{_ratio_label(ratio_tol)} must be <= {ratio_tol:g}."
            )
        print("[PASS] NPU results are consistent with CPU.")
        return 0
    except (OSError, ValueError) as error:
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
