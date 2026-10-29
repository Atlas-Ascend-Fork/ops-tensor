#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify MX-FP4 values/scales, row padding and unchanged uncovered output rows."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import write_metrics_json


GROUP_SIZE = 32
OUTPUT_SENTINEL = 0xA5
SCALE_EXP_TOL = 1
DEQUANT_TOL_FACTOR = 1.1
# FP4 E2M1: bit 3 is the sign; bits 0-2 encode 0, 0.5, 1, 1.5, 2, 3, 4, 6.
# Decode directly so verification also works without ml_dtypes.float4_e2m1fn.
FP4_LUT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def read_exact(path, dtype, elements):
    expected_bytes = elements * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    data = np.fromfile(path, dtype=dtype)
    if data.size != elements:
        raise ValueError(f"{path}: expected {elements} elements, got {data.size}")
    return data


def verify(golden_dir, actual_dir, m, n):
    golden_dir, actual_dir = Path(golden_dir), Path(actual_dir)
    metadata = json.loads((golden_dir / "case.json").read_text(encoding="utf-8"))
    if (metadata["m"], metadata["n"]) != (m, n):
        raise ValueError("CLI dimensions do not match case.json")
    active = metadata["effective_m"]
    if m <= 0 or n <= 0 or n % GROUP_SIZE or not 0 <= active <= m:
        raise ValueError("Invalid dimensions or effective_m in case.json")
    scale_columns = ((n + 63) // 64) * 2
    golden = read_exact(golden_dir / "golden_ref.bin", np.float32, m * n).reshape(m, n)
    golden_scale = read_exact(
        golden_dir / "golden_scale.bin", np.uint8, m * scale_columns
    ).reshape(m, -1)
    packed = read_exact(actual_dir / "npu_y.bin", np.uint8, m * n // 2).reshape(
        m, n // 2
    )
    scales = read_exact(
        actual_dir / "npu_scale.bin", np.uint8, m * scale_columns
    ).reshape(m, -1)

    y_errors, scale_errors, max_y_diff, max_scale_diff = 0, 0, 0.0, 0.0
    if active:
        actual_scale = scales[:active, : n // GROUP_SIZE]
        expected_scale = golden_scale[:active, : n // GROUP_SIZE]
        scale_diff = np.abs(
            actual_scale.astype(np.int32) - expected_scale.astype(np.int32)
        )
        # E8M0=255 denotes NaN and is never valid for these finite random inputs.
        scale_errors = int(
            np.count_nonzero((scale_diff > SCALE_EXP_TOL) | (actual_scale == 255))
        )
        max_scale_diff = float(np.max(scale_diff))
        codes = np.empty((active, n), dtype=np.uint8)
        codes[:, 0::2] = packed[:active] & 0x0F
        codes[:, 1::2] = packed[:active] >> 4
        # FP64 also keeps malformed large exponents finite for diagnostic metrics.
        steps = np.repeat(
            np.exp2(actual_scale.astype(np.float64) - 127.0), GROUP_SIZE, axis=1
        )
        dequant = FP4_LUT[codes] * steps
        error = np.abs(dequant - golden[:active])
        # FP4 E2M1's largest half-ULP is one shared scale. The margin also covers
        # BF16 rounding of MMAD and clipping, as in the FlatQuant example.
        y_errors = int(
            np.count_nonzero(
                ~np.isfinite(dequant) | (error > DEQUANT_TOL_FACTOR * steps + 1.0e-9)
            )
        )
        max_y_diff = float(np.max(error / steps))

    padding = scales[:active, n // GROUP_SIZE :]
    padding_errors = int(np.count_nonzero(padding))
    tail_errors = int(np.count_nonzero(packed[active:] != OUTPUT_SENTINEL))
    tail_errors += int(np.count_nonzero(scales[active:] != OUTPUT_SENTINEL))
    results = [
        ("y", y_errors, active * n, max_y_diff),
        ("scale", scale_errors, active * (n // GROUP_SIZE), max_scale_diff),
        ("scale_padding", padding_errors, padding.size, float(padding_errors)),
        (
            "untouched_rows",
            tail_errors,
            packed[active:].size + scales[active:].size,
            float(tail_errors),
        ),
    ]
    metrics = []
    for name, errors, total, maximum in results:
        ratio = errors / total if total else 0.0
        print(
            f"[verify] {name}: max abs diff: {maximum:.6e}, error ratio: {ratio:.6f} ({errors}/{total})"
        )
        metrics.append(
            {
                "name": name,
                "max_abs_diff": maximum,
                "error_ratio": ratio,
                "status": "pass" if errors == 0 else "fail",
            }
        )
    passed = all(metric["status"] == "pass" for metric in metrics)
    write_metrics_json(metrics, "pass" if passed else "fail", str(actual_dir))
    print(
        "[PASS] RotateQuant output matches CPU reference."
        if passed
        else "[FAIL] RotateQuant output differs."
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify RotateQuant MX-FP4 outputs.")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--actual-dir", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    return 0 if verify(args.golden_dir, args.actual_dir, args.m, args.n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
