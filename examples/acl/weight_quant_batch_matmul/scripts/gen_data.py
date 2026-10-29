#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate deterministic A16W8 inputs and golden with staged antiquant rounding."""

import argparse
from pathlib import Path

import ml_dtypes
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for axis in ("m", "k", "n"):
        parser.add_argument(f"--{axis}", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    for flag in ("trans-b", "per-tensor", "offset", "bias"):
        parser.add_argument(f"--{flag}", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.m, args.k, args.n) <= 0:
        parser.error("M, K and N must be positive")
    dtype = np.float16 if args.dtype == "fp16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(20260918)
    a = (rng.integers(-8, 9, (args.m, args.k)).astype(np.float32) / 8).astype(dtype)
    b = rng.integers(-8, 9, (args.k, args.n), dtype=np.int8)
    count = 1 if args.per_tensor else args.n
    scale = (rng.integers(1, 5, count).astype(np.float32) / 16).astype(dtype)
    offset = (rng.integers(-2, 3, count).astype(np.float32) / 4).astype(dtype)
    bias = (rng.integers(-4, 5, args.n).astype(np.float32) / 8).astype(dtype)
    # Cast, add offset and multiply scale round separately in the VF output dtype.
    converted = b.astype(dtype)
    if args.offset:
        converted = (converted.astype(np.float32) + offset.astype(np.float32)).astype(
            dtype
        )
    converted = (converted.astype(np.float32) * scale.astype(np.float32)).astype(dtype)
    golden = a.astype(np.float32) @ converted.astype(np.float32)
    if args.bias:
        golden += bias.astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "a": a,
        "b": b.T.copy() if args.trans_b else b,
        "scale": scale,
        "offset": offset,
        "bias": bias,
        "golden_c": golden.astype(dtype),
    }
    for name, values in arrays.items():
        values.tofile(args.output_dir / f"{name}.bin")


if __name__ == "__main__":
    main()
