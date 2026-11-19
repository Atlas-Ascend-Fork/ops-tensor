#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate transposed WeightNZ A8W4 inputs, with optional per-expert smoothing."""

import argparse
from pathlib import Path

import ml_dtypes
import numpy as np
from gen_gmmsq_data import compute_swiglu, quantize_golden


def create_weight(experts, k, n):
    columns = np.arange(n)
    values = np.where(columns % 2 == 0, 1.0, 0.5).astype(np.float32)
    values[n // 2 :] *= np.where(columns[n // 2 :] % 3 == 0, -1.0, 1.0)
    weight = np.broadcast_to(values, (experts, k, n)).copy()
    codes = np.where(np.abs(weight) == 1.0, 2, 1).astype(np.uint8)
    codes |= np.where(weight < 0, 8, 0).astype(np.uint8)
    # ZN storage [E, K/64, N/16, 16, 64], two FP4 values per byte.
    nz = codes.transpose(0, 2, 1).reshape(experts, n // 16, 16, k // 64, 64)
    nz = nz.transpose(0, 3, 1, 2, 4).copy().reshape(-1, 2)
    return weight, nz[:, 0] | (nz[:, 1] << 4)


def generate(args):
    e, m, n, k = args.experts, args.m, args.n, args.k
    if not (1 <= e <= 8 and e <= m <= 4096 and 256 <= n <= 4096 and 256 <= k <= 4096):
        raise ValueError("invalid example shape")
    if n % 128 or k % 128:
        raise ValueError("K/N must be 128-aligned")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lengths = np.ones(e, dtype=np.int64)
    lengths[-1] = m - e + 1
    groups = lengths if args.group_type else np.cumsum(lengths)
    x = np.zeros((m, k), dtype=np.float32)
    x[:, 0] = np.resize(np.array([1.0, 4.0, -1.0], dtype=np.float32), m)
    x[:, -1] = x[:, 0]
    weight, packed = create_weight(e, k, n)
    scales_x = np.ones((m, k // 32), dtype=np.float32)
    scales_w = np.ones((e, k // 32, n), dtype=np.float32)
    y = compute_swiglu(lengths, x, scales_x, weight, scales_w, 7.0, 1.702, 1.0)
    smooth = np.exp2(
        (np.arange(e)[:, None] + np.arange(n // 2)[None, :]) % 3 - 1
    ).astype(np.float32)
    if args.smooth:
        row_smooth = np.repeat(smooth, lengths, axis=0)
        y = (y.astype(np.float32) * row_smooth).astype(ml_dtypes.bfloat16)
    quantized, scale = quantize_golden(y, args.algorithm)
    arrays = {
        "x": x.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
        "weight": packed,
        "x_scale": np.full((m, k // 64, 2), 127, dtype=np.uint8),
        "weight_scale": np.full((e, n, k // 64, 2), 127, dtype=np.uint8),
        "group_list": groups,
        "smooth": smooth,
        "golden_y": quantized,
        "golden_y_scale": scale,
    }
    for name, array in arrays.items():
        array.tofile(output / (name + ".bin"))


def main():
    parser = argparse.ArgumentParser()
    for name in ("experts", "m", "n", "k"):
        parser.add_argument("--" + name, type=int, required=True)
    for name in ("algorithm", "smooth", "group-type"):
        parser.add_argument("--" + name, type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
