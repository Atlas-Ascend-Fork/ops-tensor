#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate native QGMM MXA8W4 inputs and an FP32 reference output."""

import argparse
from pathlib import Path

import numpy as np

from gen_data import OUTPUT_DTYPE_MAP, TYPE_INFO, _format_weight, _pack_fp4
from gen_wqgmm_mx_a8w4_data import (
    MX_GROUP_SIZE,
    MX_SCALE_EXPONENT_BIAS,
    WEIGHT_TYPES,
    align_up,
    decode_scale,
    parse_group_list,
)

FP4_C0 = 64
SCALE_K_PAIR = 2 * MX_GROUP_SIZE


def generate(args):
    if not (
        1 <= args.groups <= 1024
        and 1 <= args.m <= 1048576
        and 1 <= args.n <= 4096
        and 2 <= args.k <= 4096
        and args.k % 2 == 0
    ):
        raise ValueError("invalid MXA8W4 example dimensions")
    group_list, lengths = parse_group_list(
        args.group_list, args.groups, args.m, args.list_type
    )

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_k = align_up(args.k, SCALE_K_PAIR) // MX_GROUP_SIZE
    a_values, a_codes, _ = TYPE_INFO["mxfp8_e4m3"]
    a_indices = rng.integers(a_codes.size, size=(args.m, args.k), dtype=np.uint8)
    scale_a = rng.integers(
        MX_SCALE_EXPONENT_BIAS - 1,
        MX_SCALE_EXPONENT_BIAS + 2,
        size=(args.m, scale_k),
        dtype=np.uint8,
    )
    scale_b = rng.integers(
        MX_SCALE_EXPONENT_BIAS - 1,
        MX_SCALE_EXPONENT_BIAS + 2,
        size=(args.groups, args.n, scale_k),
        dtype=np.uint8,
    )

    bias = None
    if args.bias_dtype != "none":
        bias = (rng.integers(-4, 5, size=(args.groups, args.n)) * 0.125).astype(
            OUTPUT_DTYPE_MAP[args.bias_dtype]
        )
        bias.tofile(output_dir / "bias.bin")

    weight = WEIGHT_TYPES["float4_e2m1"]
    golden = np.empty((args.m, args.n), dtype=np.float32)
    row = 0
    with (output_dir / "input_b.bin").open("wb") as weight_file:
        for group, length in enumerate(lengths):
            codes = rng.integers(
                weight["codes"].size, size=(args.k, args.n), dtype=np.uint8
            )
            zn = _format_weight(codes, "zn", args.k, args.n, FP4_C0)
            _pack_fp4(zn).tofile(weight_file)
            if length == 0:
                continue

            rows = slice(row, row + length)
            a_scale = np.repeat(decode_scale(scale_a[rows]), MX_GROUP_SIZE, axis=1)[
                :, : args.k
            ]
            b_scale = np.repeat(decode_scale(scale_b[group].T), MX_GROUP_SIZE, axis=0)[
                : args.k
            ]
            golden[rows] = (a_values[a_indices[rows]] * a_scale) @ (
                weight["values"][codes] * b_scale
            )
            if bias is not None:
                golden[rows] += bias[group].astype(np.float32)
            row += length

    a_codes[a_indices].tofile(output_dir / "input_a.bin")
    scale_a.tofile(output_dir / "scale_a.bin")
    scale_b.tofile(output_dir / "scale_b.bin")
    group_list.tofile(output_dir / "group_list.bin")
    golden.tofile(output_dir / "golden_c.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dim in ("groups", "m", "n", "k"):
        parser.add_argument(f"--{dim}", type=int, required=True)
    parser.add_argument(
        "--bias-dtype", choices=("none", *OUTPUT_DTYPE_MAP), required=True
    )
    parser.add_argument("--list-type", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
