#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Generate a small deterministic GMMAQ MX input set.

The example uses zero MX values and E8M0 scale=1.  This keeps the golden result
byte-exact (GELU(0)=0) while still exercising the grouped Blaze kernel,
activation epilogue and output-scale path for both FP8 and FP4.
"""

import argparse
import os

import numpy as np


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def mx_scale_elements(dimension):
    return (dimension + 63) // 64 * 2


def main():
    parser = argparse.ArgumentParser(description="Generate GMMAQ MX example data")
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=("mxfp8_e4m3", "mxfp4_e2m1"), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if min(args.group_num, args.m, args.n, args.k) <= 0:
        raise ValueError("group_num, M, N and K must be positive")
    if args.k % 16 != 0:
        raise ValueError("K must be a multiple of 16")
    c0 = 64 if args.dtype == "mxfp4_e2m1" else 32
    if args.n % c0 != 0:
        raise ValueError(f"N must be a multiple of {c0} for this fixed NZ example")

    group_list = np.asarray(
        [int(value) for value in args.group_list.split(";")], dtype=np.int64
    )
    if group_list.size != args.group_num or np.any(group_list != args.m):
        raise ValueError("length group-list must contain one M value per group")

    os.makedirs(args.output_dir, exist_ok=True)
    values_per_group = args.m * args.k
    weight_elements = align(args.k, 16) * align(args.n, c0)
    value_bytes = (
        (values_per_group + 1) // 2
        if args.dtype.startswith("mxfp4")
        else values_per_group
    )
    weight_bytes = (
        (weight_elements + 1) // 2
        if args.dtype.startswith("mxfp4")
        else weight_elements
    )
    scale_k = mx_scale_elements(args.k)
    scale_n = mx_scale_elements(args.n)

    # Zero payloads and exponent 127 (E8M0 value 1.0) make the expected output exact.
    np.zeros(args.group_num * value_bytes, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "input_x.bin")
    )
    np.zeros(args.group_num * weight_bytes, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "input_weight.bin")
    )
    np.full(args.group_num * args.m * scale_k, 127, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "x_scale.bin")
    )
    np.full(args.group_num * args.n * scale_k, 127, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "weight_scale.bin")
    )
    group_list.tofile(os.path.join(args.output_dir, "group_list.bin"))

    output_elements = args.group_num * args.m * args.n
    output_bytes = (
        (output_elements + 1) // 2
        if args.dtype.startswith("mxfp4")
        else output_elements
    )
    np.zeros(output_bytes, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "golden_y.bin")
    )
    # A zero activation has exponent zero; the MX epilogue emits the minimum
    # E8M0 exponent for this empty dynamic range.
    np.zeros(args.group_num * args.m * scale_n, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "golden_y_scale.bin")
    )


if __name__ == "__main__":
    main()
