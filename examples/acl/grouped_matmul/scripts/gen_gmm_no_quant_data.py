#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate deterministic no-quant grouped matmul inputs and a NumPy golden result.

Covers the aclnnGroupedMatmulV5 (Ascend 950) non-quant scenario matrix:
groupType -1/0/2, groupListType 0/1/2, weight format ND/NZ, single/multi tensor lists.
"""

import argparse
import os

import ml_dtypes
import numpy as np

DTYPE_MAP = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}
BIAS_VALUE = 0.25


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def parse_ints(text):
    return [int(item) for item in text.split(";") if item != ""]


def format_nz(weight, c0):
    """Convert a logical [k, n] row-major matrix into the NZ frame [n/c0, k/16, 16, c0]."""
    k_len, n_len = weight.shape
    padded = np.zeros((align_up(k_len, 16), align_up(n_len, c0)), dtype=weight.dtype)
    padded[:k_len, :n_len] = weight
    return (
        padded.reshape(padded.shape[0] // 16, 16, padded.shape[1] // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def parse_group_list(text, group_num, group_list_type):
    """Return (actual_indices, split_sizes) for the given groupListType."""
    values = parse_ints(text)
    if group_list_type == 0:
        if len(values) != group_num:
            raise ValueError("cumsum groupList length must equal groupNum")
        bounds = [0] + values
        sizes = [bounds[i + 1] - bounds[i] for i in range(group_num)]
        return list(range(group_num)), sizes
    if group_list_type == 1:
        if len(values) != group_num:
            raise ValueError("count groupList length must equal groupNum")
        return list(range(group_num)), values
    if len(values) != 2 * group_num:
        raise ValueError("sparse groupList must hold groupNum [index, size] pairs")
    return values[0::2], values[1::2]


def write_tensor(path, matrix):
    matrix.tofile(path)


def write_weight(path, matrix, weight_format, c0):
    if weight_format == "nz":
        format_nz(matrix, c0).tofile(path)
    else:
        matrix.tofile(path)


class Inputs:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.a_chunks = []  # storage-order chunks concatenated into input_a.bin
        self.b_chunks = []  # storage-order chunks concatenated into input_b.bin
        self.b_files = []  # per-group logical weight matrices (multi-weight case)
        self.bias_matrix = (
            None  # [groupNum, n] for single weight, else per-group vectors
        )
        self.bias_files = []
        self.golden_chunks = []


def generate_m_split(args, np_dtype, rng):
    """groupType=0: M-axis grouping (s-s-s / s-m-s / m-m-s)."""
    actual, sizes = parse_group_list(
        args.group_list, args.group_num, args.group_list_type
    )
    total_m = sum(sizes)
    if total_m != args.m:
        raise ValueError(f"groupList sizes sum to {total_m}, expected m={args.m}")
    if (
        args.group_list_type == 2
        and any(size == 0 for size in sizes[:-1])
        and sizes[-1] != 0
    ):
        raise ValueError("sparse groupList must front-load non-zero groups")

    weights = [
        rng.uniform(-0.08, 0.08, size=(args.k, args.n)).astype(np_dtype)
        for _ in range(args.group_num)
    ]
    biases = [
        np.full(args.n, BIAS_VALUE if args.is_bias else 0.0, np_dtype)
        for _ in range(args.group_num)
    ]
    inputs = Inputs(args.output_dir)

    if args.single_x:
        x = rng.uniform(-0.08, 0.08, size=(total_m, args.k)).astype(np_dtype)
        inputs.a_chunks.append(x)
        x_rows = x.astype(np.float32)
        golden = []
        row_start = 0
        for index, size in enumerate(sizes):
            block = x_rows[row_start : row_start + size] @ weights[
                actual[index]
            ].astype(np.float32)
            if args.is_bias:
                block = block + biases[actual[index]].astype(np.float32)
            golden.append(block.astype(np_dtype))
            row_start += size
        inputs.golden_chunks = golden
    else:
        x_ms = parse_ints(args.x_ms)
        if len(x_ms) != args.group_num or sum(x_ms) != args.m:
            raise ValueError("xMs must describe one m per group and sum to m")
        golden = []
        for index, x_m in enumerate(x_ms):
            x = rng.uniform(-0.08, 0.08, size=(x_m, args.k)).astype(np_dtype)
            inputs.a_chunks.append(x)
            block = x.astype(np.float32) @ weights[index].astype(np.float32)
            if args.is_bias:
                block = block + biases[index].astype(np.float32)
            golden.append(block.astype(np_dtype))
        # m-m-s: the kernel resolves per-group m from each x descriptor (groupType normalized to
        # no-split), so the golden simply concatenates all x_i @ weight_i blocks in group order.
        inputs.golden_chunks = golden

    if args.single_weight:
        inputs.b_chunks = list(weights)
        inputs.bias_matrix = np.stack(biases) if args.is_bias else None
    else:
        inputs.b_files = weights
        inputs.bias_files = biases if args.is_bias else None
    return inputs, args.single_x


def generate_k_split(args, np_dtype, rng):
    """groupType=2: K-axis grouping (s-s-s with 3D y / s-m-m with multi y)."""
    if args.is_bias:
        raise ValueError("K-axis grouping does not support bias")
    _, sizes = parse_group_list(args.group_list, args.group_num, args.group_list_type)
    if sum(sizes) != args.k:
        raise ValueError(f"groupList sizes sum to {sum(sizes)}, expected k={args.k}")

    # x is stored transposed as [K, M]; each group contributes a [k_g, M] segment.
    segments = [
        rng.uniform(-0.08, 0.08, size=(args.m, size)).astype(np_dtype) for size in sizes
    ]
    inputs = Inputs(args.output_dir)
    inputs.a_chunks = [segment.T for segment in segments]

    golden = []
    weight_segments = []
    k_start = 0
    if args.single_weight:
        weight = rng.uniform(-0.08, 0.08, size=(args.k, args.n)).astype(np_dtype)
        for size in sizes:
            weight_segments.append(weight[k_start : k_start + size])
            k_start += size
        # ND-only path (K-axis grouping rejects NZ): concatenating the row segments rebuilds the
        # whole [K, N] matrix shared by all groups.
        inputs.b_chunks = weight_segments
    else:
        for size in sizes:
            weight_segments.append(
                rng.uniform(-0.08, 0.08, size=(size, args.n)).astype(np_dtype)
            )
        inputs.b_files = weight_segments
    for segment, weight_segment in zip(segments, weight_segments):
        golden.append(
            (segment.astype(np.float32) @ weight_segment.astype(np.float32)).astype(
                np_dtype
            )
        )
    inputs.golden_chunks = golden
    return inputs, True


def generate_no_split(args, np_dtype, rng):
    """groupType=-1: m-m-m without groupList; each x_i @ weight_i is an independent matmul."""
    x_ms = parse_ints(args.x_ms)
    if len(x_ms) != args.group_num or sum(x_ms) != args.m:
        raise ValueError("xMs must describe one m per group and sum to m")
    inputs = Inputs(args.output_dir)
    for index, x_m in enumerate(x_ms):
        x = rng.uniform(-0.08, 0.08, size=(x_m, args.k)).astype(np_dtype)
        weight = rng.uniform(-0.08, 0.08, size=(args.k, args.n)).astype(np_dtype)
        inputs.a_chunks.append(x)
        block = x.astype(np.float32) @ weight.astype(np.float32)
        if args.is_bias:
            bias = np.full(args.n, BIAS_VALUE, np_dtype)
            block = block + bias.astype(np.float32)
            inputs.bias_files.append(bias)
        inputs.b_files.append(weight)
        inputs.golden_chunks.append(block.astype(np_dtype))
    return inputs, False


def write_outputs(args, inputs, single_x):
    os.makedirs(args.output_dir, exist_ok=True)
    if single_x:
        np.concatenate(inputs.a_chunks).tofile(
            os.path.join(args.output_dir, "input_a.bin")
        )
    else:
        for index, chunk in enumerate(inputs.a_chunks):
            chunk.tofile(os.path.join(args.output_dir, f"input_a_{index}.bin"))
    if inputs.b_files:
        for index, weight in enumerate(inputs.b_files):
            write_weight(
                os.path.join(args.output_dir, f"input_b_{index}.bin"),
                weight,
                args.weight_format,
                c0_of(args),
            )
    else:
        storage = [
            format_nz(w, c0_of(args)) if args.weight_format == "nz" else w
            for w in inputs.b_chunks
        ]
        np.concatenate(storage).tofile(os.path.join(args.output_dir, "input_b.bin"))
    if inputs.bias_matrix is not None:
        inputs.bias_matrix.tofile(os.path.join(args.output_dir, "bias.bin"))
    if inputs.bias_files:
        for index, bias in enumerate(inputs.bias_files):
            bias.tofile(os.path.join(args.output_dir, f"bias_{index}.bin"))
    if args.has_group_list:
        values = parse_ints(args.group_list)
        np.asarray(values, np.int64).tofile(
            os.path.join(args.output_dir, "group_list.bin")
        )
    np.concatenate([chunk.reshape(-1) for chunk in inputs.golden_chunks]).tofile(
        os.path.join(args.output_dir, "golden_c.bin")
    )


def c0_of(args):
    elem_size = np.dtype(DTYPE_MAP[args.dtype]).itemsize
    return 32 // elem_size


def validate_args(args):
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {args.dtype}")
    if args.weight_format not in ("nd", "nz"):
        raise ValueError(f"unsupported weight format: {args.weight_format}")
    if args.group_type not in (-1, 0, 2):
        raise ValueError(f"unsupported groupType: {args.group_type}")
    if args.group_list_type not in (0, 1, 2):
        raise ValueError(f"unsupported groupListType: {args.group_list_type}")
    if args.group_list_type == 2 and args.group_type != 0:
        raise ValueError("sparse groupList (type 2) only supports M-axis grouping")
    if args.group_type == -1 and args.has_group_list:
        raise ValueError("no-split (m-m-m) must not carry a groupList")
    if args.group_type == 2 and args.is_bias:
        raise ValueError("K-axis grouping does not support bias")
    if args.weight_format == "nz":
        if args.group_type == 2:
            raise ValueError("K-axis grouping does not support NZ weight")
        c0 = c0_of(args)
        if args.k % 16 != 0 or args.n % c0 != 0:
            raise ValueError("NZ weight requires k % 16 == 0 and n % c0 == 0")


def main():
    parser = argparse.ArgumentParser(
        description="Generate no-quant grouped matmul inputs"
    )
    parser.add_argument("--group-type", type=int, required=True)
    parser.add_argument("--group-list-type", type=int, required=True)
    parser.add_argument("--single-x", type=int, choices=(0, 1), required=True)
    parser.add_argument("--single-weight", type=int, choices=(0, 1), required=True)
    parser.add_argument("--single-y", type=int, choices=(0, 1), required=True)
    parser.add_argument("--weight-format", choices=("nd", "nz"), required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), required=True)
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--is-bias", type=int, choices=(0, 1), required=True)
    parser.add_argument("--has-group-list", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--x-ms", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate_args(args)
    np_dtype = DTYPE_MAP[args.dtype]
    rng = np.random.default_rng(args.seed)
    if args.group_type == 0:
        inputs, single_x = generate_m_split(args, np_dtype, rng)
    elif args.group_type == 2:
        inputs, single_x = generate_k_split(args, np_dtype, rng)
    else:
        inputs, single_x = generate_no_split(args, np_dtype, rng)
    write_outputs(args, inputs, single_x)
    print(
        f"[INFO] generated no-quant GMM data: groupType={args.group_type}, "
        f"groupListType={args.group_list_type}, weightFormat={args.weight_format}, "
        f"dtype={args.dtype}, groups={args.group_num}"
    )


if __name__ == "__main__":
    main()
