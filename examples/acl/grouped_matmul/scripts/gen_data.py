#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate deterministic QGMM MX inputs and a NumPy golden result."""

import argparse
import os

import ml_dtypes
import numpy as np


MX_SCALE_ONE = np.uint8(0x7F)
TYPE_INFO = {
    "mxfp8_e4m3": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x30, 0xB0, 0x38, 0xB8], np.uint8),
        32,
    ),
    "mxfp8_e5m2": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x38, 0xB8, 0x3C, 0xBC], np.uint8),
        32,
    ),
    "mxfp4_e2m1": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x1, 0x9, 0x2, 0xA], np.uint8),
        64,
    ),
    "mxfp4_e1m2": (
        np.array([0.5, -0.5, 1.0, -1.0], np.float32),
        np.array([0x2, 0xA, 0x4, 0xC], np.uint8),
        64,
    ),
}
OUTPUT_DTYPE_MAP = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _pack_fp4(values):
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    return flat[0::2] | (flat[1::2] << 4)


def _encode(values, dtype):
    return (
        _pack_fp4(values)
        if dtype.startswith("mxfp4")
        else np.asarray(values, np.uint8).reshape(-1)
    )


def _format_weight(codes, layout_b, k, n, c0):
    if layout_b == "nd":
        return codes.reshape(-1)
    if layout_b == "dn":
        return codes.T.reshape(-1)
    if layout_b == "nz":
        padded = np.zeros((_align_up(k, 16), _align_up(n, c0)), np.uint8)
        padded[:k, :n] = codes
        return (
            padded.reshape(-1, 16, padded.shape[1] // c0, c0)
            .transpose(2, 0, 1, 3)
            .reshape(-1)
        )
    padded = np.zeros((_align_up(n, 16), _align_up(k, c0)), np.uint8)
    padded[:n, :k] = codes.T
    return (
        padded.reshape(-1, 16, padded.shape[1] // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def _parse_group_list(text, group_num, total_size, list_type):
    values = np.asarray([int(value) for value in text.split(";")], np.int64)
    if list_type == 2:
        if values.size != group_num * 2:
            raise ValueError("sparse group-list must contain e index/length pairs")
        pairs = values.reshape(group_num, 2)
        if sorted(pairs[:, 0].tolist()) != list(range(group_num)):
            raise ValueError(
                "sparse group-list indices must be a permutation of [0, e)"
            )
        lengths = pairs[:, 1]
    elif list_type == 0:
        if (
            values.size != group_num
            or np.any(values < 0)
            or np.any(np.diff(values) < 0)
        ):
            raise ValueError("offset group-list must contain e nondecreasing offsets")
        lengths = np.diff(np.concatenate((np.zeros(1, np.int64), values)))
    else:
        if values.size != group_num:
            raise ValueError("length group-list must contain e lengths")
        lengths = values
    if np.any(lengths <= 0):
        raise ValueError("group-list lengths must be positive")
    if int(lengths.sum()) != total_size:
        raise ValueError("group-list must cover the complete grouped axis")
    return values, lengths


def _write_weights(output_dir, encoded, scales, single_w):
    if single_w == 0:
        for index, (weight, scale) in enumerate(zip(encoded, scales)):
            weight.tofile(os.path.join(output_dir, f"input_b_{index}.bin"))
            scale.tofile(os.path.join(output_dir, f"scale_b_{index}.bin"))
        return
    np.concatenate(encoded).tofile(os.path.join(output_dir, "input_b.bin"))
    np.concatenate(scales).tofile(os.path.join(output_dir, "scale_b.bin"))


def _validate_args(args):
    if args.group_num <= 0 or args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("group-num, m, n and k must be positive")
    weight_nz = args.layout_b in ("nz", "zn")
    if weight_nz and (args.k == 1 or args.n == 1):
        raise ValueError("weight NZ/ZN requires k and n to be greater than 1")
    if args.dtype == "mxfp4_e1m2" and not weight_nz:
        raise ValueError("mxfp4_e1m2 is supported only with weight NZ/ZN")
    if args.dtype.startswith("mxfp4"):
        if args.k % 2 or args.k == 2:
            raise ValueError("mxfp4 requires k to be even and not equal to 2")
        if args.layout_b in ("nd", "nz") and args.n % 2:
            raise ValueError(
                "mxfp4 with non-transposed weight ND/NZ requires n to be even"
            )
    if args.group_type != 2:
        return
    if (
        args.dtype.startswith("mxfp4")
        or args.layout_b != "nd"
        or args.single_w != 1
        or args.is_bias != 0
        or args.group_list_type == 2
    ):
        raise ValueError(
            "K-axis grouping supports MXFP8, single weight ND, no bias and Length/Offset Group List"
        )


def generate(args):
    _validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    values, codes, c0 = TYPE_INFO[args.dtype]
    rng = np.random.default_rng(args.seed)
    k_grouped = args.group_type == 2
    if k_grouped:
        group_list, k_lengths = _parse_group_list(
            args.group_list, args.group_num, args.k, args.group_list_type
        )
        a_indices = [
            rng.integers(0, values.size, size=(args.m, int(group_k)))
            for group_k in k_lengths
        ]
        b_indices = [
            rng.integers(0, values.size, size=(int(group_k), args.n))
            for group_k in k_lengths
        ]
        input_a = _encode(
            np.concatenate([codes[index].T.reshape(-1) for index in a_indices]),
            args.dtype,
        )
        encoded_b = [
            _encode(codes[index].reshape(-1), args.dtype) for index in b_indices
        ]
        scale_slots = args.k // 64 + args.group_num
        scale_a = np.full((scale_slots, args.m, 2), MX_SCALE_ONE, np.uint8)
        scale_b_storage = np.full((scale_slots, args.n, 2), MX_SCALE_ONE, np.uint8)
        golden_groups = []
        cumulative_k = 0
        for group_index, (group_k, a_index, b_index) in enumerate(
            zip(k_lengths, a_indices, b_indices)
        ):
            scale_count = (int(group_k) + 31) // 32
            scale_a_codes = rng.integers(
                0x7D, 0x80, size=(args.m, scale_count), dtype=np.uint8
            )
            scale_b_codes = rng.integers(
                0x7D, 0x80, size=(scale_count, args.n), dtype=np.uint8
            )
            scale_start = cumulative_k // 64 + group_index
            for scale_index in range(scale_count):
                slot = scale_start + scale_index // 2
                lane = scale_index % 2
                scale_a[slot, :, lane] = scale_a_codes[:, scale_index]
                scale_b_storage[slot, :, lane] = scale_b_codes[scale_index, :]
            a_scale = np.exp2(scale_a_codes.astype(np.int16) - 127)
            b_scale = np.exp2(scale_b_codes.astype(np.int16) - 127)
            k_scale_index = np.arange(int(group_k)) // 32
            scaled_a = values[a_index] * a_scale[:, k_scale_index]
            scaled_b = values[b_index] * b_scale[k_scale_index, :]
            golden_groups.append(scaled_a @ scaled_b)
            cumulative_k += int(group_k)
        golden = np.stack(golden_groups)
        scale_a = scale_a.reshape(-1)
        scale_b = [scale_b_storage.reshape(-1)]
    else:
        group_list, m_groups = _parse_group_list(
            args.group_list,
            args.group_num,
            args.group_num * args.m,
            args.group_list_type,
        )
        a_indices = [
            rng.integers(0, values.size, size=(int(group_m), args.k))
            for group_m in m_groups
        ]
        b_indices = rng.integers(0, values.size, size=(args.group_num, args.k, args.n))
        a_values = [values[index] for index in a_indices]
        b_values = values[b_indices]
        input_a = _encode(
            np.concatenate([codes[index].reshape(-1) for index in a_indices]),
            args.dtype,
        )
        golden = np.concatenate(
            [a_values[index] @ b_values[index] for index in range(args.group_num)],
            axis=0,
        )
        encoded_b = [
            _encode(
                _format_weight(
                    codes[b_indices[index]], args.layout_b, args.k, args.n, c0
                ),
                args.dtype,
            )
            for index in range(args.group_num)
        ]
        scale_k = _align_up(args.k, 64) // 32
        scale_a = np.full(args.group_num * args.m * scale_k, MX_SCALE_ONE, np.uint8)
        scale_b = [
            np.full(args.n * scale_k, MX_SCALE_ONE, np.uint8)
            for _ in range(args.group_num)
        ]
    bias = np.full((args.group_num, args.n), 0.25 if args.is_bias else 0.0, np.float32)
    if args.is_bias:
        golden += bias[0] if k_grouped else np.repeat(bias, m_groups, axis=0)

    input_a.tofile(os.path.join(args.output_dir, "input_a.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    bias.tofile(os.path.join(args.output_dir, "bias.bin"))
    group_list.tofile(os.path.join(args.output_dir, "group_list.bin"))
    _write_weights(args.output_dir, encoded_b, scale_b, args.single_w)
    golden.astype(OUTPUT_DTYPE_MAP[args.output_dtype]).tofile(
        os.path.join(args.output_dir, "golden_c.bin")
    )


def main():
    parser = argparse.ArgumentParser(description="Generate QGMM MX example data")
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(TYPE_INFO), required=True)
    parser.add_argument(
        "--output-dtype", choices=tuple(OUTPUT_DTYPE_MAP), required=True
    )
    parser.add_argument("--layout-b", choices=("nd", "dn", "nz", "zn"), required=True)
    parser.add_argument("--group-list-type", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--group-type", type=int, choices=(0, 2), required=True)
    parser.add_argument("--single-w", type=int, choices=(0, 1), required=True)
    parser.add_argument("--is-bias", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
