#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate deterministic grouped MX A8W4 inputs and FP16/BF16 golden data."""

import argparse
import os

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
MX_SCALE_EXPONENT_BIAS = 127
MX_GROUP_SIZE = 32
NZ_K0 = 32
NZ_N0 = 16
FP4_PACK_FACTOR = 2
GROUP_LIST_TYPE_SPARSE = 2
SPARSE_GROUP_LIST_ITEM_STRIDE = 2
DEFAULT_SEED = 20260829
OUTPUT_DTYPES = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
}
WEIGHT_TYPES = {
    "float4_e2m1": {
        "codes": np.arange(0x10, dtype=np.uint8),
        "values": np.asarray(
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
        ),
    },
    "float4_e1m2": {
        "codes": np.arange(0x10, dtype=np.uint8),
        "values": np.asarray(
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                -0.0,
                -0.25,
                -0.5,
                -0.75,
                -1.0,
                -1.25,
                -1.5,
                -1.75,
            ],
            dtype=np.float32,
        ),
    },
}


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def parse_group_list(text, group_num, total_m, group_list_type):
    values = np.asarray([int(item) for item in text.split(";")], dtype=np.int64)
    entry_count = group_num * (
        SPARSE_GROUP_LIST_ITEM_STRIDE
        if group_list_type == GROUP_LIST_TYPE_SPARSE
        else 1
    )
    if values.size != entry_count:
        raise ValueError(f"groupList must contain exactly {entry_count} entries")
    if np.any(values < 0):
        raise ValueError("groupList entries must be nonnegative")
    if group_list_type == GROUP_LIST_TYPE_SPARSE:
        values = values.reshape(group_num, SPARSE_GROUP_LIST_ITEM_STRIDE)
        experts, counts = values[:, 0], values[:, 1]
        if sorted(experts.tolist()) != list(range(group_num)):
            raise ValueError(
                "sparse groupList indices must be a permutation of [0, groupNum)"
            )
        active = counts > 0
        if np.any(active[1:] & ~active[:-1]):
            raise ValueError("sparse groupList must place all zero-size groups last")
        if np.any(np.diff(experts[active]) <= 0) or np.any(
            np.diff(experts[~active]) <= 0
        ):
            raise ValueError(
                "active and zero-size groups must each be ordered by expert index"
            )
        if int(counts.sum()) != total_m:
            raise ValueError("sparse groupList counts must sum to totalM")
        # The existing generator stores weights by expert index. Restore counts to
        # that order so X/Y remain contiguous while W, ScaleB and Bias select experts.
        lengths = np.zeros(group_num, dtype=np.int64)
        lengths[experts] = counts
    elif group_list_type == 0:
        if np.any(np.diff(values) < 0):
            raise ValueError("offset groupList must be nondecreasing")
        if int(values[-1]) != total_m:
            raise ValueError("the last offset must equal totalM")
        lengths = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), values)))
    else:
        if int(values.sum()) != total_m:
            raise ValueError("length groupList entries must sum to totalM")
        lengths = values.copy()
    return values, lengths


def pack_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % FP4_PACK_FACTOR != 0:
        raise ValueError("the FP4 packing axis must have an even length")
    low = np.bitwise_and(codes[..., 0::2], 0x0F)
    high = np.left_shift(np.bitwise_and(codes[..., 1::2], 0x0F), 4)
    return np.bitwise_or(low, high).astype(np.uint8)


def format_weight_zn(codes):
    k, n = codes.shape
    k_aligned = align_up(k, NZ_K0)
    n_aligned = align_up(n, NZ_N0)
    padded = np.zeros((k_aligned, n_aligned), dtype=np.uint8)
    padded[:k, :n] = codes
    zn = padded.reshape(k_aligned // NZ_K0, NZ_K0, n_aligned // NZ_N0, NZ_N0)
    return pack_fp4(zn.transpose(0, 2, 3, 1)).reshape(-1)


def make_scale(shape, phase):
    coordinates = np.indices(shape, dtype=np.int64).sum(axis=0)
    exponents = (coordinates + phase) % 3 - 1
    return (exponents + MX_SCALE_EXPONENT_BIAS).astype(np.uint8)


def decode_scale(scale):
    exponents = scale.astype(np.int16) - MX_SCALE_EXPONENT_BIAS
    return np.exp2(exponents.astype(np.float32))


def make_bias(n, group_index, enabled, output_dtype):
    if not enabled:
        return np.zeros(n, dtype=output_dtype)
    values = np.where((np.arange(n) + group_index) % 2 == 0, 0.5, -0.5)
    return values.astype(output_dtype)


def validate_args(args):
    if args.group_num <= 0 or args.total_m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("groupNum, totalM, N and K must be positive")
    if args.single_w == 0 and args.group_num > 128:
        raise ValueError("TensorList mode supports at most 128 experts")
    if args.single_w == 1 and args.group_num > 1024:
        raise ValueError("contiguous mode supports at most 1024 experts")
    if args.k % 64 != 0 or args.n % 64 != 0:
        raise ValueError("this example requires K % 64 == 0 and N % 64 == 0")


def generate(args):
    validate_args(args)
    group_list, group_lengths = parse_group_list(
        args.group_list, args.group_num, args.total_m, args.group_list_type
    )
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    output_dtype = OUTPUT_DTYPES[args.c_dtype]
    weight_type = WEIGHT_TYPES[args.weight_dtype]
    scale_k = align_up(args.k, 64) // MX_GROUP_SIZE
    activation_values = np.asarray([0.5, 1.0, -1.0, 2.0], dtype=np.float32)
    activation = rng.choice(activation_values, size=(args.total_m, args.k)).astype(
        FP8_E4M3FN
    )
    scale_a = make_scale((args.total_m, scale_k), 0)
    golden = np.zeros((args.total_m, args.n), dtype=np.float32)

    packed_weights = []
    scale_b_storage = []
    biases = []
    m_offset = 0
    for group_index, group_m in enumerate(group_lengths.tolist()):
        value_indices = (
            np.arange(args.k, dtype=np.uint64)[:, None]
            + np.arange(args.n, dtype=np.uint64)[None, :]
            + group_index
        ) % weight_type["codes"].size
        weight_codes = weight_type["codes"][value_indices]
        weight_values = weight_type["values"][value_indices]
        packed_weight = format_weight_zn(weight_codes)
        scale_b = make_scale((scale_k, args.n), group_index)
        bias = make_bias(args.n, group_index, args.is_bias == 1, output_dtype)

        if group_m > 0:
            m_end = m_offset + group_m
            a_scale = np.repeat(
                decode_scale(scale_a[m_offset:m_end]), MX_GROUP_SIZE, axis=1
            )[:, : args.k]
            b_scale = np.repeat(decode_scale(scale_b), MX_GROUP_SIZE, axis=0)[
                : args.k, :
            ]
            golden[m_offset:m_end] = (
                activation[m_offset:m_end].astype(np.float32) * a_scale
            ) @ (weight_values * b_scale)
            if args.is_bias == 1:
                golden[m_offset:m_end] += bias.astype(np.float32)
            m_offset = m_end

        if args.single_w == 0:
            packed_weight.tofile(
                os.path.join(args.output_dir, f"input_b_{group_index}.bin")
            )
            scale_b.T.copy().tofile(
                os.path.join(args.output_dir, f"scale_b_{group_index}.bin")
            )
            if args.is_bias == 1:
                bias.tofile(os.path.join(args.output_dir, f"bias_{group_index}.bin"))
        else:
            packed_weights.append(packed_weight)
            scale_b_storage.append(scale_b.T.copy().view(np.uint8))
            if args.is_bias == 1:
                biases.append(bias)

    if args.single_w == 1:
        np.concatenate(packed_weights).tofile(
            os.path.join(args.output_dir, "input_b.bin")
        )
        np.concatenate(scale_b_storage).tofile(
            os.path.join(args.output_dir, "scale_b.bin")
        )
        if args.is_bias == 1:
            np.concatenate(biases).tofile(os.path.join(args.output_dir, "bias.bin"))

    activation.view(np.uint8).tofile(os.path.join(args.output_dir, "input_a.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    group_list.tofile(os.path.join(args.output_dir, "group_list.bin"))
    golden.astype(output_dtype).tofile(os.path.join(args.output_dir, "golden_c.bin"))
    print(
        "[SUCCESS] generated grouped MX A8W4 data: "
        f"E={args.group_num}, M={args.total_m}, K={args.k}, N={args.n}, "
        f"weight={args.weight_dtype}, output={args.c_dtype}, singleW={args.single_w}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grouped FP8 E4M3 x packed FP4 E2M1/E1M2 data and golden"
    )
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--total-m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--weight-dtype", choices=tuple(WEIGHT_TYPES), required=True)
    parser.add_argument("--c-dtype", choices=tuple(OUTPUT_DTYPES), required=True)
    parser.add_argument("--is-bias", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--group-list-type",
        type=int,
        choices=(0, 1, GROUP_LIST_TYPE_SPARSE),
        required=True,
    )
    parser.add_argument("--single-w", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
