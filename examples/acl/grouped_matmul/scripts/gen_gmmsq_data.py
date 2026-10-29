#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate deterministic E4M3FN Weight-NZ/ZN SwiGLU mode-2 example data."""

import argparse
import os

import ml_dtypes
import numpy as np


MX_SCALE_ONE = np.uint8(0x7F)
MX_GROUP_SIZE = 32
MX_STORAGE_GROUP_SIZE = 64
FP8_MAX = np.float32(448.0)
CEIL_DIV_ADJUSTMENT = 1
LAST_AXIS = -1
SINGLE_GROUP_BOUNDARY = 1
NZ_K0 = 16
NZ_C0 = 32
WEIGHT_K_AXIS = 1
WEIGHT_N_AXIS = 2
SCALE_PAIR_SIZE = 2
SWIGLU_SPLIT_FACTOR = 2
SIGMOID_DENOMINATOR_OFFSET = np.float32(1.0)
BF16_ABSOLUTE_MASK = 0x7FFF
BF16_EXPONENT_MASK = 0x7F80
BF16_MANTISSA_MASK = 0x007F
BF16_MANTISSA_BITS = 7
E4M3_MAX_EXPONENT_BITS = 0x0400
E4M3_MAX_EXPONENT = 8
E4M3_MAX_MANTISSA = 0x0060
BF16_RECIPROCAL_EXPONENT_BIAS = 0x7F00
SCALE_ALG_CUBLAS = 1
HEAD_X_VALUE = 1.0
TAIL_X_VALUE = 0.5
TAIL_WEIGHT_FACTOR = 0.25
DATA_PATTERN_PERIOD = 5
DATA_PATTERN_CENTER = 2
ACT_GROUP_FACTOR = 2
GATE_COLUMN_FACTOR = 3
ROW_SCALE_CODE_BASE = 123
WEIGHT_SCALE_CODE_BASE = 126
E8M0_EXPONENT_BIAS = 127
TAIL_SCALE_EXPONENT_ADJUSTMENT = 1
MAX_EXAMPLE_GROUPS = 8
MIN_EXAMPLE_K = 1
WEIGHT_FRACTAL_PERMUTATION = (2, 0, 1, 3)
SCALE_PAIR_PERMUTATION = (0, 1, 3, 2)
TRANSPOSE_SCALE_PERMUTATION = (0, 2, 1)


def align_up(value, alignment):
    return (value + alignment - CEIL_DIV_ADJUSTMENT) // alignment * alignment


def parse_group_offsets(text, group_num, total_m):
    offsets = np.asarray([int(value) for value in text.split(";")], np.int64)
    if (
        offsets.size != group_num
        or np.any(offsets < 0)
        or np.any(offsets > total_m)
        or np.any(np.diff(offsets) < 0)
        or int(offsets[LAST_AXIS]) != total_m
    ):
        raise ValueError(
            "group-list must contain group_num nondecreasing offsets within [0, m] ending at m"
        )
    lengths = np.diff(
        np.concatenate((np.zeros(SINGLE_GROUP_BOUNDARY, np.int64), offsets))
    )
    return offsets, lengths


def encode_e4m3(values):
    return np.asarray(values, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)


def format_weight(codes, layout_b, k, n):
    if layout_b == "nz":
        padded = np.zeros((align_up(k, NZ_K0), align_up(n, NZ_C0)), np.uint8)
        padded[:k, :n] = codes
        return (
            padded.reshape(
                LAST_AXIS, NZ_K0, padded.shape[WEIGHT_K_AXIS] // NZ_C0, NZ_C0
            )
            .transpose(WEIGHT_FRACTAL_PERMUTATION)
            .reshape(LAST_AXIS)
        )
    padded = np.zeros((align_up(n, NZ_K0), align_up(k, NZ_C0)), np.uint8)
    padded[:n, :k] = codes.T
    return (
        padded.reshape(LAST_AXIS, NZ_K0, padded.shape[WEIGHT_K_AXIS] // NZ_C0, NZ_C0)
        .transpose(WEIGHT_FRACTAL_PERMUTATION)
        .reshape(LAST_AXIS)
    )


def format_weight_scale(codes, layout_b):
    group_num, scale_k, n = codes.shape
    if layout_b == "nz":
        if scale_k % SCALE_PAIR_SIZE != 0:
            raise ValueError("ScaleBND requires an even logical scale K")
        return (
            codes.reshape(group_num, scale_k // SCALE_PAIR_SIZE, SCALE_PAIR_SIZE, n)
            .transpose(SCALE_PAIR_PERMUTATION)
            .reshape(LAST_AXIS)
        )
    return codes.transpose(TRANSPOSE_SCALE_PERMUTATION).reshape(LAST_AXIS)


def compute_swiglu(
    group_lengths,
    x_values,
    x_scale_values,
    weight_values,
    weight_scale_values,
    clamp_limit,
    glu_alpha,
    glu_bias,
):
    output = []
    row_start = 0
    k_scale_indices = np.arange(x_values.shape[WEIGHT_K_AXIS]) // MX_GROUP_SIZE
    output_n = weight_values.shape[WEIGHT_N_AXIS] // SWIGLU_SPLIT_FACTOR
    for group_index, group_m in enumerate(group_lengths):
        row_end = row_start + int(group_m)
        scaled_x = (
            x_values[row_start:row_end]
            * x_scale_values[row_start:row_end][:, k_scale_indices]
        )
        scaled_weight = (
            weight_values[group_index]
            * weight_scale_values[group_index][k_scale_indices]
        )
        matmul = scaled_x @ scaled_weight
        act = matmul[:, :output_n]
        gate = matmul[:, output_n:]
        act = np.minimum(act, np.float32(clamp_limit))
        gate = np.clip(gate, -clamp_limit, clamp_limit) + np.float32(glu_bias)
        swish = act / (
            SIGMOID_DENOMINATOR_OFFSET + np.exp(-np.float32(glu_alpha) * act)
        )
        output.append((swish * gate).astype(ml_dtypes.bfloat16))
        row_start = row_end
    return np.concatenate(output, axis=0)


def quantize_golden(values_bf16, scale_alg):
    m, n = values_bf16.shape
    if n % MX_GROUP_SIZE != 0:
        raise ValueError("output N must be a multiple of 32")
    aligned_n = align_up(n, MX_STORAGE_GROUP_SIZE)
    padded = np.zeros((m, aligned_n), dtype=ml_dtypes.bfloat16)
    padded[:, :n] = values_bf16
    bits = padded.view(np.uint16).reshape(m, aligned_n // MX_GROUP_SIZE, MX_GROUP_SIZE)
    max_bits = np.max(bits & np.uint16(BF16_ABSOLUTE_MASK), axis=LAST_AXIS).astype(
        np.int32
    )
    if scale_alg == 0:
        # The shared OCP Tile reduces exponent bits, so BF16 subnormals have a zero exponent.
        nonzero = (max_bits & BF16_EXPONENT_MASK) != 0
        max_exp_bits = np.maximum(max_bits & BF16_EXPONENT_MASK, E4M3_MAX_EXPONENT_BITS)
        scale_code = (max_exp_bits - E4M3_MAX_EXPONENT_BITS) >> BF16_MANTISSA_BITS
    elif scale_alg == SCALE_ALG_CUBLAS:
        nonzero = max_bits != 0
        exponent = max_bits >> BF16_MANTISSA_BITS
        mantissa = max_bits & BF16_MANTISSA_MASK
        exponent += mantissa > E4M3_MAX_MANTISSA
        scale_code = np.maximum(exponent - E4M3_MAX_EXPONENT, 0)
    else:
        raise ValueError(f"unsupported scale algorithm: {scale_alg}")
    scale_code = np.where(nonzero, scale_code, 0).astype(np.uint8)

    reciprocal_bits = np.where(
        nonzero,
        BF16_RECIPROCAL_EXPONENT_BIAS
        - (scale_code.astype(np.int32) << BF16_MANTISSA_BITS),
        0,
    ).astype(np.uint16)
    reciprocal = reciprocal_bits.view(ml_dtypes.bfloat16)
    scaled = (
        padded.reshape(m, aligned_n // MX_GROUP_SIZE, MX_GROUP_SIZE)
        * reciprocal[:, :, None]
    ).astype(ml_dtypes.bfloat16)
    scaled_fp32 = scaled.astype(np.float32)
    if not np.all(np.isfinite(scaled_fp32)) or np.any(np.abs(scaled_fp32) > FP8_MAX):
        raise ValueError("generated data exceeds the finite E4M3FN range")
    quantized = scaled_fp32.astype(ml_dtypes.float8_e4m3fn)
    y = quantized.reshape(m, aligned_n)[:, :n].copy().view(np.uint8)
    y_scale = scale_code.reshape(m, aligned_n // MX_STORAGE_GROUP_SIZE, SCALE_PAIR_SIZE)
    return y, y_scale


def generate(args):
    if min(args.group_num, args.m, args.n, args.k) <= 0:
        raise ValueError("group-num, m, n and k must be positive")
    if (
        args.k <= MIN_EXAMPLE_K
        or args.n % SWIGLU_SPLIT_FACTOR != 0
        or (args.n // SWIGLU_SPLIT_FACTOR) % MX_GROUP_SIZE != 0
    ):
        raise ValueError("full n must be even and output n must be a multiple of 32")
    if args.group_num > MAX_EXAMPLE_GROUPS or args.scale_alg not in (
        0,
        SCALE_ALG_CUBLAS,
    ):
        raise ValueError("the example supports at most 8 groups and scale-alg 0/1")
    attrs = (args.clamp_limit, args.glu_alpha, args.glu_bias, args.dst_type_max)
    if (
        not all(
            np.isfinite(value) and abs(value) <= float(np.finfo(np.float32).max)
            for value in attrs
        )
        or args.clamp_limit <= 0.0
        or np.float32(args.clamp_limit) <= 0.0
        or args.dst_type_max != 0.0
    ):
        raise ValueError(
            "attributes must be finite float32 values with positive clamp-limit and dst-type-max=0"
        )
    offsets, group_lengths = parse_group_offsets(
        args.group_list, args.group_num, args.m
    )
    output_n = args.n // SWIGLU_SPLIT_FACTOR
    scale_k = align_up(args.k, MX_STORAGE_GROUP_SIZE) // MX_GROUP_SIZE

    os.makedirs(args.output_dir, exist_ok=True)
    x_values = np.zeros((args.m, args.k), np.float32)
    x_values[:, 0] = HEAD_X_VALUE
    x_values[:, LAST_AXIS] = TAIL_X_VALUE
    x_codes = encode_e4m3(x_values)

    columns = np.arange(output_n, dtype=np.int64)
    acts = np.stack(
        [
            (
                (columns + ACT_GROUP_FACTOR * group) % DATA_PATTERN_PERIOD
                - DATA_PATTERN_CENTER
            ).astype(np.float32)
            for group in range(args.group_num)
        ]
    )
    gates = np.stack(
        [
            (
                (GATE_COLUMN_FACTOR * columns + group) % DATA_PATTERN_PERIOD
                - DATA_PATTERN_CENTER
            ).astype(np.float32)
            for group in range(args.group_num)
        ]
    )
    tail_acts = np.stack(
        [
            (
                (
                    (ACT_GROUP_FACTOR * columns + group) % DATA_PATTERN_PERIOD
                    - DATA_PATTERN_CENTER
                )
                * TAIL_WEIGHT_FACTOR
            ).astype(np.float32)
            for group in range(args.group_num)
        ]
    )
    tail_gates = np.stack(
        [
            (
                (
                    (columns + GATE_COLUMN_FACTOR * group) % DATA_PATTERN_PERIOD
                    - DATA_PATTERN_CENTER
                )
                * TAIL_WEIGHT_FACTOR
            ).astype(np.float32)
            for group in range(args.group_num)
        ]
    )
    logical_weights = np.zeros((args.group_num, args.k, args.n), np.float32)
    formatted_weights = []
    for group in range(args.group_num):
        logical_weights[group, 0, :output_n] = acts[group]
        logical_weights[group, 0, output_n:] = gates[group]
        logical_weights[group, LAST_AXIS, :output_n] = tail_acts[group]
        logical_weights[group, LAST_AXIS, output_n:] = tail_gates[group]
        formatted_weights.append(
            format_weight(
                encode_e4m3(logical_weights[group]), args.layout_b, args.k, args.n
            )
        )

    row_scale_codes = (
        ROW_SCALE_CODE_BASE + np.arange(args.m, dtype=np.int64) % DATA_PATTERN_PERIOD
    ).astype(np.uint8)
    x_scale_codes = np.repeat(row_scale_codes[:, None], scale_k, axis=WEIGHT_K_AXIS)
    tail_scale_index = (args.k - CEIL_DIV_ADJUSTMENT) // MX_GROUP_SIZE
    x_scale_codes[:, tail_scale_index] = np.maximum(
        x_scale_codes[:, tail_scale_index].astype(np.int16)
        - TAIL_SCALE_EXPONENT_ADJUSTMENT,
        0,
    ).astype(np.uint8)
    x_scale_values = np.exp2(
        x_scale_codes.astype(np.int16) - E8M0_EXPONENT_BIAS
    ).astype(np.float32)

    full_columns = np.arange(args.n, dtype=np.int64)
    half_tag = full_columns >= output_n
    weight_scale_codes = np.empty((args.group_num, scale_k, args.n), dtype=np.uint8)
    for group in range(args.group_num):
        for scale_index in range(scale_k):
            weight_scale_codes[group, scale_index] = (
                WEIGHT_SCALE_CODE_BASE
                + (full_columns + half_tag + scale_index // SCALE_PAIR_SIZE + group)
                % SCALE_PAIR_SIZE
            )
    weight_scale_values = np.exp2(
        weight_scale_codes.astype(np.int16) - E8M0_EXPONENT_BIAS
    ).astype(np.float32)
    formatted_weight_scale = format_weight_scale(weight_scale_codes, args.layout_b)

    golden_bf16 = compute_swiglu(
        group_lengths,
        x_values,
        x_scale_values,
        logical_weights,
        weight_scale_values,
        args.clamp_limit,
        args.glu_alpha,
        args.glu_bias,
    )
    golden_y, golden_scale = quantize_golden(golden_bf16, args.scale_alg)

    x_codes.tofile(os.path.join(args.output_dir, "input_x.bin"))
    np.concatenate(formatted_weights).tofile(
        os.path.join(args.output_dir, "input_weight.bin")
    )
    formatted_weight_scale.tofile(os.path.join(args.output_dir, "weight_scale.bin"))
    x_scale_codes.tofile(os.path.join(args.output_dir, "x_scale.bin"))
    offsets.tofile(os.path.join(args.output_dir, "group_list.bin"))
    golden_y.tofile(os.path.join(args.output_dir, "golden_y.bin"))
    golden_scale.tofile(os.path.join(args.output_dir, "golden_y_scale.bin"))


def main():
    parser = argparse.ArgumentParser(description="Generate QGMM SwiGLU MX example data")
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--layout-b", choices=("nz", "zn"), required=True)
    parser.add_argument(
        "--scale-alg", type=int, choices=(0, SCALE_ALG_CUBLAS), required=True
    )
    parser.add_argument("--clamp-limit", type=float, required=True)
    parser.add_argument("--glu-alpha", type=float, required=True)
    parser.add_argument("--glu-bias", type=float, required=True)
    parser.add_argument("--dst-type-max", type=float, required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.dst_type_max != 0.0:
        raise ValueError("scaleAlg 0/1 examples require dst-type-max=0")
    generate(args)


if __name__ == "__main__":
    main()
