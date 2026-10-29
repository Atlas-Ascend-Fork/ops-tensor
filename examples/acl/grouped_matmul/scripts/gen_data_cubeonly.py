#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Generate deterministic QGMM Cube (INT8/FP8) inputs and CPU golden data."""

import argparse
import os

import numpy as np
import ml_dtypes

MX = ml_dtypes.float8_e4m3fn
FIXPIPE_SCALE_ONE = np.uint64(0x000040003F800000)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def parse_group_list(text, group_num, m, list_type):
    values = np.asarray([int(item) for item in text.split(";")], dtype=np.int64)
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
    if np.any(lengths <= 0) or int(lengths.sum()) != group_num * m:
        raise ValueError(
            "group-list must cover total M = groupNum * m with positive lengths"
        )
    return values


def format_b(codes, layout_b, k, n):
    """Convert one logical B matrix (K,N) to the physical layout expected by the kernel."""
    c0 = 32
    if layout_b == "nd":
        return codes.reshape(-1)
    if layout_b == "dn":
        return codes.T.reshape(-1)
    if layout_b == "nz":
        padded = np.zeros((align_up(k, 16), align_up(n, c0)), np.uint8)
        padded[:k, :n] = codes
        return (
            padded.reshape(-1, 16, padded.shape[1] // c0, c0)
            .transpose(2, 0, 1, 3)
            .reshape(-1)
        )
    padded = np.zeros((align_up(n, 16), align_up(k, c0)), np.uint8)
    padded[:n, :k] = codes.T
    return (
        padded.reshape(-1, 16, padded.shape[1] // c0, c0)
        .transpose(2, 0, 1, 3)
        .reshape(-1)
    )


def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(20260908)
    group_num = args.group_num
    total_m = group_num * args.m
    dtype = args.dtype.lower()
    x1 = args.x1_quant_mode.lower()
    x2 = args.x2_quant_mode.lower()
    per_channel = x2 in ("perchannel", "per_channel", "2")
    is_int8 = dtype in ("int8", "int8_t")
    is_fp8 = dtype in ("fp8_e4m3", "fp8_e4m3fn", "fp8_e4m3fn_t")

    if not (is_int8 or is_fp8):
        raise ValueError("unsupported dtype")
    if x1 not in ("default", "0"):
        if not is_fp8 or x1 not in ("pertensor", "per_tensor", "1"):
            raise ValueError(
                "x1QuantMode is only meaningful as PERTENSOR for FP8 double-scale"
            )
    if x2 not in ("pertensor", "per_tensor", "1", "perchannel", "per_channel", "2"):
        raise ValueError("unsupported x2QuantMode")
    if is_int8 and x1 not in ("default", "0"):
        raise ValueError("INT8 AIC_ONLY path requires x1QuantMode=DEFAULT")
    if not is_int8 and args.is_bias:
        raise ValueError("FP8 AIC_ONLY example does not support bias")

    parse_group_list(args.group_list, group_num, args.m, args.group_list_type)
    scale_a = (
        np.full(group_num, 1.0, np.float32)
        if x1 not in ("default", "0")
        else np.zeros(0, np.float32)
    )

    if is_int8:
        a = rng.integers(-4, 5, size=(total_m, args.k), dtype=np.int8)
        b = rng.integers(-4, 5, size=(group_num, args.k, args.n), dtype=np.int8)
        bias = np.zeros((group_num, args.n), dtype=np.int32)
        if args.is_bias:
            bias = np.tile(
                np.where(np.arange(args.n) % 2 == 0, 3, -5).astype(np.int32),
                (group_num, 1),
            )
        accum = np.matmul(
            a.astype(np.int32).reshape(group_num, args.m, args.k), b.astype(np.int32)
        )
        if args.is_bias:
            accum += bias[:, None, :]
        if per_channel:
            scale_b_float = np.ones((group_num, 1, args.n), np.float32)
            scale_b = np.full((group_num, args.n), FIXPIPE_SCALE_ONE, np.uint64)
        else:
            scale_b_float = np.ones((group_num, 1, 1), np.float32)
            scale_b = np.ones(group_num, np.float32)
        golden = accum * scale_b_float
        bias.tofile(os.path.join(args.output_dir, "bias.bin"))
    else:
        fp8_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], np.float32)
        fp8_codes = fp8_values.astype(MX).view(np.uint8)
        a_codes = fp8_codes[rng.integers(0, fp8_codes.size, size=(total_m, args.k))]
        b_codes = fp8_codes[
            rng.integers(0, fp8_codes.size, size=(group_num, args.k, args.n))
        ]
        a = a_codes.astype(np.uint8)
        b = b_codes.astype(np.uint8)
        a_fp32 = a.view(MX).astype(np.float32)
        b_fp32 = b.view(MX).astype(np.float32)
        bias = np.zeros((group_num, args.n), np.float32)
        if per_channel:
            scale_b_float = np.ones((group_num, 1, args.n), np.float32)
            scale_b = np.full((group_num, args.n), FIXPIPE_SCALE_ONE, np.uint64)
        else:
            scale_b_float = np.ones((group_num, 1, 1), np.float32)
            scale_b = np.ones(group_num, np.float32)
        if x1 not in ("default", "0"):
            scale_a_float = np.ones((group_num, 1, 1), np.float32)
            scale_b_float = scale_b_float * scale_a_float
        golden = np.matmul(a_fp32.reshape(group_num, args.m, args.k), b_fp32)
        golden = golden * scale_b_float
        bias.tofile(os.path.join(args.output_dir, "bias.bin"))

    a.tofile(os.path.join(args.output_dir, "input_a.bin"))
    np.concatenate(
        [format_b(b[idx], args.layout_b, args.k, args.n) for idx in range(group_num)]
    ).tofile(os.path.join(args.output_dir, "input_b.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(args.output_dir, "scale_b.bin"))
    output_dtype = getattr(args, "output_dtype", None) or (
        "fp16" if dtype == "int8" else "fp32"
    )
    output_types = {
        "fp16": np.float16,
        "float16": np.float16,
        "half": np.float16,
        "bf16": ml_dtypes.bfloat16,
        "bfloat16": ml_dtypes.bfloat16,
        "fp32": np.float32,
        "float32": np.float32,
        "float": np.float32,
    }
    golden.reshape(-1).astype(output_types[output_dtype]).tofile(
        os.path.join(args.output_dir, "golden_c.bin")
    )
    group_list = parse_group_list(
        args.group_list, group_num, args.m, args.group_list_type
    )
    group_list.tofile(os.path.join(args.output_dir, "group_list.bin"))


def main():
    parser = argparse.ArgumentParser(description="Generate QGMM Cube example data")
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--output-dtype",
        choices=[
            "fp16",
            "float16",
            "half",
            "bf16",
            "bfloat16",
            "fp32",
            "float32",
            "float",
        ],
    )
    parser.add_argument("--dtype", choices=("int8", "fp8_e4m3"), required=True)
    parser.add_argument("--layout-b", choices=("nd", "dn", "nz", "zn"), required=True)
    parser.add_argument("--x1-quant-mode", required=True)
    parser.add_argument("--x2-quant-mode", required=True)
    parser.add_argument("--is-bias", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list-type", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--single-w", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
