#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate deterministic Cube and per-tensor StreamK inputs and golden results."""

import argparse
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np


DEQ_SCALE_MASK = np.uint32(0xFFFFE000)


def parse_bool(value):
    normalized = value.lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true, false, 1, or 0")


def normalize_quant_mode(value):
    normalized = value.lower().replace("_", "")
    if normalized not in ("default", "pertensor", "perchannel"):
        raise ValueError("quant mode must be default, pertensor, or perchannel")
    return normalized


def truncate_deq_scale(value):
    """Match Fixpipe scale precision by clearing the low 13 FP32 bits."""
    scale = np.atleast_1d(np.array(value, dtype=np.float32, copy=True))
    scale_bits = scale.view(np.uint32)
    scale_bits &= DEQ_SCALE_MASK
    return scale_bits.view(np.float32)


def generate(args):
    x1_mode = normalize_quant_mode(args.x1_quant_mode)
    x2_mode = normalize_quant_mode(args.x2_quant_mode)
    a_type = args.a_type.lower()
    b_type = args.b_type.lower()
    c_type = args.c_type.lower()
    bias_type = args.bias_type.lower()
    scale_type = args.x2_scale_type.lower()
    is_pertensor_streamk = args.kernel_variant == "pertensor_streamk"
    is_hifloat8 = (
        a_type in ("hifloat8", "hifloat8_t")
        and b_type in ("hifloat8", "hifloat8_t")
        and c_type in ("bfloat16", "bfloat16_t", "bf16")
        and bias_type in ("float", "float32")
    )
    is_int8_input = (
        a_type in ("int8", "int8_t")
        and b_type in ("int8", "int8_t")
        and bias_type in ("int32", "int32_t")
    )
    is_int8 = is_int8_input and c_type in ("int32", "int32_t")
    is_int8_streamk = (
        is_pertensor_streamk
        and is_int8_input
        and c_type in ("float16", "float16_t", "half")
    )
    valid_hifloat8_mode = (
        x2_mode == "pertensor" and x1_mode in ("default", "pertensor")
    ) or (x1_mode == "default" and x2_mode == "perchannel")
    valid_int8_mode = x1_mode == "default" and x2_mode == "default"
    valid_int8_streamk_mode = x1_mode == "default" and x2_mode == "pertensor"
    valid_hifloat8_scale = scale_type in (
        ("uint64", "uint64_t") if x2_mode == "perchannel" else ("float", "float32")
    )
    if not (
        (
            not is_pertensor_streamk
            and (
                (is_hifloat8 and valid_hifloat8_mode and valid_hifloat8_scale)
                or (is_int8 and valid_int8_mode and scale_type in ("float", "float32"))
            )
        )
        or (
            is_int8_streamk
            and valid_int8_streamk_mode
            and scale_type in ("float", "float32")
        )
    ):
        raise ValueError("unsupported dtype, quant mode, or x2ScaleType combination")
    if min(args.batch, args.m, args.k, args.n) <= 0:
        raise ValueError("batch/M/K/N must be positive")
    if args.bias not in (0, args.n):
        raise ValueError("bias must be 0 or N")
    if is_pertensor_streamk and (
        args.batch != 1 or args.bias != 0 or args.trans_a or args.trans_b
    ):
        raise ValueError(
            "pertensor_streamk requires batch=1, no bias, and no transpose"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(20260803)
    scale_a = np.array([1.0], dtype=np.float32)
    scale_b = np.array([1.0], dtype=np.float32)

    if is_int8 or is_int8_streamk:
        logical_a = rng.integers(
            -8, 9, size=(args.batch, args.m, args.k), dtype=np.int8
        )
        logical_b = rng.integers(
            -8, 9, size=(args.batch, args.k, args.n), dtype=np.int8
        )
        bias = np.zeros(args.n, dtype=np.int32)
        if args.bias:
            bias = np.where(np.arange(args.n) % 2 == 0, 17, -31).astype(np.int32)
        golden = np.matmul(logical_a.astype(np.int32), logical_b.astype(np.int32))
        if args.bias:
            golden += bias.reshape(1, 1, args.n)
        if is_int8_streamk:
            scale_b = np.array([0.5], dtype=np.float32)
            golden = (golden.astype(np.float32) * scale_b[0]).astype(np.float16)
        stored_a = logical_a.transpose(0, 2, 1) if args.trans_a else logical_a
        stored_b = logical_b.transpose(0, 2, 1) if args.trans_b else logical_b
        stored_a.copy().tofile(os.path.join(args.output_dir, "input_a.bin"))
        stored_b.copy().tofile(os.path.join(args.output_dir, "input_b.bin"))
        scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
        scale_b.tofile(os.path.join(args.output_dir, "scale_b.bin"))
        bias.tofile(os.path.join(args.output_dir, "bias.bin"))
        golden.astype(np.float16 if is_int8_streamk else np.int32).tofile(
            os.path.join(args.output_dir, "golden_c.bin")
        )
        return

    import torch
    from en_dtypes import hifloat8

    logical_a = rng.integers(8, 30, size=(args.batch, args.m, args.k), dtype=np.uint8)
    logical_b = rng.integers(8, 30, size=(args.batch, args.k, args.n), dtype=np.uint8)
    a_fp32 = logical_a.view(hifloat8).astype(np.float32)
    b_fp32 = logical_b.view(hifloat8).astype(np.float32)

    bias = np.zeros(args.n, dtype=np.float32)
    if args.bias:
        bias = np.where(np.arange(args.n) % 2 == 0, 0.5, -1.0).astype(np.float32)

    accum = torch.matmul(torch.from_numpy(a_fp32), torch.from_numpy(b_fp32))
    if args.bias:
        accum += torch.from_numpy(bias).reshape(1, 1, args.n)

    if x1_mode == "pertensor":
        scale_a[0] = rng.uniform(1.0, 2.0)

    if x2_mode == "perchannel":
        scale_b = rng.uniform(0.01, 2.0, size=args.n).astype(np.float32)
        effective_scale = torch.from_numpy(truncate_deq_scale(scale_b)).reshape(
            1, 1, args.n
        )
    else:
        scale_b = np.array([rng.uniform(1.0, 2.0)], dtype=np.float32)
        combined = scale_a * scale_b if x1_mode == "pertensor" else scale_b
        effective_scale = torch.from_numpy(truncate_deq_scale(combined)).reshape(
            1, 1, 1
        )
    golden = (accum * effective_scale).to(torch.bfloat16)

    stored_a = logical_a.transpose(0, 2, 1) if args.trans_a else logical_a
    stored_b = logical_b.transpose(0, 2, 1) if args.trans_b else logical_b
    stored_a.copy().tofile(os.path.join(args.output_dir, "input_a.bin"))
    stored_b.copy().tofile(os.path.join(args.output_dir, "input_b.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(args.output_dir, "scale_b.bin"))
    bias.tofile(os.path.join(args.output_dir, "bias.bin"))
    golden.view(torch.uint16).numpy().tofile(
        os.path.join(args.output_dir, "golden_c.bin")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bias", type=int, required=True)
    parser.add_argument("--a-type", required=True)
    parser.add_argument("--b-type", required=True)
    parser.add_argument("--c-type", required=True)
    parser.add_argument("--bias-type", required=True)
    parser.add_argument("--trans-a", type=parse_bool, required=True)
    parser.add_argument("--trans-b", type=parse_bool, required=True)
    parser.add_argument("--x1-quant-mode", required=True)
    parser.add_argument("--x2-quant-mode", required=True)
    parser.add_argument("--x2-scale-type", required=True)
    parser.add_argument(
        "--kernel-variant",
        default="batch",
        choices=("batch", "without_batch", "pertensor_streamk"),
    )
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
