#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic finite-input accuracy examples; golden is derived from serialized input bytes."""

import argparse
from pathlib import Path
import numpy as np
import ml_dtypes

OUTPUT_DTYPES = {"float16": np.float16, "bfloat16": ml_dtypes.bfloat16}


def weight_dtype(name):
    if name == "hifloat8":
        from en_dtypes import hifloat8

        return hifloat8
    return {
        "int8": np.int8,
        "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
        "float8_e5m2": ml_dtypes.float8_e5m2,
    }[name]


def cast_weight(weight, name, out_dtype):
    # Preserve the two-step INT8/HiFloat8 -> half -> BF16 path.
    if out_dtype == ml_dtypes.bfloat16 and name in ("int8", "hifloat8"):
        return weight.astype(np.float16).astype(out_dtype)
    if name in ("float8_e4m3fn", "float8_e5m2"):
        return weight.astype(np.float32).astype(out_dtype)
    return weight.astype(out_dtype)


def dequantize(weight, scale, offset, name, out_dtype):
    value = cast_weight(weight, name, out_dtype)
    if offset is not None:
        value = (
            value.astype(np.float32) + offset.astype(np.float32)[:, None, :]
        ).astype(out_dtype)
    return (value.astype(np.float32) * scale.astype(np.float32)[:, None, :]).astype(
        out_dtype
    )


def parse_groups(text, groups, total_m, list_type):
    values = np.asarray([int(x) for x in text.split(";")], dtype=np.int64)
    if values.size != groups or np.any(values < 0):
        raise ValueError("group list must contain E nonnegative values")
    counts = np.diff(np.r_[np.int64(0), values]) if list_type == 0 else values.copy()
    if np.any(counts < 0) or int(counts.sum()) != total_m:
        raise ValueError("invalid group list order or total M")
    return values, counts


def generate(args):
    if min(args.groups, args.m, args.n, args.k) <= 0 or args.groups > 1024:
        raise ValueError("invalid E/M/N/K")
    if args.offset and args.weight_dtype != "int8":
        raise ValueError("offset requires int8 weight")
    if args.bias_dtype not in ("none", args.x_dtype) and not (
        args.x_dtype == "bfloat16" and args.bias_dtype == "float32"
    ):
        raise ValueError("unsupported bias dtype")
    out_dtype = OUTPUT_DTYPES[args.x_dtype]
    wdtype = weight_dtype(args.weight_dtype)
    bdtype = np.float32 if args.bias_dtype == "float32" else out_dtype
    groups, counts = parse_groups(args.group_list, args.groups, args.m, args.list_type)
    rng = np.random.default_rng(args.seed)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (args.groups, args.k, args.n)
    x = rng.uniform(-0.5, 0.5, (args.m, args.k)).astype(out_dtype)
    raw_weight = (
        rng.integers(-16, 17, shape)
        if args.weight_dtype == "int8"
        else rng.uniform(-4, 4, shape)
    )
    weight = raw_weight.astype(wdtype)
    scale = rng.uniform(0.03, 0.12, (args.groups, args.n)).astype(out_dtype)
    offset = rng.uniform(-0.5, 0.5, (args.groups, args.n)).astype(out_dtype)
    bias = rng.uniform(-0.125, 0.125, (args.groups, args.n)).astype(bdtype)
    x.tofile(directory / "x.bin")
    np.ascontiguousarray(weight.transpose(0, 2, 1) if args.trans else weight).tofile(
        directory / "weight.bin"
    )
    scale.tofile(directory / "scale.bin")
    if args.offset:
        offset.tofile(directory / "offset.bin")
    if args.bias_dtype != "none":
        bias.tofile(directory / "bias.bin")
    groups.tofile(directory / "group_list.bin")
    # Read the actual H2D bytes back, including physical NT storage; transpose exactly once.
    x = np.fromfile(directory / "x.bin", dtype=out_dtype).reshape(args.m, args.k)
    physical_shape = (args.groups, args.n, args.k) if args.trans else shape
    weight = np.fromfile(directory / "weight.bin", dtype=wdtype).reshape(physical_shape)
    weight = weight.transpose(0, 2, 1) if args.trans else weight
    scale = np.fromfile(directory / "scale.bin", dtype=out_dtype).reshape(
        args.groups, args.n
    )
    offset = (
        np.fromfile(directory / "offset.bin", dtype=out_dtype).reshape(
            args.groups, args.n
        )
        if args.offset
        else None
    )
    converted = dequantize(weight, scale, offset, args.weight_dtype, out_dtype)
    converted.tofile(
        directory / "golden_weight.bin"
    )  # Diagnostic stages, not a second device kernel.
    y = np.empty((args.m, args.n), dtype=out_dtype)
    start = 0
    for expert, count in enumerate(counts):
        end = start + int(count)
        if count:
            acc = x[start:end].astype(np.float32) @ converted[expert].astype(np.float32)
            if args.bias_dtype != "none":
                # Output-column bias; an independent FP32 reference, not the MMAD accumulation order.
                stored_bias = np.fromfile(directory / "bias.bin", dtype=bdtype).reshape(
                    args.groups, args.n
                )
                acc = acc + stored_bias[expert].astype(np.float32)
            y[start:end] = acc.astype(out_dtype)
        start = end
    if not np.isfinite(y.astype(np.float32)).all():
        raise ValueError("finite-input example produced a nonfinite golden")
    y.tofile(directory / "golden_y.bin")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("groups", "m", "n", "k"):
        p.add_argument("--" + key, type=int, required=True)
    p.add_argument("--x-dtype", choices=tuple(OUTPUT_DTYPES), required=True)
    p.add_argument(
        "--weight-dtype",
        choices=("int8", "float8_e4m3fn", "float8_e5m2", "hifloat8"),
        required=True,
    )
    p.add_argument(
        "--bias-dtype",
        choices=("none", "float16", "bfloat16", "float32"),
        required=True,
    )
    p.add_argument("--trans", type=int, choices=(0, 1), required=True)
    p.add_argument("--offset", type=int, choices=(0, 1), required=True)
    p.add_argument("--list-type", type=int, choices=(0, 1), required=True)
    p.add_argument("--group-list", required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--output-dir", required=True)
    generate(p.parse_args())


if __name__ == "__main__":
    main()
