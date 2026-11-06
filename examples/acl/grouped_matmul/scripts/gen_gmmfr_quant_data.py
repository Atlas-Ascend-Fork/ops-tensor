#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""Generate quantized GMMFR example inputs and an independent CPU reference."""

import argparse
from pathlib import Path
import numpy as np
import ml_dtypes

# Exact finite values with explicit storage encodings, including HIFLOAT8.
CODES = {
    "int8": ([1, 2, -1, -2], [1, 2, -1, -2], np.int8),
    "fp8_e4m3fn": ([0x38, 0x30, 0xB8, 0xB0], [1, 0.5, -1, -0.5], np.uint8),
    "fp8_e5m2": ([0x3C, 0x38, 0xBC, 0xB8], [1, 0.5, -1, -0.5], np.uint8),
    "hifloat8": ([0x08, 0x18, 0x88, 0x98], [1, 0.5, -1, -0.5], np.uint8),
    "fp4_e2m1": ([2, 1, 10, 9], [1, 0.5, -1, -0.5], np.uint8),
}


def pack(values):
    a = np.asarray(values, np.uint8).reshape(-1)
    if a.size % 2:
        raise ValueError("FP4 requires an even element count")
    return a[::2] | (a[1::2] << 4)


def generate(a):
    if a.quant_mode != "mx" and a.weight_format != "NZ":
        raise ValueError("pertoken/A8W4 examples require WeightNZ")
    a.mode = (
        ("mx_nd" if a.weight_format == "ND" else "mx_nz")
        if a.quant_mode == "mx"
        else a.quant_mode
    )
    bf16 = a.output_dtype == "bfloat16"
    nz_types = a.mode == "mx_nz" and (
        (a.dtype == "fp4_e2m1" and bf16 and a.logit_dtype == "bfloat16")
        or (a.dtype == "fp8_e4m3fn" and a.logit_dtype == "float32")
    )
    if (a.mode == "mx_nz" and not nz_types) or (
        a.mode != "mx_nz" and (bf16 or a.logit_dtype != "float32")
    ):
        raise ValueError("unsupported output/logit type for selected quantization mode")
    if (
        min(a.e, a.m, a.k, a.n, a.batch) <= 0
        or a.k % 64
        or a.n % 64
        or a.shared_len < 0
        or a.shared_offset < 0
        or a.shared_offset + a.shared_len > a.batch
    ):
        raise ValueError(
            "positive dimensions, K/N multiples of 64 and a valid shared-input interval are required"
        )
    allowed = {
        "pertoken": {"int8", "fp8_e4m3fn", "hifloat8"},
        "mx_nd": {"fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1"},
        "mx_nz": {"fp8_e4m3fn", "fp4_e2m1"},
        "a8w4": {"fp8_e4m3fn"},
    }
    if a.dtype not in allowed[a.mode]:
        raise ValueError("unsupported mode/dtype pair")
    counts = np.full(a.e, a.m // a.e, np.int64)
    counts[-1] += a.m - int(counts.sum())
    if a.empty_group and a.e > 1:
        counts[0] += counts[1]
        counts[1] = 0
    groups = np.cumsum(counts) if a.group_type == 0 else counts
    rng = np.random.default_rng(a.seed)
    codes, values, storage = CODES[a.dtype]
    xi = rng.integers(0, 4, (a.m, a.k))
    x = np.asarray(values, np.float32)[xi]
    xc = np.asarray(codes, storage)[xi]
    w_dtype = "fp4_e2m1" if a.mode == "a8w4" else a.dtype
    wc, wv, wt = CODES[w_dtype]
    wi = rng.integers(0, 4, (a.e, a.k, a.n))
    w = np.asarray(wv, np.float32)[wi]
    raw_w = np.asarray(wc, wt)[wi]
    if a.mode == "mx_nz":
        # Non-transposed WeightNZ: [E,N1,K1,16,N0], N0=32 (FP8) or 64 (FP4).
        n0 = 64 if w_dtype == "fp4_e2m1" else 32
        raw_w = raw_w.reshape(a.e, a.k // 16, 16, a.n // n0, n0).transpose(
            0, 3, 1, 2, 4
        )
    elif a.mode != "mx_nd":
        # Transposed WeightNZ: [E,K1,N1,16,32], K0 is 32 for INT8/FP8 and A8W4.
        raw_w = (
            raw_w.transpose(0, 2, 1)
            .reshape(a.e, a.n // 16, 16, a.k // 32, 32)
            .transpose(0, 3, 1, 2, 4)
        )
    if w_dtype == "fp4_e2m1":
        raw_w = pack(raw_w)
    if a.dtype == "fp4_e2m1":
        xc = pack(xc)
    bias = (
        np.where(np.arange(a.e * a.n) % 2, 0.25, -0.25).reshape(a.e, a.n)
        if a.bias
        else np.zeros((a.e, a.n))
    ).astype(ml_dtypes.bfloat16)
    shared = (
        np.where(np.arange(max(1, a.shared_len) * a.n) % 3, 0.5, -0.5).reshape(
            max(1, a.shared_len), a.n
        )
    ).astype(ml_dtypes.bfloat16)
    logit_type = ml_dtypes.bfloat16 if a.logit_dtype == "bfloat16" else np.float32
    logits = np.where(np.arange(a.m) % 2, 0.5, 0.25).astype(logit_type)
    indices = ((np.arange(a.m) * 7 + 3) % a.batch).astype(
        np.int32 if bf16 else np.int64
    )
    if a.mode == "pertoken":
        sa = np.where(np.arange(a.m) % 2, 0.5, 1).astype(np.float32)
        sb = (
            np.where(np.arange(a.e * a.n) % 3, 0.5, 2)
            .reshape(a.e, a.n)
            .astype(np.float32)
        )
    else:
        sa = (
            np.where(np.arange(a.m * a.k // 32) % 2, 126, 127)
            .reshape(a.m, a.k // 32)
            .astype(np.uint8)
        )
        sb = (
            np.where(np.arange(a.e * a.n * a.k // 32) % 3, 127, 128)
            .reshape(a.e, a.n, a.k // 32)
            .astype(np.uint8)
        )
    golden = np.zeros((a.batch, a.n), ml_dtypes.bfloat16 if bf16 else np.float32)
    if a.shared_len:
        golden[a.shared_offset : a.shared_offset + a.shared_len] = (
            shared[: a.shared_len].astype(np.float32) * 0.5
        )
    start = 0
    for expert, count in enumerate(counts):
        end = start + int(count)
        if a.mode == "pertoken":
            acc = (x[start:end] @ w[expert]) * sb[expert]
            if a.token_scale:
                acc *= sa[start:end, None]
        else:
            acc = np.zeros((int(count), a.n), np.float32)
            for b, lo in enumerate(range(0, a.k, 32)):
                ascale = np.exp2(sa[start:end, b].astype(np.int32) - 127).astype(
                    np.float32
                )
                bscale = np.exp2(sb[expert, :, b].astype(np.int32) - 127).astype(
                    np.float32
                )
                acc += (x[start:end, lo : lo + 32] * ascale[:, None]) @ (
                    w[expert, lo : lo + 32] * bscale
                )
        if a.bias:
            acc += bias[expert].astype(np.float32)
        if bf16:
            # Match BF16 result/logit conversion, BF16 product and per-row BF16 atomic add.
            result = acc.astype(ml_dtypes.bfloat16).astype(np.float32)
            logit = logits[start:end].astype(ml_dtypes.bfloat16).astype(np.float32)
            contribution = (result * logit[:, None]).astype(ml_dtypes.bfloat16)
            for source, row in enumerate(indices[start:end]):
                golden[row] = (
                    golden[row].astype(np.float32)
                    + contribution[source].astype(np.float32)
                ).astype(ml_dtypes.bfloat16)
        else:
            np.add.at(
                golden,
                indices[start:end],
                acc * logits[start:end, None].astype(np.float32),
            )
        start = end
    scale_b = sb
    if a.mode in ("mx_nd", "mx_nz"):
        scale_b = sb.reshape(a.e, a.n, a.k // 64, 2).transpose(0, 2, 1, 3)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "x": xc,
        "weight": raw_w,
        "scale_b": scale_b,
        "scale_a": sa,
        "bias": bias,
        "groups": groups,
        "shared": shared,
        "logit": logits,
        "rows": indices,
        "golden_y": golden,
    }.items():
        np.ascontiguousarray(value).tofile(out / (name + ".bin"))
    print(f"Generated {a.mode}/{a.dtype}: E={a.e}, M={a.m}, K={a.k}, N={a.n}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quant-mode", choices=("pertoken", "mx", "a8w4"), required=True)
    p.add_argument("--weight-format", choices=("ND", "NZ"), required=True)
    p.add_argument("--output-dtype", choices=("float32", "bfloat16"), required=True)
    p.add_argument("--logit-dtype", choices=("float32", "bfloat16"), required=True)
    p.add_argument("--dtype", choices=tuple(CODES), required=True)
    for name in ("e", "m", "k", "n", "batch"):
        p.add_argument("--" + name, type=int, required=True)
    for name in ("bias", "token-scale", "group-type", "empty-group"):
        p.add_argument(
            "--" + name,
            type=int,
            choices=(0, 1),
            default=0 if name != "token-scale" else 1,
        )
    p.add_argument("--shared-len", type=int, default=0)
    p.add_argument("--shared-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
