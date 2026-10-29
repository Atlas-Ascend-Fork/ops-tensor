#!/usr/bin/python3
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

"""Generate deterministic MX inputs and a NumPy golden result for quant_batch_matmul_mx."""

import argparse
import os
from dataclasses import dataclass

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2
FP8_E8M0 = ml_dtypes.float8_e8m0fnu
FP4_E2M1_TO_FP32 = (
    np.arange(16, dtype=np.int8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
)

GROUP_SIZE = 32

_DTYPE_MAP = {
    "fp8_e4m3": FP8_E4M3FN,
    "fp8_e5m2": FP8_E5M2,
    "fp4_e2m1": None,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# FP4 helpers
# ---------------------------------------------------------------------------


def pack_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2 != 0:
        raise ValueError("FP4 packing axis must have an even length")
    low = np.bitwise_and(codes[..., 0::2], 0x0F)
    high = np.bitwise_and(codes[..., 1::2], 0x0F)
    return np.bitwise_or(low, np.left_shift(high, 4)).astype(np.uint8)


def unpack_fp4(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = np.bitwise_and(packed, 0x0F)
    codes[..., 1::2] = np.right_shift(packed, 4)
    return codes


# ---------------------------------------------------------------------------
# Common scale helper
# ---------------------------------------------------------------------------


def make_scale(shape):
    """Create valid E8M0 values with ml_dtypes."""
    exponents = np.indices(shape, dtype=np.int64).sum(axis=0) % 4 - 1
    values = np.exp2(exponents.astype(np.float32))
    return values.astype(FP8_E8M0)


# ---------------------------------------------------------------------------
# Quantized data generation for FP8 / FP4
# ---------------------------------------------------------------------------


def generate_quantized_data(shape, dtype, rng):
    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        np_dtype = _DTYPE_MAP[dtype]
        values = rng.choice(
            np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32), size=shape
        )
        return values.astype(np_dtype), values.astype(np.float32)
    if dtype == "fp4_e2m1":
        codes = rng.integers(0, 16, size=shape, dtype=np.uint8)
        fp32_values = FP4_E2M1_TO_FP32[codes]
        if len(shape) == 2:
            rows, cols = shape
            padded = np.zeros((rows, cols + (cols % 2)), dtype=np.uint8)
            padded[:, :cols] = codes
            packed = pack_fp4(padded.reshape(rows, -1))
        else:
            packed = pack_fp4(codes.reshape(-1))
        return packed, fp32_values
    raise ValueError(f"Unsupported quantized dtype: {dtype}")


def _transpose_quantized(raw, dtype, rows, cols):
    """Transpose a quantized data array from logical (rows, cols) to physical (cols, rows)."""
    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        return raw.reshape(rows, cols).T.copy()
    if dtype == "fp4_e2m1":
        packed = np.asarray(raw, dtype=np.uint8)
        half_cols = (cols + 1) // 2
        codes = unpack_fp4(packed.reshape(rows, half_cols))
        codes = codes[:, :cols]
        codes_t = codes.T.copy()
        return pack_fp4(codes_t.reshape(-1))
    raise ValueError(f"Unsupported dtype for transpose: {dtype}")


# ---------------------------------------------------------------------------
# Scale formatting
# ---------------------------------------------------------------------------


def format_scale_a(scale_a, m):
    """Format ScaleA for scalea_nd_layout_ptn (no transA): (M, sk) row-major."""
    return scale_a


def format_scale_a_trans(scale_a, m):
    """Format ScaleA for scalea_dn_layout_ptn (transA=true)."""
    scale_k = scale_a.shape[1]
    scale_k_aligned = align(scale_k, 2)
    sa = np.zeros((m, scale_k_aligned), dtype=FP8_E8M0)
    sa[:m, :scale_k] = scale_a
    return sa.T.reshape(scale_k_aligned // 2, 2, m).transpose(0, 2, 1).reshape(-1)


def format_scale_b(scale_b, n):
    """Format ScaleB for scaleb_nd_layout_ptn (no transB): C0=2 interleave along N."""
    scale_k = scale_b.shape[0]
    scale_k_aligned = align(scale_k, 2)
    sb = np.zeros((scale_k_aligned, n), dtype=FP8_E8M0)
    sb[:scale_k, :] = scale_b
    return sb.reshape(scale_k_aligned // 2, 2, n).transpose(0, 2, 1).reshape(-1)


def format_scale_b_trans(scale_b, n):
    """Format ScaleB for scaleb_dn_layout_ptn (transB=true): (N, sk) row-major = transpose."""
    return scale_b.T.copy()


# ---------------------------------------------------------------------------
# NZ conversion
# ---------------------------------------------------------------------------

_C0_SIZE = {"fp8_e4m3": 32, "fp8_e5m2": 32, "fp4_e2m1": 64}


def _convert_to_nz(data_nd, dtype, k, n, trans_b):
    """Convert logical (K,N) ND data to NZ physical layout for GM."""
    c0 = _C0_SIZE[dtype]
    is_fp4 = dtype == "fp4_e2m1"

    if is_fp4:
        codes = unpack_fp4(data_nd.view(np.uint8).reshape(k, (n + 1) // 2))[..., :n]
        mat = codes.astype(np.uint8)
    else:
        mat = data_nd.view(np.uint8).reshape(k, n)

    k_padded = align(k, 16 if not trans_b else c0)
    n_padded = align(n, c0 if not trans_b else 16)
    padded = np.zeros((max(k, k_padded), max(n, n_padded)), dtype=np.uint8)
    padded[:k, :n] = mat

    if not trans_b:
        num_k_tiles = align(k, 16) // 16
        num_n_tiles = align(n, c0) // c0
        out = np.zeros(num_n_tiles * num_k_tiles * 16 * c0, dtype=np.uint8)
        idx = 0
        for nt in range(num_n_tiles):
            for kt in range(num_k_tiles):
                for r in range(16):
                    for c in range(c0):
                        out[idx] = padded[kt * 16 + r, nt * c0 + c]
                        idx += 1
    else:
        num_k_tiles = align(k, c0) // c0
        num_n_tiles = align(n, 16) // 16
        out = np.zeros(num_k_tiles * num_n_tiles * c0 * 16, dtype=np.uint8)
        idx = 0
        for kt in range(num_k_tiles):
            for nt in range(num_n_tiles):
                for c in range(16):
                    for r in range(c0):
                        out[idx] = padded[kt * c0 + r, nt * 16 + c]
                        idx += 1

    if is_fp4:
        return pack_fp4(out.reshape(-1))
    return out


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@dataclass
class MxGeneratedData:
    """Container for generated MX data artifacts."""

    a_raw: np.ndarray
    b_raw: np.ndarray
    scale_a: np.ndarray
    scale_b: np.ndarray
    bias_np: np.ndarray
    golden: np.ndarray


def _prepare_data(args):
    """Generate quantized data, scales, and golden result."""
    scale_k = align(args.k, 64) // GROUP_SIZE
    rng = np.random.default_rng(20260727)
    a_raw, a_fp32 = generate_quantized_data((args.m, args.k), args.a_dtype, rng)
    b_raw, b_fp32 = generate_quantized_data((args.k, args.n), args.b_dtype, rng)
    scale_a = make_scale((args.m, scale_k))
    scale_b = make_scale((scale_k, args.n))
    sa_b = np.repeat(scale_a.astype(np.float32), GROUP_SIZE, axis=1)[:, : args.k]
    sb_b = np.repeat(scale_b.astype(np.float32), GROUP_SIZE, axis=0)[: args.k, :]
    golden = np.matmul(a_fp32 * sa_b, b_fp32 * sb_b)
    bias_np = np.zeros(args.n, dtype=np.float32)
    if args.bias:
        bias_np = np.where(np.arange(args.n) % 2 == 0, 0.5, -1.0).astype(np.float32)
    golden = (golden + bias_np).astype(_DTYPE_MAP[args.c_dtype])
    return MxGeneratedData(a_raw, b_raw, scale_a, scale_b, bias_np, golden)


def _write_bin(path, arr):
    """Write an array to a .bin file, using view(np.uint8) for typed dtypes."""
    if hasattr(arr, "view"):
        arr.view(np.uint8).tofile(path)
    else:
        arr.tofile(path)


def _join_path(d, name):
    return os.path.join(d, name)


def _write_data(args, data):
    """Write all .bin artifacts to output_dir."""
    d = args.output_dir

    a_write = (
        _transpose_quantized(data.a_raw, args.a_dtype, args.m, args.k)
        if args.trans_a
        else data.a_raw
    )
    _write_bin(_join_path(d, "input_a.bin"), a_write)

    if args.format == "(ND,NZ)":
        _write_bin(
            _join_path(d, "input_b.bin"),
            _convert_to_nz(data.b_raw, args.b_dtype, args.k, args.n, args.trans_b),
        )
    else:
        b_write = (
            _transpose_quantized(data.b_raw, args.b_dtype, args.k, args.n)
            if args.trans_b
            else data.b_raw
        )
        _write_bin(_join_path(d, "input_b.bin"), b_write)

    sa_fmt = (
        format_scale_a_trans(data.scale_a, args.m)
        if args.trans_a
        else format_scale_a(data.scale_a, args.m)
    )
    sb_fmt = (
        format_scale_b_trans(data.scale_b, args.n)
        if args.trans_b
        else format_scale_b(data.scale_b, args.n)
    )
    _write_bin(_join_path(d, "scale_a.bin"), sa_fmt)
    _write_bin(_join_path(d, "scale_b.bin"), sb_fmt)

    bias_out = data.bias_np if args.bias else np.zeros(1, dtype=np.float32)
    bias_out.tofile(_join_path(d, "bias.bin"))
    np.zeros((args.m, args.n), dtype=_DTYPE_MAP[args.c_dtype]).tofile(
        _join_path(d, "initial_c.bin")
    )
    data.golden.tofile(_join_path(d, "golden_c.bin"))


def generate(args):
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("M/N must be positive")
    if args.bias not in (0, args.n):
        raise ValueError("bias must be 0 or N")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)
    data = _prepare_data(args)
    _write_data(args, data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate quant_batch_matmul_mx input and golden data."
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--a-dtype", required=True, choices=("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")
    )
    parser.add_argument(
        "--b-dtype", required=True, choices=("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")
    )
    parser.add_argument(
        "--c-dtype", required=True, choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--trans-a", action="store_true", default=False)
    parser.add_argument("--trans-b", action="store_true", default=False)
    parser.add_argument("--format", default="(ND,ND)", choices=("(ND,ND)", "(ND,NZ)"))

    generate(parser.parse_args())


if __name__ == "__main__":
    main()
