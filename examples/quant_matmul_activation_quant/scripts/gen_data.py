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

"""Generate deterministic MX inputs and compute GELU/SwiGLU/Swish + MX-quant golden data.

A is MX-quantized FP8 (fp8_e4m3 / fp8_e5m2) or FP4 (fp4_e2m1). B supports ND/NZ layouts.
Swish (including SiLU) uses FP8 only and additionally supports fp8_e5m2 B in ND. FP4*FP8 mixed dtype is not supported.
C dtype is float (L0C float32 accumulator, DualDst to UB).
Output Y dtype follows A dtype; Y_scale is fp8_e8m0.
"""

import argparse
import os

import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2
FP8_E8M0 = ml_dtypes.float8_e8m0fnu
FP4_E2M1_TO_FP32 = (
    np.arange(16, dtype=np.int8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
)

GROUP_SIZE = 32
MXFP_DIVISOR_SIZE = 64
MXFP_MULTI_BASE_SIZE = 2

_DTYPE_MAP = {
    "fp8_e4m3": FP8_E4M3FN,
    "fp8_e5m2": FP8_E5M2,
    "fp4_e2m1": None,
}

_EMAX_MAP = {
    "fp8_e4m3": 8,
    "fp8_e5m2": 15,
    "fp4_e2m1": 2,
}

_DTYPE_MAX = {
    "fp8_e4m3": 448.0,
    "fp8_e5m2": 57344.0,
    "fp4_e2m1": 6.0,
}

_C0_SIZE = {"fp8_e4m3": 32, "fp8_e5m2": 32, "fp4_e2m1": 64}


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
# Scale helper
# ---------------------------------------------------------------------------


def make_scale(shape):
    exponents = np.indices(shape, dtype=np.int64).sum(axis=0) % 4 - 1
    values = np.exp2(exponents.astype(np.float32))
    return values.astype(FP8_E8M0)


# ---------------------------------------------------------------------------
# Quantized data generation for FP8
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
# Scale formatting (matmul input scales, with C0=2 interleave)
# ---------------------------------------------------------------------------


def format_scale_a(scale_a, m):
    return scale_a


def format_scale_a_trans(scale_a, m):
    scale_k = scale_a.shape[1]
    scale_k_aligned = align(scale_k, 2)
    sa = np.zeros((m, scale_k_aligned), dtype=FP8_E8M0)
    sa[:m, :scale_k] = scale_a
    return sa.T.reshape(scale_k_aligned // 2, 2, m).transpose(0, 2, 1).reshape(-1)


def format_scale_b(scale_b, n):
    scale_k = scale_b.shape[0]
    scale_k_aligned = align(scale_k, 2)
    sb = np.zeros((scale_k_aligned, n), dtype=FP8_E8M0)
    sb[:scale_k, :] = scale_b
    return sb.reshape(scale_k_aligned // 2, 2, n).transpose(0, 2, 1).reshape(-1)


def format_scale_b_trans(scale_b, n):
    return scale_b.T.copy()


# ---------------------------------------------------------------------------
# NZ conversion (when the requested weight layout is NZ)
# ---------------------------------------------------------------------------


def _convert_to_nz(data_nd, dtype, k, n, trans_b):
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
# Activation + dynamic MX quantization (golden computation)
# ---------------------------------------------------------------------------


def _ftz_float32(values):
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        np.abs(values) < np.finfo(np.float32).tiny,
        np.copysign(np.float32(0.0), values),
        values,
    )


def gelu_tanh(x):
    """Follow the kernel's FP32 sigmoid form, preserving the negative tail."""
    values = np.asarray(x, dtype=np.float32)
    tanh_approx_factor = np.float32(1.0 / 0.044715)
    neg_sqrt_eight_over_pi = np.float32(-1.595769121 * 0.044715)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        squared = values * values
        cubed = squared * values
        exponent = (cubed + values * tanh_approx_factor) * neg_sqrt_eight_over_pi
        exp_neg = _ftz_float32(np.exp(_ftz_float32(exponent)))
        return _ftz_float32(_ftz_float32(values) / (np.float32(1.0) + exp_neg))


def swiglu(x):
    gate, linear = np.split(x, 2, axis=-1)
    with np.errstate(over="ignore"):
        return gate / (1.0 + np.exp(-gate)) * linear


def swish(x, beta=1.0):
    """Compute Swish in FP32 and flush subnormal results to signed zero."""
    values = np.asarray(x, dtype=np.float32)
    neg_beta = -np.float32(beta)
    with np.errstate(over="ignore"):
        result = values / (np.float32(1.0) + np.exp(values * neg_beta))
    return np.where(
        np.abs(result) < np.finfo(np.float32).tiny,
        np.copysign(np.float32(0.0), result),
        result,
    )


def mx_quantize(data, emax, dtype_max, scale_alg):
    """Quantize the last axis in 32-element blocks using OCP(0) or BLAS(1) scale."""
    values = np.asarray(data, dtype=np.float32)
    width = values.shape[-1]
    rows = values.reshape(-1, width)
    block_count = (width + GROUP_SIZE - 1) // GROUP_SIZE
    padded_width = block_count * GROUP_SIZE
    padded = np.zeros((rows.shape[0], padded_width), dtype=np.float32)
    padded[:, :width] = rows
    blocks = padded.reshape(rows.shape[0], block_count, GROUP_SIZE)

    abs_max = np.max(np.abs(blocks), axis=-1).astype(np.float64)
    mantissa, exponent = np.frexp(abs_max)
    exponent -= 1 + emax
    if scale_alg == 1:
        # FP8 E4M3/E5M2 qmax is 1.75 * 2**emax.
        exponent += mantissa * 2 > 1.75
    scale_codes = np.clip(exponent + 127, 0, 254).astype(np.uint8)
    scale_codes[abs_max == 0.0] = 0

    normalized = np.ldexp(
        blocks.astype(np.float64), (127 - scale_codes.astype(np.int32))[..., None]
    )
    if scale_alg == 0:
        # OCP clears the reciprocal when sharedExp == 0.
        normalized = np.where(
            scale_codes[..., None] == 0, np.copysign(0.0, blocks), normalized
        )
    normalized = np.clip(normalized, -dtype_max, dtype_max)
    quantized = normalized.reshape(rows.shape[0], padded_width)[:, :width]
    return quantized.reshape(values.shape), scale_codes.view(FP8_E8M0)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _write_bin(path, arr):
    if hasattr(arr, "view") and arr.dtype != np.uint8:
        arr.view(np.uint8).tofile(path)
    else:
        arr.tofile(path)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(args):
    if not np.isfinite(args.beta) or abs(args.beta) > np.finfo(np.float32).max:
        raise ValueError("beta must be a finite FP32 scalar")
    beta = np.float32(args.beta)
    if args.activation != "swish" and beta != np.float32(1.0):
        raise ValueError("beta can differ from 1 only for swish")
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("M/N/K must be positive")
    if args.activation not in ("silu", "swish") and args.k % 8 != 0:
        raise ValueError("GELU/SwiGLU examples require K to be a multiple of 8")
    if args.bias not in (0, args.n):
        raise ValueError("bias must be 0 or N")
    if args.b_dtype == "fp8_e5m2" and args.activation not in ("silu", "swish"):
        raise ValueError("This example enables E5M2 B only for Swish/SiLU")
    if args.activation in ("silu", "swish"):
        if args.a_dtype == "fp4_e2m1" or args.b_dtype == "fp4_e2m1":
            raise ValueError("Swish/SiLU requires FP8 inputs")
        if args.format == "(ND,NZ)" and (args.trans_a or args.b_dtype != "fp8_e4m3"):
            raise ValueError("Swish/SiLU NZ requires transA=false and E4M3 B")
    if args.activation == "swiglu":
        if (
            args.n % 64 != 0
            or args.trans_a
            or args.a_dtype == "fp4_e2m1"
            or args.b_dtype == "fp4_e2m1"
        ):
            raise ValueError(
                "SwiGLU requires N multiple of 64, transA=false, and FP8 inputs"
            )
    if args.scale_alg == 1 and args.a_dtype == "fp4_e2m1":
        raise ValueError("BLAS scale is only enabled for FP8 output in this example")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)

    scale_k = align(args.k, MXFP_DIVISOR_SIZE) // GROUP_SIZE
    output_n = args.n // 2 if args.activation == "swiglu" else args.n
    scale_n = align(output_n, MXFP_DIVISOR_SIZE) // GROUP_SIZE
    rng = np.random.default_rng(20260727)

    a_raw, a_fp32 = generate_quantized_data((args.m, args.k), args.a_dtype, rng)
    b_raw, b_fp32 = generate_quantized_data((args.k, args.n), args.b_dtype, rng)
    scale_a = make_scale((args.m, scale_k))
    scale_b = make_scale((scale_k, args.n))

    sa_b = np.repeat(scale_a.astype(np.float32), GROUP_SIZE, axis=1)[:, : args.k]
    sb_b = np.repeat(scale_b.astype(np.float32), GROUP_SIZE, axis=0)[: args.k, :]
    matmul_result = np.matmul(a_fp32 * sa_b, b_fp32 * sb_b)

    bias_np = np.zeros(args.n, dtype=np.float32)
    if args.bias:
        bias_np = np.where(np.arange(args.n) % 2 == 0, 0.5, -1.0).astype(np.float32)
    matmul_with_bias = matmul_result + bias_np

    # C dtype = float (L0C float32 accumulator, DualDst fixpipe to UB as float).
    # The epilogue performs activation in FP32 and rounds its result to BF16 before MX quantization.
    if args.activation == "swiglu":
        activation_result = swiglu(matmul_with_bias).astype(bfloat16)
    elif args.activation in ("silu", "swish"):
        activation_result = swish(matmul_with_bias, beta).astype(bfloat16)
    else:
        activation_result = gelu_tanh(matmul_with_bias).astype(bfloat16)

    # Dynamic MX quantization -> Y (out dtype = a_dtype) + Y_scale (e8m0)
    emax = _EMAX_MAP[args.a_dtype]
    dtype_max = _DTYPE_MAX[args.a_dtype]
    quantized_fp32, y_scale_flat = mx_quantize(
        activation_result.astype(np.float32),
        emax,
        dtype_max,
        args.scale_alg,
    )

    if args.a_dtype == "fp4_e2m1":
        codes = quantized_fp32.astype(ml_dtypes.float4_e2m1fn).view(np.uint8)
        if codes.size % 2 != 0:
            padded = np.zeros(codes.size + 1, dtype=np.uint8)
            padded[: codes.size] = codes
            codes = padded
        golden_y = pack_fp4(codes.reshape(-1))
    else:
        golden_y = quantized_fp32.astype(_DTYPE_MAP[args.a_dtype])
    y_scale_per_row = y_scale_flat.reshape(args.m, -1)
    golden_y_scale = np.zeros((args.m, scale_n), dtype=FP8_E8M0)
    golden_y_scale[:, : y_scale_per_row.shape[1]] = y_scale_per_row

    d = args.output_dir
    a_write = (
        _transpose_quantized(a_raw, args.a_dtype, args.m, args.k)
        if args.trans_a
        else a_raw
    )
    _write_bin(os.path.join(d, "input_a.bin"), a_write)
    if args.format == "(ND,NZ)":
        _write_bin(
            os.path.join(d, "input_b.bin"),
            _convert_to_nz(b_raw, args.b_dtype, args.k, args.n, args.trans_b),
        )
    else:
        b_write = (
            _transpose_quantized(b_raw, args.b_dtype, args.k, args.n)
            if args.trans_b
            else b_raw
        )
        _write_bin(os.path.join(d, "input_b.bin"), b_write)

    sa_fmt = (
        format_scale_a_trans(scale_a, args.m)
        if args.trans_a
        else format_scale_a(scale_a, args.m)
    )
    sb_fmt = (
        format_scale_b_trans(scale_b, args.n)
        if args.trans_b
        else format_scale_b(scale_b, args.n)
    )
    _write_bin(os.path.join(d, "scale_a.bin"), sa_fmt)
    _write_bin(os.path.join(d, "scale_b.bin"), sb_fmt)

    bias_out = bias_np if args.bias else np.zeros(1, dtype=np.float32)
    bias_out.tofile(os.path.join(d, "bias.bin"))

    golden_y.tofile(os.path.join(d, "golden_y.bin"))
    golden_y_scale.view(np.uint8).tofile(os.path.join(d, "golden_y_scale.bin"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate quant_matmul_activation_quant input and golden data."
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
    parser.add_argument("--trans-a", action="store_true", default=False)
    parser.add_argument("--trans-b", action="store_true", default=False)
    parser.add_argument("--format", default="(ND,NZ)", choices=("(ND,ND)", "(ND,NZ)"))
    parser.add_argument(
        "--activation", default="gelu", choices=("gelu", "swiglu", "silu", "swish")
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--scale-alg", type=int, default=0, choices=(0, 1))
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
