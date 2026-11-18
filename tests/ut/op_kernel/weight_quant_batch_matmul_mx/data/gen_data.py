# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3
"""Generate deterministic MX inputs and a NumPy golden result for the kernel UT."""

import argparse
import os

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP4_VALUES = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)
FP4_CODES = np.array([0x1, 0x2, 0xA, 0x4], dtype=np.uint8)
GROUP_SIZE = 32
NZ_K0 = 32
NZ_N0 = 16
MX_EXPONENT_BIAS = 127


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def pack_fp4(codes):
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError("FP4 element count must be even")
    return flat[::2] | (flat[1::2] << 4)


def format_weight(codes, layout):
    if layout == "nd":
        return pack_fp4(codes.T)

    k, n = codes.shape
    k_aligned = align(k, NZ_K0)
    n_aligned = align(n, NZ_N0)
    padded = np.zeros((k_aligned, n_aligned), dtype=np.uint8)
    padded[:k, :n] = codes
    nz = padded.reshape(k_aligned // NZ_K0, NZ_K0, n_aligned // NZ_N0, NZ_N0)
    nz = nz.transpose(0, 2, 3, 1)
    return pack_fp4(nz)


def decode_e8m0(codes):
    return np.exp2(codes.astype(np.int16) - MX_EXPONENT_BIAS).astype(np.float32)


def format_scale_nz(codes, n):
    """(scale_k, n) E8M0 codes -> NZ storage (align(n,16)/16, scale_k, 16, 2) bytes.

    NZ scale packs pairs of consecutive k-group exponents: canonical view is
    (scale_k/2, n, 2); storage (n1/16, scale_k/2, 16, 2) with trailing (16, 2)
    = 16 n-values x {even k-group, odd k-group}.
    """
    scale_k = codes.shape[0]
    if scale_k % 2 != 0:
        raise ValueError("scale_k must be even for NZ scale pairs")
    n1 = align(n, 16)
    padded = np.full((scale_k, n1), MX_EXPONENT_BIAS, dtype=np.uint8)
    padded[:, :n] = codes
    pairs = np.empty((scale_k // 2, n1, 2), dtype=np.uint8)
    pairs[:, :, 0] = padded[0::2, :]
    pairs[:, :, 1] = padded[1::2, :]
    nz = pairs.reshape(scale_k // 2, n1 // 16, 16, 2).transpose(1, 0, 2, 3)
    return np.ascontiguousarray(nz)


def generate(problem_shape, weight_layout, with_bias, output_dir, scale_layout="nd"):
    m, n, k = problem_shape
    if min(m, n, k) <= 0 or k % 8 != 0:
        raise ValueError("M/N must be positive and K must be a multiple of 8")
    if weight_layout == "nz" and n % 8 != 0:
        raise ValueError("NZ N must be a multiple of 8")

    os.makedirs(output_dir, exist_ok=True)
    scale_k = align(k, 64) // GROUP_SIZE
    group_count = (k + GROUP_SIZE - 1) // GROUP_SIZE

    row_indices, k_indices = np.indices((m, k))
    x1 = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)[
        (row_indices + k_indices) % 4
    ]
    x1 = x1.astype(FP8_E4M3FN)

    weight_k_indices, n_indices = np.indices((k, n))
    weight_pattern = (2 * weight_k_indices + n_indices) % FP4_VALUES.size
    x2 = FP4_VALUES[weight_pattern]
    x2_codes = FP4_CODES[weight_pattern]

    scale_a_codes = np.full((m, scale_k), MX_EXPONENT_BIAS, dtype=np.uint8)
    scale_b_codes = np.full((scale_k, n), MX_EXPONENT_BIAS, dtype=np.uint8)
    scale_a_codes[:, :group_count] += np.arange(group_count, dtype=np.uint8) % 2
    scale_b_codes[:group_count, :] -= (
        np.arange(group_count, dtype=np.uint8)[:, None] % 2
    )

    scale_a = np.repeat(
        decode_e8m0(scale_a_codes[:, :group_count]), GROUP_SIZE, axis=1
    )[:, :k]
    scale_b = np.repeat(
        decode_e8m0(scale_b_codes[:group_count, :]), GROUP_SIZE, axis=0
    )[:k, :]
    x1_dequant = (x1.astype(np.float32) * scale_a).astype(FP8_E4M3FN)
    x2_dequant = (x2 * scale_b).astype(FP8_E4M3FN)

    bias = np.zeros(n, dtype=np.float16)
    if with_bias:
        bias = np.where(np.arange(n) % 2 == 0, 0.5, -1.0).astype(np.float16)
    golden = np.matmul(x1_dequant.astype(bias.dtype), x2_dequant.astype(bias.dtype))
    golden = (golden + bias).astype(bias.dtype)

    x1.view(np.uint8).tofile(os.path.join(output_dir, "input_a.bin"))
    format_weight(x2_codes, weight_layout).tofile(
        os.path.join(output_dir, "input_b.bin")
    )
    scale_a_codes.tofile(os.path.join(output_dir, "scale_a.bin"))
    if scale_layout == "nz":
        format_scale_nz(scale_b_codes, n).tofile(
            os.path.join(output_dir, "scale_b.bin")
        )
    else:
        scale_b_codes.T.tofile(os.path.join(output_dir, "scale_b.bin"))
    bias.tofile(os.path.join(output_dir, "bias.bin"))
    golden.tofile(os.path.join(output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Weight Quant MX kernel UT data"
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--weight-layout", choices=("nd", "nz"), required=True)
    parser.add_argument("--scale-layout", choices=("nd", "nz"), default="nd")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    generate(
        (args.m, args.n, args.k),
        args.weight_layout,
        args.bias,
        args.output_dir,
        args.scale_layout,
    )


if __name__ == "__main__":
    main()
