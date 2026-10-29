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

"""Generate deterministic inputs and the int8 golden for the per-group A8W4 example.

Kernel contract (T-CG per-group weight dequant matmul, arch35):
    y = (x1 @ (x2 * x2Scale)) * yScale
    - x1      : FP8 E4M3FN, logical (m, k), physical row-major (m, k)
    - x2      : packed FP4 E2M1, logical (k, n)
                nd variant: transposed ND, physical (n, k) row-major, nibbles packed along k
                nz variant: NZ fractal ((16, k1), (32, n1)), nibbles packed along n
    - x2Scale : BF16 per-group scale, logical (k/group, n)
                nd variant: physical (n, k/group) row-major (dn_ext, follows transpose_x2)
                nz variant: physical (k/group, n) row-major (nd_ext)
    - yScale  : UINT64 per-channel scale packed as (fp32 bits & 0xFFFFE000) | (1 << 46)
    - y       : INT8, logical (m, n), physical row-major

Golden follows the nn-repo T_CG reference: the dequantized weight is rounded back to FP8
(simulating the AIV vector output) before the FP32 matmul, and the yScale decode keeps only
the fixpipe-truncated FP32 mantissa.
"""

import argparse
import os

import numpy as np


FP4_E2M1_TABLE = np.array(
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
)

# Exact FP8 E4M3FN bit patterns for the activation value set.
X1_VALUES = [
    0x00,
    0x30,
    0x38,
    0x40,
    0x80,
    0xB0,
    0xB8,
    0xC0,
]  # 0, .5, 1, 2, -0, -.5, -1, -2
X1_LOOKUP = {
    0x00: 0.0,
    0x30: 0.5,
    0x38: 1.0,
    0x40: 2.0,
    0x80: 0.0,
    0xB0: -0.5,
    0xB8: -1.0,
    0xC0: -2.0,
}
# Exact BF16 bit patterns for the scale value set (powers of two).
SCALE_VALUES = [0x3E80, 0x3F00, 0x3F80]  # 0.25, 0.5, 1.0
# Exact FP32 bit patterns for the yScale value set (powers of two, fixpipe-truncation safe).
Y_SCALE_VALUES = [0x3E800000, 0x3F000000, 0x3F800000, 0x40000000]  # 0.25, 0.5, 1.0, 2.0

DEQ_SCALE_MASK = np.uint32(0xFFFFE000)
U64_SCALE_FLAG_BIT = 1 << 46


def bf16_bits_to_f32(bits_u16):
    return (bits_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def u64_to_deq_scale(u64_scale):
    shape = u64_scale.shape
    deq_u32 = u64_scale.astype(np.uint32).copy()
    deq_u32 &= DEQ_SCALE_MASK
    return deq_u32.view(np.float32).reshape(shape)


def pack_u64_scale(fp32_bits):
    u64 = fp32_bits.astype(np.uint64)
    u64 |= np.uint64(U64_SCALE_FLAG_BIT)
    return u64


def pack_fp4_nd(nibbles_kn):
    """(k, n) nibbles -> physical (n, k) row-major packed bytes (pack along k)."""
    k_len, n_len = nibbles_kn.shape
    if k_len % 2 != 0:
        raise ValueError("ND packed-FP4 K must be even")
    physical = nibbles_kn.T.copy()  # (n, k)
    physical = physical.reshape(n_len, k_len // 2, 2)
    packed = (physical[:, :, 0] & 0xF) | ((physical[:, :, 1] & 0xF) << 4)
    return packed.astype(np.uint8)


def pack_fp4_nz(nibbles_kn):
    """(k, n) nibbles -> NZ fractal bytes ((16, k1), (32, n1)), pack along n.

    Physical order per fractal: 16 K rows, each holding 32 N nibbles packed into 16 bytes.
    """
    k_len, n_len = nibbles_kn.shape
    k1 = (k_len + 15) // 16
    n1 = (n_len + 31) // 32
    padded = np.zeros((k1 * 16, n1 * 32), dtype=np.uint8)
    padded[:k_len, :n_len] = nibbles_kn
    # (k1, 16, n1, 32) -> pack the inner 32 nibbles along N into 16 bytes
    blocks = padded.reshape(k1, 16, n1, 32)
    packed = np.zeros((k1, 16, n1, 16), dtype=np.uint8)
    for j in range(16):
        packed[:, :, :, j] = (blocks[:, :, :, 2 * j] & 0xF) | (
            (blocks[:, :, :, 2 * j + 1] & 0xF) << 4
        )
    # flatten in (n1, k1, 16k, 16byte) order
    out = np.zeros((k1 * 16 * n1 * 16), dtype=np.uint8)
    idx = 0
    for n1_idx in range(n1):
        for k1_idx in range(k1):
            out[idx : idx + 16 * 16] = packed[k1_idx, :, n1_idx, :].reshape(-1)
            idx += 16 * 16
    return out


def generate(args):
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M, K, and N must be positive")
    if args.k % 32 != 0:
        raise ValueError("K must be a multiple of 32 (per-group constraint)")
    if args.k % 2 != 0:
        raise ValueError("K must be even (packed FP4)")
    if args.variant == "nz" and args.n % 32 != 0:
        raise ValueError("The nz variant requires n to be a multiple of 32")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)

    rng = np.random.default_rng(20260920)
    k_group = args.k // 32

    x1_bytes = np.array(X1_VALUES, dtype=np.uint8)[
        rng.integers(0, len(X1_VALUES), size=(args.m, args.k))
    ]
    x1_f = np.vectorize(X1_LOOKUP.get)(x1_bytes).astype(np.float32)

    x2_nibbles = rng.integers(0, 7, size=(args.k, args.n)).astype(np.uint8)
    x2_nibbles = np.where(
        rng.random((args.k, args.n)) < 0.5, x2_nibbles, x2_nibbles + 8
    ).astype(np.uint8)

    scale_bits = np.array(SCALE_VALUES, dtype=np.uint16)[
        rng.integers(0, len(SCALE_VALUES), size=(k_group, args.n))
    ]
    scale_f = bf16_bits_to_f32(scale_bits)

    y_scale_bits = np.array(Y_SCALE_VALUES, dtype=np.uint32)[
        rng.integers(0, len(Y_SCALE_VALUES), size=(1, args.n))
    ]
    y_scale = pack_u64_scale(y_scale_bits)

    # Golden: per-group scale broadcast along K, BF16-domain multiply (kernel casts F4 to
    # the scale type before Mul), FP32 matmul, per-channel deq scale multiply, RINT +
    # saturating int8 cast. The nibble/bf16/power-of-two value sets are exact in both BF16
    # and FP8 E4M3FN, so the AIV cast chain is lossless on this data.
    x2_cvt = FP4_E2M1_TABLE[x2_nibbles].astype(np.float16)
    scale_br = np.repeat(scale_f.astype(np.float16), 32, axis=0)[: args.k, :]
    x2_deq = (x2_cvt * scale_br).astype(np.float32)
    accum = np.matmul(x1_f, x2_deq)
    deq_scale = u64_to_deq_scale(y_scale)
    out = accum * deq_scale
    golden = np.clip(np.rint(out), -128, 127).astype(np.int8)

    x1_bytes.tofile(os.path.join(args.output_dir, "input_a.bin"))
    if args.variant == "nz":
        pack_fp4_nz(x2_nibbles).tofile(os.path.join(args.output_dir, "input_b.bin"))
        # nd_ext scale: physical (k_group, n) row-major BF16 bit patterns
        scale_bits.tofile(os.path.join(args.output_dir, "scale_b.bin"))
    else:
        pack_fp4_nd(x2_nibbles).tofile(os.path.join(args.output_dir, "input_b.bin"))
        # dn_ext scale: logical (k_group, n) stored transposed as physical (n, k_group) BF16
        scale_bits.T.copy().tofile(os.path.join(args.output_dir, "scale_b.bin"))
    y_scale.tofile(os.path.join(args.output_dir, "y_scale.bin"))
    golden.tofile(os.path.join(args.output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("nd", "nz"))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
