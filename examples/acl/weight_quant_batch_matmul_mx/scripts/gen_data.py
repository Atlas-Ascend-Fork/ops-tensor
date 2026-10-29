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

"""Generate deterministic MXA8W4 inputs and an FP16 golden result."""

import argparse
import os
from dataclasses import dataclass

import ml_dtypes
import numpy as np


FP8_E4M3FN = ml_dtypes.float8_e4m3fn
FP8_E8M0 = ml_dtypes.float8_e8m0fnu
FP4_E2M1_TO_FP32 = (
    np.arange(16, dtype=np.int8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
)

GROUP_SIZE = 32
NZ_K0 = 32
NZ_N0 = 16
RANDOM_SEED = 20260727


@dataclass(frozen=True)
class MxDataConfig:
    m: int
    k: int
    n: int
    bias: int
    layout: str
    output_dir: str


@dataclass(frozen=True)
class MxDataArtifacts:
    activation: np.ndarray
    packed_weight: np.ndarray
    scale_a: np.ndarray
    scale_b: np.ndarray
    bias: np.ndarray
    initial_c: np.ndarray
    golden_c: np.ndarray


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def validate_config(config):
    if min(config.m, config.n, config.k) <= 0:
        raise ValueError("M, N, and K must be positive")
    if config.k % 8 != 0:
        raise ValueError("K must be a multiple of 8")
    if config.layout == "nz" and config.n % 8 != 0:
        raise ValueError("NZ N must be a multiple of 8")
    if config.bias not in (0, config.n):
        raise ValueError("bias must be 0 or N")


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


def format_weight(codes, layout):
    if layout == "nd":
        # ND B is stored as logical (N, K), with adjacent K values packed together.
        return pack_fp4(codes.T).reshape(-1)

    k, n = codes.shape
    k_aligned = align(k, NZ_K0)
    n_aligned = align(n, NZ_N0)
    padded = np.zeros((k_aligned, n_aligned), dtype=np.uint8)
    padded[:k, :n] = codes
    nz = padded.reshape(k_aligned // NZ_K0, NZ_K0, n_aligned // NZ_N0, NZ_N0)
    nz = nz.transpose(0, 2, 3, 1)
    return pack_fp4(nz).reshape(-1)


def decode_weight(packed, layout, k, n):
    if layout == "nd":
        packed_nd = np.asarray(packed, dtype=np.uint8).reshape(n, (k + 1) // 2)
        return unpack_fp4(packed_nd)[..., :k].T

    k_aligned = align(k, NZ_K0)
    n_aligned = align(n, NZ_N0)
    packed_nz = np.asarray(packed, dtype=np.uint8).reshape(
        k_aligned // NZ_K0, n_aligned // NZ_N0, NZ_N0, NZ_K0 // 2
    )
    nz_codes = unpack_fp4(packed_nz)
    return nz_codes.transpose(0, 3, 1, 2).reshape(k_aligned, n_aligned)[:k, :n]


def make_scale(shape):
    exponents = np.indices(shape, dtype=np.int64).sum(axis=0) % 4 - 1
    values = np.exp2(exponents.astype(np.float32))
    return values.astype(FP8_E8M0)


def make_bias(n, bias_size):
    if bias_size == 0:
        return np.zeros(n, dtype=np.float16)
    return np.where(np.arange(n) % 2 == 0, 0.5, -1.0).astype(np.float16)


def generate_artifacts(config):
    scale_k = align(config.k, 64) // GROUP_SIZE
    rng = np.random.default_rng(RANDOM_SEED)

    activation_values = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)
    activation = rng.choice(activation_values, size=(config.m, config.k)).astype(
        FP8_E4M3FN
    )
    weight_codes = rng.integers(0, 16, size=(config.k, config.n), dtype=np.uint8)
    packed_weight = format_weight(weight_codes, config.layout)

    scale_a = make_scale((config.m, scale_k))
    scale_b = make_scale((scale_k, config.n))
    bias = make_bias(config.n, config.bias)

    scale_a_broadcast = np.repeat(scale_a.astype(np.float32), GROUP_SIZE, axis=1)[
        :, : config.k
    ]
    scale_b_broadcast = np.repeat(scale_b.astype(np.float32), GROUP_SIZE, axis=0)[
        : config.k, :
    ]
    decoded_weight = FP4_E2M1_TO_FP32[
        decode_weight(packed_weight, config.layout, config.k, config.n)
    ]

    activation_dequant = activation.astype(np.float32) * scale_a_broadcast
    weight_dequant = decoded_weight * scale_b_broadcast
    golden_c = (
        np.matmul(activation_dequant, weight_dequant) + bias.astype(np.float32)
    ).astype(np.float16)

    return MxDataArtifacts(
        activation=activation,
        packed_weight=packed_weight,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        initial_c=np.zeros((config.m, config.n), dtype=np.float16),
        golden_c=golden_c,
    )


def write_artifacts(output_dir, artifacts):
    os.makedirs(output_dir, exist_ok=True)
    artifacts.activation.view(np.uint8).tofile(os.path.join(output_dir, "input_a.bin"))
    artifacts.packed_weight.tofile(os.path.join(output_dir, "input_b.bin"))
    artifacts.scale_a.view(np.uint8).tofile(os.path.join(output_dir, "scale_a.bin"))
    # ScaleBDN uses an N-major GM stride while its logical shape is (scale_k, N).
    artifacts.scale_b.T.view(np.uint8).tofile(os.path.join(output_dir, "scale_b.bin"))
    artifacts.bias.tofile(os.path.join(output_dir, "bias.bin"))
    artifacts.initial_c.tofile(os.path.join(output_dir, "initial_c.bin"))
    artifacts.golden_c.tofile(os.path.join(output_dir, "golden_c.bin"))


def generate(config):
    validate_config(config)
    artifacts = generate_artifacts(config)
    write_artifacts(config.output_dir, artifacts)
    print(f"[SUCCESS] Test data generated in {config.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MXA8W4 input and golden data."
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--bias", type=int, default=0)
    parser.add_argument("--layout", type=str.lower, choices=("nd", "nz"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return MxDataConfig(
        m=args.m,
        k=args.k,
        n=args.n,
        bias=args.bias,
        layout=args.layout,
        output_dir=args.output_dir,
    )


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
