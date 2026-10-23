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

import argparse
import os

import numpy as np


def cast_data(data, dtype):
    if dtype == "float16":
        storage = data.astype(np.float16)
        return storage, storage.astype(np.float32)
    # NumPy has no native BF16: round to nearest even and store the raw bits.
    bits = data.astype(np.float32).view(np.uint32).astype(np.uint64)
    storage = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return storage, (storage.astype(np.uint32) << 16).view(np.float32)


def gen_golden_data(m, k, n, trans_a, trans_b, dtype, bias_dtype):
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    rng = np.random.default_rng(42)
    a, a_fp32 = cast_data(rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32), dtype)
    b, b_fp32 = cast_data(rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32), dtype)
    accumulator = np.matmul(a_fp32, b_fp32)
    if bias_dtype != "none":
        bias_fp32 = np.linspace(-1.25, 1.25, n, dtype=np.float32)
        if bias_dtype == "float32":
            bias = bias_fp32
        else:
            bias, bias_fp32 = cast_data(bias_fp32, dtype)
        bias.tofile("./input/bias.bin")
        accumulator += bias_fp32

    # Apply sigmoid before the final FP16/BF16 conversion, matching the kernel.
    with np.errstate(over="ignore"):
        golden_fp32 = 1.0 / (1.0 + np.exp(-accumulator))
    golden, _ = cast_data(golden_fp32, dtype)
    (a.T.copy() if trans_a else a).tofile("./input/input_a.bin")
    (b.T.copy() if trans_b else b).tofile("./input/input_b.bin")
    golden.tofile("./output/cpu_output.bin")
    print("Data generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate FusedMatMul sigmoid data.")
    parser.add_argument("m", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("transA", choices=["true", "false"])
    parser.add_argument("transB", choices=["true", "false"])
    parser.add_argument("dtype", choices=["float16", "bfloat16"])
    parser.add_argument("bias_dtype", choices=["none", "same", "float32"])
    args = parser.parse_args()
    if min(args.m, args.k, args.n) <= 0:
        parser.error("M, K, N must be positive integers")
    gen_golden_data(
        args.m,
        args.k,
        args.n,
        args.transA == "true",
        args.transB == "true",
        args.dtype,
        args.bias_dtype,
    )


if __name__ == "__main__":
    main()
