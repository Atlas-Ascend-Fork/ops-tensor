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


def gen_golden_elemwise(m, k, n, trans_a, trans_b, dtype, op_type, inner_precise):
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    rng = np.random.default_rng(42)
    a, a_fp32 = cast_data(rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32), dtype)
    b, b_fp32 = cast_data(rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32), dtype)
    x3, x3_fp32 = cast_data(rng.uniform(-0.5, 0.5, (m, n)).astype(np.float32), dtype)
    accumulator = np.matmul(a_fp32, b_fp32)
    if str(inner_precise) == "1":
        # High-performance mode: the matmul result is rounded to the output dtype before the
        # elementwise fusion (legacy B16-accumulator writeout semantics).
        _, accumulator = cast_data(accumulator, dtype)
    if op_type == "mul":
        golden_fp32 = accumulator * x3_fp32
    else:
        golden_fp32 = accumulator + x3_fp32
    golden, _ = cast_data(golden_fp32, dtype)
    (a.T.copy() if trans_a else a).tofile("./input/input_a.bin")
    (b.T.copy() if trans_b else b).tofile("./input/input_b.bin")
    x3.tofile("./input/input_x3.bin")
    golden.tofile("./output/cpu_output.bin")
    print("Data generated successfully!")


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
    parser = argparse.ArgumentParser(description="Generate FusedMatMul example data.")
    parser.add_argument("m", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("transA", choices=["true", "false"])
    parser.add_argument("transB", choices=["true", "false"])
    parser.add_argument("dtype", choices=["float16", "bfloat16"])
    # Mainline contract: 7th positional is the sigmoid example's bias_dtype.
    parser.add_argument("bias_dtype", choices=["none", "same", "float32"])
    # Elementwise example appends its own parameters (op_type/inner_precise) after the
    # mainline ones; bias_dtype stays in place for it (passed as "none").
    parser.add_argument(
        "op_type", nargs="?", default="sigmoid", help="add/mul, default sigmoid"
    )
    parser.add_argument("inner_precise", nargs="?", default="1", help="0/1, default 1")
    args = parser.parse_args()
    if min(args.m, args.k, args.n) <= 0:
        parser.error("M, K, N must be positive integers")
    if args.op_type in ("add", "mul"):
        gen_golden_elemwise(
            args.m,
            args.k,
            args.n,
            args.transA == "true",
            args.transB == "true",
            args.dtype,
            args.op_type,
            args.inner_precise,
        )
    else:
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
