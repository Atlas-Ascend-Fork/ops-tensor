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
from pathlib import Path

import numpy as np

from reference import confidence_head_compute


def cast_feature(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        packed = values.astype(np.float16)
        return packed, packed.astype(np.float32)
    bits = values.view(np.uint32).astype(np.uint64)
    packed = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return packed, (packed.astype(np.uint32) << 16).view(np.float32)


def generate(args):
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    shape = (args.batch, args.n, args.k_embed)
    hidden, hidden_fp32 = cast_feature(rng.uniform(-0.25, 0.25, shape), args.dtype)
    weight, weight_fp32 = cast_feature(
        rng.uniform(-0.25, 0.25, args.k_embed + args.k_markov), args.dtype
    )
    hidden.tofile("input/hidden.bin")
    weight.tofile("input/weight.bin")
    markov_fp32 = None
    if args.has_markov == "true":
        markov, markov_fp32 = cast_feature(
            rng.uniform(-0.25, 0.25, (args.batch, args.n, args.k_markov)), args.dtype
        )
        markov.tofile("input/markov.bin")
    temperature = None
    if args.temperature != "none":
        temperature = np.array([float(args.temperature)], dtype=np.float32)
        temperature.tofile("input/temperature.bin")
    bias = None
    if args.bias == "true":
        bias = np.linspace(-1.25, 1.25, args.n, dtype=np.float32)
        bias.tofile("input/bias.bin")
    low = None if args.clamp_low == "none" else float(args.clamp_low)
    high = None if args.clamp_high == "none" else float(args.clamp_high)
    result = confidence_head_compute(
        hidden_fp32,
        weight_fp32,
        markov_fp32,
        temperature,
        bias,
        clamp_low=low,
        clamp_high=high,
    )[0]
    result.tofile("output/cpu_output.bin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ConfidenceHeadFused example inputs and FP32 golden."
    )
    for name in ("n", "k_embed", "k_markov", "batch"):
        parser.add_argument(name, type=int)
    parser.add_argument("dtype", choices=("float16", "bfloat16"))
    parser.add_argument("has_markov", choices=("true", "false"))
    parser.add_argument("temperature")
    parser.add_argument("bias", choices=("true", "false"))
    parser.add_argument("clamp_low")
    parser.add_argument("clamp_high")
    args = parser.parse_args()
    dims = (args.n, args.k_embed, args.k_markov, args.batch)
    if min(dims) < 0 or max(dims) > 0xFFFFFFFF:
        parser.error("dimensions must fit UINT32")
    if args.n * args.batch > 0xFFFFFFFF or args.k_embed + args.k_markov > 0xFFFFFFFF:
        parser.error("flattened N and total K must fit UINT32")
    if args.has_markov == "false" and args.k_markov != 0:
        parser.error("k_markov must be zero when has_markov=false")
    return args


if __name__ == "__main__":
    generate(parse_args())
