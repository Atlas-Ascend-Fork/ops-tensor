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

"""Generate deterministic data and NumPy golden outputs for BlockAttnResPrepare."""

import argparse
import os

import numpy as np


TOTAL_T = 1
TOTAL_N = 8
TOTAL_S = 2
TOTAL_D = 32
EPSILON = 1.0e-6


def generate(args):
    if not 0 <= args.valid_blocks <= TOTAL_N:
        raise ValueError(f"valid_blocks must be in [0, {TOTAL_N}]")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)
    residual = np.fromfunction(
        lambda t, n, d: (((t * 13 + n * 7 + d * 5) % 23) - 11) * 0.015625,
        (TOTAL_T, TOTAL_N, TOTAL_D),
        dtype=np.float32,
    ).astype(np.float32)
    query = np.fromfunction(
        lambda s, d: (((s * 17 + d * 3) % 19) - 9) * 0.03125,
        (TOTAL_S, TOTAL_D),
        dtype=np.float32,
    ).astype(np.float32)
    golden_max = np.full((TOTAL_T, TOTAL_S), np.finfo(np.float32).min, dtype=np.float32)
    golden_sum = np.zeros((TOTAL_T, TOTAL_S), dtype=np.float32)
    golden_output = np.zeros((TOTAL_T, TOTAL_S, TOTAL_D), dtype=np.float32)
    if args.valid_blocks > 0:
        valid_residual = residual[:, : args.valid_blocks, :]
        for token in range(TOTAL_T):
            values = valid_residual[token]
            rms = np.sqrt(np.mean(np.square(values), axis=1) + EPSILON)
            logits = np.matmul(query, values.T) / rms.reshape(1, -1)
            golden_max[token] = np.max(logits, axis=1)
            weights = np.exp(logits - golden_max[token].reshape(-1, 1))
            golden_sum[token] = np.sum(weights, axis=1)
            golden_output[token] = np.matmul(weights, values)

    residual.tofile(os.path.join(args.output_dir, "block_residual.bin"))
    query.tofile(os.path.join(args.output_dir, "effective_query.bin"))
    golden_max.tofile(os.path.join(args.output_dir, "golden_max.bin"))
    golden_output.tofile(os.path.join(args.output_dir, "golden_output.bin"))
    golden_sum.tofile(os.path.join(args.output_dir, "golden_sum.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-blocks", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
