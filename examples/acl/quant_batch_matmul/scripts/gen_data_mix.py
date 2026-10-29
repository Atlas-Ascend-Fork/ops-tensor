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

"""Generate deterministic inputs for the QBMM MIX examples."""

import argparse
import os

import numpy as np


X1_SCALE_VALUES = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32)
X2_SCALE_VALUES = np.array([2.0, 0.5, 0.25, 1.0, 4.0], dtype=np.float32)


def generate(args):
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M, K, and N must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)

    rng = np.random.default_rng(20260904)
    a = rng.integers(-4, 5, size=(args.m, args.k), dtype=np.int8)
    b = rng.integers(-4, 5, size=(args.k, args.n), dtype=np.int8)
    scale_a = X1_SCALE_VALUES[np.arange(args.m) % X1_SCALE_VALUES.size]
    scale_b = X2_SCALE_VALUES[np.arange(args.n) % X2_SCALE_VALUES.size]
    golden = (
        np.matmul(a.astype(np.int32), b.astype(np.int32)).astype(np.float32)
        * scale_a.reshape(args.m, 1)
        * scale_b.reshape(1, args.n)
    ).astype(np.float16)

    a.tofile(os.path.join(args.output_dir, "input_a.bin"))
    b.tofile(os.path.join(args.output_dir, "input_b.bin"))
    scale_a.tofile(os.path.join(args.output_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(args.output_dir, "scale_b.bin"))
    golden.tofile(os.path.join(args.output_dir, "golden_c.bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("batch", "without_batch"))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
