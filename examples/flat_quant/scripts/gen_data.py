#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate deterministic BF16 inputs and a float32 golden reference for flat_quant.

Semantics: y[b] = P1 @ x[b] @ P2 (P2 is skipped when has_p2 is false), computed in
float32 from BF16 inputs. The kernel quantizes the BF16 result to MX-FP4 (E2M1)
with per-32-element-group E8M0 scales; the golden reference is the pre-quantization
float32 matrix, quantization is checked by verify_result.py.
"""

import argparse
import os

import ml_dtypes
import numpy as np


def make_near_orthogonal(dim, rng):
    """FlatQuant transforms are near-orthogonal learned matrices; use QR for realism."""
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q.astype(np.float32)


def generate(args):
    if args.iter_batch < 1 or args.iter_batch > args.k:
        raise ValueError(f"iter_batch must be in [1, {args.k}]")
    if (
        args.m % 16 != 0
        or args.n % 16 != 0
        or (args.m * args.n) % 64 != 0
        or args.m * args.n > 4096
    ):
        raise ValueError(
            "M and N must be multiples of 16 and M*N a multiple of 64 and <= 4096"
        )
    if args.iter_batch * args.m * args.n > 32768:
        raise ValueError("iter_batch*M*N must be <= 32768")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.output_dir), "output"), exist_ok=True)

    rng = np.random.default_rng(20261008)
    x = (rng.standard_normal((args.k, args.m, args.n)) * 0.5).astype(ml_dtypes.bfloat16)
    p1 = make_near_orthogonal(args.m, rng).astype(ml_dtypes.bfloat16)
    p2 = make_near_orthogonal(args.n, rng).astype(ml_dtypes.bfloat16)

    x_f32 = x.astype(np.float32)
    p1_f32 = p1.astype(np.float32)
    p2_f32 = p2.astype(np.float32)
    temp = np.matmul(x_f32, p2_f32) if args.has_p2 else x_f32
    golden = np.matmul(p1_f32, temp).astype(np.float32)

    x.view(np.uint16).tofile(os.path.join(args.output_dir, "input_x.bin"))
    p1.view(np.uint16).tofile(os.path.join(args.output_dir, "input_p1.bin"))
    p2.view(np.uint16).tofile(os.path.join(args.output_dir, "input_p2.bin"))
    golden.tofile(os.path.join(args.output_dir, "golden_ref.bin"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate flat_quant input and golden data."
    )
    parser.add_argument(
        "--k", type=int, required=True, help="number of MxN input matrices"
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--iter-batch", type=int, required=True)
    parser.add_argument("--has-p2", type=lambda v: v.lower() == "true", required=True)
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
