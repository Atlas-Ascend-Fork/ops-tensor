#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate RotateQuant inputs and an independent NumPy CPU reference.

The example fixes scaleAlg=2, dstTypeMax=6, roundMode=rint and trans=false.
MMAD is evaluated in FP32 from FP16/BF16 inputs, then rounded to BF16 before
optional per-32-element-group clipping. golden_ref.bin is this pre-quantization
reference; golden_scale.bin contains E8M0 bytes with per-row padding.
"""

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np


SEED = 20261026
GROUP_SIZE = 32
MAX_GROUP_NUM = 1024
OUTPUT_SENTINEL = 0xA5


def parse_groups(m, text, group_list_type):
    """Return original device values and M counts; preserve empty group indexes."""
    if text == "none":
        return [], [m]
    values = [int(item) for item in text.split(";")]
    if not 1 <= len(values) <= MAX_GROUP_NUM:
        raise ValueError("groupList length must be in [1, 1024]")
    counts = []
    end = 0
    for value in values:
        next_end = value if group_list_type == 0 else end + value
        if value < 0 or next_end < end or next_end > m:
            raise ValueError("groupList must describe a nonnegative prefix of M")
        counts.append(next_end - end)
        end = next_end
    return values, counts


def validate_args(args):
    if args.m <= 0 or args.n <= 0 or args.m * args.n > np.iinfo(np.int32).max:
        raise ValueError("Require M,N > 0 and M*N <= INT32_MAX in this example")
    if args.k not in (32, 64, 128, 256) or args.n % args.k != 0:
        raise ValueError("K must be 32/64/128/256 and divide N")
    if args.b not in (1, args.n // args.k):
        raise ValueError("B must be 1 or N/K")
    if args.alpha != "none":
        alpha = float(args.alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "This example accepts alpha=none or a finite value in [0,1]"
            )
    return parse_groups(args.m, args.group_list, args.group_list_type)


def rotate_reference(x, rotations, counts, b, k):
    """For group e and segment q, multiply X[e,:,q,:] by R[e,0 or q,:,:]."""
    m, n = x.shape
    q = n // k
    result = np.zeros((m, n), dtype=np.float32)
    start = 0
    for group, count in enumerate(counts):
        end = start + count
        if count:
            rows = x[start:end].astype(np.float32).reshape(count, q, k)
            rotation = rotations[group].astype(np.float32)
            if b == 1:
                rotated = np.matmul(rows, rotation[0])
            else:
                rotated = np.matmul(rows.transpose(1, 0, 2), rotation).transpose(
                    1, 0, 2
                )
            result[start:end] = rotated.reshape(count, n)
        start = end
    return result


def apply_clamp(reference, effective_m, alpha):
    result = reference.astype(ml_dtypes.bfloat16).astype(np.float32)
    if effective_m and 0.0 < alpha < 1.0:
        groups = result[:effective_m].reshape(-1, GROUP_SIZE)
        limits = (np.max(np.abs(groups), axis=1) * alpha).astype(ml_dtypes.bfloat16)
        groups[:] = np.clip(
            groups,
            -limits.astype(np.float32)[:, None],
            limits.astype(np.float32)[:, None],
        )
    return result


def golden_scales(reference, effective_m):
    """DYN range, dstTypeMax=6: round the BF16 amax exponent and subtract two."""
    m, n = reference.shape
    columns = ((n + 63) // 64) * 2
    result = np.full((m, columns), OUTPUT_SENTINEL, dtype=np.uint8)
    if effective_m:
        groups = reference[:effective_m].reshape(-1, GROUP_SIZE)
        maxima = np.max(np.abs(groups), axis=1).astype(ml_dtypes.bfloat16)
        bits = maxima.view(np.uint16).astype(np.int32)
        # Adding 63 before removing seven fraction bits matches the DYN threshold:
        # fraction codes >=65 increment the exponent; smaller exponents clamp to 2.
        exponent = np.maximum((bits + 63) >> 7, 2) - 2
        result[:effective_m, : n // GROUP_SIZE] = exponent.astype(np.uint8).reshape(
            effective_m, -1
        )
        result[:effective_m, n // GROUP_SIZE :] = 0
    return result


def generate(args):
    values, counts = validate_args(args)
    effective_m = sum(counts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dtype = np.float16 if args.dtype == "float16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(SEED)
    x = (rng.standard_normal((args.m, args.n)) * 0.5).astype(input_dtype)
    rotations = np.empty((len(counts), args.b, args.k, args.k), dtype=input_dtype)
    for group in range(len(counts)):
        for batch in range(args.b):
            matrix, _ = np.linalg.qr(rng.standard_normal((args.k, args.k)))
            rotations[group, batch] = matrix.astype(input_dtype)
    alpha = np.array(
        [0.0 if args.alpha == "none" else float(args.alpha)], dtype=ml_dtypes.bfloat16
    )
    reference = rotate_reference(x, rotations, counts, args.b, args.k)
    reference = apply_clamp(reference, effective_m, float(alpha[0]))

    x.view(np.uint16).tofile(output_dir / "input_x.bin")
    rotations.view(np.uint16).tofile(output_dir / "input_rotation.bin")
    alpha.view(np.uint16).tofile(output_dir / "input_alpha.bin")
    np.array(values if values else [0], dtype=np.int64).tofile(
        output_dir / "input_group_list.bin"
    )
    reference.tofile(output_dir / "golden_ref.bin")
    golden_scales(reference, effective_m).tofile(output_dir / "golden_scale.bin")
    metadata = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "b": args.b,
        "dtype": args.dtype,
        "group_list": values,
        "group_list_type": args.group_list_type,
        "counts": counts,
        "effective_m": effective_m,
        "has_alpha": args.alpha != "none",
        "alpha_bf16": float(alpha[0]),
        "seed": SEED,
        "scale_alg": 2,
        "dst_type_max": 6,
        "round_mode": "rint",
        "trans": False,
        "numpy_version": np.__version__,
        "ml_dtypes_version": ml_dtypes.__version__,
    }
    (output_dir / "case.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Generated M={args.m}, N={args.n}, K={args.k}, B={args.b}, active rows={effective_m}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate RotateQuant input and CPU reference data."
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--group-list-type", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--group-list", required=True, help="none, or semicolon-separated INT64 values"
    )
    parser.add_argument(
        "--alpha", required=True, help="none, or a finite value in [0,1]"
    )
    parser.add_argument("--output-dir", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
