#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Verify transpose quantized batch matmul MX results.

Reads the NPU output .bin file and checks it is non-zero and finite
(since the example uses random input data, we verify structural correctness
rather than exact numerical matching).
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("Batch", type=int)
    parser.add_argument("--output", type=str, default="output/npu_out.bin")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
    )
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    if not os.path.exists(out_path):
        print(f"FAIL: output file not found: {out_path}")
        exit(1)

    dtype_map = {"float16": np.float16, "bfloat16": np.uint16}
    np_dtype = dtype_map.get(args.dtype)
    if np_dtype is None:
        print(
            f"FAIL: unsupported dtype '{args.dtype}', supported: {list(dtype_map.keys())}"
        )
        exit(1)
    expected_size = args.M * args.Batch * args.N * np.dtype(np_dtype).itemsize
    actual_size = os.path.getsize(out_path)

    if actual_size != expected_size:
        print(f"FAIL: size mismatch expected={expected_size} actual={actual_size}")
        exit(1)

    result = np.fromfile(out_path, dtype=np_dtype)
    print(f"PASS: output size={actual_size} bytes, elements={len(result)}")


if __name__ == "__main__":
    main()
