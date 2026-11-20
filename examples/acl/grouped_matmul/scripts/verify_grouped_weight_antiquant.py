#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Full-output finite-data smoke check. Tolerances are explicit, not product acceptance criteria."""

import argparse
from pathlib import Path
import numpy as np
import ml_dtypes


OUTPUT_DTYPES = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}


def verify(golden_path, actual_path, m, n, dtype, atol, rtol):
    if m <= 0 or n <= 0 or not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("invalid dimensions or tolerances")
    storage_dtype = OUTPUT_DTYPES[dtype]
    size = m * n * np.dtype(storage_dtype).itemsize
    for path in (golden_path, actual_path):
        if Path(path).stat().st_size != size:
            raise ValueError(f"{path}: expected exactly {size} bytes")
    expected = np.fromfile(golden_path, dtype=storage_dtype).astype(np.float32)
    actual = np.fromfile(actual_path, dtype=storage_dtype).astype(np.float32)
    error = np.abs(actual - expected)
    bad = (
        ~np.isfinite(expected)
        | ~np.isfinite(actual)
        | (error > atol + rtol * np.abs(expected))
    )
    count = int(bad.sum())
    print(
        f"elements={m * n}, mismatches={count}, atol={atol}, rtol={rtol}, max_abs={float(error.max())}"
    )
    if count:
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"output mismatch at {divmod(index, n)}: expected={expected[index]}, actual={actual[index]}"
        )
    print(
        "PASS: example tolerance check only; product precision acceptance remains separate"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("golden")
    p.add_argument("actual")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dtype", choices=tuple(OUTPUT_DTYPES), required=True)
    p.add_argument("--atol", type=float, required=True)
    p.add_argument("--rtol", type=float, required=True)
    args = p.parse_args()
    verify(args.golden, args.actual, args.m, args.n, args.dtype, args.atol, args.rtol)


if __name__ == "__main__":
    main()
