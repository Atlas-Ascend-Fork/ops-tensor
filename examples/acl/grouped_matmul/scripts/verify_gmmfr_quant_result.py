#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""Compare GMMFR outputs against the CPU reference using the selected example accuracy contract."""

import argparse
from pathlib import Path
import sys
import numpy as np
import ml_dtypes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import write_metrics_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("golden")
    p.add_argument("actual")
    p.add_argument("--quant-mode", choices=("pertoken", "mx", "a8w4"), required=True)
    p.add_argument("--weight-format", choices=("ND", "NZ"), required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    a = p.parse_args()
    nz = a.quant_mode == "mx" and a.weight_format == "NZ"
    if a.dtype == "bfloat16" and not nz:
        raise ValueError("BF16 output is supported only by the MX WeightNZ example")
    value_tol, ratio_tol = (
        ((2e-2, 2e-2) if a.dtype == "bfloat16" else (4e-3, 4e-3)) if nz else (1e-4, 0.0)
    )
    dtype = ml_dtypes.bfloat16 if a.dtype == "bfloat16" else np.float32
    g = np.fromfile(a.golden, dtype=dtype).astype(np.float32)
    x = np.fromfile(a.actual, dtype=dtype).astype(np.float32)
    if g.size == 0 or x.shape != g.shape:
        raise ValueError("empty or inconsistent output size")
    ok = (
        np.isfinite(g)
        & np.isfinite(x)
        & np.isclose(x, g, rtol=value_tol, atol=value_tol)
    )
    status = (
        "pass"
        if np.all(np.isfinite(g) & np.isfinite(x)) and float(np.mean(~ok)) <= ratio_tol
        else "fail"
    )
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(np.max(np.abs(x - g))),
                "error_ratio": float(np.mean(~ok)),
                "ratio_tol": ratio_tol,
                "status": status,
            }
        ],
        status,
        str(Path(a.actual).resolve().parent),
    )
    if status != "pass":
        raise ValueError(f"CPU golden comparison failed: {int((~ok).sum())}/{g.size}")
    print(
        f"PASS: dtype={a.dtype}, rtol=atol={value_tol}, mismatch ratio <= {ratio_tol}"
    )


if __name__ == "__main__":
    main()
