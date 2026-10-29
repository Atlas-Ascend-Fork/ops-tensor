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

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json
import argparse
from dataclasses import dataclass

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch

FULL_TENSOR_PRINT_MAX_ELEMENTS = 128

DTYPE_CONFIG = {
    "float32": {
        "np_dtype": np.float32,
        "torch_dtype": torch.float32,
        "ratio_tol": 1e-4,
    },
    "bfloat16": {
        "np_dtype": np.uint16,
        "torch_dtype": torch.bfloat16,
        "ratio_tol": 5e-3,
    },
    "float16": {"np_dtype": np.uint16, "torch_dtype": torch.float16, "ratio_tol": 5e-3},
}


@dataclass
class VerifySummaryCfg:
    m: int
    batch: int
    n: int
    dtype_str: str
    ratio_tol: float


def _compute_ratio_error_mask(
    golden_f32, npu_f32, abs_diff, non_finite_mask, dtype_str, ratio_tol
):
    if dtype_str == "float32":
        max_ab = torch.maximum(torch.abs(golden_f32), torch.abs(npu_f32))
        metric = torch.where(
            max_ab > 0,
            abs_diff / max_ab,
            torch.where(
                abs_diff == 0,
                torch.zeros_like(abs_diff),
                torch.full_like(abs_diff, float("inf")),
            ),
        )
    else:
        metric = abs_diff
    return (metric > ratio_tol) | non_finite_mask


def _ratio_label(dtype_str, ratio_tol):
    if dtype_str == "float32":
        return f"rel_err(abs(a-b)/max(a,b)) > {ratio_tol:g}"
    return f">{ratio_tol:g}"


def _print_large_tensor_summary(
    golden_tensor, npu_output_tensor, cfg: VerifySummaryCfg
):
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = cfg.m * cfg.batch * cfg.n
    no_non_finite = torch.zeros_like(g, dtype=torch.bool)
    ratio_mask = _compute_ratio_error_mask(
        g, p, abs_err, no_non_finite, cfg.dtype_str, cfg.ratio_tol
    )
    over_tol = ratio_mask.sum().item()

    print(f"\n[verify] shape=({cfg.m}, {cfg.batch}, {cfg.n}), elements={numel}")
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(
        f"  count({_ratio_label(cfg.dtype_str, cfg.ratio_tol)}): {over_tol} / {numel}"
    )


def verify_result(m, batch, n, dtype_str, is_hf32=False):
    cfg = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["float16"])
    np_dtype = cfg["np_dtype"]
    torch_dtype = cfg["torch_dtype"]
    ratio_tol = 1e-3 if is_hf32 else cfg["ratio_tol"]

    output = np.fromfile("./output/npu_out.bin", dtype=np_dtype)
    golden = np.fromfile("./output/cpu_output.bin", dtype=np_dtype)

    if output.size != golden.size:
        raise ValueError("npu output size != cpu output size")

    # C is stored in transposed-batch layout: [m, batch, n]
    npu_output_tensor = torch.from_numpy(output).view(torch_dtype).reshape(m, batch, n)
    golden_tensor = torch.from_numpy(golden).view(torch_dtype).reshape(m, batch, n)

    numel = m * batch * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print("\ncpu golden:\n", golden_tensor)
        print("npu output:\n", npu_output_tensor)
    else:
        summary_cfg = VerifySummaryCfg(
            m=m, batch=batch, n=n, dtype_str=dtype_str, ratio_tol=ratio_tol
        )
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, summary_cfg)

    golden_f32 = golden_tensor.to(torch.float32)
    npu_f32 = npu_output_tensor.to(torch.float32)
    abs_diff = torch.abs(golden_f32 - npu_f32)
    non_finite_mask = ~(
        torch.isfinite(golden_f32) & torch.isfinite(npu_f32) & torch.isfinite(abs_diff)
    )
    ratio_error_mask = _compute_ratio_error_mask(
        golden_f32, npu_f32, abs_diff, non_finite_mask, dtype_str, ratio_tol
    )

    error_count = int(ratio_error_mask.sum().item())
    error_ratio = error_count / numel if numel else 0.0

    print(f"max abs diff: {abs_diff.max().item() if numel else 0.0}")
    print(
        f"ratio error count({_ratio_label(dtype_str, ratio_tol)}): {error_count}/{numel}, "
        f"error ratio: {error_ratio:.6f}"
    )

    _status = "pass" if error_ratio <= ratio_tol else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(abs_diff.max().item()) if numel else 0.0,
                "error_ratio": float(error_ratio),
                "ratio_tol": float(ratio_tol),
                "status": _status,
            }
        ],
        _status,
        "./output",
    )

    return error_ratio <= ratio_tol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify NPU output against CPU golden."
    )
    parser.add_argument("m", type=int, help="Matrix M dimension")
    parser.add_argument("batch", type=int, help="Batch dimension")
    parser.add_argument("n", type=int, help="Matrix N dimension")
    parser.add_argument(
        "dtype", nargs="?", default="float16", help="Data type (default: float16)"
    )
    parser.add_argument("--hf32", action="store_true", help="Enable HF32")
    args = parser.parse_args()

    m = args.m
    batch = args.batch
    n = args.n
    dtype_str = args.dtype

    try:
        res = verify_result(m, batch, n, dtype_str, is_hf32=args.hf32)
        if not res:
            cfg = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["float16"])
            ratio_tol = 1e-3 if args.hf32 else cfg["ratio_tol"]
            raise ValueError(
                f"[ERROR] NPU results differ from CPU. "
                f"The ratio of points with {_ratio_label(dtype_str, ratio_tol)} "
                f"must be <= {ratio_tol}.\n"
            )
        print("[PASS] NPU results are consistent with CPU.\n")

    except Exception as e:
        print(e)
        sys.exit(1)
