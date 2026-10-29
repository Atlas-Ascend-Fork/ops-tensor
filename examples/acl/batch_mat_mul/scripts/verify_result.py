#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"),
)
from metrics import write_metrics_json
import argparse

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


def verify_batch_result(m, n, batch, dtype_str, is_hf32=False):
    cfg = DTYPE_CONFIG.get(dtype_str, DTYPE_CONFIG["float16"])
    np_dtype = cfg["np_dtype"]
    torch_dtype = cfg["torch_dtype"]
    ratio_tol = 1e-3 if is_hf32 else cfg["ratio_tol"]

    output = np.fromfile("./output/npu_out.bin", dtype=np_dtype)
    golden = np.fromfile("./output/cpu_output.bin", dtype=np_dtype)

    if output.size != golden.size:
        raise ValueError(
            f"npu output size ({output.size}) != cpu output size ({golden.size})"
        )

    total_elements = batch * m * n
    npu_output_tensor = torch.from_numpy(output).view(torch_dtype).reshape(batch, m, n)
    golden_tensor = torch.from_numpy(golden).view(torch_dtype).reshape(batch, m, n)

    golden_f32 = golden_tensor.to(torch.float32)
    npu_f32 = npu_output_tensor.to(torch.float32)
    abs_diff = torch.abs(golden_f32 - npu_f32)
    non_finite_mask = ~(
        torch.isfinite(golden_f32) & torch.isfinite(npu_f32) & torch.isfinite(abs_diff)
    )

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

    ratio_error_mask = (metric > ratio_tol) | non_finite_mask
    error_count = int(ratio_error_mask.sum().item())
    error_ratio = error_count / total_elements if total_elements else 0.0

    print(f"[verify] batch={batch}, shape=({m}, {n}), elements={total_elements}")
    print(f"max abs diff: {abs_diff.max().item() if total_elements else 0.0}")
    print(
        f"ratio error count: {error_count}/{total_elements}, error ratio: {error_ratio:.6f}"
    )

    _status = "pass" if error_ratio <= ratio_tol else "fail"
    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": float(abs_diff.max().item()) if total_elements else 0.0,
                "error_ratio": float(error_ratio),
                "ratio_tol": float(ratio_tol),
                "elements": int(total_elements),
                "status": _status,
            }
        ],
        _status,
        "./output",
    )

    return error_ratio <= ratio_tol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify batch matmul NPU output against CPU golden."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("batch", type=int)
    parser.add_argument("dtype", nargs="?", default="float16")
    parser.add_argument("--hf32", action="store_true")
    args = parser.parse_args()

    try:
        res = verify_batch_result(
            args.m, args.n, args.batch, args.dtype, is_hf32=args.hf32
        )
        if not res:
            raise ValueError("[ERROR] NPU results differ from CPU golden.")
        print("[PASS] NPU results are consistent with CPU.\n")
    except Exception as e:
        print(e)
        sys.exit(1)
