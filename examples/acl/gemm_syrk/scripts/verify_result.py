#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch

# Hybrid tolerance (atol + rtol * |golden|) per element; an element fails when
# |npu - golden| > tol. error_ratio counts failed elements and must stay below
# RATIO_TOL. Symmetry: the kernel writes the (j, i) mirror tile from the same
# accumulator, so the output must be symmetric up to one rounding step.
DTYPE_CONFIG = {
    "float16": {
        "np_dtype": np.uint16,
        "torch_dtype": torch.float16,
        "atol": 4e-3,
        "rtol": 4e-3,
        "sym_tol": 4e-3,
    },
    "bfloat16": {
        "np_dtype": np.uint16,
        "torch_dtype": torch.bfloat16,
        "atol": 2e-2,
        "rtol": 2e-2,
        "sym_tol": 2e-2,
    },
}
RATIO_TOL = 1e-3


def verify_result(m, batch, dtype_str):
    cfg = DTYPE_CONFIG[dtype_str]
    numel = batch * m * m

    output = np.fromfile("./output/npu_out.bin", dtype=cfg["np_dtype"])
    golden = np.fromfile("./output/golden_c.bin", dtype=cfg["np_dtype"])
    if output.size != numel or golden.size != numel:
        raise ValueError(
            f"file size mismatch: output={output.size}, golden={golden.size}, expected={numel}"
        )

    npu_tensor = torch.from_numpy(output).view(cfg["torch_dtype"]).reshape(batch, m, m)
    golden_tensor = (
        torch.from_numpy(golden).view(cfg["torch_dtype"]).reshape(batch, m, m)
    )

    golden_f32 = golden_tensor.to(torch.float32)
    npu_f32 = npu_tensor.to(torch.float32)
    abs_diff = torch.abs(golden_f32 - npu_f32)
    tol = cfg["atol"] + cfg["rtol"] * torch.abs(golden_f32)
    error_mask = abs_diff > tol
    error_count = int(error_mask.sum().item())
    error_ratio = error_count / numel if numel else 0.0
    max_abs_diff = float(abs_diff.max().item()) if numel else 0.0

    print(f"[verify] shape=({batch}, {m}, {m}), elements={numel}")
    print(f"max abs diff: {max_abs_diff}")
    print(
        f"error count(|diff| > atol + rtol*|golden|): {error_count}/{numel}, "
        f"error ratio: {error_ratio:.6f} (tol {RATIO_TOL})"
    )

    sym_diff = torch.abs(npu_f32 - torch.transpose(npu_f32, 1, 2))
    sym_max_diff = float(sym_diff.max().item()) if numel else 0.0
    print(f"symmetry max diff: {sym_max_diff} (tol {cfg['sym_tol']})")

    accuracy_ok = error_ratio <= RATIO_TOL
    symmetry_ok = sym_max_diff <= cfg["sym_tol"]
    status = "pass" if (accuracy_ok and symmetry_ok) else "fail"

    write_metrics_json(
        [
            {
                "name": "output",
                "max_abs_diff": max_abs_diff,
                "error_ratio": float(error_ratio),
                "ratio_tol": float(RATIO_TOL),
                "status": "pass" if accuracy_ok else "fail",
            },
            {
                "name": "symmetry",
                "max_abs_diff": sym_max_diff,
                "error_ratio": 0.0 if symmetry_ok else 1.0,
                "ratio_tol": 0.0,
                "status": "pass" if symmetry_ok else "fail",
            },
        ],
        status,
        "./output",
    )

    return accuracy_ok and symmetry_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify gemm_syrk NPU output against golden (accuracy + symmetry)."
    )
    parser.add_argument("m", type=int, help="M (=N) dimension")
    parser.add_argument("batch", type=int, help="Batch dimension")
    parser.add_argument("dtype", type=str, choices=["float16", "bfloat16"])
    args = parser.parse_args()

    try:
        res = verify_result(args.m, args.batch, args.dtype)
        if not res:
            raise ValueError(
                "[ERROR] NPU results differ from golden "
                f"(error ratio > {RATIO_TOL} or output is not symmetric).\n"
            )
        print("[PASS] NPU results are consistent with golden (and symmetric).\n")

    except Exception as e:
        print(e)
        sys.exit(1)
