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

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import numpy as np
import torch

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

VIEW_TYPE_MAP = {
    torch.float16: torch.uint16,
    torch.float32: torch.uint32,
    torch.bfloat16: torch.uint16,
}

DATA_TYPE = torch.float16


def write_artifacts(base_dir, a_data, b_data, out, dtype):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vt = VIEW_TYPE_MAP[dtype]
    a_data.view(vt).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    b_data.view(vt).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(vt).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def generate_bias(n, dtype):
    """生成 bias 数据大小为 n, 并返回 bias tensor 供 golden 计算复用"""
    lo, hi = (-1, 1) if dtype in (torch.float16, torch.bfloat16) else (0.0, 1.0)
    bias = np.random.uniform(lo, hi, n).astype(np.float32)
    bias_tensor = torch.from_numpy(bias).to(dtype)
    vt = VIEW_TYPE_MAP[dtype]
    input_dir = os.path.join(os.getcwd(), "input")
    os.makedirs(input_dir, exist_ok=True)
    bias_tensor.view(vt).numpy().tofile(os.path.join(input_dir, "bias.bin"))
    print(f"[INFO] Generated bias: {n} elements, dtype={dtype}")
    return bias_tensor


def gen_golden_data(
    m, k, n, transpose_a, transpose_b, dtype, bias_size=0, bias_tensor=None
):
    lo, hi = (-1.0, 1.0) if dtype in (torch.float16, torch.bfloat16) else (0.0, 1.0)
    a_ori = (
        np.random.uniform(lo, hi, (k, m)).astype(np.float32)
        if transpose_a
        else np.random.uniform(lo, hi, (m, k)).astype(np.float32)
    )
    b_ori = (
        np.random.uniform(lo, hi, (n, k)).astype(np.float32)
        if transpose_b
        else np.random.uniform(lo, hi, (k, n)).astype(np.float32)
    )

    a_cpu = torch.from_numpy(a_ori).to(dtype)
    b_cpu = torch.from_numpy(b_ori).to(dtype)

    a_cpu_t = a_cpu.t() if transpose_a else a_cpu
    b_cpu_t = b_cpu.t() if transpose_b else b_cpu

    if bias_size > 0 and bias_tensor is not None:
        out = torch.addmm(bias_tensor.float(), a_cpu_t.float(), b_cpu_t.float()).to(
            dtype
        )
    else:
        out = torch.matmul(a_cpu_t.float(), b_cpu_t.float()).to(dtype)

    write_artifacts(os.getcwd(), a_cpu, b_cpu, out, dtype)

    print("Data generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 6, 7, 8, 9):
        print(
            "Usage: python3 gen_data.py m k n [transA transB] [dtype] [bias] [layoutB]"
        )
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    if len(sys.argv) >= 6:
        transpose_a = sys.argv[4].lower() == "true"
        transpose_b = sys.argv[5].lower() == "true"
    else:
        transpose_a = False
        transpose_b = False

    if len(sys.argv) >= 7:
        dtype_str = sys.argv[6].lower()
        DATA_TYPE = DTYPE_MAP.get(dtype_str)
        if DATA_TYPE is None:
            print(f"Unsupported dtype: {dtype_str}, using float16")
            DATA_TYPE = torch.float16

    bias = 0
    if len(sys.argv) >= 8:
        bias = int(sys.argv[7])
        if bias != 0 and bias != n:
            print(f"Error: bias ({bias}) must equal n ({n}) or 0")
            sys.exit(1)
    if len(sys.argv) >= 9:
        layout_b = sys.argv[8]
        if layout_b not in ("ND", "NZ"):
            print(f"Error: layoutB must be 'ND' or 'NZ', got '{layout_b}'")
            sys.exit(1)

    bias_tensor = None
    if bias > 0:
        bias_tensor = generate_bias(bias, DATA_TYPE)

    gen_golden_data(m, k, n, transpose_a, transpose_b, DATA_TYPE, bias, bias_tensor)
