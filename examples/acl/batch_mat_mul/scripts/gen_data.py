#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
    lo, hi = (-1, 1) if dtype in (torch.float16, torch.bfloat16) else (0.0, 1.0)
    bias = np.random.uniform(lo, hi, (n,)).astype(np.float32)
    bias_tensor = torch.from_numpy(bias).to(dtype)
    vt = VIEW_TYPE_MAP[dtype]
    input_dir = os.path.join(os.getcwd(), "input")
    os.makedirs(input_dir, exist_ok=True)
    bias_tensor.view(vt).numpy().tofile(os.path.join(input_dir, "bias.bin"))
    print(f"[INFO] Generated bias: n={n}, dtype={dtype}")
    return bias_tensor


def gen_batch_data(
    m, k, n, batch_a, batch_b, batch_c, transpose_a, transpose_b, dtype, bias_size=0
):
    lo, hi = (-1.0, 1.0) if dtype in (torch.float16, torch.bfloat16) else (0.0, 1.0)

    a_shape = (batch_a, k, m) if transpose_a else (batch_a, m, k)
    b_shape = (batch_b, n, k) if transpose_b else (batch_b, k, n)

    a_ori = np.random.uniform(lo, hi, a_shape).astype(np.float32)
    b_ori = np.random.uniform(lo, hi, b_shape).astype(np.float32)

    a_cpu = torch.from_numpy(a_ori).to(dtype)
    b_cpu = torch.from_numpy(b_ori).to(dtype)

    a_cpu_t = a_cpu.transpose(-2, -1) if transpose_a else a_cpu
    b_cpu_t = b_cpu.transpose(-2, -1) if transpose_b else b_cpu

    a_broadcast = (
        a_cpu_t.expand(batch_c, -1, -1) if batch_a == 1 and batch_c > 1 else a_cpu_t
    )
    b_broadcast = (
        b_cpu_t.expand(batch_c, -1, -1) if batch_b == 1 and batch_c > 1 else b_cpu_t
    )

    out = torch.matmul(a_broadcast.float(), b_broadcast.float()).to(dtype)

    if bias_size > 0:
        bias_tensor = generate_bias(n, dtype)
        out = out + bias_tensor.float().to(dtype)

    a_flat = a_cpu.reshape(-1)
    b_flat = b_cpu.reshape(-1)
    out_flat = out.reshape(-1)

    write_artifacts(os.getcwd(), a_flat, b_flat, out_flat, dtype)

    print("Batch data generated successfully!")
    print(f"  A: batch={batch_a}, shape={tuple(a_cpu.shape)}")
    print(f"  B: batch={batch_b}, shape={tuple(b_cpu.shape)}")
    print(f"  C: batch={batch_c}, shape={tuple(out.shape)}")


if __name__ == "__main__":
    if len(sys.argv) < 10:
        print(
            "Usage: python3 gen_data.py m k n batch_a batch_b batch_c transA transB dtype [bias]"
        )
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    batch_a = int(sys.argv[4])
    batch_b = int(sys.argv[5])
    batch_c = int(sys.argv[6])
    transpose_a = sys.argv[7].lower() == "true"
    transpose_b = sys.argv[8].lower() == "true"
    dtype_str = sys.argv[9].lower()
    DATA_TYPE = DTYPE_MAP.get(dtype_str, torch.float16)

    bias = 0
    if len(sys.argv) >= 11:
        bias = int(sys.argv[10])

    gen_batch_data(
        m, k, n, batch_a, batch_b, batch_c, transpose_a, transpose_b, DATA_TYPE, bias
    )
