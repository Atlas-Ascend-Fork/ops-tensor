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
from dataclasses import dataclass

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


@dataclass
class GemmGenConfig:
    m: int
    k: int
    n: int
    batch: int
    trans_batch_a: bool
    dtype: torch.dtype


def write_artifacts(base_dir, a_data, b_data, out, dtype):
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vt = VIEW_TYPE_MAP[dtype]
    a_data.view(vt).numpy().tofile(os.path.join(input_dir, "input_a.bin"))
    b_data.view(vt).numpy().tofile(os.path.join(input_dir, "input_b.bin"))
    out.view(vt).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data(cfg: GemmGenConfig):
    lo, hi = (-1.0, 1.0) if cfg.dtype in (torch.float16, torch.bfloat16) else (0.0, 1.0)

    a_logical = torch.from_numpy(
        np.random.uniform(lo, hi, (cfg.batch, cfg.m, cfg.k)).astype(np.float32)
    ).to(cfg.dtype)
    b = torch.from_numpy(
        np.random.uniform(lo, hi, (cfg.batch, cfg.k, cfg.n)).astype(np.float32)
    ).to(cfg.dtype)

    out_logical = torch.matmul(a_logical.float(), b.float()).to(cfg.dtype)

    # C is always stored in transposed-batch layout: [m, batch, n]
    out_stored = out_logical.transpose(0, 1).contiguous()

    # A storage layout depends on trans_batch_a
    if cfg.trans_batch_a:
        a_stored = a_logical.transpose(0, 1).contiguous()  # [m, batch, k]
    else:
        a_stored = a_logical  # [batch, m, k]

    write_artifacts(os.getcwd(), a_stored, b, out_stored, cfg.dtype)

    print("Data generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6, 7):
        print("Usage: python3 gen_data.py m k n batch [trans_batch_a] [dtype]")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    batch = int(sys.argv[4])

    trans_batch_a = False
    if len(sys.argv) >= 6:
        trans_batch_a = sys.argv[5].lower() == "true"

    dtype_str = "float16"
    if len(sys.argv) >= 7:
        dtype_str = sys.argv[6].lower()
        DATA_TYPE = DTYPE_MAP.get(dtype_str)
        if DATA_TYPE is None:
            print(f"Unsupported dtype: {dtype_str}, using float16")
            DATA_TYPE = torch.float16
    else:
        DATA_TYPE = torch.float16

    cfg = GemmGenConfig(
        m=m,
        k=k,
        n=n,
        batch=batch,
        trans_batch_a=trans_batch_a,
        dtype=DATA_TYPE,
    )

    gen_golden_data(cfg)
