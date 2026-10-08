# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3

import argparse
import os
import numpy as np


def gen_data(m, n, k, batch, trans_batch_a=False, output_dir="./"):
    os.makedirs(output_dir, exist_ok=True)

    if trans_batch_a:
        a = np.random.randint(0, 128, size=(m * batch * k,), dtype=np.uint8)
    else:
        a = np.random.randint(0, 128, size=(batch * m * k,), dtype=np.uint8)
    b = np.random.randint(0, 128, size=(batch * k * n,), dtype=np.uint8)

    # per-channel scale encoded as uint64 (0x3F80000000000000 = 1.0f in fixpipe format)
    scale = np.array([0x3F80000000000000], dtype=np.uint64)

    # per-token scale (x1) as float32 (1.0f), used when x1QuantMode = PERTENSOR
    pertoken_scale = np.array([1.0], dtype=np.float32)

    bias = np.zeros(n, dtype=np.float32)

    a.tofile(os.path.join(output_dir, "input_a.bin"))
    b.tofile(os.path.join(output_dir, "input_b.bin"))
    scale.tofile(os.path.join(output_dir, "scale.bin"))
    pertoken_scale.tofile(os.path.join(output_dir, "pertoken_scale.bin"))
    bias.tofile(os.path.join(output_dir, "bias.bin"))

    print(
        f"Generated data: M={m}, N={n}, K={k}, batch={batch}, trans_batch_a={trans_batch_a}"
    )
    print(f"  input_a.bin: {a.nbytes} bytes")
    print(f"  input_b.bin: {b.nbytes} bytes")
    print(f"  scale.bin: {scale.nbytes} bytes")
    print(f"  pertoken_scale.bin: {pertoken_scale.nbytes} bytes")
    print(f"  bias.bin: {bias.nbytes} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TQBMM fixpipe quant (HIFP8) test data"
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trans_batch_a", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()
    gen_data(args.m, args.n, args.k, args.batch, args.trans_batch_a, args.output_dir)


if __name__ == "__main__":
    main()
