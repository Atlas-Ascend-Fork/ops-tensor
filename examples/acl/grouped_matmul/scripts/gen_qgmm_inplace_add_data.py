#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#


"""Generate quantized inputs and references for two consecutive inplace-add calls."""

import argparse
from pathlib import Path

import numpy as np

MX_SCALE_K = 32
MX_SCALE_PAIR = 2
MX_SCALE_PACK_K = MX_SCALE_K * MX_SCALE_PAIR
E8M0_ONE = 127
ENCODINGS = {
    "hif8_tc": (0x08, 0x18, 0x88, 0x98),
    "hif8_tt": (0x08, 0x18, 0x88, 0x98),
    "mxfp8": (0x38, 0x30, 0xB8, 0xB0),
    "mxfp4": (0x2, 0x1, 0xA, 0x9),
}


def encode_input(indices, scenario):
    encoded = np.array(ENCODINGS[scenario], dtype=np.uint8)[indices]
    if scenario == "mxfp4":
        return encoded[:, ::2] | (encoded[:, 1::2] << 4)
    return encoded


def make_mx_scales(rng, k, groups, width):
    rows = k // MX_SCALE_PACK_K + groups
    return rng.integers(
        E8M0_ONE - 1, E8M0_ONE + 2, size=(rows, width, MX_SCALE_PAIR), dtype=np.uint8
    )


def expand_mx_scales(scales, start, group, length):
    row_offset = start // MX_SCALE_PACK_K + group
    row_count = (length + MX_SCALE_PACK_K - 1) // MX_SCALE_PACK_K
    exponents = (
        scales[row_offset : row_offset + row_count]
        .transpose(0, 2, 1)
        .reshape(-1, scales.shape[1])
    )
    exponents = (
        np.repeat(exponents, MX_SCALE_K, axis=0)[:length].astype(np.int32) - E8M0_ONE
    )
    return np.exp2(exponents.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=ENCODINGS, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--group-lengths", required=True)
    parser.add_argument("--group-list-type", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lengths = np.array(
        [int(value) for value in args.group_lengths.split(";")], dtype=np.int64
    )
    if (
        min(args.m, args.n, args.k, args.group_num) <= 0
        or lengths.size != args.group_num
        or np.any(lengths < 0)
        or lengths.sum() != args.k
    ):
        parser.error(
            "Expected positive shapes and nonnegative group lengths summing to K"
        )
    if args.scenario == "mxfp4" and (args.m % 2 or args.n % 2):
        parser.error("MXFP4 requires even M and N for packed input")

    values = np.array([1, 0.5, -1, -0.5], dtype=np.float32)
    rng = np.random.default_rng(2026)
    a = rng.integers(len(values), size=(args.k, args.m))
    b = rng.integers(len(values), size=(args.k, args.n))
    groups = np.arange(args.group_num)
    is_mx = args.scenario in ("mxfp8", "mxfp4")
    if is_mx:
        scale1 = make_mx_scales(rng, args.k, args.group_num, args.m)
        scale2 = make_mx_scales(rng, args.k, args.group_num, args.n)
    else:
        scale1 = (groups + 1).astype(np.float32) / 4
        if args.scenario == "hif8_tc":
            scale2 = ((groups[:, None] + np.arange(args.n)) % 7 + 1).astype(
                np.float32
            ) / 8
        else:
            scale2 = np.exp2((groups % 3 - 1).astype(np.float32))
    initial = rng.choice(
        np.array([-3, -1.5, 2, 4], dtype=np.float32),
        size=(args.group_num, args.m, args.n),
    )
    delta = np.zeros_like(initial)
    ends = np.cumsum(lengths)
    start = 0
    for group, end in enumerate(ends):
        if end > start:
            a_group, b_group = values[a[start:end]], values[b[start:end]]
            if is_mx:
                a_group *= expand_mx_scales(scale1, start, group, end - start)
                b_group *= expand_mx_scales(scale2, start, group, end - start)
                delta[group] = a_group.T @ b_group
            else:
                delta[group] = (a_group.T @ b_group) * scale2[group] * scale1[group]
        start = end
    first = initial + delta
    second = first + delta

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "a": encode_input(a, args.scenario),
        "b": encode_input(b, args.scenario),
        "scale1": scale1,
        "scale2": scale2,
        "group_list": ends if args.group_list_type == 0 else lengths,
        "group_lengths": lengths,
        "y_ref": initial,
        "golden_1": first,
        "golden_2": second,
    }.items():
        array.tofile(args.output_dir / f"{name}.bin")


if __name__ == "__main__":
    main()
