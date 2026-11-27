#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Make input files and exact expected bytes for the signed-diagonal rotation example."""

import argparse
import struct
from pathlib import Path

M, N, K = 35, 544, 32
LEVEL0_COLS, LEVEL1_COLS = 512, 32
SENTINEL = 0xA5
AMPLITUDES = (1.0, -0.5, 0.25, -1.0)
# Encodings for 1/4, 1/2 and 1 times the maximum finite value in each output format.
TARGETS = {
    "fp4": (6, 0x08, {0.25: 0x03, 0.5: 0x05, 1.0: 0x07}),
    "e4m3": (448, 0x80, {0.25: 0x6E, 0.5: 0x76, 1.0: 0x7E}),
    "e5m2": (57344, 0x80, {0.25: 0x73, 0.5: 0x77, 1.0: 0x7B}),
}
GROUP_LISTS = {
    "ordinary": (),
    "count": (17, 0, 17),
    "cumsum": (17, 17, 34),
    "empty_groups": (0, 0, 0),
}


def encode_input(values, dtype):
    if dtype == "float16":
        return struct.pack(f"<{len(values)}e", *values)
    # All fixture values are exactly representable in BF16.
    return b"".join(struct.pack("<f", value)[2:] for value in values)


def rotation_sign(group, block, lane):
    return -1 if ((lane + group) % 3 == 0) != (block % 2 == 1) else 1


def row_groups(mode):
    entries = GROUP_LISTS[mode]
    if mode == "ordinary":
        return [0] * M
    if mode == "cumsum":
        entries = tuple(end - start for start, end in zip((0,) + entries, entries))
    groups = [group for group, count in enumerate(entries) for _ in range(count)]
    return groups + [None] * (M - len(groups))


def generate(args):
    per_group = 1 if args.rotations == "shared" else N // K
    groups = GROUP_LISTS[args.mode]
    active_groups = row_groups(args.mode)
    cmax, sign_mask, positive_codes = TARGETS[args.output_dtype]
    input_base = 6.0 if args.output_dtype == "fp4" else 7.0
    clamp_ratio = 0.5 if args.clamp == 1 else 1.0
    fp4 = args.output_dtype == "fp4"
    level0_count = (N + LEVEL0_COLS - 1) // LEVEL0_COLS
    level1_groups = N // LEVEL1_COLS
    level1_stride = ((level1_groups + 1) // 2) * 2
    y_stride = N // 2 if fp4 else N
    y = bytearray([SENTINEL]) * (M * y_stride)
    level0 = bytearray([SENTINEL]) * (M * level0_count * 4)
    level1 = bytearray([SENTINEL]) * (M * level1_stride)
    x = []
    rotation = []

    for group in range(max(1, len(groups))):
        for block in range(per_group):
            rotation.extend(
                rotation_sign(group, block, i) if i == j else 0.0
                for i in range(K)
                for j in range(K)
            )

    for row, group in enumerate(active_groups):
        for col in range(N):
            block0 = col // LEVEL0_COLS
            amplitude = input_base * (2 ** ((row + block0) % 2))
            value = amplitude * AMPLITUDES[col % len(AMPLITUDES)]
            x.append(value)
            if group is None:
                continue

            if col % LEVEL0_COLS == 0:
                scale0 = amplitude * clamp_ratio / cmax
                struct.pack_into(
                    "<f", level0, (row * level0_count + block0) * 4, scale0
                )
            # Each 32-wide group reaches the level0-normalized target maximum.
            # FP4 scaleAlg=0 and FP8 scaleAlg=1 both give E8M0 scale 1 for this fixture.
            block = 0 if per_group == 1 else col // K
            signed = value * rotation_sign(group, block, col % K)
            magnitude = min(abs(signed) / amplitude, clamp_ratio) / clamp_ratio
            code = positive_codes[magnitude] | (sign_mask if signed < 0 else 0)
            if fp4:
                index = row * y_stride + col // 2
                y[index] = code if col % 2 == 0 else y[index] | (code << 4)
            else:
                y[row * y_stride + col] = code

        if group is not None:
            start = row * level1_stride
            level1[start : start + level1_groups] = bytes([127]) * level1_groups
            level1[start + level1_groups : start + level1_stride] = bytes(
                level1_stride - level1_groups
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_x.bin").write_bytes(encode_input(x, args.input_dtype))
    (output / "input_rotation.bin").write_bytes(
        encode_input(rotation, args.input_dtype)
    )
    (output / "input_group_list.bin").write_bytes(
        struct.pack(f"<{len(groups)}q", *groups)
    )
    (output / "golden_y.bin").write_bytes(y)
    (output / "golden_level0_scale.bin").write_bytes(level0)
    (output / "golden_level1_scale.bin").write_bytes(level1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=GROUP_LISTS)
    parser.add_argument("input_dtype", choices=("float16", "bfloat16"))
    parser.add_argument("output_dtype", choices=TARGETS)
    parser.add_argument("rotations", choices=("shared", "per_block"))
    parser.add_argument("clamp", type=int, choices=(0, 1))
    parser.add_argument("output_dir", type=Path)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
