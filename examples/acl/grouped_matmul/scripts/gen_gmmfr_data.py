#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

"""Generate GroupedMatmulFinalizeRouting MX WeightNZ inputs and CPU golden."""

import argparse
import os

import ml_dtypes
import numpy as np


MX_SCALE_ONE = np.uint8(0x7F)
FP4_PACK_FACTOR = 2
NZ_K0 = 16
NZ_N0 = {"mxfp8_e4m3": 32, "mxfp4_e2m1": 64}
OUTPUT_DTYPES = {
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}
LOGIT_DTYPES = {
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
}
ROW_INDEX_DTYPES = {"int32": np.int32, "int64": np.int64}
TYPE_INFO = {
    "mxfp8_e4m3": {
        "codes": np.asarray([0x30, 0x38, 0xB0, 0xB8], dtype=np.uint8),
        "values": np.asarray([0.5, 1.0, -0.5, -1.0], dtype=np.float32),
    },
    "mxfp4_e2m1": {
        "codes": np.asarray([0x1, 0x2, 0x9, 0xA], dtype=np.uint8),
        "values": np.asarray([0.5, 1.0, -0.5, -1.0], dtype=np.float32),
    },
}


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def pack_fp4(codes):
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if codes.size % FP4_PACK_FACTOR != 0:
        raise ValueError("FP4 data must contain an even number of elements")
    return (codes[0::2] & 0x0F) | ((codes[1::2] & 0x0F) << 4)


def format_weight_nz(codes, dtype):
    k, n = codes.shape
    n0 = NZ_N0[dtype]
    padded = np.zeros((align_up(k, NZ_K0), align_up(n, n0)), dtype=np.uint8)
    padded[:k, :n] = codes
    storage = padded.reshape(padded.shape[0] // NZ_K0, NZ_K0, padded.shape[1] // n0, n0)
    storage = storage.transpose(2, 0, 1, 3).reshape(-1)
    return pack_fp4(storage) if dtype.startswith("mxfp4") else storage


def parse_group_list(text, group_num, total_m, group_list_type):
    values = np.asarray([int(item) for item in text.split(";")], dtype=np.int64)
    if values.size != group_num:
        raise ValueError("groupList must contain groupNum entries")
    if group_list_type == 0:
        if (
            np.any(values < 0)
            or np.any(np.diff(values) < 0)
            or int(values[-1]) != total_m
        ):
            raise ValueError("offset groupList must be nondecreasing and end at totalM")
        lengths = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), values)))
    elif group_list_type == 1:
        if np.any(values <= 0) or int(values.sum()) != total_m:
            raise ValueError("length groupList must be positive and sum to totalM")
        lengths = values
    else:
        raise ValueError("GMMFR example supports offset and length groupList only")
    return values, lengths


def encode_values(indices, dtype):
    info = TYPE_INFO[dtype]
    codes = info["codes"][indices]
    return pack_fp4(codes) if dtype.startswith("mxfp4") else codes.reshape(-1)


def scatter_add_bfloat16(output, row_indices, values):
    """Match BF16 atomic add by rounding after every routed row."""
    for source_row, output_row in enumerate(row_indices):
        output[output_row] = (
            output[output_row].astype(np.float32)
            + values[source_row].astype(np.float32)
        ).astype(ml_dtypes.bfloat16)


def generate(args):
    if args.dtype not in TYPE_INFO:
        raise ValueError(f"unsupported dtype: {args.dtype}")
    if args.output_dtype not in OUTPUT_DTYPES:
        raise ValueError(f"unsupported output dtype: {args.output_dtype}")
    if args.logit_dtype not in LOGIT_DTYPES:
        raise ValueError(f"unsupported logit dtype: {args.logit_dtype}")
    if args.row_index_dtype not in ROW_INDEX_DTYPES:
        raise ValueError(f"unsupported row index dtype: {args.row_index_dtype}")
    if (
        args.group_num <= 0
        or args.total_m <= 0
        or args.batch <= 0
        or args.n <= 0
        or args.k <= 0
    ):
        raise ValueError("groupNum, totalM, batch, n and k must be positive")
    if args.dtype.startswith("mxfp4") and (args.k % 2 != 0 or args.k == 2):
        raise ValueError("MXFP4 requires even k and k != 2")

    group_list, group_lengths = parse_group_list(
        args.group_list, args.group_num, args.total_m, args.group_list_type
    )
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    info = TYPE_INFO[args.dtype]
    indices_x = rng.integers(0, info["codes"].size, size=(args.total_m, args.k))
    x_values = info["values"][indices_x]
    scale_k = (args.k + 63) // 64 * 2

    weights = []
    weight_values = []
    for group_index in range(args.group_num):
        indices_w = rng.integers(0, info["codes"].size, size=(args.k, args.n))
        weights.append(format_weight_nz(info["codes"][indices_w], args.dtype))
        weight_values.append(info["values"][indices_w])

    x_bytes = encode_values(indices_x, args.dtype)
    x_bytes.tofile(os.path.join(args.output_dir, "input_x.bin"))
    np.full(args.total_m * scale_k, MX_SCALE_ONE, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "input_scale_x.bin")
    )
    for group_index, weight in enumerate(weights):
        weight.tofile(os.path.join(args.output_dir, f"input_weight_{group_index}.bin"))
    np.concatenate(weights).tofile(os.path.join(args.output_dir, "input_weight.bin"))
    np.full(args.group_num * args.n * scale_k, MX_SCALE_ONE, dtype=np.uint8).tofile(
        os.path.join(args.output_dir, "input_scale_weight.bin")
    )

    np.zeros(args.group_num * args.n, dtype=ml_dtypes.bfloat16).tofile(
        os.path.join(args.output_dir, "input_bias.bin")
    )
    np.zeros(args.batch * args.n, dtype=ml_dtypes.bfloat16).tofile(
        os.path.join(args.output_dir, "input_shared.bin")
    )
    group_list.tofile(os.path.join(args.output_dir, "input_group_list.bin"))

    row_index = np.arange(args.total_m, dtype=np.int64) % args.batch
    row_index.astype(ROW_INDEX_DTYPES[args.row_index_dtype]).tofile(
        os.path.join(args.output_dir, "input_row_index.bin")
    )
    logit_values = np.where(np.arange(args.total_m) % 3 == 0, 0.5, 1.0).astype(
        np.float32
    )
    logit_values.astype(LOGIT_DTYPES[args.logit_dtype]).tofile(
        os.path.join(args.output_dir, "input_logit.bin")
    )

    is_bfloat16_output = args.output_dtype == "bfloat16"
    golden_dtype = OUTPUT_DTYPES[args.output_dtype]
    golden = np.zeros((args.batch, args.n), dtype=golden_dtype)
    offset = 0
    logit_for_compute = logit_values.astype(LOGIT_DTYPES[args.logit_dtype])
    if is_bfloat16_output:
        logit_for_compute = logit_for_compute.astype(ml_dtypes.bfloat16)
    else:
        logit_for_compute = logit_for_compute.astype(np.float32)
    for group_index, group_m in enumerate(group_lengths.tolist()):
        end = offset + group_m
        gmm_output = x_values[offset:end] @ weight_values[group_index]
        if is_bfloat16_output:
            gmm_output = gmm_output.astype(ml_dtypes.bfloat16)
            weighted_output = (gmm_output * logit_for_compute[offset:end, None]).astype(
                ml_dtypes.bfloat16
            )
            scatter_add_bfloat16(golden, row_index[offset:end], weighted_output)
        else:
            np.add.at(
                golden,
                row_index[offset:end],
                gmm_output * logit_for_compute[offset:end, None],
            )
        offset = end

    golden.astype(golden_dtype).tofile(os.path.join(args.output_dir, "golden_y.bin"))
    print(
        f"[SUCCESS] generated GMMFR MX data: dtype={args.dtype}, output={args.output_dtype}, "
        f"logit={args.logit_dtype}, row_index={args.row_index_dtype}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GMMFR MX WeightNZ inputs and golden data"
    )
    parser.add_argument("--group-num", type=int, required=True)
    parser.add_argument("--total-m", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(TYPE_INFO), required=True)
    parser.add_argument("--output-dtype", choices=tuple(OUTPUT_DTYPES), required=True)
    parser.add_argument("--logit-dtype", choices=tuple(LOGIT_DTYPES), required=True)
    parser.add_argument(
        "--row-index-dtype", choices=tuple(ROW_INDEX_DTYPES), required=True
    )
    parser.add_argument("--group-list-type", type=int, choices=(0, 1), required=True)
    parser.add_argument("--group-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
