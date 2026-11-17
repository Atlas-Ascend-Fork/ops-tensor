#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only tests of example serialization, staged golden and failure reporting; no NPU execution."""

import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gen_grouped_weight_antiquant import generate, dequantize, weight_dtype
from verify_grouped_weight_antiquant import verify


class WeightAntiquantExampleData(unittest.TestCase):
    def test_all_supported_configs_and_physical_transpose(self):
        configs = 0
        with tempfile.TemporaryDirectory() as temp:
            for x in ("float16", "bfloat16"):
                for w in ("int8", "float8_e4m3fn", "float8_e5m2", "hifloat8"):
                    for offset in (0, 1) if w == "int8" else (0,):
                        for bias in ("none", x) + (
                            ("float32",) if x == "bfloat16" else ()
                        ):
                            outputs = []
                            weights = []
                            for trans in (0, 1):
                                directory = Path(temp) / str(trans)
                                args = argparse.Namespace(
                                    groups=3,
                                    m=6,
                                    k=17,
                                    n=19,
                                    x_dtype=x,
                                    weight_dtype=w,
                                    bias_dtype=bias,
                                    trans=trans,
                                    offset=offset,
                                    list_type=trans,
                                    group_list="0;2;4" if trans else "0;2;6",
                                    seed=73,
                                    output_dir=str(directory),
                                )
                                generate(args)
                                outputs.append(
                                    (directory / "golden_y.bin").read_bytes()
                                )
                                dtype = weight_dtype(w)
                                shape = (3, 19, 17) if trans else (3, 17, 19)
                                physical = np.fromfile(
                                    directory / "weight.bin", dtype=dtype
                                ).reshape(shape)
                                weights.append(
                                    np.ascontiguousarray(
                                        physical.transpose(0, 2, 1)
                                        if trans
                                        else physical
                                    ).tobytes()
                                )
                                self.assertEqual(len(outputs[-1]), 6 * 19 * 2)
                                self.assertEqual(
                                    (directory / "golden_weight.bin").stat().st_size,
                                    3 * 17 * 19 * 2,
                                )
                                with contextlib.redirect_stdout(io.StringIO()):
                                    verify(
                                        directory / "golden_y.bin",
                                        directory / "golden_y.bin",
                                        6,
                                        19,
                                        x,
                                        0,
                                        0,
                                    )
                                configs += 1
                            self.assertEqual(outputs[0], outputs[1])
                            self.assertEqual(weights[0], weights[1])
        self.assertEqual(configs, 50)

    def test_add_and_mul_round_independently(self):
        # Broad finite grid includes values for which fused FP32 affine is different.
        for dtype in (np.float16, ml_dtypes.bfloat16):
            w = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(1, 256, 1)
            offsets = np.linspace(-0.49, 0.49, 31).astype(dtype).reshape(1, 31)
            scales = np.linspace(0.03, 0.12, 31).astype(dtype).reshape(1, 31)
            staged = dequantize(w, scales, offsets, "int8", dtype)
            fused = (
                (w.astype(np.float32) + offsets.astype(np.float32)[:, None, :])
                * scales.astype(np.float32)[:, None, :]
            ).astype(dtype)
            self.assertTrue(
                np.any(staged.astype(np.float32) != fused.astype(np.float32))
            )

    def test_verifier_rejects_single_element_corruption_nonfinite_and_size(self):
        with tempfile.TemporaryDirectory() as temp:
            golden, actual = Path(temp) / "golden.bin", Path(temp) / "actual.bin"
            np.ones((3, 5), dtype=np.float16).tofile(golden)
            for bad in (2.0, float("nan"), float("inf")):
                values = np.ones(15, dtype=np.float16)
                values[7] = bad
                values.tofile(actual)
                with (
                    self.assertRaises(ValueError),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    verify(golden, actual, 3, 5, "float16", 0.003, 0.003)
            actual.write_bytes(golden.read_bytes() + b"\x00")
            with self.assertRaises(ValueError):
                verify(golden, actual, 3, 5, "float16", 0.003, 0.003)

    def test_invalid_groups_and_fp8_offset_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                groups=3,
                m=6,
                k=17,
                n=19,
                x_dtype="float16",
                weight_dtype="int8",
                bias_dtype="none",
                trans=0,
                offset=0,
                list_type=0,
                group_list="0;4;3",
                seed=73,
                output_dir=temp,
            )
            with self.assertRaises(ValueError):
                generate(args)
            args.weight_dtype, args.offset = "float8_e4m3fn", 1
            with self.assertRaises(ValueError):
                generate(args)


if __name__ == "__main__":
    unittest.main()
