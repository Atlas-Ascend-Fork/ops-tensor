# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Self-tests for the example data generator; these do not validate an NPU kernel."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import ml_dtypes
    import numpy as np
except ImportError:
    ml_dtypes = None
    np = None


OCP_SUBNORMAL_CASES = ((0x0001, 0x00), (0x8001, 0x80), (0x007F, 0x00), (0x807F, 0x80))
CUBLAS_SUBNORMAL_CASES = ((0x0001, 0x08), (0x8001, 0x88))
SIGNED_ZERO_CASES = ((0x0000, 0x00), (0x8000, 0x80))
EMPTY_GROUP_COUNT = 3
EMPTY_GROUP_TOTAL_M = 2
VALID_GROUP_OFFSETS = [0, 2, 2]
VALID_GROUP_LENGTHS = [0, 2, 0]
INVALID_GROUP_COUNT = 2
INVALID_GROUP_TOTAL_M = 2
REPO_PARENT_INDEX = 5
GOLDEN_ROW_COUNT = 1
GOLDEN_COLUMNS = 64
SCALE_PAIR_SIZE = 2
SWIGLU_SPLIT_FACTOR = 2
SCALE_ALG_CUBLAS = 1
GROUP_LIST_ELEMENT_BYTES = 8
CEIL_DIV_ADJUSTMENT = 1
MIN_NORMAL_BF16 = 0x0080
EXPECTED_MIN_NORMAL_FP8 = 0x40
REPO_ROOT = Path(__file__).resolve().parents[REPO_PARENT_INDEX]
EXAMPLE_DIR = (
    REPO_ROOT / "examples/acl/grouped_matmul/quant_grouped_matmul_swiglu_quant_mx"
)


@unittest.skipIf(np is None or ml_dtypes is None, "requires numpy and ml_dtypes")
class GmmsqExampleGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = REPO_ROOT / "examples/acl/grouped_matmul/scripts/gen_gmmsq_data.py"
        spec = importlib.util.spec_from_file_location("gmmsq_example_generator", path)
        cls.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.generator)

    @staticmethod
    def values(bits):
        return np.full((GOLDEN_ROW_COUNT, GOLDEN_COLUMNS), bits, np.uint16).view(
            ml_dtypes.bfloat16
        )

    @staticmethod
    def args(output_dir, **overrides):
        args = dict(
            group_num=2,
            m=35,
            n=192,
            k=68,
            layout_b="nz",
            scale_alg=0,
            clamp_limit=7.0,
            glu_alpha=1.0,
            glu_bias=1.0,
            dst_type_max=0.0,
            group_list="1;35",
            output_dir=output_dir,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_ocp_subnormal_group_zero_reciprocal_preserves_sign(self):
        for bits, expected in OCP_SUBNORMAL_CASES:
            with self.subTest(bits=bits):
                y, scale = self.generator.quantize_golden(self.values(bits), 0)
                np.testing.assert_array_equal(
                    y, np.full((GOLDEN_ROW_COUNT, GOLDEN_COLUMNS), expected, np.uint8)
                )
                np.testing.assert_array_equal(
                    scale,
                    np.zeros(
                        (GOLDEN_ROW_COUNT, GOLDEN_ROW_COUNT, SCALE_PAIR_SIZE), np.uint8
                    ),
                )

    def test_cublas_subnormal_group_remains_nonzero(self):
        for bits, expected in CUBLAS_SUBNORMAL_CASES:
            with self.subTest(bits=bits):
                y, scale = self.generator.quantize_golden(
                    self.values(bits), SCALE_ALG_CUBLAS
                )
                np.testing.assert_array_equal(
                    y, np.full((GOLDEN_ROW_COUNT, GOLDEN_COLUMNS), expected, np.uint8)
                )
                np.testing.assert_array_equal(
                    scale,
                    np.zeros(
                        (GOLDEN_ROW_COUNT, GOLDEN_ROW_COUNT, SCALE_PAIR_SIZE), np.uint8
                    ),
                )

    def test_ocp_minimum_normal_is_not_cleared(self):
        y, scale = self.generator.quantize_golden(self.values(MIN_NORMAL_BF16), 0)
        np.testing.assert_array_equal(
            y,
            np.full(
                (GOLDEN_ROW_COUNT, GOLDEN_COLUMNS), EXPECTED_MIN_NORMAL_FP8, np.uint8
            ),
        )
        np.testing.assert_array_equal(
            scale,
            np.zeros((GOLDEN_ROW_COUNT, GOLDEN_ROW_COUNT, SCALE_PAIR_SIZE), np.uint8),
        )

    def test_zero_groups_preserve_sign_for_both_algorithms(self):
        for algorithm in (0, SCALE_ALG_CUBLAS):
            for bits, expected in SIGNED_ZERO_CASES:
                with self.subTest(algorithm=algorithm, bits=bits):
                    y, scale = self.generator.quantize_golden(
                        self.values(bits), algorithm
                    )
                    self.assertTrue(np.all(y == expected))
                    self.assertTrue(np.all(scale == 0))

    def test_group_offsets_allow_empty_groups(self):
        offsets, lengths = self.generator.parse_group_offsets(
            "0;2;2", EMPTY_GROUP_COUNT, EMPTY_GROUP_TOTAL_M
        )
        np.testing.assert_array_equal(offsets, VALID_GROUP_OFFSETS)
        np.testing.assert_array_equal(lengths, VALID_GROUP_LENGTHS)

    def test_invalid_group_offsets_are_rejected(self):
        for text in ("-1;2", "3;2", "0;1", "0;2;2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.generator.parse_group_offsets(
                    text, INVALID_GROUP_COUNT, INVALID_GROUP_TOTAL_M
                )

    def test_invalid_attributes_are_rejected_before_writing(self):
        invalid = (
            {"clamp_limit": 0.0},
            {"clamp_limit": float("nan")},
            {"clamp_limit": 1e-100},
            {"glu_alpha": float("inf")},
            {"glu_bias": -float("inf")},
            {"glu_alpha": 1e40},
            {"dst_type_max": 1.0},
            {"scale_alg": 2},
            {"group_num": 9},
        )
        with tempfile.TemporaryDirectory() as folder:
            for override in invalid:
                output = Path(folder) / "must_not_be_created"
                with self.subTest(override=override), self.assertRaises(ValueError):
                    self.generator.generate(self.args(str(output), **override))
                self.assertFalse(output.exists())

    def test_both_csv_cases_generate_complete_outputs(self):
        with (EXAMPLE_DIR / "quant_grouped_matmul_swiglu_quant_mx.csv").open(
            newline=""
        ) as cases:
            for row in csv.DictReader(cases):
                with (
                    self.subTest(case=row["caseName"]),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    args = self.args(
                        folder,
                        group_num=int(row["groupNum"]),
                        m=int(row["m"]),
                        n=int(row["n"]),
                        k=int(row["k"]),
                        layout_b=row["layoutB"],
                        scale_alg=int(row["scaleAlg"]),
                        clamp_limit=float(row["clampLimit"]),
                        glu_alpha=float(row["gluAlpha"]),
                        glu_bias=float(row["gluBias"]),
                        dst_type_max=float(row["dstTypeMax"]),
                        group_list=row["groupList"],
                    )
                    self.generator.generate(args)
                    output_n = args.n // SWIGLU_SPLIT_FACTOR
                    self.assertEqual(
                        (Path(folder) / "golden_y.bin").stat().st_size,
                        args.m * output_n,
                    )
                    self.assertEqual(
                        (Path(folder) / "golden_y_scale.bin").stat().st_size,
                        args.m
                        * (
                            (output_n + GOLDEN_COLUMNS - CEIL_DIV_ADJUSTMENT)
                            // GOLDEN_COLUMNS
                        )
                        * SCALE_PAIR_SIZE,
                    )
                    self.assertEqual(
                        (Path(folder) / "group_list.bin").stat().st_size,
                        args.group_num * GROUP_LIST_ELEMENT_BYTES,
                    )


if __name__ == "__main__":
    unittest.main()
