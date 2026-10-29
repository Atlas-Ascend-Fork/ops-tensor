# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Self-tests for the example result verifier; these do not validate an NPU kernel."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import ml_dtypes
    import numpy as np
except ImportError:
    ml_dtypes = None
    np = None


FP8_SIGNED_ZERO_EXPECTED = [0x00, 0x80, 0x38, 0xB8]
FP8_SIGNED_ZERO_ACTUAL = [0x80, 0x00, 0x38, 0xB8]
FP8_MISMATCH_CASES = ((0x38, 0x39), (0x38, 0xB8), (0x00, 0x01))
SIGNS = (-1, 1)
NONFINITE_CODE_CASES = ((0x7F, False), (0xFF, False), (0xFF, True))
FINITE_SCALE_CODES = [0, 1, 127, 254]
SCALE_CODE_NEIGHBOUR = 1
COMPENSATING_OUTPUT_CASES = ((0x38, 0x30, False), (127, 128, True))
INVALID_FILE_SIZE_CASES = (([], []), ([0x38], []), ([0x38], [0x38, 0x38]))
REPO_PARENT_INDEX = 5
REFERENCE_MAGNITUDE = 1.0
WITHIN_TOLERANCE_DELTA = 0.5e-3
OUTSIDE_TOLERANCE_DELTA = 2e-3
TINY_NONZERO_VALUE = 1e-38
E8M0_EXPONENT_BASE = 2.0
E8M0_MIN_EXPONENT = -127
LARGE_VECTOR_ELEMENTS = 20000
CORRUPTED_ELEMENT_VALUE = 2
ALLOWED_BAD_ELEMENTS = 20
EXCESS_BAD_ELEMENTS = ALLOWED_BAD_ELEMENTS + 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_PARENT_INDEX]


@unittest.skipIf(np is None or ml_dtypes is None, "requires numpy and ml_dtypes")
class GmmsqResultVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = REPO_ROOT / "examples/acl/grouped_matmul/scripts/verify_gmmsq_result.py"
        spec = importlib.util.spec_from_file_location("gmmsq_result_verifier", path)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)

    def compare_codes(self, expected, actual, is_scale=False):
        with tempfile.TemporaryDirectory() as folder:
            golden_path = Path(folder) / "golden.bin"
            actual_path = Path(folder) / "actual.bin"
            np.asarray(expected, dtype=np.uint8).tofile(golden_path)
            np.asarray(actual, dtype=np.uint8).tofile(actual_path)
            self.verifier.compare(golden_path, actual_path, "test", is_scale)

    def test_fp8_signed_zero_is_numerically_equal(self):
        self.compare_codes(FP8_SIGNED_ZERO_EXPECTED, FP8_SIGNED_ZERO_ACTUAL)

    def test_fp8_neighbour_and_sign_errors_fail(self):
        for golden, actual in FP8_MISMATCH_CASES:
            with self.subTest(golden=golden, actual=actual):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.compare_codes([golden], [actual])

    def test_relative_tolerance_within_and_outside_in_both_directions(self):
        for sign in SIGNS:
            golden = np.float32(sign)
            within = np.float32(sign * (REFERENCE_MAGNITUDE + WITHIN_TOLERANCE_DELTA))
            outside = np.float32(sign * (REFERENCE_MAGNITUDE + OUTSIDE_TOLERANCE_DELTA))
            with self.subTest(sign=sign):
                for expected, actual in ((golden, within), (within, golden)):
                    self.verifier.compare_values([expected], [actual], "Y")
                for expected, actual in ((golden, outside), (outside, golden)):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        self.verifier.compare_values([expected], [actual], "Y")

    def test_zero_and_tiny_nonzero_do_not_use_absolute_tolerance(self):
        for expected, actual in ((0.0, TINY_NONZERO_VALUE), (TINY_NONZERO_VALUE, 0.0)):
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.verifier.compare_values([expected], [actual], "Y")

    def test_nonfinite_values_always_fail_on_either_side(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for expected, actual in (
                (value, REFERENCE_MAGNITUDE),
                (REFERENCE_MAGNITUDE, value),
                (value, value),
            ):
                with self.subTest(expected=expected, actual=actual):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        self.verifier.compare_values([expected], [actual], "Y")
        for code, is_scale in NONFINITE_CODE_CASES:
            with self.subTest(code=code, is_scale=is_scale):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.compare_codes([code], [code], is_scale)

    def test_scale_zero_code_and_extremes_are_finite_not_zero(self):
        self.compare_codes(FINITE_SCALE_CODES, FINITE_SCALE_CODES, is_scale=True)
        self.assertEqual(
            np.asarray([0], np.uint8).view(ml_dtypes.float8_e8m0fnu)[0],
            np.float32(E8M0_EXPONENT_BASE**E8M0_MIN_EXPONENT),
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.compare_codes([0], [SCALE_CODE_NEIGHBOUR], is_scale=True)

    def test_compensating_y_and_scale_errors_cannot_pass(self):
        # 1 * 1 == 0.5 * 2, but both independent outputs are wrong.
        for golden, actual, is_scale in COMPENSATING_OUTPUT_CASES:
            with self.subTest(is_scale=is_scale):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.compare_codes([golden], [actual], is_scale)

    def test_empty_truncated_and_extended_files_fail(self):
        for golden, actual in INVALID_FILE_SIZE_CASES:
            with self.subTest(golden=golden, actual=actual):
                with self.assertRaisesRegex(ValueError, "empty|size mismatch"):
                    self.compare_codes(golden, actual)

    def test_dual_threshold_allows_at_most_one_per_thousand_bad_elements(self):
        expected = np.ones(LARGE_VECTOR_ELEMENTS, np.float32)
        actual = expected.copy()
        actual[:ALLOWED_BAD_ELEMENTS] = CORRUPTED_ELEMENT_VALUE
        self.verifier.compare_values(expected, actual, "Y")
        actual[:EXCESS_BAD_ELEMENTS] = CORRUPTED_ELEMENT_VALUE
        with self.assertRaisesRegex(ValueError, "errors=21/20000"):
            self.verifier.compare_values(expected, actual, "Y")


if __name__ == "__main__":
    unittest.main()
