# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only CLI regressions using synthetic FlatQuant binary fixtures."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


MATRIX_ROWS = 16
MATRIX_COLUMNS = 16
MATRIX_ELEMENTS = MATRIX_ROWS * MATRIX_COLUMNS
GROUP_SIZE = 32
NONFINITE_SCALE = 253
HALF_SCALE = 124
VERIFIER = Path(__file__).resolve().parents[1] / "scripts" / "verify_result.py"


class FlatQuantResultVerifierTest(unittest.TestCase):
    @staticmethod
    def _run_cli(reference, packed=None, scales=None):
        batches = reference.shape[0]
        if packed is None:
            packed = np.zeros(batches * MATRIX_ELEMENTS // 2, dtype=np.uint8)
        if scales is None:
            scales = np.zeros(batches * MATRIX_ELEMENTS // GROUP_SIZE, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            golden_dir = Path(folder) / "golden"
            actual_dir = Path(folder) / "actual"
            golden_dir.mkdir()
            actual_dir.mkdir()
            reference.tofile(golden_dir / "golden_ref.bin")
            packed.tofile(actual_dir / "npu_y.bin")
            scales.tofile(actual_dir / "npu_scale.bin")
            command = [
                sys.executable,
                str(VERIFIER),
                "--golden-dir",
                str(golden_dir),
                "--actual-dir",
                str(actual_dir),
                "--k",
                str(batches),
                "--m",
                str(MATRIX_ROWS),
                "--n",
                str(MATRIX_COLUMNS),
                "--iter-batch",
                "1",
                "--has-p2",
                "false",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            metrics_file = actual_dir / "verify_metrics.json"
            metrics = (
                json.loads(metrics_file.read_text()) if metrics_file.exists() else None
            )
            return result, metrics

    def _assert_nonfinite_rejected(self, reference, packed=None, scales=None):
        result, metrics = self._run_cli(reference, packed, scales)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Golden reference contains non-finite values", result.stderr)
        self.assertNotIn("[PASS]", result.stdout)
        self.assertIsNone(metrics)

    def test_single_nan_reference_is_rejected(self):
        reference = np.zeros((1, MATRIX_ROWS, MATRIX_COLUMNS), dtype=np.float32)
        reference.flat[0] = np.nan
        scales = np.zeros(MATRIX_ELEMENTS // GROUP_SIZE, dtype=np.uint8)
        scales[0] = NONFINITE_SCALE
        self._assert_nonfinite_rejected(reference, scales=scales)

    def test_nan_in_later_batch_is_rejected(self):
        reference = np.zeros((2, MATRIX_ROWS, MATRIX_COLUMNS), dtype=np.float32)
        reference.flat[-1] = np.nan
        scales = np.zeros(2 * MATRIX_ELEMENTS // GROUP_SIZE, dtype=np.uint8)
        scales[-1] = NONFINITE_SCALE
        self._assert_nonfinite_rejected(reference, scales=scales)

    def test_all_nan_reference_is_rejected(self):
        reference = np.full((1, MATRIX_ROWS, MATRIX_COLUMNS), np.nan, dtype=np.float32)
        scales = np.full(MATRIX_ELEMENTS // GROUP_SIZE, NONFINITE_SCALE, dtype=np.uint8)
        self._assert_nonfinite_rejected(reference, scales=scales)

    def test_infinite_reference_groups_are_rejected(self):
        for value, packed_byte in ((np.inf, 0x77), (-np.inf, 0xFF)):
            with self.subTest(value=value):
                reference = np.zeros((1, MATRIX_ROWS, MATRIX_COLUMNS), dtype=np.float32)
                reference.reshape(-1)[:GROUP_SIZE] = value
                packed = np.zeros(MATRIX_ELEMENTS // 2, dtype=np.uint8)
                # E2M1 nibbles 7/15 are +/-6; +/-6 * 2**126 overflows float32.
                packed[: GROUP_SIZE // 2] = packed_byte
                scales = np.zeros(MATRIX_ELEMENTS // GROUP_SIZE, dtype=np.uint8)
                scales[0] = NONFINITE_SCALE
                self._assert_nonfinite_rejected(reference, packed, scales)

    def test_zero_reference_passes_multiple_batches(self):
        reference = np.zeros((2, MATRIX_ROWS, MATRIX_COLUMNS), dtype=np.float32)
        result, metrics = self._run_cli(reference)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(metrics["overall_status"], "pass")
        self.assertTrue(
            all(output["status"] == "pass" for output in metrics["outputs"])
        )

    def test_exact_finite_positive_and_negative_reference_pass(self):
        # E2M1 +/-4 (nibbles 6/14), with scale byte 124, represents +/-0.5.
        for value, packed_byte in ((0.5, 0x66), (-0.5, 0xEE)):
            with self.subTest(value=value):
                reference = np.full(
                    (1, MATRIX_ROWS, MATRIX_COLUMNS), value, dtype=np.float32
                )
                packed = np.full(MATRIX_ELEMENTS // 2, packed_byte, dtype=np.uint8)
                scales = np.full(
                    MATRIX_ELEMENTS // GROUP_SIZE, HALF_SCALE, dtype=np.uint8
                )
                result, metrics = self._run_cli(reference, packed, scales)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(metrics["overall_status"], "pass")

    def test_mismatched_finite_output_still_fails(self):
        reference = np.full((1, MATRIX_ROWS, MATRIX_COLUMNS), 0.5, dtype=np.float32)
        scales = np.full(MATRIX_ELEMENTS // GROUP_SIZE, HALF_SCALE, dtype=np.uint8)
        result, metrics = self._run_cli(reference, scales=scales)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(metrics["overall_status"], "fail")
        self.assertEqual(metrics["outputs"][0]["error_ratio"], 1.0)
        self.assertEqual(metrics["outputs"][1]["error_ratio"], 0.0)

    def test_wrong_output_lengths_still_fail(self):
        reference = np.zeros((1, MATRIX_ROWS, MATRIX_COLUMNS), dtype=np.float32)
        cases = (
            (np.zeros(MATRIX_ELEMENTS // 2 - 1, dtype=np.uint8), None, "NPU y size"),
            (
                None,
                np.zeros(MATRIX_ELEMENTS // GROUP_SIZE - 1, dtype=np.uint8),
                "NPU scale size",
            ),
        )
        for packed, scales, message in cases:
            with self.subTest(message=message):
                result, metrics = self._run_cli(reference, packed, scales)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertIsNone(metrics)


if __name__ == "__main__":
    unittest.main()
