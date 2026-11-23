# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Keep the merged V3 PR's final review fixes when reorganizing MX implementations."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4] / "include/blaze"


class V3ReviewContractTest(unittest.TestCase):
    def setUp(self):
        self.block = (ROOT / "gemm/block/block_mmad_weight_prologue_mx.h").read_text(
            encoding="utf-8"
        )
        self.kernel = (
            ROOT / "gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
        ).read_text(encoding="utf-8")

    def test_aggregate_header_and_direct_dependencies(self):
        aggregate = (ROOT / "gemm/tile/compute.h").read_text(encoding="utf-8")
        self.assertNotIn("__aicore__", aggregate)  # 201445126
        for source in [self.block, self.kernel]:
            self.assertIn('#include "blaze/gemm/tile/compute.h"', source)
            self.assertNotIn(
                '#include "blaze/gemm/tile/arch35/shift_w4_to_w8.h"', source
            )  # 201826295, 201836787
        self.assertLess(self.block.index("\\brief"), self.block.index("#pragma once"))

    def test_specialized_layout_and_k_policy_have_local_owners(self):
        common = (ROOT / "gemm/utils/common_utils.h").read_text(encoding="utf-8")
        self.assertNotIn("MxWeightQuantGmmL1Layout", common)  # 201840343
        self.assertIn("struct MxWeightQuantGmmL1Layout", self.block)
        self.assertNotIn("CalcWeightQuantMxKBlock", common)  # 201840689
        position = self.block.index("void CalcWeightQuantMxKBlock(")
        self.assertGreater(
            self.block.rfind("private:", 0, position),
            self.block.rfind("public:", 0, position),
        )

    def test_continuous_conversion_and_mmad_helpers_remain_private(self):
        shift = (ROOT / "gemm/tile/arch35/shift_w4_to_w8.h").read_text(encoding="utf-8")
        self.assertNotIn("ConvertContinuousPair", shift)  # 201843236
        self.assertNotIn("ContinuousWeightParams", shift)
        for name in ["ShiftW4ToW8PairVf", "MakeContinuousWeightParams"]:
            position = self.kernel.index(
                name + "(",
                self.kernel.index(
                    "private:", self.kernel.index("class GroupedWeightPrologueMx")
                ),
            )
            self.assertGreater(
                self.kernel.rfind("private:", 0, position),
                self.kernel.rfind("public:", 0, position),
            )
        for name in ["MakeMxL0Operand", "CopyMxScalesToL1", "IssueMxMmad"]:
            self.assertNotIn(name, shift)  # 201856779
            position = self.block.index(name + "(")
            self.assertGreater(
                self.block.rfind("private:", 0, position),
                self.block.rfind("public:", 0, position),
            )

    def test_sync_wrapper_preserves_both_flags_and_their_order(self):
        self.assertNotIn("CrossCoreWaitFlag", self.block)  # 201825665
        self.assertNotIn("CrossCoreSetFlag", self.block)
        for method, operation, flag in [
            ("WaitAivToAic", "WaitForVector", "SYNC_AIC_AIV_FLAG"),
            ("SetAicToAiv", "NotifyVector", "SYNC_AIV_AIC_FLAG"),
        ]:
            start = self.block.index("::" + method + "()")
            body = self.block[start : self.block.index("\n}", start)]
            prefix = f"Sync::{operation}<WQGMM_MX_SYNC_MODE, PIPE_MTE1>"
            self.assertLess(
                body.index(prefix + f"({flag} + WQGMM_MX_FLAG_ID_MAX)"),
                body.index(prefix + f"({flag})"),
            )

    def test_fast_option_explains_scope_without_changing_default(self):
        epilogue = (ROOT / "epilogue/block/block_epilogue_swiglu_mx_quant.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("bool A8W4V3Fast_ = false", epilogue)
        self.assertIn("not a per-shape speed guarantee", epilogue)  # 201851577
        self.assertIn("default false", epilogue)


if __name__ == "__main__":
    unittest.main()
