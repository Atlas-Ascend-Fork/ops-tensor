# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only metadata checks. These do not validate device compilation or NPU results."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
POLICY = ROOT / "include/blaze/gemm/policy/dispatch_policy.h"
BLOCK = ROOT / "include/blaze/gemm/block/block_mmad_weight_prologue_mx.h"
KERNEL = ROOT / "include/blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"


def metadata_source():
    policy = POLICY.read_text(encoding="utf-8")
    policy = policy[
        policy.index("enum class MxPipelineMode") : policy.index("// Finalize routing")
    ]
    block = BLOCK.read_text(encoding="utf-8")
    start = block.index("template <uint32_t Mode_, class AType, class BiasType>")
    config = block[start : block.index("namespace Block {", start)]
    # The trailing namespace is already open in the source prefix below.
    prefix = r"""
#include <cassert>
#include <cstdint>
#include <type_traits>
#define __aicore__
namespace AscendC { using TEventID = uint16_t; namespace Std = std; constexpr uint64_t BLOCK_CUBE=16; }
namespace asc { namespace te {
template <class T> constexpr uint64_t c0_element = 32 / sizeof(T);
struct copy_gm_to_l1 {}; struct copy_l1_to_l0a {}; struct copy_l1_to_l0b {};
struct copy_l1_to_biastable {}; struct copy_l0c_to_gm {};
template <class T> constexpr T make_copy(T value) { return value; }
} }
namespace Blaze { namespace Gemm {
constexpr uint64_t DOUBLE_BUFFER_COUNT=2, TRIPLE_BUFFER_COUNT=3, QUADRUPLE_BUFFER_COUNT=4;
struct KernelWqgmmMxMix {};
enum class WeightPrologueMxKSchedule { FIXED, DYNAMIC };
enum class WeightPrologueMxScaleBSource { AIC, AIV };
"""
    return prefix + policy + config


class PipelineContractTest(unittest.TestCase):
    def test_policy_is_separate_from_resource_layout(self):
        policy = POLICY.read_text(encoding="utf-8")
        block = BLOCK.read_text(encoding="utf-8")
        self.assertNotIn("struct MxWeightPipelinePolicy", policy)
        self.assertNotIn("enum class MxPipelineMode", block)
        self.assertIn("struct MxWeightPipelineConfig {", block)
        self.assertNotIn("struct Producer :", block)
        self.assertNotIn("struct Consumer :", block)
        self.assertNotIn("friend", block)
        self.assertNotIn("MxBase", block)
        self.assertNotIn("OriginalScheduler", KERNEL.read_text(encoding="utf-8"))
        self.assertNotIn("OriginalKernel", KERNEL.read_text(encoding="utf-8"))

    def test_unused_entry_removed(self):
        self.assertNotIn("RunSssUnfolded", KERNEL.read_text(encoding="utf-8"))

    def test_prologue_operations_are_private(self):
        source = KERNEL.read_text(encoding="utf-8")
        self.assertNotIn("namespace MxPrologueOperations", source)
        optimized = source[source.index("namespace MxPipeline {") :]
        producer = optimized[optimized.index("class GroupedWeightPrologueMx {") :]
        self.assertGreater(
            producer.index("__aicore__ static inline void ConvertMxPackedWeight"),
            producer.index("private:"),
        )

    def test_kernel_checks_both_mode_contracts(self):
        source = KERNEL.read_text(encoding="utf-8")
        for operand in ("BlockMmad", "BlockPrologue"):
            self.assertIn(
                f"is_same_v<typename {operand}::DispatchPolicy, PipelineConfig>", source
            )

    def test_metadata_compiles_and_mismatch_is_rejected(self):
        compiler = os.environ.get("MX_POLICY_CXX")
        if not compiler:
            self.skipTest(
                "Set MX_POLICY_CXX to a host C++17 compiler; this is not a device compiler test"
            )
        arguments = [compiler]
        if Path(compiler).stem == "zig":
            arguments.append("c++")
        source = metadata_source()
        source += r"""
template <uint32_t Mode> using C = MxWeightPipelineConfig<Mode, int8_t, int16_t>;
template <class P> struct Operand { using DispatchPolicy = P; };
template <class BlockMmad, class BlockPrologue, class PipelineConfig> struct Contract {
"""
        checks = re.findall(
            r"static_assert\(AscendC::Std::is_same_v<typename (?:BlockMmad|BlockPrologue)::DispatchPolicy, PipelineConfig>,\s*\"[^\"]+\"\);",
            KERNEL.read_text(encoding="utf-8"),
        )
        self.assertEqual(len(checks), 2)
        source += "\n".join(checks) + "\n};\n} }\nusing namespace Blaze::Gemm;\n"
        positive = r"""
int main() {
static_assert(sizeof(Contract<Operand<C<0>>,Operand<C<0>>,C<0>>) > 0, "original");
static_assert(sizeof(Contract<Operand<C<2>>,Operand<C<2>>,C<2>>) > 0, "aligned");
static_assert(sizeof(Contract<Operand<C<3>>,Operand<C<3>>,C<3>>) > 0, "triple");
static_assert(sizeof(Contract<Operand<C<4>>,Operand<C<4>>,C<4>>) > 0, "quad");
assert(C<2>::Consumer::L1_M == 256 && C<2>::Consumer::L1_N == 256);
assert(C<3>::WeightStages(false) == 3 && C<4>::WeightStages(false) == 4);
assert(C<4>::ScaleWindow(false) == 1024 && C<4>::ScaleWindow(true) == 4096);
assert(C<3>::CreditLimit(1024,256,false) == 3);
assert(C<4>::CreditLimit(1280,256,false) == 4);
assert(C<4>::CreditLimit(0,256,false) == 0 && C<4>::CreditLimit(1024,0,false) == 0);
}
"""
        cases = [
            (positive, None),
            (
                "int main(){ return sizeof(Contract<Operand<C<4>>,Operand<C<3>>,C<3>>); }",
                "Kernel pipeline mode must match BlockMmad",
            ),
            (
                "int main(){ return sizeof(Contract<Operand<C<3>>,Operand<C<4>>,C<3>>); }",
                "Kernel pipeline mode must match weight prologue",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.cpp"
            binary = Path(directory) / "check.exe"
            for body, error in cases:
                with self.subTest(error=error):
                    path.write_text(source + body, encoding="utf-8")
                    result = subprocess.run(
                        arguments + ["-std=c++17", str(path), "-o", str(binary)],
                        capture_output=True,
                        text=True,
                        errors="replace",
                        timeout=180,
                    )
                    if error is None:
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        subprocess.run([str(binary)], check=True, timeout=30)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(error, result.stderr)


if __name__ == "__main__":
    unittest.main()
