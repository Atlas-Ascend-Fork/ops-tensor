# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Address oracle for FP8 SwiGLU GM->L1 concat (not a hardware DMA test).

Run with python -m unittest discover -s examples/acl/grouped_matmul/quant_grouped_matmul_swiglu_quant_mx/tests -p 'test_*.py'.
The DMA-loop model is checked against independent scalar NZ/ZN indexing,
including nonzero N/K offsets, group offsets, tail widths and guard bytes.
"""

import unittest


NZ_K0 = 16
NZ_C0 = 32
SWIGLU_SPLIT_FACTOR = 2
NONZERO_GROUP_INDEX = 2
K_WINDOW_SIZE = 128
PROBLEM_SHAPES = ((128, 128), (384, 384), (4096, 8192))
ZN_COPY_WIDTHS = (16, 32, 64)
NZ_COPY_WIDTHS = (32, 64)


def physical(k, n, full_k, full_n, trans):
    if trans:
        return (
            ((k // NZ_C0) * (full_n // NZ_K0) + n // NZ_K0) * NZ_K0 + n % NZ_K0
        ) * NZ_C0 + k % NZ_C0
    return (
        ((n // NZ_C0) * (full_k // NZ_K0) + k // NZ_K0) * NZ_K0 + k % NZ_K0
    ) * NZ_C0 + n % NZ_C0


class SingleJumpAddresses(unittest.TestCase):
    def test_nz_and_zn(self):
        for trans in (False, True):
            for k, n in PROBLEM_SHAPES:
                for width in ZN_COPY_WIDTHS if trans else NZ_COPY_WIDTHS:
                    for j in (0, n // SWIGLU_SPLIT_FACTOR - width):
                        for k0, kc in (
                            (0, K_WINDOW_SIZE),
                            (k - K_WINDOW_SIZE, K_WINDOW_SIZE),
                        ):
                            with self.subTest(
                                trans=trans, k=k, n=n, width=width, j=j, k0=k0
                            ):
                                # A nonzero group catches accidental use of a global base.
                                group = NONZERO_GROUP_INDEX * k * n
                                src_base = group + physical(k0, j, k, n, trans)
                                seen = set()
                                for p in range(kc):
                                    for q in range(SWIGLU_SPLIT_FACTOR * width):
                                        half, column = divmod(q, width)
                                        expected_src = group + physical(
                                            k0 + p,
                                            j
                                            + half * (n // SWIGLU_SPLIT_FACTOR)
                                            + column,
                                            k,
                                            n,
                                            trans,
                                        )
                                        expected_dst = physical(
                                            p, q, kc, SWIGLU_SPLIT_FACTOR * width, trans
                                        )
                                        if trans:
                                            burst = p // NZ_C0
                                            src = (
                                                src_base
                                                + half
                                                * (n // SWIGLU_SPLIT_FACTOR)
                                                * NZ_C0
                                                + burst * n * NZ_C0
                                                + column * NZ_C0
                                                + p % NZ_C0
                                            )
                                            dst = (
                                                half * width * NZ_C0
                                                + burst
                                                * SWIGLU_SPLIT_FACTOR
                                                * width
                                                * NZ_C0
                                                + column * NZ_C0
                                                + p % NZ_C0
                                            )
                                        else:
                                            row, lane = divmod(column, NZ_C0)
                                            d = p * NZ_C0 + lane
                                            src = (
                                                src_base
                                                + half * (n // SWIGLU_SPLIT_FACTOR) * k
                                                + row * k * NZ_C0
                                                + d
                                            )
                                            dst = (
                                                half * (width // NZ_C0) * kc * NZ_C0
                                                + row * kc * NZ_C0
                                                + d
                                            )
                                        self.assertEqual(
                                            (src, dst), (expected_src, expected_dst)
                                        )
                                        self.assertNotIn(dst, seen)
                                        seen.add(dst)
                                self.assertEqual(
                                    seen, set(range(SWIGLU_SPLIT_FACTOR * width * kc))
                                )


if __name__ == "__main__":
    unittest.main()
