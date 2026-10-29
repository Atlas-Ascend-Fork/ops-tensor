# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Check the unsigned scalar rewrite of the FP8 NZ/ZN concat DMA plan.

This is a host-side arithmetic/address oracle, not a hardware DMA test. The
previous multiplication/division model remains independent of the shift model,
including rejected shapes and DMA field boundaries. The separate address oracle
remains independent. Run with python -m unittest discover -s examples/acl/grouped_matmul/quant_grouped_matmul_swiglu_quant_mx/tests -p 'test_*.py'.
"""

import random
import unittest

from test_swiglu_weight_nz_concat_addresses import physical


UNIT_BIT = 1
UINT64_WIDTH = 64
UINT64_MASK = (UNIT_BIT << UINT64_WIDTH) - UNIT_BIT
NZ_K0 = 16
NZ_C0 = 32
SWIGLU_SPLIT_FACTOR = 2
ALIGN_MASK_ADJUSTMENT = 1
MAX_CONCAT_DMA_BYTES = 0x7FFFFFFF
SCALAR_ARGUMENT_COUNT = 5
RANDOM_SEED = 20260924
RANDOM_SAMPLE_COUNT = 3000
RANDOM_WIDTH_BITS = 22
RANDOM_K_BITS = 26
RANDOM_WINDOW_BITS = 14
NONZERO_GROUP_INDEX = 2
ZN_COPY_WIDTHS = (16, 32, 64)
NZ_COPY_WIDTHS = (32, 64)
CONCAT_N_SHIFT = 1
C0_SHIFT = 5
K_ALIGN = 128
N_ALIGN = 64
MAX_DMA_BURST_COUNT = 0xFFFF
MAX_DMA_DST_STRIDE = 0xFFFFFFFF


def legacy_plan(k, n, width, window_k, l1_k, trans):
    """Return both DMA descriptors: dst/src offsets, count, bytes and strides."""
    n_align = NZ_K0 if trans else NZ_C0
    if (
        k % K_ALIGN != 0
        or n % N_ALIGN != 0
        or window_k == 0
        or window_k % K_ALIGN != 0
        or l1_k != window_k
        or width == 0
        or width % n_align != 0
        or width > n // SWIGLU_SPLIT_FACTOR
    ):
        return None
    if trans:
        bursts = SWIGLU_SPLIT_FACTOR * (window_k // NZ_C0)
        byte_count = (width * NZ_C0) & UINT64_MASK
        if bursts > MAX_DMA_BURST_COUNT or byte_count > MAX_CONCAT_DMA_BYTES:
            return None
        fields = (
            bursts // SWIGLU_SPLIT_FACTOR,
            byte_count,
            (n * NZ_C0) & UINT64_MASK,
            byte_count * SWIGLU_SPLIT_FACTOR,
        )
        return (
            (0, 0, *fields),
            (byte_count, ((n // SWIGLU_SPLIT_FACTOR) * NZ_C0) & UINT64_MASK, *fields),
        )
    rows = width // NZ_C0
    byte_count = (window_k * NZ_C0) & UINT64_MASK
    if rows > MAX_DMA_BURST_COUNT or byte_count > MAX_DMA_DST_STRIDE:
        return None
    fields = rows, byte_count, (k * NZ_C0) & UINT64_MASK, byte_count
    return (
        (0, 0, *fields),
        (
            (width * l1_k) & UINT64_MASK,
            ((n // SWIGLU_SPLIT_FACTOR) * k) & UINT64_MASK,
            *fields,
        ),
    )


def shift_plan(k, n, width, window_k, l1_k, trans):
    n_align = NZ_K0 if trans else NZ_C0
    if (
        k & (K_ALIGN - ALIGN_MASK_ADJUSTMENT)
        or n & (N_ALIGN - ALIGN_MASK_ADJUSTMENT)
        or window_k == 0
        or window_k & (K_ALIGN - ALIGN_MASK_ADJUSTMENT)
        or l1_k != window_k
        or width == 0
        or width & (n_align - ALIGN_MASK_ADJUSTMENT)
        or width > n >> CONCAT_N_SHIFT
    ):
        return None
    if trans:
        bursts_per_half = window_k >> C0_SHIFT
        byte_count = (width << C0_SHIFT) & UINT64_MASK
        if (
            bursts_per_half > MAX_DMA_BURST_COUNT >> CONCAT_N_SHIFT
            or byte_count > MAX_DMA_DST_STRIDE >> CONCAT_N_SHIFT
        ):
            return None
        fields = (
            bursts_per_half,
            byte_count,
            (n << C0_SHIFT) & UINT64_MASK,
            byte_count << CONCAT_N_SHIFT,
        )
        return (
            (0, 0, *fields),
            (
                byte_count,
                ((n >> CONCAT_N_SHIFT) << C0_SHIFT) & UINT64_MASK,
                *fields,
            ),
        )
    rows = width >> C0_SHIFT
    byte_count = (window_k << C0_SHIFT) & UINT64_MASK
    if rows > MAX_DMA_BURST_COUNT or byte_count > MAX_DMA_DST_STRIDE:
        return None
    fields = rows, byte_count, (k << C0_SHIFT) & UINT64_MASK, byte_count
    return (
        (0, 0, *fields),
        (
            (width * l1_k) & UINT64_MASK,
            ((n >> CONCAT_N_SHIFT) * k) & UINT64_MASK,
            *fields,
        ),
    )


class WeightNzScalarEquivalence(unittest.TestCase):
    def assert_same_plan(self, values):
        self.assertEqual(legacy_plan(*values), shift_plan(*values), values)

    def test_alignment_zero_and_unsigned_boundaries(self):
        values = (
            0,
            1,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            129,
            255,
            256,
            257,
            (1 << 32) - 128,
            1 << 32,
            1 << 63,
            UINT64_MASK - 127,
            UINT64_MASK,
        )
        for trans in (False, True):
            for index in range(SCALAR_ARGUMENT_COUNT):
                for value in values:
                    args = [384, 384, 64, 128, 128, trans]
                    args[index] = value
                    self.assert_same_plan(args)

    def test_dma_limits_preserve_acceptance(self):
        # Each accepted shape is the last aligned value below its field limit.
        # The next aligned value must still select the ordinary-copy fallback.
        cases = (
            (1048448, 128, 32, 1048448, 1048448, True),
            (1048576, 128, 32, 1048576, 1048576, True),
            (128, 134217728, 67108848, 128, 128, True),
            (128, 134217728, 67108864, 128, 128, True),
            (128, 4194304, 2097120, 128, 128, False),
            (128, 4194304, 2097152, 128, 128, False),
            (134217600, 128, 32, 134217600, 134217600, False),
            (134217728, 128, 32, 134217728, 134217728, False),
        )
        for index, args in enumerate(cases):
            with self.subTest(args=args):
                self.assert_same_plan(args)
                self.assertEqual(
                    shift_plan(*args) is not None, index % SWIGLU_SPLIT_FACTOR == 0
                )

    def test_seeded_unsigned_dimensions(self):
        rng = random.Random(RANDOM_SEED)
        for trans in (False, True):
            for _ in range(RANDOM_SAMPLE_COUNT):
                width = rng.randrange(UNIT_BIT, UNIT_BIT << RANDOM_WIDTH_BITS) * (
                    NZ_K0 if trans else NZ_C0
                )
                n = (
                    (SWIGLU_SPLIT_FACTOR * width + N_ALIGN - ALIGN_MASK_ADJUSTMENT)
                    // N_ALIGN
                ) * N_ALIGN
                k = rng.randrange(UNIT_BIT, UNIT_BIT << RANDOM_K_BITS) * K_ALIGN
                window_k = (
                    rng.randrange(UNIT_BIT, UNIT_BIT << RANDOM_WINDOW_BITS) * K_ALIGN
                )
                self.assert_same_plan((k, n, width, window_k, window_k, trans))

    def test_shift_plan_matches_physical_addresses(self):
        for trans in (False, True):
            for width in ZN_COPY_WIDTHS if trans else NZ_COPY_WIDTHS:
                k, n, k0, window_k = 384, 384, 256, 128
                n0 = n // SWIGLU_SPLIT_FACTOR - width
                group_base = NONZERO_GROUP_INDEX * k * n
                src_base = group_base + physical(k0, n0, k, n, trans)
                plan = shift_plan(k, n, width, window_k, window_k, trans)
                self.assertIsNotNone(plan)
                destinations = set()
                for p in range(window_k):
                    for q in range(SWIGLU_SPLIT_FACTOR * width):
                        half, column = divmod(q, width)
                        dst_off, src_off, count, size, src_step, dst_step = plan[half]
                        if trans:
                            burst, lane = divmod(p, NZ_C0)
                            byte = column * NZ_C0 + lane
                        else:
                            burst, lane = divmod(column, NZ_C0)
                            byte = p * NZ_C0 + lane
                        self.assertLess(burst, count)
                        self.assertLess(byte, size)
                        src = src_base + src_off + burst * src_step + byte
                        dst = dst_off + burst * dst_step + byte
                        self.assertEqual(
                            src,
                            group_base
                            + physical(
                                k0 + p,
                                n0 + half * (n // SWIGLU_SPLIT_FACTOR) + column,
                                k,
                                n,
                                trans,
                            ),
                        )
                        self.assertEqual(
                            dst,
                            physical(
                                p, q, window_k, SWIGLU_SPLIT_FACTOR * width, trans
                            ),
                        )
                        self.assertNotIn(dst, destinations)
                        destinations.add(dst)
                self.assertEqual(
                    destinations, set(range(SWIGLU_SPLIT_FACTOR * width * window_k))
                )


if __name__ == "__main__":
    unittest.main()
