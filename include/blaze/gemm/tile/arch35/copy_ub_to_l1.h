/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_ub_to_l1.h
 * \brief UB-to-L1 copy for padded source layouts: FP16/BF16 weight (ZN/NZ padding)
 *        and converted 8-bit weight (ZN column padding / ZN-ZN / NZ row padding).
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Tile {

// Each outer row group (ZN) or column group (NZ) must contain a contiguous span.
// The source pattern selects the grouping; destination strides must describe
// the same contiguous groups in L1. Only gaps may differ. Singleton
// dimensions may have zero stride, and source/destination shapes need not match.
// Callers provide nonempty, compatible layouts, 32-byte-aligned spans/pitches,
// nonnegative gaps fitting the instruction fields, and sufficient buffer capacity.
struct CopyPaddedUBToL1 {
    template <typename Traits, const Traits& traits, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void Copy(const DstTensor& dst, const SrcTensor& src)
    {
        // Deduce unqualified scalars through address-space-qualified pointers;
        // Tensor element types themselves can retain CCE address-space qualifiers.
        using DstType = decltype(GetL1Scalar(dst.data().get()));
        using SrcType = decltype(GetUbScalar(src.data().get()));
        static_assert(AscendC::Std::is_same_v<DstType, SrcType>,
                      "UB-to-L1 copy requires matching source and destination types");

        using SrcPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
        constexpr bool IS_ZN_ROW_PADDING = AscendC::Std::is_same_v<SrcPattern, Blaze::Gemm::zn_row_padding_layout_ptn>;
        constexpr bool IS_NZ_COL_PADDING = AscendC::Std::is_same_v<SrcPattern, Blaze::Gemm::nz_col_padding_layout_ptn>;
        // Converted 8-bit weight family: the source pattern describes the physical UB
        // layout emitted by the corresponding W4-to-W8 path.
        constexpr bool IS_ZN_UB_LAYOUT = AscendC::Std::is_same_v<SrcPattern, Weight8BitZnToZnUbLayoutPtn>;
        constexpr bool IS_ZN_COL_PADDING_8BIT = AscendC::Std::is_same_v<SrcPattern, Weight8BitDnToZnUbLayoutPtn> ||
                                                AscendC::Std::is_same_v<SrcPattern, ZnColPaddingLayoutPtn>;
        constexpr bool IS_NZ_ROW_PADDING_8BIT = AscendC::Std::is_same_v<SrcPattern, NzRowPaddingLayoutPtn>;
        static_assert(IS_ZN_ROW_PADDING || IS_NZ_COL_PADDING || IS_ZN_UB_LAYOUT || IS_ZN_COL_PADDING_8BIT ||
                          IS_NZ_ROW_PADDING_8BIT,
                      "UB-to-L1 copy requires a supported padding source layout");

        if constexpr (IS_ZN_ROW_PADDING) {
            static_assert(AscendC::Std::is_same_v<DstType, half> || AscendC::Std::is_same_v<DstType, bfloat16_t>,
                          "UB-to-L1 copy supports only FP16/BF16 elements");
            CopyZnRowPaddingToL1<DstType>(dst, src);
        } else if constexpr (IS_NZ_COL_PADDING) {
            static_assert(AscendC::Std::is_same_v<DstType, half> || AscendC::Std::is_same_v<DstType, bfloat16_t>,
                          "UB-to-L1 copy supports only FP16/BF16 elements");
            CopyNzColPaddingToL1<DstType>(dst, src);
        } else {
            static_assert(sizeof(DstType) == 1,
                          "Converted weight copy requires matching 8-bit source and destination elements");
            if constexpr (IS_NZ_ROW_PADDING_8BIT) {
                static_assert(AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nz_layout_ptn>,
                              "The NzToNz converted weight copy requires an NZ-fractal L1 destination layout");
                CopyNzRowPadding(dst, src);
            } else {
                static_assert(AscendC::Std::is_same_v<DstLayoutPattern, asc::te::zn_layout_ptn>,
                              "Converted weight copy requires a standard ZN L1 destination layout");
                if constexpr (IS_ZN_COL_PADDING_8BIT) {
                    CopyZnColPadding(dst, src);
                } else {
                    CopyZnToZnWeight(dst, src);
                }
            }
        }
    }

private:
    static constexpr int64_t BLOCK_BYTES = 32;

    template <typename Scalar>
    __aicore__ static Scalar GetL1Scalar(__cbuf__ Scalar*);

    template <typename Scalar>
    __aicore__ static Scalar GetUbScalar(__ubuf__ Scalar*);

    template <typename Scalar, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void CopyZnRowPaddingToL1(const DstTensor& dst, const SrcTensor& src)
    {
        constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(Scalar);
        const auto srcShape = asc::te::get_shape(src.layout());
        const auto srcStride = asc::te::get_stride(src.layout());
        const auto dstStride = asc::te::get_stride(dst.layout());

        // Each K group contains k0*n0*n1 contiguous weight elements; lengths and gaps use 32-byte units.
        const auto rowShape = AscendC::Std::get<0>(srcShape);
        const auto columnShape = AscendC::Std::get<1>(srcShape);
        const int64_t rowInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(rowShape));
        const int64_t columnInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(columnShape));
        const int64_t columnOuterSize = static_cast<int64_t>(AscendC::Std::get<1>(columnShape));
        const int64_t blockCount = static_cast<int64_t>(AscendC::Std::get<1>(rowShape));
        const int64_t blockLen = rowInnerSize * columnInnerSize * columnOuterSize / ELEMS_PER_BLOCK;
        const int64_t srcBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(srcStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t dstBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(dstStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t srcGap = srcBlockSpan - blockLen;
        const int64_t dstGap = dstBlockSpan - blockLen;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(),
                       static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen),
                       static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap));
    }

    template <typename Scalar, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void CopyNzColPaddingToL1(const DstTensor& dst, const SrcTensor& src)
    {
        constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(Scalar);
        const auto srcShape = asc::te::get_shape(src.layout());
        const auto srcStride = asc::te::get_stride(src.layout());
        const auto dstStride = asc::te::get_stride(dst.layout());

        // Each N group contains k0*k1*n0 contiguous weight elements; lengths and gaps use 32-byte units.
        const auto rowShape = AscendC::Std::get<0>(srcShape);
        const auto columnShape = AscendC::Std::get<1>(srcShape);
        const int64_t rowInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(rowShape));
        const int64_t rowOuterSize = static_cast<int64_t>(AscendC::Std::get<1>(rowShape));
        const int64_t columnInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(columnShape));
        const int64_t blockCount = static_cast<int64_t>(AscendC::Std::get<1>(columnShape));
        const int64_t blockLen = rowInnerSize * rowOuterSize * columnInnerSize / ELEMS_PER_BLOCK;
        const int64_t srcBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(srcStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t dstBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(dstStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t srcGap = srcBlockSpan - blockLen;
        const int64_t dstGap = dstBlockSpan - blockLen;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(),
                       static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen),
                       static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap));
    }

    // ZN-family column-padding source (Weight8BitDnToZnUbLayoutPtn /
    // ZnColPaddingLayoutPtn): k1 K-slabs, each holding nSize adjacent 32B
    // blocks (one per N column of 32 contiguous K), with the slab pitch carrying the
    // padding gap; the L1 destination is a standard ZN layout.
    template <typename T, typename U>
    __aicore__ inline static void CopyZnColPadding(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dstLayout);
        uint16_t blockCount = static_cast<uint16_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape)));
        uint32_t blockLen = static_cast<uint32_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape)));
        int64_t srcBlockSpan = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) / BLOCK_BYTE_SIZE;
        int64_t dstBlockSpan = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) / BLOCK_BYTE_SIZE;
        int64_t srcGap = srcBlockSpan - static_cast<int64_t>(blockLen);
        int64_t dstGap = dstBlockSpan - static_cast<int64_t>(blockLen);
        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen, srcGap,
                       dstGap);
    }

    template <typename T, typename U>
    __aicore__ inline static void CopyZnToZnWeight(const T& dst, const U& src)
    {
        using type = typename U::element_type;
        const auto& srcLayout = src.layout();

        // Get shape and stride tuples
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);

        // Extract dimensions from srcShape = ((c0, k1), (n0, n1))
        // Std::get<0>(srcShape) = (c0, k1), Std::get<1>(srcShape) = (n0, n1)
        uint16_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        uint16_t k1 = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint16_t n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        uint16_t n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));

        // Extract innerStride from srcStride = ((1, n1*InnerStride), (c0, InnerStride))
        // For UB2L1, use column stride (InnerStride) not row stride (n1*InnerStride)
        int64_t innerStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));

        // Total number of fractal blocks to copy
        uint16_t blockCount = k1 * n1;

        // Block length in 32B units
        uint32_t blockLen = (n0 * c0 * sizeof(type)) / AscendC::ONE_BLK_SIZE;

        // Source stride in 32B units
        int64_t srcStride = (innerStride * sizeof(type)) / AscendC::ONE_BLK_SIZE - blockLen;

        // Destination stride in 32B units (contiguous in L1)
        int64_t dstStride = 0;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen,
                       srcStride, dstStride);
    }

    // NZ-family row-padding source (NzRowPaddingLayoutPtn): the UB source holds the
    // NzToNz chunks (N1 fractals of per-group 32K x 32N slabs, 256-element chunks
    // interleaved across BL1 ping-pong buffers), the L1 destination is an NZ-fractal
    // slice inheriting the read-side frame strides; each fractal is one gap-0 run.
    template <typename T, typename U>
    __aicore__ inline static void CopyNzRowPadding(const T& dst, const U& src)
    {
        using type = typename U::element_type;
        const auto& srcLayout = src.layout();
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dst.layout());

        const int64_t blockBytes = static_cast<int64_t>(BLOCK_BYTE_SIZE);
        // Source: shape ((VEC_REG_ELEM, kGroupNum), (vLLoopNum, n1LoopNum)), strides
        // ((1, vLLoopNum * innerStride), (innerStride, vLLoopNum * innerStride * kGroupNum)).
        const int64_t chunkLen = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        const int64_t kGroupNum = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        const int64_t vLLoopNum = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        const int64_t n1LoopNum = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));
        const int64_t srcChunkStride = AscendC::Std::get<0>(AscendC::Std::get<1>(srcStrideTuple));
        const int64_t srcN1Stride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));
        // Destination: NZ-fractal strides ((C0, C0 * FRACTAL), (1, C0 * CeilAlign(kBL1, 16)));
        // the n1 pitch comes from the frame the AIC read side consumes.
        const int64_t dstN1Stride = AscendC::Std::get<1>(AscendC::Std::get<1>(dstStrideTuple));

        const int64_t blocksPerN1 = vLLoopNum * kGroupNum;
        if (blocksPerN1 == 0 || n1LoopNum == 0) {
            return;
        }
        const uint32_t blockLen = static_cast<uint32_t>(chunkLen * sizeof(type) / BLOCK_BYTE_SIZE);
        const int64_t srcGap = srcChunkStride * static_cast<int64_t>(sizeof(type)) / blockBytes -
                               static_cast<int64_t>(blockLen);
        if (dstN1Stride == blocksPerN1 * chunkLen) {
            // The covered chunks tile the destination fractals contiguously: one run.
            uint16_t blockCount = static_cast<uint16_t>(blocksPerN1 * n1LoopNum);
            asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen,
                           srcGap, 0);
            return;
        }
        // Trailing partial K group (or a partial-K bub window): the covered chunks no
        // longer tile the destination fractals contiguously, so emit one gap-0 run per
        // N1 fractal and advance both sides by their layout strides.
        __cbuf__ uint8_t* dstPtr = (__cbuf__ uint8_t*)dst.data().get();
        __ubuf__ uint8_t* srcPtr = (__ubuf__ uint8_t*)src.data().get();
        const uint16_t fractalBlockCount = static_cast<uint16_t>(blocksPerN1);
        for (int64_t n1Idx = 0; n1Idx < n1LoopNum; n1Idx++) {
            asc_copy_ub2l1((__cbuf__ void*)dstPtr, (__ubuf__ void*)srcPtr, fractalBlockCount, blockLen, srcGap, 0);
            dstPtr += dstN1Stride * static_cast<int64_t>(sizeof(type));
            srcPtr += srcN1Stride * static_cast<int64_t>(sizeof(type));
        }
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Blaze

namespace asc {
namespace te {

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1>
    : public copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1, ub_to_l1_trait_default,
                         Blaze::Gemm::Tile::CopyPaddedUBToL1, ub_to_l1_trait_default> {};

} // namespace te
} // namespace asc
