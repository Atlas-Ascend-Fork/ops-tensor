/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layout_utils.h
 * \brief
 */

#pragma once

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

// Slice 3D Layout Pattern
struct NDSliceLayoutPtn {};

// Physical UB layouts emitted by the MXA8W4 conversion. They are intentionally
// distinct from standard tensor-api patterns because their physical interleave
// and stride contracts are consumed by custom vector/copy tiles.
struct Weight8BitZnToZnUbLayoutPtn {};
struct Weight8BitDnToZnUbLayoutPtn {};

// New-style tags for the converted-weight UB layouts. ZnColPaddingLayoutPtn describes
// the same geometry as the legacy Weight8BitDnToZnUbLayoutPtn above (kept for
// compatibility); both route to the same converted-weight branch of CopyPaddedUBToL1.
// NzRowPaddingLayoutPtn covers the NZ-fractal family (N1 fractals, per-group 32K x 32N
// slabs, chunk interleave).
struct ZnColPaddingLayoutPtn {};
struct NzRowPaddingLayoutPtn {};

// IsTrans
template <typename LayoutPattern>
constexpr bool GetTransValue()
{
    constexpr bool isNonTrans = AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nd_layout_ptn,
                                                          asc::te::nd_ext_layout_ptn, asc::te::nz_layout_ptn,
                                                          asc::te::scalea_nd_layout_ptn, asc::te::scaleb_nd_layout_ptn>;
    constexpr bool isTrans = AscendC::Std::is_one_of_v<LayoutPattern, asc::te::dn_layout_ptn,
                                                       asc::te::dn_ext_layout_ptn, asc::te::zn_layout_ptn,
                                                       asc::te::scalea_dn_layout_ptn, asc::te::scaleb_dn_layout_ptn>;

    constexpr bool isKnown = isNonTrans || isTrans;
    static_assert(isKnown, "IsTrans is not implemented for this layout pattern");

    return !isNonTrans && isTrans;
}

template <typename LayoutPattern>
struct IsTrans {
    static constexpr bool value = GetTransValue<LayoutPattern>();
};

// IsWeightNz
template <typename LayoutPattern>
constexpr bool GetWeightNzValue()
{
    constexpr bool isNonWeightNz = AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nd_ext_layout_ptn,
                                                             asc::te::dn_ext_layout_ptn>;
    constexpr bool
        isWeightNz = AscendC::Std::is_one_of_v<LayoutPattern, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>;

    constexpr bool isKnown = isNonWeightNz || isWeightNz;
    static_assert(isKnown, "IsWeightNz is not implemented for this layout");

    return !isNonWeightNz && isWeightNz;
}

template <typename LayoutPattern>
struct IsWeightNz {
    static constexpr bool value = GetWeightNzValue<LayoutPattern>();
};

// IsScaleNz
template <typename LayoutPattern>
constexpr bool GetScaleNzValue()
{
    constexpr bool isScaleNz = AscendC::Std::is_same_v<LayoutPattern, asc::te::nn_layout_ptn>;
    constexpr bool isScaleNd = AscendC::Std::is_one_of_v<LayoutPattern, asc::te::scaleb_nd_layout_ptn,
                                                         asc::te::scaleb_dn_layout_ptn>;
    constexpr bool isKnown = isScaleNz || isScaleNd;
    static_assert(isKnown, "IsScaleNz is not implemented for this layout");

    return isScaleNz;
}

template <typename LayoutPattern>
struct IsScaleNz {
    static constexpr bool value = GetScaleNzValue<LayoutPattern>();
};

// HiF4 scale packs two adjacent positions per element: ScaleA along M in the DN layout and ScaleB along N
// in the ND layout. All other scale layouts (including ScaleNZ) keep the logical position.
template <bool IsHif4Type, typename LayoutScale>
__aicore__ inline constexpr int64_t GetScaleGmPos(int64_t pos)
{
    constexpr bool isPackedLayout = AscendC::Std::is_same_v<LayoutScale, asc::te::scalea_dn_layout_ptn> ||
                                    AscendC::Std::is_same_v<LayoutScale, asc::te::scaleb_nd_layout_ptn>;
    if constexpr (IsHif4Type && isPackedLayout) {
        return pos >> 1;
    }
    return pos;
}

// Build a row-major hierarchical ND (nd_ext_layout_ptn) layout with an explicit row pitch.
// Shape = ((1, rows), (1, cols)); Stride = ((0, rowPitch), (0, 1)).
template <typename T = float>
__aicore__ inline auto MakeNDExtLayout(int64_t rows, int64_t cols, int64_t rowPitch)
{
    auto shape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<1>{}, rows),
                                     asc::te::make_shape(AscendC::Std::Int<1>{}, cols));
    auto stride = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<0>{}, rowPitch),
                                       asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
    return asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(shape, stride);
}

} // namespace Gemm
} // namespace Blaze
