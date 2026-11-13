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
 * \file tile_trait.h
 * \brief
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze::Gemm::Tile {

template <typename AType, typename LayoutA, typename BType, typename LayoutB>
struct MxKZeroPaddingTrait {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    // A4W4 场景要求 K 轴为偶数，当且仅当 A 转置 B 不转置时 K 轴同时为外轴，可能为奇数，需要额外补零并对齐；
    // 其余 A4W4 场景 A / B 矩阵中总有一个 K 轴为内轴，A8W4 场景支持 K 轴为奇数，均无需额外补零和对齐操作；
    // 无需额外补零和对齐时，K_ALIGNMENT 设为 1，表示保持原 K 轴大小。
    static constexpr uint64_t K_ALIGNMENT = (Blaze::Gemm::IsFp4<AType>() && Blaze::Gemm::IsFp4<BType>() &&
                                             Blaze::Gemm::IsTrans<LayoutA>::value &&
                                             !Blaze::Gemm::IsTrans<LayoutB>::value) ?
                                                2UL :
                                                1UL;
#else
    static constexpr uint64_t K_ALIGNMENT = 64UL;
#endif
};

constexpr asc::te::mmad_trait MX_MMAD_TRAIT = asc::te::mmad_trait{0, false, false, true, asc::te::mmad_type::mx};
struct MmadTraitMX {
    using TraitType = asc::te::mmad_trait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

// Disable M padding to select the native M=1 GEMV mode.
struct MmadTraitGemv {
    using TraitType = asc::te::mmad_trait;
    static constexpr TraitType value{0, false, false, false, asc::te::mmad_type::normal};
};

constexpr asc::te::l0c_to_ub_trait COPY_L0C2UB_SPLIT_M_TRAIT = asc::te::l0c_to_ub_trait{
    asc::te::round_mode::default_round, false, false, asc::te::dual_dst_mode::split_m};
struct CopyL0C2UBTraitSplitM {
    using TraitType = asc::te::l0c_to_ub_trait;
    static constexpr const TraitType value = COPY_L0C2UB_SPLIT_M_TRAIT;
};
// 与 CopyL0C2UBTraitSplitM 完全相同，仅作为别名保留以兼容既有调用
using CopyL0C2UBTraitMixSplitM = CopyL0C2UBTraitSplitM;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
using CopyL0C2UBTraitMix = asc::te::l0c_to_ub_trait_default;
#else
using CopyL0C2UBTraitMix = CopyL0C2UBTraitSplitM;
#endif

struct CopyGM2L1TraitPreAllocation {
    using TraitType = asc::te::gm_to_l1_trait;
    static constexpr const TraitType value = asc::te::gm_to_l1_trait{true};
};

constexpr asc::te::l0c_to_gm_trait COPY_L0C2GM_RELU_TRAIT = asc::te::l0c_to_gm_trait{asc::te::round_mode::default_round,
                                                                                     true, false};
struct CopyL0C2GMTraitRelu {
    using TraitType = asc::te::l0c_to_gm_trait;
    static constexpr const TraitType value = COPY_L0C2GM_RELU_TRAIT;
};

} // namespace Blaze::Gemm::Tile
