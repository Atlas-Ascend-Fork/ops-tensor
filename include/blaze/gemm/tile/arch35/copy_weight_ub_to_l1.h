/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "blaze/gemm/tile/arch35/copy_ub_to_l1.h"

namespace Blaze {
namespace Gemm {
namespace Tile {

// Legacy entry kept for source compatibility: the implementation lives in the
// converted-weight branch of CopyPaddedUBToL1 (blaze/gemm/tile/arch35/copy_ub_to_l1.h)
// and this op only forwards to it, so existing make_copy(CopyUB2L1Weight8Bit{}) callers
// keep working unchanged.
struct CopyUB2L1Weight8Bit {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        CopyPaddedUBToL1::Copy<Tp, traits>(dst, src);
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Blaze

namespace asc {
namespace te {

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit>
    : public copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit, ub_to_l1_trait_default,
                         Blaze::Gemm::Tile::CopyUB2L1Weight8Bit, ub_to_l1_trait_default> {};

} // namespace te
} // namespace asc
