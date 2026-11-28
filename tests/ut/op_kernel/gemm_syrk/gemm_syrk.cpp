/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file gemm_syrk.cpp
 * \brief GemmSyrk Kernel UT统一入口（通用装配 / 单次搬运双入口）
 */

#pragma once

#include <cstring>
#include "gemm_syrk_basic.h"

namespace {
template <typename>
struct DependentFalse : std::false_type {};
} // namespace

enum SyrkKernelType : int8_t {
    SYRK_KERNEL_SINGLE_FETCH = 0,
};

template <SyrkKernelType KERNEL_TYPE, bool TRANS, typename ElementType>
__global__ __aicore__ void gemm_syrk_kernel_entry(GM_ADDR aGM, GM_ADDR cInGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                  GM_ADDR tilingGM)
{
    GemmSyrkUT::GemmSyrkTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(tilingData));
    if constexpr (KERNEL_TYPE == SYRK_KERNEL_SINGLE_FETCH) {
        GemmSyrkUT::GemmSyrkSingleFetchWrapper<ElementType, TRANS>(aGM, cInGM, cGM, workspaceGM, tilingData);
    } else {
        static_assert(DependentFalse<ElementType>::value, "Unsupported SyrkKernelType value");
    }
}
