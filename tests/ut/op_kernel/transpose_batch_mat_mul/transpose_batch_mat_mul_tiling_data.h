/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file transpose_batch_mat_mul_tiling_data.h
 * \brief TransposeBatchMatMul Kernel UT Tiling数据结构定义
 */

#pragma once

#include "blaze_kernel_stub.h"

#include <cstdint>

enum class TbmmL2CacheMode : std::uint32_t {
    L2_CACHE_DEFAULT = 0x00,
    A_L2_CACHE_DISABLE = 0x01,
    B_L2_CACHE_DISABLE = 0x02,
    ALL_L2_CACHE_DISABLE = 0x03,
};

#pragma pack(push, 8)
struct TbmmBasicTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t batch = 1;
    uint32_t batchSplitFactor = 1;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 1;
    uint32_t nTailMain = 1;
    uint8_t isHf32 = 0;
    uint8_t l1BufferNum = 0;
    uint8_t l0cDB = 1;
    uint8_t ubDB = 1;
    TbmmL2CacheMode l2CacheDisable = TbmmL2CacheMode::L2_CACHE_DEFAULT;
    uint32_t sliceM = 1;
    uint32_t srcNdStride = 1;
    uint32_t innerBatch = 1;
    uint32_t iterBatchL1 = 1;
    uint32_t iterBatchL0 = 1;
};
#pragma pack(pop)
