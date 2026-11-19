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
 * \file qbmm_tiling_data.h
 * \brief QBMM Kernel UT Tiling数据结构定义（打包所有子Params用于CPU debug memcpy）
 */

#pragma once

#include "blaze_kernel_stub.h"

#include <cstdint>

#pragma pack(push, 8)
struct QBMMV3TilingData {
    // ProblemShape: M, N, K, B
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;

    // BlockMmadParams (from block_mmad_a8w8_fixpipe_quant.h)
    uint64_t aGmAddr;
    uint64_t bGmAddr;
    uint64_t cGmAddr;
    uint64_t biasGmAddr;
    uint64_t scaleAGmAddr;
    uint64_t scaleBGmAddr;

    // BlockSchedulerParams (from block_scheduler_qbmm.h)
    int64_t baseM;
    int64_t baseN;
    int64_t mTailTile;
    int64_t nTailTile;
    int64_t mBaseTailSplitCnt;
    int64_t nBaseTailSplitCnt;
    int64_t mTailMain;
    int64_t nTailMain;

    // QBMMTiling (from kernel_qbmm_cube.h)
    // batch 广播维度 A：batchA1..batchA4
    uint32_t batchA1;
    uint32_t batchA2;
    uint32_t batchA3;
    uint32_t batchA4;
    // batch 广播维度 B：batchB1..batchB4
    uint32_t batchB1;
    uint32_t batchB2;
    uint32_t batchB3;
    uint32_t batchB4;
    // batch 广播维度 C（输出）：batchC1..batchC4
    uint32_t batchC1;
    uint32_t batchC2;
    uint32_t batchC3;
    uint32_t batchC4;
    // bias / 量化模式标志
    uint32_t biasThreeDim;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    // L1 载入 / buffer 配置
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t nBufferNum;
    uint32_t baseM_qbmm;
    uint32_t baseN_qbmm;
    uint32_t baseK_qbmm;
    uint32_t isBias;
    uint32_t dbL0C;
    uint32_t weightMustHitL2{1U};

    // BlockEpilogueDequant 额外字段（MIX 路径）：bias 运行时 dtype 代码（ge::DataType，如 DT_FLOAT/DT_FLOAT16/DT_BF16）
    uint32_t biasDtype;
};
#pragma pack(pop)

namespace QBMMUT {

constexpr uint32_t GE_DT_FLOAT = 0U;
constexpr uint32_t GE_DT_INT32 = 3U;

#pragma pack(push, 8)
struct QBMML0CPingpongTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t nBufferNum;
    uint32_t dbL0C;
    uint32_t isBias{0U};
    uint32_t weightMustHitL2{1U};
    uint32_t mL0{0U};
    uint32_t nL0{0U};
    uint32_t batchA{0U};
    uint32_t batchB{0U};
    uint32_t biasThreeDim{0U};
};

struct QBMMStreamKTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
    uint32_t usedCoreNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t singleCoreK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t dbL0C;
    uint32_t isBias{0U};
    uint32_t weightMustHitL2{1U};
};

struct QBMMPertensorStreamKTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
    uint32_t usedCoreNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t singleCoreK;
    uint32_t kL1;
    uint32_t isBias;
    uint32_t biasDtype;
};
#pragma pack(pop)

} // namespace QBMMUT
