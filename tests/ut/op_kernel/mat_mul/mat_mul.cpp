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
 * \file mat_mul.cpp
 * \brief MatMulV3 Kernel UT统一入口
 */

#pragma once
#include <cstring>
#include "mat_mul_stream_k.h"
#include "mat_mul_basic.h"
#include "mat_mul_a_full_load.h"
#include "mat_mul_b_full_load.h"
#include "mat_mul_fixpipe_opti.h"
#include "mat_mul_bmm_broadcast.h"
#include "mat_mul_iterbatch_broadcast.h"
#include "mat_mul_iterbatch.h"
#include "mat_mul_tiling_data.h"

namespace {
template <typename>
struct DependentFalse : std::false_type {};
} // namespace

enum OpType : int8_t {
    OP_TYPE_MATMUL_BASIC = 0,
    OP_TYPE_MATMUL_STREAMK = 1,
    OP_TYPE_MATMUL_AFULLLOAD = 2,
    OP_TYPE_MATMUL_BFULLLOAD = 3,
    OP_TYPE_MATMUL_FIXPIPE_OPT = 4,
    OP_TYPE_MATMUL_BMM_BROADCAST = 5,
    OP_TYPE_MATMUL_BMM_ITERBATCH_BROADCAST = 6,
    OP_TYPE_MATMUL_BMM_ITERBATCH = 7,
};

template <int8_t OP_TYPE, typename DTYPE_X1, typename DTYPE_X2, typename DTYPE_Y, typename DTYPE_BIAS,
          Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODE = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE = 0,
          uint64_t NON_CONTIGUOUS_TYPE = 0, typename TilingDataT = MatMulV3BasicTilingData>
__global__ __aicore__ void mat_mul_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR yGM,
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TilingDataT tilingData;
    memcpy(&tilingData, tilingGM, sizeof(TilingDataT));

    if constexpr (OP_TYPE == OP_TYPE_MATMUL_STREAMK) {
        MatMulV3UT::MatMulStreamKWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, L0C2OUT_MODE, FUSED_OP_TYPE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData, tilingData.batch);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BASIC) {
        MatMulV3UT::MatMulBasicWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, NON_CONTIGUOUS_TYPE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_AFULLLOAD) {
        MatMulV3UT::MatMulAFullLoadWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, biasGM, yGM,
                                                                                    workspaceGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BFULLLOAD) {
        MatMulV3UT::MatMulBFullLoadWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, biasGM, yGM,
                                                                                    workspaceGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_FIXPIPE_OPT) {
        MatMulV3UT::MatMulFixpipeOptWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(x1GM, x2GM, biasGM, yGM,
                                                                                     workspaceGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BMM_BROADCAST) {
        MatMulV3UT::MatMulBmmBroadCastWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
            x1GM, x2GM, biasGM, yGM, workspaceGM,
            *reinterpret_cast<const MatMulV3BmmBroadcastTilingData*>(&tilingData));
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BMM_ITERBATCH_BROADCAST) {
        MatMulV3UT::MatMulIterBatchBroadcastWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, *reinterpret_cast<const MatMulV3IterBatchTilingData*>(&tilingData));
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BMM_ITERBATCH) {
        MatMulV3UT::MatMulIterBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, L0C2OUT_MODE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, *reinterpret_cast<const MatMulV3IterBatchTilingData*>(&tilingData));
    } else {
        static_assert(DependentFalse<TilingDataT>::value, "Unsupported OP_TYPE value");
    }
}
