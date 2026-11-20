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
 * \file transpose_batch_mat_mul.cpp
 * \brief TransposeBatchMatMul Kernel UT统一入口
 */

#pragma once
#include "transpose_batch_mat_mul_basic.h"
#include "transpose_batch_mat_mul_tiling_data.h"

enum TbmmOpType : int8_t {
    OP_TYPE_TBMM_BASIC = 0,
    OP_TYPE_TBMM_TRANS_BATCH_A = 1,
    OP_TYPE_TBMM_ITER_BATCH = 2,
    OP_TYPE_TBMM_BASIC_SPLIT_K = 3,
};

template <int8_t OP_TYPE, typename DTYPE_X1, typename DTYPE_X2, typename DTYPE_Y, typename DTYPE_BIAS,
          uint64_t NON_CONTIGUOUS_TYPE =
              static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_C_TRANSPOSE),
          bool TRANS_A = false, bool TRANS_B = false, bool WEIGHT_NZ = false,
          Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODEL = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>
__global__ __aicore__ void transpose_batch_mat_mul_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR yGM,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR scaleGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    const auto* tilingData = reinterpret_cast<const TbmmBasicTilingData*>(tilingGM);
    using LayoutA = AscendC::Std::conditional_t<TRANS_A, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<
        WEIGHT_NZ, AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        AscendC::Std::conditional_t<TRANS_B, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>>;

    if constexpr (OP_TYPE == OP_TYPE_TBMM_BASIC || OP_TYPE == OP_TYPE_TBMM_TRANS_BATCH_A ||
                  OP_TYPE == OP_TYPE_TBMM_BASIC_SPLIT_K) {
        TbmmUT::TbmmBasicWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, NON_CONTIGUOUS_TYPE, LayoutA, LayoutB,
                                 OP_TYPE == OP_TYPE_TBMM_BASIC_SPLIT_K>(x1GM, x2GM, biasGM, yGM, workspaceGM,
                                                                        *tilingData, scaleGM);
    } else if constexpr (OP_TYPE == OP_TYPE_TBMM_ITER_BATCH) {
        TbmmUT::TbmmIterBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutA, LayoutB, NON_CONTIGUOUS_TYPE,
                                     L0C2OUT_MODEL>(x1GM, x2GM, biasGM, yGM, *tilingData);
    } else {
        static_assert(OP_TYPE == OP_TYPE_TBMM_BASIC || OP_TYPE == OP_TYPE_TBMM_TRANS_BATCH_A ||
                          OP_TYPE == OP_TYPE_TBMM_ITER_BATCH || OP_TYPE == OP_TYPE_TBMM_BASIC_SPLIT_K,
                      "Unsupported OP_TYPE value for transpose_batch_mat_mul_kernel_entry");
    }
}
