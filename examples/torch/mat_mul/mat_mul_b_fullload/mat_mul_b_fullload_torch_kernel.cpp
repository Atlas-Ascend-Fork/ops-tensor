/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file mat_mul_b_fullload_torch_kernel.cpp
 * @brief Blaze MatMul BFullLoad kernel and host launchers for the torch extension example.
 *
 * B matrix is fully loaded into L1 buffer, A matrix is streamed in K-tiles.
 * Suitable for scenarios where N is small enough that B fits entirely in L1.
 * Tiling policy follows acl/mat_mul/mat_mul_b_fullload (ND layout only).
 *
 * Supported dtypes: float16, bfloat16, float32
 * Layout: ND x ND -> ND (no transpose, no bias)
 */

#include <cstdint>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "kernel_basic_intf.h"

/* ========================================================================== */
/* Tiling (same policy as mat_mul_b_fullload)                                 */
/* ========================================================================== */

namespace {

struct BFullLoadTiling {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    uint32_t mTailCnt = 1;
    uint32_t nTailCnt = 1;
    static constexpr int64_t L1_BUFFER_NUM = 2;
    static constexpr int64_t L0C_DB = 1;
};

inline int64_t CeilDivI64(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline BFullLoadTiling GetTiling(int64_t m, int64_t k, int64_t n)
{
    (void)k;
    BFullLoadTiling cfg;
    cfg.baseM = 128;
    cfg.baseN = n;
    cfg.baseK = 32;
    cfg.mL1 = cfg.baseM;
    cfg.nL1 = n;
    cfg.kL1 = cfg.baseK * BFullLoadTiling::L1_BUFFER_NUM;

    int64_t tailM = m - (CeilDivI64(m, cfg.baseM) - 1) * cfg.baseM;
    if (tailM > 0 && tailM < cfg.baseM) {
        int64_t splitM = CeilDivI64(tailM, 2L);
        cfg.mTailCnt = static_cast<uint32_t>(CeilDivI64(tailM, splitM));
    }
    return cfg;
}

} // namespace

/* ========================================================================== */
/* Device-side kernel                                                         */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::B_FULL_LOAD_MODE>;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__global__ __aicore__ void mat_mul_b_fullload_torch_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, int64_t m, int64_t n,
                                                           int64_t k, BFullLoadTiling cfg)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBFullLoad<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, asc::te::nd_ext_layout_ptn, B_TYPE,
                                                    asc::te::nd_ext_layout_ptn, C_TYPE, asc::te::nd_ext_layout_ptn,
                                                    BIAS_TYPE, asc::te::nd_ext_layout_ptn>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {m, n, k, 1},
        {aGM, bGM, cGM, nullptr, nullptr, nullptr, static_cast<uint64_t>(k), static_cast<uint64_t>(cfg.mL1),
         static_cast<uint64_t>(cfg.nL1), static_cast<uint64_t>(cfg.kL1), static_cast<uint32_t>(cfg.baseM),
         static_cast<uint32_t>(cfg.baseN), static_cast<uint32_t>(cfg.baseK), BFullLoadTiling::L1_BUFFER_NUM,
         BFullLoadTiling::L0C_DB, 0, 1},
        {},
        {static_cast<uint32_t>(cfg.baseM), static_cast<uint32_t>(cfg.nL1), static_cast<uint32_t>(cfg.kL1),
         static_cast<uint32_t>(cfg.baseM), static_cast<uint32_t>(cfg.baseN), static_cast<uint32_t>(cfg.baseK),
         cfg.mTailCnt, cfg.nTailCnt, 1, 1, 1, 1, 0, Blaze::Gemm::L2_CACHE_DEFAULT}};

    MatmulKernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side launchers (extern "C", called from the torch binding)            */
/* ========================================================================== */

namespace {

template <class T>
void LaunchMatMulBFullLoad(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, int64_t m, int64_t n, int64_t k,
                           int64_t blockNum)
{
    BFullLoadTiling cfg = GetTiling(m, k, n);
    mat_mul_b_fullload_torch_kernel<T, T, T, T><<<blockNum, nullptr, stream>>>(a, b, c, m, n, k, cfg);
}

} // namespace

// Uniform launcher signature shared with the other variants (see
// mat_mul_torch_common.h); the workspace argument is unused here.
extern "C" void launch_mat_mul_b_fullload_f16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                              uint8_t* /*workspace*/, int64_t m, int64_t n, int64_t k, int64_t blockNum)
{
    LaunchMatMulBFullLoad<half>(stream, a, b, c, m, n, k, blockNum);
}

extern "C" void launch_mat_mul_b_fullload_bf16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                               uint8_t* /*workspace*/, int64_t m, int64_t n, int64_t k,
                                               int64_t blockNum)
{
    LaunchMatMulBFullLoad<bfloat16_t>(stream, a, b, c, m, n, k, blockNum);
}

extern "C" void launch_mat_mul_b_fullload_f32(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c,
                                              uint8_t* /*workspace*/, int64_t m, int64_t n, int64_t k, int64_t blockNum)
{
    LaunchMatMulBFullLoad<float>(stream, a, b, c, m, n, k, blockNum);
}
