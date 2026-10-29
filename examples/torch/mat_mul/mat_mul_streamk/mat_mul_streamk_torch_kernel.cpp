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
 * @file mat_mul_streamk_torch_kernel.cpp
 * @brief Blaze MatMul StreamK kernel and host launchers for the torch extension example.
 *
 * StreamK splits the K dimension across cores when the M*N tile grid cannot
 * fill all AIC cores. Tiling policy follows acl/mat_mul/mat_mul_streamk
 * (ND layout only). The kernel needs a GM workspace for cross-core partial
 * accumulation; it is allocated by the torch binding and passed in here.
 *
 * Supported dtypes: float16, bfloat16, float32
 * Layout: ND x ND -> ND (no transpose, no bias)
 */

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "kernel_basic_intf.h"

/* ========================================================================== */
/* Tiling (same policy as mat_mul_streamk)                                    */
/* ========================================================================== */

namespace {

struct StreamKTiling {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    int64_t skSingleK = 0;
    bool isNdFixpipe12 = false;
    static constexpr int64_t L1_BUFFER_NUM = 2;
    static constexpr int64_t L0C_DB = 1;
};

inline int64_t CeilDivI64(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t CeilAlignI64(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

// L0B capacity in bytes, and the double buffering that baseK must leave room for.
constexpr int64_t kL0BBytes = 64L * 1024L;
constexpr int64_t kL0BBufferNum = 2;
// ND x ND (transA=false, transB=false): K rows are aligned to 128 bytes.
constexpr int64_t kAlignBytes = 128L;
// Thresholds gating the ND_FIXPIPE_1_2 fixpipe path.
constexpr int64_t kNdFixpipe12MinN = 64L;
constexpr int64_t kNdFixpipe12NAlign = 16L;
constexpr int64_t kNdFixpipe12MinM = 2L;
constexpr int64_t kNdFixpipe12MinMN = 256L;

template <class T>
StreamKTiling GetTiling(int64_t m, int64_t k, int64_t n, int64_t aicNum)
{
    StreamKTiling cfg;
    // The binding returns empty outputs for zero-sized problems without
    // launching; guard here too so the tile counts below can never become
    // zero divisors.
    if (m <= 0 || n <= 0) {
        return cfg;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        cfg.baseM = 128;
        cfg.baseN = 256;
        cfg.baseK = 64;
    } else {
        cfg.baseM = 256;
        cfg.baseN = 256;
        cfg.baseK = 64;
    }

    int64_t mCnt = CeilDivI64(m, cfg.baseM);
    int64_t nCnt = CeilDivI64(n, cfg.baseN);
    int64_t totalMNCnt = mCnt * nCnt;

    if (totalMNCnt <= aicNum / 2) {
        cfg.baseM = CeilAlignI64(CeilDivI64(m, mCnt), 16L);
        cfg.baseN = CeilAlignI64(CeilDivI64(n, nCnt), 16L);
        int64_t kCnt = aicNum / totalMNCnt;
        cfg.skSingleK = CeilDivI64(k, kCnt);
    } else {
        int64_t rem = totalMNCnt % aicNum;
        int64_t kCnt = (rem > 0) ? (aicNum / rem) : 1;
        int64_t skSingleCoreK = CeilDivI64(k, kCnt);
        cfg.skSingleK = skSingleCoreK;
    }

    int64_t baseKAlignValue = kAlignBytes / static_cast<int64_t>(sizeof(T));
    int64_t kValueMax = (kL0BBytes / kL0BBufferNum / static_cast<int64_t>(sizeof(T)) / std::max(cfg.baseM, cfg.baseN)) /
                        baseKAlignValue * baseKAlignValue;
    cfg.baseK = std::min(cfg.skSingleK, kValueMax);
    cfg.mL1 = cfg.baseM;
    cfg.nL1 = cfg.baseN;
    cfg.kL1 = cfg.baseK * 2; // baseK * stepKa(2)

    cfg.isNdFixpipe12 = (n > kNdFixpipe12MinN && n % kNdFixpipe12NAlign != 0 && m > kNdFixpipe12MinM &&
                         m * n >= kNdFixpipe12MinMN);
    return cfg;
}

} // namespace

/* ========================================================================== */
/* Device-side kernel                                                         */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, Blaze::Gemm::MatMulL0C2Out L0C2OUT>
__global__ __aicore__ void mat_mul_streamk_torch_kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                        int64_t m, int64_t n, int64_t k, int64_t usedCoreNum,
                                                        StreamKTiling cfg)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<L0C2OUT, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, A_TYPE, asc::te::nd_ext_layout_ptn, B_TYPE,
                                                    asc::te::nd_ext_layout_ptn, C_TYPE, asc::te::nd_ext_layout_ptn,
                                                    BIAS_TYPE, asc::te::nd_ext_layout_ptn>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<float, C_TYPE, DispatchPolicy>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {m, n, k, 1},
        {aGM, bGM, cGM, nullptr, nullptr, workspaceGM, static_cast<uint32_t>(cfg.mL1), static_cast<uint32_t>(cfg.nL1),
         static_cast<uint32_t>(cfg.kL1), static_cast<uint32_t>(cfg.baseM), static_cast<uint32_t>(cfg.baseN),
         static_cast<uint32_t>(cfg.baseK), StreamKTiling::L1_BUFFER_NUM, StreamKTiling::L0C_DB},
        {cGM, workspaceGM},
        {static_cast<uint32_t>(usedCoreNum), static_cast<uint32_t>(cfg.baseM), static_cast<uint32_t>(cfg.baseN),
         static_cast<uint32_t>(cfg.baseK), static_cast<uint32_t>(cfg.skSingleK), static_cast<uint32_t>(cfg.kL1), 0,
         Blaze::Gemm::L2_CACHE_DEFAULT}};

    MatmulKernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side launchers (extern "C", called from the torch binding)            */
/* ========================================================================== */

namespace {

template <class T>
void LaunchMatMulStreamK(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, int64_t m,
                         int64_t n, int64_t k, int64_t blockNum)
{
    if (m == 0 || n == 0) {
        return;
    }
    StreamKTiling cfg = GetTiling<T>(m, k, n, blockNum);
    if (cfg.isNdFixpipe12) {
        mat_mul_streamk_torch_kernel<T, T, T, T, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>
            <<<blockNum, nullptr, stream>>>(a, b, c, workspace, m, n, k, blockNum, cfg);
    } else {
        mat_mul_streamk_torch_kernel<T, T, T, T, Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>
            <<<blockNum, nullptr, stream>>>(a, b, c, workspace, m, n, k, blockNum, cfg);
    }
}

} // namespace

extern "C" void launch_mat_mul_streamk_f16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                           int64_t m, int64_t n, int64_t k, int64_t blockNum)
{
    LaunchMatMulStreamK<half>(stream, a, b, c, workspace, m, n, k, blockNum);
}

extern "C" void launch_mat_mul_streamk_bf16(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                            int64_t m, int64_t n, int64_t k, int64_t blockNum)
{
    LaunchMatMulStreamK<bfloat16_t>(stream, a, b, c, workspace, m, n, k, blockNum);
}

extern "C" void launch_mat_mul_streamk_f32(aclrtStream stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
                                           int64_t m, int64_t n, int64_t k, int64_t blockNum)
{
    LaunchMatMulStreamK<float>(stream, a, b, c, workspace, m, n, k, blockNum);
}
