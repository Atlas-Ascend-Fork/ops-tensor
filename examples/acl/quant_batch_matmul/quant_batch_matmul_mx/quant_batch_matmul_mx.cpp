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
 * @file quant_batch_matmul_mx.cpp
 * @brief Unified MX Quant Batch MatMul example covering batch, L0C ping-pong, and StreamK variants.
 *
 * Exercises Blaze::Gemm::Kernel::GemmUniversal (kernel_qbmm_mx.h specialization)
 * with Blaze::Gemm::MatmulWithScaleMx dispatch policy and
 * Blaze::Gemm::Block::BlockMmad (block_mmad_qbmm_mx.h specialization).
 *
 * Both A and B are MX-quantized (FP8/FP4), each carrying an independent E8M0 scale.
 * Supported transposes: transA (A stored as (K,M)), transB (B stored as (N,K))
 * Supported formats: (ND,ND), (ND,NZ)
 *
 * Supported A/B dtypes: fp8_e4m3, fp8_e5m2, fp4_e2m1 (A and B may differ)
 * Supported C dtypes:   float16, bfloat16, float32
 * Bias dtype:           float32
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Constants                                                                  */
/* ========================================================================== */

static constexpr uint64_t GROUP_SIZE = 32UL;
static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
static constexpr int64_t BLOCK_16 = 16L;
static constexpr uint64_t C0_SIZE_B8 = 32UL;
static constexpr uint64_t C0_SIZE_B4 = 64UL;
static constexpr uint64_t STREAMK_WORKSPACE_TILE_BYTES = 256UL * 256UL * sizeof(float);
static constexpr uint64_t STREAMK_WORKSPACE_OVERHEAD_BYTES = 20UL * 1024UL * 1024UL;
static constexpr uint32_t L2_CACHE_DEFAULT_VALUE = 0U;

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL(FULL_LOAD_MODE, WITHOUT_BATCH)                                                      \
    quant_batch_matmul_mx_kernel<A_TYPE, B_TYPE, C_TYPE, LAYOUT_A, LAYOUT_B, FULL_LOAD_MODE, WITHOUT_BATCH>    \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dBias, p.dScaleA, p.dScaleB, p.dC, p.m, p.k, p.n, p.baseM, \
                                      p.baseN, p.baseK, p.kL1, p.scaleKL1, p.l1BufferNum, p.dbL0C, p.biasElements)

#define DISPATCH_TRANS(A_TYPE, B_TYPE, C_TYPE, TRANS_A, TRANS_B, IS_NZ, FULL_LOAD_MODE, WITHOUT_BATCH)        \
    do {                                                                                                      \
        if (TRANS_A) {                                                                                        \
            if (TRANS_B) {                                                                                    \
                if (IS_NZ) {                                                                                  \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, true, true, true, FULL_LOAD_MODE, WITHOUT_BATCH>(    \
                        launchParams);                                                                        \
                } else {                                                                                      \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, true, true, false, FULL_LOAD_MODE, WITHOUT_BATCH>(   \
                        launchParams);                                                                        \
                }                                                                                             \
            } else {                                                                                          \
                if (IS_NZ) {                                                                                  \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, true, false, true, FULL_LOAD_MODE, WITHOUT_BATCH>(   \
                        launchParams);                                                                        \
                } else {                                                                                      \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, true, false, false, FULL_LOAD_MODE, WITHOUT_BATCH>(  \
                        launchParams);                                                                        \
                }                                                                                             \
            }                                                                                                 \
        } else {                                                                                              \
            if (TRANS_B) {                                                                                    \
                if (IS_NZ) {                                                                                  \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, false, true, true, FULL_LOAD_MODE, WITHOUT_BATCH>(   \
                        launchParams);                                                                        \
                } else {                                                                                      \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, false, true, false, FULL_LOAD_MODE, WITHOUT_BATCH>(  \
                        launchParams);                                                                        \
                }                                                                                             \
            } else {                                                                                          \
                if (IS_NZ) {                                                                                  \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, false, false, true, FULL_LOAD_MODE, WITHOUT_BATCH>(  \
                        launchParams);                                                                        \
                } else {                                                                                      \
                    LaunchKernel<A_TYPE, B_TYPE, C_TYPE, false, false, false, FULL_LOAD_MODE, WITHOUT_BATCH>( \
                        launchParams);                                                                        \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
    } while (0)

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t bias = 0;
    std::string aDtype;
    std::string bDtype;
    std::string cDtype;
    bool transA = false;
    bool transB = false;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    int64_t kL1 = 0;
    int64_t scaleKL1 = 0;
    int64_t l1Buffers = 2;
    int64_t dbL0C = 1;
    bool aFullLoad = false;
    bool withoutBatch = false;
    bool l0cPingPong = false;
    bool streamK = false;
    std::string kernelVariant = "batch";
    std::string format = "(ND,ND)";
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 20) {
        std::fprintf(stderr,
                     "Usage: %s <m> <k> <n> <bias> <a_dtype> <b_dtype> <c_dtype>"
                     " <transA> <transB> <format> <base_m> <base_n> <base_k> <tile_k_l1> <scale_k_l1>"
                     " <l1_buffers> <db_l0c> <a_full_load> <kernel_variant>\n",
                     argv[0]);
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.bias = std::atoll(argv[4]);
    args.aDtype = argv[5];
    args.bDtype = argv[6];
    args.cDtype = argv[7];
    args.transA = ParseBool(argv[8]);
    args.transB = ParseBool(argv[9]);
    args.format = argv[10];

    args.baseM = std::atoll(argv[11]);
    args.baseN = std::atoll(argv[12]);
    args.baseK = std::atoll(argv[13]);
    args.kL1 = std::atoll(argv[14]);
    args.scaleKL1 = std::atoll(argv[15]);
    args.l1Buffers = std::atoll(argv[16]);
    args.dbL0C = std::atoll(argv[17]);
    args.aFullLoad = ParseBool(argv[18]);
    args.kernelVariant = argv[19];
    if (args.kernelVariant != "batch" && args.kernelVariant != "without_batch" &&
        args.kernelVariant != "l0c_pingpong" && args.kernelVariant != "streamk") {
        std::fprintf(stderr, "Error: kernel_variant must be batch, without_batch, l0c_pingpong, or streamk.\n");
        return false;
    }
    args.withoutBatch = args.kernelVariant == "without_batch";
    args.l0cPingPong = args.kernelVariant == "l0c_pingpong";
    args.streamK = args.kernelVariant == "streamk";

    // Validation
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::fprintf(stderr, "Error: M, K, N must be positive integers.\n");
        return false;
    }
    if (args.bias != 0 && args.bias != args.n) {
        std::fprintf(stderr, "Error: bias (%ld) must equal n (%ld) or be 0.\n", args.bias, args.n);
        return false;
    }
    if (args.aDtype != "fp8_e4m3" && args.aDtype != "fp8_e5m2" && args.aDtype != "fp4_e2m1") {
        std::fprintf(stderr, "Error: A dtype must be fp8_e4m3, fp8_e5m2, or fp4_e2m1.\n");
        return false;
    }
    if (args.bDtype != "fp8_e4m3" && args.bDtype != "fp8_e5m2" && args.bDtype != "fp4_e2m1") {
        std::fprintf(stderr, "Error: B dtype must be fp8_e4m3, fp8_e5m2, or fp4_e2m1.\n");
        return false;
    }
    bool aIsFp4 = args.aDtype == "fp4_e2m1";
    bool bIsFp4 = args.bDtype == "fp4_e2m1";
    if (aIsFp4 != bIsFp4) {
        std::fprintf(stderr, "Error: FP4*FP8 mixed dtype is not supported by hardware (got a=%s b=%s).\n",
                     args.aDtype.c_str(), args.bDtype.c_str());
        return false;
    }
    if (args.format != "(ND,ND)" && args.format != "(ND,NZ)") {
        std::fprintf(stderr, "Error: format must be (ND,ND) or (ND,NZ) (got '%s').\n", args.format.c_str());
        return false;
    }
    if (args.cDtype != "float16" && args.cDtype != "bfloat16" && args.cDtype != "float32") {
        std::fprintf(stderr, "Error: C dtype must be float16, bfloat16, or float32 (got '%s').\n", args.cDtype.c_str());
        return false;
    }
    if (args.l0cPingPong) {
        const bool validL0cPingPong = args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3" &&
                                      args.cDtype == "float16" && !args.transA && !args.transB &&
                                      args.format == "(ND,ND)" && args.bias == 0 && args.dbL0C == 2 &&
                                      args.scaleKL1 == args.kL1;
        if (!validL0cPingPong) {
            std::fprintf(stderr, "Error: l0c_pingpong requires fp8_e4m3*fp8_e4m3 -> float16, ND/ND, no transpose, "
                                 "no bias, db_l0c=2, and scale_k_l1=tile_k_l1.\n");
            return false;
        }
    }
    if (args.streamK) {
        const bool validStreamK = args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e5m2" && args.cDtype == "float16" &&
                                  !args.transA && !args.transB && args.format == "(ND,ND)" && args.bias == 0 &&
                                  !args.aFullLoad && args.k % static_cast<int64_t>(MXFP_DIVISOR_SIZE) == 0;
        if (!validStreamK) {
            std::fprintf(stderr, "Error: streamk requires fp8_e4m3*fp8_e5m2 -> float16, ND/ND, no transpose, "
                                 "no bias, a_full_load=false, and K aligned to 64.\n");
            return false;
        }
    }
    return true;
}

/* ========================================================================== */
/* Dtype utilities                                                            */
/* ========================================================================== */

static bool IsFp4Dtype(const std::string& dtype) { return dtype == "fp4_e2m1"; }

static size_t DtypeSize(const std::string& dtype)
{
    if (dtype == "fp4_e2m1" || dtype == "fp8_e4m3" || dtype == "fp8_e5m2") {
        return 1;
    }
    if (dtype == "float16" || dtype == "bfloat16") {
        return 2;
    }
    if (dtype == "float32") {
        return 4;
    }
    std::fprintf(stderr, "Unknown dtype: %s\n", dtype.c_str());
    return 0;
}

static size_t ElementCountToBytes(int64_t count, const std::string& dtype)
{
    if (IsFp4Dtype(dtype)) {
        return static_cast<size_t>((count + 1) / 2) * DtypeSize(dtype);
    }
    return static_cast<size_t>(count) * DtypeSize(dtype);
}

/* ========================================================================== */
/* NZ format utilities (for (ND,NZ) format)                                   */
/* ========================================================================== */

static uint64_t GetC0Size(const std::string& dtype) { return IsFp4Dtype(dtype) ? C0_SIZE_B4 : C0_SIZE_B8; }

static int64_t CalcNZElementCount(int64_t k, int64_t n, const std::string& dtype, bool transB)
{
    uint64_t c0Size = GetC0Size(dtype);
    if (transB) {
        int64_t kCeil = CeilAlign(k, static_cast<int64_t>(c0Size));
        int64_t nCeil = CeilAlign(n, BLOCK_16);
        return kCeil * nCeil;
    }
    int64_t kCeil = CeilAlign(k, BLOCK_16);
    int64_t nCeil = CeilAlign(n, static_cast<int64_t>(c0Size));
    return kCeil * nCeil;
}

/* ========================================================================== */
/* Device-side kernel wrapper (template-based)                                */
/* ========================================================================== */

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

template <class AType, class BType, class CType, class LayoutA, class LayoutB, uint64_t FullLoadMode, bool WithoutBatch>
__global__ __aicore__ void quant_batch_matmul_mx_kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm,
                                                        GM_ADDR scaleBGm, GM_ADDR cGm, int64_t m, int64_t k, int64_t n,
                                                        uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t kL1,
                                                        uint64_t scaleKL1, uint64_t l1BufferNum, uint64_t dbL0C,
                                                        uint64_t biasElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BiasType = float;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ScheduleType = AscendC::Std::conditional_t<WithoutBatch, Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch,
                                                     Blaze::Gemm::KernelMmadWithScaleMx>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, ScheduleType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      BlockScheduler>;
    typename Kernel::Params params{};
    params.problemShape = ProblemShape{m, n, k, 1};
    params.mmadParams = {aGm, bGm, cGm, biasGm, scaleAGm, scaleBGm};
    params.l1Params = {kL1, scaleKL1, l1BufferNum};
    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    if constexpr (!WithoutBatch) {
        params.qbmmParams.batchA1 = 1U;
        params.qbmmParams.batchA2 = 1U;
        params.qbmmParams.batchA3 = 1U;
        params.qbmmParams.batchA4 = 1U;
        params.qbmmParams.batchB1 = 1U;
        params.qbmmParams.batchB2 = 1U;
        params.qbmmParams.batchB3 = 1U;
        params.qbmmParams.batchB4 = 1U;
        params.qbmmParams.batchC1 = 1U;
        params.qbmmParams.batchC2 = 1U;
        params.qbmmParams.batchC3 = 1U;
        params.qbmmParams.batchC4 = 1U;
        params.qbmmParams.biasThreeDim = 0U;
    }
    params.qbmmParams.baseM = static_cast<uint32_t>(baseM);
    params.qbmmParams.baseN = static_cast<uint32_t>(baseN);
    params.qbmmParams.baseK = static_cast<uint32_t>(baseK);
    params.qbmmParams.isBias = biasElements != 0U ? 1U : 0U;
    params.qbmmParams.dbL0C = static_cast<uint32_t>(dbL0C);
    params.qbmmParams.bMustHitL2 = 1U;
    Kernel kernel;
    kernel(params);
}

template <uint64_t FullLoadMode>
__global__ __aicore__ void quant_batch_matmul_mx_l0c_pingpong_kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleAGm,
                                                                     GM_ADDR scaleBGm, GM_ADDR cGm, int64_t m,
                                                                     int64_t k, int64_t n, uint64_t baseM,
                                                                     uint64_t baseN, uint64_t baseK, uint64_t kL1,
                                                                     uint64_t l1BufferNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, Layout,
                                                                                Layout, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, Layout, BType, Layout, half, Layout, float,
                                                    Layout>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      BlockScheduler>;

    typename Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams = {aGm, bGm, cGm, nullptr, scaleAGm, scaleBGm};
    params.l1Params = {kL1, kL1, l1BufferNum};
    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    params.qbmmParams.batchA1 = 1U;
    params.qbmmParams.batchA2 = 1U;
    params.qbmmParams.batchA3 = 1U;
    params.qbmmParams.batchA4 = 1U;
    params.qbmmParams.batchB1 = 1U;
    params.qbmmParams.batchB2 = 1U;
    params.qbmmParams.batchB3 = 1U;
    params.qbmmParams.batchB4 = 1U;
    params.qbmmParams.batchC1 = 1U;
    params.qbmmParams.batchC2 = 1U;
    params.qbmmParams.batchC3 = 1U;
    params.qbmmParams.batchC4 = 1U;
    params.qbmmParams.biasThreeDim = 0U;
    params.qbmmParams.baseM = static_cast<uint32_t>(baseM);
    params.qbmmParams.baseN = static_cast<uint32_t>(baseN);
    params.qbmmParams.baseK = static_cast<uint32_t>(baseK);
    params.qbmmParams.isBias = 0U;
    params.qbmmParams.dbL0C = 2U;
    params.qbmmParams.bMustHitL2 = 1U;
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void quant_batch_matmul_mx_streamk_kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR scaleAGm,
                                                                GM_ADDR scaleBGm, GM_ADDR cGm, GM_ADDR workspaceGm,
                                                                int64_t m, int64_t k, int64_t n, uint32_t usedCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e5m2_t;
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpiloguePolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, Layout, BType, Layout, half, Layout, float,
                                                    Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, half, EpiloguePolicy>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams = {aGm, bGm, cGm, nullptr, scaleAGm, scaleBGm};
    params.epilogueParams = {cGm, workspaceGm};
    params.schParams = {usedCoreNum, m, 32, 64, 64, 64, 0U, L2_CACHE_DEFAULT_VALUE};
    params.qbmmParams = {64U, 1U, 1U};
    Kernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side kernel launcher                                                  */
/* ========================================================================== */

namespace {

struct LaunchParams {
    uint8_t* dA;
    uint8_t* dB;
    uint8_t* dBias;
    uint8_t* dScaleA;
    uint8_t* dScaleB;
    uint8_t* dC;
    int64_t m, n, k;
    int64_t blockNum;
    uint64_t baseM, baseN, baseK;
    uint64_t kL1, scaleKL1, l1BufferNum, dbL0C, biasElements;
    aclrtStream stream;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, bool TransA, bool TransB, bool IsNzFormat, uint64_t FullLoadMode,
          bool WithoutBatch>
void LaunchKernel(const LaunchParams& p)
{
    using BiasType = float;
    using LAYOUT_A = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LAYOUT_B = std::conditional_t<
        IsNzFormat, std::conditional_t<TransB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>>;

    LAUNCH_KERNEL_IMPL(FullLoadMode, WithoutBatch);
}

template <class A_TYPE, class B_TYPE, uint64_t FullLoadMode, bool WithoutBatch>
void LaunchByC(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    if (args.cDtype == "float32") {
        DISPATCH_TRANS(A_TYPE, B_TYPE, float, args.transA, args.transB, isNzFormat, FullLoadMode, WithoutBatch);
    } else if (args.cDtype == "float16") {
        DISPATCH_TRANS(A_TYPE, B_TYPE, half, args.transA, args.transB, isNzFormat, FullLoadMode, WithoutBatch);
    } else if (args.cDtype == "bfloat16") {
        DISPATCH_TRANS(A_TYPE, B_TYPE, bfloat16_t, args.transA, args.transB, isNzFormat, FullLoadMode, WithoutBatch);
    } else {
        std::fprintf(stderr, "Unsupported C dtype: %s\n", args.cDtype.c_str());
        std::exit(1);
    }
}

template <class A_TYPE, class B_TYPE, bool WithoutBatch>
void LaunchByFullLoad(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    if (args.aFullLoad) {
        LaunchByC<A_TYPE, B_TYPE, Blaze::Gemm::A_FULL_LOAD_MODE, WithoutBatch>(args, isNzFormat, launchParams);
    } else {
        LaunchByC<A_TYPE, B_TYPE, Blaze::Gemm::NONE_FULL_LOAD_MODE, WithoutBatch>(args, isNzFormat, launchParams);
    }
}

void LaunchWithoutBatch(const CliArgs& args, const LaunchParams& launchParams)
{
    if (args.aFullLoad) {
        LaunchKernel<fp8_e4m3fn_t, fp8_e4m3fn_t, half, false, false, false, Blaze::Gemm::A_FULL_LOAD_MODE, true>(
            launchParams);
    } else {
        LaunchKernel<fp8_e4m3fn_t, fp8_e4m3fn_t, half, false, false, false, Blaze::Gemm::NONE_FULL_LOAD_MODE, true>(
            launchParams);
    }
}

void LaunchL0cPingPong(const CliArgs& args, const LaunchParams& p)
{
    if (args.aFullLoad) {
        quant_batch_matmul_mx_l0c_pingpong_kernel<Blaze::Gemm::A_FULL_LOAD_MODE><<<p.blockNum, 0, p.stream>>>(
            p.dA, p.dB, p.dScaleA, p.dScaleB, p.dC, p.m, p.k, p.n, p.baseM, p.baseN, p.baseK, p.kL1, p.l1BufferNum);
    } else {
        quant_batch_matmul_mx_l0c_pingpong_kernel<Blaze::Gemm::NONE_FULL_LOAD_MODE><<<p.blockNum, 0, p.stream>>>(
            p.dA, p.dB, p.dScaleA, p.dScaleB, p.dC, p.m, p.k, p.n, p.baseM, p.baseN, p.baseK, p.kL1, p.l1BufferNum);
    }
}

void LaunchStreamK(const LaunchParams& p, uint8_t* workspace)
{
    const uint32_t usedCoreNum = static_cast<uint32_t>(p.blockNum);
    quant_batch_matmul_mx_streamk_kernel<<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dScaleA, p.dScaleB, p.dC, workspace,
                                                                      p.m, p.k, p.n, usedCoreNum);
}

} // namespace

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);

    int64_t blockNum = GetAicCoreNum();
    if (blockNum <= 0) {
        std::cerr << "blockNum cannot be less than 0, but current: " << blockNum << std::endl;
        return;
    }

    const uint64_t scaleK = static_cast<uint64_t>(CeilDiv(args.k, static_cast<int64_t>(MXFP_DIVISOR_SIZE))) *
                            (MXFP_DIVISOR_SIZE / GROUP_SIZE);
    const size_t aSize = ElementCountToBytes(args.m * args.k, args.aDtype);
    bool isNzFormat = (args.format == "(ND,NZ)");
    size_t bSize;
    if (isNzFormat) {
        int64_t nzElementCount = CalcNZElementCount(args.k, args.n, args.bDtype, args.transB);
        bSize = ElementCountToBytes(nzElementCount, args.bDtype);
    } else {
        bSize = ElementCountToBytes(args.k * args.n, args.bDtype);
    }
    const size_t biasSize = args.bias > 0 ? static_cast<size_t>(args.bias) * sizeof(float) : 1;
    const size_t scaleASize = static_cast<size_t>(args.m) * scaleK;
    const size_t scaleBSize = static_cast<size_t>(args.n) * scaleK;
    const size_t cSize = static_cast<size_t>(args.m * args.n) * DtypeSize(args.cDtype);

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    // Allocate host buffers
    std::vector<uint8_t> hostA(aSize);
    std::vector<uint8_t> hostB(bSize);
    std::vector<uint8_t> hostC(cSize, 0);
    std::vector<uint8_t> hostBias(biasSize, 0);
    std::vector<uint8_t> hostScaleA(scaleASize);
    std::vector<uint8_t> hostScaleB(scaleBSize);

    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), aSize)) {
        std::cerr << "Failed to read input A" << std::endl;
        return;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), bSize)) {
        std::cerr << "Failed to read input B" << std::endl;
        return;
    }
    if (!ReadFile(inputDir + "/scale_a.bin", hostScaleA.data(), scaleASize)) {
        std::cerr << "Failed to read scale A" << std::endl;
        return;
    }
    if (!ReadFile(inputDir + "/scale_b.bin", hostScaleB.data(), scaleBSize)) {
        std::cerr << "Failed to read scale B" << std::endl;
        return;
    }
    if (args.bias > 0) {
        if (!ReadFile(inputDir + "/bias.bin", hostBias.data(), biasSize)) {
            std::cerr << "Failed to read bias" << std::endl;
            return;
        }
    }
    if (!ReadFile(inputDir + "/initial_c.bin", hostC.data(), cSize)) {
        std::cerr << "Failed to read initial C" << std::endl;
        return;
    }

    // Allocate device buffers
    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceBias{nullptr};
    uint8_t* deviceScaleA{nullptr};
    uint8_t* deviceScaleB{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    size_t workspaceSize = 0U;

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleA), scaleASize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleB), scaleBSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (args.streamK) {
        workspaceSize = static_cast<size_t>(blockNum) * STREAMK_WORKSPACE_TILE_BYTES + STREAMK_WORKSPACE_OVERHEAD_BYTES;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(deviceWorkspace, workspaceSize, 0, workspaceSize));
    }

    ACL_CHECK(aclrtMemcpy(deviceA, aSize, hostA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, bSize, hostB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, cSize, hostC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBias, biasSize, hostBias.data(), biasSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, scaleASize, hostScaleA.data(), scaleASize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, scaleBSize, hostScaleB.data(), scaleBSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Print execution summary
    std::cout << "============================================================" << std::endl;
    std::cout << "  QuantBatchMatmul MX — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << std::endl;
    std::cout << "  A Dtype  : " << args.aDtype << std::endl;
    std::cout << "  B Dtype  : " << args.bDtype << std::endl;
    std::cout << "  C Dtype  : " << args.cDtype << std::endl;
    std::cout << "  transA   : " << (args.transA ? "true" : "false") << std::endl;
    std::cout << "  transB   : " << (args.transB ? "true" : "false") << std::endl;
    std::cout << "  Bias     : " << args.bias << " (float32)" << std::endl;
    std::cout << "  L0 Tile  : [" << args.baseM << ", " << args.baseN << ", " << args.baseK << "]" << std::endl;
    std::cout << "  L1 kL1   : " << args.kL1 << std::endl;
    std::cout << "  scaleKL1 : " << args.scaleKL1 << std::endl;
    std::cout << "  l1Buffers: " << args.l1Buffers << std::endl;
    std::cout << "  dbL0C    : " << args.dbL0C << std::endl;
    std::cout << "  AFullLoad: " << (args.aFullLoad ? "true" : "false") << std::endl;
    std::cout << "  Kernel   : " << args.kernelVariant << std::endl;
    std::cout << "  Format   : " << args.format << std::endl;
    std::cout << "  BlockNum : " << blockNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams launchParams = {deviceA,
                                 deviceB,
                                 deviceBias,
                                 deviceScaleA,
                                 deviceScaleB,
                                 deviceC,
                                 args.m,
                                 args.n,
                                 args.k,
                                 blockNum,
                                 static_cast<uint64_t>(args.baseM),
                                 static_cast<uint64_t>(args.baseN),
                                 static_cast<uint64_t>(args.baseK),
                                 static_cast<uint64_t>(args.kL1),
                                 static_cast<uint64_t>(args.scaleKL1),
                                 static_cast<uint64_t>(args.l1Buffers),
                                 static_cast<uint64_t>(args.dbL0C),
                                 static_cast<uint64_t>(args.bias),
                                 stream};

    if (args.l0cPingPong) {
        LaunchL0cPingPong(args, launchParams);
    } else if (args.streamK) {
        LaunchStreamK(launchParams, deviceWorkspace);
    } else if (args.withoutBatch) {
        const bool supportedWithoutBatch = args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3" &&
                                           args.cDtype == "float16" && !args.transA && !args.transB && !isNzFormat &&
                                           args.bias == 0;
        if (!supportedWithoutBatch) {
            std::fprintf(stderr,
                         "The without_batch example is fixed to fp8_e4m3*fp8_e4m3, float16, ND/ND, no transpose, "
                         "and no bias.\n");
            std::exit(1);
        }
        LaunchWithoutBatch(args, launchParams);
    } else if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3") {
        LaunchByFullLoad<fp8_e4m3fn_t, fp8_e4m3fn_t, false>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e5m2") {
        LaunchByFullLoad<fp8_e4m3fn_t, fp8_e5m2_t, false>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp8_e5m2" && args.bDtype == "fp8_e4m3") {
        LaunchByFullLoad<fp8_e5m2_t, fp8_e4m3fn_t, false>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp8_e5m2" && args.bDtype == "fp8_e5m2") {
        LaunchByFullLoad<fp8_e5m2_t, fp8_e5m2_t, false>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp4_e2m1" && args.bDtype == "fp4_e2m1") {
        LaunchByFullLoad<fp4x2_e2m1_t, fp4x2_e2m1_t, false>(args, isNzFormat, launchParams);
    } else {
        std::fprintf(stderr,
                     "Unsupported A/B dtype combination: a=%s b=%s.\n"
                     "Supported: fp8_e4m3*fp8_e4m3, fp8_e4m3*fp8_e5m2, fp8_e5m2*fp8_e4m3,\n"
                     "          fp8_e5m2*fp8_e5m2, fp4_e2m1*fp4_e2m1.\n",
                     args.aDtype.c_str(), args.bDtype.c_str());
        std::exit(1);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), cSize, deviceC, cSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Write output
    std::string outPath = outputDir + "/npu_out.bin";
    if (!WriteFile(outPath, hostC.data(), cSize)) {
        std::cerr << "Failed to write output" << std::endl;
    } else {
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(deviceScaleA));
    ACL_CHECK(aclrtFree(deviceScaleB));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

/* ========================================================================== */
/* Entry point                                                                */
/* ========================================================================== */

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseCliArgs(argc, argv, args)) {
        return 1;
    }

    Run(args);
    return 0;
}
