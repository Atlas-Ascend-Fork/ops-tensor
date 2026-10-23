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
 * @file quant_matmul_activation_quant.cpp
 * @brief CSV-driven MX quantized matmul + GELU/SwiGLU/Swish + dynamic MX quant fusion example.
 *
 * Both batch and without_batch options use the same GemmUniversal kernel with batch=1.
 *
 * Both A and B are MX-quantized (FP8 or FP4), each carrying an independent E8M0 scale.
 * AIC does MX GEMM (L0C→UB via DualDst fixpipe), AIV does activation and dynamic
 * MX quantization, outputting quantized Y (same dtype as A) + fp8_e8m0 scale.
 *
 * Supported transposes: transA (A stored as (K,M)), transB (B stored as (N,K))
 * Weight B supports ND and NZ layouts (DN/ZN when transB=true).
 *
 * For Swish in ND, A and B can independently use fp8_e4m3 or fp8_e5m2.
 * In NZ, B must use fp8_e4m3 and A must not be transposed.
 * For GELU/SwiGLU with FP8 inputs, B must use fp8_e4m3; A may use fp8_e4m3 or fp8_e5m2.
 * GELU also supports fp4_e2m1 for both A and B. Swish uses FP8 output with RINT only.
 * The silu entry selects Swish with beta=1.
 * C dtype (matmul intermediate / epilogue input): float32 (L0C accumulator, DualDst to UB).
 * Bias dtype: float32
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_gelu_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swish_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Constants                                                                  */
/* ========================================================================== */

static constexpr uint64_t GROUP_SIZE = 32UL;
static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
static constexpr int64_t BLOCK_16 = 16L;
static constexpr uint64_t C0_SIZE_B8 = 32UL;
static constexpr uint64_t C0_SIZE_B4 = 64UL;

enum class ActivationKind { GELU, SWIGLU, SWISH };

/* ========================================================================== */
/* Macros                                                                     */
/* ========================================================================== */

#define LAUNCH_KERNEL_IMPL(FULL_LOAD_MODE, ACTIVATION)                                                           \
    quant_matmul_activation_quant_kernel<A_TYPE, B_TYPE, A_TYPE, LAYOUT_A, LAYOUT_B, FULL_LOAD_MODE, ACTIVATION> \
        <<<p.blockNum, 0, p.stream>>>(p.dA, p.dB, p.dBias, p.dScaleA, p.dScaleB, p.dY, p.dYScale, p.m, p.k, p.n, \
                                      p.baseM, p.baseN, p.baseK, p.kL1, p.scaleKL1, p.l1BufferNum, p.dbL0C,      \
                                      p.biasElements, p.scaleAlg, p.beta)

#define DISPATCH_TRANS(A_TYPE, B_TYPE, TRANS_A, TRANS_B, IS_NZ, FULL_LOAD_MODE, ACTIVATION)                      \
    do {                                                                                                         \
        if (TRANS_A) {                                                                                           \
            if (TRANS_B) {                                                                                       \
                if (IS_NZ) {                                                                                     \
                    LaunchKernel<A_TYPE, B_TYPE, true, true, true, FULL_LOAD_MODE, ACTIVATION>(launchParams);    \
                } else {                                                                                         \
                    LaunchKernel<A_TYPE, B_TYPE, true, true, false, FULL_LOAD_MODE, ACTIVATION>(launchParams);   \
                }                                                                                                \
            } else {                                                                                             \
                if (IS_NZ) {                                                                                     \
                    LaunchKernel<A_TYPE, B_TYPE, true, false, true, FULL_LOAD_MODE, ACTIVATION>(launchParams);   \
                } else {                                                                                         \
                    LaunchKernel<A_TYPE, B_TYPE, true, false, false, FULL_LOAD_MODE, ACTIVATION>(launchParams);  \
                }                                                                                                \
            }                                                                                                    \
        } else {                                                                                                 \
            if (TRANS_B) {                                                                                       \
                if (IS_NZ) {                                                                                     \
                    LaunchKernel<A_TYPE, B_TYPE, false, true, true, FULL_LOAD_MODE, ACTIVATION>(launchParams);   \
                } else {                                                                                         \
                    LaunchKernel<A_TYPE, B_TYPE, false, true, false, FULL_LOAD_MODE, ACTIVATION>(launchParams);  \
                }                                                                                                \
            } else {                                                                                             \
                if (IS_NZ) {                                                                                     \
                    LaunchKernel<A_TYPE, B_TYPE, false, false, true, FULL_LOAD_MODE, ACTIVATION>(launchParams);  \
                } else {                                                                                         \
                    LaunchKernel<A_TYPE, B_TYPE, false, false, false, FULL_LOAD_MODE, ACTIVATION>(launchParams); \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
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
    bool transA = false;
    bool transB = false;
    std::string format = "(ND,NZ)";
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    int64_t kL1 = 0;
    int64_t scaleKL1 = 0;
    int64_t l1Buffers = 2;
    int64_t dbL0C = 1;
    bool aFullLoad = false;
    std::string activation = "gelu";
    uint32_t scaleAlg = 0;
    std::string kernelVariant = "batch";
    float beta = 1.0F;
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 21 && argc != 22) {
        std::fprintf(stderr,
                     "Usage: %s <m> <k> <n> <bias> <a_dtype> <b_dtype>"
                     " <transA> <transB> <format> <base_m> <base_n> <base_k> <tile_k_l1> <scale_k_l1>"
                     " <l1_buffers> <db_l0c> <a_full_load> <gelu|swiglu|silu|swish> <scale_alg>"
                     " <batch|without_batch> [beta=1]\n",
                     argv[0]);
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);
    args.n = std::atoll(argv[3]);
    args.bias = std::atoll(argv[4]);
    args.aDtype = argv[5];
    args.bDtype = argv[6];
    args.transA = ParseBool(argv[7]);
    args.transB = ParseBool(argv[8]);
    args.format = argv[9];

    args.baseM = std::atoll(argv[10]);
    args.baseN = std::atoll(argv[11]);
    args.baseK = std::atoll(argv[12]);
    args.kL1 = std::atoll(argv[13]);
    args.scaleKL1 = std::atoll(argv[14]);
    args.l1Buffers = std::atoll(argv[15]);
    args.dbL0C = std::atoll(argv[16]);
    args.aFullLoad = ParseBool(argv[17]);
    args.activation = argv[18];
    args.scaleAlg = static_cast<uint32_t>(std::atoll(argv[19]));
    args.kernelVariant = argv[20];
    if (argc == 22) {
        char* end = nullptr;
        args.beta = std::strtof(argv[21], &end);
        if (end == argv[21] || *end != '\0' || !std::isfinite(args.beta)) {
            std::fprintf(stderr, "Error: beta must be a finite FP32 scalar.\n");
            return false;
        }
    }
    if (args.activation != "swish" && args.beta != 1.0F) {
        std::fprintf(stderr, "Error: beta can differ from 1 only for swish.\n");
        return false;
    }

    // Validation
    if (args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::fprintf(stderr, "Error: M, K, N must be positive integers.\n");
        return false;
    }
    if (args.baseM <= 0) {
        std::fprintf(stderr, "Error: base_m must be a positive integer.\n");
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
    if (args.activation != "gelu" && args.activation != "swiglu" && args.activation != "silu" &&
        args.activation != "swish") {
        std::fprintf(stderr, "Error: activation must be gelu, swiglu, silu, or swish.\n");
        return false;
    }
    if (args.bDtype == "fp8_e5m2" && args.activation != "silu" && args.activation != "swish") {
        std::fprintf(stderr, "Error: this example enables E5M2 B only for Swish/SiLU.\n");
        return false;
    }
    if (args.scaleAlg > 1U) {
        std::fprintf(stderr, "Error: scale_alg must be 0 (OCP) or 1 (BLAS).\n");
        return false;
    }
    if (args.scaleAlg == 1U && aIsFp4) {
        std::fprintf(stderr, "Error: BLAS scale is only enabled for FP8 output in this example.\n");
        return false;
    }
    if (args.kernelVariant != "batch" && args.kernelVariant != "without_batch") {
        std::fprintf(stderr, "Error: kernel_variant must be batch or without_batch.\n");
        return false;
    }
    if (args.activation == "swiglu") {
        if (args.kernelVariant != "without_batch" || args.transA || aIsFp4 || bIsFp4 || args.n % 64 != 0) {
            std::fprintf(stderr,
                         "Error: SwiGLU example requires without_batch, transA=false, FP8 inputs, and N multiple "
                         "of 64.\n");
            return false;
        }
    }
    if (args.activation == "silu" || args.activation == "swish") {
        if (aIsFp4 || bIsFp4 || (args.format == "(ND,NZ)" && (args.transA || args.bDtype != "fp8_e4m3"))) {
            std::fprintf(stderr, "Error: Swish/SiLU requires FP8 inputs; NZ requires transA=false and E4M3 B.\n");
            return false;
        }
        if (args.baseM <= 0 || args.baseM > 256 || args.baseM % 16 != 0 || args.baseN <= 0 || args.baseN > 256 ||
            args.baseN % GROUP_SIZE != 0) {
            std::fprintf(stderr, "Error: Swish/SiLU requires base_m in [16,256] aligned to 16 and "
                                 "base_n in [32,256] aligned to 32.\n");
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
    if (dtype == "fp8_e4m3" || dtype == "fp8_e5m2" || dtype == "fp4_e2m1") {
        return 1;
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
/* NZ format utilities                                                       */
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

template <class AType, class BType, class CType, class LayoutA, class LayoutB, uint64_t FullLoadMode,
          ActivationKind Activation>
__global__ __aicore__ void quant_matmul_activation_quant_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR biasGm, GM_ADDR scaleAGm, GM_ADDR scaleBGm, GM_ADDR yGm, GM_ADDR yScaleGm,
    int64_t m, int64_t k, int64_t n, uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t kL1, uint64_t scaleKL1,
    uint64_t l1BufferNum, uint64_t dbL0C, uint64_t biasElements, uint32_t scaleAlg, float beta)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using OutType = CType;
    using MatmulOutType = float;
    using BiasType = float;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    constexpr bool IsSwiGlu = Activation == ActivationKind::SWIGLU;
    // Single-batch cases reuse the batch kernel with default batch parameters (all batch dims 1).
    // SwiGLU selects the concat-N layout via the policy flag (mmad produces [gate|linear], epilogue N/2).
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M, 0, IsSwiGlu>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA,
                                                                                LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, MatmulOutType,
                                                    LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = AscendC::Std::conditional_t<
        IsSwiGlu, Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutType, MatmulOutType, fp8_e8m0_t>,
        AscendC::Std::conditional_t<Activation == ActivationKind::SWISH,
                                    Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<OutType, MatmulOutType>,
                                    Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutType, MatmulOutType>>>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    typename Kernel::Params params{};
    params.problemShape = ProblemShape{m, n, k, 1};
    params.mmadParams = {aGm, bGm, yGm, biasGm, scaleAGm, scaleBGm};
    float dtypeMaxVal = 0.0f;
    if constexpr (AscendC::IsSameType<OutType, fp4x2_e2m1_t>::value) {
        dtypeMaxVal = 6.0f;
    }
    if constexpr (IsSwiGlu) {
        params.epilogueParams.yGmAddr = yGm;
        params.epilogueParams.yScaleGmAddr = yScaleGm;
        params.epilogueParams.baseM = static_cast<uint32_t>(baseM);
        params.epilogueParams.baseN = static_cast<uint32_t>(baseN >> 1);
        params.epilogueParams.scaleAlg = static_cast<uint8_t>(scaleAlg);
    } else if constexpr (Activation == ActivationKind::SWISH) {
        params.epilogueParams = {yGm,      yScaleGm, static_cast<uint32_t>(baseM), static_cast<uint32_t>(baseN),
                                 scaleAlg, beta};
    } else {
        params.epilogueParams = {yGm,
                                 yScaleGm,
                                 static_cast<uint32_t>(baseM),
                                 static_cast<uint32_t>(baseN),
                                 Blaze::Epilogue::Block::GeluAlg::TANH,
                                 static_cast<Blaze::Epilogue::Block::QuantAlg>(scaleAlg),
                                 Blaze::Epilogue::Block::ROUND_MODE_FP4::RINT,
                                 dtypeMaxVal};
    }
    params.l1Params = {kL1, scaleKL1, l1BufferNum};
    params.schParams = {static_cast<int64_t>(baseM), static_cast<int64_t>(baseN), 1, 1, 1, 1, 0, 0};
    // Batch parameters default to 1 for the example's single-batch cases.
    params.qbmmParams = {1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         0,
                         static_cast<uint32_t>(baseM),
                         static_cast<uint32_t>(baseN),
                         static_cast<uint32_t>(baseK),
                         biasElements != 0U ? 1U : 0U,
                         static_cast<uint32_t>(dbL0C)};
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
    uint8_t* dY;
    uint8_t* dYScale;
    int64_t m, n, k;
    int64_t blockNum;
    uint64_t baseM, baseN, baseK;
    uint64_t kL1, scaleKL1, l1BufferNum, dbL0C, biasElements;
    uint32_t scaleAlg;
    float beta;
    aclrtStream stream;
};

template <class A_TYPE, class B_TYPE, bool TransA, bool TransB, bool IsNzFormat, uint64_t FullLoadMode,
          ActivationKind Activation>
void LaunchKernel(const LaunchParams& p)
{
    using LAYOUT_A = std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LAYOUT_B = std::conditional_t<
        IsNzFormat, std::conditional_t<TransB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>,
        std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>>;

    if constexpr (Activation == ActivationKind::SWISH && IsNzFormat &&
                  (TransA || !std::is_same_v<B_TYPE, fp8_e4m3fn_t>)) {
        // ParseCliArgs rejects these combinations before allocating device memory.
        std::fprintf(stderr, "Unsupported Swish/SiLU WeightNZ layout/dtype combination.\n");
        std::exit(1);
    } else {
        LAUNCH_KERNEL_IMPL(FullLoadMode, Activation);
    }
}

template <class A_TYPE, class B_TYPE, uint64_t FullLoadMode, ActivationKind Activation>
void LaunchElementwiseByTrans(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    DISPATCH_TRANS(A_TYPE, B_TYPE, args.transA, args.transB, isNzFormat, FullLoadMode, Activation);
}

template <class A_TYPE, class B_TYPE, ActivationKind Activation = ActivationKind::GELU>
void LaunchElementwiseByFullLoad(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    if (args.aFullLoad) {
        LaunchElementwiseByTrans<A_TYPE, B_TYPE, Blaze::Gemm::A_FULL_LOAD_MODE, Activation>(args, isNzFormat,
                                                                                            launchParams);
    } else {
        LaunchElementwiseByTrans<A_TYPE, B_TYPE, Blaze::Gemm::NONE_FULL_LOAD_MODE, Activation>(args, isNzFormat,
                                                                                               launchParams);
    }
}

template <class A_TYPE, class B_TYPE, uint64_t FullLoadMode>
void LaunchSwiGluByLayout(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    if (args.transB) {
        if (isNzFormat) {
            LaunchKernel<A_TYPE, B_TYPE, false, true, true, FullLoadMode, ActivationKind::SWIGLU>(launchParams);
        } else {
            LaunchKernel<A_TYPE, B_TYPE, false, true, false, FullLoadMode, ActivationKind::SWIGLU>(launchParams);
        }
    } else if (isNzFormat) {
        LaunchKernel<A_TYPE, B_TYPE, false, false, true, FullLoadMode, ActivationKind::SWIGLU>(launchParams);
    } else {
        LaunchKernel<A_TYPE, B_TYPE, false, false, false, FullLoadMode, ActivationKind::SWIGLU>(launchParams);
    }
}

template <class A_TYPE, class B_TYPE>
void LaunchSwiGluByFullLoad(const CliArgs& args, bool isNzFormat, const LaunchParams& launchParams)
{
    if (args.aFullLoad) {
        LaunchSwiGluByLayout<A_TYPE, B_TYPE, Blaze::Gemm::A_FULL_LOAD_MODE>(args, isNzFormat, launchParams);
    } else {
        LaunchSwiGluByLayout<A_TYPE, B_TYPE, Blaze::Gemm::NONE_FULL_LOAD_MODE>(args, isNzFormat, launchParams);
    }
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
    if (args.aFullLoad) {
        const int64_t mBlockNum = CeilDiv(args.m, args.baseM);
        if (mBlockNum > blockNum) {
            std::cerr << "AFullLoad requires ceil(M / base_m) <= AIC core count, but got " << mBlockNum
                      << " M blocks and " << blockNum << " AIC cores." << std::endl;
            std::exit(1);
        }
        // A stays resident in L1: each core must keep the same M tile across N rounds.
        // The A-full-load scheduler requires an equal number of cores for every M tile.
        blockNum = blockNum / mBlockNum * mBlockNum;
    }

    const uint64_t scaleK = static_cast<uint64_t>(CeilDiv(args.k, static_cast<int64_t>(MXFP_DIVISOR_SIZE))) *
                            (MXFP_DIVISOR_SIZE / GROUP_SIZE);
    const int64_t outputN = args.activation == "swiglu" ? args.n / 2 : args.n;
    const uint64_t scaleN = static_cast<uint64_t>(CeilDiv(outputN, static_cast<int64_t>(MXFP_DIVISOR_SIZE))) *
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
    const size_t ySize = ElementCountToBytes(args.m * outputN, args.aDtype);
    const size_t yScaleSize = static_cast<size_t>(args.m) * scaleN;

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    // Allocate host buffers
    std::vector<uint8_t> hostA(aSize);
    std::vector<uint8_t> hostB(bSize);
    std::vector<uint8_t> hostBias(biasSize, 0);
    std::vector<uint8_t> hostScaleA(scaleASize);
    std::vector<uint8_t> hostScaleB(scaleBSize);
    std::vector<uint8_t> hostY(ySize, 0);
    std::vector<uint8_t> hostYScale(yScaleSize, 0);

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

    // Allocate device buffers
    uint8_t* deviceA{nullptr};
    uint8_t* deviceB{nullptr};
    uint8_t* deviceBias{nullptr};
    uint8_t* deviceScaleA{nullptr};
    uint8_t* deviceScaleB{nullptr};
    uint8_t* deviceY{nullptr};
    uint8_t* deviceYScale{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceBias), biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleA), scaleASize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceScaleB), scaleBSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), ySize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceYScale), yScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, aSize, hostA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, bSize, hostB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBias, biasSize, hostBias.data(), biasSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, scaleASize, hostScaleA.data(), scaleASize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, scaleBSize, hostScaleB.data(), scaleBSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // Poison outputs so missing stores (including the paired scale padding byte) fail verification.
    ACL_CHECK(aclrtMemset(deviceY, ySize, 0xFF, ySize));
    ACL_CHECK(aclrtMemset(deviceYScale, yScaleSize, 0xFF, yScaleSize));

    // Print execution summary
    std::cout << "============================================================" << std::endl;
    std::cout << "  QuantMatmulActivationQuant MX — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape    : M=" << args.m << ", K=" << args.k << ", N=" << args.n << std::endl;
    std::cout << "  A Dtype  : " << args.aDtype << " (also Y dtype; C is FP32)" << std::endl;
    std::cout << "  B Dtype  : " << args.bDtype << std::endl;
    std::cout << "  transA   : " << (args.transA ? "true" : "false") << std::endl;
    std::cout << "  transB   : " << (args.transB ? "true" : "false") << std::endl;
    std::cout << "  Format   : " << args.format << std::endl;
    std::cout << "  Bias     : " << args.bias << " (float32)" << std::endl;
    std::cout << "  L0 Tile  : [" << args.baseM << ", " << args.baseN << ", " << args.baseK << "]" << std::endl;
    std::cout << "  L1 kL1   : " << args.kL1 << std::endl;
    std::cout << "  scaleKL1 : " << args.scaleKL1 << std::endl;
    std::cout << "  l1Buffers: " << args.l1Buffers << std::endl;
    std::cout << "  dbL0C    : " << args.dbL0C << std::endl;
    std::cout << "  AFullLoad: " << (args.aFullLoad ? "true" : "false") << std::endl;
    std::cout << "  Activation: " << args.activation << std::endl;
    if (args.activation == "silu" || args.activation == "swish") {
        std::cout << "  Beta     : " << args.beta << std::endl;
    }
    std::cout << "  ScaleAlg : " << (args.scaleAlg == 0U ? "OCP" : "BLAS") << std::endl;
    std::cout << "  Kernel   : " << args.kernelVariant << std::endl;
    std::cout << "  BlockNum : " << blockNum << std::endl;
    std::cout << "============================================================" << std::endl;

    LaunchParams launchParams = {deviceA,
                                 deviceB,
                                 deviceBias,
                                 deviceScaleA,
                                 deviceScaleB,
                                 deviceY,
                                 deviceYScale,
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
                                 args.scaleAlg,
                                 args.beta,
                                 stream};

    if (args.activation == "swiglu") {
        if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3") {
            LaunchSwiGluByFullLoad<fp8_e4m3fn_t, fp8_e4m3fn_t>(args, isNzFormat, launchParams);
        } else if (args.aDtype == "fp8_e5m2" && args.bDtype == "fp8_e4m3") {
            LaunchSwiGluByFullLoad<fp8_e5m2_t, fp8_e4m3fn_t>(args, isNzFormat, launchParams);
        } else {
            std::fprintf(stderr, "Unsupported SwiGLU A/B dtype combination: a=%s b=%s.\n", args.aDtype.c_str(),
                         args.bDtype.c_str());
            std::exit(1);
        }
    } else if (args.activation == "silu" || args.activation == "swish") {
        if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3") {
            LaunchElementwiseByFullLoad<fp8_e4m3fn_t, fp8_e4m3fn_t, ActivationKind::SWISH>(args, isNzFormat,
                                                                                           launchParams);
        } else if (args.aDtype == "fp8_e5m2" && args.bDtype == "fp8_e4m3") {
            LaunchElementwiseByFullLoad<fp8_e5m2_t, fp8_e4m3fn_t, ActivationKind::SWISH>(args, isNzFormat,
                                                                                         launchParams);
        } else if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e5m2") {
            LaunchElementwiseByFullLoad<fp8_e4m3fn_t, fp8_e5m2_t, ActivationKind::SWISH>(args, isNzFormat,
                                                                                         launchParams);
        } else {
            LaunchElementwiseByFullLoad<fp8_e5m2_t, fp8_e5m2_t, ActivationKind::SWISH>(args, isNzFormat, launchParams);
        }
    } else if (args.aDtype == "fp8_e4m3" && args.bDtype == "fp8_e4m3") {
        LaunchElementwiseByFullLoad<fp8_e4m3fn_t, fp8_e4m3fn_t>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp8_e5m2" && args.bDtype == "fp8_e4m3") {
        LaunchElementwiseByFullLoad<fp8_e5m2_t, fp8_e4m3fn_t>(args, isNzFormat, launchParams);
    } else if (args.aDtype == "fp4_e2m1" && args.bDtype == "fp4_e2m1") {
        LaunchElementwiseByFullLoad<fp4x2_e2m1_t, fp4x2_e2m1_t>(args, isNzFormat, launchParams);
    } else {
        std::fprintf(stderr,
                     "Unsupported A/B dtype combination: a=%s b=%s.\n"
                     "Supported: fp8_e4m3*fp8_e4m3, fp8_e5m2*fp8_e4m3, fp4_e2m1*fp4_e2m1.\n",
                     args.aDtype.c_str(), args.bDtype.c_str());
        std::exit(1);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostY.data(), ySize, deviceY, ySize, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hostYScale.data(), yScaleSize, deviceYScale, yScaleSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // Write outputs
    std::string yPath = outputDir + "/npu_y.bin";
    std::string yScalePath = outputDir + "/npu_y_scale.bin";
    if (!WriteFile(yPath, hostY.data(), ySize)) {
        std::cerr << "Failed to write output Y" << std::endl;
    }
    if (!WriteFile(yScalePath, hostYScale.data(), yScaleSize)) {
        std::cerr << "Failed to write output Y_scale" << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceBias));
    ACL_CHECK(aclrtFree(deviceScaleA));
    ACL_CHECK(aclrtFree(deviceScaleB));
    ACL_CHECK(aclrtFree(deviceY));
    ACL_CHECK(aclrtFree(deviceYScale));
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
