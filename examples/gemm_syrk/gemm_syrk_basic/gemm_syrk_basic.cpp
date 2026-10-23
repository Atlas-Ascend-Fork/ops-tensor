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
 * @file gemm_syrk_basic.cpp
 * @brief Symmetric rank-k update (syrk) example: C = alpha * (A @ A^T) + beta * C.
 *
 * Supported dtypes: float16, bfloat16
 * Storage layout of a: (batch, m, k) ND, or the transposed (batch, k, m) when
 * trans is set (the kernel binds a DNExt (m, k) view and computes
 * C = alpha * (A^T @ A) + beta * C).
 *
 * The kernel updates C fully in place: cIn and c are bound to the same device
 * buffer, matching the aclnn-layer same-address binding of the GemmSyrk
 * operator. C must be symmetric on input so the in-place output stays
 * symmetric.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_syrk.h"
#include "blaze/gemm/block/block_scheduler_matmul_syrk.h"
#include "blaze/gemm/kernel/kernel_matmul_syrk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

/* ========================================================================== */
/* Tiling data (host -> device, single struct)                                */
/* ========================================================================== */

/*
 * The syrk tiling contract forces ONE symmetric square block:
 * baseBlock == baseM == baseN == mL1 == nL1 (16-aligned), with
 *   - baseBlock * baseBlock * sizeof(float) <= L0C_SIZE / 2
 *     (single accumulator in one L0C slot),
 *   - 2 * baseBlock * kL1 * dtypeSize <= L1_SIZE / 2
 *     (A + B images of one L1 double-buffer stage fit an L1 half),
 *   - baseBlock * baseK * dtypeSize <= L0A_SIZE / 2 (L0A ping-pong slot),
 *   - a single tail block per axis (no per-core tail splitting).
 */
#pragma pack(push, 8)
struct alignas(8) GemmSyrkTilingData {
    uint32_t m{0};
    uint32_t n{0}; // == m (syrk: N == M)
    uint32_t k{0};
    uint32_t batch{0};
    uint32_t baseBlock{0}; // baseM == baseN == mL1 == nL1, 16-aligned
    uint32_t baseK{0};     // L0 K step, 16-aligned
    uint32_t kL1{0};       // L1 K chunk (>= baseK), 16-aligned
    float alpha{1.0F};
    float beta{1.0F};
};
#pragma pack(pop)

/* ========================================================================== */
/* Device-side kernel (Blaze syrk assembly, MIX 1 AIC : 2 AIV)                */
/* ========================================================================== */

// Fixed L1 double buffering: kL1 is sized so one stage's A+B images fit an
// L1 half (enforced by the host tiling).
constexpr uint8_t SYRK_L1_STAGES = 2U;

template <typename ElementType, bool TRANS>
__global__ __aicore__ void gemm_syrk_kernel(GM_ADDR aGM, GM_ADDR cInGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                            GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    // Copy the __gm__ tiling fields to local scalars before building params.
    GemmSyrkTilingData t;
    const auto& gmTiling = *reinterpret_cast<__gm__ GemmSyrkTilingData*>(tilingGM);
    t.m = gmTiling.m;
    t.n = gmTiling.n;
    t.k = gmTiling.k;
    t.batch = gmTiling.batch;
    t.baseBlock = gmTiling.baseBlock;
    t.baseK = gmTiling.baseK;
    t.kL1 = gmTiling.kL1;
    t.alpha = gmTiling.alpha;
    t.beta = gmTiling.beta;

    using AccType = float;
    // TRANS: the DNExt (m, k) view over the transposed (k, m) storage; the
    // GM->L1 fetch auto-routes to dn2nz and yields the same dual-view image.
    using LayoutA = std::conditional_t<TRANS, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulSyrk;
    // Compact upper-triangle scheduler: stride polling over triangle-only
    // slots keeps every core busy instead of locking onto column densities.
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerSyrkTriangular<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutA, AccType,
                                                    LayoutC, ElementType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    // The syrk tiling contract collapses the block geometry into ONE size
    // (baseM == baseN == mL1 == nL1 == baseBlock), consumed as a single
    // `block` param by the block / scheduler layers.
    const uint32_t block = t.baseBlock;

    KernelParams params = {{t.m, t.n, t.k, t.batch},
                           // The block layer reads B from the single aGM source (the ZN L1 view
                           // of the A fetch). AIV sub 0 runs the (i, j) epilogue on the ND
                           // fixpipe image, sub 1 the (j, i) epilogue on the nz2dn
                           // (hardware-transposed) image.
                           {aGM, workspaceGM, t.k, block, t.kL1, t.baseK, SYRK_L1_STAGES},
                           // x3 == output == C: in-place read of beta * C and overwrite with the result.
                           {cInGM, cGM, t.alpha, t.beta},
                           // Scheduler params: the single symmetric block; no tail splitting.
                           {block}};

    params.epilogueParams.problemShape = params.problemShape;

    MatmulKernel kernel;
    kernel(params);
}

/* ========================================================================== */
/* Host-side tiling (symmetric-block caps from the hardware budgets)          */
/* ========================================================================== */

struct SyrkTiling {
    uint32_t baseBlock{16};
    uint32_t baseK{64};
    uint32_t kL1{128};
};

static constexpr int64_t BLOCK_16 = 16L;

// Largest 16-aligned block with block^2 <= capElems (integer floor sqrt,
// no libm dependency on the device link path).
static uint64_t FloorSqrt16(uint64_t capElems)
{
    uint64_t root = 16;
    while ((root + 16) * (root + 16) <= capElems) {
        root += 16;
    }
    return root;
}

static uint64_t FloorAlign16(uint64_t v) { return std::max<uint64_t>(v / 16UL * 16UL, 16UL); }

static SyrkTiling ComputeTiling(const PlatformInfo& platform, size_t dtypeSize, int64_t m)
{
    SyrkTiling tiling;
    const uint64_t l0cBlockMax = FloorSqrt16(platform.l0cSize / 2UL / sizeof(float));
    const uint64_t l1BlockMax = FloorAlign16(platform.l1Size / 2UL / (2UL * dtypeSize * tiling.kL1));
    const uint64_t l0BlockMax = FloorAlign16(platform.l0aSize / 2UL / (dtypeSize * tiling.baseK));
    const uint64_t mAlign = CeilAlign(static_cast<uint64_t>(m), static_cast<uint64_t>(BLOCK_16));
    const uint64_t hardCap = std::min({mAlign, l0cBlockMax, l1BlockMax, l0BlockMax});
    // 128 keeps the single-accumulator L0C slot small on large shapes; the
    // caps above clamp it on small / hardware-limited shapes.
    tiling.baseBlock = static_cast<uint32_t>(std::max<uint64_t>(std::min<uint64_t>(128UL, hardCap), 16UL));
    return tiling;
}

/* ========================================================================== */
/* CLI argument parsing                                                       */
/* ========================================================================== */

struct CliArgs {
    int64_t m{0};
    int64_t k{0};
    int64_t batch{1};
    double alpha{1.0};
    double beta{1.0};
    std::string dtype{"float16"};
    bool trans{false};
};

static bool ParseBool(const char* s)
{
    std::string str(s);
    return str == "true" || str == "1" || str == "True";
}

static bool ParseCliArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc < 3) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: " << argv[0] << " <m> <k> [batch] [alpha] [beta] [dtype] [trans]\n";
        return false;
    }

    args.m = std::atoll(argv[1]);
    args.k = std::atoll(argv[2]);

    if (argc >= 4) {
        args.batch = std::atoll(argv[3]);
    }
    if (argc >= 5) {
        args.alpha = std::atof(argv[4]);
    }
    if (argc >= 6) {
        args.beta = std::atof(argv[5]);
    }
    if (argc >= 7) {
        args.dtype = argv[6];
    }
    if (argc >= 8) {
        args.trans = ParseBool(argv[7]);
    }

    if (args.m <= 0 || args.k <= 0 || args.batch <= 0) {
        std::cerr << "Error: m, k and batch must be positive integers.\n";
        return false;
    }

    if (args.dtype != "float16" && args.dtype != "bfloat16") {
        std::cerr << "Error: dtype must be float16 or bfloat16 (got '" << args.dtype << "')\n";
        return false;
    }

    return true;
}

/* ========================================================================== */
/* Host-side runner                                                           */
/* ========================================================================== */

static void Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};

    PlatformInfo platform;
    if (!InitPlatformInfo(platform)) {
        std::cerr << "Failed to init platform info" << std::endl;
        return;
    }
    if (platform.aicNum == 0 || platform.aivNum != platform.aicNum * 2) {
        std::cerr << "GemmSyrk requires a MIX 1 AIC : 2 AIV device (aicNum=" << platform.aicNum
                  << ", aivNum=" << platform.aivNum << ")" << std::endl;
        return;
    }

    size_t dtypeSize = (args.dtype == "bfloat16") ? sizeof(bfloat16_t) : sizeof(half);
    SyrkTiling tiling = ComputeTiling(platform, dtypeSize, args.m);
    int64_t blockNum = static_cast<int64_t>(platform.aicNum);

    ACLDeviceGuard guard(stream);

    size_t sizeA = static_cast<size_t>(args.batch) * args.m * args.k * dtypeSize;
    size_t sizeC = static_cast<size_t>(args.batch) * args.m * args.m * dtypeSize;
    size_t sizeWorkspace = 2UL * 1024UL * 1024UL;

    std::string inputDir = "./input";
    std::string outputDir = "./output";

    struct stat st;
    std::string pathA = inputDir + "/input_a.bin";
    std::string pathC = inputDir + "/input_c.bin";
    if (stat(pathA.c_str(), &st) != 0 || stat(pathC.c_str(), &st) != 0) {
        std::cerr << "Input files not found: " << pathA << " / " << pathC << std::endl;
        return;
    }

    std::vector<uint8_t> hostA(sizeA);
    std::vector<uint8_t> hostC(sizeC);
    if (!ReadFile(pathA, hostA.data(), sizeA)) {
        std::cerr << "Failed to read input A" << std::endl;
        return;
    }
    if (!ReadFile(pathC, hostC.data(), sizeC)) {
        std::cerr << "Failed to read input C" << std::endl;
        return;
    }

    uint8_t* deviceA{nullptr};
    uint8_t* deviceC{nullptr};
    uint8_t* deviceWorkspace{nullptr};
    uint8_t* deviceTiling{nullptr};

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(
        aclrtMalloc(reinterpret_cast<void**>(&deviceTiling), sizeof(GemmSyrkTilingData), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));

    GemmSyrkTilingData tilingData;
    tilingData.m = static_cast<uint32_t>(args.m);
    tilingData.n = static_cast<uint32_t>(args.m); // syrk: N == M
    tilingData.k = static_cast<uint32_t>(args.k);
    tilingData.batch = static_cast<uint32_t>(args.batch);
    tilingData.baseBlock = tiling.baseBlock;
    tilingData.baseK = tiling.baseK;
    tilingData.kL1 = tiling.kL1;
    tilingData.alpha = static_cast<float>(args.alpha);
    tilingData.beta = static_cast<float>(args.beta);
    ACL_CHECK(aclrtMemcpy(deviceTiling, sizeof(GemmSyrkTilingData), &tilingData, sizeof(GemmSyrkTilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "============================================================" << std::endl;
    std::cout << "  GemmSyrk Basic — Execution Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Shape     : batch=" << args.batch << ", M=" << args.m << ", K=" << args.k << ", N=" << args.m
              << std::endl;
    std::cout << "  Dtype     : " << args.dtype << std::endl;
    std::cout << "  Layout    : " << (args.trans ? "DN (transposed a storage)" : "ND") << " -> ND" << std::endl;
    std::cout << "  alpha     : " << args.alpha << std::endl;
    std::cout << "  beta      : " << args.beta << std::endl;
    std::cout << "  trans     : " << (args.trans ? "true" : "false") << std::endl;
    std::cout << "  baseBlock : " << tiling.baseBlock << std::endl;
    std::cout << "  baseK     : " << tiling.baseK << std::endl;
    std::cout << "  kL1       : " << tiling.kL1 << std::endl;
    std::cout << "  BlockNum  : " << blockNum << " (MIX 1 AIC : 2 AIV)" << std::endl;
    std::cout << "============================================================" << std::endl;

    // cIn and c bind the same buffer: in-place update.
    if (args.dtype == "bfloat16") {
        if (args.trans) {
            gemm_syrk_kernel<bfloat16_t, true>
                <<<blockNum, 0, stream>>>(deviceA, deviceC, deviceC, deviceWorkspace, deviceTiling);
        } else {
            gemm_syrk_kernel<bfloat16_t, false>
                <<<blockNum, 0, stream>>>(deviceA, deviceC, deviceC, deviceWorkspace, deviceTiling);
        }
    } else {
        if (args.trans) {
            gemm_syrk_kernel<half, true>
                <<<blockNum, 0, stream>>>(deviceA, deviceC, deviceC, deviceWorkspace, deviceTiling);
        } else {
            gemm_syrk_kernel<half, false>
                <<<blockNum, 0, stream>>>(deviceA, deviceC, deviceC, deviceWorkspace, deviceTiling);
        }
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string outPath = outputDir + "/npu_out.bin";
    if (!WriteFile(outPath, hostC.data(), sizeC)) {
        std::cerr << "Failed to write output" << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceWorkspace));
    ACL_CHECK(aclrtFree(deviceTiling));
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
