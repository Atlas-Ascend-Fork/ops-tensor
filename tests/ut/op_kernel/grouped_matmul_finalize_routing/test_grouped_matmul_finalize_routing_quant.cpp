/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include "gtest/gtest.h"
#include "grouped_matmul_finalize_routing_cpu_debug_compat.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mx_mix_finalize_routing.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef __fp4e1m2x2
#undef __fp4e2m1x2
#undef __fp8e4m3
#undef Cast
#endif

namespace {
constexpr uint32_t GROUPS = 2, M = 32, K = 128, N = 128, BATCH = 16;
constexpr uint32_t BASE_M = 32, BASE_N = 64, BASE_K = 64;
constexpr uint32_t WORKSPACE_BYTES = M * N * sizeof(float);

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes)))
    {
        if (ptr_ != nullptr) {
            std::fill_n(ptr_, bytes, 0U);
        }
    }
    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_{nullptr};
};

// Fixed-shape smoke entry: instantiate the public components and check completion.
template <typename A, bool A8W4 = false, bool Deterministic = false>
__global__ __aicore__ void QuantSmoke(GM_ADDR x, GM_ADDR weight, GM_ADDR scaleB, GM_ADDR scaleA, GM_ADDR groups,
                                      GM_ADDR logit, GM_ADDR rows, GM_ADDR y, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using ND = asc::te::nd_ext_layout_ptn;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using namespace Blaze::Gemm;
    if constexpr (!A8W4) {
        using Acc = AscendC::Std::conditional_t<AscendC::Std::is_same_v<A, int8_t>, int32_t, float>;
        using Policy = MatmulWithScaleMix<0, false, KernelQgmmMixPertokenFinalizeRouting>;
        using Mmad = Block::BlockMmad<Policy, A, ND, AscendC::Std::tuple<A>, asc::te::zn_layout_ptn, Acc, ND,
                                      bfloat16_t, ND>;
        using Epi = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<
            float, Acc, float, int64_t, float, Deterministic, Blaze::Epilogue::Block::FinalizeRoutingMode::PERTOKEN>;
        using KernelType = Kernel::GemmUniversal<Shape, Mmad, Epi, Block::BlockSchedulerGmmSwatWithTailSplit>;
        typename KernelType::Params p{};
        p.problemShape = {M, N, K, 1};
        p.mmadParams.aGmAddr = x;
        p.mmadParams.bGmAddr = weight;
        p.sharedInputParams = {nullptr, y, 0, 0, N, BATCH, 1.0F};
        p.epilogueParams = {y, scaleB, scaleA, nullptr, logit, rows};
        p.gmmParams = {GROUPS, BATCH, 0, 0, 1.0F, BASE_M, BASE_N, BASE_K, K, K, 2, 0, 1, 1, 0, 1};
        if constexpr (Deterministic) {
            p.gmmParams.deterWorkspaceSize = WORKSPACE_BYTES;
            p.epilogueParams.workspaceGmAddr = workspace;
        }
        p.groupListGmAddr = groups;
        KernelType kernel;
        kernel(p);
    } else {
        using Policy = GroupedMatmulFinalizeRoutingWithWeightQuantMx;
        using Mmad = Block::BlockMmad<
            Policy, AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e8m0_t>,
            AscendC::Std::tuple<ND, asc::te::scalea_nd_layout_ptn>, AscendC::Std::tuple<fp4x2_e2m1_t, fp8_e8m0_t>,
            AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, float, ND, bfloat16_t, ND>;
        using Prologue = Kernel::GmmWqgmmFrWeightPrologue<fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t>;
        using Epi = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<
            float, float, float, int64_t, float, false, Blaze::Epilogue::Block::FinalizeRoutingMode::A8W4>;
        using Resplit = Kernel::BlockSchedulerWqgmmNResplit<asc::te::shape<uint64_t, uint64_t, uint64_t>>;
        using KernelType = Kernel::GmmWqgmmMxFinalizeRoutingKernel<
            asc::te::shape<uint64_t, uint64_t, uint64_t>, Mmad, Resplit, Prologue, Epi, Kernel::GmmWqgmmFrOutputInit>;
        typename KernelType::Params p{};
        p.problemShape = {K, N, GROUPS};
        p.mmad.aGmAddr = x;
        p.mmad.scaleAGmAddr = scaleA;
        p.mmad.scaleBGmAddr = scaleB;
        p.mmad.cGmAddr = y;
        p.scheduler = {1, 256, 0, 0, 0, 0, 1, 0, 256, N};
        p.prologue = {reinterpret_cast<__gm__ fp4x2_e2m1_t*>(weight)};
        p.epilogue = {y, nullptr, nullptr, nullptr, logit, rows};
        p.epilogue.nSize = N;
        p.outputInit = {reinterpret_cast<__gm__ float*>(y), nullptr, N, 0, 0, 1.0F};
        p.ptrGroupList = groups;
        p.groupListType = 1;
        KernelType kernel;
        kernel(p);
    }
}

template <typename A, bool A8W4 = false, bool Deterministic = false>
void RunSmoke()
{
    const size_t scaleABytes = M * (A8W4 ? K / 32 : sizeof(float));
    const size_t scaleBBytes = GROUPS * N * (A8W4 ? K / 32 : sizeof(float));
    GmBuffer x(M * K), weight(GROUPS * K * N / (A8W4 ? 2U : 1U));
    GmBuffer scaleA(scaleABytes), scaleB(scaleBBytes), groups(GROUPS * sizeof(int64_t));
    GmBuffer logit(M * sizeof(float)), rows(M * sizeof(int64_t)), y(BATCH * N * sizeof(float));
    GmBuffer workspace(Deterministic ? WORKSPACE_BYTES : 1U);
    for (auto* p : {&x, &weight, &scaleA, &scaleB, &groups, &logit, &rows, &y, &workspace}) {
        ASSERT_NE(p->Get(), nullptr);
    }
    std::fill_n(reinterpret_cast<int64_t*>(groups.Get()), GROUPS, M / GROUPS);
    std::fill_n(reinterpret_cast<float*>(logit.Get()), M, 1.0F);
    if constexpr (A8W4) {
        std::fill_n(scaleA.Get(), scaleABytes, 0x7FU);
        std::fill_n(scaleB.Get(), scaleBBytes, 0x7FU);
    } else {
        std::fill_n(reinterpret_cast<float*>(scaleA.Get()), M, 1.0F);
        std::fill_n(reinterpret_cast<float*>(scaleB.Get()), GROUPS * N, 1.0F);
    }
    for (uint32_t i = 0; i < M; ++i) {
        reinterpret_cast<int64_t*>(rows.Get())[i] = i % BATCH;
    }
    constexpr uint8_t INPUT_VALUE = AscendC::Std::is_same_v<A, int8_t> ? 1 : 0x38;
    std::fill_n(x.Get(), M * K, INPUT_VALUE);
    std::fill_n(weight.Get(), GROUPS * K * N / (A8W4 ? 2U : 1U), A8W4 ? 0x22 : INPUT_VALUE);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto entry = QuantSmoke<A, A8W4, Deterministic>;
    ASSERT_TRUE(KERNEL_RUN_KF(entry, 1U, x.Get(), weight.Get(), scaleB.Get(), scaleA.Get(), groups.Get(), logit.Get(),
                              rows.Get(), y.Get(), workspace.Get()));
}
} // namespace

TEST(GroupedMatmulFinalizeRoutingQuantTest, PertokenInt8) { RunSmoke<int8_t>(); }
TEST(GroupedMatmulFinalizeRoutingQuantTest, PertokenFp8) { RunSmoke<fp8_e4m3fn_t>(); }
TEST(GroupedMatmulFinalizeRoutingQuantTest, PertokenHifloat8) { RunSmoke<hifloat8_t>(); }
TEST(GroupedMatmulFinalizeRoutingQuantTest, PertokenInt8Deterministic) { RunSmoke<int8_t, false, true>(); }
TEST(GroupedMatmulFinalizeRoutingQuantTest, MxA8W4) { RunSmoke<fp8_e4m3fn_t, true>(); }
