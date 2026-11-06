/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mx_mix_finalize_routing.h"

namespace GmmfrExample {

inline constexpr uint32_t BASE_M = 32;
inline constexpr uint32_t BASE_N = 64;
inline constexpr uint32_t BASE_K = 64;
inline constexpr float DEFAULT_SHARED_INPUT_WEIGHT = 0.5F;
enum class Flow : uint32_t { PERTOKEN, MX_ND, MX_NZ, A8W4 };

// Dense group configuration; output/logit/index types are selected by Run template parameters.
struct Config {
    int64_t m{32};
    int64_t n{128};
    int64_t k{128};
    uint32_t groups{2};
    uint32_t batch{16};
    uint32_t cores{1};
    uint32_t hasBias{0};
    uint32_t tokenScale{1};
    uint32_t sharedLen{0};
    uint32_t sharedOffset{0};
    uint32_t groupListType{1};
    float sharedWeight{DEFAULT_SHARED_INPUT_WEIGHT};
    uint32_t workspaceRows{0};
};
struct Arguments {
    GM_ADDR x;
    GM_ADDR weight;
    GM_ADDR scaleB;
    GM_ADDR bias;
    GM_ADDR scaleA;
    GM_ADDR groupList;
    GM_ADDR sharedInput;
    GM_ADDR logit;
    GM_ADDR rowIndex;
    GM_ADDR y;
    GM_ADDR workspace{nullptr};
};

template <Flow Mode, typename A, typename B = A, bool Deterministic = false, typename Out = float,
          typename Logit = float, typename RowIndex = int64_t>
__aicore__ inline void Run(const Arguments& args, const Config& c)
{
    using ND = asc::te::nd_ext_layout_ptn;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    const auto bias = c.hasBias != 0 ? args.bias : nullptr;
    const auto shared = c.sharedLen != 0 ? args.sharedInput : nullptr;
    if constexpr (Mode == Flow::PERTOKEN) {
        using Acc = AscendC::Std::conditional_t<AscendC::Std::is_same_v<A, int8_t>, int32_t, float>;
        using Policy = Blaze::Gemm::MatmulWithScaleMix<0, false, Blaze::Gemm::KernelQgmmMixPertokenFinalizeRouting>;
        using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, A, ND, AscendC::Std::tuple<B>, asc::te::zn_layout_ptn, Acc,
                                                   ND, bfloat16_t, ND>;
        using Epi = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<
            float, Acc, float, int64_t, float, Deterministic, Blaze::Epilogue::Block::FinalizeRoutingMode::PERTOKEN>;
        using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epi, Scheduler>;
        typename Kernel::Params p{};
        p.problemShape = {c.m, c.n, c.k, 1};
        p.mmadParams.aGmAddr = args.x;
        p.mmadParams.bGmAddr = args.weight;
        p.sharedInputParams = {shared,  args.y,        c.sharedOffset, c.sharedLen, static_cast<int32_t>(c.n),
                               c.batch, c.sharedWeight};
        p.epilogueParams = {args.y, args.scaleB, c.tokenScale != 0 ? args.scaleA : nullptr,
                            bias,   args.logit,  args.rowIndex};
        p.gmmParams = {c.groups,
                       c.batch,
                       c.sharedOffset,
                       c.sharedLen,
                       c.sharedWeight,
                       BASE_M,
                       BASE_N,
                       BASE_K,
                       static_cast<uint32_t>(c.k),
                       static_cast<uint32_t>(c.k),
                       2,
                       static_cast<uint8_t>(c.hasBias),
                       1,
                       static_cast<uint8_t>(c.groupListType),
                       0,
                       c.cores};
        if constexpr (Deterministic) {
            const uint64_t windowRows = c.workspaceRows != 0 ? c.workspaceRows : ((c.m + BASE_M - 1) / BASE_M) * BASE_M;
            const uint32_t windowBytes = static_cast<uint32_t>(windowRows * c.n * sizeof(float));
            p.gmmParams.deterWorkspaceSize = windowBytes;
            p.epilogueParams.workspaceGmAddr = args.workspace;
        }
        p.groupListGmAddr = args.groupList;
        Kernel kernel;
        kernel(p);
    } else if constexpr (Mode == Flow::MX_ND || Mode == Flow::MX_NZ) {
        using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelQgmmMxMixFinalizeRouting>;
        using WeightLayout = AscendC::Std::conditional_t<Mode == Flow::MX_NZ, asc::te::nz_layout_ptn, ND>;
        using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, A, ND, B, WeightLayout, float, ND, bfloat16_t, ND>;
        using Prologue = Blaze::Gemm::Kernel::BlockPrologueFinalizeRouting<Out, bfloat16_t>;
        using Epi = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<Out, float, Logit, RowIndex>;
        using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, AscendC::Std::tuple<Prologue, Epi>, Scheduler>;
        typename Kernel::Params p{};
        p.problemShape = {c.m, c.n, c.k, 1};
        p.blockMmadParams = {args.x, args.weight, args.y, bias, args.scaleA, args.scaleB};
        p.prologueParams = {shared,  args.y,        c.sharedOffset, c.sharedLen, static_cast<int32_t>(c.n),
                            c.batch, c.sharedWeight};
        p.epilogueParams = {args.y, args.scaleB, args.scaleA, bias, args.logit, args.rowIndex};
        p.groupListGmAddr = args.groupList;
        p.gmmParams = {c.groups,
                       c.batch,
                       c.sharedOffset,
                       c.sharedLen,
                       c.sharedWeight,
                       BASE_M,
                       BASE_N,
                       BASE_K,
                       static_cast<uint32_t>(c.k),
                       static_cast<uint32_t>(c.k),
                       static_cast<uint32_t>(c.k),
                       static_cast<uint32_t>(c.k),
                       static_cast<uint8_t>(c.hasBias),
                       1,
                       static_cast<uint8_t>(c.groupListType)};
        Kernel kernel;
        kernel(p);
    } else {
        using Policy = Blaze::Gemm::GroupedMatmulFinalizeRoutingWithWeightQuantMx;
        using Mmad = Blaze::Gemm::Block::BlockMmad<
            Policy, AscendC::Std::tuple<A, fp8_e8m0_t>, AscendC::Std::tuple<ND, asc::te::scalea_nd_layout_ptn>,
            AscendC::Std::tuple<B, fp8_e8m0_t>,
            AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, float, ND, bfloat16_t, ND>;
        using Prologue = Blaze::Gemm::Kernel::GmmWqgmmFrWeightPrologue<A, B, bfloat16_t>;
        using Epi = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<
            float, float, float, int64_t, float, false, Blaze::Epilogue::Block::FinalizeRoutingMode::A8W4>;
        using Init = Blaze::Gemm::Kernel::GmmWqgmmFrOutputInit;
        using Resplit = Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<asc::te::shape<uint64_t, uint64_t, uint64_t>>;
        using Kernel = Blaze::Gemm::Kernel::GmmWqgmmMxFinalizeRoutingKernel<
            asc::te::shape<uint64_t, uint64_t, uint64_t>, Mmad, Resplit, Prologue, Epi, Init>;
        typename Kernel::Params p{};
        p.problemShape = {static_cast<uint64_t>(c.k), static_cast<uint64_t>(c.n), c.groups};
        p.mmad.aGmAddr = args.x;
        p.mmad.scaleAGmAddr = args.scaleA;
        p.mmad.scaleBGmAddr = args.scaleB;
        p.mmad.biasGmAddr = bias;
        p.mmad.cGmAddr = args.y;
        constexpr uint64_t MAIN_BLOCK_N = 256;
        constexpr int32_t BASE_M_A8W4 = 256;
        p.scheduler.mainBlockCount = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(c.n), MAIN_BLOCK_N);
        p.scheduler.mainBlockSize = MAIN_BLOCK_N;
        p.scheduler.coreNum = c.cores;
        p.scheduler.baseM = BASE_M_A8W4;
        p.scheduler.nSize = static_cast<uint64_t>(c.n);
        p.prologue = {reinterpret_cast<__gm__ B*>(args.weight)};
        p.epilogue = {args.y, nullptr, nullptr, nullptr, args.logit, args.rowIndex};
        p.epilogue.nSize = c.n;
        p.outputInit = {reinterpret_cast<__gm__ float*>(args.y),
                        reinterpret_cast<__gm__ bfloat16_t*>(shared),
                        static_cast<uint64_t>(c.n),
                        c.sharedOffset,
                        c.sharedLen,
                        c.sharedWeight};
        p.ptrGroupList = args.groupList;
        p.groupListType = c.groupListType;
        p.hasBias = c.hasBias;
        Kernel kernel;
        kernel(p);
    }
}
} // namespace GmmfrExample
