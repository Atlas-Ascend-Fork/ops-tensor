/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file fused_mat_mul_elemwise.h
 * \brief FusedMatMul elementwise add/mul Kernel UT Wrapper
 *
 * 本文件是 nn仓 matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_elemwise.h
 * (FusedMatMulAdvanced::FusedMatMulAddMulKernel, basic/全载路径) 的UT镜像, 覆盖
 * fixpipe_opti(OTF/1V2)、A全载(al1_full_load)、B全载(bl1_full_load)三种policy.
 * mmadParams/schParams/epilogueParams 的字段布局与填充语义必须与产品侧同步修改;
 * 两侧tiling结构不同(UT用FusedMatMulUT::MatMulV3BasicTilingData), 无法编译期共享,
 * 修改任一侧时需人工核对另一侧.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/epilogue/block/block_epilogue_matmul_elemwise.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "fused_mat_mul_tiling_data.h"

namespace FusedMatMulUT {

// HighPrecise=true (inner_precise = 0) keeps the accumulator in FP32 and fuses in the FP32 domain;
// HighPrecise=false (inner_precise = 1) fuses in the element domain
// (legacy MatMulMixWithoutQueActKernel MmOutTypeSelector semantics).
// The 1V2 remap (none full-load) keeps a FP32 accumulator (B16 acc + DUAL_DST_SPLIT_M device gap,
// splitM pairs two AIVs) while the fusion domain still follows HighPrecise: the matmul result is
// rounded back to ElementType before fusing with x3 (legacy B16-accumulator semantics).
template <typename ElementType, uint64_t FusedOpType, uint64_t L0C2OutMode, uint64_t FullLoadMode, bool HighPrecise,
          bool TransA, bool TransB>
__aicore__ inline void FusedMatMulAddMulWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM,
                                                GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tiling)
{
    static constexpr bool kRemapTo1V2 = (FullLoadMode == Blaze::Gemm::NONE_FULL_LOAD_MODE) &&
                                        (L0C2OutMode == Blaze::Gemm::ND_ALIG_1V2_FIXPIPE);
    using AccType = AscendC::Std::conditional_t<HighPrecise || kRemapTo1V2, float, ElementType>;
    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // B full-load: bl1_full_load policy; A full-load: al1_full_load policy (always OTF);
    // otherwise fixpipe_opti — same policy mapping as the nn-side assembly.
    using DispatchPolicy = AscendC::Std::conditional_t<
        FullLoadMode == Blaze::Gemm::B_FULL_LOAD_MODE, Blaze::Gemm::MatmulMultiBlockBFullLoad<L0C2OutMode, FusedOpType>,
        AscendC::Std::conditional_t<FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE,
                                    Blaze::Gemm::MatmulMultiBlockAFullLoad<Blaze::Gemm::A_FULL_LOAD_MODE, FusedOpType>,
                                    Blaze::Gemm::MatmulMultiBlockFixpipeOpti<L0C2OutMode, FusedOpType>>>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FullLoadMode, false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, AccType,
                                                    Layout, ElementType, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulElemwise<DispatchPolicy, ElementType, AccType,
                                                                              /*FusionSameDomain=*/!HighPrecise>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    // FP32 accumulators need the two AIVs to split M so the acc image plus x3/output staging fits in UB.
    const uint8_t splitM = AscendC::IsSameType<AccType, float>::value ? 1U : 0U;
    // ubPitchGran恒0: 行距统一走block侧c0_element<AccType>(与epilogue的ACC_ALIGN恒等),
    // 1V2所需的行距差异由AccType自然表达(half=16/float=8), 不再传16特判(见epilogue AccPitch注释).
    constexpr uint32_t ubPitchGran = 0U;

    // BlockMmad::Params layouts differ per dispatch policy (field maps see the nn-side assembly):
    //   al1_full_load(15): {a,b,c,bias, oriK, mL1,kL1, baseM,baseN,baseK, l1Stages,l0cStages, rowStride, splitM,ubDB}
    //   bl1_full_load(18): bl1布局为 {a,b,c,bias,groupList,workspace, oriK, mL1,nL1,kL1, baseM,baseN,baseK,
    //                       l1Stages,l0cStages, splitM,ubDB, rowStride}
    //   fixpipe_opti(19):  bl1布局 + ubPitchGran
    typename BlockMmad::Params mmadParams = {};
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        mmadParams = {x1GM,
                      x2GM,
                      nullptr,
                      biasGM,
                      tiling.k,
                      tiling.mL1,
                      tiling.kL1,
                      tiling.baseM,
                      tiling.baseN,
                      tiling.baseK,
                      tiling.l1BufferNum,
                      tiling.l0cDB,
                      tiling.rowStride,
                      splitM,
                      1U};
    } else if constexpr (FullLoadMode == Blaze::Gemm::B_FULL_LOAD_MODE) {
        mmadParams = {
            x1GM,         x2GM,       nullptr,    biasGM,          nullptr,      workspaceGM,  tiling.k,
            tiling.mL1,   tiling.nL1, tiling.kL1, tiling.baseM,    tiling.baseN, tiling.baseK, tiling.l1BufferNum,
            tiling.l0cDB, splitM,     1U,         tiling.rowStride};
    } else {
        mmadParams = {
            x1GM,         x2GM,       nullptr,    biasGM,           nullptr,      workspaceGM,  tiling.k,
            tiling.mL1,   tiling.nL1, tiling.kL1, tiling.baseM,     tiling.baseN, tiling.baseK, tiling.l1BufferNum,
            tiling.l0cDB, splitM,     1U,         tiling.rowStride, ubPitchGran};
    }

    // schParams为17项(至innerBatch): UT的tiling结构无aswWindowLen字段, 聚合初始化走
    // BlockSchedulerMatmulBasic::Params的windowLen默认值4; 产品侧下发tiling.aswWindowLen
    // (basic场景为0), 经调度器"0视为未下发回退默认4"处理, 两侧语义一致.
    KernelParams params = {
        {tiling.m, tiling.n, tiling.k, 1},
        mmadParams,
        {},
        {tiling.mL1, tiling.nL1, tiling.kL1, tiling.baseM, tiling.baseN, tiling.baseK, tiling.mTailCnt, tiling.nTailCnt,
         tiling.mBaseTailSplitCnt, tiling.nBaseTailSplitCnt, tiling.mTailMain, tiling.nTailMain,
         static_cast<uint8_t>(tiling.mmadParam > 0 ? 1 : 0), static_cast<uint32_t>(tiling.l2CacheDisable),
         tiling.sliceM, tiling.srcNdStride, tiling.innerBatch}};
    // Epilogue params: x3 fields + problemShape (params-only Init contract, aligned with the
    // upstream single-param convention). basic路径的elemwise epilogue无workspace数据流,
    // 不填workspaceGmAddr(与产品侧一致); workspace仅由bl1/fixpipe_opti的mmadParams消费.
    params.epilogueParams.cGmAddr = yGM;
    params.epilogueParams.x3GmAddr = x3GM;
    params.epilogueParams.x3BatchBroadcast = false;
    params.epilogueParams.x3M = tiling.m;
    params.epilogueParams.problemShape = params.problemShape;

    MatmulKernel kernel;
    kernel(params);
}

} // namespace FusedMatMulUT
