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
 * \file block_attn_res_prepare.h
 * \brief BlockAttnResPrepare mixed-kernel UT wrapper.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#if defined(ASCENDC_CPU_DEBUG)
#include "lib/matmul_intf.h"
#endif

#if defined(ASCENDC_CPU_DEBUG)
// asc_get_sub_block_num()/asc_get_sub_block_id() 直调 get_subblockdim()/get_subblockid()，
// 缺少 C++ 封装 GetTaskRation()/GetSubBlockIdx() 的 CPU 仿真分支（g_taskRation/sub_block_idx），
// UT 仿真下返回错误值会导致 AIV 行分配异常、跨核死锁。引入 blaze 头之前重定向到模拟器全局量，
// 语义与 C++ 封装对齐。必须先包含 tensor_api 内置的 c_api/sys_var/sys_var.h（与 blaze 链同源），
// 让真实定义完成解析，再做宏替换。
#include "c_api/sys_var/sys_var.h"
namespace BlazeUtStub {
inline int64_t AscGetSubBlockNum()
{
    if ASCEND_IS_AIC {
        return 1;
    }
    return g_taskRation;
}

inline int64_t AscGetSubBlockId()
{
    if ASCEND_IS_AIV {
        return sub_block_idx;
    }
    return 0;
}
} // namespace BlazeUtStub
#define asc_get_sub_block_num() (::BlazeUtStub::AscGetSubBlockNum())
#define asc_get_sub_block_id() (::BlazeUtStub::AscGetSubBlockId())
#endif

#include "blaze/attention/kernel/kernel_universal.h"

// Keep the UT launch payload plain and mirror the host tiling data contract. Generic Tensor API
// component parameters are assembled at the kernel entry instead of being copied through GM.
struct BlockAttnResPrepareTestTiling {
    uint32_t totalT{0U};
    uint32_t totalN{0U};
    uint32_t totalS{0U};
    uint32_t totalWorkUnits{0U};
    uint64_t totalD{0U};
    uint32_t usedCoreNum{0U};
    uint32_t baseS{0U};
    uint32_t baseT{0U};
    uint32_t baseD{0U};
    uint32_t baseDAlign{0U};
    uint32_t sTileNum{0U};
    uint32_t dTileNum{0U};
    uint32_t sAlign{0U};
    uint32_t nAlign{0U};
    uint32_t mm1NAlign{0U};
    uint8_t mm1L1Stages{1U};
    uint8_t vUbBufferNum{0U};
    uint64_t eWorkspaceElems{0U};
    uint64_t vUbElems{0U};
    uint64_t dotUbElems{0U};
    uint64_t reduceUbElems{0U};
    uint64_t softmaxUbElems{0U};
    uint64_t workspacePerCoreElems{0U};
    float epsilon{1.0e-6F};
};

__global__ __aicore__ void block_attn_res_prepare_kernel_entry(GM_ADDR blockResidual, GM_ADDR effectiveQuery,
                                                               GM_ADDR validBlocks, GM_ADDR softmaxMax,
                                                               GM_ADDR weightedOutput, GM_ADDR softmaxSum,
                                                               GM_ADDR workspace, GM_ADDR tilingAddress)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using Kernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
    constexpr uint32_t MM1_L0_K_MAX = 64U;
    constexpr uint32_t SINGLE_STAGE = 1U;
    const auto& tiling = *reinterpret_cast<BlockAttnResPrepareTestTiling*>(tilingAddress);
    Kernel::Params params{};
    params.problemShape = {static_cast<int64_t>(tiling.totalS), static_cast<int64_t>(tiling.totalN),
                           static_cast<int64_t>(tiling.totalD), static_cast<int64_t>(tiling.totalT)};

    params.mm1Params.aGmAddr = effectiveQuery;
    params.mm1Params.bGmAddr = blockResidual;
    params.mm1Params.cGmAddr = workspace;
    params.mm1Params.workspaceGmAddr = workspace;
    params.mm1Params.mL1 = tiling.sAlign;
    params.mm1Params.nL1 = tiling.mm1NAlign;
    params.mm1Params.kL1 = tiling.baseDAlign;
    params.mm1Params.mL0 = tiling.sAlign;
    params.mm1Params.nL0 = tiling.mm1NAlign;
    params.mm1Params.kL0 = tiling.baseD < MM1_L0_K_MAX ? tiling.baseD : MM1_L0_K_MAX;
    params.mm1Params.l1Stages = tiling.mm1L1Stages;
    params.mm1Params.l0cStages = SINGLE_STAGE;

    params.mm2Params.aGmAddr = workspace;
    params.mm2Params.bGmAddr = blockResidual;
    params.mm2Params.cGmAddr = weightedOutput;
    params.mm2Params.workspaceGmAddr = workspace;
    params.mm2Params.mL1 = tiling.sAlign;
    params.mm2Params.nL1 = tiling.baseDAlign;
    params.mm2Params.kL1 = tiling.nAlign;
    params.mm2Params.mL0 = tiling.sAlign;
    params.mm2Params.nL0 = tiling.baseDAlign;
    params.mm2Params.kL0 = tiling.nAlign;
    params.mm2Params.l1Stages = SINGLE_STAGE;
    params.mm2Params.l0cStages = SINGLE_STAGE;

    params.epilogueParams.validBlocksGmAddr = validBlocks;
    params.epilogueParams.softmaxMaxGmAddr = softmaxMax;
    params.epilogueParams.weightedOutputGmAddr = weightedOutput;
    params.epilogueParams.softmaxSumGmAddr = softmaxSum;
    params.epilogueParams.workspaceGmAddr = workspace;
    params.epilogueParams.totalD = tiling.totalD;
    params.epilogueParams.baseD = tiling.baseD;
    params.epilogueParams.baseDAlign = tiling.baseDAlign;
    params.epilogueParams.dTileNum = tiling.dTileNum;
    params.epilogueParams.sAlign = tiling.sAlign;
    params.epilogueParams.vUbBufferNum = tiling.vUbBufferNum;
    params.epilogueParams.eWorkspaceElems = tiling.eWorkspaceElems;
    params.epilogueParams.vUbElems = tiling.vUbElems;
    params.epilogueParams.dotUbElems = tiling.dotUbElems;
    params.epilogueParams.reduceUbElems = tiling.reduceUbElems;
    params.epilogueParams.softmaxUbElems = tiling.softmaxUbElems;
    params.epilogueParams.workspacePerCoreElems = tiling.workspacePerCoreElems;
    params.epilogueParams.epsilon = tiling.epsilon;

    params.schedulerParams.totalWorkUnits = tiling.totalWorkUnits;
    params.schedulerParams.usedCoreNum = tiling.usedCoreNum;
    params.schedulerParams.baseT = tiling.baseT;
    params.schedulerParams.baseS = tiling.baseS;
    params.schedulerParams.sTileNum = tiling.sTileNum;
    params.schedulerParams.mm1NAlign = tiling.mm1NAlign;

    Kernel kernel;
    kernel(params);
}
