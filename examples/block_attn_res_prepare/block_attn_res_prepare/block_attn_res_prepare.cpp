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
 * @file block_attn_res_prepare.cpp
 * @brief Executable example for Blaze::Attention::Kernel::KernelBlockAttnResPrepare.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "blaze/attention/kernel/kernel_universal.h"
#include "data_utils.h"

namespace {

constexpr uint32_t TOTAL_T = 1U;
constexpr uint32_t TOTAL_N = 8U;
constexpr uint32_t TOTAL_S = 2U;
constexpr uint64_t TOTAL_D = 32U;
constexpr uint32_t S_ALIGN = 16U;
constexpr uint32_t N_ALIGN = 16U;
constexpr uint32_t BASE_D_ALIGN = 32U;
constexpr uint32_t V_UB_BUFFER_NUM = 2U;
constexpr uint32_t SOFTMAX_STAT_NUM = 2U;
constexpr uint64_t WORKSPACE_PER_CORE_ELEMS = 512U;
constexpr uint32_t MM1_L0_K_MAX = 64U;
constexpr uint32_t SINGLE_STAGE = 1U;
constexpr float EPSILON = 1.0e-6F;

struct TilingData {
    uint32_t totalT{TOTAL_T};
    uint32_t totalN{TOTAL_N};
    uint32_t totalS{TOTAL_S};
    uint32_t totalWorkUnits{1U};
    uint64_t totalD{TOTAL_D};
    uint32_t usedCoreNum{1U};
    uint32_t baseS{S_ALIGN};
    uint32_t baseT{1U};
    uint32_t baseD{static_cast<uint32_t>(TOTAL_D)};
    uint32_t baseDAlign{BASE_D_ALIGN};
    uint32_t sTileNum{1U};
    uint32_t dTileNum{1U};
    uint32_t sAlign{S_ALIGN};
    uint32_t nAlign{N_ALIGN};
    uint32_t mm1NAlign{N_ALIGN};
    uint8_t mm1L1Stages{SINGLE_STAGE};
    uint8_t vUbBufferNum{V_UB_BUFFER_NUM};
    uint64_t eWorkspaceElems{static_cast<uint64_t>(S_ALIGN) * N_ALIGN};
    uint64_t vUbElems{static_cast<uint64_t>(TOTAL_N) * BASE_D_ALIGN};
    uint64_t dotUbElems{static_cast<uint64_t>(S_ALIGN) * N_ALIGN};
    uint64_t reduceUbElems{AscendC::VECTOR_REG_WIDTH / sizeof(float)};
    uint64_t softmaxUbElems{static_cast<uint64_t>(SOFTMAX_STAT_NUM) * S_ALIGN};
    uint64_t workspacePerCoreElems{WORKSPACE_PER_CORE_ELEMS};
    float epsilon{EPSILON};
};

__global__ __aicore__ void BlockAttnResPrepareKernel(GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks,
                                                     GM_ADDR softmaxMax, GM_ADDR weightedOutput, GM_ADDR softmaxSum,
                                                     GM_ADDR workspace, GM_ADDR tilingAddress)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using Kernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
    const auto& tiling = *reinterpret_cast<__gm__ TilingData*>(tilingAddress);

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

struct DeviceBuffers {
    uint8_t* residual{nullptr};
    uint8_t* query{nullptr};
    uint8_t* validBlocks{nullptr};
    uint8_t* max{nullptr};
    uint8_t* output{nullptr};
    uint8_t* sum{nullptr};
    uint8_t* workspace{nullptr};
    uint8_t* tiling{nullptr};
};

void AllocateAndCopy(uint8_t** device, const void* host, size_t bytes)
{
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

void FreeBuffers(DeviceBuffers& buffers)
{
    for (uint8_t* ptr : {buffers.residual, buffers.query, buffers.validBlocks, buffers.max, buffers.output, buffers.sum,
                         buffers.workspace, buffers.tiling}) {
        if (ptr != nullptr) {
            ACL_CHECK(aclrtFree(ptr));
        }
    }
}

int Run(uint64_t validN)
{
    const size_t residualBytes = static_cast<size_t>(TOTAL_T) * TOTAL_N * TOTAL_D * sizeof(float);
    const size_t queryBytes = static_cast<size_t>(TOTAL_S) * TOTAL_D * sizeof(float);
    const size_t statBytes = static_cast<size_t>(TOTAL_T) * TOTAL_S * sizeof(float);
    const size_t outputBytes = static_cast<size_t>(TOTAL_T) * TOTAL_S * TOTAL_D * sizeof(float);
    const size_t workspaceBytes = WORKSPACE_PER_CORE_ELEMS * sizeof(float);
    std::vector<uint8_t> residual(residualBytes);
    std::vector<uint8_t> query(queryBytes);
    std::vector<uint8_t> max(statBytes, 0U);
    std::vector<uint8_t> output(outputBytes, 0U);
    std::vector<uint8_t> sum(statBytes, 0U);
    std::vector<uint8_t> workspace(workspaceBytes, 0U);
    TilingData tiling;
    if (!ReadFile("./input/block_residual.bin", residual.data(), residual.size()) ||
        !ReadFile("./input/effective_query.bin", query.data(), query.size())) {
        return 1;
    }

    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);
    DeviceBuffers device;
    AllocateAndCopy(&device.residual, residual.data(), residual.size());
    AllocateAndCopy(&device.query, query.data(), query.size());
    AllocateAndCopy(&device.validBlocks, &validN, sizeof(validN));
    AllocateAndCopy(&device.max, max.data(), max.size());
    AllocateAndCopy(&device.output, output.data(), output.size());
    AllocateAndCopy(&device.sum, sum.data(), sum.size());
    AllocateAndCopy(&device.workspace, workspace.data(), workspace.size());
    AllocateAndCopy(&device.tiling, &tiling, sizeof(tiling));

    BlockAttnResPrepareKernel<<<1, 0, stream>>>(device.residual, device.query, device.validBlocks, device.max,
                                                device.output, device.sum, device.workspace, device.tiling);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(max.data(), max.size(), device.max, max.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(output.data(), output.size(), device.output, output.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(sum.data(), sum.size(), device.sum, sum.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    const bool written = WriteFile("./output/npu_max.bin", max.data(), max.size()) &&
                         WriteFile("./output/npu_output.bin", output.data(), output.size()) &&
                         WriteFile("./output/npu_sum.bin", sum.data(), sum.size());
    FreeBuffers(device);
    return written ? 0 : 1;
}

} // namespace

int main(int argc, const char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <valid_blocks>" << std::endl;
        return 1;
    }
    const int64_t parsedValidN = std::atoll(argv[1]);
    if (parsedValidN < 0 || parsedValidN > static_cast<int64_t>(TOTAL_N)) {
        std::cerr << "valid_blocks must be in [0, " << TOTAL_N << "]." << std::endl;
        return 1;
    }
    return Run(static_cast<uint64_t>(parsedValidN));
}
