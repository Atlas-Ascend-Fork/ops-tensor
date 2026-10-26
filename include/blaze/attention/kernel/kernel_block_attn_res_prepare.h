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

#include "blaze/attention/block/block_scheduler_block_attn_res_prepare.h"
#include "blaze/attention/kernel/kernel_universal_decl.h"
#include "blaze/epilogue/block/block_epilogue_block_attn_res_prepare.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Attention {
namespace Kernel {

namespace BlockAttnResPrepareDetail {
constexpr uint32_t MMAD_BLOCK_NUM = 2U;
constexpr uint32_t MM1_INDEX = 0U;
constexpr uint32_t MM2_INDEX = 1U;
constexpr int32_t S_DIM_INDEX = 0;
constexpr int32_t N_DIM_INDEX = 1;
constexpr int32_t D_DIM_INDEX = 2;
constexpr int32_t T_DIM_INDEX = 3;
constexpr int64_t MMAD_BATCH_OFFSET = 0;
constexpr uint32_t SINGLE_BUFFER_NUM = 1U;
constexpr uint32_t E_BUFFER_NUM = 2U;
constexpr uint16_t MODE4_LOCAL_FLAG_COUNT = 10U;

// Mode-4 maps the sibling AIV into the peer flag space by adding 16 to the same local logical flag ID.
constexpr uint16_t DOT_READY_FLAG = 0U;
constexpr uint16_t E_READY_FLAG = 1U;
constexpr uint16_t E_BUFFER_FREE_FLAG = 3U;
constexpr uint16_t AIV1_FLAG_OFFSET = 16U;
constexpr uint8_t WORKSPACE_STORE_EVENT_ID = 0U;
constexpr uint8_t SYNC_MODE = 4U;
static_assert(E_BUFFER_FREE_FLAG + E_BUFFER_NUM <= MODE4_LOCAL_FLAG_COUNT,
              "BlockAttnResPrepare local mode-4 flag ID exceeds the hardware range.");

struct TypedGmParams {
    __gm__ float* blockResidual{nullptr};
    __gm__ float* effectiveQuery{nullptr};
    __gm__ uint64_t* validBlocks{nullptr};
    __gm__ float* softmaxMax{nullptr};
    __gm__ float* weightedOutput{nullptr};
    __gm__ float* softmaxSum{nullptr};
    __gm__ float* workspace{nullptr};
};

template <typename T>
__aicore__ inline auto MakeBatchedDNExtLayout(int64_t batchCount, int64_t rows, int64_t columns, int64_t batchStride,
                                              int64_t columnStride)
{
    auto shape = asc::te::make_shape(batchCount,
                                     asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<1>{}, rows),
                                                         asc::te::make_shape(AscendC::Std::Int<1>{}, columns)));
    auto stride = asc::te::make_stride(
        batchStride, asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}),
                                          asc::te::make_stride(AscendC::Std::Int<0>{}, columnStride)));
    return asc::te::make_pattern_layout<asc::te::dn_ext_layout_ptn, asc::te::layout_trait_default<T>>(shape, stride);
}

class Sync {
public:
    __aicore__ inline static void NotifyDotReady()
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(DOT_READY_FLAG);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(DOT_READY_FLAG + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void WaitDotReady() { AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(DOT_READY_FLAG); }

    __aicore__ inline static void NotifyEReady(uint16_t eSlotIdx)
    {
        // Wait until the E workspace GM write has completed before publishing readiness to the consuming AIC.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(WORKSPACE_STORE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(WORKSPACE_STORE_EVENT_ID);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_S>(E_READY_FLAG + eSlotIdx);
    }

    __aicore__ inline static void WaitEReady(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(E_READY_FLAG + eSlotIdx);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(E_READY_FLAG + eSlotIdx + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void NotifyEBufferFree(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(E_BUFFER_FREE_FLAG + eSlotIdx);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(E_BUFFER_FREE_FLAG + eSlotIdx + AIV1_FLAG_OFFSET);
    }

    __aicore__ inline static void WaitEBufferFree(uint16_t eSlotIdx)
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(E_BUFFER_FREE_FLAG + eSlotIdx);
    }
};

} // namespace BlockAttnResPrepareDetail

template <class ProblemShape_, class BlockMmadTuple_, class BlockEpilogue_, class BlockScheduler_>
class AttentionUniversal<
    ProblemShape_, BlockMmadTuple_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelBlockAttnResPrepareSchedule,
                                                      typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    using DispatchPolicy = typename BlockEpilogue_::DispatchPolicy;
    using ProblemShape = ProblemShape_;
    using BlockMmadTuple = BlockMmadTuple_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using BlockInfo = typename BlockScheduler::BlockInfo;
    static_assert(AscendC::Std::tuple_size_v<BlockMmadTuple> == BlockAttnResPrepareDetail::MMAD_BLOCK_NUM,
                  "BlockAttnResPrepare MMAD tuple must contain MM1 and MM2.");
    using Mm1Block = typename AscendC::Std::tuple_element<BlockAttnResPrepareDetail::MM1_INDEX, BlockMmadTuple>::type;
    using Mm2Block = typename AscendC::Std::tuple_element<BlockAttnResPrepareDetail::MM2_INDEX, BlockMmadTuple>::type;
    using Mm1Params = typename Mm1Block::Params;
    using Mm2Params = typename Mm2Block::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MmadBlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        ProblemShape problemShape{};
        Mm1Params mm1Params{};
        Mm2Params mm2Params{};
        BlockEpilogueParams epilogueParams{};
        BlockSchedulerParams schedulerParams{};
    };
    static_assert(AscendC::Std::is_same_v<typename Mm1Block::AType, float> &&
                      AscendC::Std::is_same_v<typename Mm1Block::BType, float> &&
                      AscendC::Std::is_same_v<typename Mm1Block::CType, float>,
                  "BlockAttnResPrepare MM1 only supports FP32.");
    static_assert(AscendC::Std::is_same_v<typename Mm2Block::AType, float> &&
                      AscendC::Std::is_same_v<typename Mm2Block::BType, float> &&
                      AscendC::Std::is_same_v<typename Mm2Block::CType, float>,
                  "BlockAttnResPrepare MM2 only supports FP32.");
    static_assert(AscendC::Std::is_same_v<typename Mm1Block::LayoutA, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<typename Mm1Block::LayoutB, asc::te::dn_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<typename Mm1Block::LayoutC, asc::te::nd_ext_layout_ptn>,
                  "BlockAttnResPrepare MM1 requires ND x batched-DN -> ND layouts.");
    static_assert(AscendC::Std::is_same_v<typename Mm2Block::LayoutA, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<typename Mm2Block::LayoutB, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<typename Mm2Block::LayoutC, asc::te::nd_ext_layout_ptn>,
                  "BlockAttnResPrepare MM2 requires ND x ND -> ND layouts.");
    static_assert(AscendC::Std::is_same_v<typename BlockScheduler::ProblemShape, ProblemShape>,
                  "BlockAttnResPrepare kernel and scheduler must use the same problem shape.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::ElementType, float>,
                  "BlockAttnResPrepare epilogue only supports FP32.");
    static_assert(AscendC::Std::is_same_v<DispatchPolicy, typename BlockEpilogue::DispatchPolicy>,
                  "BlockAttnResPrepare kernel and epilogue must use the same dispatch policy.");

    __aicore__ inline void operator()(const Params& params)
    {
        Init(params);
        Run();
    }

private:
    // ---------------------- Initialization and runtime shape ----------------------
    __aicore__ inline void BindGmAddresses(const Params& params)
    {
        gm_.blockResidual = reinterpret_cast<__gm__ float*>(params.mm1Params.bGmAddr);
        gm_.effectiveQuery = reinterpret_cast<__gm__ float*>(params.mm1Params.aGmAddr);
        gm_.validBlocks = reinterpret_cast<__gm__ uint64_t*>(params.epilogueParams.validBlocksGmAddr);
        gm_.softmaxMax = reinterpret_cast<__gm__ float*>(params.epilogueParams.softmaxMaxGmAddr);
        gm_.weightedOutput = reinterpret_cast<__gm__ float*>(params.epilogueParams.weightedOutputGmAddr);
        gm_.softmaxSum = reinterpret_cast<__gm__ float*>(params.epilogueParams.softmaxSumGmAddr);
        gm_.workspace = reinterpret_cast<__gm__ float*>(params.epilogueParams.workspaceGmAddr);
    }

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = &params.problemShape;
        mm1Params_ = &params.mm1Params;
        mm2Params_ = &params.mm2Params;
        epilogueParams_ = &params.epilogueParams;
        schedulerParams_ = &params.schedulerParams;
        BindGmAddresses(params);
        blockEpilogue_.Init(params.epilogueParams);
    }

    __aicore__ inline uint64_t ReadValidBlocks() const
    {
        auto tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gm_.validBlocks),
                                           Gemm::MakeNDExtLayout<uint64_t>(1, 1, 1));
        return tensor[asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
    }

    __aicore__ inline uint32_t GetValidN() const
    {
        const uint64_t validBlocks = ReadValidBlocks();
        const uint64_t totalN = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(*problemShape_));
        return static_cast<uint32_t>(validBlocks < totalN ? validBlocks : totalN);
    }

    // AIC and its sibling AIVs share one workspace slice and therefore use the same logical core index.
    __aicore__ inline static uint32_t GetLogicalCoreIndex()
    {
        if ASCEND_IS_AIV {
            return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        }
        return AscendC::GetBlockIdx();
    }

    __aicore__ inline void Run()
    {
        const uint32_t validN = GetValidN();
        const uint32_t coreIndex = GetLogicalCoreIndex();
        BlockScheduler scheduler(*problemShape_, *schedulerParams_, validN, coreIndex);
        if (coreIndex >= scheduler.GetCoreNums()) {
            return;
        }
        if (validN == 0U) {
            ProcessEmptyBlocks(scheduler);
            return;
        }
        auto coreWorkspaceTensor = MakeCoreWorkspaceTensor(coreIndex);
        ProcessValidBlocks(scheduler, coreWorkspaceTensor);
    }

    // --------------------------- Block-level orchestration ---------------------------
    __aicore__ inline void ProcessEmptyBlocks(BlockScheduler& scheduler)
    {
        // Empty input has no Cube computation. AIV0 owns all output rows; the sibling AIV remains idle.
        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() != 0U) {
                return;
            }
            BlockInfo blockInfo{};
            while (scheduler.GetNextBlock(blockInfo)) {
                const uint32_t blockT = static_cast<uint32_t>(
                    asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockShape));
                for (uint32_t tokenIdx = 0U; tokenIdx < blockT; ++tokenIdx) {
                    ProcessEmptyToken(blockInfo, tokenIdx);
                }
            }
        }
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void ProcessValidBlocks(BlockScheduler& scheduler, const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        BlockInfo blockInfo{};
        while (scheduler.GetNextBlock(blockInfo)) {
            ProcessBlock(blockInfo, coreWorkspaceTensor, scheduler);
        }
    }

    __aicore__ inline static uint16_t GetESlotIdx(uint32_t tokenIdx)
    {
        return static_cast<uint16_t>(tokenIdx % BlockAttnResPrepareDetail::E_BUFFER_NUM);
    }

    __aicore__ inline uint64_t GetDotWorkspaceElems() const
    {
        return static_cast<uint64_t>(schedulerParams_->baseS) * mm1Params_->nL1;
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void ProcessBlock(const BlockInfo& blockInfo, const CoreWorkspaceTensor& coreWorkspaceTensor,
                                        const BlockScheduler& scheduler)
    {
        if ASCEND_IS_AIC {
            ProcessAicBlock(blockInfo, coreWorkspaceTensor);
        }
        if ASCEND_IS_AIV {
            const typename BlockScheduler::AivRowRange rowRange = scheduler.GetAivRowRange(blockInfo.blockShape);
            ProcessAivBlock(blockInfo, rowRange, coreWorkspaceTensor);
        }
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void ProcessAicBlock(const BlockInfo& blockInfo, const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        const uint32_t blockT = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockShape));
        // MM1 produces all grouped-token dot values before either AIV consumes its row slice.
        RunMm1(blockInfo, coreWorkspaceTensor);
        BlockAttnResPrepareDetail::Sync::NotifyDotReady();

        // Seed the one or two E slots. Later notifications are issued only after MM2 has consumed a reusable slot.
        const uint32_t initialBufferCount = blockT > BlockAttnResPrepareDetail::SINGLE_BUFFER_NUM ?
                                                BlockAttnResPrepareDetail::E_BUFFER_NUM :
                                                BlockAttnResPrepareDetail::SINGLE_BUFFER_NUM;
        for (uint16_t eSlotIdx = 0U; eSlotIdx < initialBufferCount; ++eSlotIdx) {
            BlockAttnResPrepareDetail::Sync::NotifyEBufferFree(eSlotIdx);
        }

        for (uint32_t tokenIdx = 0U; tokenIdx < blockT; ++tokenIdx) {
            const uint16_t eSlotIdx = GetESlotIdx(tokenIdx);
            BlockAttnResPrepareDetail::Sync::WaitEReady(eSlotIdx);
            for (uint32_t dTileIdx = 0U; dTileIdx < epilogueParams_->dTileNum; ++dTileIdx) {
                RunMm2(blockInfo, tokenIdx, dTileIdx, coreWorkspaceTensor);
            }
            if (tokenIdx + BlockAttnResPrepareDetail::E_BUFFER_NUM < blockT) {
                BlockAttnResPrepareDetail::Sync::NotifyEBufferFree(eSlotIdx);
            }
        }
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void ProcessAivBlock(const BlockInfo& blockInfo,
                                           const typename BlockScheduler::AivRowRange& rowRange,
                                           const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        const uint32_t blockT = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockShape));
        // Reduce V while MM1 runs, then alternate E slots so AIV softmax can overlap the preceding token's MM2.
        for (uint32_t tokenIdx = 0U; tokenIdx < blockT; ++tokenIdx) {
            const uint16_t eSlotIdx = GetESlotIdx(tokenIdx);
            RunVReduction(blockInfo, tokenIdx);
            if (tokenIdx == 0U) {
                BlockAttnResPrepareDetail::Sync::WaitDotReady();
            }
            BlockAttnResPrepareDetail::Sync::WaitEBufferFree(eSlotIdx);
            RunSoftmaxFinalize(blockInfo, tokenIdx, rowRange.rowStart, rowRange.rowCount, eSlotIdx,
                               coreWorkspaceTensor);
            BlockAttnResPrepareDetail::Sync::NotifyEReady(eSlotIdx);
        }
    }

    // ------------------------------ GM Tensor views ------------------------------
    template <typename T>
    __aicore__ inline static auto MakeGmTensor(__gm__ T* address, int64_t rows, int64_t columns, int64_t rowPitch)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(address),
                                    Gemm::MakeNDExtLayout<T>(rows, columns, rowPitch));
    }

    __aicore__ inline auto MakeCoreWorkspaceTensor(uint32_t coreIndex) const
    {
        const int64_t coreNum = static_cast<int64_t>(schedulerParams_->usedCoreNum);
        const int64_t perCoreElems = static_cast<int64_t>(epilogueParams_->workspacePerCoreElems);
        auto workspaceTensor = MakeGmTensor(gm_.workspace, coreNum, perCoreElems, perCoreElems);
        return workspaceTensor.slice(asc::te::make_coord(static_cast<int64_t>(coreIndex), static_cast<int64_t>(0)),
                                     asc::te::make_shape(static_cast<int64_t>(1), perCoreElems));
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline auto MakeCoreWorkspaceView(const CoreWorkspaceTensor& coreWorkspaceTensor, uint64_t elementOffset,
                                                 int64_t rows, int64_t columns, int64_t rowPitch) const
    {
        const int64_t storageElems = (rows - 1) * rowPitch + columns;
        auto storageTensor = coreWorkspaceTensor.slice(
            asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(elementOffset)),
            asc::te::make_shape(static_cast<int64_t>(1), storageElems));
        auto address = reinterpret_cast<__gm__ float*>(storageTensor.data().get());
        return MakeGmTensor(address, rows, columns, rowPitch);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline auto MakeMm1DotTensor(const CoreWorkspaceTensor& coreWorkspaceTensor,
                                            const BlockInfo& blockInfo) const
    {
        const int64_t blockS = asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockShape);
        const int64_t blockN = asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(blockInfo.blockShape);
        const int64_t blockT = asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockShape);
        const int64_t groupedN = blockT * blockN;
        return MakeCoreWorkspaceView(coreWorkspaceTensor, 0U, blockS, groupedN, mm1Params_->nL1);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline auto MakeSoftmaxDotTensor(const CoreWorkspaceTensor& coreWorkspaceTensor,
                                                const BlockInfo& blockInfo, uint32_t tokenIdx, uint32_t sRowStart,
                                                uint32_t validSRows) const
    {
        const uint32_t blockN = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(blockInfo.blockShape));
        const uint64_t offset = static_cast<uint64_t>(sRowStart) * mm1Params_->nL1 +
                                static_cast<uint64_t>(tokenIdx) * blockN;
        return MakeCoreWorkspaceView(coreWorkspaceTensor, offset, validSRows, blockN, mm1Params_->nL1);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline auto MakeEWorkspaceTensor(const CoreWorkspaceTensor& coreWorkspaceTensor, uint16_t eSlotIdx,
                                                uint32_t sRowStart, uint32_t validSRows) const
    {
        const uint64_t nAlign = mm2Params_->kL1;
        const uint64_t offset = GetDotWorkspaceElems() +
                                static_cast<uint64_t>(eSlotIdx) * epilogueParams_->eWorkspaceElems +
                                static_cast<uint64_t>(sRowStart) * nAlign;
        return MakeCoreWorkspaceView(coreWorkspaceTensor, offset, validSRows, nAlign, nAlign);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline auto MakeDummyBiasTensor(const CoreWorkspaceTensor& coreWorkspaceTensor, uint32_t columns) const
    {
        return MakeCoreWorkspaceView(coreWorkspaceTensor, 0U, 1, columns, columns);
    }

    __aicore__ inline auto MakeResidualTensor(const BlockInfo& blockInfo, uint32_t tokenIdx, uint64_t dOffset,
                                              uint64_t validD) const
    {
        const int64_t blockN = asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(blockInfo.blockShape);
        const uint64_t tokenGmIdx = static_cast<uint64_t>(
                                        asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockCoord)) +
                                    tokenIdx;
        const uint64_t totalN = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(*problemShape_));
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        __gm__ float* address = gm_.blockResidual + tokenGmIdx * totalN * totalD + dOffset;
        return MakeGmTensor(address, blockN, validD, totalD);
    }

    __aicore__ inline auto MakeOutputTensor(const BlockInfo& blockInfo, uint32_t tokenIdx, uint32_t sRowStart,
                                            uint32_t validSRows, uint64_t dOffset, uint64_t validD) const
    {
        const uint64_t tokenGmIdx = static_cast<uint64_t>(
                                        asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockCoord)) +
                                    tokenIdx;
        const uint64_t outputSStart = static_cast<uint64_t>(
                                          asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockCoord)) +
                                      sRowStart;
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        const uint64_t totalT = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(*problemShape_));
        const uint64_t outputRowPitch = totalT * totalD;
        __gm__ float* address = gm_.weightedOutput + (outputSStart * totalT + tokenGmIdx) * totalD + dOffset;
        return MakeGmTensor(address, validSRows, validD, outputRowPitch);
    }

    __aicore__ inline auto MakeStatisticTensor(__gm__ float* statistic, const BlockInfo& blockInfo, uint32_t tokenIdx,
                                               uint32_t sRowStart, uint32_t validSRows) const
    {
        const uint64_t tokenGmIdx = static_cast<uint64_t>(
                                        asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockCoord)) +
                                    tokenIdx;
        const uint64_t outputSStart = static_cast<uint64_t>(
                                          asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockCoord)) +
                                      sRowStart;
        const uint64_t totalT = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(*problemShape_));
        __gm__ float* address = statistic + outputSStart * totalT + tokenGmIdx;
        return MakeGmTensor(address, validSRows, 1, totalT);
    }

    __aicore__ inline void ProcessEmptyToken(const BlockInfo& blockInfo, uint32_t tokenIdx)
    {
        const uint32_t blockS = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockShape));
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        auto outputTensor = MakeOutputTensor(blockInfo, tokenIdx, 0U, blockS, 0U, totalD);
        auto maxTensor = MakeStatisticTensor(gm_.softmaxMax, blockInfo, tokenIdx, 0U, blockS);
        auto sumTensor = MakeStatisticTensor(gm_.softmaxSum, blockInfo, tokenIdx, 0U, blockS);
        blockEpilogue_.ProcessEmptyInput(outputTensor, maxTensor, sumTensor);
    }

    __aicore__ inline void RunVReduction(const BlockInfo& blockInfo, uint32_t tokenIdx)
    {
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        auto vTensor = MakeResidualTensor(blockInfo, tokenIdx, 0U, totalD);
        blockEpilogue_.ReduceV(vTensor);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void RunSoftmaxFinalize(const BlockInfo& blockInfo, uint32_t tokenIdx, uint32_t sRowStart,
                                              uint32_t validSRows, uint16_t eSlotIdx,
                                              const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        if (validSRows == 0U) {
            return;
        }
        auto dotTensor = MakeSoftmaxDotTensor(coreWorkspaceTensor, blockInfo, tokenIdx, sRowStart, validSRows);
        auto eWorkspaceTensor = MakeEWorkspaceTensor(coreWorkspaceTensor, eSlotIdx, sRowStart, validSRows);
        auto maxTensor = MakeStatisticTensor(gm_.softmaxMax, blockInfo, tokenIdx, sRowStart, validSRows);
        auto sumTensor = MakeStatisticTensor(gm_.softmaxSum, blockInfo, tokenIdx, sRowStart, validSRows);
        blockEpilogue_.FinalizeSoftmax(dotTensor, eWorkspaceTensor, maxTensor, sumTensor);
    }

    // ------------------------------- Cube stages -------------------------------
    template <typename CoreWorkspaceTensor>
    __aicore__ inline void RunMm1(const BlockInfo& blockInfo, const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        const int64_t blockS = asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockShape);
        const int64_t blockN = asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(blockInfo.blockShape);
        const int64_t blockT = asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockShape);
        const int64_t sOffset = asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockCoord);
        const int64_t tOffset = asc::te::get<BlockAttnResPrepareDetail::T_DIM_INDEX>(blockInfo.blockCoord);
        const uint64_t totalN = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(*problemShape_));
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        __gm__ float* q = gm_.effectiveQuery + static_cast<uint64_t>(sOffset) * totalD;
        const uint64_t groupedN = static_cast<uint64_t>(blockT) * blockN;
        __gm__ float* v = gm_.blockResidual + static_cast<uint64_t>(tOffset) * totalN * totalD;
        auto qTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(q),
            asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<float>>{}(
                blockS, static_cast<int64_t>(totalD)));
        auto vTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(v),
                                            BlockAttnResPrepareDetail::MakeBatchedDNExtLayout<float>(
                                                blockT, totalD, blockN, static_cast<int64_t>(totalN * totalD), totalD));
        auto dotTensor = MakeMm1DotTensor(coreWorkspaceTensor, blockInfo);
        auto dummyBias = MakeDummyBiasTensor(coreWorkspaceTensor, mm1Params_->nL1);
        auto& mm1 = AscendC::Std::get<BlockAttnResPrepareDetail::MM1_INDEX>(blockMmadTuple_);
        mm1.Init(*mm1Params_);
        MmadBlockShape mm1BlockShape{blockS, static_cast<int64_t>(groupedN), static_cast<int64_t>(totalD),
                                     BlockAttnResPrepareDetail::MMAD_BATCH_OFFSET};
        mm1(qTensor, vTensor, dummyBias, dotTensor, mm1BlockShape);
    }

    template <typename CoreWorkspaceTensor>
    __aicore__ inline void RunMm2(const BlockInfo& blockInfo, uint32_t tokenIdx, uint32_t dTileIdx,
                                  const CoreWorkspaceTensor& coreWorkspaceTensor)
    {
        const uint32_t blockS = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::S_DIM_INDEX>(blockInfo.blockShape));
        const uint32_t blockN = static_cast<uint32_t>(
            asc::te::get<BlockAttnResPrepareDetail::N_DIM_INDEX>(blockInfo.blockShape));
        const uint16_t eSlotIdx = GetESlotIdx(tokenIdx);
        auto eTensor = MakeEWorkspaceTensor(coreWorkspaceTensor, eSlotIdx, 0U, blockS);
        auto dummyBias = MakeDummyBiasTensor(coreWorkspaceTensor, epilogueParams_->baseDAlign);
        const uint64_t totalD = static_cast<uint64_t>(
            asc::te::get<BlockAttnResPrepareDetail::D_DIM_INDEX>(*problemShape_));
        const uint64_t dOffset = static_cast<uint64_t>(dTileIdx) * epilogueParams_->baseD;
        const uint64_t remainingD = totalD - dOffset;
        const uint32_t validD = static_cast<uint32_t>(remainingD < epilogueParams_->baseD ? remainingD :
                                                                                            epilogueParams_->baseD);
        auto vTensor = MakeResidualTensor(blockInfo, tokenIdx, dOffset, validD);
        auto outputTensor = MakeOutputTensor(blockInfo, tokenIdx, 0U, blockS, dOffset, validD);
        auto& mm2 = AscendC::Std::get<BlockAttnResPrepareDetail::MM2_INDEX>(blockMmadTuple_);
        mm2.Init(*mm2Params_);
        MmadBlockShape mm2BlockShape{static_cast<int64_t>(blockS), static_cast<int64_t>(validD),
                                     static_cast<int64_t>(blockN), BlockAttnResPrepareDetail::MMAD_BATCH_OFFSET};
        mm2(eTensor, vTensor, dummyBias, outputTensor, mm2BlockShape);
    }

    const ProblemShape* __restrict problemShape_{nullptr};
    const Mm1Params* __restrict mm1Params_{nullptr};
    const Mm2Params* __restrict mm2Params_{nullptr};
    const BlockEpilogueParams* __restrict epilogueParams_{nullptr};
    const BlockSchedulerParams* __restrict schedulerParams_{nullptr};
    BlockAttnResPrepareDetail::TypedGmParams gm_{};
    BlockMmadTuple blockMmadTuple_{};
    BlockEpilogue blockEpilogue_{};
};

// Public component stack used by the ops-transformer BlockAttnResPrepare kernel entry.
using BlockAttnResPrepareProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockAttnResPrepareMm1MmadPolicy = Gemm::MatmulMultiBlockBasic<
    0U, 0U, Gemm::KernelMmadMultiBlockBasic,
    static_cast<uint64_t>(Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_BATCHED_B)>;
using BlockAttnResPrepareMm2MmadPolicy = Gemm::MatmulMultiBlockBasic<>;
using BlockAttnResPrepareMm1 = Gemm::Block::BlockMmad<
    BlockAttnResPrepareMm1MmadPolicy, float, asc::te::nd_ext_layout_ptn, float, asc::te::dn_ext_layout_ptn, float,
    asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn>;
using BlockAttnResPrepareMm2 = Gemm::Block::BlockMmad<
    BlockAttnResPrepareMm2MmadPolicy, float, asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn, float,
    asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn>;
using BlockAttnResPrepareMmadTuple = AscendC::Std::tuple<BlockAttnResPrepareMm1, BlockAttnResPrepareMm2>;
using BlockAttnResPrepareBlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueBlockAttnResPrepare<
    float, BlockAttnResPreparePolicy>;
using BlockAttnResPrepareBlockScheduler = Block::BlockSchedulerBlockAttnResPrepare<BlockAttnResPrepareProblemShape>;
using KernelBlockAttnResPrepare = AttentionUniversal<BlockAttnResPrepareProblemShape, BlockAttnResPrepareMmadTuple,
                                                     BlockAttnResPrepareBlockEpilogue,
                                                     BlockAttnResPrepareBlockScheduler, void>;

} // namespace Kernel
} // namespace Attention
} // namespace Blaze
