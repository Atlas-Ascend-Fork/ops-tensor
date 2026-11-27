/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "c_api/spr/spr.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// Schedule original rows and complete 512-column level0 blocks. Only individual
// K-wide Fixpipe results alternate slots; the quantization workspace is private.
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadRotateDualLevelQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using SchedulerShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool IS_GROUPED = AscendC::Std::is_same_v<BlockScheduler, Block::BlockSchedulerGmmNoQuant>;
    static constexpr bool ENABLE_QUANT = DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_QUANT;

    static_assert(ENABLE_QUANT, "RotateDualLevelQuant requires the quantization epilogue execution path.");
    static_assert(AscendC::Std::is_same_v<ProblemShape, TupleShape>, "RotateDualLevelQuant requires int64 shapes.");
    static_assert(AscendC::Std::is_same_v<AType, BType> &&
                      (AscendC::Std::is_same_v<AType, half> || AscendC::Std::is_same_v<AType, bfloat16_t>),
                  "Rotation requires matching FP16/BF16 inputs.");
    static_assert(AscendC::Std::is_same_v<CType, bfloat16_t> &&
                      AscendC::Std::is_same_v<typename BlockEpilogue::DataTypeIn, bfloat16_t>,
                  "Fixpipe and normalized MX input must remain BF16.");
    static_assert(BlockEpilogue::ENABLE_LEVEL0, "RotateDualLevelQuant requires a dual-level epilogue.");

    struct Params {
        ProblemShape problemShape{}; // M, N, rotation K, physical rotation matrices per group.
        typename BlockMmad::Params mmadParams{};
        typename BlockScheduler::Params schedulerParams{};
        GM_ADDR groupListGmAddr{nullptr};
        typename BlockEpilogue::Params epilogueParams{};
    };

    __aicore__ inline void operator()(const Params& params)
    {
        // Host tiling guarantees the shape, tile geometry and UB layout contract.
        const int64_t core = Gemm::GetCurrentBlockIdx();
        const int64_t coreCount = AscendC::GetBlockNum();
        BlockMmad blockMmad;
        asc_override_strategy saturationStrategy = asc_override_strategy::USE_API;
        if ASCEND_IS_AIC {
            blockMmad.Init(params.mmadParams);
        }
        {
            BlockEpilogue epilogue;
            if ASCEND_IS_AIV {
                saturationStrategy = asc_get_saturation_strategy();
                asc_set_saturation_strategy(asc_override_strategy::USE_API);
                epilogue.Init(params.epilogueParams);
                // Seed both AIVs, including idle cores and a zero-row second AIV.
                for (uint16_t slot = 0; slot < params.mmadParams.ubDB; ++slot) {
                    Sync::NotifyCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>(Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot);
                }
            }
            if constexpr (IS_GROUPED) {
                ProcessGroups<BlockScheduler>(params, blockMmad, epilogue, core, coreCount);
            } else {
                const SchedulerShape shape{asc::te::get<MNK_M>(params.problemShape),
                                           asc::te::get<MNK_N>(params.problemShape),
                                           asc::te::get<MNK_K>(params.problemShape), 1, 1};
                BlockScheduler scheduler(shape, params.schedulerParams);
                ProcessTasks(params, scheduler, core, scheduler.GetBlockNums(), 0, 0, blockMmad, epilogue, coreCount);
            }
        } // Destroy epilogue state before restoring the Vector mode.
        if ASCEND_IS_AIC {
            // Drain the last credit of every slot, including unused initial ones.
            for (uint16_t slot = 0; slot < params.mmadParams.ubDB; ++slot) {
                Sync::WaitForVector<Sync::SYNC_MODE_INTRA, PIPE_FIX>(Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot, true);
            }
        }
        if ASCEND_IS_AIV {
            asc_set_saturation_strategy(saturationStrategy);
        }
    }

private:
    struct TileContext {
        int64_t mStart;
        int64_t nStart;
        uint32_t rows;
        uint32_t validCols;
    };

    template <class Scheduler>
    __aicore__ inline void ProcessGroups(const Params& params, BlockMmad& blockMmad, BlockEpilogue& epilogue,
                                         int64_t core, int64_t coreCount)
    {
        const auto& schedulerParams = params.schedulerParams;
        Scheduler scheduler(schedulerParams);
        auto groupList = reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr);
        int64_t rowStart = 0;
        for (uint32_t group = 0; group < schedulerParams.groupNum; ++group) {
            const int64_t entry = groupList[group];
            const int64_t m = asc::te::get<MNK_M>(params.problemShape);
            const bool cumulative = schedulerParams.groupListType == 0;
            // Validate before subtracting/adding signed device values. Drain the submitted prefix on failure.
            if ((cumulative && (entry < rowStart || entry > m)) ||
                (!cumulative && (entry < 0 || entry > m - rowStart))) {
                break;
            }
            const int64_t groupRows = cumulative ? entry - rowStart : entry;
            scheduler.UpdateNextGroup(TupleShape{groupRows, asc::te::get<MNK_N>(params.problemShape),
                                                 asc::te::get<MNK_K>(params.problemShape), 1});
            ProcessTasks(params, scheduler, scheduler.GetFirstBlockIdx(core), scheduler.GetBlockNums(), rowStart, group,
                         blockMmad, epilogue, coreCount);
            rowStart += groupRows;
        }
    }

    template <class Scheduler>
    __aicore__ inline void ProcessTasks(const Params& params, Scheduler& scheduler, int64_t first, int64_t end,
                                        int64_t rowStart, int64_t group, BlockMmad& blockMmad, BlockEpilogue& epilogue,
                                        int64_t coreCount)
    {
        for (int64_t task = first; task < end; task += coreCount) {
            const auto shape = scheduler.template GetBlockShape<false, BType>(task);
            const auto coord = scheduler.GetBlockCoord(task);
            TileContext tile{rowStart + asc::te::get<MNK_M>(coord), asc::te::get<MNK_N>(coord),
                             static_cast<uint32_t>(asc::te::get<MNK_M>(shape)),
                             static_cast<uint32_t>(asc::te::get<MNK_N>(shape))};
            ProcessTile(params, tile, group, blockMmad, epilogue);
        }
    }

    __aicore__ inline void ProcessTile(const Params& params, const TileContext& tile, int64_t group,
                                       BlockMmad& blockMmad, BlockEpilogue& epilogue)
    {
        const uint32_t k = static_cast<uint32_t>(asc::te::get<MNK_K>(params.problemShape));
        if ASCEND_IS_AIC {
            for (uint32_t slice = 0; slice < tile.validCols / k; ++slice) {
                ProduceRotation(params, tile, group, slice, blockMmad, epilogue);
            }
        }
        if ASCEND_IS_AIV {
            using BlockShape = typename BlockEpilogue::BlockShape;
            using BlockCoord = typename BlockEpilogue::BlockCoord;
            epilogue(BlockShape{tile.rows, tile.validCols, k, 1}, BlockCoord{tile.mStart, tile.nStart, 0, 0});
        }
    }

    __aicore__ inline void ProduceRotation(const Params& params, const TileContext& tile, int64_t group, uint32_t slice,
                                           BlockMmad& blockMmad, const BlockEpilogue& epilogue)
    {
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        const int64_t rotations = asc::te::get<MNK_B>(params.problemShape);
        const int64_t b = tile.nStart / k + slice;
        const int64_t rotationOffset = (group * rotations + (rotations == 1 ? 0 : b)) * k * k;
        auto a = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr) +
                                                         tile.mStart * n + b * k),
            Gemm::MakeNDExtLayout<AType>(tile.rows, k, n));
        auto rotation = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr) +
                                                         rotationOffset),
            Gemm::MakeNDExtLayout<BType>(k, k, k));
        // BlockMmad takes a GM tensor even when the null bias address disables the load.
        auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr)),
                                         Gemm::MakeNDExtLayout<BiasType>(1, k, k));
        auto output = epilogue.GetAccumulatorTensor({tile.rows, k, 0});
        TupleShape shape{tile.rows, k, k, 1};
        blockMmad(a, rotation, bias, output, shape, ENABLE_QUANT);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
