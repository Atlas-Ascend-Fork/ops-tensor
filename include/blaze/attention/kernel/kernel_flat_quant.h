/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_flat_quant.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/attention/kernel/kernel_universal.h"
#include "blaze/attention/block/block_mmad_flat_quant.h"
#include "blaze/attention/block/block_scheduler_flat_quant.h"
#include "blaze/epilogue/block/block_epilogue_flat_quant.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Attention {
namespace Kernel {

constexpr uint16_t FLAT_QUANT_SYNC_MODE = 4;
constexpr uint16_t FLAT_QUANT_SYNC_AIV_AIC_FLAG = 8;
constexpr uint16_t FLAT_QUANT_SYNC_AIC_AIV_FLAG = 9;
constexpr uint16_t FLAT_QUANT_FLAG_ID_MAX = 16;

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class AttentionUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                         AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                             KernelFlatQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline AttentionUniversal() {}
    __aicore__ inline ~AttentionUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadOp = BlockMmad;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using A_T = typename BlockMmad::A_T;
    using B_T = typename BlockMmad::B_T;
    using L0CType = typename BlockMmad::L0cType;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn,
                                                     AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn,
                                                     AscendC::Std::Int<asc::te::c0_element<BType>>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params) { Run(params); }

private:
    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        m_ = asc::te::get<Gemm::MNK_M>(problemShape_);
        n_ = asc::te::get<Gemm::MNK_N>(problemShape_);
        k_ = asc::te::get<Gemm::MNK_K>(problemShape_);
        b_ = asc::te::get<Gemm::MNK_B>(problemShape_);

        BlockMmadParams blockMmadParams = params.mmadParams;
        aGmPtr_ = reinterpret_cast<__gm__ A_T*>(blockMmadParams.aGmAddr);
        p1GmPtr_ = reinterpret_cast<__gm__ B_T*>(blockMmadParams.bGmAddr);
        p2GmPtr_ = reinterpret_cast<__gm__ B_T*>(blockMmadParams.cGmAddr);
    }

    __aicore__ inline void Run(Params& params)
    {
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        int64_t curBlockIdx = Gemm::GetCurrentBlockIdx();
        int64_t coreNums = AscendC::GetBlockNum();
        if (coreNums <= 0) {
            return;
        }
        Init(params);

        BlockScheduler bs(params.problemShape, coreNums, params.schParams);
        int64_t blockNums = bs.GetBlockNums();

        int64_t realCoreNums = bs.GetCoreNums(coreNums);
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        blockMmadOp.Init(params.mmadParams);
        params.epilogueParams.problemShape = problemShape_;
        epilogueOp.Init(params.epilogueParams);

        auto layoutA = MakeLayoutA{}(m_ * b_, k_);
        auto layoutP1 = MakeLayoutB{}(m_, m_);
        auto layoutP2 = MakeLayoutB{}(n_, n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmPtr_), layoutA);
        auto gmP1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(p1GmPtr_), layoutP1);
        auto gmP2 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(p2GmPtr_), layoutP2);

        for (int64_t tileIdx = curBlockIdx; tileIdx < blockNums; tileIdx += coreNums) {
            TupleL1L0Shape blockShape = bs.GetBlockShape(tileIdx);
            uint64_t iterBatch = asc::te::get<Gemm::MNK_B>(blockShape);
            uint64_t roundIdx = tileIdx / coreNums;
            int64_t batchOffset = bs.GetBlockCoord(tileIdx, curBlockIdx);
            if ASCEND_IS_AIC {
                if (roundIdx > 0) {
                    if ((roundIdx & 1) == 1) {
                        Gemm::Sync::WaitForVector<FLAT_QUANT_SYNC_MODE, PIPE_FIX>(FLAT_QUANT_SYNC_AIV_AIC_FLAG);
                    } else {
                        Gemm::Sync::WaitForVector<FLAT_QUANT_SYNC_MODE, PIPE_FIX>(FLAT_QUANT_SYNC_AIV_AIC_FLAG +
                                                                                  FLAT_QUANT_FLAG_ID_MAX);
                    }
                }
                int64_t rowOffset = batchOffset * m_;
                auto gmBlockA = gmA.slice(asc::te::make_coord(rowOffset, 0L), asc::te::make_shape(m_ * iterBatch, k_));

                blockMmadOp(gmBlockA, gmP1, gmP2, blockShape, tileIdx < coreNums);
                if (roundIdx % 2 == 0) {
                    Gemm::Sync::NotifyVector<FLAT_QUANT_SYNC_MODE, PIPE_FIX>(FLAT_QUANT_SYNC_AIC_AIV_FLAG);
                } else {
                    Gemm::Sync::NotifyVector<FLAT_QUANT_SYNC_MODE, PIPE_FIX>(FLAT_QUANT_SYNC_AIC_AIV_FLAG +
                                                                             FLAT_QUANT_FLAG_ID_MAX);
                }
            }
            if ASCEND_IS_AIV {
                if ((roundIdx & 1) == AscendC::GetSubBlockIdx()) {
                    Gemm::Sync::WaitForCube<FLAT_QUANT_SYNC_MODE, PIPE_V>(FLAT_QUANT_SYNC_AIC_AIV_FLAG);
                    epilogueOp(batchOffset, iterBatch);
                    if (tileIdx + coreNums < CeilAlign(blockNums, coreNums)) {
                        Gemm::Sync::NotifyCube<FLAT_QUANT_SYNC_MODE, PIPE_MTE3>(FLAT_QUANT_SYNC_AIV_AIC_FLAG);
                    }
                }
            }
        }
    }

private:
    __gm__ A_T* aGmPtr_{nullptr};
    __gm__ B_T* p1GmPtr_{nullptr};
    __gm__ B_T* p2GmPtr_{nullptr};
    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};
};

} // namespace Kernel
} // namespace Attention
} // namespace Blaze
