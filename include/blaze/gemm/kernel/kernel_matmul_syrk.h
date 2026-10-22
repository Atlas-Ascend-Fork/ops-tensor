/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_matmul_syrk.h
 * \brief Symmetric rank-k update kernel: C = alpha * (A @ A^T) + beta * C.
 *
 * Iterates the compact upper-triangle index space (BlockSchedulerSyrkTriangular,
 * every core receives slots/cores +- 1 tiles). Each processed slot drives one BlockMmadSyrk call that
 * computes C[i, j] from two shared nd2nz L1 fetches (a single Mmad chain per
 * pair, halved cube work). Both AIV sub-blocks then run the scale-add
 * epilogue in parallel over their own fp32 accumulator image: sub 0 gets
 * the (i, j) ND fixpipe, sub 1 the (j, i) DN-transposed fixpipe (hardware
 * nz2dn) and reads the genuine beta * C[j, i] rows -- no GM read-back and
 * no symmetric-C assumption anywhere. Lower-triangle slots are skipped
 * instantly, so the per-core work stays balanced while both GM->L1 traffic
 * and cube work are halved.
 *
 * Host tiling contract (see block_mmad_matmul_syrk.h): mL1 and nL1 both within
 * min(baseM, baseN), no per-core tail splitting (a single tail block per
 * axis), and baseM * baseN * sizeof(float) <= L0C_SIZE / 4 (UB hosts both
 * fp32 accumulator images plus both AIVs' staging).
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_syrk.h"
#include "blaze/gemm/block/block_scheduler_matmul_syrk.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<
                        AscendC::Std::is_same_v<KernelMmadSyrk, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using CType = typename BlockMmad::CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = typename BlockScheduler::BlockShape;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline GemmUniversal()
    {
        if ASCEND_IS_AIV {
            Sync::NotifyCube<MIX_SYNC_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }

    __aicore__ inline ~GemmUniversal()
    {
        if ASCEND_IS_AIC {
            Sync::WaitForVector<MIX_SYNC_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG, true);
        }
    }

    __aicore__ inline void operator()(const Params& params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        Init(params);
        if ASCEND_IS_AIV {
            // Both AIV sub-blocks participate: sub 0 runs the (i, j)
            // epilogue, sub 1 the transposed (j, i) epilogue.
            curBlockIdx /= AscendC::GetTaskRation();
        }

        BlockScheduler bs(params.problemShape, params.schParams);
        if (curBlockIdx >= bs.GetCoreNums()) {
            return;
        }

        BlockEpilogue epilogueOp;
        BlockMmad blockMmad;
        epilogueOp.Init(params.epilogueParams, problemShape_);
        if ASCEND_IS_AIC {
            blockMmad.Init(params.mmadParams);
        }
        MatmulProcess(params, epilogueOp, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());
    }

private:
    __aicore__ inline void MatmulProcess(const Params& params, BlockEpilogue& epilogueOp, BlockMmad& blockMmad,
                                         BlockScheduler& bs, int64_t curBlockIdx, int64_t coreNums,
                                         int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA{}(m_, k_);
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<false, AType>(blockIdx);
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            const int64_t coordM = asc::te::get<MNK_M>(blockCoord);
            const int64_t coordN = asc::te::get<MNK_N>(blockCoord);
            const int64_t batchIdx = asc::te::get<MNK_B>(blockCoord);
            const int64_t shapeM = asc::te::get<MNK_M>(blockShape);
            int64_t shapeN = asc::te::get<MNK_N>(blockShape);
            const int64_t shapeK = asc::te::get<MNK_K>(blockShape);
            shapeN = AscendC::Std::min(shapeN, static_cast<int64_t>(n_) - coordN);
            const bool isDiagonal = (coordN == coordM);
            const int64_t shapeMA = AscendC::Std::min(shapeM, static_cast<int64_t>(m_) - coordM);
            if (shapeMA <= 0 || shapeN <= 0) {
                continue;
            }
            TupleShape tileShape{shapeMA, shapeN, shapeK, 1};

            const uint64_t batchOffsetA = static_cast<uint64_t>(batchIdx) * m_ * k_;
            auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_ + batchOffsetA),
                                            layoutA);
            TupleShape tileCoord{coordM, coordN, 0, batchIdx};

            const int64_t offsetC = (batchIdx * static_cast<int64_t>(m_) + coordM) * static_cast<int64_t>(n_) + coordN;
            const int64_t offsetCTransposed = (batchIdx * static_cast<int64_t>(m_) + coordN) *
                                                  static_cast<int64_t>(n_) +
                                              coordM;

            // The fixpipe images carry the NZ-fractal padding of the L0C
            // accumulator: the fractal row is padded to M_ALIGN (2 fp32
            // elements, 8B rows) and the fractal column to C0_ALIGN (the
            // 16-element C0 frame), so the UB image bytes follow the same
            // alignment as the L0C source.
            const int64_t accBytes0 = static_cast<int64_t>(
                Blaze::Gemm::CeilAlign(static_cast<uint64_t>(shapeMA), M_ALIGN) *
                Blaze::Gemm::CeilAlign(static_cast<uint64_t>(shapeN), C0_ALIGN) * sizeof(float));
            const int64_t accBytes1 = static_cast<int64_t>(
                Blaze::Gemm::CeilAlign(static_cast<uint64_t>(shapeN), M_ALIGN) *
                Blaze::Gemm::CeilAlign(static_cast<uint64_t>(shapeMA), C0_ALIGN) * sizeof(float));
            const int64_t staging0Base = accBytes0;
            const int64_t staging0Bytes = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE) - accBytes0 -
                                          UB_TOP_RESERVED_BYTES;
            const int64_t staging1Base = accBytes1;
            const int64_t staging1Bytes = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE) - accBytes1 -
                                          UB_TOP_RESERVED_BYTES;
            auto layoutUbC = MakeLayoutC{}(shapeMA, shapeN);
            auto ubTensorC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), layoutUbC);
            auto layoutUbCT = MakeLayoutC{}(shapeN, shapeMA);
            auto ubTensorCT = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), layoutUbCT);
            if ASCEND_IS_AIC {
                Sync::WaitForVector<MIX_SYNC_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG, true);
                blockMmad(gmA, ubTensorC, tileShape, tileCoord, isDiagonal);
                Sync::NotifyVector<MIX_SYNC_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG, true);
            }
            if ASCEND_IS_AIV {
                if (AscendC::GetSubBlockIdx() == 0) {
                    epilogueOp(ubTensorC, tileShape, offsetC, false, params.schParams.block, params.schParams.block,
                               staging0Base, staging0Bytes);
                } else {
                    TupleShape tileShapeT{shapeN, shapeMA, shapeK, 1};
                    epilogueOp(ubTensorCT, tileShapeT, offsetCTransposed, false, params.schParams.block,
                               params.schParams.block, staging1Base, staging1Bytes, isDiagonal);
                }
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = params.problemShape;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    }

private:
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    // UB top headroom kept out of the epilogue staging budget: the historical
    // transpose path reserved two 512B scratch tiles there; the current nz2dn
    // fixpipe no longer reads it, the reservation stays as safety margin
    // against UB overflow.
    static constexpr int64_t UB_TOP_RESERVED_BYTES = 1024;
    // NZ-fractal alignment of the fp32 accumulator images (see accBytes).
    static constexpr uint64_t M_ALIGN = 2UL;
    static constexpr uint64_t C0_ALIGN = 16UL;

    __gm__ AType* aGmAddr_{nullptr};
    TupleShape problemShape_{};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
