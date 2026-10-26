/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_matmul_iterbatch.h
 * \brief GemmUniversal partial specialization for the IterBatch path:
 *        blocks = CeilDiv(b, iterBatchL1), strided across cores. AIC software-pipelines
 *        the GM->L1 preload and fixpipes L0C to GM (ON_THE_FLY) or to the paired AIVs'
 *        UB (ND_FIXPIPE_1_2); the AIV epilogue writes the UB slots to GM in ND layout.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/block/block_mmad_matmul_iterbatch.h"
#include "blaze/gemm/block/block_scheduler_matmul_iterbatch.h"
#include "blaze/epilogue/block/block_epilogue_iterbatch.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<
                        AscendC::Std::is_same_v<KernelIterBatch, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    using BlockSchedulerParams = typename Block::BlockSchedulerMatmulIterBatch<ProblemShape>::Params;
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schedulerParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        BlockEpilogue epilogueOp;
        BlockMmad blockMmadOp;
        int64_t curBlockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        if constexpr (BlockMmad::IS_MIX) {
            if ASCEND_IS_AIV {
                curBlockIdx /= static_cast<int64_t>(AscendC::GetTaskRation());
            }
        }
        Init(params);
        Block::BlockSchedulerMatmulIterBatch<ProblemShape> bs(params.problemShape, params.schedulerParams);
        int64_t blockNums = bs.GetBlockNums();
        int64_t realCoreNums = bs.GetCoreNums(blockNum);
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schedulerParams.isHf32);
        auto layoutA = MakeLayoutA{}(b_, m_, k_);
        auto layoutB = MakeLayoutB{}(b_, k_, n_);
        auto layoutC = MakeLayoutC{}(b_, m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
        // Enable the dual page table
        SetL2Cache(gmA, gmB, params.schedulerParams.l2CacheDisable,
                   params.mmadParams.aGmAddr == params.mmadParams.bGmAddr);
        if constexpr (BlockMmad::IS_MIX) {
            // This core's total tile count; the epilogue suppresses the release flag after its last tile.
            uint64_t mnL0Cnt = CeilDiv(m_, static_cast<uint64_t>(params.schedulerParams.baseM)) *
                               CeilDiv(n_, static_cast<uint64_t>(params.schedulerParams.baseN));
            uint64_t totalTiles = 0;
            for (int64_t t = curBlockIdx; t < blockNums; t += blockNum) {
                totalTiles += CeilDiv(static_cast<uint64_t>(asc::te::get<MNK_B>(bs.GetBlockShape(t))),
                                      static_cast<uint64_t>(params.schedulerParams.iterBatchL0)) *
                              mnL0Cnt;
            }
            epilogueOp.Init(params.epilogueParams, m_, n_, totalTiles);
        }
        BlockMmadParams mmadParams = params.mmadParams;
        blockMmadOp.Init(mmadParams);
        uint64_t firstIterBatchL1 = static_cast<uint64_t>(asc::te::get<MNK_B>(bs.GetBlockShape(curBlockIdx)));
        auto gmCurA = gmA.slice(
            asc::te::make_coord(asc::te::get<MNK_B>(bs.GetBlockCoord(curBlockIdx)), asc::te::make_coord(0, 0)),
            asc::te::make_shape(firstIterBatchL1, asc::te::make_shape(m_, k_)));
        auto gmCurB = gmB.slice(
            asc::te::make_coord(asc::te::get<MNK_B>(bs.GetBlockCoord(curBlockIdx)), asc::te::make_coord(0, 0)),
            asc::te::make_shape(firstIterBatchL1, asc::te::make_shape(k_, n_)));
        for (int64_t blockIdx = curBlockIdx; blockIdx < blockNums; blockIdx += blockNum) {
            uint64_t curIterBatchL1 = static_cast<uint64_t>(asc::te::get<MNK_B>(bs.GetBlockShape(blockIdx)));
            bool isFinalRound = blockIdx + blockNum >= blockNums;
            if ASCEND_IS_NOT_AIV {
                int64_t nextIdx = isFinalRound ? blockIdx : blockIdx + blockNum;
                bool isPreLoadRound = blockIdx == curBlockIdx;
                uint64_t nextIterBatchL1 = static_cast<uint64_t>(asc::te::get<MNK_B>(bs.GetBlockShape(nextIdx)));
                auto cCur = gmC.slice(
                    asc::te::make_coord(asc::te::get<MNK_B>(bs.GetBlockCoord(blockIdx)), asc::te::make_coord(0, 0)),
                    asc::te::make_shape(curIterBatchL1, asc::te::make_shape(m_, n_)));
                auto gmNextA = gmA.slice(
                    asc::te::make_coord(asc::te::get<MNK_B>(bs.GetBlockCoord(nextIdx)), asc::te::make_coord(0, 0)),
                    asc::te::make_shape(nextIterBatchL1, asc::te::make_shape(m_, k_)));
                auto gmNextB = gmB.slice(
                    asc::te::make_coord(asc::te::get<MNK_B>(bs.GetBlockCoord(nextIdx)), asc::te::make_coord(0, 0)),
                    asc::te::make_shape(nextIterBatchL1, asc::te::make_shape(k_, n_)));
                blockMmadOp(gmCurA, gmCurB, gmNextA, gmNextB, gmBias, cCur, curIterBatchL1, nextIterBatchL1,
                            isPreLoadRound, isFinalRound);
            }
            if constexpr (BlockMmad::IS_MIX) {
                if ASCEND_IS_AIV {
                    epilogueOp(asc::te::get<MNK_B>(bs.GetBlockCoord(blockIdx)) * m_ * n_, params.schedulerParams.baseM,
                               params.schedulerParams.baseN, curIterBatchL1, params.schedulerParams.iterBatchL0);
                }
            }
        }
        Blaze::Gemm::UnsetHF32(params.schedulerParams.isHf32);
    }

private:
    __aicore__ inline void Init(Params const& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        b_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        if (params.mmadParams.biasGmAddr != nullptr) {
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        }
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode, bool aSameAsB)
    {
        if ((l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) && !aSameAsB) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if ((l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) && !aSameAsB) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t b_{1};
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
