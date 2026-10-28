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
 * \file kernel_confidence_head_fused_mix.h
 * \brief ConfidenceHeadFused GEMV kernel using Tensor API / Blaze-style layering.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelConfidenceHeadMix, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockCoord = typename BlockScheduler::BlockCoord;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using AccType = typename BlockEpilogue::L0CDataType;
    using EpilogueShape = typename BlockEpilogue::BlockShape;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;

    static_assert(AscendC::Std::is_same_v<ProblemShape, asc::te::shape<int64_t, int64_t, int64_t, int64_t>>,
                  "ConfidenceHeadFused requires a problem shape of <M, N, KEmbed, KMarkov>.");
    static_assert(AscendC::Std::is_same_v<AccType, float> &&
                      AscendC::Std::is_same_v<typename BlockMmad::CType, AccType> &&
                      AscendC::Std::is_same_v<typename BlockEpilogue::OutputType, float>,
                  "ConfidenceHeadFused requires matching FP32 accumulation and epilogue types.");

    static constexpr bool HAS_MARKOV = BlockMmad::HAS_MARKOV;
    static constexpr bool TRANS_B = Blaze::Gemm::GetTransValue<LayoutB>();

    struct Params {
        ProblemShape problemShape;
        GM_ADDR weightGmAddr{nullptr};
        GM_ADDR hiddenGmAddr{nullptr};
        GM_ADDR markovGmAddr{nullptr};
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        Init(params);
        if (n_ == 0) {
            return;
        }
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() != PRIMARY_AIV_SUB_IDX) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        // Keep bias tiles within one batch; otherwise schedule all rows as one N dimension.
        const ProblemShape schedulerShape{1L, nPerBatch_, static_cast<int64_t>(kEmbed_ + kMarkov_),
                                          static_cast<int64_t>(n_) / nPerBatch_};
        BlockScheduler scheduler(schedulerShape, params.schParams);
        int64_t realCoreNums = scheduler.GetCoreNums();
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        BlockEpilogue blockEpilogue;
        blockEpilogue.Init(params.epilogueParams);

        // A zero reduction produces logits=0; keep bias, temperature and clamp processing in the epilogue.
        if (kEmbed_ + kMarkov_ == 0) {
            if ASCEND_IS_AIV {
                ProcessZeroReduction(scheduler, curBlockIdx, blockEpilogue);
            }
            return;
        }
        InitCrossCoreSync();

        // Complete Cube cleanup before restoring HF32 mode.
        {
            BlockMmad blockMmad;
            if ASCEND_IS_AIC {
                blockMmad.Init(params.mmadParams);
            }

            Process(params, blockMmad, scheduler, curBlockIdx, blockEpilogue);
        }
        if ASCEND_IS_AIC {
            UnsetHf32();
        }
    }

private:
    __aicore__ inline void ProcessZeroReduction(BlockScheduler& scheduler, int64_t coreIdx, BlockEpilogue& epilogue)
    {
        uint64_t loop = 0;
        const int64_t coreNums = scheduler.GetCoreNums();
        const int64_t blockNums = scheduler.GetBlockNums();
        for (int64_t blockIdx = coreIdx; blockIdx < blockNums; blockIdx += coreNums) {
            auto shape = scheduler.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto schedulerCoord = scheduler.GetBlockCoord(blockIdx);
            const int64_t coordN = asc::te::get<Blaze::Gemm::MNK_N>(schedulerCoord) +
                                   asc::te::get<Blaze::Gemm::MNK_B>(schedulerCoord) * nPerBatch_;
            const BlockCoord coord{0L, coordN, 0L, 0L};
            int64_t count = asc::te::get<Blaze::Gemm::MNK_N>(shape);
            auto input = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, AccType>(epilogue.GetUbAddr(loop++)),
                MakeNDExtLayout<AccType>(1L, count, count));
            EpilogueShape epilogueShape{1L, count, 0L, 1L};
            epilogue.template operator()<true>(input, epilogueShape, coord);
        }
    }

    __aicore__ inline void Process(Params const& params, BlockMmad& blockMmad, BlockScheduler& scheduler,
                                   int64_t curBlockIdx, BlockEpilogue& blockEpilogue)
    {
        auto layoutA = MakeLayoutA{}(1L, static_cast<int64_t>(kEmbed_ + kMarkov_));
        auto layoutB1 = MakeLayoutB{}(static_cast<int64_t>(kEmbed_), static_cast<int64_t>(n_));
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(b1GmAddr_), layoutB1);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                               reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr)),
                                           asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, nPerBatch_));

        const int64_t coreNums = scheduler.GetCoreNums();
        const int64_t blockNums = scheduler.GetBlockNums();
        if (coreNums <= 0 || blockNums <= 0) {
            return;
        }

        if ASCEND_IS_AIC {
            SetL2Cache(gmB1, kEmbed_);
        }

        if constexpr (HAS_MARKOV) {
            auto layoutB2 = MakeLayoutB{}(static_cast<int64_t>(kMarkov_), static_cast<int64_t>(n_));
            auto gmB2 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(b2GmAddr_), layoutB2);
            if ASCEND_IS_AIC {
                SetL2Cache(gmB2, kMarkov_);
            }
            ProcessTiles(blockMmad, blockEpilogue, scheduler, gmA, gmB1, gmB2, gmBias, curBlockIdx, coreNums,
                         blockNums);
        } else {
            ProcessTiles(blockMmad, blockEpilogue, scheduler, gmA, gmB1, gmB1, gmBias, curBlockIdx, coreNums,
                         blockNums);
        }

        if ASCEND_IS_AIC {
            WaitAllVecDone();
        }
    }

    template <typename TensorA, typename TensorB1, typename TensorB2, typename TensorBias>
    __aicore__ inline void ProcessTiles(BlockMmad& blockMmad, BlockEpilogue& blockEpilogue, BlockScheduler& scheduler,
                                        const TensorA& gmA, const TensorB1& gmB1, const TensorB2& gmB2,
                                        const TensorBias& gmBias, int64_t curBlockIdx, int64_t coreNums,
                                        int64_t blockNums)
    {
        uint64_t loopIdx = 0;
        for (int64_t blockIdx = curBlockIdx; blockIdx < blockNums; blockIdx += coreNums) {
            auto schedulerShape = scheduler.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto schedulerCoord = scheduler.GetBlockCoord(blockIdx);
            const int64_t coordN = asc::te::get<Blaze::Gemm::MNK_N>(schedulerCoord) +
                                   asc::te::get<Blaze::Gemm::MNK_B>(schedulerCoord) * nPerBatch_;
            const int64_t shapeN = asc::te::get<Blaze::Gemm::MNK_N>(schedulerShape);
            const BlockCoord blockCoord{0L, coordN, 0L, 0L};
            const typename BlockMmad::BlockShape blockShape{1L, shapeN,
                                                            asc::te::get<Blaze::Gemm::MNK_K>(schedulerShape), 1L};
            if (shapeN <= 0) {
                continue;
            }

            auto layoutUb = MakeNDExtLayout<AccType>(1L, shapeN, shapeN);
            uint64_t dbIdx = loopIdx & 1;
            uint64_t ubAddr = blockEpilogue.GetUbAddr(dbIdx);
            auto ubBlockC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AccType>(ubAddr),
                                                 layoutUb);

            if ASCEND_IS_AIC {
                WaitVecDone(dbIdx);
                auto gmBlockB1 = gmB1.slice(asc::te::make_coord(0L, coordN),
                                            asc::te::make_shape(static_cast<int64_t>(kEmbed_), shapeN));
                const int64_t biasN = asc::te::get<1, 1>(gmBias.layout().shape());
                auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, coordN % biasN),
                                                asc::te::make_shape(1L, shapeN));
                if constexpr (HAS_MARKOV) {
                    auto gmBlockB2 = gmB2.slice(asc::te::make_coord(0L, coordN),
                                                asc::te::make_shape(static_cast<int64_t>(kMarkov_), shapeN));
                    blockMmad(gmA, gmBlockB1, gmBlockB2, gmBlockBias, ubBlockC, blockShape);
                } else {
                    blockMmad(gmA, gmBlockB1, gmBlockB1, gmBlockBias, ubBlockC, blockShape);
                }
                SetCubeReady(dbIdx);
            }

            if ASCEND_IS_AIV {
                WaitCubeReady(dbIdx);
                EpilogueShape epilogueShape{1L, shapeN, 0L, 1L};
                blockEpilogue(ubBlockC, epilogueShape, blockCoord);
                SetVecDone(dbIdx);
            }
            ++loopIdx;
        }
    }

    __aicore__ inline void Init(Params const& params)
    {
        n_ = static_cast<uint64_t>(asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape));
        kEmbed_ = static_cast<uint64_t>(asc::te::get<Blaze::Gemm::CONFIDENCE_HEAD_K_EMBED>(params.problemShape));
        kMarkov_ = HAS_MARKOV ?
                       static_cast<uint64_t>(asc::te::get<Blaze::Gemm::CONFIDENCE_HEAD_K_MARKOV>(params.problemShape)) :
                       0UL;
        nPerBatch_ = params.epilogueParams.hasBias && params.epilogueParams.biasN != 0 ?
                         static_cast<int64_t>(params.epilogueParams.biasN) :
                         static_cast<int64_t>(n_);
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.weightGmAddr);
        b1GmAddr_ = reinterpret_cast<__gm__ BType*>(params.hiddenGmAddr);
        if constexpr (HAS_MARKOV) {
            b2GmAddr_ = reinterpret_cast<__gm__ BType*>(params.markovGmAddr);
        }
    }

    __aicore__ inline void UnsetHf32() { AscendC::SetHF32Mode(0); }

    template <typename TensorB>
    __aicore__ inline void SetL2Cache(TensorB& gmB, uint64_t k)
    {
        if (k != 0 && (k & (B_L2_CACHE_LINE_ELEMS - 1)) == 0) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        if ASCEND_IS_AIV {
            for (uint16_t i = 0; i < SYNC_DB_COUNT; ++i) {
                Sync::NotifyCube<SYNC_MODE4, PIPE_MTE3>(SYNC_VEC_TO_CUBE_FLAG + i);
            }
        }
    }

    __aicore__ inline void WaitVecDone(uint64_t dbIdx)
    {
        const uint16_t flag = static_cast<uint16_t>(SYNC_VEC_TO_CUBE_FLAG + dbIdx);
        Sync::WaitForVector<SYNC_MODE4, PIPE_FIX>(flag);
    }

    __aicore__ inline void WaitAllVecDone()
    {
        for (uint16_t i = 0; i < SYNC_DB_COUNT; ++i) {
            WaitVecDone(i);
        }
    }

    __aicore__ inline void SetCubeReady(uint64_t dbIdx)
    {
        const uint16_t flag = static_cast<uint16_t>(SYNC_CUBE_TO_VEC_FLAG + dbIdx);
        Sync::NotifyVector<SYNC_MODE4, PIPE_FIX>(flag);
    }

    __aicore__ inline void WaitCubeReady(uint64_t dbIdx)
    {
        Sync::WaitForCube<SYNC_MODE4, PIPE_V>(static_cast<uint16_t>(SYNC_CUBE_TO_VEC_FLAG + dbIdx));
    }

    __aicore__ inline void SetVecDone(uint64_t dbIdx)
    {
        Sync::NotifyCube<SYNC_MODE4, PIPE_MTE3>(static_cast<uint16_t>(SYNC_VEC_TO_CUBE_FLAG + dbIdx));
    }

private:
    static constexpr uint64_t B_L2_CACHE_LINE_ELEMS = 128UL / sizeof(BType);
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr int64_t PRIMARY_AIV_SUB_IDX = 0;
    static constexpr uint16_t SYNC_CUBE_TO_VEC_FLAG = 7;
    static constexpr uint16_t SYNC_VEC_TO_CUBE_FLAG = 9;
    static constexpr uint16_t SYNC_DB_COUNT = 2;

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* b1GmAddr_{nullptr};
    __gm__ BType* b2GmAddr_{nullptr};

    uint64_t n_{1};
    int64_t nPerBatch_{0};
    uint64_t kEmbed_{1};
    uint64_t kMarkov_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
