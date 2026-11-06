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
 * \file kernel_qgmm_mix_pertoken_finalize_routing.h
 * \brief TensorAPI kernel schedule for GroupedMatmulFinalizeRouting per-token dequant
 *        (INT8/FP8/HIFLOAT8 full quantization). AIC runs the quantized MMAD
 *        (BlockMmad with MatmulWithScaleMix, L0C -> UB split-M fixpipe); AIV runs the
 *        shared-input initialization and the dequant + logit + finalize-routing epilogue.
 */

#pragma once

#include <cstdint>

#include "kernel_basic_intf.h"
#include "tensor_api/tensor.h"

#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelQgmmMixPertokenFinalizeRouting, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockEpilogue::DataTypeOut;
    static_assert(AscendC::Std::is_one_of_v<CType, float, bfloat16_t>,
                  "GMM FR per-token initialization requires float/bfloat16_t output.");
    // The Mix block names its accumulator dtype L0CType (int32 for INT8, float otherwise).
    using MmadCType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutBias = typename BlockMmad::LayoutBias;
    static_assert((AscendC::Std::is_one_of_v<AType, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>),
                  "GMM FR per-token dequant requires quantized AType (int8/fp8/hifloat8).");
    static_assert((AscendC::Std::is_one_of_v<BType, int8_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>),
                  "GMM FR per-token dequant requires quantized BType (int8/fp8/hifloat8).");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "GMM FR per-token dequant only supports ND LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GMM FR per-token dequant only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(AscendC::Std::is_same_v<MmadCType, float> || AscendC::Std::is_same_v<MmadCType, int32_t>,
                  "GMM FR per-token dequant requires float/int32 MMAD output.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::DataTypeIn, MmadCType>,
                  "Epilogue accumulator dtype must match the MMAD output dtype.");

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using SchedulerBlockCoord = typename BlockScheduler::BlockCoord;
    using EpilogueBlockShape = typename BlockEpilogue::BlockShape;
    using EpilogueBlockCoord = typename BlockEpilogue::BlockCoord;

    struct GmmTiling {
        uint32_t groupNum;
        uint32_t batch;
        uint32_t sharedInputOffset;
        uint32_t sharedInputLen;
        float residualScale;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t l1BufferNum;
        uint8_t hasBias;
        uint8_t dbL0C;
        uint8_t groupListType;
        uint32_t deterWorkspaceSize;
        uint32_t coreNum;
    };

    struct SharedInputParams {
        GM_ADDR residualGm{nullptr};
        GM_ADDR yGmAddr{nullptr};
        uint32_t sharedInputOffset{0};
        uint32_t sharedInputLen{0};
        int32_t n{0};
        uint32_t batch{0};
        float residualScale{1.0F};
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        SharedInputParams sharedInputParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GmmTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool
        IS_WEIGHT_NZ_LAYOUT = AscendC::Std::is_one_of_v<LayoutB, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_DETER = BlockEpilogue::IS_DETER;
    static constexpr uint32_t GMM_PT_DQ_UB_DOUBLE_BUFFER_LEN = 1024 * 8;
    static constexpr uint32_t GMM_PT_DQ_ONE_CORE_ALIGN_LEN = 512;
    static constexpr uint32_t GMM_PT_DQ_ONE_CORE_ALIGN_LEN_SMALL = 32;
    // NZ/ZN weight fractal tile geometry for the grouped B operand: each stored fractal
    // covers 16 x 32 elements (16 rows by 32 columns, or transposed for ZN).
    static constexpr uint64_t GMM_PT_DQ_TILE_K_SMALL = 16;
    static constexpr uint64_t GMM_PT_DQ_TILE_N_SMALL = 16;
    static constexpr uint64_t GMM_PT_DQ_TILE_K_LARGE = 32;
    static constexpr uint64_t GMM_PT_DQ_TILE_N_LARGE = 32;
    static constexpr uint64_t GMM_PT_DQ_TILE_CAPACITY = GMM_PT_DQ_TILE_K_SMALL * GMM_PT_DQ_TILE_N_LARGE;

    static constexpr uint64_t INPUT_C0_SIZE = asc::te::c0_element<AType>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using TensorLayoutGroupList = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>;

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler& scheduler, int64_t m, int64_t hostBaseM)
    {
        const int64_t safeBaseM = hostBaseM > 0 ? hostBaseM : static_cast<int64_t>(BLOCK_CUBE);
        int64_t baseM = safeBaseM;
        if (m > 0) {
            const int64_t mCnt = CeilDiv(m, safeBaseM);
            baseM = CeilAlign(CeilDiv(m, mCnt), static_cast<int64_t>(BLOCK_CUBE));
        }
        curBaseM_ = static_cast<uint32_t>(baseM);
        scheduler.UpdateBaseM(curBaseM_);
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler)
    {
        const uint32_t mTailAlign = 1U;
        const uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(INPUT_C0_SIZE);
        scheduler.SetTailAlign(mTailAlign, nTailAlign);
    }

    __aicore__ inline void Run(const Params& params)
    {
        Init(params);
        BlockScheduler scheduler(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        if constexpr (IS_DETER) {
            deterWindow_.windowSize = params.gmmParams.deterWorkspaceSize / (static_cast<uint64_t>(n_) * sizeof(CType));
            deterWindow_.windowSize = (deterWindow_.windowSize / params.gmmParams.baseM) * params.gmmParams.baseM;
            deterWindow_.lowBoundM = deterWindow_.windowSize;
            ASCENDC_ASSERT(deterWindow_.windowSize > 0,
                           { KERNEL_LOG(KERNEL_ERROR, "Deterministic workspace must hold at least baseM rows"); });
            if (deterWindow_.windowSize == 0) {
                return;
            }
        }
        AscendC::SyncAll();
        if ASCEND_IS_AIV {
            Sync::NotifyCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>();
        }
        if ASCEND_IS_AIC {
            Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.Init(params.epilogueParams);
        }

        if (groupNum_ == 0) {
            End();
            return;
        }
        // 与 GMM A8W4 一致，用 [1, groupNum] 的 ND 扩展布局封装连续 INT64 分组列表。
        const auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
            TensorLayoutGroupList{}(1, static_cast<int64_t>(groupNum_)));
        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            const int64_t groupM = GetSplitValueFromGroupList(gmGroupList, groupIdx);
            if (groupM <= 0) {
                continue;
            }
            if constexpr (IS_DETER) {
                ProcessDeterministicGroup(params, scheduler, groupIdx, groupM);
            } else {
                ProcessSingleGroup(params, scheduler, groupIdx, groupM, preOffset_ - groupM);
                cumulativeGroupM_ += static_cast<uint64_t>(groupM);
            }
        }
        if constexpr (IS_DETER) {
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIV {
                FinalizeRoutingDeterministic(params);
                Sync::NotifyCube();
            }
            if ASCEND_IS_AIC {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }
            AscendC::SyncAll<false>();
        }
        End();
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        n_ = asc::te::get<MNK_N>(params.problemShape);
        k_ = asc::te::get<MNK_K>(params.problemShape);
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        isVecSetSyncCom_ = false;
        cumulativeGroupM_ = 0;
        deterWindow_ = {};

        if ASCEND_IS_AIV {
            sharedInputParams_ = &params.sharedInputParams;
            InitAllLocalTensor();
            InitSharedInput();
        }
        if ASCEND_IS_AIC {
            const ProblemShape mmadProblemShape{static_cast<int64_t>(gmmParams.baseM),
                                                static_cast<int64_t>(gmmParams.baseN), k_, 0};
            typename BlockMmad::Params mmadInitParams{};
            mmadInitParams.aGmAddr = params.mmadParams.aGmAddr;
            mmadInitParams.bGmAddr = params.mmadParams.bGmAddr;
            mmadInitParams.problemShape = mmadProblemShape;
            mmadInitParams.l0TileShape = BlockShape{static_cast<int64_t>(gmmParams.baseM),
                                                    static_cast<int64_t>(gmmParams.baseN),
                                                    static_cast<int64_t>(gmmParams.baseK), 0};
            mmadInitParams.kAL1 = gmmParams.kAL1;
            mmadInitParams.kBL1 = gmmParams.kBL1;
            mmadInitParams.l1BufferNum = gmmParams.l1BufferNum;
            mmadInitParams.enableL0CPingPong = gmmParams.dbL0C == DOUBLE_BUFFER_COUNT;
            mmadInitParams.biasGmAddr = nullptr;
            mmadInitParams.isBias = false;
            mmadOp_.Init(mmadInitParams);
        }
    }

    template <typename GroupListTensor_>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor_& gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == 0U) {
            const int64_t offset = gmGroupList[groupIdx];
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = gmGroupList[groupIdx];
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline int64_t GetBGroupOffset(uint32_t groupIdx) const
    {
        uint64_t singleGroupBSize = static_cast<uint64_t>(n_) * static_cast<uint64_t>(k_);
        if constexpr (IS_WEIGHT_NZ_LAYOUT) {
            if constexpr (TRANS_B) {
                singleGroupBSize = CeilDiv(static_cast<uint64_t>(k_), GMM_PT_DQ_TILE_K_LARGE) *
                                   CeilDiv(static_cast<uint64_t>(n_), GMM_PT_DQ_TILE_N_SMALL) * GMM_PT_DQ_TILE_CAPACITY;
            } else {
                singleGroupBSize = CeilDiv(static_cast<uint64_t>(n_), GMM_PT_DQ_TILE_N_LARGE) *
                                   CeilDiv(static_cast<uint64_t>(k_), GMM_PT_DQ_TILE_K_SMALL) * GMM_PT_DQ_TILE_CAPACITY;
            }
        }
        return static_cast<int64_t>(static_cast<uint64_t>(groupIdx) * singleGroupBSize);
    }

    // Deterministic mode: flush the finished workspace window into y. Rows are assigned to
    // vector cores by row index modulo so each y row has exactly one deterministic writer.
    __aicore__ inline void FinalizeRoutingDeterministic(const Params& params)
    {
        uint64_t totalM = deterWindow_.curGroupM - deterWindow_.windowStartM;
        if (totalM == 0) {
            return;
        }
        if (totalM > deterWindow_.windowSize) {
            totalM = deterWindow_.windowSize;
            deterWindow_.curGroupM = deterWindow_.windowStartM + deterWindow_.windowSize;
        }
        const uint64_t coreNumVec = static_cast<uint64_t>(params.gmmParams.coreNum) * AscendC::GetTaskRation();
        constexpr uint64_t ROW_INDEX_CHUNK = 256;
        for (uint64_t chunk = 0; chunk < totalM; chunk += ROW_INDEX_CHUNK) {
            const uint64_t count = Min(ROW_INDEX_CHUNK, totalM - chunk);
            epilogueOp_.PrefetchRowIndicesFromBase(deterWindow_.windowStartM + chunk, count);
            for (uint64_t i = 0; i < count; ++i) {
                const uint64_t outRow = epilogueOp_.GetPrefetchedRowIndex(i);
                if (outRow % coreNumVec == static_cast<uint64_t>(AscendC::GetBlockIdx())) {
                    epilogueOp_.DeterministicFlushRow(chunk + i, outRow, static_cast<uint64_t>(n_));
                }
            }
        }
    }

    __aicore__ inline void ProcessDeterministicGroup(const Params& params, BlockScheduler& scheduler, uint32_t groupIdx,
                                                     int64_t groupM)
    {
        const int64_t groupStartM = preOffset_ - groupM;
        int64_t processedM = 0;
        // All cores use the same row segments, including cores without a tile in a segment.
        while (processedM < groupM) {
            if (cumulativeGroupM_ == deterWindow_.lowBoundM) {
                AscendC::SyncAll<false>();
                if ASCEND_IS_AIV {
                    FinalizeRoutingDeterministic(params);
                }
                deterWindow_.windowStartM = cumulativeGroupM_;
                deterWindow_.lowBoundM = cumulativeGroupM_ + deterWindow_.windowSize;
                AscendC::SyncAll<false>();
            }
            const int64_t remainingRows = static_cast<int64_t>(deterWindow_.lowBoundM - cumulativeGroupM_);
            const int64_t segmentM = Min(groupM - processedM, remainingRows);
            ProcessSingleGroup(params, scheduler, groupIdx, segmentM, groupStartM + processedM);
            processedM += segmentM;
            cumulativeGroupM_ += static_cast<uint64_t>(segmentM);
            deterWindow_.curGroupM = cumulativeGroupM_;
        }
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockScheduler& scheduler, uint32_t groupIdx,
                                              int64_t groupM, int64_t mPrefixOffset)
    {
        BaseMBalance(scheduler, groupM, params.gmmParams.baseM);
        scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, n_, k_, 0});
        const EpilogueBlockCoord baseOffset{0,
                                            static_cast<int64_t>(groupIdx) * n_,
                                            static_cast<int64_t>(groupIdx) * n_,
                                            mPrefixOffset,
                                            mPrefixOffset,
                                            mPrefixOffset};

        const int64_t aOffset = mPrefixOffset * k_;
        const int64_t bOffset = GetBGroupOffset(groupIdx);

        auto layoutA = MakeLayoutA{}(groupM, k_);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + aOffset), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bBasePtr_ + bOffset), layoutB);

        SchedulerBlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        do {
            ProcessSingleBlock(gmA, gmB, scheduler, blockCoord, baseOffset);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

    template <typename TensorA_, typename TensorB_>
    __aicore__ inline void ProcessSingleBlock(const TensorA_& gmA, const TensorB_& gmB, BlockScheduler& scheduler,
                                              const SchedulerBlockCoord& blockCoord,
                                              const EpilogueBlockCoord& baseOffset)
    {
        const SchedulerBlockShape schedulerBlockShape = scheduler.GetBlockShape(blockCoord);
        const int64_t blockM = asc::te::get<MNK_M>(schedulerBlockShape);
        const int64_t blockN = asc::te::get<MNK_N>(schedulerBlockShape);
        if (blockM <= 0 || blockN <= 0) {
            return;
        }
        const int64_t mSplitOffset = asc::te::get<MNK_K>(schedulerBlockShape);
        const int64_t nSplitOffset = asc::te::get<MNK_B>(schedulerBlockShape);
        const int64_t mBlockIdx = asc::te::get<MNK_M>(blockCoord);
        const int64_t nBlockIdx = asc::te::get<MNK_N>(blockCoord);
        const int64_t mPos = mBlockIdx * static_cast<int64_t>(curBaseM_) + mSplitOffset;
        const int64_t nPos = nBlockIdx * static_cast<int64_t>(baseN_) + nSplitOffset;
        const BlockShape blockShape{blockM, blockN, k_, 0};

        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }
            auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)),
                                      asc::te::make_shape(blockM, k_));
            auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                      asc::te::make_shape(k_, blockN));
            auto ubC = epilogueOp_.GetL0c2UbTensor(blockM, blockN);
            mmadOp_(gmBlockA, gmBlockB, ubC, blockShape);
            Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
        }
        isVecSetSyncCom_ = true;

        if ASCEND_IS_AIV {
            Sync::WaitForCube();
            const EpilogueBlockShape epilogueShape{blockM, blockN, 0, 0};
            const EpilogueBlockCoord epilogueOffset{nPos, nPos, nPos, mPos, mPos, mPos};
            epilogueOp_(typename BlockEpilogue::ProblemShape{0, n_, k_}, epilogueShape, epilogueOffset, baseOffset,
                        cumulativeGroupM_ - deterWindow_.windowStartM);
            Sync::NotifyCube();
        }
    }

    __aicore__ inline void InitAllLocalTensor()
    {
        shareInputUbPingOffset_ = 0UL;
        shareInputUbPongOffset_ = shareInputUbPingOffset_ + GMM_PT_DQ_UB_DOUBLE_BUFFER_LEN * sizeof(bfloat16_t);
        ubOutPingOffset_ = shareInputUbPongOffset_ + GMM_PT_DQ_UB_DOUBLE_BUFFER_LEN * sizeof(bfloat16_t);
        ubOutPongOffset_ = ubOutPingOffset_ + GMM_PT_DQ_UB_DOUBLE_BUFFER_LEN * sizeof(CType);
    }

    __aicore__ inline void InitSharedInput()
    {
        if ASCEND_IS_AIC {
            return;
        }
        vectorCoreNum_ = static_cast<uint32_t>(AscendC::GetBlockNum() * AscendC::GetTaskRation());
        if (AscendC::GetBlockIdx() >= vectorCoreNum_) {
            return;
        }
        const uint32_t sharedInputOffset = sharedInputParams_->sharedInputOffset;
        const uint32_t sharedInputLen = sharedInputParams_->sharedInputLen;
        const uint64_t sharedInputStart = static_cast<uint64_t>(sharedInputParams_->n) * sharedInputOffset;
        if (sharedInputLen == 0) {
            return;
        }

        const uint64_t totalOutput = static_cast<uint64_t>(sharedInputParams_->n) * sharedInputLen;
        uint64_t singleCount = CeilDiv(totalOutput, static_cast<uint64_t>(vectorCoreNum_));
        const uint32_t alignLen = totalOutput > GMM_PT_DQ_ONE_CORE_ALIGN_LEN ? GMM_PT_DQ_ONE_CORE_ALIGN_LEN :
                                                                               GMM_PT_DQ_ONE_CORE_ALIGN_LEN_SMALL;
        singleCount = CeilDiv(singleCount, static_cast<uint64_t>(alignLen)) * alignLen;
        const uint64_t baseOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * singleCount;
        if (baseOffset >= totalOutput) {
            return;
        }
        if (baseOffset + singleCount > totalOutput) {
            singleCount = totalOutput - baseOffset;
        }
        const uint64_t outOffset = sharedInputStart + baseOffset;
        using ProcessSharedInput = Blaze::Epilogue::Tile::ProcessSharedInput<CType>;
        const auto layout = asc::te::make_layout(asc::te::make_shape(singleCount), asc::te::make_stride(1L));
        const auto sharedInput = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                reinterpret_cast<__gm__ bfloat16_t*>(sharedInputParams_->residualGm) + baseOffset),
            layout);
        const auto sharedOutput = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(sharedInputParams_->yGmAddr) +
                                                         outOffset),
            layout);
        ProcessSharedInput::Run(sharedInput, sharedOutput, sharedInputParams_->residualScale,
                                {shareInputUbPingOffset_, shareInputUbPongOffset_, ubOutPingOffset_, ubOutPongOffset_,
                                 GMM_PT_DQ_UB_DOUBLE_BUFFER_LEN});
    }

    struct DeterministicWindow {
        uint64_t windowSize{0};
        uint64_t windowStartM{0};
        uint64_t lowBoundM{0};
        uint64_t curGroupM{0};
    };

    BlockMmad mmadOp_;
    const SharedInputParams* sharedInputParams_{nullptr};
    uint64_t shareInputUbPingOffset_{0};
    uint64_t shareInputUbPongOffset_{0};
    uint64_t ubOutPingOffset_{0};
    uint64_t ubOutPongOffset_{0};
    uint32_t vectorCoreNum_{0};
    BlockEpilogue epilogueOp_;
    DeterministicWindow deterWindow_{};
    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    int64_t preOffset_{0};
    uint64_t cumulativeGroupM_{0};
    int64_t n_{0};
    int64_t k_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{0};
    bool isVecSetSyncCom_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
