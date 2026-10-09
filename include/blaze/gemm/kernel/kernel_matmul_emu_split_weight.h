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
 * \file kernel_matmul_emu_split_weight.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/epilogue/block/block_epilogue_muls_add.h"
#include "blaze/gemm/block/block_mmad_matmul_emu_split_weight.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

constexpr int16_t AIV_SYNC_AIC_FLAG = 0;
constexpr int16_t AIC_SYNC_AIV_FLAG = 1;
constexpr int16_t AIC_SYNC_AIV_MODE_4 = 4;
constexpr int16_t FLAG_ID_MAX = 16;

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMatmulEmuSplitWeight, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
    {
        Init(params);
        Run(params);
    }

private:
    BlockMmad blockMmadOp_;
    BlockEpilogue epilogueOp_;

    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
    __gm__ AType* xGmAddr_;
    __gm__ BType* wHighGmAddr_;
    __gm__ BType* wLowGmAddr_;

    __aicore__ inline void Init(Params& params)
    {
        m_ = asc::te::get<MNK_M>(params.problemShape);
        n_ = asc::te::get<MNK_N>(params.problemShape);
        k_ = asc::te::get<MNK_K>(params.problemShape);
        params.mmadParams.k = static_cast<uint64_t>(k_);
        xGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.xGmAddr);
        wHighGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.wHighGmAddr);
        wLowGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.wLowGmAddr);

        if ASCEND_IS_AIC {
            int64_t blockIdx = GetCurrentBlockIdx();
            int64_t usedCoreNum = static_cast<int64_t>(params.mmadParams.usedCoreNum);
            if (blockIdx >= usedCoreNum) {
                return;
            }
            blockMmadOp_.Init(params.mmadParams);
        }
        if ASCEND_IS_AIV {
            ProblemShape problemShape4 = {m_, n_, k_, 1};
            epilogueOp_.Init(params.epilogueParams, static_cast<int64_t>(params.mmadParams.baseM),
                             static_cast<int64_t>(params.mmadParams.baseN), problemShape4);
        }
    }

    __aicore__ inline void Run(Params const& params)
    {
        int64_t baseM = params.mmadParams.baseM;
        int64_t baseN = params.mmadParams.baseN;

        auto layoutA = MakeLayoutA{}(m_, k_);
        auto layoutBHigh = MakeLayoutB{}(k_, n_);
        auto layoutBLow = MakeLayoutB{}(k_, n_);
        // 对两个AIV分别做核间同步
        bool enableCVSync[2] = {false, false};
        constexpr int64_t kPos = 0L;

        uint64_t mBlockNums = CeilDiv(static_cast<uint32_t>(m_), static_cast<uint32_t>(baseM));
        uint64_t nBlockNums = CeilDiv(static_cast<uint32_t>(n_), static_cast<uint32_t>(baseN));
        uint64_t totalBlocks = mBlockNums * nBlockNums;
        int64_t curBlockIdx = GetCurrentBlockIdx();
        int64_t usedCoreNum = static_cast<int64_t>(params.mmadParams.usedCoreNum);
        uint64_t blockCount = 0;

        for (int64_t blockIdx = curBlockIdx; blockIdx < static_cast<int64_t>(totalBlocks);
             blockIdx += AscendC::GetBlockNum(), ++blockCount) {
            uint64_t targetSubBlockId = blockCount & 1UL;
            bool useSubBlockOne = targetSubBlockId == 1;
            int16_t flagOffset = static_cast<int16_t>(targetSubBlockId * FLAG_ID_MAX);

            // 切分逻辑简单，未使用独立的block_scheduler
            int64_t mPos = (blockIdx / nBlockNums) * baseM;
            int64_t nPos = (blockIdx % nBlockNums) * baseN;
            int64_t curM = (mPos + baseM > m_) ? (m_ - mPos) : baseM;
            int64_t curN = (nPos + baseN > n_) ? (n_ - nPos) : baseN;

            int64_t offsetC = mPos * n_ + nPos;
            BlockShape singleShape{curM, curN, 1, 1};

            if ASCEND_IS_AIC {
                if (enableCVSync[targetSubBlockId]) {
                    Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + flagOffset);
                }

                auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(xGmAddr_), layoutA);
                auto gmBHigh = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(wHighGmAddr_),
                                                    layoutBHigh);
                auto gmBLow = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(wLowGmAddr_),
                                                   layoutBLow);

                auto curNUbAlign = CeilAlign(curN, static_cast<int64_t>(C0_SIZE_fp32));

                constexpr uint64_t UB_HALF_BYTES = AscendC::TOTAL_UB_SIZE / DOUBLE_BUFFER_COUNT;

                auto layoutUBLow = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                              AscendC::Std::Int<C0_SIZE_L0C>>(curM, curNUbAlign);
                auto ubBlockCLow = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(0),
                                                        layoutUBLow);

                auto layoutUBHigh = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                               AscendC::Std::Int<C0_SIZE_L0C>>(curM, curNUbAlign);
                auto ubBlockCHigh = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(UB_HALF_BYTES), layoutUBHigh);

                auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(curM, k_));
                auto gmBlockBHigh = gmBHigh.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k_, curN));
                auto gmBlockBLow = gmBLow.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k_, curN));

                blockMmadOp_(gmBlockA, gmBlockBHigh, gmBlockBLow, ubBlockCHigh, ubBlockCLow, singleShape,
                             useSubBlockOne);

                enableCVSync[targetSubBlockId] = true;
                Sync::NotifyVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + flagOffset);
            }

            if ASCEND_IS_AIV {
                if (AscendC::GetSubBlockIdx() != targetSubBlockId) {
                    continue;
                }
                Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
                epilogueOp_({curM, curN, 1, 1}, offsetC);
                if (blockIdx + 2 * AscendC::GetBlockNum() < static_cast<int64_t>(totalBlocks)) {
                    Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
                }
            }
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
