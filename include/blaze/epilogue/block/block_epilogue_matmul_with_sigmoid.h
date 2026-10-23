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
 * \file block_epilogue_matmul_with_sigmoid.h
 * \brief AIV epilogue for out = sigmoid(x1 @ x2).
 */

#pragma once

#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <class DispatchPolicy_, class ElementType_>
class BlockEpilogueMatmulWithSigmoid {
public:
    using DispatchPolicy = DispatchPolicy_;
    using L0CDataType = float;
    using OutputType = ElementType_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outputGmAddr{nullptr};
        ProblemShape problemShape{};
        bool splitM{false};
    };

    __aicore__ inline BlockEpilogueMatmulWithSigmoid()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BUFFER_REUSE_EVENT);
        }
    }

    __aicore__ inline ~BlockEpilogueMatmulWithSigmoid()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BUFFER_REUSE_EVENT);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIV {
            outputGmAddr_ = params.outputGmAddr;
            m_ = asc::te::get<Blaze::Gemm::MNK_M>(params.problemShape);
            n_ = asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape);
            splitM_ = params.splitM;
        }
    }

    template <typename TensorC>
    __aicore__ inline void operator()(TensorC& ubTensor, const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        const int64_t curM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
        const int64_t curN = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        const int64_t halfM = Blaze::Gemm::CeilDiv(curM, static_cast<int64_t>(AscendC::GetTaskRation()));
        int64_t localRows = curM;
        int64_t localRowOffset = 0;
        if (splitM_) {
            const int64_t subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
            localRows = (static_cast<uint64_t>(curM) & 1UL) > 0UL ? halfM - subBlockIdx : halfM;
            localRowOffset = subBlockIdx * halfM;
        }
        const int64_t accumulatorRows = splitM_ ? halfM : Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN);
        const int64_t coordM = asc::te::get<Blaze::Gemm::MNK_M>(blockCoord);
        const int64_t coordN = asc::te::get<Blaze::Gemm::MNK_N>(blockCoord);
        const int64_t batchIdx = asc::te::get<Blaze::Gemm::MNK_B>(blockCoord);
        const int64_t dstOffset = ((batchIdx * m_ + coordM + localRowOffset) * n_) + coordN;
        if (localRows <= 0) {
            Blaze::Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            Blaze::Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            return;
        }
        Blaze::Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
        ProcessTile(ubTensor, localRows, accumulatorRows, curN, dstOffset);
        Blaze::Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
    }

private:
    template <typename TensorC>
    __aicore__ inline void ProcessTile(TensorC& ubTensor, int64_t localRows, int64_t accumulatorRows, int64_t tileN,
                                       int64_t tileGmOffset)
    {
        if (localRows <= 0 || tileN <= 0) {
            return;
        }
        const uint64_t nAlignAcc = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN),
                                                          static_cast<uint64_t>(ACC_ALIGN));
        const uint64_t nAlignOutput = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN),
                                                             static_cast<uint64_t>(OUTPUT_ALIGN));
        const uint64_t accumulatorBytes = Blaze::Gemm::CeilAlign(
            static_cast<uint64_t>(accumulatorRows) * nAlignAcc * sizeof(float), static_cast<uint64_t>(DATA_BLOCK));
        if (accumulatorBytes >= AscendC::TOTAL_UB_SIZE) {
            return;
        }
        const uint64_t outputBufferBytes = AscendC::TOTAL_UB_SIZE - accumulatorBytes;
        const uint64_t stageRowBytes = nAlignOutput * sizeof(ElementType_);
        const int64_t stageRows = static_cast<int64_t>(
            Blaze::Gemm::Min(static_cast<uint64_t>(localRows), outputBufferBytes / stageRowBytes));
        if (stageRows <= 0) {
            return;
        }

        for (int64_t stageOffset = 0; stageOffset < localRows; stageOffset += stageRows) {
            const int64_t rowsThisStage = Blaze::Gemm::Min(stageRows, localRows - stageOffset);
            const int64_t gmElemOffset = tileGmOffset + stageOffset * n_;
            const auto origin = asc::te::make_coord(0L, 0L);
            const auto validShape = asc::te::make_shape(rowsThisStage, tileN);
            auto outputLayout = Blaze::Gemm::MakeNDExtLayout<ElementType_>(rowsThisStage, tileN,
                                                                           static_cast<int64_t>(nAlignOutput));
            auto outputUb = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, ElementType_>(accumulatorBytes), outputLayout);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BUFFER_REUSE_EVENT);
            auto inputAddr = reinterpret_cast<__ubuf__ float*>(ubTensor.data().get()) + stageOffset * nAlignAcc;
            auto inputLayout = Blaze::Gemm::MakeNDExtLayout<float>(rowsThisStage, tileN,
                                                                   static_cast<int64_t>(nAlignAcc));
            auto inputUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(inputAddr), inputLayout);
            Tile::Sigmoid::run(inputUb, outputUb);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);

            auto outputGmStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType_*>(outputGmAddr_) +
                                                             gmElemOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, n_));
            auto outputGm = outputGmStorage.slice(origin, validShape);
            asc::te::copy(outputGm, outputUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BUFFER_REUSE_EVENT);
        }
    }

private:
    static constexpr int64_t SPLIT_M_ALIGN = 2;
    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t ACC_ALIGN = DATA_BLOCK / sizeof(float);
    static constexpr uint32_t OUTPUT_ALIGN = DATA_BLOCK / sizeof(ElementType_);
    static constexpr uint16_t OUTPUT_EVENT = 0;
    static constexpr uint16_t BUFFER_REUSE_EVENT = 1;
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;

    GM_ADDR outputGmAddr_{nullptr};
    int64_t m_{0};
    int64_t n_{0};
    bool splitM_{false};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
