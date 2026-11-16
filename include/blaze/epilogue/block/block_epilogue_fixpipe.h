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
 * \file block_epilogue_fixpipe.h
 * \brief
 */

#pragma once
#include "kernel_basic_intf.h"
#include "blaze/epilogue/block/block_epilogue_types.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_,
          typename FusionOp_ = Gemm::Block::DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFixpipe {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using DispatchPolicy = DispatchPolicy_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        ProblemShape problemShape{};
    };

    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t OUT_ALIGN = DATA_BLOCK / sizeof(DataTypeOut);
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr int64_t SPLIT_M_ALIGN = 2;

    __aicore__ inline BlockEpilogueFixpipe();
    __aicore__ inline void Init(Params const& params);
    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;

    // Compatibility interfaces and data members retained for existing callers.
    [[deprecated("Init接口在CANN 9.2.0版本标记废弃，将在2027年11月10日之后的版本删除，替换为Init(const Params&)")]]
    __aicore__ inline void Init(Params const& params, const ProblemShape& problemShape);
    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, bool splitM = false,
                                      int64_t baseM = 0, int64_t baseN = 0, uint64_t ubDB = 1);
    __aicore__ inline void Run(BlockShape const& blockShape, int64_t dstOffset, bool splitM, int64_t baseM,
                               int64_t baseN, uint64_t ubDB = 1);

    AscendC::LocalTensor<DataTypeIn> ubLocalTmp_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    ProblemShape problemShape_;
    uint64_t cvPingPong_{0};
};

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::BlockEpilogueFixpipe()
{}

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::Init(
    Params const& params)
{
    outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
    problemShape_ = params.problemShape;
    static_assert(sizeof(DataTypeIn) >= sizeof(DataTypeOut), "DataTypeIn size must be >= DataTypeOut size");
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline auto
BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::GetAccumulatorTensor(
    const AccumulatorTensorParams& params) const
{
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0),
                                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(params.rows, params.cols));
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::Init(
    Params const& params, const ProblemShape& problemShape)
{
    auto epilogueParams = params;
    epilogueParams.problemShape = problemShape;
    Init(epilogueParams);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::operator()(
    BlockShape const& blockShape, int64_t dstOffset, bool splitM, int64_t baseM, int64_t baseN, uint64_t ubDB)
{
    Run(blockShape, dstOffset, splitM, baseM, baseN, ubDB);
}

template <typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFixpipe<DataTypeOut_, DataTypeIn_, DispatchPolicy_, FusionOp_>::Run(
    BlockShape const& blockShape, int64_t dstOffset, bool splitM, int64_t baseM, int64_t baseN, uint64_t ubDB)
{
    int64_t mL1 = asc::te::get<Gemm::MNK_M>(blockShape);
    int64_t curM = mL1;
    if (baseM != 0) {
        curM = Blaze::Gemm::Min(curM, baseM);
    }
    int64_t halfBlockShapeM = Blaze::Gemm::CeilDiv(curM, AscendC::GetTaskRation());
    int64_t blockShapeM = curM;
    if (splitM) {
        blockShapeM = (static_cast<uint64_t>(curM) & 1UL) > 0UL ? (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                                                                  halfBlockShapeM;
    }
    int64_t nL1 = asc::te::get<Gemm::MNK_N>(blockShape);
    int64_t curBaseN = (baseN != 0) ? Blaze::Gemm::Min(nL1, baseN) : nL1;
    int64_t nL1Iter = Blaze::Gemm::CeilDiv(nL1, curBaseN);
    int64_t problemN = asc::te::get<Gemm::MNK_N>(problemShape_);
    constexpr int64_t c0Size = static_cast<int64_t>(asc::te::c0_element<DataTypeOut>);
    constexpr int64_t ubHalfElems = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn) /
                                                         Gemm::DOUBLE_BUFFER_COUNT);
    bool enablePp = (ubDB > 1);

    for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
        int64_t tileN = (nIdx + 1 == nL1Iter) ? (nL1 - curBaseN * nIdx) : curBaseN;
        int64_t blockShapeNAlign = Blaze::Gemm::CeilAlign(tileN, c0Size);
        int64_t inputSize = blockShapeM * blockShapeNAlign;
        uint16_t slot = enablePp ? static_cast<uint16_t>(cvPingPong_ & 1UL) : 0U;

        // Wait for AIC fixpipe (chunk ready) on the pipe that consumes UB first
        if constexpr (DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_RELU) {
            Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + slot);
        } else {
            Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + slot);
        }
        AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0,
                                                  AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
        // Point UB source to this chunk's ping-pong slot
        ubLocalTmp_ = ubLocal_[slot * ubHalfElems];

        if (inputSize > 0) {
            // CopyOut dstOffset along N advances per chunk; subBlock M split preserved
            int64_t offset = dstOffset + nIdx * curBaseN +
                             halfBlockShapeM * problemN * (AscendC::GetSubBlockIdx() & 0x1);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(blockShapeM),
                                                  static_cast<uint32_t>(tileN * sizeof(DataTypeOut)), 0,
                                                  static_cast<int64_t>((problemN - tileN) * sizeof(DataTypeOut)), 0};
            if constexpr (DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_RELU &&
                          !AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
                AscendC::Relu(ubLocalTmp_, ubLocalTmp_, blockShapeM * tileN);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
            }
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], ubLocalTmp_, copyParams);
        }

        // Notify AIC the UB slot is free
        Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + slot);
        cvPingPong_++;
    }
}
} // namespace Block
} // namespace Epilogue
} // namespace Blaze
