/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_rotate_quant.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"
#include "basic_api/kernel_operator_scalar_intf.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_rotate_quant.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelRotateQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

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
    using AElementType = typename BlockMmad::AElementType;
    using BElementType = typename BlockMmad::BElementType;
    using L0CType = typename BlockMmad::L0CType;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = typename BlockScheduler::BlockCoord;

    using MakeLayoutB = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn,
                                                     AscendC::Std::Int<asc::te::c0_element<BElementType>>>;

    static_assert(AscendC::Std::is_same_v<ProblemShape, TupleShape>,
                  "RotateQuant kernel requires an int64 M/N/K/B problem shape.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::DataTypeIn, bfloat16_t>,
                  "RotateQuant epilogue must consume the BF16 Fixpipe output.");

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        typename BlockScheduler::GroupParams groupParams{};
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
    {
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        int64_t curBlockIdx = Gemm::GetCurrentBlockIdx();
        int64_t coreNums = AscendC::GetBlockNum();
        if (coreNums <= 0) {
            return;
        }
        Init(params);

        BlockScheduler bs(params.problemShape, coreNums, params.schParams, params.groupParams);
        int64_t blockNums = bs.GetBlockNums();

        int64_t realCoreNums = bs.GetCoreNums(coreNums);
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        blockMmadOp.Init(params.mmadParams);

        int64_t oriOverflowMode = AscendC::GetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL,
                                                      ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>();
        if ASCEND_IS_AIV {
            AscendC::SetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL, ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>(0);
        }

        params.epilogueParams.alpha = alpha_;
        params.epilogueParams.needClamp = needClamp_;
        epilogueOp.Init(params.epilogueParams);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmPtr_), MakeLayoutAGm());
        int64_t loadedRotationOffset = -1;
        for (int64_t blockIdx = curBlockIdx, roundIdx = 0; blockIdx < blockNums; blockIdx += coreNums, roundIdx++) {
            auto blockInfo = bs.GetBlockInfo(blockIdx);
            TupleL1L0Shape blockShape = blockInfo.blockShape;
            BlockCoord blockCoord = blockInfo.blockCoord;
            if ASCEND_IS_AIC {
                if (roundIdx > 0) {
                    if ((roundIdx & 1) == 1) {
                        Gemm::Sync::WaitForVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(ROTATE_QUANT_SYNC_AIV_AIC_FLAG);
                    } else {
                        Gemm::Sync::WaitForVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(ROTATE_QUANT_SYNC_AIV_AIC_FLAG +
                                                                                    ROTATE_QUANT_FLAG_ID_MAX);
                    }
                }
                int64_t coordM = asc::te::get<Gemm::MNK_M>(blockCoord);
                int64_t coordB = asc::te::get<Gemm::MNK_B>(blockCoord);
                int64_t blkM = asc::te::get<Gemm::MNK_M>(blockShape);
                auto gmBlockA = asc::te::squeeze<0>(
                    gmA.slice(asc::te::make_coord(coordB, asc::te::make_coord(coordM, 0L)),
                              asc::te::make_shape(1L, asc::te::make_shape(blkM, k_))));
                auto gmBlockRot = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(rotGmPtr_ + blockInfo.rotationOffset),
                    MakeLayoutB{}(k_, k_));
                // Rotation changes do not reset the block buffers or the AIC/AIV round counter.
                blockMmadOp(gmBlockA, gmBlockRot, blockShape, loadedRotationOffset != blockInfo.rotationOffset);
                loadedRotationOffset = blockInfo.rotationOffset;
                if (roundIdx % 2 == 0) {
                    Gemm::Sync::NotifyVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(ROTATE_QUANT_SYNC_AIC_AIV_FLAG);
                } else {
                    Gemm::Sync::NotifyVector<ROTATE_QUANT_SYNC_MODE, PIPE_FIX>(ROTATE_QUANT_SYNC_AIC_AIV_FLAG +
                                                                               ROTATE_QUANT_FLAG_ID_MAX);
                }
            }
            if ASCEND_IS_AIV {
                if ((roundIdx & 1) == AscendC::GetSubBlockIdx()) {
                    Gemm::Sync::WaitForCube<ROTATE_QUANT_SYNC_MODE, PIPE_V>(ROTATE_QUANT_SYNC_AIC_AIV_FLAG);
                    epilogueOp(blockShape, blockCoord);
                    if (blockIdx + coreNums < blockNums) {
                        Gemm::Sync::NotifyCube<ROTATE_QUANT_SYNC_MODE, PIPE_MTE3>(ROTATE_QUANT_SYNC_AIV_AIC_FLAG);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            AscendC::SetCtrlSpr<ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL, ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL>(
                oriOverflowMode);
        }
    }

private:
    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        m_ = asc::te::get<Gemm::MNK_M>(problemShape_);
        n_ = asc::te::get<Gemm::MNK_N>(problemShape_);
        k_ = asc::te::get<Gemm::MNK_K>(problemShape_);
        b_ = asc::te::get<Gemm::MNK_B>(problemShape_);

        const BlockMmadParams& blockMmadParams = params.mmadParams;
        aGmPtr_ = reinterpret_cast<__gm__ AElementType*>(blockMmadParams.aGmAddr);
        rotGmPtr_ = reinterpret_cast<__gm__ BElementType*>(blockMmadParams.bGmAddr);
        if (params.schParams.hasAlpha == 1) {
            auto alphaLayout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_shape(1L));
            auto gmAlpha = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                    reinterpret_cast<__gm__ bfloat16_t*>(blockMmadParams.alphaGMAddr)),
                                                alphaLayout);
            alpha_ = gmAlpha[0];
            const float alphaValue = AscendC::Cast(alpha_);
            // alpha大于0并且小于1时，需要做截断
            needClamp_ = alphaValue > 0.0f && alphaValue < 1.0f;
        }
    }

    // B == 1 views x as (1, M * N/K, K); B > 1 views x as (B, M, K).
    // In the latter case batch B advances by K, while adjacent M rows advance by B * K.
    __aicore__ inline auto MakeLayoutAGm()
    {
        int64_t rows = (b_ > 1) ? m_ : m_ * (n_ / k_);
        int64_t rowStride = b_ * k_;
        auto matrixShape = asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, rows),
                                               asc::te::make_shape(asc::te::_1{}, k_));
        auto matrixStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, rowStride),
                                                 asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
        auto shape = asc::te::make_shape(b_, matrixShape);
        auto stride = asc::te::make_stride(k_, matrixStride);
        return asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<AElementType>>(
            shape, stride);
    }

private:
    static constexpr uint16_t ROTATE_QUANT_SYNC_MODE = 4;
    static constexpr uint16_t ROTATE_QUANT_SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint16_t ROTATE_QUANT_SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint16_t ROTATE_QUANT_FLAG_ID_MAX = 16;
    static constexpr uint16_t ROTATE_QUANT_FLOAT_OVERFLOW_MODE_CTRL = 60;

    __gm__ AElementType* aGmPtr_{nullptr};
    __gm__ BElementType* rotGmPtr_{nullptr};
    TupleShape problemShape_{};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t b_{0};
    bfloat16_t alpha_{0.0f};
    bool needClamp_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
