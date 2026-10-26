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
 * \file kernel_qbmm_pertensor_streamk.h
 * \brief GemmUniversal specialization for QBMM per-tensor StreamK.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

/**
 * QBMM per-tensor StreamK specialization.
 *
 * DP blocks use fixpipe dequantization and write C directly. StreamK blocks
 * write raw accumulation partials to workspace for AIV reduction.
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelQbmmPertensorMultiBlockStreamK,
                                                      typename BlockMmad_::DispatchPolicy::ScheduleType> &&
                              AscendC::Std::is_same_v<KernelQbmmPertensorMultiBlockStreamK,
                                                      typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
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
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using X2ScaleType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using WorkspaceType = typename BlockEpilogue::WorkspaceType;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, asc::te::layout_trait_default<AType>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, asc::te::layout_trait_default<BType>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, asc::te::layout_trait_default<CType>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, asc::te::layout_trait_default<BiasType>>;

private:
    static constexpr bool IS_INT8_INPUT = AscendC::Std::is_same_v<AType, int8_t> &&
                                          AscendC::Std::is_same_v<BType, int8_t>;
    static constexpr bool IS_HIFLOAT8_INPUT = AscendC::Std::is_same_v<AType, hifloat8_t> &&
                                              AscendC::Std::is_same_v<BType, hifloat8_t>;
    static constexpr bool IS_FP8_INPUT = IsFp8<AType>() && IsFp8<BType>();
    static constexpr bool IS_SUPPORTED_OUTPUT = AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>;

    static_assert(IS_INT8_INPUT || IS_HIFLOAT8_INPUT || IS_FP8_INPUT,
                  "QBMM Per-tensor StreamK: AType/BType must both be int8_t, both be hifloat8_t, or each be "
                  "fp8_e4m3fn_t/fp8_e5m2_t.");
    static_assert(IS_SUPPORTED_OUTPUT, "QBMM Per-tensor StreamK: BlockMmad::CType must be half/bfloat16_t/float.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::OutType, CType>,
                  "QBMM Per-tensor StreamK: BlockEpilogue::OutType must match BlockMmad::CType.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::WorkspaceType, typename BlockMmad::WorkspaceType>,
                  "QBMM Per-tensor StreamK: BlockEpilogue::WorkspaceType must match BlockMmad::WorkspaceType.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM Per-tensor StreamK: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(
        AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                  asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
        "QBMM Per-tensor StreamK: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");
    static_assert(AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn>,
                  "QBMM Per-tensor StreamK: LayoutC must be nd_ext_layout_ptn.");

public:
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams blockMmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        if (params.schParams.usedCoreNum <= 0 || asc::te::get<MNK_B>(params.problemShape) != 1) {
            return;
        }
        Init(params);

        BlockScheduler bs(params.problemShape, params.schParams);
        mL1_ = params.schParams.baseM;
        nL1_ = params.schParams.baseN;
        mBlockNums_ = Blaze::Gemm::CeilDiv(m_, mL1_);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n_, nL1_);
        skBlockNums_ = Blaze::Gemm::CeilDiv(k_, params.schParams.singleCoreK);

        if ASCEND_IS_AIC {
            ProcessOnAic(params, bs);
        }

        if ASCEND_IS_AIV {
            ProcessOnAiv(params, bs);
        }
    }

private:
    __aicore__ inline void ProcessOnAic(Params const& params, BlockScheduler& bs)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx < bs.GetCoreNums()) {
            ProcessAicBlocks(params, bs, curBlockIdx);
        }
        SignalAicFinish();
    }

    __aicore__ inline void ProcessAicBlocks(Params const& params, BlockScheduler& bs, int64_t curBlockIdx)
    {
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        BlockMmad blockMmad;
        BlockShape l0BlockShape{params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 0};
        blockMmad.Init(params.problemShape, l0BlockShape, params.schParams.kL1, params.schParams.kL1, 2UL,
                       QuantMode::PERTENSOR_MODE, params.blockMmadParams.biasGmAddr != nullptr, false);

        int64_t totalBlockNums = bs.GetBlockNums();
        int64_t tailSKTotalBlockNums = (mBlockNums_ * nBlockNums_) % usedCoreNum * skBlockNums_;
        int64_t totalMNBlockNumsInDP = mBlockNums_ * nBlockNums_ - (mBlockNums_ * nBlockNums_) % usedCoreNum;

        auto layoutA = MakeLayoutA{}(m_, k_);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += usedCoreNum) {
            int64_t actualBlockIdx = GetActualBlockIdx(bs, blockIdx, totalBlockNums, tailSKTotalBlockNums, usedCoreNum);
            ProcessAicBlock(params, bs, blockMmad, gmA, gmB, gmBias, gmC, actualBlockIdx, totalMNBlockNumsInDP);
        }
    }

    __aicore__ inline void SignalAicFinish() const
    {
        AscendC::CrossCoreSetFlag<AIC_ONLY_SYNC_MODE, PIPE_FIX>(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }

    __aicore__ inline int64_t GetActualBlockIdx(BlockScheduler& bs, int64_t blockIdx, int64_t totalBlockNums,
                                                int64_t tailSKTotalBlockNums, int64_t usedCoreNum) const
    {
        if (bs.CheckIsSkScene(0)) {
            return blockIdx;
        }
        bool preloadSK = blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                         Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                             Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum) - 1;
        if (preloadSK) {
            return blockIdx + usedCoreNum;
        }
        bool moveBackSK = blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                          Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                              Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum);
        return moveBackSK ? blockIdx - usedCoreNum : blockIdx;
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void ProcessAicBlock(Params const& params, BlockScheduler& bs, BlockMmad& blockMmad, TensorA& gmA,
                                           TensorB& gmB, TensorBias& gmBias, TensorC& gmC, int64_t blockIdx,
                                           int64_t totalMNBlockNumsInDP)
    {
        BlockShape singleCoreShape = bs.GetBlockShape(blockIdx);
        BlockShape singleCoreCoord = bs.GetBlockCoord(blockIdx);
        bool isSkBlock = bs.CheckIsSkScene(blockIdx);
        int64_t kSingleCore = isSkBlock ? params.schParams.singleCoreK : k_;
        int64_t alignedBaseN = Blaze::Gemm::CeilAlign(
            params.schParams.baseN, static_cast<int64_t>(AscendC::GetVecLen() / sizeof(WorkspaceType)));
        int64_t offsetWorkspace = (isSkBlock ? (blockIdx - totalMNBlockNumsInDP) : 0) * params.schParams.baseM *
                                  alignedBaseN;
        auto gmWorkSpace = MakeWorkspaceTensor(singleCoreShape, offsetWorkspace);

        auto gmBlockA = gmA.slice(
            asc::te::make_coord(asc::te::get<MNK_M>(singleCoreCoord) * mL1_,
                                asc::te::get<MNK_K>(singleCoreCoord) * kSingleCore),
            asc::te::make_shape(asc::te::get<MNK_M>(singleCoreShape), asc::te::get<MNK_K>(singleCoreShape)));
        auto gmBlockB = gmB.slice(
            asc::te::make_coord(asc::te::get<MNK_K>(singleCoreCoord) * kSingleCore,
                                asc::te::get<MNK_N>(singleCoreCoord) * nL1_),
            asc::te::make_shape(asc::te::get<MNK_K>(singleCoreShape), asc::te::get<MNK_N>(singleCoreShape)));
        auto gmBlockC = gmC.slice(
            asc::te::make_coord(asc::te::get<MNK_M>(singleCoreCoord) * mL1_,
                                asc::te::get<MNK_N>(singleCoreCoord) * nL1_),
            asc::te::make_shape(asc::te::get<MNK_M>(singleCoreShape), asc::te::get<MNK_N>(singleCoreShape)));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, asc::te::get<MNK_N>(singleCoreCoord) * nL1_),
                                        asc::te::make_shape(1L, asc::te::get<MNK_N>(singleCoreShape)));

        blockMmad(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, gmWorkSpace, singleCoreShape,
                  asc::te::get<MNK_K>(singleCoreCoord), isSkBlock);
    }

    __aicore__ inline void ProcessOnAiv(Params const& params, BlockScheduler& bs)
    {
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        uint64_t lastLoopTotalCnt = static_cast<uint64_t>((mBlockNums_ * nBlockNums_) % usedCoreNum * skBlockNums_);
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            return;
        }

        BlockEpilogue epilogueOp;
        BlockShape l1Block = {params.schParams.baseM, params.schParams.baseN, params.schParams.kL1, 1};
        BlockShape l0Block = {params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 1};
        auto epParams = params.epilogueParams;
        epParams.windowLen = params.schParams.windowLen;
        epilogueOp.Init(epParams, params.problemShape, l1Block, {mBlockNums_, nBlockNums_, skBlockNums_, 1},
                        usedCoreNum, bs.CheckIsSkScene(0), l0Block);
        epilogueOp();
    }

    __aicore__ inline void Init(Params const& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        if ASCEND_IS_AIC {
            auto mmParams = params.blockMmadParams;
            aGmAddr_ = reinterpret_cast<__gm__ AType*>(mmParams.aGmAddr);
            bGmAddr_ = reinterpret_cast<__gm__ BType*>(mmParams.bGmAddr);
            cGmAddr_ = reinterpret_cast<__gm__ CType*>(mmParams.cGmAddr);
            workspaceGmAddr_ = reinterpret_cast<__gm__ WorkspaceType*>(params.epilogueParams.workspaceGmAddr);
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(mmParams.biasGmAddr);
            InitScale(params);
        }
    }

    __aicore__ inline void InitScale(Params const& params)
    {
        const auto scalarLayout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_stride(1L));
        if (params.epilogueParams.perTokenScaleGmAddr != nullptr) {
            const auto x1Scale = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ float*>(params.epilogueParams.perTokenScaleGmAddr)),
                scalarLayout);
            const auto x2Scale = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ float*>(params.epilogueParams.scaleGmAddr)),
                scalarLayout);
            const float dequantScale = x1Scale[0] * x2Scale[0];
            const uint32_t scaleBits = *reinterpret_cast<const uint32_t*>(&dequantScale);
            scaleScalar_ = static_cast<uint64_t>(scaleBits & DEQ_SCALE_MUL_MASK);
        } else if constexpr (AscendC::IsSameType<X2ScaleType, uint64_t>::value ||
                             AscendC::IsSameType<X2ScaleType, int64_t>::value) {
            const auto x2Scale = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ uint64_t*>(params.epilogueParams.scaleGmAddr)),
                scalarLayout);
            scaleScalar_ = x2Scale[0];
        } else if constexpr (AscendC::IsSameType<X2ScaleType, bfloat16_t>::value) {
            const auto x2Scale = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ uint16_t*>(params.epilogueParams.scaleGmAddr)),
                scalarLayout);
            const uint32_t scaleBits = static_cast<uint32_t>(x2Scale[0]) << BF16_SHIFT;
            scaleScalar_ = static_cast<uint64_t>(scaleBits & DEQ_SCALE_MUL_MASK);
        } else {
            const auto x2Scale = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(
                    reinterpret_cast<__gm__ float*>(params.epilogueParams.scaleGmAddr)),
                scalarLayout);
            const float x2ScaleValue = x2Scale[0];
            const uint32_t scaleBits = *reinterpret_cast<const uint32_t*>(&x2ScaleValue);
            scaleScalar_ = static_cast<uint64_t>(scaleBits & DEQ_SCALE_MUL_MASK);
        }
    }

    template <typename SingleCoreShape>
    __aicore__ inline auto MakeWorkspaceTensor(const SingleCoreShape& singleCoreShape, int64_t offsetWorkspace)
    {
        auto workspaceStrideColumn = Blaze::Gemm::CeilAlign(
            asc::te::get<MNK_N>(singleCoreShape), static_cast<int64_t>(AscendC::GetVecLen() / sizeof(WorkspaceType)));
        auto workspaceShape = asc::te::make_shape(
            asc::te::make_shape(asc::te::_1{}, asc::te::get<MNK_M>(singleCoreShape)),
            asc::te::make_shape(asc::te::_1{}, workspaceStrideColumn));
        auto workspaceStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, workspaceStrideColumn),
                                                    asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
        auto layoutWorkspace = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn,
                                                            asc::te::layout_trait_default<WorkspaceType>>(
            workspaceShape, workspaceStride);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(workspaceGmAddr_ + offsetWorkspace),
                                    layoutWorkspace);
    }

private:
    static constexpr uint8_t AIC_ONLY_SYNC_MODE = 0;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIC_ONLY_SYNC_FLAG = 7;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint32_t BF16_SHIFT = 16U;
    static constexpr uint32_t DEQ_SCALE_MUL_MASK = 0xFFFFE000U;

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ WorkspaceType* workspaceGmAddr_{nullptr};
    uint64_t scaleScalar_{0UL};

    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
