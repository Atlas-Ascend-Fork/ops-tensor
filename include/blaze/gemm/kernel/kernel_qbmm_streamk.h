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
 * \file kernel_qbmm_streamk.h
 * \brief GemmUniversal specialization for QBMM MX StreamK.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelQbmmMultiBlockStreamK, typename BlockMmad_::DispatchPolicy::ScheduleType> &&
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    struct QBMMStreamKParams {
        uint32_t scaleKL1;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape_ problemShape;
        typename BlockMmad_::Params mmadParams;
        typename BlockEpilogue_::Params epilogueParams;
        typename BlockScheduler_::Params schParams;
        QBMMStreamKParams qbmmParams;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        Init(params);
        if (!IsValidProblem()) {
            return;
        }

        BlockScheduler_ bs(problemShape_, params.schParams);
        if ASCEND_IS_AIC {
            ProcessAic(params, bs);
        }

        if ASCEND_IS_AIV {
            ProcessAiv(params, bs);
        }
    }

private:
    using ProblemShape = ProblemShape_;
    using BlockMmadOp = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using BlockMmadParams = typename BlockMmadOp::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using AType = typename BlockMmadOp::AType;
    using BType = typename BlockMmadOp::BType;
    using CType = typename BlockMmadOp::CType;
    using BiasType = typename BlockMmadOp::BiasType;
    using ScaleType = MxScaleType<AType>;
    using LayoutA = typename BlockMmadOp::LayoutA;
    using LayoutB = typename BlockMmadOp::LayoutB;
    using LayoutC = typename BlockMmadOp::LayoutC;

    static_assert((IsFp8<AType>() && IsFp8<BType>()) || (IsFp4<AType>() && IsFp4<BType>()) ||
                      (IsHif4<AType>() && IsHif4<BType>()),
                  "QBMM MX StreamK: AType/BType must each be fp8_e4m3fn_t/fp8_e5m2_t, or each be "
                  "fp4x2_e2m1_t/fp4x2_e1m2_t/hifloat4x2_t.");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
                  "QBMM MX StreamK: BlockMmad::CType must be half/bfloat16_t/float.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::OutType, CType>,
                  "QBMM MX StreamK: BlockEpilogue::OutType must match BlockMmad::CType.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::WorkspaceType, float>,
                  "QBMM MX StreamK: BlockEpilogue::WorkspaceType must be float.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX StreamK: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QBMM MX StreamK: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");
    // Preserve the ND/DN C Tensor layouts supported by the L0C copy path.
    // A supplied epilogue must use the same output layout.
    static_assert(AscendC::Std::is_one_of_v<LayoutC, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX StreamK: LayoutC must be nd_ext_layout_ptn/dn_ext_layout_ptn.");

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_HIF4 = IsHif4<AType>();
    static constexpr int32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<GetScaleC0<AType>()>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<GetScaleC0<AType>()>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<GetScaleC0<BType>()>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<GetScaleC0<BType>()>>>;

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = params.problemShape;
        usedCoreNums_ = params.schParams.usedCoreNum;
        blockMmadParams_ = params.mmadParams;

        int64_t m = asc::te::get<MNK_M>(problemShape_);
        int64_t n = asc::te::get<MNK_N>(problemShape_);
        int64_t k = asc::te::get<MNK_K>(problemShape_);

        mBlockNums_ = Blaze::Gemm::CeilDiv(m, params.schParams.baseM);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n, params.schParams.baseN);
        skBlockNums_ = Blaze::Gemm::CeilDiv(k, params.schParams.singleCoreK);

        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr);
        scaleAGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(blockMmadParams_.scaleAGmAddr);
        scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(blockMmadParams_.scaleBGmAddr);
        workspaceGmAddr_ = reinterpret_cast<__gm__ float*>(params.epilogueParams.workspaceGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr);
    }

    __aicore__ inline bool IsValidProblem() const
    {
        return usedCoreNums_ > 0 && asc::te::get<MNK_B>(problemShape_) == 1;
    }

    __aicore__ inline int64_t GetActualBlockIdx(BlockScheduler& bs, int64_t blockIdx, int64_t blockNums,
                                                int64_t tailSKTotalBlockNums) const
    {
        if (bs.CheckIsSkScene(0)) {
            return blockIdx;
        }
        bool preloadSK = blockIdx % usedCoreNums_ < tailSKTotalBlockNums &&
                         CeilDiv(blockIdx + 1, usedCoreNums_) == CeilDiv(blockNums, usedCoreNums_) - 1;
        if (preloadSK) {
            return blockIdx + usedCoreNums_;
        }
        bool moveBackSK = blockIdx % usedCoreNums_ < tailSKTotalBlockNums &&
                          CeilDiv(blockIdx + 1, usedCoreNums_) == CeilDiv(blockNums, usedCoreNums_);
        return moveBackSK ? blockIdx - usedCoreNums_ : blockIdx;
    }

    __aicore__ inline void SignalAicFinish() const
    {
        AscendC::CrossCoreSetFlag<AIC_ONLY_SYNC_MODE, PIPE_FIX>(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_ONLY_SYNC_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }

    template <typename BlockShape>
    __aicore__ inline auto MakeWorkspaceTensor(BlockShape const& singleCoreShape, int64_t offsetWorkspace) const
    {
        auto workspaceStrideColumn0 = BlockEpilogue::DispatchPolicy::FIXP_OPTI == MatMulL0C2Out::ND_FIXPIPE_1_2 ?
                                          CeilAlign(static_cast<uint64_t>(asc::te::get<MNK_N>(singleCoreShape)),
                                                    static_cast<uint64_t>(BLOCK_BYTE_SIZE)) :
                                          asc::te::get<MNK_N>(singleCoreShape);
        auto workspaceShape = asc::te::make_shape(
            asc::te::make_shape(asc::te::_1{}, asc::te::get<MNK_M>(singleCoreShape)),
            asc::te::make_shape(asc::te::_1{}, asc::te::get<MNK_N>(singleCoreShape)));
        auto workspaceStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, workspaceStrideColumn0),
                                                    asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
        auto layoutWorkspace = asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn,
                                                            asc::te::layout_trait_default<float>>(workspaceShape,
                                                                                                  workspaceStride);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(workspaceGmAddr_ + offsetWorkspace),
                                    layoutWorkspace);
    }

    template <typename TensorB, typename BlockShape>
    __aicore__ inline void SetBL2Cache(Params const& params, BlockShape const& currentBasicBlock, TensorB& gmB) const
    {
        // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
        // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
        constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
        const bool isCurrentNAligned = TRANS_B || (asc::te::get<MNK_N>(currentBasicBlock) & cacheLineAlignMask) == 0UL;
        const bool disableWeightL2 = params.qbmmParams.bMustHitL2 == 0U &&
                                     asc::te::get<MNK_M>(currentBasicBlock) >= asc::te::get<MNK_M>(problemShape_) &&
                                     isCurrentNAligned;
        gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    }

    template <typename TensorA, typename TensorScaleA, typename TensorB, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void ProcessAicBlock(Params const& params, BlockScheduler& bs, BlockMmadOp& blockMmadOp,
                                           TensorA& gmA, TensorScaleA& gmScaleA, TensorB& gmB, TensorScaleB& gmScaleB,
                                           TensorBias& gmBias, TensorC& gmC, int64_t tmpBlockIdx, int64_t mL1,
                                           int64_t nL1, ProblemShape const& l0BlockShape, uint64_t scaleKL1)
    {
        auto singleCoreShape = bs.GetBlockShape(tmpBlockIdx);
        auto singleCoreCoord = bs.GetBlockCoord(tmpBlockIdx);
        SetBL2Cache(params, singleCoreShape, gmB);
        bool isSkScene = bs.CheckIsSkScene(tmpBlockIdx);
        int64_t kSingleCore = isSkScene ? params.schParams.singleCoreK : asc::te::get<MNK_K>(problemShape_);
        int64_t alignedBaseN = Blaze::Gemm::CeilAlign(params.schParams.baseN, static_cast<int64_t>(BLOCK_BYTE_SIZE));
        int64_t offsetWorkspace = (((tmpBlockIdx % usedCoreNums_) / skBlockNums_) * skBlockNums_ +
                                   asc::te::get<MNK_K>(singleCoreCoord)) *
                                  params.schParams.baseM * alignedBaseN;
        auto gmWorkSpace = MakeWorkspaceTensor(singleCoreShape, offsetWorkspace);
        auto scaleKL1LenSingleCore = static_cast<int64_t>(MxScaleKLen<AType>(static_cast<uint64_t>(kSingleCore)));
        int64_t scaleAPos = asc::te::get<MNK_M>(singleCoreCoord) * mL1;
        if constexpr (IS_HIF4 && TRANS_A) {
            scaleAPos = scaleAPos >> 1;
        }
        int64_t scaleBPos = asc::te::get<MNK_N>(singleCoreCoord) * nL1;
        if constexpr (IS_HIF4 && !TRANS_B) {
            scaleBPos = scaleBPos >> 1;
        }
        auto gmBlockA = gmA.slice(
            asc::te::make_coord(asc::te::get<MNK_M>(singleCoreCoord) * mL1,
                                asc::te::get<MNK_K>(singleCoreCoord) * kSingleCore),
            asc::te::make_shape(asc::te::get<MNK_M>(singleCoreShape), asc::te::get<MNK_K>(singleCoreShape)));
        auto gmBlockScaleA = gmScaleA.slice(
            asc::te::make_coord(scaleAPos, asc::te::get<MNK_K>(singleCoreCoord) * scaleKL1LenSingleCore),
            asc::te::make_shape(asc::te::get<MNK_M>(singleCoreShape), scaleKL1LenSingleCore));
        auto gmBlockB = gmB.slice(
            asc::te::make_coord(asc::te::get<MNK_K>(singleCoreCoord) * kSingleCore,
                                asc::te::get<MNK_N>(singleCoreCoord) * nL1),
            asc::te::make_shape(asc::te::get<MNK_K>(singleCoreShape), asc::te::get<MNK_N>(singleCoreShape)));
        auto gmBlockScaleB = gmScaleB.slice(
            asc::te::make_coord(asc::te::get<MNK_K>(singleCoreCoord) * scaleKL1LenSingleCore, scaleBPos),
            asc::te::make_shape(scaleKL1LenSingleCore, asc::te::get<MNK_N>(singleCoreShape)));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, asc::te::get<MNK_N>(singleCoreCoord) * nL1),
                                        asc::te::make_shape(1L, asc::te::get<MNK_N>(singleCoreShape)));
        auto gmBlockC = gmC.slice(
            asc::te::make_coord(asc::te::get<MNK_M>(singleCoreCoord) * mL1, asc::te::get<MNK_N>(singleCoreCoord) * nL1),
            asc::te::make_shape(asc::te::get<MNK_M>(singleCoreShape), asc::te::get<MNK_N>(singleCoreShape)));
        bool isBiasForBlock = biasGmAddr_ != nullptr && asc::te::get<MNK_K>(singleCoreCoord) == 0;
        blockMmadOp.Init(singleCoreShape, l0BlockShape,
                         {static_cast<uint64_t>(params.schParams.kL1), scaleKL1, DOUBLE_BUFFER_COUNT}, isBiasForBlock,
                         params.qbmmParams.dbL0C > 1);
        if (isSkScene) {
            blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmWorkSpace, singleCoreShape);
        } else {
            blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleCoreShape);
        }
    }

    __aicore__ inline void ProcessAicBlocks(Params const& params, BlockScheduler& bs, int64_t curBlockIdx)
    {
        int64_t m = asc::te::get<MNK_M>(problemShape_);
        int64_t n = asc::te::get<MNK_N>(problemShape_);
        int64_t k = asc::te::get<MNK_K>(problemShape_);

        int64_t mnBlockNums = mBlockNums_ * nBlockNums_;
        int64_t tailSKTotalBlockNums = static_cast<int64_t>((mnBlockNums % usedCoreNums_) * skBlockNums_);
        int64_t blockNums = bs.GetBlockNums();
        auto scaleKLen = MxScaleKLen<AType>(k);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), MakeLayoutA{}(m, k));
        auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_),
                                             MakeLayoutScaleA{}(m, scaleKLen));
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), MakeLayoutB{}(k, n));
        auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_),
                                             MakeLayoutScaleB{}(scaleKLen, n));
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_),
                                           asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n));
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), MakeLayoutC{}(m, n));
        BlockMmadOp blockMmadOp;
        auto scaleKL1 = static_cast<uint64_t>(params.qbmmParams.scaleKL1);
        ProblemShape l0BlockShape = {params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 1};
        for (int64_t blockIdx = curBlockIdx; blockIdx < blockNums; blockIdx += usedCoreNums_) {
            int64_t tmpBlockIdx = GetActualBlockIdx(bs, blockIdx, blockNums, tailSKTotalBlockNums);
            ProcessAicBlock(params, bs, blockMmadOp, gmA, gmScaleA, gmB, gmScaleB, gmBias, gmC, tmpBlockIdx,
                            params.schParams.baseM, params.schParams.baseN, l0BlockShape, scaleKL1);
        }
    }

    __aicore__ inline void ProcessAic(Params const& params, BlockScheduler& bs)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx < usedCoreNums_) {
            ProcessAicBlocks(params, bs, curBlockIdx);
        }
        SignalAicFinish();
    }

    __aicore__ inline void ProcessAiv(Params const& params, BlockScheduler& bs)
    {
        int64_t mnBlockNums = mBlockNums_ * nBlockNums_;
        uint64_t lastLoopTotalCnt = (mnBlockNums % usedCoreNums_) * skBlockNums_;
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            return;
        }

        ProblemShape l1BlockShape = {params.schParams.baseM, params.schParams.baseN, params.schParams.kL1, 1};
        ProblemShape l0BlockShape = {params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 1};
        BlockEpilogue epilogueOp;
        auto epParams = params.epilogueParams;
        epParams.windowLen = params.schParams.windowLen;
        epilogueOp.Init(epParams, problemShape_, l1BlockShape, {mBlockNums_, nBlockNums_, skBlockNums_, 1},
                        usedCoreNums_, bs.CheckIsSkScene(0), l0BlockShape);
        epilogueOp();
    }

    static constexpr uint8_t AIC_ONLY_SYNC_MODE = 0;
    static constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
    static constexpr uint16_t AIC_ONLY_SYNC_FLAG = 7;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint16_t FLAG_ID_MAX = 16;

    ProblemShape problemShape_{};
    int64_t usedCoreNums_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
    BlockMmadParams blockMmadParams_{};

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ ScaleType* scaleAGmAddr_{nullptr};
    __gm__ ScaleType* scaleBGmAddr_{nullptr};
    __gm__ float* workspaceGmAddr_{nullptr};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
