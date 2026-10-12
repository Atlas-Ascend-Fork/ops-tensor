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
 * \file kernel_qgmm_mx_mix_finalize_routing.h
 * \brief TensorAPI kernel schedule for GroupedMatmulFinalizeRouting MX.
 */

#pragma once

#include <cstdint>

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
namespace {
constexpr uint32_t GMM_FR_UB_DOUBLE_BUFFER_LEN = 1024 * 8;
constexpr uint32_t GMM_FR_ONE_CORE_ALIGN_LEN = 512;
constexpr uint32_t GMM_FR_ONE_CORE_ALIGN_LEN_SMALL = 32;
} // namespace

static constexpr AscendC::Reg::CastTrait CAST_GMM_FR_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

__aicore__ inline uint16_t FloatToBf16Bits(float value)
{
    union {
        float f;
        uint32_t u;
    } converter{value};
    uint32_t roundingBias = 0x7FFFU + ((converter.u >> 16) & 1U);
    return static_cast<uint16_t>((converter.u + roundingBias) >> 16);
}

#define GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS template <typename DataTypeOut_, typename DataTypeIn_>
#define GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS DataTypeOut_, DataTypeIn_

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
class BlockPrologueFinalizeRouting {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    static_assert(AscendC::Std::is_one_of_v<DataTypeOut, float, bfloat16_t>,
                  "GMM finalize routing prologue only supports float/bfloat16_t output.");
    static_assert(AscendC::Std::is_same_v<DataTypeIn, bfloat16_t>,
                  "GMM finalize routing prologue only supports bfloat16_t shared input.");

    struct Params {
        GM_ADDR residualGm{nullptr};
        GM_ADDR yGmAddr{nullptr};
        uint32_t sharedInputOffset{0};
        uint32_t sharedInputLen{0};
        int32_t n{0};
        uint32_t batch{0};
        float residualScale{1.0F};
    };

    __aicore__ inline BlockPrologueFinalizeRouting() {}
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()();

private:
    __aicore__ inline void InitAllLocalTensor();
    __aicore__ inline void Compute(uint64_t singleCount, uint64_t outOffset, uint64_t baseOffset, uint64_t curCount);
    __aicore__ inline void CopyInShareInput(uint64_t residualUbOffset, uint64_t offset, uint64_t size);
    __aicore__ inline void CopyOutShareInput(uint64_t yUbOffset, uint64_t offset, uint64_t size);
    static __simd_vf__ inline void VfDoSharedCastAndMuls(__ubuf__ DataTypeOut* dstPtr,
                                                         __ubuf__ DataTypeIn* residualLocalUbAddr, float sharedWeight,
                                                         uint16_t sharedWeightBf16, uint16_t loopNum,
                                                         uint64_t curCount);

    const Params* params_{nullptr};
    uint64_t shareInputUbPingOffset_{0};
    uint64_t shareInputUbPongOffset_{0};
    uint64_t ubOutPingOffset_{0};
    uint64_t ubOutPongOffset_{0};
    uint32_t vectorCoreNum_{0};
};

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    InitAllLocalTensor();
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::CopyInShareInput(
    uint64_t residualUbOffset, uint64_t offset, uint64_t size)
{
    const auto layout = asc::te::make_layout(asc::te::make_shape(size), asc::te::make_stride(1L));
    const auto residualGm = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ DataTypeIn*>(params_->residualGm) +
                                                     offset),
        layout);
    const auto residualUb = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(residualUbOffset), layout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_gm_to_ub{}), residualUb, residualGm);
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::CopyOutShareInput(
    uint64_t yUbOffset, uint64_t offset, uint64_t size)
{
    if (size == 0) {
        return;
    }
    const auto layout = asc::te::make_layout(asc::te::make_shape(size), asc::te::make_stride(1L));
    const auto yGm = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ DataTypeOut*>(params_->yGmAddr) + offset),
        layout);
    const auto yUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeOut>(yUbOffset), layout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), yGm, yUb);
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::InitAllLocalTensor()
{
    shareInputUbPingOffset_ = 0UL;
    shareInputUbPongOffset_ = shareInputUbPingOffset_ + GMM_FR_UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeIn);
    ubOutPingOffset_ = shareInputUbPongOffset_ + GMM_FR_UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeIn);
    ubOutPongOffset_ = ubOutPingOffset_ + GMM_FR_UB_DOUBLE_BUFFER_LEN * sizeof(DataTypeOut);
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__simd_vf__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::VfDoSharedCastAndMuls(
    __ubuf__ DataTypeOut* dstPtr, __ubuf__ DataTypeIn* residualLocalUbAddr, float sharedWeight,
    uint16_t sharedWeightBf16, uint16_t loopNum, uint64_t curCount)
{
    constexpr uint32_t vectorRegLen = AscendC::IsSameType<DataTypeOut, bfloat16_t>::value ?
                                          AscendC::VECTOR_REG_WIDTH / sizeof(float) :
                                          AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeOut);
    uint32_t elementNum = static_cast<uint32_t>(curCount);
    AscendC::Reg::MaskReg mask;
    uint16_t oneRepeatSize = vectorRegLen;
    if constexpr (AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(elementNum);
            AscendC::Reg::RegTensor<DataTypeIn> vSrcReg;
            AscendC::Reg::RegTensor<uint16_t> vSharedWeightReg;
            AscendC::Reg::RegTensor<bfloat16_t> vDstReg;
            AscendC::Reg::Duplicate<uint16_t, AscendC::Reg::MaskMergeMode::ZEROING>(vSharedWeightReg, sharedWeightBf16,
                                                                                    mask);
            AscendC::Reg::LoadAlign<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vSrcReg, residualLocalUbAddr + i * oneRepeatSize);
            AscendC::Reg::Mul<bfloat16_t, AscendC::Reg::MaskMergeMode::ZEROING>(
                vDstReg, vSrcReg, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(vSharedWeightReg), mask);
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstPtr + i * oneRepeatSize,
                                                                                         vDstReg, mask);
        }
    } else {
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<DataTypeOut>(elementNum);
            AscendC::Reg::RegTensor<DataTypeIn> vSrcReg;
            AscendC::Reg::RegTensor<DataTypeOut> vRegCast;
            AscendC::Reg::RegTensor<DataTypeOut> vDstReg;
            AscendC::Reg::LoadAlign<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                vSrcReg, residualLocalUbAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<DataTypeOut, DataTypeIn, CAST_GMM_FR_BF16_TO_FP32>(vRegCast, vSrcReg, mask);
            AscendC::Reg::Muls(vDstReg, vRegCast, sharedWeight, mask);
            AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstPtr + i * oneRepeatSize,
                                                                                          vDstReg, mask);
        }
    }
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::Compute(uint64_t singleCount,
                                                                                                uint64_t outOffset,
                                                                                                uint64_t baseOffset,
                                                                                                uint64_t curCount)
{
    uint32_t pingPongId = 0;
    for (uint32_t offset = 0; offset < singleCount; offset += curCount) {
        if (unlikely(offset + curCount > singleCount)) {
            curCount = singleCount - offset;
        }
        const uint64_t residualUbOffset = pingPongId == 0 ? shareInputUbPingOffset_ : shareInputUbPongOffset_;
        const Gemm::BufferSlot bufferSlot{0UL, static_cast<uint8_t>(pingPongId)};
        {
            auto mte2Lock = bufferSlot.LockMte2();
            CopyInShareInput(residualUbOffset, baseOffset + offset, curCount);
        }
        const uint64_t yUbOffset = pingPongId == 0 ? ubOutPingOffset_ : ubOutPongOffset_;
        auto residualLocalUbAddr = reinterpret_cast<__ubuf__ DataTypeIn*>(residualUbOffset);
        auto outUbAddr = reinterpret_cast<__ubuf__ DataTypeOut*>(yUbOffset);
        constexpr uint32_t vectorRegLen = AscendC::IsSameType<DataTypeOut, bfloat16_t>::value ?
                                              AscendC::VECTOR_REG_WIDTH / sizeof(float) :
                                              AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeOut);
        uint16_t loopNum = CeilDiv(curCount, static_cast<uint64_t>(vectorRegLen));
        uint16_t sharedWeightBf16 = FloatToBf16Bits(params_->residualScale);
        {
            auto vectorLock = bufferSlot.LockV();
            asc_vf_call<VfDoSharedCastAndMuls>(outUbAddr, residualLocalUbAddr, params_->residualScale, sharedWeightBf16,
                                               loopNum, curCount);
        }
        {
            auto mte3Lock = bufferSlot.LockMte3();
            CopyOutShareInput(yUbOffset, outOffset + offset, curCount);
        }
        pingPongId = (pingPongId + 1) & 1;
    }
}

GMM_FR_PROLOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockPrologueFinalizeRouting<GMM_FR_PROLOGUE_FUNC_LOCAL_PARAMS>::operator()()
{
    if ASCEND_IS_AIC {
        return;
    }
    vectorCoreNum_ = static_cast<uint32_t>(AscendC::GetBlockNum() * AscendC::GetTaskRation());
    if (AscendC::GetBlockIdx() >= vectorCoreNum_) {
        return;
    }
    const uint32_t sharedInputOffset = params_->sharedInputOffset;
    const uint32_t sharedInputLen = params_->sharedInputLen;
    if (sharedInputLen == 0) {
        return;
    }

    const uint64_t firstZeroSize = static_cast<uint64_t>(params_->n) * sharedInputOffset;
    const uint64_t totalOutput = static_cast<uint64_t>(params_->n) * sharedInputLen;
    uint64_t singleCount = CeilDiv(totalOutput, static_cast<uint64_t>(vectorCoreNum_));
    const uint32_t alignLen = totalOutput > GMM_FR_ONE_CORE_ALIGN_LEN ? GMM_FR_ONE_CORE_ALIGN_LEN :
                                                                        GMM_FR_ONE_CORE_ALIGN_LEN_SMALL;
    singleCount = CeilDiv(singleCount, static_cast<uint64_t>(alignLen)) * alignLen;
    const uint64_t baseOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * singleCount;
    if (baseOffset >= totalOutput) {
        return;
    }
    if (baseOffset + singleCount > totalOutput) {
        singleCount = totalOutput - baseOffset;
    }
    const uint64_t outOffset = firstZeroSize + baseOffset;
    Compute(singleCount, outOffset, baseOffset, GMM_FR_UB_DOUBLE_BUFFER_LEN);
}

template <class ProblemShape_, class BlockMmad_, class BlockPrologue_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, AscendC::Std::tuple<BlockPrologue_, BlockEpilogue_>, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelQgmmMxMixFinalizeRouting, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockPrologue = BlockPrologue_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockEpilogue::DataTypeOut;
    using MmadCType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    static constexpr bool
        IS_WEIGHT_NZ_LAYOUT = AscendC::Std::is_one_of_v<LayoutB, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>;
    static_assert(((AscendC::Std::is_one_of_v<AType, fp8_e4m3fn_t, fp8_e5m2_t> &&
                    AscendC::Std::is_one_of_v<BType, fp8_e4m3fn_t, fp8_e5m2_t>) ||
                   (AscendC::Std::is_one_of_v<AType, fp4x2_e2m1_t, fp4x2_e1m2_t> &&
                    AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>)),
                  "GMM finalize routing MX requires AType/BType to both be MXFP8 or both be MXFP4.");
    static_assert(!IS_WEIGHT_NZ_LAYOUT ||
                      (AscendC::Std::is_same_v<AType, fp8_e4m3fn_t> && AscendC::Std::is_same_v<BType, fp8_e4m3fn_t>) ||
                      (AscendC::Std::is_same_v<AType, fp4x2_e2m1_t> && AscendC::Std::is_same_v<BType, fp4x2_e2m1_t>),
                  "GMM finalize routing MX WeightNZ only supports E4M3/E4M3 or E2M1/E2M1.");
    static_assert(AscendC::Std::is_one_of_v<CType, float, bfloat16_t>,
                  "GMM finalize routing MX only supports float/bfloat16_t output.");
    static_assert(AscendC::Std::is_same_v<MmadCType, float>, "GMM finalize routing MX requires float MMAD output.");
    static_assert(AscendC::Std::is_same_v<BiasType, bfloat16_t>,
                  "GMM finalize routing MX only supports bfloat16_t bias type.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "GMM finalize routing MX only supports ND LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GMM finalize routing MX only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(AscendC::Std::is_same_v<LayoutC, asc::te::nd_ext_layout_ptn>,
                  "GMM finalize routing MX only supports ND LayoutC.");
    static_assert(AscendC::Std::is_same_v<LayoutBias, asc::te::nd_ext_layout_ptn>,
                  "GMM finalize routing MX only supports ND LayoutBias.");
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockMmadInitParams = typename BlockMmad::MmadParams;
    using BlockPrologueParams = typename BlockPrologue::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using SchedulerBlockShape = typename BlockScheduler::BlockShape;
    using SchedulerBlockCoord = typename BlockScheduler::BlockCoord;
    using EpilogueBlockShape = typename BlockEpilogue::BlockShape;
    using EpilogueBlockCoord = typename BlockEpilogue::BlockCoord;

    struct GMMTiling {
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
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t hasBias;
        uint8_t dbL0C;
        uint8_t groupListType;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams blockMmadParams;
        BlockPrologueParams prologueParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GMMTiling gmmParams;
    };

    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint64_t INPUT_C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
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
        const auto groupListLayout = asc::te::make_layout(asc::te::make_shape(static_cast<int64_t>(groupNum_)),
                                                          asc::te::make_stride(1L));
        const auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
            groupListLayout);
        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            const int64_t groupM = GetSplitValueFromGroupList(gmGroupList, groupIdx);
            if (groupM <= 0) {
                continue;
            }
            problemShape_ = ProblemShape{groupM, n_, k_, 0};
            ProcessSingleGroup(params, scheduler, groupIdx, groupM);
        }
        End();
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.blockMmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.blockMmadParams.bGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadParams.scaleBGmAddr);
        biasBasePtr_ = reinterpret_cast<__gm__ BiasType*>(params.blockMmadParams.biasGmAddr);
        hasBias_ = gmmParams.hasBias != 0;
        n_ = asc::te::get<MNK_N>(params.problemShape);
        k_ = asc::te::get<MNK_K>(params.problemShape);
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        isVecSetSyncCom_ = false;
        problemShape_ = params.problemShape;

        if ASCEND_IS_AIV {
            prologueOp_.Init(params.prologueParams);
            prologueOp_();
        }
        if ASCEND_IS_AIC {
            const BlockShape l0Shape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                     static_cast<int64_t>(gmmParams.baseK), 0};
            const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                    static_cast<uint64_t>(gmmParams.scaleKAL1)};
            const BlockMmadInitParams blockMmadParams{l0Shape, l1Params, hasBias_,
                                                      gmmParams.dbL0C == DOUBLE_BUFFER_COUNT};
            blockMmad_.Init(params.problemShape, blockMmadParams);
        }
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& gmGroupList, uint32_t groupIdx)
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

    __aicore__ inline int64_t GetBOffset(uint32_t groupIdx) const
    {
        uint64_t singleGroupBSize = static_cast<uint64_t>(n_) * static_cast<uint64_t>(k_);
        if constexpr (IsWeightNz<LayoutB>::value) {
            // ZN weight storage swaps the K/N fractal axes relative to NZ.
            if constexpr (TRANS_B) {
                singleGroupBSize = CeilAlign(static_cast<uint64_t>(n_), static_cast<uint64_t>(BLOCK_CUBE)) *
                                   CeilAlign(static_cast<uint64_t>(k_), INPUT_C0_SIZE);
            } else {
                singleGroupBSize = CeilAlign(static_cast<uint64_t>(n_), INPUT_C0_SIZE) *
                                   CeilAlign(static_cast<uint64_t>(k_), static_cast<uint64_t>(BLOCK_CUBE));
            }
        }
        if constexpr (IsFp4<BType>()) {
            singleGroupBSize >>= 1;
        }
        return static_cast<int64_t>(static_cast<uint64_t>(groupIdx) * singleGroupBSize);
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockScheduler& scheduler, uint32_t groupIdx,
                                              int64_t groupM)
    {
        const int64_t mPrefixOffset = preOffset_ - groupM;
        BaseMBalance(scheduler, groupM, params.gmmParams.baseM);
        scheduler.UpdateNextProblem(SchedulerProblemShape{groupM, n_, k_, 0});
        epilogueOp_.UpdateNextProblem(typename BlockEpilogue::ProblemShape{groupM, n_, k_});
        if ASCEND_IS_AIC {
            blockMmad_.UpdateParamsForNextProblem(problemShape_);
        }
        if ASCEND_IS_AIV {
            const EpilogueBlockCoord baseOffset{0, 0, 0, 0, mPrefixOffset, mPrefixOffset};
            epilogueOp_.UpdateGlobalAddr(baseOffset);
        }

        const int64_t scaleK = GetScaleK(k_);
        const int64_t aOffset = IsFp4<AType>() ? ((mPrefixOffset * k_) >> 1) : (mPrefixOffset * k_);
        const int64_t bOffset = GetBOffset(groupIdx);
        const int64_t scaleAOffset = mPrefixOffset * scaleK;
        const int64_t scaleBOffset = static_cast<int64_t>(groupIdx) * n_ * scaleK;
        const int64_t biasOffset = static_cast<int64_t>(groupIdx) * n_;

        auto layoutA = MakeLayoutA{}(groupM, k_);
        auto layoutScaleA = MakeLayoutScaleA{}(groupM, scaleK);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, n_);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + aOffset), layoutA);
        auto gmScaleA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleABasePtr_ + scaleAOffset), layoutScaleA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bBasePtr_ + bOffset), layoutB);
        auto gmScaleB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleBBasePtr_ + scaleBOffset), layoutScaleB);
        __gm__ BiasType* biasPtr = hasBias_ ? (biasBasePtr_ + biasOffset) :
                                              reinterpret_cast<__gm__ BiasType*>(bBasePtr_);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);

        SchedulerBlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        do {
            ProcessSingleBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, scheduler, blockCoord);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias>
    __aicore__ inline void ProcessSingleBlock(const TensorA& gmA, const TensorB& gmB, const TensorScaleA& gmScaleA,
                                              const TensorScaleB& gmScaleB, const TensorBias& gmBias,
                                              BlockScheduler& scheduler, const SchedulerBlockCoord& blockCoord)
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
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)),
                                                asc::te::make_shape(blockM, GetScaleK(k_)));
            auto gmBlockB = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                      asc::te::make_shape(k_, blockN));
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                                asc::te::make_shape(GetScaleK(k_), blockN));
            auto ubC = epilogueOp_.GetL0c2UbTensor(blockM, blockN);
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                            asc::te::make_shape(static_cast<int64_t>(1), blockN));
            blockMmad_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, ubC, blockShape);
            Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
        }
        isVecSetSyncCom_ = true;

        if ASCEND_IS_AIV {
            Sync::WaitForCube();
            const EpilogueBlockShape epilogueShape{blockM, blockN, 0, 0};
            const EpilogueBlockCoord epilogueOffset{nPos, 0, 0, 0, mPos, mPos};
            epilogueOp_(epilogueShape, epilogueOffset);
            Sync::NotifyCube();
        }
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    BlockMmad blockMmad_;
    BlockPrologue prologueOp_;
    BlockEpilogue epilogueOp_;
    ProblemShape problemShape_{};
    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBBasePtr_{nullptr};
    __gm__ BiasType* biasBasePtr_{nullptr};
    bool hasBias_{false};
    int64_t preOffset_{0};
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
