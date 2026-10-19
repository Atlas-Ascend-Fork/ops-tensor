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
 * \file block_epilogue_finalize_routing.h
 * \brief
 */

#ifndef BLAZE_BLOCK_EPILOGUE_FINALIZE_ROUTING_H
#define BLAZE_BLOCK_EPILOGUE_FINALIZE_ROUTING_H

#include <cstdint>

#if defined(__NPU_ARCH__)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

static constexpr AscendC::Reg::CastTrait CAST_FR_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait CAST_FR_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

#define BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeLogit_, typename RowIndexType_>
#define BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS \
    DataTypeOut_, DataTypeIn_, DataTypeLogit_, RowIndexType_

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
class BlockEpilogueFinalizeRouting {
private:
    static constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64L;
    static constexpr uint32_t L0C_MAX_M = 128;
    static constexpr uint32_t L0C_MAX_N = 256;
    static constexpr uint32_t BLOCK_BYTES = 256;
    static constexpr uint32_t VEC_MAX_M = 32;
    static constexpr uint32_t UB_TO_GM_ALIGN_BYTES = 32;
    static constexpr uint32_t MAX_SINGLE_MN = L0C_MAX_M * L0C_MAX_N;
    static constexpr uint32_t HALF_DB_MAX_SINGLE_MN = VEC_MAX_M * L0C_MAX_N;
    static constexpr uint32_t Y_IDX = 0;
    static constexpr uint32_t LOGIT_INDEX = 4;
    static constexpr uint64_t MAX_OUTPUT_M_UB = VEC_MAX_M;

public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeLogit = DataTypeLogit_;
    using RowIndexType = RowIndexType_;
    static_assert(AscendC::Std::is_one_of_v<DataTypeOut, float, bfloat16_t>,
                  "GMM finalize routing epilogue only supports float/bfloat16_t output.");
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float>,
                  "GMM finalize routing epilogue only supports float MMAD output.");
    static_assert(AscendC::Std::is_one_of_v<DataTypeLogit, float, bfloat16_t>,
                  "GMM finalize routing epilogue only supports float/bfloat16_t logit.");
    static_assert(AscendC::Std::is_one_of_v<RowIndexType, int32_t, int64_t>,
                  "GMM finalize routing epilogue only supports int32_t/int64_t row index.");
    static_assert((AscendC::Std::is_same_v<DataTypeOut, float> && AscendC::Std::is_same_v<RowIndexType, int64_t>) ||
                      (AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> &&
                       AscendC::Std::is_same_v<RowIndexType, int32_t>),
                  "Float output requires int64_t row index; bfloat16_t output requires int32_t row index.");

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR logitGmAddr{nullptr};
        GM_ADDR rowIndexGmAddr{nullptr};
        int32_t baseM{256};
        int32_t baseN{256};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // blk_m, blk_n, blk_k, _
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t,
                                      int64_t>;                     // y, _, _, _, logit, rowIndex
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>; // m, n, k

    __aicore__ inline BlockEpilogueFinalizeRouting() {}
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols);
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);

private:
    __aicore__ inline void CopyInLogit(uint32_t curBaseM, uint64_t offsetM,
                                       AscendC::LocalTensor<DataTypeLogit> logitUb);
    static __simd_vf__ inline void VfDoLogitMuls(uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit,
                                                 uint16_t repeatTimesRe, uint64_t singleN, uint64_t l0cAlignN,
                                                 uint64_t alignN, uint64_t vectorLength,
                                                 __ubuf__ DataTypeOut* outUbAddr, __ubuf__ DataTypeIn* l0cOutUbAddr,
                                                 __ubuf__ DataTypeLogit* logitUbAddr);
    __aicore__ inline void VectorAtomicProcess(uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM,
                                               uint64_t yOffset, uint64_t alignN,
                                               AscendC::LocalTensor<DataTypeOut> yLocal);

    // GM ADDR
    AscendC::GlobalTensor<DataTypeLogit> logitGlobal_;
    AscendC::GlobalTensor<RowIndexType> rowIndexGlobal_;
    AscendC::GlobalTensor<DataTypeOut> yGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUb_;
    AscendC::LocalTensor<DataTypeLogit> logitUbPing_;
    AscendC::LocalTensor<DataTypeLogit> logitUbPong_;
    AscendC::LocalTensor<DataTypeOut> outUbPing_;
    AscendC::LocalTensor<DataTypeOut> outUbPong_;

    const Params* params_{nullptr};
    int64_t n_{0};
    uint32_t subBlockIdx_{0};
    uint16_t yCrossPingPongId_{0};
    uint16_t logitCrossPingPongId_{0};
};

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params_->yGmAddr));
    l0cOutUb_ = AscendC::LocalTensor<DataTypeIn>(AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN);
    uint32_t afterFirstIn = MAX_SINGLE_MN * sizeof(DataTypeIn);
    logitUbPing_ = AscendC::LocalTensor<DataTypeLogit>(AscendC::TPosition::VECIN, afterFirstIn, BLOCK_BYTES);
    uint32_t afterSecondIn = afterFirstIn + BLOCK_BYTES * sizeof(DataTypeLogit);
    logitUbPong_ = AscendC::LocalTensor<DataTypeLogit>(AscendC::TPosition::VECIN, afterSecondIn, BLOCK_BYTES);
    uint32_t afterLogit = afterSecondIn + BLOCK_BYTES * sizeof(DataTypeLogit);
    outUbPing_ = AscendC::LocalTensor<DataTypeOut>(AscendC::TPosition::VECOUT, afterLogit, HALF_DB_MAX_SINGLE_MN);
    outUbPong_ = AscendC::LocalTensor<DataTypeOut>(
        AscendC::TPosition::VECOUT, afterLogit + HALF_DB_MAX_SINGLE_MN * sizeof(DataTypeOut), HALF_DB_MAX_SINGLE_MN);
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline auto
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueFinalizeRouting<
    BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor(int64_t rows, int64_t cols)
{
    constexpr uint64_t outputC0Size = asc::te::c0_element<DataTypeIn>;
    constexpr uint64_t splitMAlign = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    const uint64_t copyRows = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(rows), splitMAlign);
    const uint64_t copyCols = Blaze::Gemm::Align32(static_cast<uint64_t>(cols));
    const auto layoutOutUb = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<outputC0Size>>(
        copyRows, copyCols);
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layoutOutUb);
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<
    BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateNextProblem(const ProblemShape& problemShape)
{
    n_ = asc::te::get<Blaze::Gemm::MNK_N>(problemShape);
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueFinalizeRouting<
    BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        logitGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeLogit*>(params_->logitGmAddr) +
                                     asc::te::get<LOGIT_INDEX>(baseOffset));
        rowIndexGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ RowIndexType*>(params_->rowIndexGmAddr) +
                                        asc::te::get<LOGIT_INDEX>(baseOffset));
    }
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::CopyInLogit(
    uint32_t curBaseM, uint64_t offsetM, AscendC::LocalTensor<DataTypeLogit> logitUb)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curBaseM * sizeof(DataTypeLogit)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DataTypeLogit> padParams;
    AscendC::DataCopyPad(logitUb, logitGlobal_[offsetM], copyParams, padParams);
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VectorAtomicProcess(
    uint32_t curBaseN, uint32_t curVecBaseM, uint64_t offsetM, uint64_t yOffset, uint64_t alignN,
    AscendC::LocalTensor<DataTypeOut> yLocal)
{
    AscendC::SetAtomicAdd<DataTypeOut>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; ++i) {
        auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
        AscendC::DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN], copyParams);
    }

    AscendC::SetAtomicNone();
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__simd_vf__ inline void
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::VfDoLogitMuls(
    uint32_t offsetRe, uint32_t offsetLogit, uint16_t repeatTimesLogit, uint16_t repeatTimesRe, uint64_t singleN,
    uint64_t l0cAlignN, uint64_t alignN, uint64_t vectorLength, __ubuf__ DataTypeOut* outUbAddr,
    __ubuf__ DataTypeIn* l0cOutUbAddr, __ubuf__ DataTypeLogit* logitUbAddr)
{
    l0cOutUbAddr += offsetRe;
    logitUbAddr += offsetLogit;
    AscendC::Reg::MaskReg mask;
    if constexpr (AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            uint32_t elementNum = static_cast<uint32_t>(singleN);
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);
                AscendC::Reg::RegTensor<bfloat16_t> vRegLogitBf16, vRegResultBf16, vRegDstBf16;
                if constexpr (AscendC::IsSameType<DataTypeLogit, bfloat16_t>::value) {
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B16>(vRegLogitBf16,
                                                                                                 logitUbAddr + i);
                } else {
                    AscendC::Reg::RegTensor<float> vRegLogitFp32;
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B32>(vRegLogitFp32,
                                                                                                 logitUbAddr + i);
                    AscendC::Reg::Cast<bfloat16_t, float, CAST_FR_FP32_TO_BF16>(vRegLogitBf16, vRegLogitFp32, mask);
                }
                AscendC::Reg::RegTensor<float> vRegResultFp32;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vRegResultFp32, l0cOutUbAddr + i * l0cAlignN + j * vectorLength);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FR_FP32_TO_BF16>(vRegResultBf16, vRegResultFp32, mask);
                AscendC::Reg::Mul(vRegDstBf16, vRegLogitBf16, vRegResultBf16, mask);
                AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    outUbAddr + i * alignN + j * vectorLength, vRegDstBf16, mask);
            }
        }
    } else {
        for (uint16_t i = 0; i < repeatTimesLogit; ++i) {
            AscendC::Reg::RegTensor<float> vRegLogit;
            uint32_t elementNum = static_cast<uint32_t>(singleN);
            for (uint16_t j = 0; j < repeatTimesRe; ++j) {
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);
                if constexpr (AscendC::IsSameType<DataTypeLogit, bfloat16_t>::value) {
                    AscendC::Reg::RegTensor<bfloat16_t> vRegLogitBf16;
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B16>(vRegLogitBf16,
                                                                                                 logitUbAddr + i);
                    AscendC::Reg::Cast<float, bfloat16_t, CAST_FR_BF16_TO_FP32>(vRegLogit, vRegLogitBf16, mask);
                } else {
                    AscendC::Reg::LoadAlign<DataTypeLogit, AscendC::Reg::LoadDist::DIST_BRC_B32>(vRegLogit,
                                                                                                 logitUbAddr + i);
                }
                AscendC::Reg::RegTensor<DataTypeIn> vRegResult;
                AscendC::Reg::RegTensor<DataTypeOut> vRegDst;
                AscendC::Reg::LoadAlign<DataTypeIn, AscendC::Reg::LoadDist::DIST_NORM>(
                    vRegResult, l0cOutUbAddr + i * l0cAlignN + j * vectorLength);
                AscendC::Reg::Mul(vRegDst, vRegLogit, vRegResult, mask);
                AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    outUbAddr + i * alignN + j * vectorLength, vRegDst, mask);
            }
        }
    }
}

BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueFinalizeRouting<BLAZE_GMMFR_BLOCK_EPILOGUE_FINALIZE_ROUTING_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape& blockShape, const BlockCoord& blockCoord)
{
    const uint64_t singleM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
    const uint64_t singleN = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
    const uint32_t halfSingleM = Blaze::Gemm::CeilDiv(singleM, static_cast<uint64_t>(AscendC::GetTaskRation()));
    const uint64_t l0cAlignN = Blaze::Gemm::Align32(singleN);
    // Each UB row is copied to GM separately, so every row start must be 32-byte aligned.
    constexpr uint64_t outputAlignElements = UB_TO_GM_ALIGN_BYTES / sizeof(DataTypeOut);
    const uint64_t alignN = Blaze::Gemm::CeilAlign(singleN, outputAlignElements);
    const uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    const uint32_t mOffset = subBlockIdx_ * halfSingleM;
    constexpr uint64_t vectorLength = BLOCK_BYTES / sizeof(DataTypeIn);
    const auto repeatTimesRe = Blaze::Gemm::CeilDiv(singleN, vectorLength);
    const uint64_t logitOffset = asc::te::get<LOGIT_INDEX>(blockCoord) + mOffset;
    const uint64_t yOffset = asc::te::get<Y_IDX>(blockCoord);
    const uint16_t remainRepeatTimesLogit = (singleMInVec % MAX_OUTPUT_M_UB != 0) ? singleMInVec % MAX_OUTPUT_M_UB :
                                                                                    MAX_OUTPUT_M_UB;
    auto logitUb = logitCrossPingPongId_ == 0 ? logitUbPing_ : logitUbPong_;
    CopyInLogit(singleMInVec, logitOffset, logitUb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(logitCrossPingPongId_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(logitCrossPingPongId_);
    logitCrossPingPongId_ = (logitCrossPingPongId_ + 1) & 1;
    const uint32_t loopNumY = Blaze::Gemm::CeilDiv(singleMInVec, MAX_OUTPUT_M_UB);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    for (uint32_t i = 0; i < loopNumY; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongId_);
        const uint16_t repeatTimesLine = (i == loopNumY - 1) ? remainRepeatTimesLogit : MAX_OUTPUT_M_UB;
        __ubuf__ DataTypeOut* outUbAddr = yCrossPingPongId_ == 0 ?
                                              reinterpret_cast<__ubuf__ DataTypeOut*>(outUbPing_.GetPhyAddr()) :
                                              reinterpret_cast<__ubuf__ DataTypeOut*>(outUbPong_.GetPhyAddr());
        auto l0cOutUbAddr = reinterpret_cast<__ubuf__ DataTypeIn*>(l0cOutUb_.GetPhyAddr());
        auto logitUbAddr = reinterpret_cast<__ubuf__ DataTypeLogit*>(logitUb.GetPhyAddr());
        asc_vf_call<VfDoLogitMuls>(i * MAX_OUTPUT_M_UB * l0cAlignN, i * MAX_OUTPUT_M_UB, repeatTimesLine, repeatTimesRe,
                                   singleN, l0cAlignN, alignN, vectorLength, outUbAddr, l0cOutUbAddr, logitUbAddr);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongId_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(yCrossPingPongId_);
        auto yLocal = yCrossPingPongId_ == 0 ? outUbPing_ : outUbPong_;
        VectorAtomicProcess(singleN, repeatTimesLine, logitOffset + i * MAX_OUTPUT_M_UB, yOffset, alignN, yLocal);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(yCrossPingPongId_);
        yCrossPingPongId_ = (yCrossPingPongId_ + 1) & 1;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongId_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(logitCrossPingPongId_);
}
} // namespace Block
} // namespace Epilogue

namespace Gemm::Block {
using Epilogue::Block::BlockEpilogueFinalizeRouting;
}

} // namespace Blaze
#endif
#endif
