/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_flat_quant.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "blaze/epilogue/tile/compute.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = Fusion::DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFlatQuant {
public:
    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;
    using FusionOp = FusionOp_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        ProblemShape problemShape{};
        float dstTypeMax{0.0f};
        float invDstTypeMax{0.0f};
    };

    __aicore__ inline BlockEpilogueFlatQuant() {}

    __aicore__ inline void Init(Params const& params);
    __aicore__ inline void operator()(uint64_t startBatchIdx, uint64_t iterBatch);

private:
    static constexpr uint8_t GATHER_PATTERN = 7;
    static constexpr int32_t CEIL_SIZE = 16;
    static constexpr int32_t GROUP_SIZE = 32;
    static constexpr int32_t VEC_N_LEN = 64;
    static constexpr int32_t MN_SIZE = 64 * 1024;
    static constexpr int32_t OUT_SIZE = 32 * 1024;
    static constexpr int32_t EMAX_SIZE = 2 * 1024;
    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100; // MxQuantConfig fpEmax（沿用 flat 实现的硬编码上界）
    static constexpr uint16_t BLOCK_SCALE = 2;
    static constexpr float ZERO_FLOAT = 0.0f;
    static constexpr float SIX_FLOAT = 6.0f;
    static constexpr float SEVEN_FLOAT = 7.0f;
    static constexpr float TWELVE_FLOAT = 12.0f;
    static constexpr uint32_t STORE_UNALIGN_STRIDE_BYTES = 8;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;

    __aicore__ inline void Quant(uint64_t batchIdx, uint64_t iterIdx);
    __aicore__ inline void ClearDirtyData();
    __aicore__ inline void ClearScaleTensor();
    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src);
    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src);
    __aicore__ inline void ComputeMxQuant(AscendC::LocalTensor<bfloat16_t>& xTensor,
                                          AscendC::LocalTensor<int8_t>& yTensor,
                                          AscendC::LocalTensor<uint16_t>& eMaxTensor,
                                          AscendC::LocalTensor<int8_t>& scaleTensor,
                                          AscendC::LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB,
                                          uint64_t inputOffset);
    __aicore__ inline void ComputeTransLayout(AscendC::LocalTensor<int8_t>& scaleTensor,
                                              AscendC::LocalTensor<int8_t>& scaleBlockTensor, uint16_t m, uint16_t n);

    static __simd_vf__ inline void SaveTailVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count);
    static __simd_vf__ inline void ClearTailVf(__ubuf__ uint16_t* dstPtr, uint32_t count);
    static __simd_vf__ inline void RestoreTailVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count);

    template <typename T>
    __aicore__ inline static uint32_t GetUbByteOffset(const AscendC::LocalTensor<T>& tensor)
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.GetPhyAddr()) - asc_get_phy_buf_addr(0));
    }

    // ---- Shape ----
    struct FlatQuantShapeInfo {
        int64_t k{0};
        int64_t m{0};
        int64_t n{0};
        int64_t mCeil{0};
        int64_t nCeil{0};
    };

    FlatQuantShapeInfo shape_;

    // ---- UB ----
    // TOTAL_UB_SIZE 仅在 device 编译 pass（__NPU_ARCH__ 已定义）可用，
    // 因此 xTensor_ 在 __aicore__ 的 Init 中构造，不能在类作用域默认初始化。
    AscendC::LocalTensor<bfloat16_t> xTensor_;
    AscendC::LocalTensor<int8_t> yTensor_;
    AscendC::LocalTensor<uint16_t> eMaxTensor_;
    AscendC::LocalTensor<int8_t> scaleTensor_;
    AscendC::LocalTensor<uint16_t> deQuantScaleTensor_;
    AscendC::LocalTensor<int8_t> scaleBlockTensor_;

    // ---- GM ----
    AscendC::GlobalTensor<int8_t> cGlobal_;
    AscendC::GlobalTensor<int8_t> scaleGlobal_;

    // ---- Problem / Params ----
    ProblemShape problemShape_;
    int64_t alignM_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;

    // ---- Events ----
    uint16_t eventIdVToMte3_{2};
    uint16_t eventIdMte3ToV_{3};
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::Init(
    Params const& params)
{
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.outGmAddr));
    scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.scaleGmAddr));
    problemShape_ = params.problemShape;
    xTensor_ = AscendC::LocalTensor<bfloat16_t>{AscendC::TPosition::VECIN, 0,
                                                AscendC::TOTAL_UB_SIZE / sizeof(bfloat16_t)};

    shape_.m = asc::te::get<Gemm::MNK_M>(problemShape_);
    shape_.n = asc::te::get<Gemm::MNK_N>(problemShape_);
    shape_.k = asc::te::get<Gemm::MNK_B>(problemShape_);
    dstTypeMax_ = params.dstTypeMax;
    invDstTypeMax_ = params.invDstTypeMax;
    if (dstTypeMax_ == SIX_FLOAT) {
        addValueBit_ = ADD_VALUE_FOR_BF16_MAN1;
    } else if (dstTypeMax_ == SEVEN_FLOAT) {
        addValueBit_ = ADD_VALUE_FOR_BF16_MAN2;
    }

    shape_.mCeil = Gemm::CeilAlign(static_cast<int64_t>(shape_.m), static_cast<int64_t>(CEIL_SIZE));
    shape_.nCeil = VEC_N_LEN;
    alignM_ = Gemm::CeilDiv(static_cast<int64_t>(shape_.m * shape_.n), static_cast<int64_t>(VEC_N_LEN));

    yTensor_ = xTensor_[MN_SIZE].template ReinterpretCast<int8_t>();
    eMaxTensor_ = yTensor_[OUT_SIZE].template ReinterpretCast<uint16_t>();
    deQuantScaleTensor_ = eMaxTensor_[EMAX_SIZE];
    scaleTensor_ = deQuantScaleTensor_[EMAX_SIZE].template ReinterpretCast<int8_t>();
    scaleBlockTensor_ = scaleTensor_[EMAX_SIZE];
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearDirtyData()
{
    AscendC::GatherMaskParams params;
    params.src0BlockStride = 1;
    params.src0RepeatStride = Gemm::CeilAlign(static_cast<int64_t>(shape_.n), static_cast<int64_t>(CEIL_SIZE)) *
                              sizeof(DataTypeIn) / GROUP_SIZE;
    params.src1RepeatStride = 0;
    params.repeatTimes = shape_.m;
    uint64_t rvdCnt = 0Ul;
    AscendC::GatherMask(xTensor_, xTensor_, GATHER_PATTERN, true, shape_.n, params, rvdCnt);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearScaleTensor()
{
    AscendC::Duplicate(scaleTensor_, static_cast<int8_t>(0), EMAX_SIZE);
    AscendC::Duplicate(scaleBlockTensor_, static_cast<int8_t>(0), EMAX_SIZE);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::SaveTailVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count)
{
    AscendC::Reg::RegTensor<uint16_t> vReg;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::LoadUnAlignPre(u0, srcPtr);
    AscendC::Reg::LoadUnAlign(vReg, u0, srcPtr);
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<uint16_t>(count);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(dstPtr, vReg, mask);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearTailVf(
    __ubuf__ uint16_t* dstPtr, uint32_t count)
{
    AscendC::Reg::RegTensor<uint16_t> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, static_cast<uint16_t>(0));
    AscendC::Reg::UnalignRegForStore u1;

    constexpr uint32_t strideElems = STORE_UNALIGN_STRIDE_BYTES / sizeof(uint16_t);
    uint32_t loopCount = (count + strideElems - 1) / strideElems;
    for (uint32_t i = 0; i < loopCount; i++) {
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dstPtr, zeroReg, u1,
                                                                                          STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::RestoreTailVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count)
{
    constexpr uint32_t strideElems = STORE_UNALIGN_STRIDE_BYTES / sizeof(uint16_t);
    uint32_t loopCount = (count + strideElems - 1) / strideElems;
    AscendC::Reg::UnalignRegForStore u1;
    for (uint32_t i = 0; i < loopCount; i++) {
        AscendC::Reg::RegTensor<uint16_t> vReg;
        AscendC::Reg::UnalignRegForLoad u0;
        __ubuf__ uint16_t* curSrcPtr = srcPtr + i * strideElems;
        AscendC::Reg::LoadUnAlignPre(u0, curSrcPtr);
        AscendC::Reg::LoadUnAlign(vReg, u0, curSrcPtr);
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dstPtr, vReg, u1,
                                                                                          STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_,
                                              FusionOp_>::CopyOutputFromUbToGm(uint64_t offset,
                                                                               AscendC::LocalTensor<int8_t>& src)
{
    uint64_t alignedOffset = offset >> 1;
    copy_ubuf_to_gm_align_v2(cGlobal_[alignedOffset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, 1,
                             static_cast<uint32_t>((shape_.m * shape_.n * sizeof(int8_t)) >> 1), 0, 0, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_,
                                              FusionOp_>::CopyScaleFromUbToGm(uint64_t offset,
                                                                              AscendC::LocalTensor<int8_t>& src)
{
    uint32_t blockCount = static_cast<uint32_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(alignM_ * shape_.nCeil), Gemm::MXFP_DIVISOR_SIZE));
    copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, blockCount,
                             BLOCK_SCALE, 0, BLOCK_SCALE, 32);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ComputeMxQuant(
    AscendC::LocalTensor<bfloat16_t>& xTensor, AscendC::LocalTensor<int8_t>& yTensor,
    AscendC::LocalTensor<uint16_t>& eMaxTensor, AscendC::LocalTensor<int8_t>& scaleTensor,
    AscendC::LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB, uint64_t inputOffset)
{
    const uint32_t totalScale = static_cast<uint32_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(totalDataInUB), static_cast<uint64_t>(GROUP_SIZE)));
    const uint32_t xByteOffset = GetUbByteOffset(xTensor) +
                                 static_cast<uint32_t>(inputOffset) * static_cast<uint32_t>(sizeof(bfloat16_t));

    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalDataInUB),
                                                    static_cast<int64_t>(totalDataInUB));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalScale),
                                                     static_cast<int64_t>(totalScale));
    auto xUbTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(xByteOffset),
                                          dataLayout);
    auto maxExpTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GetUbByteOffset(eMaxTensor)), scaleLayout);
    auto yScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(scaleTensor)), scaleLayout);
    auto reciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(GetUbByteOffset(deQuantScaleTensor)), scaleLayout);
    auto yUbTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(yTensor)), dataLayout);

    // MxQuantConfig 在调用点由 Init 阶段解析的普通成员组装（tile 类型不得出现在类作用域）
    // zeroScaleOnZeroExp=true: flat 阵营，zeroMask 取原始 maxExp
    Tile::MxQuantConfig cfg{Tile::MxScaleAlg::OCP, FP4_E2M1_MAX_EXP, invDstTypeMax_, addValueBit_, true};
    if (dstTypeMax_ == ZERO_FLOAT) {
        cfg.alg = Tile::MxScaleAlg::OCP;
    } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
        cfg.alg = Tile::MxScaleAlg::DYN_DTYPE_RANGE;
    } else {
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
    }

    Tile::MxQuant<DataTypeOut> mx;
    mx.GroupMaxExp(xUbTensor, maxExpTensor, totalDataInUB, dstTypeMax_ >= SIX_FLOAT && dstTypeMax_ <= TWELVE_FLOAT);
    mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
    mx.Quantize(xUbTensor, reciprocalTensor, yUbTensor, totalDataInUB);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ComputeTransLayout(
    AscendC::LocalTensor<int8_t>& scaleTensor, AscendC::LocalTensor<int8_t>& scaleBlockTensor, uint16_t m, uint16_t n)
{
    uint16_t scaleBlockN = Gemm::CeilDiv(static_cast<uint64_t>(n), static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) * 2;

    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(m), static_cast<int64_t>(scaleBlockN),
                                                   static_cast<int64_t>(scaleBlockN));
    auto srcTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(scaleTensor)), srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(m), static_cast<int64_t>(AscendC::ONE_BLK_SIZE),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(GetUbByteOffset(scaleBlockTensor)), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransScaleLayout(srcTensor, dstTensor, m, scaleBlockN);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::Quant(
    uint64_t batchIdx, uint64_t iterIdx)
{
    int64_t mnSize = shape_.m * shape_.n;
    uint64_t yOffset = batchIdx * static_cast<uint64_t>(mnSize);
    uint64_t scaleOffset = batchIdx *
                           Gemm::CeilDiv(static_cast<uint64_t>(mnSize),
                                         static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) *
                           2;
    uint32_t totalDataInUB = static_cast<uint32_t>(mnSize);
    uint64_t inputOffset = iterIdx * totalDataInUB;
    ClearScaleTensor();

    if (shape_.n % 16 != 0) {
        ClearDirtyData();
    }

    uint32_t tailRemainder = totalDataInUB % GROUP_SIZE;
    if (tailRemainder != 0) {
        uint32_t tailSize = GROUP_SIZE - tailRemainder;
        __ubuf__ uint16_t* tailAddr = (__ubuf__ uint16_t*)xTensor_.GetPhyAddr() + inputOffset + totalDataInUB;
        __ubuf__ uint16_t* saveAddr = (__ubuf__ uint16_t*)scaleBlockTensor_.GetPhyAddr();
        AscendC::VF_CALL<SaveTailVf>(saveAddr, tailAddr, tailSize);
        AscendC::VF_CALL<ClearTailVf>(tailAddr, tailSize);
    }

    ComputeMxQuant(xTensor_, yTensor_, eMaxTensor_, scaleTensor_, deQuantScaleTensor_, totalDataInUB, inputOffset);

    if (tailRemainder != 0) {
        uint32_t tailSize = GROUP_SIZE - tailRemainder;
        __ubuf__ uint16_t* tailAddr = (__ubuf__ uint16_t*)xTensor_.GetPhyAddr() + inputOffset + totalDataInUB;
        __ubuf__ uint16_t* saveAddr = (__ubuf__ uint16_t*)scaleBlockTensor_.GetPhyAddr();
        AscendC::VF_CALL<RestoreTailVf>(tailAddr, saveAddr, tailSize);
    }

    ComputeTransLayout(scaleTensor_, scaleBlockTensor_, static_cast<uint16_t>(alignM_),
                       static_cast<uint16_t>(shape_.nCeil));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3_);

    CopyOutputFromUbToGm(yOffset, yTensor_);
    CopyScaleFromUbToGm(scaleOffset, scaleBlockTensor_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV_);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::operator()(
    uint64_t startBatchIdx, uint64_t iterBatch)
{
    for (uint64_t iter = 0; iter < iterBatch; ++iter) {
        Quant(startBatchIdx + iter, iter);
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
