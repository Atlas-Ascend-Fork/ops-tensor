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
 * \file block_epilogue_gelu_mx_quant.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "math/erf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

enum class QuantAlg : uint32_t {
    OCP = 0,
    BLAS = 1,
    DYN_DTYPE_RANGE = 2,
};

enum class GeluAlg : uint8_t {
    TANH = 0,
    ERF = 1,
};

enum class ROUND_MODE_FP4 : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t MX_SCALE_ALIGN_SIZE = 2;

constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint32_t MAX_SINGLE_SCALE_NUM = MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE;
constexpr uint16_t BF16_ADD_VALUE_MAN1 = 0x003f; // dst_TypeMax=0.0或6.0时使用
constexpr uint16_t BF16_ADD_VALUE_MAN2 = 0x001f; // dst_TypeMax=7.0时使用
// elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // 0b 0000 0100 0000 0000 右移7位为8
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780; // 0b 0000 0111 1000 0000 右移7位为15
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100; // 0b 0000 0001 0000 0000 右移7位为2
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000; // 右移7位为0

constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;

template <typename DataTypeOut_, typename DataTypeIn_>
class BlockEpilogueGeluMxQuant {
public:
    __aicore__ inline BlockEpilogueGeluMxQuant() {}

    struct Params {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        GeluAlg geluAlg{GeluAlg::TANH};
        QuantAlg quantAlg{QuantAlg::OCP};
        ROUND_MODE_FP4 fp4RoundMode{ROUND_MODE_FP4::RINT};
        float dtypeMax{0.0F};
        Params() = default;
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    // shape
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const& params);

    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    template <class T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T*>(asc_get_phy_buf_addr(0) + byteOffset);
    }

    __aicore__ inline void SetupUbLayout();

    __aicore__ inline void VFDoGeluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize);
    __aicore__ inline void TransFp4MxOutLayout(uint16_t mSize);
    __aicore__ inline void VFDoGeluAndQuantForMX(uint16_t mSize, uint16_t nSize);
    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, int64_t gmOffset);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, int64_t gmOffset);

    // ---- Params ----
    const Params* params_{nullptr};

    // ---- GM base pointers (set via UpdateGlobalAddr) ----
    __gm__ int8_t* quantOutputGmAddr_{nullptr};
    __gm__ int8_t* quantScaleGmAddr_{nullptr};

    // ---- UB byte offsets (set in SetupUbLayout) ----
    uint64_t quantOutputUbOffset_{0};
    uint64_t quantScaleOutputUbOffset_{0};
    uint64_t quantScaleBlockOutputUbOffset_{0};
    uint64_t geluResUbOffset_{0};
    uint64_t maxExpUbOffset_{0};
    uint64_t halfScaleUbOffset_{0};

    int64_t n_;
    int64_t scaleN_;
    int64_t scaleNAlign_;
    int64_t scaleBlockN_;
    uint32_t subBlockIdx_;
    uint32_t singleM_;
    uint32_t singleN_;

    uint16_t fpEmax_{0};
    float invDstTypeMax_{1.0f / 6.0f};
    float dstTypeMax_{0.0};
    uint32_t addValueBits_{BF16_ADD_VALUE_MAN1}; // 场景2，进位附加值

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::Init(Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    // 量化结果的Nan值会转变成极大值
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    params_ = &params;
    subBlockIdx_ = AscendC::GetSubBlockIdx();
    quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr);
    quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        invDstTypeMax_ = 1.0f / 448.0f;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        invDstTypeMax_ = 1.0f / 57344.0f;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
        dstTypeMax_ = params_->dtypeMax;
        invDstTypeMax_ = 1.0f / 6.0f;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
        dstTypeMax_ = params_->dtypeMax;
        invDstTypeMax_ = 1.0f / 3.5f;
    }
    if (params_->quantAlg == QuantAlg::DYN_DTYPE_RANGE && params_->dtypeMax != DIGIT_ZERO_FLOAT) {
        invDstTypeMax_ = 1.0f / params_->dtypeMax;
    }
    // 进位附加值规则统一为 gelu_tanh 语义
    addValueBits_ = params_->dtypeMax == DIGIT_SEVEN_FLOAT ? BF16_ADD_VALUE_MAN2 : BF16_ADD_VALUE_MAN1;
    SetupUbLayout();
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::SetupUbLayout()
{
    constexpr uint32_t afterIn = MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutputUbOffset_ = afterIn;
    constexpr uint32_t afterOut = afterIn + MAX_SINGLE_MN * sizeof(int8_t);
    quantScaleOutputUbOffset_ = afterOut;
    constexpr uint32_t afterIO = afterOut + MAX_SINGLE_SCALE_NUM * sizeof(int8_t);
    geluResUbOffset_ = afterIO;
    constexpr uint32_t afterIOAndGelu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    maxExpUbOffset_ = afterIOAndGelu;
    constexpr uint32_t afterIOAndGeluExp = afterIOAndGelu + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    halfScaleUbOffset_ = afterIOAndGeluExp;
    constexpr uint32_t realScaleBlockOffset = afterIOAndGeluExp + MAX_SINGLE_SCALE_NUM * sizeof(uint16_t);
    quantScaleBlockOutputUbOffset_ = realScaleBlockOffset;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yGmAddr) + asc::te::get<Y_IDX>(baseOffset);
        quantScaleGmAddr_ = reinterpret_cast<__gm__ int8_t*>(params_->yScaleGmAddr) +
                            asc::te::get<Y_SCALE_IDX>(baseOffset);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = asc::te::get<Gemm::MNK_N>(problemShape);
    scaleN_ = Gemm::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(BLOCK_SIZE));
    scaleNAlign_ = Gemm::CeilAlign(scaleN_, MX_SCALE_ALIGN_SIZE);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyOutputFromUb2Gm(uint64_t blockCount,
                                                                                                int64_t gmOffset)
{
    int64_t nValid = static_cast<int64_t>(singleN_);
    int64_t gmRowPitch = n_;

    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        nValid = nValid >> 1;
        gmRowPitch = gmRowPitch >> 1;
        gmOffset = gmOffset >> 1;
    }
    int64_t nUbAligned = static_cast<int64_t>(Gemm::Align32(static_cast<uint64_t>(nValid)));

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                      ubLayout);
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        if (static_cast<int64_t>(singleN_) % OUT_ELE_NUM_ONE_BLK != 0) {
            outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(geluResUbOffset_),
                                         ubLayout);
        }
    }
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantOutputGmAddr_ + gmOffset),
                                      gmLayout);

    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    asc::te::copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::CopyScaleFromUb2Gm(uint64_t blockCount,
                                                                                               int64_t gmOffset)
{
    int64_t nValid = static_cast<int64_t>(scaleBlockN_);
    int64_t nUbAligned = static_cast<int64_t>(AscendC::ONE_BLK_SIZE);
    int64_t gmRowPitch = scaleNAlign_;

    auto ubLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, nUbAligned);
    auto gmLayout = Gemm::MakeNDExtLayout(static_cast<int64_t>(blockCount), nValid, gmRowPitch);
    auto outUb = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleBlockOutputUbOffset_), ubLayout);
    auto outGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(quantScaleGmAddr_ + gmOffset),
                                      gmLayout);

    auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
    asc::te::copy(copyUB2GM, outGm, outUb);
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluAndQuantForMX(uint16_t mSize,
                                                                                                  uint16_t nSize)
{
    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize)); // 输入为32位对齐
    __ubuf__ bfloat16_t* geluResAddr = GetUbAddr<bfloat16_t>(geluResUbOffset_);
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<bfloat16_t> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0.0));
            constexpr uint32_t bf16Vl = asc_get_vf_len() / sizeof(bfloat16_t);
            uint32_t remainingElements = mSize * nAligned;
            uint32_t zeroOffset = 0;
            while (remainingElements > 0) {
                AscendC::Reg::MaskReg zeroMask = AscendC::Reg::UpdateMask<bfloat16_t>(remainingElements);
                AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(geluResAddr + zeroOffset,
                                                                                             zeroReg, zeroMask);
                zeroOffset += bf16Vl;
            }
        }
    }
    auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize), static_cast<int64_t>(nSize),
                                        static_cast<int64_t>(nAligned));
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0), layout);
    auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(geluResUbOffset_),
                                          layout);
    Tile::Gelu<bfloat16_t, DataTypeIn> gelu;
    if (params_->geluAlg == GeluAlg::ERF) {
        gelu.GeluErf(srcTensor, dstTensor, mSize, nSize);
    } else {
        gelu.GeluTanh(srcTensor, dstTensor, mSize, nSize);
    }

    const uint32_t totalDataInUb = mSize * nAligned;
    const uint32_t totalScaleInUb = totalDataInUb / BLOCK_SIZE;

    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalDataInUb),
                                                    static_cast<int64_t>(totalDataInUb));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalScaleInUb),
                                                     static_cast<int64_t>(totalScaleInUb));
    auto geluResTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(geluResUbOffset_), dataLayout);
    auto maxExpTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(maxExpUbOffset_),
                                             scaleLayout);
    auto yScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleOutputUbOffset_), scaleLayout);
    auto reciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(halfScaleUbOffset_), scaleLayout);
    auto yTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                        dataLayout);

    // MxQuantConfig 由 Init 阶段解析的普通成员在调用点组装（tile 类型不得出现在类作用域）
    // zeroScaleOnZeroExp=true: gelu_mx 阵营
    Tile::MxQuantConfig cfg{Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_, static_cast<uint16_t>(addValueBits_), true};

    Tile::MxQuant<DataTypeOut> mx;
    if (params_->quantAlg == QuantAlg::OCP) {
        cfg.alg = Tile::MxScaleAlg::OCP;
        mx.GroupMaxExp(geluResTensor, maxExpTensor, totalDataInUb, false);
    } else if (params_->quantAlg == QuantAlg::BLAS) {
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
        mx.GroupMaxExp(geluResTensor, maxExpTensor, totalDataInUb, true);
    } else if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
        cfg.alg = Tile::MxScaleAlg::DYN_DTYPE_RANGE;
        mx.GroupMaxExp(geluResTensor, maxExpTensor, totalDataInUb, true);
    } else {
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
        mx.GroupMaxExp(geluResTensor, maxExpTensor, totalDataInUb, true);
    }
    mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScaleInUb);

    Tile::MxQuantFp4RoundMode fp4RoundMode = Tile::MxQuantFp4RoundMode::RINT;
    if (params_->fp4RoundMode == ROUND_MODE_FP4::FLOOR) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::FLOOR;
    } else if (params_->fp4RoundMode == ROUND_MODE_FP4::ROUND) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::ROUND;
    }
    mx.Quantize(geluResTensor, reciprocalTensor, yTensor, totalDataInUb, fp4RoundMode);
    return;
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::VFDoGeluForMX(uint16_t mSize)
{
    VFDoGeluAndQuantForMX(mSize, static_cast<uint16_t>(singleN_));
}

/**
 * @brief 转换FP4 MX量化输出的数据布局
 *
 * 委托 MxQuant::TransFp4OutLayout：将FP4量化输出从 Align16(n/2) 字节行距的线性
 * 布局转为 Align32(n/2) 字节行距的块对齐布局，满足 MTE 搬运的对齐要求。
 *
 * 注意: 该函数仅在DataTypeOut为fp4x2_e2m1_t或fp4x2_e1m2_t且singleN_ < 64时被调用
 */
template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransFp4MxOutLayout(uint16_t mSize)
{
    const int64_t packedBytes = static_cast<int64_t>(singleN_) / 2;
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), packedBytes,
                                                   static_cast<int64_t>(Gemm::Align16(packedBytes)));
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantOutputUbOffset_),
                                          srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), packedBytes,
                                                   static_cast<int64_t>(Gemm::Align32(packedBytes)));
    auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(geluResUbOffset_),
                                          dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransFp4OutLayout(srcTensor, dstTensor, mSize, static_cast<uint16_t>(singleN_));
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::TransMxScaleLayout(uint16_t mSize)
{
    // scale layout: (mSize*8) -> (mSize,32)
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), scaleBlockN_, scaleBlockN_);
    auto srcTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleOutputUbOffset_), srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(quantScaleBlockOutputUbOffset_), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransScaleLayout(srcTensor, dstTensor, mSize, static_cast<uint16_t>(scaleBlockN_));
}

template <typename DataTypeOut_, typename DataTypeIn_>
__aicore__ inline void BlockEpilogueGeluMxQuant<DataTypeOut_, DataTypeIn_>::operator()(const BlockShape& blockShape,
                                                                                       const BlockCoord& blockCoord)
{
    singleM_ = asc::te::get<Gemm::MNK_M>(blockShape);
    singleN_ = asc::te::get<Gemm::MNK_N>(blockShape);
    scaleBlockN_ = Gemm::CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(BLOCK_SIZE));
    blockCoord_ = blockCoord;
    auto halfSingleM = Gemm::CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    VFDoGeluForMX(singleMInVec);
    int64_t yOffset = static_cast<int64_t>(asc::te::get<Y_IDX>(blockCoord)) +
                      static_cast<int64_t>(subBlockIdx_ * halfSingleM * n_);
    int64_t yScaleOffset = static_cast<int64_t>(asc::te::get<Y_SCALE_IDX>(blockCoord)) +
                           static_cast<int64_t>(subBlockIdx_ * halfSingleM * scaleNAlign_);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        if (static_cast<int64_t>(singleN_) % OUT_ELE_NUM_ONE_BLK != 0) {
            TransFp4MxOutLayout(singleMInVec);
        }
    }
    TransMxScaleLayout(singleMInVec);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Epilogue
} // namespace Blaze
