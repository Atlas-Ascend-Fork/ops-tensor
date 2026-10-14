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
 * \file block_epilogue_qbmm_pertensor_streamk.h
 * \brief Dedicated AIV epilogue for QBMM per-tensor StreamK.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/tile/compute.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

using AscendC::IsSameType;
using Blaze::Gemm::CeilAlign;
using Blaze::Gemm::CeilDiv;

// The dedicated epilogue owns the complete AIV path: reduce split-K workspace partials,
// apply per-tensor x2 scale and optional x1 scale/bias, cast, and write the final output.
template <class WorkspaceType_, class OutType_, class DispatchPolicy_, class X2ScaleType_ = float,
          class X1ScaleType_ = float>
class BlockEpilogueQbmmPertensorStreamK {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR perTokenScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        bool isBias{false};
        uint32_t biasDtype{0};
    };

    __aicore__ inline BlockEpilogueQbmmPertensorStreamK()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }
    __aicore__ inline ~BlockEpilogueQbmmPertensorStreamK()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    using WorkspaceType = WorkspaceType_;
    using OutType = OutType_;
    using DispatchPolicy = DispatchPolicy_;
    using X2ScaleType = X2ScaleType_;
    using X1ScaleType = X1ScaleType_;

    static constexpr uint32_t DATA_BLOCK = 32U;
    static constexpr uint32_t OUT_ALIGN = DATA_BLOCK / sizeof(OutType);
    static constexpr uint32_t DEQ_SCALE_MUL_MASK = 0xFFFFE000U;
    static constexpr uint16_t BIAS_EVENT_ID = 2U;

    __aicore__ inline static float DecodeMaskedDequantScale(uint32_t scaleBits)
    {
        scaleBits &= DEQ_SCALE_MUL_MASK;
        return *reinterpret_cast<float*>(&scaleBits);
    }

    __aicore__ inline static float MergeAndMaskDequantScale(float x2Scale, float x1Scale)
    {
        const float dequantScale = x2Scale * x1Scale;
        return DecodeMaskedDequantScale(*reinterpret_cast<const uint32_t*>(&dequantScale));
    }

private:
    // basic args
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t mL1_ = 0;
    uint64_t nL1_ = 0;
    uint64_t mCnt_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t kCnt_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t taskGroupSize_ = 0;

    struct AivParams {
        uint64_t indexParams = 0;
        uint64_t mCntIndex = 0;
        uint64_t nCntIndex = 0;
        uint64_t kCntIndex = 0;
        uint64_t curML1InAiv = 0;
        uint64_t curNL1InAiv = 0;
        uint64_t curAlignedNInAiv = 0;
    };
    AivParams aivParams_;

    uint64_t cGmOffset_ = 0;
    GM_ADDR cGmAddr_{nullptr};
    GM_ADDR workspaceGmAddr_{nullptr};
    GM_ADDR scaleGmAddr_{nullptr};
    GM_ADDR perTokenScaleGmAddr_{nullptr};
    GM_ADDR biasGmAddr_{nullptr};
    bool isBias_{false};
    uint32_t biasDtype_{0};

    struct ReductionTileParams {
        uint64_t workspaceOffset = 0;
        uint64_t validRows = 0;
        uint64_t packedElements = 0;
    };
    ReductionTileParams reductionTileParams_;

public:
    __aicore__ inline void Init(Params const& params, BlockShape blockShapeInAiv, BlockShape tileL1ShapeInAiv,
                                BlockCoord coordInAiv, uint64_t usedCoreNum, bool checkIsSkScene)
    {
        m_ = asc::te::get<Blaze::Gemm::MNK_M>(blockShapeInAiv);
        n_ = asc::te::get<Blaze::Gemm::MNK_N>(blockShapeInAiv);
        mL1_ = asc::te::get<Blaze::Gemm::MNK_M>(tileL1ShapeInAiv);
        nL1_ = asc::te::get<Blaze::Gemm::MNK_N>(tileL1ShapeInAiv);
        mCnt_ = asc::te::get<Blaze::Gemm::MNK_M>(coordInAiv);
        nCnt_ = asc::te::get<Blaze::Gemm::MNK_N>(coordInAiv);
        kCnt_ = asc::te::get<Blaze::Gemm::MNK_K>(coordInAiv);
        usedCoreNum_ = usedCoreNum;
        cGmAddr_ = params.cGmAddr;
        scaleGmAddr_ = params.scaleGmAddr;
        perTokenScaleGmAddr_ = params.perTokenScaleGmAddr;
        biasGmAddr_ = params.biasGmAddr;
        hasX1Scale_ = perTokenScaleGmAddr_ != nullptr;
        isBias_ = params.isBias;
        biasDtype_ = params.biasDtype;
        workspaceGmAddr_ = params.workspaceGmAddr;
        AscendC::ICachePreLoad(NUM_TWO);
        // Ensure cube to pair with vector, add sync flag in dp+sk scene
        if (!checkIsSkScene) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Blaze::Gemm::ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Blaze::Gemm::ZERO_FLAG);
        }
    }

    __aicore__ inline void Run()
    {
        taskGroupSize_ = AscendC::GetTaskRation() * kCnt_;
        if (taskGroupSize_ == 0UL || usedCoreNum_ == 0UL || mCnt_ == 0UL || nCnt_ == 0UL) {
            return;
        }
        UpdateAivBasicIndex();
        UpdateAivBasicBlock();
        UpdateAivParams();
        AscendC::LocalTensor<WorkspaceType> ubAddTensor{AscendC::TPosition::VECIN, 0,
                                                        AscendC::TOTAL_UB_SIZE / sizeof(WorkspaceType)};
        if (reductionTileParams_.validRows == 0) {
            return;
        }
        AcquireAuxUbForWorkspace();
        CopyWorkspaceToUb();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);

        for (uint64_t i = 1; i < kCnt_; ++i) {
            AscendC::Add(ubAddTensor, ubAddTensor, ubAddTensor[i * reductionTileParams_.packedElements],
                         reductionTileParams_.packedElements);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ReleaseAuxCopyAfterReduce();

        DequantAndStore();
    }

    __aicore__ inline void UpdateAivBasicIndex()
    {
        uint64_t newBlockIdx = AscendC::GetBlockIdx() / taskGroupSize_;
        aivParams_.kCntIndex = AscendC::GetBlockIdx() % taskGroupSize_;

        aivParams_.indexParams = newBlockIdx;
        uint64_t cGmIndex = aivParams_.indexParams + (mCnt_ * nCnt_ - mCnt_ * nCnt_ % usedCoreNum_);
        uint64_t mainWindow = AscendC::Std::min(MAIN_WINDOW, mCnt_);
        uint64_t mainRow = mCnt_ / mainWindow - 1UL;
        uint64_t tailWindow = mCnt_ - mainRow * mainWindow;
        uint64_t rowIdx = cGmIndex / nCnt_ / mainWindow;
        if (rowIdx < mainRow) {
            aivParams_.mCntIndex = rowIdx * mainWindow + cGmIndex % mainWindow;
            aivParams_.nCntIndex = (cGmIndex / mainWindow) % nCnt_;
        } else {
            rowIdx = mainRow;
            uint64_t tailIndex = cGmIndex - mainRow * mainWindow * nCnt_;
            aivParams_.mCntIndex = mainRow * mainWindow + tailIndex % tailWindow;
            aivParams_.nCntIndex = (tailIndex / tailWindow) % nCnt_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % NUM_TWO != 0UL) {
            aivParams_.nCntIndex = nCnt_ - 1UL - aivParams_.nCntIndex;
        }
    }

    __aicore__ inline void UpdateAivBasicBlock()
    {
        aivParams_.curML1InAiv = aivParams_.mCntIndex != (mCnt_ - 1) ? mL1_ : (m_ - (mCnt_ - 1) * mL1_);
        aivParams_.curNL1InAiv = aivParams_.nCntIndex != (nCnt_ - 1) ? nL1_ : (n_ - (nCnt_ - 1) * nL1_);
        aivParams_.curAlignedNInAiv = Blaze::Gemm::CeilAlign(
            aivParams_.curNL1InAiv, static_cast<uint64_t>(AscendC::GetVecLen() / sizeof(WorkspaceType)));
    }

    __aicore__ inline void UpdateAivParams()
    {
        const uint64_t rowsPerTaskGroup = Blaze::Gemm::CeilAlign(
            Blaze::Gemm::CeilDiv(aivParams_.curML1InAiv, taskGroupSize_),
            Blaze::Gemm::CeilDiv(UB2GM_SRCGAP_UNIT, aivParams_.curAlignedNInAiv));
        uint64_t rowGroupCount = Blaze::Gemm::CeilDiv(aivParams_.curML1InAiv, rowsPerTaskGroup);
        uint64_t tailRows = aivParams_.curML1InAiv - (rowGroupCount - 1) * rowsPerTaskGroup;
        const uint64_t assignedRows = aivParams_.kCntIndex >= rowGroupCount ?
                                          0UL :
                                          (aivParams_.kCntIndex == rowGroupCount - 1 ? tailRows : rowsPerTaskGroup);

        reductionTileParams_.validRows = assignedRows;
        // Calculate the workspace tile origin used by the split-K reduction.
        reductionTileParams_.workspaceOffset = (aivParams_.indexParams) * kCnt_ * BLOCK_BASE_M * BLOCK_BASE_N +
                                               aivParams_.kCntIndex * rowsPerTaskGroup * aivParams_.curAlignedNInAiv;
        // Calculate the corresponding output tile origin in C GM.
        cGmOffset_ = aivParams_.nCntIndex * nL1_ + aivParams_.mCntIndex * mL1_ * n_ +
                     aivParams_.kCntIndex * rowsPerTaskGroup * n_;
        reductionTileParams_.packedElements = Blaze::Gemm::CeilAlign(
            reductionTileParams_.validRows * aivParams_.curAlignedNInAiv, Blaze::Gemm::BLOCK_BYTE_SIZE);
    }

    __aicore__ inline void operator()() { Run(); }

    __aicore__ inline void DequantAndStore()
    {
        int64_t localM = static_cast<int64_t>(reductionTileParams_.validRows);
        int64_t localN = static_cast<int64_t>(aivParams_.curNL1InAiv);
        int64_t rowStride = static_cast<int64_t>(aivParams_.curAlignedNInAiv);
        int64_t offsetBias = static_cast<int64_t>(aivParams_.nCntIndex * nL1_);

        InitDequant(localM, localN, rowStride);
        CopyBiasToUb(localN, offsetBias);

        int64_t splitNum = AscendC::Std::min(localM, static_cast<int64_t>(DEQUANT_SPLIT_COUNT));
        int64_t mSizeForOnce = CeilDiv(localM, splitNum);
        for (int64_t i = 0; i < splitNum; ++i) {
            if (i * mSizeForOnce >= localM) {
                break;
            }
            int64_t mSize = AscendC::Std::min(mSizeForOnce, localM - i * mSizeForOnce);
            int64_t l0cOffset = i * mSizeForOnce * rowStride;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongId_);
            DequantCompute(mSize, localN, rowStride, l0cOffset);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongId_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongId_);

            int64_t gmOffset = static_cast<int64_t>(cGmOffset_) + i * mSizeForOnce * n_;
            CopyResultToGm(mSize, localN, gmOffset);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongId_);
            pingPongId_ ^= 1U;
        }
        ResetAuxCopyFlags();
    }

private:
    __aicore__ inline void InitDequant(int64_t localM, int64_t localN, int64_t rowStride)
    {
        x2ScaleScalar_ = ReadRawX2ScaleScalar();
        if (hasX1Scale_) {
            AscendC::GlobalTensor<X1ScaleType> x1Scale;
            x1Scale.SetGlobalBuffer(reinterpret_cast<__gm__ X1ScaleType*>(perTokenScaleGmAddr_));
            x1ScaleScalar_ = x1Scale.GetValue(0);
        }

        // Match the existing non-StreamK paths exactly. Fixpipe combines both per-tensor scales and masks the
        // combined value when bias is not handled by the AIV. A post-dequant bias selects the MIX semantics:
        // keep both FP32 scales unmasked, multiply them in X2/X1 order, and add bias last.
        if (!isBias_) {
            x2ScaleScalar_ = hasX1Scale_ ?
                                 MergeAndMaskDequantScale(x2ScaleScalar_, x1ScaleScalar_) :
                                 DecodeMaskedDequantScale(*reinterpret_cast<const uint32_t*>(&x2ScaleScalar_));
        }
        SetupUbLayout(localM, localN, rowStride);
    }

    __aicore__ inline float ReadRawX2ScaleScalar()
    {
        if constexpr (IsSameType<X2ScaleType, float>::value) {
            AscendC::GlobalTensor<float> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleGmAddr_));
            return x2Scale.GetValue(0);
        } else if constexpr (IsSameType<X2ScaleType, bfloat16_t>::value) {
            AscendC::GlobalTensor<uint16_t> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scaleGmAddr_));
            uint16_t raw = x2Scale.GetValue(0);
            uint32_t bits = static_cast<uint32_t>(raw) << 16;
            return *reinterpret_cast<float*>(&bits);
        } else if constexpr (IsSameType<X2ScaleType, uint64_t>::value || IsSameType<X2ScaleType, int64_t>::value) {
            AscendC::GlobalTensor<X2ScaleType> x2Scale;
            x2Scale.SetGlobalBuffer(reinterpret_cast<__gm__ X2ScaleType*>(scaleGmAddr_));
            X2ScaleType rawScale = x2Scale.GetValue(0);
            uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(rawScale));
            return *reinterpret_cast<float*>(&bits);
        }
        return 1.0F;
    }

    __aicore__ inline void SetupUbLayout(int64_t localM, int64_t localN, int64_t rowStride)
    {
        // Auxiliary buffers reuse the K-partial staging area after the in-place reduction. Run()
        // establishes the V->MTE2 dependency before scale/bias can overwrite this region.
        uint64_t offset = static_cast<uint64_t>(localM) * static_cast<uint64_t>(rowStride) * sizeof(WorkspaceType);
        if (isBias_) {
            biasUbOffset_ = offset;
            offset += CeilAlign(static_cast<uint64_t>(localN) * sizeof(float), static_cast<uint64_t>(DATA_BLOCK));
        }

        // The ping-pong buffer holds exactly the maximum rows processed by one dequant split.
        uint64_t outOnceSize = CeilDiv(static_cast<uint64_t>(localM), DEQUANT_SPLIT_COUNT) *
                               CeilAlign(static_cast<uint64_t>(localN), static_cast<uint64_t>(OUT_ALIGN)) *
                               sizeof(OutType);
        dequantPingOffset_ = CeilAlign(offset, static_cast<uint64_t>(AscendC::GetVecLen()));
        offset = dequantPingOffset_ + CeilAlign(outOnceSize, static_cast<uint64_t>(DATA_BLOCK));
        dequantPongOffset_ = CeilAlign(offset, static_cast<uint64_t>(AscendC::GetVecLen()));
    }

    __aicore__ inline void CopyWorkspaceToUb()
    {
        const int64_t rows = static_cast<int64_t>(kCnt_);
        const int64_t cols = static_cast<int64_t>(reductionTileParams_.packedElements);
        // Each split-K partial occupies one fixed-size GM tile; pack its valid prefix contiguously in UB.
        auto ubLayout = Gemm::MakeNDExtLayout(rows, cols, cols);
        auto gmLayout = Gemm::MakeNDExtLayout(rows, cols, static_cast<int64_t>(BLOCK_BASE_M * BLOCK_BASE_N));
        auto workspaceUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, WorkspaceType>(0),
                                                ubLayout);
        auto workspaceGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ WorkspaceType*>(workspaceGmAddr_) +
                                                         reductionTileParams_.workspaceOffset),
            gmLayout);
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGM2UB, workspaceUb, workspaceGm);
    }

    template <class T>
    __aicore__ inline static int64_t AlignedUbPitch(int64_t cols)
    {
        return static_cast<int64_t>(
            CeilAlign(static_cast<uint64_t>(cols) * sizeof(T), static_cast<uint64_t>(DATA_BLOCK)) / sizeof(T));
    }

    __aicore__ inline void CopyBiasToUb(int64_t localN, int64_t offsetBias)
    {
        if (isBias_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
            if (biasDtype_ == DT_FLOAT) {
                CopyBiasToUbTyped<float>(localN, offsetBias);
            } else if (biasDtype_ == DT_FLOAT16) {
                CopyBiasToUbTyped<half>(localN, offsetBias);
            } else {
                CopyBiasToUbTyped<bfloat16_t>(localN, offsetBias);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BIAS_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BIAS_EVENT_ID);
        }
    }

    __aicore__ inline void AcquireAuxUbForWorkspace()
    {
        // The workspace reduction and auxiliary inputs reuse the same UB region. Wait until the
        // previous dequant iteration has finished consuming bias before MTE2 overwrites it.
        if (isBias_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        }
    }

    __aicore__ inline void ReleaseAuxCopyAfterReduce()
    {
        // Do not let bias GM2UB overwrite K partials before the vector reduction completes.
        if (isBias_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        }
    }

    template <class ActualBiasType>
    __aicore__ inline void CopyBiasToUbTyped(int64_t localN, int64_t offsetBias)
    {
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        auto ubLayout = Gemm::MakeNDExtLayout(1, localN, AlignedUbPitch<ActualBiasType>(localN));
        auto gmLayout = Gemm::MakeNDExtLayout(1, localN, localN);
        auto biasUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ActualBiasType>(biasUbOffset_),
                                           ubLayout);
        auto biasGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                               reinterpret_cast<__gm__ ActualBiasType*>(biasGmAddr_) + offsetBias),
                                           gmLayout);
        asc::te::copy(copyGM2UB, biasUb, biasGm);
    }

    __aicore__ inline void DequantCompute(int64_t mSize, int64_t localN, int64_t rowStride, int64_t l0cOffset)
    {
        if (!isBias_ || biasDtype_ == DT_FLOAT) {
            DequantComputeTyped<float>(mSize, localN, rowStride, l0cOffset);
        } else if (biasDtype_ == DT_FLOAT16) {
            DequantComputeTyped<half>(mSize, localN, rowStride, l0cOffset);
        } else {
            DequantComputeTyped<bfloat16_t>(mSize, localN, rowStride, l0cOffset);
        }
    }

    template <class BiasDtype>
    __aicore__ inline void DequantComputeTyped(int64_t mSize, int64_t localN, int64_t rowStride, int64_t l0cOffset)
    {
        uint32_t nDstAligned = static_cast<uint32_t>(
            CeilAlign(static_cast<uint64_t>(localN), static_cast<uint64_t>(OUT_ALIGN)));
        uint64_t dequantOffset = pingPongId_ == 0U ? dequantPingOffset_ : dequantPongOffset_;
        uint16_t mSize16 = static_cast<uint16_t>(mSize);
        uint16_t nSize16 = static_cast<uint16_t>(localN);
        uint32_t nSrcAligned = static_cast<uint32_t>(rowStride);

        // Tensors for the Tile::Dequant chain. x2 is always PER_TENSOR here (the raw scale
        // dtype is resolved by ReadRawX2ScaleScalar into x2ScaleScalar_ before the tile runs), so
        // the x2/x1 vector slots receive fp32 placeholder tensors (never dereferenced).
        auto accTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, WorkspaceType>(
                                                  static_cast<uint64_t>(l0cOffset) * sizeof(WorkspaceType)),
                                              Gemm::MakeNDExtLayout(mSize, localN, rowStride));
        auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(dequantOffset),
                                              Gemm::MakeNDExtLayout(mSize, localN, static_cast<int64_t>(nDstAligned)));
        auto x2DummyTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(0),
                                                  Gemm::MakeNDExtLayout(1, 1, 1));
        auto x1DummyTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(0),
                                                  Gemm::MakeNDExtLayout(1, 1, 1));
        auto biasTensor = MakeBiasTensor<BiasDtype>(localN);

        // Dispatch matrix (design doc §5.2): x2/x1 scalars are kept separate when bias runs
        // on the AIV; without bias InitDequant pre-merges (and 0xFFFFE000-masks) x1 into
        // x2ScaleScalar_ so a single Muls reproduces the fixpipe behavior.
        using B32ToFpDequant = Tile::Dequant<Tile::B32ToFpTag>;
        using DequantMode = Tile::DequantMode;
        if (hasX1Scale_ && isBias_) {
            B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::PER_TENSOR, true, BiasDtype>(
                accTensor, dstTensor, x2DummyTensor, x1DummyTensor, biasTensor, x2ScaleScalar_, x1ScaleScalar_, mSize16,
                nSize16, nSrcAligned, nDstAligned);
        } else if (isBias_) {
            B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::NONE, true, BiasDtype>(
                accTensor, dstTensor, x2DummyTensor, x1DummyTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16, nSize16,
                nSrcAligned, nDstAligned);
        } else {
            B32ToFpDequant::template Run<DequantMode::PER_TENSOR, DequantMode::NONE, false, BiasDtype>(
                accTensor, dstTensor, x2DummyTensor, x1DummyTensor, biasTensor, x2ScaleScalar_, 1.0f, mSize16, nSize16,
                nSrcAligned, nDstAligned);
        }
    }

    // Bias single-row vector (runtime dtype) at biasUbOffset_; placeholder when no bias.
    template <class BiasDtype>
    __aicore__ inline auto MakeBiasTensor(int64_t localN)
    {
        if (isBias_) {
            return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasDtype>(biasUbOffset_),
                                        Gemm::MakeNDExtLayout(1, localN, AlignedUbPitch<BiasDtype>(localN)));
        }
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BiasDtype>(0),
                                    Gemm::MakeNDExtLayout(1, 1, 1));
    }

    __aicore__ inline void CopyResultToGm(int64_t mSize, int64_t localN, int64_t gmOffset)
    {
        uint64_t dequantOffset = pingPongId_ == 0U ? dequantPingOffset_ : dequantPongOffset_;
        uint64_t nDstAligned = CeilAlign(static_cast<uint64_t>(localN), static_cast<uint64_t>(OUT_ALIGN));
        auto ubLayout = Gemm::MakeNDExtLayout(mSize, localN, static_cast<int64_t>(nDstAligned));
        auto gmLayout = Gemm::MakeNDExtLayout(mSize, localN, static_cast<int64_t>(n_));
        auto outUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, OutType>(dequantOffset),
                                          ubLayout);
        auto outGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(cGmAddr_) + gmOffset),
            gmLayout);
        asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), outGm, outUb);
    }

    __aicore__ inline void ResetAuxCopyFlags()
    {
        if (isBias_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BIAS_EVENT_ID);
        }
    }

    static constexpr uint64_t BLOCK_BASE_M = 256UL;
    static constexpr uint64_t BLOCK_BASE_N = 256UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t MAIN_WINDOW = 4UL;
    static constexpr uint64_t UB2GM_SRCGAP_UNIT = 32UL;
    static constexpr uint64_t DEQUANT_SPLIT_COUNT = 4UL;

    bool hasX1Scale_{false};
    float x2ScaleScalar_{1.0f};
    float x1ScaleScalar_{1.0f};
    uint32_t pingPongId_{0U};
    uint64_t biasUbOffset_{0UL};
    uint64_t dequantPingOffset_{0UL};
    uint64_t dequantPongOffset_{0UL};
};
} // namespace Block
} // namespace Epilogue

} // namespace Blaze
