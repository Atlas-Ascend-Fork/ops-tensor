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
 * \file kernel_wqgmm_mix_weight_prologue.h
 * \brief MX weight-quant grouped-matmul mix kernel with its scheduler and weight prologue.
 */
#pragma once

#include <type_traits>

#include "kernel_vec_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#if defined(__NPU_ARCH__)
#include "blaze/gemm/tile/arch35/shift_w4_to_w8.h"
#endif
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/sync.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

/*!
 * \brief Device-side tile enumerator for N-resplit + split-M execution.
 *
 * The host tiling describes N as three consecutive segments: main blocks,
 * first-tail blocks, and second-tail blocks. For each grouped problem this
 * scheduler further splits M around baseM and assigns the resulting M/N tiles
 * to cube cores in a round-robin order. The starting core rotates between
 * groups so uneven expert sizes do not repeatedly load the same cores.
 */
template <typename ProblemShape_>
class BlockSchedulerWqgmmNResplit {
public:
    struct Params {
        uint64_t mainBlockCount;
        uint64_t mainBlockSize;
        uint64_t firstTailBlockCount;
        uint64_t firstTailBlockSize;
        uint64_t secondTailBlockCount;
        uint64_t secondTailBlockSize;
        uint64_t coreNum;
        uint64_t cubeNumBlocksN;
        int32_t baseM;
        uint64_t nSize;
    };

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    __aicore__ inline explicit BlockSchedulerWqgmmNResplit(const Params& params) : params_(params)
    {
        cubeBlockIdx_ = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            cubeBlockIdx_ >>= 1;
        }

        nBlockNum_ = params_.mainBlockCount + params_.firstTailBlockCount + params_.secondTailBlockCount;
        seg01Count_ = params_.mainBlockCount + params_.firstTailBlockCount;
        seg1Base_ = params_.mainBlockCount * params_.mainBlockSize;
        seg2Base_ = seg1Base_ + params_.firstTailBlockCount * params_.firstTailBlockSize;
    }

    // Initialize the per-group scheduling state before enumerating that group's tiles.
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        startBasicBlockId_ = (startBasicBlockId_ + prevTileNum_) % params_.coreNum;
        mSize_ = asc::te::get<MNK_M>(problemShape);
        mBlkNum_ = CeilDiv(mSize_, static_cast<uint64_t>(params_.baseM));
        mStep_ = CeilDiv(mSize_, mBlkNum_);
        tileNum_ = mBlkNum_ * nBlockNum_;

        const uint64_t first = cubeBlockIdx_ >= startBasicBlockId_ ? cubeBlockIdx_ : cubeBlockIdx_ + params_.coreNum;
        tileIdx_ = first - startBasicBlockId_;
        prevTileNum_ = tileNum_;
    }

    // Return the next element coordinate assigned to this cube core.
    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (tileIdx_ >= tileNum_) {
            return false;
        }

        const uint64_t mIdx = tileIdx_ / nBlockNum_;
        const uint64_t nIdx = tileIdx_ % nBlockNum_;
        const uint64_t mOffset = mIdx * mStep_;
        uint64_t nOffset = 0;

        if (nIdx < params_.mainBlockCount) {
            nOffset = nIdx * params_.mainBlockSize;
        } else if (nIdx < seg01Count_) {
            const uint64_t localIdx = nIdx - params_.mainBlockCount;
            nOffset = seg1Base_ + localIdx * params_.firstTailBlockSize;
        } else {
            const uint64_t localIdx = nIdx - seg01Count_;
            nOffset = seg2Base_ + localIdx * params_.secondTailBlockSize;
        }

        AscendC::Std::get<MNK_M>(blockCoord) = static_cast<int64_t>(mOffset);
        AscendC::Std::get<MNK_N>(blockCoord) = static_cast<int64_t>(nOffset);

        tileIdx_ += params_.coreNum;
        return true;
    }

    // Resolve the true M/N tile shape for a previously returned coordinate.
    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const
    {
        const uint64_t mOffset = static_cast<uint64_t>(asc::te::get<MNK_M>(blockCoord));
        const uint64_t nOffset = static_cast<uint64_t>(asc::te::get<MNK_N>(blockCoord));

        const uint64_t mL1Size = mOffset + mStep_ > mSize_ ? mSize_ - mOffset : mStep_;

        uint64_t segSize = params_.secondTailBlockSize;
        if (nOffset < seg1Base_) {
            segSize = params_.mainBlockSize;
        } else if (nOffset < seg2Base_) {
            segSize = params_.firstTailBlockSize;
        }
        const uint64_t nL1Size = nOffset + segSize > params_.nSize ? params_.nSize - nOffset : segSize;
        return {static_cast<int64_t>(mL1Size), static_cast<int64_t>(nL1Size), 0, 0};
    }

private:
    Params params_;

    uint64_t cubeBlockIdx_{0};

    // N-segment constants derived from host tiling.
    uint64_t seg01Count_{0};
    uint64_t seg1Base_{0};
    uint64_t seg2Base_{0};
    uint64_t nBlockNum_{0};

    // Per-group runtime state.
    uint64_t mSize_{0};
    uint64_t mBlkNum_{0};
    uint64_t mStep_{0};
    uint64_t tileNum_{0};
    uint64_t tileIdx_{0};

    // Cross-group rolling state.
    uint64_t startBasicBlockId_{0};
    uint64_t prevTileNum_{0};
};

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;

struct WeightPrologueMxBlockParams {
    uint64_t kSize;
    uint64_t kbL1Size;
    uint64_t nL1Size;
    uint64_t nOffset;
    uint64_t nAlign;
};

// Macro aliases keep this dispatch-policy specialization concise and readable.
#define WQGMM_MX_PROLOGUE_TEMPLATE_PARAM template <class OutType_, class InType_, class BiasType_, bool V3A8w4Fast>

#define WQGMM_MX_PROLOGUE_CLASS GroupedWeightPrologueMx<OutType_, InType_, BiasType_, V3A8w4Fast>

template <class OutType_, class InType_, class BiasType_, bool V3A8w4Fast = false>
class GroupedWeightPrologueMx {
public:
    using DispatchPolicy = GroupedMatmulWithWeightQuantMx;
    using OutType = OutType_;
    using InType = InType_;
    using BiasType = BiasType_;

    static constexpr uint64_t UB_MTE2_BUFFER_NUM = 4;

    struct Params {
        __gm__ InType* ptrB;
    };

    template <typename MmadParams>
    __aicore__ inline explicit GroupedWeightPrologueMx(const MmadParams& params);
    __aicore__ inline GroupedWeightPrologueMx();
    template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void operator()(const GMWeightTensorType_& gmWeightTensor,
                                      const GMScaleBTensorType_& gmScaleBTensor, const GMBiasTensorType_& gmBiasTensor,
                                      uint64_t kbL1Size, uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
                                      uint64_t nAlign);
    // V3 converts the two contiguous logical N segments together; no concat staging is needed.
    __aicore__ inline static uint64_t GetContinuousPairKBlock(uint64_t m, uint64_t n)
    {
        return m <= CONTINUOUS_PAIR_M_THRESHOLD && n * CONTINUOUS_PAIR_SIZE <= CONTINUOUS_PAIR_N_THRESHOLD ?
                   CONTINUOUS_PAIR_K_LARGE :
                   CONTINUOUS_PAIR_K_SMALL;
    }

    template <typename W>
    __aicore__ inline void RunContinuousPairRange(const W& weight, uint64_t m, uint64_t k, uint64_t n, uint64_t offset,
                                                  uint64_t fullN, uint64_t beginK, uint64_t endK)
    {
        static_assert(V3A8w4Fast, "Continuous pair pipeline requires the V3 A8W4 specialization");
        const uint64_t kb = GetContinuousPairKBlock(m, n);
        for (uint64_t pos = beginK; pos < endK; pos += kb) {
            const uint64_t kHalf = kb / CONTINUOUS_PAIR_SIZE;
            const uint64_t split = GetSubBlockIdx() * kHalf;
            const uint64_t realK = Min(kHalf, pos + split < k ? k - pos - split : 0UL);
            const uint64_t inputIndex = ubMte2LoopIdx_;
            auto leftInput = MakeWeight4BitTensor<C0_SIZE_B4>(realK, n);
            ++ubMte2LoopIdx_;
            auto rightInput = MakeWeight4BitTensor<C0_SIZE_B4>(realK, n);
            ++ubMte2LoopIdx_;
            WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (inputIndex & (UB_MTE2_BUFFER_NUM - 1)));
            WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + ((inputIndex + 1) & (UB_MTE2_BUFFER_NUM - 1)));
            CopyContinuousPairInput(weight, leftInput, rightInput, pos + split, offset, fullN, realK, n);
            if (cvLoopIdx_ >= CONTINUOUS_PAIR_CREDIT_LIMIT)
                WaitAicToAiv();
            auto leftOutput = MakeWeight8BitTensor(realK, n);
            ++ubComputeLoopIdx_;
            auto rightOutput = MakeWeight8BitTensor(realK, n);
            --ubComputeLoopIdx_;
            WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
            WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V +
                                        ((ubComputeLoopIdx_ + 1) & (WEIGHT_8BIT_BUFFER_NUM - 1)));
            Tile::ShiftW4ToW8<OutType, InType>::ConvertContinuousPair(leftInput, leftOutput, rightInput, rightOutput);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (inputIndex & (UB_MTE2_BUFFER_NUM - 1)));
            SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + ((inputIndex + 1) & (UB_MTE2_BUFFER_NUM - 1)));
            CopyContinuousPairOutput(leftOutput, rightOutput, realK, n, split);
            ubComputeLoopIdx_ += CONTINUOUS_PAIR_SIZE;
            cvLoopIdx_ += CONTINUOUS_PAIR_SIZE;
            SetAivToAic();
        }
    }

    __aicore__ inline void FinalizeVectorCompute();
    __aicore__ inline void ResumeVectorState()
    {
        for (uint16_t i = 0; i < UB_MTE2_BUFFER_NUM; ++i)
            SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + i);
        for (uint16_t i = 0; i < WEIGHT_8BIT_BUFFER_NUM; ++i)
            SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + i);
        if constexpr (!V3A8w4Fast) {
            for (uint16_t i = 0; i < MX_SCALE_OUTPUT_BUFFER_NUM; ++i)
                SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + i);
        }
    }
    __aicore__ inline ~GroupedWeightPrologueMx();

private:
    template <typename W, typename T>
    __aicore__ inline void CopyContinuousPairInput(const W& weight, const T& leftInput, const T& rightInput,
                                                   uint64_t kOffset, uint64_t offset, uint64_t fullN, uint64_t realK,
                                                   uint64_t n)
    {
        if (realK != 0) {
            auto copy = asc::te::make_copy(Tile::CopyGM2UBWeight{});
            auto left = weight.slice(asc::te::make_coord(kOffset, offset), asc::te::make_shape(realK, n));
            auto right = weight.slice(asc::te::make_coord(kOffset, offset + fullN / CONTINUOUS_PAIR_SIZE),
                                      asc::te::make_shape(realK, n));
            asc::te::copy(copy, leftInput, left);
            asc::te::copy(copy, rightInput, right);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    }

    template <typename T>
    __aicore__ inline void CopyContinuousPairOutput(const T& leftOutput, const T& rightOutput, uint64_t realK,
                                                    uint64_t n, uint64_t split)
    {
        auto copy = asc::te::make_copy(asc::te::copy_ub_to_l1{});
        pairedHalf_ = 0;
        auto leftL1 = MakeL1WeightTensor(realK, n, split);
        pairedHalf_ = 1;
        auto rightL1 = MakeL1WeightTensor(realK, n, split);
        if (realK != 0) {
            // Each K32 slab is contiguous in UB; the destination pitch skips the other branch in L1.
            const int64_t slabElements = static_cast<int64_t>(Align16(n) * asc::te::c0_element<OutType>);
            auto layout = MakeNDExtLayout<OutType>(static_cast<int64_t>(CeilDiv(realK, asc::te::c0_element<OutType>)),
                                                   slabElements, slabElements);
            asc::te::copy(copy, leftL1, asc::te::make_tensor(leftOutput.data(), layout));
            asc::te::copy(copy, rightL1, asc::te::make_tensor(rightOutput.data(), layout));
        }
        SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
        SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + ((ubComputeLoopIdx_ + 1) & (WEIGHT_8BIT_BUFFER_NUM - 1)));
    }

    static constexpr uint64_t CONTINUOUS_PAIR_SIZE = 2UL;
    static constexpr uint64_t CONTINUOUS_PAIR_M_THRESHOLD = 256UL;
    static constexpr uint64_t CONTINUOUS_PAIR_CREDIT_LIMIT = DOUBLE_BUFFER_COUNT * CONTINUOUS_PAIR_SIZE;
    static constexpr uint64_t FAST_PACKED_WEIGHT_KB = 64UL;
    static constexpr uint64_t LEGACY_PACKED_WEIGHT_KB = 80UL;
    __aicore__ inline void SetAivToAic();
    // Fixed offsets pair only with the SwiGLU tile-scoped UB-output MMAD.
    static constexpr uint64_t CONTINUOUS_PAIR_K_LARGE = 512UL;
    static constexpr uint64_t CONTINUOUS_PAIR_K_SMALL = 256UL;
    static constexpr uint64_t CONTINUOUS_PAIR_N_THRESHOLD = 128UL;
    struct V3L1Params {
        const uint64_t* weightOffsets = MxWeightQuantGmmL1Layout::WEIGHT_OFFSETS;
        const uint64_t* biasOffsets = MxWeightQuantGmmL1Layout::BIAS_L1_OFFSETS;
        const uint64_t* scaleBOffsets = MxWeightQuantGmmL1Layout::SCALE_B_OFFSETS;
        decltype(asc::te::make_shape(0L, 0L, 0L, 0L)) l1TileShape = asc::te::make_shape(
            0L, 0L, 0L, static_cast<int64_t>(MxWeightQuantGmmL1Layout::SCALE_K_WINDOW));
        uint16_t readyFlag = 1;
        uint16_t freeFlag = 0;
        bool hasBias = false;
    };
    __aicore__ inline void WaitAicToAiv();
    template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const WeightPrologueMxBlockParams& offsetParam,
                                                        const GMWeightTensorType_& gmWeightTensor,
                                                        const GMScaleBTensorType_& gmScaleBTensor,
                                                        const GMBiasTensorType_& gmBiasTensor);
    template <typename GMScaleBTensorType_>
    __aicore__ inline void HandleMxScale(const WeightPrologueMxBlockParams& offsetParam,
                                         const GMScaleBTensorType_& gmScaleBTensor, uint64_t kOffset);

    __aicore__ inline void WaitVectorToMTE2();
    __aicore__ inline void SetVectorToMTE2();
    template <typename Weight4BitTensorType_, typename Weight8BitTensorType_, typename BiasTensorType_>
    __aicore__ inline void WeightAntiQuantComputeNzNk(const Weight4BitTensorType_& weight4BitTensor,
                                                      const Weight8BitTensorType_& weight8BitTensor,
                                                      const BiasTensorType_& biasInTensor,
                                                      const BiasTensorType_& biasOutTensor, bool processBias);
    template <typename Weight8BitTensorType_, typename L1TensorType_, typename BiasTensorType_,
              typename BiasL1TensorType_>
    __aicore__ inline void CopyWeightToL1(uint64_t mte2RealK, const Weight8BitTensorType_& weight8BitTensor,
                                          const L1TensorType_& l1Tensor, const BiasTensorType_& biasOutTensor,
                                          const BiasL1TensorType_& biasL1Tensor, bool processBias);

    // GM to UB copy function
    template <typename GMWeightBaseTensorType_, typename GMBiasTensorType_, typename Weight4BitTensorType_,
              typename BiasTensorType_>
    __aicore__ inline void CopyGmToUb(uint64_t kOffset, uint64_t mte2RealK, const WeightPrologueMxBlockParams& param,
                                      const GMWeightBaseTensorType_& gmWeightBaseTensor,
                                      const GMBiasTensorType_& gmBiasTensor,
                                      const Weight4BitTensorType_& weight4BitTensor,
                                      const BiasTensorType_& biasInTensor, uint64_t biasNOffset, uint64_t biasNSize,
                                      bool processBias);

    // === Tensor Creation Helper Functions ===
    // Weight tensor creation functions
    template <uint64_t InputC0 = asc::te::c0_element<OutType>>
    __aicore__ inline auto MakeWeight4BitTensor(uint64_t mte2RealK, uint64_t nL1Size);
    __aicore__ inline auto MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size);
    __aicore__ inline auto MakeBiasUbTensor(uint64_t baseOffset, uint64_t slotIdx, uint64_t nL1Size);

    // L1 tensor creation functions
    __aicore__ inline auto MakeL1WeightTensor(uint64_t mte2RealK, uint64_t nL1Size, uint64_t l1SplitOffset);
    __aicore__ inline auto MakeL1BiasTensor(uint64_t nL1Size, uint64_t nOffset);

    uint64_t cvLoopIdx_ = 0;

    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;
    uint64_t scaleComputeLoopIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;

    // === Buffer Size Unit ===
    static constexpr uint64_t KB = 1024;

    // === Pipeline Buffer Configuration ===
    static constexpr uint64_t WEIGHT_8BIT_BUFFER_NUM = QUADRUPLE_BUFFER_NUM; // 4

    // The 20KB input slots are shared by packed weight and ScaleB staging.
    // A 4096-K ScaleB window needs 128 rows * (128 + 32) bytes = 20KB.
    static constexpr uint64_t WEIGHT_4BIT_TOTAL_SIZE = (V3A8w4Fast ? FAST_PACKED_WEIGHT_KB : LEGACY_PACKED_WEIGHT_KB) *
                                                       KB;
    static constexpr uint64_t WEIGHT_4BIT_SINGLE_BUFFER_SIZE = WEIGHT_4BIT_TOTAL_SIZE / UB_MTE2_BUFFER_NUM;

    // Four interleaved 8-bit weight output slots.
    static constexpr uint64_t WEIGHT_8BIT_TOTAL_SIZE = 128 * KB;

    // Bias input/output follow their respective MTE2/compute slot indices.
    static constexpr uint64_t BIAS_UB_SINGLE_BUFFER_SIZE = 512;
    // Match the low-level implementation: when both AIVs participate in the
    // K slice, AIV0 owns at most one 128-element bias vector and AIV1 owns the
    // remainder.  A short K tail that fits entirely on AIV0 lets that AIV
    // process the full bias vector in one fused VF invocation.
    static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128;
    static constexpr uint64_t BIAS_UB_TOTAL_SIZE = QUADRUPLE_BUFFER_NUM * BIAS_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t BIAS_UB_INPUT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE + WEIGHT_8BIT_TOTAL_SIZE;
    static constexpr uint64_t BIAS_UB_OUTPUT_OFFSET = BIAS_UB_INPUT_OFFSET + BIAS_UB_TOTAL_SIZE;

    static constexpr uint64_t MX_SCALE_TRANS_ID_OFFSET = BIAS_UB_OUTPUT_OFFSET + BIAS_UB_TOTAL_SIZE;
    static constexpr uint64_t MX_SCALE_TRANS_ID_BUFFER_SIZE = 1 * KB;
    static constexpr uint64_t MX_SCALE_OUTPUT_OFFSET = MX_SCALE_TRANS_ID_OFFSET + MX_SCALE_TRANS_ID_BUFFER_SIZE;
    static constexpr uint64_t MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE = 16 * KB;
    static constexpr uint64_t MX_SCALE_OUTPUT_BUFFER_NUM = 2;

    // Compile-time verification: active use stays within the 248KB UB hardware limit.
    static_assert(V3A8w4Fast || WEIGHT_4BIT_SINGLE_BUFFER_SIZE >= 128 * 160,
                  "ScaleB input slot must hold 128 padded N rows");
    static_assert(BIAS_UB_SINGLE_BUFFER_SIZE >= 128 * sizeof(BiasType),
                  "Bias slot must hold one AIV's maximum N slice");
    static_assert(MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE >= 128 * 128,
                  "ScaleB output slot must hold one full 4096-K window");
    static_assert(MX_SCALE_OUTPUT_OFFSET + MX_SCALE_OUTPUT_BUFFER_NUM * MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE <= 248 * KB,
                  "UB buffer total must not exceed 248KB hardware limit");

    // === UB Buffer Base Offsets ===
    static constexpr uint64_t WEIGHT_4BIT_INIT_OFFSET = 0;
    static constexpr uint64_t WEIGHT_8BIT_INIT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE;

    // === UB Buffer Offset Arrays (Compile-time computed for fast lookup) ===
    static constexpr uint64_t WEIGHT_4BIT_OFFSETS[4] = {WEIGHT_4BIT_INIT_OFFSET + 0 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 1 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 2 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
                                                        WEIGHT_4BIT_INIT_OFFSET + 3 * WEIGHT_4BIT_SINGLE_BUFFER_SIZE};

    // === Event IDs for Pipeline Synchronization ===
    constexpr static TEventID VEC_EVENT_ID_V_TO_MTE2 = 0;
    constexpr static TEventID VEC_EVENT_ID_MTE3_TO_V = 0;
    constexpr static TEventID VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V = 4;
    constexpr static TEventID EVENT_ID_MTE2_TO_V = 0;

    // === Cross-Core Synchronization Flags ===
    // In 1:2 AIC:AIV MIX mode, mode 4 maps the same AIV-local flag ID to the
    // AIC flag at the base ID for AIV0 and at the base ID + 16 for AIV1.
    static constexpr uint8_t SYNC_AIC_AIV_MODE = 4;

    static constexpr uint64_t MX_GROUP_SIZE = 32;

    // Read-only views into the owning kernel's L1 configuration.
    const uint64_t* weightOffsets_;
    const uint64_t* biasOffsets_;
    const uint64_t* scaleBOffsets_;
    uint64_t scaleKWindowSize_;
    uint16_t readyFlag_;
    uint16_t freeFlag_;
    bool hasBias_ = false;
    uint64_t pairedHalf_ = 0;
};

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename MmadParams>
__aicore__ inline WQGMM_MX_PROLOGUE_CLASS::GroupedWeightPrologueMx(const MmadParams& params)
    : weightOffsets_(params.weightOffsets),
      biasOffsets_(params.biasOffsets),
      scaleBOffsets_(params.scaleBOffsets),
      scaleKWindowSize_(AscendC::Std::get<3>(params.l1TileShape)),
      readyFlag_(params.readyFlag),
      freeFlag_(params.freeFlag),
      hasBias_(V3A8w4Fast ? false : params.hasBias)
{
    ResumeVectorState();
    if constexpr (!V3A8w4Fast) {
        // Reload on every construction: the preceding tile's FIX/epilogue may have overwritten all UB.
        auto transIdLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, Tile::MX_SCALE_TRANS_ID_SIZE);
        auto transIdGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ uint16_t*)Tile::MX_SCALE_TRANS_ID), transIdLayout);
        auto transIdUb = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(MX_SCALE_TRANS_ID_OFFSET), transIdLayout);
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGM2UB, transIdUb, transIdGm);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    }
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQGMM_MX_PROLOGUE_CLASS::GroupedWeightPrologueMx() : GroupedWeightPrologueMx(V3L1Params{})
{
    static_assert(V3A8w4Fast, "The default constructor is reserved for the V3 A8W4 prologue");
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::ComputeBasicBlockAivNdKnNzNk(const WeightPrologueMxBlockParams& param,
                                                                             const GMWeightTensorType_& gmWeightTensor,
                                                                             const GMScaleBTensorType_& gmScaleBTensor,
                                                                             const GMBiasTensorType_& gmBiasTensor)
{
    // Setup loop constants
    const uint64_t kMte2BaseSize = param.kbL1Size >> 1;
    const uint64_t l1SplitOffset = GetSubBlockIdx() * kMte2BaseSize;

    // Main processing loop
    for (uint64_t kOffset = 0; kOffset < param.kSize; kOffset += param.kbL1Size, cvLoopIdx_++) {
        // Calculate K block sizes
        uint64_t l1RealLen = (kOffset + param.kbL1Size) > param.kSize ? param.kSize - kOffset : param.kbL1Size;
        uint64_t mte2RealK = GetSubBlockIdx() == 0     ? Min(kMte2BaseSize, l1RealLen) :
                             l1RealLen > kMte2BaseSize ? l1RealLen - kMte2BaseSize :
                                                         0;

        // The paired AIC keeps two tiles in flight.  Match the low-level
        // credit protocol: the first two iterations need no credit, all later
        // iterations consume one completion token before touching L1.
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }

        // ScaleB is refreshed once per 4096-K window and shares the quad MTE2
        // input slots with weight.  Handle it before taking the weight slot.
        HandleMxScale(param, gmScaleBTensor, kOffset);

        bool isBiasSingleVector = mte2RealK == l1RealLen;
        uint64_t biasNSize = 0;
        uint64_t biasNOffset = 0;
        if (hasBias_ && kOffset == 0) {
            if (isBiasSingleVector) {
                biasNSize = param.nL1Size;
            } else {
                uint64_t vec0NSize = Min(param.nL1Size, MX_BIAS_SINGLE_VECTOR_SIZE);
                biasNSize = GetSubBlockIdx() == 0 ? vec0NSize : param.nL1Size - vec0NSize;
            }
            biasNOffset = GetSubBlockIdx() * (biasNSize != 0 ? MX_BIAS_SINGLE_VECTOR_SIZE : 0);
        }

        // Create tensors using the post-ScaleB MTE2 slot indices.
        auto weight4BitTensor = MakeWeight4BitTensor(mte2RealK, param.nL1Size);
        auto weight8BitTensor = MakeWeight8BitTensor(mte2RealK, param.nL1Size);
        auto l1Tensor = MakeL1WeightTensor(mte2RealK, param.nL1Size, l1SplitOffset);
        auto biasInTensor = MakeBiasUbTensor(BIAS_UB_INPUT_OFFSET, ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1),
                                             biasNSize);
        auto biasOutTensor = MakeBiasUbTensor(BIAS_UB_OUTPUT_OFFSET, ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1),
                                              biasNSize);
        auto biasL1Tensor = MakeL1BiasTensor(biasNSize, biasNOffset);
        bool processBias = hasBias_ && kOffset == 0 && biasNSize != 0;

        // Pipeline: Stage 1 - Wait and Load from GM
        WaitVectorToMTE2();

        CopyGmToUb(kOffset + GetSubBlockIdx() * kMte2BaseSize, mte2RealK, param, gmWeightTensor, gmBiasTensor,
                   weight4BitTensor, biasInTensor, biasNOffset, biasNSize, processBias);

        // Pipeline: Stage 2 - Compute.  AIC credit was consumed above so the
        // ScaleB load can overlap the same AIC window as weight conversion.
        WeightAntiQuantComputeNzNk(weight4BitTensor, weight8BitTensor, biasInTensor, biasOutTensor, processBias);
        SetVectorToMTE2();
        ubMte2LoopIdx_++;

        // Pipeline: Stage 3 - Copy to L1 and Signal
        CopyWeightToL1(mte2RealK, weight8BitTensor, l1Tensor, biasOutTensor, biasL1Tensor, processBias);
        SetAivToAic();
    }
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename GMScaleBTensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::HandleMxScale(const WeightPrologueMxBlockParams& param,
                                                              const GMScaleBTensorType_& gmScaleBTensor,
                                                              uint64_t kOffset)
{
    if (kOffset % scaleKWindowSize_ != 0) {
        return;
    }

    uint64_t scaleL1Buf = scaleL1BufIdx_ & (DOUBLE_BUFFER_COUNT - 1);
    scaleL1BufIdx_++;
    uint64_t vec0NSize = param.nL1Size > BLOCK_CUBE ? CeilAlign(param.nL1Size / 2, static_cast<uint64_t>(BLOCK_CUBE)) :
                                                      param.nL1Size;
    uint64_t vecNSize = GetSubBlockIdx() == 0 ? vec0NSize : param.nL1Size - vec0NSize;
    uint64_t vecNOffset = GetSubBlockIdx() * vec0NSize;
    if (vecNSize == 0) {
        return;
    }

    uint64_t scaleWindowK = Min(scaleKWindowSize_, param.kSize - kOffset);
    uint64_t scaleKReal = Align64(scaleWindowK) / MX_GROUP_SIZE;
    auto gmScaleSlice = gmScaleBTensor.slice(asc::te::make_coord(kOffset / MX_GROUP_SIZE, param.nOffset + vecNOffset),
                                             asc::te::make_shape(scaleKReal, vecNSize));

    WaitVectorToMTE2();
    // Keep the 160-byte physical row stride while slicing to the active K/N.
    auto scaleInputLayout = asc::te::make_frame_layout<asc::te::scaleb_dn_layout_ptn>(
        AscendC::Std::Int<Tile::MX_SCALE_INPUT_ROW_STRIDE>{}, vecNSize);
    auto scaleInputBase = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, fp8_e8m0_t>(
                                                   WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)]),
                                               scaleInputLayout);
    auto scaleInput = scaleInputBase.slice(asc::te::make_coord(0, 0), asc::te::make_shape(scaleKReal, vecNSize));
    auto copyGM2UBMxScale = asc::te::make_copy(Tile::CopyGM2UBMxScale{});
    asc::te::copy(copyGM2UBMxScale, scaleInput, gmScaleSlice);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

    uint64_t scaleOutputBuf = scaleComputeLoopIdx_ & (MX_SCALE_OUTPUT_BUFFER_NUM - 1);
    WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + scaleOutputBuf);
    auto scaleOutputLayout = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(scaleKReal,
                                                                                                             vecNSize);
    auto scaleOutput = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, fp8_e8m0_t>(MX_SCALE_OUTPUT_OFFSET +
                                                                 scaleOutputBuf * MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE),
        scaleOutputLayout);
    auto transIdLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, Tile::MX_SCALE_TRANS_ID_SIZE);
    auto transId = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(MX_SCALE_TRANS_ID_OFFSET), transIdLayout);
    Tile::MxScaleTranspose::Transpose(scaleOutput, scaleInput, transId);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    uint64_t scaleL1Base = scaleBOffsets_[scaleL1Buf];
    // MX ScaleB uses the hardware NN scale layout whose C0 is two E8M0
    // elements.  The generic data-type trait would select the ordinary
    // 32-byte cube C0 and is therefore not valid for nn_layout_ptn.
    auto scaleL1Layout = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<2>>(scaleKReal,
                                                                                                  param.nL1Size);
    auto scaleL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(scaleL1Base),
                                        scaleL1Layout);
    auto scaleL1Slice = scaleL1.slice(asc::te::make_coord(0, vecNOffset), asc::te::make_shape(scaleKReal, vecNSize));
    auto copyUB2L1MxScale = asc::te::make_copy(Tile::CopyUB2L1MxScale{});
    asc::te::copy(copyUB2L1MxScale, scaleL1Slice, scaleOutput);

    SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + scaleOutputBuf);
    SetVectorToMTE2();
    ubMte2LoopIdx_++;
    scaleComputeLoopIdx_++;
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::operator()(const GMWeightTensorType_& gmWeightTensor,
                                                           const GMScaleBTensorType_& gmScaleBTensor,
                                                           const GMBiasTensorType_& gmBiasTensor, uint64_t kbL1Size,
                                                           uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
                                                           uint64_t nAlign)
{
    // Type assertions - __aicore__ guarantees these types are valid
    static_assert(std::is_same_v<OutType, __fp8e4m3> || AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value,
                  "OutType must be fp8_e4m3fn_t");
    static_assert(std::is_same_v<InType, __fp4e2m1x2> || AscendC::IsSameType<InType, fp4x2_e2m1_t>::value ||
                      AscendC::IsSameType<InType, fp4x2_e1m2_t>::value,
                  "InType must be fp4x2_e2m1_t or fp4x2_e1m2_t");
    static_assert(AscendC::IsSameType<BiasType, half>::value || AscendC::IsSameType<BiasType, bfloat16_t>::value,
                  "BiasType must be half or bfloat16_t");

    WeightPrologueMxBlockParams offsetParam = {};
    offsetParam.kSize = kSize;
    offsetParam.nL1Size = nL1Size;
    offsetParam.nOffset = nOffset;
    offsetParam.kbL1Size = kbL1Size;
    offsetParam.nAlign = nAlign;
    ComputeBasicBlockAivNdKnNzNk(offsetParam, gmWeightTensor, gmScaleBTensor, gmBiasTensor);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline WQGMM_MX_PROLOGUE_CLASS::~GroupedWeightPrologueMx()
{
    uint64_t outstandingAicCredits = V3A8w4Fast ? Min(cvLoopIdx_ / CONTINUOUS_PAIR_SIZE,
                                                      static_cast<uint64_t>(DOUBLE_BUFFER_COUNT)) :
                                                  Min(cvLoopIdx_, static_cast<uint64_t>(DOUBLE_BUFFER_COUNT));
    for (uint64_t idx = 0; idx < outstandingAicCredits; ++idx) {
        WaitAicToAiv();
    }
    FinalizeVectorCompute();
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::SetAivToAic()
{
    Sync::NotifyCube<SYNC_AIC_AIV_MODE, PIPE_MTE3>(readyFlag_);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::WaitAicToAiv()
{
    Sync::WaitForCube<SYNC_AIC_AIV_MODE, PIPE_MTE3>(freeFlag_);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::WaitVectorToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::SetVectorToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename GMWeightBaseTensorType_, typename GMBiasTensorType_, typename Weight4BitTensorType_,
          typename BiasTensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::CopyGmToUb(uint64_t kOffset, uint64_t mte2RealK,
                                                           const WeightPrologueMxBlockParams& param,
                                                           const GMWeightBaseTensorType_& gmWeightBaseTensor,
                                                           const GMBiasTensorType_& gmBiasTensor,
                                                           const Weight4BitTensorType_& weight4BitTensor,
                                                           const BiasTensorType_& biasInTensor, uint64_t biasNOffset,
                                                           uint64_t biasNSize, bool processBias)
{
    if (mte2RealK > 0) {
        auto gmSliceTensor = gmWeightBaseTensor.slice(asc::te::make_coord(kOffset, param.nOffset),
                                                      asc::te::make_shape(mte2RealK, param.nL1Size));
        auto copyGM2UBWeight = asc::te::make_copy(Blaze::Gemm::Tile::CopyGM2UBWeight{});
        asc::te::copy(copyGM2UBWeight, weight4BitTensor, gmSliceTensor);
    }
    if (processBias) {
        auto gmBiasSlice = gmBiasTensor.slice(asc::te::make_coord(0, param.nOffset + biasNOffset),
                                              asc::te::make_shape(1, biasNSize));
        auto copyGM2UBBias = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGM2UBBias, biasInTensor, gmBiasSlice);
    }

    // Synchronization point after copy completes
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename Weight4BitTensorType_, typename Weight8BitTensorType_, typename BiasTensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::WeightAntiQuantComputeNzNk(
    const Weight4BitTensorType_& weight4BitTensor, const Weight8BitTensorType_& weight8BitTensor,
    const BiasTensorType_& biasInTensor, const BiasTensorType_& biasOutTensor, bool processBias)
{
    WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));

    Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor, biasInTensor, biasOutTensor, processBias);

    // Set/Wait flags AFTER compute
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <typename Weight8BitTensorType_, typename L1TensorType_, typename BiasTensorType_, typename BiasL1TensorType_>
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::CopyWeightToL1(uint64_t mte2RealK,
                                                               const Weight8BitTensorType_& weight8BitTensor,
                                                               const L1TensorType_& l1Tensor,
                                                               const BiasTensorType_& biasOutTensor,
                                                               const BiasL1TensorType_& biasL1Tensor, bool processBias)
{
    if (likely(mte2RealK > 0)) {
        // Copy weight 8-bit from UB to L1 (inlined from CopyWeight8BitForAligned)
        if constexpr (V3A8w4Fast) {
            const auto shape = asc::te::get_shape(weight8BitTensor.layout());
            const auto stride = asc::te::get_stride(weight8BitTensor.layout());
            const int64_t slabs = AscendC::Std::get<1>(AscendC::Std::get<0>(shape));
            const int64_t slabElements = AscendC::Std::get<1>(AscendC::Std::get<0>(stride));
            auto layout = MakeNDExtLayout<OutType>(slabs, slabElements, slabElements);
            auto copy = asc::te::make_copy(asc::te::copy_ub_to_l1{});
            asc::te::copy(copy, l1Tensor, asc::te::make_tensor(weight8BitTensor.data(), layout));
        } else {
            auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
            asc::te::copy(copyUB2L1, l1Tensor, weight8BitTensor);
        }
    }
    if (processBias) {
        auto copyUB2L1Bias = asc::te::make_copy(asc::te::copy_ub_to_l1{});
        asc::te::copy(copyUB2L1Bias, biasL1Tensor, biasOutTensor);
    }
    SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
    ubComputeLoopIdx_++;
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline void WQGMM_MX_PROLOGUE_CLASS::FinalizeVectorCompute()
{
    for (uint16_t idx = 0; idx < WEIGHT_8BIT_BUFFER_NUM; idx++) {
        WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < UB_MTE2_BUFFER_NUM; idx++) {
        WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + idx);
    }

    if constexpr (!V3A8w4Fast) {
        for (uint16_t idx = 0; idx < MX_SCALE_OUTPUT_BUFFER_NUM; idx++) {
            WaitFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + idx);
        }
    }
}

// === Tensor Creation Helper Function Implementations ===

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
template <uint64_t InputC0>
__aicore__ inline auto WQGMM_MX_PROLOGUE_CLASS::MakeWeight4BitTensor(uint64_t mte2RealK, uint64_t nL1Size)
{
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, InType>(
                                    WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)]),
                                asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<InputC0>>(
                                    static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size)));
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQGMM_MX_PROLOGUE_CLASS::MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size)
{
    constexpr uint64_t SLOT_STRIDE = V3A8w4Fast ? WEIGHT_8BIT_TOTAL_SIZE / WEIGHT_8BIT_BUFFER_NUM :
                                                  AscendC::VECTOR_REG_WIDTH * sizeof(OutType);
    constexpr uint64_t INNER_SIZE = AscendC::VECTOR_REG_WIDTH * (V3A8w4Fast ? 1UL : WEIGHT_8BIT_BUFFER_NUM);
    return asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, OutType>(
            WEIGHT_8BIT_INIT_OFFSET + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)) * SLOT_STRIDE),
        Blaze::Gemm::Weight8BitZnToZnUBLayout<OutType>{}(static_cast<int64_t>(mte2RealK),
                                                         static_cast<int64_t>(Align16(nL1Size)), INNER_SIZE));
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQGMM_MX_PROLOGUE_CLASS::MakeBiasUbTensor(uint64_t baseOffset, uint64_t slotIdx,
                                                                 uint64_t nL1Size)
{
    auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
        1, CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE)));
    return asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(baseOffset + slotIdx * BIAS_UB_SINGLE_BUFFER_SIZE),
        layout);
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQGMM_MX_PROLOGUE_CLASS::MakeL1WeightTensor(uint64_t mte2RealK, uint64_t nL1Size,
                                                                   uint64_t l1SplitOffset)
{
    auto l1BaseLayout = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<OutType>>(
        mte2RealK, V3A8w4Fast ? nL1Size * CONTINUOUS_PAIR_SIZE : nL1Size);
    auto l1BaseTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, OutType>(
            weightOffsets_[(V3A8w4Fast ? cvLoopIdx_ / CONTINUOUS_PAIR_SIZE : cvLoopIdx_) & (DOUBLE_BUFFER_COUNT - 1)]),
        l1BaseLayout);
    auto tile = l1BaseTensor.slice(asc::te::make_coord(l1SplitOffset, V3A8w4Fast ? pairedHalf_ * nL1Size : 0),
                                   asc::te::make_shape(mte2RealK, nL1Size));
    if constexpr (V3A8w4Fast) {
        const int64_t slabElements = static_cast<int64_t>(Align16(nL1Size) * asc::te::c0_element<OutType>);
        const auto stride = asc::te::get_stride(tile.layout());
        const int64_t pitch = AscendC::Std::get<1>(AscendC::Std::get<0>(stride));
        auto layout = MakeNDExtLayout<OutType>(static_cast<int64_t>(CeilDiv(mte2RealK, asc::te::c0_element<OutType>)),
                                               slabElements, pitch);
        return asc::te::make_tensor(tile.data(), layout);
    } else {
        return tile;
    }
}

WQGMM_MX_PROLOGUE_TEMPLATE_PARAM
__aicore__ inline auto WQGMM_MX_PROLOGUE_CLASS::MakeL1BiasTensor(uint64_t nL1Size, uint64_t nOffset)
{
    auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
        1, CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE)));
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(
                                    biasOffsets_[cvLoopIdx_ & (DOUBLE_BUFFER_COUNT - 1)] + nOffset * sizeof(BiasType)),
                                layout);
}

#undef WQGMM_MX_PROLOGUE_CLASS
#undef WQGMM_MX_PROLOGUE_TEMPLATE_PARAM

// Macro aliases keep long template declarations and definitions readable.
#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM                                              \
    template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockEpilogue_, \
              class BlockPrologue_, bool IsSingleMultiSingle_>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS                                                            \
    GmmWeightQuantMxKernel<ProblemShape_, BlockMmad_, BlockScheduler_, BlockEpilogue_, BlockPrologue_, \
                           IsSingleMultiSingle_>

/*!
 * \brief Group-level orchestrator for the MXFP8(M1 input) + FP4(weight input) grouped matmul path.
 *
 * Design reason:
 * - The MMAD compute stage requires A/B to use the same compute representation.
 * - B is stored as FP4 in GM, but AIC cannot directly convert FP4 to FP8E4M3 inside MMAD.
 * - Therefore AIV runs prologue first to convert packed FP4 weight tiles into FP8E4M3-compatible B', then AIC
 *   consumes A, B', scaleA and scaleB for accumulation.
 *
 * Specific flow:
 * - Outer loop reads cumulative M (type 0), group M counts (type 1), or [expert index, M count] rows (type 2).
 * - For each group g:
 *   C_g = MatmulMX(A_g, B'_g, scaleA_g, scaleB_g)
 *   where A_g/B'_g are consumed by MMAD with a consistent compute type, and scaleA_g/scaleB_g are MX scales.
 * - AIC branch: tile scheduling + block MMAD.
 * - AIV branch: tile prologue (FP4E2M1/FP4E1M2 -> FP8E4M3) for B.
 *
 * Key constraints:
 * 1) groupList has groupNum rows; type 2 uses two columns and places empty groups after nonempty groups,
 *    with each partition ordered by expert index.
 * 2) BlockMmad and BlockPrologue must use the same dispatch policy so they agree on tile boundaries and sync points.
 *
 * When to use:
 * - Use this kernel for the weight-quant grouped matmul path where weights are stored in FP4 and must be consumed
 *   by MMAD after prologue conversion to an A-compatible compute type.
 */
GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
class GmmWeightQuantMxKernel {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockPrologue = BlockPrologue_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    static constexpr bool HAS_EPILOGUE = !AscendC::Std::is_same_v<BlockEpilogue, void>;
    using EpilogueParamsOwner = AscendC::Std::conditional_t<HAS_EPILOGUE, BlockEpilogue, BlockPrologue>;
    using Scheduler = Block::BlockSchedulerGmmSwatWithTailSplit;
    using ScaleType = ScaleAType;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    static constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;

    static_assert(AscendC::Std::is_same_v<AType, fp8_e4m3fn_t>, "AType must be fp8_e4m3fn_t");
    static_assert(AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>,
                  "BType must be fp4x2_e2m1_t or fp4x2_e1m2_t");
    static_assert(AscendC::Std::is_same_v<ScaleAType, fp8_e8m0_t> && AscendC::Std::is_same_v<ScaleBType, fp8_e8m0_t>,
                  "ScaleAType and ScaleBType must be fp8_e8m0_t");
    static_assert(AscendC::Std::is_one_of_v<BiasType, half, bfloat16_t>, "BiasType must be half or bfloat16_t");
    static_assert(HAS_EPILOGUE || AscendC::Std::is_one_of_v<CType, half, bfloat16_t>,
                  "CType must be half or bfloat16_t");
    static_assert(AscendC::Std::is_same_v<typename BlockPrologue::OutType, AType> &&
                      AscendC::Std::is_same_v<typename BlockPrologue::InType, BType> &&
                      AscendC::Std::is_same_v<typename BlockPrologue::BiasType, BiasType>,
                  "BlockPrologue types must match BlockMmad types");
    static_assert(
        HAS_EPILOGUE ||
            AscendC::Std::is_same_v<typename BlockMmad::DispatchPolicy, typename BlockPrologue::DispatchPolicy>,
        "BlockMmad and BlockPrologue must use the same dispatch policy");
    static_assert(AscendC::Std::is_same_v<typename BlockMmad::DispatchPolicy::ScheduleType, KernelWqgmmMxMix>,
                  "GMM MX A8W4 requires KernelWqgmmMxMix scheduling");

    struct GmParams {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
        typename BlockPrologue::Params prologue;
        GM_ADDR ptrGroupList;
        uint32_t groupListType;
        uint32_t hasBias;
    };

    struct UbParams {
        GM_ADDR x{nullptr};
        GM_ADDR weight{nullptr};
        GM_ADDR xScale{nullptr};
        GM_ADDR weightScale{nullptr};
        GM_ADDR groupList{nullptr};
        int64_t m{0L};
        int64_t n{0L};
        int64_t k{0L};
        uint32_t groupNum{0};
        uint32_t groupListType{0};
        typename Scheduler::Params scheduler{};
        typename EpilogueParamsOwner::Params epilogue{};
    };

    using Params = AscendC::Std::conditional_t<HAS_EPILOGUE, UbParams, GmParams>;

    __aicore__ inline GmmWeightQuantMxKernel() = default;
    __aicore__ inline void operator()(const Params& params);

private:
    AscendC::Std::conditional_t<HAS_EPILOGUE, AscendC::Std::tuple<>, typename BlockMmad::Params> mmadParams_{};

    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t& kaL1Size, uint64_t& kbL1Size,
                                             uint64_t& l0KSize) const
    {
        constexpr uint64_t L1_K_CONFIG_512 = 512;
        constexpr uint64_t L1_K_CONFIG_256 = 256;
        constexpr uint64_t L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
        constexpr uint64_t L0_K_SMALL_TILE_THRESHOLD = 128;
        constexpr uint64_t L0_K_SMALL_TILE_SIZE = 256;
        constexpr uint64_t L0_K_LARGE_TILE_SIZE = 128;
        constexpr uint64_t A_L1_MODEL_SINGLE_BUF_SIZE = 80 * 1024;
        constexpr uint64_t A_B_LOAD_BALANCE_FACTOR = 2;
        constexpr uint64_t L1_K_DYNAMIC_CONFIG_M_THRESHOLD = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType) /
                                                              L1_K_CONFIG_512 / AscendC::BLOCK_CUBE) *
                                                             AscendC::BLOCK_CUBE;

        kbL1Size = (mL1Size <= L1_K_DYNAMIC_CONFIG_M_THRESHOLD && nL1Size <= L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                       L1_K_CONFIG_512 :
                       L1_K_CONFIG_256;

        uint64_t mL1Align = CeilAlign(mL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        uint64_t kaDepth = CeilDiv(nL1Size, mL1Align * A_B_LOAD_BALANCE_FACTOR);
        kaDepth = kaDepth == 0 ? 1 : kaDepth;

        uint64_t modelMaxKaDepth = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType)) / (mL1Align * kbL1Size);
        uint64_t actualMaxKaDepth = (mmadParams_.aBufferBytes / sizeof(AType)) / (mL1Align * kbL1Size);
        uint64_t maxKaDepth = modelMaxKaDepth < actualMaxKaDepth ? modelMaxKaDepth : actualMaxKaDepth;
        // The scheduler guarantees one kb tile fits; preserve the original zero-depth guard.
        maxKaDepth = maxKaDepth == 0 ? 1 : maxKaDepth;
        kaDepth = kaDepth < maxKaDepth ? kaDepth : maxKaDepth;
        kaL1Size = kaDepth * kbL1Size;
        l0KSize = (mL1Size <= L0_K_SMALL_TILE_THRESHOLD && nL1Size <= L0_K_SMALL_TILE_THRESHOLD) ?
                      L0_K_SMALL_TILE_SIZE :
                      L0_K_LARGE_TILE_SIZE;
    }

    static constexpr uint32_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint32_t GROUP_LIST_TYPE_LENGTH = 1U;
    static constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2U;
    static constexpr uint32_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2U;
    static constexpr uint32_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1U;
    static constexpr int32_t PROBLEM_SHAPE_K_INDEX = 1;
    static constexpr int32_t PROBLEM_SHAPE_N_INDEX = 2;
    static constexpr int32_t PROBLEM_SHAPE_GROUP_NUM_INDEX = 3;

    int64_t preOffset_{0};
    uint32_t groupListType_{GROUP_LIST_TYPE_OFFSET};

    __gm__ typename BlockMmad::AType* xGm_;
    // Single-weight mode stores a data base; multi-weight mode stores the tensor-list descriptor.
    __gm__ typename BlockMmad::ScaleBType* antiquantScaleGm_;
    __gm__ typename BlockMmad::CType* yGm_;
    __gm__ typename BlockMmad::ScaleAType* perTokenScaleGm_;

    // Single-weight mode stores a data base; multi-weight mode stores the tensor-list descriptor.
    __gm__ typename BlockPrologue::InType* weightGm_;
    __gm__ typename BlockMmad::BiasType* biasGm_;

    __gm__ int64_t* groupListGm_;

    using TensorLayoutGroupList = typename asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>;

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& groupList, uint32_t loopIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = groupList[loopIdx];
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            splitValue = groupList[loopIdx];
            preOffset_ += splitValue;
        } else {
            const uint32_t splitValueIdx = loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE +
                                           SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = groupList[splitValueIdx];
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    template <typename T_>
    __aicore__ inline __gm__ T_* GetTensorAddrFromTensorList(uint32_t tensorIdx, __gm__ T_* tensorListAddr) const
    {
        AscendC::ListTensorDesc tensorList(reinterpret_cast<__gm__ void*>(tensorListAddr));
        return tensorList.GetDataPtr<T_>(tensorIdx);
    }

public:
    // GM scaleA is row-major [M, ceil(K/64), 2]; only the L1 staging layout is ZZ.
    __aicore__ inline static auto MakeScaleATensor(const Params& params, int64_t mOffset, int64_t tileMOffset,
                                                   int64_t tileM)
    {
        const int64_t scaleK = CeilDiv(params.k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ScaleType*>(params.xScale) +
                                                         (mOffset + tileMOffset) * scaleK),
            asc::te::make_frame_layout<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(tileM,
                                                                                                               scaleK));
    }

private:
    __aicore__ inline void RunEpilogue(const Params& params)
    {
        static_assert(AscendC::Std::is_same_v<typename BlockMmad::DispatchPolicy, GroupedMatmulSwigluWithWeightQuantMx>,
                      "Deferred MX epilogue requires AIC ScaleB/deferred UB policy");
        static_assert(AscendC::Std::is_same_v<LayoutB, asc::te::zn_layout_ptn> &&
                          AscendC::Std::is_same_v<LayoutScaleB, asc::te::scaleb_dn_layout_ptn>,
                      "MX A8W4 SwiGLU only supports transposed WeightNZ and weight scale.");
        static_assert(BlockEpilogue::ENABLE_SMOOTH_SCALE,
                      "A8W4 SwiGLU requires the epilogue weight-scale restoration and smoothing specialization.");

        if ASCEND_IS_AIC {
            BlockMmad blockMmad;
            RunGroups<true>(params, &blockMmad, nullptr, nullptr);
        } else {
            auto settings = params.epilogue;
            settings.inputScale = WEIGHT_PROLOGUE_INPUT_SCALE;
            BlockEpilogue epilogue;
            epilogue.Init(settings);
            BlockPrologue prologue;
            RunGroups<false>(params, nullptr, &prologue, &epilogue);
        }
    }

private:
    static constexpr int64_t WEIGHT_CACHE_K_ALIGNMENT = 256L;
    static constexpr uint32_t CROSS_CORE_MODE = Sync::SYNC_MODE_INTRA;
    static constexpr uint32_t RESULT_READY_FLAG = 2U;
    static constexpr uint32_t RESULT_CONSUMED_FLAG = 3U;
    static constexpr uint32_t PACKED_FP4_ELEMENTS = 2U;
    static constexpr uint32_t SWIGLU_HALF_COUNT = 2U;
    static constexpr float WEIGHT_PROLOGUE_INPUT_SCALE = 64.0F;

    __aicore__ inline void SignalResultReady()
    {
        Sync::NotifyVector<CROSS_CORE_MODE, PIPE_FIX>(RESULT_READY_FLAG, true);
    }

    __aicore__ inline void WaitResultConsumed()
    {
        Sync::WaitForVector<CROSS_CORE_MODE, PIPE_FIX>(RESULT_CONSUMED_FLAG, true);
    }

    struct TileState {
        int64_t rows{0L}, cols{0L}, row{0L}, col{0L}, groupRows{0L}, groupOffset{0L};
        uint32_t expert{0};
        bool valid{false};
    };
    AscendC::Std::conditional_t<HAS_EPILOGUE, TileState, AscendC::Std::tuple<>> previous_;

    __aicore__ inline void FlushCube(const TileState& tile, BlockMmad* blockMmad)
    {
        WaitResultConsumed();
        BlockEpilogue epilogue;
        auto output = epilogue.GetAccumulatorTensor({tile.rows, tile.cols});
        blockMmad->CopyContinuousResultToUb(output, tile.rows, tile.cols);
        SignalResultReady();
    }

    __aicore__ inline void ProcessEpilogue(const Params& params, const TileState& tile, BlockEpilogue* epilogueState)
    {
        const int64_t halfN = params.n / SWIGLU_HALF_COUNT;
        const int64_t scaleN = CeilDiv(halfN, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto& epilogue = *epilogueState;
        epilogue.UpdateNextProblem(asc::te::make_shape(tile.groupRows, halfN, params.k));
        epilogue.UpdateGlobalAddr({tile.groupOffset * halfN, tile.groupOffset * scaleN});
        epilogue.SetSmoothScaleOffset(static_cast<int64_t>(tile.expert) * halfN + tile.col);
        epilogue.PrefetchSmoothScale(static_cast<uint32_t>(tile.cols));
        Sync::WaitForCube<CROSS_CORE_MODE, PIPE_V>(RESULT_READY_FLAG);
        epilogue(asc::te::make_shape(tile.rows, tile.cols, 0L, 0L),
                 {tile.row * halfN + tile.col, tile.row * scaleN + tile.col / static_cast<int64_t>(MXFP_DIVISOR_SIZE) *
                                                                       static_cast<int64_t>(MXFP_MULTI_BASE_SIZE)});
    }

    __aicore__ inline void ConsumePrevious(const Params& params, BlockPrologue& prologue, BlockEpilogue* epilogue)
    {
        prologue.FinalizeVectorCompute();
        Sync::NotifyCube<CROSS_CORE_MODE, PIPE_V>(RESULT_CONSUMED_FLAG);
        ProcessEpilogue(params, previous_, epilogue);
        prologue.ResumeVectorState();
    }

    template <bool IsAic>
    __aicore__ inline void RunGroups(const Params& params, BlockMmad* blockMmad, BlockPrologue* prologue,
                                     BlockEpilogue* epilogue)
    {
        if (params.groupListType > 1) {
            return;
        }
        const int64_t halfN = params.n / SWIGLU_HALF_COUNT;
        const int64_t scaleK = CeilDiv(params.k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto groupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupList)),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(params.groupNum)));
        Scheduler scheduler(params.scheduler);
        groupListType_ = params.groupListType;
        preOffset_ = 0;
        for (uint32_t expert = 0; expert < params.groupNum; ++expert) {
            const int64_t groupOffset = preOffset_;
            const int64_t groupM = GetSplitValueFromGroupList(groupList, expert);
            if (groupM == 0) {
                continue;
            }
            const int64_t mBlocks = CeilDiv(groupM, static_cast<int64_t>(params.scheduler.baseM));
            const int64_t balancedM = CeilAlign(CeilDiv(groupM, mBlocks), static_cast<int64_t>(BLOCK_CUBE));
            scheduler.SetBaseM(static_cast<int32_t>(balancedM));
            scheduler.UpdateNextProblem(asc::te::make_shape(groupM, halfN, params.k, 1L));
            typename Scheduler::BlockInfo block;
            while (scheduler.GetNextBlock(block)) {
                DispatchBlock<IsAic>(params, blockMmad, prologue, epilogue, block, expert, groupOffset, groupM, scaleK);
            }
        }
        if (previous_.valid) {
            if constexpr (IsAic) {
                FlushCube(previous_, blockMmad);
            } else {
                ConsumePrevious(params, *prologue, epilogue);
            }
        }
    }

    template <bool IsAic>
    __aicore__ inline void DispatchBlock(const Params& params, BlockMmad* blockMmad, BlockPrologue* prologue,
                                         BlockEpilogue* epilogue, const typename Scheduler::BlockInfo& block,
                                         uint32_t expert, int64_t mOffset, int64_t groupM, int64_t scaleK)
    {
        const auto& shape = block.blockShape;
        const auto& coord = block.blockCoord;
        const int64_t tileN = asc::te::get<MNK_N>(shape);
        if (tileN == SWIGLU_MAIN_TILE_N) {
            RunBlock<IsAic, SWIGLU_MAIN_TILE_N>(params, blockMmad, prologue, epilogue, shape, coord, expert, mOffset,
                                                groupM, scaleK);
        } else if (tileN == SWIGLU_TAIL_TILE_N) {
            RunBlock<IsAic, SWIGLU_TAIL_TILE_N>(params, blockMmad, prologue, epilogue, shape, coord, expert, mOffset,
                                                groupM, scaleK);
        } else {
            RunBlock<IsAic, 0>(params, blockMmad, prologue, epilogue, shape, coord, expert, mOffset, groupM, scaleK);
        }
    }

    static constexpr int64_t SWIGLU_MAIN_TILE_N = 128L;
    static constexpr int64_t SWIGLU_TAIL_TILE_N = 64L;

    template <bool IsAic, int64_t HalfN, class Shape, class Coord>
    __aicore__ inline void RunBlock(const Params& params, BlockMmad* blockMmad, BlockPrologue* prologueState,
                                    BlockEpilogue* epilogue, const Shape& shape, const Coord& coord, uint32_t expert,
                                    int64_t mOffset, int64_t groupM, int64_t scaleK)
    {
        const int64_t tileM = asc::te::get<MNK_M>(shape);
        const int64_t tileN = HalfN == 0L ? asc::te::get<MNK_N>(shape) : HalfN;
        const int64_t tileMOffset = asc::te::get<MNK_M>(coord);
        const int64_t tileNOffset = asc::te::get<MNK_N>(coord);
        if constexpr (IsAic) {
            if (previous_.valid)
                FlushCube(previous_, blockMmad);
            auto a = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.x) +
                                                             (mOffset + tileMOffset) * params.k),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(tileM, params.k));
            auto scaleA = MakeScaleATensor(params, mOffset, tileMOffset, tileM);
            auto scaleB = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ScaleType*>(params.weightScale) +
                                                             static_cast<int64_t>(expert) * params.n * scaleK),
                asc::te::make_frame_layout<LayoutScaleB, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(scaleK, params.n));
            blockMmad->ComputeContinuousToL0c(a, scaleA, scaleB, tileM, tileN, tileNOffset);
        } else {
            RunVectorBlock(params, prologueState, epilogue, expert, groupM, tileM, tileN, tileNOffset);
        }
        previous_ = {tileM, tileN, tileMOffset, tileNOffset, groupM, mOffset, expert, true};
    }

    __aicore__ inline void RunVectorBlock(const Params& params, BlockPrologue* prologueState, BlockEpilogue* epilogue,
                                          uint32_t expert, int64_t groupM, int64_t tileM, int64_t tileN,
                                          int64_t tileNOffset)
    {
        auto& prologue = *prologueState;
        auto weight = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType*>(params.weight) +
                                                         static_cast<int64_t>(expert) * params.k * params.n /
                                                             PACKED_FP4_ELEMENTS),
            asc::te::make_frame_layout<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>(params.k, params.n));
        if (groupM <= params.scheduler.baseM && params.k % WEIGHT_CACHE_K_ALIGNMENT == 0) {
            weight.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        }
        if (previous_.valid) {
            const int64_t kb = static_cast<int64_t>(BlockPrologue::GetContinuousPairKBlock(tileM, tileN));
            const int64_t firstK = Min(params.k, kb * static_cast<int64_t>(DOUBLE_BUFFER_COUNT));
            prologue.RunContinuousPairRange(weight, tileM, params.k, tileN, tileNOffset, params.n, 0, firstK);
            ConsumePrevious(params, prologue, epilogue);
            if (firstK < params.k) {
                prologue.RunContinuousPairRange(weight, tileM, params.k, tileN, tileNOffset, params.n, firstK,
                                                params.k);
            }
        } else {
            prologue.RunContinuousPairRange(weight, tileM, params.k, tileN, tileNOffset, params.n, 0, params.k);
        }
    }
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params& params)
{
    if constexpr (HAS_EPILOGUE) {
        RunEpilogue(params);
    } else {
        // Prepare the shared AIC/AIV parameters before constructing either block.
        const bool hasBias = params.hasBias != 0;
        constexpr uint64_t L1_M = 256;
        constexpr uint64_t L1_N = 256;
        constexpr uint64_t L1_K_CONFIG_256 = 256;
        constexpr uint64_t L0_K_CONFIG_128 = 128;
        constexpr uint64_t MX_SCALE_K_WINDOW_SIZE = 4096;
        constexpr uint16_t SYNC_AIV_AIC_FLAG = 0;
        constexpr uint16_t SYNC_AIC_AIV_FLAG = 1;
        constexpr uint64_t SCALE_AL1_OFFSET = 64 * 1024;
        constexpr uint64_t SCALE_BL1_OFFSET = 96 * 1024;
        constexpr uint64_t A_L1_OFFSET = 128 * 1024;
        constexpr uint64_t L1_BUF_OFFSET = 384 * 1024;
        constexpr uint64_t A_L1_BUF_OFFSET = 128 * 1024;
        constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128 * 1024;
        constexpr uint64_t B_L1_SINGLE_BUF_SIZE = 64 * 1024;
        constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = 4 * 1024;
        constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64 * 1024, 380 * 1024};
        mmadParams_.aGmAddr = params.mmad.aGmAddr;
        mmadParams_.scaleAGmAddr = params.mmad.scaleAGmAddr;
        mmadParams_.scaleBGmAddr = params.mmad.scaleBGmAddr;
        mmadParams_.biasGmAddr = params.mmad.biasGmAddr;
        mmadParams_.cGmAddr = params.mmad.cGmAddr;
        mmadParams_.aBufferBytes = A_L1_SINGLE_BUF_SIZE - (hasBias ? BIAS_L1_SINGLE_BUF_SIZE : 0);
        mmadParams_.bBufferBytes = B_L1_SINGLE_BUF_SIZE;
        for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
            const uint64_t biasShift = hasBias && i == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0;
            mmadParams_.weightOffsets[i] = i * L1_BUF_OFFSET;
            mmadParams_.biasOffsets[i] = BIAS_L1_OFFSETS[i];
            mmadParams_.aOffsets[i] = A_L1_OFFSET + i * A_L1_BUF_OFFSET + biasShift;
            mmadParams_.scaleAOffsets[i] = SCALE_AL1_OFFSET + i * L1_BUF_OFFSET + biasShift;
            mmadParams_.scaleBOffsets[i] = SCALE_BL1_OFFSET + i * L1_BUF_OFFSET + biasShift;
        }
        // Actual K blocking is selected for each scheduler tile.
        mmadParams_.l1TileShape = {L1_M, L1_N, L1_K_CONFIG_256, MX_SCALE_K_WINDOW_SIZE};
        mmadParams_.l0TileShape = {L1_M, L1_N, L0_K_CONFIG_128};
        mmadParams_.l1BufferNum = DOUBLE_BUFFER_COUNT;
        mmadParams_.hasBias = hasBias;
        mmadParams_.explicitL1Offsets = true;
        mmadParams_.readyFlag = SYNC_AIC_AIV_FLAG;
        mmadParams_.freeFlag = SYNC_AIV_AIC_FLAG;
        mmadParams_.seedWeightCredits = false;
        groupListType_ = params.groupListType;
        preOffset_ = 0;
        // ScaleB is consumed by AIC MMAD and by the AIV prologue that fills ScaleB L1.
        antiquantScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleBType*>(mmadParams_.scaleBGmAddr);
        // AIC branch consumes A / scales and performs MMAD accumulation.
        if ASCEND_IS_AIC {
            xGm_ = reinterpret_cast<__gm__ typename BlockMmad::AType*>(mmadParams_.aGmAddr);
            perTokenScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleAType*>(mmadParams_.scaleAGmAddr);
            yGm_ = reinterpret_cast<__gm__ typename BlockMmad::CType*>(mmadParams_.cGmAddr);
        }
        // AIV branch prepares transformed weights through the prologue path.
        if ASCEND_IS_AIV {
            weightGm_ = reinterpret_cast<__gm__ typename BlockPrologue::InType*>(params.prologue.ptrB);
            biasGm_ = reinterpret_cast<__gm__ typename BlockMmad::BiasType*>(mmadParams_.biasGmAddr);
        }
        BlockScheduler scheduler(params.scheduler);
        groupListGm_ = reinterpret_cast<__gm__ int64_t*>(params.ptrGroupList);
        const uint64_t kSize = AscendC::Std::get<PROBLEM_SHAPE_K_INDEX>(params.problemShape);
        const uint64_t nSize = AscendC::Std::get<PROBLEM_SHAPE_N_INDEX>(params.problemShape);
        const uint64_t groupNum = AscendC::Std::get<PROBLEM_SHAPE_GROUP_NUM_INDEX>(params.problemShape);
        const uint64_t scaleKSize = CeilDiv(kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const uint64_t groupListSize = groupNum *
                                       (groupListType_ == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1U);
        auto tensorGroupListGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupListGm_),
                                                      TensorLayoutGroupList{}(1, groupListSize));
        using MakeLayoutB = asc::te::frame_layout_format<
            typename BlockMmad::LayoutB, AscendC::Std::Int<asc::te::c0_element<typename BlockMmad::AType>>>;
        using MakeLayoutScaleB = asc::te::frame_layout_format<typename BlockMmad::LayoutScaleB,
                                                              AscendC::Std::Int<SCALE_C0>>;
        const auto scaleBLayout = MakeLayoutScaleB{}(scaleKSize, nSize);
        if ASCEND_IS_AIC {
            using MakeLayoutA = asc::te::frame_layout_format<typename BlockMmad::LayoutA>;
            using MakeLayoutC = asc::te::frame_layout_format<typename BlockMmad::LayoutC>;
            using MakeLayoutScaleA = asc::te::frame_layout_format<typename BlockMmad::LayoutScaleA,
                                                                  AscendC::Std::Int<SCALE_C0>>;
            BlockMmad blockMmad(mmadParams_);
            // ScaleB is now produced by AIV directly in L1. Keep one GM tensor for the BlockMmad API contract; AIC does
            // not resolve, slice, or dereference it in the group/tile loops.
            auto tensorScaleBGm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(antiquantScaleGm_),
                                                       scaleBLayout);
            for (uint32_t loopIdx = 0; loopIdx < groupNum; ++loopIdx) {
                const int64_t groupM = GetSplitValueFromGroupList(tensorGroupListGm, loopIdx);
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && groupM <= 0) {
                    break;
                }
                const uint64_t mSize = static_cast<uint64_t>(groupM);
                if (mSize > 0 && nSize > 0) {
                    const uint64_t groupMOffset = static_cast<uint64_t>(preOffset_ - groupM);
                    auto tensorAGm = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(xGm_ + groupMOffset * kSize),
                        MakeLayoutA{}(mSize, kSize));
                    auto tensorScaleAGm = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(perTokenScaleGm_ + groupMOffset * scaleKSize),
                        MakeLayoutScaleA{}(mSize, scaleKSize));
                    auto tensorCGm = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(yGm_ + groupMOffset * nSize),
                        MakeLayoutC{}(mSize, nSize));

                    scheduler.UpdateNextProblem(asc::te::make_shape(mSize, nSize, kSize));
                    typename BlockScheduler::BlockCoord blockCoord;
                    while (scheduler.GetTileIdx(blockCoord)) {
                        auto blockShape = scheduler.GetBlockShape(blockCoord);
                        auto mOffset = AscendC::Std::get<0>(blockCoord);
                        auto nOffset = AscendC::Std::get<1>(blockCoord);
                        auto mL1Size = AscendC::Std::get<0>(blockShape);
                        auto nL1Size = AscendC::Std::get<1>(blockShape);

                        auto tensorBlockAGm = tensorAGm.slice(asc::te::make_coord(mOffset, 0),
                                                              asc::te::make_shape(mL1Size, kSize));
                        auto tensorBlockScaleAGm = tensorScaleAGm.slice(asc::te::make_coord(mOffset, 0),
                                                                        asc::te::make_shape(mL1Size, scaleKSize));
                        auto tensorBlockCGm = tensorCGm.slice(asc::te::make_coord(mOffset, nOffset),
                                                              asc::te::make_shape(mL1Size, nL1Size));
                        uint64_t kaL1Size = 0;
                        uint64_t kbL1Size = 0;
                        uint64_t l0KSize = 0;
                        CalcDynamicKBlock(mL1Size, nL1Size, kaL1Size, kbL1Size, l0KSize);
                        blockMmad(tensorBlockAGm, tensorBlockScaleAGm, tensorScaleBGm, tensorBlockCGm, kaL1Size,
                                  kbL1Size, l0KSize);
                    }
                }
            }
        } else {
            using TensorLayoutBias = typename BlockMmad::LayoutBias;
            const uint64_t nAlign = CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
            const auto weightLayout = MakeLayoutB{}(static_cast<int64_t>(kSize), static_cast<int64_t>(nAlign));
            const auto biasLayout = asc::te::frame_layout_format<TensorLayoutBias>{}(static_cast<int64_t>(1),
                                                                                     static_cast<int64_t>(nSize));
            constexpr uint64_t FP4_CACHE_LINE_ELEMENT_NUM = 256;
            const bool isWeightCacheLineAligned = kSize % FP4_CACHE_LINE_ELEMENT_NUM == 0;
            const uint64_t baseM = static_cast<uint64_t>(params.scheduler.baseM);
            BlockPrologue blockPrologue(mmadParams_);
            for (uint32_t loopIdx = 0; loopIdx < groupNum; ++loopIdx) {
                const int64_t groupM = GetSplitValueFromGroupList(tensorGroupListGm, loopIdx);
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && groupM <= 0) {
                    break;
                }
                const uint64_t mSize = static_cast<uint64_t>(groupM);
                uint32_t groupIdx = loopIdx;
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
                    groupIdx = static_cast<uint32_t>(tensorGroupListGm[loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
                }
                if (mSize > 0 && nSize > 0) {
                    __gm__ typename BlockPrologue::InType* groupWeightGm = weightGm_;
                    __gm__ typename BlockMmad::ScaleBType* groupScaleBGm = antiquantScaleGm_;
                    __gm__ typename BlockMmad::BiasType* groupBiasGm = biasGm_;
                    if constexpr (IS_SINGLE_MULTI_SINGLE) {
                        groupWeightGm = GetTensorAddrFromTensorList(groupIdx, weightGm_);
                        groupScaleBGm = GetTensorAddrFromTensorList(groupIdx, antiquantScaleGm_);
                        if (params.hasBias != 0) {
                            groupBiasGm = GetTensorAddrFromTensorList(groupIdx, biasGm_);
                        }
                    } else {
                        // Each byte stores two FP4 elements.
                        groupWeightGm += static_cast<uint64_t>(groupIdx) * ((nSize * kSize) >> 1U);
                        groupScaleBGm += static_cast<uint64_t>(groupIdx) * nSize * scaleKSize;
                        if (params.hasBias != 0) {
                            groupBiasGm += static_cast<uint64_t>(groupIdx) * nSize;
                        }
                    }
                    scheduler.UpdateNextProblem(asc::te::make_shape(mSize, nSize, kSize));
                    typename BlockScheduler::BlockCoord blockCoord;
                    auto weightGmTensor = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(groupWeightGm), weightLayout);
                    auto scaleBGmTensor = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(groupScaleBGm), scaleBLayout);
                    const bool disableWeightL2 = mSize <= baseM && isWeightCacheLineAligned;
                    weightGmTensor.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable :
                                                                       asc::te::cache_mode::normal);
                    auto biasGmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupBiasGm),
                                                             biasLayout);
                    while (scheduler.GetTileIdx(blockCoord)) {
                        auto blockShape = scheduler.GetBlockShape(blockCoord);
                        auto nOffset = AscendC::Std::get<1>(blockCoord);
                        auto mL1Size = AscendC::Std::get<0>(blockShape);
                        auto nL1Size = AscendC::Std::get<1>(blockShape);
                        uint64_t kaL1Size = 0;
                        uint64_t kbL1Size = 0;
                        uint64_t l0KSize = 0;
                        CalcDynamicKBlock(mL1Size, nL1Size, kaL1Size, kbL1Size, l0KSize);
                        blockPrologue(weightGmTensor, scaleBGmTensor, biasGmTensor, kbL1Size, kSize, nL1Size, nOffset,
                                      nAlign);
                    }
                }
            }
        }
    }
}

#undef GROUPED_MATMUL_RESPLIT_KERNEL_CLASS
#undef GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
