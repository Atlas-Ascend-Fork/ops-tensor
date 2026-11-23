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
            asc_vf_call<ShiftW4ToW8PairVf>(MakeContinuousWeightParams(leftInput, leftOutput),
                                           MakeContinuousWeightParams(rightInput, rightOutput));
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
    struct ContinuousWeightParams {
        uint64_t loopKNum;
        uint64_t innerLoopNum;
        uint64_t loopKDstStride;
        uint64_t innerDstStride;
        uint64_t nRealSizeAlign;
        __ubuf__ InType_* weight4BitPhyAddr;
        __ubuf__ OutType_* weight8BitPhyAddr;
    };
    static __simd_callee__ inline void DecodeFastWeight(
        AscendC::Reg::RegTensor<int8_t>& decoded, AscendC::Reg::RegTensor<int8_t>& shifted,
        AscendC::Reg::RegTensor<int8_t>& rightShifted, AscendC::Reg::RegTensor<int8_t>& packed,
        AscendC::Reg::RegTensor<int8_t>& laneShift, AscendC::Reg::RegTensor<int8_t>& rightShift,
        AscendC::Reg::RegTensor<int8_t>& mask, AscendC::Reg::MaskReg& active)
    {
        AscendC::Reg::ShiftLeft(shifted, packed, laneShift, active);
        AscendC::Reg::ShiftRight(rightShifted, shifted, rightShift, active);
        AscendC::Reg::And(decoded, rightShifted, mask, active);
    }

    static constexpr bool IS_E1M2 = AscendC::IsSameType<InType, fp4x2_e1m2_t>::value;
    static constexpr uint32_t SHIFT_RIGHT_SIZE = IS_E1M2 ? 0x3U : 0x2U;
    static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4U;
    static constexpr uint32_t AND_MASK = IS_E1M2 ? 0x8EU : 0x9CU;
    static constexpr uint64_t FP4_VALUES_PER_BYTE = 2UL;
    static constexpr uint64_t GATHER_WORD_BYTES = sizeof(uint16_t);
    static constexpr uint64_t PACKED_K32_ROW_BYTES = C0_SIZE_B8 / FP4_VALUES_PER_BYTE;
    static constexpr uint64_t PACKED_K64_ROW_BYTES = C0_SIZE_B4 / FP4_VALUES_PER_BYTE;
    static constexpr uint64_t PACKED_K32_ROW_WORDS = PACKED_K32_ROW_BYTES / GATHER_WORD_BYTES;
    static constexpr uint64_t PACKED_K64_ROW_WORDS = PACKED_K64_ROW_BYTES / GATHER_WORD_BYTES;
    static constexpr uint64_t K32_SLABS_PER_K64 = C0_SIZE_B4 / C0_SIZE_B8;
    static constexpr uint64_t N_ROWS_PER_VF = static_cast<uint64_t>(AscendC::VECTOR_REG_WIDTH) / C0_SIZE_B8;
    static constexpr uint64_t INPUT_GROUP_WORDS = N_ROWS_PER_VF * PACKED_K64_ROW_WORDS;
    static constexpr uint32_t GATHER_WORD_COUNT = static_cast<uint32_t>(AscendC::VECTOR_REG_WIDTH) /
                                                  FP4_VALUES_PER_BYTE / GATHER_WORD_BYTES;
    static constexpr uint16_t INDEX_GROUP_WORD_SHIFT = 3;
    static_assert((1UL << INDEX_GROUP_WORD_SHIFT) == PACKED_K32_ROW_WORDS,
                  "Gather index shift must match the packed K32 row size in uint16 words");

    static __simd_callee__ inline void InitializeGather(AscendC::Reg::RegTensor<uint16_t>& gatherIndex,
                                                        AscendC::Reg::RegTensor<uint16_t>& indexGroup,
                                                        AscendC::Reg::RegTensor<int16_t>& indexShift,
                                                        AscendC::Reg::MaskReg& gatherMask)
    {
        // 64 packed uint16 words (128 bytes) cover eight N rows of one K32 slab.
        // i + ((i >> 3) << 3) skips the other K32 half of every physical K64 row.
        uint32_t gatherWords = GATHER_WORD_COUNT;
        gatherMask = AscendC::Reg::UpdateMask<uint16_t>(gatherWords);
        AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int16_t>&>(gatherIndex), static_cast<int16_t>(0));
        AscendC::Reg::Duplicate(indexShift, static_cast<int16_t>(INDEX_GROUP_WORD_SHIFT), gatherMask);
        AscendC::Reg::ShiftRight(indexGroup, gatherIndex, indexShift, gatherMask);
        AscendC::Reg::ShiftLeft(indexGroup, indexGroup, indexShift, gatherMask);
        AscendC::Reg::Add(gatherIndex, gatherIndex, indexGroup, gatherMask);
    }

    static __simd_callee__ inline void LoadPackedWeight(ContinuousWeightParams& params, uint16_t loopKIdx,
                                                        uint16_t innerLoopIdx, AscendC::Reg::RegTensor<int8_t>& wLoad,
                                                        AscendC::Reg::RegTensor<uint16_t>& gathered,
                                                        AscendC::Reg::RegTensor<uint16_t>& activeGatherIndex,
                                                        AscendC::Reg::RegTensor<uint8_t>& unused,
                                                        AscendC::Reg::MaskReg& gatherMask)
    {
        // Pointer arithmetic is in uint16 words: K64 slab, then eight N rows.
        auto src = reinterpret_cast<__ubuf__ uint16_t*>(params.weight4BitPhyAddr) +
                   (loopKIdx / K32_SLABS_PER_K64) * params.nRealSizeAlign * PACKED_K64_ROW_WORDS +
                   innerLoopIdx * INPUT_GROUP_WORDS;
        AscendC::Reg::Gather(gathered, src, activeGatherIndex, gatherMask);
        auto& packed = reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(gathered);
        // Only the gathered low 128 bytes contribute to the first Interleave result.
        AscendC::Reg::Interleave(reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(wLoad), unused, packed, packed);
    }

    static __simd_callee__ inline void ConvertWeightLoops(
        ContinuousWeightParams& params, AscendC::Reg::RegTensor<int8_t>& wShrReg,
        AscendC::Reg::RegTensor<int8_t>& wAndReg, AscendC::Reg::RegTensor<int8_t>& wLoad,
        AscendC::Reg::RegTensor<int8_t>& wShl, AscendC::Reg::RegTensor<int8_t>& wShr1,
        AscendC::Reg::RegTensor<int8_t>& wAnd, AscendC::Reg::RegTensor<int8_t>& laneShift,
        AscendC::Reg::RegTensor<uint16_t>& gatherIndex, AscendC::Reg::RegTensor<uint16_t>& activeGatherIndex,
        AscendC::Reg::RegTensor<uint16_t>& indexGroup, AscendC::Reg::RegTensor<uint16_t>& gathered,
        AscendC::Reg::RegTensor<uint8_t>& unused, AscendC::Reg::MaskReg& preg, AscendC::Reg::MaskReg& gatherMask)
    {
        for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
            // Keep Gather's UB pointer 32B aligned; select the odd K32 half through its word indices.
            AscendC::Reg::Duplicate(
                indexGroup, static_cast<uint16_t>((loopKIdx % K32_SLABS_PER_K64) * PACKED_K32_ROW_WORDS), gatherMask);
            AscendC::Reg::Add(activeGatherIndex, gatherIndex, indexGroup, gatherMask);

            for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
                LoadPackedWeight(params, loopKIdx, innerLoopIdx, wLoad, gathered, activeGatherIndex, unused,
                                 gatherMask);
                DecodeFastWeight(wAnd, wShl, wShr1, wLoad, laneShift, wShrReg, wAndReg, preg);
                AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
                AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    (__ubuf__ uint8_t*&)params.weight8BitPhyAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd,
                    aregWeightB8Out, preg);
            }
        }
    }

    static __simd_callee__ inline void ShiftW4ToW8ExtendedCallee(ContinuousWeightParams params)
    {
        AscendC::Reg::RegTensor<int8_t> wShrReg;
        AscendC::Reg::RegTensor<int8_t> wShlReg;
        AscendC::Reg::RegTensor<int8_t> wAndReg;
        AscendC::Reg::RegTensor<int8_t> wLoad;
        AscendC::Reg::RegTensor<int8_t> wShl;
        AscendC::Reg::RegTensor<int8_t> wShr1;
        AscendC::Reg::RegTensor<int8_t> wAnd;

        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, SHIFT_RIGHT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, AND_MASK, preg);

        AscendC::Reg::RegTensor<uint16_t> gatherIndex, activeGatherIndex, indexGroup, gathered;
        AscendC::Reg::RegTensor<int16_t> indexShift;
        AscendC::Reg::RegTensor<uint8_t> unused;
        AscendC::Reg::MaskReg gatherMask;
        InitializeGather(gatherIndex, indexGroup, indexShift, gatherMask);

        AscendC::Reg::RegTensor<int8_t> laneShift;
        AscendC::Reg::RegTensor<int8_t> zeroShift;
        AscendC::Reg::Duplicate(zeroShift, static_cast<int8_t>(0), preg);
        AscendC::Reg::Select(laneShift, wShlReg, zeroShift, pregVsel);

        ConvertWeightLoops(params, wShrReg, wAndReg, wLoad, wShl, wShr1, wAnd, laneShift, gatherIndex,
                           activeGatherIndex, indexGroup, gathered, unused, preg, gatherMask);
    }

    static __simd_vf__ inline void ShiftW4ToW8PairVf(ContinuousWeightParams left, ContinuousWeightParams right)
    {
        ShiftW4ToW8ExtendedCallee(left);
        ShiftW4ToW8ExtendedCallee(right);
    }

    template <typename W, typename O>
    __aicore__ static inline ContinuousWeightParams MakeContinuousWeightParams(const W& input, const O& output)
    {
        ContinuousWeightParams params;
        params.weight4BitPhyAddr = (__ubuf__ InType*)input.data().get();
        params.weight8BitPhyAddr = (__ubuf__ OutType*)output.data().get();
        params.loopKNum = AscendC::Std::get<1>(AscendC::Std::get<0>(input.layout().shape())) * K32_SLABS_PER_K64;
        params.nRealSizeAlign = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().shape())) * BLOCK_CUBE;
        params.innerDstStride = AscendC::Std::get<1>(AscendC::Std::get<1>(output.layout().stride()));
        params.innerLoopNum = (params.nRealSizeAlign * C0_SIZE_B8) / static_cast<uint64_t>(asc_get_vf_len());
        params.loopKDstStride = params.innerLoopNum * params.innerDstStride;
        return params;
    }

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

namespace Blaze {
namespace Gemm {
namespace Kernel {

namespace MxPipeline {

template <typename ProblemShape_, bool ColumnFirst = false, bool Incremental = false>
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
        const uint64_t baseM = static_cast<uint64_t>(params_.baseM);
        // Single and complete M tiles do not need a second dynamic division.
        mStep_ = mBlkNum_ <= 1 ? mSize_ : (mSize_ == mBlkNum_ * baseM ? baseM : CeilDiv(mSize_, mBlkNum_));
        tileNum_ = mBlkNum_ * nBlockNum_;

        const uint64_t first = cubeBlockIdx_ >= startBasicBlockId_ ? cubeBlockIdx_ : cubeBlockIdx_ + params_.coreNum;
        tileIdx_ = first - startBasicBlockId_;
        prevTileNum_ = tileNum_;
        // Idle cores never enumerate this group; avoid their division/remainder setup.
        if (Incremental && tileIdx_ < tileNum_) {
            const uint64_t innerCount = ColumnFirst ? mBlkNum_ : nBlockNum_;
            if (innerCount == 1) {
                outerTileIdx_ = tileIdx_;
                innerTileIdx_ = 0;
                outerTileStep_ = params_.coreNum;
                innerTileStep_ = 0;
            } else {
                outerTileIdx_ = tileIdx_ / innerCount;
                innerTileIdx_ = tileIdx_ % innerCount;
                outerTileStep_ = params_.coreNum / innerCount;
                innerTileStep_ = params_.coreNum % innerCount;
            }
        }
    }

    // Return the next element coordinate assigned to this cube core.
    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        BlockShape unusedShape;
        return GetTileIdx(blockCoord, unusedShape);
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord, BlockShape& shape)
    {
        if (tileIdx_ >= tileNum_) {
            return false;
        }

        const uint64_t innerCount = ColumnFirst ? mBlkNum_ : nBlockNum_;
        const uint64_t outer = Incremental ? outerTileIdx_ : tileIdx_ / innerCount;
        const uint64_t inner = Incremental ? innerTileIdx_ : tileIdx_ % innerCount;
        const uint64_t mIdx = ColumnFirst ? inner : outer;
        const uint64_t nIdx = ColumnFirst ? outer : inner;
        const uint64_t mOffset = mIdx * mStep_;
        uint64_t nOffset = 0UL;
        uint64_t segmentSize = params_.secondTailBlockSize;

        if (nIdx < params_.mainBlockCount) {
            nOffset = nIdx * params_.mainBlockSize;
            segmentSize = params_.mainBlockSize;
        } else if (nIdx < seg01Count_) {
            const uint64_t localIdx = nIdx - params_.mainBlockCount;
            nOffset = seg1Base_ + localIdx * params_.firstTailBlockSize;
            segmentSize = params_.firstTailBlockSize;
        } else {
            const uint64_t localIdx = nIdx - seg01Count_;
            nOffset = seg2Base_ + localIdx * params_.secondTailBlockSize;
        }

        AscendC::Std::get<MNK_M>(blockCoord) = static_cast<int64_t>(mOffset);
        AscendC::Std::get<MNK_N>(blockCoord) = static_cast<int64_t>(nOffset);

        shape = {static_cast<int64_t>(mOffset + mStep_ > mSize_ ? mSize_ - mOffset : mStep_),
                 static_cast<int64_t>(nOffset + segmentSize > params_.nSize ? params_.nSize - nOffset : segmentSize), 0,
                 0};
        AdvanceTile();
        return true;
    }

    // Resolve the true M/N tile shape for a previously returned coordinate.
    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const
    {
        const uint64_t mOffset = static_cast<uint64_t>(asc::te::get<MNK_M>(blockCoord));
        const uint64_t nOffset = static_cast<uint64_t>(asc::te::get<MNK_N>(blockCoord));
        const uint64_t rows = mOffset + mStep_ > mSize_ ? mSize_ - mOffset : mStep_;
        uint64_t columns = params_.secondTailBlockSize;
        if (nOffset < seg1Base_) {
            columns = params_.mainBlockSize;
        } else if (nOffset < seg2Base_) {
            columns = params_.firstTailBlockSize;
        }
        columns = nOffset + columns > params_.nSize ? params_.nSize - nOffset : columns;
        return {static_cast<int64_t>(rows), static_cast<int64_t>(columns), 0, 0};
    }

private:
    __aicore__ inline void AdvanceTile()
    {
        tileIdx_ += params_.coreNum;
        if constexpr (!Incremental) {
            return;
        }
        outerTileIdx_ += outerTileStep_;
        innerTileIdx_ += innerTileStep_;
        const uint64_t innerCount = ColumnFirst ? mBlkNum_ : nBlockNum_;
        if (innerTileIdx_ >= innerCount) {
            innerTileIdx_ -= innerCount;
            ++outerTileIdx_;
        }
    }

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
    uint64_t innerTileIdx_{0};
    uint64_t outerTileIdx_{0};
    uint64_t innerTileStep_{0};
    uint64_t outerTileStep_{0};

    // Cross-group rolling state.
    uint64_t startBasicBlockId_{0};
    uint64_t prevTileNum_{0};
};

template <class OutType_, class InType_, class BiasType_, uint32_t Mode_ = 0U>
class GroupedWeightPrologueMx {
public:
    using DispatchPolicy = MxWeightPipelineConfig<Mode_, OutType_, BiasType_>;
    using OutType = OutType_;
    using InType = InType_;
    using BiasType = BiasType_;
    struct Params {
        __gm__ InType* ptrB{nullptr};
    };
    using PipelineConfig = MxWeightPipelineConfig<Mode_, OutType, BiasType>;
    template <typename MmadParams>
    __aicore__ inline explicit GroupedWeightPrologueMx(const MmadParams& params)
    {
        weightOffsets_ = params.weightOffsets;
        biasOffsets_ = params.biasOffsets;
        scaleBOffsets_ = params.scaleBOffsets;
        scaleKWindowSize_ = AscendC::Std::get<SCALE_SHAPE_INDEX>(params.l1TileShape);
        readyFlag_ = params.readyFlag;
        freeFlag_ = params.freeFlag;
        hasBias_ = params.hasBias;
        InitializeVectorBuffers<true>();
    }
    __aicore__ inline explicit GroupedWeightPrologueMx(
        bool hasBias,
        uint64_t l1BufferStage = (MODE == MxPipelineMode::OperandTriple) ? TRIPLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT)
    {
        if constexpr (!(MODE == MxPipelineMode::ParameterDriven)) {
            hasBias_ = hasBias;
            if ((MODE == MxPipelineMode::Aligned) && GetSubBlockIdx() != 0) {
                return;
            }
            if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
                l1BufferStage_ = !hasBias && l1BufferStage == TRIPLE_BUFFER_COUNT ? TRIPLE_BUFFER_COUNT :
                                                                                    DOUBLE_BUFFER_COUNT;
            }
            InitializeVectorBuffers<!(MODE == MxPipelineMode::Aligned)>();
        }
    }
    template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void operator()(const GMWeightTensorType_& gmWeightTensor,
                                      const GMScaleBTensorType_& gmScaleBTensor, const GMBiasTensorType_& gmBiasTensor,
                                      uint64_t kbL1Size, uint64_t kSize, uint64_t nL1Size, uint64_t nOffset,
                                      uint64_t nAlign)
    {
        if ((MODE == MxPipelineMode::Aligned) && GetSubBlockIdx() != 0) {
            return;
        }
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
        if constexpr (!(MODE == MxPipelineMode::ParameterDriven)) {
            offsetParam.kbL1Size = CalcDynamicKBlock(kbL1Size, nL1Size);
        }
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            if (!hasBias_) {
                const uint64_t nAlignL1 = CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
                while (offsetParam.kbL1Size > WEIGHT_FRAGMENT_K &&
                       nAlignL1 * offsetParam.kbL1Size * sizeof(OutType) > QUAD_B_STRIDE) {
                    offsetParam.kbL1Size >>= 1;
                }
            }
        }
        offsetParam.nAlign = nAlign;
        ComputeBasicBlockAivNdKnNzNk(offsetParam, gmWeightTensor, gmScaleBTensor, gmBiasTensor);
    }

    __aicore__ inline ~GroupedWeightPrologueMx()
    {
        uint64_t limit = DOUBLE_BUFFER_COUNT;
        if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            limit = PipelineConfig::WeightStages(hasBias_);
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            limit = creditLimit_;
        }
        const uint64_t outstandingAicCredits = Min(cvLoopIdx_, limit);
        for (uint64_t idx = 0UL; idx < outstandingAicCredits; ++idx) {
            Sync::WaitForCube<(MODE == MxPipelineMode::ParameterDriven) ? SYNC_AIC_AIV_MODE : Sync::SYNC_MODE_INTRA,
                              PIPE_MTE3>((MODE == MxPipelineMode::ParameterDriven) ? freeFlag_ : SYNC_AIV_AIC_FLAG);
        }
        if ((!(MODE == MxPipelineMode::Aligned) || GetSubBlockIdx() == 0)) {
            DrainMxPrologueBuffers(WEIGHT_8BIT_BUFFER_NUM, UB_MTE2_BUFFER_NUM, MX_SCALE_OUTPUT_BUFFER_NUM,
                                   VEC_EVENT_ID_MTE3_TO_V, VEC_EVENT_ID_V_TO_MTE2, VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V);
        }
    }

private:
    static constexpr auto MODE = static_cast<MxPipelineMode>(Mode_);
    __aicore__ static inline void LoadMxScaleTransposeIndices(uint64_t ubOffset, uint16_t readyEvent)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, Tile::MX_SCALE_TRANS_ID_SIZE);
        auto gm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ uint16_t*)Tile::MX_SCALE_TRANS_ID), layout);
        auto ub = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(ubOffset), layout);
        auto copy = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copy, ub, gm);
        SetFlag<HardEvent::MTE2_V>(readyEvent);
        WaitFlag<HardEvent::MTE2_V>(readyEvent);
    }

    template <typename OutType, typename InType, typename Weight4BitTensorType_, typename Weight8BitTensorType_,
              typename BiasTensorType_>
    __aicore__ static inline void ConvertMxPackedWeight(const Weight4BitTensorType_& weight4BitTensor,
                                                        const Weight8BitTensorType_& weight8BitTensor,
                                                        const BiasTensorType_& biasInTensor,
                                                        const BiasTensorType_& biasOutTensor, bool processBias,
                                                        uint64_t eventId)
    {
        WaitFlag<HardEvent::MTE3_V>(eventId);

        Tile::ShiftW4ToW8<OutType, InType>(weight4BitTensor, weight8BitTensor, biasInTensor, biasOutTensor,
                                           processBias);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    template <typename Weight8BitTensorType_, typename L1TensorType_, typename BiasTensorType_,
              typename BiasL1TensorType_>
    __aicore__ static inline void StoreMxConvertedWeight(uint64_t mte2RealK,
                                                         const Weight8BitTensorType_& weight8BitTensor,
                                                         const L1TensorType_& l1Tensor,
                                                         const BiasTensorType_& biasOutTensor,
                                                         const BiasL1TensorType_& biasL1Tensor, bool processBias,
                                                         uint64_t eventId, uint64_t& computeIndex)
    {
        if (likely(mte2RealK > 0)) {
            auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyUB2L1Weight8Bit{});
            asc::te::copy(copyUB2L1, l1Tensor, weight8BitTensor);
        }
        if (processBias) {
            auto copyUB2L1Bias = asc::te::make_copy(asc::te::copy_ub_to_l1{});
            asc::te::copy(copyUB2L1Bias, biasL1Tensor, biasOutTensor);
        }
        SetFlag<HardEvent::MTE3_V>(eventId);
        computeIndex++;
    }

    template <typename OutType, typename InType, uint64_t InputC0 = asc::te::c0_element<OutType>>
    __aicore__ static inline auto MakeMxPackedWeightTensor(uint64_t mte2RealK, uint64_t nL1Size, uint64_t offset)
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, InType>(offset),
                                    asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<InputC0>>(
                                        static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size)));
    }

    template <typename BiasType>
    __aicore__ static inline auto MakeMxBiasUbTensor(uint64_t baseOffset, uint64_t slotIdx, uint64_t nL1Size,
                                                     uint64_t bufferSize)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            1, CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE)));
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(baseOffset + slotIdx * bufferSize), layout);
    }

    template <typename OutType>
    __aicore__ static inline auto MakeMxConvertedWeightTensor(uint64_t mte2RealK, uint64_t nL1Size, uint64_t baseOffset,
                                                              uint64_t slot, uint64_t bufferCount)
    {
        constexpr uint64_t vectorElements = AscendC::VECTOR_REG_WIDTH;
        const uint64_t innerSize = vectorElements * bufferCount;
        const uint64_t offset = baseOffset + slot * vectorElements * sizeof(OutType);
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, OutType>(offset),
            Blaze::Gemm::Weight8BitZnToZnUBLayout<OutType>{}(static_cast<int64_t>(mte2RealK),
                                                             static_cast<int64_t>(Align16(nL1Size)), innerSize));
    }

    __aicore__ static inline void DrainMxPrologueBuffers(uint16_t weightBuffers, uint16_t inputBuffers,
                                                         uint16_t scaleBuffers, uint16_t weightEvent,
                                                         uint16_t inputEvent, uint16_t scaleEvent)
    {
        for (uint16_t slot = 0U; slot < weightBuffers; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(weightEvent + slot);
        }
        for (uint16_t slot = 0U; slot < inputBuffers; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(inputEvent + slot);
        }
        for (uint16_t slot = 0U; slot < scaleBuffers; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(scaleEvent + slot);
        }
    }

    template <typename GMScale>
    __aicore__ static inline auto LoadMxScaleWindow(const GMScale& source, uint64_t kOffset, uint64_t nOffset,
                                                    uint64_t kSize, uint64_t nSize, uint64_t ubOffset,
                                                    uint16_t freeEvent, uint16_t readyEvent)
    {
        auto gm = source.slice(asc::te::make_coord(kOffset, nOffset), asc::te::make_shape(kSize, nSize));
        WaitFlag<HardEvent::V_MTE2>(freeEvent);
        auto layout = asc::te::make_frame_layout<asc::te::scaleb_dn_layout_ptn>(
            AscendC::Std::Int<Tile::MX_SCALE_INPUT_ROW_STRIDE>{}, nSize);
        auto base = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, fp8_e8m0_t>(ubOffset), layout);
        auto input = base.slice(asc::te::make_coord(0, 0), asc::te::make_shape(kSize, nSize));
        auto copy = asc::te::make_copy(Tile::CopyGM2UBMxScale{});
        asc::te::copy(copy, input, gm);
        SetFlag<HardEvent::MTE2_V>(readyEvent);
        WaitFlag<HardEvent::MTE2_V>(readyEvent);
        return input;
    }

    __aicore__ static inline auto MakeMxScaleOutput(uint64_t kSize, uint64_t nSize, uint64_t ubOffset,
                                                    uint16_t freeEvent)
    {
        WaitFlag<HardEvent::MTE3_V>(freeEvent);
        auto layout = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
            kSize, nSize);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, fp8_e8m0_t>(ubOffset), layout);
    }

    template <typename Input, typename Output>
    __aicore__ static inline void TransposeMxScaleWindow(const Input& input, const Output& output, uint64_t indexOffset,
                                                         uint16_t readyEvent)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, Tile::MX_SCALE_TRANS_ID_SIZE);
        auto indices = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(indexOffset),
                                            layout);
        Tile::MxScaleTranspose::Transpose(output, input, indices);
        SetFlag<HardEvent::V_MTE3>(readyEvent);
        WaitFlag<HardEvent::V_MTE3>(readyEvent);
    }

    template <typename Output>
    __aicore__ static inline void StoreMxScaleWindow(const Output& output, uint64_t kSize, uint64_t nSize,
                                                     uint64_t fullN, uint64_t nOffset, uint64_t l1Offset)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(
            kSize, fullN);
        auto l1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1Offset), layout);
        auto slice = l1.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(kSize, nSize));
        auto copy = asc::te::make_copy(Tile::CopyUB2L1MxScale{});
        asc::te::copy(copy, slice, output);
    }

    template <typename GMWeight, typename Packed>
    __aicore__ static inline void CopyMxPackedWeight(const GMWeight& source, const Packed& packed, uint64_t kOffset,
                                                     uint64_t realK, uint64_t nOffset, uint64_t nSize)
    {
        auto gm = source.slice(asc::te::make_coord(kOffset, nOffset), asc::te::make_shape(realK, nSize));
        auto copy = asc::te::make_copy(Tile::CopyGM2UBWeight{});
        asc::te::copy(copy, packed, gm);
    }

    template <typename GMWeight, typename GMBias, typename Packed, typename Bias>
    __aicore__ static inline void LoadMxPackedWeightAndBias(const GMWeight& weight, const GMBias& bias,
                                                            const Packed& packed, const Bias& biasInput,
                                                            uint64_t kOffset, uint64_t realK, uint64_t nOffset,
                                                            uint64_t nSize, uint64_t biasOffset, uint64_t biasSize,
                                                            bool processBias, uint16_t readyEvent)
    {
        if (realK > 0) {
            CopyMxPackedWeight(weight, packed, kOffset, realK, nOffset, nSize);
        }
        if (processBias) {
            auto source = bias.slice(asc::te::make_coord(0, nOffset + biasOffset), asc::te::make_shape(1, biasSize));
            auto copy = asc::te::make_copy(asc::te::copy_gm_to_ub{});
            asc::te::copy(copy, biasInput, source);
        }
        SetFlag<HardEvent::MTE2_V>(readyEvent);
        WaitFlag<HardEvent::MTE2_V>(readyEvent);
    }

    template <bool NeedsScaleIndices>
    __aicore__ inline void InitializeVectorBuffers()
    {
        for (uint16_t slot = 0; slot < UB_MTE2_BUFFER_NUM; ++slot) {
            SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + slot);
        }
        for (uint16_t slot = 0; slot < WEIGHT_8BIT_BUFFER_NUM; ++slot) {
            SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V + slot);
        }
        for (uint16_t slot = 0; slot < MX_SCALE_OUTPUT_BUFFER_NUM; ++slot) {
            SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + slot);
        }
        if constexpr (NeedsScaleIndices) {
            LoadMxScaleTransposeIndices(MX_SCALE_TRANS_ID_OFFSET, EVENT_ID_MTE2_TO_V);
        }
    }

    static constexpr uint64_t QUAD_B_STRIDE = 65536UL;
    static constexpr uint64_t QUAD_A_BASE = 311296UL;
    static constexpr uint64_t QUAD_A_STRIDE = 106496UL;
    static constexpr uint64_t QUAD_SCALE_A_BASE = 262144UL;
    static constexpr uint64_t QUAD_SCALE_B_BASE = 278528UL;
    static constexpr uint64_t QUAD_SCALE_STRIDE = 8192UL;
    static constexpr uint64_t KB = 1024UL;
    static constexpr uint64_t BIAS_L1_BYTES = 4UL * KB;
    static constexpr uint64_t DOUBLE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {128UL * KB, 256UL * KB};
    static constexpr uint64_t DOUBLE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {0UL, 384UL * KB};
    static constexpr uint64_t DOUBLE_SCALE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64UL * KB, 448UL * KB};
    static constexpr uint64_t DOUBLE_SCALE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {96UL * KB, 480UL * KB};
    static constexpr uint64_t DOUBLE_BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64UL * KB, 380UL * KB};
    static constexpr uint64_t TRIPLE_A_L1_OFFSETS[TRIPLE_BUFFER_COUNT] = {0UL, 256UL * KB, 192UL * KB};
    static constexpr uint64_t TRIPLE_B_L1_OFFSETS[TRIPLE_BUFFER_COUNT] = {64UL * KB, 320UL * KB, 448UL * KB};
    static constexpr uint64_t TRIPLE_SCALE_A_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {128UL * KB, 384UL * KB};
    static constexpr uint64_t TRIPLE_SCALE_B_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {160UL * KB, 416UL * KB};

    static constexpr uint64_t UB_MTE2_BUFFER_NUM = 4UL;
    static constexpr uint64_t WEIGHT_8BIT_BUFFER_NUM = 4UL;
    static constexpr uint64_t DEFAULT_PACKED_INPUT_BYTES = 80UL * KB;
    static constexpr uint64_t ALIGNED_PACKED_INPUT_BYTES = 64UL * KB;
    static constexpr uint64_t WEIGHT_4BIT_TOTAL_SIZE = (MODE == MxPipelineMode::Aligned) ? ALIGNED_PACKED_INPUT_BYTES :
                                                                                           DEFAULT_PACKED_INPUT_BYTES;
    static constexpr uint64_t WEIGHT_4BIT_SINGLE_BUFFER_SIZE = WEIGHT_4BIT_TOTAL_SIZE / UB_MTE2_BUFFER_NUM;
    static constexpr uint64_t WEIGHT_8BIT_TOTAL_SIZE = 128UL * KB;
    static constexpr uint64_t BIAS_UB_SINGLE_BUFFER_SIZE = 512UL;
    static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128UL;
    static constexpr uint64_t BIAS_UB_TOTAL_SIZE = 4UL * BIAS_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t BIAS_UB_INPUT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE + WEIGHT_8BIT_TOTAL_SIZE;
    static constexpr uint64_t BIAS_UB_OUTPUT_OFFSET = BIAS_UB_INPUT_OFFSET + BIAS_UB_TOTAL_SIZE;
    static constexpr uint64_t MX_SCALE_TRANS_ID_OFFSET = BIAS_UB_OUTPUT_OFFSET + BIAS_UB_TOTAL_SIZE;
    static constexpr uint64_t MX_SCALE_TRANS_ID_BUFFER_SIZE = 1UL * KB;
    static constexpr uint64_t MX_SCALE_OUTPUT_OFFSET = MX_SCALE_TRANS_ID_OFFSET + MX_SCALE_TRANS_ID_BUFFER_SIZE;
    static constexpr uint64_t MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE = 16UL * KB;
    static constexpr uint64_t MX_SCALE_OUTPUT_BUFFER_NUM = ((MODE == MxPipelineMode::ParameterDriven) ||
                                                            (MODE == MxPipelineMode::WeightQuad) ||
                                                            (MODE == MxPipelineMode::OperandTriple)) ?
                                                               DOUBLE_BUFFER_COUNT :
                                                               0UL;
    static constexpr uint64_t WEIGHT_4BIT_INIT_OFFSET = 0UL;
    static constexpr uint64_t WEIGHT_8BIT_INIT_OFFSET = WEIGHT_4BIT_TOTAL_SIZE;
    enum InputSlotIndex : uint32_t { THIRD_INPUT_SLOT = 2U, FOURTH_INPUT_SLOT = 3U };
    static constexpr uint64_t WEIGHT_4BIT_OFFSETS[UB_MTE2_BUFFER_NUM] = {
        WEIGHT_4BIT_INIT_OFFSET + 0UL * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + 1UL * WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + THIRD_INPUT_SLOT* WEIGHT_4BIT_SINGLE_BUFFER_SIZE,
        WEIGHT_4BIT_INIT_OFFSET + FOURTH_INPUT_SLOT* WEIGHT_4BIT_SINGLE_BUFFER_SIZE};
    constexpr static AscendC::TEventID VEC_EVENT_ID_V_TO_MTE2 = 0;
    constexpr static AscendC::TEventID VEC_EVENT_ID_MTE3_TO_V = 0;
    constexpr static AscendC::TEventID VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V = 4;
    constexpr static AscendC::TEventID EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
    static constexpr uint64_t MX_GROUP_SIZE = 32UL;
    static constexpr uint64_t WEIGHT_L1_INIT_OFFSET = DOUBLE_B_L1_OFFSETS[0UL];
    static constexpr uint64_t WEIGHT_L1_DB_OFFSET = DOUBLE_B_L1_OFFSETS[1UL];
    static constexpr uint64_t L1_WEIGHT_OFFSETS[DOUBLE_BUFFER_COUNT] = {WEIGHT_L1_INIT_OFFSET * sizeof(OutType),
                                                                        WEIGHT_L1_DB_OFFSET * sizeof(OutType)};
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = PipelineConfig::READY_FLAG;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = PipelineConfig::FREE_FLAG;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256UL;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512UL;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 160UL;
    static constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {DOUBLE_BIAS_L1_OFFSETS[0UL],
                                                                      DOUBLE_BIAS_L1_OFFSETS[1UL]};
    static constexpr uint64_t SCALE_B_L1_INIT_OFFSET = DOUBLE_SCALE_B_L1_OFFSETS[0UL];
    static constexpr uint64_t SCALE_B_L1_DB_OFFSET = DOUBLE_SCALE_B_L1_OFFSETS[1UL] - DOUBLE_SCALE_B_L1_OFFSETS[0UL];
    static constexpr uint64_t BIAS_L1_SINGLE_BUFFER_SIZE = BIAS_L1_BYTES;
    static constexpr uint64_t MX_SCALE_K_WINDOW_SIZE = (MODE == MxPipelineMode::WeightQuad) ?
                                                           PipelineConfig::DEFAULT_SCALE_WINDOW_K :
                                                           PipelineConfig::SCALE_WINDOW_K;
    static constexpr uint64_t DOUBLE_WEIGHT_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {DOUBLE_B_L1_OFFSETS[0UL],
                                                                               DOUBLE_B_L1_OFFSETS[1UL]};
    static constexpr uint64_t TRIPLE_WEIGHT_L1_OFFSETS[TRIPLE_BUFFER_COUNT] = {
        TRIPLE_B_L1_OFFSETS[0UL], TRIPLE_B_L1_OFFSETS[1UL], TRIPLE_B_L1_OFFSETS[2UL]};

    uint64_t cvLoopIdx_ = 0;
    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;
    uint64_t scaleComputeLoopIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;
    const uint64_t* weightOffsets_;
    const uint64_t* biasOffsets_;
    const uint64_t* scaleBOffsets_;
    uint64_t scaleKWindowSize_;
    uint16_t readyFlag_;
    uint16_t freeFlag_;
    bool hasBias_ = false;
    uint64_t l1BufferStage_{DOUBLE_BUFFER_COUNT};
    uint64_t creditLimit_{DOUBLE_BUFFER_COUNT};
    static constexpr uint64_t UB_CAPACITY_BYTES = 248UL * 1024UL;
    static constexpr uint64_t SCALE_SHAPE_INDEX = 3UL;
    static_assert(MX_SCALE_OUTPUT_OFFSET + MX_SCALE_OUTPUT_BUFFER_NUM * MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE <=
                      UB_CAPACITY_BYTES,
                  "MX UB allocation exceeds hardware capacity");
    // Share the transfer itself; callers retain their prefetch and event timing.

    template <typename GMWeightTensorType_>
    __aicore__ inline void PrefetchFragment(const GMWeightTensorType_& gmWeightTensor,
                                            const WeightPrologueMxBlockParams& param, uint64_t firstSlot,
                                            uint64_t fragIdx)
    {
        const uint64_t slot = (firstSlot + fragIdx) & (UB_MTE2_BUFFER_NUM - 1);
        WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + slot);
        auto packed = MakeMxPackedWeightTensor<OutType, InType>(WEIGHT_FRAGMENT_K, param.nL1Size,
                                                                WEIGHT_4BIT_OFFSETS[slot]);
        CopyMxPackedWeight(gmWeightTensor, packed, fragIdx * WEIGHT_FRAGMENT_K, WEIGHT_FRAGMENT_K, param.nOffset,
                           param.nL1Size);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V + slot);
    }

    __aicore__ inline uint64_t GetScaleBL1Offset(uint64_t bufferId) const
    {
        uint64_t offset = l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_SCALE_B_L1_OFFSETS[bufferId] :
                                                                  DOUBLE_SCALE_B_L1_OFFSETS[bufferId];
        return offset +
               (l1BufferStage_ == DOUBLE_BUFFER_COUNT && hasBias_ && bufferId == 0 ? BIAS_L1_SINGLE_BUFFER_SIZE : 0UL);
    }

    static constexpr uint64_t WEIGHT_FRAGMENT_K = 128UL;

    struct WeightFragment {
        uint64_t realK;
        uint64_t splitOffset;
        uint64_t biasSize;
        uint64_t biasOffset;
        bool hasBias;
    };

    __aicore__ inline WeightFragment MakeWeightFragment(const WeightPrologueMxBlockParams& param, uint64_t kOffset)
    {
        const uint64_t halfK = param.kbL1Size / DOUBLE_BUFFER_COUNT;
        const uint64_t realK = Min(param.kbL1Size, param.kSize - kOffset);
        WeightFragment fragment{};
        fragment.realK = GetSubBlockIdx() == 0 ? Min(halfK, realK) : (realK > halfK ? realK - halfK : 0);
        fragment.splitOffset = GetSubBlockIdx() * halfK;
        if (hasBias_ && kOffset == 0) {
            const uint64_t firstN = Min(param.nL1Size, MX_BIAS_SINGLE_VECTOR_SIZE);
            fragment.biasSize = fragment.realK == realK ? param.nL1Size :
                                                          (GetSubBlockIdx() == 0 ? firstN : param.nL1Size - firstN);
            fragment.biasOffset = GetSubBlockIdx() * (fragment.biasSize != 0 ? MX_BIAS_SINGLE_VECTOR_SIZE : 0);
        }
        fragment.hasBias = hasBias_ && kOffset == 0 && fragment.biasSize != 0;
        return fragment;
    }

    template <typename GMWeightTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void ConvertWeightStage(const WeightPrologueMxBlockParams& param, uint64_t kOffset,
                                              const GMWeightTensorType_& gmWeight, const GMBiasTensorType_& gmBias)
    {
        const auto fragment = MakeWeightFragment(param, kOffset);
        auto packed = MakeMxPackedWeightTensor<OutType, InType>(
            fragment.realK, param.nL1Size, WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)]);
        auto expanded = MakeMxConvertedWeightTensor<OutType>(fragment.realK, param.nL1Size, WEIGHT_8BIT_INIT_OFFSET,
                                                             ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1),
                                                             WEIGHT_8BIT_BUFFER_NUM);
        auto weightL1 = MakeL1WeightTensor(fragment.realK, param.nL1Size, fragment.splitOffset);
        auto biasInput = MakeMxBiasUbTensor<BiasType>(BIAS_UB_INPUT_OFFSET, ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1),
                                                      fragment.biasSize, BIAS_UB_SINGLE_BUFFER_SIZE);
        auto biasOutput = MakeMxBiasUbTensor<BiasType>(BIAS_UB_OUTPUT_OFFSET,
                                                       ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1),
                                                       fragment.biasSize, BIAS_UB_SINGLE_BUFFER_SIZE);
        auto biasL1 = MakeL1BiasTensor(fragment.biasSize, fragment.biasOffset);
        WaitFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
        LoadMxPackedWeightAndBias(gmWeight, gmBias, packed, biasInput, kOffset + fragment.splitOffset, fragment.realK,
                                  param.nOffset, param.nL1Size, fragment.biasOffset, fragment.biasSize,
                                  fragment.hasBias, EVENT_ID_MTE2_TO_V);
        ConvertMxPackedWeight<OutType, InType>(
            packed, expanded, biasInput, biasOutput, fragment.hasBias,
            VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
        SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
        ubMte2LoopIdx_++;
        StoreMxConvertedWeight(fragment.realK, expanded, weightL1, biasOutput, biasL1, fragment.hasBias,
                               VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)),
                               ubComputeLoopIdx_);
        Sync::NotifyCube<(MODE == MxPipelineMode::ParameterDriven) ? SYNC_AIC_AIV_MODE : Sync::SYNC_MODE_INTRA,
                         PIPE_MTE3>((MODE == MxPipelineMode::ParameterDriven) ? readyFlag_ : SYNC_AIC_AIV_FLAG);
    }

    template <typename GMWeightTensorType_>
    __aicore__ inline void ComputeAlignedFragments(const WeightPrologueMxBlockParams& param,
                                                   const GMWeightTensorType_& gmWeight)
    {
        const uint64_t fragments = param.kSize / WEIGHT_FRAGMENT_K;
        const uint64_t firstSlot = ubMte2LoopIdx_;
        uint64_t produced = 0UL;
        for (; produced < Min(static_cast<uint64_t>(DOUBLE_BUFFER_COUNT), fragments); ++produced) {
            PrefetchFragment(gmWeight, param, firstSlot, produced);
        }
        for (uint64_t index = 0UL; index < fragments; ++index) {
            const uint64_t offset = (index & 1) * WEIGHT_FRAGMENT_K;
            auto packed = MakeMxPackedWeightTensor<OutType, InType>(
                WEIGHT_FRAGMENT_K, param.nL1Size, WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)]);
            auto expanded = MakeMxConvertedWeightTensor<OutType>(
                WEIGHT_FRAGMENT_K, param.nL1Size, WEIGHT_8BIT_INIT_OFFSET,
                ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1), WEIGHT_8BIT_BUFFER_NUM);
            auto l1 = MakeL1WeightTensor(WEIGHT_FRAGMENT_K, param.nL1Size, offset);
            auto biasIn = MakeMxBiasUbTensor<BiasType>(BIAS_UB_INPUT_OFFSET, ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1),
                                                       0, BIAS_UB_SINGLE_BUFFER_SIZE);
            auto biasOut = MakeMxBiasUbTensor<BiasType>(
                BIAS_UB_OUTPUT_OFFSET, ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1), 0, BIAS_UB_SINGLE_BUFFER_SIZE);
            auto biasL1 = MakeL1BiasTensor(0, 0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
            if (produced < fragments) {
                PrefetchFragment(gmWeight, param, firstSlot, produced++);
            }
            ConvertMxPackedWeight<OutType, InType>(
                packed, expanded, biasIn, biasOut, false,
                VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)));
            SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
            ubMte2LoopIdx_++;
            if (offset == 0 && cvLoopIdx_ >= DOUBLE_BUFFER_COUNT) {
                Sync::WaitForCube<(MODE == MxPipelineMode::ParameterDriven) ? SYNC_AIC_AIV_MODE : Sync::SYNC_MODE_INTRA,
                                  PIPE_MTE3>((MODE == MxPipelineMode::ParameterDriven) ? freeFlag_ : SYNC_AIV_AIC_FLAG);
            }
            StoreMxConvertedWeight(WEIGHT_FRAGMENT_K, expanded, l1, biasOut, biasL1, false,
                                   VEC_EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (WEIGHT_8BIT_BUFFER_NUM - 1)),
                                   ubComputeLoopIdx_);
            if (offset != 0) {
                Sync::NotifyCube<(MODE == MxPipelineMode::ParameterDriven) ? SYNC_AIC_AIV_MODE : Sync::SYNC_MODE_INTRA,
                                 PIPE_MTE3>((MODE == MxPipelineMode::ParameterDriven) ? readyFlag_ : SYNC_AIC_AIV_FLAG);
                cvLoopIdx_++;
            }
        }
    }

    template <typename GMWeightTensorType_, typename GMScaleBTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const WeightPrologueMxBlockParams& param,
                                                        const GMWeightTensorType_& gmWeight,
                                                        const GMScaleBTensorType_& gmScale,
                                                        const GMBiasTensorType_& gmBias)
    {
        if constexpr ((MODE == MxPipelineMode::Aligned)) {
            ComputeAlignedFragments(param, gmWeight);
        } else {
            uint64_t limit = DOUBLE_BUFFER_COUNT;
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                limit = PipelineConfig::WeightStages(hasBias_);
            } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
                creditLimit_ = Min(l1BufferStage_, PipelineConfig::CreditLimit(param.kSize, param.kbL1Size, hasBias_));
                limit = creditLimit_;
            }
            for (uint64_t kOffset = 0UL; kOffset < param.kSize; kOffset += param.kbL1Size, cvLoopIdx_++) {
                if (cvLoopIdx_ >= limit) {
                    Sync::WaitForCube<(MODE == MxPipelineMode::ParameterDriven) ? SYNC_AIC_AIV_MODE :
                                                                                  Sync::SYNC_MODE_INTRA,
                                      PIPE_MTE3>((MODE == MxPipelineMode::ParameterDriven) ? freeFlag_ :
                                                                                             SYNC_AIV_AIC_FLAG);
                }
                HandleMxScale(param, gmScale, kOffset);
                ConvertWeightStage(param, kOffset, gmWeight, gmBias);
            }
        }
    }

    struct ScaleTransfer {
        uint64_t kOffset;
        uint64_t kSize;
        uint64_t nSize;
        uint64_t nOffset;
        uint64_t l1Slot;
        uint64_t outputSlot;
    };

    __aicore__ inline uint64_t ScaleTransferL1Base(uint64_t slot) const
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return scaleBOffsets_[slot];
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return GetScaleBL1Offset(slot);
        } else {
            const uint64_t shift = hasBias_ && slot == 0 ? BIAS_L1_SINGLE_BUFFER_SIZE : 0;
            if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
                if (!hasBias_) {
                    return QUAD_SCALE_B_BASE + slot * QUAD_SCALE_STRIDE;
                }
            }
            return SCALE_B_L1_INIT_OFFSET + slot * SCALE_B_L1_DB_OFFSET + shift;
        }
    }

    template <typename GMScaleBTensorType_>
    __aicore__ inline void HandleMxScale(const WeightPrologueMxBlockParams& param, const GMScaleBTensorType_& gmScale,
                                         uint64_t kOffset)
    {
        const uint64_t window = (MODE == MxPipelineMode::ParameterDriven) ? scaleKWindowSize_ :
                                                                            PipelineConfig::ScaleWindow(hasBias_);
        if (kOffset % window != 0) {
            return;
        }
        ScaleTransfer transfer{};
        const uint64_t slots = (MODE == MxPipelineMode::WeightQuad) && !hasBias_ ? QUADRUPLE_BUFFER_COUNT :
                                                                                   DOUBLE_BUFFER_COUNT;
        transfer.l1Slot = scaleL1BufIdx_++ & (slots - 1);
        const uint64_t firstN = param.nL1Size > BLOCK_CUBE ?
                                    CeilAlign(param.nL1Size / DOUBLE_BUFFER_COUNT, static_cast<uint64_t>(BLOCK_CUBE)) :
                                    param.nL1Size;
        transfer.nSize = GetSubBlockIdx() == 0 ? firstN : param.nL1Size - firstN;
        transfer.nOffset = GetSubBlockIdx() * firstN;
        if (transfer.nSize == 0) {
            return;
        }
        transfer.kOffset = kOffset;
        const uint64_t realK = Min(window, param.kSize - kOffset);
        transfer.kSize = (MODE == MxPipelineMode::ParameterDriven) ? Align64(realK) / MX_GROUP_SIZE :
                                                                     CeilDiv(realK, MX_GROUP_SIZE);
        transfer.outputSlot = scaleComputeLoopIdx_ & (MX_SCALE_OUTPUT_BUFFER_NUM - 1);
        auto input = LoadMxScaleWindow(
            gmScale, transfer.kOffset / MX_GROUP_SIZE, param.nOffset + transfer.nOffset, transfer.kSize, transfer.nSize,
            WEIGHT_4BIT_OFFSETS[ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)],
            VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)), EVENT_ID_MTE2_TO_V);
        auto output = MakeMxScaleOutput(
            transfer.kSize, transfer.nSize,
            MX_SCALE_OUTPUT_OFFSET + transfer.outputSlot * MX_SCALE_OUTPUT_SINGLE_BUFFER_SIZE,
            VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + transfer.outputSlot);
        TransposeMxScaleWindow(input, output, MX_SCALE_TRANS_ID_OFFSET, EVENT_ID_MTE2_TO_V);
        StoreMxScaleWindow(output, transfer.kSize, transfer.nSize, param.nL1Size, transfer.nOffset,
                           ScaleTransferL1Base(transfer.l1Slot));
        SetFlag<HardEvent::MTE3_V>(VEC_EVENT_ID_TRANS_SCALE_MTE3_TO_V + transfer.outputSlot);
        SetFlag<HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & (UB_MTE2_BUFFER_NUM - 1)));
        ubMte2LoopIdx_++;
        scaleComputeLoopIdx_++;
    }

    __aicore__ inline uint64_t WeightL1Address() const
    {
        const uint64_t slot = cvLoopIdx_ & (DOUBLE_BUFFER_COUNT - 1);
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return weightOffsets_[slot];
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return l1BufferStage_ == TRIPLE_BUFFER_COUNT ? TRIPLE_WEIGHT_L1_OFFSETS[cvLoopIdx_ % TRIPLE_BUFFER_COUNT] :
                                                           DOUBLE_WEIGHT_L1_OFFSETS[slot];
        } else if constexpr ((MODE == MxPipelineMode::WeightQuad)) {
            if (!hasBias_) {
                return (cvLoopIdx_ & (QUADRUPLE_BUFFER_COUNT - 1)) * QUAD_B_STRIDE;
            }
        }
        return L1_WEIGHT_OFFSETS[slot];
    }

    __aicore__ inline auto MakeL1WeightTensor(uint64_t mte2RealK, uint64_t nL1Size, uint64_t l1SplitOffset)
    {
        const uint64_t layoutK = (MODE == MxPipelineMode::Aligned) ? MX_A8W4_L1_K_CONFIG_256 : mte2RealK;
        auto layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<OutType>>(
            layoutK, nL1Size);
        auto tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, OutType>(WeightL1Address()),
                                           layout);
        return tensor.slice(asc::te::make_coord(l1SplitOffset, 0), asc::te::make_shape(mte2RealK, nL1Size));
    }

    __aicore__ inline uint64_t BiasL1Address() const
    {
        const uint64_t slot = cvLoopIdx_ & (DOUBLE_BUFFER_COUNT - 1);
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return biasOffsets_[slot];
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            const uint64_t biasSlot = l1BufferStage_ == TRIPLE_BUFFER_COUNT ? cvLoopIdx_ % TRIPLE_BUFFER_COUNT : slot;
            return DOUBLE_BIAS_L1_OFFSETS[biasSlot];
        } else {
            return BIAS_L1_OFFSETS[slot];
        }
    }

    __aicore__ inline auto MakeL1BiasTensor(uint64_t nL1Size, uint64_t nOffset)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            1, CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE)));
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(BiasL1Address() + nOffset * sizeof(BiasType)),
            layout);
    }

    __aicore__ inline uint64_t CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size) const
    {
        if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            if (l1BufferStage_ == TRIPLE_BUFFER_COUNT) {
                return MX_A8W4_L1_K_CONFIG_256;
            }
        }
        return (mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD &&
                nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                   MX_A8W4_L1_K_CONFIG_512 :
                   MX_A8W4_L1_K_CONFIG_256;
    }
};

// Macro aliases keep long template specializations readable in declarations/definitions.
#define GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM                                              \
    template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockEpilogue_, \
              class BlockPrologue_, bool IsSingleMultiSingle_, uint32_t PipelineStages_>

#define GROUPED_MATMUL_RESPLIT_KERNEL_CLASS                                                            \
    GmmWeightQuantMxKernel<ProblemShape_, BlockMmad_, BlockScheduler_, BlockEpilogue_, BlockPrologue_, \
                           IsSingleMultiSingle_, PipelineStages_>

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
template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockEpilogue_, class BlockPrologue_,
          bool IsSingleMultiSingle_, uint32_t PipelineStages_ = 0U>
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
    static constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;
    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
        typename BlockPrologue::Params prologue;
        GM_ADDR ptrGroupList;
        uint32_t groupListType;
        uint32_t hasBias;
    };

    __aicore__ inline GmmWeightQuantMxKernel() = default;
    __aicore__ inline void operator()(const Params& params);

private:
    static constexpr auto MODE = static_cast<MxPipelineMode>(PipelineStages_);
    template <typename AType, bool AllowFastPath>
    __aicore__ static inline void CalcMxKernelKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t aBufferBytes,
                                                     uint64_t& kaL1Size, uint64_t& kbL1Size, uint64_t& l0KSize)
    {
        constexpr uint64_t L1_K_CONFIG_512 = 512UL;
        constexpr uint64_t L1_K_CONFIG_256 = 256UL;
        constexpr uint64_t L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
        constexpr uint64_t L0_K_SMALL_TILE_THRESHOLD = 128UL;
        constexpr uint64_t L0_K_SMALL_TILE_SIZE = 256UL;
        constexpr uint64_t L0_K_LARGE_TILE_SIZE = 128UL;
        constexpr uint64_t A_L1_MODEL_SINGLE_BUF_SIZE = 80UL * 1024UL;
        constexpr uint64_t A_B_LOAD_BALANCE_FACTOR = 2UL;
        constexpr uint64_t L1_K_DYNAMIC_CONFIG_M_THRESHOLD = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType) /
                                                              L1_K_CONFIG_512 / AscendC::BLOCK_CUBE) *
                                                             AscendC::BLOCK_CUBE;

        kbL1Size = (mL1Size <= L1_K_DYNAMIC_CONFIG_M_THRESHOLD && nL1Size <= L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                       L1_K_CONFIG_512 :
                       L1_K_CONFIG_256;

        l0KSize = (mL1Size <= L0_K_SMALL_TILE_THRESHOLD && nL1Size <= L0_K_SMALL_TILE_THRESHOLD) ?
                      L0_K_SMALL_TILE_SIZE :
                      L0_K_LARGE_TILE_SIZE;
        uint64_t mL1Align = CeilAlign(mL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        if constexpr (AllowFastPath) {
            if (nL1Size <= mL1Align * A_B_LOAD_BALANCE_FACTOR) {
                kaL1Size = kbL1Size;
                return;
            }
        }
        uint64_t kaDepth = CeilDiv(nL1Size, mL1Align * A_B_LOAD_BALANCE_FACTOR);
        kaDepth = kaDepth == 0UL ? 1UL : kaDepth;

        uint64_t modelMaxKaDepth = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType)) / (mL1Align * kbL1Size);
        uint64_t actualMaxKaDepth = (aBufferBytes / sizeof(AType)) / (mL1Align * kbL1Size);
        uint64_t maxKaDepth = modelMaxKaDepth < actualMaxKaDepth ? modelMaxKaDepth : actualMaxKaDepth;
        // The scheduler guarantees one kb tile fits; preserve the original zero-depth guard.
        maxKaDepth = maxKaDepth == 0UL ? 1UL : maxKaDepth;
        kaDepth = kaDepth < maxKaDepth ? kaDepth : maxKaDepth;
        kaL1Size = kaDepth * kbL1Size;
    }

    template <typename GroupListTensor>
    __aicore__ static inline int64_t ReadMxGroupRows(const GroupListTensor& groupList, uint32_t loopIdx,
                                                     uint32_t groupListType, int64_t& preOffset)
    {
        constexpr uint32_t SPARSE_ITEM_STRIDE = 2U;
        constexpr uint32_t SPARSE_VALUE_OFFSET = 1U;
        int64_t splitValue = 0L;
        if (groupListType == 0U) {
            const int64_t offset = groupList[loopIdx];
            splitValue = offset - preOffset;
            preOffset = offset;
        } else if (groupListType == 1U) {
            splitValue = groupList[loopIdx];
            preOffset += splitValue;
        } else {
            const uint32_t splitValueIdx = loopIdx * SPARSE_ITEM_STRIDE + SPARSE_VALUE_OFFSET;
            splitValue = groupList[splitValueIdx];
            preOffset += splitValue;
        }
        return splitValue;
    }

    template <typename T_>
    __aicore__ static inline __gm__ T_* GetTensorAddrFromTensorList(uint32_t tensorIdx, __gm__ T_* tensorListAddr)
    {
        AscendC::ListTensorDesc tensorList(reinterpret_cast<__gm__ void*>(tensorListAddr));
        return tensorList.GetDataPtr<T_>(tensorIdx);
    }

    using PipelineConfig = MxWeightPipelineConfig<PipelineStages_, typename BlockMmad::AType,
                                                  typename BlockMmad::BiasType>;
    static_assert(AscendC::Std::is_same_v<typename BlockMmad::DispatchPolicy, PipelineConfig>,
                  "Kernel pipeline mode must match BlockMmad dispatch policy");
    static_assert(AscendC::Std::is_same_v<typename BlockPrologue::DispatchPolicy, PipelineConfig>,
                  "Kernel pipeline mode must match weight prologue dispatch policy");
    struct ProblemInfo {
        uint64_t k;
        uint64_t n;
        uint64_t groups;
        uint64_t scaleK;
    };
    struct GroupInputs {
        __gm__ typename BlockPrologue::InType* weight;
        __gm__ typename BlockMmad::ScaleBType* scale;
        __gm__ typename BlockMmad::BiasType* bias;
    };

    __aicore__ inline ProblemInfo GetProblemInfo(const Params& params) const
    {
        const uint64_t k = AscendC::Std::get<PROBLEM_SHAPE_K_INDEX>(params.problemShape);
        return {k, AscendC::Std::get<PROBLEM_SHAPE_N_INDEX>(params.problemShape),
                AscendC::Std::get<PROBLEM_SHAPE_GROUP_NUM_INDEX>(params.problemShape),
                CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE};
    }

    __aicore__ inline auto MakeGroupList(const ProblemInfo& info) const
    {
        const uint64_t size = info.groups *
                              (groupListType_ == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1U);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupListGm_),
                                    TensorLayoutGroupList{}(1, size));
    }

    __aicore__ inline void InitializeMmad(const Params& params)
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            constexpr uint64_t KB = 1024UL;
            constexpr uint64_t BIAS_BYTES = 4UL * KB;
            constexpr uint64_t L1_M = 256UL;
            constexpr uint64_t L1_N = 256UL;
            constexpr uint64_t K_STAGE = 256UL;
            constexpr uint64_t SCALE_WINDOW = 4096UL;
            const bool bias = params.hasBias != 0;
            const uint64_t biasBytes = bias ? BIAS_BYTES : 0;
            constexpr uint64_t weightBytes = L1_M * K_STAGE;
            constexpr uint64_t l0K = K_STAGE / DOUBLE_BUFFER_COUNT;
            for (uint64_t i = 0UL; i < DOUBLE_BUFFER_COUNT; ++i) {
                const uint64_t shift = bias && i == 0 ? BIAS_BYTES : 0;
                mmadParams_.weightOffsets[i] = i * 384UL * KB;
                mmadParams_.biasOffsets[i] = (i == 0 ? 64UL : 380UL) * KB;
                mmadParams_.aOffsets[i] = (128UL * KB) + i * (128UL * KB) + shift;
                mmadParams_.scaleAOffsets[i] = (64UL * KB) + i * (384UL * KB) + shift;
                mmadParams_.scaleBOffsets[i] = (96UL * KB) + i * (384UL * KB) + shift;
            }
            mmadParams_.aGmAddr = params.mmad.aGmAddr;
            mmadParams_.scaleAGmAddr = params.mmad.scaleAGmAddr;
            mmadParams_.scaleBGmAddr = params.mmad.scaleBGmAddr;
            mmadParams_.biasGmAddr = params.mmad.biasGmAddr;
            mmadParams_.cGmAddr = params.mmad.cGmAddr;
            mmadParams_.aBufferBytes = (128UL * KB) - biasBytes;
            mmadParams_.bBufferBytes = weightBytes;
            mmadParams_.l1TileShape = {L1_M, L1_N, K_STAGE, SCALE_WINDOW};
            mmadParams_.l0TileShape = {L1_M, L1_N, l0K};
            mmadParams_.l1BufferNum = DOUBLE_BUFFER_COUNT;
            mmadParams_.hasBias = bias;
            mmadParams_.explicitL1Offsets = true;
            mmadParams_.readyFlag = PipelineConfig::READY_FLAG;
            mmadParams_.freeFlag = PipelineConfig::FREE_FLAG;
            mmadParams_.seedWeightCredits = false;
        }
    }

    __aicore__ inline void InitializeInputs(const Params& params)
    {
        const auto& inputs = params.mmad;
        groupListType_ = params.groupListType;
        preOffset_ = 0;
        groupListGm_ = reinterpret_cast<__gm__ int64_t*>(params.ptrGroupList);
        antiquantScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleBType*>(inputs.scaleBGmAddr);
        if ASCEND_IS_AIC {
            xGm_ = reinterpret_cast<__gm__ typename BlockMmad::AType*>(inputs.aGmAddr);
            perTokenScaleGm_ = reinterpret_cast<__gm__ typename BlockMmad::ScaleAType*>(inputs.scaleAGmAddr);
            yGm_ = reinterpret_cast<__gm__ typename BlockMmad::CType*>(inputs.cGmAddr);
        }
        if ASCEND_IS_AIV {
            weightGm_ = reinterpret_cast<__gm__ typename BlockPrologue::InType*>(params.prologue.ptrB);
            biasGm_ = reinterpret_cast<__gm__ typename BlockMmad::BiasType*>(inputs.biasGmAddr);
        }
    }

    template <typename ATensor, typename ScaleATensor, typename BTensor, typename CTensor>
    __aicore__ inline void ComputeAicTile(const ProblemInfo& info, BlockMmad& block, bool bias,
                                          const typename BlockScheduler::BlockCoord& coord,
                                          const typename BlockScheduler::BlockShape& shape, const ATensor& a,
                                          const ScaleATensor& scaleA, const BTensor& scaleB, const CTensor& c)
    {
        const auto m = AscendC::Std::get<0>(coord);
        const auto n = AscendC::Std::get<1>(coord);
        const auto rows = AscendC::Std::get<0>(shape);
        const auto cols = AscendC::Std::get<1>(shape);
        auto blockA = a.slice(asc::te::make_coord(m, 0), asc::te::make_shape(rows, info.k));
        auto blockScale = scaleA.slice(asc::te::make_coord(m, 0), asc::te::make_shape(rows, info.scaleK));
        auto blockC = c.slice(asc::te::make_coord(m, n), asc::te::make_shape(rows, cols));
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            uint64_t ka = 0UL, kb = 0UL, l0 = 0UL;
            CalcMxKernelKBlock<AType, true>(rows, cols, mmadParams_.aBufferBytes, ka, kb, l0);
            block(blockA, blockScale, scaleB, blockC, ka, kb, l0);
        } else if constexpr ((MODE == MxPipelineMode::Aligned)) {
            auto blockScaleB = scaleB.slice(asc::te::make_coord(0, n), asc::te::make_shape(info.scaleK, cols));
            block(blockA, blockScale, blockScaleB, blockC, bias);
        } else {
            block(blockA, blockScale, scaleB, blockC, bias);
        }
    }

    template <typename ScaleBTensor>
    __aicore__ inline void ComputeAicGroup(const ProblemInfo& info, uint64_t rows, bool bias, BlockScheduler& scheduler,
                                           BlockMmad& block, const ScaleBTensor& scaleB)
    {
        using LA = asc::te::frame_layout_format<typename BlockMmad::LayoutA>;
        using LC = asc::te::frame_layout_format<typename BlockMmad::LayoutC>;
        using LS = asc::te::frame_layout_format<typename BlockMmad::LayoutScaleA, AscendC::Std::Int<SCALE_C0>>;
        const uint64_t offset = static_cast<uint64_t>(preOffset_) - rows;
        auto a = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(xGm_ + offset * info.k),
                                      LA{}(rows, info.k));
        auto scale = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(perTokenScaleGm_ + offset * info.scaleK),
            LS{}(rows, info.scaleK));
        auto c = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(yGm_ + offset * info.n),
                                      LC{}(rows, info.n));
        scheduler.UpdateNextProblem(asc::te::make_shape(rows, info.n, info.k));
        typename BlockScheduler::BlockCoord coord;
        typename BlockScheduler::BlockShape shape;
        while (scheduler.GetTileIdx(coord, shape)) {
            ComputeAicTile(info, block, bias, coord, shape, a, scale, scaleB, c);
        }
    }

    template <typename GroupTensor>
    __aicore__ inline void RunAicGroups(const ProblemInfo& info, const Params& params, BlockScheduler& scheduler,
                                        const GroupTensor& groups)
    {
        using LS = asc::te::frame_layout_format<typename BlockMmad::LayoutScaleB, AscendC::Std::Int<SCALE_C0>>;
        auto block = MakeMmad();
        for (uint32_t i = 0U; i < info.groups; ++i) {
            const int64_t rows = ReadMxGroupRows(groups, i, groupListType_, preOffset_);
            if (groupListType_ == GROUP_LIST_TYPE_SPARSE && rows <= 0) {
                break;
            }
            if (rows > 0 && info.n > 0) {
                auto address = antiquantScaleGm_;
                if constexpr ((MODE == MxPipelineMode::Aligned)) {
                    const uint32_t index = groupListType_ == GROUP_LIST_TYPE_SPARSE ?
                                               static_cast<uint32_t>(groups[i * SPARSE_GROUP_LIST_ITEM_STRIDE]) :
                                               i;
                    address = GroupScaleBAddress(info, index);
                }
                auto scaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(address),
                                                   LS{}(info.scaleK, info.n));
                ComputeAicGroup(info, static_cast<uint64_t>(rows), params.hasBias != 0, scheduler, block, scaleB);
            }
        }
    }

    __aicore__ inline auto GroupScaleBAddress(const ProblemInfo& info, uint32_t index) const
    {
        if constexpr (IS_SINGLE_MULTI_SINGLE) {
            return GetTensorAddrFromTensorList(index, antiquantScaleGm_);
        } else {
            return antiquantScaleGm_ + static_cast<uint64_t>(index) * info.n * info.scaleK;
        }
    }

    __aicore__ inline GroupInputs GetGroupInputs(const ProblemInfo& info, uint32_t index, bool hasBias)
    {
        GroupInputs group{weightGm_, GroupScaleBAddress(info, index), biasGm_};
        if constexpr (IS_SINGLE_MULTI_SINGLE) {
            group.weight = GetTensorAddrFromTensorList(index, weightGm_);
            if (hasBias) {
                group.bias = GetTensorAddrFromTensorList(index, biasGm_);
            }
        } else {
            group.weight += static_cast<uint64_t>(index) * ((info.n * info.k) >> 1U);
            if (hasBias) {
                group.bias += static_cast<uint64_t>(index) * info.n;
            }
        }
        return group;
    }

    template <typename WeightTensor, typename ScaleTensor, typename BiasTensor>
    __aicore__ inline void ComputeAivTile(const ProblemInfo& info, BlockPrologue& block,
                                          const typename BlockScheduler::BlockCoord& coord,
                                          const typename BlockScheduler::BlockShape& shape, uint64_t nAlign,
                                          const WeightTensor& weight, const ScaleTensor& scale, const BiasTensor& bias)
    {
        const auto n = AscendC::Std::get<1>(coord);
        const auto rows = AscendC::Std::get<0>(shape);
        const auto cols = AscendC::Std::get<1>(shape);
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            uint64_t ka = 0UL, kb = 0UL, l0 = 0UL;
            CalcMxKernelKBlock<AType, true>(rows, cols, mmadParams_.aBufferBytes, ka, kb, l0);
            block(weight, scale, bias, kb, info.k, cols, n, nAlign);
        } else {
            block(weight, scale, bias, rows, info.k, cols, n, nAlign);
        }
    }

    __aicore__ inline void ComputeAivGroup(const ProblemInfo& info, uint64_t rows, uint32_t index, const Params& params,
                                           BlockScheduler& scheduler, BlockPrologue& block)
    {
        using LB = asc::te::frame_layout_format<typename BlockMmad::LayoutB,
                                                AscendC::Std::Int<asc::te::c0_element<typename BlockMmad::AType>>>;
        using LS = asc::te::frame_layout_format<typename BlockMmad::LayoutScaleB, AscendC::Std::Int<SCALE_C0>>;
        using LBI = asc::te::frame_layout_format<typename BlockMmad::LayoutBias>;
        const uint64_t nAlign = CeilAlign(info.n, static_cast<uint64_t>(BLOCK_CUBE));
        const auto group = GetGroupInputs(info, index, params.hasBias != 0);
        auto weight = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(group.weight),
                                           LB{}(static_cast<int64_t>(info.k), static_cast<int64_t>(nAlign)));
        auto scale = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(group.scale),
                                          LS{}(info.scaleK, info.n));
        auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(group.bias),
                                         LBI{}(static_cast<int64_t>(1), static_cast<int64_t>(info.n)));
        constexpr uint64_t fp4CacheLineElements = 256UL;
        const bool disable = rows <= static_cast<uint64_t>(params.scheduler.baseM) &&
                             info.k % fp4CacheLineElements == 0;
        weight.set_l2_cache_hint(disable ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        scheduler.UpdateNextProblem(asc::te::make_shape(rows, info.n, info.k));
        typename BlockScheduler::BlockCoord coord;
        typename BlockScheduler::BlockShape shape;
        while (scheduler.GetTileIdx(coord, shape)) {
            ComputeAivTile(info, block, coord, shape, nAlign, weight, scale, bias);
        }
    }

    template <typename GroupTensor>
    __aicore__ inline void RunAivGroups(const ProblemInfo& info, const Params& params, BlockScheduler& scheduler,
                                        const GroupTensor& groups)
    {
        auto block = MakePrologue(params.hasBias != 0);
        for (uint32_t i = 0U; i < info.groups; ++i) {
            const int64_t rows = ReadMxGroupRows(groups, i, groupListType_, preOffset_);
            if (groupListType_ == GROUP_LIST_TYPE_SPARSE && rows <= 0) {
                break;
            }
            const uint32_t index = groupListType_ == GROUP_LIST_TYPE_SPARSE ?
                                       static_cast<uint32_t>(groups[i * SPARSE_GROUP_LIST_ITEM_STRIDE]) :
                                       i;
            if (rows > 0 && info.n > 0) {
                ComputeAivGroup(info, static_cast<uint64_t>(rows), index, params, scheduler, block);
            }
        }
    }

    typename BlockMmad::Params mmadParams_{};

    __aicore__ inline BlockMmad MakeMmad() const
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return BlockMmad(mmadParams_);
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return BlockMmad(TRIPLE_BUFFER_COUNT);
        } else {
            return BlockMmad{};
        }
    }

    __aicore__ inline BlockPrologue MakePrologue(bool hasBias) const
    {
        if constexpr ((MODE == MxPipelineMode::ParameterDriven)) {
            return BlockPrologue(mmadParams_);
        } else if constexpr ((MODE == MxPipelineMode::OperandTriple)) {
            return BlockPrologue(hasBias, TRIPLE_BUFFER_COUNT);
        } else {
            return BlockPrologue(hasBias);
        }
    }

    // Select K blocking for both the AIC consumer and AIV producer.

    static constexpr uint32_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint32_t GROUP_LIST_TYPE_SPARSE = 2U;
    static constexpr uint32_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2U;
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
};

GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM
__aicore__ inline void GROUPED_MATMUL_RESPLIT_KERNEL_CLASS::operator()(const Params& params)
{
    InitializeMmad(params);
    InitializeInputs(params);
    BlockScheduler scheduler(params.scheduler);
    const auto info = GetProblemInfo(params);
    const auto groups = MakeGroupList(info);
    if ASCEND_IS_AIC {
        RunAicGroups(info, params, scheduler, groups);
    } else {
        RunAivGroups(info, params, scheduler, groups);
    }
}

#undef GROUPED_MATMUL_RESPLIT_KERNEL_CLASS
#undef GROUPED_MATMUL_RESPLIT_KERNEL_TEMPLATE_PARAM

} // namespace MxPipeline

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
