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
 * \file antiquant.h
 * \brief Vector weight antiquantization tile for FP16/BF16 matrix multiplication.
 *        Antiquant selects the quantization mode, weight/output types, physical weight
 *        orientation and optional offset at compile time. Current capabilities:
 *        - Per-channel scale: one parameter per output channel N, shared along K.
 *          Weight types: INT8, FP8 E4M3/E5M2 and HiFloat8; output, scale and enabled
 *          offset use XType_ (half or bfloat16_t). W4 and other modes are not implemented.
 *        - Cast weight to XType_, optionally Add offset, then Mul scale. Add and Mul
 *          round independently in XType_; they are not a fused FP32 affine operation.
 *          INT8/HiFloat8 -> BF16 uses a half intermediate; E4M3/E5M2 uses FP32 intermediates.
 *          HasOffset_ controls the Tile calculation; the GMM caller permits offset only for INT8.
 *        - TransB_=false: physical [K,N] input (NN), N-contiguous, padded NZ output.
 *          TransB_=true: physical [N,K] input (NT), K-contiguous, padded ZN output.
 *          Both Tensor views use logical (K,N) coordinates; no extra GM transpose is performed.
 *        - W8 converts two 128-lane registers per outer iteration. OuterSize_ defaults
 *          to 128: the full VF tile is 128x256 (NN) or 256x128 (NT). InputInnerSize_
 *          is the input row pitch in weight elements, independent of the valid tail.
 *          Output groups have one padding row; their pitch comes from AntiquantTileConfig.
 *        - The caller supplies UB Tensor views covering full-width VF accesses and the
 *          padded output layout. outerSize limits K rows (NN) or N rows (NT); the inner
 *          256 lanes are still written. The caller slices the valid result for UB-to-L1 copy.
 *        - Issued by the AIV prologue (no AIC guard). Allocation, buffer reuse and pipeline
 *          synchronization belong to the caller. AntiquantTileBase adapts Tensor arguments;
 *          AntiquantImplBase keeps the VF implementation protected.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {
constexpr int32_t ANTIQUANT_DEFAULT_OUTER_SIZE = 128;

template <class WeightType_>
constexpr bool IS_ANTIQUANT_W8 = AscendC::Std::is_same_v<WeightType_, int8_t> ||
                                 AscendC::Std::is_same_v<WeightType_, float8_e4m3_t> ||
                                 AscendC::Std::is_same_v<WeightType_, float8_e5m2_t> ||
                                 AscendC::Std::is_same_v<WeightType_, hifloat8_t>;

template <bool TransB_, int32_t OuterSize_ = ANTIQUANT_DEFAULT_OUTER_SIZE, QuantMode Mode_ = QuantMode::PERCHANNEL_MODE,
          class WeightType_ = int8_t, class Enable_ = void>
struct AntiquantTileConfig {
    static_assert(Blaze::Gemm::always_false_v<WeightType_>,
                  "Antiquant configuration is not implemented for this quantization mode and weight type");
};

// Contiguous W8 width is two F16 registers. Outer size is independent of the MTE2 input allocation;
template <bool TransB_, int32_t OuterSize_, class WeightType_>
struct AntiquantTileConfig<TransB_, OuterSize_, QuantMode::PERCHANNEL_MODE, WeightType_,
                           AscendC::Std::enable_if_t<IS_ANTIQUANT_W8<WeightType_>>> {
    static constexpr int32_t F16_LANES = 128;
    static constexpr int32_t REG_COUNT = 2;
    static constexpr int32_t SECOND_INPUT_OFFSET = F16_LANES; // WeightType_ elements, not output elements.
    static constexpr int32_t INNER_SIZE = REG_COUNT * F16_LANES;
    static constexpr int32_t TILE_K = TransB_ ? INNER_SIZE : OuterSize_;
    static constexpr int32_t TILE_N = TransB_ ? OuterSize_ : INNER_SIZE;
    static constexpr int32_t OUTPUT_PADDING_ROWS = 1;
    static constexpr int32_t OUTPUT_GROUP_PITCH = (OuterSize_ + OUTPUT_PADDING_ROWS) * BLOCK_CUBE;
    static constexpr int32_t SECOND_REG_OFFSET = (OuterSize_ + OUTPUT_PADDING_ROWS) * F16_LANES;
};

template <QuantMode Mode_, class XType_, class WeightType_, bool TransB_, bool HasOffset_,
          uint32_t InputInnerSize_ = AntiquantTileConfig<TransB_, ANTIQUANT_DEFAULT_OUTER_SIZE, Mode_,
                                                         WeightType_>::INNER_SIZE,
          int32_t OuterSize_ = ANTIQUANT_DEFAULT_OUTER_SIZE>
class Antiquant;

namespace RegNS = AscendC::Reg;
using AscendC::Reg::RegTensor;

class AntiquantImplBase {
protected:
    template <typename T>
    static constexpr uint32_t VECTOR_REG_SIZE = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    static constexpr RegNS::CastTrait S8_TO_FP16_TRAIT_ODD = {
        RegNS::RegLayout::ZERO, RegNS::SatMode::UNKNOWN, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    static constexpr RegNS::CastTrait FP16_TO_BF16_TRAIT = {RegNS::RegLayout::UNKNOWN, RegNS::SatMode::UNKNOWN,
                                                            RegNS::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::CAST_RINT};

    static constexpr RegNS::CastTrait FP8_TO_FP32_TRAIT_0 = {
        RegNS::RegLayout::ZERO, RegNS::SatMode::UNKNOWN, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    static constexpr RegNS::CastTrait FP8_TO_FP32_TRAIT_2 = {
        RegNS::RegLayout::TWO, RegNS::SatMode::UNKNOWN, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    static constexpr RegNS::CastTrait FP32_TO_F16_ODD = {RegNS::RegLayout::ZERO, RegNS::SatMode::NO_SAT,
                                                         RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

    static constexpr RegNS::CastTrait FP32_TO_F16_EVEN = {
        RegNS::RegLayout::ONE, RegNS::SatMode::NO_SAT, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

    template <typename DtypeOut_, typename DtypeIn_>
    static __simd_callee__ inline void CastLowBitToF16(RegTensor<DtypeOut_>& weightF16Vreg,
                                                       RegTensor<DtypeIn_>& weightLowBitVreg, RegNS::MaskReg& maskAll)
    {
        if constexpr (AscendC::IsSameType<DtypeIn_, int4x2_t>::value) {
            static_assert(Blaze::Gemm::always_false_v<DtypeIn_>, "int4 not supported in Blaze CastLowBitToF16");
        } else if constexpr (AscendC::IsSameType<DtypeIn_, float8_e4m3_t>::value ||
                             AscendC::IsSameType<DtypeIn_, float8_e5m2_t>::value) {
            RegTensor<float> weightF32Vreg0, weightF32Vreg1;
            RegTensor<DtypeOut_> weightF16VregOdd, weightF16VregEven;
            RegNS::Cast<float, DtypeIn_, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightLowBitVreg, maskAll);
            RegNS::Cast<float, DtypeIn_, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightLowBitVreg, maskAll);
            RegNS::Cast<DtypeOut_, float, FP32_TO_F16_ODD>(weightF16VregOdd, weightF32Vreg0, maskAll);
            RegNS::Cast<DtypeOut_, float, FP32_TO_F16_EVEN>(weightF16VregEven, weightF32Vreg1, maskAll);
            RegNS::Or<uint16_t, RegNS::MaskMergeMode::ZEROING>((RegTensor<uint16_t>&)weightF16Vreg,
                                                               (RegTensor<uint16_t>&)weightF16VregOdd,
                                                               (RegTensor<uint16_t>&)weightF16VregEven, maskAll);
        } else if constexpr (AscendC::IsSameType<typename RegNS::TypeGet<DtypeOut_>::T, vector_bf16>::value) {
            if constexpr (AscendC::IsSameType<DtypeIn_, int8_t>::value ||
                          AscendC::IsSameType<DtypeIn_, hifloat8_t>::value) {
                RegTensor<half> weightFp16Vreg;
                RegNS::Cast<half, DtypeIn_, S8_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightLowBitVreg, maskAll);
                RegNS::Cast<DtypeOut_, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
            } else {
                static_assert(Blaze::Gemm::always_false_v<DtypeIn_>,
                              "Antiquant BF16 conversion requires supported INT8/FP8 weights");
            }
        } else if constexpr (AscendC::IsSameType<typename RegNS::TypeGet<DtypeOut_>::T, vector_f16>::value) {
            if constexpr (AscendC::IsSameType<DtypeIn_, int8_t>::value ||
                          AscendC::IsSameType<DtypeIn_, hifloat8_t>::value) {
                RegNS::Cast<DtypeOut_, DtypeIn_, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightLowBitVreg, maskAll);
            } else {
                static_assert(Blaze::Gemm::always_false_v<DtypeIn_>,
                              "Antiquant FP16 conversion requires supported INT8/FP8 weights");
            }
        } else {
            static_assert(Blaze::Gemm::always_false_v<DtypeOut_>, "Antiquant output must be FP16 or BF16");
        }
    }

    template <typename DtypeOut_, uint64_t OutputGroupStrideBlocks_>
    static __simd_callee__ inline void WeightF16NdRegToNzUb(__ubuf__ DtypeOut_*& weightF16PhyAddr0,
                                                            __ubuf__ DtypeOut_*& weightF16PhyAddr1,
                                                            RegTensor<DtypeOut_>& weightF16Vreg0,
                                                            RegTensor<DtypeOut_>& weightF16Vreg1,
                                                            RegNS::MaskReg& maskAll)
    {
        RegNS::DataCopy<DtypeOut_, RegNS::DataCopyMode::DATA_BLOCK_COPY, RegNS::PostLiteral::POST_MODE_UPDATE>(
            weightF16PhyAddr0, weightF16Vreg0, OutputGroupStrideBlocks_, 1, maskAll);
        RegNS::DataCopy<DtypeOut_, RegNS::DataCopyMode::DATA_BLOCK_COPY, RegNS::PostLiteral::POST_MODE_UPDATE>(
            weightF16PhyAddr1, weightF16Vreg1, OutputGroupStrideBlocks_, 1, maskAll);
    }

    // The writeback group stride comes from the selected Tile configuration, in 32-byte blocks.
    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, uint64_t OutputGroupStrideBlocks_,
              bool HasOffset_>
    static __simd_vf__ inline void AntiquantPerChannelNdNkVf(
        __ubuf__ DtypeOut_* weightF16PhyAddr0, __ubuf__ DtypeOut_* weightF16PhyAddr1,
        __ubuf__ DtypeIn_* weightLowBitPhyAddr0, __ubuf__ DtypeIn_* weightLowBitPhyAddr1,
        __ubuf__ DtypeOut_* antiQuantScaleBasePhyAddr, __ubuf__ DtypeOut_* antiQuantOffsetBasePhyAddr, uint16_t ubLoopN)
    {
        static constexpr RegNS::LoadDist LD_DIST_SCALE = RegNS::LoadDist::DIST_BRC_B16;
        static constexpr RegNS::LoadDist LD_DIST_W = RegNS::LoadDist::DIST_UNPACK_B8;
        RegTensor<DtypeOut_> antiQuantScaleVreg;
        RegTensor<DtypeOut_> antiQuantOffsetVreg;

        RegTensor<DtypeIn_> weightVreg0;
        RegTensor<DtypeIn_> weightVreg1;
        RegTensor<DtypeOut_> weightF16Vreg0;
        RegTensor<DtypeOut_> weightF16Vreg1;

        RegNS::MaskReg maskCast = RegNS::CreateMask<DtypeIn_, RegNS::MaskPattern::ALL>();
        RegNS::MaskReg maskAll = RegNS::CreateMask<DtypeOut_, RegNS::MaskPattern::ALL>();

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
            if constexpr (HasOffset_) {
                RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + ubLoopNIdx);
            }
            RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr + ubLoopNIdx);
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg0, weightLowBitPhyAddr0 + ubLoopNIdx * VfUbMte2InnerSize_);
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg1, weightLowBitPhyAddr1 + ubLoopNIdx * VfUbMte2InnerSize_);
            CastLowBitToF16(weightF16Vreg0, weightVreg0, maskCast);
            CastLowBitToF16(weightF16Vreg1, weightVreg1, maskCast);
            if constexpr (HasOffset_) {
                RegNS::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                RegNS::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
            }
            RegNS::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
            RegNS::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
            WeightF16NdRegToNzUb<DtypeOut_, OutputGroupStrideBlocks_>(weightF16PhyAddr0, weightF16PhyAddr1,
                                                                      weightF16Vreg0, weightF16Vreg1, maskAll);
        }
    }

    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, uint64_t OutputGroupStrideBlocks_,
              bool HasOffset_>
    static __simd_vf__ inline void AntiquantPerChannelNdKnVf(
        __ubuf__ DtypeOut_* weightF16PhyAddr0, __ubuf__ DtypeOut_* weightF16PhyAddr1,
        __ubuf__ DtypeIn_* weightLowBitPhyAddr0, __ubuf__ DtypeIn_* weightLowBitPhyAddr1,
        __ubuf__ DtypeOut_* antiQuantScaleBasePhyAddr, __ubuf__ DtypeOut_* antiQuantOffsetBasePhyAddr, uint16_t ubLoopK)
    {
        static constexpr RegNS::LoadDist LD_DIST_SCALE = RegNS::LoadDist::DIST_NORM;
        static constexpr RegNS::LoadDist LD_DIST_W = RegNS::LoadDist::DIST_UNPACK_B8;
        RegTensor<DtypeOut_> antiQuantScaleVreg0;
        RegTensor<DtypeOut_> antiQuantScaleVreg1;
        RegTensor<DtypeOut_> antiQuantOffsetVreg0;
        RegTensor<DtypeOut_> antiQuantOffsetVreg1;
        RegTensor<DtypeIn_> weightVreg0;
        RegTensor<DtypeIn_> weightVreg1;
        RegTensor<DtypeOut_> weightF16Vreg0;
        RegTensor<DtypeOut_> weightF16Vreg1;

        RegNS::MaskReg maskCast = RegNS::CreateMask<DtypeIn_, RegNS::MaskPattern::ALL>();
        RegNS::MaskReg maskAll = RegNS::CreateMask<DtypeOut_, RegNS::MaskPattern::ALL>();
        constexpr uint64_t REG_ELEM = VECTOR_REG_SIZE<DtypeOut_>;
        if constexpr (HasOffset_) {
            RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantOffsetVreg0, antiQuantOffsetBasePhyAddr);
            RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr + REG_ELEM);
        }
        RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantScaleVreg0, antiQuantScaleBasePhyAddr);
        RegNS::DataCopy<DtypeOut_, LD_DIST_SCALE>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr + REG_ELEM);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg0, weightLowBitPhyAddr0 + ubLoopKIdx * VfUbMte2InnerSize_);
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg1, weightLowBitPhyAddr1 + ubLoopKIdx * VfUbMte2InnerSize_);
            CastLowBitToF16(weightF16Vreg0, weightVreg0, maskCast);
            CastLowBitToF16(weightF16Vreg1, weightVreg1, maskCast);
            if constexpr (HasOffset_) {
                RegNS::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg0, maskAll);
                RegNS::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
            }
            RegNS::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
            RegNS::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
            WeightF16NdRegToNzUb<DtypeOut_, OutputGroupStrideBlocks_>(weightF16PhyAddr0, weightF16PhyAddr1,
                                                                      weightF16Vreg0, weightF16Vreg1, maskAll);
        }
    }
};

// Shared Tensor entry and compile-time dispatch. Family-specific packing and
// parameter access belong to the matching instruction adapter below.
template <QuantMode Mode_, class XType_, class WeightType_, bool TransB_, bool HasOffset_, uint32_t InputInnerSize_,
          int32_t OuterSize_>
class AntiquantTileBase : protected AntiquantImplBase {
protected:
    using Config = AntiquantTileConfig<TransB_, OuterSize_, Mode_, WeightType_>;
    static_assert(AscendC::Std::is_same_v<XType_, half> || AscendC::Std::is_same_v<XType_, bfloat16_t>);

    template <class DstTensor_, class SrcTensor_, class Param_>
    __aicore__ inline void operator()(const DstTensor_& dst, const SrcTensor_& src, const Param_& scale,
                                      const Param_& offset, uint16_t outerSize) const
    {
        if constexpr (Mode_ == QuantMode::PERCHANNEL_MODE && IS_ANTIQUANT_W8<WeightType_>) {
            RunW8PerChannel(dst, src, scale, offset, outerSize);
        } else {
            static_assert(Blaze::Gemm::always_false_v<WeightType_>, "Antiquant VF adapter is not implemented");
        }
    }

private:
    // W8 Tensor-to-VF boundary: source retains its parent row pitch and covers
    // 256 inner elements per active outer row. Destination describes the full
    // padded tile; only the caller's MTE3 view is sliced to the effective tail.
    // Scale/offset views cover the VF's full N width. Offsets below are VF
    // register operands; task, expert and tile selection remain Tensor slices.
    template <class DstTensor_, class SrcTensor_, class Param_>
    __aicore__ inline void RunW8PerChannel(const DstTensor_& dst, const SrcTensor_& src, const Param_& scale,
                                           const Param_& offset, uint16_t outerSize) const
    {
        static_assert(InputInnerSize_ >= Config::INNER_SIZE && OuterSize_ > 0);
        static_assert(VECTOR_REG_SIZE<XType_> == Config::F16_LANES);
        constexpr uint64_t OUTPUT_GROUP_STRIDE_BLOCKS = Config::OUTPUT_GROUP_PITCH / BLOCK_CUBE;
        auto out = dst.data().get();
        auto in = src.data().get();
        if constexpr (TransB_) {
            asc_vf_call<AntiquantPerChannelNdNkVf<InputInnerSize_, XType_, WeightType_, OUTPUT_GROUP_STRIDE_BLOCKS,
                                                  HasOffset_>>(out, out + Config::SECOND_REG_OFFSET, in,
                                                               in + Config::SECOND_INPUT_OFFSET, scale.data().get(),
                                                               offset.data().get(), outerSize);
        } else {
            asc_vf_call<AntiquantPerChannelNdKnVf<InputInnerSize_, XType_, WeightType_, OUTPUT_GROUP_STRIDE_BLOCKS,
                                                  HasOffset_>>(out, out + Config::SECOND_REG_OFFSET, in,
                                                               in + Config::SECOND_INPUT_OFFSET, scale.data().get(),
                                                               offset.data().get(), outerSize);
        }
    }
};

template <QuantMode Mode_, class XType_, class WeightType_, bool TransB_, bool HasOffset_, uint32_t InputInnerSize_,
          int32_t OuterSize_>
class Antiquant
    : private AntiquantTileBase<Mode_, XType_, WeightType_, TransB_, HasOffset_, InputInnerSize_, OuterSize_> {
    using Base = AntiquantTileBase<Mode_, XType_, WeightType_, TransB_, HasOffset_, InputInnerSize_, OuterSize_>;

public:
    using Base::operator();
};

} // namespace Blaze::Gemm::Tile
