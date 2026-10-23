/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_quant.h
 * \brief Tile-level MX (microscaling) dynamic quantization chain.
 *
 * Pipeline (each stage is an independent public entry):
 *   1. GroupMaxExp        : per-32-element-group max exponent / max absolute codepoint
 *   2. GenScale           : E8M0 yScale + u16 reciprocal (OCP / cuBLAS / DynDtypeRange)
 *   3. Quantize           : src * reciprocal -> saturated FP8/FP4 cast, packed int8 store
 *   4. TransScaleLayout   : packed yScale -> 32B-row-block layout for MTE copy
 *      PadScaleRows       : packed yScale -> 32B rows with explicit zero padding
 *   4'. TransFp4OutLayout : FP4 packed output re-layout (Align16 -> Align32 row pitch)
 *
 * Public interfaces accept UB tensors and call private VFs through asc_vf_call.
 * The Block manages buffer layout, synchronization and GM copies.
 *
 * Semantics notes:
 *   - OCP zeroScaleOnZeroExp=false: zero sharedExp clears the reciprocal.
 *   - OCP zeroScaleOnZeroExp=true: zero maxExp clears yScale and the reciprocal.
 *     For 0 < maxExp <= fpEmax, the reciprocal is 0 or 0x7f00, respectively.
 *   - DynDtypeRange ignores this option and clears the reciprocal when sharedExp is zero.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze::Epilogue::Tile {

enum class MxScaleAlg : uint8_t {
    OCP = 0,
    CUBLAS = 1,
    DYN_DTYPE_RANGE = 2,
};

enum class MxQuantFp4RoundMode : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

// Per-output-dtype quantization configuration, derived by the Block at Init
// time from the host params (fpEmax / invDstTypeMax / addValueBits rules are
// Block-level concerns; the tile itself is stateless).
struct MxQuantConfig {
    MxScaleAlg alg{MxScaleAlg::OCP};
    uint16_t fpEmax{0};             // E4M3: 0x0400, E5M2: 0x0780, E2M1: 0x0100, E1M2: 0x0000
    float invDstTypeMax{1.0f};      // cuBLAS: 1/dstTypeMax (host may override)
    uint16_t addValueBits{0x003f};  // Dyn: dstTypeMax==7 ? 0x001f : 0x003f
    bool zeroScaleOnZeroExp{false}; // OCP zeroMask predicate/scope (see file header)
};

// ---------------------------------------------------------------------------
// MxQuant tile: reusable MX dynamic quantization chain
//   DataTypeOut: fp8_e4m3fn_t / fp8_e5m2_t / fp4x2_e2m1_t / fp4x2_e1m2_t
//   All tensors are scanned linearly (totalCount / totalScale elements); row
//   pitch is a Block-side buffer-layout concern (activation rowPitch =
//   Align32(n), matching the Gelu tile output contract).
// ---------------------------------------------------------------------------
template <typename DataTypeOut_>
class MxQuant {
public:
    using DataTypeOut = DataTypeOut_;

    static_assert(AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value,
                  "MxQuant only supports FP8 or FP4 output.");

    __aicore__ inline MxQuant() = default;

    /*!
     * Stage 1: per-group max extraction over totalCount contiguous bf16 elements.
     * useAbs=false -> exponent-field mask (OCP); true -> absolute-value mask
     * (cuBLAS / DynDtypeRange). maxExp receives one uint16 per 32-element group
     * (totalScale = CeilDiv(totalCount, 32), plus tail space for StoreUnAlign).
     */
    template <typename SrcTensor, typename MaxExpTensor>
    __aicore__ inline void GroupMaxExp(const SrcTensor& srcTensor, const MaxExpTensor& maxExpTensor,
                                       uint32_t totalCount, bool useAbs);

    /*!
     * Stage 2: scale computation over totalScale groups. yScale receives packed
     * E8M0 bytes (DIST_PACK_B16); reciprocal receives BF16 bit patterns stored
     * as uint16_t. Algorithm selected by cfg.alg.
     */
    template <typename MaxExpTensor, typename ScaleTensor, typename ReciprocalTensor>
    __aicore__ inline void GenScale(const MaxExpTensor& maxExpTensor, const ScaleTensor& yScaleTensor,
                                    const ReciprocalTensor& reciprocalTensor, const MxQuantConfig& cfg,
                                    uint32_t totalScale);

    /*!
     * Stage 3: quantize totalCount bf16 elements (src * per-group reciprocal)
     * into packed DataTypeOut written to the int8 buffer y. FP8 vs FP4 path is
     * selected at compile time by DataTypeOut; FP4 rounding mode at runtime.
     */
    template <typename SrcTensor, typename ReciprocalTensor, typename YTensor>
    __aicore__ inline void Quantize(const SrcTensor& srcTensor, const ReciprocalTensor& reciprocalTensor,
                                    const YTensor& yTensor, uint32_t totalCount,
                                    MxQuantFp4RoundMode fp4RoundMode = MxQuantFp4RoundMode::RINT);

    /*!
     * Stage 4: yScale layout transpose. src holds scaleBlockN contiguous valid
     * scales per row; dst is an mSize x 32B row-block buffer. The Block zeroes
     * dst beforehand when stale bytes beyond the valid prefix are observable.
     */
    template <typename ScaleTensor, typename ScaleBlockTensor>
    __aicore__ inline void TransScaleLayout(const ScaleTensor& srcTensor, const ScaleBlockTensor& dstTensor,
                                            uint16_t mSize, uint16_t scaleBlockN);

    /*!
     * Repack mSize compact scale rows into 32B rows. Preserve [0, scaleCols),
     * zero [scaleCols, scaleCopyCols), and leave the rest of each dst row untouched.
     * Requires 0 < scaleCols <= scaleCopyCols <= ONE_BLK_SIZE, src row pitch
     * scaleCols, dst row pitch ONE_BLK_SIZE, and separate 32B-aligned UB buffers.
     * LoadUnAlign reads a full vector plus its 32B prefetch block: the caller
     * must keep that source window readable even on the last row. Masks do not
     * bound source reads. The Block owns buffer allocation, synchronization and
     * the decision to append the paired E8M0 zero byte on the last N tile.
     */
    template <typename ScaleTensor, typename ScaleBlockTensor>
    __aicore__ inline void PadScaleRows(const ScaleTensor& srcTensor, const ScaleBlockTensor& dstTensor, uint16_t mSize,
                                        uint32_t scaleCols, uint32_t scaleCopyCols);

    /*!
     * Stage 4' (FP4 output only): re-layout the packed FP4 output from
     * Align16(n/2)-byte row pitch to Align32(n/2)-byte row pitch so the MTE
     * copy can use an NDExt layout. Only needed when n is not 64-aligned.
     */
    template <typename YTensor, typename YBlockTensor>
    __aicore__ inline void TransFp4OutLayout(const YTensor& srcTensor, const YBlockTensor& dstTensor, uint16_t mSize,
                                             uint16_t nSize);

private:
    // ---- Quantization constants ----
    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr int16_t SHR_NUM_FOR_FP32 = 23;
    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
    static constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
    static constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
    static constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
    static constexpr uint32_t NAN_PACK = 0x00007f81;
    static constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
    static constexpr uint32_t NUMBER_ZERO = 0x00000000;
    static constexpr uint32_t NUM_TWO_FIVE_FOUR = 0x000000fe;
    static constexpr uint32_t NUMBER_HALF = 0x00400000;
    static constexpr uint32_t INTERLEAVED_REG_FACTOR = 2;
    static constexpr uint32_t HALF_REG_FACTOR = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;          // MX group size
    static constexpr uint32_t OUT_ELE_NUM_ONE_BLK = 64; // PACK4_B32 elements per block

    // ---- Cast traits (named by bit-width; RoundMode lives in AscendC, not AscendC::Reg) ----
    // Narrowing fp32 -> FP8 with saturation (clamps instead of inf).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_FP8_SAT = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // Widening 16F -> fp32, layout ZERO / ONE halves (Sat/Round ignored for widening).
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    // ---- Vf parameter structs: raw __ubuf__ pointers + shape info ----
    struct MxMaxExpVfParams {
        __ubuf__ bfloat16_t* srcAddr;
        __ubuf__ uint16_t* maxExpAddr;
        uint16_t loopCount;
        uint32_t totalCount;
    };

    struct MxScaleVfParams {
        __ubuf__ uint16_t* maxExpAddr;
        __ubuf__ uint16_t* yScaleAddr; // E8M0 packed into an int8 buffer, written as u16
        __ubuf__ uint16_t* reciprocalAddr;
        uint16_t loopCount;
        uint16_t fpEmax;
        uint16_t addValueBits;
        uint32_t totalScale;
        float invDstTypeMax;
    };

    struct MxQuantizeVfParams {
        __ubuf__ bfloat16_t* srcAddr;
        __ubuf__ uint16_t* reciprocalAddr;
        __ubuf__ int8_t* yAddr;
        uint16_t loopCount;
        uint32_t totalCount;
    };

    struct MxTransScaleVfParams {
        __ubuf__ int8_t* srcAddr;
        __ubuf__ int8_t* dstAddr;
        uint16_t mSize;
        uint16_t scaleBlockN;
    };

    struct MxPadScaleRowsVfParams {
        __ubuf__ int8_t* srcAddr;
        __ubuf__ int8_t* dstAddr;
        uint16_t mSize;
        uint32_t scaleCols;
        uint32_t scaleCopyCols;
    };

    struct MxTransFp4OutVfParams {
        __ubuf__ int8_t* srcAddr;
        __ubuf__ int8_t* dstAddr;
        uint16_t mSize;
        uint16_t copyCount;    // nSize / 2 bytes per row
        uint32_t srcRowStride; // Align16(nSize / 2)
        uint32_t dstRowStride; // Align32(nSize / 2)
    };

    // ---- Loop count derivation ----
    __aicore__ inline static uint16_t DataLoopCount(uint32_t totalCount)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        return static_cast<uint16_t>(
            Gemm::CeilDiv(totalCount, static_cast<uint32_t>(VL_FOR_16F) * INTERLEAVED_REG_FACTOR));
    }

    __aicore__ inline static uint16_t ScaleLoopCount(uint32_t totalScale, MxScaleAlg alg)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        const uint32_t lanes = alg == MxScaleAlg::CUBLAS ? static_cast<uint32_t>(VL_FOR_16F) / HALF_REG_FACTOR :
                                                           static_cast<uint32_t>(VL_FOR_16F);
        return static_cast<uint16_t>(Gemm::CeilDiv(totalScale, lanes));
    }

    // Shared by OCP and DynDtypeRange. Pass reciprocalAddr by reference so
    // POST_MODE_UPDATE advances the caller's pointer across iterations.
    static __simd_callee__ inline void ScaleReciprocalTailCallee(
        AscendC::Reg::RegTensor<uint16_t>& sharedExp, AscendC::Reg::RegTensor<uint16_t>& halfScale,
        AscendC::Reg::RegTensor<uint16_t>& scaleBias, AscendC::Reg::RegTensor<uint16_t>& nanReg,
        AscendC::Reg::RegTensor<uint16_t>& zeroReg, AscendC::Reg::RegTensor<uint16_t>& specialReg,
        AscendC::Reg::MaskReg& finiteMask, AscendC::Reg::MaskReg& zeroMask, AscendC::Reg::MaskReg& mask,
        __ubuf__ uint16_t*& reciprocalAddr, uint16_t vlHalf)
    {
        AscendC::Reg::MaskReg specialMask;
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialMask, sharedExp, scaleBias, mask);
        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, mask);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanReg, finiteMask);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroReg, zeroMask);
        AscendC::Reg::Select<uint16_t>(halfScale, specialReg, halfScale, specialMask);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalAddr, halfScale,
                                                                                        vlHalf, mask);
    }

    // ---- Vf implementations (static __simd_vf__ inline, defined in-class) ----
    template <bool FULL_BLOCK, bool USE_ABS, bool EVEN_COUNT>
    static __simd_vf__ inline void MaxExpVf(MxMaxExpVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        constexpr uint16_t ELEMENT_AFTER_REDUCE = static_cast<uint16_t>(asc_get_vf_len() / BLOCK_SIZE);
        __ubuf__ bfloat16_t* srcAddr = params.srcAddr;
        __ubuf__ uint16_t* maxExpAddr = params.maxExpAddr;
        const uint16_t loopCount = params.loopCount;
        // Track valid elements in the even/odd registers without deinterleaving masks.
        uint32_t remainingOdd = params.totalCount / INTERLEAVED_REG_FACTOR;
        uint32_t remainingEven = remainingOdd + params.totalCount % INTERLEAVED_REG_FACTOR;
        AscendC::Reg::RegTensor<bfloat16_t> value0;
        AscendC::Reg::RegTensor<bfloat16_t> value1;
        AscendC::Reg::RegTensor<uint16_t> exp0;
        AscendC::Reg::RegTensor<uint16_t> exp1;
        AscendC::Reg::RegTensor<uint16_t> maskValue;
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::RegTensor<uint16_t> groupMaxValue;
        AscendC::Reg::MaskReg maskEven = AscendC::Reg::CreateMask<bfloat16_t>();
        AscendC::Reg::MaskReg maskOdd = AscendC::Reg::CreateMask<bfloat16_t>();
        AscendC::Reg::UnalignRegForStore unalign;
        if constexpr (USE_ABS) {
            // Clear the sign bit; keep the exponent and mantissa.
            AscendC::Reg::Duplicate(maskValue, ABS_MASK_FOR_16BIT);
        } else {
            // Clear the sign and mantissa bits; keep only the exponent.
            AscendC::Reg::Duplicate(maskValue, MAX_EXP_FOR_BF16);
        }
        for (uint16_t i = 0; i < loopCount; ++i) {
            if constexpr (!FULL_BLOCK) {
                maskEven = AscendC::Reg::UpdateMask<bfloat16_t>(remainingEven);
                if constexpr (EVEN_COUNT) {
                    maskOdd = maskEven;
                } else {
                    maskOdd = AscendC::Reg::UpdateMask<bfloat16_t>(remainingOdd);
                }
            }
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(value0, value1, srcAddr,
                                                                             VL_FOR_16F * INTERLEAVED_REG_FACTOR);
            AscendC::Reg::And(exp0, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0), maskValue, maskEven);
            AscendC::Reg::And(exp1, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), maskValue, maskOdd);
            AscendC::Reg::Max(maxValue, exp0, exp1, maskEven);
            AscendC::Reg::ReduceMaxWithDataBlock(groupMaxValue, maxValue, maskEven);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, groupMaxValue, unalign, ELEMENT_AFTER_REDUCE);
        }
        AscendC::Reg::StoreUnAlignPost(maxExpAddr, unalign, 0);
    }

    template <bool ZERO_SCALE_ON_ZERO_EXP, bool FULL_BLOCK>
    static __simd_vf__ inline void ScaleOcpVf(MxScaleVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> fp8Nan;
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::RegTensor<uint16_t> zero;
        AscendC::Reg::RegTensor<uint16_t> nan;
        AscendC::Reg::RegTensor<uint16_t> special;
        AscendC::Reg::MaskReg infNanMask;
        AscendC::Reg::MaskReg zeroMask;
        auto mask = AscendC::Reg::CreateMask<uint16_t>();
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::Duplicate(maxExpValue, params.fpEmax);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zero, 0);
        AscendC::Reg::Duplicate(nan, NAN_CUSTOMIZATION);
        AscendC::Reg::Duplicate(special, SPECIAL_EXP_THRESHOLD);
        uint32_t remainingScale = params.totalScale;
        for (uint16_t i = 0; i < params.loopCount; ++i) {
            if constexpr (!FULL_BLOCK) {
                mask = AscendC::Reg::UpdateMask<uint16_t>(remainingScale);
            }
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxValue, params.maxExpAddr,
                                                                                           VL_FOR_16F);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(infNanMask, maxValue, expMask, mask);
            if constexpr (ZERO_SCALE_ON_ZERO_EXP) {
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, maxValue, zero, mask);
            }
            AscendC::Reg::Max(maxValue, maxValue, maxExpValue, mask);
            AscendC::Reg::Sub(sharedExp, maxValue, maxExpValue, mask);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, mask);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, infNanMask);
            if constexpr (ZERO_SCALE_ON_ZERO_EXP) {
                AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);
            }
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(params.yScaleAddr, scaleValue,
                                                                             VL_FOR_16F / HALF_REG_FACTOR, mask);

            if constexpr (!ZERO_SCALE_ON_ZERO_EXP) {
                AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zero, mask);
            }
            ScaleReciprocalTailCallee(sharedExp, halfScale, scaleBias, nan, zero, special, infNanMask, zeroMask, mask,
                                      params.reciprocalAddr, VL_FOR_16F);
        }
    }

    template <bool FULL_BLOCK>
    static __simd_vf__ inline void ScaleCublasVf(MxScaleVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        AscendC::Reg::RegTensor<uint16_t> max16;
        AscendC::Reg::RegTensor<uint32_t> max32;
        AscendC::Reg::RegTensor<uint32_t> exp32;
        AscendC::Reg::RegTensor<uint32_t> mantissa32;
        AscendC::Reg::RegTensor<uint32_t> expAddOne32;
        AscendC::Reg::RegTensor<uint32_t> extractExp;
        AscendC::Reg::RegTensor<uint16_t> expOut;
        AscendC::Reg::RegTensor<uint32_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> reciprocalExpOut;
        AscendC::Reg::RegTensor<float> invMax;
        AscendC::Reg::RegTensor<uint32_t> mantissaMask;
        AscendC::Reg::RegTensor<uint32_t> expMask;
        AscendC::Reg::RegTensor<uint32_t> zero;
        AscendC::Reg::RegTensor<uint32_t> scaleBias;
        AscendC::Reg::RegTensor<uint32_t> nan;
        AscendC::Reg::RegTensor<uint32_t> fp8Nan;
        AscendC::Reg::MaskReg finiteMask;
        AscendC::Reg::MaskReg nonzeroMask;
        AscendC::Reg::MaskReg predicate0;
        AscendC::Reg::MaskReg predicate1;
        AscendC::Reg::MaskReg predicate2;
        uint32_t halfCount = VL_FOR_16F / HALF_REG_FACTOR;
        auto maskB16 = AscendC::Reg::UpdateMask<uint16_t>(halfCount);
        auto maskFloat = AscendC::Reg::CreateMask<uint32_t>();
        uint32_t remainingScale = params.totalScale;

        AscendC::Reg::Duplicate(invMax, params.invDstTypeMax);
        AscendC::Reg::Duplicate(mantissaMask, MAN_MASK_FLOAT);
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::Reg::Duplicate(zero, 0);
        AscendC::Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::Reg::Duplicate(nan, NAN_PACK);
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8_IN_FP32);

        for (uint16_t i = 0; i < params.loopCount; ++i) {
            if constexpr (!FULL_BLOCK) {
                const uint32_t processedScale = remainingScale < VL_FOR_16F / HALF_REG_FACTOR ?
                                                    remainingScale :
                                                    VL_FOR_16F / HALF_REG_FACTOR;
                uint32_t b16MaskElementCount = processedScale;
                maskB16 = AscendC::Reg::UpdateMask<uint16_t>(b16MaskElementCount);
                maskFloat = AscendC::Reg::UpdateMask<uint32_t>(remainingScale);
            }
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_UNPACK_B16>(max16, params.maxExpAddr,
                                                                             VL_FOR_16F / HALF_REG_FACTOR);
            AscendC::Reg::Cast<float, bfloat16_t, CT_16F_TO_32F_ZERO>(
                reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(max16), maskFloat);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(finiteMask, max32, expMask, maskFloat);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(nonzeroMask, max32, zero, maskFloat);
            AscendC::Reg::Mul(reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32),
                              reinterpret_cast<AscendC::Reg::RegTensor<float>&>(max32), invMax, maskFloat);
            AscendC::Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            AscendC::Reg::And(mantissa32, max32, mantissaMask, maskFloat);

            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate0, max32, NUMBER_HALF, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(
                predicate1, max32, NUM_TWO_FIVE_FOUR << SHR_NUM_FOR_FP32, maskFloat);
            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(predicate2, mantissa32, NUMBER_ZERO, maskFloat);
            AscendC::Reg::MaskAnd(predicate0, predicate0, predicate1, maskFloat);
            AscendC::Reg::MaskAnd(predicate0, predicate0, predicate2, maskFloat);

            AscendC::Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            AscendC::Reg::Select(extractExp, expAddOne32, exp32, predicate0);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan, finiteMask);
            AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zero, nonzeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
                params.yScaleAddr + i * VL_FOR_16F / HALF_REG_FACTOR / HALF_REG_FACTOR, expOut, maskB16);

            AscendC::Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            AscendC::Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nan, finiteMask);
            AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zero, nonzeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(reciprocalExpOut, halfScale);
            AscendC::Reg::StoreAlign<uint16_t>(params.reciprocalAddr + i * VL_FOR_16F / HALF_REG_FACTOR,
                                               reciprocalExpOut, maskB16);
        }
    }

    template <bool FULL_BLOCK>
    static __simd_vf__ inline void ScaleDynVf(MxScaleVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::RegTensor<uint16_t> maxExpOnly;
        AscendC::Reg::RegTensor<uint16_t> roundedMaxExp;
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::RegTensor<uint16_t> addValue;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::RegTensor<uint16_t> fp8Nan;
        AscendC::Reg::RegTensor<uint16_t> zero;
        AscendC::Reg::RegTensor<uint16_t> nan;
        AscendC::Reg::RegTensor<uint16_t> special;
        AscendC::Reg::MaskReg finiteMask;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg belowRangeMask;
        auto mask = AscendC::Reg::CreateMask<uint16_t>();

        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::Duplicate(addValue, params.addValueBits);
        AscendC::Reg::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zero, 0);
        AscendC::Reg::Duplicate(nan, NAN_CUSTOMIZATION);
        AscendC::Reg::Duplicate(special, SPECIAL_EXP_THRESHOLD);
        uint32_t remainingScale = params.totalScale;

        for (uint16_t i = 0; i < params.loopCount; ++i) {
            if constexpr (!FULL_BLOCK) {
                mask = AscendC::Reg::UpdateMask<uint16_t>(remainingScale);
            }
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxValue, params.maxExpAddr,
                                                                                           VL_FOR_16F);
            AscendC::Reg::And(maxExpOnly, maxValue, expMask, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(finiteMask, maxExpOnly, expMask, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(belowRangeMask, maxExpOnly, maxExpValue, mask);
            AscendC::Reg::Add(roundedMaxExp, maxValue, addValue, mask);
            AscendC::Reg::And(roundedMaxExp, roundedMaxExp, expMask, mask);
            AscendC::Reg::Select<uint16_t>(roundedMaxExp, maxExpValue, roundedMaxExp, belowRangeMask);
            AscendC::Reg::Sub(sharedExp, roundedMaxExp, maxExpValue, mask);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, mask);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, finiteMask);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(params.yScaleAddr, scaleValue,
                                                                             VL_FOR_16F / HALF_REG_FACTOR, mask);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zero, mask);
            ScaleReciprocalTailCallee(sharedExp, halfScale, scaleBias, nan, zero, special, finiteMask, zeroMask, mask,
                                      params.reciprocalAddr, VL_FOR_16F);
        }
    }

    template <bool FULL_BLOCK>
    static __simd_vf__ inline void QuantizeFp8Vf(MxQuantizeVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        constexpr uint16_t ELEMENT_AFTER_REDUCE = static_cast<uint16_t>(asc_get_vf_len() / BLOCK_SIZE);
        uint32_t remainingOdd = params.totalCount / INTERLEAVED_REG_FACTOR;
        uint32_t remainingEven = remainingOdd + params.totalCount % INTERLEAVED_REG_FACTOR;
        uint32_t remainingOutput = params.totalCount;
        auto fullB16 = AscendC::Reg::CreateMask<bfloat16_t>();
        auto fullB32 = AscendC::Reg::CreateMask<float>();
        auto fullB8 = AscendC::Reg::CreateMask<uint8_t>();
        AscendC::Reg::MaskReg maskEven = fullB16, maskOdd = fullB16;
        auto outputMask = fullB8;
        AscendC::Reg::RegTensor<uint16_t> scale;
        AscendC::Reg::RegTensor<bfloat16_t> value0, value1;
        AscendC::Reg::RegTensor<float> fp32Value00, fp32Value01, fp32Value10, fp32Value11;
        AscendC::Reg::RegTensor<DataTypeOut> fp8Value00, fp8Value01, fp8Value10, fp8Value11;
        static constexpr AscendC::Reg::CastTrait CT_FP8_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                               AscendC::Reg::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait CT_FP8_TWO = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                               AscendC::Reg::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait CT_FP8_THREE = {
            AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < params.loopCount; ++i) {
            if constexpr (!FULL_BLOCK) {
                maskEven = AscendC::Reg::UpdateMask<bfloat16_t>(remainingEven);
                maskOdd = AscendC::Reg::UpdateMask<bfloat16_t>(remainingOdd);
                outputMask = AscendC::Reg::UpdateMask<uint8_t>(remainingOutput);
            }
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(value0, value1, params.srcAddr,
                                                                             VL_FOR_16F * INTERLEAVED_REG_FACTOR);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(scale, params.reciprocalAddr,
                                                                          ELEMENT_AFTER_REDUCE);
            AscendC::Reg::Mul(value0, value0, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale), maskEven);
            AscendC::Reg::Mul(value1, value1, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale), maskOdd);
            AscendC::Reg::Cast<float, bfloat16_t, CT_16F_TO_32F_ZERO>(fp32Value00, value0, fullB16);
            AscendC::Reg::Cast<float, bfloat16_t, CT_16F_TO_32F_ONE>(fp32Value01, value0, fullB16);
            AscendC::Reg::Cast<float, bfloat16_t, CT_16F_TO_32F_ZERO>(fp32Value10, value1, fullB16);
            AscendC::Reg::Cast<float, bfloat16_t, CT_16F_TO_32F_ONE>(fp32Value11, value1, fullB16);
            // Each conversion fills one byte of each four-byte lane; other bytes are zero.
            AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_FP8_SAT>(fp8Value00, fp32Value00, fullB32);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP8_ONE>(fp8Value10, fp32Value10, fullB32);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP8_TWO>(fp8Value01, fp32Value01, fullB32);
            AscendC::Reg::Cast<DataTypeOut, float, CT_FP8_THREE>(fp8Value11, fp32Value11, fullB32);
            auto& out0 = reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(fp8Value00);
            auto& out1 = reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(fp8Value10);
            auto& out2 = reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(fp8Value01);
            auto& out3 = reinterpret_cast<AscendC::Reg::RegTensor<uint8_t>&>(fp8Value11);
            AscendC::Reg::Add(out0, out0, out2, fullB8);
            AscendC::Reg::Add(out1, out1, out3, fullB8);
            AscendC::Reg::Add(out0, out0, out1, fullB8);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                params.yAddr, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(out0),
                VL_FOR_16F * INTERLEAVED_REG_FACTOR, outputMask);
        }
    }

    template <AscendC::RoundMode ROUND_MODE>
    static __simd_vf__ inline void QuantizeFp4Vf(MxQuantizeVfParams params)
    {
        constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
        constexpr uint16_t ELEMENT_AFTER_REDUCE = static_cast<uint16_t>(asc_get_vf_len() / BLOCK_SIZE);
        uint32_t remainingData = params.totalCount;
        AscendC::Reg::MaskReg mask0;
        AscendC::Reg::MaskReg mask1;
        // Invalid lanes are discarded by the cast/store masks; multiplication is elementwise.
        auto fullB16 = AscendC::Reg::CreateMask<bfloat16_t>();
        AscendC::Reg::RegTensor<uint16_t> scale;
        AscendC::Reg::RegTensor<bfloat16_t> value0;
        AscendC::Reg::RegTensor<bfloat16_t> value1;
        AscendC::Reg::RegTensor<DataTypeOut> fp4Value0;
        AscendC::Reg::RegTensor<DataTypeOut> fp4Value1;
        static constexpr AscendC::Reg::CastTrait castFp4 = {AscendC::Reg::RegLayout::ZERO,
                                                            AscendC::Reg::SatMode::UNKNOWN,
                                                            AscendC::Reg::MaskMergeMode::ZEROING, ROUND_MODE};

        for (uint16_t i = 0; i < params.loopCount; ++i) {
            mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
            mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(remainingData);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(value0, value1, params.srcAddr,
                                                                             VL_FOR_16F * INTERLEAVED_REG_FACTOR);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(scale, params.reciprocalAddr,
                                                                          ELEMENT_AFTER_REDUCE);
            AscendC::Reg::Mul(value0, value0, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale), fullB16);
            AscendC::Reg::Mul(value1, value1, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(scale), fullB16);
            AscendC::Reg::Interleave(value0, value1, value0, value1);
            AscendC::Reg::Cast<DataTypeOut, bfloat16_t, castFp4>(fp4Value0, value0, mask0);
            AscendC::Reg::Cast<DataTypeOut, bfloat16_t, castFp4>(fp4Value1, value1, mask1);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                params.yAddr, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp4Value0), OUT_ELE_NUM_ONE_BLK,
                mask0);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                params.yAddr, reinterpret_cast<AscendC::Reg::RegTensor<int8_t>&>(fp4Value1), OUT_ELE_NUM_ONE_BLK,
                mask1);
        }
    }

    static __simd_vf__ inline void TransScaleVf(MxTransScaleVfParams params)
    {
        for (uint16_t row = 0; row < params.mSize; ++row) {
            uint32_t elementCount = params.scaleBlockN;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int8_t>(elementCount);
            AscendC::Reg::RegTensor<int8_t> value;
            AscendC::Reg::UnalignRegForLoad unalign;
            __ubuf__ int8_t* rowSource = params.srcAddr + row * params.scaleBlockN;
            AscendC::Reg::LoadUnAlignPre(unalign, rowSource);
            AscendC::Reg::LoadUnAlign(value, unalign, rowSource);
            __ubuf__ int8_t* rowDestination = params.dstAddr + row * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(rowDestination, value, mask);
        }
    }

    static __simd_vf__ inline void PadScaleRowsVf(MxPadScaleRowsVfParams params)
    {
        auto storeMask = AscendC::Reg::UpdateMask<int8_t>(params.scaleCopyCols);
        uint32_t validCount = params.scaleCols;
        auto validMask = AscendC::Reg::UpdateMask<int8_t>(validCount);
        AscendC::Reg::RegTensor<int8_t> zero;
        AscendC::Reg::Duplicate(zero, static_cast<int8_t>(0));
        for (uint16_t row = 0; row < params.mSize; ++row) {
            AscendC::Reg::RegTensor<int8_t> value;
            AscendC::Reg::UnalignRegForLoad unalign;
            const uint64_t srcOffset = static_cast<uint64_t>(row) * params.scaleCols;
            const uint64_t dstOffset = static_cast<uint64_t>(row) * AscendC::ONE_BLK_SIZE;
            __ubuf__ int8_t* rowSource = params.srcAddr + srcOffset;
            AscendC::Reg::LoadUnAlignPre(unalign, rowSource);
            AscendC::Reg::LoadUnAlign(value, unalign, rowSource);
            // The byte after this row's valid scales may belong to the next row.
            AscendC::Reg::Select(value, value, zero, validMask);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(params.dstAddr + dstOffset, value,
                                                                                    storeMask);
        }
    }

    static __simd_vf__ inline void TransFp4OutVf(MxTransFp4OutVfParams params)
    {
        for (uint16_t mIdx = 0; mIdx < params.mSize; ++mIdx) {
            uint32_t elemNum = params.copyCount;
            AscendC::Reg::MaskReg maskOutN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignRegForLoad u0;
            __ubuf__ int8_t* srcUb = params.srcAddr + mIdx * params.srcRowStride;
            AscendC::Reg::LoadUnAlignPre(u0, srcUb);
            AscendC::Reg::LoadUnAlign(vreg0, u0, srcUb);
            __ubuf__ int8_t* dstUb = params.dstAddr + mIdx * params.dstRowStride;
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskOutN);
        }
    }
};

// ===========================================================================
// GroupMaxExp – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename SrcTensor, typename MaxExpTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::GroupMaxExp(const SrcTensor& srcTensor, const MaxExpTensor& maxExpTensor,
                                                          uint32_t totalCount, bool useAbs)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using MaxExpElementType = asc::te::get_attribute_element_type<typename MaxExpTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using MaxExpLayoutPattern = asc::te::get_layout_pattern<typename MaxExpTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, bfloat16_t>, "GroupMaxExp src must be bfloat16_t.");
    static_assert(AscendC::Std::is_same_v<MaxExpElementType, uint16_t>, "GroupMaxExp maxExp must be uint16_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<MaxExpTensor>, asc::te::location::ub>,
                  "GroupMaxExp only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<MaxExpLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GroupMaxExp only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
    MxMaxExpVfParams params{reinterpret_cast<__ubuf__ bfloat16_t*>(srcTensor.data().get()),
                            reinterpret_cast<__ubuf__ uint16_t*>(maxExpTensor.data().get()), DataLoopCount(totalCount),
                            totalCount};
    // Resolve the mode outside the VF to keep its vector loop free of runtime branches.
    if (totalCount % (static_cast<uint32_t>(VL_FOR_16F) * INTERLEAVED_REG_FACTOR) == 0) {
        if (useAbs) {
            // Full iterations: max absolute value.
            asc_vf_call<MaxExpVf<true, true, true>>(params);
        } else {
            // Full iterations: max exponent.
            asc_vf_call<MaxExpVf<true, false, true>>(params);
        }
    } else {
        if (useAbs) {
            // Handle tail: max absolute value.
            if (totalCount % INTERLEAVED_REG_FACTOR == 0) {
                asc_vf_call<MaxExpVf<false, true, true>>(params);
            } else {
                asc_vf_call<MaxExpVf<false, true, false>>(params);
            }
        } else {
            // Handle tail: max exponent.
            if (totalCount % INTERLEAVED_REG_FACTOR == 0) {
                asc_vf_call<MaxExpVf<false, false, true>>(params);
            } else {
                asc_vf_call<MaxExpVf<false, false, false>>(params);
            }
        }
    }
}

// ===========================================================================
// GenScale – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename MaxExpTensor, typename ScaleTensor, typename ReciprocalTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::GenScale(const MaxExpTensor& maxExpTensor,
                                                       const ScaleTensor& yScaleTensor,
                                                       const ReciprocalTensor& reciprocalTensor,
                                                       const MxQuantConfig& cfg, uint32_t totalScale)
{
    using MaxExpElementType = asc::te::get_attribute_element_type<typename MaxExpTensor::element_type*>;
    using ScaleElementType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;
    using ReciprocalElementType = asc::te::get_attribute_element_type<typename ReciprocalTensor::element_type*>;
    using MaxExpLayoutPattern = asc::te::get_layout_pattern<typename MaxExpTensor::layout_type>;
    using ScaleLayoutPattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
    using ReciprocalLayoutPattern = asc::te::get_layout_pattern<typename ReciprocalTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<MaxExpElementType, uint16_t>, "GenScale maxExp must be uint16_t.");
    static_assert(AscendC::Std::is_same_v<ScaleElementType, int8_t>, "GenScale yScale must be int8_t.");
    static_assert(AscendC::Std::is_same_v<ReciprocalElementType, uint16_t>, "GenScale reciprocal must be uint16_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<MaxExpTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ReciprocalTensor>, asc::te::location::ub>,
                  "GenScale only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<MaxExpLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<ScaleLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<ReciprocalLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GenScale only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
    MxScaleVfParams params{reinterpret_cast<__ubuf__ uint16_t*>(maxExpTensor.data().get()),
                           reinterpret_cast<__ubuf__ uint16_t*>(yScaleTensor.data().get()),
                           reinterpret_cast<__ubuf__ uint16_t*>(reciprocalTensor.data().get()),
                           ScaleLoopCount(totalScale, cfg.alg),
                           cfg.fpEmax,
                           cfg.addValueBits,
                           totalScale,
                           cfg.invDstTypeMax};
    if (cfg.alg == MxScaleAlg::OCP) {
        if (cfg.zeroScaleOnZeroExp) {
            if (totalScale % VL_FOR_16F == 0) {
                asc_vf_call<ScaleOcpVf<true, true>>(params);
            } else {
                asc_vf_call<ScaleOcpVf<true, false>>(params);
            }
        } else {
            if (totalScale % VL_FOR_16F == 0) {
                asc_vf_call<ScaleOcpVf<false, true>>(params);
            } else {
                asc_vf_call<ScaleOcpVf<false, false>>(params);
            }
        }
    } else if (cfg.alg == MxScaleAlg::CUBLAS) {
        if (totalScale % (VL_FOR_16F / HALF_REG_FACTOR) == 0) {
            asc_vf_call<ScaleCublasVf<true>>(params);
        } else {
            asc_vf_call<ScaleCublasVf<false>>(params);
        }
    } else {
        if (totalScale % VL_FOR_16F == 0) {
            asc_vf_call<ScaleDynVf<true>>(params);
        } else {
            asc_vf_call<ScaleDynVf<false>>(params);
        }
    }
}

// ===========================================================================
// Quantize – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename SrcTensor, typename ReciprocalTensor, typename YTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::Quantize(const SrcTensor& srcTensor,
                                                       const ReciprocalTensor& reciprocalTensor, const YTensor& yTensor,
                                                       uint32_t totalCount, MxQuantFp4RoundMode fp4RoundMode)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using ReciprocalElementType = asc::te::get_attribute_element_type<typename ReciprocalTensor::element_type*>;
    using YElementType = asc::te::get_attribute_element_type<typename YTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using ReciprocalLayoutPattern = asc::te::get_layout_pattern<typename ReciprocalTensor::layout_type>;
    using YLayoutPattern = asc::te::get_layout_pattern<typename YTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, bfloat16_t>, "Quantize src must be bfloat16_t.");
    static_assert(AscendC::Std::is_same_v<ReciprocalElementType, uint16_t>, "Quantize reciprocal must be uint16_t.");
    static_assert(AscendC::Std::is_same_v<YElementType, int8_t>, "Quantize y must be int8_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ReciprocalTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<YTensor>, asc::te::location::ub>,
                  "Quantize only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<ReciprocalLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<YLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "Quantize only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint16_t VL_FOR_16F = static_cast<uint16_t>(asc_get_vf_len() / sizeof(bfloat16_t));
    MxQuantizeVfParams params{reinterpret_cast<__ubuf__ bfloat16_t*>(srcTensor.data().get()),
                              reinterpret_cast<__ubuf__ uint16_t*>(reciprocalTensor.data().get()),
                              reinterpret_cast<__ubuf__ int8_t*>(yTensor.data().get()), DataLoopCount(totalCount),
                              totalCount};
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        if (totalCount % (static_cast<uint32_t>(VL_FOR_16F) * INTERLEAVED_REG_FACTOR) == 0) {
            asc_vf_call<QuantizeFp8Vf<true>>(params);
        } else {
            asc_vf_call<QuantizeFp8Vf<false>>(params);
        }
    } else {
        if (fp4RoundMode == MxQuantFp4RoundMode::FLOOR) {
            asc_vf_call<QuantizeFp4Vf<AscendC::RoundMode::CAST_FLOOR>>(params);
        } else if (fp4RoundMode == MxQuantFp4RoundMode::ROUND) {
            asc_vf_call<QuantizeFp4Vf<AscendC::RoundMode::CAST_ROUND>>(params);
        } else {
            asc_vf_call<QuantizeFp4Vf<AscendC::RoundMode::CAST_RINT>>(params);
        }
    }
}

// ===========================================================================
// TransScaleLayout – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename ScaleTensor, typename ScaleBlockTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::TransScaleLayout(const ScaleTensor& srcTensor,
                                                               const ScaleBlockTensor& dstTensor, uint16_t mSize,
                                                               uint16_t scaleBlockN)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename ScaleBlockTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename ScaleBlockTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, int8_t> && AscendC::Std::is_same_v<DstElementType, int8_t>,
                  "TransScaleLayout tensors must be int8_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleBlockTensor>, asc::te::location::ub>,
                  "TransScaleLayout only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "TransScaleLayout only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    MxTransScaleVfParams params{reinterpret_cast<__ubuf__ int8_t*>(srcTensor.data().get()),
                                reinterpret_cast<__ubuf__ int8_t*>(dstTensor.data().get()), mSize, scaleBlockN};
    asc_vf_call<TransScaleVf>(params);
}

// ===========================================================================
// PadScaleRows – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename ScaleTensor, typename ScaleBlockTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::PadScaleRows(const ScaleTensor& srcTensor,
                                                           const ScaleBlockTensor& dstTensor, uint16_t mSize,
                                                           uint32_t scaleCols, uint32_t scaleCopyCols)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename ScaleBlockTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename ScaleBlockTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, int8_t> && AscendC::Std::is_same_v<DstElementType, int8_t>,
                  "PadScaleRows tensors must be int8_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleBlockTensor>, asc::te::location::ub>,
                  "PadScaleRows only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "PadScaleRows only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    MxPadScaleRowsVfParams params{reinterpret_cast<__ubuf__ int8_t*>(srcTensor.data().get()),
                                  reinterpret_cast<__ubuf__ int8_t*>(dstTensor.data().get()), mSize, scaleCols,
                                  scaleCopyCols};
    asc_vf_call<PadScaleRowsVf>(params);
}

// ===========================================================================
// TransFp4OutLayout – public high-level
// ===========================================================================
template <typename DataTypeOut_>
template <typename YTensor, typename YBlockTensor>
__aicore__ inline void MxQuant<DataTypeOut_>::TransFp4OutLayout(const YTensor& srcTensor, const YBlockTensor& dstTensor,
                                                                uint16_t mSize, uint16_t nSize)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename YTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename YBlockTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename YTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename YBlockTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<SrcElementType, int8_t> && AscendC::Std::is_same_v<DstElementType, int8_t>,
                  "TransFp4OutLayout tensors must be int8_t.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<YTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<YBlockTensor>, asc::te::location::ub>,
                  "TransFp4OutLayout only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "TransFp4OutLayout only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t packedBytes = static_cast<uint32_t>(nSize) / 2;
    MxTransFp4OutVfParams params{reinterpret_cast<__ubuf__ int8_t*>(srcTensor.data().get()),
                                 reinterpret_cast<__ubuf__ int8_t*>(dstTensor.data().get()),
                                 mSize,
                                 static_cast<uint16_t>(packedBytes),
                                 static_cast<uint32_t>(Gemm::Align16(packedBytes)),
                                 static_cast<uint32_t>(Gemm::Align32(packedBytes))};
    asc_vf_call<TransFp4OutVf>(params);
}

} // namespace Blaze::Epilogue::Tile
