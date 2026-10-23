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
 * \file gelu.h
 * \brief Tile-level GELU epilogue: Tanh approximation and Erf exact.
 *
 * Design:
 *   - Public  (__aicore__): accepts make_tensor-created UB tensors, extracts
 *     raw __ubuf__ pointers via .data().get(), and delegates to Vf.
 *   - Private (__simd_vf__): Reg API register-level computation.
 *     Pure vector instructions (LoadAlign/StoreAlign, Mul, Axpy, Exp, Log, Div, Cast,
 *     Compare/Select). Both algorithms are fully in-register: one
 *     Vf launch covers the whole [mSize, nSize] tile.
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Epilogue::Tile {

// ---------------------------------------------------------------------------
// Gelu tile: reusable activation epilogue
//   Any input/output combination of float / bfloat16_t / half:
//     - non-fp32 input  : widened to fp32 (CT_16F_TO_32F) before computing
//     - fp32 output     : stored directly, no narrowing cast
//     - bf16 output     : narrowed with CT_32F_TO_16F (NO_SAT)
//     - half output     : narrowed with CT_32F_TO_16F_SAT (clamps at 65504)
//   kPreciseDiv: per-caller division mode (GELU (tanh) lineage = true:
//   error-compensation DivPrecisionImpl; GELU MX lineage = false: plain vdiv).
//   Both lineages are golden-verified with their own mode, so the divergence
//   is parameterized instead of silently unified.
// ---------------------------------------------------------------------------
template <typename DataTypeOut_, typename DataTypeIn_, bool kPreciseDiv_ = false, bool kPadToMxGroup_ = false>
class Gelu {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    static_assert(!kPadToMxGroup_ || AscendC::IsSameType<DataTypeOut, bfloat16_t>::value,
                  "MX tail padding requires BF16 output with a zeroing cast.");

    __aicore__ inline Gelu() = default;

    /*!
     * Tanh-approximation GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
     * Equivalent sigmoid form: x / (1 + exp(-2*sqrt(2/pi)*(x+0.044715*x^3)))
     * Accepts make_tensor-created UB tensors, extracts __ubuf__ for Vf.
     */
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void GeluTanh(const SrcTensor& srcTensor, const DstTensor& dstTensor, uint16_t mSize,
                                    uint16_t nSize);

    /*!
     * Erf-based GELU: 0.5*x*(1+erf(x/sqrt(2)))
     * Accepts make_tensor-created UB tensors, extracts __ubuf__ for Vf.
     * Pure register pipeline: scale x/sqrt(2), subsection polynomial erf
     * (ErfCallee) and (1+erf)*(0.5*x) assembly in a single Vf launch.
     */
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void GeluErf(const SrcTensor& srcTensor, const DstTensor& dstTensor, uint16_t mSize,
                                   uint16_t nSize);

private:
    // Constants
    static constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;
    static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
    static constexpr float ONE_OVER_SQRT_TWO = 0.707106781f;
    static constexpr float GELU_HALF = 0.5f;
    static constexpr float GELU_ONE = 1.0f;

#ifdef __NPU_ARCH__
    // Erf subsection polynomial approximation coefficients (fp32 bit patterns),
    // ported from impl/adv_api/detail/math/erf/erf_3510_impl.h (ErfAPI).
    // P1: |x| >= ERF_C0 branch, P2: |x| < ERF_C0 branch.
    static constexpr uint32_t ERF_C0 = 0x3F8060FE;
    static constexpr uint32_t ERF_P1[] = {0x38EB4C3A, 0xBAAE005B, 0x3C09919F, 0xBD24D99A,
                                          0x3E235519, 0x3F69B4F9, 0x3F210A14};
    static constexpr uint32_t ERF_P2[] = {0x38B1E96A, 0xBA574D20, 0x3BAAD5EA, 0xBCDC1BE7,
                                          0x3DE718AF, 0xBEC093AC, 0x3E0375D3};

    // Cast traits (compile-time constexpr, shared between high-level and Vf).
    // Named by bit-width: 16F covers both half and bfloat16_t.
    // Narrowing fp32 -> 16F, no saturation: safe for bf16 (shares fp32's exponent range).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // Narrowing fp32 -> 16F with saturation: for half (max 65504), clamps instead of inf.
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F_SAT = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // Widening 16F -> fp32 (half / bfloat16_t); Sat/Round are ignored for widening.
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    // Per-instantiation narrowing trait selected by output type:
    // half -> SAT (avoid inf beyond 65504); bf16 -> NO_SAT (no overflow possible);
    // float output skips the narrowing cast entirely (if constexpr at call sites).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_OUT = AscendC::IsSameType<DataTypeOut, half>::value ?
                                                                 CT_32F_TO_16F_SAT :
                                                                 CT_32F_TO_16F;

    // Division mode selected per instantiation. {ZEROING, false} is bit-identical
    // to the default Div<float> template (GetDivSpecificMode(MaskMergeMode) maps to
    // {ZEROING, false, INTRINSIC} -> plain vdiv); true switches to the
    // error-compensation precision division (2-3x instruction count per Div).
    static constexpr AscendC::Reg::DivSpecificMode GELU_DIV_MODE = {
        AscendC::Reg::MaskMergeMode::ZEROING,
        kPreciseDiv_,
    };
#endif // __NPU_ARCH__

    // Vf parameter structs: raw __ubuf__ pointers + shape info for Reg API.
    // Shared by the tanh and erf Vf entries (identical tile shape contract).
    template <typename DataTypeOut, typename DataTypeIn>
    struct GeluVfParams {
        __ubuf__ DataTypeOut* dstAddr;
        __ubuf__ DataTypeIn* srcAddr;
        uint16_t mSize;
        uint16_t nSize;
        uint16_t sizePerRepeat;
        uint16_t oneRowRepeatTimes;
        uint32_t nAligned;
    };

    /*!
     * Callee: fp32 input register → GELU (tanh) math → narrow to DataTypeOut → store.
     * Shared by GeluTanhVf (fp32 input loaded directly) and GeluTanhCastInVf
     * (16F input widened to fp32 before the call). Mirrors the
     * __simd_callee__ composition pattern of gemm/tile/arch35/shift_w4_to_w8.h.
     */
    static __simd_callee__ inline void GeluTanhCoreCallee(AscendC::Reg::RegTensor<float>& vregInput,
                                                          AscendC::Reg::MaskReg& mask, AscendC::Reg::MaskReg& storeMask,
                                                          __ubuf__ DataTypeOut* dstAddr, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> vregInputSqr;
        AscendC::Reg::RegTensor<float> vregInputCub;
        AscendC::Reg::RegTensor<float> vregOutput;
        AscendC::Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
        AscendC::Reg::Mul(vregInputCub, vregInputSqr, vregInput, mask);
        AscendC::Reg::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
        AscendC::Reg::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
        AscendC::Reg::Exp(vregInputCub, vregInputCub, mask);
        AscendC::Reg::Adds(vregInputCub, vregInputCub, GELU_ONE, mask);
        AscendC::Reg::Div<float, &GELU_DIV_MODE>(vregOutput, vregInput, vregInputCub, mask);
        if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, vregOutput, mask);
        } else {
            AscendC::Reg::RegTensor<DataTypeOut> vregOutputOut;
            AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_OUT>(vregOutputOut, vregOutput, mask);
            AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset,
                                                                                          vregOutputOut, storeMask);
        }
    }

    /*!
     * VF: fp32 input → load → delegate to GeluTanhCoreCallee.
     */
    template <bool FULL_REPEAT>
    static __simd_vf__ inline void GeluTanhVf(GeluVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        auto mask = AscendC::Reg::CreateMask<float>();

        __ubuf__ float* src = reinterpret_cast<__ubuf__ float*>(params.srcAddr);
        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            uint32_t storeCount = params.nAligned;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                if constexpr (!FULL_REPEAT) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                }
                auto storeMask = mask;
                if constexpr (kPadToMxGroup_ && !FULL_REPEAT) {
                    storeMask = AscendC::Reg::UpdateMask<float>(storeCount);
                }
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregInput, src + offset);
                GeluTanhCoreCallee(vregInput, mask, storeMask, params.dstAddr, offset);
            }
        }
    }

    /*!
     * VF: 16F input → unpack + widen to fp32 → delegate to GeluTanhCoreCallee.
     */
    template <bool FULL_REPEAT>
    static __simd_vf__ inline void GeluTanhCastInVf(GeluVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<DataTypeIn> vregInput16;
        auto mask = AscendC::Reg::CreateMask<float>();

        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            uint32_t storeCount = params.nAligned;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                if constexpr (!FULL_REPEAT) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                }
                auto storeMask = mask;
                if constexpr (kPadToMxGroup_ && !FULL_REPEAT) {
                    storeMask = AscendC::Reg::UpdateMask<float>(storeCount);
                }
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::LoadAlign<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                                             params.srcAddr + offset);
                AscendC::Reg::Cast<float, DataTypeIn, CT_16F_TO_32F>(vregInput, vregInput16, mask);
                GeluTanhCoreCallee(vregInput, mask, storeMask, params.dstAddr, offset);
            }
        }
    }

    /*!
     * Callee: dst = cmpMask ? bitsAsFp32(hiBits) : bitsAsFp32(loBits).
     * Coefficient select of the erf subsection polynomial, mirroring the
     * Duplicate + Select pairs of erf_3510_impl.h (ErfAPI).
     */
    static __simd_callee__ inline void ErfSelectConstCallee(AscendC::Reg::RegTensor<float>& dstReg, uint32_t hiBits,
                                                            uint32_t loBits, AscendC::Reg::MaskReg& cmpMask,
                                                            AscendC::Reg::MaskReg& mask)
    {
        AscendC::Reg::RegTensor<uint32_t> hiReg;
        AscendC::Reg::RegTensor<uint32_t> loReg;
        AscendC::Reg::Duplicate(hiReg, hiBits, mask);
        AscendC::Reg::Duplicate(loReg, loBits, mask);
        AscendC::Reg::Select(dstReg, (AscendC::Reg::RegTensor<float>&)hiReg, (AscendC::Reg::RegTensor<float>&)loReg,
                             cmpMask);
    }

    /*!
     * Callee: erf(x) via subsection polynomial approximation, a register-level
     * port of ErfSubsectionCompute/ErfSpecialCaseCompute in
     * impl/adv_api/detail/math/erf/erf_3510_impl.h:
     *   |x| >= C0 : y = copysign(1 - exp(f26*ln2), x)
     *   |x| <  C0 : y = f26
     * where f26 is the branch-selected Horner polynomial.
     */
    static __simd_callee__ inline void ErfCallee(AscendC::Reg::RegTensor<float>& dstReg,
                                                 AscendC::Reg::RegTensor<float>& srcReg, AscendC::Reg::MaskReg& mask)
    {
        constexpr uint32_t ERF_R1 = 0x3F800000; // 1.0f bit pattern
        constexpr uint32_t ERF_R2 = 0x80000000; // sign mask
        constexpr float ERF_LOG2_VALUE = 2.0f;  // ln2 = Log(2.0f)

        AscendC::Reg::RegTensor<float> absReg;
        AscendC::Reg::RegTensor<float> argReg;
        AscendC::Reg::RegTensor<float> accReg;
        AscendC::Reg::RegTensor<float> coeffReg;
        AscendC::Reg::RegTensor<float> tailReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<uint32_t> u32Reg;
        AscendC::Reg::MaskReg cmpMask;

        // p2 = |x| >= C0; arg = p2 ? |x| : x*x
        AscendC::Reg::Abs(absReg, srcReg, mask);
        AscendC::Reg::Duplicate(u32Reg, ERF_C0, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(cmpMask, absReg, (AscendC::Reg::RegTensor<float>&)u32Reg,
                                                           mask);
        AscendC::Reg::Mul(tailReg, srcReg, srcReg, mask);
        AscendC::Reg::Select(argReg, absReg, tailReg, cmpMask);

        // Horner: acc = P[0]; acc = acc * arg + P[i], with P[i] = p2 ? P1[i] : P2[i]
        ErfSelectConstCallee(accReg, ERF_P1[0], ERF_P2[0], cmpMask, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[1], ERF_P2[1], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[2], ERF_P2[2], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[3], ERF_P2[3], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[4], ERF_P2[4], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[5], ERF_P2[5], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);
        ErfSelectConstCallee(coeffReg, ERF_P1[6], ERF_P2[6], cmpMask, mask);
        AscendC::Reg::FusedMulDstAdd(accReg, argReg, coeffReg, mask);

        // tailArg = p2 ? -|x| : x; f26 = acc * tailArg + tailArg
        AscendC::Reg::Neg(tailReg, absReg, mask);
        AscendC::Reg::Select(tailReg, tailReg, srcReg, cmpMask);
        AscendC::Reg::FusedMulDstAdd(accReg, tailReg, tailReg, mask);

        // Special case: |x| < C0 keeps f26, otherwise 1 - exp(f26 * ln2)
        AscendC::Reg::Duplicate(u32Reg, ERF_C0, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::LT>(cmpMask, absReg, (AscendC::Reg::RegTensor<float>&)u32Reg,
                                                           mask);
        AscendC::Reg::Duplicate(expReg, ERF_LOG2_VALUE, mask);
        AscendC::Reg::Log(expReg, expReg, mask);
        AscendC::Reg::Mul(expReg, accReg, expReg, mask);
        AscendC::Reg::Exp(expReg, expReg, mask);
        AscendC::Reg::Duplicate(u32Reg, ERF_R1, mask);
        AscendC::Reg::Sub(expReg, (AscendC::Reg::RegTensor<float>&)u32Reg, expReg, mask);
        AscendC::Reg::Duplicate(u32Reg, ERF_R2, mask);
        AscendC::Reg::And(u32Reg, (AscendC::Reg::RegTensor<uint32_t>&)srcReg, u32Reg, mask);
        AscendC::Reg::Or(u32Reg, u32Reg, (AscendC::Reg::RegTensor<uint32_t>&)expReg, mask);
        AscendC::Reg::Select(dstReg, accReg, (AscendC::Reg::RegTensor<float>&)u32Reg, cmpMask);
    }

    /*!
     * Callee: fp32 input register → x/sqrt(2) → ErfCallee → (1+erf)*(0.5*x)
     * → narrow to DataTypeOut → store. Shared by GeluErfVf (fp32 input loaded
     * directly) and GeluErfCastInVf (16F input widened to fp32 before the call).
     */
    static __simd_callee__ inline void GeluErfCoreCallee(AscendC::Reg::RegTensor<float>& vregInput,
                                                         AscendC::Reg::MaskReg& mask, AscendC::Reg::MaskReg& storeMask,
                                                         __ubuf__ DataTypeOut* dstAddr, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> vregScaled;
        AscendC::Reg::RegTensor<float> vregErf;
        AscendC::Reg::RegTensor<float> vregAdds;
        AscendC::Reg::RegTensor<float> vregMuls;
        AscendC::Reg::RegTensor<float> vregOutput;
        AscendC::Reg::Muls(vregScaled, vregInput, ONE_OVER_SQRT_TWO, mask);
        ErfCallee(vregErf, vregScaled, mask);
        AscendC::Reg::Adds(vregAdds, vregErf, GELU_ONE, mask);
        AscendC::Reg::Muls(vregMuls, vregInput, GELU_HALF, mask);
        AscendC::Reg::Mul(vregOutput, vregAdds, vregMuls, mask);
        if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, vregOutput, mask);
        } else {
            AscendC::Reg::RegTensor<DataTypeOut> vregOutputOut;
            AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_OUT>(vregOutputOut, vregOutput, mask);
            AscendC::Reg::StoreAlign<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset,
                                                                                          vregOutputOut, storeMask);
        }
    }

    /*!
     * VF: fp32 input → load → delegate to GeluErfCoreCallee.
     */
    template <bool FULL_REPEAT>
    static __simd_vf__ inline void GeluErfVf(GeluVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        auto mask = AscendC::Reg::CreateMask<float>();

        __ubuf__ float* src = reinterpret_cast<__ubuf__ float*>(params.srcAddr);
        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            uint32_t storeCount = params.nAligned;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                if constexpr (!FULL_REPEAT) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                }
                auto storeMask = mask;
                if constexpr (kPadToMxGroup_ && !FULL_REPEAT) {
                    storeMask = AscendC::Reg::UpdateMask<float>(storeCount);
                }
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregInput, src + offset);
                GeluErfCoreCallee(vregInput, mask, storeMask, params.dstAddr, offset);
            }
        }
    }

    /*!
     * VF: 16F input → unpack + widen to fp32 → delegate to GeluErfCoreCallee.
     */
    template <bool FULL_REPEAT>
    static __simd_vf__ inline void GeluErfCastInVf(GeluVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<DataTypeIn> vregInput16;
        auto mask = AscendC::Reg::CreateMask<float>();

        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            uint32_t storeCount = params.nAligned;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                if constexpr (!FULL_REPEAT) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                }
                auto storeMask = mask;
                if constexpr (kPadToMxGroup_ && !FULL_REPEAT) {
                    storeMask = AscendC::Reg::UpdateMask<float>(storeCount);
                }
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::LoadAlign<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                                             params.srcAddr + offset);
                AscendC::Reg::Cast<float, DataTypeIn, CT_16F_TO_32F>(vregInput, vregInput16, mask);
                GeluErfCoreCallee(vregInput, mask, storeMask, params.dstAddr, offset);
            }
        }
    }
};

// ===========================================================================
// GeluTanh – public interface
// ===========================================================================
template <typename DataTypeOut_, typename DataTypeIn_, bool kPreciseDiv_, bool kPadToMxGroup_>
template <typename SrcTensor, typename DstTensor>
__aicore__ inline void Gelu<DataTypeOut_, DataTypeIn_, kPreciseDiv_, kPadToMxGroup_>::GeluTanh(
    const SrcTensor& srcTensor, const DstTensor& dstTensor, uint16_t mSize, uint16_t nSize)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float> || AscendC::Std::is_same_v<DataTypeIn, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeIn, half>,
                  "GeluTanh input must be float, bfloat16_t or half.");
    static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeOut, half>,
                  "GeluTanh output must be float, bfloat16_t or half.");
    static_assert(
        AscendC::Std::is_same_v<SrcElementType, DataTypeIn> && AscendC::Std::is_same_v<DstElementType, DataTypeOut>,
        "GeluTanh tensor element types must match DataTypeIn/DataTypeOut.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                  "GeluTanh only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GeluTanh only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint16_t sizePerRepeat = static_cast<uint16_t>(asc_get_vf_len() / sizeof(float));
    uint16_t oneRowRepeatTimes = Gemm::CeilDiv(static_cast<uint32_t>(nSize), static_cast<uint32_t>(sizePerRepeat));
    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
    GeluVfParams<DataTypeOut, DataTypeIn> params{reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get()),
                                                 reinterpret_cast<__ubuf__ DataTypeIn*>(srcTensor.data().get()),
                                                 mSize,
                                                 nSize,
                                                 sizePerRepeat,
                                                 oneRowRepeatTimes,
                                                 nAligned};
    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        if (nSize % sizePerRepeat == 0) {
            asc_vf_call<GeluTanhVf<true>>(params);
        } else {
            asc_vf_call<GeluTanhVf<false>>(params);
        }
    } else {
        if (nSize % sizePerRepeat == 0) {
            asc_vf_call<GeluTanhCastInVf<true>>(params);
        } else {
            asc_vf_call<GeluTanhCastInVf<false>>(params);
        }
    }
}

// ===========================================================================
// GeluErf – public interface
// ===========================================================================
template <typename DataTypeOut_, typename DataTypeIn_, bool kPreciseDiv_, bool kPadToMxGroup_>
template <typename SrcTensor, typename DstTensor>
__aicore__ inline void Gelu<DataTypeOut_, DataTypeIn_, kPreciseDiv_, kPadToMxGroup_>::GeluErf(
    const SrcTensor& srcTensor, const DstTensor& dstTensor, uint16_t mSize, uint16_t nSize)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float> || AscendC::Std::is_same_v<DataTypeIn, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeIn, half>,
                  "GeluErf input must be float, bfloat16_t or half.");
    static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeOut, half>,
                  "GeluErf output must be float, bfloat16_t or half.");
    static_assert(
        AscendC::Std::is_same_v<SrcElementType, DataTypeIn> && AscendC::Std::is_same_v<DstElementType, DataTypeOut>,
        "GeluErf tensor element types must match DataTypeIn/DataTypeOut.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                  "GeluErf only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GeluErf only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint16_t sizePerRepeat = static_cast<uint16_t>(asc_get_vf_len() / sizeof(float));
    uint16_t oneRowRepeatTimes = Gemm::CeilDiv(static_cast<uint32_t>(nSize), static_cast<uint32_t>(sizePerRepeat));
    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
    GeluVfParams<DataTypeOut, DataTypeIn> params{reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get()),
                                                 reinterpret_cast<__ubuf__ DataTypeIn*>(srcTensor.data().get()),
                                                 mSize,
                                                 nSize,
                                                 sizePerRepeat,
                                                 oneRowRepeatTimes,
                                                 nAligned};
    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        if (nSize % sizePerRepeat == 0) {
            asc_vf_call<GeluErfVf<true>>(params);
        } else {
            asc_vf_call<GeluErfVf<false>>(params);
        }
    } else {
        if (nSize % sizePerRepeat == 0) {
            asc_vf_call<GeluErfCastInVf<true>>(params);
        } else {
            asc_vf_call<GeluErfCastInVf<false>>(params);
        }
    }
}

} // namespace Blaze::Epilogue::Tile
