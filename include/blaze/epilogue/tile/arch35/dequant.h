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
 * \file dequant.h
 * \brief Vector dequantization tile, dispatched on two axes:
 *        1. class template parameter (quantized-object tag): Fp4ToFp8Tag (prologue
 *           packed-FP4 weight -> FP8, before MMAD, feeding L1 fractal layouts), B32ToFpTag
 *           (any int32/fp32 source -> fp32/bf16/fp16 with x2/x1 scales + bias; the source is
 *           not required to be an MMAD accumulator).
 *        2. Run template parameter (quantization granularity, DequantMode): per-tensor / per-channel /
 *           per-token / per-group. Dtypes and layouts are derived from the Run tensor arguments.
 *        Per-family notes:
 *        - Dequant<Fp4ToFp8Tag>: per-group only (group size in Run's GroupSize); layout dispatch
 *          (DNExt ND-trans / NZ) rides on the InputTensor type; the W4 VF kernels are private
 *          static members and are not exposed. Issued from the AIV prologue (no AIC guard).
 *        - Dequant<B32ToFpTag>: x2 scale (PER_TENSOR/PER_CHANNEL) * x1 scale (NONE/PER_TENSOR/
 *          PER_TOKEN) + bias, one VF launch per Run call, register-resident end to end. AIV-only
 *          (AIC guard retained). Do NOT unify the guards across families.
 *        - 16F output narrowing of the B32 family is NO_SAT (dequant lineage), deliberately
 *          different from Gelu's half-SAT policy; per-family precision baselines are kept.
 */

#pragma once

#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Epilogue::Tile {
namespace Reg = AscendC::Reg;
using AscendC::IsSameType;

// Quantization granularity (Run template parameter): the second dispatch axis. Each family
// supports a subset, enforced by static_assert at the Run entry.
enum class DequantMode : uint8_t {
    NONE = 0,        // axis carries no scale (B32 x1 axis only)
    PER_TENSOR = 1,  // global scalar
    PER_CHANNEL = 2, // vector scale along N (B32 x2 axis / weight N dim)
    PER_TOKEN = 3,   // vector scale along M/rows (Acc x1 axis)
    PER_GROUP = 4,   // scale along K groups (W4 weight; group size rides in Run's GroupSize)
};

// Quantized-object tags (class template parameter): the first dispatch axis. The tag selects
// the dequant semantics (object + pipeline stage); layouts and dtypes follow the tensors.
struct Fp4ToFp8Tag {}; // prologue: packed FP4 weight -> FP8, before MMAD
struct B32ToFpTag {};  // int32/fp32 -> fp32/bf16/fp16 with x2/x1 scales + bias (any source)

// Per-group W4 runtime parameters: validK floors validK / GROUP_SIZE (a trailing partial K
// group is not converted), a value no layout encodes; scaleMaskAddr carries the VF-internal
// scale-select mask blob owned by the caller's UB storage.
struct A8W4TcgDequantParams {
    uint32_t validK{0};
    __ubuf__ uint8_t* scaleMaskAddr{nullptr};
};

// Primary template: declaration only; unsupported tags fail at compile time.
template <class DequantTag>
class Dequant;

// Per-group W4 weight dequant (prologue): packed FP4 weight + per-group scale -> FP8 (group
// size 32 only). Issued from the AIV-side weight prologue, so no AIC guard here.
template <>
class Dequant<Fp4ToFp8Tag> {
public:
    static constexpr uint32_t GROUP_SIZE_32 = 32;

    template <DequantMode Gran, uint32_t GroupSize, class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void Run(const InputTensor& input, const ScaleTensor& scale, const OutTensor& output,
                                      const A8W4TcgDequantParams& params)
    {
        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;
        // Entry gate: only the supported per-group quantization shapes may proceed.
        static_assert(Gran == DequantMode::PER_GROUP, "W4 weight dequant only supports the PER_GROUP granularity.");
        static_assert(GroupSize == GROUP_SIZE_32, "Per-group W4 dequant currently only supports a group size of 32.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<InputTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<ScaleTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<OutTensor>, asc::te::location::ub>,
                      "Per-group W4 dequant only supports UB tensors.");
        static_assert(AscendC::IsSameType<XType, fp8_e4m3fn_t>::value,
                      "Per-group W4 dequant expects fp8_e4m3fn output.");
        static_assert(Blaze::Gemm::IsFp4<WType>(), "Per-group W4 dequant expects packed FP4 input.");
        static_assert(AscendC::IsSameType<SType, half>::value || AscendC::IsSameType<SType, bfloat16_t>::value,
                      "Per-group W4 dequant expects FP16/BF16 scale.");

        using InputPattern = asc::te::get_layout_pattern<typename InputTensor::layout_type>;
        constexpr bool IS_DN_TRANS_INPUT = AscendC::Std::is_same_v<InputPattern, asc::te::dn_ext_layout_ptn>;
        constexpr bool IS_NZ_INPUT = AscendC::Std::is_same_v<InputPattern, asc::te::nz_layout_ptn>;
        static_assert(IS_DN_TRANS_INPUT || IS_NZ_INPUT,
                      "Per-group W4 dequant supports DNExt (ND-trans) and NZ input layouts.");

        if constexpr (IS_DN_TRANS_INPUT) {
            RunNdTrans(input, scale, output, params);
        } else {
            RunNz<GroupSize>(input, scale, output, params);
        }
    }

private:
#ifdef __NPU_ARCH__
    static constexpr int32_t PERGROUP_OFFSET_8 = 8;
    static constexpr int32_t PERGROUP_OFFSET_FOR_4_BITS = 128;
    static constexpr int32_t OFFSET_8 = 8;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr int32_t ALIGNED_32_SIZE = 32;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4_BITS = 64;
    static constexpr int64_t C0_SIZE_B8 = static_cast<int64_t>(Blaze::Gemm::C0_SIZE_B8);
    static constexpr int32_t INT4_PACK_SHIFT = 1;
    static constexpr int64_t INT4_DTYPE_FACTOR = 2;

    static constexpr Reg::CastTrait CT_16F_TO_32F_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CT_16F_TO_32F_ONE = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CT_32F_TO_FP8_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait CT_32F_TO_FP8_TWO = {Reg::RegLayout::TWO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait CT_F4_TO_16F_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CT_BF16_TO_16F = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
#endif // __NPU_ARCH__

    template <typename xType, typename wType, typename scaleType>
    struct DequantW4Pergroup32NKParams {
        uint32_t maskWeight;
        uint16_t outerExtend;
        uint16_t innerExtend;
        uint32_t outerStrideScale;
        uint32_t outerStrideWeight;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        __ubuf__ scaleType* scaleBaseAddr;
        __ubuf__ int8_t* weightInBaseAddr0;
        __ubuf__ int8_t* weightInBaseAddr1;
        __ubuf__ xType* weightOutBaseAddr;
    };

    template <typename xType, typename wType, typename scaleType>
    struct DequantW4PergroupKNParams {
        uint32_t groupNumUb;
        uint32_t vLLoopNumInGroup;
        uint32_t n1LoopNum;
        uint32_t scaleN1Stride;
        uint32_t bubNLen;
        uint32_t weightInGroupIdStride;
        uint32_t weightInN1Stride;
        uint32_t weightOutN1Stride;
        uint32_t weightOutGroupIdStride;
        uint32_t weightOutVlStride;
        __ubuf__ scaleType* scaleBaseAddr0;
        __ubuf__ scaleType* scaleBaseAddr1;
        __ubuf__ uint8_t* scaleMaskBaseAddr;
        __ubuf__ int8_t* weightInBaseAddr0;
        __ubuf__ int8_t* weightInBaseAddr1;
        __ubuf__ xType* weightOutBaseAddr;
    };

    template <typename wType, typename scaleType>
    static __simd_callee__ inline void CastWeightF4ToF16(Reg::RegTensor<scaleType>& weightF16Reg0,
                                                         Reg::RegTensor<scaleType>& weightF16Reg1,
                                                         Reg::RegTensor<wType>& weightInReg0,
                                                         Reg::RegTensor<wType>& weightInReg1, Reg::MaskReg& maskRegALL)
    {
        if constexpr (AscendC::IsSameType<scaleType, half>::value) {
            Reg::RegTensor<bfloat16_t> weightBF16Reg0, weightBF16Reg1;
            Reg::Cast<bfloat16_t, wType, CT_F4_TO_16F_ZERO>(weightBF16Reg0, weightInReg0, maskRegALL);
            Reg::Cast<bfloat16_t, wType, CT_F4_TO_16F_ZERO>(weightBF16Reg1, weightInReg1, maskRegALL);
            Reg::Cast<scaleType, bfloat16_t, CT_BF16_TO_16F>(weightF16Reg0, weightBF16Reg0, maskRegALL);
            Reg::Cast<scaleType, bfloat16_t, CT_BF16_TO_16F>(weightF16Reg1, weightBF16Reg1, maskRegALL);
        } else {
            Reg::Cast<scaleType, wType, CT_F4_TO_16F_ZERO>(weightF16Reg0, weightInReg0, maskRegALL);
            Reg::Cast<scaleType, wType, CT_F4_TO_16F_ZERO>(weightF16Reg1, weightInReg1, maskRegALL);
        }
    }

    template <typename xType, typename wType, typename scaleType>
    static __simd_vf__ inline void DequantW4Pergroup32NK(DequantW4Pergroup32NKParams<xType, wType, scaleType> p)
    {
        Reg::RegTensor<scaleType> scaleLoad, scaleCompute0, scaleCompute1;
        Reg::RegTensor<wType> wLoad0, wLoad1;
        Reg::RegTensor<scaleType> wCvt0, wCvt1, wMul0, wMul1;
        Reg::RegTensor<xType> wCvtB8N0, wCvtB8N1, wCvtB8N2, wCvtB8N3, wSel0, wSel1, wSel2, wSel3;
        Reg::RegTensor<xType> wDIntlv0, wDIntlv1;
        Reg::RegTensor<float> wCvtF32N0, wCvtF32N1, wCvtF32N2, wCvtF32N3;

        Reg::MaskReg maskRegB4 = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegB16 = Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();
        Reg::MaskReg maskWeight;
        uint32_t maskWeightTmp;

        for (uint16_t outerIdx = 0; outerIdx < p.outerExtend; ++outerIdx) {
            maskWeightTmp = p.maskWeight;
            for (uint16_t innerIdx = 0; innerIdx < p.innerExtend; ++innerIdx) {
                Reg::AddrReg addrRegScale = Reg::CreateAddrReg<scaleType>(outerIdx, p.outerStrideScale, innerIdx,
                                                                          PERGROUP_OFFSET_8);
                Reg::AddrReg addrRegWeight = Reg::CreateAddrReg<uint8_t>(outerIdx, p.outerStrideWeight, innerIdx,
                                                                         PERGROUP_OFFSET_FOR_4_BITS);
                maskWeight = Reg::UpdateMask<xType>(maskWeightTmp);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_E2B_B16>(scaleLoad, p.scaleBaseAddr, addrRegScale);
                Reg::Interleave(scaleCompute0, scaleCompute1, scaleLoad, scaleLoad);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)wLoad0, (__ubuf__ uint8_t*)p.weightInBaseAddr0, addrRegWeight);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)wLoad1, (__ubuf__ uint8_t*)p.weightInBaseAddr1, addrRegWeight);

                CastWeightF4ToF16<wType, scaleType>(wCvt0, wCvt1, wLoad0, wLoad1, maskRegB4);

                Reg::Mul(wMul0, wCvt0, scaleCompute0, maskRegB16);
                Reg::Mul(wMul1, wCvt1, scaleCompute1, maskRegB16);
                Reg::Cast<float, scaleType, CT_16F_TO_32F_ZERO>(wCvtF32N0, wMul0, maskRegB16);
                Reg::Cast<float, scaleType, CT_16F_TO_32F_ONE>(wCvtF32N1, wMul0, maskRegB16);
                Reg::Cast<float, scaleType, CT_16F_TO_32F_ZERO>(wCvtF32N2, wMul1, maskRegB16);
                Reg::Cast<float, scaleType, CT_16F_TO_32F_ONE>(wCvtF32N3, wMul1, maskRegB16);
                Reg::Cast<xType, float, CT_32F_TO_FP8_ZERO>(wCvtB8N0, wCvtF32N0, maskRegB16);
                Reg::Cast<xType, float, CT_32F_TO_FP8_TWO>(wCvtB8N1, wCvtF32N1, maskRegB16);
                Reg::Cast<xType, float, CT_32F_TO_FP8_ZERO>(wCvtB8N2, wCvtF32N2, maskRegB16);
                Reg::Cast<xType, float, CT_32F_TO_FP8_TWO>(wCvtB8N3, wCvtF32N3, maskRegB16);
                Reg::Select(wSel0, wCvtB8N0, wCvtB8N1, maskRegVsel);
                Reg::Select(wSel1, wCvtB8N2, wCvtB8N3, maskRegVsel);
                Reg::DeInterleave(wDIntlv0, wDIntlv1, wSel0, wSel1);

                Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBaseAddr, wDIntlv0, p.dataBlockStride, p.repeatStride, maskWeight);
            }
            p.weightOutBaseAddr += p.outDimOffset;
        }
    }

    template <typename xType, typename wType, typename scaleType>
    static __simd_vf__ inline void DequantW4PergroupKN(DequantW4PergroupKNParams<xType, wType, scaleType> param)
    {
        Reg::RegTensor<scaleType> scaleRegCompute, scaleRegAssist, weightF16Reg0, weightF16Reg1;
        Reg::RegTensor<wType> weightInReg0, weightInReg1;
        Reg::RegTensor<float> weightF32Reg0, weightF32Reg1, weightF32Reg2, weightF32Reg3;
        Reg::RegTensor<xType> weightF8Reg0, weightF8Reg1, weightF8Reg2, weightF8Reg3, weightF8SelReg0, weightF8SelReg1;
        Reg::MaskReg scaleMaskReg = Reg::CreateMask<uint8_t>();
        Reg::MaskReg maskRegALL = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();

        Reg::DataCopy(scaleMaskReg, param.scaleMaskBaseAddr);
        for (uint16_t n1Idx = 0; n1Idx < param.n1LoopNum; n1Idx++) {
            for (uint16_t groupIdx = 0; groupIdx < param.groupNumUb; groupIdx++) {
                Reg::AddrReg scaleAddrReg = Reg::CreateAddrReg<scaleType>(n1Idx, param.scaleN1Stride, groupIdx,
                                                                          param.bubNLen);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegCompute, param.scaleBaseAddr0, scaleAddrReg);
                Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegAssist, param.scaleBaseAddr1, scaleAddrReg);
                Reg::Select(scaleRegCompute, scaleRegCompute, scaleRegAssist, scaleMaskReg);
                for (uint16_t vLIdx = 0; vLIdx < param.vLLoopNumInGroup; vLIdx++) {
                    Reg::AddrReg weightInAddrReg = Reg::CreateAddrReg<uint8_t>(n1Idx, param.weightInN1Stride, groupIdx,
                                                                               param.weightInGroupIdStride, vLIdx,
                                                                               PERGROUP_OFFSET_FOR_4_BITS);
                    Reg::AddrReg weightOutAddrReg = Reg::CreateAddrReg<uint8_t>(n1Idx, param.weightOutN1Stride,
                                                                                groupIdx, param.weightOutGroupIdStride,
                                                                                vLIdx, param.weightOutVlStride);
                    Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg0,
                                                                           (__ubuf__ uint8_t*)param.weightInBaseAddr0,
                                                                           weightInAddrReg);
                    Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg1,
                                                                           (__ubuf__ uint8_t*)param.weightInBaseAddr1,
                                                                           weightInAddrReg);
                    CastWeightF4ToF16<wType, scaleType>(weightF16Reg0, weightF16Reg1, weightInReg0, weightInReg1,
                                                        maskRegALL);
                    Reg::Mul(weightF16Reg0, weightF16Reg0, scaleRegCompute, maskRegALL);
                    Reg::Mul(weightF16Reg1, weightF16Reg1, scaleRegCompute, maskRegALL);

                    Reg::Cast<float, scaleType, CT_16F_TO_32F_ZERO>(weightF32Reg0, weightF16Reg0, maskRegALL);
                    Reg::Cast<float, scaleType, CT_16F_TO_32F_ONE>(weightF32Reg1, weightF16Reg0, maskRegALL);
                    Reg::Cast<float, scaleType, CT_16F_TO_32F_ZERO>(weightF32Reg2, weightF16Reg1, maskRegALL);
                    Reg::Cast<float, scaleType, CT_16F_TO_32F_ONE>(weightF32Reg3, weightF16Reg1, maskRegALL);
                    Reg::Cast<xType, float, CT_32F_TO_FP8_ZERO>(weightF8Reg0, weightF32Reg0, maskRegALL);
                    Reg::Cast<xType, float, CT_32F_TO_FP8_TWO>(weightF8Reg1, weightF32Reg1, maskRegALL);
                    Reg::Cast<xType, float, CT_32F_TO_FP8_ZERO>(weightF8Reg2, weightF32Reg2, maskRegALL);
                    Reg::Cast<xType, float, CT_32F_TO_FP8_TWO>(weightF8Reg3, weightF32Reg3, maskRegALL);
                    Reg::Select(weightF8SelReg0, weightF8Reg0, weightF8Reg1, maskRegVsel);
                    Reg::Select(weightF8SelReg1, weightF8Reg2, weightF8Reg3, maskRegVsel);

                    Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr, weightF8SelReg0,
                                                                        weightOutAddrReg, maskRegALL);
                    Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr + 128, weightF8SelReg1,
                                                                        weightOutAddrReg, maskRegALL);
                }
            }
        }
    }

    // ND-trans (bTrans == true) path: DN fp4 input -> ZN L1 fractal output.
    // Input: DNExt (k, n) packed FP4 whose physical n-rows are padded to
    // UB_ALIGN_SIZE_FOR_4_BITS. Scale: DNExt (kGroup, n) whose n-rows are padded
    // to 32B. Output: Weight8BitDnToZnUbLayoutPtn FP8.
    template <class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void RunNdTrans(const InputTensor& input, const ScaleTensor& scale,
                                             const OutTensor& output, const A8W4TcgDequantParams& params)
    {
        using ScalePattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
        using OutPattern = asc::te::get_layout_pattern<typename OutTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<ScalePattern, asc::te::dn_ext_layout_ptn>,
                      "The ND-trans dequant requires a DNExt scale tensor.");
        static_assert(AscendC::Std::is_same_v<OutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>,
                      "The ND-trans dequant requires the DnToZn converted-weight UB layout.");
        (void)params;

        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;

        const int64_t kLen = AscendC::Std::get<1>(AscendC::Std::get<0>(input.layout().shape()));
        const int64_t nLen = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().shape()));
        // DNExt (k, n): physical n-row pitch in packed-FP4 elements.
        const int64_t weightRowPitch = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().stride()));
        // DNExt (kGroup, n): physical n-row pitch in scale elements.
        const int64_t scaleRowPitch = AscendC::Std::get<1>(AscendC::Std::get<1>(scale.layout().stride()));
        // DnToZn UB layout: k1-slab stride == dataBlockStride * C0.
        const int64_t dataBlockStride = AscendC::Std::get<1>(AscendC::Std::get<0>(output.layout().stride())) /
                                        C0_SIZE_B8;

        DequantW4Pergroup32NKParams<XType, WType, SType> vfParams;
        vfParams.outerExtend = static_cast<uint16_t>(nLen);
        vfParams.innerExtend = static_cast<uint16_t>(Blaze::Gemm::CeilDiv<int64_t>(
            Blaze::Gemm::CeilAlign<int64_t>(kLen, UB_ALIGN_SIZE_FOR_4_BITS), static_cast<int64_t>(asc_get_vf_len())));
        vfParams.dataBlockStride = static_cast<uint32_t>(dataBlockStride);
        vfParams.repeatStride = vfParams.dataBlockStride * static_cast<uint32_t>(OFFSET_8);
        vfParams.outDimOffset = static_cast<int32_t>(static_cast<int64_t>(AscendC::ONE_BLK_SIZE) -
                                                     static_cast<int64_t>(vfParams.innerExtend) *
                                                         static_cast<int64_t>(vfParams.repeatStride) *
                                                         static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
        vfParams.outerStrideScale = static_cast<uint32_t>(scaleRowPitch);
        vfParams.outerStrideWeight = static_cast<uint32_t>(weightRowPitch >> INT4_PACK_SHIFT);
        vfParams.maskWeight = static_cast<uint32_t>(kLen);
        vfParams.scaleBaseAddr = (__ubuf__ SType*)scale.data().get();
        vfParams.weightInBaseAddr0 = (__ubuf__ int8_t*)input.data().get();
        vfParams.weightInBaseAddr1 = vfParams.weightInBaseAddr0 + OFFSET_64;
        vfParams.weightOutBaseAddr = (__ubuf__ XType*)output.data().get();
        asc_vf_call<DequantW4Pergroup32NK<XType, WType, SType>>(vfParams);
    }

    // NZ (bTrans == false) path: NZ fp4 input -> NZ L1 fractal output. Input: NZ-pattern
    // (k, n) packed FP4 with a tight n1-fractal pitch; Scale: NDExt (kGroup, n) padded to
    // the 32-aligned N size; Output: NzRowPaddingLayoutPtn FP8 with vector chunks
    // interleaved across BL1 ping-pong buffers.
    template <uint32_t GroupSize, class InputTensor, class ScaleTensor, class OutTensor>
    __aicore__ inline static void RunNz(const InputTensor& input, const ScaleTensor& scale, const OutTensor& output,
                                        const A8W4TcgDequantParams& params)
    {
        using ScalePattern = asc::te::get_layout_pattern<typename ScaleTensor::layout_type>;
        using OutPattern = asc::te::get_layout_pattern<typename OutTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<ScalePattern, asc::te::nd_ext_layout_ptn>,
                      "The NZ dequant requires an NDExt scale tensor.");
        static_assert(AscendC::Std::is_same_v<OutPattern, Blaze::Gemm::NzRowPaddingLayoutPtn>,
                      "The NZ dequant requires the NzToNz converted-weight UB layout.");

        using XType = asc::te::get_attribute_element_type<typename OutTensor::element_type*>;
        using WType = asc::te::get_attribute_element_type<typename InputTensor::element_type*>;
        using SType = asc::te::get_attribute_element_type<typename ScaleTensor::element_type*>;

        const int64_t n1LoopNum = AscendC::Std::get<1>(AscendC::Std::get<1>(input.layout().shape()));
        // NDExt (kGroup, n): kGroup-row pitch in scale elements (32-aligned N size).
        const int64_t scaleRowPitch = AscendC::Std::get<1>(AscendC::Std::get<0>(scale.layout().stride()));
        // NzToNz UB layout strides: (1, groupIdStride), (vlStride, n1Stride).
        const int64_t weightOutVlStride = AscendC::Std::get<0>(AscendC::Std::get<1>(output.layout().stride()));
        const int64_t weightOutGroupIdStride = AscendC::Std::get<1>(AscendC::Std::get<0>(output.layout().stride()));
        const int64_t weightOutN1Stride = AscendC::Std::get<1>(AscendC::Std::get<1>(output.layout().stride()));

        // groupNumUb intentionally floors validK / GROUP_SIZE, matching the VF group
        // loop: a trailing partial K group is not converted.
        const uint32_t groupNumUb = params.validK / GroupSize;
        DequantW4PergroupKNParams<XType, WType, SType> vfParams;
        vfParams.groupNumUb = groupNumUb;
        vfParams.vLLoopNumInGroup = static_cast<uint32_t>(static_cast<int64_t>(GroupSize) * C0_SIZE_B8 /
                                                          static_cast<int64_t>(asc_get_vf_len()));
        vfParams.n1LoopNum = static_cast<uint32_t>(n1LoopNum);
        vfParams.bubNLen = static_cast<uint32_t>(n1LoopNum * C0_SIZE_B8);
        vfParams.scaleN1Stride = static_cast<uint32_t>(scaleRowPitch / n1LoopNum);
        vfParams.weightInGroupIdStride = static_cast<uint32_t>(static_cast<int64_t>(GroupSize) * C0_SIZE_B8 /
                                                               INT4_DTYPE_FACTOR);
        // Matches the VF's floor-based group addressing: groupIdStride * groupNumUb.
        vfParams.weightInN1Stride = vfParams.weightInGroupIdStride * groupNumUb;
        vfParams.weightOutN1Stride = static_cast<uint32_t>(weightOutN1Stride);
        vfParams.weightOutGroupIdStride = static_cast<uint32_t>(weightOutGroupIdStride);
        vfParams.weightOutVlStride = static_cast<uint32_t>(weightOutVlStride);
        vfParams.scaleBaseAddr0 = (__ubuf__ SType*)scale.data().get();
        vfParams.scaleBaseAddr1 = vfParams.scaleBaseAddr0 + Blaze::Gemm::BLOCK_CUBE;
        vfParams.scaleMaskBaseAddr = params.scaleMaskAddr;
        vfParams.weightInBaseAddr0 = (__ubuf__ int8_t*)input.data().get();
        vfParams.weightInBaseAddr1 = vfParams.weightInBaseAddr0 + OFFSET_64;
        vfParams.weightOutBaseAddr = (__ubuf__ XType*)output.data().get();
        asc_vf_call<DequantW4PergroupKN<XType, WType, SType>>(vfParams);
    }
};

// 32-bit-source dequant (epilogue; the int32/fp32 input may be any source, not only MMAD
// accumulators). Pipeline (one Vf launch per Run call, register-resident
// end to end): acc (AccType, UB) -> load + int32->fp32 cast -> *x2Scale -> *x1Scale -> +bias ->
// fp32->OutType cast -> store. Row pitches (nSrcAligned / nDstAligned) stay explicit
// parameters: they are Block-side buffer-layout concerns (L0C 32B alignment vs workspace
// GetVecLen alignment both exist). Dtypes are derived from the tensor arguments; disabled mode
// slots receive placeholder tensors constructed with the real dtype (never dereferenced).
// Semantics notes:
// - 16F output narrowing is NO_SAT (dequant lineage). This is deliberately different from
//   Gelu's half-SAT policy (clamp at 65504); each tile family keeps its own precision
//   baseline. Do NOT unify across families.
// - Per-token x1Scale dtype is fixed fp32 (the DIST_BRC_B32 broadcast load multiplies fp32
//   registers).
// - The streamk no-bias "x2*x1 pre-merge + 0xFFFFE000 mantissa mask" is a Block-side Init
//   concern: the Block passes the (possibly merged) scalar via x2ScaleScalar and calls
//   Run<PER_TENSOR, NONE>; the tile stays unaware.
// - AIV-only (AIC guard): the W4 prologue family is issued from the AIV side and carries no
//   guard; do NOT unify the guards across families.
template <>
class Dequant<B32ToFpTag> {
public:
    /*!
     * Full-chain entry (compile-time dispatch of the granularity matrix; only the subset
     * each Block uses gets instantiated):
     *   accTensor    : AccType (int32_t/fp32), UB, row pitch = nSrcAligned (Block-side layout)
     *   dstTensor    : DataTypeOut (fp32/bf16/fp16), UB ping-pong, row pitch = nDstAligned
     *   x2ScaleTensor: x2Gran==PER_CHANNEL -> single-row vector (reused across rows, indexed
     *                  by vfBlock offset); PER_TENSOR -> placeholder tensor (never
     *                  dereferenced), the value travels via x2ScaleScalar
     *   x1ScaleTensor: x1Gran==PER_TOKEN -> fp32 linear mSize entries (the Block has already
     *                  applied the row offset); otherwise a placeholder tensor
     *   biasTensor   : hasBias -> BiasDtype single-row vector (reused across rows);
     *                  otherwise a placeholder tensor
     *   Unused scalar slots pass 1.0f (never read by the disabled stages).
     */
    template <DequantMode x2Gran, DequantMode x1Gran, bool hasBias, class BiasDtype, class AccTensor, class DstTensor,
              class X2ScaleTensor, class X1ScaleTensor, class BiasTensor>
    __aicore__ inline static void Run(const AccTensor& accTensor, const DstTensor& dstTensor,
                                      const X2ScaleTensor& x2ScaleTensor, const X1ScaleTensor& x1ScaleTensor,
                                      const BiasTensor& biasTensor, float x2ScaleScalar, float x1ScaleScalar,
                                      uint16_t mSize, uint16_t nSize, uint32_t nSrcAligned, uint32_t nDstAligned)
    {
        using AccType = asc::te::get_attribute_element_type<typename AccTensor::element_type*>;
        using DataTypeOut = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
        using X2ScaleType = asc::te::get_attribute_element_type<typename X2ScaleTensor::element_type*>;
        using X1ScaleElementType = asc::te::get_attribute_element_type<typename X1ScaleTensor::element_type*>;
        using BiasElementType = asc::te::get_attribute_element_type<typename BiasTensor::element_type*>;
        using AccLayoutPattern = asc::te::get_layout_pattern<typename AccTensor::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
        using X2ScaleLayoutPattern = asc::te::get_layout_pattern<typename X2ScaleTensor::layout_type>;
        using X1ScaleLayoutPattern = asc::te::get_layout_pattern<typename X1ScaleTensor::layout_type>;
        using BiasLayoutPattern = asc::te::get_layout_pattern<typename BiasTensor::layout_type>;
        static_assert(x2Gran == DequantMode::PER_TENSOR || x2Gran == DequantMode::PER_CHANNEL,
                      "B32 dequant x2 granularity must be PER_TENSOR or PER_CHANNEL.");
        static_assert(
            x1Gran == DequantMode::NONE || x1Gran == DequantMode::PER_TENSOR || x1Gran == DequantMode::PER_TOKEN,
            "B32 dequant x1 granularity must be NONE, PER_TENSOR or PER_TOKEN.");
        static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                          AscendC::Std::is_same_v<DataTypeOut, half>,
                      "B32 dequant only supports float / bfloat16_t / half output.");
        static_assert(AscendC::Std::is_same_v<AccType, int32_t> || AscendC::Std::is_same_v<AccType, float>,
                      "B32 dequant input must be int32_t or float.");
        static_assert(AscendC::Std::is_same_v<X2ScaleType, float> || AscendC::Std::is_same_v<X2ScaleType, bfloat16_t> ||
                          AscendC::Std::is_same_v<X2ScaleType, half>,
                      "B32 dequant x2Scale dtype must be float / bfloat16_t / half.");
        static_assert(AscendC::Std::is_same_v<BiasDtype, float> || AscendC::Std::is_same_v<BiasDtype, bfloat16_t> ||
                          AscendC::Std::is_same_v<BiasDtype, half>,
                      "B32 dequant bias dtype must be float / bfloat16_t / half.");
        static_assert(AscendC::Std::is_same_v<X1ScaleElementType, float>,
                      "B32 dequant x1Scale tensor must be fp32 (per-token dtype is fixed).");
        static_assert(AscendC::Std::is_same_v<BiasElementType, BiasDtype>,
                      "B32 dequant bias tensor element type must match BiasDtype.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<AccTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<X2ScaleTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<X1ScaleTensor>, asc::te::location::ub> &&
                          AscendC::Std::is_same_v<asc::te::get_mem_location<BiasTensor>, asc::te::location::ub>,
                      "B32 dequant only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<AccLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<X2ScaleLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<X1ScaleLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<BiasLayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "B32 dequant only supports NDExt tensor layouts.");
        if ASCEND_IS_AIC {
            return;
        }
        DqVfParams<DataTypeOut, AccType, X2ScaleType, BiasDtype> params{
            reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get()),
            reinterpret_cast<__ubuf__ AccType*>(accTensor.data().get()),
            reinterpret_cast<__ubuf__ X2ScaleType*>(x2ScaleTensor.data().get()),
            reinterpret_cast<__ubuf__ float*>(x1ScaleTensor.data().get()),
            reinterpret_cast<__ubuf__ BiasDtype*>(biasTensor.data().get()),
            x2ScaleScalar,
            x1ScaleScalar,
            mSize,
            nSize,
            nSrcAligned,
            nDstAligned,
            NLoopCnt<AccType>(nSize)};
        asc_vf_call<DequantVf<x2Gran, x1Gran, hasBias, DataTypeOut, AccType, X2ScaleType, BiasDtype>>(params);
    }

private:
#ifdef __NPU_ARCH__
    // ---- Cast traits (converged from DQ_CT_* / STREAMK_DQ_CT_*; named by bit-width like
    //      gelu.h / mx_quant.h; RoundMode lives in AscendC) ----
    // int32 -> fp32, 1:1 width (RegLayout/Sat irrelevant).
    static constexpr AscendC::Reg::CastTrait CT_S32_TO_32F = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    // fp32 -> 16F narrowing, NO_SAT (half and bf16 share it; deliberately NOT the Gelu
    // half-SAT policy — the dequant lineage keeps NO_SAT).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    // 16F -> fp32 widening, ZERO / ONE register halves (same name and value as mx_quant.h).
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
#endif // __NPU_ARCH__

    // ---- Vf parameter struct: raw __ubuf__ pointers + shape metadata ----
    template <typename DataTypeOut, typename AccType, typename X2ScaleType, typename BiasDtype>
    struct DqVfParams {
        __ubuf__ DataTypeOut* dstAddr;
        __ubuf__ AccType* accAddr;
        __ubuf__ X2ScaleType* x2ScaleAddr; // dereferenced only when PER_CHANNEL
        __ubuf__ float* x1ScaleAddr;       // dereferenced only when PER_TOKEN (fp32 fixed)
        __ubuf__ BiasDtype* biasAddr;      // dereferenced only when hasBias
        float x2ScaleScalar;               // used only when x2Gran==PER_TENSOR
        float x1ScaleScalar;               // used only when x1Gran==PER_TENSOR
        uint16_t mSize;
        uint16_t nSize;
        uint32_t nSrcAligned;
        uint32_t nDstAligned;
        uint16_t nLoopCnt; // derived in Run: CeilDiv(nSize, EleNumPerVf())
    };

    // ---- Stage callees (one-to-one with the source block functions) ----
    // Widen a bf16/fp16 scale/bias register to fp32 (ZERO/ONE halves + Interleave, 1:2 width
    // expansion); plain register copy when SrcType is already float. Shared by the x2Scale
    // and bias stages.
    template <typename SrcType>
    static __simd_callee__ inline void WidenOrCopyToF32Callee(AscendC::Reg::RegTensor<float>& dst,
                                                              AscendC::Reg::RegTensor<SrcType>& src,
                                                              AscendC::Reg::MaskReg& maskN,
                                                              AscendC::Reg::MaskReg& maskB16)
    {
        if constexpr (!AscendC::IsSameType<SrcType, float>::value) {
            AscendC::Reg::RegTensor<float> oneReg;
            AscendC::Reg::Cast<float, SrcType, CT_16F_TO_32F_ZERO>(dst, src, maskN);
            AscendC::Reg::Cast<float, SrcType, CT_16F_TO_32F_ONE>(oneReg, src, maskB16);
            AscendC::Reg::Interleave(dst, oneReg, dst, oneReg);
        } else {
            dst = src;
        }
    }

    // Stage 1: load one AccType block from UB + int32->fp32 cast (fp32 acc: copy).
    template <typename AccType>
    static __simd_callee__ inline void LoadAndCastAccCallee(AscendC::Reg::RegTensor<float>& dst, __ubuf__ AccType* acc,
                                                            uint32_t offset, AscendC::Reg::MaskReg& maskN)
    {
        AscendC::Reg::RegTensor<AccType> accReg;
        AscendC::Reg::DataCopy(accReg, acc + offset);
        if constexpr (AscendC::IsSameType<AccType, int32_t>::value) {
            AscendC::Reg::Cast<float, AccType, CT_S32_TO_32F>(dst, accReg, maskN);
        } else {
            dst = accReg;
        }
    }

    // Stage 2: *x2Scale (PER_TENSOR: single Muls / PER_CHANNEL: load + widen + Mul).
    template <DequantMode x2Gran, typename X2ScaleType>
    static __simd_callee__ inline void ApplyX2ScaleCallee(AscendC::Reg::RegTensor<float>& out,
                                                          AscendC::Reg::RegTensor<float>& src,
                                                          __ubuf__ X2ScaleType* scale, uint32_t offset,
                                                          AscendC::Reg::MaskReg& maskN, AscendC::Reg::MaskReg& maskB16,
                                                          float x2ScaleScalar)
    {
        if constexpr (x2Gran == DequantMode::PER_TENSOR) {
            AscendC::Reg::Muls(out, src, x2ScaleScalar, maskN);
        } else {
            AscendC::Reg::RegTensor<X2ScaleType> scaleReg;
            AscendC::Reg::DataCopy(scaleReg, scale + offset);
            AscendC::Reg::RegTensor<float> castScaleReg;
            WidenOrCopyToF32Callee<X2ScaleType>(castScaleReg, scaleReg, maskN, maskB16);
            AscendC::Reg::Mul(out, src, castScaleReg, maskN);
        }
    }

    // Stage 3: *x1Scale (PER_TENSOR: single Muls / PER_TOKEN: DIST_BRC_B32 broadcast + Mul /
    // NONE: register copy).
    template <DequantMode x1Gran>
    static __simd_callee__ inline void ApplyX1ScaleCallee(AscendC::Reg::RegTensor<float>& out,
                                                          AscendC::Reg::RegTensor<float>& src,
                                                          __ubuf__ float* perTokenScale, uint16_t mIdx,
                                                          AscendC::Reg::MaskReg& maskN, float x1ScaleScalar)
    {
        if constexpr (x1Gran == DequantMode::PER_TENSOR) {
            AscendC::Reg::Muls(out, src, x1ScaleScalar, maskN);
        } else if constexpr (x1Gran == DequantMode::PER_TOKEN) {
            AscendC::Reg::RegTensor<float> perTokenScaleReg;
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(perTokenScaleReg, perTokenScale + mIdx);
            AscendC::Reg::Mul(out, src, perTokenScaleReg, maskN);
        } else {
            out = src;
        }
    }

    // Stage 4: +bias (float direct / 16F widened to fp32 / no-bias copy).
    template <bool hasBias, typename BiasDtype>
    static __simd_callee__ inline void ApplyBiasCallee(AscendC::Reg::RegTensor<float>& out,
                                                       AscendC::Reg::RegTensor<float>& src, __ubuf__ BiasDtype* bias,
                                                       uint32_t offset, AscendC::Reg::MaskReg& maskN,
                                                       AscendC::Reg::MaskReg& maskB16)
    {
        if constexpr (hasBias) {
            AscendC::Reg::RegTensor<BiasDtype> biasReg;
            AscendC::Reg::DataCopy(biasReg, bias + offset);
            AscendC::Reg::RegTensor<float> castBiasReg;
            WidenOrCopyToF32Callee<BiasDtype>(castBiasReg, biasReg, maskN, maskB16);
            AscendC::Reg::Add(out, src, castBiasReg, maskN);
        } else {
            out = src;
        }
    }

    // Stage 5: fp32 -> DataTypeOut cast + store (float: DIST_NORM_B32 / 16F: DIST_PACK_B32).
    template <typename DataTypeOut>
    static __simd_callee__ inline void CastAndStoreCallee(__ubuf__ DataTypeOut* dst, uint32_t dstUbOffset,
                                                          AscendC::Reg::RegTensor<float>& src,
                                                          AscendC::Reg::MaskReg& maskN)
    {
        AscendC::Reg::RegTensor<DataTypeOut> castResultOutReg;
        if constexpr (!AscendC::IsSameType<DataTypeOut, float>::value) {
            AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_16F>(castResultOutReg, src, maskN);
        } else {
            castResultOutReg = src;
        }
        if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
            AscendC::Reg::DataCopy<DataTypeOut, AscendC::Reg::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                        castResultOutReg, maskN);
        } else {
            AscendC::Reg::DataCopy<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstUbOffset,
                                                                                        castResultOutReg, maskN);
        }
    }

    // ---- Vf body: verbatim instruction sequence of the source __VEC_SCOPE__ bodies
    //      (VFDoDequant / VfDoDequant), stage calls one-to-one ----
    template <DequantMode x2Gran, DequantMode x1Gran, bool hasBias, typename DataTypeOut, typename AccType,
              typename X2ScaleType, typename BiasDtype>
    static __simd_vf__ inline void DequantVf(DqVfParams<DataTypeOut, AccType, X2ScaleType, BiasDtype> params)
    {
        uint32_t eleNumPerVf = asc_get_vf_len() / sizeof(AccType);
        AscendC::Reg::MaskReg maskB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t elementNum = static_cast<uint32_t>(params.nSize);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < params.nLoopCnt; vfBlockIdx++) {
                AscendC::Reg::RegTensor<float> castSrcOutReg, mulScaleOutReg, mulPtScaleOutReg, addBiasOutReg;
                AscendC::Reg::MaskReg maskN = AscendC::Reg::UpdateMask<AccType>(elementNum);
                uint32_t blockOffset = vfBlockIdx * eleNumPerVf;

                LoadAndCastAccCallee<AccType>(castSrcOutReg, params.accAddr, mIdx * params.nSrcAligned + blockOffset,
                                              maskN);
                ApplyX2ScaleCallee<x2Gran, X2ScaleType>(mulScaleOutReg, castSrcOutReg, params.x2ScaleAddr, blockOffset,
                                                        maskN, maskB16, params.x2ScaleScalar);
                ApplyX1ScaleCallee<x1Gran>(mulPtScaleOutReg, mulScaleOutReg, params.x1ScaleAddr, mIdx, maskN,
                                           params.x1ScaleScalar);
                ApplyBiasCallee<hasBias, BiasDtype>(addBiasOutReg, mulPtScaleOutReg, params.biasAddr, blockOffset,
                                                    maskN, maskB16);
                CastAndStoreCallee<DataTypeOut>(params.dstAddr, mIdx * params.nDstAligned + blockOffset, addBiasOutReg,
                                                maskN);
            }
        }
    }

    // ---- Internal loop-count derivation ----
    template <typename AccType>
    __aicore__ inline static uint32_t EleNumPerVf()
    {
        return static_cast<uint32_t>(asc_get_vf_len() / sizeof(AccType));
    }

    template <typename AccType>
    __aicore__ inline static uint16_t NLoopCnt(uint16_t nSize)
    {
        return static_cast<uint16_t>((nSize + EleNumPerVf<AccType>() - 1) / EleNumPerVf<AccType>());
    }
};

} // namespace Blaze::Epilogue::Tile
