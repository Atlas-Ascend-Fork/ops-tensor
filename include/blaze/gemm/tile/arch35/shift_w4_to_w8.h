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
 * \file shift_w4_to_w8.h
 * \brief Convert packed W4 weights to W8 layout for vector-to-cube consumption.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/tile/arch35/scale_mx_bias.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Gemm::Tile {

template <typename OutType_, typename InType_>
class ShiftW4ToW8 {
public:
    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline ShiftW4ToW8(const Weight4BitTensorType& weight4BitTensor,
                                  const Weight8BitTensorType& weight8BitTensor)
    {
        using SrcLayoutPattern = asc::te::get_layout_pattern<typename Weight4BitTensorType::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename Weight8BitTensorType::layout_type>;
        constexpr bool IS_ZN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::zn_layout_ptn>;
        constexpr bool IS_DN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::dn_ext_layout_ptn>;
        constexpr bool IS_ZN_OUTPUT = AscendC::Std::is_same_v<DstLayoutPattern, Weight8BitZnToZnUbLayoutPtn>;
        constexpr bool
            IS_DN_OUTPUT = AscendC::Std::is_same_v<DstLayoutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>;
        static_assert(IS_ZN_WEIGHT || IS_DN_WEIGHT, "W4-to-W8 conversion only supports ZN and DNExt source layouts");
        static_assert((IS_ZN_WEIGHT && IS_ZN_OUTPUT) || (IS_DN_WEIGHT && IS_DN_OUTPUT),
                      "W4-to-W8 source and destination layouts must use the matching conversion path");

        if constexpr (IS_DN_WEIGHT) {
            ShiftW4ToW8DnImpl(weight4BitTensor, weight8BitTensor);
        } else {
            ShiftW4ToW8ZnImpl(weight4BitTensor, weight8BitTensor);
        }
    }

    // MX GMM fused path: composes bias scaling into the same VF command as W4
    // decoding.  processBias dispatches to compile-time VF instantiations so the
    // steady-state K iterations contain no bias branch or extra VF entry.
    template <typename Weight4BitTensorType, typename Weight8BitTensorType, typename BiasInTensor,
              typename BiasOutTensor>
    __aicore__ inline ShiftW4ToW8(const Weight4BitTensorType& weight4BitTensor,
                                  const Weight8BitTensorType& weight8BitTensor, const BiasInTensor& biasInTensor,
                                  const BiasOutTensor& biasOutTensor, bool processBias)
    {
        using SrcLayoutPattern = asc::te::get_layout_pattern<typename Weight4BitTensorType::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename Weight8BitTensorType::layout_type>;
        static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::zn_layout_ptn>,
                      "Fused MX bias conversion requires ZN FP4 weight input");
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, Weight8BitZnToZnUbLayoutPtn>,
                      "Fused MX bias conversion requires the ZN W8 UB layout");
        // fp8_e4m3fn_t is an alias of the __fp8e4m3 builtin on NPU; only the portable alias
        // is used here so CPU-debug host compilers without NPU builtin types can parse this.
        static_assert(AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value, "OutType must be fp8_e4m3fn_t");
        static_assert(
            AscendC::IsSameType<InType, fp4x2_e2m1_t>::value || AscendC::IsSameType<InType, fp4x2_e1m2_t>::value,
            "InType must be fp4x2_e2m1_t or fp4x2_e1m2_t");

        ShiftW4ToW8Params weightParams = MakeWeightParams(weight4BitTensor, weight8BitTensor);

        using BiasType = typename asc::te::get_attribute_element_type<typename BiasInTensor::element_type*>;
        typename ScaleMxBias<BiasType>::ScaleMxBiasParams biasParams{};
        if (processBias) {
            constexpr uint64_t VECTOR_ELEMENTS = static_cast<uint64_t>(asc_get_vf_len()) / sizeof(BiasType);
            uint64_t biasElementCount = static_cast<uint64_t>(asc::te::get_total_column_shape(biasInTensor.layout()));
            biasParams.loopNum = static_cast<uint16_t>(CeilDiv(biasElementCount, VECTOR_ELEMENTS));
            biasParams.biasInAddr = (__ubuf__ BiasType*)biasInTensor.data().get();
            biasParams.biasOutAddr = (__ubuf__ BiasType*)biasOutTensor.data().get();
            ShiftW4ToW8AndScaleBiasVf<BiasType, true>(weightParams, biasParams);
        } else {
            ShiftW4ToW8AndScaleBiasVf<BiasType, false>(weightParams, biasParams);
        }
    }

    template <typename W, typename O>
    __aicore__ inline static void ConvertContinuousPair(const W& left, const O& leftOutput, const W& right,
                                                        const O& rightOutput)
    {
        ShiftW4ToW8 converter;
        asc_vf_call<ShiftW4ToW8PairVf>(converter.MakeContinuousWeightParams(left, leftOutput),
                                       converter.MakeContinuousWeightParams(right, rightOutput));
    }

private:
    __aicore__ inline ShiftW4ToW8() = default;
    using OutType = OutType_;
    using InType = InType_;

    struct ShiftW4ToW8Params {
        uint64_t loopKNum;
        uint64_t innerLoopNum;
        uint64_t loopKDstStride;
        uint64_t innerDstStride;
        uint64_t nRealSizeAlign;
        __ubuf__ InType_* weight4BitPhyAddr;
        __ubuf__ OutType_* weight8BitPhyAddr;
    };

    struct ShiftW4DnToW8Params {
        uint16_t outExtend;
        uint16_t innerExtend;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        uint32_t maskB8Tail0;
        uint32_t maskB8Tail1;
        uint32_t inputRowStrideBytes;
        __ubuf__ int8_t* weight4BitPhyAddr;
        __ubuf__ OutType_* weight8BitPhyAddr;
        __ubuf__ OutType_* weight8BitPhyAddr1;
    };

    template <typename W, typename O>
    __aicore__ inline ShiftW4ToW8Params MakeContinuousWeightParams(const W& input, const O& output) const
    {
        auto params = MakeWeightParams(input, output);
        params.loopKNum *= K32_SLABS_PER_K64;
        return params;
    }

    // Derive kernel-loop and stride parameters directly from tensor shape/stride metadata.
    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline ShiftW4ToW8Params MakeWeightParams(const Weight4BitTensorType& weight4BitTensor,
                                                         const Weight8BitTensorType& weight8BitTensor) const
    {
        ShiftW4ToW8Params params;
        params.weight4BitPhyAddr = (__ubuf__ InType*)weight4BitTensor.data().get();
        params.weight8BitPhyAddr = (__ubuf__ OutType*)weight8BitTensor.data().get();
        params.loopKNum = AscendC::Std::get<1>(AscendC::Std::get<0>(weight4BitTensor.layout().shape()));
        params.nRealSizeAlign = AscendC::Std::get<1>(AscendC::Std::get<1>(weight4BitTensor.layout().shape())) *
                                BLOCK_CUBE;
        params.innerDstStride = AscendC::Std::get<1>(AscendC::Std::get<1>(weight8BitTensor.layout().stride()));
        params.innerLoopNum = (params.nRealSizeAlign * C0_SIZE_B8) / static_cast<uint64_t>(asc_get_vf_len());
        params.loopKDstStride = params.innerLoopNum * params.innerDstStride;
        return params;
    }

    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline void ShiftW4ToW8ZnImpl(const Weight4BitTensorType& weight4BitTensor,
                                             const Weight8BitTensorType& weight8BitTensor)
    {
        using DstLayoutPattern = asc::te::get_layout_pattern<typename Weight8BitTensorType::layout_type>;
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, Weight8BitZnToZnUbLayoutPtn>,
                      "ZN FP4 conversion requires the MX FP8 ZN UB layout");
        static_assert(AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value, "OutType must be fp8_e4m3fn_t");
        static_assert(
            AscendC::IsSameType<InType, fp4x2_e2m1_t>::value || AscendC::IsSameType<InType, fp4x2_e1m2_t>::value,
            "InType must be fp4x2_e2m1_t or fp4x2_e1m2_t");

        asc_vf_call<ShiftW4ToW8Vf>(MakeWeightParams(weight4BitTensor, weight8BitTensor));
    }

    template <typename Weight4BitTensorType, typename Weight8BitTensorType>
    __aicore__ inline void ShiftW4ToW8DnImpl(const Weight4BitTensorType& weight4BitTensor,
                                             const Weight8BitTensorType& weight8BitTensor)
    {
        using DstLayoutPattern = asc::te::get_layout_pattern<typename Weight8BitTensorType::layout_type>;
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, Blaze::Gemm::Weight8BitDnToZnUbLayoutPtn>,
                      "DN FP4 conversion requires the MX DN-to-ZN destination layout");
        static_assert(AscendC::IsSameType<OutType, fp8_e4m3fn_t>::value, "OutType must be fp8_e4m3fn_t");
        static_assert(AscendC::IsSameType<InType, fp4x2_e2m1_t>::value, "InType must be fp4x2_e2m1_t");

        const auto& inputLayout = weight4BitTensor.layout();
        auto inputShape = inputLayout.shape();
        auto inputStride = inputLayout.stride();
        uint64_t kSize = static_cast<uint64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(inputShape)));
        uint64_t nSize = static_cast<uint64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(inputShape)));
        uint64_t nAlign = Align16(nSize);
        uint64_t kAlign32 = Align32(kSize);
        uint64_t kAlign64 = Align64(kSize);
        constexpr uint64_t VECTOR_REG_WIDTH_BYTES = static_cast<uint64_t>(asc_get_vf_len());
        constexpr uint16_t FP4_PACK_SHIFT = 1U;
        constexpr uint64_t VECTOR_REG_WIDTH_FOR_4BITS = VECTOR_REG_WIDTH_BYTES << FP4_PACK_SHIFT;

        ShiftW4DnToW8Params params;
        params.outExtend = static_cast<uint16_t>(nSize);
        params.innerExtend = static_cast<uint16_t>(CeilDiv(kAlign64, VECTOR_REG_WIDTH_FOR_4BITS));
        params.dataBlockStride = static_cast<uint32_t>(nAlign + 1U);
        params.repeatStride = params.dataBlockStride * static_cast<uint32_t>(BLOCK_CUBE);
        uint64_t outputRowSpan = static_cast<uint64_t>(params.innerExtend) * params.repeatStride * BLOCK_BYTE_SIZE;
        params.outDimOffset = static_cast<int32_t>(static_cast<int64_t>(BLOCK_BYTE_SIZE) -
                                                   static_cast<int64_t>(outputRowSpan));
        params.maskB8Tail0 = static_cast<uint32_t>(Min(kAlign32 % VECTOR_REG_WIDTH_FOR_4BITS, VECTOR_REG_WIDTH_BYTES) +
                                                   kAlign32 / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH_BYTES);
        params.maskB8Tail1 = static_cast<uint32_t>(Max(static_cast<int64_t>(kAlign32 % VECTOR_REG_WIDTH_FOR_4BITS) -
                                                           static_cast<int64_t>(VECTOR_REG_WIDTH_BYTES),
                                                       static_cast<int64_t>(0)) +
                                                   kAlign32 / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH_BYTES);
        params.inputRowStrideBytes = static_cast<uint32_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(inputStride)) >>
                                                           1U);
        params.weight4BitPhyAddr = (__ubuf__ int8_t*)weight4BitTensor.data().get();
        params.weight8BitPhyAddr = (__ubuf__ OutType*)weight8BitTensor.data().get();
        params.weight8BitPhyAddr1 = params.weight8BitPhyAddr + VECTOR_REG_WIDTH_BYTES * params.dataBlockStride;

        asc_vf_call<ShiftW4DnToW8Vf>(params);
    }

    static __simd_vf__ inline void ShiftW4ToW8Vf(ShiftW4ToW8Params params) { ShiftW4ToW8Callee(params); }

    static __simd_vf__ inline void ShiftW4ToW8PairVf(ShiftW4ToW8Params left, ShiftW4ToW8Params right)
    {
        ShiftW4ToW8ExtendedCallee(left);
        ShiftW4ToW8ExtendedCallee(right);
    }

    template <typename BiasType, bool ProcessBias>
    static __simd_vf__ inline void ShiftW4ToW8AndScaleBiasVf(
        ShiftW4ToW8Params weightParams, typename ScaleMxBias<BiasType>::ScaleMxBiasParams biasParams)
    {
        ShiftW4ToW8Callee(weightParams);
        if constexpr (ProcessBias) {
            ScaleMxBias<BiasType>::ScaleMxBiasCallee(biasParams);
        }
    }

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

    static __simd_callee__ inline void LoadPackedWeight(ShiftW4ToW8Params& params, uint16_t loopKIdx,
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

    static __simd_callee__ inline void DecodePackedWeight(
        AscendC::Reg::RegTensor<int8_t>& wAnd, AscendC::Reg::RegTensor<int8_t>& wShl,
        AscendC::Reg::RegTensor<int8_t>& wShr1, AscendC::Reg::RegTensor<int8_t>& wLoad,
        AscendC::Reg::RegTensor<int8_t>& laneShift, AscendC::Reg::RegTensor<int8_t>& wShrReg,
        AscendC::Reg::RegTensor<int8_t>& wAndReg, AscendC::Reg::RegTensor<int8_t>& wShr0,
        AscendC::Reg::RegTensor<int8_t>& wShlReg, AscendC::Reg::RegTensor<int8_t>& wSel, AscendC::Reg::MaskReg& preg,
        AscendC::Reg::MaskReg& pregVsel)
    {
        DecodeFastWeight(wAnd, wShl, wShr1, wLoad, laneShift, wShrReg, wAndReg, preg);
    }

    static __simd_callee__ inline void ConvertWeightLoops(
        ShiftW4ToW8Params& params, AscendC::Reg::RegTensor<int8_t>& wShrReg, AscendC::Reg::RegTensor<int8_t>& wShlReg,
        AscendC::Reg::RegTensor<int8_t>& wAndReg, AscendC::Reg::RegTensor<int8_t>& wLoad,
        AscendC::Reg::RegTensor<int8_t>& wShl, AscendC::Reg::RegTensor<int8_t>& wShr0,
        AscendC::Reg::RegTensor<int8_t>& wShr1, AscendC::Reg::RegTensor<int8_t>& wSel,
        AscendC::Reg::RegTensor<int8_t>& wAnd, AscendC::Reg::RegTensor<int8_t>& laneShift,
        AscendC::Reg::RegTensor<uint16_t>& gatherIndex, AscendC::Reg::RegTensor<uint16_t>& activeGatherIndex,
        AscendC::Reg::RegTensor<uint16_t>& indexGroup, AscendC::Reg::RegTensor<uint16_t>& gathered,
        AscendC::Reg::RegTensor<uint8_t>& unused, AscendC::Reg::MaskReg& preg, AscendC::Reg::MaskReg& pregVsel,
        AscendC::Reg::MaskReg& gatherMask)
    {
        for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
            // Keep Gather's UB pointer 32B aligned; select the odd K32 half through its word indices.
            AscendC::Reg::Duplicate(
                indexGroup, static_cast<uint16_t>((loopKIdx % K32_SLABS_PER_K64) * PACKED_K32_ROW_WORDS), gatherMask);
            AscendC::Reg::Add(activeGatherIndex, gatherIndex, indexGroup, gatherMask);

            for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
                LoadPackedWeight(params, loopKIdx, innerLoopIdx, wLoad, gathered, activeGatherIndex, unused,
                                 gatherMask);
                DecodePackedWeight(wAnd, wShl, wShr1, wLoad, laneShift, wShrReg, wAndReg, wShr0, wShlReg, wSel, preg,
                                   pregVsel);
                AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
                AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    (__ubuf__ uint8_t*&)params.weight8BitPhyAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd,
                    aregWeightB8Out, preg);
            }
        }
    }

    static __simd_callee__ inline void ShiftW4ToW8Callee(ShiftW4ToW8Params params)
    {
        constexpr bool IS_E1M2 = AscendC::IsSameType<InType, fp4x2_e1m2_t>::value;
        constexpr uint32_t SHIFT_RIGHT_SIZE = IS_E1M2 ? 0x3U : 0x2U;
        constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
        constexpr uint32_t AND_MASK = IS_E1M2 ? 0x8EU : 0x9CU;
        AscendC::Reg::RegTensor<int8_t> wShrReg;
        AscendC::Reg::RegTensor<int8_t> wShlReg;
        AscendC::Reg::RegTensor<int8_t> wAndReg;
        AscendC::Reg::RegTensor<int8_t> wLoad;
        AscendC::Reg::RegTensor<int8_t> wShl;
        AscendC::Reg::RegTensor<int8_t> wShr0;
        AscendC::Reg::RegTensor<int8_t> wShr1;
        AscendC::Reg::RegTensor<int8_t> wSel;
        AscendC::Reg::RegTensor<int8_t> wAnd;

        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg pregVsel = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, SHIFT_RIGHT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
        AscendC::Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, AND_MASK, preg);

        for (uint16_t loopKIdx = 0; loopKIdx < params.loopKNum; ++loopKIdx) {
            for (uint16_t innerLoopIdx = 0; innerLoopIdx < params.innerLoopNum; ++innerLoopIdx) {
                // DIST_US_B8 load mode expands each packed B4 byte into lane-aligned B8 slots.
                // Packed B4 address offset (bytes) = logical element index >> 1.
                AscendC::Reg::AddrReg aregWeightB8In = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, (C0_SIZE_B8 * params.nRealSizeAlign) >> 1, innerLoopIdx, asc_get_vf_len() >> 1);
                AscendC::Reg::LoadAlign<uint8_t, AscendC::Reg::LoadDist::DIST_US_B8>(
                    (AscendC::Reg::RegTensor<uint8_t>&)wLoad, (__ubuf__ uint8_t*&)params.weight4BitPhyAddr,
                    aregWeightB8In);

                AscendC::Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
                AscendC::Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
                AscendC::Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
                AscendC::Reg::Select(wSel, wShr1, wShr0, pregVsel);
                AscendC::Reg::And(wAnd, wSel, wAndReg, preg);

                AscendC::Reg::AddrReg aregWeightB8Out = AscendC::Reg::CreateAddrReg<uint8_t>(
                    loopKIdx, params.loopKDstStride, innerLoopIdx, params.innerDstStride);
                AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    (__ubuf__ uint8_t*&)params.weight8BitPhyAddr, (AscendC::Reg::RegTensor<uint8_t>&)wAnd,
                    aregWeightB8Out, preg);
            }
        }
    }

    static __simd_callee__ inline void ShiftW4ToW8ExtendedCallee(ShiftW4ToW8Params params)
    {
        AscendC::Reg::RegTensor<int8_t> wShrReg;
        AscendC::Reg::RegTensor<int8_t> wShlReg;
        AscendC::Reg::RegTensor<int8_t> wAndReg;
        AscendC::Reg::RegTensor<int8_t> wLoad;
        AscendC::Reg::RegTensor<int8_t> wShl;
        AscendC::Reg::RegTensor<int8_t> wShr0;
        AscendC::Reg::RegTensor<int8_t> wShr1;
        AscendC::Reg::RegTensor<int8_t> wSel;
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

        ConvertWeightLoops(params, wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd, laneShift,
                           gatherIndex, activeGatherIndex, indexGroup, gathered, unused, preg, pregVsel, gatherMask);
    }

    static __simd_vf__ inline void ShiftW4DnToW8Vf(ShiftW4DnToW8Params params)
    {
        namespace MicroAPI = AscendC::MicroAPI;
        constexpr uint32_t E2M1_EM_SHIFT_RIGHT_SIZE = 0x2;
        constexpr uint32_t E2M1_SIGN_SHIFT_LEFT_SIZE = 0x4;
        constexpr uint32_t FP4_SIGN_MASK = 0x80;
        constexpr uint32_t E2M1_EM_MASK = 0x1C;
        constexpr uint64_t VECTOR_REG_WIDTH_BYTES = static_cast<uint64_t>(asc_get_vf_len());
        __ubuf__ OutType* weightOutUbAddr = params.weight8BitPhyAddr;
        __ubuf__ OutType* weightOutUbAddr1 = params.weight8BitPhyAddr1;
        MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sign0, sign1, emShr, emShl, signShl, out0, out1;
        MicroAPI::RegTensor<int8_t> emShrBits, emShlBits, signBits;
        MicroAPI::RegTensor<uint8_t> emMask, signMask;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(emShrBits, E2M1_EM_SHIFT_RIGHT_SIZE, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(emShlBits, E2M1_EM_SHIFT_RIGHT_SIZE, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(signBits, E2M1_SIGN_SHIFT_LEFT_SIZE, preg);
        MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(emMask, E2M1_EM_MASK, preg);
        MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(signMask, FP4_SIGN_MASK, preg);

        for (uint16_t outIdx = 0; outIdx < params.outExtend; ++outIdx) {
            uint32_t maskWeight0Tmp = params.maskB8Tail0;
            uint32_t maskWeight1Tmp = params.maskB8Tail1;
            for (uint16_t repeatIdx = 0; repeatIdx < params.innerExtend; ++repeatIdx) {
                MicroAPI::MaskReg maskB8Tail0 = MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                MicroAPI::MaskReg maskB8Tail1 = MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                MicroAPI::AddrReg aregWeightB8 = MicroAPI::CreateAddrReg<uint8_t>(outIdx, params.inputRowStrideBytes,
                                                                                  repeatIdx, VECTOR_REG_WIDTH_BYTES);
                MicroAPI::LoadAlign(wLoad0, (__ubuf__ uint8_t*&)params.weight4BitPhyAddr, aregWeightB8);

                MicroAPI::ShiftRight(emShr, wLoad0, emShrBits, preg);
                MicroAPI::And(emShr, emShr, emMask, preg);
                MicroAPI::ShiftLeft(emShl, wLoad0, emShlBits, preg);
                MicroAPI::And(emShl, emShl, emMask, preg);
                MicroAPI::ShiftLeft(signShl, wLoad0, signBits, preg);
                MicroAPI::And(sign0, signShl, signMask, preg);
                MicroAPI::And(sign1, wLoad0, signMask, preg);
                MicroAPI::Or(out0, emShr, sign1, preg);
                MicroAPI::Or(out1, emShl, sign0, preg);
                MicroAPI::Interleave(wDIntlv0, wDIntlv1, out1, out0);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                     MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint8_t*&)weightOutUbAddr,
                                                                              wDIntlv0, params.dataBlockStride,
                                                                              params.repeatStride, maskB8Tail0);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                     MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint8_t*&)weightOutUbAddr1,
                                                                              wDIntlv1, params.dataBlockStride,
                                                                              params.repeatStride, maskB8Tail1);
            }
            weightOutUbAddr += params.outDimOffset;
            weightOutUbAddr1 += params.outDimOffset;
        }
    }
};

} // namespace Blaze::Gemm::Tile

// MX weight-prologue helpers share this existing weight-conversion implementation header.
#if defined(__NPU_ARCH__) || defined(ASCENDC_CPU_DEBUG)
namespace Blaze::Gemm::Tile {

/*!
 * \brief Shared MX operand operations; buffer ownership and fences belong to the caller.
 */
template <typename T, bool OperandB>
__aicore__ inline auto MakeMxL0Operand(uint64_t offset, uint64_t rows, uint64_t columns)
{
    using Layout = AscendC::Std::conditional_t<OperandB, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0b, asc::te::location::l0a>;
    auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<asc::te::c0_element<T>>>(rows, columns);
    return asc::te::make_tensor(asc::te::make_mem_ptr<Location, T>(offset), layout);
}

template <typename Source, typename Destination, typename Copy, typename Row, typename Column, typename Rows,
          typename Columns>
__aicore__ inline void CopyMxOperandTile(const Copy& copy, const Destination& destination, const Source& source,
                                         Row row, Column column, Rows rows, Columns columns)
{
    auto tile = source.slice(asc::te::make_coord(row, column), asc::te::make_shape(rows, columns));
    asc::te::copy(copy, destination, tile);
}

__aicore__ inline auto MakeMxBiasTable(uint64_t offset, uint64_t columns)
{
    auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1L),
                                                                         static_cast<int64_t>(Align16(columns)));
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(offset), layout);
}

template <typename Index = int64_t, typename SourceA, typename SourceB, typename DestinationA, typename DestinationB>
__aicore__ inline void CopyMxScalesToL1(const SourceA& a, const SourceB& b, const DestinationA& outA,
                                        const DestinationB& outB, uint64_t rows, uint64_t columns, uint64_t kOffset,
                                        uint64_t kColumns, uint64_t nOffset = 0)
{
    constexpr uint64_t HALF_COUNT = 2UL;
    auto copy = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    CopyMxOperandTile(copy, outA, a, 0, static_cast<Index>(kOffset), static_cast<Index>(rows),
                      static_cast<Index>(kColumns));
    const uint64_t partColumns = columns / HALF_COUNT;
    const uint64_t sourceStride = asc::te::get_total_column_shape(b.layout()) / HALF_COUNT;
    for (uint64_t part = 0; part < HALF_COUNT; ++part) {
        auto destination = outB.slice(
            asc::te::make_coord(0, static_cast<Index>(part * partColumns)),
            asc::te::make_shape(static_cast<Index>(kColumns), static_cast<Index>(partColumns)));
        CopyMxOperandTile(copy, destination, b, static_cast<Index>(kOffset),
                          static_cast<Index>(nOffset + part * sourceStride), static_cast<Index>(kColumns),
                          static_cast<Index>(partColumns));
    }
}

template <typename C, typename A, typename B>
__aicore__ inline void IssueMxMmad(const asc::te::mmad_params& params, const C& c, const A& a, const B& b)
{
    auto atom = asc::te::make_mmad(asc::te::mmad_operation{}, MmadTraitMX{}).with(params);
    asc::te::mmad(atom, c, a, b);
}

constexpr uint64_t MX_L0_SCALE_ADDRESS_SHIFT = 4UL;

__aicore__ inline void UpdateMxMmadParams(asc::te::mmad_params& params, uint16_t k, bool firstK, bool lastK,
                                          bool needBias)
{
    params.k = k;
    params.unit_flag = lastK ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
    params.init_with_zero = firstK && !needBias;
}

template <typename Scale, bool OperandB, typename Index = int64_t, typename Tensor>
__aicore__ inline void CopyMxScaleToL0(const Tensor& source, uint64_t l0Offset, uint64_t rows, uint64_t columns,
                                       uint64_t kOffset, uint64_t outerOffset = 0)
{
    using Layout = AscendC::Std::conditional_t<OperandB, asc::te::nn_layout_ptn, asc::te::zz_layout_ptn>;
    using Location = AscendC::Std::conditional_t<OperandB, asc::te::location::l0scaleb, asc::te::location::l0scalea>;
    using Copy = AscendC::Std::conditional_t<OperandB, asc::te::copy_l1_to_l0scaleb, asc::te::copy_l1_to_l0scalea>;
    auto layout = asc::te::make_frame_layout<Layout, AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>>(rows, columns);
    auto destination = asc::te::make_tensor(
        asc::te::make_mem_ptr<Location, Scale>(l0Offset >> MX_L0_SCALE_ADDRESS_SHIFT), layout);
    const Index offset = static_cast<Index>(kOffset);
    if constexpr (OperandB) {
        CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, offset, static_cast<Index>(outerOffset),
                          static_cast<Index>(rows), static_cast<Index>(columns));
    } else {
        CopyMxOperandTile(asc::te::make_copy(Copy{}), destination, source, static_cast<Index>(outerOffset), offset,
                          static_cast<Index>(rows), static_cast<Index>(columns));
    }
}

} // namespace Blaze::Gemm::Tile

#endif
