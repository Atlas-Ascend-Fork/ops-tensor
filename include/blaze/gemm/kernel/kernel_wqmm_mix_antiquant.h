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
 * \file kernel_wqmm_mix_antiquant.h
 * \brief AIV dequantization of 8-bit weights and the corresponding GemmUniversal specialization.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/block/block_mmad_wqmm_mix_weight_prologue.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

using AscendC::Reg::RegTensor;

namespace RegNS = AscendC::Reg;

template <typename XType_, typename WeightType_, typename AntiquantScaleType_, uint32_t UbMte2InnerSize_,
          uint32_t UbMte2BufNum_, bool TransB_, QuantMode AntiquantType_, bool HasAntiquantOffset_,
          typename SyncProtocol_, typename SharedMemProvider_>
class KernelWqmmMixAntiquantPrologue {
    static_assert(AscendC::IsSameType<XType_, half>::value || AscendC::IsSameType<XType_, bfloat16_t>::value,
                  "KernelWqmmMixAntiquantPrologue requires FP16/BF16 output");
    static_assert(AscendC::IsSameType<WeightType_, int8_t>::value ||
                      AscendC::IsSameType<WeightType_, float8_e4m3_t>::value ||
                      AscendC::IsSameType<WeightType_, hifloat8_t>::value,
                  "KernelWqmmMixAntiquantPrologue requires int8, float8_e4m3 or hifloat8 input");
    static_assert(AntiquantType_ == QuantMode::PERTENSOR_MODE || AntiquantType_ == QuantMode::PERCHANNEL_MODE,
                  "KernelWqmmMixAntiquantPrologue supports per-tensor or per-channel antiquantization");
    static_assert(AscendC::IsSameType<AntiquantScaleType_, XType_>::value,
                  "KernelWqmmMixAntiquantPrologue requires scale/offset type to match X");
    static_assert(UbMte2BufNum_ == 2 || UbMte2BufNum_ == 4,
                  "KernelWqmmMixAntiquantPrologue supports only 2 or 4 input buffers");

public:
    struct Params {
        uint64_t kbL1Size{0};
        uint64_t weightStride{0};
        asc_load_l2_cache_mode weightCacheMode{asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM};
        XType_ scaleValue{0};
        XType_ offsetValue{0};
    };

    __aicore__ inline explicit KernelWqmmMixAntiquantPrologue(const Params& params,
                                                              const SharedMemProvider_& sharedMemProvider)
        : sharedMemProvider_(sharedMemProvider)
    {
        Init(params.kbL1Size);
        weightStride_ = params.weightStride;
        weightCacheMode_ = params.weightCacheMode;
        if constexpr (AntiquantType_ == QuantMode::PERTENSOR_MODE) {
            scaleValue_ = params.scaleValue;
            offsetValue_ = params.offsetValue;
        }
    }

    __aicore__ inline ~KernelWqmmMixAntiquantPrologue() { Finalize(); }

    __aicore__ inline void operator()(__gm__ WeightType_* blockB, __gm__ AntiquantScaleType_* blockScale,
                                      __gm__ AntiquantScaleType_* blockOffset, uint64_t tileN, uint64_t kSize)
    {
        ProcessPrologueBlock(blockB, blockScale, blockOffset, tileN, kSize);
    }

private:
    static constexpr uint64_t UB_OUT_BUF_NUM = DOUBLE_BUFFER_COUNT;
    // UB capacities in KiB; all N tiles, including resplit tails, must fit.
    static constexpr uint64_t UB_IN_SIZE = 174;
    static constexpr uint64_t UB_OUT_SIZE = 66;
    static constexpr uint64_t SCALE_SIZE = 4;
    static constexpr uint64_t OFFSET_SIZE = 4;
    // Two FP16/BF16 registers cover 256 elements along the contiguous axis.
    // The outer axis contains 64 elements in the padded NZ/ZN output layout.
    static constexpr uint64_t VF_N = TransB_ ? 64 : 256;
    static constexpr uint64_t VF_K = TransB_ ? 256 : 64;

    template <typename T>
    static constexpr uint32_t VECTOR_REG_SIZE = AscendC::VECTOR_REG_WIDTH /
                                                sizeof(typename AscendC::Std::remove_cvref_t<T>);

    static __aicore__ inline uint64_t CalcRealLen(uint64_t fullLen, uint64_t offset, uint64_t tileLen)
    {
        return offset + tileLen > fullLen ? fullLen - offset : tileLen;
    }

    static constexpr RegNS::CastTrait S8_TO_FP16_TRAIT_ODD = {
        RegNS::RegLayout::ZERO, RegNS::SatMode::UNKNOWN, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    static constexpr RegNS::CastTrait FP16_TO_BF16_TRAIT = {RegNS::RegLayout::UNKNOWN, RegNS::SatMode::UNKNOWN,
                                                            RegNS::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::CAST_RINT};

    static constexpr RegNS::CastTrait S4_TO_FP16_TRAIT_ODD = {
        RegNS::RegLayout::ZERO, RegNS::SatMode::UNKNOWN, RegNS::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

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
        } else if constexpr (AscendC::IsSameType<DtypeIn_, float8_e4m3_t>::value) {
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
                              "WQMM BF16 conversion supports INT8, HiFloat8, and FP8 E4M3 weights");
            }
        } else if constexpr (AscendC::IsSameType<typename RegNS::TypeGet<DtypeOut_>::T, vector_f16>::value) {
            if constexpr (AscendC::IsSameType<DtypeIn_, int8_t>::value ||
                          AscendC::IsSameType<DtypeIn_, hifloat8_t>::value) {
                RegNS::Cast<DtypeOut_, DtypeIn_, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightLowBitVreg, maskAll);
            } else {
                static_assert(Blaze::Gemm::always_false_v<DtypeIn_>,
                              "WQMM FP16 conversion supports INT8, HiFloat8, and FP8 E4M3 weights");
            }
        } else {
            static_assert(Blaze::Gemm::always_false_v<DtypeOut_>, "WQMM dequantization output must be FP16 or BF16");
        }
    }

    template <typename DtypeOut_, uint64_t OuterSize_>
    static __simd_callee__ inline void WeightF16NdRegToNzUb(__ubuf__ DtypeOut_*& weightF16PhyAddr0,
                                                            __ubuf__ DtypeOut_*& weightF16PhyAddr1,
                                                            RegTensor<DtypeOut_>& weightF16Vreg0,
                                                            RegTensor<DtypeOut_>& weightF16Vreg1,
                                                            RegNS::MaskReg& maskAll)
    {
        RegNS::DataCopy<DtypeOut_, RegNS::DataCopyMode::DATA_BLOCK_COPY, RegNS::PostLiteral::POST_MODE_UPDATE>(
            weightF16PhyAddr0, weightF16Vreg0, OuterSize_ + 1, 1, maskAll);
        RegNS::DataCopy<DtypeOut_, RegNS::DataCopyMode::DATA_BLOCK_COPY, RegNS::PostLiteral::POST_MODE_UPDATE>(
            weightF16PhyAddr1, weightF16Vreg1, OuterSize_ + 1, 1, maskAll);
    }

    // (N_, K_) describe the shared tile geometry in all four VF signatures;
    // each orientation uses only its outer dimension for NZ/ZN writeback pitch.
    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, int32_t N_, int32_t K_,
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
            WeightF16NdRegToNzUb<DtypeOut_, N_>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1,
                                                maskAll);
        }
    }

    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, int32_t N_, int32_t K_,
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
            WeightF16NdRegToNzUb<DtypeOut_, K_>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1,
                                                maskAll);
        }
    }

    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, int32_t N_, int32_t K_,
              bool HasOffset_>
    static __simd_vf__ inline void AntiquantPerTensorNdKnVf(__ubuf__ DtypeOut_* weightF16PhyAddr0,
                                                            __ubuf__ DtypeOut_* weightF16PhyAddr1,
                                                            __ubuf__ DtypeIn_* weightLowBitPhyAddr0,
                                                            __ubuf__ DtypeIn_* weightLowBitPhyAddr1,
                                                            DtypeOut_ scaleValue, DtypeOut_ offsetValue,
                                                            uint16_t ubLoopK)
    {
        static constexpr RegNS::LoadDist LD_DIST_W = RegNS::LoadDist::DIST_UNPACK_B8;
        RegTensor<DtypeOut_> antiQuantScaleVreg0;
        RegTensor<DtypeOut_> antiQuantScaleVreg1;
        RegTensor<DtypeIn_> weightVreg0;
        RegTensor<DtypeIn_> weightVreg1;
        RegTensor<DtypeOut_> weightF16Vreg0;
        RegTensor<DtypeOut_> weightF16Vreg1;

        RegNS::MaskReg maskCast = RegNS::CreateMask<DtypeIn_, RegNS::MaskPattern::ALL>();
        RegNS::MaskReg maskAll = RegNS::CreateMask<DtypeOut_, RegNS::MaskPattern::ALL>();

        RegNS::Duplicate(antiQuantScaleVreg0, scaleValue);
        RegNS::Duplicate(antiQuantScaleVreg1, scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg0, weightLowBitPhyAddr0 + ubLoopKIdx * VfUbMte2InnerSize_);
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg1, weightLowBitPhyAddr1 + ubLoopKIdx * VfUbMte2InnerSize_);
            CastLowBitToF16(weightF16Vreg0, weightVreg0, maskCast);
            CastLowBitToF16(weightF16Vreg1, weightVreg1, maskCast);
            if constexpr (HasOffset_) {
                RegNS::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
                RegNS::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
            }
            RegNS::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
            RegNS::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
            WeightF16NdRegToNzUb<DtypeOut_, K_>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1,
                                                maskAll);
        }
    }

    template <uint32_t VfUbMte2InnerSize_, typename DtypeOut_, typename DtypeIn_, int32_t N_, int32_t K_,
              bool HasOffset_>
    static __simd_vf__ inline void AntiquantPerTensorNdNkVf(__ubuf__ DtypeOut_* weightF16PhyAddr0,
                                                            __ubuf__ DtypeOut_* weightF16PhyAddr1,
                                                            __ubuf__ DtypeIn_* weightLowBitPhyAddr0,
                                                            __ubuf__ DtypeIn_* weightLowBitPhyAddr1,
                                                            DtypeOut_ scaleValue, DtypeOut_ offsetValue,
                                                            uint16_t ubLoopN)
    {
        static constexpr RegNS::LoadDist LD_DIST_W = RegNS::LoadDist::DIST_UNPACK_B8;
        RegTensor<DtypeOut_> antiQuantScaleVreg;
        RegTensor<DtypeIn_> weightVreg0;
        RegTensor<DtypeIn_> weightVreg1;
        RegTensor<DtypeOut_> weightF16Vreg0;
        RegTensor<DtypeOut_> weightF16Vreg1;

        RegNS::MaskReg maskCast = RegNS::CreateMask<DtypeIn_, RegNS::MaskPattern::ALL>();
        RegNS::MaskReg maskAll = RegNS::CreateMask<DtypeOut_, RegNS::MaskPattern::ALL>();

        RegNS::Duplicate(antiQuantScaleVreg, scaleValue);

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg0, weightLowBitPhyAddr0 + ubLoopNIdx * VfUbMte2InnerSize_);
            RegNS::DataCopy<DtypeIn_, LD_DIST_W>(weightVreg1, weightLowBitPhyAddr1 + ubLoopNIdx * VfUbMte2InnerSize_);
            CastLowBitToF16(weightF16Vreg0, weightVreg0, maskCast);
            CastLowBitToF16(weightF16Vreg1, weightVreg1, maskCast);
            if constexpr (HasOffset_) {
                RegNS::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
                RegNS::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
            }
            RegNS::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
            RegNS::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
            WeightF16NdRegToNzUb<DtypeOut_, N_>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1,
                                                maskAll);
        }
    }

    using SyncProtocol = SyncProtocol_;

    // Quantized weight capacities are in bytes.
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE = UB_IN_SIZE * 1024;
    // Output, antiquantScale and antiquantOffset capacities are in 16-bit elements.
    static constexpr uint64_t HIGH_BIT_DATA_UB_TOTAL_SIZE = UB_OUT_SIZE * 512;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_TOTAL_SIZE = SCALE_SIZE * 512;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_TOTAL_SIZE = OFFSET_SIZE * 512;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE = WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE /
                                                                           UbMte2BufNum_;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE = ANTIQUANT_SCALE_UB_TOTAL_SIZE / UbMte2BufNum_;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE = ANTIQUANT_OFFSET_UB_TOTAL_SIZE / UbMte2BufNum_;
    static constexpr uint64_t HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE = HIGH_BIT_DATA_UB_TOTAL_SIZE / UB_OUT_BUF_NUM;
    struct Storage {
        BufferSlot ubWeight;
        BufferSlot ubOutput;
        BufferSlot ubScale;
        BufferSlot ubOffset;
        BufferSlot inputSlots[UbMte2BufNum_];
        BufferSlot outputSlots[UB_OUT_BUF_NUM];

        __aicore__ inline void Init()
        {
            const uint64_t outputBase = WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE;
            const uint64_t scaleBase = outputBase + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(XType_);
            ubWeight = {0UL, 0U};
            ubOutput = {outputBase, 1U};
            ubScale = {scaleBase, 2U};
            ubOffset = {scaleBase + ANTIQUANT_SCALE_UB_TOTAL_SIZE * sizeof(AntiquantScaleType_), 3U};
            for (uint32_t index = 0; index < UbMte2BufNum_; ++index) {
                inputSlots[index] = {index * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE, static_cast<uint8_t>(index)};
            }
            for (uint32_t index = 0; index < UB_OUT_BUF_NUM; ++index) {
                outputSlots[index] = {outputBase + index * HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE * sizeof(XType_),
                                      static_cast<uint8_t>(UbMte2BufNum_ + index)};
            }
        }

        __aicore__ inline const BufferSlot& GetInputSlot(uint32_t index) const { return inputSlots[index]; }

        __aicore__ inline const BufferSlot& GetOutputSlot(uint32_t index) const { return outputSlots[index]; }
    };

    uint64_t kbL1Size_;
    uint64_t weightStride_;
    asc_load_l2_cache_mode weightCacheMode_;
    uint64_t nL1Size_;
    uint64_t ubMte2LoopIdx_ = 0;
    uint64_t ubCalLoopId_ = 0;
    uint64_t cvLoopIdx_ = 0;
    uint64_t l1SplitVecOffset_;
    XType_ scaleValue_ = 0;
    XType_ offsetValue_ = 0;
    Storage storage_;
    SharedMemProvider_ sharedMemProvider_;

    // Typed UB pointers use element offsets from each buffer's byte address.
    // Layouts describe the physical regions used for UB-to-L1 copies.
    __aicore__ inline __ubuf__ WeightType_* GetInputWeight(uint32_t bufferId) const
    {
        return reinterpret_cast<__ubuf__ WeightType_*>(storage_.GetInputSlot(bufferId).Addr());
    }

    __aicore__ inline __ubuf__ XType_* GetConvertedWeight(uint32_t bufferId) const
    {
        return reinterpret_cast<__ubuf__ XType_*>(storage_.GetOutputSlot(bufferId).Addr());
    }

    __aicore__ inline __ubuf__ AntiquantScaleType_* GetScale(uint32_t bufferId) const
    {
        return reinterpret_cast<__ubuf__ AntiquantScaleType_*>(storage_.ubScale.Addr()) +
               bufferId * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE;
    }

    __aicore__ inline __ubuf__ AntiquantScaleType_* GetOffset(uint32_t bufferId) const
    {
        return reinterpret_cast<__ubuf__ AntiquantScaleType_*>(storage_.ubOffset.Addr()) +
               bufferId * ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE;
    }

    static __aicore__ inline auto MakeSrcLayout(int64_t kSize, int64_t nSize, int64_t srcPitch)
    {
        if constexpr (TransB_) {
            return Blaze::Gemm::ZnRowPaddingUBLayout<XType_>{}(kSize, nSize, srcPitch);
        } else {
            return Blaze::Gemm::NzColPaddingUBLayout<XType_>{}(kSize, nSize, srcPitch);
        }
    }

    static __aicore__ inline auto MakeDstLayout(int64_t kSize, int64_t nSize)
    {
        using Trait = asc::te::layout_trait_default<XType_>;
        if constexpr (TransB_) {
            return asc::te::make_frame_layout<asc::te::zn_layout_ptn, Trait>(kSize, nSize);
        } else {
            return asc::te::make_frame_layout<asc::te::nz_layout_ptn, Trait>(kSize, nSize);
        }
    }

    __aicore__ inline void CopyConvertedWeight(uint32_t bufferId, uint64_t elementOffset, __ubuf__ XType_* src,
                                               uint64_t realN, uint64_t realK, uint64_t l1N, uint64_t l1K) const
    {
        auto dst = sharedMemProvider_(bufferId).get() + elementOffset;
        // Describe weight groups in (K,N) coordinates. The UB pitch is in elements
        // and includes a 32-byte padding block, also for tail tiles.
        constexpr uint64_t ELEMS_PER_BLOCK = Blaze::Gemm::BLOCK_CUBE;
        const int64_t srcPitch = static_cast<int64_t>(((TransB_ ? VF_N : VF_K) + 1UL) * ELEMS_PER_BLOCK);
        const auto srcLayout = MakeSrcLayout(static_cast<int64_t>(realK), static_cast<int64_t>(realN), srcPitch);
        const auto dstLayout = MakeDstLayout(static_cast<int64_t>(l1K), static_cast<int64_t>(l1N));
        const auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(src), srcLayout);
        const auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1>(dst), dstLayout);
        auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});
        asc::te::copy(copyUB2L1, dstTensor, srcTensor);
    }

    __aicore__ inline void Init(uint64_t kbL1Size)
    {
        kbL1Size_ = kbL1Size;
        nL1Size_ = 0;
        storage_.Init();
    }

    __aicore__ inline void Finalize()
    {
        if (cvLoopIdx_ > 0U) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1U) {
            WaitAicToAiv();
        }
    }

    __aicore__ inline void WaitAicToAiv()
    {
        AscendC::CrossCoreWaitFlag<SyncProtocol::MODE, PIPE_MTE3>(
            SyncProtocol::AIC_FREE_FLAG + (AscendC::GetSubBlockIdx() == 1 ? SyncProtocol::FLAG_ID_MAX : 0));
    }
    __aicore__ inline void SetAivToAic()
    {
        AscendC::CrossCoreSetFlag<SyncProtocol::MODE, PIPE_MTE3>(
            SyncProtocol::AIV_READY_FLAG + (AscendC::GetSubBlockIdx() == 1 ? SyncProtocol::FLAG_ID_MAX : 0));
    }

    __aicore__ inline void ProcessPrologueBlockNdKn(__gm__ WeightType_* gB, __gm__ AntiquantScaleType_* gScale,
                                                    __gm__ AntiquantScaleType_* gOffset, uint64_t tileN, uint64_t kSize)
    {
        nL1Size_ = tileN;
        uint64_t kSplitSize = kbL1Size_ >> 1;
        for (uint64_t kL1Offset = 0; kL1Offset < kSize; kL1Offset += kbL1Size_, cvLoopIdx_++) {
            uint64_t l1RealK = CalcRealLen(kSize, kL1Offset, kbL1Size_);
            uint64_t coreKOffset = AscendC::GetSubBlockIdx() * kSplitSize;
            uint64_t mte2RealK = AscendC::GetSubBlockIdx() == 0 ? Min(kSplitSize, l1RealK) :
                                 l1RealK > kSplitSize           ? l1RealK - kSplitSize :
                                                                  0;

            auto gBTile = gB + (kL1Offset + coreKOffset) * weightStride_;
            CopyGmToUb(gBTile, gScale, gOffset, 0UL, tileN, mte2RealK);

            // The two L1 weight buffers are initially free; reuse waits for the AIC.
            // Each cross-core flag signals one handoff and cannot count repeated sets.
            if (cvLoopIdx_ > 1U) {
                WaitAicToAiv();
            }
            if (mte2RealK > 0) {
                uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (UbMte2BufNum_ - 1);
                const auto& inputSlot = storage_.GetInputSlot(mte2BufIdx);
                auto inputVectorLock = inputSlot.LockV();
                auto ubBInB8 = GetInputWeight(mte2BufIdx);
                auto ubScale = GetScale(mte2BufIdx);
                auto ubOffset = GetOffset(mte2BufIdx);

                for (uint64_t vfKOffset = 0; vfKOffset < mte2RealK; vfKOffset += VF_K) {
                    uint64_t vfRealK = CalcRealLen(mte2RealK, vfKOffset, VF_K);
                    for (uint64_t nOffset = 0; nOffset < tileN; nOffset += VF_N) {
                        uint64_t realN = CalcRealLen(tileN, nOffset, VF_N);
                        uint64_t outputBufIdx = ubCalLoopId_ & (UB_OUT_BUF_NUM - 1U);
                        const auto& outputSlot = storage_.GetOutputSlot(outputBufIdx);
                        auto regBOut = GetConvertedWeight(outputBufIdx);
                        auto ubBInB8Sub = ubBInB8 + vfKOffset * UbMte2InnerSize_ + nOffset;
                        {
                            auto outputVectorLock = outputSlot.LockV();
                            if constexpr (AntiquantType_ == QuantMode::PERTENSOR_MODE) {
                                asc_vf_call<AntiquantPerTensorNdKnVf<UbMte2InnerSize_, XType_, WeightType_, VF_N, VF_K,
                                                                     HasAntiquantOffset_>>(
                                    regBOut, regBOut + (VF_K + 1) * VECTOR_REG_SIZE<XType_>, ubBInB8Sub,
                                    ubBInB8Sub + VECTOR_REG_SIZE<XType_>, scaleValue_, offsetValue_,
                                    static_cast<uint16_t>(vfRealK));
                            } else {
                                asc_vf_call<AntiquantPerChannelNdKnVf<UbMte2InnerSize_, XType_, WeightType_, VF_N, VF_K,
                                                                      HasAntiquantOffset_>>(
                                    regBOut, regBOut + (VF_K + 1) * VECTOR_REG_SIZE<XType_>, ubBInB8Sub,
                                    ubBInB8Sub + VECTOR_REG_SIZE<XType_>, ubScale + nOffset, ubOffset + nOffset,
                                    static_cast<uint16_t>(vfRealK));
                            }
                        }

                        uint32_t l1BufferId = cvLoopIdx_ & (DOUBLE_BUFFER_COUNT - 1U);
                        uint64_t offset = CeilAlign(l1RealK, Blaze::Gemm::BLOCK_CUBE) * nOffset +
                                          (coreKOffset + vfKOffset) * Blaze::Gemm::BLOCK_CUBE;
                        {
                            auto outputMte3Lock = outputSlot.LockMte3();
                            CopyConvertedWeight(l1BufferId, offset, regBOut, realN, vfRealK, realN, l1RealK);
                        }
                        ubCalLoopId_++;
                    }
                }
            }
            SetAivToAic();
        }
    }

    __aicore__ inline void ProcessPrologueBlock(__gm__ WeightType_* gB, __gm__ AntiquantScaleType_* gScale,
                                                __gm__ AntiquantScaleType_* gOffset, uint64_t tileN, uint64_t kSize)
    {
        if constexpr (!TransB_) {
            ProcessPrologueBlockNdKn(gB, gScale, gOffset, tileN, kSize);
            return;
        }
        nL1Size_ = tileN;
        uint64_t vec0Mte2RealN = nL1Size_ >> 1;
        uint64_t l1RealN = AscendC::GetSubBlockIdx() == 0 ? vec0Mte2RealN : nL1Size_ - vec0Mte2RealN;
        l1SplitVecOffset_ = AscendC::GetSubBlockIdx() * vec0Mte2RealN;

        for (uint64_t kMte2Offset = 0; kMte2Offset < kSize; kMte2Offset += UbMte2InnerSize_) {
            uint64_t mte2RealK = CalcRealLen(kSize, kMte2Offset, UbMte2InnerSize_);
            uint64_t localNOffset = l1SplitVecOffset_;
            auto gBTile = gB + localNOffset * weightStride_ + kMte2Offset;
            CopyGmToUb(gBTile, gScale, gOffset, localNOffset, l1RealN, mte2RealK);

            uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (UbMte2BufNum_ - 1);
            auto ubBInB8 = GetInputWeight(mte2BufIdx);
            auto ubScale = GetScale(mte2BufIdx);
            auto ubOffset = GetOffset(mte2BufIdx);
            const auto& inputSlot = storage_.GetInputSlot(mte2BufIdx);
            auto inputVectorLock = inputSlot.LockV();

            for (uint64_t kWeightLowBitUbOffset = 0; kWeightLowBitUbOffset < mte2RealK;
                 kWeightLowBitUbOffset += kbL1Size_, cvLoopIdx_++) {
                uint64_t l1RequireVfComputeRealK = CalcRealLen(mte2RealK, kWeightLowBitUbOffset, kbL1Size_);
                if (cvLoopIdx_ > 1U) {
                    WaitAicToAiv();
                }
                constexpr uint64_t VF_K_NDNK = VF_K;
                for (uint64_t vfKOffset = 0; vfKOffset < l1RequireVfComputeRealK; vfKOffset += VF_K_NDNK) {
                    uint64_t vfRealK = CalcRealLen(l1RequireVfComputeRealK, vfKOffset, VF_K_NDNK);
                    for (uint64_t nOffset = 0; nOffset < l1RealN; nOffset += VF_N) {
                        uint64_t realN = CalcRealLen(l1RealN, nOffset, VF_N);
                        uint64_t outputBufIdx = ubCalLoopId_ & (UB_OUT_BUF_NUM - 1U);
                        const auto& outputSlot = storage_.GetOutputSlot(outputBufIdx);
                        auto regBOut = GetConvertedWeight(outputBufIdx);
                        auto ubBInB8SubL1 = ubBInB8 + nOffset * UbMte2InnerSize_ + kWeightLowBitUbOffset + vfKOffset;
                        {
                            auto outputVectorLock = outputSlot.LockV();
                            if constexpr (AntiquantType_ == QuantMode::PERTENSOR_MODE) {
                                asc_vf_call<AntiquantPerTensorNdNkVf<UbMte2InnerSize_, XType_, WeightType_, VF_N, VF_K,
                                                                     HasAntiquantOffset_>>(
                                    regBOut, regBOut + (VF_N + 1) * VECTOR_REG_SIZE<XType_>, ubBInB8SubL1,
                                    ubBInB8SubL1 + VECTOR_REG_SIZE<XType_>, scaleValue_, offsetValue_,
                                    static_cast<uint16_t>(realN));
                            } else {
                                asc_vf_call<AntiquantPerChannelNdNkVf<UbMte2InnerSize_, XType_, WeightType_, VF_N, VF_K,
                                                                      HasAntiquantOffset_>>(
                                    regBOut, regBOut + (VF_N + 1) * VECTOR_REG_SIZE<XType_>, ubBInB8SubL1,
                                    ubBInB8SubL1 + VECTOR_REG_SIZE<XType_>, ubScale + nOffset, ubOffset + nOffset,
                                    static_cast<uint16_t>(realN));
                            }
                        }

                        uint32_t l1BufferId = cvLoopIdx_ & (DOUBLE_BUFFER_COUNT - 1U);
                        uint64_t offset = vfKOffset * CeilAlign(nL1Size_, Blaze::Gemm::BLOCK_CUBE) +
                                          (l1SplitVecOffset_ + nOffset) * Blaze::Gemm::BLOCK_CUBE;
                        {
                            auto outputMte3Lock = outputSlot.LockMte3();
                            CopyConvertedWeight(l1BufferId, offset, regBOut, realN, vfRealK, nL1Size_, vfRealK);
                        }
                        ubCalLoopId_++;
                    }
                }
                SetAivToAic();
            }
        }
    }

    __aicore__ inline void CopyGmToUb(__gm__ WeightType_* gB, __gm__ AntiquantScaleType_* gScale,
                                      __gm__ AntiquantScaleType_* gOffset, uint64_t offsetN, uint64_t ubMte2NSize,
                                      uint64_t ubMte2KSize)
    {
        if (ubMte2NSize == 0 || ubMte2KSize == 0) {
            ubMte2LoopIdx_++;
            return;
        }
        uint64_t inputBufIdx = ubMte2LoopIdx_ & (UbMte2BufNum_ - 1U);
        const auto& inputSlot = storage_.GetInputSlot(inputBufIdx);
        auto inputMte2Lock = inputSlot.LockMte2();
        uint64_t blockCount = TransB_ ? ubMte2NSize : ubMte2KSize;
        uint64_t blockLen = TransB_ ? ubMte2KSize : ubMte2NSize;
        // GM and UB pitches are byte distances between consecutive row starts.
        // Use constant padding and the configured weight cache mode.
        asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(GetInputWeight(inputBufIdx)),
                             reinterpret_cast<__gm__ uint8_t*>(gB), static_cast<uint16_t>(blockCount),
                             static_cast<uint32_t>(blockLen), 0, 0, true, weightCacheMode_, weightStride_,
                             static_cast<uint32_t>(UbMte2InnerSize_));

        if constexpr (AntiquantType_ != QuantMode::PERTENSOR_MODE) {
            uint32_t scaleBytes = static_cast<uint32_t>(ubMte2NSize * sizeof(AntiquantScaleType_));
            asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint16_t*>(GetScale(inputBufIdx)),
                                 reinterpret_cast<__gm__ uint16_t*>(gScale + offsetN), 1, scaleBytes, 0, 0, true,
                                 asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, uint64_t{scaleBytes}, scaleBytes);
            if constexpr (HasAntiquantOffset_) {
                asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint16_t*>(GetOffset(inputBufIdx)),
                                     reinterpret_cast<__gm__ uint16_t*>(gOffset + offsetN), 1, scaleBytes, 0, 0, true,
                                     asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, uint64_t{scaleBytes}, scaleBytes);
            }
        }
        ubMte2LoopIdx_++;
    }
};

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<BlockEpilogue_, void> &&
        AscendC::Std::is_same_v<KernelMmadAPrefetchBAntiquant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleType = typename BlockMmad::ScaleType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    static_assert(AscendC::Std::is_same_v<AType, CType>, "WQMM Blaze requires X and Y to have the same type");
    static_assert(AscendC::Std::is_same_v<BiasType, AType> || AscendC::Std::is_same_v<BiasType, float>,
                  "WQMM Blaze bias must be X type or FP32");

    struct PrologueParams {
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        GM_ADDR offsetGmAddr{nullptr};
        uint32_t weightL2Cacheable{0};
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        uint64_t aPreloadSize{0};
        uint64_t aElementCount{0};
        PrologueParams prologueParams;
        BlockSchedulerParams schedulerParams;
    };

    __aicore__ inline GemmUniversal() = default;
    __aicore__ inline ~GemmUniversal() = default;

    __aicore__ inline void operator()(const Params& params) const { Execute(params); }

private:
    // LayoutB uses (N,K) coordinates for weight: ND stores N x K, DN stores K x N.
    // The matrix multiplication treats weight as a logical (K,N) matrix.
    static constexpr bool TRANS_B = !IsTrans<LayoutB>::value;
    static constexpr uint32_t N_RANGE_MAIN = 0;        // N0: main blocks.
    static constexpr uint32_t N_RANGE_FIRST_TAIL = 1;  // N1: first tail blocks.
    static constexpr uint32_t N_RANGE_SECOND_TAIL = 2; // N2: second tail blocks after resplitting.

    struct SharedWeightMemProvider {
        BlockMmad& blockMmad;

        __aicore__ inline auto operator()(uint32_t bufferId) const
        {
            return blockMmad.template GetShareMemPtr<asc::te::location::l1, AType>(bufferId);
        }
    };

    __aicore__ inline static void Execute(const Params& params)
    {
        BlockMmad blockMmad;
        blockMmad.Init(params.mmadParams);
        SharedWeightMemProvider getSharedWeightMem{blockMmad};
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        if ASCEND_IS_AIV {
            RunAiv(params, scheduler, getSharedWeightMem);
        }
        if ASCEND_IS_AIC {
            RunAic(params, scheduler, blockMmad);
        }
    }

    template <typename SharedMemProvider>
    __aicore__ inline static void RunAiv(const Params& params, const BlockScheduler& scheduler,
                                         const SharedMemProvider& sharedMemProvider)
    {
        using WeightPrologue = KernelWqmmMixAntiquantPrologue<
            AType, BType, ScaleType, DispatchPolicy::UB_MTE2_INNER_SIZE, DispatchPolicy::UB_MTE2_BUFFER_NUM, TRANS_B,
            DispatchPolicy::ANTIQUANT_TYPE, DispatchPolicy::HAS_ANTIQUANT_OFFSET, typename DispatchPolicy::SyncProtocol,
            SharedMemProvider>;
        const uint64_t n = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        const uint64_t k = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        AType scaleValue{0};
        AType offsetValue{0};
        if constexpr (DispatchPolicy::ANTIQUANT_TYPE == QuantMode::PERTENSOR_MODE) {
            scaleValue = static_cast<AType>(reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.scaleGmAddr)[0]);
            if constexpr (DispatchPolicy::HAS_ANTIQUANT_OFFSET) {
                offsetValue = static_cast<AType>(
                    reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.offsetGmAddr)[0]);
            }
        }

        typename WeightPrologue::Params prologueParams{
            .kbL1Size = static_cast<uint64_t>(asc::te::get<3>(params.mmadParams.l1TileShape)),
            .weightStride = TRANS_B ? k : n,
            .weightCacheMode = params.prologueParams.weightL2Cacheable ? asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM :
                                                                         asc_load_l2_cache_mode::NOTALLOC_KEEP,
            .scaleValue = scaleValue,
            .offsetValue = offsetValue};
        WeightPrologue blockPrologue(prologueParams, sharedMemProvider);

        auto blockNums = scheduler.GetBlockNums();
        for (uint64_t mIdx = 0; mIdx < asc::te::get<0>(blockNums); ++mIdx) {
            // Block counts are ordered as (M, N0, N1, N2); N counts start at tuple index 1.
            ProcessAivNRange<N_RANGE_MAIN>(blockPrologue, scheduler, params, k,
                                           asc::te::get<N_RANGE_MAIN + 1>(blockNums));
            ProcessAivNRange<N_RANGE_FIRST_TAIL>(blockPrologue, scheduler, params, k,
                                                 asc::te::get<N_RANGE_FIRST_TAIL + 1>(blockNums));
            ProcessAivNRange<N_RANGE_SECOND_TAIL>(blockPrologue, scheduler, params, k,
                                                  asc::te::get<N_RANGE_SECOND_TAIL + 1>(blockNums));
        }
    }

    __aicore__ inline static void RunAic(const Params& params, const BlockScheduler& scheduler, BlockMmad& blockMmad)
    {
        const int64_t m = static_cast<int64_t>(asc::te::get<MNK_M>(params.problemShape));
        const int64_t n = static_cast<int64_t>(asc::te::get<MNK_N>(params.problemShape));
        const int64_t k = static_cast<int64_t>(asc::te::get<MNK_K>(params.problemShape));
        auto gmA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)),
            asc::te::frame_layout_format<LayoutA>{}(m, k));
        auto gmC = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)),
            asc::te::frame_layout_format<LayoutC>{}(m, n));
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                               reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr)),
                                           asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, n));

        PreloadA(gmA, params);
        auto blockNums = scheduler.GetBlockNums();
        for (uint64_t mIdx = 0; mIdx < asc::te::get<0>(blockNums); ++mIdx) {
            uint64_t coordM = scheduler.GetBlockCoordM(mIdx);
            uint64_t tileM = scheduler.GetBlockShapeM(coordM);
            auto blockA = gmA.slice(asc::te::make_coord(coordM, 0UL),
                                    asc::te::make_shape(tileM, static_cast<uint64_t>(k)));
            ProcessAicNRange<N_RANGE_MAIN>(blockMmad, scheduler, blockA, gmBias, gmC, coordM, tileM,
                                           asc::te::get<N_RANGE_MAIN + 1>(blockNums), params.mmadParams.hasBias);
            ProcessAicNRange<N_RANGE_FIRST_TAIL>(blockMmad, scheduler, blockA, gmBias, gmC, coordM, tileM,
                                                 asc::te::get<N_RANGE_FIRST_TAIL + 1>(blockNums),
                                                 params.mmadParams.hasBias);
            ProcessAicNRange<N_RANGE_SECOND_TAIL>(blockMmad, scheduler, blockA, gmBias, gmC, coordM, tileM,
                                                  asc::te::get<N_RANGE_SECOND_TAIL + 1>(blockNums),
                                                  params.mmadParams.hasBias);
        }
    }

    template <typename TensorA>
    __aicore__ inline static void PreloadA(const TensorA& tensorA, const Params& params)
    {
        // Keep the L1 bias reservation consistent with BlockMmad::InitBufferSlots.
        constexpr uint64_t BIAS_L1_BYTES = 4096UL;
        uint64_t xOffset = AscendC::GetBlockIdx() * params.aPreloadSize;
        if (params.aPreloadSize == 0 || xOffset >= params.aElementCount) {
            return;
        }
        uint64_t preloadCount = Min(params.aPreloadSize, params.aElementCount - xOffset);
        uint64_t nBase = static_cast<uint64_t>(asc::te::get<MNK_N>(params.mmadParams.l1TileShape));
        uint64_t kL1Size = static_cast<uint64_t>(asc::te::get<3>(params.mmadParams.l1TileShape));
        uint64_t l1AOffset = nBase * kL1Size * sizeof(AType) + (params.mmadParams.hasBias ? BIAS_L1_BYTES : 0UL);
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, preloadCount);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                            reinterpret_cast<__gm__ AType*>(tensorA.data().get()) + xOffset),
                                        layout);
        auto l1A = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1AOffset), layout);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        asc::te::copy(copyGM2L1, l1A, gmA);
        // Preload warms x in L2 and waits for local MTE2 completion.
        // The input copy in BlockMmad provides the MTE2-to-MTE1 synchronization for computation.
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    template <uint32_t N_RANGE, typename WeightPrologue>
    __aicore__ inline static void ProcessAivNRange(WeightPrologue& blockPrologue, const BlockScheduler& scheduler,
                                                   const Params& params, uint64_t k, uint64_t nBlockNum)
    {
        for (uint64_t nIdx = 0; nIdx < nBlockNum; ++nIdx) {
            uint64_t coordN = scheduler.template GetBlockCoordN<N_RANGE>(nIdx);
            uint64_t tileN = scheduler.template GetBlockShapeN<N_RANGE>(coordN);
            auto blockWeight = reinterpret_cast<__gm__ BType*>(params.prologueParams.bGmAddr) +
                               (TRANS_B ? coordN * k : coordN);
            auto blockScale = reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.scaleGmAddr);
            auto blockOffset = reinterpret_cast<__gm__ ScaleType*>(params.prologueParams.offsetGmAddr);
            if constexpr (DispatchPolicy::ANTIQUANT_TYPE != QuantMode::PERTENSOR_MODE) {
                blockScale += coordN;
                if constexpr (DispatchPolicy::HAS_ANTIQUANT_OFFSET) {
                    blockOffset += coordN;
                }
            }
            blockPrologue(blockWeight, blockScale, blockOffset, tileN, k);
        }
    }

    template <uint32_t N_RANGE, typename TensorA, typename TensorBias, typename TensorC>
    __aicore__ inline static void ProcessAicNRange(BlockMmad& blockMmad, const BlockScheduler& scheduler,
                                                   const TensorA& blockA, const TensorBias& gmBias, TensorC& gmC,
                                                   uint64_t coordM, uint64_t tileM, uint64_t nBlockNum, bool hasBias)
    {
        for (uint64_t nIdx = 0; nIdx < nBlockNum; ++nIdx) {
            uint64_t coordN = scheduler.template GetBlockCoordN<N_RANGE>(nIdx);
            uint64_t tileN = scheduler.template GetBlockShapeN<N_RANGE>(coordN);
            uint64_t biasOffset = hasBias ? coordN : 0UL;
            auto blockBias = gmBias.slice(asc::te::make_coord(0UL, biasOffset), asc::te::make_shape(1UL, tileN));
            auto blockC = gmC.slice(asc::te::make_coord(coordM, coordN), asc::te::make_shape(tileM, tileN));
            blockMmad(blockA, blockBias, blockC, asc::te::make_shape(tileM, tileN));
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
