/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_rotate_quant.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/epilogue/block/block_epilogue_types.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/tile/compute.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = Fusion::DefaultFusion<DataTypeOut_, DataTypeIn_>, bool EnableLevel0_ = false>
class BlockEpilogueRotateQuant {
public:
    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;
    using FusionOp = FusionOp_;
    static constexpr bool ENABLE_LEVEL0 = EnableLevel0_;

    // The dual-level kernel reserves this complete MX workspace in each AIV.
    // Its collect and FP32 level0 scale buffers must be placed after this region.
    static constexpr uint32_t NORMALIZED_COLS = 512;
    static constexpr uint32_t MAX_NORMALIZED_ROWS = 32;
    static constexpr uint32_t NORMALIZED_WORKSPACE_BYTES = 54 * 1024;
    static_assert(AscendC::IsSameType<DataTypeIn, bfloat16_t>::value,
                  "BlockEpilogueRotateQuant requires BF16 input from Fixpipe.");
    static_assert(sizeof(DataTypeScale) == sizeof(int8_t), "RotateQuant requires one-byte MX scales.");

    static_assert(AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value,
                  "BlockEpilogueRotateQuant requires MX FP4/FP8 output.");

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

private:
    struct MxParams {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t> problemShape{};
        bfloat16_t alpha{0.0f};
        bool needClamp{false};
        int64_t axis{0};
        int64_t roundMode{0};
        int64_t scaleAlg{0};
        float dstTypeMax{0.0f};
        float invDstTypeMax{0.0f};
        uint32_t ubDB{1};
        bool splitM{false};
    };

    struct Level0Params {
        GM_ADDR scaleGmAddr{nullptr};
        int64_t scalesPerRow{0};
        uint32_t tileRows{0};
        uint32_t rotK{0};
        uint32_t workspaceBase{0}; // Byte offset of the private workspace within this AIV's UB.
        uint32_t inputPitch{0};    // BF16 elements between rows in a Fixpipe slot.
        uint32_t collectUbOffset{0};
        uint32_t level0ScaleUbOffset{0};
        bfloat16_t alpha{0.0f};
        bool needClamp{false};
    };

    struct DualLevelParams {
        MxParams mx{};
        Level0Params level0{};
    };

public:
    using Params = AscendC::Std::conditional_t<EnableLevel0_, DualLevelParams, MxParams>;

    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    __aicore__ inline BlockEpilogueRotateQuant();
    __aicore__ inline ~BlockEpilogueRotateQuant();
    __aicore__ inline void Init(Params const& params);
    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);

private:
    // Original matrix coordinates for this AIV's quantization tile.
    struct RotateQuantTileContext {
        int64_t mStart;
        int64_t nStart;
        uint32_t rows;
        uint32_t validCols;
    };

    __aicore__ inline void ConfigureMx(const MxParams& params);
    __aicore__ inline void RunLevel0Tile(const RotateQuantTileContext& tile, const Level0Params& level0);
    __aicore__ inline void NormalizeTile(const RotateQuantTileContext& tile, const Level0Params& level0);
    __aicore__ inline void StoreLevel0Scale(const RotateQuantTileContext& tile, const Level0Params& level0);
    __aicore__ inline void ComputeNormalizedTile(const RotateQuantTileContext& tile);
    __aicore__ inline void StoreNormalizedTile(const RotateQuantTileContext& tile);

    __aicore__ inline static constexpr uint32_t WorkspaceBytes(uint32_t maxRows)
    {
        return (SCALE_BLOCK_UB_OFFSET + maxRows * SCALE_STORE_STRIDE + SCALE_STORE_STRIDE - 1) / SCALE_STORE_STRIDE *
               SCALE_STORE_STRIDE;
    }

    __aicore__ inline uint32_t WaitForFixpipe() const;
    __aicore__ inline void ReleaseFixpipe(uint32_t slot, bool hasMte3Dependency);
    __aicore__ inline void Drain();
    __aicore__ inline void SetWorkspace(uint32_t ubBaseOffset, uint32_t writebackEventId);
    static constexpr uint32_t WORKSPACE_COUNT = 2;
    static constexpr int64_t MODE_RINT = 4;
    static constexpr int64_t MODE_ROUND = 0;
    static constexpr int64_t MODE_FLOOR = 1;

    static constexpr int32_t GROUP_SIZE = 32;
    static constexpr int32_t MN_SIZE = 16 * 1024;
    static constexpr int32_t EMAX_SIZE = 512;
    // All buffer offsets are relative to this instance's workspace base.
    static constexpr uint32_t X_UB_OFFSET = 0;
    static constexpr uint32_t Y_UB_OFFSET = X_UB_OFFSET + MN_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t LIMIT_UB_OFFSET = Y_UB_OFFSET + MN_SIZE * sizeof(int8_t);
    static constexpr uint32_t MAX_EXP_UB_OFFSET = LIMIT_UB_OFFSET + EMAX_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t RECIPROCAL_UB_OFFSET = MAX_EXP_UB_OFFSET + EMAX_SIZE * sizeof(uint16_t);
    static constexpr uint32_t SCALE_UB_OFFSET = RECIPROCAL_UB_OFFSET + EMAX_SIZE * sizeof(uint16_t);
    static constexpr uint32_t SCALE_BLOCK_UB_OFFSET = SCALE_UB_OFFSET + EMAX_SIZE * sizeof(int8_t);

    static constexpr float SIX_FLOAT = 6.0f;
    static constexpr float SEVEN_FLOAT = 7.0f;
    static constexpr uint32_t SCALE_STORE_STRIDE = 32;

    static_assert(MAX_NORMALIZED_ROWS * NORMALIZED_COLS <= MN_SIZE,
                  "Normalized tiles must fit the existing MX input buffer.");
    static_assert(MAX_NORMALIZED_ROWS * (NORMALIZED_COLS / GROUP_SIZE) <= EMAX_SIZE,
                  "Normalized tiles must fit the existing MX scale buffers.");
    static_assert(SCALE_BLOCK_UB_OFFSET + MAX_NORMALIZED_ROWS * SCALE_STORE_STRIDE + 512 <= NORMALIZED_WORKSPACE_BYTES,
                  "Reserve scale rows and a readable vector guard inside the MX workspace.");

    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
    static constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;

    __aicore__ inline void InitGlobalBuffer(MxParams const& params);
    __aicore__ inline void InitProblemShape(const ProblemShape& problemShape);
    __aicore__ inline void InitQuantParams(bfloat16_t alpha, bool needClamp, int64_t axis, int64_t roundMode,
                                           int64_t scaleAlg, float dstTypeMax, float invDstTypeMax);
    __aicore__ inline void CopyUbToGm(__gm__ int8_t* dst, uint32_t srcUbOffset, uint16_t rows, uint32_t columns,
                                      uint32_t srcRowPitch, uint32_t dstRowPitch);
    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, uint16_t blockCount, uint32_t dataSize,
                                                uint32_t dstStride);
    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, uint32_t srcUbOffset, uint16_t blockCount,
                                               uint32_t dataSize, uint32_t dstStride, uint32_t srcStride);
    __aicore__ inline void CopyPadScaleFromUbToGm(uint64_t offset, uint32_t nKRatio, uint32_t nkIdx, uint32_t blkM);

    __aicore__ inline void ComputeMxQuant(uint32_t totalDataInUB);
    __aicore__ inline void ClearScaleBlock(uint32_t blkM);
    __aicore__ inline void ComputeTransLayout(uint16_t mSize, uint16_t scaleBlockN);
    __aicore__ inline void TransFp4Output(uint16_t mSize, uint16_t kSize);
    __aicore__ inline void PadScale(uint32_t nKRatio, uint32_t nkIdx, uint32_t blkM);

    static __simd_vf__ inline void PadScaleVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr,
                                              uint32_t nKRatio, uint16_t firstNum, uint16_t lastNum,
                                              uint16_t rowLoopNum);
    static __simd_vf__ inline void ZeroNormalizedPaddingVf(__ubuf__ uint8_t* scaleBlkAddr, uint16_t rows,
                                                           uint16_t paddingColumn);
    static __simd_callee__ inline void DataCopyByCount(__ubuf__ int8_t* dstAddr, __ubuf__ int8_t* srcAddr,
                                                       uint32_t count);

    __aicore__ inline void DrainWorkspace(uint32_t writebackEventId);
    __aicore__ inline void Quant(uint64_t offset, int64_t blkM, int64_t k);

    // Input READY/ACK use events 0 and 1. Event 2 orders all three private-workspace stores.
    static constexpr uint16_t OUTPUT_EVENT = 2;
    static constexpr uint32_t GROUPS_PER_BLOCK = NORMALIZED_COLS / GROUP_SIZE;
    static constexpr uint32_t VECTOR_ELEMENTS = 128;
    static constexpr uint32_t SCALE_ROW_BYTES = 32;
    static constexpr uint16_t ABS_MASK = 0x7fff;
    static constexpr uint16_t BF16_INF = 0x7f80;
    static constexpr AscendC::Reg::DivSpecificMode DUAL_LEVEL_DIV_MODE = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                          AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    static constexpr uint32_t FP32_MIN_NORMAL = 0x00800000;
    static constexpr float MAX_VALUE = AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ?
                                           6.0f :
                                           (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ? 448.0f : 57344.0f);
    static constexpr AscendC::Reg::CastTrait CAST_BF16_TO_FP32_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CAST_BF16_TO_FP32_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    static __simd_vf__ inline void CollectSliceVf(__ubuf__ bfloat16_t* source, __ubuf__ bfloat16_t* collect,
                                                  uint32_t rows, uint32_t inputPitch, uint32_t rotK)
    {
        AscendC::Reg::RegTensor<bfloat16_t> value;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < rotK; col += VECTOR_ELEMENTS) {
                uint32_t remaining = rotK - col;
                auto mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                AscendC::Reg::LoadAlign(value, source + row * inputPitch + col);
                AscendC::Reg::StoreAlign(collect + row * rotK + col, value, mask);
            }
        }
    }

    static __simd_vf__ inline void GatherVf(__ubuf__ bfloat16_t* collect, __ubuf__ bfloat16_t* input, uint32_t rows,
                                            uint32_t tileRows, uint32_t rotK, uint32_t validCols)
    {
        static_assert(!EnableLevel0_ || asc_get_vf_len() / sizeof(bfloat16_t) == VECTOR_ELEMENTS,
                      "The 512-element level0 implementation requires A5 256-byte vector registers.");
        AscendC::Reg::RegTensor<bfloat16_t> value;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t block = 0; block < NORMALIZED_COLS / rotK; ++block) {
                for (uint32_t column = 0; column < rotK; column += VECTOR_ELEMENTS) {
                    uint32_t remaining = rotK - column;
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    if (block * rotK < validCols) {
                        AscendC::Reg::LoadAlign(value, collect + (block * tileRows + row) * rotK + column);
                    } else {
                        AscendC::Reg::Duplicate(value, static_cast<bfloat16_t>(0));
                    }
                    AscendC::Reg::StoreAlign(input + row * NORMALIZED_COLS + block * rotK + column, value, mask);
                }
            }
        }
    }

    static __simd_vf__ inline void ClampVf(__ubuf__ bfloat16_t* input, __ubuf__ bfloat16_t* limits, uint32_t loopCount)
    {
        AscendC::Reg::RegTensor<bfloat16_t> value0, value1, limit, negLimit;
        AscendC::Reg::RegTensor<uint16_t> signMask;
        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(signMask, static_cast<uint16_t>(0x8000));
        for (uint32_t i = 0; i < loopCount; ++i) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(value0, value1, input,
                                                                             VECTOR_ELEMENTS * 2);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(limit, limits, 8);
            AscendC::Reg::Xor(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(negLimit),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(limit), signMask, mask);
            AscendC::Reg::Max(value0, value0, negLimit, mask);
            AscendC::Reg::Max(value1, value1, negLimit, mask);
            AscendC::Reg::Min(value0, value0, limit, mask);
            AscendC::Reg::Min(value1, value1, limit, mask);
            AscendC::Reg::Interleave(value0, value1, value0, value1);
            AscendC::Reg::StoreAlign(input - VECTOR_ELEMENTS * 2, value0, mask);
            AscendC::Reg::StoreAlign(input - VECTOR_ELEMENTS, value1, mask);
        }
    }

    static __simd_callee__ inline void FiniteAbs(AscendC::Reg::RegTensor<uint16_t>& result,
                                                 AscendC::Reg::RegTensor<bfloat16_t>& value,
                                                 AscendC::Reg::RegTensor<uint16_t>& absMask,
                                                 AscendC::Reg::RegTensor<uint16_t>& inf,
                                                 AscendC::Reg::RegTensor<uint16_t>& zero, AscendC::Reg::MaskReg& mask)
    {
        AscendC::Reg::MaskReg nonFinite;
        AscendC::Reg::And(result, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value), absMask, mask);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::GE>(nonFinite, result, inf, mask);
        AscendC::Reg::Select<uint16_t>(result, zero, result, nonFinite);
    }

    static __simd_callee__ inline void NormalizeVector(__ubuf__ bfloat16_t* dst,
                                                       AscendC::Reg::RegTensor<bfloat16_t>& value,
                                                       AscendC::Reg::RegTensor<float>& scale)
    {
        AscendC::Reg::RegTensor<float> fp32Zero, fp32One;
        AscendC::Reg::RegTensor<bfloat16_t> resultZero, resultOne, zero;
        AscendC::Reg::MaskReg mask16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg mask32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::Duplicate(zero, static_cast<bfloat16_t>(0));
        AscendC::Reg::Compare<bfloat16_t, AscendC::CMPMODE::EQ>(zeroMask, value, zero, mask16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ZERO>(fp32Zero, value, mask16);
        AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ONE>(fp32One, value, mask16);
        if constexpr (EnableLevel0_ && !AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            AscendC::Reg::Div<float, &DUAL_LEVEL_DIV_MODE>(fp32Zero, fp32Zero, scale, mask32);
            AscendC::Reg::Div<float, &DUAL_LEVEL_DIV_MODE>(fp32One, fp32One, scale, mask32);
        } else {
            AscendC::Reg::Div(fp32Zero, fp32Zero, scale, mask32);
            AscendC::Reg::Div(fp32One, fp32One, scale, mask32);
        }
        AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(resultZero, fp32Zero, mask32);
        AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(resultOne, fp32One, mask32);
        AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(
            reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(resultZero),
            reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(resultZero));
        AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(
            reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(resultOne),
            reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(resultOne));
        AscendC::Reg::Interleave(resultZero, resultOne, resultZero, resultOne);
        // Restore the original signed zero instead of retaining the result of 0/0.
        AscendC::Reg::Select<bfloat16_t>(resultZero, value, resultZero, zeroMask);
        AscendC::Reg::StoreAlign(dst, resultZero, mask16);
    }

    // Stage 2 for dual FP4 with a custom denominator. The FP32 quotient decides the E8M0 boundary.
    static __simd_vf__ inline void GenDualFp4CublasScaleVf(__ubuf__ uint16_t* maxima, __ubuf__ uint8_t* scales,
                                                           __ubuf__ uint16_t* reciprocals, uint32_t totalScale,
                                                           float dstTypeMax)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint32_t LANES = asc_get_vf_len() / sizeof(float);
        constexpr int16_t FP32_MANTISSA_BITS = 23;
        constexpr int16_t BF16_MANTISSA_BITS = 7;
        Reg::RegTensor<float> maximum, divisor;
        Reg::RegTensor<uint16_t> unpacked, packed;
        Reg::RegTensor<uint32_t> exponent, fraction, nextExponent, code, inverse;
        Reg::RegTensor<uint32_t> fractionMask, zero, nanCode, nanInverse, inverseBias;
        Reg::MaskReg finite, nonzero, roundUp, predicate;
        Reg::Duplicate(divisor, dstTypeMax);
        Reg::Duplicate(fractionMask, 0x007fffffU);
        Reg::Duplicate(zero, 0U);
        Reg::Duplicate(nanCode, 0xffU);
        Reg::Duplicate(nanInverse, 0x7f81U);
        Reg::Duplicate(inverseBias, 0x7f00U);

        for (uint32_t offset = 0; offset < totalScale; offset += LANES) {
            uint32_t remaining = totalScale - offset;
            uint32_t storeCount = remaining < LANES ? remaining : LANES;
            auto storeMask = Reg::UpdateMask<uint16_t>(storeCount);
            auto mask = Reg::UpdateMask<uint32_t>(remaining);
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>(unpacked, maxima + offset);
            Reg::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ZERO>(
                maximum, reinterpret_cast<Reg::RegTensor<bfloat16_t>&>(unpacked), mask);
            auto& bits = reinterpret_cast<Reg::RegTensor<uint32_t>&>(maximum);
            Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(finite, bits, 0x7f800000U, mask);
            Reg::CompareScalar<uint32_t, AscendC::CMPMODE::NE>(nonzero, bits, 0U, mask);
            Reg::Div<float, &DUAL_LEVEL_DIV_MODE>(maximum, maximum, divisor, mask);

            Reg::ShiftRights(exponent, bits, FP32_MANTISSA_BITS, mask);
            Reg::And(fraction, bits, fractionMask, mask);
            // FP4 rounds up only normalized, in-range quotients with a nonzero fraction.
            Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(roundUp, exponent, 0U, mask);
            Reg::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(predicate, exponent, 254U, mask);
            Reg::MaskAnd(roundUp, roundUp, predicate, mask);
            Reg::CompareScalar<uint32_t, AscendC::CMPMODE::NE>(predicate, fraction, 0U, mask);
            Reg::MaskAnd(roundUp, roundUp, predicate, mask);
            Reg::Adds(nextExponent, exponent, 1U, mask);
            Reg::Select(code, nextExponent, exponent, roundUp);
            Reg::Select(code, code, nanCode, finite);
            Reg::Select(code, code, zero, nonzero);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(packed, code);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                reinterpret_cast<__ubuf__ uint16_t*>(scales + offset), packed, storeMask);

            Reg::ShiftLefts(inverse, code, BF16_MANTISSA_BITS, mask);
            Reg::Sub(inverse, inverseBias, inverse, mask);
            Reg::Select(inverse, inverse, nanInverse, finite);
            Reg::Select(inverse, inverse, zero, nonzero);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(packed, inverse);
            Reg::StoreAlign<uint16_t>(reciprocals + offset, packed, storeMask);
        }
    }

    static __simd_vf__ inline void NormalizeVf(__ubuf__ bfloat16_t* input, __ubuf__ float* level0Scale, uint32_t rows)
    {
        AscendC::Reg::RegTensor<bfloat16_t> x0, x1, x2, x3;
        AscendC::Reg::RegTensor<uint16_t> abs0, abs1, abs2, abs3, absMask, inf, zero16;
        AscendC::Reg::RegTensor<uint32_t> minNormal, zero32;
        AscendC::Reg::RegTensor<float> scale;
        AscendC::Reg::MaskReg mask16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg mask32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg underflow;
        AscendC::Reg::Duplicate(absMask, ABS_MASK);
        AscendC::Reg::Duplicate(inf, BF16_INF);
        AscendC::Reg::Duplicate(zero16, static_cast<uint16_t>(0));
        AscendC::Reg::Duplicate(zero32, static_cast<uint32_t>(0));
        AscendC::Reg::Duplicate(minNormal, FP32_MIN_NORMAL);
        constexpr bool useFp8Quotient = EnableLevel0_ && !AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value;
        AscendC::Reg::RegTensor<float> divisor;
        if constexpr (useFp8Quotient) {
            AscendC::Reg::Duplicate(divisor, MAX_VALUE);
        }
        for (uint32_t row = 0; row < rows; ++row) {
            auto rowAddr = input + row * NORMALIZED_COLS;
            AscendC::Reg::LoadAlign(x0, rowAddr);
            AscendC::Reg::LoadAlign(x1, rowAddr + VECTOR_ELEMENTS);
            AscendC::Reg::LoadAlign(x2, rowAddr + VECTOR_ELEMENTS * 2);
            AscendC::Reg::LoadAlign(x3, rowAddr + VECTOR_ELEMENTS * 3);
            // Follow DynamicDualLevelMxQuant: only the amax statistic excludes Inf/NaN.
            FiniteAbs(abs0, x0, absMask, inf, zero16, mask16);
            FiniteAbs(abs1, x1, absMask, inf, zero16, mask16);
            FiniteAbs(abs2, x2, absMask, inf, zero16, mask16);
            FiniteAbs(abs3, x3, absMask, inf, zero16, mask16);
            AscendC::Reg::Max(abs0, abs0, abs1, mask16);
            AscendC::Reg::Max(abs2, abs2, abs3, mask16);
            AscendC::Reg::Max(abs0, abs0, abs2, mask16);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(abs0, abs0, mask16);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ZERO>(
                scale, reinterpret_cast<AscendC::Reg::RegTensor<bfloat16_t>&>(abs0), mask16);
            if constexpr (useFp8Quotient) {
                AscendC::Reg::Div<float, &DUAL_LEVEL_DIV_MODE>(scale, scale, divisor, mask32);
            } else {
                AscendC::Reg::Muls(scale, scale, 1.0f / MAX_VALUE, mask32);
            }
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(
                underflow, reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(scale), minNormal, mask32);
            AscendC::Reg::Select<float>(scale, reinterpret_cast<AscendC::Reg::RegTensor<float>&>(zero32), scale,
                                        underflow);
            auto scaleAddr = level0Scale + row * (SCALE_ROW_BYTES / sizeof(float));
            AscendC::Reg::UnalignRegForStore unalign;
            AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, scale, unalign,
                                                                                           1);
            AscendC::Reg::StoreUnAlignPost<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, unalign, 0);
            AscendC::Reg::Duplicate(scale, scale, mask32);
            // All 512 original values are already in registers before any in-place write.
            NormalizeVector(rowAddr, x0, scale);
            NormalizeVector(rowAddr + VECTOR_ELEMENTS, x1, scale);
            NormalizeVector(rowAddr + VECTOR_ELEMENTS * 2, x2, scale);
            NormalizeVector(rowAddr + VECTOR_ELEMENTS * 3, x3, scale);
        }
    }

    struct NoLevel0Params {};
    AscendC::Std::conditional_t<EnableLevel0_, Level0Params, NoLevel0Params> level0_{};

    // ---- Shape ----
    struct RotateQuantShapeInfo {
        int64_t m;
        int64_t n;
        int64_t b;
    };

    RotateQuantShapeInfo shape_;

    // ---- GM ----
    __gm__ int8_t* cGlobal_{nullptr};
    __gm__ int8_t* scaleGlobal_{nullptr};

    // ---- Quant Params ----
    bfloat16_t alpha_ = 0.0f;
    bool needClamp_ = false;
    int64_t axis_ = 0;
    int64_t roundMode_ = 0;
    int64_t scaleAlg_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t fpEmax_ = 0;
    uint16_t addValueBits_ = 0;
    uint32_t ubBaseOffset_ = 0;
    uint32_t writebackEventId_ = 0;
    bool deferWriteback_ = false;
    bool pendingWriteback_[WORKSPACE_COUNT] = {false, false};
    bool inputDoubleBuffered_{false};
    uint32_t ubSlotBytes_{AscendC::TOTAL_UB_SIZE};
    bool splitM_{false};
    uint64_t inputSequence_{0};
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::ComputeNormalizedTile(
    const RotateQuantTileContext& tile)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (tile.rows == 0 || tile.validCols == 0) {
        return;
    }

    // Every row has a complete 512-value physical stride, including a zero tail.
    // It is therefore safe to scan the tile linearly without mixing MX groups.
    AscendC::PipeBarrier<PIPE_V>();
    ComputeMxQuant(tile.rows * NORMALIZED_COLS);
    AscendC::PipeBarrier<PIPE_V>();
    ClearScaleBlock(tile.rows);
    AscendC::PipeBarrier<PIPE_V>();
    ComputeTransLayout(static_cast<uint16_t>(tile.rows), NORMALIZED_COLS / GROUP_SIZE);
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t scalesPerRow = Gemm::CeilDiv<int64_t>(shape_.n, GROUP_SIZE);
    const uint32_t validScales = tile.validCols / GROUP_SIZE;
    const bool padLastScale = tile.nStart + tile.validCols == shape_.n && (scalesPerRow & 1) != 0;
    if (padLastScale) {
        // A zero-valued padded MX group does not define the public padding byte.
        // Set the byte explicitly after transposing the full 16-scale source rows.
        asc_vf_call<ZeroNormalizedPaddingVf>(reinterpret_cast<__ubuf__ uint8_t*>(ubBaseOffset_ + SCALE_BLOCK_UB_OFFSET),
                                             static_cast<uint16_t>(tile.rows), static_cast<uint16_t>(validScales));
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::StoreNormalizedTile(const RotateQuantTileContext& tile)
{
    if (tile.rows == 0 || tile.validCols == 0) {
        return;
    }
    const uint32_t scalesPerRow = Gemm::CeilDiv<int64_t>(shape_.n, GROUP_SIZE);
    const uint32_t storedScalesPerRow = scalesPerRow + (scalesPerRow & 1);
    const uint32_t validScales = tile.validCols / GROUP_SIZE;
    const bool padLastScale = tile.nStart + tile.validCols == shape_.n && (scalesPerRow & 1) != 0;
    constexpr uint32_t ELEMENTS_PER_BYTE = AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ? 2 : 1;
    const uint64_t yOffset = (static_cast<uint64_t>(tile.mStart) * shape_.n + tile.nStart) / ELEMENTS_PER_BYTE;
    const uint64_t scaleOffset = static_cast<uint64_t>(tile.mStart) * storedScalesPerRow + tile.nStart / GROUP_SIZE;

    CopyUbToGm(cGlobal_ + yOffset, Y_UB_OFFSET, static_cast<uint16_t>(tile.rows), tile.validCols / ELEMENTS_PER_BYTE,
               NORMALIZED_COLS / ELEMENTS_PER_BYTE, static_cast<uint32_t>(shape_.n / ELEMENTS_PER_BYTE));
    CopyScaleFromUbToGm(scaleOffset, SCALE_BLOCK_UB_OFFSET, static_cast<uint16_t>(tile.rows),
                        validScales + static_cast<uint32_t>(padLastScale), storedScalesPerRow, SCALE_STORE_STRIDE);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                           EnableLevel0_>::BlockEpilogueRotateQuant()
{}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                           EnableLevel0_>::~BlockEpilogueRotateQuant()
{
    if ASCEND_IS_AIV {
        Drain();
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline auto
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::GetAccumulatorTensor(
    const AccumulatorTensorParams& params) const
{
    const int64_t alignN = Gemm::CeilAlign(params.cols, static_cast<int64_t>(Gemm::BLOCK_CUBE));
    // BlockMmad takes the slot-0 view and adds its ping-pong offset when issuing Fixpipe.
    const uint32_t offset = X_UB_OFFSET + params.slot * (AscendC::TOTAL_UB_SIZE / WORKSPACE_COUNT);
    return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(offset),
                                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(params.rows, alignN));
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::Init(Params const& params)
{
    if constexpr (EnableLevel0_) {
        level0_ = params.level0;
        const uint32_t base = level0_.workspaceBase;
        const bool validWorkspace = base % 32 == 0 &&
                                    static_cast<uint64_t>(base) + NORMALIZED_WORKSPACE_BYTES <= AscendC::TOTAL_UB_SIZE;
        ASCENDC_ASSERT(validWorkspace, { KERNEL_LOG(KERNEL_ERROR, "Invalid normalized MX workspace"); });
        ubBaseOffset_ = base;
        ConfigureMx(params.mx);
    } else {
        const uint32_t slotBytes = params.ubDB == 2 ? AscendC::TOTAL_UB_SIZE / 2 : AscendC::TOTAL_UB_SIZE;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        const bool validInput = (params.ubDB == 1 || params.ubDB == 2) && !params.splitM;
#else
        const bool validInput = params.ubDB == 1 || params.ubDB == 2;
#endif
        const bool validWorkspace = validInput && WorkspaceBytes(0) <= slotBytes;
        ASCENDC_ASSERT(validWorkspace, { KERNEL_LOG(KERNEL_ERROR, "Invalid RotateQuant epilogue workspace"); });
        SetWorkspace(0, 0);
        ConfigureMx(params);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::ConfigureMx(const MxParams& params)
{
    inputDoubleBuffered_ = params.ubDB == 2;
    ubSlotBytes_ = inputDoubleBuffered_ ? AscendC::TOTAL_UB_SIZE / 2 : AscendC::TOTAL_UB_SIZE;
    deferWriteback_ = inputDoubleBuffered_ && !EnableLevel0_;
    splitM_ = params.splitM;
    inputSequence_ = 0;
    InitGlobalBuffer(params);
    InitProblemShape(params.problemShape);
    InitQuantParams(params.alpha, !EnableLevel0_ && params.needClamp, params.axis, params.roundMode, params.scaleAlg,
                    params.dstTypeMax, params.invDstTypeMax);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline uint32_t
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::WaitForFixpipe() const
{
    const uint32_t slot = inputDoubleBuffered_ ? inputSequence_ & 1U : 0U;
    Gemm::Sync::WaitForCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(Gemm::Sync::FIXPIPE_AIC_READY_FLAG_BASE + slot);
    return slot;
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::ReleaseFixpipe(uint32_t slot, bool hasMte3Dependency)
{
    if (!hasMte3Dependency) {
        // Order Vector reads before ACK when no GM store already orders MTE3.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
    }
    Gemm::Sync::NotifyCube<Gemm::Sync::SYNC_MODE_INTRA, PIPE_MTE3>(Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot);
    ++inputSequence_;
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::SetWorkspace(uint32_t ubBaseOffset,
                                                                             uint32_t writebackEventId)
{
    // The consumer advances once for every Fixpipe chunk, including zero-row lanes.
    ubBaseOffset_ = ubBaseOffset;
    writebackEventId_ = writebackEventId;
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::Drain()
{
    for (uint32_t eventId = 0; eventId < WORKSPACE_COUNT; ++eventId) {
        DrainWorkspace(eventId);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::DrainWorkspace(uint32_t writebackEventId)
{
    if (pendingWriteback_[writebackEventId]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(writebackEventId);
        pendingWriteback_[writebackEventId] = false;
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::Quant(uint64_t offset, int64_t blkM, int64_t k)
{
    DrainWorkspace(writebackEventId_);
    const bool validWorkspace = blkM > 0 && k > 0 &&
                                static_cast<uint64_t>(blkM) * static_cast<uint64_t>(k) <= MN_SIZE &&
                                WorkspaceBytes(static_cast<uint32_t>(blkM)) <= ubSlotBytes_;
    ASCENDC_ASSERT(validWorkspace, { KERNEL_LOG(KERNEL_ERROR, "RotateQuant epilogue tile exceeds its workspace"); });
    uint32_t mnSize = blkM * k;
    uint64_t yOffset = offset * static_cast<uint64_t>(k);
    ComputeMxQuant(mnSize);

    uint32_t nScaleSize = Gemm::CeilDiv<int64_t>(shape_.n, GROUP_SIZE);
    uint32_t isOdd = nScaleSize & 1;
    nScaleSize += isOdd;
    if (shape_.b > 1) {
        uint64_t mIdx = offset / shape_.b;
        uint64_t bIdx = offset % shape_.b;
        uint64_t scaleOffset = mIdx * nScaleSize + Gemm::CeilDiv<uint64_t>(bIdx * k, GROUP_SIZE);
        uint16_t scaleBlockN = Gemm::CeilDiv<int64_t>(k, GROUP_SIZE);
        uint16_t dataSize = scaleBlockN + ((bIdx == shape_.b - 1) ? isOdd : 0);

        // CopyScaleFromUbToGm 按行距 SCALE_STORE_STRIDE(32B) 读取, 清零必须覆盖每行完整行距,
        // 否则最后一个 bIdx 的 pad 字节(行 m 的 m*32+1 处)读到未初始化 UB, 污染 scale 输出
        ClearScaleBlock(blkM);
        ComputeTransLayout(blkM, scaleBlockN);
        if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
            // A packed K=32 row occupies only 16 bytes. Give MTE3 a 32-byte UB row pitch.
            if (k < 64 && blkM > 1) {
                AscendC::PipeBarrier<PIPE_V>();
                TransFp4Output(blkM, k);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        CopyOutputFromUbToGm(yOffset, blkM, k, shape_.n);
        CopyScaleFromUbToGm(scaleOffset, SCALE_BLOCK_UB_OFFSET, blkM, dataSize, nScaleSize, SCALE_STORE_STRIDE);
    } else if (isOdd == 1) {
        uint32_t nKRatio = shape_.n / k;
        uint64_t mIdx = offset / nKRatio;
        uint32_t nkIdx = offset % nKRatio;
        uint64_t scaleOffset = mIdx * nScaleSize + nkIdx;
        ClearScaleBlock(blkM);
        PadScale(nKRatio, nkIdx, blkM);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        CopyOutputFromUbToGm(yOffset, 1, mnSize, 0);
        CopyPadScaleFromUbToGm(scaleOffset, nKRatio, nkIdx, blkM);
    } else {
        uint64_t scaleOffset = offset * Gemm::CeilDiv<int64_t>(k, GROUP_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(writebackEventId_);
        CopyOutputFromUbToGm(yOffset, 1, mnSize, 0);
        uint32_t scaleSize = Gemm::CeilDiv<uint32_t>(mnSize, GROUP_SIZE);
        CopyScaleFromUbToGm(scaleOffset, SCALE_UB_OFFSET, 1, scaleSize, 0, scaleSize);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(writebackEventId_);
    if (deferWriteback_) {
        // The other workspace can execute Vector work while this slot is written to GM.
        pendingWriteback_[writebackEventId_] = true;
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(writebackEventId_);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::operator()(const BlockShape& blockShape,
                                                                           const BlockCoord& blockCoord)
{
    const int64_t fullM = asc::te::get<Gemm::MNK_M>(blockShape);
    const int64_t rowsPerAiv = splitM_ ? Gemm::CeilDiv(fullM, int64_t{2}) : fullM;
    const int64_t rowStart = static_cast<int64_t>(AscendC::GetSubBlockIdx()) * rowsPerAiv;
    const int64_t localRows = Gemm::Max(int64_t{0}, Gemm::Min(rowsPerAiv, fullM - rowStart));
    if constexpr (EnableLevel0_) {
        const RotateQuantTileContext tile{asc::te::get<Gemm::MNK_M>(blockCoord) + rowStart,
                                          asc::te::get<Gemm::MNK_N>(blockCoord), static_cast<uint32_t>(localRows),
                                          static_cast<uint32_t>(asc::te::get<Gemm::MNK_N>(blockShape))};
        RunLevel0Tile(tile, level0_);
    } else {
        const uint32_t slot = WaitForFixpipe();
        SetWorkspace(slot * ubSlotBytes_, slot);
        DrainWorkspace(slot);
        if (localRows > 0) {
            const int64_t k = asc::te::get<Gemm::MNK_N>(blockShape);
            const uint64_t offset = (asc::te::get<Gemm::MNK_M>(blockCoord) + rowStart) * shape_.b +
                                    asc::te::get<Gemm::MNK_B>(blockCoord);
            Quant(offset, localRows, k);
        }
        ReleaseFixpipe(slot, localRows > 0);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::InitGlobalBuffer(MxParams const& params)
{
    cGlobal_ = reinterpret_cast<__gm__ int8_t*>(params.outGmAddr);
    scaleGlobal_ = reinterpret_cast<__gm__ int8_t*>(params.scaleGmAddr);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::InitProblemShape(const ProblemShape& problemShape)
{
    shape_.m = asc::te::get<Gemm::MNK_M>(problemShape);
    shape_.n = asc::te::get<Gemm::MNK_N>(problemShape);
    shape_.b = asc::te::get<Gemm::MNK_B>(problemShape);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::InitQuantParams(bfloat16_t alpha, bool needClamp,
                                                                                int64_t axis, int64_t roundMode,
                                                                                int64_t scaleAlg, float dstTypeMax,
                                                                                float invDstTypeMax)
{
    alpha_ = alpha;
    needClamp_ = needClamp;
    axis_ = axis;
    roundMode_ = roundMode;
    scaleAlg_ = scaleAlg;
    dstTypeMax_ = dstTypeMax;
    invDstTypeMax_ = invDstTypeMax;

    // 进位附加值规则统一为 gelu_tanh 语义（DYN 路径使用）
    addValueBits_ = dstTypeMax_ == SEVEN_FLOAT ? ADD_VALUE_FOR_BF16_MAN2 : ADD_VALUE_FOR_BF16_MAN1;

    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        // FP8 时 host 强制 dstTypeMax=0（下刷的 invDstTypeMax 为 0），cuBLAS 路径按输出 dtype 取 1/dstTypeMax
        invDstTypeMax_ = 1.0f / 448.0f;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        invDstTypeMax_ = 1.0f / 57344.0f;
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::CopyUbToGm(__gm__ int8_t* dst, uint32_t srcUbOffset,
                                                                           uint16_t rows, uint32_t columns,
                                                                           uint32_t srcRowPitch, uint32_t dstRowPitch)
{
    auto ubLayout = Gemm::MakeNDExtLayout<int8_t>(rows, columns, srcRowPitch);
    auto gmLayout = Gemm::MakeNDExtLayout<int8_t>(rows, columns, dstRowPitch);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + srcUbOffset),
                                    ubLayout);
    auto output = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(dst), gmLayout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), output, src);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::CopyOutputFromUbToGm(
    uint64_t offset, uint16_t blockCount, uint32_t dataSize, uint32_t dstStride)
{
    uint32_t srcUbOffset = Y_UB_OFFSET;
    uint32_t srcStride = dataSize;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        offset >>= 1;
        dataSize >>= 1;
        dstStride >>= 1;
        srcStride = dataSize;
        if (blockCount > 1 && dataSize < SCALE_STORE_STRIDE) {
            srcUbOffset = X_UB_OFFSET;
            srcStride = SCALE_STORE_STRIDE;
        }
    }
    CopyUbToGm(cGlobal_ + offset, srcUbOffset, blockCount, dataSize, srcStride, dstStride == 0 ? dataSize : dstStride);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void
BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_, EnableLevel0_>::CopyScaleFromUbToGm(
    uint64_t offset, uint32_t srcUbOffset, uint16_t blockCount, uint32_t dataSize, uint32_t dstStride,
    uint32_t srcStride)
{
    CopyUbToGm(scaleGlobal_ + offset, srcUbOffset, blockCount, dataSize, srcStride,
               dstStride == 0 ? dataSize : dstStride);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::CopyPadScaleFromUbToGm(uint64_t offset,
                                                                                       uint32_t nKRatio, uint32_t nkIdx,
                                                                                       uint32_t blkM)
{
    uint16_t firstNum = (nKRatio - nkIdx <= blkM) ? nKRatio - nkIdx : blkM;
    uint16_t rowLoopNum = (firstNum < blkM) ? (blkM - firstNum) / nKRatio : 0;
    uint16_t lastNum = (firstNum < blkM) ? (blkM - firstNum) % nKRatio : 0;

    uint32_t dataSize = firstNum + ((nkIdx + firstNum == nKRatio) ? 1 : 0);
    uint32_t srcOffset = SCALE_BLOCK_UB_OFFSET;
    CopyScaleFromUbToGm(offset, srcOffset, 1, dataSize, dataSize, dataSize);
    offset += dataSize;
    srcOffset += Gemm::CeilAlign<uint32_t>(firstNum + 1, SCALE_STORE_STRIDE);

    if (rowLoopNum > 0) {
        dataSize = nKRatio + 1;
        uint32_t srcStride = Gemm::CeilAlign<uint32_t>(dataSize, SCALE_STORE_STRIDE);
        CopyScaleFromUbToGm(offset, srcOffset, rowLoopNum, dataSize, dataSize, srcStride);
        offset += rowLoopNum * dataSize;
        srcOffset += rowLoopNum * srcStride;
    }

    if (lastNum > 0) {
        CopyScaleFromUbToGm(offset, srcOffset, 1, lastNum, lastNum, lastNum);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::ComputeMxQuant(uint32_t totalDataInUB)
{
    const uint32_t totalScale = Gemm::CeilDiv<uint32_t>(totalDataInUB, GROUP_SIZE);
    auto dataLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalDataInUB),
                                                    static_cast<int64_t>(totalDataInUB));
    auto scaleLayout = Gemm::MakeNDExtLayout<int8_t>(1, static_cast<int64_t>(totalScale),
                                                     static_cast<int64_t>(totalScale));
    auto srcTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(ubBaseOffset_ + X_UB_OFFSET), dataLayout);
    auto quantYTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + Y_UB_OFFSET), dataLayout);
    auto limitTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(ubBaseOffset_ + LIMIT_UB_OFFSET), scaleLayout);
    auto maxExpTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(ubBaseOffset_ + MAX_EXP_UB_OFFSET), scaleLayout);
    auto yScaleTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_UB_OFFSET), scaleLayout);
    auto reciprocalTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(ubBaseOffset_ + RECIPROCAL_UB_OFFSET), scaleLayout);

    // MxQuantConfig 由 Init 阶段解析的普通成员在调用点组装（tile 类型不得出现在类作用域）
    // zeroScaleOnZeroExp=false: rotate/gelu_tanh 阵营（sharedExp 判定，仅清零 reciprocal）
    Tile::MxQuantConfig cfg{Tile::MxScaleAlg::OCP, fpEmax_, invDstTypeMax_, addValueBits_, false};
    // The original RotateQuant only rounds subnormal scales up for FP8.
    cfg.roundSubnormalScaleUp = !AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value;
    if (scaleAlg_ == 1) { // per-channel cuBLAS（仅 FP8）
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
    } else if (scaleAlg_ == 0) {
        cfg.alg = Tile::MxScaleAlg::OCP;
    } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
        cfg.alg = Tile::MxScaleAlg::DYN_DTYPE_RANGE;
    } else { // FP4 cuBLAS: dstTypeMax in [6, 12], excluding 6 and 7.
        cfg.alg = Tile::MxScaleAlg::CUBLAS;
    }

    Tile::MxQuant<DataTypeOut> mx;
    if (needClamp_) {
        // 钳位链路: limit = absMax * alpha；useExp: OCP 取 limit 指数字段，其余取 limit 全位
        mx.GroupMaxExp(srcTensor, maxExpTensor, limitTensor, totalDataInUB, alpha_, scaleAlg_ == 0);
    } else {
        // useAbs: OCP 取指数字段，cuBLAS/Dyn 取绝对值位
        mx.GroupMaxExp(srcTensor, maxExpTensor, totalDataInUB, scaleAlg_ != 0);
    }
    if constexpr (EnableLevel0_) {
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (EnableLevel0_ && AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        cfg.zeroScaleOnZeroExp = cfg.alg == Tile::MxScaleAlg::OCP;
        if (cfg.alg == Tile::MxScaleAlg::CUBLAS) {
            if ASCEND_IS_AIV {
                asc_vf_call<GenDualFp4CublasScaleVf>(
                    reinterpret_cast<__ubuf__ uint16_t*>(maxExpTensor.data().get()),
                    reinterpret_cast<__ubuf__ uint8_t*>(yScaleTensor.data().get()),
                    reinterpret_cast<__ubuf__ uint16_t*>(reciprocalTensor.data().get()), totalScale, dstTypeMax_);
            }
        } else {
            mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
        }
    } else {
        mx.GenScale(maxExpTensor, yScaleTensor, reciprocalTensor, cfg, totalScale);
    }
    if constexpr (EnableLevel0_) {
        AscendC::PipeBarrier<PIPE_V>();
    }

    Tile::MxQuantFp4RoundMode fp4RoundMode = Tile::MxQuantFp4RoundMode::RINT;
    if (roundMode_ == MODE_FLOOR) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::FLOOR;
    } else if (roundMode_ == MODE_ROUND) {
        fp4RoundMode = Tile::MxQuantFp4RoundMode::ROUND;
    }
    if (needClamp_) {
        mx.Quantize(srcTensor, reciprocalTensor, limitTensor, quantYTensor, totalDataInUB, fp4RoundMode);
    } else {
        mx.Quantize(srcTensor, reciprocalTensor, quantYTensor, totalDataInUB, fp4RoundMode);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::ClearScaleBlock(uint32_t blkM)
{
    uint32_t count = blkM * SCALE_STORE_STRIDE;
    auto layout = Gemm::MakeNDExtLayout<int8_t>(1, count, count);
    auto scaleBlock = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_BLOCK_UB_OFFSET), layout);
    Gemm::Tile::FillUb<int8_t>::FillWithValue(scaleBlock, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::ComputeTransLayout(uint16_t mSize, uint16_t scaleBlockN)
{
    // scale 布局: (M, scaleBlockN) -> (M, 32B 行块)，供 MTE 按行距 SCALE_STORE_STRIDE 搬运
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize), static_cast<int64_t>(scaleBlockN),
                                                   static_cast<int64_t>(scaleBlockN));
    auto srcTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_UB_OFFSET), srcLayout);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(static_cast<int64_t>(mSize),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE),
                                                   static_cast<int64_t>(AscendC::ONE_BLK_SIZE));
    auto dstTensor = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_BLOCK_UB_OFFSET), dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransScaleLayout(srcTensor, dstTensor, mSize, scaleBlockN);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::TransFp4Output(uint16_t mSize, uint16_t kSize)
{
    uint32_t packedBytes = kSize / 2;
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, packedBytes, Gemm::Align16(packedBytes));
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(mSize, packedBytes, Gemm::Align32(packedBytes));
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + Y_UB_OFFSET),
                                    srcLayout);
    auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + X_UB_OFFSET),
                                    dstLayout);
    Tile::MxQuant<DataTypeOut> mx;
    mx.TransFp4OutLayout(src, dst, mSize, kSize);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::PadScale(uint32_t nKRatio, uint32_t nkIdx,
                                                                         uint32_t blkM)
{
    if ASCEND_IS_AIC {
        return;
    }
    auto srcLayout = Gemm::MakeNDExtLayout<int8_t>(1, blkM, blkM);
    auto dstLayout = Gemm::MakeNDExtLayout<int8_t>(1, blkM * SCALE_STORE_STRIDE, blkM * SCALE_STORE_STRIDE);
    auto src = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_UB_OFFSET), srcLayout);
    auto dst = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + SCALE_BLOCK_UB_OFFSET), dstLayout);
    uint16_t firstNum = (nKRatio - nkIdx <= blkM) ? nKRatio - nkIdx : blkM;
    uint16_t rowLoopNum = (firstNum < blkM) ? (blkM - firstNum) / nKRatio : 0;
    uint16_t lastNum = (firstNum < blkM) ? (blkM - firstNum) % nKRatio : 0;
    asc_vf_call<PadScaleVf>(reinterpret_cast<__ubuf__ int8_t*>(src.data().get()),
                            reinterpret_cast<__ubuf__ int8_t*>(dst.data().get()), nKRatio, firstNum, lastNum,
                            rowLoopNum);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__simd_vf__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                 EnableLevel0_>::PadScaleVf(__ubuf__ int8_t* scaleAddr,
                                                                            __ubuf__ int8_t* scaleBlkAddr,
                                                                            uint32_t nKRatio, uint16_t firstNum,
                                                                            uint16_t lastNum, uint16_t rowLoopNum)
{
    // ClearScaleBlock provides the zero padding; each staged row starts at a 32-byte boundary.
    DataCopyByCount(scaleBlkAddr, scaleAddr, firstNum);
    scaleAddr += firstNum;
    scaleBlkAddr += (firstNum + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;

    for (uint16_t mIdx = 0; mIdx < rowLoopNum; ++mIdx) {
        DataCopyByCount(scaleBlkAddr, scaleAddr, nKRatio);
        scaleAddr += nKRatio;
        scaleBlkAddr += (nKRatio + SCALE_STORE_STRIDE) / SCALE_STORE_STRIDE * SCALE_STORE_STRIDE;
    }

    if (lastNum > 0) {
        DataCopyByCount(scaleBlkAddr, scaleAddr, lastNum);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__simd_callee__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                     EnableLevel0_>::DataCopyByCount(__ubuf__ int8_t* dstAddr,
                                                                                     __ubuf__ int8_t* srcAddr,
                                                                                     uint32_t count)
{
    constexpr uint32_t VF_LEN_8 = asc_get_vf_len() / sizeof(uint8_t);
    uint16_t loopNum = (count + VF_LEN_8 - 1) / VF_LEN_8;
    AscendC::Reg::RegTensor<int8_t> value;
    AscendC::Reg::UnalignReg unalign;
    for (uint16_t i = 0; i < loopNum; ++i) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<int8_t>(count);
        AscendC::Reg::LoadUnAlignPre(unalign, srcAddr + i * VF_LEN_8);
        AscendC::Reg::LoadUnAlign(value, unalign, srcAddr + i * VF_LEN_8);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstAddr + i * VF_LEN_8, value, mask);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__simd_vf__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                 EnableLevel0_>::ZeroNormalizedPaddingVf(__ubuf__ uint8_t* scaleBlkAddr,
                                                                                         uint16_t rows,
                                                                                         uint16_t paddingColumn)
{
    AscendC::Reg::RegTensor<uint8_t> zero;
    AscendC::Reg::Duplicate(zero, 0);
    for (uint16_t row = 0; row < rows; ++row) {
        auto rowAddr = scaleBlkAddr + static_cast<uint32_t>(row) * SCALE_STORE_STRIDE + paddingColumn;
        AscendC::Reg::UnalignRegForStore unalign;
        AscendC::Reg::StoreUnAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(rowAddr, zero, unalign, 1);
        AscendC::Reg::StoreUnAlignPost<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(rowAddr, unalign, 0);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::RunLevel0Tile(const RotateQuantTileContext& tile,
                                                                              const Level0Params& level0)
{
    // Both AIVs consume one credit per K-wide slice, including the zero-row lane.
    const uint32_t slices = tile.validCols / level0.rotK;
    for (uint32_t slice = 0; slice < slices; ++slice) {
        const uint32_t slot = WaitForFixpipe();
        if (tile.rows != 0) {
            auto src = reinterpret_cast<__ubuf__ bfloat16_t*>(slot * ubSlotBytes_);
            auto dst = reinterpret_cast<__ubuf__ bfloat16_t*>(ubBaseOffset_ + level0.collectUbOffset) +
                       slice * level0.tileRows * level0.rotK;
            asc_vf_call<CollectSliceVf>(src, dst, tile.rows, level0.inputPitch, level0.rotK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        ReleaseFixpipe(slot, false);
    }
    if (tile.rows == 0 || tile.validCols == 0) {
        return;
    }

    NormalizeTile(tile, level0);
    ComputeNormalizedTile(tile);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
    StoreLevel0Scale(tile, level0);
    StoreNormalizedTile(tile);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_EVENT);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::NormalizeTile(const RotateQuantTileContext& tile,
                                                                              const Level0Params& level0)
{
    auto collectLayout = Gemm::MakeNDExtLayout<bfloat16_t>(1, level0.tileRows * NORMALIZED_COLS,
                                                           level0.tileRows * NORMALIZED_COLS);
    auto collect = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(ubBaseOffset_ + level0.collectUbOffset),
        collectLayout);
    auto inputLayout = Gemm::MakeNDExtLayout<bfloat16_t>(tile.rows, NORMALIZED_COLS, NORMALIZED_COLS);
    auto input = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(ubBaseOffset_ + X_UB_OFFSET), inputLayout);
    auto scaleLayout = Gemm::MakeNDExtLayout<float>(tile.rows, 1, SCALE_ROW_BYTES / sizeof(float));
    auto scale = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, float>(ubBaseOffset_ + level0.level0ScaleUbOffset), scaleLayout);
    auto inputAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(input.data().get());

    // The collect allocation includes a full vector read guard for K=32/64 loads.
    asc_vf_call<GatherVf>(reinterpret_cast<__ubuf__ bfloat16_t*>(collect.data().get()), inputAddr, tile.rows,
                          level0.tileRows, level0.rotK, tile.validCols);
    AscendC::PipeBarrier<PIPE_V>();

    if (level0.needClamp) {
        auto groupLayout = Gemm::MakeNDExtLayout<int8_t>(1, tile.rows * GROUPS_PER_BLOCK, tile.rows * GROUPS_PER_BLOCK);
        auto limit = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(ubBaseOffset_ + LIMIT_UB_OFFSET), groupLayout);
        auto maxExp = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, uint16_t>(ubBaseOffset_ + MAX_EXP_UB_OFFSET), groupLayout);
        Tile::MxQuant<DataTypeOut> mx;
        // Only the BF16 limits are consumed here; MX computes its own exponents after level0.
        mx.GroupMaxExp(input, maxExp, limit, tile.rows * NORMALIZED_COLS, level0.alpha, false);
        AscendC::PipeBarrier<PIPE_V>();
        asc_vf_call<ClampVf>(inputAddr, reinterpret_cast<__ubuf__ bfloat16_t*>(limit.data().get()), tile.rows * 2);
        AscendC::PipeBarrier<PIPE_V>();
    }

    asc_vf_call<NormalizeVf>(inputAddr, reinterpret_cast<__ubuf__ float*>(scale.data().get()), tile.rows);
    // MX re-reads the BF16 normalization output from the same UB region.
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_, bool EnableLevel0_>
__aicore__ inline void BlockEpilogueRotateQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_,
                                                EnableLevel0_>::StoreLevel0Scale(const RotateQuantTileContext& tile,
                                                                                 const Level0Params& level0)
{
    const uint64_t gmOffset = (static_cast<uint64_t>(tile.mStart) * level0.scalesPerRow +
                               tile.nStart / NORMALIZED_COLS) *
                              sizeof(float);
    auto ubLayout = Gemm::MakeNDExtLayout<int8_t>(tile.rows, sizeof(float), SCALE_ROW_BYTES);
    auto gmLayout = Gemm::MakeNDExtLayout<int8_t>(tile.rows, sizeof(float), level0.scalesPerRow * sizeof(float));
    auto src = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubBaseOffset_ + level0.level0ScaleUbOffset), ubLayout);
    auto dst = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int8_t*>(level0.scaleGmAddr) + gmOffset),
        gmLayout);
    asc::te::copy(asc::te::make_copy(asc::te::copy_ub_to_gm{}), dst, src);
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
