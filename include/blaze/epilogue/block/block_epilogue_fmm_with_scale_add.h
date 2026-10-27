/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_fmm_with_scale_add.h
 * \brief AIV epilogue for out = alpha * acc + beta * x3.
 *
 * cuBLAS / torch addmm contract for beta == 0: "If beta == 0, then C does not
 * have to be a valid input" -- the x3 (C) read and the beta * x3 add are
 * skipped entirely (NOT computed as 0 * x3), so NaN/Inf in x3 never reaches
 * the output; out = alpha * acc. This also saves the x3 GM->UB traffic.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {
namespace Detail {

#ifdef __NPU_ARCH__
constexpr AscendC::Reg::CastTrait FMM_WITH_SCALE_ADD_B16_TO_FP32_ZERO = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait FMM_WITH_SCALE_ADD_B16_TO_FP32_ONE = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait FMM_WITH_SCALE_ADD_FP32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#endif // __NPU_ARCH__

constexpr float FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE = 1.0F;

template <typename ElementType_>
struct FmmWithScaleAddVfParams {
    __ubuf__ float* accAddr{nullptr};
    __ubuf__ ElementType_* x3Addr{nullptr};
    uint32_t rowsThisStage{0};
    uint32_t curN{0};
    uint32_t nAlignAcc{0};
    uint32_t nAlignElement{0};
    uint32_t vlF32{0};
    uint32_t vfLoops{0};
    float alpha{FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
    float beta{FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
};

} // namespace Detail

template <class DispatchPolicy_, class ElementType_>
class BlockEpilogueFmmWithScaleAdd {
public:
    using DispatchPolicy = DispatchPolicy_;
    using L0CDataType = float;
    using X3Type = ElementType_;
    using OutputType = ElementType_;
    using ComputeType = float;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR x3GmAddr{nullptr};
        GM_ADDR outputGmAddr{nullptr};
        float alpha{Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
        float beta{Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
        ProblemShape problemShape{};
        bool splitM{false};
        int64_t baseM{0};
        int64_t baseN{0};
    };

    __aicore__ inline BlockEpilogueFmmWithScaleAdd()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(X3_SKIP_EVENT);
        }
    }

    __aicore__ inline ~BlockEpilogueFmmWithScaleAdd()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(X3_SKIP_EVENT);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIV {
            x3GmAddr_ = params.x3GmAddr;
            outputGmAddr_ = params.outputGmAddr;
            // Alpha and beta are scalar attributes copied into tilingData by the host.
            alpha_ = params.alpha;
            beta_ = params.beta;
            hasAlphaScale_ = alpha_ != Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE;
            hasBetaScale_ = beta_ != Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE;
            betaZero_ = beta_ == 0.0F;
            m_ = asc::te::get<Blaze::Gemm::MNK_M>(params.problemShape);
            n_ = asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape);
            splitM_ = params.splitM;
            baseM_ = params.baseM;
            baseN_ = params.baseN;
        }
    }

    template <typename TensorC>
    __aicore__ inline void operator()(TensorC& ubTensor, const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        const int64_t coordM = asc::te::get<Blaze::Gemm::MNK_M>(blockCoord);
        const int64_t coordN = asc::te::get<Blaze::Gemm::MNK_N>(blockCoord);
        const int64_t batchIdx = asc::te::get<Blaze::Gemm::MNK_B>(blockCoord);
        const int64_t dstOffset = (batchIdx * m_ + coordM) * n_ + coordN;
        (*this)(ubTensor, blockShape, dstOffset, splitM_, baseM_, baseN_);
    }

    template <typename TensorC>
    __aicore__ inline void operator()(TensorC& ubTensor, const BlockShape& blockShape, int64_t dstOffset, bool splitM,
                                      int64_t baseM, int64_t baseN, int64_t stagingBaseOffset = -1,
                                      int64_t stagingBytes = -1, bool isDiagonal = false)
    {
        int64_t curM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
        if (baseM != 0) {
            curM = Blaze::Gemm::Min(curM, baseM);
        }
        const int64_t halfM = Blaze::Gemm::CeilDiv(curM, static_cast<int64_t>(AscendC::GetTaskRation()));
        int64_t localRows = curM;
        int64_t localRowOffset = 0;
        if (splitM) {
            const int64_t subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
            localRows = (static_cast<uint64_t>(curM) & 1UL) > 0UL ? halfM - subBlockIdx : halfM;
            localRowOffset = subBlockIdx * halfM;
        }
        // Fixpipe pads M before DUAL_DST_SPLIT_M. Both AIVs therefore reserve halfM physical rows even when the
        // second AIV has one fewer valid row for an odd M.
        const int64_t accumulatorRows = splitM ? halfM : Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN);
        const int64_t nL1 = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        const int64_t curBaseN = baseN != 0 ? Blaze::Gemm::Min(nL1, baseN) : nL1;
        const int64_t nL1Iter = Blaze::Gemm::CeilDiv(nL1, curBaseN);

        for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
            const int64_t tileN = nIdx + 1 == nL1Iter ? nL1 - curBaseN * nIdx : curBaseN;
            if (localRows <= 0) {
                // No vector work is issued for this AIV. Keep the ready/free handshake on PIPE_MTE3 so that the
                // free flag cannot bypass a ready wait queued on another pipeline.
                Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
                continue;
            }
            Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
            ProcessTile(ubTensor, localRows, accumulatorRows, tileN, dstOffset + nIdx * curBaseN + localRowOffset * n_,
                        stagingBaseOffset, stagingBytes, isDiagonal);
            Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }

private:
    static constexpr int64_t SPLIT_M_ALIGN = 2;
    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t ACC_ALIGN = DATA_BLOCK / sizeof(float);
    static constexpr uint32_t ELEMENT_ALIGN = DATA_BLOCK / sizeof(ElementType_);
    static constexpr uint16_t X3_EVENT = 0;
    static constexpr uint16_t OUTPUT_EVENT = 1;
    static constexpr uint16_t BUFFER_REUSE_EVENT = 2;
    static constexpr uint16_t X3_SKIP_EVENT = 3;
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    using VfParams = Detail::FmmWithScaleAddVfParams<ElementType_>;

    template <typename TensorC>
    __aicore__ inline void ProcessTile(TensorC& ubTensor, int64_t localRows, int64_t accumulatorRows, int64_t tileN,
                                       int64_t tileGmOffset, int64_t stagingBaseOffset, int64_t stagingBytes,
                                       bool isDiagonal = false)
    {
        if (localRows <= 0 || tileN <= 0) {
            return;
        }
        const uint64_t nAlignAcc = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN),
                                                          static_cast<uint64_t>(ACC_ALIGN));
        const uint64_t nAlignElement = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN),
                                                              static_cast<uint64_t>(ELEMENT_ALIGN));
        const uint64_t accumulatorBytes = Blaze::Gemm::CeilAlign(
            static_cast<uint64_t>(accumulatorRows) * nAlignAcc * sizeof(float), static_cast<uint64_t>(DATA_BLOCK));
        if (accumulatorBytes >= AscendC::TOTAL_UB_SIZE) {
            return;
        }
        // Place x3 immediately after the accumulator. The output reuses the x3 buffer in place.
        // Callers that host several accumulator images in UB pass an explicit staging
        // window; the default places staging right after this tile's accumulator.
        const uint64_t x3OutputBufferBytes = stagingBytes >= 0 ? static_cast<uint64_t>(stagingBytes) :
                                                                 AscendC::TOTAL_UB_SIZE - accumulatorBytes;
        const uint64_t stageRowBytes = nAlignElement * sizeof(ElementType_);
        const int64_t stageRows = static_cast<int64_t>(
            Blaze::Gemm::Min(static_cast<uint64_t>(localRows), x3OutputBufferBytes / stageRowBytes));
        if (stageRows <= 0) {
            return;
        }
        const uint32_t vlF32 = AscendC::GetVecLen() / sizeof(float);
        const uint32_t vfLoops = static_cast<uint32_t>(
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(tileN), static_cast<uint64_t>(vlF32)));

        for (int64_t stageOffset = 0; stageOffset < localRows; stageOffset += stageRows) {
            const int64_t rowsThisStage = Blaze::Gemm::Min(stageRows, localRows - stageOffset);
            const int64_t gmElemOffset = tileGmOffset + stageOffset * n_;
            const auto origin = asc::te::make_coord(0L, 0L);
            const auto validShape = asc::te::make_shape(rowsThisStage, tileN);

            auto x3UbStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, ElementType_>(
                    stagingBaseOffset >= 0 ? static_cast<uint64_t>(stagingBaseOffset) : accumulatorBytes),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage,
                                                                       static_cast<int64_t>(nAlignElement)));
            auto x3Ub = x3UbStorage.slice(origin, validShape);
            auto x3GmStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType_*>(x3GmAddr_) +
                                                             gmElemOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, n_));
            auto x3Gm = x3GmStorage.slice(origin, validShape);

            if (!betaZero_) {
                // x3 and output share the same UB region. Wait for the previous UB2GM before overwriting it.
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
                auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
                asc::te::copy(copyGmToUb, x3Ub, x3Gm);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(X3_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(X3_EVENT);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(X3_SKIP_EVENT);
            }

            auto accAddr = reinterpret_cast<__ubuf__ float*>(ubTensor.data().get()) + stageOffset * nAlignAcc;
            VfParams vfParams{accAddr,
                              reinterpret_cast<__ubuf__ ElementType_*>(x3Ub.data().get()),
                              static_cast<uint32_t>(rowsThisStage),
                              static_cast<uint32_t>(tileN),
                              static_cast<uint32_t>(nAlignAcc),
                              static_cast<uint32_t>(nAlignElement),
                              vlF32,
                              vfLoops,
                              alpha_,
                              beta_};
            if (betaZero_) {
                // cuBLAS semantics: out = alpha * acc, the x3 (C) addend is neither loaded nor added.
                if (hasAlphaScale_) {
                    asc_vf_call<FmmScaleAddVf<true, false, true>>(vfParams);
                } else {
                    asc_vf_call<FmmScaleAddVf<false, false, true>>(vfParams);
                }
            } else if (hasAlphaScale_ && hasBetaScale_) {
                asc_vf_call<FmmScaleAddVf<true, true>>(vfParams);
            } else if (hasAlphaScale_) {
                asc_vf_call<FmmScaleAddVf<true, false>>(vfParams);
            } else if (hasBetaScale_) {
                asc_vf_call<FmmScaleAddVf<false, true>>(vfParams);
            } else {
                asc_vf_call<FmmScaleAddVf<false, false>>(vfParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);

            auto outputGmStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType_*>(outputGmAddr_) +
                                                             gmElemOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, n_));
            auto outputGm = outputGmStorage.slice(origin, validShape);
            auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
            if (!isDiagonal) {
                asc::te::copy(copyUbToGm, outputGm, x3Ub);
            }
            if (betaZero_) {
                // Release the staging region for the next vector write (no MTE2
                // load in between, so the MTE3 -> V event carries the ordering).
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(X3_SKIP_EVENT);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
            }
        }
    }

    template <bool hasAlphaScale, bool hasBetaScale, bool skipX3 = false>
    static __simd_vf__ inline void FmmScaleAddVf(const VfParams params)
    {
        AscendC::Reg::MaskReg allB16 = AscendC::Reg::CreateMask<ElementType_, AscendC::Reg::MaskPattern::ALL>();
        for (uint32_t row = 0; row < params.rowsThisStage; ++row) {
            __ubuf__ float* accRow = params.accAddr + row * params.nAlignAcc;
            __ubuf__ ElementType_* x3Row = params.x3Addr + row * params.nAlignElement;
            __ubuf__ ElementType_* outRow = params.x3Addr + row * params.nAlignElement;

            for (uint32_t i = 0; i < params.vfLoops; ++i) {
                const uint32_t col = i * params.vlF32;
                const uint32_t remain = params.curN - col;
                uint32_t valid = remain < params.vlF32 ? remain : params.vlF32;
                AscendC::Reg::MaskReg maskF32 = AscendC::Reg::UpdateMask<float>(valid);

                AscendC::Reg::RegTensor<float> acc;
                AscendC::Reg::RegTensor<ElementType_> x3Packed;
                AscendC::Reg::RegTensor<float> x3Zero;
                AscendC::Reg::RegTensor<float> x3One;
                AscendC::Reg::RegTensor<float> outF32;
                AscendC::Reg::RegTensor<ElementType_> outB16;

                AscendC::Reg::DataCopy(acc, accRow + col);
                if constexpr (!skipX3) {
                    if constexpr (AscendC::IsSameType<ElementType_, float>::value) {
                        AscendC::Reg::DataCopy(x3Zero, x3Row + col);
                    } else {
                        AscendC::Reg::DataCopy(x3Packed, x3Row + col);
                        AscendC::Reg::Cast<float, ElementType_, Detail::FMM_WITH_SCALE_ADD_B16_TO_FP32_ZERO>(
                            x3Zero, x3Packed, maskF32);
                        AscendC::Reg::Cast<float, ElementType_, Detail::FMM_WITH_SCALE_ADD_B16_TO_FP32_ONE>(
                            x3One, x3Packed, allB16);
                        AscendC::Reg::Interleave(x3Zero, x3One, x3Zero, x3One);
                    }
                }

                if constexpr (skipX3) {
                    if constexpr (hasAlphaScale) {
                        AscendC::Reg::Muls(outF32, acc, params.alpha, maskF32);
                    } else {
                        AscendC::Reg::Muls(outF32, acc, 1.0F, maskF32);
                    }
                } else if constexpr (hasAlphaScale && hasBetaScale) {
                    AscendC::Reg::RegTensor<float> accScaled;
                    AscendC::Reg::RegTensor<float> x3Scaled;
                    AscendC::Reg::Muls(accScaled, acc, params.alpha, maskF32);
                    AscendC::Reg::Muls(x3Scaled, x3Zero, params.beta, maskF32);
                    AscendC::Reg::Add(outF32, accScaled, x3Scaled, maskF32);
                } else if constexpr (hasAlphaScale) {
                    AscendC::Reg::RegTensor<float> accScaled;
                    AscendC::Reg::Muls(accScaled, acc, params.alpha, maskF32);
                    AscendC::Reg::Add(outF32, accScaled, x3Zero, maskF32);
                } else if constexpr (hasBetaScale) {
                    AscendC::Reg::RegTensor<float> x3Scaled;
                    AscendC::Reg::Muls(x3Scaled, x3Zero, params.beta, maskF32);
                    AscendC::Reg::Add(outF32, acc, x3Scaled, maskF32);
                } else {
                    AscendC::Reg::Add(outF32, acc, x3Zero, maskF32);
                }

                if constexpr (AscendC::IsSameType<ElementType_, float>::value) {
                    AscendC::Reg::DataCopy<ElementType_, AscendC::Reg::StoreDist::DIST_NORM_B32>(outRow + col, outF32,
                                                                                                 maskF32);
                } else {
                    AscendC::Reg::Cast<ElementType_, float, Detail::FMM_WITH_SCALE_ADD_FP32_TO_B16>(outB16, outF32,
                                                                                                    maskF32);
                    AscendC::Reg::DataCopy<ElementType_, AscendC::Reg::StoreDist::DIST_PACK_B32>(outRow + col, outB16,
                                                                                                 maskF32);
                }
            }
        }
    }

    GM_ADDR x3GmAddr_{nullptr};
    GM_ADDR outputGmAddr_{nullptr};
    int64_t m_{0};
    int64_t n_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    bool splitM_{false};
    float alpha_{Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
    float beta_{Detail::FMM_WITH_SCALE_ADD_DEFAULT_SCALE_VALUE};
    bool hasAlphaScale_{false};
    bool hasBetaScale_{false};
    bool betaZero_{false};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
