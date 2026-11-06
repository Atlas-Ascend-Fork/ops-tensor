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
 * \file kernel_wqgmm_mx_mix_finalize_routing.h
 * \brief TensorAPI kernel for GroupedMatmulFinalizeRouting MX A8W4 weight quant
 *        (FP8 E4M3FN activation x FP4 E2M1 WeightNZ). AIV runs the weight dequant
 *        prologue (FP4 -> FP8 + bias x(1/64) pre-scale -> L1),
 *        the output init (y zero-fill + shared input), and the finalize-routing epilogue
 *        (x64 logit muls + row-index scatter with atomic add). AIC runs the MX MMAD
 *        and ScaleB GM-to-L1 copy, then fixpipes L0C to UB with
 *        split-M for the vector epilogue.
 *
 * W8 conversion and split-M output reuse the original UB arena in separate
 * phases. AIV preloads two K strips before releasing the previous result; AIC
 * waits only before fixpipe, allowing the next MMAD to overlap the epilogue.
 */

#pragma once

#include <cstdint>
#include "blaze/gemm/utils/sync.h"

#include "kernel_vec_intf.h"
#include "tensor_api/tensor.h"

#include "blaze/epilogue/tile/compute.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

#define GMM_FR_WQ_PROLOGUE_TEMPLATE_PARAM template <typename OutType_, typename InType_, typename BiasType_>

/*!
 * \brief AIV weight dequant prologue for the A8W4 finalize-routing kernel.
 *
 * Converts packed FP4 weight tiles to FP8 with the fused bias pre-scale (x 1/64, so the
 * epilogue's x64 restores the bias magnitude) and signals the paired AIC per K tile.
 * W8 and fixpipe reuse one UB arena. Before consuming the previous result, AIV
 * preloads the next two B strips; the remaining strips follow its epilogue.
 */
GMM_FR_WQ_PROLOGUE_TEMPLATE_PARAM
class GmmWqgmmFrWeightPrologue {
private:
    static constexpr uint64_t GMM_FR_WQ_WEIGHT_READY_FLAG = 1;
    static constexpr TEventID GMM_FR_WQ_EV_V_TO_MTE2 = 0;
    static constexpr TEventID GMM_FR_WQ_EV_MTE3_TO_V = 0;
    static constexpr TEventID GMM_FR_WQ_EV_MTE2_TO_V = 0;
    static constexpr uint64_t GMM_FR_WQ_UB_KB = 1024;
    static constexpr uint64_t GMM_FR_WQ_VEC_REG_BYTES = 256;
    // Match the low-level UB arena. W8 and split-M results share [80KB, 240KB)
    // in separate phases; the VCV handoff protects the transition.
    static constexpr uint64_t MTE2_SLOT_NUM = 4;
    static constexpr uint64_t W8_SLOT_NUM = 4;
    static constexpr uint64_t W4_TOTAL_SIZE = 80 * GMM_FR_WQ_UB_KB;
    static constexpr uint64_t W4_SLOT_SIZE = W4_TOTAL_SIZE / MTE2_SLOT_NUM;
    static constexpr uint64_t W8_BASE_OFFSET = W4_TOTAL_SIZE;
    static constexpr uint64_t W8_TOTAL_SIZE = 160 * GMM_FR_WQ_UB_KB;
    static constexpr uint64_t W8_ROW_PITCH = GMM_FR_WQ_VEC_REG_BYTES * W8_SLOT_NUM;
    static constexpr uint64_t BIAS_IN_OFFSET = W8_BASE_OFFSET + W8_TOTAL_SIZE;
    static constexpr uint64_t BIAS_SLOT_SIZE = 256;
    static constexpr uint64_t BIAS_OUT_OFFSET = BIAS_IN_OFFSET + MTE2_SLOT_NUM * BIAS_SLOT_SIZE;
    static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128;
    static constexpr uint64_t L1_WEIGHT_OFFSETS[2] = {0, 384 * GMM_FR_WQ_UB_KB};
    static constexpr uint64_t BIAS_L1_OFFSETS[2] = {64 * GMM_FR_WQ_UB_KB, 380 * GMM_FR_WQ_UB_KB};
    static constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = 4 * GMM_FR_WQ_UB_KB;

public:
    using OutType = OutType_;
    using InType = InType_;
    using BiasType = BiasType_;
    static_assert(AscendC::Std::is_same_v<OutType, fp8_e4m3fn_t>, "Weight conversion output must be FP8 E4M3FN");
    static_assert(AscendC::Std::is_one_of_v<InType, fp4x2_e2m1_t, fp4x2_e1m2_t>,
                  "Weight conversion input must be packed FP4");
    static_assert(AscendC::Std::is_same_v<BiasType, bfloat16_t>, "Weight conversion bias must be BF16");

    struct Params {
        __gm__ InType* ptrB{nullptr};
    };

    __aicore__ inline explicit GmmWqgmmFrWeightPrologue(bool hasBias = false) : hasBias_(hasBias)
    {
        for (uint16_t idx = 0; idx < MTE2_SLOT_NUM; ++idx) {
            SetFlag<HardEvent::V_MTE2>(GMM_FR_WQ_EV_V_TO_MTE2 + idx);
        }
        for (uint16_t idx = 0; idx < W8_SLOT_NUM; ++idx) {
            SetFlag<HardEvent::MTE3_V>(GMM_FR_WQ_EV_MTE3_TO_V + idx);
        }
    }

    __aicore__ inline ~GmmWqgmmFrWeightPrologue()
    {
        // Up to two AIC L1 credits may still be outstanding when the tile list ends.
        const uint64_t outstandingAicCredits = Min(cvLoopIdx_, static_cast<uint64_t>(2));
        for (uint64_t idx = 0; idx < outstandingAicCredits; ++idx) {
            WaitAicToAiv();
        }
        for (uint16_t idx = 0; idx < W8_SLOT_NUM; ++idx) {
            WaitFlag<HardEvent::MTE3_V>(GMM_FR_WQ_EV_MTE3_TO_V + idx);
        }
        for (uint16_t idx = 0; idx < MTE2_SLOT_NUM; ++idx) {
            WaitFlag<HardEvent::V_MTE2>(GMM_FR_WQ_EV_V_TO_MTE2 + idx);
        }
    }

    template <typename GMWeightTensorType_, typename GMBiasTensorType_, typename DrainPrevious_>
    __aicore__ inline void operator()(const GMWeightTensorType_& gmWeightTensor, const GMBiasTensorType_& gmBiasTensor,
                                      uint64_t kSize, uint64_t kbL1Size, uint64_t nL1Size, uint64_t nOffset,
                                      uint64_t nAlign, bool isWeightL2Disable, DrainPrevious_&& drainPrevious)
    {
        const uint64_t kMte2BaseSize = kbL1Size / 2;
        const uint64_t l1SplitOffset = GetSubBlockIdx() * kMte2BaseSize;
        const uint64_t nL1Align = CeilAlign(nL1Size, static_cast<uint64_t>(BLOCK_CUBE));

        if (kSize == 0) {
            return;
        }
        uint64_t kOffset = 0;
        do {
            const uint64_t l1RealLen = (kOffset + kbL1Size) > kSize ? kSize - kOffset : kbL1Size;
            const uint64_t mte2RealK = GetSubBlockIdx() == 0 ?
                                           Min(kMte2BaseSize, l1RealLen) :
                                           (l1RealLen > kMte2BaseSize ? l1RealLen - kMte2BaseSize : 0);
            if (kOffset == 2 * kbL1Size) {
                drainPrevious();
            }
            uint64_t biasNSize = 0;
            uint64_t biasNOffset = 0;
            if (hasBias_ && kOffset == 0) {
                const uint64_t vec0NSize = Min(nL1Size, MX_BIAS_SINGLE_VECTOR_SIZE);
                biasNSize = GetSubBlockIdx() == 0 ? vec0NSize : nL1Size - vec0NSize;
                biasNOffset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
            }
            const bool processBias = hasBias_ && kOffset == 0 && biasNSize != 0;

            WaitVectorToMte2();
            CopyGmToUb(kOffset + l1SplitOffset, mte2RealK, nL1Size, nOffset, gmWeightTensor, gmBiasTensor, biasNOffset,
                       biasNSize, processBias, isWeightL2Disable);
            // The paired AIC keeps two L1 weight tiles in flight.
            if (cvLoopIdx_ > 1) {
                WaitAicToAiv();
            }
            WeightAntiQuantComputeNzNk(mte2RealK, nL1Size, biasNSize, processBias);
            SetVectorToMte2();

            CopyWeightToL1(mte2RealK, nL1Size, nL1Align, l1SplitOffset, processBias, biasNOffset, biasNSize);
            SetAivToAic();
            kOffset += kbL1Size;
            cvLoopIdx_++;
        } while (kOffset < kSize);
        if (kSize <= 2 * kbL1Size) {
            drainPrevious();
        }
    }

private:
    __aicore__ inline void SetAivToAic()
    {
        Sync::NotifyCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>(GMM_FR_WQ_WEIGHT_READY_FLAG);
    }

    __aicore__ inline void WaitAicToAiv()
    {
        Sync::WaitForCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>(GMM_FR_WQ_AIC_FREE_FLAG);
    }

    __aicore__ inline void WaitVectorToMte2()
    {
        WaitFlag<HardEvent::V_MTE2>(GMM_FR_WQ_EV_V_TO_MTE2 + (mte2LoopIdx_ & (MTE2_SLOT_NUM - 1)));
    }

    __aicore__ inline void SetVectorToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(GMM_FR_WQ_EV_V_TO_MTE2 + (mte2LoopIdx_ & (MTE2_SLOT_NUM - 1)));
        mte2LoopIdx_++;
    }

    template <typename GMWeightTensorType_, typename GMBiasTensorType_>
    __aicore__ inline void CopyGmToUb(uint64_t kGmOffset, uint64_t mte2RealK, uint64_t nL1Size, uint64_t nOffset,
                                      const GMWeightTensorType_& gmWeightTensor, const GMBiasTensorType_& gmBiasTensor,
                                      uint64_t biasNOffset, uint64_t biasNSize, bool processBias,
                                      bool isWeightL2Disable)
    {
        const uint64_t w4Slot = mte2LoopIdx_ & (MTE2_SLOT_NUM - 1);
        if (mte2RealK > 0) {
            auto gmSliceTensor = gmWeightTensor.slice(asc::te::make_coord(kGmOffset, nOffset),
                                                      asc::te::make_shape(mte2RealK, nL1Size));
            if (isWeightL2Disable) {
                gmSliceTensor.set_l2_cache_hint(asc::te::cache_mode::disable);
            }
            const auto w4Layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                             AscendC::Std::Int<asc::te::c0_element<OutType>>>(
                static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size));
            auto w4Tensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, InType>(W4_SLOT_SIZE * w4Slot), w4Layout);
            auto copyGM2UBWeight = asc::te::make_copy(Tile::CopyGM2UBWeight{});
            asc::te::copy(copyGM2UBWeight, w4Tensor, gmSliceTensor);
        }
        if (processBias) {
            auto gmBiasSlice = gmBiasTensor.slice(asc::te::make_coord(0, nOffset + biasNOffset),
                                                  asc::te::make_shape(1, biasNSize));
            const auto biasLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1, CeilAlign(biasNSize, static_cast<uint64_t>(BLOCK_CUBE)));
            auto biasTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(BIAS_IN_OFFSET + w4Slot * BIAS_SLOT_SIZE),
                biasLayout);
            auto copyGM2UBBias = asc::te::make_copy(asc::te::copy_gm_to_ub{});
            asc::te::copy(copyGM2UBBias, biasTensor, gmBiasSlice);
        }
        SetFlag<HardEvent::MTE2_V>(GMM_FR_WQ_EV_MTE2_TO_V);
        WaitFlag<HardEvent::MTE2_V>(GMM_FR_WQ_EV_MTE2_TO_V);
    }

    __aicore__ inline auto MakeWeight8BitTensor(uint64_t mte2RealK, uint64_t nL1Size)
    {
        const uint64_t slotOffset = W8_BASE_OFFSET + (computeLoopIdx_ & (W8_SLOT_NUM - 1)) * GMM_FR_WQ_VEC_REG_BYTES;
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, OutType>(slotOffset),
            Weight8BitZnToZnUBLayout<OutType>{}(static_cast<int64_t>(mte2RealK), static_cast<int64_t>(Align16(nL1Size)),
                                                W8_ROW_PITCH));
    }

    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t mte2RealK, uint64_t nL1Size, uint64_t biasNSize,
                                                      bool processBias)
    {
        WaitFlag<HardEvent::MTE3_V>(GMM_FR_WQ_EV_MTE3_TO_V + (computeLoopIdx_ & (W8_SLOT_NUM - 1)));
        const uint64_t w4Slot = mte2LoopIdx_ & (MTE2_SLOT_NUM - 1);
        const auto w4Layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                         AscendC::Std::Int<asc::te::c0_element<OutType>>>(
            static_cast<int64_t>(mte2RealK), static_cast<int64_t>(nL1Size));
        auto w4Tensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, InType>(W4_SLOT_SIZE * w4Slot), w4Layout);
        auto w8Tensor = MakeWeight8BitTensor(mte2RealK, nL1Size);
        const auto biasLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, Align16(biasNSize));
        auto biasInTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(BIAS_IN_OFFSET + w4Slot * BIAS_SLOT_SIZE),
            biasLayout);
        auto biasOutTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(
                BIAS_OUT_OFFSET + (computeLoopIdx_ & (W8_SLOT_NUM - 1)) * BIAS_SLOT_SIZE),
            biasLayout);
        Tile::ShiftW4ToW8<OutType, InType>(w4Tensor, w8Tensor, biasInTensor, biasOutTensor, processBias);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void CopyWeightToL1(uint64_t mte2RealK, uint64_t nL1Size, uint64_t nL1Align,
                                          uint64_t l1SplitOffset, bool processBias, uint64_t biasNOffset,
                                          uint64_t biasNSize)
    {
        if (likely(mte2RealK > 0)) {
            auto l1BaseLayout = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
                                                           asc::te::layout_trait_default<OutType>>(mte2RealK, nL1Size);
            auto l1BaseTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, OutType>(
                                                         L1_WEIGHT_OFFSETS[cvLoopIdx_ & 1] + l1SplitOffset * nL1Align),
                                                     l1BaseLayout);
            auto w8Tensor = MakeWeight8BitTensor(mte2RealK, nL1Size);
            auto copyUB2L1 = asc::te::make_copy(Tile::CopyUB2L1Weight8Bit{});
            asc::te::copy(copyUB2L1, l1BaseTensor, w8Tensor);
        }
        if (processBias) {
            const auto biasL1Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1, CeilAlign(biasNSize, static_cast<uint64_t>(BLOCK_CUBE)));
            auto biasL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(
                                                   BIAS_L1_OFFSETS[cvLoopIdx_ & 1] + biasNOffset * sizeof(BiasType)),
                                               biasL1Layout);
            auto biasOutTensor = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, BiasType>(
                    BIAS_OUT_OFFSET + (computeLoopIdx_ & (W8_SLOT_NUM - 1)) * BIAS_SLOT_SIZE),
                biasL1Layout);
            auto copyUB2L1Bias = asc::te::make_copy(asc::te::copy_ub_to_l1{});
            asc::te::copy(copyUB2L1Bias, biasL1, biasOutTensor);
        }
        SetFlag<HardEvent::MTE3_V>(GMM_FR_WQ_EV_MTE3_TO_V + (computeLoopIdx_ & (W8_SLOT_NUM - 1)));
        computeLoopIdx_++;
    }

    static constexpr uint64_t GMM_FR_WQ_AIC_FREE_FLAG = 0;

    bool hasBias_{false};
    uint64_t cvLoopIdx_{0};
    uint64_t mte2LoopIdx_{0};
    uint64_t computeLoopIdx_{0};
};

#undef GMM_FR_WQ_PROLOGUE_TEMPLATE_PARAM

/*!
 * \brief Writes the scaled shared input to y. The caller must zero y before invocation.
 */
class GmmWqgmmFrOutputInit {
private:
    static constexpr uint64_t GMM_FR_WQ_UB_KB = 1024;

    static constexpr uint64_t SHARED_INPUT_BUFFER_BYTES = 16 * GMM_FR_WQ_UB_KB;

public:
    struct Params {
        __gm__ float* ptrY{nullptr};
        __gm__ bfloat16_t* ptrSharedInput{nullptr};
        uint64_t nSize{0};
        uint64_t sharedInputOffset{0};
        uint64_t sharedInputLen{0};
        float sharedInputWeight{0.0F};
    };

    __aicore__ inline GmmWqgmmFrOutputInit() {}

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        const uint64_t sharedStart = params.sharedInputOffset * params.nSize;
        const uint64_t sharedSize = params.sharedInputLen * params.nSize;
        InitSharedInput(params, sharedStart, sharedSize);
    }

private:
    __aicore__ inline void InitSharedInput(const Params& params, uint64_t sharedStart, uint64_t sharedSize)
    {
        if (sharedSize == 0) {
            return;
        }
        const uint64_t workerNum = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetTaskRation();
        const uint64_t workerIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t chunkElems = SHARED_INPUT_BUFFER_BYTES / sizeof(float);
        using ProcessSharedInput = Blaze::Epilogue::Tile::ProcessSharedInput<float, false>;
        for (uint64_t offset = workerIdx * chunkElems; offset < sharedSize; offset += workerNum * chunkElems) {
            const uint64_t curElems = Min(chunkElems, sharedSize - offset);
            const auto layout = asc::te::make_layout(asc::te::make_shape(curElems), asc::te::make_stride(1L));
            const auto sharedInput = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(params.ptrSharedInput + offset), layout);
            const auto sharedOutput = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(params.ptrY + sharedStart + offset), layout);
            ProcessSharedInput::Run(sharedInput, sharedOutput, params.sharedInputWeight,
                                    {SHARED_INPUT_BUFFER_BYTES, SHARED_INPUT_BUFFER_BYTES, 0, 0, chunkElems});
        }
    }
};

/*!
 * \brief Group-level orchestrator for the MX A8W4 finalize-routing path.
 *
 * Scheduling (mirrors the legacy GMMFRWeightQuantResplitController): each group's M axis
 * is split around base tiles of 256 rows, the N axis walks 256-wide main blocks, and
 * tiles are assigned round-robin to cube cores with a rotating start core between
 * groups. AIC: MMAD + fixpipe. AIV: weight prologue (deferred epilogue), output init.
 */
template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockPrologue_, class BlockEpilogue_,
          class BlockOutputInit_>
class GmmWqgmmMxFinalizeRoutingKernel {
private:
    // Preserve the low-level M partition; output temporarily reuses the W8 arena.
    static constexpr uint64_t GMM_FR_WQ_M_L1 = 256;
    static constexpr uint64_t L1_K_BASE = 256;
    static constexpr uint64_t L1_K_EXTENDED = 512;
    static constexpr uint64_t L0_K_BASE = 128;
    static constexpr uint64_t L0_K_EXTENDED = 256;
    static constexpr uint64_t EXTENDED_L1_K_MAX_M_WITH_BIAS = 240;
    static constexpr uint64_t NARROW_TILE_N = 128;
    static constexpr uint64_t SHORT_TILE_M = 128;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128 * 1024;
    static constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = 4 * 1024;
    static constexpr uint64_t GMM_FR_WQ_FIXPIPE_READY_FLAG = 2;
    static constexpr uint64_t GMM_FR_WQ_OUTPUT_FREE_FLAG = 3;
    static constexpr uint64_t GMM_FR_WQ_UB_KB = 1024;

public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockPrologue = BlockPrologue_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockOutputInit = BlockOutputInit_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    static_assert(AscendC::Std::is_same_v<AType, fp8_e4m3fn_t>, "AType must be fp8_e4m3fn_t");
    static_assert(AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>,
                  "BType must be fp4x2_e2m1_t or fp4x2_e1m2_t");
    static_assert(AscendC::Std::is_same_v<ScaleAType, fp8_e8m0_t> && AscendC::Std::is_same_v<ScaleBType, fp8_e8m0_t>,
                  "ScaleAType and ScaleBType must be fp8_e8m0_t");
    static_assert(AscendC::Std::is_same_v<CType, float>, "A8W4 finalize routing requires float MMAD output.");
    static_assert(AscendC::Std::is_same_v<typename BlockMmad::LayoutA, asc::te::nd_ext_layout_ptn>,
                  "A8W4 finalize routing requires ND activation");
    static_assert(AscendC::Std::is_same_v<typename BlockMmad::LayoutB, asc::te::zn_layout_ptn>,
                  "A8W4 finalize routing requires transposed WeightNZ");
    static_assert(AscendC::Std::is_same_v<BiasType, bfloat16_t>, "A8W4 finalize routing requires BF16 bias");

    struct Params {
        ProblemShape problemShape; // {k, n, groupNum}
        typename BlockMmad::Params mmad;
        typename BlockScheduler::Params scheduler;
        typename BlockPrologue::Params prologue;
        typename BlockEpilogue::Params epilogue;
        typename BlockOutputInit::Params outputInit;
        GM_ADDR ptrGroupList;
        uint32_t groupListType;
        uint32_t hasBias;
    };

    __aicore__ inline GmmWqgmmMxFinalizeRoutingKernel() = default;
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint64_t K_ALIGNMENT = 64;

    template <typename GroupListTensor_>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor_& groupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = groupList[groupIdx];
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = groupList[groupIdx];
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void ReleaseOutputArena() const
    {
        Sync::NotifyCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>(GMM_FR_WQ_OUTPUT_FREE_FLAG);
    }

    __aicore__ inline void SetFixpipeReady() const
    {
        // Mode 4 addresses each AIV sub-block separately from the AIC side.
        Sync::NotifyVector<Sync::SYNC_MODE_INTRA, PIPE_FIX>(GMM_FR_WQ_FIXPIPE_READY_FLAG, true);
    }

    __aicore__ inline void WaitFixpipeReady() const
    {
        Sync::WaitForCube<Sync::SYNC_MODE_INTRA, PIPE_V>(GMM_FR_WQ_FIXPIPE_READY_FLAG);
    }

    __aicore__ inline static typename BlockMmad::Params MakeMmadParams(bool hasBias)
    {
        // 初始 L1/L0 几何和 Scale 窗口长度，沿用原 GMM 的配置上限。
        constexpr uint64_t L1_N = 256;
        constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;

        // 保留原 GMM 的 L1 分区及常量命名，地址偏移和缓冲容量均以字节为单位。
        constexpr uint64_t SCALE_AL1_OFFSET = 64 * 1024;
        constexpr uint64_t SCALE_BL1_OFFSET = 96 * 1024;
        constexpr uint64_t A_L1_OFFSET = 128 * 1024;
        constexpr uint64_t L1_BUF_OFFSET = 384 * 1024;
        constexpr uint64_t A_L1_BUF_OFFSET = 128 * 1024;
        constexpr uint64_t B_L1_SINGLE_BUF_SIZE = 64 * 1024;
        constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64 * 1024, 380 * 1024};
        constexpr uint16_t SYNC_AIV_AIC_FLAG = 0;
        constexpr uint16_t SYNC_AIC_AIV_FLAG = 1;

        typename BlockMmad::Params p{};
        // 初始化存储规划所需的几何参数；实际 K 分块由 GMM Kernel 随当前 tile 传入。
        p.l1TileShape = {GMM_FR_WQ_M_L1, L1_N, L1_K_BASE, MX_SCALE_K_L1_SIZE};
        p.l0TileShape = {GMM_FR_WQ_M_L1, L1_N, L0_K_BASE};
        p.l1BufferNum = DOUBLE_BUFFER_COUNT;
        p.hasBias = hasBias;
        p.aBufferBytes = A_L1_SINGLE_BUF_SIZE - (hasBias ? BIAS_L1_SINGLE_BUF_SIZE : 0);
        p.bBufferBytes = B_L1_SINGLE_BUF_SIZE;
        // 与配套 GMM AIV 使用相同的固定 L1 地址及 ready/free 标志。
        p.explicitL1Offsets = true;
        p.readyFlag = SYNC_AIC_AIV_FLAG;
        p.freeFlag = SYNC_AIV_AIC_FLAG;
        // GMM AIV 初始已有两个可写 B 槽位，此处再次发送许可会使同一槽位被重复授予。
        p.seedWeightCredits = false;
        // Bias0 位于第 0 组 Scale/A 分区之前，因此该组视图整体后移 4 KiB。
        // Bias1 占用第 1 个 A 槽位的尾部；两个 A 槽位都预留 Bias 空间，但只有第 0 组需要后移。
        for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; ++i) {
            uint64_t biasShift = hasBias && i == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0;
            p.weightOffsets[i] = i * L1_BUF_OFFSET;
            p.biasOffsets[i] = BIAS_L1_OFFSETS[i];
            p.aOffsets[i] = A_L1_OFFSET + i * A_L1_BUF_OFFSET + biasShift;
            p.scaleAOffsets[i] = SCALE_AL1_OFFSET + i * L1_BUF_OFFSET + biasShift;
            p.scaleBOffsets[i] = SCALE_BL1_OFFSET + i * L1_BUF_OFFSET + biasShift;
        }
        return p;
    }

    __aicore__ inline void Run(const Params& params)
    {
        groupListType_ = params.groupListType;
        preOffset_ = 0;
        const uint64_t kSize = static_cast<uint64_t>(AscendC::Std::get<0>(params.problemShape));
        const uint64_t nSize = static_cast<uint64_t>(AscendC::Std::get<1>(params.problemShape));
        const uint64_t groupNum = static_cast<uint64_t>(AscendC::Std::get<2>(params.problemShape));
        const uint64_t scaleKSize = CeilDiv(kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT);
        const uint64_t scaleKGmSize = kAlignSize / MXFP_GROUP_SIZE_LOCAL;

        BlockScheduler scheduler(params.scheduler);
        BlockEpilogue blockEpilogue;
        BlockMmad blockMmad{MakeMmadParams(params.hasBias != 0)};
        BlockOutputInit outputInit{};

        if ASCEND_IS_AIV {
            blockEpilogue.Init(params.epilogue);
            outputInit(params.outputInit);
        }
        AscendC::SyncAll();
        // Seed weight-buffer events only after output initialization has drained its DMA events.
        BlockPrologue blockPrologue(params.hasBias != 0);

        const uint64_t groupListSize = groupNum;
        auto groupListLayout = asc::te::make_layout(asc::te::make_shape(static_cast<int64_t>(groupListSize)),
                                                    asc::te::make_stride(1L));
        auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.ptrGroupList)),
            groupListLayout);
        using MakeLayoutA = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>;
        using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn,
                                                              AscendC::Std::Int<SCALE_C0>>;
        using MakeLayoutB = asc::te::frame_layout_format<asc::te::zn_layout_ptn,
                                                         AscendC::Std::Int<asc::te::c0_element<AType>>>;
        using MakeLayoutScaleB = asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn,
                                                              AscendC::Std::Int<SCALE_C0>>;
        using MakeLayoutBias = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>;

        // Deferred epilogue state: the tile whose fixpipe result is pending on this AIV.
        bool hasPendingTile = false;
        uint64_t pendingM = 0;
        uint64_t pendingN = 0;
        uint64_t pendingMOffset = 0;
        uint64_t pendingNOffset = 0;
        uint64_t pendingGroupM = 0;

        for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
            const int64_t groupM = GetSplitValueFromGroupList(gmGroupList, groupIdx);
            const uint64_t mSize = static_cast<uint64_t>(groupM);
            if (mSize == 0 || nSize == 0) {
                continue;
            }
            const uint64_t groupMOffset = static_cast<uint64_t>(preOffset_ - groupM);
            scheduler.UpdateNextProblem(asc::te::make_shape(mSize, nSize, kSize));
            typename BlockScheduler::BlockCoord blockCoord;
            __gm__ BType* groupWeightGm = reinterpret_cast<__gm__ BType*>(params.prologue.ptrB) +
                                          static_cast<uint64_t>(groupIdx) * ((nSize * kSize) >> 1U);
            __gm__ ScaleBType* groupScaleBGm = reinterpret_cast<__gm__ ScaleBType*>(params.mmad.scaleBGmAddr) +
                                               static_cast<uint64_t>(groupIdx) * nSize * scaleKSize;
            __gm__ BiasType* groupBiasGm = reinterpret_cast<__gm__ BiasType*>(params.mmad.biasGmAddr) +
                                           static_cast<uint64_t>(groupIdx) * nSize;
            if (params.hasBias == 0) {
                groupBiasGm = reinterpret_cast<__gm__ BiasType*>(params.prologue.ptrB);
            }
            const uint64_t nAlign = CeilAlign(nSize, static_cast<uint64_t>(BLOCK_CUBE));
            const auto weightLayout = MakeLayoutB{}(static_cast<int64_t>(kSize), static_cast<int64_t>(nAlign));
            const auto scaleBLayout = MakeLayoutScaleB{}(static_cast<int64_t>(scaleKGmSize),
                                                         static_cast<int64_t>(nSize));
            const auto biasLayout = MakeLayoutBias{}(static_cast<int64_t>(1), static_cast<int64_t>(nSize));
            auto weightGmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupWeightGm),
                                                       weightLayout);
            auto scaleBGmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupScaleBGm),
                                                       scaleBLayout);
            auto biasGmTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(groupBiasGm),
                                                     biasLayout);
            constexpr uint64_t FP4_CACHE_LINE_ELEMENT_NUM = 256;
            const bool isWeightCacheLineAligned = kSize % FP4_CACHE_LINE_ELEMENT_NUM == 0;
            const bool disableWeightL2 = mSize <= GMM_FR_WQ_M_L1 && isWeightCacheLineAligned;

            if (!scheduler.GetTileIdx(blockCoord)) {
                continue;
            }
            do {
                const auto blockShape = scheduler.GetBlockShape(blockCoord);
                const uint64_t mOffset = static_cast<uint64_t>(AscendC::Std::get<0>(blockCoord));
                const uint64_t nOffset = static_cast<uint64_t>(AscendC::Std::get<1>(blockCoord));
                const uint64_t mL1Size = static_cast<uint64_t>(AscendC::Std::get<0>(blockShape));
                const uint64_t nL1Size = static_cast<uint64_t>(AscendC::Std::get<1>(blockShape));
                if (mL1Size == 0 || nL1Size == 0) {
                    continue;
                }
                uint64_t kaL1Size = 0;
                uint64_t kbL1Size = 0;
                uint64_t l0KSize = 0;
                CalcKBlock(mL1Size, nL1Size, params.hasBias != 0, kaL1Size, kbL1Size, l0KSize);
                if ASCEND_IS_AIC {
                    auto xGm = reinterpret_cast<__gm__ AType*>(params.mmad.aGmAddr) + groupMOffset * kSize;
                    auto tensorAGm = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(xGm),
                        MakeLayoutA{}(static_cast<int64_t>(mSize), static_cast<int64_t>(kSize)));
                    auto scaleAGm = reinterpret_cast<__gm__ ScaleAType*>(params.mmad.scaleAGmAddr) +
                                    groupMOffset * scaleKSize;
                    auto tensorScaleAGm = asc::te::make_tensor(
                        asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGm),
                        MakeLayoutScaleA{}(static_cast<int64_t>(mSize), static_cast<int64_t>(scaleKSize)));
                    auto tensorBlockAGm = tensorAGm.slice(
                        asc::te::make_coord(static_cast<int64_t>(mOffset), 0),
                        asc::te::make_shape(static_cast<int64_t>(mL1Size), static_cast<int64_t>(kSize)));
                    auto tensorBlockScaleAGm = tensorScaleAGm.slice(
                        asc::te::make_coord(static_cast<int64_t>(mOffset), 0),
                        asc::te::make_shape(static_cast<int64_t>(mL1Size), static_cast<int64_t>(scaleKSize)));
                    auto ubC = MakeFixpipeUbTensor(mL1Size, nL1Size);
                    auto blockScaleB = scaleBGmTensor.slice(
                        asc::te::make_coord(0L, static_cast<int64_t>(nOffset)),
                        asc::te::make_shape(static_cast<int64_t>(scaleKGmSize), static_cast<int64_t>(nL1Size)));
                    blockMmad(tensorBlockAGm, tensorBlockScaleAGm, blockScaleB, ubC, kaL1Size, kbL1Size, l0KSize);
                    SetFixpipeReady();
                }
                if ASCEND_IS_AIV {
                    auto drainPrevious = [&]() __aicore__ {
                        blockEpilogue.PrefetchRoutingMetadata(mL1Size, mOffset, groupMOffset);
                        if (hasPendingTile) {
                            ReleaseOutputArena();
                            WaitFixpipeReady();
                            blockEpilogue(
                                typename BlockEpilogue::ProblemShape{0, static_cast<int64_t>(nSize),
                                                                     static_cast<int64_t>(kSize)},
                                typename BlockEpilogue::BlockShape{static_cast<int64_t>(pendingM),
                                                                   static_cast<int64_t>(pendingN), 0, 0},
                                typename BlockEpilogue::BlockCoord{static_cast<int64_t>(pendingNOffset), 0, 0, 0,
                                                                   static_cast<int64_t>(pendingMOffset),
                                                                   static_cast<int64_t>(pendingMOffset)},
                                typename BlockEpilogue::BlockCoord{0, 0, 0, 0, static_cast<int64_t>(pendingGroupM),
                                                                   static_cast<int64_t>(pendingGroupM)});
                            hasPendingTile = false;
                        }
                    };
                    blockPrologue(weightGmTensor, biasGmTensor, kSize, kbL1Size, nL1Size, nOffset, nAlign,
                                  disableWeightL2, drainPrevious);
                    pendingM = mL1Size;
                    pendingN = nL1Size;
                    pendingMOffset = mOffset;
                    pendingNOffset = nOffset;
                    pendingGroupM = groupMOffset;
                    hasPendingTile = true;
                }
            } while (scheduler.GetTileIdx(blockCoord));
        }
        if ASCEND_IS_AIV {
            if (hasPendingTile) {
                ReleaseOutputArena();
                WaitFixpipeReady();
                blockEpilogue(
                    typename BlockEpilogue::ProblemShape{0, static_cast<int64_t>(nSize), static_cast<int64_t>(kSize)},
                    typename BlockEpilogue::BlockShape{static_cast<int64_t>(pendingM), static_cast<int64_t>(pendingN),
                                                       0, 0},
                    typename BlockEpilogue::BlockCoord{static_cast<int64_t>(pendingNOffset), 0, 0, 0,
                                                       static_cast<int64_t>(pendingMOffset),
                                                       static_cast<int64_t>(pendingMOffset)},
                    typename BlockEpilogue::BlockCoord{0, 0, 0, 0, static_cast<int64_t>(pendingGroupM),
                                                       static_cast<int64_t>(pendingGroupM)});
            }
        }
    }

    __aicore__ inline auto MakeFixpipeUbTensor(uint64_t mRows, uint64_t nCols)
    {
        constexpr uint64_t FIXPIPE_BASE = 80 * GMM_FR_WQ_UB_KB;
        // Mirror the MX finalize-routing GetL0c2UbTensor contract: an NDExt frame layout
        // whose row pitch is 32-element aligned, consumed by copy_l0c_to_ub with split-M.
        const uint64_t copyRows = CeilAlign(mRows, static_cast<uint64_t>(2));
        const uint64_t copyCols = Blaze::Gemm::Align32(nCols);
        const auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(copyRows),
                                                                                   static_cast<int64_t>(copyCols));
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(FIXPIPE_BASE), layout);
    }

    __aicore__ inline void CalcKBlock(uint64_t m, uint64_t n, bool hasBias, uint64_t& ka, uint64_t& kb,
                                      uint64_t& k0) const
    {
        const uint64_t maxMForExtendedK = hasBias ? EXTENDED_L1_K_MAX_M_WITH_BIAS : GMM_FR_WQ_M_L1;
        kb = m <= maxMForExtendedK && n <= NARROW_TILE_N ? L1_K_EXTENDED : L1_K_BASE;
        const uint64_t mAlign = Align16(m);
        const uint64_t aCapacityBytes = A_L1_SINGLE_BUF_SIZE - (hasBias ? BIAS_L1_SINGLE_BUF_SIZE : 0);
        const uint64_t capacity = aCapacityBytes / (mAlign * kb);
        ka = Min(CeilDiv(n, mAlign * 2), capacity) * kb;
        k0 = m <= SHORT_TILE_M && n <= NARROW_TILE_N ? L0_K_EXTENDED : L0_K_BASE;
    }

    static constexpr uint64_t MXFP_GROUP_SIZE_LOCAL = 32;
    int64_t preOffset_{0};
    uint32_t groupListType_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
