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
 * \file block_mmad_wqgmm_mx_mix.h
 * \brief MMAD block for the MX weight-quant grouped-matmul mix kernel.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

static constexpr uint64_t WQGMM_MX_GROUP_SIZE = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
static constexpr uint64_t WQGMM_MX_SYNC_MODE = 4;
static constexpr uint64_t WQGMM_MX_FLAG_ID_MAX = 16;

// Macro aliases keep the specialization declaration compact for this single dispatch-policy binding.
#define BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM                                                              \
    template <class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_, \
              class LayoutC_, class BiasType_, class LayoutBias_>

#define BLAZE_WQGMM_MX_MIX_MMAD_CLASS                                                                         \
    BlockMmad<GroupedMatmulWithWeightQuantMx, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, \
              LayoutC_, BiasType_, LayoutBias_>

/*!
 * \brief AIC tile-compute unit for one tile inside one group in the weight-quant grouped matmul pipeline.
 *
 * Design reason:
 * - This class handles MMAD compute for a single-group tile only.
 * - AIC does not support direct FP4E2M1/FP4E1M2 -> FP8E4M3 conversion in the MMAD path.
 * - Therefore B must be preprocessed by prologue first, and this class consumes the converted B tiles.
 *
 * Distinctive behaviors:
 * 1) It synchronizes with prologue through cross-core flags, and the sync semantics must match prologue exactly.
 * 2) It uses dynamic kL1 splitting (kaL1/kbL1) and this policy is aligned with prologue's K-window organization.
 * 3) It uses dynamic kL0 splitting inside each kL1 tile to balance compute and memory movement.
 * 4) A/scaleA/scaleB/B use different transfer granularities by design:
 *    - scaleA/scaleB are moved with MX_SCALE_K_L1_SIZE = 4096 K-window,
 *    - typical kaL1/kbL1 are 256/512,
 *    - this larger 4096 window is used to organize scale transfer at 128B cacheline-friendly granularity,
 *      improving effective bandwidth and reuse.
 *
 * Key constraints:
 * 1) A must satisfy ND format.
 * 2) B must be prologue-converted ZN format and use the same compute type as AType.
 * 3) Scale layout must satisfy MX_DIVISOR_SIZE = 64.
 *
 * When to use:
 * - Use this block on the MX weight-quant grouped-matmul path when dynamic kL1/kL0 blocking is needed to increase
 *   per-tile data workload
 *   and overall throughput.
 */
BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
class BLAZE_WQGMM_MX_MIX_MMAD_CLASS {
public:
    using DispatchPolicy = GroupedMatmulWithWeightQuantMx;

    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using BiasType = BiasType_;
    using CType = CType_;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;

    static_assert(asc::te::is_satisfied_ptn_format_v<
                  decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((AType*)0),
                                                asc::te::frame_layout_format<LayoutA>{}(0UL, 0UL))),
                  asc::te::nd_ext_layout_ptn>);

    constexpr static int32_t C0_SIZE = asc::te::c0_element<AType>;
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;

    // Parameters are initialized by the kernel wrapper and passed to this block-level compute unit.
    struct Params {
        GM_ADDR ptrA{nullptr};
        GM_ADDR ptrScaleA{nullptr};
        GM_ADDR ptrScaleB{nullptr};
        GM_ADDR ptrBias{nullptr};
        GM_ADDR ptrC{nullptr};
    };

    __aicore__ inline BlockMmad();
    template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
    __aicore__ inline void operator()(const TensorA_& tensorA, const TensorScaleA_& tensorScaleA,
                                      const TensorScaleB_& tensorScaleB, const TensorC_& tensorC, bool hasBias);
    __aicore__ inline ~BlockMmad();

private:
    struct BlockMmadOffsetParam {
        uint64_t mL1Size;
        uint64_t kaL1Size;
        uint64_t kbL1Size;
        uint64_t l0KSize;
        uint64_t nL1Size;
        uint64_t kSize;
    };

    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t& kaL1Size,
                                             uint64_t& kbL1Size) const;
    template <typename TensorL0C_>
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam& param,
                                         const TensorL0C_& tensorL0C);
    __aicore__ inline void WaitAMTE1ToMTE2();
    __aicore__ inline void SetMTE1ToMTE2();
    __aicore__ inline void WaitScaleMTE1ToMTE2();
    __aicore__ inline void SetScaleMTE1ToMTE2();
    template <typename TensorA_>
    __aicore__ inline void CopyAGmToL1(const TensorA_& tensorA, const BlockMmadOffsetParam& param, int64_t kaGmOffset);
    template <typename TensorScaleA_>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA_& tensorScaleA, const BlockMmadOffsetParam& param,
                                             uint64_t kbL1Offset);
    template <typename TensorC_, typename TensorL0C_>
    __aicore__ inline void CopyCL0c2Gm(const TensorC_& tensorC, const TensorL0C_& tensorL0C);

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 = typename asc::te::frame_layout_format<asc::te::zz_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 = typename asc::te::frame_layout_format<asc::te::nn_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;

    using MakeLayoutAL0 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL0 = asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL0 = typename asc::te::frame_layout_format<asc::te::zz_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL0 = typename asc::te::frame_layout_format<asc::te::nn_layout_ptn,
                                                                     AscendC::Std::Int<SCALE_C0>>;

    static constexpr uint64_t L1_M = 256;
    static constexpr uint64_t L1_N = 256;

    static constexpr uint64_t L1_K_CONFIG_512 = 512;
    static constexpr uint64_t L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;
    static constexpr uint64_t L1_K_DYNAMIC_CONFIG_N_THRESHOLD = L1_N >> 1;
    // Keep the A-side staging model aligned with the low-level GMM implementation. The model reserves 80KB for
    // each A buffer and chooses ka depth by balancing the ND2NZ A load against the converted-B L1 load.
    static constexpr uint64_t A_L1_MODEL_SINGLE_BUF_SIZE = 80 * 1024;
    static constexpr uint64_t A_B_LOAD_BALANCE_FACTOR = 2;
    static constexpr uint64_t L1_K_DYNAMIC_CONFIG_M_THRESHOLD = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType) /
                                                                 L1_K_CONFIG_512 / BLOCK_CUBE) *
                                                                BLOCK_CUBE;

    uint64_t aL1BufIdx_ = 0;
    uint64_t bL1BufIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    bool hasBias_ = false;

    using TensorAL1 = decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(0UL),
                                                    MakeLayoutAL1{}(16UL, 16UL)));
    using TensorScaleAL1 = decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, ScaleAType>(0UL),
                                                         MakeLayoutScaleAL1{}(16UL, 16UL)));
    using TensorScaleBL1 = decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, ScaleBType>(0UL),
                                                         MakeLayoutScaleBL1{}(16UL, 16UL)));

    TensorAL1 tensorAL1_;
    TensorScaleAL1 tensorScaleAL1_;
    TensorScaleBL1 tensorScaleBL1_;

    // --- sync ---
    // 2 buffer
    static constexpr TEventID EVENT_IDS_MTE1_TO_MTE2 = 0;
    // 2 buffer
    static constexpr TEventID EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 = 2;
    static constexpr TEventID EVENT_ID_MTE1_TO_MTE2 = 4;
    static constexpr TEventID EVENT_ID_MTE2_TO_MTE1 = 0;
    // Match the original basic API event assignment: M_MTE1 ping/pong uses
    // slots 3/4 and the shared MTE1_M fence uses slot 3.
    static constexpr TEventID EVENT_ID_M_TO_MTE1 = 3;
    static constexpr TEventID EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;

    /**
     * L1 512KB Memory Map
     * * Segment 1 [0KB - 256KB]:
     * [0k]    [64k]      [96k]    [128k]                        [256k]
     * |--B B0---|--scA0---|--scB0---|----------- A (Part 1) -------|
     *     (64KB)     (32KB)    (32KB)            (128KB)
     * * Segment 2 [256KB - 512KB]:
     * [256k]                    [384k]        [448k]   [480k]   [512k]
     * |-------- A (Part 2) -------|---- B B1 ----|--scA1--|--scB1--|
     *          (128KB)               (64KB)        (32KB)    (32KB)
     * * Note: A is a contiguous 256KB block spanning the middle of the buffer.
     */
    static constexpr uint64_t SCALE_AL1_OFFSET = 64 * 1024;
    static constexpr uint64_t SCALE_BL1_OFFSET = 96 * 1024;
    static constexpr uint64_t A_L1_OFFSET = 128 * 1024;
    static constexpr uint64_t L1_BUF_OFFSET = 384 * 1024;
    static constexpr uint64_t A_L1_BUF_OFFSET = 128 * 1024;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128 * 1024;
    static constexpr uint64_t BIAS_L1_SINGLE_BUF_SIZE = 4 * 1024;
    static constexpr uint64_t BIAS_L1_OFFSETS[DOUBLE_BUFFER_COUNT] = {64 * 1024, 380 * 1024};
    // Match the original basic API: each L0 ping/pong operand owns a 2KB BT slot.
    static constexpr uint64_t BIAS_TABLE_SLOT_SIZE = 2 * 256 * sizeof(float);

    /**
     * L0 64KB Memory Map (double-buffered B tiles)
     * [0k]      [32KB]    [64KB]
     * |--- B0 ---|--- B1 ---|
     *    (32KB)     (32KB)
     */
    static constexpr uint64_t L0_BUF_OFFSET = 32 * 1024;

    // Tensor API atoms carry no runtime state. Keeping them as class constants avoids rebuilding the descriptors
    // in every K iteration while preserving the same generated data-movement/MMAD operations.
    static constexpr auto COPY_GM_TO_L1_ATOM = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    static constexpr auto COPY_L1_TO_L0A_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
    static constexpr auto COPY_L1_TO_L0B_ATOM = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
    static constexpr auto COPY_L1_TO_BT_ATOM = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
    static constexpr auto COPY_L0C_TO_GM_ATOM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
    static constexpr auto MMAD_MX_ATOM = asc::te::mmad_atom<
        asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{};
};

} // namespace Block

namespace Block {

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorL0C_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize,
                                                                    const BlockMmadOffsetParam& param,
                                                                    const TensorL0C_& tensorL0C)
{
    auto tensorBL1 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(bL1BufIdx_ * L1_BUF_OFFSET),
        asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>{}(kbL1RealSize,
                                                                                           param.nL1Size));
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    // Decide the MMAD accumulation mode for this K-tile.
    bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    bool isFirstGmK = kbOffset == 0;
    asc::te::mmad_params params;
    params.m = static_cast<uint16_t>(param.mL1Size);
    params.n = static_cast<uint16_t>(param.nL1Size);
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += param.l0KSize) {
        bool isLastL1K = l1KOffset + param.l0KSize >= kbL1RealSize;
        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : param.l0KSize;
        uint64_t realL0ScaleK = CeilDiv(realL0k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);

        auto layoutAL0 = MakeLayoutAL0{}(param.mL1Size, realL0k);
        auto tensorAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutAL0);
        auto tensorBlockAL1 = tensorAL1_.slice(asc::te::make_coord(0, (l1KOffset + kbOffset) % param.kaL1Size),
                                               asc::te::make_shape(param.mL1Size, realL0k));
        asc::te::copy(COPY_L1_TO_L0A_ATOM, tensorAL0, tensorBlockAL1);

        uint64_t scaleKOffset = ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MXFP_DIVISOR_SIZE) *
                                MXFP_MULTI_BASE_SIZE;
        auto layoutScaleAL0 = MakeLayoutScaleAL0{}(param.mL1Size, realL0ScaleK);
        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, ScaleAType>((l0BufIdx_ * L0_BUF_OFFSET) >> 4),
            layoutScaleAL0);
        auto tensorBlockScaleAL1 = tensorScaleAL1_.slice(asc::te::make_coord(0, scaleKOffset),
                                                         asc::te::make_shape(param.mL1Size, realL0ScaleK));
        auto copyL12L0ScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        asc::te::copy(copyL12L0ScaleA, tensorScaleAL0, tensorBlockScaleAL1);

        auto layoutBL0 = MakeLayoutBL0{}(realL0k, param.nL1Size);
        auto tensorBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(l1KOffset, 0),
                                              asc::te::make_shape(realL0k, param.nL1Size));
        asc::te::copy(COPY_L1_TO_L0B_ATOM, tensorBL0, tensorBlockBL1);

        auto layoutScaleBL0 = MakeLayoutScaleBL0{}(realL0ScaleK, param.nL1Size);
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, ScaleBType>((l0BufIdx_ * L0_BUF_OFFSET) >> 4),
            layoutScaleBL0);
        auto tensorBlockScaleBL1 = tensorScaleBL1_.slice(asc::te::make_coord(scaleKOffset, 0),
                                                         asc::te::make_shape(realL0ScaleK, param.nL1Size));
        auto copyL12L0ScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        asc::te::copy(copyL12L0ScaleB, tensorScaleBL0, tensorBlockScaleBL1);

        bool needBias = hasBias_ && isFirstGmK && l1KOffset == 0;
        params.k = static_cast<uint16_t>(realL0k);
        params.unit_flag = (isLastGmK && isLastL1K) ? asc::te::unit_flag_mode::enable_update :
                                                      asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = isFirstGmK && l1KOffset == 0 && !needBias;
        if (needBias) {
            // Bias is consumed only by the first L0 tile. Keep all three descriptors out of the no-bias and
            // subsequent-K hot path instead of rebuilding them for every L0 iteration.
            auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
                1, CeilAlign(param.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)));
            auto tensorBiasL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(BIAS_L1_OFFSETS[bL1BufIdx_]), layoutBias);
            auto tensorBiasBT = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::bias, float>(l0BufIdx_ * BIAS_TABLE_SLOT_SIZE), layoutBias);
            asc::te::copy(COPY_L1_TO_BT_ATOM, tensorBiasBT, tensorBiasL1);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            auto Mmad = MMAD_MX_ATOM.with(params);
            asc::te::mmad(Mmad, tensorL0C, tensorAL0, tensorBL0, tensorBiasBT);
        } else {
            SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
            auto Mmad = MMAD_MX_ATOM.with(params);
            asc::te::mmad(Mmad, tensorL0C, tensorAL0, tensorBL0);
        }

        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + l0BufIdx_);
        l0BufIdx_ ^= 1;
    }
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::WaitAMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::WaitScaleMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleL1BufIdx_);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetScaleMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + scaleL1BufIdx_);
    scaleL1BufIdx_ ^= 1;
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + aL1BufIdx_);
    aL1BufIdx_ ^= 1;
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorA_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CopyAGmToL1(const TensorA_& tensorA,
                                                                  const BlockMmadOffsetParam& param, int64_t kaGmOffset)
{
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
    auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kaGmOffset), asc::te::make_shape(param.mL1Size, kaL1RealSize));
    tensorAL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(
            A_L1_OFFSET + aL1BufIdx_ * A_L1_BUF_OFFSET + (hasBias_ && aL1BufIdx_ == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0)),
        layoutAL1);
    asc::te::copy(COPY_GM_TO_L1_ATOM, tensorAL1_, gmBlockA);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorScaleA_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CopyMxScaleGmToL1(const TensorScaleA_& tensorScaleA,
                                                                        const BlockMmadOffsetParam& param,
                                                                        uint64_t kbL1Offset)
{
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / WQGMM_MX_GROUP_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > param.kSize ?
                                    (param.kSize - kbL1Offset) / WQGMM_MX_GROUP_SIZE :
                                    scaleKL1StandardLen;
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    tensorScaleAL1_ = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(
                                               SCALE_AL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET +
                                               (hasBias_ && scaleL1BufIdx_ == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0)),
                                           layoutScaleAL1);
    auto gmBlockScaleA = tensorScaleA.slice(asc::te::make_coord(0, kbL1Offset / WQGMM_MX_GROUP_SIZE),
                                            asc::te::make_shape(param.mL1Size, scaleKL1RealSize));
    asc::te::copy(COPY_GM_TO_L1_ATOM, tensorScaleAL1_, gmBlockScaleA);

    // ScaleB is produced by the paired AIV prologue directly in this L1 buffer. AIC only materializes the tensor
    // view consumed by L1->L0B; copying ScaleB from GM here would serialize it with ScaleA/A on the AIC MTE2 pipe.
    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    tensorScaleBL1_ = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(
                                               SCALE_BL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET +
                                               (hasBias_ && scaleL1BufIdx_ == 0 ? BIAS_L1_SINGLE_BUF_SIZE : 0)),
                                           layoutScaleBL1);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorC_, typename TensorL0C_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CopyCL0c2Gm(const TensorC_& tensorC, const TensorL0C_& tensorL0C)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    asc::te::copy(COPY_L0C_TO_GM_ATOM.with(asc::te::l0c_to_gm_params{asc::te::unit_flag_mode::enable_update}), tensorC,
                  tensorL0C, FP32_64_AS_UINT64);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline BLAZE_WQGMM_MX_MIX_MMAD_CLASS::BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
    }
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline BLAZE_WQGMM_MX_MIX_MMAD_CLASS::~BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_COUNT; i++) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MX_SCALE_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_IDS_MTE1_TO_MTE2 + i);
    }
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size,
                                                                        uint64_t& kaL1Size, uint64_t& kbL1Size) const
{
    kbL1Size = (mL1Size <= L1_K_DYNAMIC_CONFIG_M_THRESHOLD && nL1Size <= L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                   L1_K_CONFIG_512 :
                   L1_K_CONFIG_256;

    uint64_t mL1Align = CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t kaDepth = CeilDiv(nL1Size, mL1Align * A_B_LOAD_BALANCE_FACTOR);
    kaDepth = kaDepth == 0 ? 1 : kaDepth;

    uint64_t modelMaxKaDepth = (A_L1_MODEL_SINGLE_BUF_SIZE / sizeof(AType)) / (mL1Align * kbL1Size);
    uint64_t actualAL1SingleBufSize = A_L1_SINGLE_BUF_SIZE - (hasBias_ ? BIAS_L1_SINGLE_BUF_SIZE : 0);
    uint64_t actualMaxKaDepth = (actualAL1SingleBufSize / sizeof(AType)) / (mL1Align * kbL1Size);
    uint64_t maxKaDepth = modelMaxKaDepth < actualMaxKaDepth ? modelMaxKaDepth : actualMaxKaDepth;
    // The scheduler limits mL1Size so one kb tile always fits. Keep the guard explicit to avoid a zero ka size if
    // that contract is relaxed later.
    maxKaDepth = maxKaDepth == 0 ? 1 : maxKaDepth;
    kaDepth = kaDepth < maxKaDepth ? kaDepth : maxKaDepth;
    kaL1Size = kaDepth * kbL1Size;
}

/*
 * kaL1Size % kbL1Size == 0
 * scaleL1Size % kbL1Size == 0
 */
BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
template <typename TensorA_, typename TensorScaleA_, typename TensorScaleB_, typename TensorC_>
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::operator()(const TensorA_& tensorA,
                                                                 const TensorScaleA_& tensorScaleA,
                                                                 const TensorScaleB_& tensorScaleB,
                                                                 const TensorC_& tensorC, bool hasBias)
{
    // ScaleB is an AIV-produced L1 operand. Keep the public tensor argument for the kernel API contract, but do not
    // dereference it on AIC.
    static_cast<void>(tensorScaleB);
    hasBias_ = hasBias;
    BlockMmadOffsetParam blockParam = {};
    blockParam.mL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(tensorC.layout());
    blockParam.kSize = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorA.layout());
    blockParam.nL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(
        tensorC.layout());
    CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, blockParam.kaL1Size, blockParam.kbL1Size);
    blockParam.l0KSize = (blockParam.mL1Size <= 128 && blockParam.nL1Size <= 128) ? 256 : 128;
    // L0C shape and address are invariant for the complete K traversal. Reuse one tensor descriptor for every
    // K-L1 tile and for the final fixpipe writeback.
    auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(blockParam.mL1Size,
                                                                                                   blockParam.nL1Size);
    auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);
    for (uint64_t kbGmOffset = 0; kbGmOffset < blockParam.kSize; kbGmOffset += blockParam.kbL1Size, bL1BufIdx_ ^= 1) {
        uint64_t kbL1RealSize = (kbGmOffset + blockParam.kbL1Size) >= blockParam.kSize ? blockParam.kSize - kbGmOffset :
                                                                                         blockParam.kbL1Size;
        if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
            WaitScaleMTE1ToMTE2();
            CopyMxScaleGmToL1(tensorScaleA, blockParam, kbGmOffset);
        }

        if (kbGmOffset % blockParam.kaL1Size == 0) {
            WaitAMTE1ToMTE2();
            CopyAGmToL1(tensorA, blockParam, kbGmOffset);
        }

        WaitAivToAic();
        ProcessTileL1(kbGmOffset, kbL1RealSize, blockParam, tensorL0C);
        uint64_t nextKbGmOffset = kbGmOffset + blockParam.kbL1Size;
        if (nextKbGmOffset % blockParam.kaL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetMTE1ToMTE2();
        }
        if (nextKbGmOffset % MX_SCALE_K_L1_SIZE == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetScaleMTE1ToMTE2();
        }
        SetAicToAiv();
    }
    CopyCL0c2Gm(tensorC, tensorL0C);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::WaitAivToAic()
{
    CrossCoreWaitFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + WQGMM_MX_FLAG_ID_MAX);
    CrossCoreWaitFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
__aicore__ inline void BLAZE_WQGMM_MX_MIX_MMAD_CLASS::SetAicToAiv()
{
    CrossCoreSetFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + WQGMM_MX_FLAG_ID_MAX);
    CrossCoreSetFlag<WQGMM_MX_SYNC_MODE, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef BLAZE_WQGMM_MX_MIX_MMAD_TEMPLATE_PARAM
#undef BLAZE_WQGMM_MX_MIX_MMAD_CLASS
} // namespace Block
} // namespace Gemm
} // namespace Blaze
