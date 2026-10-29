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
 * \file kernel_qbmm_mx_without_batch.h
 * \brief
 */

#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS                                              \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                  \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleMxWithoutBatch, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using ScaleType = MxScaleType<AType>;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;

    static_assert((IsFp8<AType>() && IsFp8<BType>()) || (IsFp4<AType>() && IsFp4<BType>()) ||
                      (IsHif4<AType>() && IsHif4<BType>()),
                  "QBMM MX Without Batch: AType/BType must each be fp8_e4m3fn_t/fp8_e5m2_t, or each be "
                  "fp4x2_e2m1_t/fp4x2_e1m2_t/hifloat4x2_t.");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
                  "QBMM MX Without Batch: BlockMmad::CType must be half/bfloat16_t/float.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX Without Batch: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(
        AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                  asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
        "QBMM MX Without Batch: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");
    // Preserve the ND/DN C Tensor layouts supported by the L0C copy path.
    // A supplied epilogue must use the same output layout.
    static_assert(AscendC::Std::is_one_of_v<LayoutC, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM MX Without Batch: LayoutC must be nd_ext_layout_ptn/dn_ext_layout_ptn.");

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr bool IS_HIF4 = IsHif4<AType>();
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t SCALE_N_ALIGN = IsScaleNz<LayoutScaleB>::value ? MATMUL_MNK_ALIGN : 1;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<GetScaleC0<AType>()>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<GetScaleC0<AType>()>>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, AscendC::Std::Int<GetScaleC0<BType>()>>;

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void Process(const Params& params, BlockScheduler& bs);

    // Process one block, keeping Process compact.
    template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
              class GmTensorC>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos,
                                           int64_t nPos, int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

    BlockMmad mmadOp_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ ScaleType* scaleAGmAddr_;
    __gm__ ScaleType* scaleBGmAddr_;
};

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    const bool isBias = qbmmParams.isBias == 1;
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
    if (isBias) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }

    BlockScheduler bs(problemShape, params.schParams);

    const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
    mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias, qbmmParams.dbL0C > 1);

    Process(params, bs);
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB>
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::SetBL2Cache(
    const ProblemShape& problemShape, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN, uint32_t bMustHitL2,
    TensorB& gmB)
{
    // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
    // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                 isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
template <class GmTensorA, class GmTensorB, class GmTensorScaleA, class GmTensorScaleB, class GmTensorBias,
          class GmTensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::ProcessOneBlock(
    const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorScaleA& gmScaleA, const GmTensorScaleB& gmScaleB,
    const GmTensorBias& gmBias, const GmTensorC& gmC, const BlockShape& singleShape, int64_t mPos, int64_t nPos,
    int64_t baseM, int64_t baseN, int64_t k, int64_t scaleKLen)
{
    constexpr int64_t kPos = 0L; // K is not split, so the K coordinate is 0.
    int64_t scaleAPos = mPos;
    if constexpr (IS_HIF4 && TRANS_A) {
        scaleAPos = mPos >> 1;
    }
    int64_t scaleBPos = nPos;
    if constexpr (IS_HIF4 && !TRANS_B) {
        scaleBPos = nPos >> 1;
    }
    auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
    auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(scaleAPos, kPos), asc::te::make_shape(baseM, scaleKLen));
    auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, baseN));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, scaleBPos), asc::te::make_shape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
    auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(baseM, baseN));

    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
}

QBMM_MX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_WITHOUT_BATCH_KERNEL_TEM_PARAMS>::Process(const Params& params,
                                                                                       BlockScheduler& bs)
{
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    const auto scaleKLen = MxScaleKLen<AType>(k);
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    auto layoutC = MakeLayoutC{}(m, n);
    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
    if constexpr (IS_ATOMIC_ADD) {
        gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
    }

    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    if ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        bs.UpdateTailTile(mTailTile, nTailTile);
    }

    BlockCoord blockCoord;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape = bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                                           WEIGHT_NZ, SCALE_N_ALIGN>(blockCoord);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            break;
        }
        bs.GetTileCoord(blockCoord, mPos, nPos);
        SetBL2Cache(problemShape, baseM, baseN, params.qbmmParams.bMustHitL2, gmB);
        ProcessOneBlock(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, singleShape, mPos, nPos, baseM, baseN, k, scaleKLen);
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
