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
 * \file kernel_qbmm_mix_without_batch.h
 * \brief MIX template kernel without batch: AIC cube compute + AIV dequant epilogue (Tensor API).
 */
#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MIX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_WITHOUT_BATCH_KERNEL_TEM_PARAMS                                              \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                   \
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<KernelMmadWithScaleMixWithoutBatch, \
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MIX_WITHOUT_BATCH_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MIX_WITHOUT_BATCH_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using OutType = typename BlockEpilogue::OutType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

private:
    static constexpr bool IS_INT8_INPUT = AscendC::Std::is_same_v<AType, int8_t> &&
                                          AscendC::Std::is_same_v<BType, int8_t>;
    static constexpr bool IS_HIFLOAT8_INPUT = AscendC::Std::is_same_v<AType, hifloat8_t> &&
                                              AscendC::Std::is_same_v<BType, hifloat8_t>;
    static constexpr bool IS_FP8_INPUT = IsFp8<AType>() && IsFp8<BType>();
    static constexpr bool IS_SUPPORTED_OUTPUT = AscendC::Std::is_one_of_v<OutType, half, bfloat16_t, float>;

    static_assert(IS_INT8_INPUT || IS_HIFLOAT8_INPUT || IS_FP8_INPUT,
                  "QBMM Mix Without Batch: AType/BType must both be int8_t, both be hifloat8_t, or each be "
                  "fp8_e4m3fn_t/fp8_e5m2_t.");
    static_assert(IS_SUPPORTED_OUTPUT, "QBMM Mix Without Batch: BlockEpilogue::OutType must be half/bfloat16_t/float.");
    static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::L0CType, L0CType>,
                  "QBMM Mix Without Batch: BlockEpilogue::L0CType must match BlockMmad::L0CType.");
    static_assert(AscendC::Std::is_one_of_v<LayoutA, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>,
                  "QBMM Mix Without Batch: LayoutA must be nd_ext_layout_ptn/dn_ext_layout_ptn.");
    static_assert(
        AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                  asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
        "QBMM Mix Without Batch: LayoutB must be nd_ext_layout_ptn/dn_ext_layout_ptn/nz_layout_ptn/zn_layout_ptn.");

public:
    struct QBMMTiling {
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        BlockSchedulerParams schParams;
        EpilogueParams epilogueParams;
        QBMMTiling qbmmParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        BlockScheduler bs(params.problemShape, params.schParams);
        if ASCEND_IS_AIC {
            mmOp_.Init(params.mmParams);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.Init(params.epilogueParams);
        }
        Run(params, bs);
    }

private:
    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB)
    {
        if ASCEND_IS_AIC {
            // 0x7f: 128-element alignment for 128-byte B matrix GM streaming
            constexpr uint64_t cacheLineAlignMask = 0x7fUL;
            const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
            const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) &&
                                         isCurrentNAligned;
            gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }

    // Process one block on AIC(cube) and AIV(dequant), keeping Run compact.
    // hasBlock is only used by AIC WaitForVector; AIV does not read it.
    template <class GmTensorA, class GmTensorB, class GmTensorBias>
    __aicore__ inline void ProcessOneBlock(const GmTensorA& gmA, const GmTensorB& gmB, const GmTensorBias& gmBlockBias,
                                           const BlockShape& singleShape, int64_t mPos, int64_t nPos, int64_t curM,
                                           int64_t curN, int64_t k, int64_t n, int64_t l0cUbBaseOffset, bool hasBlock)
    {
        constexpr int64_t kPos = 0;
        if ASCEND_IS_AIC {
            if (hasBlock) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }
            auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(curM, k));
            auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, curN));

            // DATA_BLOCK=32 matches BlockEpilogueDequant::DATA_BLOCK.
            constexpr int64_t l0cAlign = BLOCK_BYTE_SIZE / sizeof(L0CType);
            const int64_t curNAligned = Blaze::Gemm::CeilAlign(curN, l0cAlign);
            const int64_t curMAligned = Blaze::Gemm::CeilAlign(curM, static_cast<int64_t>(2));
            auto layoutUbC = asc::te::make_frame_layout<asc::te::nd_layout_ptn>(curMAligned, curNAligned);
            auto ubC = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(l0cUbBaseOffset * sizeof(L0CType)), layoutUbC);
            mmOp_(gmBlockA, gmBlockB, ubC, singleShape, gmBlockBias);
            Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
        }
        if ASCEND_IS_AIV {
            Sync::WaitForCube();
            epilogueOp_(curM, curN, nPos, mPos, nPos, mPos * n + nPos, l0cUbBaseOffset);
            Sync::NotifyCube();
        }
    }

    __aicore__ inline void Run(const Params& params, BlockScheduler& bs)
    {
        const int64_t m = asc::te::get<MNK_M>(params.problemShape);
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);

        auto layoutA = MakeLayoutA{}(m, k);
        auto layoutB = MakeLayoutB{}(k, n);
        auto gmA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)),
            layoutA);
        auto gmB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)),
            layoutB);
        auto biasGmBase = reinterpret_cast<__gm__ int32_t*>(params.mmParams.biasGmAddr);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                     asc::te::layout_trait_default<int32_t>>(1L, n);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmBase), layoutBias);

        if ((bs.GetEndBlockIdx() + 1) * params.schParams.mTailTile * params.schParams.nTailTile <=
            AscendC::GetBlockNum()) {
            bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
        }

        BlockCoord blockCoord;
        int64_t mPos = 0;
        int64_t nPos = 0;
        bool hasBlock = false;
        while (bs.GetTileIdx(blockCoord)) {
            BlockShape singleShape = bs.template GetBlockShape<QuantMode::DEFAULT, QuantMode::DEFAULT, WEIGHT_NZ>(
                blockCoord);
            if (asc::te::get<IDX_M_TILEIDX>(singleShape) <= 0 || asc::te::get<IDX_N_TILEIDX>(singleShape) <= 0) {
                break;
            }
            bs.GetTileCoord(blockCoord, mPos, nPos);
            const int64_t curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
            const int64_t curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
            const int64_t l0cUbBaseOffset = 0;
            if ASCEND_IS_AIC {
                SetBL2Cache(params.problemShape, curM, curN, params.qbmmParams.bMustHitL2, gmB);
            }
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, params.mmParams.isBias ? nPos : 0L),
                                            asc::te::make_shape(1L, curN));
            ProcessOneBlock(gmA, gmB, gmBlockBias, singleShape, mPos, nPos, curM, curN, k, n, l0cUbBaseOffset,
                            hasBlock);
            hasBlock = true;
        }
        if ASCEND_IS_AIC {
            if (hasBlock) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }
        }
    }

    BlockMmad mmOp_;
    BlockEpilogue epilogueOp_;

    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
