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
 * \file kernel_qgmm_swiglu_mx.h
 * \brief QGMM MX kernel specialization that fuses mode-2 SwiGLU and MXFP8 output quantization.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

/*!
 * \brief AIC/AIV fused QGMM MX kernel for paired left/right N halves.
 *
 * AIC computes both halves into UB and AIV applies SwiGLU followed by MX quantization. The caller must validate
 * all required GM addresses and attributes before entering this low-level component; it does not return ACLNN
 * status codes. The current V3 integration accepts mode 2, scale algorithms 0/1 and dstTypeMax 0 only.
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelGmmSwiGluMixMx, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
private:
    static constexpr int64_t SCALE_CACHE_MASK = 0xff;
    static constexpr uint32_t DEFAULT_TAIL_ALIGNMENT = 1;
    static constexpr int64_t GROUP_LIST_ELEMENT_STRIDE = 1;
    static constexpr uint32_t SWIGLU_N_SPLIT_SHIFT = 1;
    static constexpr uint8_t SINGLE_WEIGHT_TENSOR = 1;
    static constexpr int64_t BIAS_ROW_COUNT = 1;
    static constexpr uint32_t UINT64_ADDRESS_SHIFT = 3;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    static_assert((IsFp4<AType>() && IsFp4<BType>()) || (IsFp8<AType>() && IsFp8<BType>()),
                  "QGMM SwiGLU MX only supports same bit-width MXFP4/MXFP8 AType and BType.");
    static_assert(AscendC::Std::is_same_v<CType, float> && AscendC::Std::is_same_v<BiasType, float>,
                  "QGMM SwiGLU MX only supports float CType and BiasType.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "QGMM SwiGLU MX only supports ND LayoutA.");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QGMM SwiGLU MX only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutBias, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "QGMM SwiGLU MX does not support NZ/ZN LayoutC or LayoutBias.");
    static_assert(AscendC::IsSameType<CType, typename BlockEpilogue::DataTypeIn>::value,
                  "BlockMmad UB output type must match BlockEpilogue input type.");
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;

    using BlockMmadAddressParams = typename BlockMmad::Params;
    using BlockMmadInitParams = typename BlockMmad::MmadParams;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using EpilogueProblemShape = typename BlockEpilogue::ProblemShape;
    using EpilogueBlockShape = typename BlockEpilogue::BlockShape;
    using EpilogueOutputOffsets = typename BlockEpilogue::OutputOffsets;

    /*! \brief Runtime grouped-matmul tiling consumed by the fused specialization. */
    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t dbL0C;
        int8_t groupType;
        uint8_t groupListType;
        uint8_t singleW;
    };

    /*! \brief Kernel launch parameters. */
    struct Params {
        ProblemShape problemShape;                     //!< Compatibility field; gmmParams supplies authoritative M/N/K.
        BlockMmadAddressParams blockMmadAddressParams; //!< A/B/ScaleA/ScaleB addresses; C/Bias are unused here.
        BlockEpilogueParams epilogueParams;            //!< Required Y/YScale addresses and fused attributes.
        GM_ADDR groupListGmAddr;                       //!< Required when groupNum is nonzero.
        GMMTiling gmmParams;                           //!< Authoritative grouped-matmul shape and tiling.
    };

    __aicore__ inline GemmUniversal() {}

    __aicore__ inline ~GemmUniversal() {}

    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    static constexpr uint64_t INPUT_C0_SIZE = IsFp4<AType>() ? Blaze::Gemm::C0_SIZE_B4 : Blaze::Gemm::C0_SIZE_B8;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<INPUT_C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    __aicore__ inline void SyncAicToAiv()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void SyncAivToAic()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
        }
    }

    __aicore__ inline void WaitForAiv()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void WaitForAic()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForAiv();
            }
        }
    }

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& bs)
    {
        uint32_t mTailAlign = DEFAULT_TAIL_ALIGNMENT;
        uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(INPUT_C0_SIZE);
        bs.SetTailAlign(mTailAlign, nTailAlign);
    }

    __aicore__ inline void BaseMBalance(BlockScheduler& scheduler, int64_t m, int64_t baseM)
    {
        if (m <= 0) {
            return;
        }
        const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
        const int64_t mCnt = CeilDiv(m, safeBaseM);
        const int64_t balancedBaseM = CeilDiv(m, mCnt);
        curBaseM_ = static_cast<uint32_t>(CeilAlign(balancedBaseM, static_cast<int64_t>(BLOCK_CUBE)));
        scheduler.UpdateBaseM(curBaseM_);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, TensorScaleB& gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        if constexpr (TRANS_B) {
            if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
            } else {
                gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
            }
        } else {
            if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
            } else {
                gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
            }
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const auto groupListLayout = asc::te::make_layout(asc::te::make_shape(static_cast<int64_t>(groupNum_)),
                                                          asc::te::make_stride(GROUP_LIST_ELEMENT_STRIDE));
        const auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr)),
            groupListLayout);
        const auto& gmmParams = params.gmmParams;
        BlockScheduler bs(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(bs);

        for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
            if (!SetMNK(gmGroupList, groupIdx)) {
                continue;
            }
            const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
            const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
            const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
            BaseMBalance(bs, problemM, gmmParams.baseM);
            if ASCEND_IS_AIC {
                blockMmad.UpdateParamsForNextProblem(problemShape_);
            }
            bs.UpdateNextProblem(SchedulerShape{problemM, problemN >> SWIGLU_N_SPLIT_SHIFT, problemK, 0});
            ProcessSingleGroup(bs, groupIdx);
        }
        End();
    }

    __aicore__ inline void Init(const Params& params)
    {
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.blockMmadAddressParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.blockMmadAddressParams.bGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadAddressParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.blockMmadAddressParams.scaleBGmAddr);
        singleW_ = (gmmParams.singleW == SINGLE_WEIGHT_TENSOR);

        const ProblemShape initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;

        const BlockShape l0TileShape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                     static_cast<int64_t>(gmmParams.baseK), 0};
        const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                static_cast<uint64_t>(gmmParams.scaleKAL1)};
        const BlockMmadInitParams blockMmadInitParams{l0TileShape, l1Params, false,
                                                      gmmParams.dbL0C == DOUBLE_BUFFER_COUNT};
        if ASCEND_IS_AIC {
            blockMmad.Init(initProblemShape, blockMmadInitParams);
        }

        blockEpilogue.Init(params.epilogueParams);

        const EpilogueProblemShape epilogueProblemShape{asc::te::get<MNK_M>(problemShape_),
                                                        asc::te::get<MNK_N>(problemShape_) >> SWIGLU_N_SPLIT_SHIFT,
                                                        asc::te::get<MNK_K>(problemShape_)};
        blockEpilogue.UpdateNextProblem(epilogueProblemShape);
    }

    template <typename GroupListTensor>
    __aicore__ inline bool SetMNK(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(gmGroupList, groupIdx);
        const ProblemShape initProblemShape{splitValue, asc::te::get<MNK_N>(problemShape_),
                                            asc::te::get<MNK_K>(problemShape_), 0};
        problemShape_ = initProblemShape;
        if (splitValue <= 0) {
            return false;
        }
        return true;
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor& gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        const int64_t groupValue = gmGroupList[groupIdx];
        // groupListType supports only 0 (cumulative offsets/cumsum) and 1 (per-group counts).
        if (groupListType_ == 0) {
            splitValue = groupValue - preOffset_;
            preOffset_ = groupValue;
        } else {
            splitValue = groupValue;
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void ProcessSingleGroup(BlockScheduler& bs, uint32_t groupIdx)
    {
        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        const int64_t baseN = static_cast<int64_t>(baseN_);

        const typename BlockScheduler::SwigluGroupParams groupParams{groupIdx, preOffset_, problemShape_, singleW_};
        bs.UpdateSwigluGroup(groupParams);
        typename BlockScheduler::SwigluBlockInfo blockInfo;
        if (!bs.GetNextSwigluBlock(blockInfo)) {
            return;
        }
        const typename BlockScheduler::SwigluGroupInfo& groupInfo = bs.GetSwigluGroupInfo();

        EpilogueOutputOffsets baseOutputOffsets{groupInfo.outputOffset, groupInfo.outputScaleOffset};
        blockEpilogue.UpdateGlobalAddr(baseOutputOffsets);

        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, groupInfo.inputScaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(groupInfo.inputScaleK, problemN);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(BIAS_ROW_COUNT, problemN);

        int64_t aOffset = groupInfo.aOffset;
        if constexpr (IsFp4<AType>()) {
            aOffset >>= 1;
        }
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + aOffset), layoutA);
        auto gmScaleA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleABasePtr_ + groupInfo.scaleAOffset), layoutScaleA);
        int64_t weightOffset = groupInfo.bOffset;
        if constexpr (IsWeightNz<LayoutB>::value) {
            if (singleW_) {
                // Only a single batched FRACTAL_NZ tensor needs a physical per-group stride.
                // TensorList entries already point to each group and retain the scheduler's zero offset.
                const uint64_t weightGroupElements = TRANS_B ?
                                                         CeilDiv(static_cast<uint64_t>(problemK), INPUT_C0_SIZE) *
                                                             CeilDiv(static_cast<uint64_t>(problemN), C0_SIZE_fp16) *
                                                             C0_SIZE_fp16 * INPUT_C0_SIZE :
                                                         CeilDiv(static_cast<uint64_t>(problemK), C0_SIZE_fp16) *
                                                             CeilDiv(static_cast<uint64_t>(problemN), INPUT_C0_SIZE) *
                                                             C0_SIZE_fp16 * INPUT_C0_SIZE;
                weightOffset = static_cast<int64_t>(groupIdx * weightGroupElements);
            }
        }
        if constexpr (IsFp4<BType>()) {
            weightOffset >>= 1;
        }
        auto gmB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                (singleW_ ? bBasePtr_ : GetTensorAddrFromList(groupIdx, bBasePtr_)) + weightOffset),
            layoutB);
        auto gmScaleB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(
                (singleW_ ? scaleBBasePtr_ : GetTensorAddrFromList(groupIdx, scaleBBasePtr_)) + groupInfo.scaleBOffset),
            layoutScaleB);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr_), layoutBias);

        SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_), baseN);

        do {
            const int64_t blockM = asc::te::get<MNK_M>(blockInfo.blockShape);
            const int64_t blockN = asc::te::get<MNK_N>(blockInfo.blockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const BlockShape& singleShape = blockInfo.blockShape;

            auto gmBlockA = gmA.slice(asc::te::make_coord(blockInfo.aMOffset, static_cast<int64_t>(0)),
                                      asc::te::make_shape(blockM, problemK));
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(blockInfo.aMOffset, static_cast<int64_t>(0)),
                                                asc::te::make_shape(blockM, groupInfo.inputScaleK));
            auto gmBlockBLeft = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), blockInfo.bLeftNOffset),
                                          asc::te::make_shape(problemK, blockN));
            auto gmBlockBRight = gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), blockInfo.bRightNOffset),
                                           asc::te::make_shape(problemK, blockN));
            auto gmBlockScaleBLeft = gmScaleB.slice(
                asc::te::make_coord(static_cast<int64_t>(0), blockInfo.bLeftNOffset),
                asc::te::make_shape(groupInfo.inputScaleK, blockN));
            auto gmBlockScaleBRight = gmScaleB.slice(
                asc::te::make_coord(static_cast<int64_t>(0), blockInfo.bRightNOffset),
                asc::te::make_shape(groupInfo.inputScaleK, blockN));

            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForAiv();
                }
                auto ubOut = blockEpilogue.GetConcatL0c2UbTensor(blockM, blockN);
                if constexpr (IsWeightNz<LayoutB>::value) {
                    // Mode 2 stores activation and gate in separate N halves.
                    blockMmad(gmBlockA, gmBlockBLeft, gmBlockBRight, gmBlockScaleA, gmBlockScaleBLeft,
                              gmBlockScaleBRight, ubOut, singleShape);
                } else {
                    blockMmad(gmBlockA, gmBlockBLeft, gmBlockScaleA, gmBlockScaleBLeft, ubOut, singleShape);
                }
                SyncAicToAiv();
            }
            isVecSetSyncCom_ = true;

            if ASCEND_IS_AIV {
                WaitForAic();
                const EpilogueBlockShape epiBlockShape = blockInfo.epilogueBlockShape;
                const EpilogueOutputOffsets outputOffsets{blockInfo.outputOffsets.outputOffset,
                                                          blockInfo.outputOffsets.outputScaleOffset};
                blockEpilogue(epiBlockShape, outputOffsets);
                SyncAivToAic();
            }
        } while (bs.GetNextSwigluBlock(blockInfo));
    }

    template <typename T>
    __aicore__ inline __gm__ T* GetTensorAddrFromList(uint16_t index, __gm__ T* tensorPtr)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> UINT64_ADDRESS_SHIFT);
        return reinterpret_cast<__gm__ T*>(*(retPtr + index));
    }

    BlockMmad blockMmad;
    BlockEpilogue blockEpilogue;
    ProblemShape problemShape_{};

    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ BiasType* biasPtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};

    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{0};
    bool isVecSetSyncCom_{false};
    bool singleW_{true};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
