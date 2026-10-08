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
 * \file block_mmad_a8w8_fixpipe_quant.h
 * \brief A8W8 quantized matmul block with fixpipe dequant (Tensor API)
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::SetFlag;
using AscendC::WaitFlag;

template <uint64_t FullLoadMode_, bool AtomicAdd_, class ScheduleType_, class GmmArrayPtr_, uint64_t NonContiguousType_,
          class AType_, class LayoutA_, class BTypeTuple_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_>
class BlockMmad<MatmulWithScaleFixpipeQuant<FullLoadMode_, AtomicAdd_, ScheduleType_, GmmArrayPtr_, NonContiguousType_>,
                AType_, LayoutA_, BTypeTuple_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using CType = CType_;
    using BiasType = BiasType_;
    using X2ScaleType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using L0CType = AscendC::Std::conditional_t<AscendC::IsSameType<AType, int8_t>::value, int32_t, float>;
    using DispatchPolicy = MatmulWithScaleFixpipeQuant<FullLoadMode_, AtomicAdd_, ScheduleType_, GmmArrayPtr_,
                                                       NonContiguousType_>;
    using WorkspaceType = L0CType;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_INT_SCALE = IsSameType<X2ScaleType, uint64_t>::value ||
                                         IsSameType<X2ScaleType, int64_t>::value;
    static constexpr bool
        IS_PERTENSOR_STREAMK = AscendC::Std::is_same_v<ScheduleType_, KernelQbmmPertensorMultiBlockStreamK>;
    static constexpr bool
        IS_GROUPED_FIXPIPE = AscendC::Std::is_same_v<ScheduleType_, KernelGroupedMmadWithScaleFixpipeQuant>;
    static constexpr bool STREAMK_BIAS_IN_MMAD = IsSameType<BiasType, int32_t>::value ||
                                                 (!IsSameType<AType, int8_t>::value && IS_INT_SCALE &&
                                                  IsSameType<BiasType, float>::value);
    static constexpr bool BIAS_IN_MMAD = !IS_PERTENSOR_STREAMK || STREAMK_BIAS_IN_MMAD;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
        // Keep new fields after the original GM addresses to preserve six-address aggregate initialization.
        uint64_t oriK{0};
        uint64_t kAL1{0};
        uint64_t kBL1{0};
        uint64_t l1BufNum{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        QuantMode quantMode{QuantMode::DEFAULT};
        bool isBias{false};
        bool enableL0cPingPong{false};
    };

    __aicore__ inline BlockMmad()
    {
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        SetFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        SetFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        WaitFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        WaitFlag<HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = params.oriK;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.isBias;
        l1BufNum_ = params.l1BufNum;
        enableL0cPingPong_ = params.enableL0cPingPong;
        const uint64_t kAL1 = params.kAL1;
        const uint64_t kBL1 = params.kBL1;
        uint64_t x2ScaleL1OneBuffer = 0UL;
        if constexpr (IS_GROUPED_FIXPIPE) {
            if (params.quantMode == QuantMode::PERCHANNEL_MODE || params.quantMode == QuantMode::PERGROUP_MODE) {
                x2ScaleL1OneBuffer = GetScaleL1Bytes(baseN_);
            }
        } else if (params.quantMode == QuantMode::PERCHANNEL_MODE) {
            x2ScaleL1OneBuffer = baseN_ * sizeof(uint64_t);
        }
        uint64_t biasL1OneBuffer = 0UL;
        if constexpr (IS_GROUPED_FIXPIPE) {
            biasL1OneBuffer = GetBiasL1Bytes(baseN_, isBias_);
        } else if (isBias_) {
            biasL1OneBuffer = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            kBL1_ = kBL1;
            kAL1_ = kBL1_;
            kL1_ = kBL1;
            kL1Iter_ = CeilDiv(k_, kL1_);
        } else {
            if (l1BufNum_ == DOUBLE_BUFFER_COUNT) {
                kAL1_ = kAL1;
                kBL1_ = kBL1;
                kAL1Iter_ = CeilDiv(k_, kAL1_);
                kBL1Iter_ = CeilDiv(k_, kBL1_);
                if (kAL1 == kBL1) {
                    kL1_ = kAL1;
                    kL1Iter_ = CeilDiv(k_, kL1_);
                }
            } else {
                kL1_ = Min(kAL1, kBL1);
                kL1Iter_ = CeilDiv(k_, kL1_);
                kAL1_ = kL1_;
                kBL1_ = kL1_;
            }
        }
        const uint64_t aL1OneBuffer = GetAL1BufferSize();
        const uint64_t bL1OneBuffer = GetBL1BufferSize();
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            InitNoneFullLoadL1BufferOffsets(aL1OneBuffer, bL1OneBuffer, x2ScaleL1OneBuffer, biasL1OneBuffer);
        } else {
            InitAFullLoadL1BufferOffsets(aL1OneBuffer, bL1OneBuffer, x2ScaleL1OneBuffer, biasL1OneBuffer);
        }
    }

    // Compatibility overload for existing callers of the original multi-argument Init interface.
    __aicore__ inline void Init(const ProblemShape& problemShape, const BlockShape& l0TileShape, const uint64_t& kAL1,
                                const uint64_t& kBL1, const uint64_t& l1BufNum, QuantMode quantMode, bool isBias,
                                bool enableL0cPingPong)
    {
        Params params{};
        params.oriK = static_cast<uint64_t>(asc::te::get<IDX_K_IDX>(problemShape));
        params.kAL1 = kAL1;
        params.kBL1 = kBL1;
        params.l1BufNum = l1BufNum;
        params.mL0 = static_cast<uint32_t>(asc::te::get<IDX_M_IDX>(l0TileShape));
        params.nL0 = static_cast<uint32_t>(asc::te::get<IDX_N_IDX>(l0TileShape));
        params.kL0 = static_cast<uint32_t>(asc::te::get<IDX_K_IDX>(l0TileShape));
        params.quantMode = quantMode;
        params.isBias = isBias;
        params.enableL0cPingPong = enableL0cPingPong;
        Init(params);
    }

    template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmB, TScale scaleGlobal, TensorBias gmBias, TensorC gmC,
                                      BlockShape singleShape)
    {
        Process(gmA, gmB, scaleGlobal, gmBias, gmC, gmC, singleShape, 0, false);
    }

    template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC,
              typename TensorWorkspace>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmB, TScale scaleGlobal, TensorBias gmBias, TensorC gmC,
                                      TensorWorkspace gmWorkspace, BlockShape singleShape, int64_t kCntIndex,
                                      bool isSkBlock)
    {
        static_assert(IS_PERTENSOR_STREAMK, "StreamK output requires the per-tensor StreamK schedule.");
        // Only the StreamK scheduler stores the current K span in singleShape[2]. The non-StreamK QBMM scheduler
        // uses that field for the M-tail split offset and must keep the full-K loop state initialized by Init().
        UpdateKLoop(asc::te::get<IDX_K_IDX>(singleShape));
        Process(gmA, gmB, scaleGlobal, gmBias, gmC, gmWorkspace, singleShape, kCntIndex, isSkBlock);
    }

private:
    template <typename TensorA, typename TensorB, typename TScale, typename TensorBias, typename TensorC,
              typename TensorWorkspace>
    __aicore__ inline void Process(TensorA gmA, TensorB gmB, TScale scaleGlobal, TensorBias gmBias, TensorC gmC,
                                   TensorWorkspace gmWorkspace, BlockShape singleShape, int64_t kCntIndex,
                                   bool isSkBlock)
    {
        uint64_t curML1 = asc::te::get<IDX_M_TILEIDX>(singleShape);
        uint64_t curNL1 = asc::te::get<IDX_N_TILEIDX>(singleShape);
        processBias_ = BIAS_IN_MMAD && isBias_ && (!IS_PERTENSOR_STREAMK || !isSkBlock || kCntIndex == 0);
        bool useFixpipeOutput = !IS_PERTENSOR_STREAMK || !isSkBlock;
        asc::te::mmad_params mmadParams;
        mmadParams.m = static_cast<uint16_t>(curML1);
        mmadParams.n = static_cast<uint16_t>(curNL1);
        constexpr uint64_t halfL0cSize = AscendC::TOTAL_L0C_SIZE >> 1;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * halfL0cSize;
        uint16_t scaleL1BufId = scaleLoopCnt_ & 1;
        uint16_t biasBufId = biasLoopCnt_ & 1;

        auto layoutL0C = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>(curML1,
                                                                                                            curNL1);
        auto c1Local = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, L0CType>(l0cOffset),
                                            layoutL0C);

        auto layoutX2L1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                     asc::te::layout_trait_default<uint64_t>>(1, curNL1);
        auto tensorX2L1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, uint64_t>(l1BufferX2ScaleOffset_[scaleL1BufId]), layoutX2L1);

        PrepareScale(scaleGlobal, tensorX2L1, useFixpipeOutput, scaleL1BufId);

        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                       asc::te::layout_trait_default<BiasType>>(
            1UL, processBias_ ? curNL1 : 1UL);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(processBias_ ? l1BufferBiasOffset_[biasBufId] : 0UL),
            layoutBiasL1);
        PrepareBias(gmBias, tensorBiasL1, biasBufId);

        RunMmad(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);

        CopyL0CToGm<TScale>(gmC, gmWorkspace, c1Local, tensorX2L1, isSkBlock);

        FinalizeProcess<TScale>(useFixpipeOutput, scaleL1BufId, biasBufId);
    }

    template <typename TScale, typename TensorScale>
    __aicore__ inline void PrepareScale(TScale scaleGlobal, TensorScale& tensorX2L1, bool useFixpipeOutput,
                                        uint16_t scaleL1BufId)
    {
        if (!useFixpipeOutput) {
            return;
        }
        if constexpr (IsSameType<TScale, uint64_t>::value) {
            scalarScale_ = scaleGlobal;
        } else if constexpr (asc::te::is_attr_tensor_v<TScale> && !IsSameType<CType, int32_t>::value) {
            WaitFlag<HardEvent::FIX_MTE2>(scaleL1BufId);
            auto copyGM2L1Scale = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            asc::te::copy(copyGM2L1Scale, tensorX2L1, scaleGlobal);
            SetFlag<HardEvent::MTE2_FIX>(scaleL1BufId);
            WaitFlag<HardEvent::MTE2_FIX>(scaleL1BufId);
        }
    }

    template <typename TensorBias, typename TensorBiasL1>
    __aicore__ inline void PrepareBias(TensorBias gmBias, TensorBiasL1& tensorBiasL1, uint16_t biasBufId)
    {
        if (!processBias_) {
            return;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
        SetFlag<HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
    }

    template <typename TensorA, typename TensorB, typename TensorL0C, typename TensorBiasL1>
    __aicore__ inline void RunMmad(TensorA gmA, TensorB gmB, asc::te::mmad_params& mmadParams, TensorL0C& c1Local,
                                   uint64_t curML1, uint64_t curNL1, TensorBiasL1& tensorBiasL1, uint16_t biasBufId)
    {
        if (kAL1_ == kBL1_) {
            IterateABL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        } else if (kAL1_ > kBL1_) {
            IterateAL1BL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        } else {
            IterateBL1AL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1, tensorBiasL1, biasBufId);
        }
    }

    template <typename TScale>
    __aicore__ inline void FinalizeProcess(bool useFixpipeOutput, uint16_t scaleL1BufId, uint16_t biasBufId)
    {
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
        if constexpr (asc::te::is_attr_tensor_v<TScale> && !IsSameType<CType, int32_t>::value) {
            if (useFixpipeOutput) {
                SetFlag<HardEvent::FIX_MTE2>(scaleL1BufId);
                scaleLoopCnt_++;
            }
        }
        if (processBias_) {
            SetFlag<HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
            biasLoopCnt_++;
        }
    }

public:
    // A new GM K slice requires the corresponding K-dependent loop state.
    __aicore__ inline void UpdateParamsForKSlice(uint64_t curK)
    {
        static_assert(IS_GROUPED_FIXPIPE, "UpdateParamsForKSlice is only supported by grouped fixpipe kernels.");
        UpdateKLoop(curK);
    }

private:
    __aicore__ inline void UpdateKLoop(uint64_t curK)
    {
        k_ = curK;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            kL1Iter_ = CeilDiv(k_, kL1_);
            if constexpr (IS_GROUPED_FIXPIPE) {
                // A full-load caching cannot span different per-group A slices.
                abL1LoopCnt_ = 0;
            }
        } else if (l1BufNum_ == DOUBLE_BUFFER_COUNT) {
            kAL1Iter_ = CeilDiv(k_, kAL1_);
            kBL1Iter_ = CeilDiv(k_, kBL1_);
            if (kAL1_ == kBL1_) {
                kL1Iter_ = CeilDiv(k_, kL1_);
            }
        } else {
            kL1Iter_ = CeilDiv(k_, kL1_);
        }
    }

    template <typename TScale, typename TensorC, typename TensorWorkspace, typename TensorL0C, typename TensorScale>
    __aicore__ inline void CopyL0CToGm(TensorC gmC, TensorWorkspace gmWorkspace, TensorL0C& tensorL0C,
                                       TensorScale& tensorX2L1, bool isSkBlock)
    {
        constexpr bool isScalarScale = IsSameType<TScale, uint64_t>::value;
        constexpr bool isPerChannelScale = asc::te::is_attr_tensor_v<TScale>;
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        if constexpr (IS_PERTENSOR_STREAMK) {
            if (isSkBlock) {
                asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)),
                              gmWorkspace, tensorL0C);
                return;
            }
        }
        if constexpr (IsSameType<CType, int32_t>::value) {
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC,
                          tensorL0C);
        } else if constexpr (isScalarScale) {
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC,
                          tensorL0C, scalarScale_);
        } else if constexpr (isPerChannelScale) {
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC,
                          tensorL0C, tensorX2L1);
        }
    }

    static constexpr uint64_t C0_SIZE = asc::te::c0_element<AType>;
    static constexpr uint16_t BIAS_BUFFER_FLAG_0 = 4;
    static constexpr uint16_t BIAS_BUFFER_FLAG_1 = 5;
    static constexpr uint16_t X2_SCALE_BUFFER_FLAG_0 = 0;
    static constexpr uint16_t X2_SCALE_BUFFER_FLAG_1 = 1;
    static constexpr uint32_t L0_INIT_MODE_ABL1 = 0U;
    static constexpr uint32_t L0_INIT_MODE_SPLIT = 1U;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;

    __aicore__ inline static uint64_t GetAL1Bytes(uint64_t baseM, uint64_t kAL1)
    {
        return (TRANS_A ? CeilAlign(baseM, C0_SIZE) * CeilAlign(kAL1, BLOCK_CUBE) :
                          CeilAlign(baseM, BLOCK_CUBE) * CeilAlign(kAL1, C0_SIZE)) *
               sizeof(AType);
    }

    __aicore__ inline static uint64_t GetBL1Bytes(uint64_t baseN, uint64_t kBL1)
    {
        return (TRANS_B ? CeilAlign(kBL1, C0_SIZE) * CeilAlign(baseN, BLOCK_CUBE) :
                          CeilAlign(kBL1, BLOCK_CUBE) * CeilAlign(baseN, C0_SIZE)) *
               sizeof(BType);
    }

    __aicore__ inline uint64_t GetAL1BufferSize() const
    {
        const uint64_t bufferK = DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE ? k_ : kAL1_;
        if constexpr (IS_GROUPED_FIXPIPE) {
            return GetAL1Bytes(baseM_, bufferK);
        } else if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            const uint64_t alignedM = CeilAlign(baseM_, TRANS_A ? C0_SIZE : BLOCK_CUBE);
            const uint64_t alignedK = CeilAlign(bufferK, TRANS_A ? BLOCK_CUBE : C0_SIZE);
            return alignedM * alignedK;
        } else {
            return baseM_ * bufferK;
        }
    }

    __aicore__ inline uint64_t GetBL1BufferSize() const
    {
        if constexpr (IS_GROUPED_FIXPIPE) {
            return GetBL1Bytes(baseN_, kBL1_);
        } else {
            return baseN_ * kBL1_;
        }
    }

    __aicore__ inline static uint64_t GetScaleL1Bytes(uint64_t baseN) { return Align32(baseN * sizeof(uint64_t)); }

    __aicore__ inline static uint64_t GetBiasL1Bytes(uint64_t baseN, bool hasBias)
    {
        return hasBias ? Align32(baseN * sizeof(BiasType)) : 0UL;
    }

    template <typename C1Tensor, typename TensorAL0, typename TensorBL0, typename TensorBt>
    __aicore__ inline void ExecuteMmad(asc::te::mmad_params& params, C1Tensor& c1Local, TensorAL0 l0aLocal,
                                       TensorBL0 l0bLocal, TensorBt tensorBt, bool needBias)
    {
        using MmadAtomT = asc::te::mmad_atom<
            asc::te::mmad_traits<asc::te::mmad_operation, asc::te::mmad_trait_default>>;
        if constexpr (BIAS_IN_MMAD) {
            if (needBias) {
                asc::te::mmad(MmadAtomT{}.with(params), c1Local, l0aLocal, l0bLocal, tensorBt);
                return;
            }
        }
        asc::te::mmad(MmadAtomT{}.with(params), c1Local, l0aLocal, l0bLocal);
    }

    __aicore__ inline uint16_t GetL1BufferId(uint64_t loopCnt)
    {
        // Three-buffer mode is not power-of-two, so bitmask wrapping would skip buffer 1.
        return static_cast<uint16_t>(l1BufNum_ == TRIPLE_BUFFER_COUNT ? loopCnt % TRIPLE_BUFFER_COUNT :
                                                                        (loopCnt & (l1BufNum_ - 1)));
    }

    __aicore__ inline void InitNoneFullLoadL1BufferOffsets(uint64_t aL1OneBuffer, uint64_t bL1OneBuffer,
                                                           uint64_t x2ScaleL1OneBuffer, uint64_t biasL1OneBuffer)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        if (l1BufNum_ == TRIPLE_BUFFER_COUNT) {
            // L1 space: A0|A2|B0|B2|scale0|bias0|...|A1|B1|scale1|bias1
            l1BufferAOffset_[0] = 0UL;
            l1BufferAOffset_[THIRD_BUFFER_ID] = l1BufferAOffset_[0] + aL1OneBuffer;
            l1BufferBOffset_[0] = l1BufferAOffset_[THIRD_BUFFER_ID] + aL1OneBuffer;
            l1BufferBOffset_[THIRD_BUFFER_ID] = l1BufferBOffset_[0] + bL1OneBuffer;
            l1BufferX2ScaleOffset_[0] = l1BufferBOffset_[THIRD_BUFFER_ID] + bL1OneBuffer;
            l1BufferBiasOffset_[0] = l1BufferX2ScaleOffset_[0] + x2ScaleL1OneBuffer;
            l1BufferAOffset_[1] = Max(halfL1Size, l1BufferBiasOffset_[0] + biasL1OneBuffer);
            l1BufferBOffset_[1] = l1BufferAOffset_[1] + aL1OneBuffer;
            l1BufferX2ScaleOffset_[1] = l1BufferBOffset_[1] + bL1OneBuffer;
            l1BufferBiasOffset_[1] = l1BufferX2ScaleOffset_[1] + x2ScaleL1OneBuffer;
            return;
        }
        if constexpr (IS_GROUPED_FIXPIPE) {
            if (l1BufNum_ == DOUBLE_BUFFER_COUNT) {
                // Each fixed GMM stage occupies one complete half of L1.
                for (uint16_t bufferId = 0; bufferId < DOUBLE_BUFFER_COUNT; ++bufferId) {
                    const uint64_t stageBase = halfL1Size * bufferId;
                    l1BufferAOffset_[bufferId] = stageBase;
                    l1BufferBOffset_[bufferId] = stageBase + aL1OneBuffer;
                    l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer;
                    l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer;
                }
                return;
            }
        }
        const uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            const uint64_t l1Offset = halfL1Size * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer * l1HalfBufNum + bL1OneBuffer * (bufferId >> 1);
        }
        for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer * l1HalfBufNum;
            l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer;
        }
    }

    __aicore__ inline void InitAFullLoadL1BufferOffsets(uint64_t aL1OneBuffer, uint64_t bL1OneBuffer,
                                                        uint64_t x2ScaleL1OneBuffer, uint64_t biasL1OneBuffer)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        if (l1BufNum_ == TRIPLE_BUFFER_COUNT) {
            // L1 space: B0|B2|scale0|bias0|A|...|B1|scale1|bias1
            l1BufferBOffset_[0] = 0UL;
            l1BufferBOffset_[THIRD_BUFFER_ID] = l1BufferBOffset_[0] + bL1OneBuffer;
            l1BufferX2ScaleOffset_[0] = l1BufferBOffset_[THIRD_BUFFER_ID] + bL1OneBuffer;
            l1BufferBiasOffset_[0] = l1BufferX2ScaleOffset_[0] + x2ScaleL1OneBuffer;
            l1BufferAOffset_[0] = l1BufferBiasOffset_[0] + biasL1OneBuffer;
            l1BufferBOffset_[1] = Max(halfL1Size, l1BufferAOffset_[0] + aL1OneBuffer);
            l1BufferX2ScaleOffset_[1] = l1BufferBOffset_[1] + bL1OneBuffer;
            l1BufferBiasOffset_[1] = l1BufferX2ScaleOffset_[1] + x2ScaleL1OneBuffer;
            return;
        }
        const uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        l1BufferAOffset_[0] = bL1OneBuffer * l1HalfBufNum + x2ScaleL1OneBuffer + biasL1OneBuffer;
        const uint64_t b1Offset = l1BufferAOffset_[0] + aL1OneBuffer >= halfL1Size ?
                                      l1BufferAOffset_[0] + aL1OneBuffer :
                                      halfL1Size;
        for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer * (bufferId >> 1);
        }
        for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer * l1HalfBufNum;
            l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer;
        }
    }

    template <typename TensorAL1, typename TensorBL1, typename TensorBiasL1, typename C1Tensor>
    __aicore__ inline void Iterate(asc::te::mmad_params& mmadParams, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
                                   const TensorBiasL1& tensorBiasL1, C1Tensor& c1Local, uint64_t curML1,
                                   uint64_t curNL1, uint64_t curInnerKL1, uint64_t aKPrefix, uint64_t bKPrefix,
                                   bool isL1LastRound, uint32_t initMode, uint64_t kL1OuterIdx, uint64_t kL1InnerIdx,
                                   uint16_t biasBufId)
    {
        const uint64_t baseK = baseK_;
        const uint64_t kL0Iter = CeilDiv(curInnerKL1, baseK);
        const bool needBiasInL1 = processBias_ && kL1OuterIdx == 0 && kL1InnerIdx == 0;
        const bool initCMatrixInL1 = kL1OuterIdx == 0 && (initMode == L0_INIT_MODE_ABL1 || kL1InnerIdx == 0);
        constexpr uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE >> 1;
        uint64_t l0PingPong = l0PingPong_;

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, Align16(curNL1));
        const uint64_t biasBtOffset = baseN_ * biasBufId * sizeof(BiasType);
        auto tensorBt = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, BiasType>(biasBtOffset),
                                             layoutBt);

        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            const uint64_t kL0Offset = static_cast<uint64_t>(iter1) * baseK;
            const uint64_t curKL0 = Min(baseK, curInnerKL1 - kL0Offset);
            const uint64_t l0PingPongId = l0PingPong & 1;
            const uint64_t l0Offset = halfL0Size * l0PingPongId;

            auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
                curML1, curKL0);
            auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
                curKL0, curNL1);
            auto l0aLocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                                                 layoutAL0);
            auto l0bLocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset),
                                                 layoutBL0);

            const uint16_t mte1WaitMFlag = static_cast<uint16_t>(l0PingPongId + M_MTE1_FLAG_0);
            WaitFlag<HardEvent::M_MTE1>(mte1WaitMFlag);

            auto aL1Sub = tensorAL1.slice(asc::te::make_coord(0, aKPrefix + kL0Offset),
                                          asc::te::make_shape(curML1, curKL0));
            asc::te::copy(copyL12L0A, l0aLocal, aL1Sub);

            auto bL1Sub = tensorBL1.slice(asc::te::make_coord(bKPrefix + kL0Offset, 0),
                                          asc::te::make_shape(curKL0, curNL1));
            asc::te::copy(copyL12L0B, l0bLocal, bL1Sub);

            const bool needBias = needBiasInL1 && iter1 == 0;
            if (needBias) {
                auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
                asc::te::copy(copyL12BT, tensorBt, tensorBiasL1);
            }

            SetFlag<HardEvent::MTE1_M>(l0PingPongId);
            WaitFlag<HardEvent::MTE1_M>(l0PingPongId);

            mmadParams.k = static_cast<uint16_t>(curKL0);
            mmadParams.unit_flag = (isL1LastRound && iter1 + 1 == kL0Iter) ? asc::te::unit_flag_mode::enable_update :
                                                                             asc::te::unit_flag_mode::enable_keep;
            mmadParams.init_with_zero = initCMatrixInL1 && iter1 == 0;

            ExecuteMmad(mmadParams, c1Local, l0aLocal, l0bLocal, tensorBt, needBias);

            SetFlag<HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong++;
        }
        l0PingPong_ = l0PingPong;
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateABL1(TensorA gmA, TensorB gmB, asc::te::mmad_params& mmadParams, C1Tensor& c1Local,
                                       uint64_t curML1, uint64_t curNL1, const TensorBiasL1& tensorBiasL1,
                                       uint16_t biasBufId)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        const uint64_t k = k_;
        const uint64_t kL1 = kL1_;
        const uint64_t kAL1 = kAL1_;
        const uint64_t kBL1 = kBL1_;
        const uint64_t kL1Iter = kL1Iter_;
        const uint64_t aL1KStride = CeilAlign(curML1, TRANS_A ? C0_SIZE : BLOCK_CUBE);
        uint64_t abL1LoopCnt = abL1LoopCnt_;

        for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
            const uint64_t curKL1 = (iter0 == kL1Iter - 1) ? (k - iter0 * kL1) : kL1;
            const uint16_t l1BufId = GetL1BufferId(abL1LoopCnt);
            WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t offsetAL1;
            if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
                offsetAL1 = l1BufferAOffset_[0] + iter0 * kL1 * aL1KStride;
            } else {
                offsetAL1 = l1BufferAOffset_[l1BufId];
            }
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
            auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAL1),
                                                  layoutAL1);
            if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
                auto gmTileA = gmA.slice(asc::te::make_coord(0UL, iter0 * kAL1), asc::te::make_shape(curML1, curKL1));
                asc::te::copy(copyGM2L1, tensorAL1, gmTileA);
            } else {
                if (abL1LoopCnt < kL1Iter) {
                    auto gmTileA = gmA.slice(asc::te::make_coord(0UL, iter0 * kL1),
                                             asc::te::make_shape(curML1, curKL1));
                    asc::te::copy(copyGM2L1, tensorAL1, gmTileA);
                }
            }

            const uint64_t offsetBL1 = l1BufferBOffset_[l1BufId];
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
            auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBL1),
                                                  layoutBL1);
            auto gmTileB = gmB.slice(asc::te::make_coord(iter0 * kBL1, 0UL), asc::te::make_shape(curKL1, curNL1));
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKL1, 0UL, 0UL,
                    iter0 == kL1Iter - 1, L0_INIT_MODE_ABL1, iter0, 0UL, biasBufId);

            SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt++;
        }
        abL1LoopCnt_ = abL1LoopCnt;
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateAL1BL1(TensorA gmA, TensorB gmB, asc::te::mmad_params& mmadParams, C1Tensor& c1Local,
                                         uint64_t curML1, uint64_t curNL1, const TensorBiasL1& tensorBiasL1,
                                         uint16_t biasBufId)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

        for (uint64_t kAIter = 0; kAIter < kAL1Iter_; ++kAIter) {
            uint64_t curKAL1 = (kAIter == kAL1Iter_ - 1) ? (k_ - kAIter * kAL1_) : kAL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongId_);

            uint64_t offsetAL1 = l1BufferAOffset_[aPingPongId_];
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
            auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAL1),
                                                  layoutAL1);
            auto gmTileA = gmA.slice(asc::te::make_coord(0UL, kAIter * kAL1_), asc::te::make_shape(curML1, curKAL1));
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongId_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongId_);

            uint64_t kBL1IterLocal = CeilDiv(curKAL1, kBL1_);
            for (uint64_t kBIter = 0; kBIter < kBL1IterLocal; ++kBIter) {
                uint64_t curKBL1 = (kBIter == kBL1IterLocal - 1) ? (curKAL1 - kBIter * kBL1_) : kBL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongId_);

                uint64_t offsetBL1 = l1BufferBOffset_[bPingPongId_];
                auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
                auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBL1),
                                                      layoutBL1);
                uint64_t bKStart = kAIter * kAL1_ + kBIter * kBL1_;
                auto gmTileB = gmB.slice(asc::te::make_coord(bKStart, 0UL), asc::te::make_shape(curKBL1, curNL1));
                asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongId_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongId_);

                uint64_t aKPrefix = kBIter * kBL1_;
                Iterate(mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKBL1, aKPrefix, 0UL,
                        (kAIter == kAL1Iter_ - 1) && (kBIter == kBL1IterLocal - 1), L0_INIT_MODE_SPLIT, kAIter, kBIter,
                        biasBufId);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongId_);
                bPingPongId_ = bPingPongId_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongId_);
            aPingPongId_ = aPingPongId_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    template <typename TensorA, typename TensorB, typename C1Tensor, typename TensorBiasL1>
    __aicore__ inline void IterateBL1AL1(TensorA gmA, TensorB gmB, asc::te::mmad_params& mmadParams, C1Tensor& c1Local,
                                         uint64_t curML1, uint64_t curNL1, const TensorBiasL1& tensorBiasL1,
                                         uint16_t biasBufId)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

        for (uint64_t kBIter = 0; kBIter < kBL1Iter_; ++kBIter) {
            uint64_t curKBL1 = (kBIter == kBL1Iter_ - 1) ? (k_ - kBIter * kBL1_) : kBL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongId_);

            uint64_t offsetBL1 = l1BufferBOffset_[bPingPongId_];
            auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
            auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBL1),
                                                  layoutBL1);
            auto gmTileB = gmB.slice(asc::te::make_coord(kBIter * kBL1_, 0UL), asc::te::make_shape(curKBL1, curNL1));
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongId_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongId_);

            uint64_t kAL1IterLocal = CeilDiv(curKBL1, kAL1_);
            for (uint64_t kAIter = 0; kAIter < kAL1IterLocal; ++kAIter) {
                uint64_t curKAL1 = (kAIter == kAL1IterLocal - 1) ? (curKBL1 - kAIter * kAL1_) : kAL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongId_);

                uint64_t offsetAL1 = l1BufferAOffset_[aPingPongId_];
                auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
                auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAL1),
                                                      layoutAL1);
                uint64_t aKStart = kBIter * kBL1_ + kAIter * kAL1_;
                auto gmTileA = gmA.slice(asc::te::make_coord(0UL, aKStart), asc::te::make_shape(curML1, curKAL1));
                asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongId_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongId_);

                uint64_t bKPrefix = kAIter * kAL1_;
                Iterate(mmadParams, tensorAL1, tensorBL1, tensorBiasL1, c1Local, curML1, curNL1, curKAL1, 0UL, bKPrefix,
                        (kBIter == kBL1Iter_ - 1) && (kAIter == kAL1IterLocal - 1), L0_INIT_MODE_SPLIT, kBIter, kAIter,
                        biasBufId);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongId_);
                aPingPongId_ = aPingPongId_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongId_);
            bPingPongId_ = bPingPongId_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    uint64_t k_{0};
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kAL1Iter_{0};
    uint64_t kBL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint16_t aPingPongId_{0};
    uint16_t bPingPongId_{0};
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t biasLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t l1BufferAOffset_[QUADRUPLE_BUFFER_COUNT] = {0UL};
    uint64_t l1BufferBOffset_[QUADRUPLE_BUFFER_COUNT] = {0UL};
    uint64_t l1BufferX2ScaleOffset_[DOUBLE_BUFFER_COUNT] = {0UL};
    uint64_t l1BufferBiasOffset_[DOUBLE_BUFFER_COUNT] = {0UL};
    uint64_t scalarScale_{0UL};
    bool enableL0cPingPong_{false};
    bool isBias_{false};
    bool processBias_{false};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
