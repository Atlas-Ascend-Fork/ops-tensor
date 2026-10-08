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
 * \file kernel_tqbmm.h
 * \brief TQBMM tensor-api kernels. All variants share the TQBMM 3D batch loop + PERM_X1
 *        traversal and differ only in the BlockMmad / BlockEpilogue / schedule type:
 *        - KernelMmadMultiBlockTQBMM            : MX quant, AIC-only (per-group e8m0 scales)
 *        - KernelMmadMultiBlockTQBMMFixpipeQuant: HIFP8/FP8 per-tensor fixpipe quant, AIC-only
 *        - KernelMmadMultiBlockTQBMMMx          : FP8 per-token dequant, AIC + AIV epilogue
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

namespace TqbmmDetail {

using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

// One tile of the TQBMM batch traversal, consumed by the kernels' ProcessBlock.
struct BlockTask {
    BlockCoord coord;
    BlockShape shape;
    int64_t mPos{0};
    int64_t nPos{0};
    uint64_t batchIdx{0};
    int64_t baseM{0};
    int64_t baseN{0};
};

template <typename Ptn, typename T, typename IntType>
__aicore__ inline auto MakeBatchLayout(uint64_t batch, uint64_t row, uint64_t col, uint64_t batchStride,
                                       uint64_t rowStride)
{
    return asc::te::make_pattern_layout<Ptn, asc::te::layout_trait<T, IntType>>(
        asc::te::make_shape(batch, asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, row),
                                                       asc::te::make_shape(asc::te::_1{}, col))),
        asc::te::make_stride(batchStride, asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, rowStride),
                                                               asc::te::make_stride(asc::te::_0{}, asc::te::_1{}))));
}

// A: perm_x1=[1,0,2] 时物理 [M, B, K]（m 非连续），否则 [B, M, K] 连续。
template <class LayoutA, class AType, int64_t C0Size>
__aicore__ inline auto MakeBatchLayoutA(uint64_t batch, uint64_t m, uint64_t k, bool isPermX1)
{
    const uint64_t aBatchStride = isPermX1 ? k : (m * k);
    const uint64_t aMStride = isPermX1 ? (batch * k) : k;
    return MakeBatchLayout<LayoutA, AType, AscendC::Std::Int<C0Size>>(batch, m, k, aBatchStride, aMStride);
}

// C: 物理 [M, B, N]（B 内嵌 M 平面，B 步长 = N，M 步长 = B * N）。
template <class LayoutC, class CType>
__aicore__ inline auto MakeBatchLayoutC(uint64_t batch, uint64_t m, uint64_t n)
{
    return MakeBatchLayout<LayoutC, CType, AscendC::Std::Int<asc::te::c0_element<CType>>>(batch, m, n, n, batch * n);
}

template <class BlockMmad>
__aicore__ inline bool IsPermX1()
{
    return BlockMmad::DispatchPolicy::NON_CONTIGUOUS_TYPE ==
           static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1);
}

template <typename TensorB>
__aicore__ inline void SetBL2Cache(uint64_t problemM, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN,
                                   uint32_t bMustHitL2, bool transB, uint64_t cacheLineAlignMask, TensorB& gmB)
{
    const bool isCurrentNAligned = transB || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 = bMustHitL2 == 0U && currentBasicBlockM >= problemM && isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

// 三个 TQBMM 特化共享的 3D batch 遍历：per-batch tail tile 调度 + block 循环。
// KernelT 需提供：void ProcessBlock(<tensors...>, const BlockTask&)。
template <QuantMode AQuantMode, QuantMode BQuantMode, bool WeightNz, class KernelT, class BlockSchedulerT,
          class SchParamsT, class... TensorTs>
__aicore__ inline void TraverseBatchTiles(KernelT& kernel, BlockSchedulerT& bs, const SchParamsT& schParams,
                                          uint64_t batch, bool& needUpdateTail, TensorTs&... tensors)
{
    const int64_t singleBatchBlockCnt = bs.GetTotalCnt();
    const int64_t coreNums = AscendC::GetBlockNum();
    const uint64_t tailRoundStart = coreNums > 0 ?
                                        (static_cast<uint64_t>(singleBatchBlockCnt) * batch / coreNums) * coreNums :
                                        0UL;

    for (uint64_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
        const bool isTailRound = (batchIdx + 1) * singleBatchBlockCnt > tailRoundStart;
        if (needUpdateTail || (isTailRound && ((bs.GetEndBlockIdx() + 1) + (batch - batchIdx - 1) * bs.GetTotalCnt()) *
                                                      schParams.mTailTile * schParams.nTailTile <=
                                                  coreNums)) {
            needUpdateTail = true;
            bs.UpdateTailTile(schParams.mTailTile, schParams.nTailTile);
        }

        BlockCoord blockCoord;
        int64_t mPos = 0L;
        int64_t nPos = 0L;
        while (bs.GetTileIdx(blockCoord)) {
            BlockShape singleShape = bs.template GetBlockShape<AQuantMode, BQuantMode, WeightNz>(blockCoord);
            const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
            const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
            if (baseM <= 0 || baseN <= 0) {
                break;
            }
            bs.GetTileCoord(blockCoord, mPos, nPos);
            const BlockTask task{blockCoord, singleShape, mPos, nPos, batchIdx, baseM, baseN};
            kernel.ProcessBlock(tensors..., task);
        }
        bs.UpdateNextBatchBlockRoundParams();
    }
}

} // namespace TqbmmDetail

// ============================ MX quant (AIC only) ============================

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTQBMM, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static_assert(AscendC::Std::is_one_of_v<AscendC::Std::tuple<AType, BType, CType>,
                                            AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t>,
                                            AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e4m3fn_t, half>,
                                            AscendC::Std::tuple<fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t>,
                                            AscendC::Std::tuple<fp4x2_e2m1_t, fp4x2_e2m1_t, half>>,
                  "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be NZLayoutPtn or ZNLayoutPtn");

    struct TQBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
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
        TQBMMTiling tqbmmParams;
        Params() = default;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicAdd<float>();
        }
        Init(params);

        const auto& problemShape = params.problemShape;
        const auto& tqbmmParams = params.tqbmmParams;

        BlockScheduler bs(problemShape, params.schParams);

        const BlockShape l0BlockShape{static_cast<int64_t>(tqbmmParams.baseM), static_cast<int64_t>(tqbmmParams.baseN),
                                      static_cast<int64_t>(tqbmmParams.baseK), 0};
        mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, tqbmmParams.dbL0C > 1);

        auto layoutA = TqbmmDetail::MakeBatchLayoutA<LayoutA, AType, C0_SIZE>(batch_, m_, k_, isPermX1_);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = TqbmmDetail::MakeBatchLayoutC<LayoutC, CType>(batch_, m_, n_);

        const uint64_t scaleABatchStride = isPermX1_ ? scaleKLen_ : m_ * scaleKLen_;
        const uint64_t scaleAMStride = isPermX1_ ? batch_ * scaleKLen_ : scaleKLen_;
        auto layoutScaleA = TqbmmDetail::MakeBatchLayout<ScaleALayoutPtn, AscendC::fp8_e8m0_t,
                                                         AscendC::Std::Int<SCALE_C0>>(batch_, m_, scaleKLen_,
                                                                                      scaleABatchStride, scaleAMStride);
        auto layoutScaleB = MakeLayoutScaleB{}(batch_, scaleKLen_, n_);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(n_));

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
        auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        TqbmmDetail::TraverseBatchTiles<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, WEIGHT_NZ>(
            *this, bs, params.schParams, batch_, needUpdateTail_, gmA, gmB, gmScaleA, gmScaleB, gmC, gmBias);

        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }

    // 每个 (batch, m, n) block 的处理体，由 TqbmmDetail::TraverseBatchTiles 调用。
    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorC,
              typename TensorBias>
    __aicore__ inline void ProcessBlock(TensorA& gmA, TensorB& gmB, TensorScaleA& gmScaleA, TensorScaleB& gmScaleB,
                                        TensorC& gmC, TensorBias& gmBias, const TqbmmDetail::BlockTask& task)
    {
        constexpr int64_t kPos = 0L;
        if constexpr (IS_ATOMIC_ADD) {
            gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        TqbmmDetail::SetBL2Cache(m_, task.baseM, task.baseN, bMustHitL2_, TRANS_B, IsFp4<BType>() ? 0xffUL : 0x7fUL,
                                 gmB);

        auto subTensorA = gmA.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, kPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(task.baseM, static_cast<int64_t>(k_))));
        auto gmBlockA = asc::te::squeeze<0>(subTensorA);

        auto subScaleA = gmScaleA.slice(
            asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, kPos)),
            asc::te::make_shape(1L, asc::te::make_shape(task.baseM, static_cast<int64_t>(scaleKLen_))));
        auto gmBlockScaleA = asc::te::squeeze<0>(subScaleA);

        auto subTensorB = gmB.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(kPos, task.nPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(k_), task.baseN)));
        auto gmBlockB = asc::te::squeeze<0>(subTensorB);

        auto subScaleB = gmScaleB.slice(
            asc::te::make_coord(task.batchIdx, asc::te::make_coord(kPos, task.nPos)),
            asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(scaleKLen_), task.baseN)));
        auto gmBlockScaleB = asc::te::squeeze<0>(subScaleB);

        auto subTensorC = gmC.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, task.nPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(task.baseM, task.baseN)));
        auto gmBlockC = asc::te::squeeze<0>(subTensorC);

        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, task.nPos), asc::te::make_shape(1L, task.baseN));

        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, task.shape);
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;

    using ScaleALayoutPtn = asc::te::scalea_nd_layout_ptn;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;

    __aicore__ inline void Init(const Params& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        isBias_ = false;
        isPermX1_ = TqbmmDetail::IsPermX1<BlockMmad>();
        bMustHitL2_ = params.tqbmmParams.bMustHitL2;
        scaleKLen_ = Blaze::Gemm::CeilDiv(k_, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    }

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    BlockMmad mmadOp_;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    uint64_t scaleKLen_{0};
    uint32_t bMustHitL2_{1U};
    bool isBias_{false};
    bool isPermX1_{false};
    bool needUpdateTail_{false};
};

// ==================== HIFP8/FP8 fixpipe quant (AIC only) ====================

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTQBMMFixpipeQuant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using X2ScaleType = typename BlockMmad::X2ScaleType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static_assert(AscendC::Std::is_one_of_v<AscendC::Std::tuple<AType, BType, CType, BiasType>,
                                            AscendC::Std::tuple<hifloat8_t, hifloat8_t, half, float>,
                                            AscendC::Std::tuple<hifloat8_t, hifloat8_t, bfloat16_t, float>,
                                            AscendC::Std::tuple<hifloat8_t, hifloat8_t, float, float>,
                                            AscendC::Std::tuple<hifloat8_t, hifloat8_t, hifloat8_t, float>>,
                  "Unsupported (AType, BType, CType, BiasType) combination for TQBMM fixpipe quant");
    static_assert(AscendC::Std::is_one_of_v<X2ScaleType, uint64_t, int64_t, bfloat16_t, float>,
                  "TQBMM fixpipe quant only supports uint64_t/int64_t/bfloat16_t/float ScaleGmType.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be NZLayoutPtn or ZNLayoutPtn");

    struct TQBMMParams {
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
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
        BlockSchedulerParams schParams;
        TQBMMParams tqbmmParams;
        Params() = default;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicAdd<float>();
        }
        Init(params);

        const auto& problemShape = params.problemShape;
        const auto& tqbmmParams = params.tqbmmParams;

        BlockScheduler bs(problemShape, params.schParams);

        const BlockShape l0BlockShape{static_cast<int64_t>(tqbmmParams.baseM), static_cast<int64_t>(tqbmmParams.baseN),
                                      static_cast<int64_t>(tqbmmParams.baseK), 0};
        BlockMmadParams blockMmadParams = params.mmadParams;
        blockMmadParams.oriK = static_cast<uint64_t>(asc::te::get<MNK_K>(problemShape));
        blockMmadParams.kAL1 = tqbmmParams.kAL1;
        blockMmadParams.kBL1 = tqbmmParams.kBL1;
        blockMmadParams.l1BufNum = tqbmmParams.nBufferNum;
        blockMmadParams.mL0 = tqbmmParams.baseM;
        blockMmadParams.nL0 = tqbmmParams.baseN;
        blockMmadParams.kL0 = tqbmmParams.baseK;
        blockMmadParams.quantMode = static_cast<QuantMode>(tqbmmParams.x2QuantMode);
        blockMmadParams.isBias = tqbmmParams.isBias != 0U;
        blockMmadParams.enableL0cPingPong = tqbmmParams.dbL0C > 1;
        mmadOp_.Init(blockMmadParams);

        auto layoutA = TqbmmDetail::MakeBatchLayoutA<LayoutA, AType, C0_SIZE>(batch_, m_, k_, isPermX1_);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = TqbmmDetail::MakeBatchLayoutC<LayoutC, CType>(batch_, m_, n_);
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(n_));
        auto layoutScale = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
                                                      asc::te::layout_trait_default<X2ScaleType>>(1, n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
        auto gmScale = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleGmAddr_), layoutScale);

        TqbmmDetail::TraverseBatchTiles<QuantMode::DEFAULT, QuantMode::DEFAULT, WEIGHT_NZ>(
            *this, bs, params.schParams, batch_, needUpdateTail_, gmA, gmB, gmScale, gmC, gmBias);

        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }

    // 每个 (batch, m, n) block 的处理体，由 TqbmmDetail::TraverseBatchTiles 调用。
    template <typename TensorA, typename TensorB, typename TensorScale, typename TensorC, typename TensorBias>
    __aicore__ inline void ProcessBlock(TensorA& gmA, TensorB& gmB, TensorScale& gmScale, TensorC& gmC,
                                        TensorBias& gmBias, const TqbmmDetail::BlockTask& task)
    {
        constexpr int64_t kPos = 0L;
        if constexpr (IS_ATOMIC_ADD) {
            gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        TqbmmDetail::SetBL2Cache(m_, task.baseM, task.baseN, bMustHitL2_, TRANS_B, 0x7fUL, gmB);

        auto subTensorA = gmA.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, kPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(task.baseM, static_cast<int64_t>(k_))));
        auto gmBlockA = asc::te::squeeze<0>(subTensorA);

        auto subTensorB = gmB.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(kPos, task.nPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(k_), task.baseN)));
        auto gmBlockB = asc::te::squeeze<0>(subTensorB);

        auto subTensorC = gmC.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, task.nPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(task.baseM, task.baseN)));
        auto gmBlockC = asc::te::squeeze<0>(subTensorC);

        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, task.nPos), asc::te::make_shape(1L, task.baseN));

        if (isPerChannel_) {
            auto gmBlockScale = gmScale.slice(asc::te::make_coord(0L, task.nPos), asc::te::make_shape(1L, task.baseN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScale, gmBlockBias, gmBlockC, task.shape);
        } else {
            mmadOp_(gmBlockA, gmBlockB, scaleScalar_, gmBlockBias, gmBlockC, task.shape);
        }
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;
    static constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
    static constexpr uint32_t LEFT_SHIFT_16 = 16;

    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;

    template <typename T>
    __aicore__ inline T ReadGmScalar(GM_ADDR gmAddr)
    {
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1L, 1L);
        auto tensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ T*>(gmAddr)), layout);
        return tensor[asc::te::make_coord(0L, 0L)];
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        isBias_ = params.tqbmmParams.isBias != 0U;
        isPermX1_ = TqbmmDetail::IsPermX1<BlockMmad>();
        bMustHitL2_ = params.tqbmmParams.bMustHitL2;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);

        const auto x2QuantMode = static_cast<QuantMode>(params.tqbmmParams.x2QuantMode);
        const auto x1QuantMode = static_cast<QuantMode>(params.tqbmmParams.x1QuantMode);
        isPerChannel_ = (x2QuantMode == QuantMode::PERCHANNEL_MODE);

        if (isPerChannel_) {
            scaleGmAddr_ = reinterpret_cast<__gm__ X2ScaleType*>(params.mmadParams.scaleBGmAddr);
        } else if (x1QuantMode == QuantMode::PERTENSOR_MODE) {
            float deqScale = ReadGmScalar<float>(params.mmadParams.scaleAGmAddr) *
                             ReadGmScalar<float>(params.mmadParams.scaleBGmAddr);
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
            scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        } else if (x2QuantMode == QuantMode::PERTENSOR_MODE) {
            if constexpr (AscendC::IsSameType<X2ScaleType, uint64_t>::value ||
                          AscendC::IsSameType<X2ScaleType, int64_t>::value) {
                scaleScalar_ = ReadGmScalar<uint64_t>(params.mmadParams.scaleBGmAddr);
            } else if constexpr (AscendC::IsSameType<X2ScaleType, bfloat16_t>::value) {
                uint16_t uint16Scale = ReadGmScalar<uint16_t>(params.mmadParams.scaleBGmAddr);
                uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << LEFT_SHIFT_16);
                scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
            } else {
                uint32_t uint32Scale = ReadGmScalar<uint32_t>(params.mmadParams.scaleBGmAddr);
                scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
            }
        }
    }

    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ BiasType* biasGmAddr_{nullptr};
    __gm__ X2ScaleType* scaleGmAddr_{nullptr};

    BlockMmad mmadOp_;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    bool isBias_{false};
    bool isPermX1_{false};
    bool isPerChannel_{false};
    bool needUpdateTail_{false};
    uint32_t bMustHitL2_{1U};
    uint64_t scaleScalar_{0};
};

// ==================== FP8 per-token dequant mix (AIC + AIV) ====================

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTQBMMMx, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using L0CType = typename BlockMmad::L0CType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using EpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::Std::is_one_of_v<
                      AscendC::Std::tuple<AType, BType>, AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e4m3fn_t>,
                      AscendC::Std::tuple<fp8_e5m2_t, fp8_e5m2_t>, AscendC::Std::tuple<fp8_e5m2_t, fp8_e4m3fn_t>,
                      AscendC::Std::tuple<fp8_e4m3fn_t, fp8_e5m2_t>, AscendC::Std::tuple<hifloat8_t, hifloat8_t>>,
                  "Unsupported (AType, BType) combination for TQBMM mix");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA cannot be NZLayoutPtn or ZNLayoutPtn");

    struct TQBMMParams {
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        TQBMMParams tqbmmParams;
        EpilogueParams epilogueParams;
        Params() = default;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        Init(params);

        const auto& problemShape = params.problemShape;
        const auto& tqbmmParams = params.tqbmmParams;

        BlockScheduler bs(problemShape, params.schParams);

        if ASCEND_IS_AIC {
            const BlockShape l0BlockShape{static_cast<int64_t>(tqbmmParams.baseM),
                                          static_cast<int64_t>(tqbmmParams.baseN),
                                          static_cast<int64_t>(tqbmmParams.baseK), 0};
            BlockMmadParams blockMmadParams = params.mmadParams;
            blockMmadParams.problemShape = problemShape;
            blockMmadParams.l0TileShape = l0BlockShape;
            blockMmadParams.kAL1 = tqbmmParams.kAL1;
            blockMmadParams.kBL1 = tqbmmParams.kBL1;
            blockMmadParams.l1BufferNum = tqbmmParams.nBufferNum;
            blockMmadParams.enableL0CPingPong = tqbmmParams.dbL0C > 1;
            mmadOp_.Init(blockMmadParams);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.Init(params.epilogueParams);
        }

        auto layoutA = TqbmmDetail::MakeBatchLayoutA<LayoutA, AType, C0_SIZE>(batch_, m_, k_, isPermX1_);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);

        TqbmmDetail::TraverseBatchTiles<QuantMode::DEFAULT, QuantMode::DEFAULT, WEIGHT_NZ>(
            *this, bs, params.schParams, batch_, needUpdateTail_, gmA, gmB);

        if ASCEND_IS_AIC {
            if (!isFirstBlock_) {
                WaitForVector();
            }
        }
    }

    // 每个 (batch, m, n) block 的处理体，由 TqbmmDetail::TraverseBatchTiles 调用。
    template <typename TensorA, typename TensorB>
    __aicore__ inline void ProcessBlock(TensorA& gmA, TensorB& gmB, const TqbmmDetail::BlockTask& task)
    {
        constexpr int64_t kPos = 0L;
        if ASCEND_IS_AIC {
            TqbmmDetail::SetBL2Cache(m_, task.baseM, task.baseN, bMustHitL2_, TRANS_B, 0x7fUL, gmB);
        }

        auto subTensorA = gmA.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(task.mPos, kPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(task.baseM, static_cast<int64_t>(k_))));
        auto gmBlockA = asc::te::squeeze<0>(subTensorA);

        auto subTensorB = gmB.slice(asc::te::make_coord(task.batchIdx, asc::te::make_coord(kPos, task.nPos)),
                                    asc::te::make_shape(1L, asc::te::make_shape(static_cast<int64_t>(k_), task.baseN)));
        auto gmBlockB = asc::te::squeeze<0>(subTensorB);

        ProcessOneBlock(gmBlockA, gmBlockB, task.shape, task.mPos, task.nPos, task.baseM, task.baseN, task.batchIdx);
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = asc::te::c0_element<AType>;

    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;

    template <typename TensorA, typename TensorB>
    __aicore__ inline void ProcessOneBlock(const TensorA& gmA, const TensorB& gmB, const BlockShape& singleShape,
                                           int64_t mPos, int64_t nPos, int64_t curM, int64_t curN, uint64_t batchIdx)
    {
        constexpr int64_t l0cUbBaseOffset = 0;
        if ASCEND_IS_AIC {
            if (!isFirstBlock_) {
                WaitForVector();
            }
            constexpr int64_t l0cAlign = BLOCK_BYTE_SIZE / sizeof(L0CType);
            const int64_t curNAligned = Blaze::Gemm::CeilAlign(curN, l0cAlign);
            const int64_t curMAligned = Blaze::Gemm::CeilAlign(curM, static_cast<int64_t>(2));
            auto layoutUbC = asc::te::make_frame_layout<asc::te::nd_layout_ptn>(curMAligned, curNAligned);
            auto ubC = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, L0CType>(l0cUbBaseOffset * sizeof(L0CType)), layoutUbC);
            mmadOp_(gmA, gmB, ubC, singleShape);
            NotifyVector();
            isFirstBlock_ = false;
        }
        if ASCEND_IS_AIV {
            WaitForCube();
            int64_t offsetScale = nPos;
            int64_t offsetPtScale = mPos;
            int64_t offsetBias = nPos;
            int64_t offsetC = mPos * static_cast<int64_t>(batch_) * static_cast<int64_t>(n_) +
                              static_cast<int64_t>(batchIdx) * static_cast<int64_t>(n_) + nPos;
            epilogueOp_(curM, curN, offsetScale, offsetPtScale, offsetBias, offsetC, l0cUbBaseOffset);
            NotifyCube();
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        isPermX1_ = TqbmmDetail::IsPermX1<BlockMmad>();
        bMustHitL2_ = params.tqbmmParams.bMustHitL2;
        if ASCEND_IS_AIC {
            aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
            bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        }
    }

    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    __gm__ AType* aGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    bool isPermX1_{false};
    bool isFirstBlock_{true};
    bool needUpdateTail_{false};
    uint32_t bMustHitL2_{1U};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
