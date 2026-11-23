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
 * \file kernel_tbmm_basic.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"
#include "c_api/spr/spr.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTBMM, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    static constexpr bool NEED_EPILOGUE = !AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>;
    static constexpr bool ENABLE_QUANT = DispatchPolicy::FUSED_OP_TYPE == Gemm::OP_TYPE_QUANT;
    static_assert(NEED_EPILOGUE == ENABLE_QUANT,
                  "TBMM epilogue and dispatch policy must select the same execution path");

    // mmad
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;

    static_assert(
        NEED_EPILOGUE ||
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<AType, BType, CType, BiasType>, AscendC::Std::tuple<half, half, half, half>,
                AscendC::Std::tuple<half, half, half, float>, AscendC::Std::tuple<half, half, signed char, half>,
                AscendC::Std::tuple<half, half, signed char, float>,
                AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>,
                AscendC::Std::tuple<float, float, float, float>>,
        "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be nz_layout_ptn or zn_layout_ptn");

    struct Params {
        // Standard TBMM view: (M, N, K, batch, splitBatch).
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
    {
        if constexpr (!NEED_EPILOGUE) {
            if ASCEND_IS_AIV {
                return;
            }
        } else {
            static_assert(AscendC::Std::is_same_v<typename BlockEpilogue::DataTypeIn, CType>,
                          "BlockMmad output and epilogue input types must match");
        }
        Init(params);
        const int64_t coreNums = AscendC::GetBlockNum();
        if (coreNums <= 0) {
            return;
        }
        BlockScheduler bs(params.problemShape, params.schParams);
        const int64_t core = NEED_EPILOGUE ? Gemm::GetCurrentBlockIdx() : AscendC::GetBlockIdx();
        const int64_t totalBlocks = bs.GetBlockNums();
        if (core >= bs.GetCoreNums()) {
            return;
        }

        asc_override_strategy saturationStrategy = asc_override_strategy::USE_API;
        if constexpr (NEED_EPILOGUE) {
            if ASCEND_IS_AIV {
                if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() != 0) {
                    return;
                }
            }
        }

        Blaze::Gemm::SetHF32(params.schParams.isHf32);
        BlockMmad blockMmad;
        {
            BlockEpilogue epilogue;
            if constexpr (NEED_EPILOGUE) {
                if ASCEND_IS_AIV {
                    saturationStrategy = asc_get_saturation_strategy();
                    asc_set_saturation_strategy(asc_override_strategy::USE_API);
                    epilogue.Init(params.epilogueParams);
                    // Every participating AIV supplies one initial credit per UB slot.
                    for (uint16_t slot = 0; slot < params.mmadParams.ubDB; ++slot) {
                        Sync::NotifyCube<Sync::SYNC_MODE_INTRA, PIPE_MTE3>(Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot);
                    }
                }
            }
            if ASCEND_IS_AIC {
                blockMmad.Init(params.mmadParams);
            }

            MatmulProcess(params, blockMmad, bs, epilogue, core, coreNums, totalBlocks);
        } // Finish pending epilogue writebacks before restoring the Vector mode.
        if constexpr (NEED_EPILOGUE) {
            if ASCEND_IS_AIV {
                asc_set_saturation_strategy(saturationStrategy);
            }
            if ASCEND_IS_AIC {
                // Consume the final returned credit, or the initial credit of an unused slot.
                for (uint16_t slot = 0; slot < params.mmadParams.ubDB; ++slot) {
                    Sync::WaitForVector<Sync::SYNC_MODE_INTRA, PIPE_FIX>(Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot,
                                                                         params.mmadParams.splitM != 0);
                }
            }
        } else {
            UnsetHf32();
        }
    }

private:
    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                         BlockEpilogue& epilogue, int64_t curBlockIdx, int64_t coreNums,
                                         int64_t totalBlockNums)
    {
        uint64_t batchStrideA = TRANS_BATCH_A ? k_ : m_ * k_;
        uint64_t mStrideA = TRANS_BATCH_A ? batch_ * k_ : k_;

        auto layoutA = MakeNDBatchLayout<AType>(batch_, m_, k_, batchStrideA, mStrideA);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = MakeNDBatchLayout<CType>(batch_, m_, n_, n_, batch_ * n_);

        uint64_t innerBatch = Blaze::Gemm::CeilDiv(batch_, batchSplitFactor_);
        auto splitBatchLayoutC = asc::te::make_pattern_layout<
            asc::te::nd_layout_ptn, asc::te::layout_trait<CType, AscendC::Std::Int<asc::te::c0_element<CType>>>>(
            asc::te::make_shape(batchSplitFactor_, innerBatch, asc::te::make_shape(m_, n_)),
            asc::te::make_stride(m_ * innerBatch * n_, n_, asc::te::make_stride(innerBatch * n_, asc::te::_1{})));

        auto layoutBias = MakeLayoutBias{}(1L, n_);
        // A,B,C Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto splitBatchGmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_),
                                                  splitBatchLayoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
        if constexpr (!NEED_EPILOGUE) {
            SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
        }

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx); // 非全载
            auto blockCoord = bs.GetBlockCoord(blockIdx);                          // (m, n, k, b)
            auto coordM = asc::te::get<MNK_M>(blockCoord);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            auto shapeM = asc::te::get<MNK_M>(blockShape);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);
            if (shapeM <= 0 || shapeN <= 0) {
                continue;
            }
            auto batchIdx = asc::te::get<MNK_B>(blockCoord);
            auto gmBlockA = asc::te::squeeze<0>(gmA.slice(AscendC::MakeCoord(batchIdx, AscendC::MakeCoord(coordM, 0L)),
                                                          AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeK))));
            auto gmBlockB = asc::te::squeeze<0>(gmB.slice(AscendC::MakeCoord(batchIdx, AscendC::MakeCoord(0L, coordN)),
                                                          AscendC::MakeShape(1L, AscendC::MakeShape(shapeK, shapeN))));
            auto gmBlockBias = gmBias.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));
            if ASCEND_IS_AIC {
                if constexpr (NEED_EPILOGUE) {
                    auto ubC = epilogue.GetAccumulatorTensor({shapeM, shapeN, 0});
                    blockMmad(gmBlockA, gmBlockB, gmBlockBias, ubC, blockShape, ENABLE_QUANT);
                } else {
                    auto gmBlockC = asc::te::squeeze<0>(
                        gmC.slice(AscendC::MakeCoord(batchIdx, AscendC::MakeCoord(coordM, coordN)),
                                  AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeN))));
                    auto splitBatchGmBlockC = asc::te::squeeze<0, 1>(splitBatchGmC.slice(
                        AscendC::MakeCoord(batchIdx / innerBatch, batchIdx % innerBatch,
                                           AscendC::MakeCoord(coordM, coordN)),
                        AscendC::MakeShape(asc::te::_1{}, asc::te::_1{}, AscendC::MakeShape(shapeM, shapeN))));
                    __gm__ uint64_t* gmScalePtr = scaleGmAddr_ == nullptr ? nullptr :
                                                                            scaleGmAddr_ + batchIdx * n_ + coordN;
                    if (batchSplitFactor_ > 1) {
                        blockMmad(gmBlockA, gmBlockB, gmBlockBias, splitBatchGmBlockC, blockShape, gmScalePtr);
                    } else {
                        blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape, gmScalePtr);
                    }
                }
            }
            if constexpr (NEED_EPILOGUE) {
                if ASCEND_IS_AIV {
                    epilogue(blockShape, blockCoord);
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline auto MakeNDBatchLayout(uint64_t batch, uint64_t row, uint64_t col, uint64_t batchStride,
                                             uint64_t colStride)
    {
        return asc::te::make_pattern_layout<asc::te::nd_layout_ptn,
                                            asc::te::layout_trait<T, AscendC::Std::Int<asc::te::c0_element<T>>>>(
            asc::te::make_shape(batch, asc::te::make_shape(row, col)),
            asc::te::make_stride(batchStride, asc::te::make_stride(colStride, asc::te::_1{})));
    }

    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        batchSplitFactor_ = static_cast<uint64_t>(asc::te::get<MNK_SplitB>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
        if constexpr (!NEED_EPILOGUE) {
            scaleGmAddr_ = reinterpret_cast<__gm__ uint64_t*>(blockMmadParams.scaleGmAddr);
        }
    }

    __aicore__ inline void UnsetHf32() { AscendC::SetHF32Mode(0); }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = BlockMmad::NON_CONTIGUOUS_TYPE;
    static constexpr bool TRANS_BATCH_A = (NON_CONTIGUOUS_TYPE ==
                                           static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1));
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;  // 可选输入，直接初始化
    __gm__ uint64_t* scaleGmAddr_ = nullptr; // 可选输入，int8输出量化scale

    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    uint64_t batchSplitFactor_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
