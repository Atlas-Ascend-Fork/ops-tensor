/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"
#include "blaze/gemm/tile/compute.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/sync.h"

namespace Blaze::Gemm::Kernel {

// Parameter views keep quantization geometry separate from expert/task scheduling.
template <QuantMode Mode_>
struct GroupedAntiquantParameterViews;

template <>
struct GroupedAntiquantParameterViews<QuantMode::PERCHANNEL_MODE> {
    static constexpr int64_t PARAM_ROWS = 1;
    // K arguments are part of the view contract; per-channel parameters are invariant along K.

    __aicore__ static inline auto MakeGmLayout(int64_t groups, int64_t, int64_t n)
    {
        return asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(groups, n);
    }

    __aicore__ static inline auto MakeUbLayout(int64_t n)
    {
        return asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(PARAM_ROWS, n);
    }

    template <class Tensor_>
    __aicore__ static inline auto SelectTask(const Tensor_& tensor, int64_t expert, int64_t nStart, int64_t, int64_t n)
    {
        return tensor.slice(asc::te::make_coord(expert, nStart), asc::te::make_shape(PARAM_ROWS, n));
    }

    template <class Tensor_>
    __aicore__ static inline auto SelectInput(const Tensor_& tensor, int64_t, int64_t, int64_t n)
    {
        return tensor.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(PARAM_ROWS, n));
    }

    template <class Tensor_>
    __aicore__ static inline auto SelectTile(const Tensor_& tensor, int64_t, int64_t nStart, int64_t, int64_t n)
    {
        return tensor.slice(asc::te::make_coord(0L, nStart), asc::te::make_shape(PARAM_ROWS, n));
    }
};

// Each dataflow supplies its storage geometry and Tile independently of the pipeline.
template <class BlockMmad_, QuantMode Mode_ = BlockMmad_::DispatchPolicy::ANTIQUANT_TYPE, class Enable_ = void>
struct GroupedWeightAntiquantConfig {
    static_assert(Blaze::Gemm::always_false_v<BlockMmad_>, "Unsupported grouped weight antiquant configuration");
};

#if defined(__NPU_ARCH__)
template <class BlockMmad_>
struct GroupedWeightAntiquantConfig<BlockMmad_, QuantMode::PERCHANNEL_MODE,
                                    AscendC::Std::enable_if_t<Tile::IS_ANTIQUANT_W8<typename BlockMmad_::BType>>> {
    using AType = typename BlockMmad_::AType;
    using BType = typename BlockMmad_::BType;
    using Policy = typename BlockMmad_::DispatchPolicy;
    using ParameterViews = GroupedAntiquantParameterViews<QuantMode::PERCHANNEL_MODE>;
    static_assert(!Policy::HAS_ANTIQUANT_OFFSET || AscendC::Std::is_same_v<BType, int8_t>, "FP8 offset is unsupported");
    static constexpr bool TRANS_B = !IsTrans<typename BlockMmad_::LayoutB>::value;
    static constexpr int64_t INPUT_K = 256;
    static constexpr int64_t INPUT_N = 256;
    static constexpr int32_t VF_OUTER_SIZE = Tile::ANTIQUANT_DEFAULT_OUTER_SIZE;
    using VfConfig = Tile::AntiquantTileConfig<TRANS_B, VF_OUTER_SIZE, Policy::ANTIQUANT_TYPE, BType>;
    static constexpr int64_t VF_K = VfConfig::TILE_K;
    static constexpr int64_t VF_N = VfConfig::TILE_N;
    static constexpr uint32_t INPUT_INNER_SIZE = TRANS_B ? INPUT_K : INPUT_N;
    static constexpr uint64_t INPUT_BYTES = INPUT_K * INPUT_N * sizeof(BType);
    static constexpr uint64_t SCALE_BYTES = ParameterViews::PARAM_ROWS * INPUT_N * sizeof(AType);
    static constexpr uint64_t OFFSET_BYTES = SCALE_BYTES;
    static constexpr uint64_t INPUT_SLOT_BYTES = INPUT_BYTES + SCALE_BYTES + OFFSET_BYTES;
    static constexpr uint64_t OUTPUT_SLOT_BYTES = (VF_OUTER_SIZE + VfConfig::OUTPUT_PADDING_ROWS) *
                                                  VfConfig::INNER_SIZE * sizeof(AType);
    static constexpr uint64_t OUTPUT_BASE = Policy::UB_MTE2_BUFFER_NUM * INPUT_SLOT_BYTES;
    static constexpr uint64_t BUFFER_ALIGNMENT_BYTES = 32UL;
    static_assert(INPUT_K % VF_K == 0 && INPUT_N % VF_N == 0, "Full VF input views must fit each UB input slot");
    static_assert(VF_K % BLOCK_CUBE == 0 && VF_N % BLOCK_CUBE == 0, "BL1 fragment starts must preserve C0 alignment");
    static_assert(INPUT_INNER_SIZE >= VfConfig::INNER_SIZE, "VF reads require the full contiguous input width");
    static_assert(INPUT_BYTES % BUFFER_ALIGNMENT_BYTES == 0 && SCALE_BYTES % BUFFER_ALIGNMENT_BYTES == 0 &&
                      OFFSET_BYTES % BUFFER_ALIGNMENT_BYTES == 0 && OUTPUT_SLOT_BYTES % BUFFER_ALIGNMENT_BYTES == 0,
                  "UB weight, parameters and output slots must be 32-byte aligned");

    using InputPattern = AscendC::Std::conditional_t<TRANS_B, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;

    __aicore__ static inline auto MakeWeightLayout(int64_t groups, int64_t k, int64_t n)
    {
        return asc::te::make_frame_layout<InputPattern>(groups, k, n);
    }

    __aicore__ static inline auto MakeInputLayout()
    {
        return asc::te::make_frame_layout<InputPattern>(INPUT_K, INPUT_N);
    }

    __aicore__ static inline auto MakeOutputLayout()
    {
        using Layout = AscendC::Std::conditional_t<TRANS_B, ZnRowPaddingUBLayout<AType>, NzColPaddingUBLayout<AType>>;
        return Layout{}(VF_K, VF_N, VfConfig::OUTPUT_GROUP_PITCH);
    }

    template <class DstTensor_, class SrcTensor_, class ParamTensor_>
    __aicore__ static inline void Convert(const DstTensor_& dst, const SrcTensor_& src, const ParamTensor_& scale,
                                          const ParamTensor_& offset, int64_t k, int64_t n)
    {
        Tile::Antiquant<Policy::ANTIQUANT_TYPE, AType, BType, TRANS_B, Policy::HAS_ANTIQUANT_OFFSET, INPUT_INNER_SIZE,
                        VF_OUTER_SIZE>{}(dst, src, scale, offset, static_cast<uint16_t>(TRANS_B ? n : k));
    }
};
#endif

// One BL1 segment owns one UB input stage; output stages rotate per conversion.
template <class BlockMmad_>
class GroupedWeightAntiquant {
public:
    using AType = typename BlockMmad_::AType;
    using BType = typename BlockMmad_::BType;
    using Policy = typename BlockMmad_::DispatchPolicy;
    using SyncProtocol = typename Policy::SyncProtocol;
    using Config = GroupedWeightAntiquantConfig<BlockMmad_>;
    using ParameterViews = typename Config::ParameterViews;
    static_assert(Policy::AIV_NUM == 1, "Grouped weight antiquant preprocessing requires MIX_AIC_1_1");
    static_assert(Policy::UB_MTE2_BUFFER_NUM == 2 && Policy::BL1_BUFFER_NUM == 2,
                  "Grouped weight antiquant uses two input, output and BL1 slots");
    static constexpr int64_t VF_K = Config::VF_K;
    static constexpr int64_t VF_N = Config::VF_N;

    template <class TensorB_, class TensorScale_>
    __aicore__ inline void operator()(BlockMmad_& mm, const TensorB_& weight, const TensorScale_& scale,
                                      const TensorScale_& offset, int64_t k, int64_t n, int64_t kL1)
    {
        if (k == 0 || n == 0) {
            return;
        }
        // Prime this task only; never speculate across an expert/task boundary.
        LoadInput(weight, scale, offset, 0, Min(kL1, k), n, inputIterations_, 0);
        for (int64_t kStart = 0; kStart < k; kStart += kL1, ++iterations_, ++inputIterations_) {
            const int64_t realK = Min(kL1, k - kStart);
            const int64_t nextK = kStart + kL1;
            // Submit the next input stage before current Vector work. Its own slot lock
            // protects reuse; no BL1 slot is touched by this prefetch.
            if (nextK < k) {
                LoadInput(weight, scale, offset, nextK, Min(kL1, k - nextK), n, inputIterations_ + 1, nextK / kL1);
            }
            const uint32_t bank = iterations_ % BL1_BUFFER_COUNT;
            if (iterations_ >= BL1_BUFFER_COUNT) {
                WaitFree();
            }
            auto l1 = mm.GetWeightL1Tensor(bank, realK, n);
            ConvertInput(l1, realK, n);
            // Exactly one READY after all copies composing this BL1 segment.
            Sync::NotifyCube<SyncProtocol::MODE, PIPE_MTE3>(SyncProtocol::AIV_READY_FLAG);
        }
    }

    __aicore__ inline void Drain() const
    {
        const uint64_t pending = Min(iterations_, uint64_t{BL1_BUFFER_COUNT});
        for (uint64_t i = 0; i < pending; ++i) {
            WaitFree();
        }
    }

private:
    static constexpr uint32_t UB_BUFFER_COUNT = Policy::UB_MTE2_BUFFER_NUM;
    static constexpr uint32_t BL1_BUFFER_COUNT = Policy::BL1_BUFFER_NUM;
    static constexpr uint32_t INPUT_MUTEX_BASE = 0;
    static constexpr uint32_t OUTPUT_MUTEX_BASE = INPUT_MUTEX_BASE + UB_BUFFER_COUNT;

    __aicore__ inline auto MakeInput(uint32_t bank) const
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, BType>(inputSlots_[bank].Addr()),
                                    Config::MakeInputLayout());
    }

    __aicore__ inline auto MakeScale(uint32_t bank) const
    {
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, AType>(inputSlots_[bank].Addr() + Config::INPUT_BYTES),
            ParameterViews::MakeUbLayout(Config::INPUT_N));
    }

    __aicore__ inline auto MakeOffset(uint32_t bank) const
    {
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AType>(
                                        inputSlots_[bank].Addr() + Config::INPUT_BYTES + Config::SCALE_BYTES),
                                    ParameterViews::MakeUbLayout(Config::INPUT_N));
    }

    __aicore__ inline void WaitFree() const
    {
        Sync::WaitForCube<SyncProtocol::MODE, PIPE_MTE3>(SyncProtocol::AIC_FREE_FLAG);
    }

    template <class TensorB_, class TensorScale_>
    __aicore__ inline void LoadInput(const TensorB_& weight, const TensorScale_& scale, const TensorScale_& offset,
                                     int64_t kStart, int64_t k, int64_t n, uint64_t inputId, uint64_t taskSegment)
    {
        const uint32_t bank = inputId % UB_BUFFER_COUNT;
        auto input = MakeInput(bank);
        auto lock = inputSlots_[bank].LockMte2();
        const auto copy = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copy, input.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(k, n)),
                      weight.slice(asc::te::make_coord(kStart, 0L), asc::te::make_shape(k, n)));
        if constexpr (Policy::ANTIQUANT_TYPE == QuantMode::PERCHANNEL_MODE) {
            // Refill both slots at the start of every output task; subsequent K segments reuse them.
            if (taskSegment >= UB_BUFFER_COUNT) {
                return;
            }
        }
        auto scaleUb = MakeScale(bank);
        asc::te::copy(copy, ParameterViews::SelectInput(scaleUb, 0L, k, n),
                      ParameterViews::SelectInput(scale, kStart, k, n));
        if constexpr (Policy::HAS_ANTIQUANT_OFFSET) {
            auto offsetUb = MakeOffset(bank);
            asc::te::copy(copy, ParameterViews::SelectInput(offsetUb, 0L, k, n),
                          ParameterViews::SelectInput(offset, kStart, k, n));
        }
    }

    template <class DstTensor_>
    __aicore__ inline void ConvertInput(const DstTensor_& l1, int64_t k, int64_t n)
    {
        const uint32_t bank = inputIterations_ % UB_BUFFER_COUNT;
        auto input = MakeInput(bank);
        auto scale = MakeScale(bank);
        auto offset = MakeOffset(bank);
        // Hold this input stage until BOTH VFs have read their weight/parameters.
        auto inputLock = inputSlots_[bank].LockV();
        for (int64_t ki = 0; ki < k; ki += VF_K) {
            for (int64_t ni = 0; ni < n; ni += VF_N) {
                const int64_t tileK = Min(VF_K, k - ki);
                const int64_t tileN = Min(VF_N, n - ni);
                // Describe the full register-width access, not only the valid tail.
                auto src = input.slice(asc::te::make_coord(ki, ni), asc::te::make_shape(VF_K, VF_N));
                auto scaleTile = ParameterViews::SelectTile(scale, ki, ni, VF_K, VF_N);
                auto offsetTile = ParameterViews::SelectTile(offset, ki, ni, VF_K, VF_N);
                auto dst = l1.slice(asc::te::make_coord(ki, ni), asc::te::make_shape(tileK, tileN));
                ConvertOutput(src, dst, scaleTile, offsetTile, tileK, tileN);
            }
        }
    }

    template <class SrcTensor_, class DstTensor_, class ScaleTensor_>
    __aicore__ inline void ConvertOutput(const SrcTensor_& src, const DstTensor_& dst, const ScaleTensor_& scale,
                                         const ScaleTensor_& offset, int64_t k, int64_t n)
    {
        const uint32_t bank = outputIterations_ % UB_BUFFER_COUNT;
        const auto& outputSlot = outputSlots_[bank];
        auto output = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AType>(outputSlot.Addr()),
                                           Config::MakeOutputLayout());
        {
            auto outputLock = outputSlot.LockV();
            Config::Convert(output, src, scale, offset, k, n);
        }
        {
            auto outputLock = outputSlot.LockMte3();
            auto outputTile = output.slice(asc::te::make_coord(0L, 0L), asc::te::make_shape(k, n));
            asc::te::copy(asc::te::make_copy(Tile::CopyPaddedUBToL1{}), dst, outputTile);
        }
        ++outputIterations_;
    }

    BufferSlot inputSlots_[UB_BUFFER_COUNT]{{0, INPUT_MUTEX_BASE}, {Config::INPUT_SLOT_BYTES, INPUT_MUTEX_BASE + 1}};
    BufferSlot outputSlots_[UB_BUFFER_COUNT]{{Config::OUTPUT_BASE, OUTPUT_MUTEX_BASE},
                                             {Config::OUTPUT_BASE + Config::OUTPUT_SLOT_BYTES, OUTPUT_MUTEX_BASE + 1}};
    uint64_t iterations_{0};
    uint64_t inputIterations_{0};
    uint64_t outputIterations_{0};
};

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelWqgmmMixAntiquant, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using Prologue = GroupedWeightAntiquant<BlockMmad>;
    using AntiquantConfig = typename Prologue::Config;
    using ParameterViews = typename Prologue::ParameterViews;
    static_assert(DispatchPolicy::AIV_NUM == 1, "Grouped weight antiquant kernel requires MIX_AIC_1_1");
    static constexpr uint32_t GROUP_LIST_CUMSUM = 0;
    static_assert(AscendC::Std::is_same_v<BlockEpilogue_, Blaze::Epilogue::Block::BlockEpilogueEmpty>,
                  "Grouped weight antiquant ASW does not support fused epilogues");

    // Caller contract for active problems (see docs/API/gemm/kernel/kernel_wqgmm_mix_antiquant.md):
    // kSize equals problemShape.K; scheduler, L1 and L0 agree on the M/N base tile.
    // L0 K is positive and at most 64; KaL1 == KbL1 == 4 * L0 K <= 256, baseN <= 256.
    // Base tiles follow the documented 16-element alignment and bias-dependent M limits.
    // Required TensorList descriptors and their data, plus groupList, cover the declared shapes.
    // Enabled bias/offset have valid storage; group counts are nonnegative and sum to M.
    // Launch MIX_AIC_1_1 on the target platform with at least 259 KiB of available UB.
    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmParams;
        typename BlockScheduler::Params schedulerParams;
        GM_ADDR x{nullptr};
        GM_ADDR weight{nullptr};
        GM_ADDR bias{nullptr};
        GM_ADDR scale{nullptr};
        GM_ADDR offset{nullptr};
        GM_ADDR groupList{nullptr};
        GM_ADDR y{nullptr};
        uint32_t groupNum{0};
        uint32_t groupListType{GROUP_LIST_CUMSUM};
    };

    __aicore__ inline void operator()(const Params& params)
    {
        const int64_t m = asc::te::get<MNK_M>(params.problemShape);
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        if (m <= 0 || n <= 0 || k <= 0 || params.groupNum == 0) {
            return;
        }
        const int64_t groups = params.groupNum;
        // List/GM_ADDR parsing ends here. Batch and block selection below use Tensor views only.
        auto x = MakeListTensor<AType>(params.x, asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(m, k));
        auto weight = MakeListTensor<BType>(params.weight, AntiquantConfig::MakeWeightLayout(groups, k, n));
        auto paramLayout = ParameterViews::MakeGmLayout(groups, k, n);
        auto scale = MakeListTensor<AType>(params.scale, paramLayout);
        auto offset = MakeListTensor<AType>(DispatchPolicy::HAS_ANTIQUANT_OFFSET ? params.offset : params.scale,
                                            paramLayout);
        auto bias = MakeListTensor<BiasType>(params.bias,
                                             asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(groups, n));
        auto y = MakeListTensor<AType>(params.y, asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(m, n));
        auto groupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t*>(params.groupList)),
            asc::te::make_layout(asc::te::make_shape(groups)));

        BlockMmad mm;
        mm.Init(params.mmParams);
        Prologue prologue;
        BlockScheduler scheduler(params.schedulerParams);
        // N starts must align to one FP16 fractal; M can split at any row.
        scheduler.SetTailAlign(1, BLOCK_CUBE);
        int64_t groupStart = 0;
        for (int64_t group = 0; group < groups; ++group) {
            const int64_t value = groupList[group];
            const int64_t groupM = params.groupListType == GROUP_LIST_CUMSUM ? value - groupStart : value;
            if (groupM == 0) {
                continue;
            }
            scheduler.UpdateNextProblem(ProblemShape{groupM, n, k, 1});
            scheduler.UpdateTailTile(1, 1);
            scheduler.UpdateTailTile();
            auto groupWeight = asc::te::squeeze<0>(weight.slice(asc::te::make_coord(group, asc::te::make_coord(0L, 0L)),
                                                                asc::te::make_shape(1L, asc::te::make_shape(k, n))));
            typename BlockScheduler::BlockInfo block;
            while (scheduler.GetNextBlock(block)) {
                const int64_t bm = asc::te::get<MNK_M>(block.blockShape);
                const int64_t bn = asc::te::get<MNK_N>(block.blockShape);
                if (bm == 0 || bn == 0) {
                    continue;
                }
                const int64_t mi = groupStart + asc::te::get<MNK_M>(block.blockCoord);
                const int64_t ni = asc::te::get<MNK_N>(block.blockCoord);
                if ASCEND_IS_AIV {
                    auto wTile = groupWeight.slice(asc::te::make_coord(0L, ni), asc::te::make_shape(k, bn));
                    auto sTile = ParameterViews::SelectTask(scale, group, ni, k, bn);
                    auto oTile = ParameterViews::SelectTask(offset, group, ni, k, bn);
                    prologue(mm, wTile, sTile, oTile, k, bn,
                             asc::te::get<BlockMmad::L1_KB_INDEX>(params.mmParams.l1TileShape));
                } else {
                    auto aTile = x.slice(asc::te::make_coord(mi, 0L), asc::te::make_shape(bm, k));
                    auto cTile = y.slice(asc::te::make_coord(mi, ni), asc::te::make_shape(bm, bn));
                    if (params.mmParams.hasBias) {
                        auto bTile = bias.slice(asc::te::make_coord(group, ni), asc::te::make_shape(1L, bn));
                        mm(aTile, bTile, cTile, block.blockShape);
                    } else {
                        // The absent-bias Tensor is only a placeholder: never offset or read it.
                        mm(aTile, bias, cTile, block.blockShape);
                    }
                }
            }
            groupStart += groupM;
        }
        if ASCEND_IS_AIV {
            prologue.Drain();
        }
    }

private:
    template <class T_, class Layout_>
    __aicore__ inline auto MakeListTensor(GM_ADDR list, const Layout_& layout) const
    {
        __gm__ T_* ptr = nullptr;
        if (list != nullptr) {
            AscendC::ListTensorDesc desc(reinterpret_cast<__gm__ void*>(list));
            ptr = desc.GetDataPtr<T_>(0);
        }
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(ptr), layout);
    }
};
} // namespace Blaze::Gemm::Kernel
