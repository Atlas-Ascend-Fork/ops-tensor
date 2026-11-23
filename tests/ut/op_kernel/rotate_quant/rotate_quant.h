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
 * \file rotate_quant.h
 * \brief RotateQuant Blaze kernel UT assembly and smoke-entry.
 *
 * As in the FlatQuant UT, the real epilogue participates in public assembly checks.
 * Kernel smoke runs the scheduler, BlockMmad and the Fixpipe handshake with a test epilogue.
 * A capture epilogue exposes the BF16 Cube result. SDKs with MX RegBase simulation can
 * also run the production epilogue and compare the packed output and scale bytes.
 */

#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "blaze/gemm/kernel/kernel_universal.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

namespace RotateQuantUT {

using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
using TbmmProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
constexpr int64_t BLOCK_M = 16;
constexpr int64_t MAX_BASE_K = 64;
constexpr int64_t GROUP_LIST_CUMSUM = 0;
constexpr int64_t GROUP_LIST_COUNT = 1;
constexpr int64_t ROUND_MODE_RINT = 4;
constexpr uint32_t CAPTURE_GUARD_BYTES = 32;

// Match the real epilogue's complete per-slot workspace, including scale row padding.
__aicore__ inline constexpr uint32_t EpilogueWorkspaceBytes(uint32_t maxRows)
{
    return (52736U + 32U * maxRows + 31U) / 32U * 32U;
}

template <uint32_t Slots_, bool Grouped_, typename Params>
__aicore__ inline void ConfigureUbDB(Params& params)
{
    static_assert(Slots_ == 1 || Slots_ == 2);
    if constexpr (Grouped_) {
        params.mmParams.ubDB = Slots_;
    } else {
        params.mmadParams.ubDB = Slots_;
    }
}

struct RotateQuantTilingData {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t b{1};
    int64_t mL1{BLOCK_M};
    int64_t baseK{MAX_BASE_K};
    int64_t groupNum{0};
    int64_t groupListType{GROUP_LIST_COUNT};
    int64_t hasAlpha{0};
    int64_t roundMode{ROUND_MODE_RINT};
    int64_t scaleAlg{0};
    float dstTypeMax{0.0f};
    float invDstTypeMax{0.0f};
};

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_,
          bool SplitM_ = true, uint32_t PipelineStages_ = 1>
struct RotateQuantBlazeTypes {
    static_assert(PipelineStages_ == 1 || PipelineStages_ == 2);
    using KernelProblemShape = AscendC::Std::conditional_t<EnableGrouped_, ProblemShape, TbmmProblemShape>;
    using GroupedScheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    using TbmmScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<TbmmProblemShape, 0, false, true, true>;
    using BlockScheduler = AscendC::Std::conditional_t<EnableGrouped_, GroupedScheduler, TbmmScheduler>;
    static constexpr uint64_t OUTPUT_MODEL = SplitM_ ? Blaze::Gemm::ND_ALIG_1V2_FIXPIPE :
                                                       Blaze::Gemm::ND_ALIG_1V1_FIXPIPE;
    using TbmmPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
        OUTPUT_MODEL, Blaze::Gemm::OP_TYPE_QUANT, Blaze::Gemm::KernelMmadMultiBlockTBMM,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1)>;
    using GroupedPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<OUTPUT_MODEL, Blaze::Gemm::OP_TYPE_QUANT,
                                                                   Blaze::Gemm::KernelGroupedMmadNoQuant, 0>;
    using DispatchPolicy = AscendC::Std::conditional_t<EnableGrouped_, GroupedPolicy, TbmmPolicy>;
    using TbmmLayout = asc::te::nd_ext_layout_ptn;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, XType_, TbmmLayout, XType_, TbmmLayout, bfloat16_t,
                                                    TbmmLayout, XType_, TbmmLayout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<bfloat16_t, YType_, ScaleType_>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<KernelProblemShape, BlockMmad, BlockEpilogue,
                                                            BlockScheduler>;
};

// Test-only consumer used by the smoke and BF16 capture epilogues.
class FixpipeInputForTest {
public:
    __aicore__ inline void Init(uint32_t ubDB, bool splitM)
    {
        ubDB_ = ubDB;
        splitM_ = splitM;
        sequence_ = 0;
    }

    __aicore__ inline uint32_t Wait() const
    {
        const uint32_t slot = ubDB_ == 2 ? sequence_ & 1U : 0U;
        Blaze::Gemm::Sync::WaitForCube<Blaze::Gemm::Sync::SYNC_MODE_INTRA, PIPE_V>(
            Blaze::Gemm::Sync::FIXPIPE_AIC_READY_FLAG_BASE + slot);
        return slot;
    }

    __aicore__ inline void Release(uint32_t slot, bool hasMte3Dependency = false)
    {
        if (!hasMte3Dependency) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        }
        Blaze::Gemm::Sync::NotifyCube<Blaze::Gemm::Sync::SYNC_MODE_INTRA, PIPE_MTE3>(
            Blaze::Gemm::Sync::FIXPIPE_AIV_ACK_FLAG_BASE + slot);
        ++sequence_;
    }

    __aicore__ inline uint32_t Base(uint32_t slot) const { return slot * SlotBytes(); }
    __aicore__ inline uint32_t SlotBytes() const
    {
        return ubDB_ == 2 ? AscendC::TOTAL_UB_SIZE / 2 : AscendC::TOTAL_UB_SIZE;
    }
    __aicore__ inline bool SplitM() const { return splitM_; }
    __aicore__ inline bool IsDoubleBuffered() const { return ubDB_ == 2; }

private:
    uint32_t ubDB_{1};
    bool splitM_{false};
    uint64_t sequence_{0};
};

struct Bf16AccumulatorForTest {
    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;
    // A nonzero offset makes the capture test detect a kernel that bypasses the epilogue's view.
    static constexpr uint32_t INPUT_UB_OFFSET = 512;

    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const
    {
        const uint32_t offset = INPUT_UB_OFFSET + params.slot * (AscendC::TOTAL_UB_SIZE / 2);
        return asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(offset),
                                    asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(params.rows, params.cols));
    }
};

template <typename YType_, typename ScaleType_>
class KernelSmokeEpilogue : public Bf16AccumulatorForTest {
public:
    using DataTypeIn = bfloat16_t;
    using DataTypeOut = YType_;
    using DataTypeScale = ScaleType_;
    using PublicEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<DataTypeIn, DataTypeOut, DataTypeScale>;
    using Params = typename PublicEpilogue::Params;
    using BlockShape = typename PublicEpilogue::BlockShape;
    using BlockCoord = typename PublicEpilogue::BlockCoord;

    __aicore__ inline void Init(const Params& params) { input_.Init(params.ubDB, params.splitM); }
    __aicore__ inline void operator()(const BlockShape&, const BlockCoord&)
    {
        const auto slot = input_.Wait();
        input_.Release(slot);
    }

private:
    FixpipeInputForTest input_;
};

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_,
          bool SplitM_ = true, uint32_t PipelineStages_ = 1>
struct KernelSmokeTypes {
    using PublicTypes = RotateQuantBlazeTypes<XType_, YType_, ScaleType_, Layout_, EnableGrouped_, SplitM_,
                                              PipelineStages_>;
    using BlockEpilogue = KernelSmokeEpilogue<YType_, ScaleType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename PublicTypes::KernelProblemShape,
                                                      typename PublicTypes::BlockMmad, BlockEpilogue,
                                                      typename PublicTypes::BlockScheduler>;
};

// Capture the Cube result before quantization without instantiating FP4/FP8 Cast.
// Use the same consumer protocol and full-block contract as the public epilogue.
class Bf16CaptureEpilogue : public Bf16AccumulatorForTest {
public:
    using DataTypeIn = bfloat16_t;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    struct Params {
        GM_ADDR outputGm{nullptr};
        int64_t batch{1};
        int64_t rotationSize{0};
        GM_ADDR slotVisitsGm{nullptr};
        uint32_t ubDB{1};
        bool splitM{false};
    };

    __aicore__ inline ~Bf16CaptureEpilogue() { Drain(); }

    __aicore__ inline static constexpr uint32_t WorkspaceBytes(uint32_t maxRows)
    {
        return EpilogueWorkspaceBytes(maxRows);
    }

    __aicore__ inline void Init(const Params& params)
    {
        Drain();
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.outputGm));
        batch_ = params.batch;
        rotationSize_ = params.rotationSize;
        input_.Init(params.ubDB, params.splitM);
        SetWorkspace(0, 0);
        deferWriteback_ = input_.IsDoubleBuffered();
        slotVisits_ = reinterpret_cast<__gm__ uint64_t*>(params.slotVisitsGm);
    }

    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        const uint32_t slot = input_.Wait();
        SetWorkspace(input_.Base(slot), slot);
        DrainWorkspace(eventId_);
        if (slotVisits_ != nullptr) {
            slotVisits_[AscendC::GetBlockIdx() * 2 + slot] = 1;
        }
        const int64_t fullM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
        const int64_t perAiv = input_.SplitM() ? Blaze::Gemm::CeilDiv(fullM, int64_t{2}) : fullM;
        const int64_t start = AscendC::GetSubBlockIdx() * perAiv;
        const int64_t rows = Blaze::Gemm::Max(int64_t{0}, Blaze::Gemm::Min(perAiv, fullM - start));
        const int64_t row = asc::te::get<Blaze::Gemm::MNK_M>(blockCoord) + start;
        const int64_t batch = asc::te::get<Blaze::Gemm::MNK_B>(blockCoord);
        if (rows > 0) {
            AscendC::LocalTensor<bfloat16_t> intermediate{
                AscendC::TPosition::VECIN, ubBaseOffset_ + INPUT_UB_OFFSET,
                static_cast<uint32_t>((input_.SlotBytes() - INPUT_UB_OFFSET) / sizeof(bfloat16_t))};
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(rotationSize_ * sizeof(bfloat16_t)), 0,
                static_cast<int64_t>((batch_ - 1) * rotationSize_ * sizeof(bfloat16_t)), 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId_);
            AscendC::DataCopyPad<bfloat16_t>(output_[(row * batch_ + batch) * rotationSize_], intermediate, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId_);
            pending_[eventId_] = true;
            if (!deferWriteback_) {
                DrainWorkspace(eventId_);
            }
        }
        input_.Release(slot, rows > 0);
    }

    __aicore__ inline void SetWorkspace(uint32_t ubBaseOffset, uint32_t writebackEventId)
    {
        ubBaseOffset_ = ubBaseOffset;
        eventId_ = writebackEventId;
    }

private:
    __aicore__ inline void Drain()
    {
        if ASCEND_IS_AIV {
            for (uint32_t eventId = 0; eventId < 2; ++eventId) {
                DrainWorkspace(eventId);
            }
        }
    }

    __aicore__ inline void DrainWorkspace(uint32_t eventId)
    {
        if (pending_[eventId]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
            pending_[eventId] = false;
        }
    }

    AscendC::GlobalTensor<bfloat16_t> output_;
    int64_t batch_{1};
    int64_t rotationSize_{0};
    uint32_t ubBaseOffset_{0};
    uint8_t eventId_{0};
    bool deferWriteback_{false};
    bool pending_[2]{false, false};
    __gm__ uint64_t* slotVisits_{nullptr};
    FixpipeInputForTest input_;
};

template <bool SplitM_, typename Params>
__aicore__ inline void ConfigureCubeParams(Params& mm, GM_ADDR xGm, GM_ADDR rotationGm,
                                           const RotateQuantTilingData& tiling)
{
    mm.aGmAddr = xGm;
    mm.bGmAddr = rotationGm;
    mm.oriK = tiling.k;
    mm.mL1 = mm.mL0 = tiling.mL1;
    mm.nL1 = mm.nL0 = tiling.k;
    mm.kL1 = tiling.k;
    mm.kL0 = tiling.baseK;
    mm.l1Stages = mm.l0cStages = 2;
    mm.splitM = SplitM_;
    mm.ubDB = 1;
}

template <bool SplitM_, typename Params>
__aicore__ inline void ConfigureGroupedParams(Params& params, GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR groupsGm,
                                              const RotateQuantTilingData& tiling)
{
    params.problemShape = {tiling.m, tiling.n, tiling.k, tiling.b};
    ConfigureCubeParams<SplitM_>(params.mmParams, xGm, rotationGm, tiling);
    params.mmParams.groupListGmAddr = groupsGm;
    params.gmmParams = {
        static_cast<uint32_t>(tiling.groupNum), 0, static_cast<uint32_t>(tiling.groupListType), 1, 1, 1, 0, 0};
    params.schedulerParams.baseM = static_cast<int32_t>(tiling.mL1);
    params.schedulerParams.baseN = static_cast<int32_t>(tiling.k);
    params.schedulerParams.mTailCnt = params.schedulerParams.nTailCnt = 1;
    params.schedulerParams.groupType = 0;
    params.schedulerParams.groupNum = static_cast<uint32_t>(tiling.groupNum);
    params.schedulerParams.groupListType = static_cast<uint32_t>(tiling.groupListType);
    params.schedulerParams.initialM = tiling.b == 1 ? tiling.m * (tiling.n / tiling.k) : tiling.m;
    params.schedulerParams.singleX = params.schedulerParams.singleWeight = params.schedulerParams.singleY = true;
    params.schedulerParams.weightElementSize = sizeof(uint16_t);
    params.schedulerParams.matrixView = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant::MatrixView::BatchTransposedA;
    params.schedulerParams.splitM = SplitM_;
}

template <typename XType_, bool SplitM_, uint32_t PipelineStages_ = 1, bool EnableGrouped_ = false>
__aicore__ inline void RunRotateQuantBf16Capture(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR outputGm,
                                                 const RotateQuantTilingData& tiling, GM_ADDR groupsGm = nullptr)
{
    using Types = RotateQuantBlazeTypes<XType_, fp4x2_e2m1_t, int8_t, asc::te::nd_ext_layout_ptn, EnableGrouped_,
                                        SplitM_, PipelineStages_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<typename Types::KernelProblemShape, typename Types::BlockMmad,
                                                      Bf16CaptureEpilogue, typename Types::BlockScheduler>;
    typename Kernel::Params params{};
    if constexpr (EnableGrouped_) {
        static_assert(Kernel::ENABLE_QUANT);
        ConfigureGroupedParams<SplitM_>(params, xGm, rotationGm, groupsGm, tiling);
    } else {
        static_assert(Kernel::ENABLE_QUANT);
        params.problemShape = {tiling.b == 1 ? tiling.m * (tiling.n / tiling.k) : tiling.m, tiling.k, tiling.k,
                               tiling.b, 1};
        params.schParams.mL1 = tiling.mL1;
        params.schParams.nL1 = params.schParams.kL1 = tiling.k;
        params.schParams.mBaseTailSplitCnt = params.schParams.nBaseTailSplitCnt = 1;
        params.schParams.mTailCnt = params.schParams.nTailCnt = 1;
        ConfigureCubeParams<SplitM_>(params.mmadParams, xGm, rotationGm, tiling);
    }
    params.epilogueParams = {outputGm, tiling.b, tiling.k, nullptr, PipelineStages_, SplitM_};
    if constexpr (PipelineStages_ == 2) {
        params.epilogueParams.slotVisitsGm = outputGm + tiling.m * tiling.n * sizeof(bfloat16_t) + CAPTURE_GUARD_BYTES;
    }
    ConfigureUbDB<PipelineStages_, EnableGrouped_>(params);
    Kernel kernel;
    kernel(params);
}

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_,
          bool SplitM_ = true, uint32_t PipelineStages_ = 1, bool RealEpilogue_ = false>
__aicore__ inline void RunRotateQuantBlaze(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm, GM_ADDR groupListGm,
                                           GM_ADDR yGm, GM_ADDR scaleGm, const RotateQuantTilingData& tiling)
{
    using PublicTypes = RotateQuantBlazeTypes<XType_, YType_, ScaleType_, Layout_, EnableGrouped_, SplitM_,
                                              PipelineStages_>;
    using SmokeTypes = KernelSmokeTypes<XType_, YType_, ScaleType_, Layout_, EnableGrouped_, SplitM_, PipelineStages_>;
    using Kernel = AscendC::Std::conditional_t<RealEpilogue_, typename PublicTypes::MatmulKernel,
                                               typename SmokeTypes::Kernel>;
    const int64_t m = tiling.m;
    const int64_t n = tiling.n;
    const int64_t k = tiling.k;
    const int64_t b = tiling.b;
    const int64_t mL1 = tiling.mL1;
    typename Kernel::Params params{};
    params.epilogueParams = {yGm,
                             scaleGm,
                             {m, n, k, b},
                             {},
                             false,
                             -1,
                             tiling.roundMode,
                             tiling.scaleAlg,
                             tiling.dstTypeMax,
                             tiling.invDstTypeMax,
                             PipelineStages_,
                             SplitM_};
    if constexpr (EnableGrouped_) {
        ConfigureGroupedParams<SplitM_>(params, xGm, rotationGm, groupListGm, tiling);
    } else {
        params.problemShape = {b == 1 ? m * (n / k) : m, k, k, b, 1};
        params.schParams.mL1 = mL1;
        params.schParams.nL1 = params.schParams.kL1 = k;
        params.schParams.mBaseTailSplitCnt = params.schParams.nBaseTailSplitCnt = 1;
        params.schParams.mTailCnt = params.schParams.nTailCnt = 1;
        ConfigureCubeParams<SplitM_>(params.mmadParams, xGm, rotationGm, tiling);
    }
    ConfigureUbDB<PipelineStages_, EnableGrouped_>(params);
    if (tiling.hasAlpha == 1 && alphaGm != nullptr) {
        auto layout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_shape(1L));
        auto alpha = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(alphaGm)), layout);
        params.epilogueParams.alpha = alpha[0];
        const float value = AscendC::Cast(params.epilogueParams.alpha);
        params.epilogueParams.needClamp = value > 0.0f && value < 1.0f;
    }
    Kernel kernel;
    kernel(params);
}

} // namespace RotateQuantUT

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_,
          bool SplitM_ = true, uint32_t PipelineStages_ = 1, bool RealEpilogue_ = false>
__global__ __aicore__ void RotateQuantKernelEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm, GM_ADDR groupListGm,
                                                  GM_ADDR yGm, GM_ADDR scaleGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto& tiling = *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm);
    RotateQuantUT::RunRotateQuantBlaze<XType_, YType_, ScaleType_, Layout_, EnableGrouped_, SplitM_, PipelineStages_,
                                       RealEpilogue_>(xGm, rotationGm, alphaGm, groupListGm, yGm, scaleGm, tiling);
}

template <typename XType_, typename YType_, typename ScaleType_, typename Layout_, bool EnableGrouped_,
          uint32_t PipelineStages_ = 1, bool RealEpilogue_ = false>
__global__ __aicore__ void RotateQuantSingleAivKernelEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR alphaGm,
                                                           GM_ADDR groupListGm, GM_ADDR yGm, GM_ADDR scaleGm,
                                                           GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    const auto& tiling = *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm);
    RotateQuantUT::RunRotateQuantBlaze<XType_, YType_, ScaleType_, Layout_, EnableGrouped_, false, PipelineStages_,
                                       RealEpilogue_>(xGm, rotationGm, alphaGm, groupListGm, yGm, scaleGm, tiling);
}

template <typename XType_, bool SplitM_ = true, uint32_t PipelineStages_ = 1>
__global__ __aicore__ void RotateQuantBf16CaptureEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR outputGm,
                                                       GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RotateQuantUT::RunRotateQuantBf16Capture<XType_, SplitM_, PipelineStages_>(
        xGm, rotationGm, outputGm, *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm));
}

// A separate entry declares the real one-AIV launch contract. The CPU runner's
// actual task ratio is probed before this entry is used for a numerical test.
template <typename XType_, uint32_t PipelineStages_ = 1>
__global__ __aicore__ void RotateQuantBf16CaptureSingleAivEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR outputGm,
                                                                GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    RotateQuantUT::RunRotateQuantBf16Capture<XType_, false, PipelineStages_>(
        xGm, rotationGm, outputGm, *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm));
}

template <typename XType_, bool SplitM_ = true, uint32_t PipelineStages_ = 1>
__global__ __aicore__ void RotateQuantGroupedBf16CaptureEntry(GM_ADDR xGm, GM_ADDR rotationGm, GM_ADDR groupsGm,
                                                              GM_ADDR outputGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RotateQuantUT::RunRotateQuantBf16Capture<XType_, SplitM_, PipelineStages_, true>(
        xGm, rotationGm, outputGm, *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm), groupsGm);
}

template <typename XType_, uint32_t PipelineStages_ = 1>
__global__ __aicore__ void RotateQuantGroupedBf16CaptureSingleAivEntry(GM_ADDR xGm, GM_ADDR rotationGm,
                                                                       GM_ADDR groupsGm, GM_ADDR outputGm,
                                                                       GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    RotateQuantUT::RunRotateQuantBf16Capture<XType_, false, PipelineStages_, true>(
        xGm, rotationGm, outputGm, *reinterpret_cast<const RotateQuantUT::RotateQuantTilingData*>(tilingGm), groupsGm);
}

__global__ __aicore__ void RotateQuantSingleAivLaunchProbe(GM_ADDR ratiosGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIV {
        const uint32_t index = AscendC::GetBlockIdx();
        if (index < AscendC::GetBlockNum() * 2) {
            // A real 1:1 group has only sub-block zero. Mark any second AIV
            // explicitly, even if a runner reports a misleading task ratio.
            reinterpret_cast<__gm__ int64_t*>(ratiosGm)[index] = AscendC::GetSubBlockIdx() == 0 ?
                                                                     AscendC::GetTaskRation() :
                                                                     -2;
        }
    }
}
