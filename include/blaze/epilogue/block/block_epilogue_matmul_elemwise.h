/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_matmul_elemwise.h
 * \brief AIC+AIV basic epilogue for FusedMatMul elementwise add/mul: out = acc + x3 or out = acc * x3.
 *
 * The accumulator (AccType, produced by the AIC fixpipe into UB) is combined elementwise with
 * x3 (ElementType, read from GM) and stored to GM as ElementType. The add/mul kind is taken from
 * DispatchPolicy::FUSED_OP_TYPE (OP_TYPE_ADD / OP_TYPE_MUL).
 * Fusion domain (mirrors the legacy Cmct::Gemm::Block::BlockEpilogueElementwise + FusionAdd/FusionMul
 * pair and the BlockEpilogueMatmulStreamK FusionSameDomain contract):
 *  - AccType != ElementType && !FusionSameDomain (inner_precise = high precision): x3 is upcast to
 *    AccType, the fusion runs in the FP32 domain and the result is cast back with round-to-nearest-even.
 *  - AccType == ElementType (inner_precise = high performance, OTF): the fusion runs in the element
 *    domain (the accumulator is already B16).
 *  - AccType != ElementType && FusionSameDomain (inner_precise = high performance remapped to the
 *    1V2 dual-AIV flow): the accumulator stays FP32 on the wire (B16 acc + DUAL_DST_SPLIT_M has a
 *    device-side gap), but the matmul result is rounded to ElementType before fusing with x3, so the
 *    fusion matches the legacy B16-accumulator semantics (bf16-style accumulate with the output dtype).
 *
 * This class serves the basic (non-streamK) kernels only and is fully self-contained; the streamK
 * path uses BlockEpilogueMatmulStreamK directly (its Run() dispatches on FUSED_OP_TYPE).
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <class DispatchPolicy_, class ElementType_, class AccType_,
          bool FusionSameDomain_ = AscendC::IsSameType<AccType_, ElementType_>::value>
class BlockEpilogueMatmulElemwise {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ElementType = ElementType_;
    using AccType = AccType_;
    static constexpr bool FUSION_SAME_DOMAIN = FusionSameDomain_;
    // Same-dtype fusion runs in the element domain (legacy high_performance mode); mixed dtype runs in
    // AccType domain with a final CAST_RINT (legacy high_precision mode).
    using FusionAccType = typename AscendC::Std::conditional<AscendC::IsSameType<AccType, ElementType>::value,
                                                             ElementType, float>::type;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::IsSameType<ElementType, half>::value ||
                      AscendC::IsSameType<ElementType, bfloat16_t>::value ||
                      AscendC::IsSameType<ElementType, float>::value,
                  "BlockEpilogueMatmulElemwise only supports half/bfloat16_t/float element type");
    static_assert(AscendC::IsSameType<AccType, ElementType>::value || AscendC::IsSameType<AccType, float>::value,
                  "Mixed domain fusion requires a float accumulator");
    static_assert(DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_ADD ||
                      DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL,
                  "BlockEpilogueMatmulElemwise requires FUSED_OP_TYPE to be OP_TYPE_ADD or OP_TYPE_MUL");

    // Params-only Init contract: problemShape rides inside Params (mirrors the upstream
    // single-param convention, e.g. BlockEpilogueFmmWithScaleAdd / scale_add / sigmoid).
    struct Params {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR x3GmAddr{nullptr};
        bool x3BatchBroadcast{false};
        int64_t x3M{0};
        ProblemShape problemShape{};
    };

    __aicore__ inline BlockEpilogueMatmulElemwise()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
        }
    }

    __aicore__ inline ~BlockEpilogueMatmulElemwise()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIV {
            x3GmAddr_ = params.x3GmAddr;
            x3BatchBroadcast_ = params.x3BatchBroadcast && params.x3M > 0;
            x3M_ = params.x3M;
            cGmAddr_ = params.cGmAddr;
            n_ = asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape);
        }
    }

    __aicore__ inline void operator()(const BlockShape& blockShape, int64_t dstOffset, bool splitM, int64_t baseM,
                                      int64_t baseN, uint64_t ubDB = 1)
    {
        (void)ubDB; // single accumulator buffer; x3/output staging follows it in UB
        const int64_t totalM = asc::te::get<Blaze::Gemm::MNK_M>(blockShape);
        const int64_t nL1 = asc::te::get<Blaze::Gemm::MNK_N>(blockShape);
        const int64_t curBaseN = baseN != 0 ? Blaze::Gemm::Min(nL1, baseN) : nL1;
        const int64_t nL1Iter = Blaze::Gemm::CeilDiv(nL1, curBaseN);
        const int64_t mL1Iter = baseM > 0 ? Blaze::Gemm::CeilDiv(totalM, baseM) : 1;

        // Create internal UB view: accumulator is at UB offset 0 (where AIC fixpipe writes).
        auto ubInternal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, AccType>(0),
                                               asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1, 1));
        __ubuf__ AccType* ubBase = reinterpret_cast<__ubuf__ AccType*>(ubInternal.data().get());

        if (mL1Iter > 1) {
            // A full-load (al1): AIC handshake is per M-tile (each writes tileM × fullN to UB).
            // One handshake per M tile; N chunks processed without extra handshakes.
            for (int64_t mIdx = 0; mIdx < mL1Iter; ++mIdx) {
                const int64_t tileM = mIdx + 1 == mL1Iter ? totalM - baseM * mIdx : baseM;
                const int64_t halfM = Blaze::Gemm::CeilDiv(tileM, static_cast<int64_t>(AscendC::GetTaskRation()));
                int64_t localRows = tileM;
                int64_t localRowOffset = 0;
                if (splitM) {
                    const int64_t subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
                    localRows = (static_cast<uint64_t>(tileM) & 1UL) > 0UL ? halfM - subBlockIdx : halfM;
                    localRowOffset = subBlockIdx * halfM;
                }
                const int64_t accumulatorRows = splitM ? halfM : Blaze::Gemm::CeilAlign(tileM, SPLIT_M_ALIGN);
                if (localRows <= 0) {
                    Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                    Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
                    continue;
                }
                Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
                for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
                    const int64_t tileN = nIdx + 1 == nL1Iter ? nL1 - curBaseN * nIdx : curBaseN;
                    ProcessTile(ubBase, localRows, accumulatorRows, tileN,
                                dstOffset + mIdx * baseM * n_ + nIdx * curBaseN + localRowOffset * n_);
                }
                Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            }
            return;
        }

        // Basic/bl1 path: AIC handshake is per N-chunk (each writes fullM × tileN to UB).
        int64_t curM = totalM;
        if (baseM != 0) {
            curM = Blaze::Gemm::Min(curM, baseM);
        }
        const int64_t halfM = Blaze::Gemm::CeilDiv(curM, static_cast<int64_t>(AscendC::GetTaskRation()));
        int64_t localRows = curM;
        int64_t localRowOffset = 0;
        if (splitM) {
            const int64_t subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
            localRows = (static_cast<uint64_t>(curM) & 1UL) > 0UL ? halfM - subBlockIdx : halfM;
            localRowOffset = subBlockIdx * halfM;
        }
        // The AIC fixpipe pads M to SPLIT_M_ALIGN before DUAL_DST_SPLIT_M, so both AIVs address
        // halfM physical rows even when the second AIV owns one fewer valid row for an odd M.
        const int64_t accumulatorRows = splitM ? halfM : Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN);
        for (int64_t nIdx = 0; nIdx < nL1Iter; ++nIdx) {
            const int64_t tileN = nIdx + 1 == nL1Iter ? nL1 - curBaseN * nIdx : curBaseN;
            if (localRows <= 0) {
                // No vector work is issued by this AIV; keep the ready/free handshake on PIPE_MTE3 so the
                // free flag cannot bypass a ready wait queued on another pipeline.
                Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
                continue;
            }
            Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
            ProcessTile(ubBase, localRows, accumulatorRows, tileN, dstOffset + nIdx * curBaseN + localRowOffset * n_);
            Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }

private:
    static constexpr int64_t SPLIT_M_ALIGN = 2;
    static constexpr uint32_t DATA_BLOCK = 32;
    static constexpr uint32_t ACC_ALIGN = DATA_BLOCK / sizeof(AccType);
    static constexpr uint32_t ELEMENT_ALIGN = DATA_BLOCK / sizeof(ElementType);
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    // UB event ids used by the staging pipeline (per-HardEvent-pair scoped).
    static constexpr uint16_t X3_EVENT = 0;
    static constexpr uint16_t OUTPUT_EVENT = 1;
    static constexpr uint16_t BUFFER_REUSE_EVENT = 2;

#ifdef __NPU_ARCH__
    static constexpr AscendC::Reg::CastTrait B16_TO_FP32_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait B16_TO_FP32_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait FP32_TO_B16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
#endif // __NPU_ARCH__

    // UB staging layout for the fusion vf kernel: acc/x3/output share one packed view
    // (the output reuses the x3 buffer in place).
    struct FusedElemwiseVfParams {
        __ubuf__ FusionAccType* accAddr{nullptr};
        __ubuf__ ElementType* x3Addr{nullptr};
        __ubuf__ ElementType* outAddr{nullptr};
        uint32_t rowsThisStage{0};
        uint32_t curN{0};
        uint32_t accPitch{0};
        uint32_t elemPitch{0};
        uint32_t vlAcc{0};
        uint32_t vfLoops{0};
    };

    template <typename Params>
    static __simd_vf__ inline void FusedElemwiseVf(const Params params)
    {
        for (uint32_t row = 0; row < params.rowsThisStage; ++row) {
            __ubuf__ FusionAccType* accRow = params.accAddr + row * params.accPitch;
            __ubuf__ ElementType* x3Row = params.x3Addr + row * params.elemPitch;
            __ubuf__ ElementType* outRow = params.outAddr + row * params.elemPitch;
            for (uint32_t i = 0; i < params.vfLoops; ++i) {
                const uint32_t col = i * params.vlAcc;
                const uint32_t remain = params.curN - col;
                uint32_t valid = remain < params.vlAcc ? remain : params.vlAcc;
                // Unified FP32-domain compute: RegTensor B16 uses NORM layout (1 elem/32-bit slot),
                // while UB B16 data is PACKED (2 elem/slot). Direct packed B16 load into RegTensor
                // reads every other element (stride-2). Fix: unpack B16→FP32 (Cast ZERO/ONE +
                // Interleave, same ceremony as x3), fuse in FP32, CAST_RINT back to ElementType.
                // Semantics: data path stays B16 (UB footprint), arithmetic always in FP32; the
                // fusion domain is controlled by FusionSameDomain (see the acc load below).
                AscendC::Reg::MaskReg allB16 = AscendC::Reg::CreateMask<ElementType, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::MaskReg maskF32 = AscendC::Reg::UpdateMask<float>(valid);
                AscendC::Reg::RegTensor<float> accF;
                AscendC::Reg::RegTensor<ElementType> accPacked;
                AscendC::Reg::RegTensor<float> accZero;
                AscendC::Reg::RegTensor<float> accOne;
                AscendC::Reg::RegTensor<ElementType> x3Packed;
                AscendC::Reg::RegTensor<float> x3Zero;
                AscendC::Reg::RegTensor<float> x3One;
                AscendC::Reg::RegTensor<float> outAcc;
                AscendC::Reg::RegTensor<ElementType> outElem;
                // 1. Load acc: FP32 direct (AccType=float) or B16 unpack (AccType=B16)
                if constexpr (AscendC::IsSameType<AccType, float>::value) {
                    AscendC::Reg::DataCopy(accF, accRow + col);
                    if constexpr (FUSION_SAME_DOMAIN && !AscendC::IsSameType<ElementType, float>::value) {
                        // Same-domain fusion with an FP32 accumulator (high_performance remapped to the
                        // 1V2 dual-AIV flow): round the matmul result to ElementType first so the x3
                        // fusion matches the legacy B16-accumulator semantics (acc quantized to the
                        // output dtype before fusing, like the legacy fixpipe B16 writeout).
                        AscendC::Reg::RegTensor<ElementType> accElem;
                        AscendC::Reg::Cast<ElementType, float, FP32_TO_B16>(accElem, accF, maskF32);
                        AscendC::Reg::Cast<float, ElementType, B16_TO_FP32_ZERO>(accF, accElem, maskF32);
                    }
                } else {
                    AscendC::Reg::DataCopy(accPacked, accRow + col);
                    AscendC::Reg::Cast<float, ElementType, B16_TO_FP32_ZERO>(accZero, accPacked, maskF32);
                    AscendC::Reg::Cast<float, ElementType, B16_TO_FP32_ONE>(accOne, accPacked, allB16);
                    AscendC::Reg::Interleave(accF, accOne, accZero, accOne);
                }
                // 2. Load x3: FP32 direct or B16 unpack (same for both modes)
                if constexpr (AscendC::IsSameType<ElementType, float>::value) {
                    AscendC::Reg::DataCopy(x3Zero, x3Row + col);
                } else {
                    AscendC::Reg::DataCopy(x3Packed, x3Row + col);
                    AscendC::Reg::Cast<float, ElementType, B16_TO_FP32_ZERO>(x3Zero, x3Packed, maskF32);
                    AscendC::Reg::Cast<float, ElementType, B16_TO_FP32_ONE>(x3One, x3Packed, allB16);
                    AscendC::Reg::Interleave(x3Zero, x3One, x3Zero, x3One);
                }
                // 3. Fuse in FP32 domain
                if constexpr (DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL) {
                    AscendC::Reg::Mul(outAcc, accF, x3Zero, maskF32);
                } else {
                    AscendC::Reg::Add(outAcc, accF, x3Zero, maskF32);
                }
                // 4. Cast back to ElementType and store
                if constexpr (AscendC::IsSameType<ElementType, float>::value) {
                    AscendC::Reg::DataCopy<ElementType, AscendC::Reg::StoreDist::DIST_NORM_B32>(outRow + col, outAcc,
                                                                                                maskF32);
                } else {
                    AscendC::Reg::Cast<ElementType, float, FP32_TO_B16>(outElem, outAcc, maskF32);
                    AscendC::Reg::DataCopy<ElementType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outRow + col, outElem,
                                                                                                maskF32);
                }
            }
        }
    }

    __aicore__ inline uint32_t AccPitch(uint32_t tileN) const
    {
        // 行距统一按ACC_ALIGN(=DATA_BLOCK/sizeof(AccType))对齐, 与block侧CopyOutFromL0C2UB的
        // c0_element<AccType>写入恒等, 与主线BlockEpilogueFixpipe读取语义一致; 1V2所需行距
        // 差异由AccType自然表达(half=16/float=8), 无需L0C2OUT_MODEL感知.
        return static_cast<uint32_t>(
            Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN), static_cast<uint64_t>(ACC_ALIGN)));
    }

    __aicore__ inline void ProcessTile(__ubuf__ AccType* ubBase, int64_t localRows, int64_t accumulatorRows,
                                       int64_t tileN, int64_t tileGmOffset)
    {
        if (localRows <= 0 || tileN <= 0) {
            return;
        }
        const uint32_t accPitch = AccPitch(static_cast<uint32_t>(tileN));
        const uint32_t elemPitch = static_cast<uint32_t>(
            Blaze::Gemm::CeilAlign(static_cast<uint64_t>(tileN), static_cast<uint64_t>(ELEMENT_ALIGN)));
        const uint64_t accumulatorBytes = Blaze::Gemm::CeilAlign(
            static_cast<uint64_t>(accumulatorRows) * accPitch * sizeof(AccType), static_cast<uint64_t>(DATA_BLOCK));
        if (accumulatorBytes >= AscendC::TOTAL_UB_SIZE) {
            return;
        }
        // x3/output staging lives right after the accumulator image; the output reuses the x3 buffer in place.
        const uint64_t x3OutputBufferBytes = AscendC::TOTAL_UB_SIZE - accumulatorBytes;
        const uint64_t stageRowBytes = elemPitch * sizeof(ElementType);
        const int64_t stageRows = static_cast<int64_t>(
            Blaze::Gemm::Min(static_cast<uint64_t>(localRows), x3OutputBufferBytes / stageRowBytes));
        if (stageRows <= 0) {
            // 静默跳过, 对齐主线scale_add epilogue的UB不足边界处理(无设备侧assert先例);
            // 正常tiling下调度器按UB容量切块, 此处为极端shape防御.
            return;
        }
        // Arithmetic runs in the FP32 domain (vlAcc=8 per 256-bit register); B16 acc is unpacked
        // inside the vf kernel (Cast ZERO/ONE + Interleave) before fusion.
        const uint32_t vlAcc = AscendC::GetVecLen() / sizeof(float);
        const uint32_t vfLoops = static_cast<uint32_t>(
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(tileN), static_cast<uint64_t>(vlAcc)));

        for (int64_t stageOffset = 0; stageOffset < localRows; stageOffset += stageRows) {
            const int64_t rowsThisStage = Blaze::Gemm::Min(stageRows, localRows - stageOffset);
            const int64_t gmElemOffset = tileGmOffset + stageOffset * n_;
            // x3/output share the same UB staging region. Wait for the previous UB2GM read to finish
            // before overwriting the region with the next GM2UB write (mirrors the scale/add epilogue).
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
            CopyX3Stage(accumulatorBytes, rowsThisStage, tileN, elemPitch, gmElemOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(X3_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(X3_EVENT);

            auto accAddr = ubBase + stageOffset * accPitch;
            auto x3Addr = reinterpret_cast<__ubuf__ ElementType*>(ubBase) + accumulatorBytes / sizeof(ElementType);
            if constexpr (!AscendC::IsSameType<AccType, float>::value) {
                // B16累加器: flat count融合(B16域直接加减, 对齐legacy语义, 与原VF路径位等价——
                // 两者均为RINT(acc+op x3)于B16). AccType==ElementType且accPitch==elemPitch,
                // 整stage为一段连续区, 一条count指令替代per-8元素寄存器仪式.
                AscendC::LocalTensor<ElementType> ubAll{AscendC::TPosition::VECIN, 0,
                                                        AscendC::TOTAL_UB_SIZE / sizeof(ElementType)};
                const int64_t accOff = stageOffset * accPitch;
                const int64_t x3Off = static_cast<int64_t>(accumulatorBytes / sizeof(ElementType));
                const int64_t fuseCount = rowsThisStage * elemPitch;
                // MTE2(x3)→计算排序由前置MTE2_V token承担(CMCT同款先例);
                if constexpr (DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL) {
                    AscendC::Mul(ubAll[x3Off], ubAll[x3Off], ubAll[accOff], fuseCount);
                } else {
                    AscendC::Add(ubAll[x3Off], ubAll[x3Off], ubAll[accOff], fuseCount);
                }
                // 计算→MTE3(输出拷贝)排序: V管屏障+token(对齐CMCT, 每stage省两次PIPE_ALL全管线排空)
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            } else {
                FusedElemwiseVfParams vfParams{accAddr,
                                               x3Addr,
                                               x3Addr,
                                               static_cast<uint32_t>(rowsThisStage),
                                               static_cast<uint32_t>(tileN),
                                               accPitch,
                                               elemPitch,
                                               vlAcc,
                                               vfLoops};
                asc_vf_call<FusedElemwiseVf<FusedElemwiseVfParams>>(vfParams);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(OUTPUT_EVENT);
            }

            CopyOutStage(accumulatorBytes, rowsThisStage, tileN, elemPitch, gmElemOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BUFFER_REUSE_EVENT);
        }
    }

    __aicore__ inline void CopyX3Stage(uint64_t accumulatorBytes, int64_t rowsThisStage, int64_t tileN,
                                       uint32_t elemPitch, int64_t gmElemOffset)
    {
        const auto origin = asc::te::make_coord(0L, 0L);
        const auto validShape = asc::te::make_shape(rowsThisStage, tileN);
        auto x3UbStorage = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, ElementType>(accumulatorBytes),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, static_cast<int64_t>(elemPitch)));
        auto x3Ub = x3UbStorage.slice(origin, validShape);
        auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        if (!x3BatchBroadcast_) {
            auto x3GmStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType*>(x3GmAddr_) +
                                                             gmElemOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, n_));
            auto x3Gm = x3GmStorage.slice(origin, validShape);
            asc::te::copy(copyGmToUb, x3Ub, x3Gm);
            return;
        }
        // Broadcast: fold the global output row index onto x3M rows of x3 and copy contiguous row segments.
        const int64_t firstRow = gmElemOffset / n_;
        const int64_t colOffset = gmElemOffset % n_;
        int64_t row = 0;
        while (row < rowsThisStage) {
            const int64_t x3Row = (firstRow + row) % x3M_;
            const int64_t segRows = Blaze::Gemm::Min(rowsThisStage - row, x3M_ - x3Row);
            auto segUbStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, ElementType>(
                    accumulatorBytes + static_cast<uint64_t>(row) * elemPitch * sizeof(ElementType)),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(segRows, static_cast<int64_t>(elemPitch)));
            auto segUb = segUbStorage.slice(origin, asc::te::make_shape(segRows, tileN));
            auto segGmStorage = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType*>(x3GmAddr_) +
                                                             x3Row * n_ + colOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(segRows, n_));
            auto segGm = segGmStorage.slice(origin, asc::te::make_shape(segRows, tileN));
            asc::te::copy(copyGmToUb, segUb, segGm);
            row += segRows;
        }
    }

    __aicore__ inline void CopyOutStage(uint64_t accumulatorBytes, int64_t rowsThisStage, int64_t tileN,
                                        uint32_t elemPitch, int64_t gmElemOffset)
    {
        const auto origin = asc::te::make_coord(0L, 0L);
        const auto validShape = asc::te::make_shape(rowsThisStage, tileN);
        auto outUbStorage = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, ElementType>(accumulatorBytes),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, static_cast<int64_t>(elemPitch)));
        auto outUb = outUbStorage.slice(origin, validShape);
        auto outGmStorage = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementType*>(cGmAddr_) +
                                                         gmElemOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rowsThisStage, n_));
        auto outGm = outGmStorage.slice(origin, validShape);
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        asc::te::copy(copyUbToGm, outGm, outUb);
    }

    GM_ADDR cGmAddr_{nullptr};
    GM_ADDR x3GmAddr_{nullptr};
    int64_t n_{0};
    int64_t x3M_{0};
    bool x3BatchBroadcast_{false};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
