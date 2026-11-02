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
 * \file block_epilogue_matmul_streamk.h
 * \brief
 */

#pragma once
#include "kernel_basic_intf.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

// FusionSameDomain_: true = fuse in OutType domain (opType=add/mul, acc cast first);
// false = fuse in FP32 domain. Tail default keeps existing three-arg instantiations.
template <class WorkspaceType_, class OutType_, class DispatchPolicy_, bool FusionSameDomain_ = true>
class BlockEpilogueMatmulStreamK {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t windowLen = 4; // ASW滑窗长度，0视为未下发回退默认4
        // Elementwise fusion input (effective when FUSED_OP_TYPE is OP_TYPE_ADD/OP_TYPE_MUL);
        // tail-appended so the existing aggregate initializations stay unchanged.
        GM_ADDR x3GmAddr{nullptr};
        bool x3BatchBroadcast{false};
        int64_t x3M{0};
    };

    __aicore__ inline BlockEpilogueMatmulStreamK()
    {
        if ASCEND_IS_AIV {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FUSION_BUFFER_REUSE_EVENT);
        }
    }
    __aicore__ inline ~BlockEpilogueMatmulStreamK()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FUSION_BUFFER_REUSE_EVENT);
        }
    }

    using WorkspaceType = WorkspaceType_;
    using OutType = OutType_;
    using DispatchPolicy = DispatchPolicy_;

    AscendC::GlobalTensor<OutType> cGlobal_;
    AscendC::GlobalTensor<WorkspaceType> workspaceGlobal_;

    // basic args
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t mL1_ = 0;
    uint64_t nL1_ = 0;
    uint64_t mCnt_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t kCnt_ = 0;
    uint64_t bCnt_ = 0;
    uint64_t round_ = 1;
    uint64_t usedCoreNum_ = 0;
    uint64_t aivMte2Num_ = 0;

    struct AivParams {
        uint64_t indexParams = 0;
        uint64_t mCntIndex = 0;
        uint64_t nCntIndex = 0;
        uint64_t kCntIndex = 0;
        uint64_t bCntIndex = 0;
        uint64_t curML1InAiv = 0;
        uint64_t curNL1InAiv = 0;
        uint64_t curAlignedNInAiv = 0;
    };
    AivParams aivParams_;

    uint64_t mBurstBase_ = 0;

    struct CopyGm2UbParams {
        uint64_t offsetWorkspaceGM = 0;
        uint64_t kCnt = 0;
        uint64_t mBurstOri = 0;
        uint64_t mBurst = 0;
        uint64_t burstLen = 0;
        uint64_t srcGap = 0;
    };
    CopyGm2UbParams copyGm2UbParams_;

    struct CopyUb2GmParams {
        uint64_t offsetCGm = 0;
        uint64_t mLength = 0;
        uint64_t burstLen = 0;
        uint64_t dstGap = 0;
        uint64_t srcGap = 0;
    };
    CopyUb2GmParams copyUb2GmParams_;

    // StreamK Init: original seven-parameter contract (unchanged, called by the shared streamK
    // kernels). Only the x3/cGmAddr bookkeeping is appended for the ADD/MUL fusion path.
    __aicore__ inline void Init(Params const& params, BlockShape blockShapeInAiv, BlockShape tileL1ShapeInAiv,
                                BlockCoord coordInAiv, uint64_t usedCoreNum, bool checkIsSkScene,
                                BlockShape tileL0Shape)
    {
        m_ = asc::te::get<Blaze::Gemm::MNK_M>(blockShapeInAiv);
        n_ = asc::te::get<Blaze::Gemm::MNK_N>(blockShapeInAiv);
        mL1_ = asc::te::get<Blaze::Gemm::MNK_M>(tileL1ShapeInAiv);
        nL1_ = asc::te::get<Blaze::Gemm::MNK_N>(tileL1ShapeInAiv);
        mCnt_ = asc::te::get<Blaze::Gemm::MNK_M>(coordInAiv);
        nCnt_ = asc::te::get<Blaze::Gemm::MNK_N>(coordInAiv);
        kCnt_ = asc::te::get<Blaze::Gemm::MNK_K>(coordInAiv);
        bCnt_ = asc::te::get<Blaze::Gemm::MNK_B>(coordInAiv);
        baseM_ = asc::te::get<Blaze::Gemm::MNK_M>(tileL0Shape);
        baseN_ = asc::te::get<Blaze::Gemm::MNK_N>(tileL0Shape);
        alignBaseN_ = Blaze::Gemm::CeilAlign(baseN_, Blaze::Gemm::BLOCK_BYTE_SIZE);
        windowLen_ = params.windowLen == 0 ? 4 : params.windowLen; // 0视为未下发，回退默认4，与scheduler约定一致
        usedCoreNum_ = usedCoreNum;
        // Decrease tile size of per vector core to prevent data race of cube and vector
        aivMte2Num_ = checkIsSkScene ? AscendC::GetTaskRation() : AscendC::BLOCK_CUBE;
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(params.cGmAddr));
        workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ WorkspaceType*>(params.workspaceGmAddr));
        AscendC::ICachePreLoad(NUM_TWO);
        // Ensure cube to pair with vector, add sync flag in dp+sk scene
        if (!checkIsSkScene) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Blaze::Gemm::ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Blaze::Gemm::ZERO_FLAG);
        }
        // Elementwise fusion (ADD/MUL) bookkeeping: raw addresses for the x3 copy-out staging.
        cGmAddr_ = params.cGmAddr;
        x3GmAddr_ = params.x3GmAddr;
        x3BatchBroadcast_ = params.x3BatchBroadcast && params.x3M > 0;
        x3M_ = params.x3M;
    }

    __aicore__ inline void Run()
    {
        UpdateAivBasicIndex();
        UpdateAivBasicBlock();
        for (uint64_t index = 0; index < aivMte2Num_; ++index) {
            UpdateAivParams(index);
            if (copyGm2UbParams_.mBurst == 0) {
                return;
            }
            if constexpr (DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_ADD ||
                          DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL) {
                // Elementwise add/mul fusion: row-wise workspace read + K-split reduce + x3 fuse.
                FuseAndOut();
                continue;
            }

            // Shared stage (non-fused paths only): flat 1D read + flat reduce — mainline original.
            AscendC::LocalTensor<float> ubAddTensor{AscendC::TPosition::VECIN, 0,
                                                    AscendC::TOTAL_UB_SIZE / sizeof(float)};
            AscendC::DataCopyExtParams dataCopyExtParams{
                static_cast<uint16_t>(copyGm2UbParams_.kCnt),
                static_cast<uint32_t>(copyGm2UbParams_.burstLen * sizeof(float)),
                static_cast<uint32_t>(copyGm2UbParams_.srcGap * sizeof(float)), 0, 0};
            AscendC::DataCopyPad<float>(ubAddTensor, workspaceGlobal_[copyGm2UbParams_.offsetWorkspaceGM],
                                        dataCopyExtParams, {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);

            for (uint64_t i = 1; i < copyGm2UbParams_.kCnt; ++i) {
                AscendC::Add(ubAddTensor, ubAddTensor, ubAddTensor[i * copyGm2UbParams_.burstLen],
                             copyGm2UbParams_.burstLen);
            }

            AscendC::DataCopyExtParams ub2gmExtParams{
                static_cast<uint16_t>(copyUb2GmParams_.mLength),
                static_cast<uint32_t>(copyUb2GmParams_.burstLen * sizeof(OutType)),
                static_cast<uint32_t>(copyUb2GmParams_.srcGap * sizeof(OutType) / UB2GM_SRCGAP_UNIT),
                static_cast<uint32_t>(copyUb2GmParams_.dstGap * sizeof(OutType)), 0};

            if constexpr (DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_RELU &&
                          (sizeof(OutType) != sizeof(half) || AscendC::IsSameType<OutType, bfloat16_t>::value)) {
                AscendC::Relu(ubAddTensor, ubAddTensor, copyGm2UbParams_.burstLen);
            }
            if constexpr (sizeof(OutType) == sizeof(half)) {
                AscendC::LocalTensor<OutType> ubCastDst{AscendC::TPosition::VECIN, 0,
                                                        AscendC::TOTAL_UB_SIZE / sizeof(OutType)};
                AscendC::Cast(ubCastDst, ubAddTensor, AscendC::RoundMode::CAST_RINT, copyGm2UbParams_.burstLen);
                if constexpr (DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_RELU &&
                              !AscendC::IsSameType<OutType, bfloat16_t>::value) {
                    // Relu not support bfloat16_t
                    AscendC::Relu(ubCastDst, ubCastDst, copyGm2UbParams_.burstLen);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Blaze::Gemm::ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Blaze::Gemm::ZERO_FLAG);
                AscendC::DataCopyPad<OutType,
                                     (DispatchPolicy::FIXP_OPTI == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2) ?
                                         AscendC::PaddingMode::Normal :
                                         AscendC::PaddingMode::Compact>(cGlobal_[copyUb2GmParams_.offsetCGm], ubCastDst,
                                                                        ub2gmExtParams);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Blaze::Gemm::ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Blaze::Gemm::ZERO_FLAG);
                AscendC::DataCopyPad<OutType,
                                     (DispatchPolicy::FIXP_OPTI == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2) ?
                                         AscendC::PaddingMode::Normal :
                                         AscendC::PaddingMode::Compact>(cGlobal_[copyUb2GmParams_.offsetCGm],
                                                                        ubAddTensor, ub2gmExtParams);
            }
        }
    }

    __aicore__ inline void UpdateAivBasicIndex()
    {
        uint64_t newBlockIdx = AscendC::GetBlockIdx() / (AscendC::GetTaskRation() * kCnt_);
        aivParams_.kCntIndex = AscendC::GetBlockIdx() % (AscendC::GetTaskRation() * kCnt_);
        aivParams_.bCntIndex = newBlockIdx / (mCnt_ * nCnt_);
        aivParams_.indexParams = newBlockIdx;
        uint64_t cGmIndex = (aivParams_.indexParams + (bCnt_ * mCnt_ * nCnt_ - bCnt_ * mCnt_ * nCnt_ % usedCoreNum_)) %
                            (mCnt_ * nCnt_);
        uint64_t mainWindow = AscendC::Std::min(windowLen_, mCnt_);
        uint64_t mainRow = mCnt_ / mainWindow - 1UL;
        uint64_t tailWindow = mCnt_ - mainRow * mainWindow;
        uint64_t rowIdx = cGmIndex / nCnt_ / mainWindow;
        if (rowIdx < mainRow) {
            aivParams_.mCntIndex = rowIdx * mainWindow + cGmIndex % mainWindow;
            aivParams_.nCntIndex = (cGmIndex / mainWindow) % nCnt_;
        } else {
            rowIdx = mainRow;
            uint64_t tailIndex = cGmIndex - mainRow * mainWindow * nCnt_;
            aivParams_.mCntIndex = mainRow * mainWindow + tailIndex % tailWindow;
            aivParams_.nCntIndex = (tailIndex / tailWindow) % nCnt_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % NUM_TWO != 0UL) {
            aivParams_.nCntIndex = nCnt_ - 1UL - aivParams_.nCntIndex;
        }
    }

    __aicore__ inline void UpdateAivBasicBlock()
    {
        if (round_ < NUM_TWO) {
            aivParams_.curML1InAiv = aivParams_.mCntIndex != (mCnt_ - 1) ? mL1_ : (m_ - (mCnt_ - 1) * mL1_);
            aivParams_.curNL1InAiv = aivParams_.nCntIndex != (nCnt_ - 1) ? nL1_ : (n_ - (nCnt_ - 1) * nL1_);
            if constexpr (DispatchPolicy::FIXP_OPTI == Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2) {
                aivParams_.curAlignedNInAiv = Blaze::Gemm::CeilAlign(aivParams_.curNL1InAiv,
                                                                     static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
            } else {
                aivParams_.curAlignedNInAiv = aivParams_.curNL1InAiv;
            }
        }
    }

    __aicore__ inline void UpdateAivParams(uint64_t index)
    {
        mBurstBase_ = Blaze::Gemm::CeilAlign(
            Blaze::Gemm::CeilDiv(aivParams_.curML1InAiv, kCnt_ * AscendC::GetTaskRation()),
            Blaze::Gemm::CeilDiv(UB2GM_SRCGAP_UNIT, aivParams_.curAlignedNInAiv));
        uint64_t mBurstCnt = Blaze::Gemm::CeilDiv(aivParams_.curML1InAiv, mBurstBase_);
        uint64_t mBurstTail = aivParams_.curML1InAiv - (mBurstCnt - 1) * mBurstBase_;
        if (aivParams_.kCntIndex >= mBurstCnt) {
            copyGm2UbParams_.mBurstOri = 0;
        } else {
            copyGm2UbParams_.mBurstOri = (aivParams_.kCntIndex == mBurstCnt - 1) ? mBurstTail : mBurstBase_;
        }

        copyGm2UbParams_.kCnt = kCnt_;
        copyGm2UbParams_.mBurst = Blaze::Gemm::CeilDiv(copyGm2UbParams_.mBurstOri, aivMte2Num_);
        // Calculate init address of workspace for moving into UB.
        copyGm2UbParams_.offsetWorkspaceGM = (aivParams_.indexParams) * kCnt_ * baseM_ * alignBaseN_ +
                                             (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) *
                                                 aivParams_.curAlignedNInAiv;
        // Calculate init address of GM for moving out to GM.
        copyUb2GmParams_.offsetCGm = aivParams_.nCntIndex * nL1_ + aivParams_.mCntIndex * mL1_ * n_ +
                                     (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) * n_ +
                                     aivParams_.bCntIndex * m_ * n_;
        uint64_t singleCnt = 1;
        if (index == singleCnt - 1) {
            copyGm2UbParams_.mBurst = copyGm2UbParams_.mBurstOri - (singleCnt - 1) * copyGm2UbParams_.mBurst;
        } else if (index >= singleCnt) {
            copyGm2UbParams_.mBurst = 0;
        }
        // datasize for moving in ub, align to 32B
        copyGm2UbParams_.burstLen = Blaze::Gemm::CeilAlign(copyGm2UbParams_.mBurst * aivParams_.curAlignedNInAiv,
                                                           Blaze::Gemm::BLOCK_BYTE_SIZE);
        // gap of src between cur burst and next burst
        copyGm2UbParams_.srcGap = baseM_ * alignBaseN_ - copyGm2UbParams_.burstLen;

        // args for ub2gm
        copyUb2GmParams_.mLength = copyGm2UbParams_.mBurst;
        copyUb2GmParams_.burstLen = aivParams_.curNL1InAiv;
        copyUb2GmParams_.dstGap = n_ - aivParams_.curNL1InAiv;
        copyUb2GmParams_.srcGap = aivParams_.curAlignedNInAiv - aivParams_.curNL1InAiv;
    }

    __aicore__ inline void operator()() { Run(); }

private:
    // ---- Elementwise add/mul fusion support (FUSED_OP_TYPE == OP_TYPE_ADD/OP_TYPE_MUL) ----
    // StreamK path only: the basic-path epilogue (BlockEpilogueMatmulElemwise, block_epilogue_matmul_elemwise.h)
    // is fully self-contained and does not inherit or reuse these members.
    static constexpr uint32_t FUSION_DATA_BLOCK = 32;
    // UB event ids used by the fusion staging. Flag ids are scoped per HardEvent pair, so they do
    // not clash with ZERO_FLAG used by the non-fused Run() path.
    static constexpr uint16_t FUSION_X3_EVENT = 0;
    static constexpr uint16_t FUSION_OUTPUT_EVENT = 1;
    static constexpr uint16_t FUSION_BUFFER_REUSE_EVENT = 2;
#ifdef __NPU_ARCH__
    static constexpr AscendC::Reg::CastTrait FUSION_B16_TO_FP32_ZERO = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait FUSION_B16_TO_FP32_ONE = {
        AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait FUSION_FP32_TO_B16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
#endif // __NPU_ARCH__
    // Fusion compute domain: OutType (same-domain, high-performance mode) or FP32 (high-precision mode).
    using FusionAccType = typename AscendC::Std::conditional<FusionSameDomain_, OutType, float>::type;
    static constexpr bool FUSION_IS_MUL = DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL;

    // UB staging layout for the fusion vf kernel: acc/x3/output share one packed view
    // (the output reuses the x3 buffer in place).
    struct FusedElemwiseVfParams {
        __ubuf__ FusionAccType* accAddr{nullptr};
        __ubuf__ OutType* x3Addr{nullptr};
        __ubuf__ OutType* outAddr{nullptr};
        uint32_t rowsThisStage{0};
        uint32_t curN{0};
        uint32_t accPitch{0};
        uint32_t elemPitch{0};
        uint32_t vlAcc{0};
        uint32_t vfLoops{0};
    };

    // In-place FP32 -> OutType acc cast params (streamK same-domain mode).
    struct CastAccVfParams {
        __ubuf__ float* srcAddr{nullptr};
        __ubuf__ OutType* dstAddr{nullptr};
        uint32_t totalElems{0};
        uint32_t vlF32{0};
        uint32_t vfLoops{0};
    };

    template <typename Params>
    static __simd_vf__ inline void FusedElemwiseVf(const Params params)
    {
        for (uint32_t row = 0; row < params.rowsThisStage; ++row) {
            __ubuf__ FusionAccType* accRow = params.accAddr + row * params.accPitch;
            __ubuf__ OutType* x3Row = params.x3Addr + row * params.elemPitch;
            __ubuf__ OutType* outRow = params.outAddr + row * params.elemPitch;
            for (uint32_t i = 0; i < params.vfLoops; ++i) {
                const uint32_t col = i * params.vlAcc;
                const uint32_t remain = params.curN - col;
                uint32_t valid = remain < params.vlAcc ? remain : params.vlAcc;
                // Unified FP32-domain fusion: RegTensor B16 uses NORM layout while UB B16 is
                // PACKED. Direct packed B16 load reads every other element (stride-2). Fix:
                // unpack B16→FP32 (Cast ZERO/ONE + Interleave), fuse in FP32, CAST_RINT back.
                AscendC::Reg::MaskReg allB16 = AscendC::Reg::CreateMask<OutType, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::MaskReg maskF32 = AscendC::Reg::UpdateMask<float>(valid);
                AscendC::Reg::RegTensor<float> accF;
                AscendC::Reg::RegTensor<OutType> accPacked;
                AscendC::Reg::RegTensor<float> accZero;
                AscendC::Reg::RegTensor<float> accOne;
                AscendC::Reg::RegTensor<OutType> x3Packed;
                AscendC::Reg::RegTensor<float> x3Zero;
                AscendC::Reg::RegTensor<float> x3One;
                AscendC::Reg::RegTensor<float> outAcc;
                AscendC::Reg::RegTensor<OutType> outElem;
                // 1. Load acc: FP32 direct (FusionAccType=float) or B16 unpack (B16)
                if constexpr (AscendC::IsSameType<FusionAccType, float>::value) {
                    AscendC::Reg::DataCopy(accF, accRow + col);
                } else {
                    AscendC::Reg::DataCopy(accPacked, accRow + col);
                    AscendC::Reg::Cast<float, OutType, FUSION_B16_TO_FP32_ZERO>(accZero, accPacked, maskF32);
                    AscendC::Reg::Cast<float, OutType, FUSION_B16_TO_FP32_ONE>(accOne, accPacked, allB16);
                    AscendC::Reg::Interleave(accF, accOne, accZero, accOne);
                }
                // 2. Load x3: FP32 direct or B16 unpack
                if constexpr (AscendC::IsSameType<OutType, float>::value) {
                    AscendC::Reg::DataCopy(x3Zero, x3Row + col);
                } else {
                    AscendC::Reg::DataCopy(x3Packed, x3Row + col);
                    AscendC::Reg::Cast<float, OutType, FUSION_B16_TO_FP32_ZERO>(x3Zero, x3Packed, maskF32);
                    AscendC::Reg::Cast<float, OutType, FUSION_B16_TO_FP32_ONE>(x3One, x3Packed, allB16);
                    AscendC::Reg::Interleave(x3Zero, x3One, x3Zero, x3One);
                }
                // 3. Fuse in FP32 domain
                if constexpr (FUSION_IS_MUL) {
                    AscendC::Reg::Mul(outAcc, accF, x3Zero, maskF32);
                } else {
                    AscendC::Reg::Add(outAcc, accF, x3Zero, maskF32);
                }
                // 4. Cast back to OutType and store
                if constexpr (AscendC::IsSameType<OutType, float>::value) {
                    AscendC::Reg::DataCopy<OutType, AscendC::Reg::StoreDist::DIST_NORM_B32>(outRow + col, outAcc,
                                                                                            maskF32);
                } else {
                    AscendC::Reg::Cast<OutType, float, FUSION_FP32_TO_B16>(outElem, outAcc, maskF32);
                    AscendC::Reg::DataCopy<OutType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outRow + col, outElem,
                                                                                            maskF32);
                }
            }
        }
    }

    // Same-domain (high-performance) mode: in-place cast of the reduced FP32 accumulator to the
    // OutType domain before the fusion vf. Reg-level loop must run in a __simd_vf__ context.
    template <typename Params>
    static __simd_vf__ inline void CastAccFp32ToOutVf(const Params params)
    {
        for (uint32_t ci = 0; ci < params.vfLoops; ++ci) {
            const uint32_t remain = params.totalElems - ci * params.vlF32;
            uint32_t valid = remain < params.vlF32 ? remain : params.vlF32;
            AscendC::Reg::MaskReg maskF32 = AscendC::Reg::UpdateMask<float>(valid);
            AscendC::Reg::RegTensor<float> accF;
            AscendC::Reg::RegTensor<OutType> accE;
            AscendC::Reg::DataCopy(accF, params.srcAddr + ci * params.vlF32);
            AscendC::Reg::Cast<OutType, float, FUSION_FP32_TO_B16>(accE, accF, maskF32);
            AscendC::Reg::DataCopy<OutType, AscendC::Reg::StoreDist::DIST_PACK_B32>(params.dstAddr + ci * params.vlF32,
                                                                                    accE, maskF32);
        }
    }

    // StreamK elementwise fusion (self-contained): 2D row-aware workspace read (CMCT model) ->
    // 2D reduce -> [same-domain cast] -> x3 fusion -> write GM.
    // The 2D read re-arranges UB rows to an aligned pitch (CeilAlign(nValid, 16 for B16)),
    // ensuring 32-byte-aligned VF register access and consistent row layout for the fusion.
    __aicore__ inline void FuseAndOut()
    {
        const uint64_t rows = copyGm2UbParams_.mBurst;
        const uint64_t nValid = aivParams_.curNL1InAiv;
        const uint64_t kCnt = copyGm2UbParams_.kCnt;
        if (rows == 0 || nValid == 0) {
            return;
        }

        // Aligned row pitch: 32-byte aligned for the VF register-level DataCopy (B16=16, FP32=8).
        const uint64_t alignedPitch = Blaze::Gemm::CeilAlign(
            aivParams_.curAlignedNInAiv, static_cast<uint64_t>(FUSION_DATA_BLOCK / sizeof(OutType)));
        // All kCnt partial-result images are staged in UB before the row-wise reduce, so the
        // accumulator footprint must cover every image, not just image 0. Missing this makes the
        // x3/output staging overlap images 1..kCnt-1 whenever the K-split count is greater than 1
        // (the common StreamK case), silently corrupting the fused result.
        const uint64_t accBytes = kCnt * rows * alignedPitch * sizeof(float);
        const uint64_t elemAlign = alignedPitch; // staging pitch matches acc pitch (already aligned)
        const uint64_t stageRowBytes = elemAlign * sizeof(OutType);
        if (accBytes >= AscendC::TOTAL_UB_SIZE || stageRowBytes > AscendC::TOTAL_UB_SIZE - accBytes) {
            return;
        }
        const int64_t stageRows = static_cast<int64_t>(Blaze::Gemm::Max(
            static_cast<uint64_t>(1), Blaze::Gemm::Min(rows, (AscendC::TOTAL_UB_SIZE - accBytes) / stageRowBytes)));

        // ── 1. 2D row-aware workspace read (CMCT model) ──
        // Per-image per-row DataCopyPad: reads nValid valid elements per row, writes to UB
        // at alignedPitch row stride. srcGap matches the workspace's row padding.
        // Per-row calls avoid DataCopyPad's dstGap semantics ambiguity (dstGap≠0 cases fail).
        AscendC::LocalTensor<float> ubAcc{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(float)};
        const uint64_t wsSrcGap = aivParams_.curAlignedNInAiv - nValid;
        for (uint64_t img = 0; img < kCnt; ++img) {
            for (uint64_t row = 0; row < rows; ++row) {
                AscendC::DataCopyExtParams wsRowParams{1, static_cast<uint32_t>(nValid * sizeof(float)),
                                                       static_cast<uint32_t>(wsSrcGap * sizeof(float)), 0, 0};
                AscendC::DataCopyPad<float>(
                    ubAcc[(img * rows + row) * alignedPitch],
                    workspaceGlobal_[copyGm2UbParams_.offsetWorkspaceGM + img * baseM_ * alignBaseN_ +
                                     row * aivParams_.curAlignedNInAiv],
                    wsRowParams, {false, 0, 0, 0});
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Blaze::Gemm::ZERO_FLAG);

        // ── 2. 2D reduce: fold images 1..kCnt-1 into image 0, row by row, valid elements only ──
        for (uint64_t img = 1; img < kCnt; ++img) {
            for (uint64_t row = 0; row < rows; ++row) {
                AscendC::Add(ubAcc[row * alignedPitch], ubAcc[row * alignedPitch],
                             ubAcc[(img * rows + row) * alignedPitch], nValid);
            }
        }
        // Drain the V-pipe Adds before the SIMT cast below (asc_vf_call is not V-pipe ordered).
        AscendC::PipeBarrier<PIPE_ALL>();

        // ── 3. Same-domain cast: FP32 -> OutType in-place (covers valid + padding, harmless) ──
        if constexpr (FusionSameDomain_ && !AscendC::IsSameType<OutType, float>::value) {
            CastAccVfParams castParams{};
            castParams.srcAddr = reinterpret_cast<__ubuf__ float*>(ubAcc.GetPhyAddr());
            castParams.dstAddr = reinterpret_cast<__ubuf__ OutType*>(ubAcc.GetPhyAddr());
            castParams.totalElems = static_cast<uint32_t>(rows * alignedPitch);
            castParams.vlF32 = AscendC::GetVecLen() / sizeof(float);
            castParams.vfLoops = (castParams.totalElems + castParams.vlF32 - 1U) / castParams.vlF32;
            asc_vf_call<CastAccFp32ToOutVf<CastAccVfParams>>(castParams);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // ── 4. Stage loop: x3 read -> VF fusion -> output write (same as before but alignedPitch) ──
        for (int64_t stageOffset = 0; stageOffset < static_cast<int64_t>(rows); stageOffset += stageRows) {
            const int64_t rowsThisStage = Blaze::Gemm::Min(stageRows, static_cast<int64_t>(rows) - stageOffset);
            const int64_t gmElemOffset = copyUb2GmParams_.offsetCGm + stageOffset * n_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FUSION_BUFFER_REUSE_EVENT);
            CopyX3Stage(accBytes, rowsThisStage, nValid, elemAlign, gmElemOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FUSION_X3_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FUSION_X3_EVENT);

            auto accAddr = reinterpret_cast<__ubuf__ FusionAccType*>(ubAcc.GetPhyAddr()) + stageOffset * alignedPitch;
            auto x3Addr = reinterpret_cast<__ubuf__ OutType*>(reinterpret_cast<uintptr_t>(ubAcc.GetPhyAddr()) +
                                                              accBytes);
            FusedElemwiseVfParams vfParams{accAddr,
                                           x3Addr,
                                           x3Addr,
                                           static_cast<uint32_t>(rowsThisStage),
                                           static_cast<uint32_t>(nValid),
                                           static_cast<uint32_t>(alignedPitch),
                                           static_cast<uint32_t>(elemAlign),
                                           AscendC::GetVecLen() / sizeof(float),
                                           static_cast<uint32_t>(Blaze::Gemm::CeilDiv(
                                               nValid, static_cast<uint64_t>(AscendC::GetVecLen() / sizeof(float))))};
            asc_vf_call<FusedElemwiseVf<FusedElemwiseVfParams>>(vfParams);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FUSION_OUTPUT_EVENT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FUSION_OUTPUT_EVENT);

            CopyOutStage(accBytes, rowsThisStage, nValid, elemAlign, gmElemOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FUSION_BUFFER_REUSE_EVENT);
        }
    }

    // Shared staging copy used by both data paths: the fused basic epilogues (staged copy-out in
    // ProcessTile) and the streamK fuse->out flow (FuseAndOut). accBytes is the
    // 32B-aligned staging start (after the accumulator image); gmElemOffset is the element offset
    // of the current tile in the output GM; nValid is the tile's valid N.
    __aicore__ inline void CopyX3Stage(uint64_t accBytes, int64_t rows, int64_t nValid, uint64_t elemPitch,
                                       int64_t gmElemOffset)
    {
        const auto origin = asc::te::make_coord(0L, 0L);
        const auto validShape = asc::te::make_shape(rows, nValid);
        auto x3Ub = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, OutType>(accBytes),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rows, static_cast<int64_t>(elemPitch)));
        auto x3UbTile = x3Ub.slice(origin, validShape);
        auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        if (!x3BatchBroadcast_) {
            auto x3Gm = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                 reinterpret_cast<__gm__ OutType*>(x3GmAddr_) + gmElemOffset),
                                             asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rows, n_));
            auto x3GmTile = x3Gm.slice(origin, validShape);
            asc::te::copy(copyGmToUb, x3UbTile, x3GmTile);
            return;
        }
        // Broadcast locating must fold the column offset onto the x3 row (consistent with the
        // basic-path staging): x3's (x3Row, col) = row folding + column part of gmElemOffset.
        // The streamK offsetCGm carries the N-tile column offset; skipping it reads wrong x3 columns.
        const int64_t firstRow = gmElemOffset / n_;
        const int64_t colOffset = gmElemOffset % n_;
        int64_t row = 0;
        while (row < rows) {
            const int64_t x3Row = (firstRow + row) % x3M_;
            const int64_t segRows = Blaze::Gemm::Min(rows - row, x3M_ - x3Row);
            auto segUb = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::ub, OutType>(accBytes + static_cast<uint64_t>(row) *
                                                                                     elemPitch * sizeof(OutType)),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(segRows, static_cast<int64_t>(elemPitch)));
            auto segGm = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(x3GmAddr_) + x3Row * n_ +
                                                             colOffset),
                asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(segRows, n_));
            auto segUbTile = segUb.slice(origin, asc::te::make_shape(segRows, nValid));
            auto segGmTile = segGm.slice(origin, asc::te::make_shape(segRows, nValid));
            asc::te::copy(copyGmToUb, segUbTile, segGmTile);
            row += segRows;
        }
    }

    __aicore__ inline void CopyOutStage(uint64_t accBytes, int64_t rows, int64_t nValid, uint64_t elemPitch,
                                        int64_t gmElemOffset)
    {
        const auto origin = asc::te::make_coord(0L, 0L);
        const auto validShape = asc::te::make_shape(rows, nValid);
        auto outUb = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, OutType>(accBytes),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rows, static_cast<int64_t>(elemPitch)));
        auto outUbTile = outUb.slice(origin, validShape);
        auto outGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ OutType*>(cGmAddr_) + gmElemOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(rows, n_));
        auto outGmTile = outGm.slice(origin, validShape);
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        asc::te::copy(copyUbToGm, outGmTile, outUbTile);
    }

    // Elementwise fusion (ADD/MUL) fields: raw addresses used by the x3 staging copies.
    GM_ADDR cGmAddr_{nullptr};
    GM_ADDR x3GmAddr_{nullptr};
    int64_t x3M_{0};
    bool x3BatchBroadcast_{false};

    uint64_t baseM_ = 0;
    uint64_t baseN_ = 0;
    uint64_t alignBaseN_ = 0;
    uint64_t windowLen_ = 4;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t UB2GM_SRCGAP_UNIT = 32UL;
};
} // namespace Block
} // namespace Epilogue

namespace Gemm::Block {
using Epilogue::Block::BlockEpilogueMatmulStreamK;
}

} // namespace Blaze
