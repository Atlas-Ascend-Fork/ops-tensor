// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
/*!
 * \file block_epilogue_dual_level_scale.h
 * \brief FP32 two-level MX scaling and accumulation for split-M GMM.
 */
#pragma once

#include "kernel_operator.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/epilogue/block/block_epilogue_types.h"

namespace Blaze::Epilogue::Block {
template <class Output_>
class BlockEpilogueDualLevelScale {
public:
    using AccumulatorTensorParams = Blaze::Epilogue::Block::AccumulatorTensorParams;

    struct Params {
        __gm__ float* scaleA{nullptr}; // A 的 level0 scale 地址。
        __gm__ float* scaleB{nullptr}; // B 的 level0 scale 地址。
        __gm__ float* bias{nullptr};
        __gm__ Output_* output{nullptr};
        int64_t n;
        int64_t baseM;
        int64_t baseN;
        int64_t segmentCount;
    };

    // Only an AIV that completed Init owns these local events.
    __aicore__ inline ~BlockEpilogueDualLevelScale();

    // Set the UB view on AIC and AIV. Only AIV binds GM tensors and initializes local events.
    template <int64_t PartialRows, int64_t PartialCols, int64_t RowAlignment, uint32_t PartialBufferCount>
    __aicore__ inline void Init(const Params& params);

    __aicore__ inline auto GetAccumulatorTensor(const AccumulatorTensorParams& params) const;

    // AIV consumes all partials for one scheduled output block. Offsets are in GM elements.
    template <class PartialTiles_, class Channel_>
    __aicore__ inline void operator()(const asc::te::shape<int64_t, int64_t, int64_t, int64_t>& shape,
                                      int64_t scaleAOffset, int64_t scaleBOffset, int64_t biasOffset,
                                      int64_t outputOffset, bool hasBias, Channel_& channel);

private:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct BlockParams {
        int64_t scaleAOffset;
        int64_t scaleBOffset;
        int64_t biasOffset;
        int64_t outputOffset;
        bool hasBias;
    };

    static constexpr int64_t SPLIT_M_AIV_COUNT = 2;

    using Output = Output_;
    static constexpr int64_t SCALE0_WINDOW = 32; // Level0 segments cached in the scale A UB window.
    static constexpr int64_t FP32_FRAME_C0 = 8;
    static constexpr uint32_t DMA_BLOCK_BYTES = AscendC::ONE_BLK_SIZE;
    static constexpr uint16_t REGISTER_COLS = 64;
    static constexpr uint8_t EVENT_ID = 0;

    static_assert(AscendC::Std::is_one_of_v<Output, half, bfloat16_t>, "Dual-level output must be FP16 or BF16.");

    __aicore__ inline void LoadScales(const BlockParams& tile, int64_t rows, int64_t cols, int64_t segmentIdx);
    template <class PartialTile_, class Channel_>
    __aicore__ inline void Consume(const PartialTile_& partial, int64_t segmentIdx, const BlockParams& tile,
                                   Channel_& channel);

    struct RowRange {
        int64_t first;
        int64_t count;
    };

    static __aicore__ inline RowRange SplitRows(int64_t rows);

    __aicore__ inline uint32_t GetAccumulatorIndex(int64_t m, int64_t n) const;

    __aicore__ inline void LoadScaleAWindow(int64_t scaleAOffset, int64_t tileRows, int64_t segmentIdx,
                                            const AscendC::DataCopyPadExtParams<float>& pad);

    __aicore__ inline void Accumulate(int64_t m, int64_t n, int64_t cols, int64_t segmentIdx, uint32_t buffer,
                                      int64_t rows, uint32_t accumulatorIndex, bool isLast, bool hasBias);

    __aicore__ inline void StoreOutput(int64_t outputOffset, int64_t rows, int64_t cols, uint32_t accumulatorIndex);

    // FP32 partial and accumulator rows use the same ND pitch.
    // On the final K segment, the converted row is stored at the same byte address.
    class ScaleTile {
    public:
        static __aicore__ inline void Run(__ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0,
                                          __ubuf__ float* b0, __ubuf__ float* bias, uint16_t rows, uint16_t cols,
                                          uint16_t partialPitch, uint16_t aPitch, bool first, bool final, bool hasBias);

    private:
#ifdef __NPU_ARCH__
        static constexpr AscendC::Reg::CastTrait CAST_OUTPUT = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        template <bool First>
        static __simd_callee__ inline void LoadSum(AscendC::Reg::RegTensor<float>& sum, __ubuf__ float* acc,
                                                   AscendC::Reg::MaskReg& mask);

        template <bool Final, bool Wide, bool HasBias>
        static __simd_callee__ inline void StoreSums(__ubuf__ float* acc, AscendC::Reg::RegTensor<float>& sum0,
                                                     AscendC::Reg::RegTensor<float>& sum1,
                                                     AscendC::Reg::RegTensor<float>& bias0,
                                                     AscendC::Reg::RegTensor<float>& bias1,
                                                     AscendC::Reg::MaskReg& mask);

        // Compute A0*B0 before MulAddDst; add bias only after the last K segment.
        template <bool First, bool Final, bool Wide, bool HasBias>
        static __simd_vf__ inline void ScaleAccumulate(__ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0,
                                                       __ubuf__ float* b0, __ubuf__ float* bias, uint16_t rows,
                                                       uint16_t partialPitch, uint16_t aPitch);

        template <bool Final, bool Wide>
        static __aicore__ inline void DispatchScale(__ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0,
                                                    __ubuf__ float* b0, __ubuf__ float* bias, uint16_t rows,
                                                    uint16_t partialPitch, uint16_t aPitch, bool first, bool hasBias);
#endif
    };

    AscendC::LocalTensor<float> partial_;
    AscendC::LocalTensor<float> accumulator_;
    AscendC::LocalTensor<float> scaleAUb_;
    AscendC::LocalTensor<float> scaleBUb_;
    AscendC::LocalTensor<float> biasUb_;
    AscendC::GlobalTensor<float> scaleAGm_;
    AscendC::GlobalTensor<float> scaleBGm_;
    AscendC::GlobalTensor<float> biasGm_;
    AscendC::GlobalTensor<Output> outputGm_;
    int64_t partialRows_{0};
    int64_t partialCols_{0};
    int64_t partialRowAlignment_{0};
    int64_t aivRows_{0};
    uint32_t partialElems_{0};
    uint32_t accumulatorOffset_{0};
    int64_t mTiles_{0};
    int64_t n_{0};
    int64_t segmentCount_{0};
    int64_t scaleARowPitch_{0};
    bool initialized_{false};
};

template <class Output_>
template <int64_t PartialRows, int64_t PartialCols, int64_t RowAlignment, uint32_t PartialBufferCount>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::Init(const Params& params)
{
    static_assert(RowAlignment == SPLIT_M_AIV_COUNT && PartialRows > 0 && PartialRows % SPLIT_M_AIV_COUNT == 0,
                  "The split-M partial must provide an equal number of aligned rows per AIV.");
    static_assert(PartialCols == REGISTER_COLS || PartialCols == 2 * REGISTER_COLS,
                  "The scale VF supports one or two FP32 registers per UB row.");
    constexpr int64_t aivRows = PartialRows / SPLIT_M_AIV_COUNT;
    constexpr uint32_t partialBytes = aivRows * PartialCols * sizeof(float);
    static_assert(partialBytes % DMA_BLOCK_BYTES == 0 && PartialCols * sizeof(float) % DMA_BLOCK_BYTES == 0,
                  "Partial and output rows must use a DMA-aligned UB layout.");
    partialRows_ = PartialRows;
    partialCols_ = PartialCols;
    partialRowAlignment_ = RowAlignment;
    aivRows_ = aivRows;
    partialElems_ = partialBytes / sizeof(float);
    if ASCEND_IS_AIC {
        return;
    }

    const int64_t baseN = params.baseN;
    mTiles_ = Gemm::CeilDiv(params.baseM, PartialRows);
    const int64_t nTiles = Gemm::CeilDiv(baseN, PartialCols);
    const uint32_t accumulatorBytes = mTiles_ * nTiles * partialBytes;
    const uint32_t scaleABytes = mTiles_ * aivRows * SCALE0_WINDOW * sizeof(float);
    accumulatorOffset_ = PartialBufferCount * partialBytes;
    const uint32_t scaleAOffset = accumulatorOffset_ + accumulatorBytes;
    const uint32_t scaleBOffset = scaleAOffset + scaleABytes;
    const uint32_t biasOffset = scaleBOffset + baseN * sizeof(float);
    partial_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, 0, PartialBufferCount * partialElems_);
    accumulator_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, accumulatorOffset_,
                                               accumulatorBytes / sizeof(float));
    scaleAUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, scaleAOffset, scaleABytes / sizeof(float));
    scaleBUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, scaleBOffset, baseN);
    biasUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, biasOffset, baseN);
    scaleAGm_.SetGlobalBuffer(params.scaleA);
    scaleBGm_.SetGlobalBuffer(params.scaleB);
    biasGm_.SetGlobalBuffer(params.bias);
    outputGm_.SetGlobalBuffer(params.output);
    n_ = params.n;
    segmentCount_ = params.segmentCount;
    scaleARowPitch_ = Gemm::Min<int64_t>(segmentCount_, SCALE0_WINDOW);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
    initialized_ = true;
}

template <class Output_>
__aicore__ inline auto BlockEpilogueDualLevelScale<Output_>::GetAccumulatorTensor(
    const AccumulatorTensorParams& params) const
{
    return asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::ub, float>(params.slot * partialElems_ *
                                                            static_cast<uint32_t>(sizeof(float))),
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<FP32_FRAME_C0>>(
            Gemm::CeilAlign(params.rows, partialRowAlignment_), partialCols_));
}

template <class Output_>
template <class PartialTiles_, class Channel_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::operator()(const BlockShape& shape, int64_t scaleAOffset,
                                                                        int64_t scaleBOffset, int64_t biasOffset,
                                                                        int64_t outputOffset, bool hasBias,
                                                                        Channel_& channel)
{
    const BlockParams tile{scaleAOffset, scaleBOffset, biasOffset, outputOffset, hasBias};
    const int64_t rows = asc::te::get<Gemm::MNK_M>(shape);
    const int64_t cols = asc::te::get<Gemm::MNK_N>(shape);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
    for (int64_t segmentIdx = 0; segmentIdx < segmentCount_; ++segmentIdx) {
        LoadScales(tile, rows, cols, segmentIdx);
        PartialTiles_ partials(rows, cols);
        typename PartialTiles_::Tile partial;
        while (partials.Next(partial)) {
            Consume(partial, segmentIdx, tile, channel);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
}

template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::LoadScales(const BlockParams& tile, int64_t rows,
                                                                        int64_t cols, int64_t segmentIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
    const AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    if (segmentIdx % SCALE0_WINDOW == 0) {
        LoadScaleAWindow(tile.scaleAOffset, rows, segmentIdx, pad);
    }
    const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(cols * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(scaleBUb_, scaleBGm_[tile.scaleBOffset + segmentIdx * n_], copy, pad);
    if (segmentIdx == 0 && tile.hasBias) {
        AscendC::DataCopyPad(biasUb_, biasGm_[tile.biasOffset], copy, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);
}

template <class Output_>
template <class PartialTile_, class Channel_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::Consume(const PartialTile_& partial, int64_t segmentIdx,
                                                                     const BlockParams& tile, Channel_& channel)
{
    const auto rows = SplitRows(partial.rows);
    const int64_t m = partial.m;
    const int64_t n = partial.n;
    const bool isLast = segmentIdx + 1 == segmentCount_;
    const uint32_t accumulatorIndex = GetAccumulatorIndex(m, n);
    channel.WaitReady();
    if (rows.count != 0) {
        Accumulate(m, n, partial.cols, segmentIdx, channel.BufferIndex(), rows.count, accumulatorIndex, isLast,
                   tile.hasBias);
    }
    // Release Fixpipe input on PIPE_V even for a zero-row AIV. Output DMA owns only the accumulator.
    channel.Release();
    if (isLast && rows.count != 0) {
        const int64_t outputOffset = tile.outputOffset + (m + rows.first) * n_ + n;
        StoreOutput(outputOffset, rows.count, partial.cols, accumulatorIndex);
    }
}

template <class Output_>
__aicore__ inline BlockEpilogueDualLevelScale<Output_>::~BlockEpilogueDualLevelScale()
{
    if ASCEND_IS_AIV {
        if (initialized_) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
        }
    }
}

template <class Output_>
__aicore__ inline typename BlockEpilogueDualLevelScale<Output_>::RowRange
BlockEpilogueDualLevelScale<Output_>::SplitRows(int64_t rows)
{
    const int64_t firstHalf = (rows + 1) >> 1;
    return AscendC::GetSubBlockIdx() == 0 ? RowRange{0, firstHalf} : RowRange{firstHalf, rows - firstHalf};
}

template <class Output_>
__aicore__ inline uint32_t BlockEpilogueDualLevelScale<Output_>::GetAccumulatorIndex(int64_t m, int64_t n) const
{
    return (n / partialCols_ * mTiles_ + m / partialRows_) * partialElems_;
}

template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::LoadScaleAWindow(
    int64_t scaleAOffset, int64_t tileRows, int64_t segmentIdx, const AscendC::DataCopyPadExtParams<float>& pad)
{
    const int64_t span = Gemm::Min<int64_t>(SCALE0_WINDOW, segmentCount_ - segmentIdx);
    const uint32_t spanBytes = span * sizeof(float);
    for (int64_t m = 0; m < tileRows; m += partialRows_) {
        const auto rows = SplitRows(Gemm::Min(partialRows_, tileRows - m));
        if (rows.count == 0) {
            continue;
        }
        const int64_t ubIndex = m / partialRows_ * aivRows_ * scaleARowPitch_;
        const int64_t gmIndex = scaleAOffset + (m + rows.first) * segmentCount_ + segmentIdx;
        if (segmentCount_ <= SCALE0_WINDOW) {
            // The complete scale rows are contiguous for short K.
            const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(rows.count * spanBytes), 0, 0, 0};
            AscendC::DataCopyPad(scaleAUb_[ubIndex], scaleAGm_[gmIndex], copy, pad);
        } else {
            const uint32_t pitchBytes = SCALE0_WINDOW * sizeof(float);
            const uint32_t spanBlocks = Gemm::CeilDiv(spanBytes, DMA_BLOCK_BYTES);
            const AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows.count), spanBytes,
                                                  static_cast<uint32_t>((segmentCount_ - span) * sizeof(float)),
                                                  pitchBytes / DMA_BLOCK_BYTES - spanBlocks, 0};
            AscendC::DataCopyPad(scaleAUb_[ubIndex], scaleAGm_[gmIndex], copy, pad);
        }
    }
}

template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::Accumulate(int64_t m, int64_t n, int64_t cols,
                                                                        int64_t segmentIdx, uint32_t buffer,
                                                                        int64_t rows, uint32_t accumulatorIndex,
                                                                        bool isLast, bool hasBias)
{
    const int64_t scaleAIndex = m / partialRows_ * aivRows_ * scaleARowPitch_ + segmentIdx % SCALE0_WINDOW;
    ScaleTile::Run(reinterpret_cast<__ubuf__ float*>(partial_.GetPhyAddr()) + buffer * partialElems_,
                   reinterpret_cast<__ubuf__ float*>(accumulator_.GetPhyAddr()) + accumulatorIndex,
                   reinterpret_cast<__ubuf__ float*>(scaleAUb_.GetPhyAddr()) + scaleAIndex,
                   reinterpret_cast<__ubuf__ float*>(scaleBUb_.GetPhyAddr()) + n,
                   reinterpret_cast<__ubuf__ float*>(biasUb_.GetPhyAddr()) + n, rows, cols, partialCols_,
                   scaleARowPitch_, segmentIdx == 0, isLast, hasBias);
}

template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::StoreOutput(int64_t outputOffset, int64_t rows,
                                                                         int64_t cols, uint32_t accumulatorIndex)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
    // The last VF casts in place while retaining the FP32 accumulator's byte pitch.
    const uint32_t outputRowBytes = partialCols_ * sizeof(float);
    const int64_t outputRowPitch = outputRowBytes / sizeof(Output);
    AscendC::LocalTensor<Output> out{
        AscendC::TPosition::VECOUT, accumulatorOffset_ + accumulatorIndex * static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(Gemm::CeilAlign((rows - 1) * outputRowPitch + cols,
                                              static_cast<int64_t>(DMA_BLOCK_BYTES / sizeof(Output))))};
    const uint32_t validBytes = cols * sizeof(Output);
    const uint32_t validBlocks = Gemm::CeilDiv(validBytes, DMA_BLOCK_BYTES);
    const AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), validBytes,
                                          outputRowBytes / DMA_BLOCK_BYTES - validBlocks,
                                          static_cast<uint32_t>((n_ - cols) * sizeof(Output)), 0};
    AscendC::DataCopyPad(outputGm_[outputOffset], out, copy);
}

#ifdef __NPU_ARCH__
template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::Run(
    __ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0, __ubuf__ float* b0, __ubuf__ float* bias,
    uint16_t rows, uint16_t cols, uint16_t partialPitch, uint16_t aPitch, bool first, bool final, bool hasBias)
{
    if (final) {
        if (cols > REGISTER_COLS) {
            DispatchScale<true, true>(partial, acc, a0, b0, bias, rows, partialPitch, aPitch, first, hasBias);
        } else {
            DispatchScale<true, false>(partial, acc, a0, b0, bias, rows, partialPitch, aPitch, first, hasBias);
        }
    } else if (cols > REGISTER_COLS) {
        DispatchScale<false, true>(partial, acc, a0, b0, bias, rows, partialPitch, aPitch, first, false);
    } else {
        DispatchScale<false, false>(partial, acc, a0, b0, bias, rows, partialPitch, aPitch, first, false);
    }
}

template <class Output_>
template <bool First>
__simd_callee__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::LoadSum(
    AscendC::Reg::RegTensor<float>& sum, __ubuf__ float* acc, AscendC::Reg::MaskReg& mask)
{
    if constexpr (First) {
        AscendC::Reg::Duplicate(sum, 0.0F, mask);
    } else {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sum, acc);
    }
}

template <class Output_>
template <bool Final, bool Wide, bool HasBias>
__simd_callee__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::StoreSums(
    __ubuf__ float* acc, AscendC::Reg::RegTensor<float>& sum0, AscendC::Reg::RegTensor<float>& sum1,
    AscendC::Reg::RegTensor<float>& bias0, AscendC::Reg::RegTensor<float>& bias1, AscendC::Reg::MaskReg& mask)
{
    namespace R = AscendC::Reg;
    if constexpr (Final) {
        if constexpr (HasBias) {
            R::Add(sum0, sum0, bias0, mask);
            if constexpr (Wide) {
                R::Add(sum1, sum1, bias1, mask);
            }
        }
        R::RegTensor<Output> out0, out1;
        R::Cast<Output, float, CAST_OUTPUT>(out0, sum0, mask);
        if constexpr (Wide) {
            R::Cast<Output, float, CAST_OUTPUT>(out1, sum1, mask);
        }
        R::StoreAlign<Output, R::StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ Output*>(acc), out0, mask);
        if constexpr (Wide) {
            R::StoreAlign<Output, R::StoreDist::DIST_PACK_B32>(reinterpret_cast<__ubuf__ Output*>(acc) + REGISTER_COLS,
                                                               out1, mask);
        }
    } else {
        R::StoreAlign<float, R::StoreDist::DIST_NORM_B32>(acc, sum0, mask);
        if constexpr (Wide) {
            R::StoreAlign<float, R::StoreDist::DIST_NORM_B32>(acc + REGISTER_COLS, sum1, mask);
        }
    }
}

template <class Output_>
template <bool First, bool Final, bool Wide, bool HasBias>
__simd_vf__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::ScaleAccumulate(
    __ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0, __ubuf__ float* b0, __ubuf__ float* bias,
    uint16_t rows, uint16_t partialPitch, uint16_t aPitch)
{
    namespace R = AscendC::Reg;
    R::MaskReg mask = R::CreateMask<uint8_t, R::MaskPattern::ALL>();
    R::RegTensor<float> column0, column1, bias0, bias1;
    R::LoadAlign<float, R::LoadDist::DIST_NORM>(column0, b0);
    if constexpr (Wide) {
        R::LoadAlign<float, R::LoadDist::DIST_NORM>(column1, b0 + REGISTER_COLS);
    }
    if constexpr (Final && HasBias) {
        R::LoadAlign<float, R::LoadDist::DIST_NORM>(bias0, bias);
        if constexpr (Wide) {
            R::LoadAlign<float, R::LoadDist::DIST_NORM>(bias1, bias + REGISTER_COLS);
        }
    }
    for (uint16_t r = 0; r < rows; ++r) {
        R::RegTensor<float> row, scale0, scale1, value0, value1, sum0, sum1;
        R::LoadAlign<float, R::LoadDist::DIST_BRC_B32>(row, a0 + r * aPitch);
        R::LoadAlign<float, R::LoadDist::DIST_NORM>(value0, partial + r * partialPitch);
        LoadSum<First>(sum0, acc + r * partialPitch, mask);
        if constexpr (Wide) {
            R::LoadAlign<float, R::LoadDist::DIST_NORM>(value1, partial + r * partialPitch + REGISTER_COLS);
            LoadSum<First>(sum1, acc + r * partialPitch + REGISTER_COLS, mask);
        }
        R::Mul(scale0, row, column0, mask);
        if constexpr (Wide) {
            R::Mul(scale1, row, column1, mask);
        }
        R::MulAddDst(sum0, value0, scale0, mask);
        if constexpr (Wide) {
            R::MulAddDst(sum1, value1, scale1, mask);
        }
        StoreSums<Final, Wide, HasBias>(acc + r * partialPitch, sum0, sum1, bias0, bias1, mask);
    }
    if constexpr (!Final) {
        R::LocalMemBar<R::MemType::VEC_STORE, R::MemType::VEC_LOAD>();
    }
}

template <class Output_>
template <bool Final, bool Wide>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::DispatchScale(
    __ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0, __ubuf__ float* b0, __ubuf__ float* bias,
    uint16_t rows, uint16_t partialPitch, uint16_t aPitch, bool first, bool hasBias)
{
    if (first) {
        if (Final && hasBias) {
            asc_vf_call<ScaleAccumulate<true, Final, Wide, true>>(partial, acc, a0, b0, bias, rows, partialPitch,
                                                                  aPitch);
        } else {
            asc_vf_call<ScaleAccumulate<true, Final, Wide, false>>(partial, acc, a0, b0, bias, rows, partialPitch,
                                                                   aPitch);
        }
    } else {
        if (Final && hasBias) {
            asc_vf_call<ScaleAccumulate<false, Final, Wide, true>>(partial, acc, a0, b0, bias, rows, partialPitch,
                                                                   aPitch);
        } else {
            asc_vf_call<ScaleAccumulate<false, Final, Wide, false>>(partial, acc, a0, b0, bias, rows, partialPitch,
                                                                    aPitch);
        }
    }
}
#else
template <class Output_>
__aicore__ inline void BlockEpilogueDualLevelScale<Output_>::ScaleTile::Run(
    __ubuf__ float* partial, __ubuf__ float* acc, __ubuf__ float* a0, __ubuf__ float* b0, __ubuf__ float* bias,
    uint16_t rows, uint16_t cols, uint16_t partialPitch, uint16_t aPitch, bool first, bool final, bool hasBias)
{}
#endif
} // namespace Blaze::Epilogue::Block
