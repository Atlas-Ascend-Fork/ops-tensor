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
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {

/* block schedule policies */
struct KernelMmadWithScaleMx {};                       // Multi-block with Mx scale
struct KernelGroupedMmadWithScaleMx {};                // Grouped multi-block with Mx scale
struct KernelMmadWithScaleMxWithoutBatch {};           // Multi-block with Mx scale, without batch broadcast
struct KernelGroupedMmadWithScaleMxActivationQuant {}; // Grouped Mx matmul with AIV activation and quantization
struct KernelMmadWithScaleMxActivationQuant {}; // Multi-block with Mx scale, AIC+AIV fusion (gelu/swiglu + mx quant)
struct KernelMmadWithScaleFixpipeQuant {};      // Multi-block with fixpipe quant scale (A8W8 fixpipe)
struct KernelMmadWithScaleFixpipeQuantWithoutBatch {}; // Multi-block with fixpipe quant scale, without batch broadcast
struct KernelGroupedMmadWithScaleFixpipeQuant {};      // Grouped S8S4 with fixpipe per-channel/per-group scale
struct KernelGroupedMmadFixpipeQuant {};         // Grouped multi-block with fixpipe quant scale (A8W8 fixpipe GMM)
struct KernelMmadWithScaleMix {};                // Multi-block with fixpipe mix scale (WeightNZ)
struct KernelMmadWithScaleMixWithoutBatch {};    // Multi-block with fixpipe mix scale (WeightNZ), without batch
struct KernelMultiBlockStreamK {};               // Multi-tile transfer with K-axis spliting and caching
struct KernelQbmmMultiBlockStreamK {};           // QBMM MX StreamK schedule
struct KernelQbmmPertensorMultiBlockStreamK {};  // QBMM per-tensor StreamK schedule
struct KernelMmadMultiBlockBasic {};             // Multi-tile basic
struct KernelMmadWithEpilogue {};                // Matmul with a dedicated AIV epilogue
struct KernelMmadFmmWithScaleAdd {};             // Fused matmul with scale/add epilogue
struct KernelMmadSyrk {};                        // Symmetric rank-k update, single nd2nz fetch per row-block
struct KernelIterBatchBroadcast {};              // Multi-tile batchMatmul broadcast + iterbatch
struct KernelIterBatch {};                       // Multi-tile batchMatmul iterbatch
struct KernelMmadMultiBlockBmmBroadcast {};      // Multi-tile batchMatmul broadcast
struct KernelMmadMultiBlockAFullLoad {};         // Multi-tile aFullLoad
struct KernelMmadMultiBlockBFullLoad {};         // Multi-tile fullLoad
struct KernelMmadMultiBlockFixpipeOpti {};       // Multi-tile FixpipeOpti
struct KernelMmadMultiBlockTBMM {};              // tbmm schedule
struct KernelMmadMultiBlockTQBMM {};             // tqbmm schedule
struct KernelMmadMultiBlockTQBMMFixpipeQuant {}; // tqbmm fixpipe quant schedule (HIFP8)
struct KernelMmadMultiBlockTQBMMMx {};           // tqbmm mix schedule (FP8 per-token dequant)
struct KernelMixWithWeightPrologue {};           // Mix matmul with AIV weight preprocessing
struct KernelMixWithWeightPergroupPrologue {};   // Mix matmul with AIV per-group weight dequant preprocessing
struct KernelWqgmmMxMix {};                      // Grouped MX mix kernel with AIV weight preprocessing
struct KernelGmmSwiGluMixMx {};                  // MIX AIC+AIV schedule for GroupedMatmul + SwiGLU + MX quant
struct KernelMatmulEmuSplitWeight {};            // Double bf16 matmul to simulate fp32 (AIC+AIV)
struct KernelMmadWithScaleMxMix {};              // Multi-block with Mx scale, epilogue after block mmad
struct KernelQgmmMxMixFinalizeRouting {};        // MIX AIC+AIV schedule for GroupedMatmulFinalizeRouting MX
struct KernelGroupedMmadNoQuant {};              // Grouped multi-block without quantization
struct KernelMmadAPrefetchBAntiquant {};         // AIV weight dequantization and AIC matrix multiplication

enum class MatmulOutputMode : std::uint8_t { OVERWRITE = 0, INPLACE_ADD = 1 };
enum class MatMulL0C2Out : std::uint8_t { ON_THE_FLY = 0, ND_FIXPIPE_1_1 = 1, ND_FIXPIPE_1_2 = 2 };

/**
 * @struct MatmulWithScaleFixpipeQuant
 * @brief Quantized fixpipe matmul with scale and fixpipe dequant (Tensor API / Blaze)
 * @param [in] FullLoadMode_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] AtomicAdd_: whether to enable atomic add on output
 * @param [in] ScheduleType_: kernel schedule tag, default KernelMmadWithScaleFixpipeQuant
 * @param [in] GmmArrayPtr_: grouped shape array pointer type, preserving the caller's address space
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleFixpipeQuant,
          class GmmArrayPtr_ = __gm__ int32_t*, uint64_t NonContiguousType_ = 0>
struct MatmulWithScaleFixpipeQuant {
    using ScheduleType = ScheduleType_;
    using GmmArrayPtr = GmmArrayPtr_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulWithScaleMix
 * @brief Mix fixpipe matmul with scale and fixpipe dequant for WeightNZ (Tensor API / Blaze)
 * @param [in] FullLoadMode_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] AtomicAdd_: whether to enable atomic add on output
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMix,
          uint64_t NonContiguousType_ = 0>
struct MatmulWithScaleMix {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulWithScaleMx
 * @brief Mx Matrix multiplication with scaleA and scaleB
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMx,
          uint64_t L0C2UBMode_ = L0C2UB_MODE_NONE, uint64_t NonContiguousType_ = 0, bool ConcatN_ = false>
struct MatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
    static constexpr uint64_t L0C2UB_MODE = L0C2UBMode_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
    // Concat-N layout: the mmad produces [gate|linear] concatenated N columns, so the
    // epilogue consumes an N/2-wide output. Orthogonal to the kernel family tag.
    static constexpr bool CONCAT_N = ConcatN_;
};

/**
 * @struct MatmulWithWeightQuantMx
 * @brief Weight-only MX matrix multiplication with AIV weight conversion.
 */
struct MatmulWithWeightQuantMx {
    using ScheduleType = KernelMixWithWeightPrologue;
};

/**
 * @struct MatmulWithWeightQuantPergroup
 * @brief T-CG weight-only per-group quantized matrix multiplication.
 *        AIV converts FP4 weights (FLOAT4_E2M1) + per-group scale (BFLOAT16/FLOAT16) to FP8, writes to L1.
 *        AIC performs FP8 x FP8 matmul with per-channel output quant via yScale (UINT64/INT64) in fixpipe.
 *        Formula: out = (x1 @ (x2 * x2Scale)) * yScale
 */
struct MatmulWithWeightQuantPergroup {
    using ScheduleType = KernelMixWithWeightPergroupPrologue;
    struct SyncProtocol {
        static constexpr uint16_t MODE = 4U;
        static constexpr uint16_t AIV_READY_FLAG = 1U;
        static constexpr uint16_t AIC_FREE_FLAG = 2U;
        static constexpr uint16_t FLAG_ID_MAX = 16U;
    };
};

/**
 * @struct GroupedMatmulWithWeightQuantMx
 * @brief Dispatch policy for grouped MX weight-quant matrix
 * multiplication.
 */
struct GroupedMatmulWithWeightQuantMx {
    using ScheduleType = KernelWqgmmMxMix;
};

/**
 * @struct GroupedMatmulWithScaleMx
 * @brief Grouped Mx matrix multiplication with scaleA and scaleB
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelGroupedMmadWithScaleMx>
struct GroupedMatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulWithScaleMxL0CPingpong
 * @brief Mx matrix multiplication with L0C ping-pong perf schedule.
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMx>
struct MatmulWithScaleMxL0CPingpong {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr MatMulL0C2Out FIXP_OPTI = FixpOpti_;
};

/**
 * @struct MatmulMultiBlockWithStreamKSplitK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 *  @param [in] KernelSchedule_: mmad dispatch policy
 */
template <MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, bool IsSplitSinglecoreK_ = true,
          class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamKSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr MatMulL0C2Out FIXP_OPTI = FixpOpti_;
    static constexpr bool IS_SPLIT_SINGLECORE_K = IsSplitSinglecoreK_;
};

/**
 * @struct MatmulMultiBlockBasic
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: matmul support non-contiguous scene such as: slice, transpose
 * @param [in] OutputMode_: grouped matmul output mode, default is overwrite
 */
template <uint64_t FullLoadMode_ = 0, uint64_t FusedOpType_ = 0, class KernelSchedule_ = KernelMmadMultiBlockBasic,
          uint64_t NonContiguousType_ = 0, MatmulOutputMode OutputMode_ = MatmulOutputMode::OVERWRITE>
struct MatmulMultiBlockBasic {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
    static constexpr MatmulOutputMode OUTPUT_MODE = OutputMode_;
};

/**
 * @struct MatmulEmuSplitWeightPolicy
 * @brief Dual matmul add dispatch policy for AIC+AIV fused dual-weight matmul (Tensor API / Blaze)
 */
struct MatmulEmuSplitWeightPolicy {
    using ScheduleType = KernelMatmulEmuSplitWeight;
};

/**
 * @struct BatchMatmulIterbatchBroadcast
 * @brief Matrix multiplication with batch broadcast, no quant, implemented based on Layout
 * @param [in] ABroadcast: A tensor needs broadcast
 * @param [in] BBroadcast: B tensor needs broadcast
 */
template <bool ABroadcast_, bool BBroadcast_>
struct MatmulIterBatchBroadcast {
    using ScheduleType = KernelIterBatchBroadcast;
    static constexpr bool A_BROADCAST = ABroadcast_;
    static constexpr bool B_BROADCAST = BBroadcast_;
};

/**
 * @struct MatmulIterBatch
 * @brief BatchMatMul with iterbatch L1/L0 pipelining and cross-tile preload,
 *        ON_THE_FLY or ND_FIXPIPE_1_2 out mode (B non-contiguous not supported)
 * @param [in] FixpOpt_: L0C out mode, ON_THE_FLY (AIC fixpipe to GM) or ND_FIXPIPE_1_2 (MIX,
 *        AIC fixpipe to AIV UB slot + AIV ND epilogue)
 * @param [in] FusedOpType_: post-mmad fusion op type, default 0 (OP_TYPE_EMPTY); reserved,
 *        no consumer yet
 * @param [in] NonContiguousType_: reserved for the B non-contiguous scenario,
 *        default 0 (contiguous); no consumer yet
 */
template <MatMulL0C2Out FixpOpt_ = MatMulL0C2Out::ON_THE_FLY, uint64_t FusedOpType_ = 0,
          uint64_t NonContiguousType_ = 0>
struct MatmulIterBatch {
    using ScheduleType = KernelIterBatch;
    static constexpr MatMulL0C2Out FIXP_OPT = FixpOpt_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockBasicSplitK
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: 0 indicates support for continuity
 */
template <uint64_t FullLoadMode_ = 0, bool IsSplitSinglecoreK_ = true,
          class KernelSchedule_ = KernelMmadMultiBlockBasic, uint64_t NonContiguousType_ = 0>
struct MatmulMultiBlockBasicSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_SPLIT_SINGLECORE_K = IsSplitSinglecoreK_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockAFullLoad
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(A_FULL_LOAD_MODE)
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <uint64_t FullLoadMode_ = A_FULL_LOAD_MODE, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockAFullLoad, uint64_t NonContiguousType_ = 0>
struct MatmulMultiBlockAFullLoad {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockBFullLoad
 * @brief Matrix multiplication B full-load structure, with optional fixpipe output mode
 * @param [in] L0C2OutModel_: mode of L0C out mode, default is ON_THE_FLY(L0C2GM)
 * @param [in] FusedOpType_: execute fusion after mmad, default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */

template <uint64_t L0C2OutModel_ = ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockBFullLoad>
struct MatmulMultiBlockBFullLoad {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = B_FULL_LOAD_MODE;
    static constexpr uint64_t L0C2OUT_MODEL = L0C2OutModel_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
};

/**
 * @struct MatmulMultiBlockFixpipeOpti
 * @brief Matrix multiplication fixpipe optimization without B full-load
 * @param [in] FusedOpType_: execute fusion after mmad, default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <uint64_t L0C2OutModel_ = ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockFixpipeOpti>
struct MatmulMultiBlockFixpipeOpti {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = NONE_FULL_LOAD_MODE;
    static constexpr uint64_t L0C2OUT_MODEL = L0C2OutModel_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr MatmulOutputMode OUTPUT_MODE = MatmulOutputMode::OVERWRITE;
};

/**
 * @struct MatmulWithWeightAntiquant
 * @brief Blaze dispatch policy for 8-bit weight dequantization followed by AIC matrix multiplication.
 * @param [in] AivNum_: number of AIV sub-blocks per AIC; must be 2
 * @param [in] UbMte2InnerSize_: weight input UB row pitch in bytes
 * @param [in] UbMte2BufNum_: number of prologue input buffers; only 2 or 4 are supported
 * @param [in] AntiquantType_: per-tensor or per-channel antiquantization
 * @param [in] HasAntiquantOffset_: whether antiquantOffset is enabled
 */
template <uint64_t AivNum_ = 2, uint32_t UbMte2InnerSize_ = 512, uint32_t UbMte2BufNum_ = 2,
          QuantMode AntiquantType_ = QuantMode::PERCHANNEL_MODE, bool HasAntiquantOffset_ = false>
struct MatmulWithWeightAntiquant {
    using ScheduleType = KernelMmadAPrefetchBAntiquant;
    struct SyncProtocol {
        static constexpr uint16_t MODE = 4;
        static constexpr uint16_t AIV_READY_FLAG = 9;
        static constexpr uint16_t AIC_FREE_FLAG = 8;
        static constexpr uint16_t FLAG_ID_MAX = 16;
    };

    static constexpr uint64_t AIV_NUM = AivNum_;
    static constexpr uint32_t UB_MTE2_INNER_SIZE = UbMte2InnerSize_;
    static constexpr uint32_t UB_MTE2_BUFFER_NUM = UbMte2BufNum_;
    static constexpr QuantMode ANTIQUANT_TYPE = AntiquantType_;
    static constexpr bool HAS_ANTIQUANT_OFFSET = HasAntiquantOffset_;

    static_assert(AivNum_ == 2, "Weight antiquantization requires two AIV sub-blocks");
    static_assert(UbMte2BufNum_ == 2 || UbMte2BufNum_ == 4,
                  "Weight antiquantization supports only 2 or 4 UB input buffers");
    static_assert(AntiquantType_ == QuantMode::PERTENSOR_MODE || AntiquantType_ == QuantMode::PERCHANNEL_MODE,
                  "Weight antiquantization supports only per-tensor or per-channel quantization");
};

/**
 * @struct MatmulSyrk
 * @brief Symmetric rank-k update (C = alpha * (A @ A^T) + beta * C) policy.
 *
 * The syrk block loads each A row-block from GM into L1 exactly once per
 * (row-block pair, k-chunk) via a single nd2nz CopyGM2L1. The NZ arrangement
 * of a row-block X(m, k) is byte-identical to the ZN arrangement of X^T(k, m),
 * so one L1 image feeds both cube inputs: an NZ view sources L0A
 * (CopyL12L0A) and a ZN view sources L0B (CopyL12L0B). Upper-triangle tile
 * pairs (i, j) / (j, i) share the two row-block fetches, halving the total
 * GM->L1 traffic versus a generic matmul composition.
 */
struct MatmulSyrk {
    using ScheduleType = KernelMmadSyrk;
};

} // namespace Gemm
} // namespace Blaze
