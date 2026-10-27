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

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

constexpr uint16_t EPILOGUE_ZERO_FLAG = 0;

class BlockEpilogueMulsAdd {
public:
    using OutputType = float;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR resultGmAddr{nullptr};
        float scale{0.00390625f};
    };

    __aicore__ inline BlockEpilogueMulsAdd()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EPILOGUE_ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EPILOGUE_ZERO_FLAG);
    }

    __aicore__ inline ~BlockEpilogueMulsAdd()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EPILOGUE_ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EPILOGUE_ZERO_FLAG);
    }

    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape const& problemShape)
    {
        cLocalLow_ = AscendC::LocalTensor<float>{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(float)};
        cLocalHigh_ = cLocalLow_[AscendC::TOTAL_UB_SIZE / Blaze::Gemm::DOUBLE_BUFFER_COUNT / sizeof(float)];

        problemShape_ = problemShape;
        scale_ = params.scale;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(params.resultGmAddr));
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset)
    {
        int64_t blockShapeM = asc::te::get<0>(blockShape);
        int64_t blockShapeN = asc::te::get<1>(blockShape);
        int64_t alignN = Blaze::Gemm::CeilAlign(blockShapeN, static_cast<int64_t>(Blaze::Gemm::C0_SIZE_fp32));
        int64_t n = asc::te::get<1>(problemShape_);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EPILOGUE_ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EPILOGUE_ZERO_FLAG);

        int32_t calCount = static_cast<int32_t>(blockShapeM * alignN);
        AscendC::Muls(cLocalLow_, cLocalLow_, scale_, calCount);
        AscendC::Add(cLocalHigh_, cLocalLow_, cLocalHigh_, calCount);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EPILOGUE_ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EPILOGUE_ZERO_FLAG);

        uint16_t blockCount = static_cast<uint16_t>(blockShapeM);
        uint32_t blockLen = static_cast<uint32_t>(blockShapeN * sizeof(float));
        uint32_t dstStride = static_cast<uint32_t>((n - blockShapeN) * sizeof(float));
        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, 0, dstStride, 0};
        AscendC::DataCopyPad<OutputType>(outputGlobal_[dstOffset], cLocalHigh_, copyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EPILOGUE_ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EPILOGUE_ZERO_FLAG);
    }

private:
    AscendC::LocalTensor<float> cLocalLow_;
    AscendC::LocalTensor<float> cLocalHigh_;
    AscendC::GlobalTensor<OutputType> outputGlobal_;
    ProblemShape problemShape_;
    float scale_{0.00390625f};
};

} // namespace Block
} // namespace Epilogue

namespace Gemm::Block {
using Epilogue::Block::BlockEpilogueMulsAdd;
}

} // namespace Blaze
