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
 * \file block_epilogue_iterbatch.h
 * \brief AIV-side ND epilogue for the IterBatch path (ND_FIXPIPE_1_2)
 */

#pragma once
#include "kernel_basic_intf.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_>
class BlockEpilogueIterbatch {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t SYNC_OFFSET = 2;

    __aicore__ inline BlockEpilogueIterbatch() {}

    struct Params {
        GM_ADDR cGmAddr{nullptr};
    };

    __aicore__ inline void Init(const Params& params, uint64_t m, uint64_t n, uint64_t totalTiles)
    {
        cGmAddr_ = reinterpret_cast<__gm__ DataTypeOut*>(params.cGmAddr);
        m_ = m;
        n_ = n;
        totalTiles_ = totalTiles;
        subBlockIdx_ = static_cast<uint64_t>(AscendC::GetSubBlockIdx()) & 0x1;
    }

    __aicore__ inline void operator()(int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1,
                                      uint64_t mainIterBatchL0)
    {
        uint64_t mL0Cnt = Gemm::CeilDiv(m_, baseM);
        uint64_t nL0Cnt = Gemm::CeilDiv(n_, baseN);
        uint64_t stepIterBatchL1L0 = Gemm::CeilDiv(curIterBatchL1, mainIterBatchL0);
        for (uint64_t iter1 = 0; iter1 < stepIterBatchL1L0; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == stepIterBatchL1L0) ? (curIterBatchL1 - mainIterBatchL0 * iter1) :
                                                                         mainIterBatchL0;
            for (uint64_t iterNL0 = 0; iterNL0 < nL0Cnt; ++iterNL0) {
                uint64_t curNL0 = (iterNL0 == nL0Cnt - 1) ? (n_ - (nL0Cnt - 1) * baseN) : baseN;
                for (uint64_t iterML0 = 0; iterML0 < mL0Cnt; ++iterML0) {
                    // Each sub-block consumes only the tiles routed into its own UB window (tile parity).
                    if ((l0CEventId_ & 0x1) == subBlockIdx_) {
                        uint64_t curML0 = (iterML0 == mL0Cnt - 1) ? (m_ - (mL0Cnt - 1) * baseM) : baseM;
                        uint64_t offsetCGMOfCopyOut = iter1 * mainIterBatchL0 * m_ * n_ + iterML0 * baseM * n_ +
                                                      iterNL0 * baseN;
                        // Compact frames: UB row pitch is the fixpipe frame width (alignNL0), GM row pitch is n_.
                        uint64_t rows = curIterBatchL0 * curML0;
                        uint64_t alignNL0 = Gemm::CeilAlign(curNL0, Gemm::BLOCK_CUBE);
                        auto ubOut = asc::te::make_tensor(
                            asc::te::make_mem_ptr<asc::te::location::ub, DataTypeIn>(0),
                            Gemm::MakeNDExtLayout<DataTypeIn>(static_cast<int64_t>(rows), static_cast<int64_t>(curNL0),
                                                              static_cast<int64_t>(alignNL0)));
                        auto gmOut = asc::te::make_tensor(
                            asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_ + static_cast<uint64_t>(offsetC) +
                                                                         offsetCGMOfCopyOut),
                            Gemm::MakeNDExtLayout<DataTypeOut>(static_cast<int64_t>(rows), static_cast<int64_t>(curNL0),
                                                               static_cast<int64_t>(n_)));
                        auto copyUB2GM = asc::te::make_copy(asc::te::copy_ub_to_gm{});
                        Gemm::Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(0x0);
                        asc::te::copy(copyUB2GM, gmOut, ubOut);
                        // Suppress the release after this sub-block's last tile so every set/wait pair closes exactly.
                        if (l0CEventId_ + SYNC_OFFSET < totalTiles_) {
                            Gemm::Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_OFFSET);
                        }
                    }
                    l0CEventId_++;
                }
            }
        }
    }

private:
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t totalTiles_{0};
    uint64_t subBlockIdx_{0};
    uint64_t l0CEventId_{0};
    __gm__ DataTypeOut* cGmAddr_ = nullptr;
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
