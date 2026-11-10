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
#include <cstdint>
namespace WeightQuantBatchMatmulUT {
struct CopyInstruction {
    uint64_t calls = 0, src = 0, dst = 0;
    uint16_t count = 0, length = 0, srcGap = 0, dstGap = 0;
    uint8_t sid = 0;
};
inline thread_local CopyInstruction* observedCopy = nullptr;
// Active only on AIV while a weight-antiquant execution test is running.
// tikicpulib selects the calling AIV's bank automatically; the device policy
// encodes that selection explicitly in bit 4. The instruction wrapper removes
// that bit before forwarding and records every real SDK handoff.
struct AntiquantSyncTrace {
    uint64_t sets = 0, waits = 0, normalizedSets = 0, normalizedWaits = 0, unexpectedFlags = 0;
};
inline thread_local AntiquantSyncTrace* antiquantSyncTrace = nullptr;
} // namespace WeightQuantBatchMatmulUT
