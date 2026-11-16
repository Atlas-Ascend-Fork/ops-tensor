/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "tensor_api/tensor.h"

namespace Blaze::Epilogue::Block {

using TileShape = asc::te::shape<int64_t, int64_t>;

struct OutputOffsets {
    int64_t yOffset{0};
    int64_t yScaleOffset{0};
};

struct AccumulatorTensorParams {
    int64_t rows{0};
    int64_t cols{0};
    uint32_t slot{0};
    int64_t batch{1};
};

} // namespace Blaze::Epilogue::Block
