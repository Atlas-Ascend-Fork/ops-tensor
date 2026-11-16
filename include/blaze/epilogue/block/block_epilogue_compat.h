/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_compat.h
 * \brief Compile-time adapters for block epilogue interface versions.
 */

#pragma once

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

template <typename BlockEpilogue_, typename EpilogueParams_, typename = void>
struct HasParamsOnlyInit {
    static constexpr bool value = false;
};

template <typename BlockEpilogue_, typename EpilogueParams_>
struct HasParamsOnlyInit<BlockEpilogue_, EpilogueParams_,
                         asc::te::void_t<decltype(AscendC::Std::declval<BlockEpilogue_&>().Init(
                             AscendC::Std::declval<const EpilogueParams_&>()))>> {
    static constexpr bool value = true;
};

template <typename BlockEpilogue_, typename = void>
struct HasGetAccumulatorTensor {
    static constexpr bool value = false;
};

template <typename BlockEpilogue_>
struct HasGetAccumulatorTensor<
    BlockEpilogue_, asc::te::void_t<decltype(AscendC::Std::declval<const BlockEpilogue_&>().GetAccumulatorTensor(
                        AscendC::Std::declval<const typename BlockEpilogue_::AccumulatorTensorParams&>()))>> {
    static constexpr bool value = true;
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
