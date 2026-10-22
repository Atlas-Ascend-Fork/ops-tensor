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
 * \file kernel_universal.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

/**
 * @class GemmUniversal
 * @brief
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class GemmUniversal {
    static_assert(always_false_v<BlockEpilogue_> && always_false_v<BlockMmad_>,
                  "GemmUniversal is not implemented for this combination of BlockMmad, BlockEpilogue and "
                  "BlockScheduler");
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
#include "blaze/gemm/kernel/kernel_batch_matmul_broadcast.h"
#include "blaze/gemm/kernel/kernel_batch_matmul_iterbatch_broadcast.h"
#include "blaze/gemm/kernel/kernel_matmul_iterbatch.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_al1_full_load.h"
#include "blaze/gemm/kernel/kernel_matmul_bl1_full_load.h"
#include "blaze/gemm/kernel/kernel_matmul_fixpipe_opti.h"
#include "blaze/gemm/kernel/kernel_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_matmul_with_scale_add.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube_without_batch.h"
#include "blaze/gemm/kernel/kernel_qgmm_cube.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/kernel/kernel_tbmm_basic.h"
#include "blaze/gemm/kernel/kernel_grouped_matmul.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.h"
