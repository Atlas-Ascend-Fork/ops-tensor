/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_universal.h
 * \brief Aggregate header for Attention kernels. Pulls in the AttentionUniversal primary template
 * declaration (kernel_universal_decl.h) and then all aggregated kernel_xx.h specialization
 * fragments. Users of these kernels include this header only; the fragments themselves include
 * kernel_universal_decl.h so they stay self-contained without including this header back.
 */
#pragma once

#include "blaze/attention/kernel/kernel_universal_decl.h"

#include "blaze/attention/kernel/kernel_flat_quant.h"
#include "blaze/attention/kernel/kernel_block_attn_res_prepare.h"
