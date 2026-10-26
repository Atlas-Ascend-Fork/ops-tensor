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
 * \file block_mmad.h
 * \brief Aggregate header for attention BlockMmad. Pulls in the BlockMmad primary template declaration
 * (block_mmad_decl.h) and then all aggregated block_mmad_xx.h specialization fragments. Users of
 * these blocks include this header only; the fragments themselves include block_mmad_decl.h
 * so they stay self-contained without including this header back.
 */
#pragma once

#include "blaze/attention/block/block_mmad_decl.h"

#include "blaze/attention/block/block_mmad_flat_quant.h"
