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
 * \brief Aggregate header for GEMM BlockMmad. Pulls in the BlockMmad primary template declaration
 * (block_mmad_decl.h) and then every block_mmad_xx.h specialization fragment in this directory.
 * Users of these blocks include this header only; the fragments themselves include block_mmad_decl.h
 * so they stay self-contained without including this header back.
 */
#pragma once

#include "blaze/gemm/block/block_mmad_decl.h"

#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_mmad_confidence_head_fused.h"
#include "blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_mmad_matmul_basic_split_k.h"
#include "blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
#include "blaze/gemm/block/block_mmad_matmul_emu_split_weight.h"
#include "blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h"
#include "blaze/gemm/block/block_mmad_matmul_iterbatch.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk_split_k.h"
#include "blaze/gemm/block/block_mmad_matmul_syrk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/block/block_mmad_wqmm_mix_prologue_fixpipe_quant.h"
#include "blaze/gemm/block/block_mmad_wqmm_mix_weight_prologue.h"
