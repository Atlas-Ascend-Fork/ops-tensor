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
 * \brief Aggregate header for GEMM kernels. Pulls in the GemmUniversal primary template declaration
 * (kernel_universal_decl.h) and then every kernel_xx.h specialization fragment in this directory.
 * Users of these kernels include this header only; the fragments themselves include
 * kernel_universal_decl.h so they stay self-contained without including this header back.
 * The only kernel headers not aggregated here are kernel_wqgmm_mix_weight_prologue.h, which defines
 * the standalone GmmWeightQuantMxKernel class rather than a GemmUniversal specialization, and
 * kernel_gmm_fixpipe_quant.h, which is a deprecated rename shim for kernel_qgmm_mix_fixpipe_quant.h;
 * users of those two include them directly.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_universal_decl.h"

#include "blaze/gemm/kernel/kernel_batch_matmul_broadcast.h"
#include "blaze/gemm/kernel/kernel_batch_matmul_iterbatch_broadcast.h"
#include "blaze/gemm/kernel/kernel_confidence_head_fused_mix.h"
#include "blaze/gemm/kernel/kernel_grouped_matmul.h"
#include "blaze/gemm/kernel/kernel_matmul_al1_full_load.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_bl1_full_load.h"
#include "blaze/gemm/kernel/kernel_matmul_emu_split_weight.h"
#include "blaze/gemm/kernel/kernel_matmul_fixpipe_opti.h"
#include "blaze/gemm/kernel/kernel_matmul_iterbatch.h"
#include "blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h"
#include "blaze/gemm/kernel/kernel_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_matmul_syrk.h"
#include "blaze/gemm/kernel/kernel_matmul_with_epilogue.h"
#include "blaze/gemm/kernel/kernel_matmul_with_scale_add.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/gemm/kernel/kernel_qgmm_cube.h"
#include "blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_mix_finalize_routing.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/gemm/kernel/kernel_tbmm_basic.h"
#include "blaze/gemm/kernel/kernel_tqbmm.h"
#include "blaze/gemm/kernel/kernel_wqmm_mix_antiquant.h"
#include "blaze/gemm/kernel/kernel_wqmm_mix_pergroup.h"
