/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file blaze_kernel_stub.h
 * \brief CANN 9.1.0 兼容性补丁 - 为 Kernel UT 提供缺失的定义
 *
 * ascend950 (arch3510, __NPU_ARCH__=3510) 的 CANN 9.1.0 CPU debug 头文件缺少以下定义:
 * 1. __biasbuf__ 地址空间限定符 (tensor_api 使用但 tikicpulib 未定义)
 * 2. POS_LOWEST / POS_HIGHEST 常量 (C API 使用, 定义在 bisheng compiler 中,
 *    但 x86_64 CPU debug 模式不引入 bisheng 头文件)
 *    NOTE: MODE_ZEROING / MODE_MERGING / MODE_UNKNOWN 已存在于 stub_fun.h 的 Literal enum,
 *    不需重复定义, 否则会与 Literal enum 成员冲突 ("redeclared as different kind of entity")
 * 3. CPU debug头文件缺失的predicate pattern常量
 * 4. 融合操作类型常量
 * 5. BLOCK_SIZE 常量 (blaze epilogue 使用但未定义, 值为32字节对齐粒度)
 *
 * POS_LOWEST/POS_HIGHEST 定义为 constexpr int32_t (值为 bisheng Pos::LOWEST=0, Pos::HIGHEST=1),
 *    可匹配 stub_fun.h 中 vdup 等 5 参数 overload 的 int32_t type 参数;
 * MODE_* 由 stub_fun.h Literal enum 提供 (MODE_ZEROING=0, MODE_MERGING=63, MODE_UNKNOWN=65),
 * 可匹配 Literal 或 int32_t mode 参数的 overload.
 *
 */

#pragma once

#include <cstdint>
#include "stub_fun.h"

#ifndef __biasbuf__
#define __biasbuf__
#endif

using float8_e4m3_t = fp8_e4m3fn_t;
using float8_e5m2_t = fp8_e5m2_t;
using float4_e1m2x2_t = fp4x2_e1m2_t;
using float4_e2m1x2_t = fp4x2_e2m1_t;
using float8_e8m0_t = fp8_e8m0_t;

// copy_gm_to_cbuf_v2 9参数重载 (带 cache_mode 参数)
// cann stub_fun.h 只提供了 8 参数版本，tensor_api 的 asc_copy_gm2l1_impl 需要此重载
#ifndef COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
#define COPY_GM_TO_CBUF_V2_WITH_CACHE_MODE
inline void copy_gm_to_cbuf_v2(void* dst, void* src, uint8_t sid, uint32_t n_burst, uint32_t len_burst,
                               uint8_t pad_func_mode, uint8_t cache_mode, uint64_t src_stride, uint32_t dst_stride)
{}
#endif

#ifndef ASCENDC_HIF4_H
// hifloat4x2_t 占位类型（真实类型见 960 工具链 bisheng __clang_cce_types.h / tikicpulib kernel_hif4.h）
struct hifloat4x2_t {};
#endif

// AscendC::hif4_scale 仅在 Ascend960 (__NPU_ARCH__ 9201/9202) 的 HIF4 路径被引用
// （见 common_utils.h 的 MxScaleType）；CANN 9.2.0+ 的 kernel_utils_constants.h 已提供
// 真实定义，无条件占位会与其冲突，故仅 960 HIF4 路径需要占位
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
namespace AscendC {

// AscendC::hif4_scale 占位类型（真实定义见 960 tensor_api），HIF4 关闭时不会被选用
struct hif4_scale {};

} // namespace AscendC
#endif

// copy_ubuf_to_gm_align_v2 (8 参数，dst 为 GM 物理地址的重载)
// tikicpulib 现有重载与该调用形式不匹配，block_epilogue_flat_quant.h 的 UB→GM 回写需要此桩
#ifndef COPY_UBUF_TO_GM_ALIGN_V2_STUB
#define COPY_UBUF_TO_GM_ALIGN_V2_STUB
inline void copy_ubuf_to_gm_align_v2(const void* dst, const void* src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst,
                                     uint8_t cacheMode, uint64_t dstStride, uint32_t srcStride)
{}
#endif
