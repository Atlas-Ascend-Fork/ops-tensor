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
 * \file sync.h
 * \brief Cross-core synchronization primitives for AIC<->AIV handshake.
 */
#pragma once

#include "kernel_basic_intf.h"
#include "c_api/sync/sync.h"

// bisheng --asc-aicore-lang 混合编译（host/device 双边）时，host 侧不定义 __NPU_ARCH__，
// 此时编译器内置 pipe_t 枚举中缺少 PIPE_FIX 成员，而本文件的模板默认实参
// （pipe_t Pipe = PIPE_FIX）在 host 侧同样需要解析，否则混合编译报
// "use of undeclared identifier 'PIPE_FIX'"。因此用 ifndef __NPU_ARCH__ 判定非 device 编译，
// 若 PIPE_FIX 缺失则添加兜底定义；device 侧（__NPU_ARCH__ 已定义）使用真实枚举成员，
// 取值与编译器内置定义一致（PIPE_FIX = 10）。必须位于上述头文件 include 之后，保证 pipe_t 类型已定义。
#ifndef __NPU_ARCH__
#ifndef PIPE_FIX
#define PIPE_FIX (pipe_t)(10)
#endif
#endif

namespace Blaze {
namespace Gemm {
namespace Sync {

// 同步模式与 syncID 定义集中于此。
// 注意：以下数值仅为本文内部模板分发的 tag，并非硬件 ffts mode 字段值
// （硬件： inter=0, subblock=1, block=2； intra 走 set_intra_block 指令，不占用 ffts mode）。
// 块内广播：AIC 通知本 AI Core(block) 内全部 AIV，适用 __mix__(1, N) 任务；
//   arrive 走 ffts(mode=2)，wait 走 wait_flag_dev，须与 arrive 配对同一 flag_id。
constexpr uint16_t SYNC_MODE_BLOCK = 0; // 对应 asc_sync_block_arrive / asc_sync_block_wait
// 跨 block 通知：信号发给本 block 之外的 AI Core/AIV，适用多核分块协同
//   （如 StreamK 分块间依赖）；arrive 走 ffts(mode=0)，wait 走 wait_flag_dev。
constexpr uint16_t SYNC_MODE_INTER = 2; // 对应 asc_sync_inter_arrive / asc_sync_inter_wait
// 块内点对点：AIC 与单个 AIV 之间的握手，直接走 set_intra_block/wait_intra_block 指令，
//   不经过 ffts；Blaze 中 AIC<->AIV 同步的默认模式（fixpipe slot 握手等）。
constexpr uint16_t SYNC_MODE_INTRA = 4; // 对应 asc_sync_intra_arrive / asc_sync_intra_wait

// AIC<->AIV 握手的自定义 flag（通道）ID。数值本身无硬件含义，0~15 任意取值皆可
// （ffts flag 字段 4 bit；intra 的 sync_id 更宽，但沿用同一约定）。仅要求遵守：
//   1) 同一次握手的 arrive/wait 两侧必须传同一 id（下方默认形参已按方向绑定）；
//   2) AIC->AIV 与 AIV->AIC 两个方向必须用不同 id，否则会互相误唤醒；
//   3) 16~31 保留给 3510 CV1/CV2 混布场景的重复信号（isMixCV1V2 时按
//      id + MIX_FLAG_ID_MAX 再发一份）；自定义 id 请用 0~15，并避开
//      FIXPIPE_* 握手已占用的 4~7。
constexpr uint16_t MIX_AIC_SYNC_AIV_FLAG = 0; // 通道 0：AIC -> AIV（NotifyVector / WaitForCube 默认 id）
constexpr uint16_t MIX_AIV_SYNC_AIC_FLAG = 1; // 通道 1：AIV -> AIC（NotifyCube / WaitForVector 默认 id）
constexpr uint16_t MIX_FLAG_ID_MAX = 16;      // id 空间分界：[0,15] 用户段；[16,31] CV1V2 复制段

// Fixpipe L0C->UB producer-consumer handshake. The slot index is added to each base flag.
constexpr uint16_t FIXPIPE_AIV_ACK_FLAG_BASE = 4;   // AIV -> AIC: the UB slot can be reused.
constexpr uint16_t FIXPIPE_AIC_READY_FLAG_BASE = 6; // AIC -> AIV: the UB slot contains valid data.
constexpr uint16_t FIXPIPE_MAX_SLOT_COUNT = 2;

namespace detail {

template <uint16_t Mode, pipe_t Pipe>
__aicore__ inline void Arrive(uint16_t id)
{
    if constexpr (Mode == SYNC_MODE_INTRA) { // 默认 Mode = 4
        asc_sync_intra_arrive(Pipe, id);
    } else if constexpr (Mode == SYNC_MODE_INTER) { // Mode = 2
        asc_sync_inter_arrive(Pipe, id);
    } else {
        asc_sync_block_arrive(Pipe, id);
    }
}

template <uint16_t Mode, pipe_t Pipe>
__aicore__ inline void Wait(uint16_t id)
{
    if constexpr (Mode == SYNC_MODE_INTRA) {
        asc_sync_intra_wait(Pipe, id);
    } else if constexpr (Mode == SYNC_MODE_INTER) {
        asc_sync_inter_wait(Pipe, id);
    } else {
        asc_sync_block_wait(Pipe, id);
    }
}

} // namespace detail

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_FIX>
__aicore__ inline void NotifyVector(uint16_t id = MIX_AIC_SYNC_AIV_FLAG, bool isMixCV1V2 = false)
{
    if ASC_IS_AIC {
        detail::Arrive<Mode, Pipe>(id);
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if (isMixCV1V2) {
            detail::Arrive<Mode, Pipe>(id + MIX_FLAG_ID_MAX);
        }
#endif
    }
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_FIX>
__aicore__ inline void WaitForVector(uint16_t id = MIX_AIV_SYNC_AIC_FLAG, bool isMixCV1V2 = false)
{
    if ASC_IS_AIC {
        detail::Wait<Mode, Pipe>(id);
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if (isMixCV1V2) {
            detail::Wait<Mode, Pipe>(id + MIX_FLAG_ID_MAX);
        }
#endif
    }
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_V>
__aicore__ inline void NotifyCube(uint16_t id = MIX_AIV_SYNC_AIC_FLAG)
{
    if ASC_IS_AIV {
        detail::Arrive<Mode, Pipe>(id);
    }
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_V>
__aicore__ inline void WaitForCube(uint16_t id = MIX_AIC_SYNC_AIV_FLAG)
{
    if ASC_IS_AIV {
        detail::Wait<Mode, Pipe>(id);
    }
}

} // namespace Sync
} // namespace Gemm
} // namespace Blaze
