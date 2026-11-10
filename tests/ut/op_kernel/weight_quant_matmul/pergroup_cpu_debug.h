/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full license.
 */
#pragma once
#include <cstring>
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_struct.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace QBMMPerGroupA8W4UT::CpuDebug {
struct TileTrace {
    uint64_t a, c, scale;
    int64_t m, n, k;
};
struct Trace {
    uint64_t loads = 0, stores = 0, maskLoads = 0, maskWords = 0, tiles = 0;
    TileTrace blocks[64]{};
};
inline thread_local Trace* trace = nullptr;
inline uintptr_t maskAddress = 0;
// Observe the kernel-to-BlockMmad interface and forward to the unchanged block.
// This validates tile-relative GM views even when the PV shim omits MMAD data.
template <class Base>
struct ObservedBlock : Base {
    template <class A, class Scale, class C, class Shape>
    void operator()(const A& a, const Scale& scale, const C& c, const Shape& shape)
    {
        if (trace) {
            auto slot = trace->tiles++;
            if (slot < 64) {
                trace->blocks[slot] = {reinterpret_cast<uintptr_t>(a.data().get()),
                                       reinterpret_cast<uintptr_t>(c.data().get()),
                                       reinterpret_cast<uintptr_t>(scale.data().get()),
                                       asc::te::get<0>(shape),
                                       asc::te::get<1>(shape),
                                       asc::te::get<2>(shape)};
            }
        }
        Base::operator()(a, scale, c, shape);
    }
};

template <class T>
T* Physical(T* source)
{
    uintptr_t offset = reinterpret_cast<uintptr_t>(source);
    if (offset >= AscendC::TOTAL_UB_SIZE)
        return source;
    auto base = offset / AscendC::ONE_BLK_SIZE * AscendC::ONE_BLK_SIZE;
    AscendC::LocalTensor<uint8_t> tensor(AscendC::TPosition::VECCALC, base, AscendC::TOTAL_UB_SIZE - base);
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(tensor.GetPhyAddr()) + offset - base);
}
} // namespace QBMMPerGroupA8W4UT::CpuDebug
namespace AscendC::Reg {
template <class T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, class U>
void PergroupDataCopy(U& dst, T* src)
{
    if (QBMMPerGroupA8W4UT::CpuDebug::trace)
        ++QBMMPerGroupA8W4UT::CpuDebug::trace->loads;
    DataCopy<T, dist>(dst, QBMMPerGroupA8W4UT::CpuDebug::Physical(src));
}
template <class T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, class U>
void PergroupDataCopy(U& dst, T* src, AddrReg address)
{
    if (QBMMPerGroupA8W4UT::CpuDebug::trace)
        ++QBMMPerGroupA8W4UT::CpuDebug::trace->loads;
    DataCopy<T, dist>(dst, QBMMPerGroupA8W4UT::CpuDebug::Physical(src), address);
}
inline void PergroupDataCopy(MaskReg& dst, uint8_t* src)
{
    using namespace QBMMPerGroupA8W4UT::CpuDebug;
    auto* physical = Physical(src);
    // The kernel initializes its mask by a direct device-address store. The
    // fixture maps only that scratch page; transfer its actual bytes to the
    // simulator's UB before forwarding the real register load.
    if (reinterpret_cast<uintptr_t>(src) == maskAddress) {
        constexpr size_t bytes = AscendC::VECTOR_REG_WIDTH / 8;
        std::memcpy(physical, src, bytes);
        if (trace) {
            ++trace->maskLoads;
            for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i)
                trace->maskWords += reinterpret_cast<uint64_t*>(src)[i] == 0x00000000ffffffffULL;
        }
    }
    DataCopy(dst, physical);
}
template <class T = DefaultType, DataCopyMode mode, PostLiteral post, class U>
void PergroupDataCopy(T*& dst, U& src, uint32_t block, uint32_t repeat, MaskReg& mask)
{
    if (QBMMPerGroupA8W4UT::CpuDebug::trace)
        ++QBMMPerGroupA8W4UT::CpuDebug::trace->stores;
    auto* physical = QBMMPerGroupA8W4UT::CpuDebug::Physical(dst);
    auto* start = physical;
    DataCopy<T, mode, post>(physical, src, block, repeat, mask);
    dst += physical - start;
}
template <class T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, class U>
void PergroupDataCopy(T* dst, U& src, AddrReg address, MaskReg& mask)
{
    if (QBMMPerGroupA8W4UT::CpuDebug::trace)
        ++QBMMPerGroupA8W4UT::CpuDebug::trace->stores;
    DataCopy<T, dist>(QBMMPerGroupA8W4UT::CpuDebug::Physical(dst), src, address, mask);
}
template <class T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, class U>
void PergroupDataCopy(T* dst, U& src, MaskReg& mask)
{
    if (QBMMPerGroupA8W4UT::CpuDebug::trace)
        ++QBMMPerGroupA8W4UT::CpuDebug::trace->stores;
    DataCopy<T, dist>(QBMMPerGroupA8W4UT::CpuDebug::Physical(dst), src, mask);
}
} // namespace AscendC::Reg
#define DataCopy PergroupDataCopy
#include "blaze/epilogue/tile/arch35/dequant.h"
#undef DataCopy
