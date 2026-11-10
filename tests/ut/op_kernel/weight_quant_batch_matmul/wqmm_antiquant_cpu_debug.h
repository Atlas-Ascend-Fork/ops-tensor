/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full license.
 */
#pragma once
#include "wqmm_copy_observer.h"
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "reg_compute/kernel_reg_compute_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/tile/datamove.h"
#include "blaze/gemm/block/block_mmad_wqmm_mix_weight_prologue.h"
#include "blaze/gemm/kernel/kernel_universal_decl.h"

namespace WeightQuantBatchMatmulUT {
struct VectorTrace {
    uint64_t loads = 0, stores = 0;
    AntiquantSyncTrace sync;
};
inline thread_local VectorTrace* vectorTrace = nullptr;
template <typename T>
T* PhysicalUb(T* address)
{
    auto offset = reinterpret_cast<uintptr_t>(address);
    if (offset >= AscendC::TOTAL_UB_SIZE)
        return address;
    auto base = offset / AscendC::ONE_BLK_SIZE * AscendC::ONE_BLK_SIZE;
    AscendC::LocalTensor<uint8_t> tensor(AscendC::TPosition::VECCALC, base, AscendC::TOTAL_UB_SIZE - base);
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(tensor.GetPhyAddr()) + offset - base);
}
} // namespace WeightQuantBatchMatmulUT
namespace AscendC::Reg {
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
void WqmmDataCopy(U& dst, T* src)
{
    if (WeightQuantBatchMatmulUT::vectorTrace)
        ++WeightQuantBatchMatmulUT::vectorTrace->loads;
    DataCopy<T, dist>(dst, WeightQuantBatchMatmulUT::PhysicalUb(src));
}
template <typename T = DefaultType, DataCopyMode mode, PostLiteral post, typename U>
void WqmmDataCopy(T*& dst, U& src, uint32_t blockStride, uint32_t repeatStride, MaskReg& mask)
{
    if (WeightQuantBatchMatmulUT::vectorTrace)
        ++WeightQuantBatchMatmulUT::vectorTrace->stores;
    auto* physical = WeightQuantBatchMatmulUT::PhysicalUb(dst);
    auto* begin = physical;
    DataCopy<T, mode, post>(physical, src, blockStride, repeatStride, mask);
    dst += physical - begin;
}
} // namespace AscendC::Reg
// Only this production header sees the shim. Dependencies have already been
// parsed, so Tensor API / AscendC::DataCopy overloads remain intact.
#define DataCopy WqmmDataCopy
#include "blaze/gemm/kernel/kernel_wqmm_mix_antiquant.h"
#undef DataCopy
