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
 * \file grouped_matmul_finalize_routing_cpu_debug_compat.h
 * \brief CPU-debug type, bias-constant and register-cast compatibility for GMMFR UT.
 */
#pragma once

#if defined(ASCENDC_CPU_DEBUG)
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "grouped_matmul_mx_a8w4/grouped_matmul_mx_a8w4_cpu_debug_stub.h"

namespace AscendC::Reg {

// CPU debug implements enum-based CastImpl overloads. Forward the device CastTrait
// options to those SDK implementations without replacing the conversion itself.
// A class adapter avoids the GCC 13 ICE for reference non-type parameters in function templates.
template <typename T_, typename U_, const CastTrait& Trait_>
class GmmfrCpuCast {
public:
    GmmfrCpuCast(RegTensor<T_>& dst, RegTensor<U_>& src, MaskReg& mask)
    {
        if constexpr (AscendC::IsSameType<T_, bfloat16_t>::value && AscendC::IsSameType<U_, float>::value) {
            CastImpl<T_, U_, Trait_.roundMode, Trait_.layoutMode, Trait_.mrgMode>(dst, src, mask);
        } else if constexpr (AscendC::IsSameType<T_, float>::value && AscendC::IsSameType<U_, int32_t>::value) {
            CastImpl<T_, U_, Trait_.roundMode, Trait_.mrgMode>(dst, src, mask);
        } else if constexpr (Trait_.roundMode == RoundMode::UNKNOWN && Trait_.satMode == SatMode::UNKNOWN) {
            CastImpl<T_, U_, Trait_.layoutMode, Trait_.mrgMode>(dst, src, mask);
        } else {
            CastImpl<T_, U_, Trait_.roundMode, Trait_.satMode, Trait_.layoutMode, Trait_.mrgMode>(dst, src, mask);
        }
    }
};

} // namespace AscendC::Reg

namespace AscendC {

// The Cast macro also reaches AscendC::Cast calls in the aggregate headers.
template <typename... Args>
inline decltype(auto) GmmfrCpuCast(Args&&... args)
{
    return Cast(static_cast<Args&&>(args)...);
}

} // namespace AscendC

// The including UT header undefines these immediately after the component includes.
#define Cast GmmfrCpuCast
#define __fp8e4m3 fp8_e4m3fn_t
#define __fp4e2m1x2 fp4x2_e2m1_t
#define __fp4e1m2x2 fp4x2_e1m2_t
#endif
