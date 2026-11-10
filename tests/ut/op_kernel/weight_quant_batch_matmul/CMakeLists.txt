# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

AddOpTestCase(weight_quant_batch_matmul "ascend950pr_9599" "")

# Observe the instruction boundary without replacing Tensor API or the SDK
# implementation. The wrapper forwards every call to the real CPU-debug stub.
# The aggregate executable is created later. Scope this option to ascend950;
# use a directory property to retain the framework's CMake 3.16 compatibility.
get_filename_component(_wqmm_ut_parent "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set_property(DIRECTORY "${_wqmm_ut_parent}" APPEND PROPERTY LINK_OPTIONS
    "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,ops_tensor_kernel_ut_ascend950>:LINKER:--wrap=_Z17copy_ubuf_to_cbufPvS_htttt>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,ops_tensor_kernel_ut_ascend950>:LINKER:--wrap=_Z15set_intra_block6pipe_tm>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:NAME>,ops_tensor_kernel_ut_ascend950>:LINKER:--wrap=_Z16wait_intra_block6pipe_tm>")
