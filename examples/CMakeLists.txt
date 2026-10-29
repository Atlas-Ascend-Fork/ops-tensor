# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor Examples — Shared CMake Configuration (L1)
#
# This is a standalone CMake project that builds all example executables.
# It uses the Blaze header-only GEMM framework and links only ACL runtime libs.
##############################################################################

cmake_minimum_required(VERSION 3.16)
project(ops_tensor_examples LANGUAGES CXX)

##############################################################################
# 1. ASC Compiler Discovery
##############################################################################

find_package(ASC REQUIRED)
enable_language(ASC)

##############################################################################
# 2. NPU Architecture Configuration (default: dav-3510 for Ascend950)
##############################################################################

if(NOT DEFINED ASCEND_NPU_ARCH)
    set(ASCEND_NPU_ARCH "dav-3510")
endif()
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>)

# glibc >= 2.39 enables __float128/_Float128 declarations in bits/floatn.h based on the
# __clang__ version, but the bisheng/ccec device pass cannot codegen them. Pretend to be
# the Intel SYCL compiler so glibc sets __HAVE_FLOAT128 to 0 (only floatn.h reads these
# two macros; nothing in glibc, CANN, or this project uses them otherwise).
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-D__INTEL_LLVM_COMPILER=20250000>
    $<$<COMPILE_LANGUAGE:ASC>:-DSYCL_LANGUAGE_VERSION=202001>
)

##############################################################################
# 3. Path Configuration
##############################################################################

set(OPS_TENSOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(BLAZE_INCLUDE_DIR ${OPS_TENSOR_ROOT}/include)
set(EXAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/acl/common)

include(${OPS_TENSOR_ROOT}/cmake/variables.cmake)
set(ASCEND_ARCH_DIR "${ARCH}-linux")

message(STATUS "Examples CANN arch dir: ${ASCEND_ARCH_DIR}")

##############################################################################
# 4. Public Include Directories
##############################################################################
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${BLAZE_INCLUDE_DIR}/tensor_api/include>
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${BLAZE_INCLUDE_DIR}/tensor_api>
)

include_directories(
    ${BLAZE_INCLUDE_DIR}           # Blaze header-only framework
    ${BLAZE_INCLUDE_DIR}/blaze     # blaze/gemm/kernel/ etc.
    ${BLAZE_INCLUDE_DIR}/tensor_api/include  # tensor_api/tensor.h etc.
    ${BLAZE_INCLUDE_DIR}/tensor_api  # impl/tensor_api/algorithm/ etc.
    ${EXAMPLES_COMMON_DIR}         # data_utils.h etc.
    ${ASCEND_HOME_PATH}/include    # ACL core headers
    ${ASCEND_HOME_PATH}/include/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc               # compiler internal headers referenced by tensor_api impl (include/utils/base/sys_macros.h etc.),
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include       # tensor_api impl dependency headers
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling  # platform_ascendc.h
    ${ASCEND_HOME_PATH}/asc/impl/c_api  # instr_impl/ header resolution
    ${ASCEND_HOME_PATH}/asc/impl/utils  # impl/utils header resolution
    ${ASCEND_HOME_PATH}/compiler/ascendc/include/highlevel_api  # matmul_intf.h for Tensor API traits
)

##############################################################################
# 5. Public Link Directories and Libraries
##############################################################################

link_directories(${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64)
link_libraries(ascendcl ascendc_runtime runtime mmpa c_sec dl)

##############################################################################
# 6. Core Macro: Register an Example Executable
#
# Usage:
#   ops_example_add_executable(target_name source1.cpp [source2.cpp ...])
#
# This macro:
#   - Marks source files as ASC language (compiled by bisheng)
#   - Creates an executable target
#   - Links platform + tiling_api for GetCoreNumAic()
#   - Installs to bin/ directory
##############################################################################

macro(ops_example_add_executable NAME)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE ASC)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE platform tiling_api)
    install(TARGETS ${NAME} DESTINATION bin)
endmacro()

##############################################################################
# 7. Load Operator Subdirectories
#
# C++ (ACL 直调) 样例统一收在 acl/ 子目录；torch/ 下的 Python 样例为可选
# 依赖（torch/torch_npu），各自独立构建，不在此处加载。
##############################################################################

add_subdirectory(acl/mat_mul)
add_subdirectory(acl/fused_mat_mul)
add_subdirectory(acl/confidence_head_fused)
add_subdirectory(acl/batch_mat_mul)
add_subdirectory(acl/block_attn_res_prepare)
add_subdirectory(acl/transpose_batch_mat_mul)
add_subdirectory(acl/quant_batch_matmul)
add_subdirectory(acl/weight_quant_matmul)
add_subdirectory(acl/weight_quant_batch_matmul_mx)
add_subdirectory(acl/weight_quant_batch_matmul)
add_subdirectory(acl/grouped_matmul)
add_subdirectory(acl/quant_matmul_activation_quant)
add_subdirectory(acl/transpose_quant_batch_matmul)
add_subdirectory(acl/gemm_syrk)
add_subdirectory(acl/flat_quant)
add_subdirectory(acl/rotate_quant)
