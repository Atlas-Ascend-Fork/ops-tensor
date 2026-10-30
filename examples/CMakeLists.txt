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
# 2. Chip Version Configuration
#
# 版本表：<SoC 短名>|<--npu-arch 取值>
#   - 样例注册中的 VER、run.sh / build.sh 的 --soc 均使用短名
#   - 新增芯片版本时只需在此追加一行
#   - 未指定 ASCEND_SOC 时使用 EXAMPLES_SOC_DEFAULT
##############################################################################

set(EXAMPLES_SOC_TABLE
    "ascend950|dav-3510"
    "ascend960|dav-9201"
    # "ascendXXX|dav-ZZZZ"
)
set(EXAMPLES_SOC_DEFAULT "ascend950")

if(NOT DEFINED ASCEND_SOC OR "${ASCEND_SOC}" STREQUAL "")
    set(ASCEND_SOC "${EXAMPLES_SOC_DEFAULT}")
endif()

set(ASCEND_NPU_ARCH "")
foreach(_soc_row IN LISTS EXAMPLES_SOC_TABLE)
    string(REPLACE "|" ";" _soc_cols "${_soc_row}")
    list(LENGTH _soc_cols _soc_cols_len)
    if(NOT _soc_cols_len EQUAL 2)
        message(FATAL_ERROR "EXAMPLES_SOC_TABLE 行格式错误（应为 2 列）：${_soc_row}")
    endif()
    list(GET _soc_cols 0 _soc_name)
    list(GET _soc_cols 1 _soc_arch)
    if("${_soc_name}" STREQUAL "${ASCEND_SOC}")
        set(ASCEND_NPU_ARCH "${_soc_arch}")
    endif()
endforeach()
unset(_soc_row)
unset(_soc_cols)
unset(_soc_cols_len)
unset(_soc_name)
unset(_soc_arch)

if(NOT ASCEND_NPU_ARCH)
    message(FATAL_ERROR
        "未知芯片版本 ASCEND_SOC=${ASCEND_SOC}，请在 examples/CMakeLists.txt 的 EXAMPLES_SOC_TABLE 中登记")
endif()

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>)

# bisheng（clang 15）把 __bf16 建模为整数型 builtin，导致 `bfloat16_t x{0.0f}` 这类
# 常量初始化被误判为收窄转换（-Wc++11-narrowing，默认即 error），而 0.0f -> __bf16
# 是精确可表示的，不存在精度损失。此处仅对 ASC 设备编译关闭该诊断。
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-narrowing>)
message(STATUS "Examples chip version: ${ASCEND_SOC} (npu-arch=${ASCEND_NPU_ARCH})")

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
#   ops_example_add_executable(<target> <sources...>)
#   ops_example_add_executable(<target> VER <version> <sources...>)
#
#   <version> is the SoC short name from EXAMPLES_SOC_TABLE (e.g. ascend950).
#   Multiple versions: comma-separated inside quotes
#     ops_example_add_executable(foo VER "ascend950,ascendXXX" foo/foo.cpp)
#
#   Without VER an example is version-agnostic and is built for any target.
#   With VER an example is built only when ASCEND_SOC is in <version>.
#
#   The configure-time scan is recorded in
#     ${CMAKE_BINARY_DIR}/examples_manifest.txt
#   as "target|version|status|source_dir" lines, so external tooling can query
#   which examples exist for the current version without parsing CMake files.
##############################################################################

set(OPS_EXAMPLE_MANIFEST "${CMAKE_BINARY_DIR}/examples_manifest.txt")
file(WRITE "${OPS_EXAMPLE_MANIFEST}" "# target|soc|status|source_dir\n")

macro(ops_example_add_executable NAME)
    set(_ex_srcs ${ARGN})
    set(_ex_vers "")

    # Optional leading "VER <version>" pair
    list(LENGTH _ex_srcs _ex_len)
    if(_ex_len GREATER 1)
        list(GET _ex_srcs 0 _ex_kw)
        if("${_ex_kw}" STREQUAL "VER")
            list(GET _ex_srcs 1 _ex_ver)
            string(REPLACE "," ";" _ex_vers "${_ex_ver}")
            list(REMOVE_AT _ex_srcs 0 1)
        endif()
    endif()

    string(REPLACE ";" "," _ex_ver_str "${_ex_vers}")
    if(NOT _ex_ver_str)
        set(_ex_ver_str "-")
    endif()

    set(_ex_enabled TRUE)
    if(_ex_vers)
        list(FIND _ex_vers "${ASCEND_SOC}" _ex_hit)
        if(_ex_hit EQUAL -1)
            set(_ex_enabled FALSE)
        endif()
    endif()

    if(_ex_enabled)
        set_source_files_properties(${_ex_srcs} PROPERTIES LANGUAGE ASC)
        add_executable(${NAME} ${_ex_srcs})
        target_link_libraries(${NAME} PRIVATE platform tiling_api)
        install(TARGETS ${NAME} DESTINATION bin)
        file(APPEND "${OPS_EXAMPLE_MANIFEST}"
             "${NAME}|${_ex_ver_str}|enabled|${CMAKE_CURRENT_SOURCE_DIR}\n")
    else()
        file(APPEND "${OPS_EXAMPLE_MANIFEST}"
             "${NAME}|${_ex_ver_str}|skipped|${CMAKE_CURRENT_SOURCE_DIR}\n")
    endif()
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
