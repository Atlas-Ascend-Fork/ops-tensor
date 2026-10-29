# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# torch/mat_mul — Blaze kernel with a PyTorch frontend
#
# Standalone CMake project (torch/torch_npu are optional dependencies, so this
# is deliberately NOT wired into examples/CMakeLists.txt). Produces a Python
# loadable module: ops_tensor_examples.so
#
# Build (CANN env + torch_npu required):
#   source <cann>/set_env.sh
#   cmake -S . -B build && cmake --build build -j
# Run:
#   python3 test_mat_mul.py
##############################################################################

cmake_minimum_required(VERSION 3.16)
project(ops_tensor_torch_examples LANGUAGES CXX)

##############################################################################
# 1. ASC Compiler Discovery (for the Blaze kernel source)
##############################################################################

find_package(ASC REQUIRED)
enable_language(ASC)

if(NOT DEFINED ASCEND_NPU_ARCH)
    set(ASCEND_NPU_ARCH "dav-3510")
endif()
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASCEND_NPU_ARCH}>)

# glibc >= 2.39 float128 workaround for the ccec device pass (see examples/CMakeLists.txt)
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-D__INTEL_LLVM_COMPILER=20250000>
    $<$<COMPILE_LANGUAGE:ASC>:-DSYCL_LANGUAGE_VERSION=202001>
)

##############################################################################
# 2. Path Configuration
##############################################################################

set(OPS_TENSOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(BLAZE_INCLUDE_DIR ${OPS_TENSOR_ROOT}/include)

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH not found. Fix: source <cann>/set_env.sh first.")
endif()

include(${OPS_TENSOR_ROOT}/cmake/variables.cmake)
set(ASCEND_ARCH_DIR "${ARCH}-linux")

##############################################################################
# 3. Torch / torch_npu / Python Discovery (via the active python3)
##############################################################################

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# COMMAND_ERROR_IS_FATAL requires CMake 3.19; stay compatible with 3.16 by
# checking RESULT_VARIABLE explicitly and failing with the command's stderr.
function(python_query OUT_VAR CODE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "${CODE}"
        OUTPUT_VARIABLE OUT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RC ERROR_VARIABLE ERR)
    if(NOT RC EQUAL 0)
        message(FATAL_ERROR "python3 -c \"${CODE}\" failed with exit code ${RC}: ${ERR}")
    endif()
    set(${OUT_VAR} ${OUT} PARENT_SCOPE)
endfunction()

python_query(TORCH_PATH "import torch, os; print(os.path.dirname(torch.__file__))")
python_query(TORCH_NPU_PATH "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))")
python_query(PYTHON_INCLUDE_DIR "import sysconfig; print(sysconfig.get_paths()['include'])")
python_query(GLIBCXX_USE_CXX11_ABI "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

message(STATUS "Torch path: ${TORCH_PATH}")
message(STATUS "Torch NPU path: ${TORCH_NPU_PATH}")
message(STATUS "GLIBCXX_USE_CXX11_ABI: ${GLIBCXX_USE_CXX11_ABI}")

##############################################################################
# 4. Include / Link Directories
##############################################################################

add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${BLAZE_INCLUDE_DIR}/tensor_api/include>
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${BLAZE_INCLUDE_DIR}/tensor_api>
)

include_directories(
    ${BLAZE_INCLUDE_DIR}
    ${BLAZE_INCLUDE_DIR}/blaze
    ${BLAZE_INCLUDE_DIR}/tensor_api/include
    ${BLAZE_INCLUDE_DIR}/tensor_api
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/op_common/
    ${ASCEND_HOME_PATH}/pkg_inc/base/
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    ${ASCEND_HOME_PATH}/asc/impl/c_api
    ${ASCEND_HOME_PATH}/asc/impl/utils
    ${ASCEND_HOME_PATH}/compiler/ascendc/include/highlevel_api
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_NPU_PATH}/include
    ${PYTHON_INCLUDE_DIR}
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)

##############################################################################
# 5. The Extension Module
#
# One MODULE library per repo: each variant subdirectory (mat_mul_basic,
# mat_mul_a_fullload, ...) contributes a pair of sources via target_sources:
#   *_torch_kernel.cpp  : ASC language (device kernel + <<<>>> launchers)
#   *_torch_binding.cpp : CXX language (torch registration, host-only)
# Final link uses the CXX driver; ASC objects are plain ELF with the device
# image embedded, and resolve against the linked CANN runtime libs.
##############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NOTE: the LANGUAGE source property is scoped to the directory where the
# target is created, so the ASC marking must happen here, not in the
# variant subdirectories.
set(KERNEL_SOURCES
    mat_mul_basic/mat_mul_basic_torch_kernel.cpp
    mat_mul_a_fullload/mat_mul_a_fullload_torch_kernel.cpp
    mat_mul_b_fullload/mat_mul_b_fullload_torch_kernel.cpp
    mat_mul_streamk/mat_mul_streamk_torch_kernel.cpp
)
set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE ASC)

add_library(ops_tensor_examples MODULE)

set_target_properties(ops_tensor_examples PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LINKER_LANGUAGE CXX
)

target_compile_definitions(ops_tensor_examples PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
)

target_link_libraries(ops_tensor_examples PRIVATE
    torch torch_cpu c10 torch_npu
    ascendcl ascendc_runtime runtime mmpa c_sec dl
    platform tiling_api
)

add_subdirectory(mat_mul_basic)
add_subdirectory(mat_mul_a_fullload)
add_subdirectory(mat_mul_b_fullload)
add_subdirectory(mat_mul_streamk)
