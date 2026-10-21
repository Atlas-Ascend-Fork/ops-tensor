# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

if(NOT OPS_TENSOR_ROOT)
    set(OPS_TENSOR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${OPS_TENSOR_ROOT}/third_party CACHE STRING "CANN third party lib path")
endif()

include(${OPS_TENSOR_ROOT}/cmake/third_party/fetch_cann_cmake.cmake)

project(ops_tensor_kernel_ut LANGUAGES CXX)

# 代码覆盖率选项
option(ENABLE_GCOV "Enable code coverage for kernel UT" OFF)

set(ALL_SUPPORTED_SOCS "ascend950")

if(NOT KERNEL_UT_SOC)
    set(KERNEL_UT_SOC_LIST ${ALL_SUPPORTED_SOCS})
else()
    string(REPLACE "," ";" KERNEL_UT_SOC_LIST "${KERNEL_UT_SOC}")
endif()

function(map_compute_unit compute_unit compute_unit_long)
    set(_keys "ascend950")
    set(_values "ascend950pr_9599")
    list(FIND _keys ${compute_unit} _idx)
    if(NOT _idx EQUAL -1)
        list(GET _values ${_idx} _mapped)
        set(${compute_unit_long} ${_mapped} PARENT_SCOPE)
    else()
        set(${compute_unit_long} ${compute_unit} PARENT_SCOPE)
    endif()
endfunction()

set(UT_EXE_NAME "ops_tensor_kernel_ut")

if(NOT ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH not set. Source CANN set_env.sh first.")
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SYSTEM_PREFIX "aarch64-linux")
else()
    set(SYSTEM_PREFIX "x86_64-linux")
endif()
set(UT_KERNEL_INC ${CMAKE_CURRENT_SOURCE_DIR})

add_cann_third_party(gtest)

list(APPEND CMAKE_PREFIX_PATH
    "${ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake"
    "${ASCEND_HOME_PATH}/compiler/ascendc_kernel_cmake"
)

find_package(tikicpulib REQUIRED)

option(KERNEL_UT_TENSOR_API_DEBUG_CHECK "Enable tensor_api CPU-sim strict checks (ASCENDC_DEBUG)" OFF)
if(NOT KERNEL_UT_TENSOR_API_DEBUG_CHECK)
    set(_TIKICPULIB_SOC_TARGETS
        tikicpulib_ascend950pr_9599
        tikicpulib_ascend960DT_96887)
    foreach(_t IN LISTS _TIKICPULIB_SOC_TARGETS)
        if(TARGET ${_t})
            get_target_property(_defs ${_t} INTERFACE_COMPILE_DEFINITIONS)
            if(_defs)
                string(REPLACE "ASCENDC_DEBUG=1;" "" _defs "${_defs}")
                string(REPLACE ";ASCENDC_DEBUG=1" "" _defs "${_defs}")
                string(REPLACE "ASCENDC_DEBUG=1" "" _defs "${_defs}")
                set_target_properties(${_t} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_defs}")
            endif()
        endif()
    endforeach()
    unset(_defs)
    unset(_t)
endif()

# gcov 覆盖率 INTERFACE 库
if(ENABLE_GCOV)
    add_library(kernel_ut_gcov INTERFACE)
    target_compile_options(kernel_ut_gcov INTERFACE
        -fprofile-arcs -ftest-coverage
    )
    target_link_options(kernel_ut_gcov INTERFACE
        -fprofile-arcs -ftest-coverage
    )
    target_link_libraries(kernel_ut_gcov INTERFACE -lgcov)
    message(STATUS "Kernel UT code coverage: ENABLED")
endif()

set_property(GLOBAL PROPERTY KERNEL_UT_REG_NAMES "")

function(AddOpTestCase opName supportedSocVersions extraCompileOptions)
    file(GLOB _test_srcs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
    if(NOT _test_srcs)
        message(WARNING "  [opkernel] ${opName}: no test_*.cpp found, skipping")
        return()
    endif()

    set_property(GLOBAL APPEND PROPERTY KERNEL_UT_REG_NAMES "${opName}")
    set_property(GLOBAL PROPERTY KERNEL_UT_REG_${opName}_SRCS "${_test_srcs}")
    set_property(GLOBAL PROPERTY KERNEL_UT_REG_${opName}_SOCS "${supportedSocVersions}")
    set_property(GLOBAL PROPERTY KERNEL_UT_REG_${opName}_FLAGS "${extraCompileOptions}")
    set_property(GLOBAL PROPERTY KERNEL_UT_REG_${opName}_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

    message(STATUS "  [opkernel] Registered ${opName} (supports: ${supportedSocVersions})")
endfunction()

if(OP_KERNEL_OPS)
    string(REPLACE "," ";" OP_LIST ${OP_KERNEL_OPS})
    foreach(OP_NAME ${OP_LIST})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${OP_NAME}/CMakeLists.txt)
            add_subdirectory(${OP_NAME})
        else()
            message(WARNING "  [opkernel] Operator '${OP_NAME}' not found, skipping")
        endif()
    endforeach()
else()
    file(GLOB OP_SUBDIRS CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(OP_DIR ${OP_SUBDIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${OP_DIR} AND
           EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${OP_DIR}/CMakeLists.txt)
            add_subdirectory(${OP_DIR})
        endif()
    endforeach()
endif()

get_property(REG_NAMES GLOBAL PROPERTY KERNEL_UT_REG_NAMES)
if(NOT REG_NAMES)
    message(FATAL_ERROR "No kernel UT operators registered")
endif()

set(BUILT_SOC_LIST "")

foreach(_soc_short ${KERNEL_UT_SOC_LIST})
    map_compute_unit(${_soc_short} _soc_long)

    set(_matched_ops "")
    foreach(_op ${REG_NAMES})
        get_property(_op_socs GLOBAL PROPERTY KERNEL_UT_REG_${_op}_SOCS)
        list(FIND _op_socs "${_soc_long}" _idx)
        if(NOT _idx EQUAL -1)
            list(APPEND _matched_ops ${_op})
        endif()
    endforeach()

    if(NOT _matched_ops)
        message(STATUS "  [opkernel] No operators matched SoC=${_soc_short}(${_soc_long}), skipping")
        continue()
    endif()

    set(_obj_targets "")
    foreach(_op ${_matched_ops})
        get_property(_op_srcs GLOBAL PROPERTY KERNEL_UT_REG_${_op}_SRCS)
        get_property(_op_flags GLOBAL PROPERTY KERNEL_UT_REG_${_op}_FLAGS)
        get_property(_op_dir GLOBAL PROPERTY KERNEL_UT_REG_${_op}_DIR)

        set(_tgt "opkernel_${_op}_${_soc_short}")
        add_library(${_tgt} OBJECT ${_op_srcs})

        target_compile_options(${_tgt} PRIVATE
            -g
            -DUT_SOC_VERSION="${_soc_long}"
            -DKERNELUT=1
            -DBLAZE_UT=1
            -DASCENDC_CPU_DEBUG=1
            -DUT_KERNEL_SRC_DIR="${UT_KERNEL_INC}"
            -D_GLIBCXX_USE_CXX11_ABI=0
            -fPIC
            -std=c++17
            ${_op_flags}
        )

        target_include_directories(${_tgt} PRIVATE
            ${OPS_TENSOR_ROOT}/include
            ${OPS_TENSOR_ROOT}/include/tensor_api/include
            ${OPS_TENSOR_ROOT}/include/tensor_api
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/c_api
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/c_api
            ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/utils
            ${ASCEND_HOME_PATH}/include/base
            ${ASCEND_HOME_PATH}/include/register
            ${ASCEND_HOME_PATH}/include
            ${UT_KERNEL_INC}
            ${_op_dir}
        )

        # GTest::gtest 自带 include 目录与 third_party_gtest 构建依赖
        target_link_libraries(${_tgt} PRIVATE
            tikicpulib::${_soc_long}
            GTest::gtest
        )
        if(ENABLE_GCOV)
            target_link_libraries(${_tgt} PRIVATE kernel_ut_gcov)
        endif()

        list(APPEND _obj_targets "$<TARGET_OBJECTS:${_tgt}>")
    endforeach()

    set(_exe "${UT_EXE_NAME}_${_soc_short}")
    add_executable(${_exe}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_op_kernel_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/k3_pvwrap.cpp
        ${_obj_targets}
    )

    target_compile_options(${_exe} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

    target_link_libraries(${_exe} PRIVATE
        GTest::gtest
        GTest::gtest_main
        tikicpulib::${_soc_long}
        pthread
    )
    if(ENABLE_GCOV)
        target_link_libraries(${_exe} PRIVATE kernel_ut_gcov)
    endif()

    target_include_directories(${_exe} PRIVATE
        ${OPS_TENSOR_ROOT}/include
        ${OPS_TENSOR_ROOT}/include/tensor_api/include
        ${OPS_TENSOR_ROOT}/include/tensor_api
        ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
        ${ASCEND_HOME_PATH}/include
        ${UT_KERNEL_INC}
    )

    target_link_directories(${_exe} PRIVATE
        ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
        ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/simulator/dav_3510/lib
    )

    list(APPEND BUILT_SOC_LIST "${_soc_short}")
    message(STATUS "  [opkernel] Built SoC=${_soc_short}(${_soc_long}): ${_matched_ops}")
endforeach()

if(NOT BUILT_SOC_LIST)
    message(FATAL_ERROR "No SoC targets produced any executables. KERNEL_UT_SOC=${KERNEL_UT_SOC_LIST}")
endif()

message(STATUS "ops_tensor Kernel UT configured: SoCs=[${BUILT_SOC_LIST}]")

# ============================================================================
# 代码覆盖率收集 target
# ============================================================================
if(ENABLE_GCOV)
    set(_cov_report ${CMAKE_BINARY_DIR}/cov_report)
    set(_cov_data ${_cov_report}/coverage.info)
    set(_gen_cov_script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/generate_cpp_cov.sh)

    # 收集所有可执行文件 target 名作为依赖
    set(_ut_exe_targets "")
    foreach(_soc ${BUILT_SOC_LIST})
        list(APPEND _ut_exe_targets "${UT_EXE_NAME}_${_soc}")
    endforeach()

    add_custom_target(collect_coverage_data
        COMMAND bash ${_gen_cov_script}
                ${CMAKE_BINARY_DIR} ${_cov_data} ${_cov_report} ${ASCEND_HOME_PATH}
        DEPENDS ${_ut_exe_targets}
        COMMENT "Collect coverage data with lcov and generate HTML report"
    )
    message(STATUS "collect_coverage_data target added (depends: ${_ut_exe_targets})")
endif()
