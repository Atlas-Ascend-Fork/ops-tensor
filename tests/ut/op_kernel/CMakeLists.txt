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

# ============================================================================
# SoC 扩展表 —— 新增 SoC 时**只需在此追加一行**，其余逻辑全部自动派生
#   每行 4 列，用 "|" 分隔：
#     短名          KERNEL_UT_SOC / build.sh --soc 的取值，也是用例架构目录的归属标签
#     compute_unit  CANN 的架构标识（tikicpulib / SoC 版本对应关系）
#     架构目录      算子目录下仅编入该 SoC 的子目录名（用例级隔离）
#     仿真器目录    $ASCEND_HOME_PATH/<sys>/simulator/<该值>/lib
# ============================================================================
set(KERNEL_UT_SOC_TABLE
    "ascend950|ascend950pr_9599|arch35|dav_3510"
    "ascend960|ascend960DT_968B7|arch92|dav_9201"
)

set(ALL_SUPPORTED_SOCS "")
set(KERNEL_UT_ARCH_DIRS "")
foreach(_soc_row IN LISTS KERNEL_UT_SOC_TABLE)
    string(REPLACE "|" ";" _soc_cols "${_soc_row}")
    list(LENGTH _soc_cols _soc_cols_len)
    if(NOT _soc_cols_len EQUAL 4)
        message(FATAL_ERROR "KERNEL_UT_SOC_TABLE 行格式错误（应为 4 列）：${_soc_row}")
    endif()
    list(GET _soc_cols 0 _soc_short)
    list(GET _soc_cols 1 _soc_long)
    list(GET _soc_cols 2 _soc_arch_dir)
    list(GET _soc_cols 3 _soc_sim_arch)
    list(APPEND ALL_SUPPORTED_SOCS "${_soc_short}")
    list(APPEND KERNEL_UT_ARCH_DIRS "${_soc_arch_dir}")
    set(KERNEL_UT_SOC_${_soc_short}_LONG "${_soc_long}")
    set(KERNEL_UT_SOC_${_soc_short}_ARCH_DIR "${_soc_arch_dir}")
    set(KERNEL_UT_SOC_${_soc_short}_SIM_ARCH "${_soc_sim_arch}")
    set(KERNEL_UT_SOC_LONG_${_soc_long}_SHORT "${_soc_short}")
endforeach()
unset(_soc_row)
unset(_soc_cols)
unset(_soc_cols_len)
unset(_soc_short)
unset(_soc_long)
unset(_soc_arch_dir)
unset(_soc_sim_arch)

if(NOT KERNEL_UT_SOC)
    set(KERNEL_UT_SOC_LIST ${ALL_SUPPORTED_SOCS})
else()
    string(REPLACE "," ";" KERNEL_UT_SOC_LIST "${KERNEL_UT_SOC}")
endif()

# 短名（或 compute_unit 长名）→ compute_unit 长名
function(map_compute_unit compute_unit compute_unit_long)
    if(DEFINED KERNEL_UT_SOC_${compute_unit}_LONG)
        set(${compute_unit_long} "${KERNEL_UT_SOC_${compute_unit}_LONG}" PARENT_SCOPE)
    else()
        set(${compute_unit_long} "${compute_unit}" PARENT_SCOPE)
    endif()
endfunction()

# 抽取裸 SoC 名：短名原样返回，compute_unit 长名反查回短名
function(map_soc_tag soc_name soc_tag)
    if(DEFINED KERNEL_UT_SOC_${soc_name}_LONG)
        set(${soc_tag} "${soc_name}" PARENT_SCOPE)
    elseif(DEFINED KERNEL_UT_SOC_LONG_${soc_name}_SHORT)
        set(${soc_tag} "${KERNEL_UT_SOC_LONG_${soc_name}_SHORT}" PARENT_SCOPE)
    else()
        set(${soc_tag} "${soc_name}" PARENT_SCOPE)
    endif()
endfunction()

# SoC → 用例架构目录（未知 SoC 返回空，此时只编入顶层通用用例）
function(map_soc_to_arch_dir soc_tag arch_dir)
    if(DEFINED KERNEL_UT_SOC_${soc_tag}_ARCH_DIR)
        set(${arch_dir} "${KERNEL_UT_SOC_${soc_tag}_ARCH_DIR}" PARENT_SCOPE)
    else()
        set(${arch_dir} "" PARENT_SCOPE)
    endif()
endfunction()

# SoC → 仿真器架构目录（未知 SoC 返回空）
function(map_soc_to_sim_arch soc_tag sim_arch)
    if(DEFINED KERNEL_UT_SOC_${soc_tag}_SIM_ARCH)
        set(${sim_arch} "${KERNEL_UT_SOC_${soc_tag}_SIM_ARCH}" PARENT_SCOPE)
    else()
        set(${sim_arch} "" PARENT_SCOPE)
    endif()
endfunction()

# CANN 是否为该 SoC 提供 CPU 调试运行时（libcpudebug.so）。
function(soc_has_cpudebug_runtime soc_long has_runtime)
    if(TARGET tikicpulib_${soc_long})
        set(${has_runtime} TRUE PARENT_SCOPE)
    else()
        set(${has_runtime} FALSE PARENT_SCOPE)
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
    # 从 SoC 扩展表派生，无需单独维护
    set(_TIKICPULIB_SOC_TARGETS "")
    foreach(_soc_short ${ALL_SUPPORTED_SOCS})
        list(APPEND _TIKICPULIB_SOC_TARGETS "tikicpulib_${KERNEL_UT_SOC_${_soc_short}_LONG}")
    endforeach()
    unset(_soc_short)
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
    # 用例源一律以 test_*.cpp 命名（顶层与架构目录同规则）：
    #   顶层 test_*.cpp           通用用例
    #   <arch>/test_*.cpp         该架构专属用例
    #   <arch>/*.cpp（非 test_）   该架构专属支撑源，同样只编入对应 SoC
    file(GLOB _test_srcs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
    set(_arch_aux_srcs "")
    foreach(_arch_dir ${KERNEL_UT_ARCH_DIRS})
        file(GLOB _dir_tests CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_arch_dir}/test_*.cpp)
        file(GLOB _dir_all CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_arch_dir}/*.cpp)
        if(_dir_tests)
            list(APPEND _test_srcs ${_dir_tests})
        endif()
        if(_dir_all)
            set(_dir_aux ${_dir_all})
            if(_dir_tests)
                list(REMOVE_ITEM _dir_aux ${_dir_tests})
            endif()
            if(_dir_aux)
                list(APPEND _arch_aux_srcs ${_dir_aux})
                list(LENGTH _dir_aux _n)
                message(STATUS "  [opkernel] ${opName}/${_arch_dir}: ${_n} aux source(s) compiled for this SoC only")
            endif()
        endif()
    endforeach()
    if(_arch_aux_srcs)
        list(APPEND _test_srcs ${_arch_aux_srcs})
    endif()

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
set(SKIPPED_SOC_LIST "")

foreach(_soc_short ${KERNEL_UT_SOC_LIST})
    map_compute_unit(${_soc_short} _soc_long)
    map_soc_tag(${_soc_short} _soc_tag)
    soc_has_cpudebug_runtime(${_soc_long} _has_runtime)

    if(NOT _has_runtime)
        message(STATUS
            "  [opkernel] SoC=${_soc_short}(${_soc_long}): 当前 CANN 未提供该架构的 CPU-debug "
            "运行时(libcpudebug.so)与头文件分支，编译/运行跳过")
        list(APPEND SKIPPED_SOC_LIST "${_soc_short}")
        continue()
    endif()

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

        # 用例级 SoC 隔离：顶层用例进全部 SoC，架构目录用例仅进对应 SoC
        map_soc_to_arch_dir(${_soc_tag} _soc_arch_dir)
        set(_soc_srcs "")
        foreach(_src ${_op_srcs})
            get_filename_component(_src_dir "${_src}" DIRECTORY)
            get_filename_component(_src_dir_name "${_src_dir}" NAME)
            if(NOT _src_dir_name IN_LIST KERNEL_UT_ARCH_DIRS)
                list(APPEND _soc_srcs "${_src}")
            elseif(_src_dir_name STREQUAL _soc_arch_dir)
                list(APPEND _soc_srcs "${_src}")
            endif()
        endforeach()

        if(NOT _soc_srcs)
            message(STATUS "  [opkernel] ${_op}: no case for ${_soc_tag}(${_soc_arch_dir}), skipped on SoC=${_soc_short}")
            continue()
        endif()

        set(_tgt "opkernel_${_op}_${_soc_short}")
        add_library(${_tgt} OBJECT ${_soc_srcs})

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

    # 各 SoC 的运行时/仿真库目录（查 SoC 扩展表）
    map_soc_to_sim_arch(${_soc_tag} _soc_sim_arch)
    if(_soc_sim_arch)
        set(_soc_sim_lib_dir "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/simulator/${_soc_sim_arch}/lib")
    else()
        set(_soc_sim_lib_dir "")
    endif()

    target_link_directories(${_exe} PRIVATE
        ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
        ${_soc_sim_lib_dir}
    )

    list(APPEND BUILT_SOC_LIST "${_soc_short}")
    message(STATUS "  [opkernel] Built SoC=${_soc_short}(${_soc_long}): ${_matched_ops}")
endforeach()

if(NOT BUILT_SOC_LIST)
    message(FATAL_ERROR
        "No SoC produced any executable. KERNEL_UT_SOC=${KERNEL_UT_SOC_LIST}\n"
        "  Skipped (CANN lacks CPU-debug runtime): [${SKIPPED_SOC_LIST}]\n")
endif()

message(STATUS "ops_tensor Kernel UT configured: SoCs=[${BUILT_SOC_LIST}] skipped=[${SKIPPED_SOC_LIST}]")

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
