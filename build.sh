#!/bin/bash
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
# ops-tensor Build Script
#
# 使用方法:
#   ./build.sh [OPTIONS]
#
# 选项:
#   --ops=OP_LIST       指定要编译的算子列表 (逗号分隔)
#   --run               编译后执行测试
#   --pkg               编译并打包成 .run 文件
#   --soc=SOC           指定目标 SoC 型号 (当前仅支持: Ascend950, 支持小写输入)
#   -j[N]               编译线程数，默认为 8，例如: -j16
#   --test-timeout=N    测试超时时间（秒），默认为 300
#   --force-submodule   强制更新 tensor_api submodule 至 superproject 记录的 pinned commit
#                       （默认已初始化时跳过；配合 --opkernel -u / --examples 使用）
#   -h, --help          显示帮助信息
#
# 行为说明:
#   无参数                编译所有算子，不执行测试
#   --ops=add             只编译 add 算子，不执行测试
#   --ops=add,sub         编译 add 和 sub 算子，不执行测试
#   --run                 编译所有算子，并执行所有算子的测试
#   --ops=add --run       编译 add 算子，并执行 add 算子的测试
#   --ops=add,sub --run   编译 add,sub 算子，并执行这些算子的测试
#   --pkg                 编译所有算子并打包成 .run 文件 (默认 SoC: Ascend950)
#   --ops=add --pkg       编译 add 算子并打包成 .run 文件
#   --soc=ascend950 --pkg  为 Ascend950 芯片打包 (支持小写)
#   -j16                  使用 16 个线程编译
#   --opkernel -u          构建并执行 Kernel UT (必须组合使用)
#                          执行前会自动初始化 tensor_api submodule（使用 pinned commit）
#   --opkernel -u --ops=OP 构建并执行指定算子的 Kernel UT (tests/ut/op_kernel/ 下子目录名)
#
# 示例:
#   ./build.sh                        # 编译所有算子 (默认 8 线程)
#   ./build.sh --ops=add              # 只编译 add 算子
#   ./build.sh -j16                   # 使用 16 线程编译所有算子
#   ./build.sh --run                  # 编译所有算子并执行测试
#   ./build.sh --ops=add --run        # 编译 add 算子并执行测试
#   ./build.sh --pkg                  # 编译所有算子并打包 (默认: Ascend950)
#   ./build.sh --ops=add --pkg        # 编译 add 算子并打包
#   ./build.sh --soc=ascend950 --pkg  # 为 Ascend950 打包 (支持小写输入)
#   ./build.sh --opkernel -u --ops=mat_mul              # 构建并执行 mat_mul Kernel UT
#   ./build.sh --opkernel -u                            # 构建并执行所有 Kernel UT
#
# 支持的 SoC 型号:
#   Ascend950 (默认)
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
BUILD_OPERATORS="all"
RUN_TESTS=false
ENABLE_PACKAGE=false
SOC_NAME="Ascend950"
SOC_EXPLICIT=false
BUILD_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREAD_NUM=8  # 默认编译线程数
THREAD_EXPLICIT=false  # 用户是否显式指定 -j（examples 模式仅显式时透传，否则 run.sh 自适应）
CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
TEST_TIMEOUT=300  # 默认测试超时时间（秒）
BUILD_OUT_DIR=build_out
VERBOSE=""
FORCE_SUBMODULE=false  # 强制更新 tensor_api submodule 至 pinned commit
CANN_3RD_LIB_PATH="${SCRIPT_DIR}/third_party"
OP_KERNEL_MODE=false
EXAMPLES_MODE=false
EXAMPLE_TARGET=""
COV=false  # 代码覆盖率（仅 Kernel UT 模式支持）

# 设置 _ASCEND_INSTALL_PATH（优先级：ASCEND_INSTALL_PATH > ASCEND_HOME_PATH > 默认值）
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi

# SoC 名称标准化函数（首字母大写，其余小写）
normalize_soc_name() {
    local soc="$1"
    # 转换为小写，然后首字母大写
    echo "${soc}" | sed 's/.*/\L&/; s/^./\U&/; s/.$/\U&/'
}

# SoC 名称映射到完整的 SOC_VERSION（CANN ASC 编译器要求的格式）
get_soc_version() {
    local soc_name="$1"
    local soc_lower=$(echo "$soc_name" | tr '[:upper:]' '[:lower:]')

    # 映射表：友好名称 -> CANN 期望的完整版本标识
    case "$soc_lower" in
        "ascend950")
            echo "ascend950dt_9595"
            ;;
        "ascend910b"|"ascend910_b")
            echo "ascend910b3"
            ;;
        "ascend910_93")
            echo "ascend910_93"
            ;;
        "ascend910")
            echo "ascend910"
            ;;
        "ascend310p"|"ascend310_p")
            echo "ascend310p"
            ;;
        "ascend310b"|"ascend310_b")
            echo "ascend310b"
            ;;
        *)
            # 如果没有映射，直接使用小写名称
            echo "$soc_lower"
            ;;
    esac
}


# 检查 ASCEND 环境变量
check_ascend_env() {
    log_info "Checking ASCEND environment..."

    if [ -z "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH environment variable is not set!"
        echo ""
        echo "Please source the CANN environment script:"
        echo "  source /usr/local/Ascend/cann/set_env.sh"
        echo ""
        echo "Then run build.sh again."
        exit 1
    fi

    if [ ! -d "${ASCEND_HOME_PATH}" ]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        exit 1
    fi

    log_success "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
}

# 日志函数（统一接口）
log() {
    local level=$1
    local msg=$2
    local color=""
    local prefix=""

    case $level in
        info)
            color="${BLUE}"
            prefix="[INFO]"
            ;;
        success)
            color="${GREEN}"
            prefix="[SUCCESS]"
            ;;
        warning)
            color="${YELLOW}"
            prefix="[WARNING]"
            ;;
        error)
            color="${RED}"
            prefix="[ERROR]"
            ;;
        verbose)
            if [ "$VERBOSE" != true ]; then
                return
            fi
            color="${BLUE}"
            prefix="[VERBOSE]"
            ;;
        *)
            color="${NC}"
            prefix="[LOG]"
            ;;
    esac

    echo -e "${color}${prefix}${NC} ${msg}"
}

# 兼容性别名
log_info() { log info "$1"; }
log_success() { log success "$1"; }
log_warning() { log warning "$1"; }
log_error() { log error "$1"; }
log_verbose() { log verbose "$1"; }

# 显示帮助信息
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --ops=OP_LIST       Specify operators to build (comma-separated)
  --run               Run tests after build
  --pkg               Build package (.run file)
  --soc=SOC           Target SoC model (default: Ascend950, case-insensitive)
  -j[N]               Number of compile threads (default: 8), e.g., -j16
  --test-timeout=N    Test timeout in seconds (default: 300)
  --force-submodule   Force update tensor_api submodule to the pinned commit
                      recorded in the superproject (instead of skipping when
                      already initialized). Used with --opkernel -u / --examples.
  -h, --help          Show this help message
  -v, --verbose       Verbose output
  --make_clean        Clean build artifacts"

Supported SoC models:
  Ascend950    (dav-3510, default)

Note: Other SoC models (Ascend910B, Ascend910_93, Ascend910, Ascend310P, Ascend310B, etc.)
      are not supported in the current version. Support for more models will be added in future releases.

Examples:
  $(basename "$0")                          # Build all operators (default 8 threads)
  $(basename "$0") --ops=add                # Build only 'add' operator
  $(basename "$0") --ops=add,sub            # Build 'add' and 'sub' operators
  $(basename "$0") -j16                     # Build with 16 threads
  $(basename "$0") --run                    # Build all and run tests
  $(basename "$0") --ops=add --run          # Build 'add' and run tests
  $(basename "$0") --pkg                    # Build package (default SoC: Ascend950)
  $(basename "$0") --ops=add --pkg          # Build 'add' and package
  $(basename "$0") --soc=Ascend950 --pkg    # Build package for Ascend950
  $(basename "$0") --soc=ascend950 --pkg    # Build package (lowercase also works)
  $(basename "$0") --test-timeout=600 --run  # Run tests with 600s timeout
  $(basename "$0") --make_clean             # Clean build artifacts

Kernel UT:
  --opkernel -u          Build and run kernel unit tests (required combination)
                         Automatically initializes tensor_api submodule before build
  --opkernel -u --ops=OP Build and run kernel UT for specified operators
                         Operators are subdirectories under tests/ut/op_kernel/
                         Examples: mat_mul, quant_batch_matmul
  --cov                Enable code coverage for kernel UT (must be used with --opkernel -u)

Kernel UT Examples:
  $(basename "$0") --opkernel -u --ops=mat_mul
  $(basename "$0") --opkernel -u --ops=mat_mul,quant_batch_matmul
  $(basename "$0") --opkernel -u

Examples Mode:
  --examples              Build and run all examples (uses built-in shapes)
                           Automatically initializes tensor_api submodule before build
  --ops=X                 Specify operator for examples (e.g., mat_mul)
  --target=Y              Specify example for examples (e.g., mat_mul_streamk)
  -j[N]                   (optional) Parallel build jobs in examples mode.
                           Default (no -j): auto = min(cores, targets, ~RAM/16GB)

  $(basename "$0") --examples                                    # All examples
  $(basename "$0") --examples --ops=mat_mul                      # mat_mul all examples
  $(basename "$0") --examples --ops=mat_mul --target=mat_mul_streamk  # Single example

EOF
}

# 解析逗号分隔的算子列表
parse_operators() {
    local ops_str=$1
    if [ "$ops_str" = "all" ]; then
        echo "all"
    else
        IFS=',' read -ra OPS <<< "$ops_str"
        printf '%s\n' "${OPS[@]}"
    fi
}

# 获取所有可用的算子
get_available_operators() {
    find "$SCRIPT_DIR/src" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort
}

# 验证指定的算子是否存在
validate_operators() {
    local ops=($1)
    local available_ops=($(get_available_operators))
    local invalid_ops=()

    for op in "${ops[@]}"; do
        local found=false
        for avail_op in "${available_ops[@]}"; do
            if [ "$op" = "$avail_op" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            invalid_ops+=("$op")
        fi
    done

    if [ ${#invalid_ops[@]} -gt 0 ]; then
        log_error "Invalid operators: ${invalid_ops[*]}"
        log_info "Available operators: ${available_ops[*]}"
        exit 1
    fi
}

get_available_kernel_ut_operators() {
    find "$SCRIPT_DIR/tests/ut/op_kernel" -mindepth 1 -maxdepth 1 -type d \
        -exec test -f {}/CMakeLists.txt \; -printf "%f\n" 2>/dev/null | sort
}

validate_kernel_ut_operators() {
    local ops=($1)
    local available_ops=($(get_available_kernel_ut_operators))
    local invalid_ops=()

    for op in "${ops[@]}"; do
        local found=false
        for avail_op in "${available_ops[@]}"; do
            if [ "$op" = "$avail_op" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            invalid_ops+=("$op")
        fi
    done

    if [ ${#invalid_ops[@]} -gt 0 ]; then
        log_error "Invalid kernel UT operators: ${invalid_ops[*]}"
        log_info "Available kernel UT operators: ${available_ops[*]}"
        exit 1
    fi
}

# 清理构建目录
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned"
    else
        log_warning "Build directory does not exist, nothing to clean"
    fi
}

clean_build_out() {
    if [ -d "${BUILD_OUT_DIR}" ]; then
        rm -rf "${BUILD_OUT_DIR}"
        log_success "build_out directory cleaned"
    else
        log_info "build_out directory does not exist, nothing to clean"
    fi
}

# 构建项目
build_project() {
    log_info "Starting build..."

    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 配置CMake
    log_info "Configuring CMake..."
    local cmake_args=()

    if [ "$BUILD_OPERATORS" != "all" ]; then
        cmake_args+=(-DENABLED_OPERATORS="${BUILD_OPERATORS}")
        log_info "Enabled operators: ${BUILD_OPERATORS}"
    else
        log_info "All operators enabled"
    fi

    # 如果需要打包，启用打包
    if [ "$ENABLE_PACKAGE" = true ]; then
        cmake_args+=(-DENABLE_PACKAGE=ON)
        log_info "Package generation: ENABLED"
    fi

    # 传递 SoC 型号和 ASCEND 路径
    # SOC_VERSION: ASC 编译器期望简化格式（如 ascend950，不是 ascend950dt_9595）
    # ASCEND_SOC: init_env.cmake 用于映射 NPU 架构
    SOC_NAME_LOWER=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
    cmake_args+=(-DSOC_VERSION="${SOC_NAME_LOWER}")
    cmake_args+=(-DASCEND_SOC="${SOC_NAME}")
    cmake_args+=(-DASCEND_HOME_PATH="${ASCEND_HOME_PATH}")
    log_info "Target SoC: ${SOC_NAME}"
    log_info "  -> SOC_VERSION: ${SOC_NAME_LOWER} (for ASC compiler)"
    log_info "  -> ASCEND_SOC: ${SOC_NAME} (for NPU arch mapping)"
    log_info "ASCEND path: ${ASCEND_HOME_PATH}"

    cmake_args+=(-DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}")

    cmake "${cmake_args[@]}" ..

    # 编译
    log_info "Compiling with ${THREAD_NUM} threads..."
    cmake --build . -j${THREAD_NUM} ${VERBOSE}

    if [ $? -eq 0 ]; then
        log_success "Build succeeded"
    else
        log_error "Build failed"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# 打包
build_package() {
    log_info "Building package..."

    local PKG_NAME="cann-${SOC_NAME_LOWER}-ops-tensor_1.0.0_linux-${ARCH}.run"
    local BUILD_OUT_DIR="${SCRIPT_DIR}/build_out"

    mkdir -p "${BUILD_OUT_DIR}"

    # 创建临时目录
    local TMP_DIR=$(mktemp -d)
    log_info "Working directory: ${TMP_DIR}"

    # 准备目录结构
    mkdir -p "${TMP_DIR}/ops_tensor"

    # 复制 include 目录
    cp -r "${SCRIPT_DIR}/include" "${TMP_DIR}/ops_tensor/"

    # 创建安装脚本
    cat > "${TMP_DIR}/install.sh" << 'EOF'
#!/bin/bash
set -e

# 安装路径优先级：ASCEND_INSTALL_PATH > 默认路径
if [ -n "${ASCEND_INSTALL_PATH}" ]; then
    INSTALL_BASE="${ASCEND_INSTALL_PATH}"
else
    INSTALL_BASE="/usr/local/Ascend/cann"
fi

TARGET_DIR="${INSTALL_BASE}/ops_tensor"

echo "Installing ops-tensor header-only library..."
echo "Target: ${TARGET_DIR}"

mkdir -p "${TARGET_DIR}"
cp -r ops_tensor/include "${TARGET_DIR}/"

echo "Installation completed."
exit 0

__ARCHIVE_BELOW__
EOF

    # 打包成 tar.gz
    cd "${TMP_DIR}"
    tar -czf archive.tar.gz ops_tensor

    # 合成 .run 文件
    cat "${TMP_DIR}/install.sh" "${TMP_DIR}/archive.tar.gz" > "${BUILD_OUT_DIR}/${PKG_NAME}"
    chmod +x "${BUILD_OUT_DIR}/${PKG_NAME}"

    # 清理
    rm -rf "${TMP_DIR}"
    cd "${SCRIPT_DIR}"

    if [ -f "${BUILD_OUT_DIR}/${PKG_NAME}" ]; then
        log_success "Package created: ${BUILD_OUT_DIR}/${PKG_NAME}"
    else
        log_error "Package creation failed"
        exit 1
    fi
}

# 运行测试
# 仓库唯一的测试套件是 Kernel UT（tests/ut/op_kernel），
# 统一委托 build_kernel_ut 完成构建与执行（RUN_TESTS=true 时自动运行）
run_tests() {
    build_kernel_ut
}

collect_coverage() {
    log_info "Collecting code coverage data..."

    local _build_dir="${SCRIPT_DIR}/build/kernel_ut"
    local _gcov_data="${_build_dir}/gcov_data"
    local _cov_report="${_build_dir}/cov_report"
    local _cov_info="${_cov_report}/coverage.info"
    local _gen_script="${SCRIPT_DIR}/tests/ut/op_kernel/cmake/scripts/generate_cpp_cov.sh"

    if [ ! -f "$_gen_script" ]; then
        log_error "Coverage script not found: $_gen_script"
        exit 1
    fi

    # GCOV_PREFIX 导致 .gcda 写到 gcov_data 目录，而 .gcno 在 build 目录原位置
    # lcov 需要 .gcda 和 .gcno 在同目录，配对复制
    log_info "Pairing .gcno files with .gcda..."
    local _paired=0
    for gcda in $(find "$_gcov_data" -name "*.gcda"); do
        local orig_path="${gcda#$_gcov_data}"
        local gcno_src="${orig_path%.gcda}.gcno"
        local gcno_dst="${gcda%.gcda}.gcno"
        if [ -f "$gcno_src" ]; then
            cp "$gcno_src" "$gcno_dst"
            _paired=$((_paired + 1))
        fi
    done
    log_info "Paired ${_paired} .gcno files"

    bash "$_gen_script" "$_build_dir" "$_cov_info" "$_cov_report" "$ASCEND_HOME_PATH" "$(dirname "$SCRIPT_DIR")"
    if [ $? -ne 0 ]; then
        log_error "Coverage report generation failed"
        exit 1
    fi

    log_success "Coverage report: ${_cov_report}/index.html"

    local _pkg="${SCRIPT_DIR}/build/kernel_ut/coverage.tar.gz"
    tar -czf "$_pkg" -C "$(dirname "$_cov_report")" cov_report
    log_success "Coverage package: ${_pkg}"
}

run_kernel_ut() {
    log_info "Running Kernel UT tests (timeout: ${TEST_TIMEOUT}s)..."

    # 设置 GCOV_PREFIX 让 .gcda 写到独立目录，避免 fork 子进程并发写冲突
    if [ "$COV" = true ]; then
        local _gcov_dir="${SCRIPT_DIR}/build/kernel_ut/gcov_data"
        rm -rf "$_gcov_dir"
        mkdir -p "$_gcov_dir"
        export GCOV_PREFIX="$_gcov_dir"
        log_info "GCOV_PREFIX set to: $_gcov_dir"
    fi

    local executables=($(ls ops_tensor_kernel_ut_* 2>/dev/null | grep -v '\.' || true))
    if [ ${#executables[@]} -eq 0 ]; then
        log_error "No kernel UT executables found"
        exit 1
    fi

    log_info "Found ${#executables[@]} executable(s): ${executables[*]}"

    local failed=0
    for exe in "${executables[@]}"; do
        log_info "Running ${exe}..."
        set +e
        timeout -k 1s ${TEST_TIMEOUT}s ./${exe} 2>&1
        local test_result=$?
        set -e

        if [ $test_result -ge 124 ]; then
            log_error "${exe} timeout (${TEST_TIMEOUT}s exceeded)"
            failed=$((failed + 1))
        elif [ $test_result -ne 0 ]; then
            log_error "${exe} failed (exit code: $test_result)"
            failed=$((failed + 1))
        else
            log_success "${exe} passed"
        fi
    done

    if [ $failed -gt 0 ]; then
        log_error "${failed}/${#executables[@]} SoC test(s) failed"
        exit 1
    else
        log_success "Kernel UT all SoC tests passed"
    fi

    if [ "$COV" = true ]; then
        collect_coverage
    fi
}

# tensor_api submodule 公共工具（Kernel UT / Examples 共用，定义 ensure_tensor_api_submodule）
source "${SCRIPT_DIR}/scripts/submodule_utils.sh"

build_kernel_ut() {
    log_info "Starting Kernel UT build..."

    ensure_tensor_api_submodule "${SCRIPT_DIR}" "${FORCE_SUBMODULE}" || exit 1

    local KERNEL_UT_SRC="${SCRIPT_DIR}/tests/ut/op_kernel"
    local KERNEL_UT_BUILD="${SCRIPT_DIR}/build/kernel_ut"

    mkdir -p "$KERNEL_UT_BUILD"
    cd "$KERNEL_UT_BUILD"

    [ -f "CMakeCache.txt" ] && rm -f CMakeCache.txt

    local cmake_args=()
    cmake_args+=(-DASCEND_HOME_PATH="${ASCEND_HOME_PATH}")
    cmake_args+=(-DOPS_TENSOR_ROOT="${SCRIPT_DIR}")

    if [ "$SOC_EXPLICIT" = true ]; then
        local _soc_lower=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
        cmake_args+=(-DKERNEL_UT_SOC="${_soc_lower}")
    fi

    if [ "$BUILD_OPERATORS" != "all" ]; then
        cmake_args+=(-DOP_KERNEL_OPS="${BUILD_OPERATORS}")
        log_info "Kernel UT operators: ${BUILD_OPERATORS}"
    else
        log_info "Kernel UT operators: all"
    fi

    if [ "$SOC_EXPLICIT" = true ]; then
        log_info "Kernel UT SoC: ${_soc_lower}"
    else
        log_info "Kernel UT SoC: all"
    fi

    cmake_args+=(-DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}")

    if [ "$COV" = true ]; then
        cmake_args+=(-DENABLE_GCOV=true)
        log_info "Code coverage: ENABLED"
    fi

    log_info "Configuring Kernel UT CMake..."
    cmake "${cmake_args[@]}" "$KERNEL_UT_SRC"

    if [ $? -ne 0 ]; then
        log_error "Kernel UT CMake configuration failed"
        exit 1
    fi

    log_info "Compiling Kernel UT with ${THREAD_NUM} threads..."
    cmake --build . -j${THREAD_NUM} ${VERBOSE}

    if [ $? -ne 0 ]; then
        log_error "Kernel UT build failed"
        exit 1
    fi

    log_success "Kernel UT build succeeded"

    if [ "$RUN_TESTS" = true ]; then
        run_kernel_ut
    fi

    cd "$SCRIPT_DIR"
}

# 构建并运行 Examples
build_examples() {
    local examples_dir="${SCRIPT_DIR}/examples"

    if [ ! -d "$examples_dir" ]; then
        log_error "Examples directory not found: $examples_dir"
        exit 1
    fi

    # 初始化 tensor_api submodule（Examples 编译依赖）
    # examples/CMakeLists.txt 的 include 路径指向 ${OPS_TENSOR_ROOT}/include/tensor_api/
    # 若 submodule 未初始化，编译会失败（不会回退到 CANN 环境）
    ensure_tensor_api_submodule "${SCRIPT_DIR}" "${FORCE_SUBMODULE}" || exit 1

    # 1. Validate parameters
    if [ -n "$EXAMPLE_TARGET" ] && [ "$BUILD_OPERATORS" = "all" ]; then
        log_error "--target must be paired with --ops (specify which operator the target belongs to)"
        exit 1
    fi

    # 2. Delegate to common/run.sh (handles discovery, build, run, and summary)
    local common_run="${SCRIPT_DIR}/examples/acl/common/run.sh"
    local run_args=()
    if [ -n "$BUILD_OPERATORS" ] && [ "$BUILD_OPERATORS" != "all" ]; then
        run_args+=("--ops=${BUILD_OPERATORS}")
    fi
    if [ -n "$EXAMPLE_TARGET" ]; then
        run_args+=("--target=${EXAMPLE_TARGET}")
    fi
    # -j 透传：仅显式指定时传给 run.sh（未指定时 run.sh 自适应，避免默认 8 限制大核机器）
    if [ "$THREAD_EXPLICIT" = true ]; then
        run_args+=("-j${THREAD_NUM}")
        log_info "Examples build parallelism: ${THREAD_NUM} job(s)"
    fi
    if [ "$FORCE_SUBMODULE" = true ]; then
        run_args+=("--force-submodule")
    fi
    bash "$common_run" "${run_args[@]}"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --ops=*)
                # 提取等号后的值
                BUILD_OPERATORS="${1#*=}"
                shift
                ;;
            --run)
                RUN_TESTS=true
                shift
                ;;
            --pkg)
                ENABLE_PACKAGE=true
                shift
                ;;
            --soc=*)
                SOC_INPUT="${1#*=}"
                SOC_NAME=$(normalize_soc_name "${SOC_INPUT}")
                SOC_EXPLICIT=true
                shift
                ;;
            -j*)
                # 提取线程数
                if [[ "$1" == "-j" ]]; then
                    # -j N 的形式
                    if [[ -z "$2" ]]; then
                        log_error "Missing thread number after -j"
                        exit 1
                    fi
                    if [[ "$2" == -* ]]; then
                        log_error "Invalid thread number: $2 (did you mean -j$N?)"
                        exit 1
                    fi
                    THREAD_NUM="$2"
                    shift 2
                else
                    # -jN 的形式
                    THREAD_NUM="${1#-j}"
                    shift
                fi
                if [[ ! "$THREAD_NUM" =~ ^[0-9]+$ ]]; then
                    log_error "non-integer argument:$THREAD_NUM"
                    exit 1
                fi
                THREAD_EXPLICIT=true
                ;;
            --test-timeout=*)
                # 提取测试超时时间
                TEST_TIMEOUT="${1#*=}"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --make_clean)
                clean_build
                clean_build_out
                exit 0
                ;;
            -v|--verbose)
                VERBOSE="-v"
                shift
                ;;
            --cann_3rd_lib_path=*)
                CANN_3RD_LIB_PATH="$(realpath ${1#*=})"
                shift
                ;;
            --cov)
                COV=true
                shift
                ;;
            --opkernel)
                OP_KERNEL_MODE=true
                shift
                ;;
            -u)
                RUN_TESTS=true
                shift
                ;;
            --examples)
                EXAMPLES_MODE=true
                shift
                ;;
            --force-submodule)
                FORCE_SUBMODULE=true
                shift
                ;;
            --target=*)
                EXAMPLE_TARGET="${1#*=}"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查依赖项
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=0

    # 检查cmake
    if ! command -v cmake &> /dev/null; then
        log_error "cmake is not installed"
        missing_deps=$((missing_deps + 1))
    else
        local cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
        log_success "cmake is installed (version: $cmake_version)"
    fi

    # 检查make
    if ! command -v make &> /dev/null; then
        log_warning "make is not installed, will use ninja or other generator"
    else
        local make_version=$(make --version | head -n1)
        log_success "make is installed ($make_version)"
    fi

    # 检查编译器 (gcc/g++)
    if ! command -v g++ &> /dev/null; then
        log_error "g++ is not installed"
        missing_deps=$((missing_deps + 1))
    else
        local gcc_version=$(g++ --version | head -n1)
        log_success "g++ is installed ($gcc_version)"
    fi

    # 检查ASC编译器（Kernel UT 不需要 bisheng）
    if [ "$OP_KERNEL_MODE" != true ]; then
        if ! command -v bisheng &> /dev/null; then
            log_error "bisheng compiler (bisheng) is not installed"
            missing_deps=$((missing_deps + 1))
        else
            local asc_version=$(bisheng --version | head -n1)
            log_success "bisheng compiler is installed ($asc_version)"
        fi
    else
        log_info "Skipping bisheng compiler check (Kernel UT mode)"
    fi

    # 检查Python (用于测试脚本)
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        if [ "$EXAMPLES_MODE" = true ]; then
            log_error "Python is not installed, required for examples mode"
            missing_deps=$((missing_deps + 1))
        else
            log_warning "Python is not installed, some test scripts may not run"
        fi
    else
        local python_cmd="python3"
        if ! command -v python3 &> /dev/null; then
            python_cmd="python"
        fi
        local python_version=$($python_cmd --version 2>&1)
        log_success "Python is installed ($python_version)"
    fi

    if [ $missing_deps -gt 0 ]; then
        log_error "Missing $missing_deps required dependencies, please install and retry"
        exit 1
    fi

    # ASCEND 环境检查（强制要求）
    check_ascend_env

    log_success "All dependencies checked"
}

# 主函数
main() {
    echo "=========================================="
    echo "    ops-tensor Build Script"
    echo "=========================================="
    echo ""

    parse_arguments "$@"

    # --opkernel validation
    if [ "$OP_KERNEL_MODE" = true ]; then
        if [ "$RUN_TESTS" != true ]; then
            log_error "--opkernel requires -u (only --opkernel -u is supported)"
            exit 1
        fi
        if [ "$ENABLE_PACKAGE" = true ]; then
            log_error "--opkernel cannot be used with --pkg"
            exit 1
        fi
    fi

    # --examples validation
    if [ "$EXAMPLES_MODE" = true ]; then
        if [ "$RUN_TESTS" = true ]; then
            log_error "--examples cannot be used with --run"
            exit 1
        fi
        if [ "$ENABLE_PACKAGE" = true ]; then
            log_error "--examples cannot be used with --pkg"
            exit 1
        fi
        if [ "$OP_KERNEL_MODE" = true ]; then
            log_error "--examples cannot be used with --opkernel"
            exit 1
        fi
    fi

    # 依赖检查
    check_dependencies

    # 检查并调整线程数
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
        log_warning "Thread num $THREAD_NUM over core num $CORE_NUMS, adjust to $CORE_NUMS"
        THREAD_NUM=$CORE_NUMS
    fi

    # 验证算子
    if [ "$BUILD_OPERATORS" != "all" ]; then
        IFS=',' read -ra OPS <<< "$BUILD_OPERATORS"
        if [ "$OP_KERNEL_MODE" = true ]; then
            validate_kernel_ut_operators "${OPS[*]}"
        elif [ "$EXAMPLES_MODE" != true ]; then
            validate_operators "${OPS[*]}"
        fi
    fi

    # 显示执行计划
    echo ""
    log_info "========== Build Plan =========="
    if [ "$OP_KERNEL_MODE" = true ]; then
        log_info "Build mode: Kernel UT"
        if [ "$BUILD_OPERATORS" = "all" ]; then
            log_info "Kernel UT operators: all"
        else
            log_info "Kernel UT operators: ${BUILD_OPERATORS}"
        fi
        if [ "$SOC_EXPLICIT" = true ]; then
            log_info "Kernel UT SoC: $(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')"
        else
            log_info "Kernel UT SoC: all"
        fi
        if [ "$RUN_TESTS" = true ]; then
            log_success "Run Kernel UT tests: YES"
            log_info "Test timeout: ${TEST_TIMEOUT}s"
        else
            log_info "Run Kernel UT tests: NO (build only)"
        fi
    else
        if [ "$BUILD_OPERATORS" = "all" ]; then
            log_info "Build operators: all"
        else
            log_info "Build operators: ${BUILD_OPERATORS}"
        fi

        if [ "$RUN_TESTS" = true ]; then
            log_success "Run tests: YES"
            log_info "Test timeout: ${TEST_TIMEOUT}s"
            if [ "$BUILD_OPERATORS" != "all" ]; then
                log_info "Test operators: ${BUILD_OPERATORS}"
            else
                log_info "Test operators: all built operators"
            fi
        else
            log_info "Run tests: NO"
        fi

        if [ "$ENABLE_PACKAGE" = true ]; then
            log_success "Build package: YES"
            log_info "Target SoC: ${SOC_NAME}"
        else
            log_info "Build package: NO"
        fi
    fi
    echo "=================================="
    echo ""

    if [ "$EXAMPLES_MODE" = true ]; then
        build_examples
        log_success "All examples operations completed!"
        exit 0
    fi

    if [ "$OP_KERNEL_MODE" = true ]; then
        build_kernel_ut
    else
        build_project

        if [ "$RUN_TESTS" = true ]; then
            run_tests
        fi

        if [ "$ENABLE_PACKAGE" = true ]; then
            build_package
        fi
    fi

    log_success "All operations completed!"
}

# 运行主函数
main "$@"
