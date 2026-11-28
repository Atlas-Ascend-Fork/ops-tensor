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
# 本仓库为 header-only 库（include/blaze/，无 src/ 算子目录），
# 构建入口保留以下三种模式：
#   1. Kernel UT：构建（可选执行）tests/ut/op_kernel/ 下的算子内核单元测试
#   2. Examples ：构建并运行 examples/acl/ 下的端到端样例
#   3. Package ：配置根工程并将 include/ 打包为 .run 自解压安装包（CI 编译入口）
#
# 使用方法:
#   ./build.sh [OPTIONS]
#
# 模式（必选其一）:
#   --opkernel [-u]     构建 Kernel UT；-u 表示构建后执行测试
#   --examples          构建并运行 Examples（内置 shape，自动初始化 tensor_api submodule）
#   --pkg               配置根工程并打包 header-only 库为 .run 安装包（不支持 --ops）
#
# 通用选项:
#   --ops=OP_LIST       指定算子列表 (逗号分隔)
#                       Kernel UT 模式: tests/ut/op_kernel/ 下子目录名 (如 mat_mul)
#                       Examples 模式:  examples/acl/ 下子目录名 (如 mat_mul)
#   --target=TARGET     指定 Example 名称，逗号分隔 (仅 --examples 模式)
#   --soc=SOC           指定目标 SoC 型号（支持小写与逗号分隔多值，默认 Ascend950）
#   -j[N]               编译线程数，默认为 8，例如: -j16
#   --test-timeout=N    Kernel UT 测试超时时间（秒），默认为 300
#   --cov               开启代码覆盖率统计（仅 --opkernel 模式）
#   --force-submodule   强制更新 tensor_api submodule 至 superproject 记录的 pinned commit
#                       （默认已初始化时跳过）
#   --cann_3rd_lib_path=PATH  指定三方库路径（默认 third_party/）
#   -v, --verbose       详细输出
#   --make_clean        清理构建目录
#   -h, --help          显示帮助信息
#
# 示例:
#   ./build.sh --opkernel -u                             # 构建并执行所有 Kernel UT
#   ./build.sh --opkernel -u --ops=mat_mul               # 构建并执行 mat_mul Kernel UT
#   ./build.sh --opkernel -u --ops=mat_mul,quant_batch_matmul
#   ./build.sh --opkernel                                # 仅构建所有 Kernel UT（不执行）
#   ./build.sh --examples                                # 构建并运行所有样例
#   ./build.sh --examples --ops=mat_mul                  # mat_mul 的所有样例
#   ./build.sh --examples --ops=mat_mul --target=mat_mul_streamk  # 单个样例
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
SOC_NAME="Ascend950"
SOC_EXPLICIT=false
BUILD_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREAD_NUM=8  # 默认编译线程数
THREAD_EXPLICIT=false  # 用户是否显式指定 -j（examples 模式仅显式时透传，否则 run.sh 自适应）
CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
TEST_TIMEOUT=300  # 默认测试超时时间（秒）
VERBOSE=""
FORCE_SUBMODULE=false  # 强制更新 tensor_api submodule 至 pinned commit
CANN_3RD_LIB_PATH="${SCRIPT_DIR}/third_party"
OP_KERNEL_MODE=false
EXAMPLES_MODE=false
PKG_MODE=false
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
Usage: $(basename "$0") [OPTIONS] (--opkernel [-u] | --examples | --pkg)

Build modes (exactly one required):
  --opkernel           Build kernel unit tests (tests/ut/op_kernel/)
  -u                   With --opkernel: also run the tests after build
  --examples           Build and run examples (examples/acl/, built-in shapes)
                       Both modes automatically initialize the tensor_api
                       submodule before build (pinned commit)
  --pkg                Configure the root project and package the header-only
                       library (include/) into a self-extracting .run
                       installer (default SoC: Ascend950; --ops not supported)

Common options:
  --ops=OP_LIST        Operators (comma-separated)
                       Kernel UT: subdirectories under tests/ut/op_kernel/
                       Examples:  subdirectories under examples/acl/
  --target=TARGET      Example name(s), comma-separated (--examples only)
  --soc=SOC            Target SoC model (default: Ascend950, case-insensitive)
  -j[N]                Number of compile threads (default: 8), e.g., -j16
  --test-timeout=N     Kernel UT test timeout in seconds (default: 300)
  --cov                Enable code coverage for kernel UT (--opkernel only)
  --force-submodule    Force update tensor_api submodule to the pinned commit
                       recorded in the superproject (instead of skipping when
                       already initialized)
  --cann_3rd_lib_path=PATH
                       Third-party lib path (default: third_party/)
  -v, --verbose        Verbose output
  --make_clean         Clean build artifacts
  -h, --help           Show this help message

Kernel UT examples:
  $(basename "$0") --opkernel -u                            # Build and run all kernel UTs
  $(basename "$0") --opkernel -u --ops=mat_mul              # Build and run mat_mul kernel UT
  $(basename "$0") --opkernel -u --ops=mat_mul,quant_batch_matmul
  $(basename "$0") --opkernel -u --soc=ascend950 --ops=mat_mul
  $(basename "$0") --opkernel                               # Build only (no run)
  $(basename "$0") --opkernel -u --test-timeout=600         # Run with 600s timeout

Examples mode:
  $(basename "$0") --examples                                    # All examples
  $(basename "$0") --examples --ops=mat_mul                      # mat_mul all examples
  $(basename "$0") --examples --ops=mat_mul --target=mat_mul_streamk  # Single example

Package mode:
  $(basename "$0") --pkg                                         # Package for Ascend950 (default)
  $(basename "$0") --soc=ascend950 --pkg                        # Package for Ascend950 (explicit)

Note:
  This repository is a header-only library (include/blaze/, no src/ directory).
  Kernel UT is the single entry for kernel-level verification; Examples is the
  entry for end-to-end feature verification; --pkg only configures the root
  project and packages the headers (CI compile entry). The legacy normal-build
  and --run entries have been removed.
EOF
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
    if [ -d "${SCRIPT_DIR}/build_out" ]; then
        rm -rf "${SCRIPT_DIR}/build_out"
        log_success "build_out directory cleaned"
    else
        log_info "build_out directory does not exist, nothing to clean"
    fi
}

# 配置根工程（header-only，无编译目标；作用是走一遍根工程 CMake 配置，
# 校验工程配置链路可用并与 CI 既有 --pkg 打包链路保持一致）
build_project() {
    log_info "Starting build..."

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    log_info "Configuring CMake..."
    local cmake_args=()

    if [ "$PKG_MODE" = true ]; then
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

# 打包 header-only 库：include/ → 自解压 .run 安装包
build_package() {
    log_info "Building package..."

    local PKG_NAME="cann-${SOC_NAME_LOWER}-ops-tensor_1.0.0_linux-${ARCH}.run"
    local BUILD_OUT_DIR="${SCRIPT_DIR}/build_out"

    mkdir -p "${BUILD_OUT_DIR}"

    local TMP_DIR=$(mktemp -d)
    log_info "Working directory: ${TMP_DIR}"

    mkdir -p "${TMP_DIR}/ops_tensor"

    cp -r "${SCRIPT_DIR}/include" "${TMP_DIR}/ops_tensor/"

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

    cd "${TMP_DIR}"
    tar -czf archive.tar.gz ops_tensor

    cat "${TMP_DIR}/install.sh" "${TMP_DIR}/archive.tar.gz" > "${BUILD_OUT_DIR}/${PKG_NAME}"
    chmod +x "${BUILD_OUT_DIR}/${PKG_NAME}"

    rm -rf "${TMP_DIR}"
    cd "$SCRIPT_DIR"

    if [ -f "${BUILD_OUT_DIR}/${PKG_NAME}" ]; then
        log_success "Package created: ${BUILD_OUT_DIR}/${PKG_NAME}"
    else
        log_error "Package creation failed"
        exit 1
    fi
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

        if [ "$test_result" -eq 124 ]; then
            log_error "${exe} timeout (${TEST_TIMEOUT}s exceeded)"
            failed=$((failed + 1))
        elif [ "$test_result" -eq 137 ]; then
            # timeout -k and an external SIGKILL share the same exit status.
            log_error "${exe} killed (exit code: 137; timeout escalation or external SIGKILL)"
            failed=$((failed + 1))
        elif [ "$test_result" -ne 0 ]; then
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

    # Kernel UT 目标 SoC：未指定 --soc 时仅构建/执行 Ascend950（保持既有默认行为）
    local _ut_soc_list="ascend950"
    if [ "$SOC_EXPLICIT" = true ]; then
        _ut_soc_list=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
    fi
    cmake_args+=(-DKERNEL_UT_SOC="${_ut_soc_list}")

    if [ "$BUILD_OPERATORS" != "all" ]; then
        cmake_args+=(-DOP_KERNEL_OPS="${BUILD_OPERATORS}")
        log_info "Kernel UT operators: ${BUILD_OPERATORS}"
    else
        log_info "Kernel UT operators: all"
    fi

    if [ "$SOC_EXPLICIT" = true ]; then
        log_info "Kernel UT SoC: ${_ut_soc_list} (from --soc=${SOC_NAME})"
    else
        log_info "Kernel UT SoC: ${_ut_soc_list} (default)"
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
    # 芯片版本透传（默认 ascend950，由 --soc 指定）
    local _ex_soc
    _ex_soc=$(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')
    run_args+=("--soc=${_ex_soc}")
    log_info "Examples chip version: ${_ex_soc}"

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
            --pkg)
                PKG_MODE=true
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

    # 检查ASC编译器（Kernel UT 不需要 bisheng，Examples 需要）
    if [ "$OP_KERNEL_MODE" = true ]; then
        log_info "Skipping bisheng compiler check (Kernel UT mode)"
    else
        if ! command -v bisheng &> /dev/null; then
            log_error "bisheng compiler (bisheng) is not installed"
            missing_deps=$((missing_deps + 1))
        else
            local asc_version=$(bisheng --version | head -n1)
            log_success "bisheng compiler is installed ($asc_version)"
        fi
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

    # 模式校验：必须显式选择 --opkernel、--examples 或 --pkg
    if [ "$OP_KERNEL_MODE" != true ] && [ "$EXAMPLES_MODE" != true ] && [ "$PKG_MODE" != true ]; then
        log_error "No build mode specified"
        log_info "Use '--opkernel [-u]' for Kernel UT, '--examples' for examples, or '--pkg' for packaging"
        log_info "Run '$(basename "$0") --help' for details"
        exit 1
    fi

    # --opkernel / --examples / --pkg 互斥
    if [ "$OP_KERNEL_MODE" = true ] && [ "$EXAMPLES_MODE" = true ]; then
        log_error "--opkernel cannot be used with --examples"
        exit 1
    fi
    if [ "$OP_KERNEL_MODE" = true ] && [ "$PKG_MODE" = true ]; then
        log_error "--opkernel cannot be used with --pkg"
        exit 1
    fi
    if [ "$EXAMPLES_MODE" = true ] && [ "$PKG_MODE" = true ]; then
        log_error "--examples cannot be used with --pkg"
        exit 1
    fi

    # --ops 仅对 --opkernel / --examples 有效（--pkg 为全库打包，不支持按算子过滤）
    if [ "$BUILD_OPERATORS" != "all" ] && [ "$PKG_MODE" = true ]; then
        log_error "--ops cannot be used with --pkg (package always contains the whole include/)"
        exit 1
    fi

    # -u 仅对 --opkernel 有效
    if [ "$RUN_TESTS" = true ] && [ "$OP_KERNEL_MODE" != true ]; then
        log_error "-u only works with --opkernel (Kernel UT mode)"
        exit 1
    fi

    # --target 仅对 --examples 有效
    if [ -n "$EXAMPLE_TARGET" ] && [ "$EXAMPLES_MODE" != true ]; then
        log_error "--target only works with --examples"
        exit 1
    fi

    # --cov 仅对 --opkernel 有效
    if [ "$COV" = true ] && [ "$OP_KERNEL_MODE" != true ]; then
        log_error "--cov only works with --opkernel (Kernel UT mode)"
        exit 1
    fi

    # 依赖检查
    check_dependencies

    # 检查并调整线程数
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
        log_warning "Thread num $THREAD_NUM over core num $CORE_NUMS, adjust to $CORE_NUMS"
        THREAD_NUM=$CORE_NUMS
    fi

    # 验证算子（Kernel UT 模式在 build.sh 校验；Examples 模式由 run.sh 校验）
    if [ "$BUILD_OPERATORS" != "all" ] && [ "$OP_KERNEL_MODE" = true ]; then
        IFS=',' read -ra OPS <<< "$BUILD_OPERATORS"
        validate_kernel_ut_operators "${OPS[*]}"
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
            log_info "Kernel UT SoC: ascend950 (default)"
        fi
        if [ "$RUN_TESTS" = true ]; then
            log_success "Run Kernel UT tests: YES"
            log_info "Test timeout: ${TEST_TIMEOUT}s"
        else
            log_info "Run Kernel UT tests: NO (build only)"
        fi
        if [ "$COV" = true ]; then
            log_info "Code coverage: YES"
        fi
    elif [ "$PKG_MODE" = true ]; then
        log_info "Build mode: Package (header-only)"
        log_success "Build package: YES"
        log_info "Target SoC: ${SOC_NAME}"
    else
        log_info "Build mode: Examples"
        if [ "$BUILD_OPERATORS" = "all" ]; then
            log_info "Examples operators: all"
        else
            log_info "Examples operators: ${BUILD_OPERATORS}"
        fi
        if [ -n "$EXAMPLE_TARGET" ]; then
            log_info "Examples targets: ${EXAMPLE_TARGET}"
        fi
        log_info "Examples chip version: $(echo "${SOC_NAME}" | tr '[:upper:]' '[:lower:]')"
    fi
    echo "=================================="
    echo ""

    if [ "$EXAMPLES_MODE" = true ]; then
        build_examples
        log_success "All examples operations completed!"
        exit 0
    fi

    if [ "$PKG_MODE" = true ]; then
        build_project
        build_package
        log_success "All operations completed!"
        return
    fi

    build_kernel_ut

    log_success "All operations completed!"
}

# 运行主函数
main "$@"
