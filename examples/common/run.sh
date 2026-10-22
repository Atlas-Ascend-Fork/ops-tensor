#!/bin/bash
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
# ops-tensor Examples — Unified Runner
# See --help for full option list.
#
# Two-phase execution: all selected example targets are built up front
# (in parallel, see build_selected_targets), then examples run through a
# parallel worker pool (see run_selected_examples). Per-example data dirs
# ({example}/input|output) keep concurrent runs independent.
##############################################################################

set -euo pipefail

# ── Key Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${EXAMPLES_DIR}")"
BUILD_DIR="${EXAMPLES_DIR}/build"
PARALLEL_LOG_DIR="${BUILD_DIR}/logs"
BUILD_STATUS_FILE="${PARALLEL_LOG_DIR}/build_status.txt"
RUN_LOG_DIR="${PARALLEL_LOG_DIR}/run"
RUN_STATUS_FILE="${PARALLEL_LOG_DIR}/run_status.txt"
RUN_CASE_PY="${SCRIPT_DIR}/run_case.py"

# ── Color Definitions ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
# stderr：discover_examples 在 $(...) 中调用，stdout 会被吞掉，错误必须走 stderr 才可见
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── CLI State ────────────────────────────────────────────────────────────────
OPS_NAME=""
TARGET=""
CASE_FILE=""
TI_ARG=""
SKIP_BUILD=false
BUILD_ONLY=false
FORCE_SUBMODULE=false
JOBS=""

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: bash examples/common/run.sh [OPTIONS]

Compile and run ops-tensor example examples.

Options:
  --ops=<names>       Operator directory name(s), comma-separated.
                      Multiple: --ops=mat_mul,grouped_matmul (runs all examples under each).
                      Single or omitted: --target may be specified.
  --target=<names>    Example name(s), comma-separated.
                      Multiple: --target=mat_mul_basic,mat_mul_streamk
                      (requires single --ops, --ti not allowed).
  --case=<path>       CSV file with test cases (equals form only).
                      If omitted, auto-discovers {target}.csv in the example dir.
  --ti=<N>            Run only test case at index N (0-based).
  --ti=<N-M>          Run test cases from index N to M (inclusive).
                      Only allowed with single --ops and single --target.
  --skip-build       Skip the CMake build stage.
  --build-only       Only build; do not run or verify.
  --force-submodule  Force update the tensor_api submodule to the pinned commit
                     recorded in the superproject (instead of skipping when
                     already initialized). Applied once before running, also
                     effective with --skip-build / --build-only.
  -j<N>, --jobs=<N>  Parallel jobs for the build phase AND the example run
                     phase (each phase computes its own auto default).
                     Build default: min(nproc, targets, ~RAM_GB/16).
                     Run default: 4 x NPU device count (8 when device count
                     is undetectable, e.g. in containers), capped by the
                     number of examples. Explicit values are capped at nproc.
                     A single example (or -j1) runs serially with console
                     output. Parallel runs write full per-example logs to
                     build/logs/run/ and echo per-case results (PASS/FAIL
                     + accuracy) to the console tagged '[example] ...'.
  -h, --help         Show this help message and exit.

Discovery rules:
  --ops=A --target=X  Run single example: examples/{A}/{X}/
  --ops=A --target=X,Y  Run examples/{A}/{X}/ and examples/{A}/{Y}/
  --ops=A,B           Run all examples under examples/{A}/ and examples/{B}/
  --ops=A             Run all examples under examples/{A}/
  (none)              Run all examples across all operators

  Directories named 'common/', 'scripts/', and 'build/' are always skipped.

Constraints:
  --ops multiple      --target not allowed, --ti not allowed
  --target multiple   --ops must be single, --ti not allowed
  --ti                Requires single --ops and single --target

Examples:
  bash examples/common/run.sh --ops=mat_mul
  bash examples/common/run.sh --ops=mat_mul,grouped_matmul
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic,mat_mul_streamk
  bash examples/common/run.sh --ops=mat_mul --target=mat_mul_basic --ti=0-5
  bash examples/common/run.sh --ops=mat_mul -j8
  bash examples/common/run.sh --skip-build --ops=mat_mul --target=mat_mul_basic
EOF
}

# ── Argument Parsing ─────────────────────────────────────────────────────────
# Strict: only --key=value forms. No positional args, no --case path (space).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ops=*)
            OPS_NAME="${1#*=}"
            shift
            ;;
        --target=*)
            TARGET="${1#*=}"
            shift
            ;;
        --case=*)
            CASE_FILE="${1#*=}"
            shift
            ;;
        --ti=*)
            TI_ARG="${1#*=}"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --force-submodule)
            FORCE_SUBMODULE=true
            shift
            ;;
        -j)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                log_error "Missing job count after -j"
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
        *)
            log_error "Positional arguments are not supported: '$1'"
            echo ""
            usage
            exit 1
            ;;
    esac
done

# ── Job Count Validation ─────────────────────────────────────────────────────
if [[ -n "$JOBS" && ! "$JOBS" =~ ^[0-9]+$ ]]; then
    log_error "Invalid parallel job count: '${JOBS}' (expected a positive integer, e.g. -j8)"
    exit 1
fi
if [[ "$JOBS" == "0" ]]; then
    log_error "Parallel job count must be >= 1 (got -j0)"
    exit 1
fi

# ── Multi-value Validation ──────────────────────────────────────────────────
OPS_MULTI=false
TARGET_MULTI=false

if [[ -n "$OPS_NAME" && "$OPS_NAME" == *","* ]]; then
    OPS_MULTI=true
fi
if [[ -n "$TARGET" && "$TARGET" == *","* ]]; then
    TARGET_MULTI=true
fi

if [[ -n "$CASE_FILE" ]]; then
    if [[ -z "$OPS_NAME" || -z "$TARGET" || "$OPS_MULTI" == true || "$TARGET_MULTI" == true ]]; then
        log_error "--case requires single --ops and single --target"
        exit 1
    fi
fi

if [[ "$OPS_MULTI" == true ]]; then
    if [[ -n "$TARGET" ]]; then
        log_error "--target is not allowed when --ops has multiple values"
        exit 1
    fi
    if [[ -n "$TI_ARG" ]]; then
        log_error "--ti is not allowed when --ops has multiple values"
        exit 1
    fi
fi

if [[ "$TARGET_MULTI" == true ]]; then
    if [[ "$OPS_MULTI" == true ]]; then
        log_error "--ops must be single when --target has multiple values"
        exit 1
    fi
    if [[ -z "$OPS_NAME" ]]; then
        log_error "--ops is required when --target is specified"
        exit 1
    fi
    if [[ -n "$TI_ARG" ]]; then
        log_error "--ti is not allowed when --target has multiple values"
        exit 1
    fi
fi

# ── Pre-flight Checks ───────────────────────────────────────────────────────
preflight() {
    local missing=0

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        log_error "ASCEND_HOME_PATH is not set. Please source CANN set_env.sh first."
        missing=1
    elif [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        missing=1
    else
        log_info "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
    fi

    if ! command -v bisheng &>/dev/null; then
        log_error "bisheng compiler not found in PATH"
        missing=1
    else
        log_info "bisheng: $(bisheng --version 2>&1 | head -1)"
    fi

    if ! command -v g++ &>/dev/null; then
        log_error "g++ compiler not found in PATH"
        missing=1
    else
        log_info "g++: $(g++ --version 2>&1 | head -1)"
    fi

    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found in PATH"
        missing=1
    else
        log_info "python3: $(python3 --version 2>&1)"
    fi

    if ! command -v cmake &>/dev/null; then
        log_error "cmake not found in PATH"
        missing=1
    else
        log_info "cmake: $(cmake --version | head -1)"
    fi

    if [[ $missing -ne 0 ]]; then
        log_error "Pre-flight checks failed. Please fix the above issues and retry."
        exit 1
    fi

    log_success "All pre-flight checks passed"
}

# ── Example Discovery ────────────────────────────────────────────────────────
# Outputs lines of "ops_name/target_name" pairs.
# Skips: common/, scripts/, and non-directory entries.
discover_examples() {
    local ops_filter="$1"
    local target_filter="$2"

    local -a ops_list=()
    local _op
    if [[ -n "$ops_filter" ]]; then
        IFS=',' read -ra ops_list <<< "$ops_filter"
        for _op in "${ops_list[@]}"; do
            _op="${_op## }"
            _op="${_op%% }"
            if [[ ! -d "${EXAMPLES_DIR}/${_op}" ]]; then
                log_error "Operator directory not found: ${EXAMPLES_DIR}/${_op}"
                exit 1
            fi
        done
    else
        local op_dir op_name
        for op_dir in "${EXAMPLES_DIR}"/*/; do
            [[ -d "$op_dir" ]] || continue
            op_name="$(basename "$op_dir")"
            [[ "$op_name" == "common" || "$op_name" == "scripts" || "$op_name" == "build" ]] && continue
            ops_list+=("$op_name")
        done
    fi

    local -a targets_list=()
    if [[ -n "$target_filter" ]]; then
        IFS=',' read -ra targets_list <<< "$target_filter"
    fi

    local _t
    for _op in "${ops_list[@]}"; do
        if [[ ${#targets_list[@]} -eq 0 ]]; then
            _discover_op_examples "$_op"
        else
            for _t in "${targets_list[@]}"; do
                _t="${_t## }"
                _t="${_t%% }"
                if [[ ! -d "${EXAMPLES_DIR}/${_op}/${_t}" ]]; then
                    log_error "Example directory not found: ${EXAMPLES_DIR}/${_op}/${_t}"
                    exit 1
                fi
                echo "${_op}/${_t}"
            done
        fi
    done
}

_discover_op_examples() {
    local op_name="$1"
    local op_dir="${EXAMPLES_DIR}/${op_name}"
    local example_dir example_name
    for example_dir in "${op_dir}"/*/; do
        [[ -d "$example_dir" ]] || continue
        example_name="$(basename "$example_dir")"
        [[ "$example_name" == "common" || "$example_name" == "scripts" || "$example_name" == "build" ]] && continue
        echo "${op_name}/${example_name}"
    done
}

# ── Build ────────────────────────────────────────────────────────────────────
source "${REPO_ROOT}/scripts/submodule_utils.sh"

configure_build() {
    if ! ensure_tensor_api_submodule "${REPO_ROOT}"; then
        log_error "Failed to initialize tensor_api submodule"
        return 1
    fi

    log_info "Configuring CMake (build dir: ${BUILD_DIR})..."
    cmake -B "${BUILD_DIR}" \
        -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
        "${EXAMPLES_DIR}"
}

build_target() {
    local target="$1"

    log_info "Building target: ${target}"
    if ! cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"; then
        log_error "Build FAILED for target: ${target}"
        return 1
    fi

    log_success "Build succeeded for target: ${target}"
}

# ── Parallel Build (phase 1) ────────────────────────────────────────────────
# GNU make serializes multiple command-line goals (make -jN A B C runs them one
# after another), and `cmake --build --target A B C` inherits that behavior
# with the Makefiles generator. Parallelism is therefore achieved by launching
# one `cmake --build --target X` per target from bash.
# (Measured: 3 targets ≈ 121s sequential vs ≈ 44s with this approach.)

resolve_build_jobs() {
    local target_count="$1"
    local core_count
    core_count=$(nproc 2>/dev/null || echo 4)

    if [[ -n "$JOBS" ]]; then
        if (( JOBS > core_count )); then
            echo "$core_count"
        else
            echo "$JOBS"
        fi
        return
    fi

    local jobs=$core_count
    if (( target_count < jobs )); then
        jobs=$target_count
    fi

    # ~16GB per concurrent ASC TU (measured peak RSS of the heaviest TU: 13.5GB)
    local mem_cap=0
    if [[ -r /proc/meminfo ]]; then
        mem_cap=$(awk '/^MemTotal:/ {print int($2 / 1048576 / 16)}' /proc/meminfo 2>/dev/null || echo 0)
        mem_cap=${mem_cap:-0}
    fi
    if (( mem_cap > 0 && mem_cap < jobs )); then
        jobs=$mem_cap
    fi

    if (( jobs < 1 )); then
        jobs=1
    fi
    echo "$jobs"
}

build_selected_targets() {
    local -a targets=("$@")
    local n=${#targets[@]}
    local jobs
    jobs=$(resolve_build_jobs "$n")
    local t0
    t0=$(date +%s)

    if (( jobs == 1 )); then
        log_info "Building ${n} target(s) serially (jobs=1)"
        local t
        for t in "${targets[@]}"; do
            build_target "$t" || return 1
        done
        return 0
    fi

    mkdir -p "${PARALLEL_LOG_DIR}"
    : > "${BUILD_STATUS_FILE}"
    log_info "Building ${n} target(s) with up to ${jobs} parallel job(s)"
    log_info "Per-target build logs: ${PARALLEL_LOG_DIR}/"

    local t
    for t in "${targets[@]}"; do
        while (( $(jobs -pr | wc -l) >= jobs )); do
            sleep 0.2
        done
        (
            s=$(date +%s)
            if cmake --build "${BUILD_DIR}" --target "${t}" \
                     >"${PARALLEL_LOG_DIR}/${t}.log" 2>&1; then
                echo "OK ${t} $(( $(date +%s) - s ))" >> "${BUILD_STATUS_FILE}"
            else
                echo "FAIL ${t} $(( $(date +%s) - s ))" >> "${BUILD_STATUS_FILE}"
            fi
        ) &
    done
    wait || true

    local fail_list
    fail_list=$(awk '$1 == "FAIL" {printf "%s ", $2}' "${BUILD_STATUS_FILE}")
    local ok_count
    ok_count=$(grep -c '^OK' "${BUILD_STATUS_FILE}" || true)

    if [[ -n "$fail_list" ]] || (( ok_count != n )); then
        log_error "Parallel build FAILED (built ${ok_count}/${n})${fail_list:+; failed: ${fail_list}}"
        log_error "Full per-target logs: ${PARALLEL_LOG_DIR}/"
        log_info "Re-running builds serially to surface the failing target's output (built targets are no-ops)..."
        for t in "${targets[@]}"; do
            build_target "$t" || return 1
        done
        return 1
    fi

    awk '$1 == "OK" {printf "  [BUILT] %-48s %ss\n", $2, $3}' "${BUILD_STATUS_FILE}"
    log_success "Built ${n} target(s) in $(( $(date +%s) - t0 ))s (jobs: ${jobs})"
    return 0
}

# ── Run Single example ─────────────────────────────────────────────────────
# Arguments: ops_name example_name [prefix]
# Returns: 0 on PASS, 1 on FAIL
# With a prefix, run_case.py echoes per-case result lines to stderr tagged
# '[prefix] ...'; the caller decides where stdout/stderr each end up.
run_example() {
    local ops_name="$1"
    local example_name="$2"
    local prefix="${3:-}"

    local example_dir="${EXAMPLES_DIR}/${ops_name}/${example_name}"
    local exec_path="${BUILD_DIR}/${ops_name}/${example_name}"
    local conf_path="${example_dir}/${example_name}.conf"

    # ── Resolve CSV ──────────────────────────────────────────────────────
    local csv_file=""
    if [[ -n "$CASE_FILE" ]]; then
        csv_file="$CASE_FILE"
    else
        csv_file="${example_dir}/${example_name}.csv"
    fi

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: ${csv_file}"
        return 1
    fi

    # Building happens up front in build_selected_targets (phase 1);
    # this function only runs/verifies from here on.

    if [[ "$BUILD_ONLY" == true ]]; then
        return 0
    fi

    # ── Validate executable ──────────────────────────────────────────────
    if [[ ! -x "$exec_path" ]]; then
        log_error "Executable not found: ${exec_path}"
        log_error "Build first or remove --skip-build."
        return 1
    fi

    # ── Validate .conf ───────────────────────────────────────────────────
    if [[ ! -f "$conf_path" ]]; then
        log_error "Config file not found: ${conf_path}"
        return 1
    fi

    # ── Prepare result path ──────────────────────────────────────────────
    local result_file="${csv_file%.csv}_result.csv"

    # ── Run via run_case.py ──────────────────────────────────────────────
    local run_args=("$exec_path" "$csv_file" "$result_file" "$conf_path")
    if [[ -n "$TI_ARG" ]]; then
        run_args+=("--ti=${TI_ARG}")
    fi
    if [[ -n "$prefix" ]]; then
        run_args+=("--prefix=${prefix}")
    fi

    local run_rc=0
    python3 "${RUN_CASE_PY}" "${run_args[@]}" || run_rc=$?

    # ── Cleanup ──────────────────────────────────────────────────────────
    cleanup_data "$example_dir"

    return $run_rc
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup_data() {
    local example_dir="$1"
    if [[ -d "${example_dir}/input" ]]; then
        rm -rf "${example_dir}/input"
    fi
    if [[ -d "${example_dir}/output" ]]; then
        rm -rf "${example_dir}/output"
    fi
}

# ── Parallel Run (phase 2) ──────────────────────────────────────────────────
# Examples are independent (per-example input/ output/ data dirs), so they can
# run concurrently through a worker pool, mirroring build_selected_targets.
# All example binaries hardcode aclrtSetDevice(0); with >1 detected NPU device,
# each worker exports ASCEND_RT_VISIBLE_DEVICES to spread load across devices
# (the env var remaps device 0 to the assigned physical device per process).

detect_device_count() {
    local count
    count=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l)
    if (( count == 0 )) && command -v npu-smi &>/dev/null; then
        count=$(npu-smi info -l 2>/dev/null | awk '/Total Count/ {print $NF}')
        count="${count:-0}"
    fi
    echo "${count:-0}"
}

resolve_run_jobs() {
    local example_count="$1"
    local core_count
    core_count=$(nproc 2>/dev/null || echo 4)

    if [[ -n "$JOBS" ]]; then
        if (( JOBS > core_count )); then
            echo "$core_count"
        else
            echo "$JOBS"
        fi
        return
    fi

    local dev_count
    dev_count=$(detect_device_count)

    local jobs=8
    if (( dev_count >= 1 )); then
        jobs=$(( dev_count * 4 ))
    fi
    if (( example_count < jobs )); then
        jobs=$example_count
    fi
    if (( core_count < jobs )); then
        jobs=$core_count
    fi
    if (( jobs < 1 )); then
        jobs=1
    fi
    echo "$jobs"
}

run_selected_examples() {
    local -a examples=("$@")
    local n=${#examples[@]}
    local jobs
    jobs=$(resolve_run_jobs "$n")
    if (( n <= 1 )); then
        jobs=1
    fi

    mkdir -p "${PARALLEL_LOG_DIR}"
    : > "${RUN_STATUS_FILE}"

    local example_path ops_name example_name log_file rc t0 duration
    if (( jobs == 1 )); then
        log_info "Running ${n} example(s) serially (jobs=1)"
        for example_path in "${examples[@]}"; do
            ops_name="${example_path%%/*}"
            example_name="${example_path##*/}"
            log_info "Running ${ops_name}/${example_name}"
            t0=$(date +%s)
            set +e
            run_example "$ops_name" "$example_name"
            rc=$?
            set -e
            duration=$(( $(date +%s) - t0 ))
            if [[ $rc -eq 0 ]]; then
                echo "OK ${ops_name}/${example_name} ${duration}" >> "${RUN_STATUS_FILE}"
                log_success "[PASS] ${ops_name}/${example_name}"
            else
                echo "FAIL ${ops_name}/${example_name} ${duration}" >> "${RUN_STATUS_FILE}"
                log_error "[FAIL] ${ops_name}/${example_name}"
            fi
        done
    else
        local dev_count
        dev_count=$(detect_device_count)
        mkdir -p "${RUN_LOG_DIR}"
        log_info "Running ${n} example(s) with up to ${jobs} parallel job(s)"
        log_info "Per-example run logs: ${RUN_LOG_DIR}/"

        local dev_idx=0
        for example_path in "${examples[@]}"; do
            ops_name="${example_path%%/*}"
            example_name="${example_path##*/}"
            log_file="${RUN_LOG_DIR}/${ops_name}_${example_name}.log"
            while (( $(jobs -pr | wc -l) >= jobs )); do
                sleep 0.2
            done
            (
                if (( dev_count > 1 )); then
                    export ASCEND_RT_VISIBLE_DEVICES=$(( dev_idx % dev_count ))
                fi
                t0=$(date +%s)
                # stdout -> per-example log; stderr (run_case.py's prefixed
                # per-case result echo) -> console, live.
                if run_example "$ops_name" "$example_name" "$example_name" > "$log_file"; then
                    duration=$(( $(date +%s) - t0 ))
                    echo "OK ${ops_name}/${example_name} ${duration}" >> "${RUN_STATUS_FILE}"
                    log_success "[PASS] ${ops_name}/${example_name} (${duration}s)"
                else
                    duration=$(( $(date +%s) - t0 ))
                    echo "FAIL ${ops_name}/${example_name} ${duration}" >> "${RUN_STATUS_FILE}"
                    log_error "[FAIL] ${ops_name}/${example_name} (${duration}s, log: ${log_file})"
                fi
            ) &
            dev_idx=$((dev_idx + 1))
        done
        wait || true
    fi

    local total_pass total_fail
    total_pass=$(grep -c '^OK' "${RUN_STATUS_FILE}" || true)
    total_fail=$(grep -c '^FAIL' "${RUN_STATUS_FILE}" || true)

    echo ""
    echo "=========================================="
    echo "  Summary"
    echo "=========================================="
    awk '{ printf "  [%s] %-55s %ss\n", $1, $2, $3 }' "${RUN_STATUS_FILE}"
    echo "------------------------------------------"
    log_info "Total: ${n}  |  PASS: ${total_pass}  |  FAIL: ${total_fail}  |  jobs: ${jobs}"
    echo "=========================================="

    if (( total_fail > 0 || total_pass + total_fail != n )); then
        if (( jobs > 1 )); then
            log_error "Per-example run logs: ${RUN_LOG_DIR}/"
        fi
        log_error "Some examples FAILED"
        return 1
    fi

    log_success "All examples PASSED"
    return 0
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo "=========================================="
    echo "  ops-tensor Examples Unified Runner"
    echo "=========================================="

    preflight

    # ── Force submodule sync (once, independent of --skip-build) ────────
    if [[ "$FORCE_SUBMODULE" == true ]]; then
        if ! ensure_tensor_api_submodule "${REPO_ROOT}" true; then
            log_error "Failed to force-update tensor_api submodule"
            exit 1
        fi
        FORCE_SUBMODULE=false
    fi

    # ── Discover examples ───────────────────────────────────────────────
    local examples
    examples="$(discover_examples "$OPS_NAME" "$TARGET")"

    if [[ -z "$examples" ]]; then
        log_error "No examples found (ops=${OPS_NAME:-<all>}, target=${TARGET:-<all>})"
        exit 1
    fi

    local example_count
    example_count="$(echo "$examples" | wc -l)"
    log_info "Discovered ${example_count} example(s)"

    # ── Configure once for all examples ─────────────────────────────────
    if [[ "$SKIP_BUILD" == true ]]; then
        log_info "Skipping build (--skip-build)"
    else
        if ! configure_build; then
            log_error "CMake configuration failed"
            exit 1
        fi

        # Phase 1: build every selected target up front, in parallel.
        local -a all_targets=()
        local _target_name
        while IFS= read -r _target_name; do
            if [[ -n "$_target_name" ]]; then
                all_targets+=("$_target_name")
            fi
        done < <(printf '%s\n' "$examples" | awk -F'/' 'NF >= 2 && !seen[$2]++ {print $2}')

        if ! build_selected_targets "${all_targets[@]}"; then
            exit 1
        fi
    fi
    if [[ "$BUILD_ONLY" == true ]]; then
        log_info "Build-only mode: no execution after build"
    fi

    # ── Run examples (phase 2: parallel worker pool) ────────────────────
    local -a example_list=()
    local _example_path
    while IFS= read -r _example_path; do
        if [[ -n "$_example_path" ]]; then
            example_list+=("$_example_path")
        fi
    done <<< "$examples"

    if ! run_selected_examples "${example_list[@]}"; then
        exit 1
    fi
}

main "$@"
