#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor — Shared tensor_api Submodule Utilities
#
# Single source of truth for initializing / force-syncing the tensor_api
# submodule (include/tensor_api) to the pinned commit recorded in the
# superproject. Consumed by:
#   - build.sh                 (--opkernel -u / --examples modes)
#   - examples/acl/common/run.sh   (unified examples runner)
#
# Usage:
#   source "${REPO_ROOT}/scripts/submodule_utils.sh"
#   ensure_tensor_api_submodule "${repo_root}"          # init if missing
#   ensure_tensor_api_submodule "${repo_root}" true     # force sync to pinned commit
#
# Caller contract:
#   - Must define log_info / log_success / log_error functions before sourcing
#   - Pass repo root (absolute path) as $1, optional force flag as $2
#   - Returns non-zero on failure; the caller decides whether to exit
#   - Compatible with both `set -e` and `set -euo pipefail` environments
##############################################################################

# 初始化 tensor_api submodule（使用 superproject 记录的 pinned commit）
# examples/CMakeLists.txt 的 include 路径只指向 submodule，不会回退到 CANN 环境
# force=true：跳过"已存在即跳过"检查，git submodule update --force 同步到
# pinned commit（丢弃 submodule 内本地修改），并校验最终 id 与 gitlink 一致
ensure_tensor_api_submodule() {
    local repo_root="${1:-.}"
    local force="${2:-false}"
    local _submod_dir="${repo_root}/include/tensor_api"
    local _repo_url="https://gitcode.com/cann/asc-devkit.git"
    local _repo_branch="feature/tensor_api_from_9.0.0"

    if [ "${force}" != true ] && [ -d "$_submod_dir" ] && [ -n "$(ls -A "$_submod_dir" 2>/dev/null)" ]; then
        log_info "tensor_api already exists, skip init"
        return 0
    fi

    if [ "${force}" = true ]; then
        log_info "Forcing tensor_api submodule update to pinned commit..."
    else
        log_info "Initializing tensor_api submodule..."
    fi

    local _fail_msg=""
    if ! command -v git &> /dev/null; then
        _fail_msg="git is not installed"
    elif ! (cd "${repo_root}" && git rev-parse --is-inside-work-tree &> /dev/null); then
        _fail_msg="not a git repository: ${repo_root}"
    else
        local _update_flags=(--init --recursive)
        if [ "${force}" = true ]; then
            _update_flags+=(--force)
        fi
        local _rc=0
        (cd "${repo_root}" && git submodule update "${_update_flags[@]}" include/tensor_api 2>&1) || _rc=$?
        if [ "${_rc}" -ne 0 ]; then
            _fail_msg="git submodule update failed (exit code: ${_rc})"
        elif [ -z "$(ls -A "${_submod_dir}" 2>/dev/null)" ]; then
            _fail_msg="tensor_api directory is empty after init"
        elif [ "${force}" = true ]; then
            local _pinned_id
            local _actual_id
            _pinned_id=$(cd "${repo_root}" && git ls-files -s include/tensor_api | awk '{print $2}')
            _actual_id=$(git -C "${_submod_dir}" rev-parse HEAD 2>/dev/null || true)
            if [ -n "${_pinned_id}" ] && [ "${_pinned_id}" != "${_actual_id}" ]; then
                _fail_msg="tensor_api is at ${_actual_id:-unknown}, expected pinned commit ${_pinned_id}"
            fi
        fi
    fi

    if [ -n "$_fail_msg" ]; then
        log_error "${_fail_msg}"
        log_error "Manually download and place at: ${_submod_dir}"
        log_error "Repo: ${_repo_url} (branch: ${_repo_branch})"
        return 1
    fi

    if [ "${force}" = true ]; then
        log_success "tensor_api submodule updated to pinned commit $(git -C "${_submod_dir}" rev-parse --short HEAD 2>/dev/null)"
    else
        log_success "tensor_api submodule initialized"
    fi
}
