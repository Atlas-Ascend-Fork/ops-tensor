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
# ops-tensor — pre-commit 本地钩子的 Python 启动器
#
# 优先使用 python3；若当前环境没有 python3（例如仅提供 python 别名），
# 退化为 python 执行，避免钩子因解释器名缺失而直接失败。
#
# 用法：bash scripts/pre_commit_python.sh <python_script> [args...]
##############################################################################

set -uo pipefail

for python_cmd in python3 python; do
    if command -v "${python_cmd}" >/dev/null 2>&1; then
        exec "${python_cmd}" "$@"
    fi
done

echo "[pre-commit] ERROR: 未找到 python3 或 python，请先安装 Python 3.7 及以上版本。" >&2
exit 2
