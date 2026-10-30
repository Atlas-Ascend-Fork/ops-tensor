#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""classify_rule.yaml 登记校验（pre-commit 守卫）。

规则：
  - 仅校验 include/blaze/ 下新增（含重命名落点）的文件；
  - 仅校验 .h / .cpp 有效实现文件，.md 等文档不参与；
  - 文件若未被 classify_rule.yaml 登记（显式文件条目或目录条目均可）则校验失败。

用法：
  python3 scripts/check_classify_rule.py          # 校验暂存区新增文件
  python3 scripts/check_classify_rule.py --all    # 校验工作树全部 include/blaze 文件

退出码：0 通过；1 存在未登记文件；2 环境/文件异常。
"""

import argparse
import os
import re
import subprocess
import sys

RULE_FILE = "classify_rule.yaml"
GOVERNED_PREFIX = "include/blaze/"
GUARDED_EXTS = (".h", ".cpp")
# classify_rule.yaml 中的路径带 monorepo 前缀，需归一化为仓库相对路径
RULE_PREFIX_CANDIDATES = ("ops/ops-tensor/",)

ENTRY_RE = re.compile(r"^\s*-\s+(\S+)\s*$")


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if out.returncode != 0:
        print("Error: 当前目录不在 git 仓库内", file=sys.stderr)
        sys.exit(2)
    return out.stdout.strip()


def load_rule_entries(root):
    """返回 (显式文件集合, 目录条目列表)。"""
    rule_path = os.path.join(root, RULE_FILE)
    if not os.path.isfile(rule_path):
        print(f"Error: 未找到 {RULE_FILE}", file=sys.stderr)
        sys.exit(2)

    files, dirs = set(), []
    with open(rule_path, encoding="utf-8") as fh:
        for line in fh:
            matched = ENTRY_RE.match(line)
            if not matched:
                continue
            entry = matched.group(1)
            for prefix in RULE_PREFIX_CANDIDATES:
                if entry.startswith(prefix):
                    entry = entry[len(prefix) :]
                    break
            if entry.endswith("/"):
                dirs.append(entry)
            else:
                files.add(entry)
    return files, dirs


def staged_new_files(root):
    """暂存区中新增/重命名落点的文件（A + R 的新路径）。"""
    out = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=AR",
            "--",
            GOVERNED_PREFIX,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print("Error: 读取暂存区失败", file=sys.stderr)
        sys.exit(2)
    return [p for p in out.stdout.splitlines() if p]


def worktree_files(root):
    base = os.path.join(root, GOVERNED_PREFIX)
    result = []
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            full = os.path.join(dirpath, name)
            result.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(result)


def is_guarded(path):
    """仅 include/blaze 下的 .h/.cpp 参与校验。"""
    return path.startswith(GOVERNED_PREFIX) and path.endswith(GUARDED_EXTS)


def main():
    parser = argparse.ArgumentParser(description="classify_rule.yaml 登记校验")
    parser.add_argument(
        "--all",
        action="store_true",
        help="校验工作树下全部 include/blaze 文件（默认仅校验暂存区新增文件）",
    )
    args = parser.parse_args()

    root = repo_root()
    rule_files, rule_dirs = load_rule_entries(root)

    candidates = worktree_files(root) if args.all else staged_new_files(root)

    missing = []
    for path in candidates:
        if not is_guarded(path):
            continue
        if path in rule_files:
            continue
        if any(path.startswith(d) for d in rule_dirs):
            continue
        missing.append(path)

    if missing:
        print(
            f"classify_rule.yaml 未登记以下 {len(missing)} 个新增文件：",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print(
            "请在 classify_rule.yaml 对应业务块的 style: 列表中补充该文件，"
            "或为其所在目录添加目录条目（以 / 结尾）。",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
