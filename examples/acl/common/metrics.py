#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""Shared metrics utilities for ops-tensor examples.

Provides:
- write_metrics_json:  write verify_metrics.json (called by verify_result.py)
- parse_stdout_metrics: fallback parse of verify stdout (called by run_case.py
  when verify_metrics.json is absent, e.g. new verify without JSON output)
"""

import json
import os
import re

METRICS_FILENAME = "verify_metrics.json"

_MAX_ABS_DIFF_PATTERNS = [
    re.compile(r"max abs diff:\s*([0-9.eE+-]+)"),
    re.compile(r"max_abs_error=([0-9.eE+-]+)"),
]

_ERROR_RATIO_PATTERNS = [
    re.compile(r"error ratio:\s*([0-9.eE+-]+)"),
    re.compile(r"ratio=([0-9.eE+-]+)"),
]


def write_metrics_json(outputs, overall_status, output_dir="./output"):
    metrics = {"outputs": outputs, "overall_status": overall_status}
    path = os.path.join(output_dir, METRICS_FILENAME)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _try_match_float(content, patterns):
    for pat in patterns:
        m = pat.search(content)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _parse_ratio_tol(content):
    found = None
    for line in content.splitlines():
        if "count" in line.lower():
            m = re.search(r">\s*([0-9]+\.?[0-9eE+-]*)", line)
            if m:
                try:
                    found = float(m.group(1))
                except ValueError:
                    pass
    return found


def _parse_output_block(name, content, status):
    max_abs_diff = _try_match_float(content, _MAX_ABS_DIFF_PATTERNS)
    error_ratio = _try_match_float(content, _ERROR_RATIO_PATTERNS)
    if max_abs_diff is None and error_ratio is None:
        return None
    ratio_tol = _parse_ratio_tol(content)
    result = {"name": name, "status": status}
    if max_abs_diff is not None:
        result["max_abs_diff"] = max_abs_diff
    if error_ratio is not None:
        result["error_ratio"] = error_ratio
    if ratio_tol is not None:
        result["ratio_tol"] = ratio_tol
    return result


def parse_stdout_metrics(stdout, rc):
    """Fallback: parse verify stdout when verify_metrics.json is absent.

    Returns {"outputs": [...], "overall_status": ...} or None.
    Handles single-output, multi-output ([verify] name: blocks), and
    grouped ([INFO] group N: blocks) verify scripts.
    """
    if not stdout:
        return None
    status = "pass" if rc == 0 else "fail"
    outputs = []

    verify_blocks = re.split(r"\[verify\]\s*(\S+?):", stdout)
    if len(verify_blocks) > 1:
        for i in range(1, len(verify_blocks), 2):
            name = verify_blocks[i]
            content = verify_blocks[i + 1] if i + 1 < len(verify_blocks) else ""
            out = _parse_output_block(name, content, status)
            if out:
                outputs.append(out)

    if not outputs:
        group_blocks = re.split(r"\[INFO\]\s*group\s+(\d+):", stdout)
        if len(group_blocks) > 1:
            for i in range(1, len(group_blocks), 2):
                name = f"group_{group_blocks[i]}"
                content = group_blocks[i + 1] if i + 1 < len(group_blocks) else ""
                out = _parse_output_block(name, content, status)
                if out:
                    outputs.append(out)

    if not outputs:
        out = _parse_output_block("output", stdout, status)
        if out:
            outputs.append(out)

    if not outputs:
        return None
    return {"outputs": outputs, "overall_status": status}
