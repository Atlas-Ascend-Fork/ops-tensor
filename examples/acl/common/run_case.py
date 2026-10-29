#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Generic CSV-driven batch execution engine for ops-tensor examples.

Reads a .conf file (INI format) to route parameters from a CSV file into
gen_data -> kernel -> verify three-stage execution.

.conf format (INI with three sections):

    [gen_data]
    params = m, k, n, --dtype:dtype, $OUTPUT_DIR
    bool_flags = ++trans-a:transA

    [kernel]
    params = m, k, n, dtype

    [verify]
    params = m, n, dtype
    bool_flags = ++hf32:hf32

Param entry syntax (comma-separated within each field):
    "column_name"           positional arg: passes CSV value as-is
    "--flag:column_name"    flag arg: appends "--flag {csv_value}"
    "++flag:column_name"    boolean flag: appends "--flag" only if CSV value is true
    "$OUTPUT_DIR"           runtime token: injected by run_case.py

Usage:
    python3 run_case.py <executable> <csv_file> <result_file> <conf_file> [options]
"""

import argparse
import configparser
import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import METRICS_FILENAME, parse_stdout_metrics


# ---------------------------------------------------------------------------
# .conf parsing
# ---------------------------------------------------------------------------


def load_conf(conf_path):
    """Parse a .conf file and return a ConfigParser instance."""
    conf = configparser.ConfigParser()
    conf.read(conf_path, encoding="utf-8")
    return conf


def parse_entries(raw):
    """Split a comma-separated .conf field into stripped, non-empty tokens."""
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def get_section_entries(conf, section):
    """Return (params_list, bool_flags_list) for a .conf section."""
    params = parse_entries(conf.get(section, "params", fallback=""))
    bool_flags = parse_entries(conf.get(section, "bool_flags", fallback=""))
    return params, bool_flags


# ---------------------------------------------------------------------------
# Value resolution
# ---------------------------------------------------------------------------


def resolve_value(spec, row, runtime_vars):
    """Resolve a value specification into a string.

    Supported specs:
        column_name           → CSV value as-is
        $TOKEN                → runtime value (e.g. $OUTPUT_DIR)
        $TOKEN/path           → runtime value with path suffix (e.g. $OUTPUT_DIR/file.bin)
        ="literal"            → literal string
    """
    if spec.startswith("="):
        return spec[1:]

    if spec.startswith("$"):
        if "/" in spec:
            token, path_suffix = spec.split("/", 1)
            if token not in runtime_vars:
                raise ValueError(f"Unknown runtime token: {token}")
            return os.path.join(runtime_vars[token], path_suffix)
        if spec not in runtime_vars:
            raise ValueError(f"Unknown runtime token: {spec}")
        return runtime_vars[spec]

    if spec not in row:
        raise ValueError(
            f"Column '{spec}' not found in CSV. Available columns: {list(row.keys())}"
        )
    return row[spec].strip()


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def build_args(entries, row, runtime_vars):
    """Translate .conf entries into command-line arguments using CSV row data.

    Entry types:
        column_name           → positional arg: CSV value
        $TOKEN                → positional arg: runtime value
        $TOKEN/path           → positional arg: runtime value + path
        ="literal"            → positional arg: literal string
        --flag:spec           → flag arg: --flag {resolved_value}
        ++flag:column         → boolean flag: --flag only if CSV value is "true"
    """
    args = []

    for entry in entries:
        if entry.startswith("++"):
            # Boolean flag: ++flag:column
            inner = entry[2:]
            if ":" not in inner:
                raise ValueError(
                    f"Invalid boolean flag syntax: '{entry}'. Expected '++flag:column'"
                )
            flag, col = inner.split(":", 1)
            if col not in row:
                raise ValueError(
                    f"Column '{col}' referenced by '{entry}' not found in CSV. "
                    f"Available columns: {list(row.keys())}"
                )
            if row[col].strip().lower() == "true":
                args.append(f"--{flag}")

        elif entry.startswith("--"):
            # Flag with value: --flag:value_spec
            inner = entry[2:]
            if ":" not in inner:
                raise ValueError(
                    f"Invalid flag syntax: '{entry}'. "
                    f"Expected '--flag:column' or '--flag:$TOKEN' etc."
                )
            flag, value_spec = inner.split(":", 1)
            value = resolve_value(value_spec, row, runtime_vars)
            args.append(f"--{flag}")
            args.append(value)

        else:
            # Positional argument: value_spec
            value = resolve_value(entry, row, runtime_vars)
            args.append(value)

    return args


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(conf_path, conf):
    """Derive scripts/, gen_data.py, verify_result.py, and output/ from .conf location.

    Layout:
        examples/{ops}/{example}/{example}.conf   <- conf_path
        examples/{ops}/scripts/                    <- scripts_dir (gen/verify scripts)
        examples/{ops}/{example}/output/           <- $OUTPUT_DIR (per-example data)
        examples/{ops}/{example}/input/            <- $INPUT_DIR (per-example data)

    Optional [scripts] section in .conf overrides script filenames:
        [scripts]
        gen_data=gen_data.py
        verify=verify_result.py

    Scripts must reside under examples/{ops}/scripts/.
    """
    example_dir = os.path.dirname(os.path.abspath(conf_path))
    ops_dir = os.path.dirname(example_dir)
    scripts_dir = os.path.join(ops_dir, "scripts")

    gen_data_name = conf.get("scripts", "gen_data", fallback="gen_data.py")
    verify_name = conf.get("scripts", "verify", fallback="verify_result.py")

    gen_data_py = os.path.join(scripts_dir, gen_data_name)
    verify_py = os.path.join(scripts_dir, verify_name)
    output_dir = os.path.join(example_dir, "output")
    input_dir = os.path.join(example_dir, "input")
    return scripts_dir, gen_data_py, verify_py, output_dir, input_dir


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def run_subprocess(cmd, cwd=None, timeout=300, settle=0):
    """Run a subprocess and return (returncode, combined_output).

    Args:
        settle: seconds to wait after process exits before returning.
            Used for kernel processes that release NPU device resources
            asynchronously (aclrtResetDevice + aclFinalize).
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if settle > 0:
            time.sleep(settle)
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT after {0}s".format(timeout)
    except Exception as exc:
        return -1, str(exc)


# ---------------------------------------------------------------------------
# --ti index parsing
# ---------------------------------------------------------------------------


def parse_ti(ti_value):
    """Parse the --ti argument into a set of 0-based indices.
    Returns None to indicate 'run all'.
    """
    if ti_value is None:
        return None
    if "-" in ti_value:
        parts = ti_value.split("-", 1)
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            print(
                f"Error: Invalid --ti range: '{ti_value}'. Use N or N-M.",
                file=sys.stderr,
            )
            sys.exit(1)
        if start > end:
            print(f"Error: --ti range start > end: {start} > {end}", file=sys.stderr)
            sys.exit(1)
        return set(range(start, end + 1))
    try:
        return {int(ti_value)}
    except ValueError:
        print(
            f"Error: Invalid --ti index: '{ti_value}'. Must be an integer.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metrics extraction from verify_metrics.json
# ---------------------------------------------------------------------------


def _load_metrics(output_dir, verify_stdout, rc):
    metrics_path = os.path.join(output_dir, METRICS_FILENAME)
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] failed to parse {metrics_path}: {exc}", file=sys.stderr)
    return parse_stdout_metrics(verify_stdout, rc)


def _extract_accuracy(metrics):
    if not metrics or not metrics.get("outputs"):
        return ""
    err_values = [o.get("error_ratio") for o in metrics["outputs"]]
    if any(e is None for e in err_values):
        return ""
    worst_err = max(err_values)
    return f"{(1.0 - worst_err) * 100:.2f}%"


def _format_metrics_json(metrics):
    """Convert error_ratio→accuracy, ratio_tol→tol accuracy (both percentage)."""
    if not metrics:
        return ""
    result = {"outputs": [], "overall_status": metrics.get("overall_status", "fail")}
    for o in metrics.get("outputs", []):
        entry = {"name": o.get("name", "output"), "status": o.get("status", "fail")}
        if o.get("max_abs_diff") is not None:
            entry["max_abs_diff"] = o["max_abs_diff"]
        err = o.get("error_ratio")
        if err is not None:
            entry["accuracy"] = f"{(1.0 - err) * 100:.2f}%"
        tol = o.get("ratio_tol")
        if tol is not None:
            entry["ratio_tol"] = f"{(1.0 - tol) * 100:.2f}%"
        result["outputs"].append(entry)
    return json.dumps(result, ensure_ascii=False)


def _format_metrics_lines(metrics):
    """Format per-output accuracy/tolerance/max-diff lines (PASS and FAIL paths)."""
    lines = []
    if not (metrics and metrics.get("outputs")):
        return lines
    for out in metrics["outputs"]:
        name = out.get("name", "output")
        parts = []
        if out.get("error_ratio") is not None:
            parts.append(f"accuracy={(1.0 - out['error_ratio']) * 100:.2f}%")
        if out.get("ratio_tol") is not None:
            parts.append(f"tol {(1.0 - out['ratio_tol']) * 100:.2f}%")
        if out.get("max_abs_diff") is not None:
            parts.append(f"max_abs_diff={out['max_abs_diff']:.6e}")
        if parts:
            lines.append(f"  {name}: {', '.join(parts)}")
    return lines


def _print_case_lines(lines, prefix):
    """Print per-case result lines.

    Lines always go to stdout. When a prefix is set, each line is also echoed
    to stderr as '[prefix] line' so a parallel runner can redirect stdout to a
    per-example log file while still streaming case results to the console.
    """
    for line in lines:
        print(line)
        if prefix:
            print(f"[{prefix}] {line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Per-case execution
# ---------------------------------------------------------------------------


def run_single_case(
    executable, row, conf, example_dir, gen_data_py, verify_py, output_dir, input_dir
):
    """Execute one test case through gen_data -> kernel -> verify.

    All three stages run with cwd = the example directory, so that CWD-relative
    scripts and kernels (./input, ./output) resolve to the per-example data
    directories instead of the shared {op}/scripts/ directory.
    """
    runtime_vars = {"$OUTPUT_DIR": output_dir, "$INPUT_DIR": input_dir}

    gen_params, gen_bools = get_section_entries(conf, "gen_data")
    kernel_params, kernel_bools = get_section_entries(conf, "kernel")
    verify_params, verify_bools = get_section_entries(conf, "verify")

    # -- Stage 1: gen_data ---------------------------------------------------
    gen_args = build_args(gen_params + gen_bools, row, runtime_vars)
    gen_cmd = [sys.executable, gen_data_py] + gen_args
    rc, out = run_subprocess(gen_cmd, cwd=example_dir)
    if rc != 0:
        return "FAIL", "gen_data", out, None

    # -- Stage 2: kernel binary ----------------------------------------------
    kernel_args = build_args(kernel_params + kernel_bools, row, runtime_vars)
    kernel_cmd = [executable] + kernel_args
    rc, out = run_subprocess(kernel_cmd, cwd=example_dir, settle=1)
    if rc != 0:
        return "FAIL", "kernel", out, None

    # -- Stage 3: verify -----------------------------------------------------
    verify_args = build_args(verify_params + verify_bools, row, runtime_vars)
    verify_cmd = [sys.executable, verify_py] + verify_args
    rc, out = run_subprocess(verify_cmd, cwd=example_dir)

    metrics = _load_metrics(output_dir, out, rc)

    if rc != 0:
        return "FAIL", "verify", out, metrics

    return "PASS", "verify", "" if metrics else out, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generic CSV-driven batch execution engine for ops-tensor examples.",
        epilog=(
            "Example:\n"
            "  python3 run_case.py ./build/mat_mul_basic/mat_mul_basic \\\n"
            "      cases.csv result.csv mat_mul_basic.conf --ti=0-5\n"
            "\n"
            "The .conf file (INI format) defines how CSV columns map to command\n"
            "arguments for each stage (gen_data, kernel, verify).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("executable", help="Path to the compiled kernel binary")
    parser.add_argument("csv_file", help="Path to the CSV test-case file")
    parser.add_argument("result_file", help="Path to write the result CSV")
    parser.add_argument(
        "conf_file", help="Path to the .conf file (INI format) for parameter routing"
    )
    parser.add_argument(
        "--ti",
        default=None,
        metavar="N|N-M",
        help="Run only specific test index(es): N or N-M (0-based)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        metavar="TAG",
        help="Echo per-case result lines to stderr as '[TAG] ...' so parallel "
        "runners can stream them to the console while stdout goes to a log",
    )

    args = parser.parse_args()

    # -- Validate inputs -----------------------------------------------------
    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.conf_file):
        print(f"Error: Config file not found: {args.conf_file}", file=sys.stderr)
        print(
            "Hint: Each example needs a .conf file. "
            "See the task documentation for the format.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(args.executable):
        print(f"Error: Executable not found: {args.executable}", file=sys.stderr)
        sys.exit(1)
    # Kernels run with cwd = example dir; resolve against this process's cwd first.
    args.executable = os.path.abspath(args.executable)

    # -- Parse .conf ---------------------------------------------------------
    conf = load_conf(args.conf_file)

    # -- Resolve paths -------------------------------------------------------
    scripts_dir, gen_data_py, verify_py, output_dir, input_dir = resolve_paths(
        args.conf_file, conf
    )
    # Subprocess cwd; must match where $INPUT_DIR/$OUTPUT_DIR live (see run_single_case).
    example_dir = os.path.dirname(os.path.abspath(args.conf_file))

    if not os.path.isfile(gen_data_py):
        print(f"Error: gen_data script not found: {gen_data_py}", file=sys.stderr)
        print(
            f"Hint: Check [scripts] gen_data in {args.conf_file}, "
            f"file must be under {scripts_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(verify_py):
        print(f"Error: verify script not found: {verify_py}", file=sys.stderr)
        print(
            f"Hint: Check [scripts] verify in {args.conf_file}, "
            f"file must be under {scripts_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Parse --ti filter ---------------------------------------------------
    ti_indices = parse_ti(args.ti)

    # -- Read CSV ------------------------------------------------------------
    with open(args.csv_file, "r", newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    if not all_rows:
        print("No test cases found in CSV file.")
        sys.exit(0)

    # -- Apply --ti filter ---------------------------------------------------
    if ti_indices is not None:
        indexed_rows = [(i, row) for i, row in enumerate(all_rows) if i in ti_indices]
    else:
        indexed_rows = list(enumerate(all_rows))

    if not indexed_rows:
        print(
            f"No test cases match --ti={args.ti} "
            f"(CSV has {len(all_rows)} rows, indices 0-{len(all_rows) - 1})"
        )
        sys.exit(0)

    total = len(indexed_rows)
    print(f"Found {len(all_rows)} test case(s) in CSV, running {total}")
    print("=" * 70)

    # -- Ensure output directory exists --------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    # -- Execute cases -------------------------------------------------------
    results = []
    pass_count = 0
    fail_count = 0

    for seq, (idx, row) in enumerate(indexed_rows, 1):
        casename = (row.get("casename") or row.get("caseName") or f"case_{idx}").strip()
        try:
            status, stage, message, metrics = run_single_case(
                args.executable,
                row,
                conf,
                example_dir,
                gen_data_py,
                verify_py,
                output_dir,
                input_dir,
            )
        except ValueError as exc:
            status, stage, message, metrics = "FAIL", "config", str(exc), None

        if status == "PASS":
            pass_count += 1
            case_lines = [f"[PASS] {casename}"]
            case_lines.extend(_format_metrics_lines(metrics))
        else:
            fail_count += 1
            case_lines = [f"[FAIL] {casename} (stage={stage})"]
            if message:
                preview = message[:500]
                if len(message) > 500:
                    preview += "..."
                case_lines.append(f"  Error: {preview}")
            case_lines.extend(_format_metrics_lines(metrics))
        _print_case_lines(case_lines, args.prefix)

        results.append(
            {
                "casename": casename,
                "status": status,
                "stage": stage,
                "message": message,
                "accuracy": _extract_accuracy(metrics),
                "metrics_json": _format_metrics_json(metrics),
            }
        )

    # -- Write result CSV ----------------------------------------------------
    with open(args.result_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "casename",
                "status",
                "stage",
                "message",
                "accuracy",
                "metrics_json",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # -- Summary -------------------------------------------------------------
    if fail_count > 0:
        print("\n" + "=" * 70)
        print("  Failed Cases")
        print("=" * 70)
        for r in results:
            if r["status"] != "PASS":
                print(f"  [FAIL] {r['casename']} (stage={r['stage']})")
        print("-" * 70)
        print(f"  Total: {len(results)}  |  PASS: {pass_count}  |  FAIL: {fail_count}")
        print(f"  Results written to: {args.result_file}")
        print("=" * 70)
        sys.exit(1)

    print(f"\nAll {pass_count} case(s) passed. Results: {args.result_file}")


if __name__ == "__main__":
    main()
