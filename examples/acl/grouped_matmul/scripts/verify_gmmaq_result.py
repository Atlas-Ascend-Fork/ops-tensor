#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Byte-exact verification for the deterministic GMMAQ MX example."""

import argparse


def compare(expected_path, actual_path, name):
    with open(expected_path, "rb") as expected_file:
        expected = expected_file.read()
    with open(actual_path, "rb") as actual_file:
        actual = actual_file.read()
    if expected != actual:
        limit = min(len(expected), len(actual))
        first = next((i for i in range(limit) if expected[i] != actual[i]), limit)
        raise ValueError(
            f"{name} mismatch: expected={len(expected)} bytes, actual={len(actual)} bytes, "
            f"first_mismatch={first}"
        )
    print(f"[PASS] {name}: {len(actual)} bytes are identical")


def main():
    parser = argparse.ArgumentParser(description="Verify GMMAQ MX output")
    parser.add_argument("golden_y")
    parser.add_argument("actual_y")
    parser.add_argument("golden_scale")
    parser.add_argument("actual_scale")
    args = parser.parse_args()
    compare(args.golden_y, args.actual_y, "Y")
    compare(args.golden_scale, args.actual_scale, "YScale")


if __name__ == "__main__":
    main()
