#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare all three device outputs with the exact fixture bytes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import write_metrics_json


def verify(output_dir):
    metrics = []
    for name in ("y", "level0_scale", "level1_scale"):
        golden = (output_dir / f"golden_{name}.bin").read_bytes()
        try:
            actual = (output_dir / f"npu_{name}.bin").read_bytes()
        except OSError as error:
            print(f"[verify] {name}: {error}")
            metrics.append({"name": name, "status": "fail"})
            continue

        byte_errors = abs(len(actual) - len(golden)) + sum(
            g != a for g, a in zip(golden, actual)
        )
        status = "pass" if byte_errors == 0 else "fail"
        metrics.append(
            {
                "name": name,
                "status": status,
                "error_ratio": byte_errors / max(len(golden), len(actual), 1),
                "ratio_tol": 0.0,
            }
        )
        print(
            f"[verify] {name}: {status}, byte differences={byte_errors}, expected={len(golden)}, actual={len(actual)}"
        )

    passed = all(output["status"] == "pass" for output in metrics)
    write_metrics_json(metrics, "pass" if passed else "fail", output_dir)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return 0 if verify(parser.parse_args().output_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
