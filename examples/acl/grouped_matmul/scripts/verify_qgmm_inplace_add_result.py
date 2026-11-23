#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#


"""Verify both inplace-add launches, including preservation of empty groups."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    directory = parser.parse_args().data_dir
    initial = np.fromfile(directory / "y_ref.bin", dtype=np.float32)
    lengths = np.fromfile(directory / "group_lengths.bin", dtype=np.int64)
    if (
        not lengths.size
        or not initial.size
        or initial.size % lengths.size
        or np.any(initial == 0)
    ):
        raise SystemExit(
            "Invalid fixture: expected nonzero initial YRef in every group"
        )

    empty_groups = lengths == 0
    initial = initial.reshape(lengths.size, -1)
    for launch in (1, 2):
        golden = np.fromfile(directory / f"golden_{launch}.bin", dtype=np.float32)
        actual = np.fromfile(directory / f"npu_out_{launch}.bin", dtype=np.float32)
        if actual.size != initial.size or golden.size != initial.size:
            raise SystemExit(f"Launch {launch}: unexpected output size")
        actual = actual.reshape(initial.shape)
        golden = golden.reshape(initial.shape)
        if np.array_equal(golden[~empty_groups], initial[~empty_groups]):
            raise SystemExit(f"Launch {launch}: fixture must exercise a nonzero update")
        if not np.array_equal(actual[empty_groups], initial[empty_groups]):
            raise SystemExit(f"Launch {launch}: empty groups changed")
        if not np.array_equal(actual, golden):
            index = tuple(np.argwhere(actual != golden)[0])
            raise SystemExit(
                f"Launch {launch}: mismatch at {index}: expected {golden[index]}, got {actual[index]}"
            )
        print(
            f"PASS: launch {launch}, {actual.size} FP32 values and {np.count_nonzero(empty_groups)} empty groups"
        )


if __name__ == "__main__":
    main()
