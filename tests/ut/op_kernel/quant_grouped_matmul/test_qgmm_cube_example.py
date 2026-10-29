"""Check example precision policy and CSV data generation without an NPU."""

import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import numpy as np
import ml_dtypes

root = pathlib.Path(__file__).resolve().parents[4]
scripts = root / "examples/acl/grouped_matmul/scripts"
verify = scripts / "verify_result_cubeonly.py"
count = 0
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    (d / "output").mkdir()
    for name, dtype, size, tol in [
        ("fp16", np.float16, 1000, 1e-3),
        ("bf16", ml_dtypes.bfloat16, 1000, 1e-3),
        ("fp32", np.float32, 10000, 1e-4),
    ]:
        golden = np.zeros(size, dtype=dtype)
        golden.tofile(d / "golden")
        below = np.asarray(tol / 2, dtype=dtype)
        above = np.asarray(tol * 2, dtype=dtype)
        edge = np.asarray(tol, dtype=dtype)
        edge_below = (
            edge
            if float(edge) <= float(np.float32(tol))
            else np.nextafter(edge, np.asarray(0, dtype=dtype))
        )
        edge_above = np.nextafter(edge_below, np.asarray(np.inf, dtype=dtype))
        for value, errors, passed in [
            (edge_below, size, True),
            (edge_above, size, False),
            (below, size, True),
            (above, 1, True),
            (above, 2, False),
            (np.nan, 2, False),
            (np.inf, 2, False),
        ]:
            actual = golden.copy()
            actual[:errors] = value
            actual.tofile(d / "actual")
            result = subprocess.run(
                [
                    sys.executable,
                    str(verify),
                    str(d / "golden"),
                    str(d / "actual"),
                    "--groups",
                    "1",
                    "--m",
                    "1",
                    "--n",
                    str(size),
                    "--dtype",
                    name,
                ],
                cwd=d,
                capture_output=True,
                text=True,
            )
            assert (result.returncode == 0) == passed, result.stdout + result.stderr
            metrics = json.loads((d / "output/verify_metrics.json").read_text())
            assert ("fail" in json.dumps(metrics)) == (not passed)
            count += 1
        golden[:-1].tofile(d / "actual")
        result = subprocess.run(
            [
                sys.executable,
                str(verify),
                str(d / "golden"),
                str(d / "actual"),
                "--groups",
                "1",
                "--m",
                "1",
                "--n",
                str(size),
                "--dtype",
                name,
            ],
            cwd=d,
            capture_output=True,
        )
        assert result.returncode != 0
        count += 1
    csv_path = (
        root
        / "examples/acl/grouped_matmul/quant_grouped_matmul_cubeonly/quant_grouped_matmul_cubeonly.csv"
    )
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    mapping = {
        "group-num": "groupNum",
        "m": "m",
        "n": "n",
        "k": "k",
        "dtype": "dtype",
        "output-dtype": "cType",
        "layout-b": "layoutB",
        "x1-quant-mode": "x1QuantMode",
        "x2-quant-mode": "x2QuantMode",
        "is-bias": "isBias",
        "group-list-type": "groupListType",
        "single-w": "singleW",
        "group-list": "groupList",
    }
    for row in rows:
        output = d / row["caseName"]
        cmd = [
            sys.executable,
            str(scripts / "gen_data_cubeonly.py"),
            "--output-dir",
            str(output),
        ]
        for flag, key in mapping.items():
            cmd += ["--" + flag, row[key]]
        subprocess.run(cmd, check=True, capture_output=True)
        width = 4 if row["cType"] == "fp32" else 2
        assert (output / "golden_c.bin").stat().st_size == int(row["groupNum"]) * int(
            row["m"]
        ) * int(row["n"]) * width
        subprocess.run(
            [
                sys.executable,
                str(verify),
                str(output / "golden_c.bin"),
                str(output / "golden_c.bin"),
                "--groups",
                row["groupNum"],
                "--m",
                row["m"],
                "--n",
                row["n"],
                "--dtype",
                row["cType"],
            ],
            cwd=d,
            check=True,
            capture_output=True,
        )
print(f"{count} precision/shape cases and {len(rows)} CSV data pipelines passed")
