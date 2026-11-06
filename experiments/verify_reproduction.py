"""Check artifact integrity and compare the scientific outputs from two completed runs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from thesis_portfolio.config import load_config
from thesis_portfolio.provenance import file_hash

RTOL = 1e-7
ATOL = 1e-9


def verify_integrity(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"Run is not complete: {run}")
    config = load_config(run / "resolved_config.json")
    archives = {f"replication_{i:04d}.npz" for i in range(config.replications)}
    expected = archives | {
        "resolved_config.json",
        "optimizer_diagnostics.json",
        "provenance.json",
        "seeds.json",
        "metrics.csv",
        "weights.csv",
        "summary.csv",
        "run.log",
        "source_snapshot.zip",
    }
    hashes = manifest["artifact_sha256"]
    if set(hashes) != expected:
        raise ValueError(f"Incomplete or unexpected artifact list in {run}")
    for name, checksum in hashes.items():
        if file_hash(run / name) != checksum:
            raise ValueError(f"SHA-256 mismatch: {run / name}")
    return config.to_dict()


def compare_table(left: Path, right: Path, filename: str, labels: tuple[str, ...]) -> None:
    def read(run):
        with (run / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = sorted(reader, key=lambda row: tuple(row[label] for label in labels))
            return reader.fieldnames, rows

    columns, left_rows = read(left)
    other_columns, right_rows = read(right)
    if columns != other_columns or len(left_rows) != len(right_rows):
        raise ValueError(f"{filename} has a different layout or row count")
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        for column in columns:
            x, y = left_row[column], right_row[column]
            if column in labels or "" in (x, y):
                if x != y:
                    raise ValueError(f"{filename} differs in {column}: {x!r} vs {y!r}")
            else:
                np.testing.assert_allclose(
                    float(x),
                    float(y),
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=False,
                    err_msg=f"{filename}: {column}",
                )


def verify_reproduction(left: Path, right: Path) -> None:
    if left.resolve() == right.resolve():
        raise ValueError("Supply two distinct run directories")
    first = verify_integrity(left)
    second = verify_integrity(right)
    if first != second:
        raise ValueError("Resolved configurations differ")
    for i in range(first["replications"]):
        name = f"replication_{i:04d}.npz"
        with (
            np.load(left / name, allow_pickle=False) as a,
            np.load(right / name, allow_pickle=False) as b,
        ):
            if set(a.files) != set(b.files):
                raise ValueError(f"Array names differ in {name}")
            for key in a.files:
                if a[key].shape != b[key].shape:
                    raise ValueError(f"Array shape differs in {name}/{key}")
                np.testing.assert_allclose(
                    a[key], b[key], rtol=RTOL, atol=ATOL, equal_nan=False, err_msg=f"{name}/{key}"
                )
    for name, labels in (
        ("metrics.csv", ("replication", "mode", "strategy")),
        ("weights.csv", ("replication", "mode", "strategy")),
        ("summary.csv", ("mode", "strategy", "metric", "replications")),
    ):
        compare_table(left, right, name, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    try:
        verify_reproduction(args.original, args.replay)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Verified artifact checksums and reproduced arrays, weights and metrics (rtol={RTOL}, atol={ATOL})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
