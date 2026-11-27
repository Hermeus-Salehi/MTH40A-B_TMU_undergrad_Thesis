"""Execute independent recipes with a fresh local project kernel and save evidence."""

import argparse
import json
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    args = parser.parse_args()
    os.environ.update(
        THESIS_PROFILE=args.profile,
        THESIS_PROJECT_ROOT=str(root),
        MPLCONFIGDIR=str(root / "tmp/mpl"),
        IPYTHONDIR=str(root / "tmp/ipython"),
        JUPYTER_PATH=str(root / ".venv/share/jupyter"),
        JUPYTER_RUNTIME_DIR=str(root / "tmp/jupyter"),
    )
    names = args.notebooks or [
        "notebooks/foundations/01_m5_mathematical_checks.ipynb",
        "notebooks/experiments/01_parameter_recovery.ipynb",
        "notebooks/modelling/01_iid_gaussian.ipynb",
        "notebooks/modelling/02_garch_gaussian.ipynb",
        "notebooks/modelling/03_gjr_gaussian.ipynb",
        "notebooks/experiments/02_monte_carlo_convergence.ipynb",
        "notebooks/experiments/03_market_stress.ipynb",
        "notebooks/experiments/04_three_regime_comparison.ipynb",
    ]
    report_path = (
        root / "documentation/implementation_tasks" / f"notebook_execution_{args.profile}.json"
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else []
    report = [row for row in report if row["notebook"] not in names]
    for name in names:
        path = root / name
        notebook = nbformat.read(path, as_version=4)
        start = time.monotonic()
        record = {"notebook": name, "profile": args.profile, "status": "running"}
        print(f"Executing {name}", flush=True)
        try:
            NotebookClient(
                notebook,
                timeout=900,
                kernel_name="thesis",
                resources={"metadata": {"path": str(root)}},
            ).execute()
            record.update(
                status="passed", code_cells=sum(c.cell_type == "code" for c in notebook.cells)
            )
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            nbformat.write(notebook, path)
            report.append(record)
            dest = (
                root
                / "documentation/implementation_tasks"
                / f"notebook_execution_{args.profile}.json"
            )
            dest.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
