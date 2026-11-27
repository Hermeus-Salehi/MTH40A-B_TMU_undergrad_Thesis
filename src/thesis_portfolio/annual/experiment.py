"""Orchestration uses the same explicit functions demonstrated by the notebooks."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from . import visualization as plots
from .config import PublicSpec, load_run_config, oracle_state, project_root, stream
from .diagnostics import (
    mathematical_backstops,
    parameter_comparison,
    recurrence_report,
    residual_diagnostics,
)
from .estimation import fit_market_model
from .portfolio import evaluate_portfolios, optimize_portfolios
from .simulation import generate_history, oracle_continue, simulate_conditional_paths


def execution_profile(config, profile):
    cfg = deepcopy(config)
    if profile == "smoke":
        cfg["status"] = "smoke"
        cfg["history"]["observations"] = 300
        cfg["history"]["selection_status"] = "smoke_only_not_calibrated"
        cfg["simulation"].update(optimization_paths=500, evaluation_paths=1000, retained_paths=32)
    elif profile != "pilot":
        raise ValueError("profile must be smoke or pilot")
    return cfg


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_value(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "items"):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + "\n")


def begin_run(root, cfg, oracle_family, stress="baseline"):
    root = Path(root)
    run_id = f"m05-{oracle_family}-{stress}-{cfg['status']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    path = root / "results" / run_id
    for sub in ["fit", "portfolios", "evaluation", "validation", "privileged", "scenarios"]:
        (path / sub).mkdir(parents=True, exist_ok=False)
    (root / "figures" / run_id).mkdir(parents=True)
    write_json(
        path / "manifest.json",
        {
            "run_id": run_id,
            "status": "incomplete",
            "oracle": oracle_family,
            "stress": stress,
            "profile": cfg["status"],
        },
    )
    (path / "privileged/resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    public = {k: v for k, v in cfg.items() if k not in ("oracle", "seed")}
    (path / "resolved_public_config.yaml").write_text(yaml.safe_dump(public, sort_keys=False))
    return path


def save_run(
    root, path, cfg, private, fitted_models, forecasts, portfolios, evaluation, tables, figures
):
    """Save observed evidence, separately labelled truth, checksums and actual statuses."""
    path, root = Path(path), Path(root)
    manifest = json.loads((path / "manifest.json").read_text())
    np.savez_compressed(
        path / "privileged/history.npz",
        returns=private.observed.returns,
        variance=private.variances,
        shocks=private.shocks,
        innovations=private.innovations,
    )
    np.savez_compressed(path / "scenarios/oracle_terminal.npz", terminal=evaluation.terminal)
    validation = {
        "mathematical_backstops": mathematical_backstops(),
        "history": recurrence_report(private),
        "research_status": cfg["history"]["selection_status"],
        "limitations": [
            "History adequacy thresholds not approved; no claim of calibration success.",
            "Conditional evaluation uncertainty excludes estimation uncertainty.",
            "Model-specific ERC targets may differ across forecasting families.",
            "Source scaling and stress values are provisional.",
            "MLE standard errors not reported; finite-history boundary estimates are flagged.",
        ],
    }
    constraints, weights, metrics = [], [], []
    for family, fit in fitted_models.items():
        fit.parameters.to_csv(path / f"fit/{family}_parameters.csv", index_label="asset")
        write_json(path / f"fit/{family}_optimizer_diagnostics.json", fit.diagnostics)
        residual_diagnostics(private.observed, fit).to_csv(
            path / f"fit/{family}_residual_diagnostics.csv", index=False
        )
        parameter_comparison(private, fit).to_csv(
            path / f"privileged/{family}_truth_comparison.csv", index_label="asset"
        )
        np.savez_compressed(
            path / f"fit/{family}_state.npz",
            residuals=fit.residuals,
            variances=fit.variances,
            correlation=fit.state.correlation,
            next_variance=fit.state.next_variance,
        )
        np.savez_compressed(
            path / f"scenarios/{family}_forecast_terminal.npz", terminal=forecasts[family].terminal
        )
        p = portfolios[family]
        constraints.append(p.constraints.assign(forecast_family=family))
        weights.append(
            pd.DataFrame(dict(p.weights), index=private.observed.assets)
            .rename_axis("asset")
            .reset_index()
            .assign(forecast_family=family)
        )
        p.forecast_metrics.to_csv(
            path / f"evaluation/{family}_forecast_thresholds.csv", index=False
        )
        write_json(path / f"portfolios/{family}_solvers.json", p.diagnostics)
        metrics.append(tables[family].assign(forecast_family=family))
        validation[family] = {
            "return_error": float(abs(p.constraints.return_residual).max()),
            "forecast_sha256": digest(forecasts[family].terminal),
            "minimum_forecast_eigenvalue": float(np.linalg.eigvalsh(p.covariance).min()),
            "fit_boundary_flags": [d.get("boundary", False) for d in fit.diagnostics],
        }
    pd.concat(constraints).to_csv(path / "portfolios/constraints.csv", index=False)
    pd.concat(weights).to_csv(path / "portfolios/weights.csv", index=False)
    pd.concat(metrics).to_csv(path / "evaluation/metrics.csv", index=False)
    write_json(path / "validation/report.json", validation)
    figure_index = []
    for name, fig in figures.items():
        target = root / "figures" / path.name / f"{name}.png"
        fig.savefig(target, dpi=150, bbox_inches="tight")
        figure_index.append(str(target.relative_to(root)))
    task_map = root / "documentation/implementation_tasks/m5_task_map.csv"
    if task_map.exists():
        shutil.copy2(task_map, path / "validation/task_map.csv")
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "src/thesis_portfolio").rglob("*.py")
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest.update(
        status="completed",
        scientific_status="provisional; calibration/count adequacy not certified",
        horizon=252,
        policy="buy_and_hold",
        target_policy="erc_fixed_return",
        history_count=len(private.observed.returns),
        optimization_paths=cfg["simulation"]["optimization_paths"],
        evaluation_paths=len(evaluation.terminal),
        forecast_families=list(fitted_models),
        observed_sha256=digest(private.observed.returns),
        evaluation_sha256=digest(evaluation.terminal),
        figures=figure_index,
        git_revision=revision,
        source_hashes=source_hashes,
        environment={
            "python": platform.python_version(),
            **{
                n: importlib.metadata.version(n) for n in ["numpy", "scipy", "pandas", "matplotlib"]
            },
        },
        seed_policy="SHA256 named child streams; full seed/config only in privileged artifact",
    )
    manifest["artifact_sha256"] = {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def make_figures(private, fits, forecasts, portfolios, evaluation):
    figures = {}
    for family, fit in fits.items():
        figures[f"{family}_history"] = plots.history_figure(private.observed, fit, private)
        figures[f"{family}_acf"] = plots.acf_figure(
            residual_diagnostics(private.observed, fit), private.observed.assets[0]
        )
        figures[f"{family}_future_fan"] = plots.fan_figure(forecasts[family], family)
        figures[f"{family}_portfolios"] = plots.portfolio_figure(
            portfolios[family], private.observed.assets
        )
        figures[f"{family}_losses"] = plots.loss_figure(
            portfolios[family], forecasts[family], evaluation, family
        )
        if family == "gjr":
            figures["gjr_news_impact"] = plots.news_impact_figure(fit.state)
    return figures


def run_experiment(root, cfg, oracle_family, *, stress="baseline", make_plots=True):
    """Batch runner. Learning notebooks call these component functions explicitly."""
    if stress not in ("baseline", "volatility", "correlation", "combined"):
        raise ValueError("unsupported stress")
    path = begin_run(root, cfg, oracle_family, stress)
    try:
        state = oracle_state(
            cfg,
            oracle_family,
            volatility_multiplier=2 if stress in ("volatility", "combined") else 1,
            correlation=0.70 if stress in ("correlation", "combined") else None,
        )
        # Matching stream names across stress cases intentionally pair exogenous shocks.
        private = generate_history(
            state, cfg["history"]["observations"], stream(cfg["seed"], f"{oracle_family}/history")
        )
        families = [oracle_family] if oracle_family == "gaussian" else [oracle_family, "gaussian"]
        fits, forecasts, portfolios = {}, {}, {}
        sim = cfg["simulation"]
        for family in families:
            fits[family] = fit_market_model(private.observed, PublicSpec(family, **cfg["fit"]))
            forecasts[family] = simulate_conditional_paths(
                fits[family].state,
                252,
                sim["optimization_paths"],
                stream(cfg["seed"], f"{oracle_family}/forecast/{family}"),
                batch_size=sim["batch_size"],
                retain=sim["retained_paths"],
            )
            portfolios[family] = optimize_portfolios(
                forecasts[family].terminal,
                cfg["risk"]["optimization_confidence"],
                tuple(cfg["risk"]["reporting"]),
            )
        # Every model has frozen allocations before any evaluation scenarios are generated.
        evaluation = oracle_continue(
            private,
            252,
            sim["evaluation_paths"],
            stream(cfg["seed"], f"{oracle_family}/evaluation"),
            batch_size=sim["batch_size"],
            retain=sim["retained_paths"],
        )
        tables = {f: evaluate_portfolios(p, evaluation.terminal) for f, p in portfolios.items()}
        figures = (
            make_figures(private, fits, forecasts, portfolios, evaluation) if make_plots else {}
        )
        manifest = save_run(
            root, path, cfg, private, fits, forecasts, portfolios, evaluation, tables, figures
        )
        for fig in figures.values():
            plt.close(fig)
        return path, manifest
    except Exception as exc:
        manifest = json.loads((path / "manifest.json").read_text())
        manifest.update(status="failed", error=str(exc))
        write_json(path / "manifest.json", manifest)
        (path / "validation/failure.txt").write_text(traceback.format_exc())
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--oracle", choices=["gaussian", "garch", "gjr"], default="garch")
    parser.add_argument(
        "--stress",
        choices=["baseline", "volatility", "correlation", "combined"],
        default="baseline",
    )
    args = parser.parse_args()
    root = project_root(args.root)
    cfg = execution_profile(load_run_config(root / "configs/annual/pilot.yaml"), args.profile)
    path, _ = run_experiment(root, cfg, args.oracle, stress=args.stress)
    print(path)


if __name__ == "__main__":
    main()
