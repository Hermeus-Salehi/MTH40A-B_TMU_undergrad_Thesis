"""Recovery, numerical convergence and stress studies using shared components."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PublicSpec, oracle_state, stream
from .diagnostics import parameter_comparison
from .estimation import fit_market_model
from .experiment import run_experiment, write_json
from .models import MarketState
from .portfolio import evaluate_portfolios, optimize_portfolios
from .simulation import generate_history, oracle_continue, simulate_conditional_paths


def source_fixture(family):
    """Source-exact model-unit fixtures; never used for investable wealth paths."""
    if family == "garch":
        omega, alpha, beta, gamma = 0.0001, 0.2, 0.5, 0.0
    elif family == "gjr":
        omega, alpha, beta, gamma = 0.02, 0.03, 0.9, 0.06
    else:
        raise ValueError("source fixture requires garch or gjr")
    h = omega / (1 - alpha - beta - gamma / 2)
    return MarketState(
        family, [0.0], [omega], [alpha], [beta], [gamma], [[1.0]], [h], [0.0], ("Source_asset",)
    )


def recovery_study(lengths=(300, 600, 1000, 1200), seeds=(20261127,), spec_options=None):
    """Nested prefixes per fixed seed. Failures are rows, not rejected draws."""
    from dataclasses import replace

    from .models import ObservedHistory, PrivateHistory

    rows = []
    for family in ("garch", "gjr"):
        state = source_fixture(family)
        for seed in seeds:
            full = generate_history(
                state, max(lengths), stream(seed, f"calibration/{family}"), investable=False
            )
            for n in lengths:
                row = {
                    "family": family,
                    "seed": seed,
                    "observations": n,
                    "success": False,
                    "units": "source-exact model units, not investable decimal returns",
                }
                observed = ObservedHistory(full.observed.returns[:n], state.assets)
                endpoint = replace(state, variance=full.variances[n - 1], shock=full.shocks[n - 1])
                private = PrivateHistory(
                    observed, full.variances[:n], full.shocks[:n], full.innovations[:n], endpoint
                )
                try:
                    fit = fit_market_model(observed, PublicSpec(family, **(spec_options or {})))
                    row.update(parameter_comparison(private, fit).iloc[0].to_dict())
                    row.update(
                        success=True,
                        boundary=fit.diagnostics[0]["boundary"],
                        standard_errors=fit.diagnostics[0]["standard_errors"],
                    )
                except (ValueError, RuntimeError) as exc:
                    row["failure"] = str(exc)
                rows.append(row)
    return pd.DataFrame(rows)


def history_selection(table, reference=1000):
    """No invented scientific threshold. Integration may use the recorded fallback."""
    return {
        "selected_n": reference,
        "status": "provisional_reference_not_calibrated",
        "reason": "Recovery adequacy criteria require researcher review; no automatic success claim.",
        "fits": len(table),
        "failed_fits": int((~table.success).sum()),
    }


def convergence_study(
    cfg, family="garch", forecast_counts=(500, 1000), evaluation_counts=(1000, 2000)
):
    """Forecast counts first, then fixed-weight evaluation counts; no oracle tuning."""
    private = generate_history(
        oracle_state(cfg, family),
        cfg["history"]["observations"],
        stream(cfg["seed"], f"{family}/history"),
    )
    fit = fit_market_model(private.observed, PublicSpec(family, **cfg["fit"]))
    forecast = simulate_conditional_paths(
        fit.state,
        252,
        max(forecast_counts),
        stream(cfg["seed"], f"{family}/forecast/{family}"),
        batch_size=cfg["simulation"]["batch_size"],
        retain=0,
    )
    allocations, rows = {}, []
    for count in forecast_counts:
        p = optimize_portfolios(
            forecast.terminal[:count],
            cfg["risk"]["optimization_confidence"],
            tuple(cfg["risk"]["reporting"]),
        )
        allocations[count] = p
        for strategy, w in p.weights.items():
            rows.append(
                {
                    "phase": "forecast",
                    "count": count,
                    "strategy": strategy,
                    "target": p.target,
                    **{f"weight_{i}": v for i, v in enumerate(w)},
                    "status": "numerical comparison, not certified adequacy",
                }
            )
    # The largest forecast count is prespecified, not selected using evaluation outcomes.
    selected = allocations[max(forecast_counts)]
    evaluation = oracle_continue(
        private,
        252,
        max(evaluation_counts),
        stream(cfg["seed"], f"{family}/evaluation"),
        batch_size=cfg["simulation"]["batch_size"],
        retain=0,
    )
    evaluation_tables = []
    for count in evaluation_counts:
        evaluation_tables.append(
            evaluate_portfolios(selected, evaluation.terminal[:count]).assign(
                count=count, phase="evaluation"
            )
        )
    return pd.DataFrame(rows), pd.concat(evaluation_tables, ignore_index=True)


def study_directory(root, name):
    path = (
        Path(root)
        / "results"
        / f"m05-{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    )
    path.mkdir(parents=True)
    return path


def stress_study(root, cfg, families=("garch", "gjr")):
    records = []
    for family in families:
        for stress in ("baseline", "volatility", "correlation", "combined"):
            path, _ = run_experiment(root, cfg, family, stress=stress, make_plots=False)
            frame = pd.read_csv(path / "evaluation/metrics.csv")
            records.append(frame.assign(oracle=family, stress=stress, run_id=path.name))
    return pd.concat(records, ignore_index=True)


def save_study(path, tables, metadata):
    for name, frame in tables.items():
        frame.to_csv(path / f"{name}.csv", index=False)
    write_json(
        path / "manifest.json",
        {"status": "completed", "research_status": "provisional", **metadata},
    )
