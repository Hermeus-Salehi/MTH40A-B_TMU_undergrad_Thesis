"""One complete Gaussian experiment with a strict historical-information boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_portfolio.config import ExperimentConfig
from thesis_portfolio.estimation import estimate_gaussian
from thesis_portfolio.optimization import (
    OptimizationResult,
    cvar,
    equal_risk_contribution,
    markowitz,
)
from thesis_portfolio.provenance import capture_provenance, file_hash, write_json
from thesis_portfolio.randomness import rng_for, seed_metadata
from thesis_portfolio.risk import portfolio_returns, summarize_returns
from thesis_portfolio.simulation import gaussian_returns


@dataclass(frozen=True)
class FittedPortfolios:
    mean: np.ndarray
    covariance: np.ndarray
    scenarios: np.ndarray
    portfolios: dict[str, dict[str, OptimizationResult]]
    controlled_target: float


def _require_valid_simple_returns(returns: np.ndarray, sample: str) -> None:
    if np.any(returns <= -1):
        raise ValueError(
            f"{sample} includes an asset return <= -100%; change the market assumption or scale"
        )


def fit_portfolios(
    history: np.ndarray,
    *,
    n_scenarios: int,
    confidence: float,
    risk_aversion: float,
    comparison_modes: tuple[str, ...],
    rng: np.random.Generator,
) -> FittedPortfolios:
    """Fit using history only; neither true market parameters nor evaluation data are inputs."""
    _require_valid_simple_returns(history, "Gaussian history")
    mean, covariance = estimate_gaussian(history)
    scenarios = gaussian_returns(mean, covariance, n_scenarios, rng)
    _require_valid_simple_returns(scenarios, "Fitted Gaussian optimization sample")
    erc = equal_risk_contribution(covariance)
    target = float(mean @ erc.weights)
    portfolios = {}
    for mode in comparison_modes:
        if mode not in ("controlled", "independent"):
            raise ValueError(f"Unsupported comparison mode: {mode}")
        kwargs = (
            {"target_return": target, "target_constraint": "equal"} if mode == "controlled" else {}
        )
        portfolios[mode] = {
            "markowitz": markowitz(mean, covariance, risk_aversion=risk_aversion, **kwargs),
            "erc": erc,
            "cvar": cvar(scenarios, confidence=confidence, mean=mean, **kwargs),
        }
    return FittedPortfolios(mean, covariance, scenarios, portfolios, target)


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty result table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> list[dict]:
    result = []
    metrics = [k for k in rows[0] if k not in ("replication", "mode", "strategy")]
    for mode, strategy in sorted({(r["mode"], r["strategy"]) for r in rows}):
        group = [r for r in rows if r["mode"] == mode and r["strategy"] == strategy]
        for metric in metrics:
            values = np.asarray([row[metric] for row in group])
            result.append(
                {
                    "mode": mode,
                    "strategy": strategy,
                    "metric": metric,
                    "replications": len(group),
                    "mean": float(values.mean()),
                    "monte_carlo_standard_error": (
                        float(values.std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else None
                    ),
                }
            )
    return result


def run_experiment(config: ExperimentConfig, output_root: str | Path = "results") -> Path:
    """Run independently seeded replications and preserve inputs, outputs and failure state."""
    # Revalidate direct dataclass construction too.
    config = ExperimentConfig.from_dict(config.to_dict())
    resolved = config.to_dict()
    config_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_root).resolve() / f"{config.name}-{timestamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger(f"thesis_portfolio.run.{output.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "config_sha256": config_hash,
        "started_utc": timestamp,
        "protocol": "fixed_target_daily_rebalance_zero_cost",
    }
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "provenance.json", capture_provenance(output))
        logger.info(
            "Starting %s with %s independent replications", config.name, config.replications
        )
        metrics_rows, weight_rows, diagnostics, seeds = [], [], {}, []
        for replication in range(config.replications):
            logger.info("Replication %s: simulate history and fit portfolios", replication)
            history = gaussian_returns(
                config.market.mean,
                config.market.covariance,
                config.history_observations,
                rng_for(config.seed, replication, "history"),
            )
            fitted = fit_portfolios(
                history,
                n_scenarios=config.optimization_scenarios,
                confidence=config.confidence,
                risk_aversion=config.risk_aversion,
                comparison_modes=config.comparison_modes,
                rng=rng_for(config.seed, replication, "optimization"),
            )
            # Evaluation is generated after allocation and shared across every strategy/mode.
            evaluation = gaussian_returns(
                config.market.mean,
                config.market.covariance,
                config.evaluation_observations,
                rng_for(config.seed, replication, "evaluation"),
            )
            _require_valid_simple_returns(evaluation, "Gaussian evaluation")
            arrays = {
                "history": history,
                "optimization_scenarios": fitted.scenarios,
                "evaluation": evaluation,
                "estimated_mean": fitted.mean,
                "estimated_covariance": fitted.covariance,
            }
            diagnostics[str(replication)] = {}
            for mode, portfolios in fitted.portfolios.items():
                diagnostics[str(replication)][mode] = {}
                for strategy, result in portfolios.items():
                    realized = portfolio_returns(evaluation, result.weights)
                    metrics = summarize_returns(
                        realized,
                        confidence=config.confidence,
                        loss_threshold=config.loss_threshold,
                        periods_per_year=config.periods_per_year,
                    )
                    metrics["estimated_return"] = float(fitted.mean @ result.weights)
                    metrics_rows.append(
                        {"replication": replication, "mode": mode, "strategy": strategy, **metrics}
                    )
                    weight_rows.append(
                        {
                            "replication": replication,
                            "mode": mode,
                            "strategy": strategy,
                            **dict(zip(config.market.asset_names, result.weights.tolist())),
                        }
                    )
                    arrays[f"{mode}_{strategy}_weights"] = result.weights
                    arrays[f"{mode}_{strategy}_returns"] = realized
                    diagnostics[str(replication)][mode][strategy] = {
                        "objective": result.objective,
                        "solver": result.diagnostics,
                        "return_target": fitted.controlled_target if mode == "controlled" else None,
                        "return_constraint": (
                            ("reference" if strategy == "erc" else "equality")
                            if mode == "controlled"
                            else None
                        ),
                    }
            np.savez_compressed(output / f"replication_{replication:04d}.npz", **arrays)
            seeds.extend(
                seed_metadata(config.seed, replication, s)
                for s in ("history", "optimization", "evaluation")
            )
        _csv(output / "metrics.csv", metrics_rows)
        _csv(output / "weights.csv", weight_rows)
        _csv(output / "summary.csv", _summary(metrics_rows))
        write_json(output / "optimizer_diagnostics.json", diagnostics)
        write_json(output / "seeds.json", seeds)
        logger.info(
            "Completed all replications; sample tail counts and replication counts remain provisional"
        )
        handler.flush()
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {
            p.name: file_hash(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        return output
    except Exception as error:
        logger.exception("Experiment failed; partial outputs must not be used as a completed run")
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
