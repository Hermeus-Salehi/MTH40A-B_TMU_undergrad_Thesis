"""Reuse validated daily solvers on annual terminal scenarios and annual moments."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..optimization import cvar, equal_risk_contribution, markowitz, risk_contributions
from ..risk import empirical_var_cvar
from .models import readonly


def annual_moments(terminal):
    x = np.asarray(terminal)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite annual scenario matrix required")
    return x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False, ddof=1))


@dataclass(frozen=True)
class FrozenPortfolios:
    weights: object
    mean: np.ndarray
    covariance: np.ndarray
    target: float
    thresholds: object
    constraints: pd.DataFrame
    forecast_metrics: pd.DataFrame
    diagnostics: object


def freeze_portfolios(solutions, mean, covariance, terminal, confidences):
    """Validate ALL constraints and freeze allocations before oracle evaluation."""
    if set(solutions) != {"ERC", "Markowitz", "CVaR"}:
        raise ValueError("all three strategies required")
    target = float(mean @ solutions["ERC"].weights)
    rows, risk_rows, thresholds = [], [], {}
    weights = {}
    for strategy, solution in solutions.items():
        w = readonly(solution.weights)
        if np.min(w) < 0 or abs(w.sum() - 1) > 1e-8 or abs(mean @ w - target) > 1e-8:
            raise ValueError("portfolio constraint failure")
        shares = risk_contributions(w, covariance) / np.sqrt(w @ covariance @ w)
        if strategy == "ERC" and np.max(np.abs(shares - 1 / len(w))) > 1e-8:
            raise ValueError("exact ERC failed")
        rows.append(
            {
                "strategy": strategy,
                "fitted_mean": float(mean @ w),
                "target": target,
                "return_residual": float(mean @ w - target),
                "budget_residual": float(w.sum() - 1),
                "risk_share_error": float(np.max(np.abs(shares - 1 / len(w)))),
            }
        )
        weights[strategy] = w
        for q in confidences:
            var, tail = empirical_var_cvar(-terminal @ w, q)
            thresholds[(strategy, q)] = var
            risk_rows.append(
                {
                    "strategy": strategy,
                    "confidence": q,
                    "mean": float(mean @ w),
                    "volatility": float(np.sqrt(w @ covariance @ w)),
                    "var": var,
                    "cvar": tail,
                }
            )
    ew, mw, cw = [weights[k] for k in ("ERC", "Markowitz", "CVaR")]
    if mw @ covariance @ mw > ew @ covariance @ ew + 1e-8:
        raise ValueError("minimum-variance objective exceeds feasible ERC")
    optimization_q = solutions["CVaR"].diagnostics["confidence"]
    if (
        empirical_var_cvar(-terminal @ cw, optimization_q)[1]
        > empirical_var_cvar(-terminal @ ew, optimization_q)[1] + 1e-8
    ):
        raise ValueError("minimum CVaR exceeds feasible ERC")
    return FrozenPortfolios(
        MappingProxyType(weights),
        readonly(mean),
        readonly(covariance),
        target,
        MappingProxyType(thresholds),
        pd.DataFrame(rows),
        pd.DataFrame(risk_rows),
        MappingProxyType({s: r.diagnostics for s, r in solutions.items()}),
    )


def optimize_portfolios(terminal, confidence=0.95, reporting=(0.95, 0.99)):
    mu, cov = annual_moments(terminal)
    erc = equal_risk_contribution(cov)
    target = float(mu @ erc.weights)
    mv = markowitz(mu, cov, target_return=target, target_constraint="equal")
    tail = cvar(
        terminal, confidence=confidence, mean=mu, target_return=target, target_constraint="equal"
    )
    return freeze_portfolios(
        {"ERC": erc, "Markowitz": mv, "CVaR": tail}, mu, cov, terminal, reporting
    )


def evaluate_portfolios(frozen, terminal):
    """Each row is an independent conditional year, NOT a successive trading day."""
    x = np.asarray(terminal)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] != len(frozen.mean) or not np.isfinite(x).all():
        raise ValueError("invalid evaluation terminal scenarios")
    rows = []
    for (strategy, q), threshold in frozen.thresholds.items():
        w = frozen.weights[strategy]
        returns = x @ w
        var, tail = empirical_var_cvar(-returns, q)
        exceed = int(np.count_nonzero(-returns > threshold))
        interval = binomtest(exceed, len(x)).proportion_ci(confidence_level=0.95)
        forecast = frozen.forecast_metrics.query("strategy == @strategy and confidence == @q").iloc[
            0
        ]
        rows.append(
            {
                "strategy": strategy,
                "confidence": q,
                "fitted_target": frozen.target,
                "oracle_mean": float(returns.mean()),
                "oracle_volatility": float(returns.std(ddof=1)),
                "oracle_var": var,
                "oracle_cvar": tail,
                "forecast_var": threshold,
                "forecast_cvar": forecast["cvar"],
                "var_error": threshold - var,
                "cvar_error": forecast["cvar"] - tail,
                "exceedances": exceed,
                "exceedance_frequency": exceed / len(x),
                "nominal_tail": 1 - q,
                "exception_ci_low": interval.low,
                "exception_ci_high": interval.high,
                "tail_mass": len(x) * (1 - q),
                "concentration": float(w @ w),
                "paths": len(x),
                "uncertainty_scope": "conditional on this history and these frozen weights",
            }
        )
    return pd.DataFrame(rows)


def frontier(mean, covariance, points=16):
    """Diagnostic fixed-return curve; never used to replace the ERC target."""
    targets = np.linspace(min(mean), max(mean), points)
    rows = []
    for target in targets:
        result = markowitz(mean, covariance, target_return=float(target), target_constraint="equal")
        rows.append({"mean": target, "volatility": np.sqrt(result.objective)})
    # A zero mean makes the no-target solver exactly the GMV problem.
    gmv = markowitz(np.zeros_like(mean), covariance)
    return pd.DataFrame(rows), {
        "gmv_mean": float(mean @ gmv.weights),
        "gmv_volatility": float(np.sqrt(gmv.weights @ covariance @ gmv.weights)),
    }
