"""Deterministic identities and clearly labelled descriptive statistical checks."""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .models import variance_step
from .simulation import aggregate_terminal_returns


def acf(values, lags=20):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) <= lags or not np.isfinite(x).all():
        raise ValueError("finite series longer than lag count required")
    y = x - x.mean()
    denominator = y @ y
    if denominator <= 0:
        raise ValueError("ACF undefined for a constant series")
    return np.array([1.0] + [y[k:] @ y[:-k] / denominator for k in range(1, lags + 1)])


def residual_diagnostics(history, fitted, lags=20):
    rows = []
    for i, asset in enumerate(history.assets):
        series = {
            "centered_returns": history.returns[:, i] - history.returns[:, i].mean(),
            "standardized_residuals": fitted.standardized[:, i],
        }
        for name, values in series.items():
            for squared in (False, True):
                correlations = acf(values**2 if squared else values, lags)
                n = len(values)
                statistic = (
                    n * (n + 2) * np.sum(correlations[1:] ** 2 / (n - np.arange(1, lags + 1)))
                )
                p = float(chi2.sf(statistic, lags))
                for lag, value in enumerate(correlations):
                    rows.append(
                        {
                            "asset": asset,
                            "series": name,
                            "squared": squared,
                            "lag": lag,
                            "acf": value,
                            "ljung_box": statistic,
                            "reference_p": p,
                            "interpretation": "descriptive chi-square reference; fitted-parameter adjustment not applied",
                            "multiple_testing": "no binary acceptance; all diagnostics reported",
                        }
                    )
    return pd.DataFrame(rows)


def mathematical_backstops():
    checks = {}
    checks["garch_scalar"] = bool(np.isclose(variance_step(0.2, 0.1, 0.01, 0.2, 0.5), 0.068))
    checks["constant_variance"] = bool(np.isclose(variance_step(100.0, 10.0, 0.04, 0.0, 0.0), 0.04))
    positive = variance_step(0.2, 0.1, 0.01, 0.2, 0.5, 0.1)
    negative = variance_step(-0.2, 0.1, 0.01, 0.2, 0.5, 0.1)
    checks["asymmetry"] = bool(np.isclose(negative - positive, 0.1 * 0.2**2))
    checks["gjr_nests_garch"] = bool(np.isclose(variance_step(-0.2, 0.1, 0.01, 0.2, 0.5, 0), 0.068))
    paths = np.array([[[0.1, -0.1], [-0.1, 0.1]]])
    terminal = aggregate_terminal_returns(paths)
    checks["buy_hold"] = bool(np.allclose(terminal, [[-0.01, -0.01]]))
    checks["not_daily_rebalanced"] = bool(not np.isclose((terminal @ [0.5, 0.5])[0], 0.0))
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def recurrence_report(private):
    s = private.endpoint
    expected = variance_step(
        private.shocks[:-1], private.variances[:-1], s.omega, s.alpha, s.beta, s.gamma
    )
    error = float(np.max(np.abs(expected - private.variances[1:])))
    if error > 1e-12:
        raise AssertionError("recurrence misalignment")
    return {
        "maximum_recurrence_error": error,
        "minimum_variance": float(private.variances.min()),
        "initialization": "long-run variance, no discarded prefix",
    }


def parameter_comparison(private, fitted):
    """Researcher-only truth access after allocation, except dedicated recovery fixtures."""
    truth = private.endpoint
    result = fitted.parameters.copy()
    for column, value in [
        ("mu", truth.mean),
        ("omega", truth.omega),
        ("alpha", truth.alpha),
        ("beta", truth.beta),
        ("gamma", truth.gamma),
        ("persistence", truth.persistence),
        ("long_run_variance", truth.long_run_variance),
    ]:
        result[f"true_{column}"] = value
        result[f"error_{column}"] = result[column] - value
    result["volatility_normalized_rmse"] = np.sqrt(
        np.mean((np.sqrt(fitted.variances) - np.sqrt(private.variances)) ** 2, axis=0)
    ) / np.sqrt(truth.long_run_variance)
    result["next_variance_error"] = fitted.state.next_variance - truth.next_variance
    return result
