"""Transparent conditional Gaussian MLE with analytic filter gradients.

The initial variance is the observed sample variance, fixed across candidate
parameters. All observations, including the first, contribute to likelihood.
Independent marginal likelihoods are fitted first; correlation is estimated
from aligned standardized residuals (not a joint multivariate MLE).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from ..estimation import estimate_gaussian
from .config import PublicSpec
from .models import MarketState, ObservedHistory, correlation_factor, readonly


def filter_variance(x, theta, initial):
    """theta=(mu, omega, alpha, beta, gamma); h[0] is training-only initialization."""
    mu, omega, alpha, beta, gamma = theta
    eps = np.asarray(x) - mu
    forcing = omega + (alpha + gamma * (eps[:-1] < 0)) * eps[:-1] ** 2
    h = np.r_[initial, lfilter([1.0], [1.0, -beta], forcing, zi=[beta * initial])[0]]
    return eps, h


def likelihood(theta, x, initial, *, gradient=False):
    """Gaussian negative log likelihood and exact parameter gradient."""
    eps, h = filter_variance(x, theta, initial)
    if np.any(h <= 0) or not np.isfinite(h).all():
        raise ValueError("invalid variance during likelihood evaluation")
    value = 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + eps**2 / h)
    if not gradient:
        return float(value)
    _, _, alpha, beta, gamma = theta
    # Differentiate the recurrence; h[0] is fixed, so its derivatives are zero.
    prev = eps[:-1]
    direct = np.column_stack(
        (
            -2 * (alpha + gamma * (prev < 0)) * prev,
            np.ones(len(prev)),
            prev**2,
            h[:-1],
            prev**2 * (prev < 0),
        )
    )
    dh = np.vstack((np.zeros(5), lfilter([1.0], [1.0, -beta], direct, axis=0)))
    grad = 0.5 * np.sum((1 / h - eps**2 / h**2)[:, None] * dh, axis=0)
    grad[0] -= np.sum(eps / h)
    return float(value), grad


@dataclass(frozen=True)
class MarginalFit:
    parameters: np.ndarray
    residuals: np.ndarray
    variances: np.ndarray
    diagnostics: dict


def fit_volatility_model(returns, spec: PublicSpec):
    """Fit only observations. No oracle parameters, states, seeds, or run config."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("at least 20 finite observations required for volatility fitting")
    if spec.family not in ("garch", "gjr"):
        raise ValueError("univariate volatility fitting requires garch or gjr")
    scale = float(x.std(ddof=1))
    if scale <= np.finfo(float).eps:
        raise ValueError("constant observations cannot identify volatility dynamics")
    y = x / scale
    initial = float(y.var(ddof=1))
    bounds = [
        (None, None),
        (1e-9, None),
        (0.0, 1.0),
        (0.0, 1.0),
        # The persistence restriction bounds gamma below 2, not below 1.
        (0.0, 2.0) if spec.family == "gjr" else (0.0, 0.0),
    ]
    constraint = {
        "type": "ineq",
        "fun": lambda p: 0.999999 - p[2] - p[3] - 0.5 * p[4],
        "jac": lambda p: np.array([0.0, 0.0, -1.0, -1.0, -0.5]),
    }
    attempts = []
    # Fixed starts are numerical options, independent of the true generating values.
    for a, b, g in ((0.05, 0.90, 0.04), (0.10, 0.70, 0.10), (0.20, 0.30, 0.02)):
        g = g if spec.family == "gjr" else 0.0
        start = np.array([y.mean(), initial * (1 - a - b - g / 2), a, b, g])
        result = minimize(
            lambda p: likelihood(p, y, initial, gradient=True),
            start,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": spec.fit_tolerance, "maxiter": spec.maxiter},
        )
        attempts.append(result)
    valid = [
        r for r in attempts if r.success and np.isfinite(r.fun) and constraint["fun"](r.x) >= -1e-10
    ]
    if not valid:
        raise RuntimeError("MLE failed: " + "; ".join(str(r.message) for r in attempts))
    best = min(valid, key=lambda r: r.fun)
    theta = best.x.copy()
    eps, h = filter_variance(y, theta, initial)
    # Scale means/shocks by s and variance quantities by s**2.
    theta[0] *= scale
    theta[1] *= scale**2
    p = theta[2] + theta[3] + theta[4] / 2
    diagnostics = {
        "success": True,
        "nll": float(best.fun + len(x) * np.log(scale)),
        "iterations": int(best.nit),
        "scale": scale,
        "initial_variance": initial * scale**2,
        "persistence": p,
        "boundary": bool(
            p > 0.999 or min(theta[2:5] if spec.family == "gjr" else theta[2:4]) < 1e-7
        ),
        "standard_errors": "not reported: boundary-aware inference deferred",
        "attempts": [
            {
                "success": bool(r.success),
                "nll_scaled": float(r.fun) if np.isfinite(r.fun) else None,
                "message": str(r.message),
            }
            for r in attempts
        ],
    }
    return MarginalFit(readonly(theta), readonly(eps * scale), readonly(h * scale**2), diagnostics)


@dataclass(frozen=True)
class FittedMarket:
    state: MarketState
    residuals: np.ndarray
    variances: np.ndarray
    parameters: pd.DataFrame
    diagnostics: tuple

    @property
    def standardized(self):
        return self.residuals / np.sqrt(self.variances)


def fit_market_model(history: ObservedHistory, spec: PublicSpec):
    if not isinstance(history, ObservedHistory) or not isinstance(spec, PublicSpec):
        raise TypeError("fitter accepts ObservedHistory and PublicSpec only")
    x = history.returns
    n, a = x.shape
    if spec.family == "gaussian":
        mu, covariance = estimate_gaussian(x)
        var = np.diag(covariance)
        if np.any(var <= 0):
            raise ValueError("positive Gaussian variance required")
        parameters = np.column_stack((mu, var, np.zeros((a, 3))))
        residuals = x - mu
        variances = np.broadcast_to(var, x.shape).copy()
        diagnostics = tuple({"success": True, "method": "sample moments"} for _ in range(a))
    else:
        fits = [fit_volatility_model(x[:, i], spec) for i in range(a)]
        parameters = np.array([f.parameters for f in fits])
        residuals = np.column_stack([f.residuals for f in fits])
        variances = np.column_stack([f.variances for f in fits])
        diagnostics = tuple(f.diagnostics for f in fits)
    standardized = residuals / np.sqrt(variances)
    corr = np.atleast_2d(np.corrcoef(standardized, rowvar=False)) if a > 1 else np.ones((1, 1))
    correlation_factor(corr)
    mu, omega, alpha, beta, gamma = parameters.T
    state = MarketState(
        spec.family,
        mu,
        omega,
        alpha,
        beta,
        gamma,
        corr,
        variances[-1],
        residuals[-1],
        history.assets,
    )
    table = pd.DataFrame(
        parameters, index=history.assets, columns=["mu", "omega", "alpha", "beta", "gamma"]
    )
    table["persistence"] = state.persistence
    table["long_run_variance"] = state.long_run_variance
    return FittedMarket(state, readonly(residuals), readonly(variances), table, diagnostics)
