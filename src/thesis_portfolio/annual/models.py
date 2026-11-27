"""Explicit market states and the common GARCH/GJR mathematical machinery."""

from dataclasses import dataclass

import numpy as np


def readonly(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def correlation_factor(correlation):
    """Validate and factor a positive-definite correlation; never repair silently."""
    r = np.asarray(correlation, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.isfinite(r).all():
        raise ValueError("correlation must be a finite square matrix")
    if not np.allclose(r, r.T, atol=1e-12, rtol=0) or not np.allclose(np.diag(r), 1):
        raise ValueError("correlation must be symmetric with unit diagonal")
    if np.any(np.abs(r) > 1 + 1e-12):
        raise ValueError("correlation outside [-1, 1]")
    return np.linalg.cholesky(r)


def variance_step(previous_shock, previous_variance, omega, alpha, beta, gamma=0):
    """h[t+1] from epsilon[t] and h[t]; gamma=0 is GARCH."""
    shock, h = np.asarray(previous_shock), np.asarray(previous_variance)
    # The indicator uses a centered shock, never an uncentered return.
    squared = shock**2
    result = omega + alpha * squared + beta * h + gamma * squared * (shock < 0)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("nonpositive or nonfinite next variance")
    return result


@dataclass(frozen=True)
class MarketState:
    """One explicit parameter/state record, with no RNG or data access."""

    family: str
    mean: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    correlation: np.ndarray
    variance: np.ndarray
    shock: np.ndarray
    assets: tuple[str, ...]

    def __post_init__(self):
        if self.family not in ("gaussian", "garch", "gjr"):
            raise ValueError("unsupported marginal family")
        n = len(self.assets)
        if not n or len(set(self.assets)) != n:
            raise ValueError("asset identifiers must be unique and nonempty")
        for name in ("mean", "omega", "alpha", "beta", "gamma", "variance", "shock"):
            a = readonly(getattr(self, name))
            if a.shape != (n,) or not np.isfinite(a).all():
                raise ValueError(f"invalid {name} vector")
            object.__setattr__(self, name, a)
        if np.any(self.omega <= 0) or np.any(self.variance <= 0):
            raise ValueError("positive intercept and state variance required")
        if np.any(self.alpha < 0) or np.any(self.beta < 0) or np.any(self.alpha + self.gamma < 0):
            raise ValueError("invalid variance coefficients")
        if np.any(self.persistence >= 1):
            raise ValueError("finite-variance stationarity required")
        if self.family in ("garch", "gaussian") and np.any(self.gamma != 0):
            raise ValueError("gamma must be zero outside GJR")
        if self.family == "gaussian" and np.any(self.alpha + self.beta != 0):
            raise ValueError("Gaussian variances must be constant")
        r = readonly(self.correlation)
        if r.shape != (n, n):
            raise ValueError("correlation asset dimension mismatch")
        correlation_factor(r)
        object.__setattr__(self, "correlation", r)

    @property
    def persistence(self):
        return self.alpha + self.beta + self.gamma / 2

    @property
    def long_run_variance(self):
        return self.omega / (1 - self.persistence)

    @property
    def next_variance(self):
        return variance_step(
            self.shock, self.variance, self.omega, self.alpha, self.beta, self.gamma
        )


@dataclass(frozen=True)
class ObservedHistory:
    """Only this observation-only value object is passed to the estimator."""

    returns: np.ndarray
    assets: tuple[str, ...]

    def __post_init__(self):
        x = readonly(self.returns)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != len(self.assets):
            raise ValueError("history must have shape (N >= 2, A)")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite observations")
        object.__setattr__(self, "returns", x)


@dataclass(frozen=True)
class PrivateHistory:
    observed: ObservedHistory
    variances: np.ndarray
    shocks: np.ndarray
    innovations: np.ndarray
    endpoint: MarketState
