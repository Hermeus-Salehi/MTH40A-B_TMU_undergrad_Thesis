"""Chronological histories and conditional futures with annual aggregation."""

from dataclasses import dataclass, replace

import numpy as np

from .models import (
    MarketState,
    ObservedHistory,
    PrivateHistory,
    correlation_factor,
    readonly,
    variance_step,
)


def validate_simple_returns(returns):
    if not np.isfinite(returns).all() or np.any(returns <= -1):
        raise ValueError(
            "invalid decimal simple return: nonfinite or <= -1; no clipping/resampling"
        )


def generate_history(state: MarketState, n, rng, *, investable=True):
    """First observation uses h_bar; all N rows are retained. Source fixtures may
    use original model units with investable=False, and are never aggregated."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("history length must be an integer >= 2")
    z = rng.standard_normal((n, len(state.assets))) @ correlation_factor(state.correlation).T
    h, eps = np.empty_like(z), np.empty_like(z)
    h[0] = state.variance
    for t in range(n):
        if t:
            h[t] = variance_step(
                eps[t - 1], h[t - 1], state.omega, state.alpha, state.beta, state.gamma
            )
        eps[t] = np.sqrt(h[t]) * z[t]
    returns = state.mean + eps
    if investable:
        validate_simple_returns(returns)
    observed = ObservedHistory(returns, state.assets)
    endpoint = replace(state, variance=h[-1], shock=eps[-1])
    return PrivateHistory(observed, readonly(h), readonly(eps), readonly(z), endpoint)


def aggregate_terminal_returns(paths):
    """Compound EACH ASSET first; multiplying terminal returns by initial weights
    then implements buy-and-hold, not daily restoration of target weights."""
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("paths must have shape (scenario, day, asset)")
    validate_simple_returns(x)
    with np.errstate(over="raise", invalid="raise"):
        terminal = np.expm1(np.log1p(x).sum(axis=1))
    if not np.isfinite(terminal).all():
        raise ValueError("nonfinite terminal return")
    return terminal


@dataclass(frozen=True)
class FutureSample:
    terminal: np.ndarray
    retained_paths: np.ndarray
    first_variance: np.ndarray
    horizon: int


def simulate_conditional_paths(state, horizon, count, rng, *, batch_size=256, retain=32):
    """Fresh futures share a fixed endpoint. Scenario-major draws preserve nested
    prefixes and batch-size invariance without storing every full trajectory."""
    for x in (horizon, count, batch_size):
        if isinstance(x, bool) or not isinstance(x, int) or x < 1:
            raise ValueError("positive integer horizon, count, batch size required")
    if not isinstance(retain, int) or retain < 0:
        raise ValueError("retain must be nonnegative")
    n = len(state.assets)
    factor = correlation_factor(state.correlation)
    first = state.next_variance
    terminal = np.empty((count, n))
    kept = np.empty((min(count, retain), horizon, n))
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        z = rng.standard_normal((size, horizon, n)) @ factor.T
        h = np.broadcast_to(first, (size, n)).copy()
        paths = np.empty_like(z)
        for t in range(horizon):
            eps = np.sqrt(h) * z[:, t]
            paths[:, t] = state.mean + eps
            if t + 1 < horizon:
                h = variance_step(eps, h, state.omega, state.alpha, state.beta, state.gamma)
        terminal[start : start + size] = aggregate_terminal_returns(paths)
        k = max(0, min(size, len(kept) - start))
        if k:
            kept[start : start + k] = paths[:k]
    return FutureSample(readonly(terminal), readonly(kept), readonly(first), horizon)


def oracle_continue(private_history, horizon, count, rng, **kwargs):
    """Privileged evaluator entry point; called only after allocation is frozen."""
    if not isinstance(private_history, PrivateHistory):
        raise TypeError("oracle evaluator requires private history")
    return simulate_conditional_paths(private_history.endpoint, horizon, count, rng, **kwargs)
