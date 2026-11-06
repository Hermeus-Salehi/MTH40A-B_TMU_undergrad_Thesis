"""Risk metrics for decimal simple returns and losses equal to minus returns."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, name: str, minimum: int = 1) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be a finite one-dimensional array with at least {minimum} entries"
        )
    return array


def empirical_var_cvar(losses: ArrayLike, confidence: float) -> tuple[float, float]:
    """Return empirical loss VaR and CVaR at confidence ``0 < confidence < 1``.

    VaR is the left generalized inverse of the empirical CDF. CVaR uses the
    Rockafellar-Uryasev expression ``VaR + mean(max(loss - VaR, 0))/(1-a)``.
    Thus a fractional observation at the tail boundary receives the proper
    mass; averaging every observation greater than or equal to VaR is NOT used.
    Values may be negative when the portfolio makes gains in every scenario.
    """
    values = _vector(losses, "losses")
    if isinstance(confidence, bool) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be a finite number strictly between zero and one")
    var = float(np.quantile(values, confidence, method="inverted_cdf"))
    cvar = var + float(np.maximum(values - var, 0.0).mean()) / (1.0 - confidence)
    if not np.isfinite(cvar):
        raise ValueError("CVaR overflowed; check the scale of losses")
    return var, cvar


def portfolio_returns(asset_returns: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
    """Return period returns for long-only, fully invested constant target weights.

    ``asset_returns @ weights`` means resetting the target at each period's
    start, with zero transaction costs. It does not represent buy-and-hold
    weights drifting with asset prices.
    """
    returns = np.asarray(asset_returns, dtype=float)
    allocation = _vector(weights, "weights")
    if returns.ndim != 2 or returns.shape[0] < 1 or returns.shape[1] != allocation.size:
        raise ValueError("asset_returns must have shape (observations, number of weights)")
    if not np.all(np.isfinite(returns)):
        raise ValueError("asset_returns must be finite")
    if np.any(allocation < 0) or not np.isclose(allocation.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("weights must be nonnegative and sum to one")
    result = returns @ allocation
    if not np.all(np.isfinite(result)):
        raise ValueError("portfolio returns overflowed")
    return result


def summarize_returns(
    returns: ArrayLike,
    *,
    confidence: float = 0.95,
    loss_threshold: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Summarize an ordered return path; all returns/risks are decimal units.

    ``mean_return``, sample ``volatility``, ``var``, ``cvar`` and the loss
    threshold refer to one period. ``loss_frequency`` counts loss strictly
    greater than the threshold. Annualized mean is the arithmetic mean times
    periods/year, not a compounded forecast; volatility uses square-root-time
    scaling, an iid convention suitable for the Gaussian baseline. Drawdown
    includes initial wealth 1. Metrics require at least two periods and reject
    returns <= -1 because positive-wealth compounding would cease to apply.
    """
    values = _vector(returns, "returns", minimum=2)
    if np.any(values <= -1):
        raise ValueError(
            "wealth metrics require returns greater than -1; Gaussian simple returns are unbounded"
        )
    if isinstance(loss_threshold, bool) or not np.isfinite(loss_threshold) or loss_threshold < 0:
        raise ValueError("loss_threshold must be a finite nonnegative per-period loss")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, Integral)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be a positive integer")
    var, cvar = empirical_var_cvar(-values, confidence)
    mean = float(values.mean())
    volatility = float(values.std(ddof=1))
    log_wealth = np.concatenate(([0.0], np.cumsum(np.log1p(values))))
    peaks = np.maximum.accumulate(log_wealth)
    drawdown = float(np.max(-np.expm1(log_wealth - peaks)))
    with np.errstate(over="ignore"):
        cumulative_return = float(np.expm1(log_wealth[-1]))
    result = {
        "mean_return": mean,
        "volatility": volatility,
        "var": var,
        "cvar": cvar,
        "loss_frequency": float(np.mean(-values > loss_threshold)),
        "annualized_mean": mean * int(periods_per_year),
        "annualized_volatility": volatility * float(np.sqrt(periods_per_year)),
        "cumulative_return": cumulative_return,
        "max_drawdown": drawdown,
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("risk summary overflowed; check return magnitudes and path length")
    return result
