"""Estimators receive historical observations, never market-generating parameters."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_gaussian(history: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return sample means and unbiased sample covariance (``ddof=1``).

    Rows are observations and columns are assets, including the single-asset
    case. The caller owns the historical cutoff; no evaluation data should be
    included in ``history``. Singular covariance is returned honestly, without
    undisclosed shrinkage or regularization.
    """
    observations = np.asarray(history, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 1:
        raise ValueError("history must have shape (at least 2 observations, at least 1 asset)")
    if not np.all(np.isfinite(observations)):
        raise ValueError("history must contain only finite observations")
    mean = observations.mean(axis=0)
    covariance = np.atleast_2d(np.cov(observations, rowvar=False, ddof=1))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("estimation overflowed; check the scale of observations")
    return mean, covariance
