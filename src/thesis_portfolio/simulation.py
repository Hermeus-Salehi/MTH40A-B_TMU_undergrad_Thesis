"""Correlated Gaussian return simulation with explicit random-number ownership."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_returns(
    mean: ArrayLike,
    covariance: ArrayLike,
    n_observations: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate ``(observations, assets)`` simple, per-period decimal returns.

    Singular positive-semidefinite covariances are supported by an eigenvalue
    factorization. Only negative eigenvalues at floating-point roundoff scale
    are clipped; materially indefinite matrices raise ``ValueError``. Gaussian
    simple returns are unbounded below, so extreme configurations can produce
    returns below -100%; downstream wealth calculations must reject those.
    """
    mu = np.asarray(mean, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or mu.size == 0 or not np.all(np.isfinite(mu)):
        raise ValueError("mean must be a nonempty finite one-dimensional array")
    if cov.shape != (mu.size, mu.size) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be finite and match the asset count")
    if isinstance(n_observations, bool) or not isinstance(n_observations, Integral):
        raise ValueError("n_observations must be a positive integer")
    if n_observations < 1:
        raise ValueError("n_observations must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")

    scale = float(np.max(np.abs(cov)))
    tolerance = 100 * np.finfo(float).eps * max(1, mu.size) * scale
    if np.max(np.abs(cov - cov.T)) > tolerance:
        raise ValueError("covariance must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2)
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    # The principal square root is unchanged if LAPACK flips eigenvector signs
    # or rotates a basis inside a repeated eigenspace. This stabilizes seed replay.
    factor = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    samples = rng.standard_normal((int(n_observations), mu.size)) @ factor.T + mu
    if not np.all(np.isfinite(samples)):
        raise ValueError("simulation overflowed; use finite, numerically reasonable parameters")
    return samples
