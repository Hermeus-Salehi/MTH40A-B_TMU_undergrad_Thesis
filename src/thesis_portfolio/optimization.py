"""Long-only portfolio optimizers with explicit numerical validation.

All returns, means, and covariance matrices must describe the same period and
use decimal simple returns. Population parameters are deliberately not accepted
through a market object: callers pass estimates or fitted-model scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import linprog, minimize


class OptimizationError(RuntimeError):
    """The mathematical problem was infeasible or a numerical solve failed."""


@dataclass(frozen=True)
class OptimizationResult:
    """A validated solution, its objective, and serializable solver diagnostics."""

    weights: NDArray[np.float64]
    objective: float
    diagnostics: dict[str, Any]


def _vector(value: ArrayLike, name: str, size: int | None = None) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    if size is not None and vector.size != size:
        raise ValueError(f"{name} must have length {size}")
    return vector


def _covariance(value: ArrayLike, *, positive_definite: bool = False) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("covariance must be a nonempty finite square matrix")
    scale = float(np.max(np.abs(matrix)))
    tolerance = max(scale, np.finfo(float).tiny) * 1e-10
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    if positive_definite and eigenvalues[0] <= tolerance:
        raise ValueError("ERC requires a numerically positive-definite covariance matrix")
    return matrix


def _target(
    mean: NDArray[np.float64],
    target_return: float | None,
    target_constraint: Literal["minimum", "equal"],
) -> float | None:
    if target_constraint not in ("minimum", "equal"):
        raise ValueError("target_constraint must be 'minimum' or 'equal'")
    if target_return is None:
        return None
    target = float(target_return)
    if not np.isfinite(target):
        raise ValueError("target_return must be finite")
    scale = max(float(np.max(np.abs(mean))), abs(target), np.finfo(float).tiny)
    if target > float(np.max(mean)) + scale * 1e-10:
        raise OptimizationError(
            "Return target is infeasible for a long-only, fully invested portfolio"
        )
    if target_constraint == "equal" and target < float(np.min(mean)) - scale * 1e-10:
        raise OptimizationError(
            "Equal return target is infeasible for a long-only, fully invested portfolio"
        )
    return target


def _solver_weights(raw: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    """Replace only machine-roundoff negatives from a bound-constrained solve.

    On the unit simplex, a dot product over n weights can accumulate O(n*eps)
    absolute error. Keep the raw minimum for auditing and refuse a larger
    bound violation. The caller rechecks the budget, target and objective
    with the actual weights it returns.
    """
    minimum = float(np.min(raw))
    if not np.all(np.isfinite(raw)) or minimum < -64 * raw.size * np.finfo(float).eps:
        raise OptimizationError(
            "Solver weights violate nonnegativity beyond floating-point roundoff"
        )
    return np.maximum(raw, 0.0), {
        "raw_minimum_weight": minimum,
        "roundoff_weights_zeroed": int(np.count_nonzero(raw < 0)),
    }


def _validate_solution(
    weights: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    target_return: float | None = None,
    target_constraint: Literal["minimum", "equal"] = "minimum",
) -> dict[str, float]:
    if not np.all(np.isfinite(weights)):
        raise OptimizationError("Solver returned nonfinite weights")
    budget_error = abs(float(np.sum(weights)) - 1.0)
    minimum_weight = float(np.min(weights))
    if budget_error > 1e-8 or minimum_weight < 0:
        raise OptimizationError(
            "Solver returned a portfolio violating budget or nonnegativity constraints"
        )
    diagnostics = {"budget_error": budget_error, "minimum_weight": minimum_weight}
    if mean is not None:
        expected_return = float(mean @ weights)
        diagnostics["estimated_return"] = expected_return
        if target_return is not None:
            shortfall = max(0.0, target_return - expected_return)
            target_error = (
                abs(target_return - expected_return) if target_constraint == "equal" else shortfall
            )
            scale = max(float(np.max(np.abs(mean))), abs(target_return), np.finfo(float).tiny)
            diagnostics["target_shortfall"] = shortfall
            diagnostics["target_error"] = target_error
            if target_error > scale * 1e-7:
                raise OptimizationError("Solver returned a portfolio violating its return target")
    return diagnostics


def _result(
    weights: NDArray[np.float64], objective: float, diagnostics: dict[str, Any]
) -> OptimizationResult:
    if not np.isfinite(objective):
        raise OptimizationError("Solver returned a nonfinite objective")
    solution = np.array(weights, dtype=float, copy=True)
    solution.setflags(write=False)
    return OptimizationResult(solution, float(objective), diagnostics)


def markowitz(
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    target_return: float | None = None,
    risk_aversion: float = 1.0,
    target_constraint: Literal["minimum", "equal"] = "minimum",
) -> OptimizationResult:
    """Optimize mean-variance utility, or variance subject to a return floor.

    Without a target, minimize ``0.5 * risk_aversion * w.T @ covariance @ w
    - mean @ w``. With a target, minimize ``w.T @ covariance @ w`` subject to
    ``mean @ w >= target_return`` (``target_constraint='minimum'``), or exact
    equality (``target_constraint='equal'``). Both enforce ``w >= 0`` and
    ``sum(w) = 1``.
    ``risk_aversion`` is only used by the independent utility formulation.
    Objectives and constraints are scaled internally, without changing their
    mathematical solution. The returned objective uses the original units.
    """
    matrix = _covariance(covariance)
    mu = _vector(mean, "mean", matrix.shape[0])
    risk_aversion = float(risk_aversion)
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive and finite")
    target = _target(mu, target_return, target_constraint)
    n_assets = mu.size
    initial = np.full(n_assets, 1.0 / n_assets)
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(n_assets)}
    ]
    if target is None:
        quadratic = 0.5 * risk_aversion
        linear = mu
        objective_name = "negative_mean_variance_utility"
    else:
        quadratic = 1.0
        linear = np.zeros(n_assets)
        objective_name = "variance"
        mean_scale = max(float(np.max(np.abs(mu))), abs(target), np.finfo(float).tiny)
        # A constant mean creates a redundant equality with the budget; omitting
        # it avoids an artificial singular SLSQP constraint system.
        if target_constraint != "equal" or float(np.ptp(mu)) > mean_scale * 1e-12:
            constraints.append(
                {
                    "type": "eq" if target_constraint == "equal" else "ineq",
                    "fun": lambda w: float((mu @ w - target) / mean_scale),
                    "jac": lambda w: mu / mean_scale,
                }
            )
        if target_constraint == "equal" and float(np.ptp(mu)) > 0.0:
            high = int(np.argmax(mu))
            low = int(np.argmin(mu))
            high_weight = float(np.clip((target - mu[low]) / (mu[high] - mu[low]), 0.0, 1.0))
            initial = np.zeros(n_assets)
            initial[high] = high_weight
            initial[low] = 1.0 - high_weight
        elif float(mu @ initial) < target:
            initial = np.zeros(n_assets)
            initial[int(np.argmax(mu))] = 1.0
    objective_scale = max(
        quadratic * float(np.max(np.abs(matrix))),
        float(np.max(np.abs(linear))),
        np.finfo(float).tiny,
    )

    def objective(weights: NDArray[np.float64]) -> float:
        return float(
            (quadratic * (weights @ matrix @ weights) - linear @ weights) / objective_scale
        )

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return (2.0 * quadratic * matrix @ weights - linear) / objective_scale

    solved = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not solved.success:
        raise OptimizationError(
            f"Markowitz optimization failed (status {solved.status}): {solved.message}"
        )
    weights, roundoff = _solver_weights(solved.x)
    diagnostics: dict[str, Any] = {
        "solver": "scipy.optimize.SLSQP",
        "success": True,
        "status": int(solved.status),
        "message": str(solved.message),
        "iterations": int(solved.nit),
        "objective_name": objective_name,
        "objective_scale": objective_scale,
        "target_constraint": target_constraint,
        **roundoff,
        **_validate_solution(weights, mu, target, target_constraint),
    }
    return _result(weights, objective(weights) * objective_scale, diagnostics)


def risk_contributions(weights: ArrayLike, covariance: ArrayLike) -> NDArray[np.float64]:
    """Return Euler contributions to volatility; their sum is portfolio volatility."""
    matrix = _covariance(covariance)
    allocation = _vector(weights, "weights", matrix.shape[0])
    variance = float(allocation @ matrix @ allocation)
    if variance <= 0:
        raise ValueError("risk contributions require positive portfolio variance")
    return allocation * (matrix @ allocation) / np.sqrt(variance)


def equal_risk_contribution(covariance: ArrayLike) -> OptimizationResult:
    """Solve positive equal-risk budgeting, then normalize to a fully invested portfolio.

    For scaled covariance C and b_i = 1/n, minimize
    ``0.5*x.T@C@x - sum(b_i*log(x_i))`` over positive x. Its first-order
    condition is ``x_i*(C@x)_i = b_i``. Damped Newton steps with an Armijo
    line search retain positivity. Normalization preserves equal risk shares.
    Singular covariance is rejected explicitly, without silent regularization.

    The returned objective is the maximum absolute deviation of a risk share
    from 1/n. The internal convex objective is included in diagnostics.
    """
    matrix = _covariance(covariance, positive_definite=True)
    covariance_scale = float(np.max(np.diag(matrix)))
    scaled = matrix / covariance_scale
    n_assets = scaled.shape[0]
    budgets = np.full(n_assets, 1.0 / n_assets)
    x = np.sqrt(budgets / np.diag(scaled))

    def objective(z: NDArray[np.float64]) -> float:
        return float(0.5 * (z @ scaled @ z) - budgets @ np.log(z))

    converged = False
    for iteration in range(1000):
        gradient = scaled @ x - budgets / x
        hessian = scaled + np.diag(budgets / x**2)
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError as exc:
            raise OptimizationError("ERC Newton system could not be solved") from exc
        error = float(np.max(np.abs(x * (scaled @ x) - budgets)))
        if error <= 1e-11:
            converged = True
            break
        step = 1.0
        declining = direction < 0
        if np.any(declining):
            step = min(step, float(0.99 * np.min(-x[declining] / direction[declining])))
        current_objective = objective(x)
        slope = float(gradient @ direction)
        for _ in range(80):
            candidate = x + step * direction
            if np.all(candidate > 0):
                candidate_objective = objective(candidate)
                # A true Newton decrease can fall below the resolution of f.
                # Permit a rounded tie only when the independent residual falls.
                rounding_allowance = 8 * np.finfo(float).eps * max(1.0, abs(current_objective))
                residual_improves = (
                    float(np.max(np.abs(candidate * (scaled @ candidate) - budgets))) < error
                )
                if candidate_objective <= current_objective + 1e-4 * step * slope or (
                    candidate_objective <= current_objective + rounding_allowance
                    and residual_improves
                ):
                    x = candidate
                    break
            step *= 0.5
        else:
            raise OptimizationError("ERC line search failed to find a descending positive step")
    if not converged:
        raise OptimizationError("ERC optimization did not converge within 1000 Newton iterations")
    weights = x / np.sum(x)
    contributions = risk_contributions(weights, matrix)
    risk_shares = contributions / np.sum(contributions)
    share_error = float(np.max(np.abs(risk_shares - budgets)))
    if share_error > 1e-8:
        raise OptimizationError("ERC solution failed the equal-risk-contribution check")
    diagnostics: dict[str, Any] = {
        "solver": "damped_newton_risk_budget",
        "success": True,
        "status": 0,
        "message": "Risk-budget first-order conditions satisfied",
        "iterations": iteration,
        "objective_name": "maximum_absolute_risk_share_error",
        "convex_objective": objective(x),
        "covariance_scale": covariance_scale,
        "risk_share_error": share_error,
        "risk_shares": risk_shares.tolist(),
        **_validate_solution(weights),
    }
    return _result(weights, share_error, diagnostics)


def cvar(
    scenarios: ArrayLike,
    *,
    confidence: float = 0.95,
    mean: ArrayLike | None = None,
    target_return: float | None = None,
    target_constraint: Literal["minimum", "equal"] = "minimum",
) -> OptimizationResult:
    """Minimize empirical CVaR of losses ``-scenarios @ weights`` via a sparse LP.

    Every scenario has equal probability. The Rockafellar-Uryasev objective is
    ``eta + sum(u)/(N*(1-confidence))``, with ``u >= loss-eta`` and ``u >= 0``.
    eta is unrestricted, allowing negative VaR/CVaR when all outcomes are gains.
    A return floor (``target_constraint='minimum'``) or equality (``'equal'``)
    uses ``mean`` or, when omitted, the
    scenario mean. The caller should pass its historical mean estimate when
    using a shared target across optimizers.
    """
    returns = np.asarray(scenarios, dtype=float)
    if returns.ndim != 2 or min(returns.shape) == 0 or not np.all(np.isfinite(returns)):
        raise ValueError("scenarios must be a nonempty finite observations-by-assets matrix")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    n_scenarios, n_assets = returns.shape
    mu = returns.mean(axis=0) if mean is None else _vector(mean, "mean", n_assets)
    target = _target(mu, target_return, target_constraint)
    return_scale = max(
        float(np.max(np.abs(returns))), float(np.max(np.abs(mu))), np.finfo(float).tiny
    )
    scaled_returns = returns / return_scale
    # Variables are [weights (n), eta (1), excess losses (N)].
    inequality = sparse.hstack(
        [
            -sparse.csr_matrix(scaled_returns),
            -sparse.csr_matrix(np.ones((n_scenarios, 1))),
            -sparse.eye(n_scenarios),
        ],
        format="csr",
    )
    upper = np.zeros(n_scenarios)
    equality = sparse.csr_matrix(
        np.concatenate([np.ones(n_assets), np.zeros(1 + n_scenarios)])[None, :]
    )
    equality_rhs = np.array([1.0])
    if target is not None:
        target_scale = max(float(np.max(np.abs(mu))), abs(target), np.finfo(float).tiny)
        target_row = sparse.csr_matrix(
            np.concatenate([-mu / target_scale, np.zeros(1 + n_scenarios)])[None, :]
        )
        if target_constraint == "equal":
            equality = sparse.vstack([equality, target_row], format="csr")
            equality_rhs = np.append(equality_rhs, -target / target_scale)
        else:
            inequality = sparse.vstack([inequality, target_row], format="csr")
            upper = np.append(upper, -target / target_scale)
    objective = np.concatenate(
        [np.zeros(n_assets), [1.0], np.full(n_scenarios, 1.0 / (n_scenarios * (1.0 - confidence)))]
    )
    bounds = [(0.0, 1.0)] * n_assets + [(None, None)] + [(0.0, None)] * n_scenarios
    solved = linprog(
        objective,
        A_ub=inequality,
        b_ub=upper,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if not solved.success:
        raise OptimizationError(
            f"CVaR optimization failed (status {solved.status}): {solved.message}"
        )
    if not np.all(np.isfinite(solved.x)):
        raise OptimizationError("CVaR solver returned nonfinite variables")
    weights, roundoff = _solver_weights(solved.x[:n_assets])
    constraint_diagnostics = _validate_solution(weights, mu, target, target_constraint)
    lp_variables = np.concatenate([weights, solved.x[n_assets:]])
    inequality_violation = max(0.0, float(np.max(inequality @ lp_variables - upper)))
    if inequality_violation > 1e-7:
        raise OptimizationError("CVaR solution violates the scenario constraints")
    losses = -scaled_returns @ weights
    quantile = float(np.quantile(losses, confidence, method="inverted_cdf"))
    empirical_objective = quantile + float(np.mean(np.maximum(losses - quantile, 0.0))) / (
        1.0 - confidence
    )
    if not np.isclose(float(solved.fun), empirical_objective, rtol=1e-7, atol=1e-8):
        raise OptimizationError(
            "CVaR LP objective disagrees with the empirical tail-risk calculation"
        )
    diagnostics: dict[str, Any] = {
        "solver": "scipy.optimize.linprog.highs",
        "success": True,
        "status": int(solved.status),
        "message": str(solved.message),
        "iterations": int(solved.nit),
        "objective_name": "empirical_cvar_loss",
        "confidence": confidence,
        "return_scale": return_scale,
        "eta": float(solved.x[n_assets] * return_scale),
        "scaled_inequality_violation": inequality_violation,
        "scenario_count": n_scenarios,
        "target_constraint": target_constraint,
        **roundoff,
        **constraint_diagnostics,
    }
    return _result(weights, empirical_objective * return_scale, diagnostics)
