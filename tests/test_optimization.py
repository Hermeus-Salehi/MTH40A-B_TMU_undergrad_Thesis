"""Analytical and adversarial checks of portfolio optimization mathematics."""

from types import SimpleNamespace

import numpy as np
import pytest

from thesis_portfolio import optimization
from thesis_portfolio.optimization import (
    OptimizationError,
    cvar,
    equal_risk_contribution,
    markowitz,
    risk_contributions,
)
from thesis_portfolio.risk import portfolio_returns


def test_markowitz_diagonal_minimum_variance_matches_analytic_weights():
    covariance = np.diag([1.0, 4.0, 9.0])
    expected = 1.0 / np.diag(covariance)
    expected /= expected.sum()
    result = markowitz(np.zeros(3), covariance, target_return=0.0)
    np.testing.assert_allclose(result.weights, expected, atol=2e-7)
    assert result.objective == pytest.approx(float(expected @ covariance @ expected), rel=1e-10)


def test_markowitz_utility_and_return_floor_are_distinct_formulations():
    mean = np.array([0.1, 0.2])
    covariance = np.diag([1.0, 4.0])
    independent = markowitz(mean, covariance)
    controlled = markowitz(mean, covariance, target_return=0.15)
    np.testing.assert_allclose(independent.weights, [0.78, 0.22], atol=1e-7)
    np.testing.assert_allclose(controlled.weights, [0.5, 0.5], atol=1e-7)
    assert mean @ controlled.weights >= 0.15 - 1e-10


def test_markowitz_scaling_preserves_small_objective_solution():
    base = markowitz([0.1, 0.2], np.diag([1.0, 4.0]))
    tiny = markowitz([1e-9, 2e-9], np.diag([1e-8, 4e-8]))
    np.testing.assert_allclose(tiny.weights, base.weights, atol=1e-8)
    assert tiny.objective == pytest.approx(base.objective * 1e-8)


def test_markowitz_maximum_feasible_return_target():
    result = markowitz([0.01, 0.02], np.diag([0.01, 0.04]), target_return=0.02)
    np.testing.assert_allclose(result.weights, [0.0, 1.0], atol=1e-7)


def test_controlled_equal_return_is_enforced_even_below_minimum_risk_return():
    mean = np.array([0.01, 0.03])
    covariance = np.diag([0.04, 0.01])
    result = markowitz(mean, covariance, target_return=0.015, target_constraint="equal")
    np.testing.assert_allclose(result.weights, [0.75, 0.25], atol=1e-8)
    scenarios = np.array([[-0.1, -0.01], [0.1, 0.01]])
    tail = cvar(
        scenarios, confidence=0.5, mean=mean, target_return=0.015, target_constraint="equal"
    )
    np.testing.assert_allclose(tail.weights, [0.75, 0.25], atol=1e-8)


@pytest.mark.parametrize("target", [0.01, 0.03])
def test_equal_return_target_boundaries(target):
    result = markowitz([0.01, 0.03], np.eye(2), target_return=target, target_constraint="equal")
    tail = cvar(
        [[-0.1, 0.1], [0.1, -0.1]],
        mean=[0.01, 0.03],
        target_return=target,
        target_constraint="equal",
    )
    expected = [1.0, 0.0] if target == 0.01 else [0.0, 1.0]
    np.testing.assert_allclose(result.weights, expected, atol=1e-8)
    np.testing.assert_allclose(tail.weights, expected, atol=1e-8)


def test_constant_mean_equal_target_has_no_redundant_slsqp_constraint():
    result = markowitz(
        [0.01, 0.01], np.diag([1.0, 4.0]), target_return=0.01, target_constraint="equal"
    )
    np.testing.assert_allclose(result.weights, [0.8, 0.2], atol=1e-7)


def test_equal_target_below_minimum_mean_is_infeasible():
    with pytest.raises(OptimizationError, match="infeasible"):
        markowitz([0.01, 0.02], np.eye(2), target_return=0.0, target_constraint="equal")
    with pytest.raises(OptimizationError, match="infeasible"):
        cvar([[0.01, 0.02]], target_return=0.0, target_constraint="equal")


@pytest.mark.parametrize("factor", [1e-12, 1.0, 1e12])
def test_erc_diagonal_covariance_gives_inverse_volatility_weights(factor):
    covariance = factor * np.diag([0.01, 0.04, 0.09])
    expected = 1.0 / np.sqrt(np.diag(covariance))
    expected /= expected.sum()
    result = equal_risk_contribution(covariance)
    np.testing.assert_allclose(result.weights, expected, atol=1e-9)
    contributions = risk_contributions(result.weights, covariance)
    np.testing.assert_allclose(contributions / contributions.sum(), np.full(3, 1 / 3), atol=1e-9)
    assert contributions.sum() == pytest.approx(
        np.sqrt(result.weights @ covariance @ result.weights)
    )


def test_erc_correlated_and_permuted_assets():
    covariance = np.array([[0.04, 0.006, -0.002], [0.006, 0.09, 0.012], [-0.002, 0.012, 0.16]])
    result = equal_risk_contribution(covariance)
    permutation = [2, 0, 1]
    permuted = equal_risk_contribution(covariance[np.ix_(permutation, permutation)])
    np.testing.assert_allclose(permuted.weights, result.weights[permutation], atol=1e-9)
    assert result.diagnostics["risk_share_error"] < 1e-8


def test_erc_rejects_singular_covariance_without_silent_regularization():
    with pytest.raises(ValueError, match="positive-definite"):
        equal_risk_contribution(np.ones((2, 2)))


def test_erc_converges_when_objective_decrease_reaches_roundoff():
    covariance = np.array(
        [
            [9.7224896802376002e-05, 2.1524171009268211e-05, -1.1140957603869725e-04],
            [2.1524171009268211e-05, 3.3415024552461276e-04, -2.1217897797859070e-05],
            [-1.1140957603869725e-04, -2.1217897797859070e-05, 1.3389687363709896e-04],
        ]
    )
    result = equal_risk_contribution(covariance)
    assert result.diagnostics["risk_share_error"] < 1e-9
    assert result.diagnostics["iterations"] < 30


def test_cvar_balances_complementary_losses():
    scenarios = np.array([[-0.1, 0.02], [0.02, -0.1], [0.05, 0.05], [0.05, 0.05]])
    result = cvar(scenarios, confidence=0.5)
    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-9)
    assert result.objective == pytest.approx(0.04)


def test_cvar_fractional_tail_probability_and_negative_eta():
    fractional = cvar(np.array([[0.0], [-1.0], [-2.0], [-3.0]]), confidence=0.625)
    assert fractional.objective == pytest.approx(8.0 / 3.0)
    gains = cvar(np.array([[0.01], [0.02], [0.03], [0.04]]), confidence=0.5)
    assert gains.objective == pytest.approx(-0.015)
    assert gains.diagnostics["eta"] < 0.0


def test_cvar_return_target_uses_explicit_estimated_mean():
    scenarios = np.array([[-0.01, -0.1], [0.01, 0.1], [-0.02, -0.2], [0.02, 0.2]])
    result = cvar(scenarios, confidence=0.5, mean=[0.01, 0.03], target_return=0.02)
    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-8)
    assert result.diagnostics["estimated_return"] == pytest.approx(0.02)


def test_cvar_scale_and_permutation_invariance():
    scenarios = np.array([[-0.1, 0.02], [0.02, -0.1], [0.05, 0.05], [0.05, 0.05]])
    base = cvar(scenarios, confidence=0.5)
    scaled = cvar(scenarios[:, ::-1] * 1e-9, confidence=0.5)
    np.testing.assert_allclose(scaled.weights, base.weights[::-1], atol=1e-9)
    assert scaled.objective == pytest.approx(base.objective * 1e-9)


@pytest.mark.parametrize("optimizer", ["markowitz", "cvar"])
def test_infeasible_target_is_explicit(optimizer):
    with pytest.raises(OptimizationError, match="infeasible"):
        if optimizer == "markowitz":
            markowitz([0.01, 0.02], np.eye(2), target_return=0.03)
        else:
            cvar([[0.01, 0.02], [0.01, 0.02]], mean=[0.01, 0.02], target_return=0.03)


@pytest.mark.parametrize(
    "covariance", [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [1, 1]]), np.array([[np.nan]])]
)
def test_invalid_covariances_are_rejected(covariance):
    with pytest.raises(ValueError):
        markowitz(np.zeros(covariance.shape[0]), covariance)


def test_bad_dimensions_and_confidence_are_rejected():
    with pytest.raises(ValueError, match="length"):
        markowitz([0.1], np.eye(2))
    with pytest.raises(ValueError, match="observations-by-assets"):
        cvar([0.1, 0.2])
    with pytest.raises(ValueError, match="confidence"):
        cvar([[0.1]], confidence=1.0)
    with pytest.raises(ValueError, match="positive portfolio variance"):
        risk_contributions([1.0], [[0.0]])


@pytest.mark.parametrize(
    "solver_name,call",
    [
        ("minimize", lambda: markowitz([0.0, 0.0], np.eye(2))),
        ("linprog", lambda: cvar([[0.01, 0.02], [-0.02, -0.01]])),
    ],
)
def test_solver_failures_never_return_fallback_weights(monkeypatch, solver_name, call):
    failed = SimpleNamespace(success=False, status=4, message="Deliberate test failure")
    monkeypatch.setattr(optimization, solver_name, lambda *args, **kwargs: failed)
    with pytest.raises(OptimizationError, match="Deliberate test failure"):
        call()


def test_single_asset_and_read_only_solution():
    for result in [
        markowitz([0.01], [[0.02]]),
        equal_risk_contribution([[0.02]]),
        cvar([[0.01], [-0.02]]),
    ]:
        np.testing.assert_array_equal(result.weights, [1.0])
        with pytest.raises(ValueError):
            result.weights[0] = 0.5


@pytest.mark.parametrize("negative", [-2e-15, -1e-7])
def test_roundoff_on_lp_bound_is_distinguished_from_invalid_short_positions(monkeypatch, negative):
    # HiGHS can return a tiny negative basic variable at an attainable boundary.
    solved = SimpleNamespace(
        x=np.array([negative, 1.0 - negative, 0.0, 0.0]),
        success=True,
        fun=0.0,
        status=0,
        message="optimal",
        nit=1,
    )
    monkeypatch.setattr(optimization, "linprog", lambda *args, **kwargs: solved)
    scenarios = np.zeros((1, 2))
    if negative < -1e-12:
        with pytest.raises(OptimizationError, match="nonnegativity"):
            cvar(scenarios)
        return
    result = cvar(scenarios)
    assert result.weights[0] == 0.0
    assert result.diagnostics["raw_minimum_weight"] == negative
    assert result.diagnostics["roundoff_weights_zeroed"] == 1
    assert portfolio_returns(scenarios, result.weights)[0] == 0.0
