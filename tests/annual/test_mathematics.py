import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from thesis_portfolio.annual.config import PublicSpec, load_run_config, oracle_state, stream
from thesis_portfolio.annual.diagnostics import acf, mathematical_backstops
from thesis_portfolio.annual.estimation import filter_variance, fit_market_model, likelihood
from thesis_portfolio.annual.models import MarketState, ObservedHistory
from thesis_portfolio.annual.portfolio import evaluate_portfolios, optimize_portfolios
from thesis_portfolio.annual.simulation import (
    aggregate_terminal_returns,
    generate_history,
    simulate_conditional_paths,
)


def config():
    from thesis_portfolio.annual.config import project_root

    return load_run_config(project_root() / "configs/annual/pilot.yaml")


def test_known_answers():
    assert all(mathematical_backstops().values())
    # The expected sequence is calculated independently, not with the recurrence helper.
    x = np.array([0.2, -0.1, 0.3])
    eps, h = filter_variance(x, [0.0, 0.01, 0.2, 0.5, 0.1], 0.1)
    np.testing.assert_allclose(h, [0.1, 0.068, 0.047])
    expected = 0.5 * np.sum(np.log(2 * np.pi * h) + eps**2 / h)
    assert likelihood([0.0, 0.01, 0.2, 0.5, 0.1], x, 0.1) == pytest.approx(expected)


def test_likelihood_gradient():
    x = np.array([0.2, -0.1, 0.3, -0.4, 0.05, 0.8])
    theta = np.array([0.01, 0.02, 0.1, 0.7, 0.12])
    _, grad = likelihood(theta, x, 0.1, gradient=True)
    numerical = approx_derivative(lambda p: likelihood(p, x, 0.1), theta).ravel()
    np.testing.assert_allclose(grad, numerical, atol=1e-5, rtol=1e-5)


def test_paths_endpoint_prefix_batch_and_aggregation():
    cfg = config()
    private = generate_history(oracle_state(cfg, "garch"), 120, stream(5, "history"))
    state = private.endpoint
    a = simulate_conditional_paths(state, 4, 10, stream(6, "future"), batch_size=3, retain=10)
    b = simulate_conditional_paths(state, 4, 20, stream(6, "future"), batch_size=8, retain=10)
    np.testing.assert_array_equal(a.terminal, b.terminal[:10])
    np.testing.assert_array_equal(a.first_variance, state.next_variance)
    np.testing.assert_allclose(a.terminal, np.prod(1 + a.retained_paths, axis=1) - 1)


def test_invalid_returns_never_filtered():
    with pytest.raises(ValueError, match="no clipping"):
        aggregate_terminal_returns(np.array([[[-1.0, 0.2]]]))


def test_no_oracle_input_and_fit_scale_equivariance():
    p = generate_history(oracle_state(config(), "garch"), 300, stream(12, "history"))
    with pytest.raises(TypeError):
        fit_market_model(p, PublicSpec("garch"))
    # Univariate fit keeps the regression test inexpensive and isolates unit scaling.
    obs = ObservedHistory(p.observed.returns[:, :1], ("A",))
    base = fit_market_model(obs, PublicSpec("garch"))
    scaled = fit_market_model(ObservedHistory(obs.returns * 100, ("A",)), PublicSpec("garch"))
    np.testing.assert_allclose(base.state.mean * 100, scaled.state.mean, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(base.state.omega * 10000, scaled.state.omega, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(base.variances * 10000, scaled.variances, rtol=2e-4)


def test_gaussian_annual_exact_return_and_no_evaluation_leakage():
    x = stream(17, "scenario").normal([0.03, 0.04, 0.05], [0.1, 0.2, 0.3], (200, 3))
    p = optimize_portfolios(x)
    original = {k: v.copy() for k, v in p.weights.items()}
    metrics = evaluate_portfolios(p, x + 0.2)
    assert abs(p.constraints.return_residual).max() < 1e-8
    assert len(metrics) == 6
    for key in p.weights:
        np.testing.assert_array_equal(p.weights[key], original[key])
        with pytest.raises(ValueError):
            p.weights[key][0] = 0.5


def test_bad_states_and_constant_series():
    with pytest.raises(ValueError):
        MarketState("dcc", [0], [0.1], [0], [0], [0], [[1]], [0.1], [0], ("A",))
    with pytest.raises(ValueError, match="constant"):
        acf(np.ones(50))
