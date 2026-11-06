import numpy as np
import pytest

from thesis_portfolio.risk import empirical_var_cvar, portfolio_returns, summarize_returns


@pytest.mark.parametrize(
    ("losses", "confidence", "expected"),
    [
        ([0, 1, 2, 3], 0.625, (2, 8 / 3)),  # tail contains 1.5 observations
        ([0, 1, 2, 3], 0.5, (1, 2.5)),
        ([0, 1, 1, 3], 0.5, (1, 2)),  # threshold ties must not inflate tail mass
        ([1, 1, 1], 0.95, (1, 1)),
        ([-4, -3, -2, -1], 0.75, (-2, -1)),  # gains must not be clipped at zero
        ([2], 0.99, (2, 2)),
    ],
)
def test_empirical_tail_risk_against_hand_calculation(losses, confidence, expected):
    assert empirical_var_cvar(losses, confidence) == pytest.approx(expected)


def test_cvar_obeys_translation_and_positive_scaling():
    losses = np.array([-0.02, 0.001, 0.01, 0.02, 0.06])
    original = np.array(empirical_var_cvar(losses, 0.72))
    np.testing.assert_allclose(empirical_var_cvar(losses * 3 + 0.1, 0.72), original * 3 + 0.1)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, np.nan, np.inf, True])
def test_invalid_confidence_raises(confidence):
    with pytest.raises(ValueError):
        empirical_var_cvar([0, 1], confidence)


@pytest.mark.parametrize("losses", [[], [[1, 2]], [1, np.nan], [np.inf]])
def test_invalid_losses_raise(losses):
    with pytest.raises(ValueError):
        empirical_var_cvar(losses, 0.95)


def test_constant_target_portfolio_and_units():
    result = portfolio_returns([[0.1, -0.1], [-0.2, 0.4]], [0.75, 0.25])
    np.testing.assert_allclose(result, [0.05, -0.05], atol=1e-15)
    metrics = summarize_returns(result, confidence=0.5, loss_threshold=0.04, periods_per_year=4)
    assert metrics["mean_return"] == pytest.approx(0, abs=1e-16)
    assert metrics["volatility"] == pytest.approx(np.sqrt(0.005))
    assert metrics["annualized_volatility"] == pytest.approx(2 * np.sqrt(0.005))
    assert metrics["loss_frequency"] == 0.5
    assert metrics["cumulative_return"] == pytest.approx(-0.0025)
    assert metrics["max_drawdown"] == pytest.approx(0.05)


def test_drawdown_includes_initial_wealth_and_recovers():
    result = summarize_returns([-0.2, 0.25, -0.1, -0.1])
    assert result["max_drawdown"] == pytest.approx(0.2)
    assert result["cumulative_return"] == pytest.approx(-0.19)


def test_loss_frequency_uses_strict_explicit_threshold():
    assert summarize_returns([-0.02, -0.03, 0.01], loss_threshold=0.02)[
        "loss_frequency"
    ] == pytest.approx(1 / 3)


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.4, 0.4], [1], [0.5, np.nan]])
def test_invalid_portfolio_weights_raise(weights):
    with pytest.raises(ValueError):
        portfolio_returns([[0.1, 0.2]], weights)


@pytest.mark.parametrize("returns", [[-1, 0.1], [-1.1, 0.1], [0.1], [0.1, np.nan]])
def test_invalid_wealth_paths_raise(returns):
    with pytest.raises(ValueError):
        summarize_returns(returns)
