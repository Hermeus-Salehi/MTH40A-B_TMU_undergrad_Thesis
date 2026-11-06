import numpy as np
import pytest

from thesis_portfolio.simulation import gaussian_returns


def test_gaussian_population_moments_with_sampling_tolerances():
    mean = np.array([0.0005, -0.0001, 0.0003])
    vol = np.array([0.01, 0.02, 0.015])
    correlation = np.array([[1, 0.4, -0.2], [0.4, 1, 0.1], [-0.2, 0.1, 1]])
    covariance = correlation * np.outer(vol, vol)
    count = 100_000
    returns = gaussian_returns(mean, covariance, count, np.random.default_rng(136))
    # Gaussian mean SE and Wishart covariance SE; six SE avoids brittle tests.
    np.testing.assert_array_less(np.abs(returns.mean(axis=0) - mean), 6 * vol / np.sqrt(count))
    covariance_se = np.sqrt((covariance**2 + np.outer(vol**2, vol**2)) / (count - 1))
    np.testing.assert_array_less(
        np.abs(np.cov(returns, rowvar=False) - covariance), 6 * covariance_se
    )


def test_singular_covariance_has_exact_linear_dependency():
    returns = gaussian_returns([0, 0], [[0.01, 0.02], [0.02, 0.04]], 100, np.random.default_rng(3))
    np.testing.assert_allclose(returns[:, 1], 2 * returns[:, 0], atol=1e-14)


def test_single_asset_and_deterministic_zero_variance():
    returns = gaussian_returns([0.002], [[0]], 3, np.random.default_rng(3))
    np.testing.assert_array_equal(returns, [[0.002], [0.002], [0.002]])


@pytest.mark.parametrize(
    "covariance",
    [
        [[1, 2], [2, 1]],  # indefinite
        [[0.01, 0.002], [0.001, 0.01]],  # asymmetric
        [[1, np.nan], [np.nan, 1]],
        [[1]],
        [[-1e-12, 0], [0, 1e-12]],  # invalid even on a small absolute scale
    ],
)
def test_invalid_covariances_raise(covariance):
    with pytest.raises(ValueError):
        gaussian_returns([0, 0], covariance, 10, np.random.default_rng(3))


@pytest.mark.parametrize("count", [0, -1, 1.2, True])
def test_invalid_observation_count_raises(count):
    with pytest.raises(ValueError):
        gaussian_returns([0], [[1]], count, np.random.default_rng(3))


def test_generator_is_required_and_inputs_are_preserved():
    mean = np.array([0.1])
    covariance = np.array([[0.2]])
    with pytest.raises(TypeError):
        gaussian_returns(mean, covariance, 2, None)
    gaussian_returns(mean, covariance, 2, np.random.default_rng(2))
    np.testing.assert_array_equal(mean, [0.1])
    np.testing.assert_array_equal(covariance, [[0.2]])
