import numpy as np
import pytest

from thesis_portfolio.estimation import estimate_gaussian


def test_sample_moments_against_hand_calculation():
    history = np.array([[0, 0], [1, 2], [2, 4]], dtype=float)
    mean, covariance = estimate_gaussian(history)
    np.testing.assert_array_equal(mean, [1, 2])
    np.testing.assert_array_equal(covariance, [[1, 2], [2, 4]])
    np.testing.assert_array_equal(history, [[0, 0], [1, 2], [2, 4]])


def test_single_asset_preserves_matrix_shape():
    mean, covariance = estimate_gaussian([[0.0], [0.02], [0.04]])
    assert mean.shape == (1,)
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == pytest.approx(0.0004)


def test_future_observations_do_not_affect_estimate_of_history_slice():
    observations = np.array([[0, 1], [1, 0], [2, 3], [4, 5]], dtype=float)
    before = estimate_gaussian(observations[:3])
    observations[3] = [10000, -10000]
    after = estimate_gaussian(observations[:3])
    for first, second in zip(before, after, strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "history", [[], [1, 2], [[1, 2]], [[], []], [[1], [np.nan]], [[1], [np.inf]]]
)
def test_invalid_history_raises(history):
    with pytest.raises(ValueError):
        estimate_gaussian(history)
