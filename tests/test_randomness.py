import json

import numpy as np
import pytest

from thesis_portfolio.randomness import rng_for, seed_metadata


def test_repeatable_order_independent_streams():
    expected = rng_for(42, 0, "evaluation").normal(size=20)
    rng_for(42, 0, "optimization").normal(size=10_000)
    rng_for(42, 1, "evaluation").normal(size=500)
    np.testing.assert_array_equal(expected, rng_for(42, 0, "evaluation").normal(size=20))


def test_streams_replications_and_root_seeds_differ():
    identities = [
        (42, 0, "history"),
        (42, 0, "optimization"),
        (42, 0, "evaluation"),
        (42, 1, "history"),
        (43, 0, "history"),
    ]
    draws = [rng_for(*identity).integers(0, 2**32, size=20).tobytes() for identity in identities]
    assert len(set(draws)) == len(draws)


def test_global_rng_state_is_not_changed():
    before = np.random.get_state()
    rng_for(42, 0, "history").normal(size=100)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_metadata_reconstructs_generator():
    metadata = json.loads(json.dumps(seed_metadata(2**100, 7, "optimization")))
    sequence = np.random.SeedSequence(
        metadata["root_seed"],
        spawn_key=metadata["spawn_key"],
        pool_size=metadata["seed_sequence_pool_size"],
    )
    restored = np.random.Generator(np.random.PCG64(sequence))
    np.testing.assert_array_equal(
        restored.normal(size=20), rng_for(2**100, 7, "optimization").normal(size=20)
    )


@pytest.mark.parametrize(
    "identity",
    [
        (-1, 0, "history"),
        (1, -1, "history"),
        (True, 0, "history"),
        (1, 0.5, "history"),
        (1, 0, "unknown"),
    ],
)
def test_bad_identity_raises(identity):
    with pytest.raises(ValueError):
        rng_for(*identity)
