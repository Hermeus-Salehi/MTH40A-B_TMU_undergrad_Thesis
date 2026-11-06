"""Order-independent random streams derived from one recorded root seed.

Each (root seed, replication, stream) identifies a fresh stream. Re-requesting
that identity restarts it; callers should retain the generator when continuing
a draw sequence. No function reads or changes NumPy's legacy global RNG.
"""

from numbers import Integral

import numpy as np

STREAM_IDS = {"history": 0, "optimization": 1, "evaluation": 2}


def _validate(seed: int, replication: int, stream: str) -> tuple[int, int, int]:
    for name, value in (("seed", seed), ("replication", replication)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if stream not in STREAM_IDS:
        raise ValueError(f"stream must be one of {tuple(STREAM_IDS)}")
    return int(seed), int(replication), STREAM_IDS[stream]


def rng_for(seed: int, replication: int, stream: str) -> np.random.Generator:
    """Create a PCG64 generator with a stable, explicitly identified stream.

    ``SeedSequence`` mixes the root seed with ``(replication, stream_id)`` as
    its spawn key. Execution order and the number of draws in other streams
    therefore cannot alter this stream.
    """
    root_seed, replication_id, stream_id = _validate(seed, replication, stream)
    sequence = np.random.SeedSequence(root_seed, spawn_key=(replication_id, stream_id))
    return np.random.Generator(np.random.PCG64(sequence))


def seed_metadata(seed: int, replication: int, stream: str) -> dict:
    """Return JSON-compatible information sufficient to recreate this stream."""
    root_seed, replication_id, stream_id = _validate(seed, replication, stream)
    return {
        "root_seed": root_seed,
        "replication": replication_id,
        "stream": stream,
        "stream_id": stream_id,
        "spawn_key": [replication_id, stream_id],
        "seed_sequence_pool_size": 4,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
    }
