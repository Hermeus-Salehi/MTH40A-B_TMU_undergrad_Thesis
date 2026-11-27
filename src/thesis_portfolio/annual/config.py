"""Strict annual YAML loading and separate public/private settings."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .models import MarketState


@dataclass(frozen=True)
class PublicSpec:
    family: str
    maxiter: int = 1000
    fit_tolerance: float = 1e-9

    def __post_init__(self):
        if self.family not in ("gaussian", "garch", "gjr"):
            raise ValueError("unsupported fitted family")
        if self.maxiter < 1 or not 0 < self.fit_tolerance < 1:
            raise ValueError("invalid fitting options")


def load_run_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    required = {
        "name",
        "seed",
        "history",
        "portfolio",
        "simulation",
        "risk",
        "oracle",
        "fit",
        "status",
    }
    if set(cfg) != required:
        raise ValueError(f"configuration fields must be {sorted(required)}")
    if cfg["portfolio"] != {
        "horizon": 252,
        "policy": "buy_and_hold",
        "target_policy": "erc_fixed_return",
    }:
        raise ValueError("only 252-day buy-and-hold ERC equality is implemented")
    for val in [
        cfg["history"]["observations"],
        cfg["simulation"]["optimization_paths"],
        cfg["simulation"]["evaluation_paths"],
        cfg["simulation"]["batch_size"],
    ]:
        if isinstance(val, bool) or not isinstance(val, int) or val < 2:
            raise ValueError("counts must be integers >= 2")
    if cfg["history"]["discarded_prefix"] != 0:
        raise ValueError("active experiment retains all history; burn-in is unsupported")
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for q in [cfg["risk"]["optimization_confidence"], *cfg["risk"]["reporting"]]:
        if not 0 < q < 1:
            raise ValueError("confidence must lie in (0, 1)")
    PublicSpec("garch", **cfg["fit"])
    return cfg


def stream(seed, name):
    """Named streams do not depend on call order or Python's randomized hash."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))


def oracle_state(cfg, family, *, volatility_multiplier=1.0, correlation=None):
    """Build privileged source-derived scaled dynamics; never pass cfg to a fitter."""
    o = cfg["oracle"]
    assets = tuple(o["assets"])
    mean, sigma = np.array(o["mean"]), np.array(o["volatility"]) * volatility_multiplier
    if sigma.shape != mean.shape or np.any(sigma <= 0) or volatility_multiplier <= 0:
        raise ValueError("invalid volatility scaling")
    n = len(assets)
    rho = o["equicorrelation"] if correlation is None else correlation
    r = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(r, 1.0)
    dynamics = o["dynamics"][family]
    a, b, g = [np.full(n, dynamics[k]) for k in ("alpha", "beta", "gamma")]
    h = sigma**2
    omega = h * (1 - a - b - g / 2)
    return MarketState(family, mean, omega, a, b, g, r, h, np.zeros(n), assets)


def project_root(explicit=None):
    """An explicit root or the editable installation root, never a cwd mutation."""
    root = Path(explicit).resolve() if explicit else Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").exists():
        raise ValueError("supply the project root containing pyproject.toml")
    return root
