"""Validated, explicit experiment settings; no silent unknown keys or unit conversion."""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _keys(data: dict, expected: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a table/object")
    missing, extra = expected - data.keys(), data.keys() - expected
    if missing or extra:
        raise ValueError(f"{name}: missing keys {sorted(missing)}; unknown keys {sorted(extra)}")


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class MarketConfig:
    asset_names: tuple[str, ...]
    mean: tuple[float, ...]
    volatility: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]

    @property
    def covariance(self) -> np.ndarray:
        vol = np.asarray(self.volatility)
        return np.outer(vol, vol) * np.asarray(self.correlation)

    @classmethod
    def from_dict(cls, data: dict) -> MarketConfig:
        _keys(data, {"asset_names", "mean", "volatility", "correlation"}, "market")
        names = data["asset_names"]
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("asset_names must be a nonempty list")
        if any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("asset_names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("asset_names must be unique")
        if set(names) & {"replication", "strategy", "mode"}:
            raise ValueError(
                "asset_names cannot use reserved output columns: replication, mode, strategy"
            )
        n = len(names)
        vectors = []
        for key in ("mean", "volatility"):
            values = data[key]
            if not isinstance(values, (list, tuple)) or len(values) != n:
                raise ValueError(f"market.{key} must have one entry per asset")
            vectors.append(tuple(_number(v, f"market.{key}") for v in values))
        mean, vol = vectors
        if any(v <= 0 for v in vol):
            raise ValueError("market.volatility must be strictly positive")
        raw = data["correlation"]
        if not isinstance(raw, (list, tuple)) or len(raw) != n:
            raise ValueError("correlation must be an n_assets by n_assets matrix")
        corr = []
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != n:
                raise ValueError("correlation must be an n_assets by n_assets matrix")
            corr.append(tuple(_number(x, "correlation") for x in row))
        a = np.asarray(corr)
        if not np.allclose(a, a.T, rtol=0, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(a), 1, rtol=0, atol=1e-12):
            raise ValueError("correlation diagonal must be one")
        if np.max(np.abs(a)) > 1 + 1e-12:
            raise ValueError("correlations must be between -1 and 1")
        # The Gaussian generator supports PSD; this three-strategy example needs PD for ERC.
        if np.linalg.eigvalsh(a)[0] <= 1e-10:
            raise ValueError("this ERC experiment requires a positive-definite correlation matrix")
        # Canonicalize accepted roundoff before deriving the covariance matrix.
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 1.0)
        market = cls(tuple(names), mean, vol, tuple(tuple(row) for row in a.tolist()))
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            covariance = market.covariance
        if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
            raise ValueError("volatility must produce a finite covariance with positive variances")
        return market


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    seed: int
    replications: int
    history_observations: int
    optimization_scenarios: int
    evaluation_observations: int
    periods_per_year: int
    comparison_modes: tuple[str, ...]
    risk_aversion: float
    confidence: float
    loss_threshold: float
    market: MarketConfig

    def to_dict(self) -> dict:
        # Return JSON-native lists rather than dataclass tuples.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        _keys(data, set(cls.__dataclass_fields__), "experiment")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        name = data["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("name must be 1–64 letters, digits, underscores or hyphens")
        values = {"schema_version": 1, "name": name}
        for key, minimum in (
            ("seed", 0),
            ("replications", 1),
            ("history_observations", 2),
            ("optimization_scenarios", 2),
            ("evaluation_observations", 2),
            ("periods_per_year", 1),
        ):
            values[key] = _integer(data[key], key, minimum)
        modes = data["comparison_modes"]
        if not isinstance(modes, (list, tuple)) or not modes:
            raise ValueError("comparison_modes must be a nonempty list")
        if any(m not in ("independent", "controlled") for m in modes):
            raise ValueError("comparison_modes supports only independent and controlled")
        if len(set(modes)) != len(modes):
            raise ValueError("comparison_modes cannot contain duplicates")
        values["comparison_modes"] = tuple(modes)
        for key in ("risk_aversion", "confidence", "loss_threshold"):
            values[key] = _number(data[key], key)
        if values["risk_aversion"] <= 0:
            raise ValueError("risk_aversion must be positive")
        if not 0 < values["confidence"] < 1:
            raise ValueError("confidence must lie strictly between zero and one")
        if values["loss_threshold"] < 0:
            raise ValueError("loss_threshold must be nonnegative")
        values["market"] = MarketConfig.from_dict(data["market"])
        if values["history_observations"] <= len(values["market"].asset_names):
            raise ValueError(
                "history_observations must exceed asset count for full-rank estimation"
            )
        return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("rb") as handle:
        data = json.load(handle) if path.suffix == ".json" else tomllib.load(handle)
    return ExperimentConfig.from_dict(data)
