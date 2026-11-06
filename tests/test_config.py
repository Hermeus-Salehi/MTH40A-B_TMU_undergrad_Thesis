"""Scientific settings and configuration entry points."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from thesis_portfolio.config import ExperimentConfig, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gaussian_smoke.toml"


def test_toml_and_saved_json_are_interchangeable(tmp_path):
    original = load_config(CONFIG)
    saved = tmp_path / "resolved.json"
    saved.write_text(json.dumps(original.to_dict()))
    assert load_config(saved) == original
    command = [sys.executable, "-m", "thesis_portfolio", "validate", "--config", str(saved)]
    validated = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(validated.stdout) == original.to_dict()
    saved.write_text("{}")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "missing keys" in rejected.stderr


@pytest.mark.parametrize(
    "key,value",
    [
        ("rebalancing_interval", 21),
        ("confidence", 1),
        ("confidence", 10**400),
        ("history_observations", 3),
        ("replications", True),
        ("comparison_modes", ["adaptive"]),
    ],
)
def test_reject_invalid_experiment_settings(key, value):
    settings = load_config(CONFIG).to_dict()
    settings[key] = value
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(settings)


@pytest.mark.parametrize("reserved", ["mode", "replication", "strategy"])
def test_asset_names_cannot_overwrite_result_identifiers(reserved):
    settings = load_config(CONFIG).to_dict()
    settings["market"]["asset_names"][0] = reserved
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(settings)


@pytest.mark.parametrize("issue", ["covariance_overflow", "duplicate_name", "perfect_correlation"])
def test_reject_unusable_market(issue):
    settings = load_config(CONFIG).to_dict()
    market = settings["market"]
    if issue == "covariance_overflow":
        market["volatility"][0] = 1e200
    elif issue == "duplicate_name":
        market["asset_names"][1] = market["asset_names"][0]
    else:
        market["correlation"] = [[1.0] * 4 for _ in range(4)]
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(settings)
