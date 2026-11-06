"""End-to-end checks on the information used to fit and evaluate portfolios."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thesis_portfolio import experiment
from thesis_portfolio.config import load_config
from thesis_portfolio.optimization import OptimizationError
from thesis_portfolio.simulation import gaussian_returns

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gaussian_smoke.toml"


@pytest.fixture(scope="module")
def config():
    return replace(
        load_config(CONFIG),
        name="integration",
        history_observations=90,
        optimization_scenarios=85,
        evaluation_observations=33,
        replications=2,
    )


@pytest.fixture(scope="module")
def baseline(config, tmp_path_factory):
    return experiment.run_experiment(config, tmp_path_factory.mktemp("baseline"))


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_manifest_traces_saved_data_to_configuration_and_source(baseline, config):
    manifest = json.loads((baseline / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert load_config(baseline / "resolved_config.json") == config
    hashes = manifest["artifact_sha256"]
    expected = {
        "resolved_config.json",
        "replication_0000.npz",
        "replication_0001.npz",
        "provenance.json",
        "source_snapshot.zip",
        "optimizer_diagnostics.json",
        "metrics.csv",
        "weights.csv",
        "summary.csv",
        "seeds.json",
        "run.log",
    }
    assert expected <= set(hashes)
    for filename, digest in hashes.items():
        assert hashlib.sha256((baseline / filename).read_bytes()).hexdigest() == digest
    provenance = json.loads((baseline / "provenance.json").read_text())
    assert provenance["packages"]["numpy"] == np.__version__
    assert provenance["source_sha256"]


def test_fitted_scenarios_can_be_replayed_from_history_and_recorded_seeds(baseline, config):
    metadata = json.loads((baseline / "seeds.json").read_text())
    assert len({tuple(s["spawn_key"]) for s in metadata}) == config.replications * 3
    for rep in range(config.replications):
        with np.load(baseline / f"replication_{rep:04d}.npz", allow_pickle=False) as data:
            history = data["history"]
            fitted_mean = history.mean(axis=0)
            fitted_covariance = np.cov(history, rowvar=False)
            np.testing.assert_allclose(data["estimated_mean"], fitted_mean)
            np.testing.assert_allclose(data["estimated_covariance"], fitted_covariance)
            seed = next(
                s for s in metadata if s["replication"] == rep and s["stream"] == "optimization"
            )
            sequence = np.random.SeedSequence(
                seed["root_seed"],
                spawn_key=seed["spawn_key"],
                pool_size=seed["seed_sequence_pool_size"],
            )
            reconstituted = gaussian_returns(
                fitted_mean,
                fitted_covariance,
                config.optimization_scenarios,
                np.random.Generator(np.random.PCG64(sequence)),
            )
            np.testing.assert_allclose(
                data["optimization_scenarios"], reconstituted, atol=1e-14, rtol=1e-12
            )


def test_evaluation_metrics_and_target_matches_can_be_recomputed(baseline, config):
    metrics = rows(baseline / "metrics.csv")
    weights = {
        (row["replication"], row["mode"], row["strategy"]): row
        for row in rows(baseline / "weights.csv")
    }
    diagnostics = json.loads((baseline / "optimizer_diagnostics.json").read_text())
    assert len(metrics) == len(weights) == config.replications * 2 * 3
    for row in metrics:
        rep, mode, strategy = row["replication"], row["mode"], row["strategy"]
        with np.load(baseline / f"replication_{int(rep):04d}.npz", allow_pickle=False) as data:
            recorded = weights[(rep, mode, strategy)]
            w = np.array([float(recorded[name]) for name in config.market.asset_names])
            np.testing.assert_array_equal(w, data[f"{mode}_{strategy}_weights"])
            assert np.min(w) >= -1e-8
            assert np.sum(w) == pytest.approx(1.0, abs=1e-8)
            realized = data["evaluation"] @ w
            np.testing.assert_allclose(data[f"{mode}_{strategy}_returns"], realized)
            assert float(row["mean_return"]) == pytest.approx(realized.mean())
            assert float(row["volatility"]) == pytest.approx(realized.std(ddof=1))
            assert float(row["cumulative_return"]) == pytest.approx(np.prod(1 + realized) - 1)
            # Compute fractional tail mass directly from order statistics, independently
            # of the positive-part identity used by the implementation.
            losses = np.sort(-realized)[::-1]
            tail_mass = len(losses) * (1 - config.confidence)
            count = int(tail_mass)
            tail_mean = (losses[:count].sum() + (tail_mass - count) * losses[count]) / tail_mass
            assert float(row["cvar"]) == pytest.approx(tail_mean)
            details = diagnostics[rep][mode][strategy]
            assert details["solver"]["success"]
            predicted = data["estimated_mean"] @ w
            assert float(row["estimated_return"]) == pytest.approx(predicted)
            if mode == "controlled":
                erc_target = data["estimated_mean"] @ data["controlled_erc_weights"]
                assert details["return_target"] == pytest.approx(erc_target)
                assert predicted == pytest.approx(erc_target, abs=1e-10)
                assert details["return_constraint"] == (
                    "reference" if strategy == "erc" else "equality"
                )
            else:
                assert details["return_target"] is None


def test_replay_preserves_numeric_outputs_and_first_replication(baseline, tmp_path):
    replay_config = load_config(baseline / "resolved_config.json")
    rerun = experiment.run_experiment(replay_config, tmp_path)
    for tabular in ("metrics.csv", "weights.csv", "seeds.json", "summary.csv"):
        assert (rerun / tabular).read_bytes() == (baseline / tabular).read_bytes()
    first_only = experiment.run_experiment(replace(replay_config, replications=1), tmp_path)
    for rep in range(replay_config.replications):
        with np.load(baseline / f"replication_{rep:04d}.npz") as original:
            candidates = [rerun, first_only] if rep == 0 else [rerun]
            for candidate in candidates:
                with np.load(candidate / f"replication_{rep:04d}.npz") as repeated:
                    for key in original.files:
                        np.testing.assert_array_equal(repeated[key], original[key])


def test_changing_only_evaluation_returns_changes_metrics_but_not_allocations(
    baseline, config, tmp_path, monkeypatch
):
    def shifted_market(mean, covariance, n_observations, rng):
        draws = gaussian_returns(mean, covariance, n_observations, rng)
        if n_observations == config.evaluation_observations:
            return draws + 0.01
        return draws

    monkeypatch.setattr(experiment, "gaussian_returns", shifted_market)
    shifted = experiment.run_experiment(config, tmp_path)
    assert (baseline / "weights.csv").read_bytes() == (shifted / "weights.csv").read_bytes()
    for original, changed in zip(
        rows(baseline / "metrics.csv"), rows(shifted / "metrics.csv"), strict=True
    ):
        # All fully invested portfolios must experience the identical daily shock.
        assert float(changed["mean_return"]) - float(original["mean_return"]) == pytest.approx(0.01)
        assert float(changed["cvar"]) - float(original["cvar"]) == pytest.approx(-0.01)


def test_scenario_count_does_not_change_history_or_evaluation(baseline, config, tmp_path):
    more = experiment.run_experiment(replace(config, optimization_scenarios=120), tmp_path)
    for rep in range(config.replications):
        name = f"replication_{rep:04d}.npz"
        with np.load(baseline / name) as original, np.load(more / name) as expanded:
            for key in ("history", "evaluation"):
                np.testing.assert_array_equal(expanded[key], original[key])
            assert len(expanded["optimization_scenarios"]) == 120


def test_solver_failure_records_failed_run(config, tmp_path, monkeypatch):
    def failed_solver(*args, **kwargs):
        raise OptimizationError("controlled solver failure for test")

    monkeypatch.setattr(experiment, "cvar", failed_solver)
    with pytest.raises(OptimizationError, match="controlled solver failure"):
        experiment.run_experiment(config, tmp_path)
    (run,) = tmp_path.iterdir()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "completed_utc" not in manifest
    assert "controlled solver failure" in manifest["error"]
    assert "controlled solver failure" in (run / "run.log").read_text()
    assert not (run / "summary.csv").exists()


@pytest.mark.parametrize(
    "size_field", ["history_observations", "optimization_scenarios", "evaluation_observations"]
)
def test_invalid_simple_return_fails_in_each_stream(config, tmp_path, monkeypatch, size_field):
    def invalid_market(mean, covariance, n_observations, rng):
        draws = gaussian_returns(mean, covariance, n_observations, rng)
        if n_observations == getattr(config, size_field):
            draws[0, 0] = -1.0
        return draws

    monkeypatch.setattr(experiment, "gaussian_returns", invalid_market)
    with pytest.raises(ValueError):
        experiment.run_experiment(config, tmp_path)
    (run,) = tmp_path.iterdir()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
