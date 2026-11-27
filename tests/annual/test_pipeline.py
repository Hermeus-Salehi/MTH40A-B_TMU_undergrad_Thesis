import json
from copy import deepcopy

import numpy as np
import pandas as pd

from thesis_portfolio.annual.config import load_run_config, project_root, stream
from thesis_portfolio.annual.experiment import execution_profile, run_experiment


def test_small_pipeline_replay_and_evaluation_count_invariance(tmp_path):
    cfg = execution_profile(load_run_config(project_root() / "configs/annual/pilot.yaml"), "smoke")
    cfg["simulation"].update(optimization_paths=60, evaluation_paths=80)
    first, m1 = run_experiment(tmp_path, cfg, "gaussian", make_plots=False)
    second, m2 = run_experiment(tmp_path, cfg, "gaussian", make_plots=False)
    assert m1["observed_sha256"] == m2["observed_sha256"]
    assert m1["evaluation_sha256"] == m2["evaluation_sha256"]
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "portfolios/weights.csv"),
        pd.read_csv(second / "portfolios/weights.csv"),
    )
    changed = deepcopy(cfg)
    changed["simulation"]["evaluation_paths"] = 90
    third, _ = run_experiment(tmp_path, changed, "gaussian", make_plots=False)
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "portfolios/weights.csv"), pd.read_csv(third / "portfolios/weights.csv")
    )
    public = (first / "resolved_public_config.yaml").read_text()
    assert "oracle:" not in public and "\nseed:" not in public
    assert json.loads((first / "manifest.json").read_text())["status"] == "completed"


def test_named_streams_are_independent_of_order():
    a = stream(8, "history").standard_normal(10)
    stream(8, "evaluation").standard_normal(100)
    np.testing.assert_array_equal(a, stream(8, "history").standard_normal(10))
