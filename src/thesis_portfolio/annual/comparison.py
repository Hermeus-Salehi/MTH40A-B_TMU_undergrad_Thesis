"""Read explicitly identified evidence runs; never silently choose a latest run."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_comparison(root, references):
    """Compare compatible annual runs and retain both oracle and forecast labels."""
    root = Path(root)
    records, signatures = [], []
    for oracle, run_id in references.items():
        if Path(run_id).name != run_id:
            raise ValueError("run references must be explicit directory identifiers")
        path = root / "results" / run_id
        m = json.loads((path / "manifest.json").read_text())
        if m["status"] != "completed" or m["oracle"] != oracle or m["stress"] != "baseline":
            raise ValueError("comparison requires completed matching baseline runs")
        signatures.append(
            (
                m["history_count"],
                m["horizon"],
                m["policy"],
                m["target_policy"],
                m["optimization_paths"],
                m["evaluation_paths"],
            )
        )
        frame = pd.read_csv(path / "evaluation/metrics.csv")
        records.append(frame.assign(oracle=oracle, run_id=run_id))
    if len(set(signatures)) != 1:
        raise ValueError("incompatible history lengths, horizons, policies or scenario counts")
    return pd.concat(records, ignore_index=True)


def regime_dashboard(table):
    """Market changes and forecast-model changes are shown in different panels."""
    view = table[table.confidence == 0.95]
    matched = view[view.oracle == view.forecast_family]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for strategy, group in matched.groupby("strategy", sort=False):
        axes[0, 0].plot(group.oracle, group.oracle_cvar, marker="o", label=strategy)
        axes[0, 1].plot(group.oracle, group.oracle_mean, marker="o", label=strategy)
    targets = matched.drop_duplicates("oracle")
    axes[0, 1].plot(
        targets.oracle,
        targets.fitted_target,
        color="black",
        ls="--",
        marker="x",
        label="Model-specific target",
    )
    axes[0, 0].set(ylabel="95% annual CVaR", title="Matched models across oracle regimes")
    axes[0, 1].set(ylabel="Annual return", title="Oracle means and fitted targets")
    for ax, oracle in zip(axes[1], ("garch", "gjr")):
        same = view[view.oracle == oracle]
        names = ["ERC", "Markowitz", "CVaR"]
        for j, family in enumerate((oracle, "gaussian")):
            family_data = same[same.forecast_family == family].set_index("strategy").loc[names]
            ax.bar(
                np.arange(3) + (j - 0.5) * 0.35,
                family_data.oracle_cvar,
                width=0.35,
                label=f"{family} forecast",
            )
        ax.set_xticks(np.arange(3), names)
        ax.set(
            ylabel="95% annual CVaR", title=f"Shared {oracle.upper()} history and oracle futures"
        )
        ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle("Annual evidence: targets match within each fitted model, not across models")
    return fig
