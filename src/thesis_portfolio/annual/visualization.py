"""Actual experiment figures; functions return figures without hidden simulation."""

import matplotlib.pyplot as plt
import numpy as np

from ..optimization import risk_contributions
from .models import variance_step
from .portfolio import frontier


def history_figure(observed, fitted, private=None):
    t = np.arange(1 - len(observed.returns), 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(t, observed.returns[:, 0], lw=0.6)
    axes[0].set(
        ylabel="Daily simple return",
        title=f"{fitted.state.family}: observed history, {observed.assets[0]}",
    )
    axes[1].plot(t, np.sqrt(fitted.variances[:, 0]), label="Fitted")
    if private is not None:
        axes[1].plot(
            t, np.sqrt(private.variances[:, 0]), alpha=0.7, label="Oracle: researcher-only"
        )
    for ax in axes:
        ax.axvline(0, color="black", ls="--", lw=0.8)
    axes[1].set(xlabel="Observed day (formation at 0)", ylabel="Daily conditional volatility")
    axes[1].legend()
    return fig


def acf_figure(table, asset):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for label in ["centered_returns", "standardized_residuals"]:
        x = table[
            (table.asset == asset) & table.squared & (table.series == label) & (table.lag > 0)
        ]
        ax.plot(x.lag, x.acf, marker=".", label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set(
        xlabel="Lag in trading days",
        ylabel="Squared-series ACF",
        title=f"Volatility dependence: {asset}",
    )
    ax.legend()
    return fig


def news_impact_figure(state):
    shocks = np.linspace(-4, 4, 151) * np.sqrt(state.long_run_variance[0])
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for gamma, label in [(0.0, "Symmetric term"), (state.gamma[0], "Fitted asymmetric term")]:
        h = variance_step(
            shocks, state.long_run_variance[0], state.omega[0], state.alpha[0], state.beta[0], gamma
        )
        ax.plot(shocks, h, label=label)
    ax.set(
        xlabel="Previous centered daily shock",
        ylabel="Next daily variance",
        title="News impact at the same prior variance",
    )
    ax.legend()
    return fig


def fan_figure(futures, family):
    wealth = np.concatenate(
        (
            np.ones((len(futures.retained_paths), 1)),
            np.cumprod(1 + futures.retained_paths[:, :, 0], axis=1),
        ),
        axis=1,
    )
    q = np.quantile(wealth, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    t = np.arange(futures.horizon + 1)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.fill_between(t, q[0], q[4], alpha=0.2, label="5–95% subset quantiles")
    ax.fill_between(t, q[1], q[3], alpha=0.35, label="25–75% subset quantiles")
    ax.plot(t, q[2], label="Subset median")
    ax.set(
        xlabel="Future trading day",
        ylabel="Asset 1 wealth per initial dollar",
        title=f"{family}: illustrative fan from {len(wealth)} retained paths (not full-sample tail estimates)",
    )
    ax.legend()
    return fig


def portfolio_figure(frozen, assets):
    curve, gmv = frontier(frozen.mean, frozen.covariance)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    x = np.arange(len(assets))
    for j, (strategy, w) in enumerate(frozen.weights.items()):
        axes[0].bar(x + (j - 1) * 0.25, w, width=0.25, label=strategy)
        rc = risk_contributions(w, frozen.covariance)
        axes[1].bar(x + (j - 1) * 0.25, rc / rc.sum(), width=0.25, label=strategy)
        axes[2].scatter(np.sqrt(w @ frozen.covariance @ w), frozen.mean @ w, label=strategy)
    for ax in axes[:2]:
        ax.set_xticks(x, assets, rotation=30)
    axes[0].set(ylabel="Initial capital weight", title="Annual buy-and-hold allocation")
    axes[1].set(ylabel="Share of fitted annual volatility", title="Risk contributions")
    axes[1].axhline(1 / len(assets), color="black", ls="--", lw=0.6)
    axes[2].plot(curve.volatility, curve["mean"], lw=1)
    axes[2].scatter(
        gmv["gmv_volatility"], gmv["gmv_mean"], marker="x", color="black", label="GMV diagnostic"
    )
    axes[2].axhline(frozen.target, ls="--", lw=0.6, color="grey")
    axes[2].set(
        xlabel="Fitted annual volatility",
        ylabel="Fitted annual mean",
        title=f"Equality target {frozen.target:.2%}",
    )
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    return fig


def loss_figure(frozen, forecasts, evaluation, family):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, (strategy, w) in zip(axes, frozen.weights.items()):
        ax.hist(-forecasts.terminal @ w, bins=60, density=True, alpha=0.4, label="Fitted")
        ax.hist(-evaluation.terminal @ w, bins=60, density=True, alpha=0.4, label="Oracle")
        row = frozen.forecast_metrics.query("strategy == @strategy").iloc[0]
        ax.axvline(row["var"], color="black", ls="--", label=f"Forecast VaR {row.confidence:.0%}")
        ax.axvline(row.cvar, color="red", ls=":", label="Forecast CVaR")
        ax.set(xlabel="Annual loss (decimal)", ylabel="Density", title=f"{family} · {strategy}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Forecast M={len(forecasts.terminal)}; oracle M={len(evaluation.terminal)}")
    return fig


def recovery_figure(table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    good = table[table.success]
    for family in good.family.unique():
        x = good[good.family == family]
        axes[0].plot(x.observations, abs(x.error_persistence), marker="o", label=family)
        axes[1].plot(x.observations, x.volatility_normalized_rmse, marker="o", label=family)
    axes[0].set(xlabel="Observed history length", ylabel="Absolute persistence error")
    axes[1].set(xlabel="Observed history length", ylabel="Normalized volatility RMSE")
    axes[0].legend()
    fig.suptitle("Source-exact recovery; finite-sample diagnostics, not acceptance thresholds")
    return fig
