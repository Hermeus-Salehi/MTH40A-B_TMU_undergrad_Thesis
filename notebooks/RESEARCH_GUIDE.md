# Read and run the annual research notebooks

Each notebook answers one research question, runs independently, and shows the
intermediate results. Reusable calculations live in
`src/thesis_portfolio/annual/`; the existing portfolio optimizers are reused.

| Read order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Mathematical checks](foundations/01_m5_mathematical_checks.ipynb) | Variance, likelihood, asymmetry, and buy-and-hold identities |
| 2 | [Parameter recovery](experiments/01_parameter_recovery.ipynb) | Source-exact nested histories and honest historical-length selection status |
| 3 | [Annual Gaussian](modelling/01_iid_gaussian.ipynb) | Compatible annual baseline, separate from daily M4 |
| 4 | [GARCH](modelling/02_garch_gaussian.ipynb) | Fitted dynamic volatility and a Gaussian comparator on the same history |
| 5 | [GJR-GARCH](modelling/03_gjr_gaussian.ipynb) | Asymmetric volatility and the same controlled comparison |
| 6 | [Monte Carlo precision](experiments/02_monte_carlo_convergence.ipynb) | Separate forecast-count and fixed-weight evaluation-count sensitivity |
| 7 | [Market stress](experiments/03_market_stress.ipynb) | Volatility-only, correlation-only, and combined changes with fresh fitted histories |
| 8 | [Three-regime evidence](experiments/04_three_regime_comparison.ipynb) | Read explicit saved run IDs and compare annual outcomes |

## Environment

From the project root:

```bash
uv sync --frozen
.venv/bin/python -m ipykernel install --prefix .venv --name thesis --display-name "Python (thesis)"
```

Open notebooks in your existing notebook editor and select this environment.
The package is installed in editable mode by `uv sync`. No notebook needs a
previous notebook's kernel, a `sys.path` edit, or a working-directory change.
For a non-editable installation, supply `THESIS_PROJECT_ROOT` explicitly.

For unattended fresh-kernel execution:

```bash
.venv/bin/python scripts/execute_research_notebooks.py
.venv/bin/python scripts/execute_research_notebooks.py --profile pilot notebooks/modelling/02_garch_gaussian.ipynb
```

The execution script sets the project kernel search path and records actual
success/failure, cell counts and elapsed times. Jupyter uses local communication
ports; restricted execution environments may require permission to start kernels.

## Configuration and evidence

`configs/annual/pilot.yaml` is the source of run settings. An explicit smoke
profile changes history length and scenario counts; its resolved settings are
saved. The pilot remains scientifically provisional even when all code executes.
The notebooks currently contain executed outputs; rerunning creates a new run ID.

The core control is **exact ERC first**. Its fitted annual mean determines the
equality target for all three strategies within a model/run. Different fitted
models can produce different targets. The comparison notebook reports these
targets alongside oracle risk and return, rather than claiming equal return across
models or market regimes.

Each result directory contains:

- `manifest.json`: status, counts, policy, source hashes, environment, data hashes.
- `fit/`: estimates, filtering states, correlation, optimizer and residual diagnostics.
- `portfolios/`: weights, solver diagnostics, exact-return constraint checks.
- `evaluation/`: frozen forecast thresholds and actual oracle annual risk metrics.
- `privileged/`: separately labelled oracle truth and truth-versus-fit diagnostics.
- `validation/`: deterministic checks, limitations, and task mapping.
- `scenarios/`: compressed terminal scenarios, not every full simulated trajectory.

Publication plots are under `figures/<run_id>/`. Recovery, convergence and stress
comparison notebooks also save their study-specific figures alongside their tables.
The three-regime recipe uses explicit versioned run references in
`documentation/implementation_tasks/reference_runs.json`; it never selects an
unspecified latest run. Update these references deliberately after a new experiment.

The selected reference runs, including their synthetic oracle artifacts, are
published with the repository. Their inventory is
`documentation/implementation_tasks/published_evidence.json`. Other generated
runs remain ignored by Git unless deliberately selected as reference evidence.

## Granular task mapping

The earlier 443 task IDs are preserved in
`documentation/implementation_tasks/m5_task_map.csv`. Reusable implementation
details are now module-owned and mapped to meaningful notebook demonstrations;
they are not 443 duplicated code cells. Each active stable cell has its own
Markdown purpose, inputs and outputs. The active-cell inventory and test records
show exactly what is implemented and executed.

The original daily notebook and its original source hash are preserved under
`archive/m4_daily/`. Its daily Gaussian formulas and daily-rebalancing assumptions
must not be applied to the annual compounded scenarios.

## Interpretation still belongs to the researcher

Review parameter recovery and boundary estimates before scientific interpretation.
History adequacy criteria are not yet specified; N=1000 is a labelled integration
reference. MLE standard errors and across-history inference are not claimed.
Residual tests are descriptive. A small smoke run proves execution, not a reliable
tail-risk ranking. No failed fit or invalid return is repaired by favorable reseeding.
