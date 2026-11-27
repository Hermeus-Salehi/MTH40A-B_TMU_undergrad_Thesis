# Milestone 5 hybrid implementation report

## Scope delivered

Implemented the annual Gaussian migration and GARCH/GJR core in the existing `thesis_portfolio` package. Reusable mathematics is separated from independently executable research notebooks. Exact ERC determines a model-specific annual equality target for minimum variance and minimum CVaR. Forecasting uses observations only; evaluation continues from the true oracle endpoint after weights and thresholds are frozen.

## Entry points

- [Research notebook guide](../../notebooks/RESEARCH_GUIDE.md)
- [GARCH recipe](../../notebooks/modelling/02_garch_gaussian.ipynb)
- [GJR recipe](../../notebooks/modelling/03_gjr_gaussian.ipynb)
- [Three-regime evidence](../../notebooks/experiments/04_three_regime_comparison.ipynb)
- [Canonical annual modules](../../src/thesis_portfolio/annual/)
- [Granular task map](m5_task_map.csv)

## Files and responsibilities

`src/thesis_portfolio/annual/` contains explicit state/observation contracts, YAML configuration, chronological and conditional simulation, conditional Gaussian likelihood/MLE, annual aggregation and portfolio composition, diagnostics, plotting, study orchestration, saved-run comparison, and artifact persistence. Existing portfolio solvers and empirical-tail functions are reused unchanged.

`notebooks/foundations/`, `notebooks/modelling/`, and `notebooks/experiments/` contain eight recipes with 134 executed active code cells. The 443 former granular task IDs are retained as module-owned or demonstration aliases; they are not represented as 443 duplicated implementation cells. The module/detail consolidation is explicit in the task map and decision log.

Updated `pyproject.toml`, `uv.lock`, generated `requirements.txt`, the root README, notebook navigation, research design, decision log, and persistent `AGENTS.md`. Added focused tests under `tests/annual/`. Run settings are in `configs/annual/pilot.yaml`.

## Actual validation

- `MPLCONFIGDIR=/tmp/thesis-m5-mpl .venv/bin/python -m pytest -q`: **121 passed**, including the existing baseline tests and nine new annual tests.
- `ruff check` on the annual modules, new tests and authoring/execution scripts: **all checks passed**.
- `scripts/execute_research_notebooks.py`: seven initial recipes passed from fresh kernels with smoke settings. The eighth saved-evidence recipe passed using explicit pilot references.
- The three model recipes, recovery, convergence and stress recipes passed at the configured pilot profile after the final GJR-bound correction.
- Pilot baseline settings: **N=1000, H=252, Mopt=10000, Meval=50000**. Recovery covers 300/600/1000/1200 nested histories; the eight source-exact fits completed for the prespecified seed.
- The pilot numerical-count study uses 10000/20000 forecast scenarios and 25000/50000 evaluation scenarios with weights fixed for evaluation comparisons.
- All eight GARCH/GJR baseline and stress combinations completed at pilot counts. Each dynamic-market run includes a Gaussian comparator on the same history and oracle futures.
- A scratch copy of the original daily M4 notebook passed **all 506 code cells** in a fresh kernel. The original and archive retain the recorded identical SHA256.
- All 134 active recipe cells have executed outputs, unique stable tags, an immediately preceding Markdown explanation, and no error outputs.
- Checksums of the referenced pilot result artifacts were recomputed successfully.
- Representative volatility, risk-share/frontier, loss, and final dashboard figures were visually inspected.

Execution evidence is in [smoke execution](notebook_execution_smoke.json), [pilot execution](notebook_execution_pilot.json), and [final notebook audit](final_notebook_audit.json).

## Reference pilot runs

| Oracle | Run ID | Largest fitted-return equality residual | ERC share error |
| --- | --- | ---: | ---: |
| gaussian | `m05-gaussian-baseline-provisional_pilot-20261127T163448-2a76f04c` | 0 | 1.14e-13 |
| garch | `m05-garch-baseline-provisional_pilot-20261127T163455-9c7a1345` | 1.04e-17 | 5.48e-13 |
| gjr | `m05-gjr-baseline-provisional_pilot-20261127T163503-222a7cac` | 2.78e-17 | 1.92e-12 |

Each run has fit tables, optimizer diagnostics, initial weights, equality checks, forecast thresholds, oracle metrics, validation records, terminal scenarios, resolved settings, and source/data hashes under `results/<run_id>/`. Its publication plots are under `figures/<run_id>/`. Explicit references are stored in [reference_runs.json](reference_runs.json).

The final dashboard is [three_regime_dashboard.png](../../results/m05-three-regime-comparison-20261127T163643-e5d4d826/three_regime_dashboard.png). Its source tables and run references are stored beside it.

## Corrections and retained limitations

An initial integration test caught conversion of a read-only weight mapping into a table; this was corrected. Final review removed an unnecessary gamma<=1 fitting bound: the GJR stationarity constraint now allows the appropriate range below 2. Boundary diagnostics include gamma=0. The affected pilot recipes were rerun. A temporary automatic-approval usage limit prevented the first fresh-kernel request; the same request succeeded after the user requested continuation.

There are no remaining software execution blockers in the delivered core. Scientific adequacy is deliberately not certified: history-length acceptance criteria were not supplied, so N=1000 remains a labelled provisional reference. Final counts, source scaling and stress settings remain provisional. MLE standard errors are not reported, residual p-values are descriptive, and conditional evaluation intervals exclude across-history estimation uncertainty. No favorable reseeding, approximate ERC substitution, or oracle-informed fitting is used.

The original daily notebook was not edited. Its runnable archive is in `archive/m4_daily/`; earlier daily conventions remain available only for regression. Copulas, the full forecasting cross-grid, rebalancing, and replicated M7 inference remain deferred.

## Next research step

Review the recovery table and agree an adequacy criterion before treating a historical length as calibrated. Review numerical-count sensitivity before fixing final scenario counts. Then use the already implemented recipes to rerun with the approved settings; no architecture redesign is required.
