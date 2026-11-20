# Run the Milestone 4 Gaussian notebook

`Milestone_4_Gaussian_Baseline.ipynb` is a self-contained notebook: it uses no local
project-module imports or external data. It contains 503 numbered subtasks, each
with a Markdown explanation and a code cell, plus three setup cells.

Tasks 33–68 implement the core Gaussian baseline. Tasks 69–74 expose the
reproducibility, known-answer, invalid-input, moment, ERC and Gaussian tail checks.
Tasks 75–84 at the bottom generate and export the expanded visual outputs.
Only Milestone 4 is implemented.

## Start JupyterLab on your Mac

Use Python 3.13 for the pinned environment validated here. In Terminal:

```bash
cd "/Users/morty/Documents/ChatGPT/Thesis Under Prof. Rubtsov"
python3.13 -m venv .venv-notebook
source .venv-notebook/bin/activate
python -m pip install -r notebooks/requirements.txt
python -m ipykernel install --sys-prefix --name thesis-gaussian --display-name "Thesis Gaussian"
jupyter lab notebooks/Milestone_4_Gaussian_Baseline.ipynb
```

On another computer, replace the `cd` path with your repository location.
If your Python 3.13 command is `python3`, use that command to create the environment.
Package installation needs an internet connection; the notebook itself runs offline.
The numerical packages are pinned to the versions used for validation. JupyterLab
is the browser interface and is allowed to update within version 4.

Select the **Thesis Gaussian** kernel, then choose **Kernel → Restart Kernel and
Run All Cells…**. Let execution finish, then save the notebook. Start from the top
after changing settings: later cells depend on earlier cells.

For later sessions, only activate the environment and launch JupyterLab again:

```bash
cd "/Users/morty/Documents/ChatGPT/Thesis Under Prof. Rubtsov"
source .venv-notebook/bin/activate
jupyter lab notebooks/Milestone_4_Gaussian_Baseline.ipynb
```

## Find the diagrams

The last cell prints the absolute output directory. Each run creates a new
timestamped folder under `milestone4_results/` in the kernel's working directory,
normally the directory containing the notebook. Figures are displayed inline and
saved inside that run's `figures/` folder.

The nine new figure files each have PNG, PDF and SVG versions:

| Filename stem | Output |
| --- | --- |
| `efficient_frontier` | Fitted frontier and the three strategies in both modes |
| `portfolio_allocations` | Capital weights by asset and strategy |
| `gaussian_simulation_diagnostics` | Gaussian marginal, Q–Q, scatter and correlation diagnostics |
| `covariance_heatmaps` | Target, sample and error covariance matrices |
| `erc_risk_contributions` | ERC capital weights and fitted volatility risk shares |
| `tail_loss_distributions` | Evaluation loss distributions with 95% VaR/CVaR markers |
| `performance_comparison` | Return, risk, exceedance and concentration table |
| `gaussian_dashboard_independent` | Primary four-panel comparison dashboard |
| `gaussian_dashboard_controlled` | Four-panel comparison at a common fitted return |

Start with `gaussian_dashboard_independent.png`. Use PDF or SVG for scalable
thesis figures. The original three quick plots are also retained as PNGs.
Full-precision CSV tables and `validation_report.json` are in the run folder.
The manifest records checksums for the output artifacts.

## Interpret the checks

Deterministic backstops raise an error if an identity or input rule fails.
Statistical diagnostics use sampling uncertainty and report PASS or FLAG; a flag
requires investigation and must not be removed by searching for a favorable seed.
ERC equality is checked on its fitted covariance. Frontier points use fitted
moments; performance and loss distributions use independent evaluation data.
Tail frequencies use frozen training thresholds. These are synthetic results from
one fitted experiment, not evidence of a universal winning strategy.

The updated notebook was validated by executing all 506 code cells in a fresh
IPython process: 192 deterministic checks passed and no statistical flags appeared
for the supplied configuration. The validation environment did not run a browser
Jupyter server; use the steps above to launch that interface locally.

Jupyter references: [installation](https://jupyter.org/install) and
[kernel commands](https://jupyterlab.readthedocs.io/en/4.6.x/user/commands.html).
