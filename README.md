# MTH 40A/B thesis · Portfolio robustness and adaptability

How much downside risk does a portfolio retain when the market behaves differently from the model used to choose its weights? This repository provides the first computational step toward that thesis question. It runs a controlled Gaussian experiment with three long-only, fully invested strategies: Markowitz mean–variance, equal risk contribution (ERC), and conditional value-at-risk (CVaR). Each strategy learns from historical observations and faces the same subsequent market path.

Part 3 of the research checklist delivers the Python package, dependency lock, experiment configurations, tests, and run records. The included Gaussian run exercises the whole process on synthetic assets. Its parameters and sample sizes are development choices. The thesis design and the mathematical conventions are documented in the [experiment specification](docs/experiment_spec.md) and [model specification](docs/model_spec.md).

## Run the example

Use Git and `uv`. The project selects Python 3.13.2 through `.python-version` and installs the package and test tools in a local `.venv` using the checked-in `uv.lock`. [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
git clone --branch codex/part3-research-infrastructure https://github.com/Hermeus-Salehi/MTH40A-B_TMU_undergrad_Thesis.git
cd MTH40A-B_TMU_undergrad_Thesis
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen thesis-portfolio validate --config configs/gaussian_smoke.toml
uv run --frozen thesis-portfolio run --config configs/gaussian_smoke.toml --output results
```

Validation prints the resolved configuration. A successful run prints a `Completed:` directory and the path to `metrics.csv`. With the supplied settings, expect 18 rows of metrics and 18 rows of weights, one for each of 3 replications × 2 comparison modes × 3 strategies. A replication means repeating the **whole experiment**, including drawing new history and fitting new portfolios.

## Follow one replication

1. **Generate historical observations.** The config gives each asset a daily arithmetic mean and volatility and specifies correlations between assets. With `D` the diagonal matrix of volatilities and `C` the correlation matrix, the generating covariance is `Σ = D C D`. The simulator uses the symmetric square root `A = Q diag(√λ) Qᵀ` of `Σ` and draws independent daily vectors `r = μ + A z` with standard normal `z`. The result is a matrix with one day per row and one asset per column. All returns are decimals, so `0.01` means 1%.
2. **Estimate what the portfolio manager could know.** Given `H` historical rows, `estimate_gaussian` returns the column means `μ̂` and the sample covariance `Σ̂` calculated with denominator `H − 1`. The fitting function accepts the history, settings, and an optimization random generator. Its interface excludes the true market moments and the future evaluation path.
3. **Choose weights.** ERC and Markowitz use the fitted covariance. Markowitz also uses the fitted mean. CVaR receives `M` new, equally likely scenarios drawn from `N(μ̂, Σ̂)`. These scenarios depend on the historical estimates and use fresh random draws. The two comparison modes use these same fitted inputs.
4. **Evaluate out of sample.** After fitting, the market simulator draws a new, ordered evaluation path using the generating moments. Every strategy in both modes receives this exact path. Each target weight vector `w` is fixed throughout evaluation. At the start of each day, holdings are restored to `w` at zero transaction cost, producing the day’s simple portfolio return `rₚ = rᵀw`. The 21-day re-optimization interval in the proposal belongs to the later adaptive experiment.
5. **Save the evidence.** The runner records the input settings, fitted moments, scenario matrices, weights, realized returns, metrics, solver diagnostics, and source snapshot. Each run gets its own directory. Only runs whose manifest has `"status": "completed"` have finished successfully.

Random streams are addressed by **root seed, replication index, and purpose**. NumPy’s `SeedSequence` receives the root seed and spawn key `(replication, stream_id)`, and initializes a `PCG64` generator. Stable IDs are `history = 0`, `optimization = 1`, and `evaluation = 2`. Increasing `M` changes the CVaR scenario sample while preserving the history and evaluation draws. Adding replications preserves the earlier streams. `seeds.json` records each identifier and the generator details. [NumPy random stream generation](https://numpy.org/doc/stable/reference/random/parallel.html)

## How weights are chosen

Every solver uses `wᵢ ≥ 0` and `Σᵢ wᵢ = 1`. All objective calculations use daily units. The `independent` mode lets each strategy pursue its own objective. The `controlled` mode compares estimated risk at one common **fitted expected return**. First compute ERC from `Σ̂`. Then calculate `m = μ̂ᵀwERC` and impose the equality `μ̂ᵀw = m` on Markowitz and CVaR. ERC supplies a feasible allocation for that target. Realized returns can still differ because the portfolios hold different weights.

| Strategy | Independent objective | Controlled objective |
| --- | --- | --- |
| Markowitz | Minimize `(γ/2) wᵀΣ̂w − μ̂ᵀw`. The configurable `γ` sets the penalty on variance. | Minimize `wᵀΣ̂w` at the common fitted return `m`. |
| ERC | Give each asset an equal share of portfolio volatility. | Use the same ERC portfolio as the return reference. |
| CVaR | Minimize empirical loss CVaR over fitted-model scenarios. | Minimize the same scenario CVaR with `μ̂ᵀw = m`, using the historical mean estimate. |

**Markowitz.** SciPy’s SLSQP solver uses the analytical gradient and explicitly enforces the budget, bounds, and applicable return constraint. The code scales the objective for numerical conditioning and reports it in the original units. It checks solver status, the weight sum, the minimum weight, and any target residual. [Markowitz, *Portfolio Selection*](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)

**ERC.** Let `σₚ = √(wᵀΣ̂w)`. Asset `i` contributes `RCᵢ = wᵢ(Σ̂w)ᵢ / σₚ` to volatility. These contributions sum to `σₚ`. With `n` assets, ERC seeks `RCᵢ / σₚ = 1/n`. For the solver, scale `Σ̂` by its largest diagonal entry to form `S`, set `bᵢ = 1/n`, and solve the positive-variable convex problem

$$
\min_{x_i>0}\ \tfrac12 x^\top Sx-\sum_i b_i\log x_i .
$$

Its gradient is `Sx − b/x` and its Hessian is `S + diag(bᵢ/xᵢ²)`. The damped Newton iteration solves for a step and shortens it to keep `x` positive. At the minimum, each component satisfies `xᵢ(Sx)ᵢ = bᵢ`. Normalizing `w = x/Σᵢxᵢ` preserves those relative risk shares. The implementation checks the maximum absolute share error against `10⁻⁸`. It requires numerically positive-definite fitted covariance. [Maillard, Roncalli and Teiletche](https://doi.org/10.3905/jpm.2010.36.4.060) · [Spinu’s Newton formulation](https://doi.org/10.2139/ssrn.2297383)

**CVaR.** Let `sⱼ` be a scenario return vector, `Lⱼ(w) = −sⱼᵀw` its portfolio loss, and `a` the confidence level. Introduce one threshold `η` and one nonnegative excess loss `uⱼ` per scenario. The sparse linear program passed to SciPy/HiGHS is

$$
\min_{w,\eta,u}\ \eta+\frac{\sum_{j=1}^M u_j}{M(1-a)}
\qquad\text{subject to}\quad u_j\ge L_j(w)-\eta,\quad u_j\ge0,
$$

together with the budget, weight bounds, and any controlled return equality. The threshold can be negative. After solving, the code independently calculates empirical CVaR from the returned weights and checks that it agrees with the solver objective. This objective describes **optimization scenarios**. The `cvar` in `metrics.csv` describes the **evaluation path**. [Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk*](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)

## Read the risk measures

For the `T` realized daily portfolio returns, set `Lₜ = −rₚ,ₜ`. VaR at confidence `a` is the smallest observed loss whose empirical cumulative probability reaches `a`. CVaR is `VaR + mean(max(Lₜ − VaR, 0)) / (1 − a)`. This gives the boundary observation its proper fraction of the tail, including when losses tie. [Rockafellar and Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)

For a hand calculation, take four losses of 0%, 1%, 2%, and 3% at 62.5% confidence. VaR is 2%. The upper 37.5% tail contains 1.5 observations’ worth of mass. It includes all of the 3% loss and half of the 2% loss, giving CVaR `(3% + 0.5 × 2%) / 1.5 = 2.6667%`. In the smoke run, 252 daily observations at 95% confidence give 12.6 observations’ worth of tail mass per evaluation path.

| Output | Meaning |
| --- | --- |
| `estimated_return`, `mean_return` | Fitted expected return `μ̂ᵀw` versus the arithmetic mean of the realized daily portfolio returns. |
| `volatility`, `var`, `cvar` | Realized daily sample standard deviation (`T − 1` denominator) and one-day empirical loss measures. |
| `loss_frequency` | Fraction of evaluation days whose loss **exceeds** the common daily threshold. |
| `annualized_mean`, `annualized_volatility` | Arithmetic mean multiplied by periods/year, and daily volatility multiplied by the square root of periods/year under the IID Gaussian convention. |
| `cumulative_return`, `max_drawdown` | Compounded simple return `∏ₜ(1+rₚ,ₜ)−1` and the largest fractional decline from a running peak, including starting wealth 1. |

A Gaussian distribution can produce an asset return at or below −100%. The runner explicitly fails such an evaluation path because the positive-wealth interpretation would break down.

## Change an experiment

Copy the smoke TOML under a new name, edit the values, validate it, then pass its path to `run`. The development configuration increases the replication count to 100. [Smoke configuration](configs/gaussian_smoke.toml) · [Development configuration](configs/gaussian_development.toml)

| Configuration | Smoke setting and effect |
| --- | --- |
| `[market]` | Four synthetic assets. `asset_names` sets array and output order. `mean`, `volatility`, and `correlation` define the generating Gaussian distribution. |
| `seed`, `replications` | Seed `20261030`, 3 independent repetitions of the entire experiment, indexed from 0. |
| `history_observations`, `optimization_scenarios`, `evaluation_observations` | 1,000 days of history, 2,000 fitted-model scenarios, 252 out-of-sample days per replication. |
| `comparison_modes`, `risk_aversion` | Both `independent` and `controlled`. Risk aversion `γ = 5` applies to independent Markowitz. |
| `confidence`, `loss_threshold`, `periods_per_year` | 0.95 tail confidence, a 0.02 daily loss threshold, and 252 periods/year for annualization. |

The config loader rejects unknown or missing keys. Asset names must be unique. Volatilities must be positive. The correlation matrix must be symmetric with unit diagonal and numerically positive definite for this three-strategy experiment. History length must exceed asset count for full-rank estimation to be possible.

## Trace a result to its inputs

| Run artifact | Use |
| --- | --- |
| `manifest.json`, `run.log` | Check completion, run protocol, timestamps, config digest, artifact hashes, and error messages. |
| `resolved_config.json`, `seeds.json` | Recover the effective settings and stream identities. The resolved JSON is a valid input to `run`. |
| `metrics.csv`, `weights.csv` | Join by `replication`, `mode`, and `strategy` to see a portfolio’s allocation and measured performance. |
| `summary.csv` | Per-strategy and per-mode metric means across replications, with Monte Carlo standard error `s/√B` when `B > 1`. For strategy contrasts, pair the original rows by replication before taking differences. |
| `replication_0000.npz`, etc. | Each archive contains history, fitted moments, optimization scenarios, shared evaluation returns, and the weights and daily realized returns for every mode and strategy. |
| `optimizer_diagnostics.json` | Inspect objective name/value, solver status and iterations, constraint residuals, the controlled target, and ERC shares. |
| `provenance.json`, `source_snapshot.zip` | Identify the code revision and dirty state when available, Python environment, dependency versions, solver backends, and hashes of the captured source and imported package. |

For an example from the supplied smoke config, take seed `20261030`, replication `0`, and `controlled` mode. ERC assigned weights `[0.431582, 0.239773, 0.168344, 0.160302]` in asset order A–D. Their shares of fitted portfolio volatility were each 25%. All three strategies matched the daily fitted return `0.000690381381`. On the shared evaluation path the ERC portfolio’s 95% daily CVaR was `0.018780255` (1.8780%). These rounded values identify one run of the development example.

The following example selects the latest completed smoke run containing the controlled comparison. It reconstructs replication 0’s ERC portfolio path from asset returns and weights, computes CVaR from the losses, and checks the corresponding CSV entry. Run it from the repository root in a POSIX shell, or run the Python body in an interpreter from `uv run --frozen python`.

```sh
uv run --frozen python - <<'PY'
import csv
import json
from pathlib import Path
import numpy as np

completed = []
for path in Path("results").glob("gaussian_smoke-*/manifest.json"):
    manifest = json.loads(path.read_text())
    if manifest["status"] == "completed":
        settings = json.loads((path.parent / "resolved_config.json").read_text())
        if "controlled" in settings["comparison_modes"]:
            completed.append((manifest["completed_utc"], path.parent))
if not completed:
    raise SystemExit("Run the smoke experiment first.")
run = max(completed)[1]
config = json.loads((run / "resolved_config.json").read_text())
with np.load(run / "replication_0000.npz", allow_pickle=False) as data:
    w = data["controlled_erc_weights"]
    daily = data["evaluation"] @ w
    np.testing.assert_allclose(daily, data["controlled_erc_returns"])
    day_one_contributions = data["evaluation"][0] * w
    np.testing.assert_allclose(day_one_contributions.sum(), daily[0])
    covariance = data["estimated_covariance"]
    risk_shares = w * (covariance @ w) / (w @ covariance @ w)

loss = -daily
a = config["confidence"]
var = np.quantile(loss, a, method="inverted_cdf")
cvar = var + np.maximum(loss - var, 0).mean() / (1 - a)
with (run / "metrics.csv").open(newline="") as handle:
    row = next(r for r in csv.DictReader(handle) if
               (r["replication"], r["mode"], r["strategy"]) == ("0", "controlled", "erc"))
np.testing.assert_allclose(cvar, float(row["cvar"]), rtol=1e-10, atol=1e-12)
print(run, "controlled ERC CVaR:", cvar)
print("Daily return contributions on day 1:", day_one_contributions)
print("Shares of fitted portfolio volatility:", risk_shares)
PY
```

The first attribution is in return units and sums to that day’s return. The second is a dimensionless share of fitted volatility and sums to one. For a hand-check on the distinction, two uncorrelated assets with volatilities 1% and 2% have ERC weights ⅔ and ⅓. Each accounts for half of portfolio volatility. On a day when their returns are +1% and −2%, their weighted return contributions are +⅔% and −⅔%, yielding a realized return of zero.

## Reproduce a completed run

Use the exact resolved config, source version, and locked environment from the original run. Substitute the directories printed by the two `run` commands for the original and replay paths below.

```sh
original='results/<original run directory>'
uv run --frozen thesis-portfolio run --config "$original/resolved_config.json" --output results
replay='results/<replay run directory>'
uv run --frozen python experiments/verify_reproduction.py "$original" "$replay"
```

The checker verifies each run’s artifact hashes, requires matching resolved configs, and compares the scientific tables and numerical arrays with relative tolerance `1e-7` and absolute tolerance `1e-9`. Each replay has its own timestamps, directory, and provenance. Recording the platform and package versions helps diagnose numerical differences across environments.

## Where to read the code

The initial repository contained a title-only README and eleven empty Python files. The implementation moves the working logic into the installable `src/thesis_portfolio` package. The old empty root modules were removed, and `main.py` now forwards to the command-line interface. For example, `python main.py run --config configs/gaussian_smoke.toml` works after installing the package. Simulation, estimation, optimization, evaluation, and experiment storage each have their own module so that later volatility models can reuse the allocation and reporting code.

| Module or file | Responsibility |
| --- | --- |
| `src/thesis_portfolio/config.py` | Validate experiment settings, dimensions, units implied by the schema, and market constraints. |
| `randomness.py`, `simulation.py`, `estimation.py` in the package | Derive independent streams, sample correlated daily returns, and estimate moments from history. |
| `optimization.py`, `risk.py` | Solve the three allocation problems, validate solutions, construct realized portfolio returns, and compute risk measures. |
| `experiment.py` | `fit_portfolios` owns the historical information boundary. `run_experiment` creates the shared evaluation path, collects replications, and writes the run record. |
| `provenance.py`, `cli.py` | Snapshot the executed source and environment, hash artifacts, and expose `validate` and `run`. |
| `tests/` | Check Gaussian moments against sampling error, singular covariance behaviour, hand-computed estimates and tail metrics, reproducible random streams, solver failures, analytic portfolio solutions, and end-to-end experiment integrity. |

For validation details and extension points, see the [source package](src/thesis_portfolio), [test suite](tests), [mathematical specification](docs/model_spec.md), and [experimental protocol](docs/experiment_spec.md).

## Verification and completion record

On November 6, 2026, a fresh virtual environment installed the frozen dependency lock with Python 3.13.2. All **112 automated tests passed**. Ruff lint and formatting checks passed. A new run reproduced the original run's scientific arrays and CSV tables within the checker tolerances, and all recorded artifact hashes verified. The inspection example above was executed successfully. The 100-replication development configuration also completed, producing 600 portfolio evaluation rows across both comparison modes.

| Part 3 checklist item | Delivered implementation |
| --- | --- |
| Repository organization | Installable source package, `tests`, `configs`, `experiments`, `docs`, and generated `results` directories. |
| Reproducible environment | Python patch selection, `pyproject.toml`, frozen `uv.lock`, and a hash-pinned `requirements.txt` export. |
| Experiment configuration | Strict TOML and saved JSON validation, configurable synthetic market and sample sizes. |
| Seeded randomness | Explicit PCG64 generators with separate historical, optimization, and evaluation streams for each replication. |
| Return generation, allocation, and risk | Correlated Gaussian sampling, historical estimation, three optimizers, and daily/path risk measurements. |
| Automated checks | Analytic portfolio cases, statistical simulation checks, discrete-tail examples, failed solves, replay, and information-boundary tests. |
| Logging and saved results | Completion/failure manifests, logs, arrays, tables, solver diagnostics, source snapshots, and checksums. |
| Reproduction documentation | This guide, mathematical and experimental specifications, and the executable reproduction checker. |

Several tests address scientific errors that could otherwise produce plausible output. A counterfactual evaluation path shifts every asset's daily returns by 1 percentage point and verifies that weights stay identical while portfolio means rise and loss CVaRs fall by that amount. Another test changes the number of optimization scenarios and checks that historical and evaluation draws remain identical. Discrete-tail tests compare CVaR with direct fractional weighting of sorted losses. Diagonal covariance tests compare ERC weights with inverse-volatility allocations and minimum-variance weights with inverse-variance allocations.

The optimizers report numerical diagnostics alongside their portfolios. Negative weights within `64 × asset_count × machine_epsilon` are treated as floating-point roundoff, set to zero, and recorded. Budget and return constraints are checked again afterward. Larger negative weights, infeasible targets, and unsuccessful solves raise explicit errors. A failed run retains its manifest and traceback in the log for investigation.

The proposed scientific defaults are visible in the configuration and specifications. They include four synthetic assets, independent Markowitz risk aversion 5, 95% tail confidence, daily restoration of fixed weights, and zero transaction costs. Supervisory review and later sensitivity experiments determine the final thesis settings.

## How this leads into the thesis

Part 4 builds on the Gaussian example with larger validation runs, scenario-count convergence, comparison tables, and figures. Part 5 adds GARCH and GJR-GARCH simulation and estimation, rolling forecasts, and adaptive re-optimization on common realized paths. The trading schedule, estimation windows, and transaction-cost model need explicit definitions for that stage. Copulas and real-data validation are extensions. The robustness study uses repeated simulations and paired comparisons to quantify uncertainty across regimes.
