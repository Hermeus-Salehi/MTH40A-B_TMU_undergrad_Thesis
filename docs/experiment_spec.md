# Gaussian experiment protocol

The Part 3 experiment compares Markowitz, ERC, and CVaR using estimated inputs
and independent evaluation data. It exercises the research infrastructure
needed for the thesis on robustness and adaptability. The proposal is pending
supervisory review. The implementation equations and course references are in
the [model specification](model_spec.md).

## Development settings

The smoke configuration gives the complete asset parameters in daily decimal
units. The sample sizes and strategy settings below are provisional research
choices. Asset names are unique. The columns `replication`, `mode`, and
`strategy` are reserved for result metadata.

| Setting | Value | Use |
| --- | --- | --- |
| Market | 4 synthetic risky assets | Distinct means, volatilities, and correlations. |
| History | 1,000 days | One estimation date at the end of the history. |
| CVaR training scenarios | 2,000 one-day draws | Estimate the tail objective under the fitted model. |
| Evaluation | 252 trading days | One new ordered path in each replication. |
| Replications | 3 for the smoke run, 100 for development | Adjust the final sample size using Monte Carlo uncertainty. |
| Confidence level | 95% | One-day loss VaR and CVaR. |
| Extreme-loss threshold | 2% | Count days whose portfolio loss strictly exceeds 2%. |
| Independent Markowitz risk aversion | 5 | Sets the trade-off between fitted mean and variance. |
| Portfolio constraints | Long-only, fully invested | Identical constraints for every strategy. |
| Trading | Fixed target weights, restored daily at zero cost | Each portfolio chooses weights once, before evaluation. |

The proposal's 21-day interval applies to future re-optimization. Rebalancing
restores the weights already chosen. Re-optimization estimates new target
weights from the information available at a later decision date.

## Procedure for each replication

1. Derive three random streams from the root seed, replication index, and
   stable identifiers for history, optimization, and evaluation.
2. Generate historical returns under the configured true Gaussian market. Fit
   means and covariance from those observations.
3. Generate CVaR training scenarios from the fitted Gaussian model using the
   optimization stream. The information used for allocations ends here.
4. Construct the portfolios for each comparison group below.
5. Draw a new path from the true Gaussian market with the evaluation stream.
   Apply every portfolio to this same path in the same order.
6. Save the inputs, fitted moments, weights, returns, solver diagnostics, and
   metrics. Record run status, configuration, seeds, code provenance, environment,
   and artifact checksums.

The evaluation draws are independent of history and of optimization randomness.
The optimization draws depend on history through the fitted moments. Changing
the number of optimization draws preserves the market history and evaluation
path. Extending a run to more replications preserves earlier stream identities.

## Comparisons

**Independent strategies.** Markowitz minimizes its configured mean-variance
objective, ERC balances volatility contributions, and CVaR minimizes scenario
tail loss. Each uses its own objective under the common constraints. Compare
realized returns alongside risk measures because fitted returns can differ.

**Controlled fitted expected returns.** Compute ERC once and take its estimated
mean return as the common target `m = mu_hat' w_ERC`. Solve minimum variance and
minimum CVaR with the equality `mu_hat' w = m` in both problems. The ERC weights
satisfy that equality by construction, so both problems have a feasible
allocation. Use the historical estimate of expected returns for both, including
CVaR. The achieved fitted returns should agree within solver tolerance. Each
portfolio's future realized return is measured separately.

This design gives a Gaussian validation check. If fitted returns are exactly
matched, each Gaussian portfolio has the same mean loss. Its population CVaR
is that mean plus a positive constant times its volatility. Thus the population
CVaR and variance problems have the same minimizer in this comparison. The
finite-scenario CVaR portfolio approximates the population solution. A large
weight discrepancy warrants a scenario-count sensitivity check.

## Reading and reproducing results

A completed manifest identifies an individual run and gives checksums for the
files needed to audit it. A failed or interrupted run has a different status.
Use the resolved configuration and recorded environment to rerun the experiment.
Check numeric arrays, weights, and metrics for agreement within numerical
tolerance. Timestamps and paths identify separate executions.

The Monte Carlo standard error in the summary measures variation of each
reported statistic across independent replications. For a difference between
strategies, calculate one difference per shared path and estimate uncertainty
across those paired differences. At 95% confidence, 252 days offer only 12.6
observations' worth of empirical tail mass per replication. The smoke run
checks execution. Final comparisons need an accuracy criterion, adequate
replication count, and sensitivity analysis. If tuning is introduced, use
separate validation observations and reserve the evaluation data for scoring.

## Completion and later stages

Part 3 is complete when a reader can install the project in a clean
environment, run the tests, reproduce a small Gaussian comparison, and trace
its outputs to source code and inputs. Acceptance checks cover the following.

- Numerical tests exercise analytic allocations, feasibility, risk shares,
  and hand-computable tail losses.
- End-to-end tests replay fitted scenarios, recompute out-of-sample metrics
  from the saved shared path, and check matching of fitted expected returns.
- Changes confined to the evaluation observations leave the weights unchanged.
- Results reproduce from recorded settings and streams. Failed solves and
  invalid return paths produce an explicit failure record.

Part 4 expands Gaussian validation and the baseline comparisons. Part 5 adds
GARCH/GJR-GARCH volatility, estimation, forecasting, and rolling evaluation.
The robustness experiment will carry fixed Gaussian targets into new market
regimes. The adaptability experiment will compare fixed targets, periodically
updated Gaussian targets, and portfolios updated with volatility forecasts.
Use the same history cutoffs, trading schedule, realized paths, and cost
assumptions within each comparison.

Future specifications should settle estimation windows, forecast and evaluation
horizons, market calibration, transaction costs, and Monte Carlo accuracy.
Compare unconditional moments across regimes empirically. Matching innovation
correlations alone may yield different unconditional return correlations under
asset-specific dynamic volatilities. Copulas and real-market validation are
extensions under the research checklist.
