# Active annual research design

The active experiment follows the supplied **M4,5 revisions V2.0** and **Hybrid
Architecture Instructions**, dated 27 November 2026. The original daily M4
experiment remains a separately runnable mathematical benchmark.

For each oracle market and seed, generate one chronological multivariate history.
Initialize variance at its long-run value and retain every observation. Pass an
`ObservedHistory` and `PublicSpec` to the estimator; neither object contains true
parameters, latent state, evaluation data, or oracle random seeds.

Estimate Gaussian daily moments or fit constant-mean GARCH/GJR marginal conditional
Gaussian likelihoods. Estimate constant cross-asset innovation correlation from
aligned fitted standardized residuals. This is marginal MLE followed by residual
correlation estimation, not joint multivariate MLE. The numerical implementation
uses an observed sample-variance initializer, includes the first observation in the
conditional likelihood, and scales each series by its observed standard deviation.
Means and residuals are converted back by that scale; intercepts and variance
states by its square. Deterministic starts do not use oracle truth. Failed fits
stop integrated runs; calibration tables retain failures.

Generate 252-day fitted futures from the fitted terminal state. Aggregate each
asset separately using `product(1 + daily_return) - 1`. Estimate annual means and
covariance from that terminal scenario matrix. Each strategy chooses initial
capital weights once; its annual return is the dot product with asset terminal
returns. There is no daily restoration of weights and no re-optimization.

Solve exact ERC first under a positive-definite annual covariance, positive equal
risk budgets, and the standard long-only unit-budget constraints. These conditions
give a unique normalized risk-budget allocation. Its fitted annual expected return
is therefore the maximum over the singleton exact ERC feasible set. Use this
model/run-specific value as an **equality**, not a floor, for minimum variance and
minimum CVaR. No external universal target or approximate ERC substitution is used.

Freeze all weights and forecast risk thresholds before generating oracle futures.
Those futures continue from the true endpoint with fresh shocks. All strategies
and fitted families within an oracle run use the same held-out terminal scenarios.
Targets differ between fitted families and histories; report those differences
alongside realized return and risk. Conditional exception intervals describe future
Monte Carlo uncertainty given one history, not parameter uncertainty across histories.

The initial grid comprises Gaussian-on-Gaussian, GARCH-on-GARCH, GJR-on-GJR, and a
Gaussian forecaster on each dynamic-market history. Normal, volatility-only,
correlation-only, and combined stress cases are implemented separately. Full
cross-family grids, copulas, rebalancing, and the M7 replication campaign are deferred.

## Provisional choices and scientific limits

- The scaled five-asset pilot and correlation 0.20 are coding defaults, not a
  calibrated five-stock market. Source dimensionless dynamics are preserved.
- The source-exact fixtures use GARCH `(0.0001, 0.20, 0.50, 0)` and GJR
  `(0.02, 0.03, 0.90, 0.06)`, transcribed from the supplied methodology. Their
  original model units are kept separate from investable decimal-return paths.
- The scaled GARCH and GJR pilots have persistence 0.70 and 0.96 respectively.
  Equal target long-run variance does not isolate asymmetry or match full
  distributions or unconditional return correlations.
- The recovery recipe studies nested 300/600/1000/1200-observation prefixes.
  Adequacy criteria were not supplied. The integration reference remains explicitly
  provisional at N=1000; no successful calibration is claimed. Smoke uses N=300.
- Pilot path counts are 10000 forecast and 50000 evaluation years. Separate
  convergence checks compare forecast and evaluation counts without oracle tuning.
- Source-exact recovery uses one prespecified calibration seed by default. It is
  illustrative evidence, not replicated inference or a guarantee of recovery.
- The likelihood reports parameter estimates, optimizer attempts and boundary
  flags. Standard errors are not reported; boundary-aware inference remains a
  limitation rather than an invented Hessian-based precision claim.
- Residual Ljung–Box p-values are descriptive chi-square references. Fitted-parameter
  adjustments and multiple-testing acceptance claims are not made.
- Gaussian daily simple returns can be invalid for positive wealth. Integrated
  runs stop on any return at or below -1 or a nonfinite value; they never clip,
  delete, or resample the offending path.
- Daily analytical Gaussian tail formulas remain daily tests. They are not used
  as theoretical benchmarks for compounded annual portfolio losses.

The code is complete only to its tested numerical scope. Scientific selection of
N, final counts, source mapping, and inferential thresholds remains researcher work.
