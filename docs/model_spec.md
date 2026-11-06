# Mathematical implementation specification

The Part 3 baseline compares three portfolios in an IID Gaussian market.
This specification records the model used by the example. Supervisor review,
final experiment settings, and later model stages are tracked in the
[experiment specification](experiment_spec.md).

## Return units and information available to each component

A row contains one day's arithmetic simple returns in decimal units. A value of
`0.01` means a 1% gain. Columns follow the configured order of assets.

| Component | Inputs | Outputs |
| --- | --- | --- |
| Market simulator | Generating mean, covariance, seed stream, observation count | Historical or evaluation returns |
| Estimator | Historical returns through the allocation date | Sample mean and covariance |
| CVaR scenario generator | Fitted mean and covariance, separate seed stream | Equally probable optimization scenarios |
| Optimizers | Fitted moments or fitted-model scenarios, portfolio constraints | Weights, objective, solver diagnostics |
| Evaluator | Selected weights and a new shared market path | Realized returns and metrics |

Each portfolio satisfies `w_i >= 0` and `sum(w) = 1`. Its daily return is
`r_p = r @ w` and its loss is `L = -r_p`. Positive loss means a decline.
Negative loss values denote gains. The confidence level `a = 0.95` selects
the upper 5% of the loss distribution.

## Gaussian simulation and estimation

Let `D` be the diagonal matrix of daily volatilities and `C` the correlation
matrix. The Gaussian simulator uses

```text
Sigma = D C D
r_t   = mu + A z_t,     z_t ~ N(0,I), independently across t
A A'  = Sigma.
```

The simulator checks finiteness, symmetry, and positive semidefiniteness.
For `Sigma = Q diag(lambda) Q'`, it uses the symmetric principal square root
`A = Q diag(sqrt(max(lambda, 0))) Q'`. This factor supports singular
covariances and is invariant to eigenvector signs and choice of basis within
repeated eigenspaces, up to roundoff. Material negative eigenvalues raise an
error. The three-strategy experiment requires a positive definite covariance
for ERC and enough historical observations for full-rank estimation.

Given `H` historical observations, the allocations depend exclusively on the
fitted moments

```text
mu_hat    = sum_t r_t / H
Sigma_hat = sum_t (r_t - mu_hat)(r_t - mu_hat)' / (H-1).
```

CVaR uses scenarios drawn from `N(mu_hat, Sigma_hat)`. More scenarios improve
the numerical approximation under a given historical fit. More historical
observations address a separate source of error in the fitted moments.

## Markowitz mean-variance

In the independent comparison, solve

```text
minimize   (gamma/2) w' Sigma_hat w - mu_hat' w
subject to w >= 0, sum(w) = 1.
```

The configurable risk aversion `gamma > 0` controls the variance penalty
relative to estimated daily return. Its numerical meaning depends on return
units. The example uses `gamma = 5`.

With a specified target return `m`, minimize `w' Sigma_hat w` under the
same portfolio constraints and a return constraint. The API accepts either a
minimum (`mu_hat' w >= m`) or equality (`mu_hat' w = m`). The controlled
comparison uses equality. [Markowitz, *Portfolio Selection* (1952)](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).

## Equal risk contribution

The contribution of asset `i` to portfolio volatility is

```text
sigma_p = sqrt(w' Sigma_hat w)
RC_i    = w_i (Sigma_hat w)_i / sigma_p
sum_i RC_i = sigma_p.
```

ERC chooses a portfolio with `RC_i / sigma_p = 1/n` for each of the `n`
assets. The solver takes a positive scalar multiple `C` of the fitted
covariance for numerical conditioning. It minimizes
`x' C x / 2 - sum_i log(x_i) / n` over positive `x` using damped Newton steps.
The first-order conditions give `x_i (Cx)_i = 1/n`. Normalizing
`w = x / sum(x)` preserves equal risk shares. The solver checks the maximum
absolute deviation from the required share. With a positive diagonal
covariance, the analytic weights are proportional to inverse asset volatility.
[Maillard, Roncalli, and Teiletche, *On the Properties of Equally-Weighted Risk Contributions Portfolios*](https://www.thierry-roncalli.com/download/erc.pdf).

## Scenario CVaR

For `M` equally probable training scenarios `s_j`, solve

```text
minimize   eta + sum_j u_j / (M (1-a))
subject to u_j >= -s_j' w - eta,  u_j >= 0,
           w >= 0,  sum(w) = 1.
```

The threshold `eta` can take either sign. In controlled mode the solver also
enforces `mu_hat' w = m` using the historical mean estimate. Independent mode
uses the portfolio constraints alone. The API also supports the minimum-return
inequality. The solver implements the sparse linear program and checks its
objective against the empirical CVaR at the returned weights. A numerical or
feasibility failure stops the run. [Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk* (2000)](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf).

## Evaluation metrics

For `T` out-of-sample losses and empirical distribution `F_T`,

```text
VaR_a  = inf {x : F_T(x) >= a}
CVaR_a = VaR_a + sum_t max(loss_t - VaR_a, 0) / (T (1-a)).
```

This assigns exactly the required upper-tail mass, including fractional mass
from observations tied at the boundary. For instance, losses of 1%, 2%, 2%,
and 8% at 62.5% confidence have VaR 2%. The tail spans 1.5 observations, so
CVaR is `(8% + 0.5 * 2%) / 1.5 = 6%`.
[Rockafellar and Uryasev, *Conditional Value-at-Risk for General Loss Distributions* (2002)](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf).

| Output | Definition |
| --- | --- |
| `estimated_return` | Historical `mu_hat' w`. Use this to check the controlled target. |
| `mean_return` | Arithmetic mean of daily realized returns on the evaluation path. |
| `volatility` | Daily sample standard deviation with denominator `T-1`. |
| `var`, `cvar` | One-day loss measures at the configured confidence level. |
| `loss_frequency` | Fraction of days with `-r_p > loss_threshold`. All portfolios use the same threshold. |
| `annualized_mean` | Daily arithmetic mean times the number of periods per year. |
| `annualized_volatility` | Daily volatility times the square root of periods per year, using the IID scaling convention. |
| `cumulative_return` | `product_t(1 + r_p,t) - 1` along the ordered path. |
| `max_drawdown` | Largest `1 - W_t / max_(0<=s<=t) W_s`, including initial wealth `W_0 = 1`. |

The evaluator restores fixed target weights every day at zero cost. It
calculates path metrics in the chronological order of the simulated days.
Gaussian simple returns are unbounded. The standalone generator returns draws
from the specified law. In a research run, any asset return at or below -100%
in the historical, optimization, or evaluation sample fails the run. The wealth
summary independently rejects portfolio returns at or below that boundary.

## Course-note cross-check

The course source is Alexey Rubtsov's *Portfolio Optimization (with MATLAB
examples)*, supplied as *Portfolio theory.pdf*. These section references identify
the formulas reviewed for the baseline.

| Course location | Correspondence |
| --- | --- |
| Section 1.1, Eqs. (1.1)–(1.2) | Portfolio mean and variance. |
| Section 1.3, Eq. (1.16), p. 7 | Minimum variance at a target expected return with long-only, fully invested weights. |
| Section 2, p. 9 | Estimation from historical data and simulation using fitted moments. |
| Section 4, Eqs. (4.2)–(4.8), pp. 12–13 | Euler contributions to volatility and equal risk under budget and nonnegativity constraints. |
| Section 5, Eqs. (5.3)–(5.6), p. 16 | Scenario CVaR minimization with budget, nonnegativity, and minimum-return constraints. |

The example makes two specific design choices. Independent Markowitz uses the
utility objective and configurable risk aversion above. Controlled CVaR
tightens the notes' minimum-return constraint to equality at ERC's fitted mean
return. The primary CVaR papers above specify the loss sign, normalization, and
discrete boundary calculation used for implementation of the notes' Section 5.
