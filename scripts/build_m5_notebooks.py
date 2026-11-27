"""Author the research recipes and a traceable legacy-task-to-hybrid map.

This is an authoring utility, not a second implementation of the mathematics.
"""

import csv
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELL_RECORDS = []


class Recipe:
    def __init__(self, relative, title, question, context):
        self.relative = relative
        self.cells = [
            nbf.v4.new_markdown_cell(
                f"# {title}\n\n**Research question.** {question}\n\n{context}\n\n"
                "This recipe runs independently in the installed project environment. The default is a labelled smoke experiment; "
                "set `THESIS_PROFILE=pilot` before starting the kernel for the configured pilot counts. "
                "Source functions are reusable and directly linked below. Outputs report actual calculations, not assumed findings."
            )
        ]

    def heading(self, title, text=""):
        self.cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{text}"))

    def cell(self, sid, title, purpose, inputs, outputs, code, source="", aliases=""):
        prose = f"### {title}\n\n{purpose}\n\n**Inputs:** {inputs}.  \n**Created / outputs:** {outputs}."
        if source:
            prose += f"  \n**Canonical implementation:** `{source}`."
        self.cells.append(nbf.v4.new_markdown_cell(prose))
        self.cells.append(
            nbf.v4.new_code_cell(
                f"# {sid} — {title}\n# Inputs: {inputs}\n# Created: {outputs}\n" + code + "\n",
                metadata={"tags": [sid], "legacy_aliases": aliases.split(",") if aliases else []},
            )
        )
        CELL_RECORDS.append(
            {
                "stable_id": sid,
                "notebook": self.relative,
                "cell_tag": sid,
                "source": source,
                "legacy_task_groups": aliases,
                "purpose": title,
            }
        )

    def save(self):
        notebook = nbf.v4.new_notebook(
            cells=self.cells,
            metadata={
                "kernelspec": {
                    "display_name": "Python (thesis)",
                    "language": "python",
                    "name": "thesis",
                },
                "language_info": {"name": "python"},
            },
        )
        nbf.validate(notebook)
        nbf.write(notebook, ROOT / self.relative)


IMPORTS = """import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from thesis_portfolio.annual.config import PublicSpec, load_run_config, oracle_state, project_root, stream
from thesis_portfolio.annual.simulation import generate_history, simulate_conditional_paths, oracle_continue
from thesis_portfolio.annual.estimation import fit_market_model
from thesis_portfolio.annual.portfolio import annual_moments, freeze_portfolios, evaluate_portfolios
from thesis_portfolio.optimization import equal_risk_contribution, markowitz, cvar, risk_contributions
from thesis_portfolio.annual.diagnostics import mathematical_backstops, residual_diagnostics, parameter_comparison, recurrence_report
from thesis_portfolio.annual.experiment import execution_profile, begin_run, save_run
from thesis_portfolio.annual import visualization as plots"""


def model_recipe(family, stem):
    prefix = f"M5.{family.upper()}" if family != "gaussian" else "M4A.GAUSSIAN"
    r = Recipe(
        f"notebooks/modelling/{stem}.ipynb",
        f"{family.upper()} annual portfolio research",
        "How do the three portfolio risk objectives behave at the exact ERC expected return under this fitted model?",
        "The market supplies one observed history. Forecasting sees that history only; true parameters and terminal state are reserved for the evaluator. "
        "Annual terminal asset returns describe one allocation held for 252 days. The original daily Gaussian notebook is preserved separately.",
    )
    r.heading(
        "1. Reproducible specification",
        "The 1000-observation pilot reference is provisional. Source-exact recovery lives in a separate research recipe; no history length is declared scientifically adequate without agreed criteria.",
    )
    r.cell(
        prefix + ".SETUP.01",
        "Import the shared research machinery",
        "Imports load definitions without running another notebook.",
        "installed package",
        "named functions",
        IMPORTS,
        "thesis_portfolio.annual",
        "S",
    )
    r.cell(
        prefix + ".SETUP.02",
        "Load the run configuration",
        "Public fitting options are separated from the privileged oracle specification.",
        "project installation, profile",
        "root, cfg",
        "root = project_root(os.environ.get('THESIS_PROJECT_ROOT'))\ncfg = execution_profile(load_run_config(root / 'configs/annual/pilot.yaml'), os.environ.get('THESIS_PROFILE', 'smoke'))\ndisplay({k:v for k,v in cfg.items() if k not in ('oracle','seed')})",
        "annual.config.load_run_config",
        "S",
    )
    r.cell(
        prefix + ".SETUP.03",
        "Open a new evidence record",
        "A unique directory begins incomplete and becomes completed only after artifacts are saved.",
        "root, cfg",
        "run_path",
        f"oracle_family = '{family}'\nrun_path = begin_run(root, cfg, oracle_family)\ndisplay(run_path)",
        "annual.experiment.begin_run",
        "103",
    )
    r.heading(
        "2. One observed timeline",
        "The oracle is a synthetic market, not an investment decision-maker. No burn-in prefix is discarded. The first future day will continue from the final historical residual and variance.",
    )
    r.cell(
        prefix + ".HISTORY.01",
        "Construct the privileged oracle",
        "Source dimensionless dynamics are preserved; the intercept is scaled to the provisional daily variance.",
        "private configuration",
        "oracle_parameters",
        "oracle_parameters = oracle_state(cfg, oracle_family)",
        "annual.config.oracle_state",
        "69,70,71,72,81,82",
    )
    r.cell(
        prefix + ".HISTORY.02",
        "Generate one chronological history",
        "Each row is one aligned observation across assets; latent state remains in the private record.",
        "oracle_parameters, history stream",
        "private_history",
        "private_history = generate_history(oracle_parameters, cfg['history']['observations'], stream(cfg['seed'], f'{oracle_family}/history'))",
        "annual.simulation.generate_history",
        "74,85",
    )
    r.cell(
        prefix + ".HISTORY.03",
        "Expose observations only",
        "This is the complete information interface seen by the estimator.",
        "private_history observation export",
        "history",
        "history = private_history.observed\ndisplay(pd.DataFrame(history.returns, columns=history.assets).head())\nprint('Observed shape:', history.returns.shape)",
        "annual.models.ObservedHistory",
        "75,87,88,89",
    )
    r.heading(
        "3. Estimate the forecasting model",
        "The conditional Gaussian likelihood sums log variance and squared standardized residuals. Variance initialization and numerical scaling use observations only. GARCH/GJR are fitted marginally by MLE; constant cross-asset correlation is then estimated from aligned standardized residuals. No oracle-assisted optimizer is present.",
    )
    r.cell(
        prefix + ".ESTIMATION.01",
        "Fit the observed history",
        "All fitted means, variance coefficients, filtered states, and correlation come from observations.",
        "history, public family/options",
        "fitted",
        "public_spec = PublicSpec(oracle_family, **cfg['fit'])\nfitted = fit_market_model(history, public_spec)",
        "annual.estimation.fit_market_model",
        "91",
    )
    r.cell(
        prefix + ".ESTIMATION.02",
        "Inspect parameter estimates",
        "Boundary estimates are retained and flagged; source truth is not used to repair them.",
        "fitted",
        "parameter table and convergence diagnostics",
        "display(fitted.parameters)\ndisplay(pd.DataFrame(fitted.diagnostics))",
        "annual.estimation.FittedMarket",
        "91",
    )
    r.cell(
        prefix + ".ESTIMATION.03",
        "Inspect estimated dependence",
        "The forecast uses one constant correlation matrix estimated from standardized residuals. Conditional covariance still changes with volatility.",
        "fitted",
        "estimated correlation and eigenvalues",
        "display(pd.DataFrame(fitted.state.correlation, index=history.assets, columns=history.assets))\nprint(np.linalg.eigvalsh(fitted.state.correlation))",
        "annual.estimation.fit_market_model",
        "90,91",
    )
    r.cell(
        prefix + ".DIAGNOSTICS.01",
        "Measure residual dependence",
        "ACF and Ljung–Box references are descriptive. Fitted-parameter adjustments are not claimed, and no seed is discarded to obtain a favorable p-value.",
        "history, fitted",
        "residual_report",
        'residual_report = residual_diagnostics(history, fitted)\ndisplay(residual_report.drop_duplicates(["asset","series","squared"])[["asset","series","squared","reference_p"]])',
        "annual.diagnostics.residual_diagnostics",
        "78,79,91",
    )
    r.cell(
        prefix + ".DIAGNOSTICS.02",
        "Plot volatility dependence",
        "Compare squared observed returns with squared fitted standardized residuals.",
        "residual_report",
        "ACF figure",
        "acf_plot = plots.acf_figure(residual_report, history.assets[0])\nplt.show()",
        "annual.visualization.acf_figure",
        "78,80,86,102",
    )
    r.heading(
        "4. Conditional annual futures",
        "For each scenario and asset, $X_i=\\prod_{t=1}^{252}(1+r_{t,i})-1$. Portfolio terminal return is $Xw$. Compounding weighted daily returns would instead impose daily rebalancing and is excluded.",
    )
    r.cell(
        prefix + ".FORECAST.01",
        "Check the formation-date state",
        "The first future variance is computed from the last fitted residual and its associated variance, not reset to a long-run value.",
        "fitted endpoint",
        "first_variance",
        'first_variance = fitted.state.next_variance\ndisplay(pd.Series(first_variance, index=history.assets, name="First future daily variance"))',
        "annual.models.MarketState.next_variance",
        "92",
    )
    r.cell(
        prefix + ".FORECAST.02",
        "Generate and aggregate annual futures",
        "The batched simulator retains annual terminal scenarios and a small path subset. It stops on invalid simple returns rather than clipping or resampling.",
        "fitted state, fresh forecast stream",
        "forecast",
        "sim = cfg['simulation']\nforecast = simulate_conditional_paths(fitted.state, 252, sim['optimization_paths'], stream(cfg['seed'], f'{oracle_family}/forecast/{oracle_family}'), batch_size=sim['batch_size'], retain=sim['retained_paths'])\nprint('Terminal scenarios:', forecast.terminal.shape)",
        "annual.simulation.simulate_conditional_paths",
        "93,94",
    )
    r.cell(
        prefix + ".FORECAST.03",
        "Estimate common annual moments",
        "All three strategies receive the same annual mean and covariance estimated from these fitted terminal scenarios.",
        "forecast.terminal",
        "annual_mean, annual_covariance",
        'annual_mean, annual_covariance = annual_moments(forecast.terminal)\ndisplay(pd.Series(annual_mean, index=history.assets, name="Fitted annual mean"))\ndisplay(pd.DataFrame(annual_covariance, index=history.assets, columns=history.assets))',
        "annual.portfolio.annual_moments",
        "93,94",
    )
    r.cell(
        prefix + ".FORECAST.04",
        "Check tail-scenario mass",
        "More future scenarios improve numerical precision; they do not add historical information. Smoke tail counts are deliberately small.",
        "forecast counts, confidence levels",
        "tail count table",
        "display(pd.DataFrame({'confidence':cfg['risk']['reporting'], 'forecast_tail_mass':[len(forecast.terminal)*(1-q) for q in cfg['risk']['reporting']]}))",
        "annual.simulation.FutureSample",
        "94",
    )
    r.cell(
        prefix + ".FORECAST.05",
        "Show conditional future paths",
        "Quantiles in this fan come only from the explicitly labelled retained path subset. Full-sample tail metrics are computed separately.",
        "forecast",
        "fan_plot",
        "fan_plot = plots.fan_figure(forecast, oracle_family)\nplt.show()",
        "annual.visualization.fan_figure",
        "102",
    )
    r.heading(
        "5. Exact ERC first, then equal expected return",
        "With positive-definite covariance and standard long-only unit-budget constraints, the exact ERC allocation is unique. Its fitted annual mean is therefore the maximum over that singleton exact ERC set. It becomes the equality target for the other objectives; it is not a universal user-entered percentage.",
    )
    r.cell(
        prefix + ".ERC.01",
        "Solve exact annual ERC",
        "The canonical convex risk-budgeting solver is reused without approximation.",
        "annual_covariance",
        "erc",
        "erc = equal_risk_contribution(annual_covariance)\ndisplay(pd.Series(erc.weights, index=history.assets))",
        "thesis_portfolio.optimization.equal_risk_contribution",
        "95",
    )
    r.cell(
        prefix + ".ERC.02",
        "Verify exact risk shares",
        "Euler contributions must sum to portfolio volatility and each normalized share must equal 1/A.",
        "erc, annual_covariance",
        "risk_shares",
        "contributions = risk_contributions(erc.weights, annual_covariance)\nrisk_shares = contributions / np.sqrt(erc.weights @ annual_covariance @ erc.weights)\nnp.testing.assert_allclose(risk_shares, 1/len(history.assets), atol=1e-8)\ndisplay(pd.Series(risk_shares, index=history.assets))",
        "thesis_portfolio.optimization.risk_contributions",
        "95,98",
    )
    r.cell(
        prefix + ".TARGET.01",
        "Freeze the model-specific fitted target",
        "Equal fitted return controls the within-model strategy comparison; realized means need not match.",
        "annual_mean, erc",
        "r_star",
        'r_star = float(annual_mean @ erc.weights)\nprint(f"ERC-derived annual target: {r_star:.6%}")',
        "annual.portfolio.freeze_portfolios",
        "95",
    )
    r.cell(
        prefix + ".MARKOWITZ.01",
        "Minimize variance at the ERC target",
        "The target is an equality, including when it lies below the GMV expected return.",
        "annual moments, r_star",
        "mv",
        "mv = markowitz(annual_mean, annual_covariance, target_return=r_star, target_constraint='equal')\ndisplay(mv.diagnostics)",
        "thesis_portfolio.optimization.markowitz",
        "96",
    )
    r.cell(
        prefix + ".CVAR.01",
        "Minimize annual scenario CVaR",
        "The sparse linear program uses one loss per simulated year and the identical annual mean in its return equality.",
        "forecast.terminal, annual_mean, r_star",
        "tail",
        "tail = cvar(forecast.terminal, confidence=cfg['risk']['optimization_confidence'], mean=annual_mean, target_return=r_star, target_constraint='equal')\ndisplay(tail.diagnostics)",
        "thesis_portfolio.optimization.cvar",
        "97",
    )
    r.cell(
        prefix + ".FREEZE.01",
        "Validate and freeze the comparison",
        "Check exact ERC, all budget/return constraints, and that optimized in-sample objectives do not exceed feasible ERC objectives. Freeze forecast thresholds before evaluation.",
        "three solutions, forecast scenarios",
        "frozen",
        "solutions = {'ERC':erc, 'Markowitz':mv, 'CVaR':tail}\nfrozen = freeze_portfolios(solutions, annual_mean, annual_covariance, forecast.terminal, cfg['risk']['reporting'])\ndisplay(frozen.constraints)\ndisplay(frozen.forecast_metrics)",
        "annual.portfolio.freeze_portfolios",
        "98",
    )
    if family != "gaussian":
        r.heading(
            "6. Gaussian comparator on the same observations",
            "This changes the forecasting model while keeping the oracle history and eventual evaluation futures identical. Its ERC-derived target may differ, so the comparison reports risk together with return.",
        )
        r.cell(
            prefix + ".COMPARATOR.01",
            "Fit the Gaussian comparator",
            "Daily sample moments are estimated from exactly the same observations.",
            "history",
            "gaussian_fit",
            "gaussian_fit = fit_market_model(history, PublicSpec('gaussian'))\ndisplay(gaussian_fit.parameters)",
            "annual.estimation.fit_market_model",
            "89,94,101",
        )
        r.cell(
            prefix + ".COMPARATOR.02",
            "Generate Gaussian annual forecasts",
            "The comparator also compounds 252 daily asset returns before optimization.",
            "gaussian_fit, independent forecast stream",
            "gaussian_forecast",
            "gaussian_forecast = simulate_conditional_paths(gaussian_fit.state, 252, sim['optimization_paths'], stream(cfg['seed'], f'{oracle_family}/forecast/gaussian'), batch_size=sim['batch_size'], retain=sim['retained_paths'])",
            "annual.simulation.simulate_conditional_paths",
            "94",
        )
        r.cell(
            prefix + ".COMPARATOR.03",
            "Optimize and freeze the comparator",
            "This helper follows the same exact-ERC-first sequence demonstrated above. It accepts terminal forecasts only.",
            "gaussian_forecast",
            "gaussian_frozen",
            "from thesis_portfolio.annual.portfolio import optimize_portfolios\ngaussian_frozen = optimize_portfolios(gaussian_forecast.terminal, cfg['risk']['optimization_confidence'], tuple(cfg['risk']['reporting']))\ndisplay(gaussian_frozen.constraints)",
            "annual.portfolio.optimize_portfolios",
            "95,96,97,98",
        )
    r.heading(
        "7. Held-out oracle evaluation",
        "Only now are fresh oracle future shocks drawn. Every strategy and fitted family in this run receives the same terminal scenarios. These are independent conditional future years, not successive years to compound together.",
    )
    r.cell(
        prefix + ".EVALUATION.01",
        "Draw shared conditional oracle futures",
        "The evaluator continues from the true endpoint; no oracle input was used by the fitter or optimizer.",
        "private history, fresh evaluation stream, frozen allocations",
        "evaluation",
        "evaluation = oracle_continue(private_history, 252, sim['evaluation_paths'], stream(cfg['seed'], f'{oracle_family}/evaluation'), batch_size=sim['batch_size'], retain=sim['retained_paths'])\nprint('Held-out annual scenarios:', evaluation.terminal.shape)",
        "annual.simulation.oracle_continue",
        "99",
    )
    r.cell(
        prefix + ".EVALUATION.02",
        "Measure annual risk and forecast exceptions",
        "Exception uncertainty is conditional on this history and these fixed weights. It excludes estimation uncertainty across histories.",
        "frozen, evaluation.terminal",
        "metrics",
        "metrics = evaluate_portfolios(frozen, evaluation.terminal)\ndisplay(metrics)",
        "annual.portfolio.evaluate_portfolios",
        "100",
    )
    if family != "gaussian":
        r.cell(
            prefix + ".EVALUATION.03",
            "Compare forecasts on common oracle outcomes",
            "Model-specific fitted targets are displayed explicitly; this is not an equal-target comparison between fitted families.",
            "gaussian_frozen, shared evaluation",
            "gaussian_metrics, comparison",
            "gaussian_metrics = evaluate_portfolios(gaussian_frozen, evaluation.terminal)\ncomparison = pd.concat([metrics.assign(forecast_family=oracle_family), gaussian_metrics.assign(forecast_family='gaussian')])\ndisplay(comparison[['forecast_family','strategy','confidence','fitted_target','oracle_mean','oracle_cvar']])",
            "annual.portfolio.evaluate_portfolios",
            "101",
        )
    r.heading(
        "8. Evidence and interpretation",
        "The following truth overlays belong to the researcher/evaluator layer and are produced after allocations are frozen. They are not additional fitting inputs.",
    )
    r.cell(
        prefix + ".VALIDATION.01",
        "Check recurrence identities",
        "A deterministic recursion failure is an implementation error; finite-sample parameter discrepancy is a separate research finding.",
        "private_history",
        "validation reports",
        "display(mathematical_backstops())\ndisplay(recurrence_report(private_history))",
        "annual.diagnostics",
        "73,76,83,84,86",
    )
    r.cell(
        prefix + ".VALIDATION.02",
        "Compare true and fitted parameters",
        "The comparison shows finite-history estimation error without using truth to choose or repair the fitted model.",
        "private_history, fitted",
        "truth_comparison",
        "truth_comparison = parameter_comparison(private_history, fitted)\ndisplay(truth_comparison)",
        "annual.diagnostics.parameter_comparison",
        "77,86,102",
    )
    r.cell(
        prefix + ".FIGURE.01",
        "Overlay fitted and oracle historical volatility",
        "The evaluator can inspect where the filtered state tracks or misses the known synthetic state.",
        "history, fitted, private_history",
        "history_plot",
        "history_plot = plots.history_figure(history, fitted, private_history)\nplt.show()",
        "annual.visualization.history_figure",
        "80,86,102",
    )
    if family == "gjr":
        r.cell(
            prefix + ".FIGURE.02",
            "Show the fitted negative-shock response",
            "At a fixed previous variance, the negative-minus-positive response is gamma times squared shock magnitude.",
            "fitted.state",
            "news_plot",
            "news_plot = plots.news_impact_figure(fitted.state)\nplt.show()",
            "annual.visualization.news_impact_figure",
            "84,102",
        )
    r.cell(
        prefix + ".FIGURE.03",
        "Show allocations and the annual frontier",
        "The frontier is a diagnostic; it does not change the chosen ERC equality target.",
        "frozen, asset labels",
        "portfolio_plot",
        "portfolio_plot = plots.portfolio_figure(frozen, history.assets)\nplt.show()",
        "annual.visualization.portfolio_figure",
        "96,102",
    )
    r.cell(
        prefix + ".FIGURE.04",
        "Compare forecast and oracle annual losses",
        "Annual loss distributions are empirical compounded outcomes, so daily Gaussian tail formulas are not used here.",
        "frozen, forecast, evaluation",
        "loss_plot",
        "loss_plot = plots.loss_figure(frozen, forecast, evaluation, oracle_family)\nplt.show()",
        "annual.visualization.loss_figure",
        "102",
    )
    r.cell(
        prefix + ".INTERPRETATION.01",
        "Summarize this run without claiming a winner",
        "Equal fitted means are a controlled input, whereas oracle means and risk measures are measured outcomes. Small smoke tail counts do not establish a scientific ranking.",
        "metrics, frozen",
        "observed summary",
        "print(f'Largest fitted-return equality error: {abs(frozen.constraints.return_residual).max():.3g}')\nprint(f'Oracle mean range: {metrics.oracle_mean.min():.2%} to {metrics.oracle_mean.max():.2%}')\nprint('History selection:', cfg['history']['selection_status'])\nprint('This run does not certify calibration adequacy or an across-history strategy ranking.')",
        "annual.portfolio.evaluate_portfolios",
        "100,101",
    )
    fits = (
        "{oracle_family:fitted}"
        if family == "gaussian"
        else "{oracle_family:fitted, 'gaussian':gaussian_fit}"
    )
    forecasts = (
        "{oracle_family:forecast}"
        if family == "gaussian"
        else "{oracle_family:forecast, 'gaussian':gaussian_forecast}"
    )
    ports = (
        "{oracle_family:frozen}"
        if family == "gaussian"
        else "{oracle_family:frozen, 'gaussian':gaussian_frozen}"
    )
    metrics = (
        "{oracle_family:metrics}"
        if family == "gaussian"
        else "{oracle_family:metrics, 'gaussian':gaussian_metrics}"
    )
    r.cell(
        prefix + ".SAVE.01",
        "Save the reproducible evidence bundle",
        "The manifest records actual counts, units, source hashes, solver diagnostics, scenario hashes, and provisional research status.",
        "validated outputs and figures",
        "manifest",
        "figures = {'history':history_plot, 'acf':acf_plot, 'fan':fan_plot, 'portfolios':portfolio_plot, 'losses':loss_plot}"
        + ("\nfigures['news_impact'] = news_plot" if family == "gjr" else "")
        + f"\nmanifest = save_run(root, run_path, cfg, private_history, {fits}, {forecasts}, {ports}, evaluation, {metrics}, figures)\ndisplay(manifest['run_id'])\nprint(manifest['status'], manifest['scientific_status'])",
        "annual.experiment.save_run",
        "103",
    )
    r.cells.append(
        nbf.v4.new_markdown_cell(
            "## Discussion\n\nUse the saved table to distinguish estimated return equality from realized performance. "
            "Review the residual diagnostics, parameter errors and any boundary flags before interpretation. "
            "The next evidence is recovery across historical lengths and Monte Carlo convergence, followed by the registered stress cases. "
            "Rebalancing, copulas and the full cross-family grid remain outside this experiment."
        )
    )
    r.save()


for family, stem in [
    ("gaussian", "01_iid_gaussian"),
    ("garch", "02_garch_gaussian"),
    ("gjr", "03_gjr_gaussian"),
]:
    model_recipe(family, stem)

r = Recipe(
    "notebooks/foundations/01_m5_mathematical_checks.ipynb",
    "Mathematical backstops for dynamic volatility",
    "Which exact identities must hold before interpreting stochastic output?",
    "These small fixtures isolate formulas, units, endpoint timing and annual wealth. They are independent of every model notebook.",
)
r.cell(
    "M5.FOUNDATIONS.SETUP",
    "Import validation machinery",
    "Use the same production functions on hand-computable inputs.",
    "installed package",
    "functions",
    IMPORTS
    + "\nfrom thesis_portfolio.annual.models import variance_step\nfrom thesis_portfolio.annual.estimation import filter_variance, likelihood\nfrom thesis_portfolio.annual.simulation import aggregate_terminal_returns",
    "annual.diagnostics",
    "S",
)
r.cell(
    "M5.FOUNDATIONS.RECURRENCE",
    "Check the scalar variance formula",
    "For epsilon=.2, h=.1, omega=.01, alpha=.2, beta=.5, the next variance is .068.",
    "fixed scalar inputs",
    "verified next variance",
    "answer = variance_step(.2,.1,.01,.2,.5)\nassert np.isclose(answer,.068)\nprint(answer)",
    "annual.models.variance_step",
    "73,76",
)
r.cell(
    "M5.FOUNDATIONS.ASYMMETRY",
    "Check equal-magnitude shocks",
    "The GJR difference must be gamma epsilon squared; gamma=0 must nest GARCH.",
    "fixed shocks and state",
    "mathematical report",
    "display(mathematical_backstops())",
    "annual.diagnostics.mathematical_backstops",
    "83,84,86",
)
r.cell(
    "M5.FOUNDATIONS.LIKELIHOOD",
    "Check likelihood against a manual series",
    "The first observed variance is a fixed training initializer and all observations contribute.",
    "three observed values, candidate parameters",
    "filtered variances and likelihood",
    "x = np.array([.2,-.1,.3])\ntheta = np.array([0.,.01,.2,.5,.1])\neps,h = filter_variance(x,theta,.1)\nnp.testing.assert_allclose(h,[.1,.068,.047])\nmanual = .5*np.sum(np.log(2*np.pi*h)+eps**2/h)\nassert np.isclose(manual,likelihood(theta,x,.1))\nprint(h, manual)",
    "annual.estimation.likelihood",
    "91",
)
r.cell(
    "M5.FOUNDATIONS.GRADIENT",
    "Check the analytic likelihood gradient",
    "A central finite-difference comparison independently checks the differentiated recurrence.",
    "likelihood, candidate parameters",
    "gradient error",
    "from scipy.optimize._numdiff import approx_derivative\n_, analytic = likelihood(theta,x,.1,gradient=True)\nnumerical = approx_derivative(lambda p: likelihood(p,x,.1),theta).ravel()\nnp.testing.assert_allclose(analytic,numerical,rtol=1e-4,atol=1e-4)\nprint(np.max(abs(analytic-numerical)))",
    "annual.estimation.likelihood",
    "91",
)
r.cell(
    "M5.FOUNDATIONS.WEALTH",
    "Distinguish buy-and-hold from daily rebalancing",
    "Each asset gains 10% and loses 10% in opposite order. Both finish at -1%; restoring equal weights daily would produce zero.",
    "two-day two-asset fixture",
    "annual aggregation identity",
    "paths = np.array([[[.1,-.1],[-.1,.1]]])\nterminal = aggregate_terminal_returns(paths)\nnp.testing.assert_allclose(terminal @ [.5,.5],[-.01])\nprint('Buy-and-hold:',terminal @ [.5,.5])",
    "annual.simulation.aggregate_terminal_returns",
    "93,99",
)
r.cell(
    "M5.FOUNDATIONS.LIMITS",
    "Plot the news-impact fixture",
    "Compare the symmetric and asymmetric response at the same prior variance.",
    "source-exact GJR fixture",
    "news impact figure",
    "from thesis_portfolio.annual.studies import source_fixture\nfixture = source_fixture('gjr')\nfig = plots.news_impact_figure(fixture)\nplt.show()",
    "annual.visualization.news_impact_figure",
    "84",
)
r.cell(
    "M5.FOUNDATIONS.SAVE",
    "Save the backstop evidence",
    "Save actual check outcomes and a figure in a unique study directory.",
    "reports, figure",
    "study record",
    "from thesis_portfolio.annual.studies import study_directory, save_study\nroot = project_root(os.environ.get('THESIS_PROJECT_ROOT'))\npath = study_directory(root,'foundations')\nsave_study(path, {'checks':pd.DataFrame([mathematical_backstops()])}, {'scope':'deterministic fixtures'})\nfig.savefig(path/'news_impact.png',dpi=150,bbox_inches='tight')\nprint(path)",
    "annual.studies.save_study",
    "103",
)
r.save()

r = Recipe(
    "notebooks/experiments/01_parameter_recovery.ipynb",
    "Parameter recovery and historical window selection",
    "How much information do nested observed histories provide about the source dynamics?",
    "Each family and seed contributes one long timeline; candidate windows are nested chronological prefixes. Source-exact model units are preserved and are not interpreted as decimal investable returns.",
)
r.cell(
    "M5.RECOVERY.SETUP",
    "Load calibration functions",
    "Calibration truth is restricted to the researcher layer; the fitter sees observed values only.",
    "installed package",
    "configuration and functions",
    IMPORTS
    + "\nfrom thesis_portfolio.annual.studies import recovery_study, history_selection, study_directory, save_study\nroot = project_root(os.environ.get('THESIS_PROJECT_ROOT'))\ncfg = load_run_config(root/'configs/annual/pilot.yaml')",
    "annual.studies",
    "91A",
)
r.cell(
    "M5.RECOVERY.FIXTURES",
    "Inspect source-exact fixtures",
    "These values are transcribed from the supplied revision; source-derived annual pilots scale omega separately.",
    "source specifications",
    "parameter table",
    "from thesis_portfolio.annual.studies import source_fixture\ndisplay(pd.DataFrame([{ 'family':f,'omega':source_fixture(f).omega[0], 'alpha':source_fixture(f).alpha[0], 'beta':source_fixture(f).beta[0], 'gamma':source_fixture(f).gamma[0], 'long_run_variance':source_fixture(f).long_run_variance[0]} for f in ('garch','gjr')]))",
    "annual.studies.source_fixture",
    "69,81,91A",
)
r.cell(
    "M5.RECOVERY.RUN",
    "Fit all prespecified nested prefixes",
    "Failed fits stay in the table. No reseeding seeks a favorable parameter estimate.",
    "length grid, fixed calibration seed",
    "recovery",
    "lengths = (300,600,1000,1200)\nrecovery = recovery_study(lengths, seeds=(cfg['seed'],), spec_options=cfg['fit'])\ndisplay(recovery)",
    "annual.studies.recovery_study",
    "91A",
)
r.cell(
    "M5.RECOVERY.ERRORS",
    "Inspect persistence and volatility error",
    "A single seed illustrates finite-history uncertainty and cannot establish reliable parameter recovery on its own.",
    "recovery table",
    "selected error columns",
    "display(recovery[['family','observations','success','error_persistence','error_long_run_variance','volatility_normalized_rmse','next_variance_error']])",
    "annual.diagnostics.parameter_comparison",
    "91A",
)
r.cell(
    "M5.RECOVERY.PLOT",
    "Plot recovery across historical lengths",
    "Errors need not decrease monotonically for every prefix. Interpret the entire table, including boundary estimates.",
    "recovery",
    "recovery figure",
    "recovery_plot = plots.recovery_figure(recovery)\nplt.show()",
    "annual.visualization.recovery_figure",
    "91A,102",
)
r.cell(
    "M5.RECOVERY.SELECTION",
    "Record selection status honestly",
    "No numerical adequacy criteria were approved. Retain 1000 as a labelled integration reference rather than claiming a calibrated minimum.",
    "recovery, configured reference",
    "selection",
    "selection = history_selection(recovery, cfg['history']['reference'])\ndisplay(selection)",
    "annual.studies.history_selection",
    "91A",
)
r.cell(
    "M5.RECOVERY.SAVE",
    "Save calibration evidence separately",
    "Production histories use different named streams and are never prefixes of these calibration histories.",
    "recovery, selection, figure",
    "calibration run",
    "path = study_directory(root,'recovery')\nsave_study(path, {'recovery':recovery}, {'selection':selection,'seeds':[cfg['seed']],'lengths':lengths})\nrecovery_plot.savefig(path/'recovery.png',dpi=150,bbox_inches='tight')\nprint(path)",
    "annual.studies.save_study",
    "91A,103",
)
r.save()

r = Recipe(
    "notebooks/experiments/02_monte_carlo_convergence.ipynb",
    "Forecast and evaluation Monte Carlo precision",
    "Which numerical quantities change when the number of simulated future years increases?",
    "Historical observations and fitted parameters remain fixed. Forecast-count comparisons occur before evaluation; evaluation-count comparisons use frozen weights. The pilot grid is 10000/20000 forecast and 25000/50000 evaluation paths; smoke uses smaller counts.",
)
r.cell(
    "M5.CONVERGENCE.SETUP",
    "Load the numerical experiment",
    "These scenario counts describe future years, not historical sample size.",
    "profile and configuration",
    "cfg and helpers",
    IMPORTS
    + "\nfrom thesis_portfolio.annual.studies import convergence_study, study_directory, save_study\nroot=project_root(os.environ.get('THESIS_PROJECT_ROOT'))\ncfg=execution_profile(load_run_config(root/'configs/annual/pilot.yaml'),os.environ.get('THESIS_PROFILE','smoke'))",
    "annual.studies",
    "101A",
)
r.cell(
    "M5.CONVERGENCE.COUNTS",
    "Declare count grids before evaluation",
    "The largest forecast count is prespecified; oracle performance does not select the portfolio or count.",
    "run profile",
    "forecast_counts, evaluation_counts",
    "forecast_counts = (500,1000) if cfg['status']=='smoke' else (10000,20000)\nevaluation_counts = (1000,2000) if cfg['status']=='smoke' else (25000,50000)\nprint(forecast_counts,evaluation_counts)",
    "annual.studies.convergence_study",
    "101A",
)
r.cell(
    "M5.CONVERGENCE.RUN",
    "Run the two distinct count checks",
    "The orchestrator freezes the fitted market, constructs nested forecast scenarios, optimizes before evaluation, and evaluates fixed selected weights on nested oracle scenarios.",
    "cfg and count grids",
    "forecast_table, evaluation_table",
    "forecast_table,evaluation_table=convergence_study(cfg,'garch',forecast_counts,evaluation_counts)",
    "annual.studies.convergence_study",
    "101A",
)
r.cell(
    "M5.CONVERGENCE.FORECAST",
    "Inspect forecast count sensitivity",
    "Changing forecast scenarios changes estimated annual moments and therefore can change r_star. Show target changes with allocations.",
    "forecast_table",
    "forecast evidence",
    "display(forecast_table)",
    "annual.studies.convergence_study",
    "101A",
)
r.cell(
    "M5.CONVERGENCE.EVALUATION",
    "Inspect fixed-weight evaluation precision",
    "Intervals quantify exception uncertainty conditional on this history, not uncertainty over fitted histories.",
    "evaluation_table",
    "evaluation evidence",
    "display(evaluation_table[['count','strategy','confidence','fitted_target','oracle_mean','oracle_cvar','exception_ci_low','exception_ci_high']])",
    "annual.portfolio.evaluate_portfolios",
    "101A",
)
r.cell(
    "M5.CONVERGENCE.PLOT",
    "Plot evaluation tail estimates",
    "Changes are empirical count sensitivity, not a promise of monotonic convergence.",
    "evaluation_table",
    "convergence_plot",
    "convergence_plot,ax=plt.subplots(figsize=(8,4),layout='constrained')\nfor strategy,group in evaluation_table[evaluation_table.confidence==.95].groupby('strategy'):\n    ax.plot(group['count'],group.oracle_cvar,marker='o',label=strategy)\nax.set(xlabel='Oracle future years',ylabel='95% annual CVaR',title='Fixed-weight evaluation count sensitivity')\nax.legend()\nplt.show()",
    "annual.studies.convergence_study",
    "101A,102",
)
r.cell(
    "M5.CONVERGENCE.SAVE",
    "Save count-sensitivity evidence",
    "No accepted adequacy threshold is invented from these comparisons.",
    "tables, count grids",
    "study artifacts",
    "path=study_directory(root,'convergence')\nsave_study(path,{'forecast':forecast_table,'evaluation':evaluation_table},{'forecast_counts':forecast_counts,'evaluation_counts':evaluation_counts,'adequacy':'not certified'})\nconvergence_plot.savefig(path/'convergence.png',dpi=150,bbox_inches='tight')\nprint(path)",
    "annual.studies.save_study",
    "103",
)
r.save()

r = Recipe(
    "notebooks/experiments/03_market_stress.ipynb",
    "Shared-history forecasting comparisons under market stress",
    "How do fitted allocations and annual downside risk change when oracle volatility or correlation increases?",
    "Matched and Gaussian forecasters use the same observations and oracle futures within each run. Each stress configuration creates a new history and refits. Matched seeds pair the underlying shocks deliberately. This is the approved small stress grid, not the full M7 replication study.",
)
r.cell(
    "M5.STRESS.SETUP",
    "Load the stress study",
    "Baseline settings and provisional stress settings are recorded separately.",
    "configuration, profile",
    "cfg and helpers",
    IMPORTS
    + "\nfrom thesis_portfolio.annual.studies import stress_study,study_directory,save_study\nroot=project_root(os.environ.get('THESIS_PROJECT_ROOT'))\ncfg=execution_profile(load_run_config(root/'configs/annual/pilot.yaml'),os.environ.get('THESIS_PROFILE','smoke'))",
    "annual.studies",
    "101",
)
r.cell(
    "M5.STRESS.DESIGN",
    "Display the registered controls",
    "Doubling daily volatility multiplies omega by four without changing alpha, beta or gamma. Correlation rises from .20 to .70 separately and jointly.",
    "revision defaults",
    "stress_design",
    "stress_design=pd.DataFrame({'case':['baseline','volatility','correlation','combined'],'volatility_multiplier':[1,2,1,2],'equicorrelation':[.2,.2,.7,.7]})\ndisplay(stress_design)",
    "annual.experiment.run_experiment",
    "101",
)
r.cell(
    "M5.STRESS.RUN",
    "Execute the registered stress grid",
    "The experiment orchestrator invokes the same demonstrated functions. Each run freezes all forecasting-model allocations before one shared oracle evaluation.",
    "cfg, two dynamic oracle families",
    "stress_results",
    "stress_results=stress_study(root,cfg)\ndisplay(stress_results[['oracle','stress','forecast_family','strategy','fitted_target','oracle_mean','oracle_cvar']])",
    "annual.studies.stress_study",
    "101",
)
r.cell(
    "M5.STRESS.PLOT",
    "Plot risk alongside changing targets",
    "Targets may differ across models and stress histories. This plot does not imply an equal-target match across those runs.",
    "stress_results",
    "stress_plot",
    "stress_plot,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')\nview=stress_results[(stress_results.confidence==.95)&(stress_results.forecast_family==stress_results.oracle)]\nfor (oracle,strategy),group in view.groupby(['oracle','strategy']):\n    axes[0].plot(group.stress,group.oracle_cvar,marker='o',label=f'{oracle}/{strategy}')\n    axes[1].plot(group.stress,group.fitted_target,marker='o',label=f'{oracle}/{strategy}')\naxes[0].set(ylabel='95% annual oracle CVaR',title='Matched forecasting families')\naxes[1].set(ylabel='Fitted annual equality target',title='Targets are model/run-specific')\naxes[0].legend(fontsize=7)\nplt.show()",
    "annual.studies.stress_study",
    "102",
)
r.cell(
    "M5.STRESS.SAVE",
    "Save the stress comparison and run links",
    "Every row retains the underlying run ID for its configuration, diagnostics and weights.",
    "stress_results, design, figure",
    "study artifacts",
    "path=study_directory(root,'stress-comparison')\nsave_study(path,{'comparisons':stress_results,'design':stress_design},{'profile':cfg['status'],'interpretation':'single-history conditional comparisons; not M7 replicated inference'})\nstress_plot.savefig(path/'stress.png',dpi=150,bbox_inches='tight')\nprint(path)",
    "annual.studies.save_study",
    "103",
)
r.save()

# Every former granular ID remains searchable. Module implementation details are
# recorded as module-owned aliases, rather than recreated as hundreds of trivial
# notebook assignments. Primary recipe cells remain meaningful active operations.
old = ROOT / "notebooks/Milestone_5_Granular_Roadmap_Revised.txt"
legacy = re.findall(r"^# (M5\.(S|\d+[A-Z]?)\.\d+)\. (.+)$", old.read_text(), re.M)
rows = []
for sid, group, description in legacy:
    candidates = [c for c in CELL_RECORDS if group in c["legacy_task_groups"].split(",")]
    if not candidates:
        raise RuntimeError(f"unmapped task {sid}")
    primary = next(
        (
            c
            for c in candidates
            if (group in ("81", "82", "83", "84", "85", "86") and "GJR" in c["stable_id"])
            or (group == "91A" and "RECOVERY" in c["stable_id"])
            or (group == "101A" and "CONVERGENCE" in c["stable_id"])
        ),
        candidates[0],
    )
    rows.append(
        {
            "original_task_id": sid,
            "description": description,
            "stable_id": primary["stable_id"],
            "notebook": primary["notebook"],
            "cell_tag": primary["cell_tag"],
            "source_function": primary["source"],
            "test_ids": "tests/annual/test_mathematics.py; tests/annual/test_pipeline.py; tests/test_optimization.py",
            "evidence": "results/<run_id>/manifest.json; fit/; portfolios/; evaluation/; validation/",
            "status": "implemented; execution evidence recorded separately",
            "change_reason": "Hybrid migration: reusable internals owned by canonical module; original granular alias retained at primary demonstration. Scientific adequacy decisions remain provisional.",
        }
    )
dest = ROOT / "documentation/implementation_tasks"
with (dest / "m5_task_map.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
with (dest / "m5_active_cells.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CELL_RECORDS[0].keys())
    writer.writeheader()
    writer.writerows(CELL_RECORDS)
print(
    json.dumps(
        {
            "notebooks": sorted({x["notebook"] for x in CELL_RECORDS}),
            "active_cells": len(CELL_RECORDS),
            "legacy_aliases": len(rows),
        },
        indent=2,
    )
)
