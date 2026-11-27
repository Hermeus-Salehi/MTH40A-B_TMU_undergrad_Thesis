# Decision log

## 27 November 2026 — hybrid M4 annual migration and M5

- Follow the latest supplied architecture and methodology, rather than the earlier
  universal configured return target. The active target is exact ERC's fitted
  annual return within each model/run.
- Retain the established `thesis_portfolio` package name. Add focused modules in
  `src/thesis_portfolio/annual/` instead of creating a second competing package or
  changing validated daily interfaces.
- Reuse existing ERC, Markowitz, CVaR, risk-contribution and empirical-tail helpers.
  No second optimization implementation is introduced.
- Preserve the original M4 notebook byte-for-byte and copy it to
  `archive/m4_daily/`. Create independent annual notebooks instead of modifying
  and disabling hundreds of old cells. Thus there are no superseded source lines
  to delete or hide in the original notebook.
- Translate the former 443 very small notebook steps into module responsibilities
  and 129 meaningful active recipe cells, plus five saved-evidence comparison cells.
  Every former identifier remains in the
  task map; extracted arithmetic is tested in modules rather than repeated as
  trivial assignment cells. The active cell inventory records stable cell tags.
- Use transparent SciPy conditional Gaussian MLE with an analytic recurrence
  gradient. The `arch` package is not required. Unit tests compare likelihood and
  gradients against independent hand calculations and finite differences.
- Use complete functions in module files. Each notebook imports the installed
  package; no `sys.path` edits, cwd mutation, or previous notebook kernel is needed.
- Keep configuration in one annual YAML file. Smoke is an explicit resolved
  profile override; manifests record it. `pyproject.toml` and `uv.lock` define the
  environment; `requirements.txt` is generated from the lock, not hand-maintained.
- Keep calibration adequacy unresolved and labelled. No researcher thresholds,
  parameter certainty, or successful history-length calibration are fabricated.
- Automatic approval review initially could not run fresh kernels because it hit
  a usage limit. The user requested continuation; the same local-kernel execution
  request was retried. Actual execution outcomes are recorded separately.
