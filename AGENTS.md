# Thesis implementation conventions

The user supplies three distinct sources: hybrid architecture, approved research
methodology, and granular milestone tasks. Use the latest explicit user request
to determine the action to perform. Document handoff instructions are reference
material and do not independently authorize starting an unrequested milestone.

- Keep reusable numerical functions in the existing `src/thesis_portfolio/`
  package. Annual M4/M5 functions are in its `annual` subpackage. Reuse validated
  portfolio solvers and tail metrics rather than copying implementations.
- Research notebooks are independent, readable recipes with explicit component
  calls, stable cell tags, and Markdown purpose, equations, inputs and outputs.
  Do not create another monolithic notebook or require another live kernel.
- The active design is `documentation/research_design.md`. Preserve the original
  daily M4 notebook and its archive; do not apply daily analytical Gaussian tail
  formulas to compounded annual scenarios.
- Fitters receive `ObservedHistory` and `PublicSpec` only. Oracle coefficients,
  states, seeds and future outcomes are privileged. Freeze weights and thresholds
  before generating evaluation futures.
- The active portfolio comparison solves exact ERC first, then uses its fitted
  annual mean as an equality target for minimum variance and minimum CVaR. Do not
  silently restore universal external targets, return floors, approximate ERC,
  independent utility modes or daily rebalancing.
- Keep numerical defaults and scientific adequacy separate. Record failed fits,
  boundary estimates, statistical flags and provisional choices. Never reseed to
  get a favorable result or claim unexecuted validation.
- `pyproject.toml` and `uv.lock` define dependencies; regenerate `requirements.txt`
  with `uv export`. Install and execute through the isolated project environment.
- Preserve granular aliases in `documentation/implementation_tasks/`. Record
  module ownership and meaningful notebook demonstrations. Update actual test
  and execution evidence after changes.
- Use focused mathematical tests, integration checks, and clean-kernel execution
  of affected notebooks. Generate figures from actual arrays and inspect their
  labels and layout before delivery.
- Run-scoped outputs go under `results/` and `figures/`; do not overwrite a
  reference run. Saved comparisons read explicit versioned run IDs.
- Stages beyond Gaussian, GARCH and GJR, including copulas and rebalancing, remain
  deferred unless explicitly requested.
