# Milestone 5 traceability

`m5_task_map.csv` retains all 443 identifiers from the revised granular roadmap.
It records the canonical owner and the primary notebook demonstration of each
former operation. Reusable arithmetic has moved to modules; the notebook does not
repeat each implementation line. Related module-level aliases can point to the
same meaningful demonstration. This is an explicit granularity migration, not
443 independently executed notebook cells.

`m5_active_cells.csv` inventories the active stable notebook cell tags and their
source functions. Every active cell has a Markdown explanation immediately above
it, identifies inputs and outputs, and performs a meaningful research operation.

The task map's implementation status does **not** claim numerical validation of
every scientific statement. Actual tests and notebook execution reports provide
validation evidence. Scientific adequacy decisions, optional diagnostics and
inference limitations are documented in the research design and run manifests.

`notebook_execution_smoke.json` and `notebook_execution_pilot.json`, when present,
record actual fresh-kernel executions. Run-specific manifests record source hashes,
resolved settings, data hashes, fit/solver diagnostics, counts, and figures.

The source instructions are:

1. `Thesis_Codex_Hybrid_Architecture_Instructions.docx` — software organization.
2. `M4,5 revisions.docx` — supplied V2.0 research methodology.
3. `Milestone_5_Granular_Roadmap_Revised.txt` — original granular aliases.

The canonical architecture reconciles proposed `src/thesis/` names with the
existing importable `src/thesis_portfolio/` package. It does not implement the
deferred stages 04–10.
