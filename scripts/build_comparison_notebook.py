"""Author the saved-evidence comparison recipe without rebuilding other notebooks."""

import csv
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = "notebooks/experiments/04_three_regime_comparison.ipynb"
cells = [
    nbf.v4.new_markdown_cell("""# Three regimes and shared-history forecasting evidence

**Research question.** How do annual allocations and downside risk differ when
both market dynamics and the forecasting model are made explicit?

This notebook reads explicitly identified, completed annual runs. It does not
depend on another notebook's live kernel. Targets match between strategies within
each fitted model; they can differ between models and oracle regimes.

The top panels compare matched models across oracle regimes. The bottom panels
hold the dynamic oracle fixed and compare its matched volatility forecaster with
a Gaussian forecaster on the same observed history and held-out futures.
""")
]
records = []


def cell(suffix, title, purpose, inputs, outputs, code, source):
    sid = "M5.COMPARISON." + suffix
    cells.append(
        nbf.v4.new_markdown_cell(
            f"### {title}\n\n{purpose}\n\n**Inputs:** {inputs}.  \n**Created / outputs:** {outputs}.  \n**Canonical implementation:** `{source}`."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            f"# {sid} — {title}\n# Inputs: {inputs}\n# Created: {outputs}\n" + code,
            metadata={"tags": [sid]},
        )
    )
    records.append(
        {
            "stable_id": sid,
            "notebook": NAME,
            "cell_tag": sid,
            "source": source,
            "legacy_task_groups": "101,102,103",
            "purpose": title,
        }
    )


cell(
    "REFERENCES",
    "Read explicit evidence references",
    "Every run is named by its immutable directory ID. No automatic latest-run lookup occurs.",
    "project root, reference_runs.json",
    "root, references",
    """import os
import json
import matplotlib.pyplot as plt
from IPython.display import display
from thesis_portfolio.annual.config import project_root
from thesis_portfolio.annual.comparison import load_comparison, regime_dashboard
from thesis_portfolio.annual.studies import study_directory, save_study
root = project_root(os.environ.get('THESIS_PROJECT_ROOT'))
references = json.loads((root/'documentation/implementation_tasks/reference_runs.json').read_text())
display(references)""",
    "annual.comparison.load_comparison",
)
cell(
    "LOAD",
    "Validate and combine annual results",
    "The loader requires matching histories, horizons, policies and scenario counts. Both oracle and forecast-family labels remain attached.",
    "references",
    "comparison",
    """comparison = load_comparison(root,references)
display(comparison[['oracle','forecast_family','strategy','confidence','fitted_target','oracle_mean','oracle_cvar']])""",
    "annual.comparison.load_comparison",
)
cell(
    "TARGETS",
    "Inspect the expected-return controls",
    "There is one target within each fitted model/run, not a universal percentage across all rows.",
    "comparison",
    "target table",
    """targets = comparison[['oracle','forecast_family','fitted_target']].drop_duplicates()
display(targets)""",
    "annual.comparison.load_comparison",
)
cell(
    "DASHBOARD",
    "Display the three-regime dashboard",
    "These conditional results describe the saved histories. They do not establish an across-history ranking or isolate asymmetry when persistence also differs.",
    "comparison",
    "dashboard",
    """dashboard = regime_dashboard(comparison)
plt.show()""",
    "annual.comparison.regime_dashboard",
)
cell(
    "SAVE",
    "Save comparison evidence and provenance",
    "The manifest records explicit source run IDs. Reproduction requires those saved artifacts or their recorded generating configurations.",
    "comparison, references, dashboard",
    "comparison study",
    """path = study_directory(root,'three-regime-comparison')
save_study(path,{'comparison':comparison,'targets':targets},{'source_run_ids':references,'return_control':'within fitted model only'})
dashboard.savefig(path/'three_regime_dashboard.png',dpi=150,bbox_inches='tight')
print(path)""",
    "annual.studies.save_study",
)
cells.append(
    nbf.v4.new_markdown_cell("""## Discussion

Read oracle mean and downside risk together when targets change. Cross-regime
differences combine changes in market dynamics with estimation and target effects.
Shared-history Gaussian comparisons hold the oracle outcomes fixed, but their
fitted targets can still differ. The recovery and convergence studies provide
numerical context; replicated scientific inference belongs to Milestone 7.
""")
)
nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python (thesis)", "language": "python", "name": "thesis"}
    },
)
nbf.validate(nb)
nbf.write(nb, ROOT / NAME)
index = ROOT / "documentation/implementation_tasks/m5_active_cells.csv"
with index.open() as f:
    old = [row for row in csv.DictReader(f) if row["notebook"] != NAME]
with index.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(old + records)
print(NAME)
