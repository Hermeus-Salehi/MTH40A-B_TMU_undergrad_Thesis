# Generated research figures

Each experiment creates a unique `figures/<run_id>/` directory. Figure paths and
numeric provenance are recorded by the corresponding `results/<run_id>/manifest.json`.
These generated files are excluded from version control by default. Regenerate
them using the installed package and the recorded resolved configuration.

The explicitly selected reference evidence listed in
`documentation/implementation_tasks/published_evidence.json` is versioned so the
executed comparison notebooks can be reviewed immediately after cloning.
