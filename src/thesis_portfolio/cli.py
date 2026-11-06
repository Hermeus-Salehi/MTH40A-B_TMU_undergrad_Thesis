"""Command-line entry point for validated, repeatable research experiments."""

import argparse
import json
import sys

from thesis_portfolio.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MTH40 portfolio research infrastructure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a fixed-target Gaussian experiment")
    run.add_argument("--config", required=True, help="TOML config or saved resolved_config.json")
    run.add_argument("--output", default="results", help="Parent directory for a new unique run")
    validate = subparsers.add_parser("validate", help="Validate and print resolved settings")
    validate.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps(config.to_dict(), indent=2))
        else:
            from thesis_portfolio.experiment import run_experiment

            output = run_experiment(config, args.output)
            print(f"Completed: {output}")
            print(f"Metrics: {output / 'metrics.csv'}")
            print(
                "Development experiment: do not interpret these results as final thesis conclusions."
            )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
