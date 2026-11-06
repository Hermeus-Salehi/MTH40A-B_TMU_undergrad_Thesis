"""Capture the executed implementation, source snapshot, environment and artifact hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_provenance(output: Path) -> dict:
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    # An installed wheel may run anywhere. Only attribute checkout metadata to
    # an implementation that is actually imported from that checkout's src/.
    source_checkout = (
        package == root / "src" / "thesis_portfolio" and (root / "pyproject.toml").is_file()
    )
    root_files = (
        "pyproject.toml",
        "uv.lock",
        "requirements.txt",
        ".python-version",
        ".gitignore",
        "README.md",
        "main.py",
    )
    sources = []
    if source_checkout:
        sources.extend(
            root / name
            for name in root_files
            if (root / name).is_file() and not (root / name).is_symlink()
        )
        for folder in ("src", "configs", "docs", "tests", "experiments", ".github"):
            sources.extend(
                p
                for p in (root / folder).rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and p.suffix in {".py", ".md", ".toml", ".json", ".yml", ".yaml"}
            )
    source_hashes = {}
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(sources)):
            relative = path.relative_to(root).as_posix()
            source_hashes[relative] = file_hash(path)
            archive.write(path, relative)
        # Capture imported code too, even when running a non-editable wheel.
        for path in sorted(package.glob("*.py")):
            relative = "executed_package/" + path.name
            source_hashes[relative] = file_hash(path)
            archive.write(path, relative)
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    distributions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    # Avoid attributing a parent repo's commit to an unpacked source archive.
    repo_root = _git(root, "rev-parse", "--show-toplevel") if source_checkout else None
    git_checkout = repo_root is not None and Path(repo_root).resolve() == root
    status = (
        _git(root, "status", "--porcelain", "--untracked-files=normal") if git_checkout else None
    )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(distributions.items())),
        "source_layout": "checkout" if source_checkout else "installed_package",
        "git_commit": _git(root, "rev-parse", "HEAD") if git_checkout else None,
        "git_branch": _git(root, "branch", "--show-current") if git_checkout else None,
        "git_dirty": None if status is None else bool(status),
        "git_status": status,
        "source_sha256": source_digest,
        "source_files_sha256": source_hashes,
        "solver_backends": {
            "markowitz": "SciPy SLSQP",
            "cvar": "SciPy HiGHS",
            "erc": "damped Newton",
        },
        "reproducibility": "Numerical agreement within documented tolerances; metadata timestamps vary.",
    }
