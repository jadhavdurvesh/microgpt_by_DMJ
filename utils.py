"""
Saudade v4 — shared helpers for reproducibility metadata.

Nothing here is architecture — just the "make experiments reproducible"
items from the roadmap: git commit, a cheap dataset fingerprint (hashing
a 400MB+ corpus on every checkpoint would be wasteful, so this hashes a
bounded sample plus the file size), and a tiny JSON-metrics writer for the
experiments/ tracking convention.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def get_git_commit(repo_dir="."):
    """Best-effort short git commit hash, or None if not in a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def dataset_fingerprint(path, sample_bytes=1_000_000):
    """
    Cheap fingerprint for a large file: size + a hash of the first and last
    `sample_bytes`. Not cryptographically meaningful — just enough to notice
    "this checkpoint was trained on a different train.bin than I think".
    Hashing the full multi-hundred-MB corpus on every checkpoint would be
    wasted I/O for the same purpose.
    """
    path = Path(path)
    if not path.exists():
        return None

    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(0, size - sample_bytes))
            h.update(f.read(sample_bytes))

    return {"size_bytes": size, "sample_sha256": h.hexdigest()[:16]}


def save_experiment_snapshot(experiment_name, base_dir, files_to_copy, metrics=None):
    """
    Copies the given files into experiments/<experiment_name>/ and writes
    metrics.json there, per the roadmap's experiment-tracking convention:

        experiments/
        ├── v4_baseline/
        │   ├── config.py
        │   ├── train_log.txt
        │   └── metrics.json
    """
    import shutil

    exp_dir = Path(base_dir) / "experiments" / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for f in files_to_copy:
        f = Path(f)
        if f.exists():
            shutil.copy2(f, exp_dir / f.name)

    if metrics is not None:
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return exp_dir
