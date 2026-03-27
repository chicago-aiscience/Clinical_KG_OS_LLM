"""Repository-root paths (resolved from this file, not the process cwd)."""

from pathlib import Path


def project_root() -> Path:
    """Project root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def transcripts_dir() -> Path:
    """Evaluation bundle: per-patient dirs with ``.txt`` and ``*_standard_answer.json``."""
    return project_root() / "data" / "transcripts"
