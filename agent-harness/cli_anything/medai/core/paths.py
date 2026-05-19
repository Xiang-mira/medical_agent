from __future__ import annotations

from pathlib import Path


def project_root_from_harness() -> Path:
    # .../agent-harness/cli_anything/medai/core/paths.py -> project root
    return Path(__file__).resolve().parents[4]


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root_from_harness() / p).resolve()
