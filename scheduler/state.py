from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, utc_now, write_json_atomic


def init_run_dirs(run_dir: Path) -> None:
    for name in ("manifests", "generated_slurm", "logs", "status", "outputs", "checkpoints", "summary"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def status_path(run_dir: Path, task_id: str, array_index: int | None = None) -> Path:
    suffix = f"_{array_index}" if array_index is not None else ""
    return run_dir / "status" / f"{task_id}{suffix}.json"


def write_status(run_dir: Path, task_id: str, status: str, **extra: Any) -> Path:
    payload = {"task_id": task_id, "status": status, "updated_at": utc_now(), **extra}
    path = status_path(run_dir, task_id, extra.get("array_index"))
    write_json_atomic(path, payload)
    return path


def read_statuses(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "status").glob("*.json")):
        try:
            rows.append(read_json(path))
        except Exception as exc:
            rows.append({"path": str(path), "status": "invalid", "error": str(exc)})
    return rows
