from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASE_ID_RE = re.compile(r"^BDMAP_[A-Za-z0-9]+$")


class SchedulerError(RuntimeError):
    """Raised when the scheduler must fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def git_snapshot(cwd: Path = ROOT) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def ensure_not_raw_data_write_path(path: Path, image_root: Path | None, mask_root: Path | None) -> None:
    resolved = path.expanduser().resolve()
    for raw_root in (image_root, mask_root):
        if raw_root is None:
            continue
        root = raw_root.expanduser().resolve()
        if resolved == root or root in resolved.parents:
            raise SchedulerError(f"Refusing to write inside raw data root: {resolved}")


def normalize_mask_stem(filename: str) -> str:
    if not filename.endswith(".nii.gz"):
        raise SchedulerError(f"Expected .nii.gz mask name, got {filename!r}")
    return filename[:-7].lstrip("_")


def is_gt_like_key(key: str) -> bool:
    token = key.lower()
    return any(
        marker in token
        for marker in (
            "gt",
            "mask_root",
            "annotation_folder",
            "eval_reference",
            "label_path",
            "labels",
            "segmentations",
        )
    )
