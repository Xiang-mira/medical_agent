#!/usr/bin/env python3
"""Check the HF private migration repo against the local asset manifest."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "Xiang-mira/MedIA-Agentic-AI-Private-HPC"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "docs/migration/hf_asset_manifest.tsv"
REQUIRED_MIGRATION_FILES = {
    "migration/asset_manifest.json",
    "migration/hf_asset_manifest.tsv",
    "migration/hf_upload_plan.tsv",
    "migration/excluded_manifest.tsv",
    "migration/HPC_MIGRATION_AUDIT.md",
    "README.md",
}


def expected_paths(manifest: Path) -> set[str]:
    paths: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("destination") == "hf_private":
                paths.add(row["repo_path"])
    return paths


def list_remote_files(repo_id: str, retries: int) -> set[str]:
    api = HfApi()
    for attempt in range(1, retries + 1):
        try:
            return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception:
            if attempt == retries:
                raise
            sleep_sec = min(120, 10 * attempt)
            print(f"retrying remote file listing after {sleep_sec}s")
            time.sleep(sleep_sec)
    raise RuntimeError("unreachable")


def is_forbidden_remote_path(path: str) -> str | None:
    parts = path.split("/")
    lowered = path.lower()
    name = parts[-1]
    if path.startswith("migration/"):
        return None
    if ".cache" in parts or "__pycache__" in parts:
        return "cache"
    if "third_party/PanTS-main/tars" in path or "third_party/PanTS-main/data" in path or "data/PanTS" in path:
        return "PanTS data/tars"
    if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        return "medical image volume"
    if (lowered.endswith(".npy") or lowered.endswith(".npz")) and not path.startswith("VoxTell/embeddings/"):
        return "probability/array volume"
    if name.endswith(".csv") and ("UCSF" in name or name.startswith("JHH") or name.startswith("UW")):
        return "institutional patient metadata csv"
    if "input_csv" in parts or any(part.startswith("input_csv_") for part in parts):
        return "institutional input_csv metadata"
    if any(token in lowered for token in ("/_debug", "/_tmp", "/_verify")):
        return "temporary output"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    expected = expected_paths(args.manifest)
    remote = list_remote_files(args.repo_id, retries=args.retries)
    missing = sorted(expected - remote)
    migration_missing = sorted(REQUIRED_MIGRATION_FILES - remote)
    forbidden = [
        {"repo_path": path, "reason": reason}
        for path in sorted(remote)
        if (reason := is_forbidden_remote_path(path))
    ]
    payload = {
        "status": "passed" if not missing and not migration_missing and not forbidden else "failed",
        "repo_id": args.repo_id,
        "manifest_files": len(expected),
        "remote_files": len(remote),
        "missing_count": len(missing),
        "migration_missing_count": len(migration_missing),
        "forbidden_count": len(forbidden),
        "missing": missing[:200],
        "migration_missing": migration_missing,
        "forbidden": forbidden[:200],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
