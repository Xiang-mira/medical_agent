#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))

from cli_anything.medai.core.totalseg_runner import (  # noqa: E402
    TASK_OFFLINE_REQUIREMENTS,
    preflight_totalseg_offline_assets,
    resolve_totalseg_home,
)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _totalseg_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("TotalSegmentator")
    except Exception:
        return ""


def build_brain_ventricle_manifest(
    *,
    home: Path | None = None,
    output_manifest: Path,
    include_sha256: bool = True,
) -> dict[str, Any]:
    home = (home or resolve_totalseg_home()).resolve()
    requirement = TASK_OFFLINE_REQUIREMENTS["brain_structures"]
    preflight = preflight_totalseg_offline_assets(["brain_structures"], home=home)
    files: list[dict[str, Any]] = []
    for dataset in requirement["required_datasets"]:
        dataset_root = home / "nnunet" / "results" / dataset
        if not dataset_root.exists():
            continue
        for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
            row = {
                "dataset": dataset,
                "relative_path": str(path.relative_to(home)),
                "size_bytes": path.stat().st_size,
            }
            if include_sha256:
                row["sha256"] = _sha256(path)
            files.append(row)
    manifest = {
        "status": "READY" if preflight["status"] == "ok" else "INCOMPLETE",
        "generated_at": _utc_timestamp(),
        "totalsegmentator_version": _totalseg_version(),
        "totalseg_home": str(home),
        "canonical_target": requirement["canonical_target"],
        "task_name": requirement["task"],
        "task_id": requirement["task_id"],
        "source_output_name": requirement["source_output_name"],
        "source_class_id": requirement["source_class_id"],
        "license_required": requirement["license_required"],
        "license_present": preflight["license_present"],
        "download_method": "official_totalseg_download_weights_-t_brain_structures",
        "required_datasets": requirement["required_datasets"],
        "missing_datasets": preflight["missing_datasets"],
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "weight_storage_decision": "WEIGHTS_NOT_COMMITTED_LICENSE_RESTRICTION",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_brain_ventricle_manifest(*, home: Path | None, manifest_path: Path) -> dict[str, Any]:
    home = (home or resolve_totalseg_home()).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in manifest.get("files", []) or []:
        rel = str(row.get("relative_path") or "")
        path = home / rel
        if not path.exists():
            missing.append(rel)
            continue
        expected_size = int(row.get("size_bytes") or -1)
        if expected_size >= 0 and path.stat().st_size != expected_size:
            mismatches.append({"relative_path": rel, "reason": "size_mismatch"})
            continue
        expected_hash = row.get("sha256")
        if expected_hash and _sha256(path) != expected_hash:
            mismatches.append({"relative_path": rel, "reason": "sha256_mismatch"})
    preflight = preflight_totalseg_offline_assets(["brain_structures"], home=home)
    status = "READY" if preflight["status"] == "ok" and not missing and not mismatches else "FAILED"
    return {
        "status": status,
        "home": str(home),
        "manifest": str(manifest_path),
        "missing_files": missing,
        "mismatches": mismatches,
        "offline_preflight": preflight,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/verify the licensed TotalSegmentator brain_ventricle offline bundle.")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-sha256", action="store_true", help="Record sizes only; not recommended for final bundle manifests.")
    args = parser.parse_args()
    if args.verify:
        result = verify_brain_ventricle_manifest(home=args.home, manifest_path=args.manifest)
    else:
        result = build_brain_ventricle_manifest(
            home=args.home,
            output_manifest=args.manifest,
            include_sha256=not bool(args.no_sha256),
        )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
