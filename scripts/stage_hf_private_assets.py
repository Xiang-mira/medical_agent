#!/usr/bin/env python3
"""Stage and upload private HPC migration assets with HF large-folder upload.

The staging directory mirrors the target Hugging Face repo layout exactly. Files
listed in the HF asset manifest are hard-linked by default so a 29+ GiB restore
set does not require another 29+ GiB of local disk space.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Xiang-mira/MedIA-Agentic-AI-Private-HPC"
DEFAULT_MANIFEST = ROOT / "docs/migration/hf_asset_manifest.tsv"
DEFAULT_STAGE_DIR = ROOT / ".hf_hpc_upload_staging"
MIGRATION_FILES = [
    "asset_manifest.json",
    "hf_asset_manifest.tsv",
    "hf_upload_plan.tsv",
    "excluded_manifest.tsv",
    "HPC_MIGRATION_AUDIT.md",
]
MARKER_NAME = ".hf_hpc_stage_marker"
README_TEXT = """# MedIA Agentic AI Private HPC Assets

Private restore assets for the DISCOVERY/OnDemand HPC migration.

This repository is generated from `docs/migration/hf_asset_manifest.tsv` in the
GitHub project. It intentionally excludes PanTS data/tarballs, raw medical image
volumes, PHI-like institutional CSV metadata, caches, and temporary outputs.
See `migration/HPC_MIGRATION_AUDIT.md` for the restore policy and audit counts.
"""


@dataclass(frozen=True)
class Asset:
    repo_path: str
    source_path: str
    size_bytes: int
    sha256: str
    reason: str


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_manifest(path: Path) -> list[Asset]:
    rows: list[Asset] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("destination") != "hf_private":
                continue
            rows.append(
                Asset(
                    repo_path=row["repo_path"],
                    source_path=row["source_path"],
                    size_bytes=int(row.get("size_bytes") or 0),
                    sha256=row.get("sha256") or "",
                    reason=row.get("reason") or "",
                )
            )
    return rows


def migration_assets(manifest_path: Path) -> list[Asset]:
    assets: list[Asset] = []
    migration_dir = manifest_path.parent
    for name in MIGRATION_FILES:
        source = migration_dir / name
        if not source.exists():
            continue
        assets.append(
            Asset(
                repo_path=f"migration/{name}",
                source_path=rel(source),
                size_bytes=source.stat().st_size,
                sha256="",
                reason="migration audit manifest",
            )
        )
    return assets


def readme_asset(stage_dir: Path) -> Asset:
    path = stage_dir / "README.md"
    return Asset(
        repo_path="README.md",
        source_path="",
        size_bytes=len(README_TEXT.encode("utf-8")),
        sha256="",
        reason="HF private repo overview",
    )


def ensure_stage_dir(stage_dir: Path, prune: bool) -> None:
    stage_dir = stage_dir.resolve()
    if stage_dir == ROOT.resolve() or stage_dir == Path(stage_dir.anchor):
        raise ValueError(f"refusing unsafe stage directory: {stage_dir}")
    if stage_dir.exists() and not stage_dir.is_dir():
        raise ValueError(f"stage path is not a directory: {stage_dir}")
    marker = stage_dir / MARKER_NAME
    if stage_dir.exists() and prune and not marker.exists() and any(stage_dir.iterdir()):
        raise ValueError(
            f"refusing to prune unmarked staging directory: {stage_dir}. "
            f"Expected marker {MARKER_NAME}."
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"purpose": "medai-hpc-hf-private-staging", "root": str(ROOT)}) + "\n",
        encoding="utf-8",
    )


def should_ignore_stage_file(path: Path, stage_dir: Path) -> bool:
    rel_parts = path.relative_to(stage_dir).parts
    return rel_parts[0] in {".cache", MARKER_NAME}


def prune_stage(stage_dir: Path, expected_repo_paths: set[str]) -> tuple[int, int]:
    removed_files = 0
    removed_dirs = 0
    if not stage_dir.exists():
        return removed_files, removed_dirs
    for path in sorted(stage_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if should_ignore_stage_file(path, stage_dir):
            continue
        rel_path = path.relative_to(stage_dir).as_posix()
        if path.is_file() or path.is_symlink():
            if rel_path not in expected_repo_paths:
                path.unlink()
                removed_files += 1
        elif path.is_dir():
            try:
                path.rmdir()
                removed_dirs += 1
            except OSError:
                pass
    return removed_files, removed_dirs


def link_or_copy(source: Path, dest: Path, copy_fallback: bool) -> str:
    if not source.exists():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.samefile(source) and dest.stat().st_size == source.stat().st_size:
                return "existing"
        except OSError:
            pass
        if dest.stat().st_size == source.stat().st_size and dest.stat().st_mtime_ns == source.stat().st_mtime_ns:
            return "existing"
        dest.unlink()
    try:
        os.link(source, dest)
        return "hardlinked"
    except OSError as exc:
        if not copy_fallback:
            raise RuntimeError(
                f"hardlink failed for {source} -> {dest}: {exc}. "
                "Use --copy-fallback only when enough free disk is available."
            ) from exc
        shutil.copy2(source, dest)
        return "copied"


def stage_assets(
    assets: list[Asset],
    stage_dir: Path,
    *,
    prune: bool,
    copy_fallback: bool,
) -> dict[str, int]:
    ensure_stage_dir(stage_dir, prune=prune)
    expected = {asset.repo_path for asset in assets}
    expected.add("README.md")
    if prune:
        removed_files, removed_dirs = prune_stage(stage_dir, expected)
    else:
        removed_files = removed_dirs = 0

    counts = {"existing": 0, "hardlinked": 0, "copied": 0, "written": 0}
    for asset in assets:
        source = ROOT / asset.source_path
        dest = stage_dir / asset.repo_path
        action = link_or_copy(source, dest, copy_fallback=copy_fallback)
        counts[action] += 1

    readme = stage_dir / "README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != README_TEXT:
        readme.write_text(README_TEXT, encoding="utf-8")
        counts["written"] += 1
    print(
        json.dumps(
            {
                "status": "staged",
                "stage_dir": str(stage_dir),
                "assets": len(assets),
                "actions": counts,
                "pruned_files": removed_files,
                "pruned_dirs": removed_dirs,
            },
            indent=2,
        )
    )
    return counts


def is_forbidden_repo_path(path: str) -> str | None:
    parts = path.split("/")
    lowered = path.lower()
    name = parts[-1]
    if path in {MARKER_NAME, "README.md"} or path.startswith("migration/"):
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


def iter_stage_content_files(stage_dir: Path) -> Iterable[Path]:
    for path in stage_dir.rglob("*"):
        if path.is_dir():
            continue
        if should_ignore_stage_file(path, stage_dir):
            continue
        yield path


def verify_stage(stage_dir: Path, manifest_assets: list[Asset], migration: list[Asset]) -> dict[str, object]:
    expected_manifest = {asset.repo_path: asset for asset in manifest_assets}
    expected_extra = {asset.repo_path: asset for asset in migration}
    expected_extra["README.md"] = readme_asset(stage_dir)
    expected_all = set(expected_manifest) | set(expected_extra)

    actual_files = {path.relative_to(stage_dir).as_posix(): path for path in iter_stage_content_files(stage_dir)}
    missing = sorted(path for path in expected_all if path not in actual_files)
    extra = sorted(path for path in actual_files if path not in expected_all)
    size_mismatches = []
    for repo_path, asset in expected_manifest.items():
        path = actual_files.get(repo_path)
        if path is None:
            continue
        size = path.stat().st_size
        if size != asset.size_bytes:
            size_mismatches.append({"repo_path": repo_path, "actual": size, "expected": asset.size_bytes})

    forbidden = []
    for repo_path in sorted(actual_files):
        reason = is_forbidden_repo_path(repo_path)
        if reason:
            forbidden.append({"repo_path": repo_path, "reason": reason})

    manifest_bytes = sum(asset.size_bytes for asset in manifest_assets)
    staged_manifest_bytes = sum(
        actual_files[path].stat().st_size for path in expected_manifest if path in actual_files
    )
    payload: dict[str, object] = {
        "status": "passed" if not missing and not extra and not size_mismatches and not forbidden else "failed",
        "stage_dir": str(stage_dir),
        "manifest_files": len(expected_manifest),
        "manifest_bytes": manifest_bytes,
        "staged_manifest_bytes": staged_manifest_bytes,
        "extra_expected_files": len(expected_extra),
        "actual_content_files": len(actual_files),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "size_mismatch_count": len(size_mismatches),
        "forbidden_count": len(forbidden),
        "missing": missing[:200],
        "extra": extra[:200],
        "size_mismatches": size_mismatches[:200],
        "forbidden": forbidden[:200],
    }
    print(json.dumps(payload, indent=2))
    return payload


def upload_large_folder(repo_id: str, stage_dir: Path, num_workers: int, print_report_every: int) -> None:
    api = HfApi()
    for attempt in range(1, 6):
        try:
            api.upload_large_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=stage_dir,
                private=True,
                ignore_patterns=[
                    ".cache/*",
                    "*/.cache/*",
                    MARKER_NAME,
                ],
                num_workers=num_workers,
                print_report=True,
                print_report_every=print_report_every,
            )
            return
        except Exception:
            if attempt == 5:
                raise
            sleep_sec = min(120, 15 * attempt)
            print(f"upload_large_folder failed; retrying after {sleep_sec}s")
            time.sleep(sleep_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--prune-stage", action="store_true")
    ap.add_argument("--hardlink", action="store_true", default=True, help="Hardlink manifest files into staging.")
    ap.add_argument("--copy-fallback", action="store_true", help="Copy files only if hardlinking fails.")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--upload-large-folder", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--print-report-every", type=int, default=60)
    args = ap.parse_args()

    stage_dir = args.stage_dir.resolve()
    manifest_assets = read_manifest(args.manifest)
    migration = migration_assets(args.manifest)
    all_stage_assets = manifest_assets + migration

    if args.verify_only:
        payload = verify_stage(stage_dir, manifest_assets, migration)
        return 0 if payload["status"] == "passed" else 2

    if args.upload_large_folder:
        payload = verify_stage(stage_dir, manifest_assets, migration)
        if payload["status"] != "passed":
            raise SystemExit("staging verification failed; not uploading")
        upload_large_folder(args.repo_id, stage_dir, args.num_workers, args.print_report_every)
        return 0

    stage_assets(
        all_stage_assets,
        stage_dir,
        prune=args.prune_stage,
        copy_fallback=args.copy_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
