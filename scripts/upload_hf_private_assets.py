#!/usr/bin/env python3
"""Create and upload the selected private HPC migration assets to Hugging Face."""
from __future__ import annotations

import argparse
import csv
import time
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Xiang-mira/MedIA-Agentic-AI-Private-HPC"
DEFAULT_PLAN = ROOT / "docs/migration/hf_upload_plan.tsv"
DEFAULT_MIGRATION_DIR = ROOT / "docs/migration"
DEFAULT_ASSET_MANIFEST = ROOT / "docs/migration/hf_asset_manifest.tsv"

OUTPUT_ALLOW_PATTERNS = ["*.json", "*.jsonl", "*.csv", "*.yaml", "*.yml", "*.txt", "*.log", "*.md"]
OUTPUT_IGNORE_PATTERNS = [
    "*.nii",
    "*.nii.gz",
    "*.npy",
    "*.npz",
    "*.pt",
    "*.pth",
    "*/student_predictions/*",
    "*/raw_predictions/*",
    "*/standard_dataset/*",
    "*/annotation_versions/*",
    "*smoke*",
    "*dryrun*",
    "*bad*",
    "*/_debug*",
    "*/_tmp*",
    "*/_verify*",
]

GENERAL_IGNORE_PATTERNS = [
    "*/.cache/*",
    "*/__pycache__/*",
    "*__pycache__*",
    "*/UCSF*.csv",
    "*/JHH*.csv",
    "*/UW*.csv",
    "*.nii",
    "*.nii.gz",
    "*.dcm",
    "*.mha",
    "*.mhd",
]


def upload_manifest_files(api: HfApi, repo_id: str, migration_dir: Path) -> None:
    for name in [
        "asset_manifest.json",
        "hf_asset_manifest.tsv",
        "hf_upload_plan.tsv",
        "excluded_manifest.tsv",
        "HPC_MIGRATION_AUDIT.md",
    ]:
        path = migration_dir / name
        if path.exists():
            retry_hf_call(
                api.upload_file,
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(path),
                path_in_repo=f"migration/{name}",
            )


def retry_hf_call(fn, *args, retries: int = 5, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == retries:
                raise
            sleep_sec = min(60, 5 * attempt)
            print(f"retrying Hugging Face API call after {sleep_sec}s")
            time.sleep(sleep_sec)
    raise RuntimeError("unreachable")


def ensure_repo(repo_id: str) -> None:
    retry_hf_call(create_repo, repo_id, repo_type="model", private=True, exist_ok=True)


def remote_files(api: HfApi, repo_id: str) -> set[str]:
    for attempt in range(1, 6):
        try:
            return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except Exception:
            if attempt == 5:
                raise
            sleep_sec = min(60, 5 * attempt)
            print(f"retrying remote file listing after {sleep_sec}s")
            time.sleep(sleep_sec)
    raise RuntimeError("unreachable")


def manifest_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))


def upload_output_manifest_rows(
    api: HfApi,
    repo_id: str,
    rows: list[dict[str, str]],
    path_prefix: str,
    chunk_size: int,
    retries: int,
    rate_limit_waits: int,
) -> None:
    existing = remote_files(api, repo_id)
    todo = [
        row for row in rows
        if row.get("repo_path", "").startswith(path_prefix.rstrip("/") + "/")
        and row.get("repo_path") not in existing
    ]
    print(f"formal output upload: prefix={path_prefix} missing_files={len(todo)}")
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        operations = [
            CommitOperationAdd(path_in_repo=row["repo_path"], path_or_fileobj=str(ROOT / row["source_path"]))
            for row in chunk
        ]
        if not operations:
            continue
        attempt = 0
        rate_waits = 0
        while True:
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=operations,
                    commit_message=f"Upload filtered formal outputs {path_prefix} {start + 1}-{start + len(chunk)}",
                )
                break
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429 and "repository commits" in str(exc):
                    rate_waits += 1
                    if rate_waits > rate_limit_waits:
                        raise
                    sleep_sec = 3900
                    print(
                        "hit Hugging Face repository commit rate limit; "
                        f"sleeping {sleep_sec}s before resuming"
                    )
                    time.sleep(sleep_sec)
                    continue
                attempt += 1
                if attempt > retries:
                    raise
                sleep_sec = min(60, 5 * attempt)
                print(f"retrying {path_prefix} chunk {start // chunk_size + 1} after {sleep_sec}s")
                time.sleep(sleep_sec)
            except Exception:
                attempt += 1
                if attempt > retries:
                    raise
                sleep_sec = min(60, 5 * attempt)
                print(f"retrying {path_prefix} chunk {start // chunk_size + 1} after {sleep_sec}s")
                time.sleep(sleep_sec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--migration-dir", type=Path, default=DEFAULT_MIGRATION_DIR)
    ap.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--rate-limit-waits", type=int, default=2)
    ap.add_argument("--outputs-only", action="store_true")
    ap.add_argument("--manifests-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.plan.open(encoding="utf-8"), delimiter="\t"))
    if args.dry_run:
        for row in rows:
            print(f"{row['kind']}\t{row['local_path']}\t{row['path_in_repo']}")
        return 0

    ensure_repo(args.repo_id)
    api = HfApi()
    asset_rows = manifest_rows(args.asset_manifest)

    if args.manifests_only:
        upload_manifest_files(api, args.repo_id, args.migration_dir)
        return 0

    for row in rows:
        kind = row["kind"]
        local = ROOT / row["local_path"]
        path_in_repo = row["path_in_repo"]
        if not local.exists():
            print(f"SKIP missing: {local}")
            continue
        if args.outputs_only and kind != "formal_outputs":
            continue
        if kind == "file":
            retry_hf_call(
                api.upload_file,
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
            )
        elif kind == "formal_outputs":
            upload_output_manifest_rows(
                api=api,
                repo_id=args.repo_id,
                rows=asset_rows,
                path_prefix=path_in_repo,
                chunk_size=args.chunk_size,
                retries=args.retries,
                rate_limit_waits=args.rate_limit_waits,
            )
        elif not args.outputs_only:
            retry_hf_call(
                api.upload_folder,
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(local),
                path_in_repo=path_in_repo,
                ignore_patterns=GENERAL_IGNORE_PATTERNS,
            )

    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(
            "# MedIA Agentic AI Private HPC Assets\n\n"
            "Private restore assets for the DISCOVERY/OnDemand HPC migration. "
            "See `migration/HPC_MIGRATION_AUDIT.md` and the GitHub repository README.\n",
            encoding="utf-8",
        )
        retry_hf_call(
            api.upload_file,
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
        )
    upload_manifest_files(api, args.repo_id, args.migration_dir)
    print(f"Uploaded selected assets to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
