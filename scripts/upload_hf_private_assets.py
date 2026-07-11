#!/usr/bin/env python3
"""Create and upload the selected private HPC migration assets to Hugging Face."""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Xiang-mira/MedIA-Agentic-AI-Private-HPC"
DEFAULT_PLAN = ROOT / "docs/migration/hf_upload_plan.tsv"
DEFAULT_MIGRATION_DIR = ROOT / "docs/migration"

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
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(path),
                path_in_repo=f"migration/{name}",
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--migration-dir", type=Path, default=DEFAULT_MIGRATION_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.plan.open(encoding="utf-8"), delimiter="\t"))
    if args.dry_run:
        for row in rows:
            print(f"{row['kind']}\t{row['local_path']}\t{row['path_in_repo']}")
        return 0

    create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api = HfApi()

    for row in rows:
        kind = row["kind"]
        local = ROOT / row["local_path"]
        path_in_repo = row["path_in_repo"]
        if not local.exists():
            print(f"SKIP missing: {local}")
            continue
        if kind == "file":
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
            )
        elif kind == "formal_outputs":
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(local),
                path_in_repo=path_in_repo,
                allow_patterns=OUTPUT_ALLOW_PATTERNS,
                ignore_patterns=OUTPUT_IGNORE_PATTERNS,
            )
        else:
            api.upload_folder(
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
        api.upload_file(
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
