#!/usr/bin/env python3
"""Safely link teacher-provided Google Drive checkpoint folders into the project.

This script creates symlinks only. It never copies large checkpoint weights and never writes into
or modifies the teacher's shared checkpoint folders.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_FOLDERS = [
    "CADS_series",
    "MOOSE_series",
    "nnUNet_private",
    "UNEST",
    "VSmTrans",
    "ATLAS-Net",
    "ATLASNet",
    "ATLAS_Net",
]


def safe_remove_link_or_empty_target(path: Path, overwrite: bool) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"
    if not overwrite:
        return "exists_keep"
    if path.is_symlink() or path.is_file():
        path.unlink()
        return "removed_existing_link_or_file"
    # Avoid deleting real non-empty directories created by user.
    if path.is_dir():
        try:
            if not any(path.iterdir()):
                path.rmdir()
                return "removed_empty_dir"
        except Exception:
            pass
        return "exists_real_dir_keep"
    return "exists_unknown_keep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-checkpoints", default="/content/drive/MyDrive/checkpoints")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--folders", default=",".join(DEFAULT_FOLDERS), help="Comma-separated folder names to link")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_checkpoints).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    checkpoint_dir = (project_root / args.checkpoint_dir).resolve()
    folders = [x.strip() for x in args.folders.split(",") if x.strip()]

    result = {
        "stage": "prepare_colab_checkpoint_links",
        "drive_checkpoints": str(drive_root),
        "project_root": str(project_root),
        "checkpoint_dir": str(checkpoint_dir),
        "dry_run": args.dry_run,
        "links": [],
        "copied_files": [],
        "warnings": [],
    }

    if not project_root.exists():
        result["status"] = "error"
        result["error"] = f"Project root does not exist: {project_root}"
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    if not drive_root.exists():
        result["status"] = "error"
        result["error"] = f"Drive checkpoint root does not exist: {drive_root}"
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2

    if not args.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        src = drive_root / folder
        dst = checkpoint_dir / folder
        item = {"name": folder, "src": str(src), "dst": str(dst), "src_exists": src.exists()}
        if not src.exists():
            item["status"] = "missing_source"
            result["links"].append(item)
            continue
        if args.dry_run:
            item["status"] = "would_link"
            result["links"].append(item)
            continue
        removal_status = safe_remove_link_or_empty_target(dst, args.overwrite)
        item["preexisting_target"] = removal_status
        if dst.exists() and not dst.is_symlink():
            item["status"] = "kept_existing_real_directory"
            result["links"].append(item)
            continue
        os.symlink(src, dst, target_is_directory=True)
        item["status"] = "linked"
        item["resolved"] = str(dst.resolve())
        result["links"].append(item)

    # Copy only the tiny Excel map into configs. This is safe and useful because it keeps the
    # project config synchronized with the teacher's latest map without changing the shared file.
    src_map = drive_root / "class_checkpoint_map.xlsx"
    dst_map = project_root / "configs" / "class_checkpoint_map.xlsx"
    if src_map.exists():
        if args.dry_run:
            result["copied_files"].append({"src": str(src_map), "dst": str(dst_map), "status": "would_copy"})
        else:
            dst_map.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_map, dst_map)
            result["copied_files"].append({"src": str(src_map), "dst": str(dst_map), "status": "copied"})
    else:
        result["warnings"].append("class_checkpoint_map.xlsx not found under drive checkpoint root")

    result["status"] = "success"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
