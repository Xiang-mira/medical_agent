#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import csv
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> int:
    p = argparse.ArgumentParser(description="Build lightweight handoff package directories for rename and generated-label delivery.")
    p.add_argument("--rename-dir", type=Path, default=Path("rename_delivery"))
    p.add_argument("--generated-dir", type=Path, default=Path("generated_labels_100cases"))
    p.add_argument("--mapping-file", type=Path, default=Path("configs/dataset_delivery/organ_rename_mapping.csv.example"))
    p.add_argument("--dry-run-report", type=Path, default=Path("reports/dataset_delivery/rename/dry_run_report.csv"))
    args = p.parse_args()
    args.rename_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("rename_anatomical_labels.py"), args.rename_dir / "rename_anatomical_labels.py")
    mapping_fields = ["source_name", "target_name", "status", "reason", "notes"]
    if args.mapping_file.exists():
        with args.mapping_file.open("r", encoding="utf-8-sig", newline="") as src, (args.rename_dir / "organ_rename_mapping.csv").open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=mapping_fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                if (row.get("status") or "").strip() == "confirmed":
                    writer.writerow(row)
    else:
        (args.rename_dir / "organ_rename_mapping.csv").write_text(",".join(mapping_fields) + "\n", encoding="utf-8")
    if args.dry_run_report.exists():
        shutil.copy2(args.dry_run_report, args.rename_dir / "dry_run_report.csv")
    else:
        (args.rename_dir / "dry_run_report.csv").write_text("case_id,source_name,target_name,source_path,target_path,mode,status,reason\n", encoding="utf-8")
    if not (args.rename_dir / "README.md").exists():
        shutil.copy2(Path("docs/dataset_delivery_373.md"), args.rename_dir / "README.md")
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    if not (args.generated_dir / "README.md").exists():
        shutil.copy2(Path("docs/dataset_delivery_373.md"), args.generated_dir / "README.md")
    for name in ("manifest.csv", "validation_report.csv", "failed_cases.csv", "copy_conflicts.csv"):
        target = args.generated_dir / name
        if not target.exists():
            target.write_text("", encoding="utf-8")
    print(f"built {args.rename_dir} and {args.generated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
