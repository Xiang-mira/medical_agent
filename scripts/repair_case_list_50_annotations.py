#!/usr/bin/env python3
"""Fill missing PanTS annotation_folder cells from the local case-id layout."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_segmentation(case_id: str) -> Path | None:
    candidates = [
        ROOT / "data" / "PanTS" / "LabelTr" / case_id / "segmentations",
        ROOT / "third_party" / "PanTS-main" / "data" / "LabelTr" / case_id / "segmentations",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output", default="", help="Default: update the input file in place.")
    args = ap.parse_args()

    src = Path(args.case_list).resolve()
    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "annotation_folder" not in fieldnames:
        fieldnames.append("annotation_folder")

    repaired = []
    missing = []
    for row in rows:
        case_id = (row.get("case_id") or "").strip()
        current = (row.get("annotation_folder") or "").strip()
        if current:
            continue
        resolved = resolve_segmentation(case_id)
        if resolved is None:
            missing.append(case_id)
            continue
        row["annotation_folder"] = str(resolved)
        repaired.append(case_id)

    dst = Path(args.output).resolve() if args.output else src
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "stage": "repair_case_list_50_annotations",
        "status": "success" if not missing else "warning",
        "case_list": str(src),
        "output": str(dst),
        "num_rows": len(rows),
        "num_repaired": len(repaired),
        "repaired_cases": repaired,
        "missing_cases": missing,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
