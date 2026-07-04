#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.student_postprocess import process_student_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply anatomy containment post-processing to prompt-student masks.")
    ap.add_argument("--input-root", type=Path, required=True, help="Raw student predictions: case_id/*.nii.gz")
    ap.add_argument("--output-root", type=Path, required=True, help="Postprocessed output root; raw input is never overwritten")
    ap.add_argument("--parent-root", type=Path, action="append", default=[], help="Root containing reliable parent masks. Can be repeated.")
    ap.add_argument("--teacher-root", type=Path, default=None, help="Alias for an additional parent-root")
    ap.add_argument("--case-list", type=Path, default=None)
    ap.add_argument("--organs", default="", help="Optional comma-separated organ subset")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "configs/organ_taxonomy.json")
    ap.add_argument("--policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-roi-masks", action="store_true", help="Do not write parent ROI masks for review")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    parent_roots = [p.resolve() for p in args.parent_root]
    if args.teacher_root is not None:
        parent_roots.append(args.teacher_root.resolve())
    if not parent_roots:
        raise SystemExit(
            "At least one reliable --teacher-root/--parent-root is required; "
            "raw student masks are not an implicit parent fallback."
        )
    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    summary = process_student_root(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        parent_roots=parent_roots,
        taxonomy_path=args.taxonomy.resolve(),
        policy_path=args.policy.resolve(),
        case_list=args.case_list.resolve() if args.case_list else None,
        organs=organs or None,
        max_cases=args.max_cases,
        overwrite=args.overwrite,
        write_roi_masks=not args.no_roi_masks,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
