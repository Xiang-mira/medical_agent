#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.student_postprocess import component_stats, load_yaml, organ_group


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate raw vs containment-postprocessed student masks.")
    ap.add_argument("--raw-root", type=Path, required=True, help="Raw student predictions: case_id/*.nii.gz")
    ap.add_argument("--post-root", type=Path, required=True, help="Postprocessed predictions: case_id/*.nii.gz")
    ap.add_argument("--teacher-root", type=Path, default=None, help="Teacher/reference masks root")
    ap.add_argument("--gt-root", type=Path, default=None, help="Optional trusted GT masks root")
    ap.add_argument("--case-list", type=Path, default=None)
    ap.add_argument("--organs", default="", help="Optional comma-separated organ subset")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    ap.add_argument("--top-visuals", type=int, default=20)
    return ap.parse_args()


def case_ids(case_list: Path | None, raw_root: Path) -> list[str]:
    if case_list and case_list.exists():
        with case_list.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            col = "case_id" if "case_id" in rows[0] else next(iter(rows[0]))
            return [str(row.get(col) or "") for row in rows if row.get(col)]
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir())


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def candidate_paths(root: Path | None, case_id: str, organ: str, *, gt: bool = False) -> list[Path]:
    if root is None:
        return []
    if gt:
        return [root / case_id / "segmentations" / f"{organ}.nii.gz", root / case_id / f"{organ}.nii.gz"]
    return [root / case_id / "updated" / f"{organ}.nii.gz", root / case_id / f"{organ}.nii.gz", root / "cases" / case_id / "updated" / f"{organ}.nii.gz", root / "cases" / case_id / f"{organ}.nii.gz"]


def read_bool(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    if reference is not None and (img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-4)):
        img = resample_from_to(img, reference, order=0)
    return img, np.asanyarray(img.dataobj) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    av = int(a.sum()); bv = int(b.sum())
    if av + bv == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / (av + bv))


def ratio(pred: np.ndarray, ref: np.ndarray) -> float | None:
    rv = int(ref.sum())
    return float(pred.sum() / rv) if rv else None


def copy_review_masks(row: dict[str, Any], visual_root: Path) -> None:
    import shutil
    case_id = row["case_id"]
    organ = row["organ"]
    out = visual_root / case_id / organ
    out.mkdir(parents=True, exist_ok=True)
    for key, name in [
        ("raw_path", "raw_student.nii.gz"),
        ("post_path", "postprocessed_student.nii.gz"),
        ("reference_path", "reference.nii.gz"),
    ]:
        p = Path(str(row.get(key) or ""))
        if p.exists():
            shutil.copy2(p, out / name)
    roi = Path(str(row.get("parent_roi_path") or ""))
    if roi.exists():
        shutil.copy2(roi, out / "parent_organ_roi.nii.gz")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_yaml(args.policy)
    cases = case_ids(args.case_list, args.raw_root)
    wanted = {x.strip() for x in args.organs.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    for case_id in cases:
        raw_case = args.raw_root / case_id
        post_case = args.post_root / case_id
        if not raw_case.exists():
            continue
        for raw_path in sorted(raw_case.glob("*.nii.gz")):
            organ = raw_path.name[:-7]
            if wanted and organ not in wanted:
                continue
            post_path = post_case / f"{organ}.nii.gz"
            if not post_path.exists():
                continue
            raw_img, raw = read_bool(raw_path)
            _, post = read_bool(post_path, raw_img)
            gt_path = first_existing(candidate_paths(args.gt_root, case_id, organ, gt=True))
            teacher_path = first_existing(candidate_paths(args.teacher_root, case_id, organ))
            ref_path = gt_path or teacher_path
            ref_kind = "gt" if gt_path else ("teacher" if teacher_path else "")
            ref = None
            if ref_path is not None:
                _, ref = read_bool(ref_path, raw_img)
            before_stats = component_stats(raw)
            after_stats = component_stats(post)
            fp_removed = int(np.logical_and(raw, ~post).sum())
            row: dict[str, Any] = {
                "case_id": case_id,
                "organ": organ,
                "organ_group": organ_group(organ),
                "raw_path": str(raw_path),
                "post_path": str(post_path),
                "reference_kind": ref_kind,
                "reference_path": str(ref_path) if ref_path else "",
                "parent_roi_path": str(args.post_root / "parent_rois" / case_id / f"{organ}_allowed_roi.nii.gz"),
                "before_voxels": int(raw.sum()),
                "after_voxels": int(post.sum()),
                "false_positive_voxels_removed": fp_removed,
                "connected_component_count_before": before_stats.get("component_count"),
                "connected_component_count_after": after_stats.get("component_count"),
                "largest_component_ratio_before": before_stats.get("largest_component_ratio"),
                "largest_component_ratio_after": after_stats.get("largest_component_ratio"),
            }
            if ref is not None:
                row.update({
                    "before_postprocessing_dice": dice(ref, raw),
                    "after_postprocessing_dice": dice(ref, post),
                    "before_volume_ratio": ratio(raw, ref),
                    "after_volume_ratio": ratio(post, ref),
                    "dice_delta": dice(ref, post) - dice(ref, raw),
                })
            else:
                row.update({
                    "before_postprocessing_dice": None,
                    "after_postprocessing_dice": None,
                    "before_volume_ratio": None,
                    "after_volume_ratio": None,
                    "dice_delta": None,
                })
            rows.append(row)

    required_columns = [
        "case_id", "organ", "organ_group", "raw_path", "post_path", "reference_kind", "reference_path",
        "parent_roi_path", "before_voxels", "after_voxels", "false_positive_voxels_removed",
        "connected_component_count_before", "connected_component_count_after",
        "largest_component_ratio_before", "largest_component_ratio_after",
        "before_postprocessing_dice", "after_postprocessing_dice",
        "before_volume_ratio", "after_volume_ratio", "dice_delta",
    ]
    df = pd.DataFrame(rows, columns=required_columns)
    per_case_path = args.output_dir / "student_postprocess_before_after_per_case_organ.csv"
    df.to_csv(per_case_path, index=False)
    summary_path = args.output_dir / "student_postprocess_before_after_organ_summary.csv"
    summary_columns = [
        "organ", "n", "before_postprocessing_dice", "after_postprocessing_dice",
        "before_volume_ratio", "after_volume_ratio", "false_positive_voxels_removed",
        "connected_component_count_before", "connected_component_count_after",
        "largest_component_ratio_before", "largest_component_ratio_after",
    ]
    if not df.empty:
        summary = df.groupby("organ").agg(
            n=("organ", "size"),
            before_postprocessing_dice=("before_postprocessing_dice", "mean"),
            after_postprocessing_dice=("after_postprocessing_dice", "mean"),
            before_volume_ratio=("before_volume_ratio", "mean"),
            after_volume_ratio=("after_volume_ratio", "mean"),
            false_positive_voxels_removed=("false_positive_voxels_removed", "sum"),
            connected_component_count_before=("connected_component_count_before", "mean"),
            connected_component_count_after=("connected_component_count_after", "mean"),
            largest_component_ratio_before=("largest_component_ratio_before", "mean"),
            largest_component_ratio_after=("largest_component_ratio_after", "mean"),
        ).reset_index()
        summary.to_csv(summary_path, index=False)
        visual_candidates = df.sort_values(["false_positive_voxels_removed", "dice_delta"], ascending=[False, False]).head(max(0, args.top_visuals))
        visual_root = args.output_dir / "visuals"
        for row in visual_candidates.to_dict("records"):
            copy_review_masks(row, visual_root)
    else:
        pd.DataFrame(columns=summary_columns).to_csv(summary_path, index=False)
    report = {
        "stage": "student_postprocess_before_after_evaluation",
        "metric_warning": "GT metrics are true accuracy only when --gt-root is trusted. Teacher metrics are consistency metrics.",
        "raw_root": str(args.raw_root),
        "post_root": str(args.post_root),
        "teacher_root": str(args.teacher_root) if args.teacher_root else None,
        "gt_root": str(args.gt_root) if args.gt_root else None,
        "rows": len(rows),
        "reference_kind_counts": Counter(str(r.get("reference_kind") or "missing") for r in rows),
        "policy_version": policy.get("version") if isinstance(policy, dict) else None,
        "outputs": {
            "per_case_organ": str(per_case_path),
            "organ_summary": str(args.output_dir / "student_postprocess_before_after_organ_summary.csv"),
            "visuals": str(args.output_dir / "visuals"),
        },
    }
    (args.output_dir / "student_postprocess_before_after_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
