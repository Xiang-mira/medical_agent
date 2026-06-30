#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generic teacher/student/GT segmentation evaluation chain.")
    ap.add_argument("--case-list", type=Path, default=None, help="CSV with case_id column. Defaults to student prediction dirs.")
    ap.add_argument("--student-root", type=Path, required=True, help="case_id/*.nii.gz student predictions.")
    ap.add_argument("--teacher-root", type=Path, default=None, help="case_id/updated/*.nii.gz or case_id/*.nii.gz teacher masks.")
    ap.add_argument("--gt-root", type=Path, default=None, help="case_id/segmentations/*.nii.gz or case_id/*.nii.gz GT masks.")
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--organ-map", type=Path, default=None, help="Optional JSON/CSV rows mapping organ,student,teacher,gt.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--volume-policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_targets(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("target_organs"), list):
        return list(doc["target_organs"])
    if isinstance(doc, list):
        return [str(x) for x in doc]
    raise ValueError(f"Cannot read target organs from {path}")


def load_cases(case_list: Path | None, student_root: Path) -> list[str]:
    if case_list and case_list.exists():
        df = pd.read_csv(case_list)
        col = "case_id" if "case_id" in df.columns else df.columns[0]
        return [str(x) for x in df[col].dropna().tolist()]
    return sorted(p.name for p in student_root.iterdir() if p.is_dir())


def load_mapping(path: Path | None, targets: list[str]) -> list[dict[str, str]]:
    if path is None:
        return [{"organ": t, "student": t, "teacher": t, "gt": t} for t in targets]
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("mappings", doc) if isinstance(doc, dict) else doc
    else:
        rows = pd.read_csv(path).to_dict("records")
    out: list[dict[str, str]] = []
    for row in rows:
        organ = str(row.get("organ") or row.get("target_organ") or row.get("pants_organ"))
        out.append({
            "organ": organ,
            "student": str(row.get("student") or row.get("student_organ") or row.get("target_match_preferred") or organ),
            "teacher": str(row.get("teacher") or row.get("teacher_organ") or row.get("target_match_preferred") or organ),
            "gt": str(row.get("gt") or row.get("gt_organ") or row.get("pants_organ") or organ),
        })
    return out


def candidate_mask_paths(root: Path | None, case_id: str, name: str, *, teacher: bool = False, gt: bool = False) -> list[Path]:
    if root is None:
        return []
    paths = []
    if teacher:
        paths.extend([root / case_id / "updated" / f"{name}.nii.gz", root / case_id / f"{name}.nii.gz"])
    elif gt:
        paths.extend([root / case_id / "segmentations" / f"{name}.nii.gz", root / case_id / f"{name}.nii.gz"])
    else:
        paths.append(root / case_id / f"{name}.nii.gz")
    # A common model-output naming convention.
    safe = name.replace("(", "").replace(")", "").replace(" ", "_")
    paths.append(root / case_id / f"ct_segment_the_{safe}.nii.gz")
    return paths


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def binarize(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr
    unique = np.unique(arr)
    if len(unique) <= 3 and set(unique.tolist()).issubset({0, 1, 255}):
        return arr > 0
    return arr >= threshold


def dice(a: np.ndarray, b: np.ndarray) -> float:
    av = int(a.sum())
    bv = int(b.sum())
    if av + bv == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / (av + bv))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def organ_group(organ: str) -> str:
    s = organ.lower()
    if any(x in s for x in ["vein", "artery", "vessel", "duct", "cava", "aorta", "postcava"]):
        return "vessel_or_duct"
    if any(x in s for x in ["stomach", "colon", "duodenum", "bowel", "intestine", "rectum"]):
        return "hollow_gi"
    if any(x in s for x in ["rib", "femur", "bone", "clavicula", "scapula", "sternum", "vertebra"]):
        return "bone_or_fragmented_structure"
    if any(x in s for x in ["liver", "spleen", "kidney", "pancreas", "lung", "adrenal"]):
        return "core_large_organ"
    if any(x in s for x in ["bladder", "prostate", "uterus", "gonad"]):
        return "partial_fov"
    return "longtail_uncertain"


def volume_thresholds(policy: dict[str, Any], group: str) -> tuple[float, float]:
    vg = policy.get("volume_gate", {}) if isinstance(policy, dict) else {}
    groups = vg.get("groups", {}) if isinstance(vg, dict) else {}
    default_min = float(vg.get("default_min_ratio", 0.2)) if isinstance(vg, dict) else 0.2
    default_max = float(vg.get("default_max_ratio", 3.0)) if isinstance(vg, dict) else 3.0
    spec = groups.get(group, {}) if isinstance(groups, dict) else {}
    return float(spec.get("min_ratio", default_min)), float(spec.get("max_ratio", default_max))


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    vox = int(mask.sum())
    if vox == 0:
        return {"component_count": 0, "largest_component_voxels": 0, "largest_component_fraction": 0.0}
    if ndi is None:
        return {"component_count": None, "largest_component_voxels": None, "largest_component_fraction": None}
    lab, n = ndi.label(mask)
    if n == 0:
        return {"component_count": 0, "largest_component_voxels": 0, "largest_component_fraction": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    return {"component_count": int(n), "largest_component_voxels": largest, "largest_component_fraction": float(largest / vox)}


def metric_block(prefix: str, reference: np.ndarray | None, pred: np.ndarray | None) -> dict[str, Any]:
    if reference is None or pred is None or reference.shape != pred.shape:
        return {f"{prefix}_available": False}
    rv = int(reference.sum())
    pv = int(pred.sum())
    ratio = (float(pv / rv) if rv > 0 else None)
    return {
        f"{prefix}_available": True,
        f"{prefix}_dice": dice(reference, pred),
        f"{prefix}_iou": iou(reference, pred),
        f"{prefix}_reference_voxels": rv,
        f"{prefix}_prediction_voxels": pv,
        f"{prefix}_volume_ratio": ratio,
    }


def gate_flags(*, teacher_voxels: int | None, student_voxels: int | None, ratio: float | None, min_ratio: float, max_ratio: float) -> list[str]:
    flags = []
    if teacher_voxels is None or student_voxels is None:
        flags.append("missing_teacher_or_student")
        return flags
    if teacher_voxels > 0 and student_voxels == 0:
        flags.append("empty_failure")
    if teacher_voxels == 0 and student_voxels > 0:
        flags.append("false_positive_present")
    if ratio is not None:
        if ratio > max_ratio:
            flags.append("oversegmentation_volume_ratio")
        if ratio < min_ratio:
            flags.append("undersegmentation_volume_ratio")
    return flags


def threshold_sweep(student_raw: np.ndarray | None, reference: np.ndarray | None, thresholds: list[float]) -> list[dict[str, Any]]:
    if student_raw is None or reference is None or student_raw.shape != reference.shape:
        return []
    rows = []
    ref = binarize(reference)
    for th in thresholds:
        pred = binarize(student_raw, th)
        rows.append({
            "threshold": th,
            "dice": dice(ref, pred),
            "iou": iou(ref, pred),
            "reference_voxels": int(ref.sum()),
            "prediction_voxels": int(pred.sum()),
            "volume_ratio": float(pred.sum() / ref.sum()) if ref.sum() > 0 else None,
        })
    return rows


def main() -> int:
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    policy = load_yaml(args.volume_policy)
    targets = load_targets(args.target_config)
    cases = load_cases(args.case_list, args.student_root)
    mappings = load_mapping(args.organ_map, targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for case_id in cases:
        for m in mappings:
            organ = m["organ"]
            group = organ_group(organ)
            min_ratio, max_ratio = volume_thresholds(policy, group)
            student_path = first_existing(candidate_mask_paths(args.student_root, case_id, m["student"]))
            teacher_path = first_existing(candidate_mask_paths(args.teacher_root, case_id, m["teacher"], teacher=True))
            gt_path = first_existing(candidate_mask_paths(args.gt_root, case_id, m["gt"], gt=True))
            student_raw = read_array(student_path) if student_path else None
            teacher_raw = read_array(teacher_path) if teacher_path else None
            gt_raw = read_array(gt_path) if gt_path else None
            student = binarize(student_raw) if student_raw is not None else None
            teacher = binarize(teacher_raw) if teacher_raw is not None else None
            gt = binarize(gt_raw) if gt_raw is not None else None
            row: dict[str, Any] = {
                "case_id": case_id,
                "organ": organ,
                "organ_group": group,
                "student_name": m["student"],
                "teacher_name": m["teacher"],
                "gt_name": m["gt"],
                "student_path": str(student_path) if student_path else "",
                "teacher_path": str(teacher_path) if teacher_path else "",
                "gt_path": str(gt_path) if gt_path else "",
                "student_exists": student_path is not None,
                "teacher_exists": teacher_path is not None,
                "gt_exists": gt_path is not None,
                "volume_min_ratio": min_ratio,
                "volume_max_ratio": max_ratio,
            }
            row.update(metric_block("student_teacher", teacher, student))
            row.update(metric_block("teacher_gt", gt, teacher))
            row.update(metric_block("student_gt", gt, student))
            if student is not None:
                row.update({f"student_{k}": v for k, v in component_stats(student).items()})
            teacher_voxels = int(teacher.sum()) if teacher is not None else None
            student_voxels = int(student.sum()) if student is not None else None
            ratio = row.get("student_teacher_volume_ratio")
            flags = gate_flags(
                teacher_voxels=teacher_voxels,
                student_voxels=student_voxels,
                ratio=ratio,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
            row["qc_flags"] = ";".join(flags)
            row["volume_gate_status"] = "pass" if not flags else "review"
            rows.append(row)

            ref_for_sweep = gt_raw if gt_raw is not None else teacher_raw
            ref_kind = "gt" if gt_raw is not None else ("teacher" if teacher_raw is not None else "")
            for srow in threshold_sweep(student_raw, ref_for_sweep, thresholds):
                sweep_rows.append({"case_id": case_id, "organ": organ, "reference_kind": ref_kind, **srow})

    per_case = pd.DataFrame(rows)
    for optional_col in [
        "student_teacher_dice",
        "teacher_gt_dice",
        "student_gt_dice",
        "student_teacher_volume_ratio",
        "student_gt_volume_ratio",
    ]:
        if optional_col not in per_case.columns:
            per_case[optional_col] = np.nan
    per_case.to_csv(args.output_dir / "evaluation_chain_per_case_organ.csv", index=False)
    if sweep_rows:
        sweep = pd.DataFrame(sweep_rows)
        sweep.to_csv(args.output_dir / "threshold_sweep_case_organ.csv", index=False)
        best = sweep.sort_values(["case_id", "organ", "dice"], ascending=[True, True, False]).groupby(["case_id", "organ"]).head(1)
        best.to_csv(args.output_dir / "threshold_sweep_best_per_case_organ.csv", index=False)
        best_org = best.groupby("organ").agg(best_threshold_median=("threshold", "median"), best_dice_mean=("dice", "mean"), best_volume_ratio_mean=("volume_ratio", "mean")).reset_index()
        best_org.to_csv(args.output_dir / "threshold_sweep_best_by_organ.csv", index=False)

    summary_rows = []
    for organ, g in per_case.groupby("organ"):
        summary_rows.append({
            "organ": organ,
            "organ_group": g["organ_group"].iloc[0],
            "n": len(g),
            "student_teacher_mean_dice": pd.to_numeric(g.get("student_teacher_dice"), errors="coerce").mean(),
            "teacher_gt_mean_dice": pd.to_numeric(g.get("teacher_gt_dice"), errors="coerce").mean(),
            "student_gt_mean_dice": pd.to_numeric(g.get("student_gt_dice"), errors="coerce").mean(),
            "student_teacher_mean_volume_ratio": pd.to_numeric(g.get("student_teacher_volume_ratio"), errors="coerce").mean(),
            "student_gt_mean_volume_ratio": pd.to_numeric(g.get("student_gt_volume_ratio"), errors="coerce").mean(),
            "overseg_cases": int(g["qc_flags"].astype(str).str.contains("oversegmentation_volume_ratio").sum()),
            "underseg_cases": int(g["qc_flags"].astype(str).str.contains("undersegmentation_volume_ratio").sum()),
            "empty_failure_cases": int(g["qc_flags"].astype(str).str.contains("empty_failure").sum()),
            "false_positive_cases": int(g["qc_flags"].astype(str).str.contains("false_positive_present").sum()),
            "review_cases": int((g["volume_gate_status"] == "review").sum()),
        })
    organ_summary = pd.DataFrame(summary_rows).sort_values(["student_gt_mean_dice", "student_teacher_mean_dice"], ascending=[False, False])
    organ_summary.to_csv(args.output_dir / "evaluation_chain_organ_summary.csv", index=False)
    case_summary = per_case.groupby("case_id").agg(
        organs=("organ", "size"),
        student_teacher_mean_dice=("student_teacher_dice", "mean"),
        teacher_gt_mean_dice=("teacher_gt_dice", "mean"),
        student_gt_mean_dice=("student_gt_dice", "mean"),
        review_cases=("volume_gate_status", lambda s: int((s == "review").sum())),
    ).reset_index()
    case_summary.to_csv(args.output_dir / "evaluation_chain_case_summary.csv", index=False)
    group_summary = organ_summary.groupby("organ_group").agg(
        organs=("organ", "size"),
        student_teacher_mean_dice=("student_teacher_mean_dice", "mean"),
        teacher_gt_mean_dice=("teacher_gt_mean_dice", "mean"),
        student_gt_mean_dice=("student_gt_mean_dice", "mean"),
        overseg_cases=("overseg_cases", "sum"),
        review_cases=("review_cases", "sum"),
    ).reset_index()
    group_summary.to_csv(args.output_dir / "organ_group_summary.csv", index=False)
    block_flags = (
        per_case["qc_flags"].astype(str).str.contains("oversegmentation_volume_ratio")
        | per_case["qc_flags"].astype(str).str.contains("empty_failure")
        | per_case["qc_flags"].astype(str).str.contains("false_positive_present")
    )
    blocklist = per_case.loc[block_flags, [
        "case_id", "organ", "student_name", "teacher_name", "organ_group",
        "qc_flags", "student_teacher_volume_ratio", "volume_min_ratio", "volume_max_ratio",
        "student_path", "teacher_path",
    ]].copy()
    if not blocklist.empty:
        blocklist["block_reason"] = blocklist["qc_flags"]
    blocklist.to_csv(args.output_dir / "next_round_blocklist.csv", index=False)
    report = {
        "metric_warning": "GT metrics are true accuracy only when supplied GT is trusted. Student-teacher metrics are consistency metrics.",
        "cases": len(cases),
        "organs": len(mappings),
        "rows": len(rows),
        "volume_gate_flag_counts": Counter(flag for row in rows for flag in str(row.get("qc_flags", "")).split(";") if flag),
        "outputs": {
            "per_case_organ": str(args.output_dir / "evaluation_chain_per_case_organ.csv"),
            "organ_summary": str(args.output_dir / "evaluation_chain_organ_summary.csv"),
            "case_summary": str(args.output_dir / "evaluation_chain_case_summary.csv"),
            "group_summary": str(args.output_dir / "organ_group_summary.csv"),
            "next_round_blocklist": str(args.output_dir / "next_round_blocklist.csv"),
        },
    }
    (args.output_dir / "evaluation_chain_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
