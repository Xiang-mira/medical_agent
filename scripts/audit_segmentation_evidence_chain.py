#!/usr/bin/env python3
"""CPU-only Student/Teacher/GT evidence-chain audit.

This script reads existing masks only. It never runs inference, training,
LabelCritic, Qwen, ShapeKit, torch, or CUDA.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from audit_autolabel_candidate_selection import binary_dice_paths, read_json, reference_kind, write_json
except ImportError:  # Allows importing this script as scripts.audit_segmentation_evidence_chain in tests.
    from scripts.audit_autolabel_candidate_selection import binary_dice_paths, read_json, reference_kind, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTEP = ROOT / "outputs/round1_373_hierarchical_repair_20260620/estep"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fieldnames})


def load_selection_rows(estep: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((estep / "cases").glob("*/pseudo_label_selection.json")):
        doc = read_json(path, {})
        for row in doc.get("selection_rows", []) or []:
            if isinstance(row, dict) and row.get("case_id") and row.get("organ"):
                rows.append({"selection_path": str(path), **row})
    return rows


def load_selected_by_key(estep: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((estep / "annotation_versions").glob("*/selection_metadata.json")):
        doc = read_json(path, {})
        case_id = str(doc.get("case_id") or path.parent.name)
        for row in doc.get("selected_organs", []) or []:
            if isinstance(row, dict) and row.get("organ"):
                out[(case_id, str(row["organ"]))] = {"metadata_path": str(path), **row}
    return out


def existing_first(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_student_mask(student_dir: Path | None, case_id: str, organ: str) -> tuple[Path | None, str | None]:
    if student_dir is None:
        return None, "student_dir_not_provided"
    if not student_dir.exists():
        return None, "student_dir_missing"
    direct_candidates = [
        student_dir / case_id / f"{organ}.nii.gz",
        student_dir / case_id / "segmentations" / f"{organ}.nii.gz",
        student_dir / case_id / "student_predictions" / f"{organ}.nii.gz",
        student_dir / "student_predictions" / case_id / f"{organ}.nii.gz",
        student_dir / "student_predictions" / case_id / "segmentations" / f"{organ}.nii.gz",
        student_dir / case_id / "predictions" / f"{organ}.nii.gz",
    ]
    found = existing_first(direct_candidates)
    if found:
        return found, None
    case_root = student_dir / case_id
    if case_root.exists():
        matches = sorted(case_root.rglob(f"{organ}.nii.gz"))
        if matches:
            return matches[0], None
    return None, "student_mask_missing"


def selected_pseudo_path(selection: dict[str, Any], selected_meta: dict[str, Any]) -> str | None:
    return (
        selection.get("selected_prediction")
        or selected_meta.get("mask_path")
        or selected_meta.get("final_mask")
        or selected_meta.get("mask")
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU-only evidence chain audit for Student, Teachers, selected pseudo-labels, and GT.")
    ap.add_argument("--estep", default=str(DEFAULT_ESTEP))
    ap.add_argument("--student-dir", default="", help="Optional existing student prediction root.")
    ap.add_argument("--output-dir", default="", help="Default: <estep>/audits/segmentation_evidence_chain")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--organs", default="", help="Optional comma-separated organ subset.")
    ap.add_argument("--skip-dice", action="store_true", help="Write coverage tables without reading NIfTI arrays.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    estep = Path(args.estep).resolve()
    student_dir = Path(args.student_dir).resolve() if args.student_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else estep / "audits" / "segmentation_evidence_chain"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = load_selection_rows(estep)
    if args.max_cases > 0:
        keep_cases = sorted({str(r["case_id"]) for r in selection_rows})[: args.max_cases]
        selection_rows = [r for r in selection_rows if str(r["case_id"]) in set(keep_cases)]
    if args.organs:
        keep_organs = {x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()}
        selection_rows = [r for r in selection_rows if str(r.get("organ")) in keep_organs]

    selected_by_key = load_selected_by_key(estep)
    student_vs_gt: list[dict[str, Any]] = []
    student_vs_pseudo: list[dict[str, Any]] = []
    teacher_vs_gt: list[dict[str, Any]] = []
    student_minus_teacher: list[dict[str, Any]] = []

    for selection in selection_rows:
        case_id = str(selection["case_id"])
        organ = str(selection["organ"])
        selected_meta = selected_by_key.get((case_id, organ), {})
        ref = selection.get("reference") or selected_meta.get("reference")
        ref_path = Path(str(ref)) if ref else None
        ref_kind = reference_kind(ref)
        gt_available = bool(ref_path and ref_path.exists() and ref_kind == "pants_gt")
        pseudo_path = selected_pseudo_path(selection, selected_meta)
        pseudo_exists = bool(pseudo_path and Path(str(pseudo_path)).exists())
        student_path, student_missing_reason = find_student_mask(student_dir, case_id, organ)
        student_exists = student_path is not None

        dice_student_gt = None if args.skip_dice else binary_dice_paths(student_path, ref_path)
        dice_student_pseudo = None if args.skip_dice else binary_dice_paths(student_path, pseudo_path)
        student_vs_gt.append({
            "case_id": case_id,
            "organ": organ,
            "student_mask": str(student_path) if student_path else "",
            "gt_mask": str(ref_path) if ref_path else "",
            "reference_kind": ref_kind,
            "gt_available": gt_available,
            "dice_student_vs_gt": dice_student_gt if gt_available else None,
            "not_evaluable_reason": None if gt_available and student_exists and dice_student_gt is not None else not_evaluable_reason(gt_available, student_exists, student_missing_reason, ref_path),
        })
        student_vs_pseudo.append({
            "case_id": case_id,
            "organ": organ,
            "student_mask": str(student_path) if student_path else "",
            "pseudo_mask": str(pseudo_path) if pseudo_path else "",
            "pseudo_available": pseudo_exists,
            "dice_student_vs_pseudo": dice_student_pseudo if pseudo_exists and student_exists else None,
            "not_evaluable_reason": None if pseudo_exists and student_exists and dice_student_pseudo is not None else (student_missing_reason or "pseudo_mask_missing"),
        })

        teacher_scores: list[tuple[str, float | None, str]] = []
        for candidate in selection.get("candidate_predictions", []) or []:
            if not isinstance(candidate, dict):
                continue
            model = str(candidate.get("model") or "unknown")
            pred = str(candidate.get("prediction") or "")
            dice_teacher_gt = None if args.skip_dice else binary_dice_paths(pred, ref_path)
            teacher_scores.append((model, dice_teacher_gt, pred))
            teacher_vs_gt.append({
                "case_id": case_id,
                "organ": organ,
                "teacher": model,
                "teacher_mask": pred,
                "gt_mask": str(ref_path) if ref_path else "",
                "reference_kind": ref_kind,
                "gt_available": gt_available,
                "dice_teacher_vs_gt": dice_teacher_gt if gt_available else None,
                "not_evaluable_reason": None if gt_available and dice_teacher_gt is not None else ("gt_missing" if not gt_available else "teacher_or_geometry_unreadable"),
                "is_selected_teacher": model == str(selection.get("selected_model")),
            })
        valid_teacher_scores = [(m, d, p) for m, d, p in teacher_scores if d is not None]
        best_teacher = max(valid_teacher_scores, key=lambda item: item[1]) if valid_teacher_scores else (None, None, None)
        student_minus_teacher.append({
            "case_id": case_id,
            "organ": organ,
            "student_mask": str(student_path) if student_path else "",
            "best_teacher": best_teacher[0],
            "best_teacher_mask": best_teacher[2],
            "dice_student_vs_gt": dice_student_gt if gt_available else None,
            "best_teacher_vs_gt": best_teacher[1] if gt_available else None,
            "student_minus_best_teacher_gt": (dice_student_gt - best_teacher[1]) if gt_available and dice_student_gt is not None and best_teacher[1] is not None else None,
            "student_exceeds_best_teacher": (dice_student_gt > best_teacher[1]) if gt_available and dice_student_gt is not None and best_teacher[1] is not None else None,
            "not_evaluable_reason": None if gt_available and dice_student_gt is not None and best_teacher[1] is not None else "missing_student_gt_or_teacher_gt",
        })

    write_csv(output_dir / "student_vs_gt.csv", student_vs_gt)
    write_csv(output_dir / "teacher_vs_gt.csv", teacher_vs_gt)
    write_csv(output_dir / "student_vs_pseudo.csv", student_vs_pseudo)
    write_csv(output_dir / "student_minus_best_teacher.csv", student_minus_teacher)
    summary = build_summary(student_vs_gt, student_vs_pseudo, teacher_vs_gt, student_minus_teacher, args.skip_dice)
    write_json(output_dir / "evidence_chain_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "student_vs_gt_evaluable", "student_vs_pseudo_evaluable", "teacher_vs_gt_evaluable", "student_exceeds_best_teacher_count"]}, indent=2))
    return 0


def not_evaluable_reason(gt_available: bool, student_exists: bool, student_reason: str | None, ref_path: Path | None) -> str:
    if not gt_available:
        return "gt_missing_or_not_real_gt"
    if not ref_path or not ref_path.exists():
        return "gt_path_missing"
    if not student_exists:
        return student_reason or "student_mask_missing"
    return "geometry_or_mask_unreadable"


def mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def build_summary(
    student_vs_gt: list[dict[str, Any]],
    student_vs_pseudo: list[dict[str, Any]],
    teacher_vs_gt: list[dict[str, Any]],
    student_minus_teacher: list[dict[str, Any]],
    skip_dice: bool,
) -> dict[str, Any]:
    svg = [float(r["dice_student_vs_gt"]) for r in student_vs_gt if r.get("dice_student_vs_gt") is not None]
    svp = [float(r["dice_student_vs_pseudo"]) for r in student_vs_pseudo if r.get("dice_student_vs_pseudo") is not None]
    tvg = [float(r["dice_teacher_vs_gt"]) for r in teacher_vs_gt if r.get("dice_teacher_vs_gt") is not None]
    deltas = [float(r["student_minus_best_teacher_gt"]) for r in student_minus_teacher if r.get("student_minus_best_teacher_gt") is not None]
    return {
        "stage": "segmentation_evidence_chain_audit",
        "status": "success",
        "skip_dice": skip_dice,
        "student_vs_gt_rows": len(student_vs_gt),
        "student_vs_gt_evaluable": len(svg),
        "mean_dice_student_vs_gt": mean_or_none(svg),
        "student_vs_pseudo_rows": len(student_vs_pseudo),
        "student_vs_pseudo_evaluable": len(svp),
        "mean_dice_student_vs_pseudo": mean_or_none(svp),
        "teacher_vs_gt_rows": len(teacher_vs_gt),
        "teacher_vs_gt_evaluable": len(tvg),
        "mean_dice_teacher_vs_gt": mean_or_none(tvg),
        "student_minus_best_teacher_evaluable": len(deltas),
        "mean_student_minus_best_teacher_gt": mean_or_none(deltas),
        "student_exceeds_best_teacher_count": sum(1 for r in student_minus_teacher if r.get("student_exceeds_best_teacher") is True),
        "not_evaluable_reasons": dict(Counter(str(r.get("not_evaluable_reason")) for r in [*student_vs_gt, *student_vs_pseudo, *teacher_vs_gt, *student_minus_teacher] if r.get("not_evaluable_reason"))),
        "accuracy_warning": "Student-vs-pseudo is distillation consistency. Student-vs-GT is real segmentation performance only when reference_kind=pants_gt.",
        "gpu_policy": "CPU-only audit; does not call inference, training, LabelCritic, Qwen, ShapeKit, torch, or CUDA.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
