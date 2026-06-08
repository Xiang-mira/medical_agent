#!/usr/bin/env python3
"""Mine case/organ pairs the 3D prompt student fails to learn.

This compares student predictions against the first-round selected best
pseudo-labels. The resulting metrics are pseudo-label consistency signals, not
true accuracy. Expert-label evaluation is a separate optional output that is
only produced when --fine-label-root is explicitly configured with real expert
labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CASE_LIST = PROJECT_ROOT / "data_manifest/case_list_50_tumor.csv"
TARGET_CONFIG = PROJECT_ROOT / "configs/student_3d_prompt_target_organs.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find student-vs-Round1 pseudo-label failure cases.")
    ap.add_argument("--student-rounds", default="1", help="Comma-separated student rounds to evaluate, e.g. 1,2,3.")
    ap.add_argument("--reference-round", type=int, default=1, help="Round containing selected pseudo labels.")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--case-list", default=str(CASE_LIST))
    ap.add_argument("--target-config", default=str(TARGET_CONFIG))
    ap.add_argument("--organs", default="", help="Optional comma-separated organ subset.")
    ap.add_argument("--reference-root", default="", help="Override reference root. Default: outputs/round<reference>/estep/annotation_versions.")
    ap.add_argument("--fine-label-root", default="", help="Optional expert fine-label root for true fine_label_eval metrics.")
    ap.add_argument("--student-root-template", default="", help="Optional template with {round}, default outputs/round{round}/student_predictions.")
    ap.add_argument("--dice-threshold", type=float, default=0.5)
    ap.add_argument("--volume-ratio-min", type=float, default=0.25)
    ap.add_argument("--volume-ratio-max", type=float, default=4.0)
    ap.add_argument("--output-dir", default="", help="Default: outputs/round<last_student_round>/failure_mining.")
    return ap.parse_args()


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("case_id")]


def load_organs(target_config: Path, subset: str) -> list[str]:
    if subset.strip():
        return [x.strip() for x in subset.split(",") if x.strip()]
    doc = json.loads(target_config.read_text(encoding="utf-8"))
    return list(doc.get("target_organs", []))


def load_selection_meta(reference_root: Path, case_id: str, organ: str) -> dict[str, Any]:
    path = reference_root / case_id / "selection_metadata.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for item in doc.get("selected_organs", []) or []:
        if isinstance(item, dict) and item.get("organ") == organ:
            return item
    return {}


def labelcritic_summary(selection_meta: dict[str, Any]) -> dict[str, Any]:
    records = selection_meta.get("labelcritic_records") or selection_meta.get("critic_records") or []
    winners = []
    decision_paths = []
    parse_statuses = []
    for record in records:
        if not isinstance(record, dict):
            continue
        decision = record.get("decision") or {}
        if decision.get("winner"):
            winners.append(decision.get("winner"))
        if decision.get("parse_status"):
            parse_statuses.append(decision.get("parse_status"))
        if record.get("output_json"):
            decision_paths.append(record.get("output_json"))
    final_winner = None
    if selection_meta.get("selection_method") == "label_critic":
        final_winner = selection_meta.get("selected_model")
    elif selection_meta.get("selection_method") in {"label_critic_fallback", "critic_disabled_fallback"}:
        final_winner = "fallback"
    return {
        "labelcritic_winners": winners,
        "labelcritic_final_winner": final_winner,
        "labelcritic_decision_paths": decision_paths,
        "labelcritic_parse_statuses": parse_statuses,
        "labelcritic_fallback_reason": selection_meta.get("fallback_reason"),
    }


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.asanyarray(nib.load(str(path)).dataobj) > 0
    except Exception:
        return None


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    total = int(a.sum()) + int(b.sum())
    return round(2 * inter / total, 6) if total > 0 else 1.0


def compare_pair(
    *,
    round_idx: int,
    case_id: str,
    organ: str,
    student_mask_path: Path,
    reference_mask_path: Path,
    selection_meta: dict[str, Any],
    dice_threshold: float,
    volume_ratio_min: float,
    volume_ratio_max: float,
) -> dict[str, Any]:
    student = load_mask(student_mask_path)
    ref = load_mask(reference_mask_path)
    flags: list[str] = []
    status = "ok"
    dsc: float | None = None
    student_voxels: int | None = None
    reference_voxels: int | None = None
    volume_ratio: float | None = None

    if ref is None:
        flags.append("missing_reference_pseudo_label")
    if student is None:
        flags.append("missing_student_prediction")
    if ref is not None:
        reference_voxels = int(ref.sum())
        if reference_voxels == 0:
            flags.append("empty_reference")
    if student is not None:
        student_voxels = int(student.sum())
        if student_voxels == 0:
            flags.append("empty_student")

    if student is not None and ref is not None:
        if student.shape != ref.shape:
            flags.append("shape_mismatch")
        else:
            dsc = dice(student, ref)
            if dsc < dice_threshold:
                flags.append("low_student_vs_pseudo_dice")
            if reference_voxels and reference_voxels > 0:
                volume_ratio = round(float(student_voxels or 0) / float(reference_voxels), 6)
                if volume_ratio < volume_ratio_min:
                    flags.append("student_volume_too_small")
                if volume_ratio > volume_ratio_max:
                    flags.append("student_volume_too_large")

    source_flags = list(selection_meta.get("review_flags", []) or [])
    quality_flags = list(selection_meta.get("quality_flags", []) or [])
    for flag in source_flags + quality_flags:
        if flag and flag not in flags:
            flags.append(f"source_{flag}")

    if flags:
        status = "review"
    review_reasons = []
    for flag in flags:
        review_reasons.append(str(flag))
    if selection_meta.get("selection_status") == "fallback":
        review_reasons.append("round1_selection_fallback")
    if selection_meta.get("shapekit_status") not in {None, "success", "skipped_dry_run"}:
        review_reasons.append(f"round1_shapekit_{selection_meta.get('shapekit_status')}")
    critic_info = labelcritic_summary(selection_meta)

    return {
        "round": round_idx,
        "case_id": case_id,
        "organ": organ,
        "status": status,
        "student_vs_pseudo_dice": dsc,
        "pseudo_consistency_dice": dsc,
        "student_voxels": student_voxels,
        "reference_voxels": reference_voxels,
        "volume_ratio": volume_ratio,
        "flags": flags,
        "student_mask": str(student_mask_path),
        "reference_mask": str(reference_mask_path),
        "selected_model": selection_meta.get("selected_model"),
        "source_model": selection_meta.get("source_model", selection_meta.get("selected_model")),
        "candidate_models": selection_meta.get("candidate_models", []),
        "candidate_count": selection_meta.get("candidate_count"),
        "selection_method": selection_meta.get("selection_method"),
        "selection_status": selection_meta.get("selection_status"),
        "fallback_reason": selection_meta.get("fallback_reason"),
        "quality_status": selection_meta.get("quality_status"),
        "review_reasons": review_reasons,
        "shapekit_status": selection_meta.get("shapekit_status"),
        "shapekit_reason": selection_meta.get("shapekit_reason"),
        **critic_info,
        "ground_truth_status": selection_meta.get("ground_truth_status", "pseudo_label_candidate"),
        "metric_family": "pseudo_consistency",
        "metric_scope": "student_vs_selected_pseudo_label_consistency",
    }


def compare_fine_label_pair(
    *,
    round_idx: int,
    case_id: str,
    organ: str,
    student_mask_path: Path,
    fine_label_mask_path: Path,
    dice_threshold: float,
    volume_ratio_min: float,
    volume_ratio_max: float,
) -> dict[str, Any]:
    row = compare_pair(
        round_idx=round_idx,
        case_id=case_id,
        organ=organ,
        student_mask_path=student_mask_path,
        reference_mask_path=fine_label_mask_path,
        selection_meta={"ground_truth_status": "expert_fine_label"},
        dice_threshold=dice_threshold,
        volume_ratio_min=volume_ratio_min,
        volume_ratio_max=volume_ratio_max,
    )
    row["metric_family"] = "fine_label_eval"
    row["metric_scope"] = "student_vs_expert_fine_label"
    row["ground_truth_status"] = "expert_fine_label"
    row["fine_label_mask"] = str(fine_label_mask_path)
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["round"]), str(row["organ"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (round_idx, organ), organ_rows in sorted(grouped.items()):
        dices = [float(r["student_vs_pseudo_dice"]) for r in organ_rows if r.get("student_vs_pseudo_dice") is not None]
        review_rows = [r for r in organ_rows if r.get("status") == "review"]
        summary.append({
            "round": round_idx,
            "organ": organ,
            "n": len(organ_rows),
            "n_with_dice": len(dices),
            "mean_student_vs_pseudo_dice": round(float(np.mean(dices)), 6) if dices else None,
            "min_student_vs_pseudo_dice": round(float(np.min(dices)), 6) if dices else None,
            "mean_pseudo_consistency_dice": round(float(np.mean(dices)), 6) if dices else None,
            "min_pseudo_consistency_dice": round(float(np.min(dices)), 6) if dices else None,
            "review_count": len(review_rows),
            "review_rate": round(len(review_rows) / len(organ_rows), 6) if organ_rows else 0.0,
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) or ["round", "case_id", "organ"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            })


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    rounds = [int(x.strip()) for x in args.student_rounds.split(",") if x.strip()]
    reference_root = Path(args.reference_root).resolve() if args.reference_root else output_root / f"round{args.reference_round}" / "estep" / "annotation_versions"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else output_root / f"round{rounds[-1]}" / "failure_mining"
    cases = load_cases(Path(args.case_list).resolve())
    organs = load_organs(Path(args.target_config).resolve(), args.organs)

    rows: list[dict[str, Any]] = []
    fine_label_rows: list[dict[str, Any]] = []
    fine_label_root = Path(args.fine_label_root).resolve() if args.fine_label_root else None
    for round_idx in rounds:
        if args.student_root_template:
            student_root = Path(args.student_root_template.format(round=round_idx)).resolve()
        else:
            student_root = output_root / f"round{round_idx}" / "student_predictions"
        for case in cases:
            case_id = case["case_id"]
            for organ in organs:
                rows.append(compare_pair(
                    round_idx=round_idx,
                    case_id=case_id,
                    organ=organ,
                    student_mask_path=student_root / case_id / f"{organ}.nii.gz",
                    reference_mask_path=reference_root / case_id / "updated" / f"{organ}.nii.gz",
                    selection_meta=load_selection_meta(reference_root, case_id, organ),
                    dice_threshold=args.dice_threshold,
                    volume_ratio_min=args.volume_ratio_min,
                    volume_ratio_max=args.volume_ratio_max,
                ))
                if fine_label_root:
                    fine_label_rows.append(compare_fine_label_pair(
                        round_idx=round_idx,
                        case_id=case_id,
                        organ=organ,
                        student_mask_path=student_root / case_id / f"{organ}.nii.gz",
                        fine_label_mask_path=fine_label_root / case_id / "updated" / f"{organ}.nii.gz",
                        dice_threshold=args.dice_threshold,
                        volume_ratio_min=args.volume_ratio_min,
                        volume_ratio_max=args.volume_ratio_max,
                    ))

    review_rows = [r for r in rows if r.get("status") == "review"]
    organ_summary = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "student_failure_cases.csv", review_rows)
    write_csv(output_dir / "student_failure_all_comparisons.csv", rows)
    write_csv(output_dir / "student_failure_organ_summary.csv", organ_summary)
    if fine_label_rows:
        write_csv(output_dir / "fine_label_eval_all_comparisons.csv", fine_label_rows)
        write_csv(output_dir / "fine_label_eval_organ_summary.csv", summarize(fine_label_rows))
    result = {
        "stage": "student_failure_mining",
        "status": "success",
        "metric_scope": "student_vs_selected_pseudo_label_consistency",
        "metric_family": "pseudo_consistency",
        "pseudo_consistency_definition": "student prediction compared with selected pseudo label; not expert-label accuracy",
        "accuracy_warning": "These are not true accuracy metrics. Expert-label evaluation is separate and only produced through --fine-label-root.",
        "student_rounds": rounds,
        "reference_round": args.reference_round,
        "reference_root": str(reference_root),
        "num_cases": len(cases),
        "num_organs": len(organs),
        "num_comparisons": len(rows),
        "num_review_items": len(review_rows),
        "dice_threshold": args.dice_threshold,
        "volume_ratio_min": args.volume_ratio_min,
        "volume_ratio_max": args.volume_ratio_max,
        "outputs": {
            "review_csv": str(output_dir / "student_failure_cases.csv"),
            "all_csv": str(output_dir / "student_failure_all_comparisons.csv"),
            "organ_summary_csv": str(output_dir / "student_failure_organ_summary.csv"),
            "fine_label_eval_all_csv": str(output_dir / "fine_label_eval_all_comparisons.csv") if fine_label_rows else None,
            "fine_label_eval_summary_csv": str(output_dir / "fine_label_eval_organ_summary.csv") if fine_label_rows else None,
            "json": str(output_dir / "student_failure_cases.json"),
        },
        "fine_label_eval": {
            "status": "success" if fine_label_rows else "not_configured",
            "fine_label_root": str(fine_label_root) if fine_label_root else None,
            "num_comparisons": len(fine_label_rows),
            "metric_scope": "student_vs_expert_fine_label",
            "note": "Configure --fine-label-root only when expert labels are available.",
        },
        "review_items": review_rows[:500],
        "organ_summary": organ_summary,
    }
    (output_dir / "student_failure_cases.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "num_comparisons": len(rows),
        "num_review_items": len(review_rows),
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
