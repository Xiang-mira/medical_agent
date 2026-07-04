#!/usr/bin/env python3
"""Build a targeted Round2 repair/blocklist queue from existing audits.

This script is intentionally CPU/file-only. It does not run inference, does
not call LabelCritic/ShapeKit, and does not copy expert GT into training
labels. Its main artifact is a case-organ blocklist that prevents known bad
student masks from entering the next E-step competition.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"
DEFAULT_FAILURE_QUEUE = {
    "PanTS_00000270": ["pancreas", "stomach", "spleen", "liver", "colon", "small_bowel"],
    "PanTS_00000246": ["duodenum", "small_bowel"],
    "PanTS_00000100": ["colon", "small_bowel", "kidney_left", "adrenal_gland_left"],
}
HOLLOW_GI = {"colon", "duodenum", "small_bowel", "intestine", "stomach"}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metric_rows(path: Path, key: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id") or "")
            organ = str(row.get("organ") or "")
            if not case_id or not organ:
                continue
            value = row.get(key)
            parsed: float | None
            try:
                parsed = float(value) if value not in ("", None) else None
            except Exception:
                parsed = None
            out[(case_id, organ)] = {**row, key: parsed}
    return out


def selection_row(annotation_root: Path, case_id: str, organ: str) -> dict[str, Any]:
    doc = read_json(annotation_root / case_id / "selection_metadata.json", {})
    for collection in ("selected_organs", "selection_rows"):
        for row in doc.get(collection, []) or []:
            if isinstance(row, dict) and str(row.get("organ") or "") == organ:
                return {"metadata_collection": collection, **row}
    return {}


def selected_mask(annotation_root: Path, case_id: str, organ: str) -> str:
    row = selection_row(annotation_root, case_id, organ)
    for key in ("final_mask", "mask_path", "mask", "selected_prediction"):
        value = str(row.get(key) or "")
        if value and Path(value).is_file():
            return value
    fallback = annotation_root / case_id / "updated" / f"{organ}.nii.gz"
    return str(fallback) if fallback.is_file() else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Default: <run-root>/round2/metrics/evidence_chain_core_organs_vs_round1_reference_raw",
    )
    parser.add_argument("--round2-annotation-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--case-organ",
        action="append",
        default=[],
        help="Additional case:organ repair item. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    evidence_dir = (
        args.evidence_dir.resolve()
        if args.evidence_dir
        else run_root / "round2" / "metrics" / "evidence_chain_core_organs_vs_round1_reference_raw"
    )
    annotation_root = (
        args.round2_annotation_root.resolve()
        if args.round2_annotation_root
        else run_root / "round2" / "estep" / "annotation_versions"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_root / "round2" / "targeted_repairs"
    )

    queue: dict[str, set[str]] = {case: set(organs) for case, organs in DEFAULT_FAILURE_QUEUE.items()}
    for item in args.case_organ:
        if ":" not in item:
            raise SystemExit(f"--case-organ must be case:organ, got {item!r}")
        case_id, organ = [x.strip() for x in item.split(":", 1)]
        if case_id and organ:
            queue.setdefault(case_id, set()).add(organ)

    student_gt = read_metric_rows(evidence_dir / "student_vs_gt.csv", "dice_student_vs_gt")
    student_pseudo = read_metric_rows(evidence_dir / "student_vs_pseudo.csv", "dice_student_vs_pseudo")
    deltas = read_metric_rows(evidence_dir / "student_minus_best_teacher.csv", "student_minus_best_teacher_gt")

    rows: list[dict[str, Any]] = []
    for case_id in sorted(queue):
        for organ in sorted(queue[case_id]):
            key = (case_id, organ)
            sel = selection_row(annotation_root, case_id, organ)
            reasons = ["targeted_round2_failure_queue"]
            if organ in HOLLOW_GI:
                reasons.append("hollow_gi_requires_conservative_postprocess")
            gt_dice = student_gt.get(key, {}).get("dice_student_vs_gt")
            pseudo_dice = student_pseudo.get(key, {}).get("dice_student_vs_pseudo")
            delta = deltas.get(key, {}).get("student_minus_best_teacher_gt")
            if isinstance(gt_dice, float) and gt_dice < 0.7:
                reasons.append("student_vs_gt_below_0.70")
            if isinstance(pseudo_dice, float) and pseudo_dice < 0.7:
                reasons.append("student_vs_pseudo_below_0.70")
            if isinstance(delta, float) and delta < -0.05:
                reasons.append("student_behind_best_teacher_gt_by_gt_0.05")
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "reason": ";".join(dict.fromkeys(reasons)),
                "action": "block_student_competition_use_teacher_or_selected_fallback",
                "student_vs_gt_dice": gt_dice if gt_dice is not None else "",
                "student_vs_pseudo_dice": pseudo_dice if pseudo_dice is not None else "",
                "student_minus_best_teacher_gt": delta if delta is not None else "",
                "round2_selection_status": sel.get("selection_status", ""),
                "round2_selected_model": sel.get("selected_model", ""),
                "round2_grade": sel.get("grade", ""),
                "round2_target_type": sel.get("target_type", ""),
                "round2_selected_mask": selected_mask(annotation_root, case_id, organ),
                "gt_used_for_training": False,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    blocklist = output_dir / "student_case_organ_blocklist.csv"
    queue_csv = output_dir / "round2_targeted_repair_queue.csv"
    fields = [
        "case_id", "organ", "reason", "action", "student_vs_gt_dice",
        "student_vs_pseudo_dice", "student_minus_best_teacher_gt",
        "round2_selection_status", "round2_selected_model", "round2_grade",
        "round2_target_type", "round2_selected_mask", "gt_used_for_training",
    ]
    write_csv(queue_csv, rows, fields)
    write_csv(blocklist, rows, ["case_id", "organ", "reason", "action"])
    report = {
        "stage": "round2_targeted_repair_queue",
        "status": "success",
        "run_root": str(run_root),
        "evidence_dir": str(evidence_dir),
        "round2_annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "queue_csv": str(queue_csv),
        "case_organ_blocklist": str(blocklist),
        "items": len(rows),
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "gt_used_for_training": False,
        "policy": "Known bad student case-organs are blocked from next-round competition; selected teacher/pseudo labels remain the fallback.",
    }
    write_json(output_dir / "round2_targeted_repair_queue.json", {**report, "rows": rows})
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
