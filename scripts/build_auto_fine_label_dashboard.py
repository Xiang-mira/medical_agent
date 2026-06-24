#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build 373-organ auto fine-label and student capability dashboards.")
    ap.add_argument("--run-output", required=True, help="run-loop output folder containing annotation_versions/")
    ap.add_argument("--target-config", default=str(ROOT / "configs/student_3d_prompt_target_organs.json"))
    ap.add_argument("--student-summary", default="", help="Optional student_inference_summary.json")
    ap.add_argument("--failure-json", default="", help="Optional student_failure_cases.json")
    ap.add_argument("--output-dir", default="", help="Default: <run-output>/dashboards")
    return ap.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) or ["organ"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            })


def iter_case_metadata(annotation_versions: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(annotation_versions.glob("*/selection_metadata.json")):
        doc = load_json(meta_path, {})
        case_id = str(doc.get("case_id") or meta_path.parent.name)
        selected: list[dict[str, Any]] = []
        for item in doc.get("selected_organs", []) or []:
            if isinstance(item, dict) and item.get("organ"):
                selected.append({"case_id": case_id, **item})
        selection_rows: list[dict[str, Any]] = []
        for item in doc.get("selection_rows", []) or []:
            if isinstance(item, dict) and item.get("organ"):
                selection_rows.append({"case_id": case_id, **item})
        gap_rows: list[dict[str, Any]] = []
        for item in doc.get("gap_rows", []) or []:
            if isinstance(item, dict) and item.get("organ"):
                gap_rows.append({"case_id": case_id, **item})
        cases[case_id] = {
            "case_id": case_id,
            "ct_path": doc.get("ct_path"),
            "selected_organs": selected,
            "selection_rows": selection_rows,
            "gap_rows": gap_rows,
        }
    return cases


def iter_selected(annotation_versions: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in iter_case_metadata(annotation_versions).values():
        rows.extend(case["selected_organs"])
    return rows


def resolve_annotation_versions(run_output: Path) -> Path:
    """Resolve common E-step output layouts to the annotation_versions folder."""
    candidates = [
        run_output / "annotation_versions",
        run_output / "estep" / "annotation_versions",
        run_output / "round1" / "estep" / "annotation_versions",
    ]
    if run_output.name == "annotation_versions":
        candidates.insert(0, run_output)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return run_output / "annotation_versions"


def load_student_summary(summary_path: Path | None) -> dict[str, Any]:
    if not summary_path:
        return {}
    return load_json(summary_path, {})


def load_student_status(summary_path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    doc = load_student_summary(summary_path)
    if not doc:
        return out
    for result in doc.get("results", []) or []:
        case_id = result.get("case_id")
        for organ, status in (result.get("per_organ_status", {}) or {}).items():
            if case_id:
                out[f"{case_id}::{organ}"] = status
    return out


def load_failure_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    doc = load_json(path, {})
    rows = doc.get("review_items") or []
    return [r for r in rows if isinstance(r, dict)]


def capability_status(*, coverage: int, cases: int, mean_reliability: float | None, empty_rate: float, unresolved: int) -> str:
    if cases == 0 or coverage == 0:
        return "unresolved"
    coverage_rate = coverage / cases
    score = mean_reliability or 0.0
    if coverage_rate >= 0.8 and score >= 0.75 and empty_rate <= 0.1 and unresolved == 0:
        return "learned"
    if coverage_rate >= 0.5 and score >= 0.5:
        return "partial"
    if coverage > 0:
        return "weak"
    return "unresolved"


def build_case_organ_scores(
    selected_rows: list[dict[str, Any]],
    *,
    case_metadata: dict[str, dict[str, Any]] | None = None,
    target_organs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten selected label metadata into the agent's case-organ score table."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in selected_rows:
        case_id = str(row.get("case_id"))
        organ = str(row.get("organ"))
        seen.add((case_id, organ))
        labelcritic_records = row.get("labelcritic_records") or []
        labelcritic_used = bool(labelcritic_records)
        labelcritic_uncertain = any(
            (record.get("decision") or {}).get("winner") == "uncertain"
            for record in labelcritic_records
            if isinstance(record, dict)
        )
        rows.append({
            "case_id": case_id,
            "organ": organ,
            "mask_path": row.get("mask_path") or row.get("final_mask") or row.get("mask"),
            "auto_fine_label_reliability_score": row.get("auto_fine_label_reliability_score"),
            "estimated_reliability": row.get("estimated_reliability", row.get("evidence_confidence")),
            "evidence_confidence": row.get("evidence_confidence"),
            "evidence_scores": row.get("evidence_scores", {}),
            "missing_evidence": row.get("missing_evidence", []),
            "decision_status": row.get("decision_status"),
            "decision_reasons": row.get("decision_reasons", []),
            "target_type": row.get("target_type", "hard"),
            "probability_mask_path": row.get("probability_mask_path"),
            "voxel_uncertainty_path": row.get("voxel_uncertainty_path"),
            "independent_family_count": row.get("independent_family_count", 0),
            "family_membership": row.get("family_membership", {}),
            "conflict_score": row.get("conflict_score"),
            "winner_margin": row.get("winner_margin"),
            "scoring_schema_version": row.get("scoring_schema_version", "legacy"),
            "grade": row.get("grade"),
            "training_weight": row.get("training_weight"),
            "auto_fine_label_status": row.get("auto_fine_label_status"),
            "label_maturity_level": row.get("label_maturity_level"),
            "quality_status": row.get("quality_status"),
            "selected_model": row.get("selected_model"),
            "source_model": row.get("source_model"),
            "selection_method": row.get("selection_method"),
            "selection_status": row.get("selection_status"),
            "selected_pseudo_consistency_dice": row.get("selected_pseudo_consistency_dice") or row.get("selected_dice"),
            "selected_candidate_qc_status": row.get("selected_candidate_qc_status"),
            "selected_candidate_qc_score": row.get("selected_candidate_qc_score"),
            "shapekit_status": row.get("shapekit_status"),
            "route_primary_teacher": row.get("route_primary_teacher"),
            "route_backup_teachers": row.get("route_backup_teachers", []),
            "route_competition_teachers": row.get("route_competition_teachers", []),
            "route_confidence": row.get("route_confidence"),
            "candidate_mode": row.get("candidate_mode"),
            "labelcritic_used": labelcritic_used,
            "labelcritic_uncertain": labelcritic_uncertain,
            "labelcritic_compare_used": row.get("labelcritic_compare_used"),
            "labelcritic_compare_reason": row.get("labelcritic_compare_reason"),
            "labelcritic_compare_skipped_reason": row.get("labelcritic_compare_skipped_reason"),
            "labelcritic_grade_used": row.get("labelcritic_grade_used"),
            "labelcritic_grade_reason": row.get("labelcritic_grade_reason"),
            "labelcritic_grade_skipped_reason": row.get("labelcritic_grade_skipped_reason"),
            "auto_grade": row.get("auto_grade"),
            "auto_grade_accept": row.get("auto_grade_accept"),
            "auto_grade_swapped": row.get("auto_grade_swapped"),
            "teacher_lineage": row.get("teacher_lineage", []),
            "distillation_eligible": row.get("distillation_eligible"),
            "distillation_exclusion_reason": row.get("distillation_exclusion_reason"),
            "student_training_priority": row.get("student_training_priority"),
            "label_confidence": row.get("label_confidence"),
            "review_flags": row.get("review_flags", []),
            "quality_flags": row.get("quality_flags", []),
            "candidate_models": row.get("candidate_models", []),
            "comparison_candidate_models": row.get("comparison_candidate_models", []),
            "ground_truth_status": row.get("ground_truth_status", "machine_generated_candidate"),
            "metric_family": row.get("metric_family", "pseudo_consistency"),
            "requested_canonical_id": row.get("requested_canonical_id"),
            "resolved_canonical_id": row.get("resolved_canonical_id"),
            "comparison_family": row.get("comparison_family"),
            "identity_status": row.get("identity_status", "legacy_unverified"),
            "identity_mismatch_reasons": row.get("identity_mismatch_reasons", []),
            "accuracy_warning": "Auto-label quality score, not expert ground-truth accuracy.",
        })
    if case_metadata and target_organs:
        for case_id, case in sorted(case_metadata.items()):
            selection_by_organ = {
                str(item.get("organ")): item
                for item in case.get("selection_rows", [])
                if isinstance(item, dict) and item.get("organ")
            }
            gaps_by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for gap in case.get("gap_rows", []):
                if isinstance(gap, dict) and gap.get("organ"):
                    gaps_by_organ[str(gap.get("organ"))].append(gap)
            for organ in target_organs:
                key = (str(case_id), str(organ))
                if key in seen:
                    continue
                selection = selection_by_organ.get(str(organ), {})
                gaps = gaps_by_organ.get(str(organ), [])
                rows.append({
                    "case_id": case_id,
                    "organ": organ,
                    "ct_path": case.get("ct_path"),
                    "mask_path": None,
                    "auto_fine_label_reliability_score": 0.0,
                    "grade": "D",
                    "training_weight": 0.0,
                    "auto_fine_label_status": "unresolved",
                    "label_maturity_level": "L0",
                    "quality_status": "unresolved",
                    "selected_model": selection.get("selected_model"),
                    "source_model": selection.get("source_model") or selection.get("selected_model"),
                    "selection_method": selection.get("selection_method"),
                    "selection_status": selection.get("selection_status", "unresolved"),
                    "selected_pseudo_consistency_dice": selection.get("selected_pseudo_consistency_dice") or selection.get("selected_dice"),
                    "selected_candidate_qc_status": selection.get("selected_candidate_qc_status"),
                    "selected_candidate_qc_score": selection.get("selected_candidate_qc_score"),
                    "shapekit_status": selection.get("shapekit_status"),
                    "labelcritic_used": bool(selection.get("labelcritic_records") or selection.get("critic_records")),
                    "labelcritic_uncertain": any(
                        (record.get("decision") or {}).get("winner") == "uncertain"
                        for record in (selection.get("labelcritic_records") or selection.get("critic_records") or [])
                        if isinstance(record, dict)
                    ),
                    "auto_grade": selection.get("auto_grade"),
                    "auto_grade_accept": selection.get("auto_grade_accept"),
                    "auto_grade_swapped": selection.get("auto_grade_swapped"),
                    "review_flags": sorted({flag for gap in gaps for flag in (gap.get("review_flags") or [])}),
                    "quality_flags": ["unresolved_missing_final_label"],
                    "candidate_models": selection.get("candidate_models", []),
                    "comparison_candidate_models": selection.get("comparison_candidate_models", []),
                    "ground_truth_status": "machine_generated_candidate",
                    "metric_family": selection.get("metric_family", "pseudo_consistency"),
                    "gap_types": [gap.get("gap_type") for gap in gaps],
                    "gap_reasons": [gap.get("reason") for gap in gaps],
                    "accuracy_warning": "Unresolved auto-label score, not expert ground-truth accuracy.",
                })
    return rows


def main() -> int:
    args = parse_args()
    run_output = Path(args.run_output).resolve()
    target_config = Path(args.target_config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_output / "dashboards"
    target_doc = load_json(target_config, {})
    target_organs = [str(x) for x in target_doc.get("target_organs", [])]
    annotation_versions = resolve_annotation_versions(run_output)
    case_metadata = iter_case_metadata(annotation_versions)
    selected_rows = [row for case in case_metadata.values() for row in case["selected_organs"]]
    identity_excluded_rows = [row for row in selected_rows if row.get("identity_status") != "valid"]
    selected_rows = [row for row in selected_rows if row.get("identity_status") == "valid"]
    selected_by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_ids = sorted(case_metadata)
    for row in selected_rows:
        selected_by_organ[str(row.get("organ"))].append(row)

    student_summary_path = Path(args.student_summary).resolve() if args.student_summary else None
    student_summary_doc = load_student_summary(student_summary_path)
    student_status = load_student_status(student_summary_path)
    student_checkpoint_status = student_summary_doc.get("checkpoint_status") or student_summary_doc.get("training_status") or "missing"
    failure_rows = load_failure_rows(Path(args.failure_json).resolve() if args.failure_json else None)
    failure_by_organ = Counter(str(r.get("organ")) for r in failure_rows if r.get("organ"))

    organ_dashboard: list[dict[str, Any]] = []
    grade_totals = Counter()
    for organ in target_organs:
        rows = selected_by_organ.get(organ, [])
        grades = Counter(str(r.get("grade", "D")) for r in rows)
        grade_totals.update(grades)
        reliabilities = [
            float(r["auto_fine_label_reliability_score"])
            for r in rows
            if r.get("auto_fine_label_reliability_score") is not None
        ]
        student_items = [
            status
            for key, status in student_status.items()
            if key.endswith(f"::{organ}")
        ]
        empty_count = sum(1 for item in student_items if item.get("empty_mask") is True or item.get("status") == "empty")
        failed_count = sum(1 for item in student_items if item.get("status") in {"failed", "timed_out", "missing", "unreadable"})
        prompt_success_count = sum(1 for item in student_items if item.get("status") == "success")
        student_prev_selected_count = sum(1 for r in rows if r.get("selected_model") == "student_prev" or r.get("source_model") == "student_prev")
        student_vs_teacher_conflict_count = sum(
            1 for r in rows
            if "student_prev" in [str(x) for x in (r.get("candidate_models") or [])]
            and (r.get("selected_model") not in {"student_prev", None})
        )
        shapekit_fallback = sum(1 for r in rows if r.get("shapekit_status") in {"fallback_original", "unsupported_target", "failed"})
        labelcritic_uncertain = sum(
            1 for r in rows
            if any((rec.get("decision") or {}).get("winner") == "uncertain" for rec in (r.get("labelcritic_records") or []) if isinstance(rec, dict))
        )
        compare_used_count = sum(1 for r in rows if r.get("labelcritic_compare_used") is True)
        grade_used_count = sum(1 for r in rows if r.get("labelcritic_grade_used") is True)
        high_conflict_count = sum(1 for r in rows if r.get("selection_method") in {"label_critic", "label_critic_inconclusive"})
        training_excluded_count = sum(1 for r in rows if r.get("distillation_eligible") is False or float(r.get("training_weight") or 0.0) <= 0.0)
        single_family_count = sum(1 for r in rows if int(r.get("independent_family_count") or 0) <= 1)
        soft_label_count = sum(1 for r in rows if r.get("target_type") == "soft")
        provisional_count = sum(1 for r in rows if r.get("target_type") == "provisional")
        rejected_count = sum(1 for r in rows if r.get("target_type") == "rejected")
        evidence_missing = Counter(
            key for r in rows for key in (r.get("missing_evidence") or [])
        )
        grade_reject_count = sum(1 for r in rows if r.get("auto_grade_accept") is False)
        route_bands = Counter(str(r.get("route_confidence") or "unknown") for r in rows)
        mean_reliability = round(sum(reliabilities) / len(reliabilities), 6) if reliabilities else None
        missing_case_count = max(0, len(case_ids) - len(rows))
        unresolved = grades.get("D", 0) + missing_case_count
        student_count = len(student_items)
        empty_rate = round(empty_count / student_count, 6) if student_count else 0.0
        organ_dashboard.append({
            "organ": organ,
            "coverage": len(rows),
            "cases": len(case_ids),
            "A_count": grades.get("A", 0),
            "B_count": grades.get("B", 0),
            "C_count": grades.get("C", 0),
            "D_count": unresolved,
            "selected_D_count": grades.get("D", 0),
            "missing_case_count": missing_case_count,
            "mean_reliability_score": mean_reliability,
            "unresolved_count": unresolved,
            "shapekit_fallback_count": shapekit_fallback,
            "labelcritic_uncertain_count": labelcritic_uncertain,
            "compare_used_count": compare_used_count,
            "grade_used_count": grade_used_count,
            "high_conflict_count": high_conflict_count,
            "training_excluded_count": training_excluded_count,
            "single_family_count": single_family_count,
            "soft_label_count": soft_label_count,
            "provisional_count": provisional_count,
            "rejected_count": rejected_count,
            "missing_evidence_counts": dict(evidence_missing),
            "labelcritic_grade_reject_count": grade_reject_count,
            "route_confidence_band": route_bands.most_common(1)[0][0] if route_bands else "unknown",
            "student_checkpoint_status": student_checkpoint_status,
            "student_prompt_success_count": prompt_success_count,
            "student_empty_prediction_count": empty_count,
            "student_empty_prediction_rate": empty_rate,
            "student_failed_prediction_count": failed_count,
            "student_prev_selected_count": student_prev_selected_count,
            "student_vs_teacher_conflict_count": student_vs_teacher_conflict_count,
            "student_review_failure_count": failure_by_organ.get(organ, 0),
            "status": capability_status(
                coverage=len(rows),
                cases=len(case_ids),
                mean_reliability=mean_reliability,
                empty_rate=empty_rate,
                unresolved=unresolved,
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    case_organ_scores = build_case_organ_scores(
        selected_rows,
        case_metadata=case_metadata,
        target_organs=target_organs,
    )
    write_csv(output_dir / "case_organ_label_scores.csv", case_organ_scores)
    (output_dir / "case_organ_label_scores.json").write_text(json.dumps({
        "stage": "case_organ_label_scores",
        "status": "success",
        "rows": len(case_organ_scores),
        "annotation_versions": str(annotation_versions),
        "accuracy_warning": "Rows are auto-label quality scores and audit metadata, not expert ground-truth accuracy.",
        "items": case_organ_scores,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "organ_capability_dashboard.csv", organ_dashboard)
    (output_dir / "organ_capability_dashboard.json").write_text(json.dumps({
        "stage": "organ_capability_dashboard",
        "status": "success" if len(organ_dashboard) == 373 else "failed",
        "target_organs": len(target_organs),
        "rows": len(organ_dashboard),
        "accuracy_warning": "Dashboard measures auto-label consistency and process quality, not expert ground-truth accuracy.",
        "items": organ_dashboard,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    student_dashboard = {
        "stage": "student_capability_dashboard",
        "status": "success",
        "student_summary": str(student_summary_path) if student_summary_path else None,
        "checkpoint_status": student_checkpoint_status,
        "model_dir": student_summary_doc.get("model_dir"),
        "num_student_case_organs": len(student_status),
        "num_prompt_success": sum(1 for item in student_status.values() if item.get("status") == "success"),
        "num_empty_predictions": sum(1 for item in student_status.values() if item.get("empty_mask") is True or item.get("status") == "empty"),
        "num_failed_predictions": sum(1 for item in student_status.values() if item.get("status") in {"failed", "timed_out", "missing", "unreadable"}),
        "num_failure_review_items": len(failure_rows),
        "note": "Student outputs are candidates for Round2, not automatic replacements.",
    }
    (output_dir / "student_capability_dashboard.json").write_text(json.dumps(student_dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "student_capability_dashboard.csv", [student_dashboard])

    summary = {
        "stage": "auto_fine_label_dataset_summary",
        "status": "success",
        "run_output": str(run_output),
        "annotation_versions": str(annotation_versions),
        "target_config": str(target_config),
        "target_organs": len(target_organs),
        "num_cases": len(case_ids),
        "num_selected_labels": len(selected_rows),
        "case_organ_score_rows": len(case_organ_scores),
        "identity_excluded_selected_rows": len(identity_excluded_rows),
        "case_organ_score_expected_rows": len(case_metadata) * len(target_organs),
        "formal_expected_cases": 50,
        "formal_expected_case_organ_rows": 50 * len(target_organs),
        "formal_missing_cases": max(0, 50 - len(case_metadata)),
        "formal_complete_50_cases": len(case_metadata) >= 50,
        "grade_counts": {grade: grade_totals.get(grade, 0) for grade in ["A", "B", "C", "D"]},
        "target_type_counts": dict(Counter(str(r.get("target_type") or "legacy") for r in selected_rows)),
        "single_family_labels": sum(1 for r in selected_rows if int(r.get("independent_family_count") or 0) <= 1),
        "missing_evidence_counts": dict(Counter(key for r in selected_rows for key in (r.get("missing_evidence") or []))),
        "scoring_schema_versions": dict(Counter(str(r.get("scoring_schema_version") or "legacy") for r in selected_rows)),
        "completion_criteria": {
            "organ_dashboard_rows_equal_373": len(organ_dashboard) == 373,
            "case_organ_label_scores_written": True,
            "case_organ_label_scores_include_unresolved": True,
            "formal_50_case_run_complete": len(case_metadata) >= 50,
            "expert_ground_truth_claimed": False,
            "student_outputs_candidate_only": True,
        },
    }
    (output_dir / "auto_fine_label_dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "success", "output_dir": str(output_dir), "rows": len(organ_dashboard)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
