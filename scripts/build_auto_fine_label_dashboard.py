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


def iter_selected(annotation_versions: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(annotation_versions.glob("*/selection_metadata.json")):
        doc = load_json(meta_path, {})
        for item in doc.get("selected_organs", []) or []:
            if isinstance(item, dict) and item.get("organ"):
                rows.append({
                    "case_id": doc.get("case_id") or meta_path.parent.name,
                    **item,
                })
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


def load_student_status(summary_path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not summary_path:
        return out
    doc = load_json(summary_path, {})
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


def main() -> int:
    args = parse_args()
    run_output = Path(args.run_output).resolve()
    target_config = Path(args.target_config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_output / "dashboards"
    target_doc = load_json(target_config, {})
    target_organs = [str(x) for x in target_doc.get("target_organs", [])]
    annotation_versions = resolve_annotation_versions(run_output)
    selected_rows = iter_selected(annotation_versions)
    selected_by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_ids = sorted({str(r.get("case_id")) for r in selected_rows if r.get("case_id")})
    for row in selected_rows:
        selected_by_organ[str(row.get("organ"))].append(row)

    student_summary_path = Path(args.student_summary).resolve() if args.student_summary else None
    student_status = load_student_status(student_summary_path)
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
        shapekit_fallback = sum(1 for r in rows if r.get("shapekit_status") in {"fallback_original", "unsupported_target", "failed"})
        labelcritic_uncertain = sum(
            1 for r in rows
            if any((rec.get("decision") or {}).get("winner") == "uncertain" for rec in (r.get("labelcritic_records") or []) if isinstance(rec, dict))
        )
        mean_reliability = round(sum(reliabilities) / len(reliabilities), 6) if reliabilities else None
        unresolved = grades.get("D", 0) + max(0, len(case_ids) - len(rows))
        student_count = len(student_items)
        empty_rate = round(empty_count / student_count, 6) if student_count else 0.0
        organ_dashboard.append({
            "organ": organ,
            "coverage": len(rows),
            "cases": len(case_ids),
            "A_count": grades.get("A", 0),
            "B_count": grades.get("B", 0),
            "C_count": grades.get("C", 0),
            "D_count": grades.get("D", 0),
            "mean_reliability_score": mean_reliability,
            "unresolved_count": unresolved,
            "shapekit_fallback_count": shapekit_fallback,
            "labelcritic_uncertain_count": labelcritic_uncertain,
            "student_empty_prediction_count": empty_count,
            "student_empty_prediction_rate": empty_rate,
            "student_failed_prediction_count": failed_count,
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
        "num_student_case_organs": len(student_status),
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
        "grade_counts": {grade: grade_totals.get(grade, 0) for grade in ["A", "B", "C", "D"]},
        "completion_criteria": {
            "organ_dashboard_rows_equal_373": len(organ_dashboard) == 373,
            "expert_ground_truth_claimed": False,
            "student_outputs_candidate_only": True,
        },
    }
    (output_dir / "auto_fine_label_dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "success", "output_dir": str(output_dir), "rows": len(organ_dashboard)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
