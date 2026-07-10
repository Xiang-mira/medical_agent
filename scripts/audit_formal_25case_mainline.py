#!/usr/bin/env python3
"""Static/project-contract audit before launching the 25-case formal run."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

MAINLINE_FILES = [
    ROOT / "scripts" / "run_em_training.py",
    ROOT / "agent-harness" / "cli_anything" / "medai" / "core" / "multimodel_loop.py",
    ROOT / "agent-harness" / "cli_anything" / "medai" / "core" / "voxtell_student.py",
]

DISALLOWED_MAINLINE_PATTERNS = [
    (re.compile(r"MEDAI_GT_ROOT"), "MEDAI_GT_ROOT"),
    (re.compile(r"--gt-root"), "--gt-root"),
    (re.compile(r"student_vs_gt\.csv"), "student_vs_gt.csv"),
    (re.compile(r"teacher_vs_gt\.csv"), "teacher_vs_gt.csv"),
    (re.compile(r"metric_target[\"']?\s*[:=]\s*[\"']GT[\"']"), "metric_target=GT"),
    (re.compile(r"dice_[A-Za-z0-9_]*_vs_gt"), "dice_*_vs_gt"),
]

REQUIRED_MAINLINE_SNIPPETS = [
    ("scripts/run_em_training.py", "use_annotation_folder_reference=False", "E-step disables annotation_folder/GT reference use"),
    ("scripts/run_em_training.py", "compute_round_pseudo_qc_chain", "pseudo-only QC chain exists"),
    ("scripts/run_em_training.py", "student_predictions_shapekit", "student masks pass through ShapeKit"),
    ("scripts/run_em_training.py", "student_training_convergence_audit_not_passed", "Round2 fails closed when student training audit fails"),
    ("agent-harness/cli_anything/medai/core/multimodel_loop.py", "student_shapekit_required", "student_prev must have per-organ ShapeKit success"),
    ("agent-harness/cli_anything/medai/core/multimodel_loop.py", "labelcritic_gate_fail_closed", "LabelCritic gate is fail-closed"),
    ("agent-harness/cli_anything/medai/core/voxtell_student.py", "row.pop(\"ground_truth_status\", None)", "legacy ground_truth_status is not written to new student manifest rows"),
]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_case_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [str(row.get("case_id") or "").strip() for row in csv.DictReader(handle) if str(row.get("case_id") or "").strip()]


def target_count() -> int:
    doc = read_json(ROOT / "configs" / "student_3d_prompt_target_organs.json", {})
    return len([x for x in doc.get("target_organs", []) if str(x).strip()])


def audit_static_source() -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    legacy_compat: list[dict[str, str]] = []
    for path in MAINLINE_FILES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(ROOT))
        for pattern, label in DISALLOWED_MAINLINE_PATTERNS:
            if pattern.search(text):
                failures.append({"file": rel, "pattern": label})
        if "ground_truth_status" in text:
            legacy_compat.append({
                "file": rel,
                "note": "legacy reader compatibility only; allowed if converted/removed before new artifact writes",
            })
    missing_required: list[dict[str, str]] = []
    for rel, snippet, reason in REQUIRED_MAINLINE_SNIPPETS:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        if snippet not in text:
            missing_required.append({"file": rel, "snippet": snippet, "reason": reason})
    return {
        "status": "passed" if not failures and not missing_required else "failed",
        "disallowed_mainline_hits": failures,
        "missing_required_snippets": missing_required,
        "legacy_compatibility_mentions": legacy_compat,
        "interpretation": "No mainline GT metrics/dependencies are allowed; legacy field readers may remain only as compatibility shims.",
    }


def audit_case_list(path: Path, expected_cases: int) -> dict[str, Any]:
    ids = read_case_ids(path) if path.exists() else []
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    return {
        "status": "passed" if len(ids) == expected_cases and len(set(ids)) == expected_cases and "annotation_folder" not in text else "failed",
        "case_list": str(path),
        "case_count": len(ids),
        "unique_case_count": len(set(ids)),
        "expected_case_count": expected_cases,
        "has_annotation_folder_column": "annotation_folder" in text.splitlines()[0] if text else False,
        "case_ids": ids,
        "policy": "Formal 25-case case list should include CT paths only; annotation_folder/GT references are intentionally omitted.",
    }


def audit_reuse_report(output_root: Path) -> dict[str, Any]:
    report_path = output_root / "teacher_assets" / "round1_25case_reuse" / "prepare_round1_25case_reuse.json"
    report = read_json(report_path, {})
    status = "passed" if report.get("status") == "passed" and len(report.get("reused_case_ids") or []) == 10 and len(report.get("new_case_ids") or []) == 15 else "failed"
    return {
        "status": status,
        "report_path": str(report_path),
        "prepare_status": report.get("status"),
        "reused_case_count": len(report.get("reused_case_ids") or []),
        "new_case_count": len(report.get("new_case_ids") or []),
        "case_list_policy": report.get("case_list_policy"),
        "reuse_policy": report.get("reuse_policy"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-list", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-cases", type=int, default=25)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    source = audit_static_source()
    case_list = audit_case_list(args.case_list, args.expected_cases)
    reuse = audit_reuse_report(args.output_root)
    target_space = {
        "status": "passed" if target_count() == 373 else "failed",
        "target_count": target_count(),
        "policy": "Every formal case must materialize one manifest row per configured target.",
    }
    failures = [
        name for name, section in {
            "source": source,
            "case_list": case_list,
            "reuse": reuse,
            "target_space": target_space,
        }.items()
        if section.get("status") != "passed"
    ]
    payload = {
        "stage": "formal_25case_mainline_audit",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "source": source,
        "case_list": case_list,
        "reuse": reuse,
        "target_space": target_space,
        "shape_student_gate_policy": "Raw student masks are audit-only; Round2 uses postprocessed student candidates and blocks per-organ student replacement without ShapeKit success evidence.",
        "metric_policy": "All formal metrics are pseudo-label consistency/student convergence, never GT accuracy.",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
