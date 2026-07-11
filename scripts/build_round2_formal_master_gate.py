#!/usr/bin/env python3
"""Build the final fail-closed Round2 gate after student and LabelCritic checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/em_round1_25case_pseudo_label_20260709"
DEFAULT_TRAINSET_AUDIT_ROOT = (
    DEFAULT_OUTPUT_ROOT / "round1/trainset_pseudo_consistency_full_mstep_lr3e-5_negative_fix_reaudit"
)
DEFAULT_REPAIR_SUMMARY = DEFAULT_OUTPUT_ROOT / "round1/estep/repair/key_organ_coverage_diagnosis_after_repair.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "round2_formal_master_gate_after_trainset_and_labelcritic.json"


def read_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return doc if isinstance(doc, dict) else {}, True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def benchmark_passed(doc: dict[str, Any]) -> bool:
    if doc.get("status") == "passed":
        return True
    return any(bool(row.get("pass")) for row in doc.get("rows", []) if isinstance(row, dict))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(value)
    except Exception:
        return default


def repair_ready(doc: dict[str, Any]) -> bool:
    return doc.get("status") in {"ready_for_cached_reselection", "passed", "ready"}


def build_master_gate(
    trainset_gate_path: Path,
    negative_safe_summary_path: Path,
    negative_manifest_audit_path: Path,
    labelcritic_summary_path: Path,
    cached_reselection_summary_path: Path,
) -> dict[str, Any]:
    trainset_gate, trainset_exists = read_json(trainset_gate_path)
    negative_safe, negative_safe_exists = read_json(negative_safe_summary_path)
    negative_audit, negative_audit_exists = read_json(negative_manifest_audit_path)
    labelcritic, labelcritic_exists = read_json(labelcritic_summary_path)
    reselection, reselection_exists = read_json(cached_reselection_summary_path)

    checks = {
        "trainset_consistency_passed": trainset_exists and trainset_gate.get("status") == "passed"
        and trainset_gate.get("round2_progression_allowed") is True,
        "post_suppression_negative_fp_zero": negative_safe_exists
        and safe_int(negative_safe.get("pre_suppression_nonempty_negative_count")) >= 0
        and safe_int(negative_safe.get("suppressed_negative_candidate_count")) >= 0
        and safe_int(
            (trainset_gate.get("summary") or {}).get("negative_false_positive_count")
            if isinstance(trainset_gate.get("summary"), dict)
            else None,
            default=1,
        )
        == 0,
        "suspicious_abdominal_negatives_handled": negative_audit_exists
        and (
            safe_int(negative_audit.get("suspicious_abdominal_negative_count")) == 0
            or negative_audit.get("suspicious_abdominal_all_withheld_required") is True
        ),
        "labelcritic_known_better_benchmark_passed": labelcritic_exists and benchmark_passed(labelcritic),
        "cached_reselection_ready": reselection_exists and repair_ready(reselection),
    }
    missing_inputs = [
        name
        for name, exists in {
            "trainset_gate": trainset_exists,
            "negative_safe_summary": negative_safe_exists,
            "negative_manifest_audit": negative_audit_exists,
            "labelcritic_benchmark_summary": labelcritic_exists,
            "cached_reselection_summary": reselection_exists,
        }.items()
        if not exists
    ]
    block_reasons: list[str] = []
    if missing_inputs:
        block_reasons.extend(f"missing_{name}" for name in missing_inputs)
    for key, passed in checks.items():
        if not passed:
            block_reasons.append(key.replace("_passed", "_failed").replace("_zero", "_nonzero"))

    status = "passed" if not block_reasons and all(checks.values()) else "blocked"
    payload = {
        "stage": "round2_formal_master_gate_after_trainset_and_labelcritic",
        "status": status,
        "round2_formal_replacement_allowed": status == "passed",
        "block_reasons": sorted(set(block_reasons)),
        "checks": checks,
        "inputs": {
            "trainset_gate": str(trainset_gate_path),
            "negative_safe_summary": str(negative_safe_summary_path),
            "negative_manifest_audit": str(negative_manifest_audit_path),
            "labelcritic_benchmark_summary": str(labelcritic_summary_path),
            "cached_reselection_summary": str(cached_reselection_summary_path),
        },
        "summary": {
            "trainset_gate_status": trainset_gate.get("status"),
            "post_suppression_negative_false_positive_count": (trainset_gate.get("summary") or {}).get(
                "negative_false_positive_count"
            )
            if isinstance(trainset_gate.get("summary"), dict)
            else None,
            "pre_suppression_negative_false_positive_count": negative_safe.get(
                "pre_suppression_nonempty_negative_count"
            ),
            "negative_safe_suppressed_count": negative_safe.get("suppressed_negative_candidate_count"),
            "suspicious_abdominal_negative_count": negative_audit.get("suspicious_abdominal_negative_count"),
            "withheld_required_count": negative_audit.get("withheld_required_count"),
            "labelcritic_status": labelcritic.get("status"),
            "labelcritic_blocker": labelcritic.get("blocker"),
            "cached_reselection_status": reselection.get("status"),
            "cached_reselection_blocker": reselection.get("blocker"),
        },
        "policy": (
            "Formal Round2 replacement starts only after trainset pseudo-consistency, negative candidate safety, "
            "negative source audit, LabelCritic known-better benchmark, and cached reselection gates all pass."
        ),
        "teacher_inference_rerun": False,
        "metric_target": "selected_pseudo_label",
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trainset-gate",
        type=Path,
        default=DEFAULT_TRAINSET_AUDIT_ROOT / "round2_progression_gate_after_trainset_consistency.json",
    )
    ap.add_argument(
        "--negative-safe-summary",
        type=Path,
        default=DEFAULT_TRAINSET_AUDIT_ROOT / "negative_safe_postprocess_summary.json",
    )
    ap.add_argument(
        "--negative-manifest-audit",
        type=Path,
        default=DEFAULT_TRAINSET_AUDIT_ROOT / "negative_absent_manifest_source_audit.json",
    )
    ap.add_argument("--labelcritic-summary", type=Path, required=True)
    ap.add_argument("--cached-reselection-summary", type=Path, default=DEFAULT_REPAIR_SUMMARY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    payload = build_master_gate(
        args.trainset_gate,
        args.negative_safe_summary,
        args.negative_manifest_audit,
        args.labelcritic_summary,
        args.cached_reselection_summary,
    )
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
