#!/usr/bin/env python3
"""Rebuild a round_summary.json from already-written artifacts.

This is a no-inference/no-training repair utility for interrupted/resumed EM
runs where the E-step gate, M-step, or student prediction artifacts completed
after the summary was last persisted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def student_contract(student_summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in student_summary.get("results", []) or [] if isinstance(row, dict)]
    target_organs_value = student_summary.get("target_organs")
    if isinstance(target_organs_value, int):
        target_organ_count = target_organs_value
    elif isinstance(target_organs_value, list):
        target_organ_count = len(target_organs_value)
    else:
        target_organ_count = 0
    total_masks = sum(int(row.get("num_masks") or 0) for row in rows)
    standardized = sum(len(row.get("standardized_masks") or []) for row in rows)
    official = sum(len(row.get("official_output_masks") or []) for row in rows)
    missing = 0
    extra = 0
    failed = 0
    empty = 0
    nonempty = 0
    case_rows = []
    for row in rows:
        expected = row.get("expected_masks") if isinstance(row.get("expected_masks"), dict) else {}
        row_missing = len(expected.get("missing", []) or [])
        row_extra = len(expected.get("extra", []) or [])
        row_failed = int(row.get("num_failed_organs") or 0)
        row_empty = int(row.get("num_empty_masks") or 0)
        row_nonempty = int(row.get("num_success_nonempty") or 0)
        missing += row_missing
        extra += row_extra
        failed += row_failed
        empty += row_empty
        nonempty += row_nonempty
        case_rows.append({
            "case_id": row.get("case_id"),
            "status": row.get("status"),
            "num_masks": int(row.get("num_masks") or 0),
            "standardized_masks": len(row.get("standardized_masks") or []),
            "official_output_masks": len(row.get("official_output_masks") or []),
            "missing": row_missing,
            "extra": row_extra,
            "failed_organs": row_failed,
            "empty_masks": row_empty,
            "nonempty_masks": row_nonempty,
        })
    return {
        "num_cases": int(student_summary.get("num_cases") or len(rows)),
        "result_cases": len(rows),
        "target_organs": target_organ_count,
        "total_masks": total_masks,
        "standardized_masks": standardized,
        "official_output_masks": official,
        "missing_masks": missing,
        "extra_masks": extra,
        "failed_organs": failed,
        "empty_masks": empty,
        "nonempty_masks": nonempty,
        "empty_mask_rate": round(empty / total_masks, 6) if total_masks else None,
        "nonempty_mask_rate": round(nonempty / total_masks, 6) if total_masks else None,
        "contract_success": bool(rows and missing == 0 and extra == 0 and failed == 0),
        "case_rows": case_rows,
    }


def rebuild_round_summary(run_root: Path, round_idx: int, *, backup: bool = True) -> dict[str, Any]:
    run_root = run_root.resolve()
    round_root = run_root / f"round{round_idx}"
    summary_path = round_root / "round_summary.json"
    old_summary = read_json(summary_path, {})
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    estep_root = round_root / "estep"
    mstep_root = round_root / "mstep"
    student_root = round_root / "student_predictions"
    dashboard_root = estep_root / "dashboards"
    metrics_root = round_root / "metrics"

    formal_gate = read_json(estep_root / "formal_gate.json", {})
    estep_run_summary = read_json(estep_root / "run_summary.json", {})
    mstep_result = read_json(mstep_root / "voxtell_prompt_mstep_result.json", {})
    train_result = read_json(mstep_root / "voxtell_prompt_train_result.json", {})
    quality_gate = read_json(mstep_root / "voxtell_student_quality_gate.json", {})
    student_summary = read_json(student_root / "student_inference_summary.json", {})
    dashboard_summary = read_json(dashboard_root / "auto_fine_label_dataset_summary.json", {})
    student_dashboard = read_json(dashboard_root / "student_capability_dashboard.json", {})
    postprocess_dir = round_root / "student_predictions_postprocessed"
    postprocess_summary_path = postprocess_dir / "organ_type_postprocess_summary.json"
    if not postprocess_summary_path.is_file():
        postprocess_dir = round_root / "student_predictions_targeted_postprocessed"
        postprocess_summary_path = postprocess_dir / "organ_type_postprocess_summary.json"
        if not postprocess_summary_path.is_file():
            postprocess_summary_path = postprocess_dir / "student_containment_postprocess_summary.json"
    postprocess_summary = read_json(postprocess_summary_path, {})
    targeted_queue = read_json(round_root / "targeted_repairs" / "round2_targeted_repair_queue.json", {})
    evidence_core_round2 = read_json(metrics_root / "evidence_chain_core_organs" / "evidence_chain_summary.json", {})
    evidence_core_vs_round1 = read_json(metrics_root / "evidence_chain_core_organs_vs_round1_reference_raw" / "evidence_chain_summary.json", {})
    evidence_targeted_postprocessed = read_json(metrics_root / "evidence_chain_targeted_postprocessed_vs_round1_reference" / "evidence_chain_summary.json", {})
    round_allowlist = read_json(metrics_root / "student_round3_gate" / "student_round2_organ_allowlist.json", {})

    contract = student_contract(student_summary)
    checkpoint = (
        mstep_result.get("finetuned_checkpoint")
        or train_result.get("finetuned_checkpoint")
        or str(mstep_root / "model_finetune.pth")
    )
    checkpoint_path = Path(str(checkpoint)) if checkpoint else None
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.is_file())
    eligible = bool(
        mstep_result.get("eligible_for_next_round_prompt_student")
        or mstep_result.get("checkpoint_eligible_for_next_round")
        or student_summary.get("checkpoint_eligible_for_next_round")
    )

    success = bool(
        formal_gate.get("status") == "success"
        and mstep_result.get("status") == "success"
        and checkpoint_exists
        and eligible
        and contract.get("contract_success")
    )

    summary = {
        "round": round_idx,
        "timestamp": timestamp,
        "rebuilt_from_artifacts": True,
        "rebuild_policy": "no_inference_no_training_summary_repair",
        "previous_summary_status": {
            "estep_status": old_summary.get("estep_status"),
            "mstep_status": old_summary.get("mstep_status"),
            "student_status": old_summary.get("student_status"),
            "success": old_summary.get("success"),
        },
        "estep_status": estep_run_summary.get("status") or old_summary.get("estep_status"),
        "estep_formal_gate": formal_gate,
        "formal_gate_path": str(estep_root / "formal_gate.json"),
        "mstep_status": mstep_result.get("status") or train_result.get("status"),
        "mstep_result_path": str(mstep_root / "voxtell_prompt_mstep_result.json"),
        "mstep_training_status": mstep_result.get("training_status"),
        "student_backend": old_summary.get("student_backend", "voxtell_style_3d_prompt"),
        "finetuned_checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_eligible_for_next_round": eligible,
        "student_status": student_summary.get("status"),
        "student_summary_path": str(student_root / "student_inference_summary.json"),
        "student_contract": contract,
        "student_accuracy_warning": student_summary.get("accuracy_warning"),
        "dashboard_summary": {
            "path": str(dashboard_root / "auto_fine_label_dataset_summary.json"),
            "status": dashboard_summary.get("status"),
            "target_organs": dashboard_summary.get("target_organs"),
            "num_selected_labels": dashboard_summary.get("num_selected_labels"),
            "grade_counts": dashboard_summary.get("grade_counts"),
            "completion_criteria": dashboard_summary.get("completion_criteria"),
        },
        "student_capability_dashboard": {
            "path": str(dashboard_root / "student_capability_dashboard.json"),
            "status": student_dashboard.get("status"),
            "checkpoint_status": student_dashboard.get("checkpoint_status"),
            "num_prompt_success": student_dashboard.get("num_prompt_success"),
            "num_empty_predictions": student_dashboard.get("num_empty_predictions"),
            "num_failed_predictions": student_dashboard.get("num_failed_predictions"),
        },
        "postprocess_summary": {
            "path": str(postprocess_summary_path),
            "status": postprocess_summary.get("status"),
            "processed_masks": postprocess_summary.get("processed_masks"),
            "parent_roots": postprocess_summary.get("parent_roots"),
        },
        "targeted_repair_queue": {
            "path": str(round_root / "targeted_repairs" / "round2_targeted_repair_queue.json"),
            "status": targeted_queue.get("status"),
            "items": targeted_queue.get("items"),
            "case_organ_blocklist": targeted_queue.get("case_organ_blocklist"),
        },
        "next_round_student_gate": {
            "path": str(metrics_root / "student_round3_gate" / "student_round2_organ_allowlist.json"),
            "status": round_allowlist.get("status"),
            "schema_version": round_allowlist.get("schema_version"),
            "allowed_count": round_allowlist.get("allowed_count"),
            "allowed_case_organ_count": round_allowlist.get("allowed_case_organ_count"),
            "routing_contract": round_allowlist.get("routing_contract"),
        },
        "metrics": {
            "metric_family": "pseudo_consistency_and_gt_when_available",
            "accuracy_warning": "Pseudo-label consistency is not expert accuracy; student-vs-GT is real only for rows with GT.",
            "core_organs_vs_round2_reference": evidence_core_round2,
            "core_organs_vs_round1_reference": evidence_core_vs_round1,
            "targeted_postprocessed_vs_round1_reference": evidence_targeted_postprocessed,
        },
        "success": success,
    }

    if backup and summary_path.exists():
        backup_path = round_root / "summary_backups" / f"round_summary.before_rebuild_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, backup_path)
        summary["previous_summary_backup"] = str(backup_path)
    write_json(summary_path, summary)
    return {"status": "success", "summary_path": str(summary_path), "success": success, "mstep_status": summary["mstep_status"], "student_status": summary["student_status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--round", dest="round_idx", type=int, default=2)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    result = rebuild_round_summary(args.run_root, args.round_idx, backup=not args.no_backup)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
