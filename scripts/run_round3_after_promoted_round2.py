#!/usr/bin/env python3
"""Deprecated Round3 launcher.

Kept importable for historical preflight/audit helpers, but direct execution and
programmatic launch are disabled for the current formal-lite chain.  The current
Round2 repair is no_material_update and reuses Round1; there is no promoted new
Round2 checkpoint to seed Round3.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_em_training as em


ROUND_IDX = 3
MIN_FREE_GIB = 10.0
DISABLED_REASON = (
    "scripts/run_round3_after_promoted_round2.py is disabled for the current "
    "formal-lite chain. The audited Round2 repair is no_material_update and "
    "reuses the promoted Round1 checkpoint; do not launch Round3 from the old "
    "promoted-Round2 helper."
)


def fail_disabled_launcher() -> None:
    raise SystemExit(DISABLED_REASON)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _case_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [str(row.get("case_id") or "").strip() for row in csv.DictReader(handle) if row.get("case_id")]


def round3_preflight(case_list: Path) -> dict:
    root = em.OUTPUT_ROOT
    registry = read_json(root / "checkpoint_promotion_registry.json")
    current_r2 = (registry.get("rounds") or {}).get("2") or {}
    history = registry.get("history") or []
    blocked_history = [
        item for item in history
        if str(item.get("round")) == "2" and item.get("status") == "competition_blocked"
    ]
    round2_gate = read_json(root / "round2" / "estep" / "formal_gate.json")
    round2_manifest = read_json(root / "round2" / "mstep" / "voxtell_prompt_student_manifest.json")
    round2_mstep = read_json(root / "round2" / "mstep" / "voxtell_prompt_mstep_result.json")
    gate_summary = round2_manifest.get("training_gate_summary") or {}
    cumulative = round2_manifest.get("cumulative_manifest_summary") or {}
    promoted_model = em.resolve_promoted_student_model(2)
    round2_inference_audit = read_json(root / "round2" / "student_predictions" / "inference_audit.json")
    round2_scope = read_json(root / "round2" / "case_scope_preflight.json")
    case_provenance = em.case_list_provenance(case_list)
    free_bytes = shutil.disk_usage(root).free
    minimum_free_bytes = int(MIN_FREE_GIB * 1024**3)

    reasons: list[str] = []
    if round2_gate.get("status") != "success":
        reasons.append(f"round2_gate_not_success:{round2_gate.get('reason')}")
    if current_r2.get("status") != "promoted":
        reasons.append(f"round2_current_status_not_promoted:{current_r2.get('status')}")
    if not blocked_history:
        reasons.append("no_competition_blocked_round2_history")
    if promoted_model is None:
        reasons.append("promoted_round2_model_incomplete")
    elif current_r2.get("checkpoint_sha256") != em._sha256_file(promoted_model / "fold_0" / "checkpoint_final.pth"):
        reasons.append("promoted_round2_checkpoint_hash_mismatch")
    if gate_summary.get("negative_absent_excluded_from_positive_quota") is not True:
        reasons.append("negative_absent_positive_quota_audit_missing")
    if cumulative.get("missing_historical_trainable_positive_organs"):
        reasons.append("historical_replay_deficits_present")
    if (round2_mstep.get("retention_audit") or {}).get("status") != "passed":
        reasons.append("round2_retention_audit_not_passed")
    if not round2_mstep.get("eligible_for_next_round_prompt_student", round2_mstep.get("checkpoint_eligible_for_next_round", False)):
        reasons.append("round2_checkpoint_not_next_round_eligible")
    if round2_inference_audit.get("status") != "passed":
        reasons.append("round2_inference_audit_not_passed")
    audit_provenance = round2_inference_audit.get("provenance") or {}
    if audit_provenance.get("case_list_sha256") != case_provenance.get("sha256"):
        reasons.append("round2_round3_case_list_hash_mismatch")
    if round2_scope.get("status") != "passed" or set(round2_scope.get("expected_case_ids") or []) != set(_case_ids(case_list)):
        reasons.append("round2_case_scope_not_reusable")
    case_ids = _case_ids(case_list)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        reasons.append("round_requires_nonempty_unique_cohort")
    if free_bytes < minimum_free_bytes:
        reasons.append("insufficient_disk_space")

    preflight = {
        "stage": "round3_preflight",
        "status": "ready" if not reasons else "failed",
        "reasons": reasons,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(root),
        "case_list": case_provenance,
        "disk": {
            "free_bytes": free_bytes,
            "free_gib": round(free_bytes / 1024**3, 2),
            "minimum_free_bytes": minimum_free_bytes,
            "minimum_free_gib": MIN_FREE_GIB,
        },
        "round2_gate": {
            "status": round2_gate.get("status"),
            "reason": round2_gate.get("reason"),
            "ab_nonzero": round2_gate.get("ab_nonzero"),
            "cohort_failures": round2_gate.get("cohort_key_organ_coverage_failures"),
            "historical_replay_deficits": round2_gate.get("historical_replay_deficits"),
        },
        "round2_manifest": {
            "path": str(root / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"),
            "num_trainable_positive_items": round2_manifest.get("num_trainable_positive_items"),
            "grade_counts": round2_manifest.get("grade_counts"),
            "all_grade_counts": round2_manifest.get("all_grade_counts"),
            "negative_absent_excluded_from_positive_quota": gate_summary.get("negative_absent_excluded_from_positive_quota"),
            "cumulative_manifest_summary": cumulative,
        },
        "round2_promotion_current": current_r2,
        "round2_competition_blocked_history_count": len(blocked_history),
        "round2_retention_audit": round2_mstep.get("retention_audit"),
        "round2_inference_audit": round2_inference_audit,
        "round2_case_scope": round2_scope,
        "promoted_round2_model": str(promoted_model) if promoted_model else None,
    }
    write_json(root / "round3" / "round3_preflight.json", preflight)
    return preflight


def write_state(status: str, **extra) -> None:
    write_json(
        em.OUTPUT_ROOT / "round3" / "round3_continuation_state.json",
        {
            "stage": "round3_from_promoted_round2",
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        },
    )


def run_round3(case_list: Path) -> int:
    fail_disabled_launcher()
    em.CASE_LIST = case_list
    em.ensure_evaluation_protocol()
    em.write_round_run_spec(ROUND_IDX)
    preflight = round3_preflight(case_list)
    if preflight.get("status") != "ready":
        write_state("blocked_by_preflight", preflight=preflight)
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 2
    competition_root = em.materialize_student_competition_root(ROUND_IDX)
    if competition_root.get("status") != "success":
        write_state("blocked_by_competition_root", preflight=preflight, competition_root=competition_root)
        print(json.dumps(competition_root, indent=2, ensure_ascii=False))
        return 2
    validated_root, competition_validation = em._validated_student_prediction_root_for_next_round(ROUND_IDX)
    if competition_validation.get("status") != "success":
        write_state(
            "blocked_by_competition_root_validation",
            preflight=preflight,
            competition_root=competition_root,
            competition_validation=competition_validation,
        )
        print(json.dumps(competition_validation, indent=2, ensure_ascii=False))
        return 2
    write_state(
        "running_estep",
        preflight=preflight,
        competition_root=competition_root,
        competition_validation=competition_validation,
        validated_competition_root=str(validated_root),
    )
    estep_result = em.run_estep(ROUND_IDX)
    dashboard = em.build_round_label_scoring_dashboard(ROUND_IDX)
    manifest = em.build_student_dataset(ROUND_IDX)
    gate = em.formal_estep_gate(ROUND_IDX, estep_result, manifest)
    if gate.get("status") != "success":
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="competition_blocked",
            reason=f"round3_gate_failed:{gate.get('reason')}",
            mstep_result={},
            manifest_path=manifest,
            summary_path=em.OUTPUT_ROOT / "round3" / "round_summary.json",
        )
        write_state("blocked_by_gate", gate=gate, manifest=str(manifest))
        return 1

    write_state("running_mstep", gate=gate, manifest=str(manifest))
    mstep_result = em.run_student_mstep(ROUND_IDX, manifest, global_consolidation=False)
    if mstep_result.get("status") != "success":
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="competition_blocked",
            reason=f"round3_mstep_failed:{mstep_result.get('training_status') or mstep_result.get('reason')}",
            mstep_result=mstep_result,
            manifest_path=manifest,
            summary_path=em.OUTPUT_ROOT / "round3" / "round_summary.json",
        )
        write_state("mstep_failed", mstep_result=mstep_result)
        return 1

    write_state("running_student_inference", mstep_result=mstep_result)
    inference_summary = em.save_student_predictions(ROUND_IDX)
    inference_audit = inference_summary.get("inference_audit") or {}
    if inference_audit.get("status") != "passed":
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="competition_blocked",
            reason="round3_inference_audit_failed",
            mstep_result=mstep_result,
            manifest_path=manifest,
            summary_path=em.OUTPUT_ROOT / "round3" / "round_summary.json",
            inference_audit=inference_audit,
        )
        write_state("inference_audit_failed", inference_audit=inference_audit)
        return 1
    postprocess_summary = em.apply_round_organ_type_postprocess(ROUND_IDX)
    round_metrics = em.compute_round_metrics(ROUND_IDX)
    evaluation_chain = em.compute_round_evaluation_chain(ROUND_IDX)
    round_metrics["evaluation_chain"] = evaluation_chain
    success = bool(
        estep_result.get("status") == "success"
        and gate.get("status") == "success"
        and mstep_result.get("status") == "success"
        and mstep_result.get("eligible_for_next_round_prompt_student", mstep_result.get("checkpoint_eligible_for_next_round", False))
        and (mstep_result.get("retention_audit") or {}).get("status") == "passed"
        and inference_audit.get("status") == "passed"
        and evaluation_chain.get("status") == "success"
        and postprocess_summary.get("status") == "success"
    )
    summary_path = em.OUTPUT_ROOT / "round3" / "round_summary.json"
    promotion = em.record_checkpoint_promotion(
        ROUND_IDX,
        status="promoted" if success else "competition_blocked",
        reason=None if success else "round3_verification_failed",
        mstep_result=mstep_result,
        manifest_path=manifest,
        summary_path=summary_path,
        inference_audit=inference_audit,
    )
    summary = {
        "round": ROUND_IDX,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "estep_status": estep_result.get("status"),
        "estep_formal_gate": gate,
        "mstep_status": mstep_result.get("status"),
        "student_backend": em.STUDENT_BACKEND,
        "mstep_type": "local_update",
        "finetuned_checkpoint": mstep_result.get("finetuned_checkpoint"),
        "inference_audit": inference_audit,
        "round3_preflight": preflight,
        "metrics": {
            "overall_mean_dsc": round_metrics.get("overall_mean_dsc"),
            "overall_mean_pseudo_consistency_dsc": round_metrics.get("overall_mean_dsc"),
            "metric_family": round_metrics.get("metric_family", "pseudo_consistency"),
            "metric_scope": round_metrics.get("metric_scope", "student_vs_selected_pseudo_label"),
            "accuracy_warning": round_metrics.get("accuracy_warning", "Not true accuracy."),
            "top5_organs": round_metrics.get("top5_organs", []),
            "bottom5_organs": round_metrics.get("bottom5_organs", []),
            "evaluation_chain": evaluation_chain,
            "organ_type_postprocess": postprocess_summary,
        },
        "label_scoring_dashboard": dashboard,
        "reliability_weights": em._student_manifest_weight_summary(ROUND_IDX),
        "success": success,
        "checkpoint_promotion": promotion,
    }
    write_json(summary_path, summary)
    write_state("complete" if success else "verification_failed", summary=summary)
    print(json.dumps({"status": "success" if success else "failed", "summary": str(summary_path)}, indent=2))
    return 0 if success else 1


def main() -> int:
    fail_disabled_launcher()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", type=Path, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    case_list = args.case_list.expanduser().resolve()
    if not case_list.is_file():
        print(json.dumps({"status": "failed", "reason": f"case_list_missing:{case_list}"}, indent=2))
        return 2
    em.CASE_LIST = case_list
    if args.preflight_only:
        preflight = round3_preflight(case_list)
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight.get("status") == "ready" else 2
    return run_round3(case_list)


if __name__ == "__main__":
    raise SystemExit(main())
