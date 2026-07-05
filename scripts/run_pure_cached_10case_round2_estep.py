#!/usr/bin/env python3
"""Run Round2 E-step/manifest audit for the pure-cached 10-case formal-lite run.

This helper is intentionally audit-only.  It may rebuild the Round2 E-step,
manifest, formal gate, and novelty audit, but it must not launch Round2 M-step
or student inference.  Training is allowed only through the formal repaired
Round2 M-step entry after a material-update novelty decision.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))
sys.path.insert(0, str(ROOT / "scripts"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    import scripts.run_em_training as em

    run_root = em.OUTPUT_ROOT
    audit_path = run_root / "round2_launch_audit.json"
    cache = em._round_teacher_cache_dirs(1)
    mstep_path = run_root / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    mstep = json.loads(mstep_path.read_text(encoding="utf-8")) if mstep_path.exists() else {}
    student_pred_root, student_gate = em._validated_student_prediction_root_for_next_round(2)
    round2_root = run_root / "round2"
    round2_clean = not round2_root.exists() or not any(round2_root.iterdir())
    case_rows = em.load_case_rows()
    case_ids = [str(row.get("case_id") or "") for row in case_rows]
    target_organs = em.load_student_target_organs()
    missing_teacher_cache = sorted(set(em.ALL_TEACHERS) - set(cache))
    round1_selected_root = em._round_selected_pseudo_label_root(1)
    selected_case_ids = {
        path.parent.name
        for path in round1_selected_root.glob("*/selection_metadata.json")
    } if round1_selected_root.exists() else set()
    round1_case_root = run_root / "round1" / "estep" / "cases"
    valid_hierarchical_cases = {
        case_id
        for case_id in case_ids
        if em._valid_hierarchical_manifest(
            round1_case_root / case_id / "hierarchical_inference_plan.json"
        )
    }
    previous_student_model = run_root / "round1" / "mstep" / "voxtell_finetuned_model"
    previous_checkpoint_complete = (
        (previous_student_model / "plans.json").is_file()
        and (previous_student_model / "fold_0" / "checkpoint_final.pth").is_file()
    )
    vllm_online = em.check_vllm_server()
    preflight_reasons = []
    if len(case_ids) != 10 or len(set(case_ids)) != 10 or any(not item for item in case_ids):
        preflight_reasons.append("case_list_must_contain_10_unique_nonempty_case_ids")
    if len(target_organs) != 373 or len(set(target_organs)) != 373:
        preflight_reasons.append("target_space_must_contain_373_unique_organs")
    if missing_teacher_cache:
        preflight_reasons.append(f"teacher_cache_incomplete:{missing_teacher_cache}")
    if selected_case_ids != set(case_ids):
        preflight_reasons.append("round1_selected_case_set_mismatch")
    if valid_hierarchical_cases != set(case_ids):
        preflight_reasons.append("round1_hierarchical_manifest_set_incomplete")
    if not previous_checkpoint_complete:
        preflight_reasons.append("round1_student_checkpoint_incomplete")
    if not mstep.get("eligible_for_next_round_prompt_student"):
        preflight_reasons.append("round1_student_checkpoint_not_quality_eligible")
    if student_gate.get("status") != "success":
        preflight_reasons.append("student_organ_gate_failed")
    expected_student_masks = int(student_gate.get("allowed_case_organ_count") or 0)
    if expected_student_masks <= 0:
        expected_student_masks = len(case_ids) * len(student_gate.get("allowed_organs", []))
    if int(student_gate.get("validated_masks") or 0) != expected_student_masks:
        preflight_reasons.append("student_mask_count_mismatch")
    if not round2_clean:
        preflight_reasons.append("round2_not_clean")
    if em.ENABLE_CRITIC and not vllm_online and not em.DEBUG_ALLOW_NO_LABELCRITIC:
        preflight_reasons.append("labelcritic_vllm_offline")
    preflight = {
        "stage": "round2_launch_preflight",
        "status": "ready" if not preflight_reasons else "failed",
        "reasons": preflight_reasons,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(run_root),
        "case_list": str(em.CASE_LIST),
        "teacher_inference_mode": em.TEACHER_INFERENCE_MODE,
        "round1_teacher_cache_count": len(cache),
        "round1_teacher_cache_keys": sorted(cache)[:50],
        "round1_teacher_cache_missing": missing_teacher_cache,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "target_organ_count": len(target_organs),
        "target_organs_unique": len(set(target_organs)) == len(target_organs),
        "round1_selected_root": str(round1_selected_root),
        "round1_selected_case_count": len(selected_case_ids),
        "round1_valid_hierarchical_case_count": len(valid_hierarchical_cases),
        "round1_student_checkpoint_eligible": bool(mstep.get("eligible_for_next_round_prompt_student")),
        "round1_student_checkpoint_complete": previous_checkpoint_complete,
        "round1_student_prediction_root": str(student_pred_root),
        "round1_student_predictions_exist": student_pred_root.exists() and any(p.is_dir() for p in student_pred_root.iterdir()),
        "round1_student_organ_gate": student_gate,
        "round2_clean_start": round2_clean,
        "quality_contract_version": em.QUALITY_CONTRACT_VERSION,
        "fov_policy_version": em.FOV_POLICY_VERSION,
        "labelcritic_vllm_online": vllm_online,
        "forbidden_aborted_round2_glob": str(run_root / "round2_aborted_*"),
        "models_to_run_expected": 0,
        "fresh_teacher_inference_allowed": False,
    }
    write_json(audit_path, preflight)
    if preflight["status"] != "ready":
        return 2
    if os.getenv("MEDAI_PREFLIGHT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        preflight["status"] = "success"
        preflight["preflight_only"] = True
        write_json(audit_path, preflight)
        return 0

    result = em.run_estep(2)
    dashboard = em.build_round_label_scoring_dashboard(2)
    manifest = em.build_student_dataset(2)
    gate = em.formal_estep_gate(2, result, manifest)
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    novelty = manifest_doc.get("novelty_audit") or {}
    mstep = {
        "status": "not_run",
        "reason": "audit_only_launcher_disables_round2_mstep",
        "required_next_step": (
            "Use the formal repaired Round2 M-step launcher only if novelty "
            "decision is material_update and max_steps > 0."
        ),
        "novelty_audit": novelty,
    }
    prediction_stage = {
        "status": "not_run",
        "reason": "audit_only_launcher_disables_student_inference",
    }
    final = {
        **preflight,
        "stage": "round2_estep_manifest_gate_novelty_audit_completed",
        "status": "success" if (
            result.get("status") == "success"
            and gate.get("status") == "success"
            and manifest.exists()
        ) else "failed",
        "estep_result": result,
        "label_scoring_dashboard": dashboard,
        "round2_manifest": str(manifest),
        "round2_formal_gate": gate,
        "round2_novelty_audit": novelty,
        "round2_mstep": mstep,
        "round2_student_predictions": prediction_stage,
    }
    write_json(audit_path, final)
    return 0 if final["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
