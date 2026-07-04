#!/usr/bin/env python3
"""Start Round2 E-step for the pure-cached 10-case formal-lite EM run."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/teacher1/JHU-project1/medical_agent")
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
    preflight = {
        "stage": "round2_launch_preflight",
        "status": "ready" if (
            cache
            and mstep.get("eligible_for_next_round_prompt_student")
            and student_gate.get("status") == "success"
            and round2_clean
        ) else "failed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(run_root),
        "case_list": str(em.CASE_LIST),
        "teacher_inference_mode": em.TEACHER_INFERENCE_MODE,
        "round1_teacher_cache_count": len(cache),
        "round1_teacher_cache_keys": sorted(cache)[:50],
        "round1_student_checkpoint_eligible": bool(mstep.get("eligible_for_next_round_prompt_student")),
        "round1_student_prediction_root": str(student_pred_root),
        "round1_student_predictions_exist": student_pred_root.exists() and any(p.is_dir() for p in student_pred_root.iterdir()),
        "round1_student_organ_gate": student_gate,
        "round2_clean_start": round2_clean,
        "forbidden_aborted_round2_glob": str(run_root / "round2_aborted_*"),
        "models_to_run_expected": 0,
        "fresh_teacher_inference_allowed": False,
    }
    write_json(audit_path, preflight)
    if preflight["status"] != "ready":
        return 2

    result = em.run_estep(2)
    dashboard = em.build_round_label_scoring_dashboard(2)
    manifest = em.build_student_dataset(2)
    gate = em.formal_estep_gate(2, result, manifest)
    mstep = {"status": "blocked", "reason": "round2_formal_gate_failed"}
    prediction_stage = {"status": "not_run"}
    if gate.get("status") == "success":
        mstep = em.run_student_mstep(2, manifest, global_consolidation=False)
        if mstep.get("status") == "success":
            em.save_student_predictions(2)
            prediction_stage = {
                "status": "success",
                "root": str(em.OUTPUT_ROOT / "round2" / "student_predictions"),
            }
        else:
            prediction_stage = {
                "status": "blocked",
                "reason": "round2_mstep_failed_or_not_quality_eligible",
            }
    final = {
        **preflight,
        "stage": "round2_estep_mstep_and_student_refresh_completed",
        "status": "success" if (
            result.get("status") == "success"
            and gate.get("status") == "success"
            and mstep.get("status") == "success"
            and prediction_stage.get("status") == "success"
        ) else "failed",
        "estep_result": result,
        "label_scoring_dashboard": dashboard,
        "round2_manifest": str(manifest),
        "round2_formal_gate": gate,
        "round2_mstep": mstep,
        "round2_student_predictions": prediction_stage,
    }
    write_json(audit_path, final)
    return 0 if final["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
