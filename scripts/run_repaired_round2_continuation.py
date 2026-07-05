#!/usr/bin/env python3
"""Deprecated repaired-Round2 continuation launcher.

This file is kept importable for historical audit helpers/tests, but direct
execution is disabled.  The current audited Round2 repair is a no-material-update
decision that reuses the promoted Round1 checkpoint; this script's inference and
Round3 launch paths are therefore unsafe for the current chain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_em_training as em


ROUND_IDX = 2
MIN_FREE_GIB = 15.0
DISABLED_REASON = (
    "scripts/run_repaired_round2_continuation.py is disabled for the current "
    "formal-lite Round2 repair. The audited replay/novelty decision is "
    "no_material_update, so the correct action is to reuse the promoted Round1 "
    "checkpoint and not launch continuation inference or Round3 from this helper."
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


def update_state(status: str, **extra) -> None:
    write_json(
        em.OUTPUT_ROOT / "round2" / "repaired_round2_continuation_state.json",
        {
            "stage": "repaired_round2_continuation",
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit_sha": em._git_commit_sha(),
            **extra,
        },
    )


def read_case_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [str(row.get("case_id") or "").strip() for row in csv.DictReader(handle) if row.get("case_id")]


def validate_case_scope(case_list: Path, manifest_path: Path) -> dict:
    expected = read_case_ids(case_list)
    manifest = read_json(manifest_path)
    manifest_cases = sorted({
        str(row.get("case_id") or "").strip()
        for row in manifest.get("items") or []
        if isinstance(row, dict) and row.get("case_id")
    })
    estep_root = em.OUTPUT_ROOT / "round2" / "estep" / "annotation_versions"
    estep_cases = sorted(path.name for path in estep_root.iterdir() if path.is_dir()) if estep_root.exists() else []
    expected_set = set(expected)
    failures = []
    if not expected or len(expected) != len(expected_set):
        failures.append("case_list_must_be_nonempty_and_unique")
    if set(manifest_cases) != expected_set:
        failures.append("manifest_case_scope_mismatch")
    if set(estep_cases) != expected_set:
        failures.append("estep_case_scope_mismatch")
    return {
        "stage": "round2_case_scope_preflight",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "case_list": str(case_list),
        "case_list_sha256": em._sha256_file(case_list),
        "expected_case_ids": expected,
        "manifest_case_ids": manifest_cases,
        "estep_case_ids": estep_cases,
    }


def disk_preflight(path: Path, minimum_gib: float = MIN_FREE_GIB) -> dict:
    free = shutil.disk_usage(path).free
    minimum = int(minimum_gib * 1024**3)
    return {
        "stage": "disk_preflight",
        "status": "passed" if free >= minimum else "failed",
        "free_bytes": free,
        "free_gib": round(free / 1024**3, 2),
        "minimum_free_bytes": minimum,
        "minimum_free_gib": minimum_gib,
    }


def validate_existing_mstep(manifest_path: Path) -> tuple[dict, dict]:
    mstep = read_json(em.OUTPUT_ROOT / "round2" / "mstep" / "voxtell_prompt_mstep_result.json")
    checkpoint = Path(str(mstep.get("inference_checkpoint") or ""))
    retention = mstep.get("retention_audit") or {}
    cumulative = read_json(manifest_path).get("cumulative_manifest_summary") or {}
    failures = []
    if mstep.get("status") != "success" or mstep.get("training_status") != "completed":
        failures.append("mstep_not_completed_successfully")
    if not checkpoint.is_file():
        failures.append("inference_checkpoint_missing")
    if not mstep.get("eligible_for_next_round_prompt_student", mstep.get("checkpoint_eligible_for_next_round", False)):
        failures.append("checkpoint_not_next_round_eligible")
    if retention.get("status") != "passed":
        failures.append("retention_audit_not_passed")
    if float(retention.get("official_retention_weight") or 0.0) != 0.2:
        failures.append("retention_weight_not_0_2")
    try:
        retention_loss = float(retention.get("mean_retention_loss"))
    except Exception:
        retention_loss = float("nan")
    if not math.isfinite(retention_loss) or retention_loss <= 0:
        failures.append("retention_loss_not_nonzero_finite")
    if retention.get("teacher_frozen_eval") is not True:
        failures.append("retention_teacher_not_frozen_eval")
    if retention.get("source_checkpoint_hash_ok") is not True:
        failures.append("retention_source_hash_mismatch")
    if cumulative.get("missing_historical_trainable_positive_organs"):
        failures.append("historical_replay_deficits_present")
    audit = {
        "stage": "existing_repaired_round2_mstep_preflight",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": em._sha256_file(checkpoint) if checkpoint.is_file() else None,
        "retention_audit": retention,
        "historical_replay_deficits": cumulative.get("missing_historical_trainable_positive_organs") or [],
    }
    return mstep, audit


def write_three_way_comparison(round_metrics: dict, evaluation_chain: dict) -> dict:
    old_path = em.OUTPUT_ROOT / "round2" / "ROUND1_ROUND2_STUDENT_COMPARISON_20260704.json"
    old = read_json(old_path)
    old_metrics = old.get("metrics") or {}
    round1 = ((old_metrics.get("round1_raw") or {}).get("summary") or {})
    bad_round2 = ((old_metrics.get("round2_raw") or {}).get("summary") or {})
    repaired = evaluation_chain
    comparison = {
        "stage": "round1_bad_round2_repaired_round2_comparison",
        "status": "success" if repaired.get("status") == "success" else "partial",
        "metric_warning": "GT metrics are real only where PanTS GT is available; pseudo metrics measure consistency.",
        "round1_baseline": round1,
        "original_bad_round2": bad_round2,
        "repaired_round2": repaired,
        "repaired_round2_pseudo_consistency": {
            "overall_mean_dsc": round_metrics.get("overall_mean_dsc"),
            "metric_scope": round_metrics.get("metric_scope"),
        },
    }
    output = em.OUTPUT_ROOT / "round2" / "ROUND1_BAD_ROUND2_REPAIRED_ROUND2_COMPARISON.json"
    write_json(output, comparison)
    comparison["path"] = str(output)
    return comparison


def screen_exists(prefix: str) -> bool:
    proc = subprocess.run(
        ["screen", "-ls"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return any(prefix in line for line in (proc.stdout or "").splitlines())


def launch_round3_screen(case_list: Path) -> dict:
    fail_disabled_launcher()
    prefix = "round3_repaired_10case_"
    state_path = em.OUTPUT_ROOT / "round3" / "round3_continuation_state.json"
    prior_state = read_json(state_path)
    if screen_exists(prefix) or prior_state.get("status") in {
        "running_estep", "running_mstep", "running_student_inference", "screen_started",
    }:
        return {"status": "already_running", "state": prior_state}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    screen_name = f"{prefix}{stamp}"
    log_path = em.OUTPUT_ROOT / "round3" / "round3_screen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {ROOT} && "
        f"MEDAI_OUTPUT_ROOT={em.OUTPUT_ROOT} "
        f"MEDAI_CASE_LIST={case_list} "
        "MEDAI_MAX_STEPS=2000 MEDAI_FORMAL_MIN_STEPS=2000 "
        "MEDAI_VOXTELL_TRAINING_PROFILE=quality_weighted_ablation "
        "MEDAI_OFFICIAL_RETENTION_WEIGHT=0.2 MEDAI_MSTEP_BATCH_SIZE=1 "
        "MEDAI_SAVE_EVERY=0 PYTHONUNBUFFERED=1 "
        f"{sys.executable} scripts/run_round3_after_promoted_round2.py "
        f"--case-list {case_list} >> {log_path} 2>&1"
    )
    proc = subprocess.run(
        ["screen", "-dmS", screen_name, "bash", "-lc", command],
        cwd=str(ROOT),
        check=False,
    )
    launch = {
        "status": "screen_started" if proc.returncode == 0 else "screen_launch_failed",
        "screen_name": screen_name,
        "log_path": str(log_path),
        "case_list": str(case_list),
        "case_list_sha256": em._sha256_file(case_list),
        "return_code": proc.returncode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(em.OUTPUT_ROOT / "round3" / "round3_launch_audit.json", launch)
    if proc.returncode == 0:
        write_json(state_path, {
            "stage": "round3_from_promoted_round2",
            "status": "screen_started",
            **launch,
        })
    return launch


def main() -> int:
    fail_disabled_launcher()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--resume-existing-mstep", action="store_true")
    parser.add_argument("--verification-only", action="store_true")
    parser.add_argument("--launch-round3", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=MIN_FREE_GIB)
    args = parser.parse_args()

    case_list = args.case_list.expanduser().resolve()
    if not case_list.is_file():
        raise SystemExit(f"Missing required case list: {case_list}")
    em.CASE_LIST = case_list
    em.ensure_evaluation_protocol()
    em.write_round_run_spec(ROUND_IDX)
    manifest_path = em.OUTPUT_ROOT / "round2" / "mstep" / "voxtell_prompt_student_manifest.json"
    gate_path = em.OUTPUT_ROOT / "round2" / "estep" / "formal_gate.json"
    summary_path = em.OUTPUT_ROOT / "round2" / "round_summary.json"
    gate = read_json(gate_path)
    scope = validate_case_scope(case_list, manifest_path)
    disk = disk_preflight(em.OUTPUT_ROOT, args.minimum_free_gib)
    snapshot = em.OUTPUT_ROOT / "round2" / "case_list_snapshot.csv"
    shutil.copy2(case_list, snapshot)
    write_json(em.OUTPUT_ROOT / "round2" / "case_scope_preflight.json", scope)
    write_json(em.OUTPUT_ROOT / "round2" / "disk_preflight.json", disk)
    if gate.get("status") != "success" or scope["status"] != "passed" or disk["status"] != "passed":
        update_state("blocked_by_preflight", gate_status=gate.get("status"), case_scope=scope, disk=disk)
        return 2

    if not args.resume_existing_mstep:
        update_state("blocked_requires_explicit_mstep_mode")
        return 2
    mstep_result, mstep_audit = validate_existing_mstep(manifest_path)
    write_json(em.OUTPUT_ROOT / "round2" / "mstep" / "existing_mstep_resume_audit.json", mstep_audit)
    if mstep_audit["status"] != "passed":
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="competition_blocked",
            reason="existing_repaired_round2_mstep_preflight_failed",
            mstep_result=mstep_result,
            manifest_path=manifest_path,
            summary_path=summary_path,
        )
        update_state("blocked_by_mstep_preflight", mstep_audit=mstep_audit)
        return 2

    if args.verification_only:
        inference_audit = em.audit_prompt_student_predictions(ROUND_IDX)
        postprocess_summary = read_json(
            em.OUTPUT_ROOT
            / "round2"
            / "student_predictions_postprocessed"
            / "organ_type_postprocess_summary.json"
        )
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="candidate",
            reason="verification_retry_reference_fixed",
            mstep_result=mstep_result,
            manifest_path=manifest_path,
            summary_path=summary_path,
            inference_audit=inference_audit,
        )
    else:
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="candidate",
            reason="clean_10case_inference_pending",
            mstep_result=mstep_result,
            manifest_path=manifest_path,
            summary_path=summary_path,
        )
        update_state("running_student_inference", mstep_audit=mstep_audit, case_scope=scope)
        inference_summary = em.save_student_predictions(ROUND_IDX)
        inference_audit = inference_summary.get("inference_audit") or {}
        postprocess_summary = {}
    if inference_audit.get("status") != "passed":
        em.record_checkpoint_promotion(
            ROUND_IDX,
            status="competition_blocked",
            reason="clean_10case_inference_audit_failed",
            mstep_result=mstep_result,
            manifest_path=manifest_path,
            summary_path=summary_path,
            inference_audit=inference_audit,
        )
        update_state("inference_audit_failed", inference_audit=inference_audit)
        return 1

    update_state("running_verification", inference_audit=inference_audit)
    if not args.verification_only:
        postprocess_summary = em.apply_round_organ_type_postprocess(ROUND_IDX)
    if postprocess_summary.get("status") != "success":
        update_state("verification_failed_missing_postprocess", postprocess_summary=postprocess_summary)
        return 1
    round_metrics = em.compute_round_metrics(ROUND_IDX)
    evaluation_chain = em.compute_round_evaluation_chain(ROUND_IDX)
    round_metrics["evaluation_chain"] = evaluation_chain
    comparison = write_three_way_comparison(round_metrics, evaluation_chain)
    success = bool(
        gate.get("status") == "success"
        and mstep_result.get("status") == "success"
        and (mstep_result.get("retention_audit") or {}).get("status") == "passed"
        and inference_audit.get("status") == "passed"
        and evaluation_chain.get("status") == "success"
        and postprocess_summary.get("status") == "success"
    )
    summary = {
        "round": ROUND_IDX,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "estep_status": "success",
        "estep_formal_gate": gate,
        "mstep_status": mstep_result.get("status"),
        "student_backend": em.STUDENT_BACKEND,
        "mstep_type": "global_consolidation",
        "finetuned_checkpoint": mstep_result.get("finetuned_checkpoint"),
        "inference_audit": inference_audit,
        "case_scope_preflight": scope,
        "existing_mstep_resume_audit": mstep_audit,
        "metrics": {
            "overall_mean_dsc": round_metrics.get("overall_mean_dsc"),
            "overall_mean_pseudo_consistency_dsc": round_metrics.get("overall_mean_dsc"),
            "metric_family": round_metrics.get("metric_family", "pseudo_consistency"),
            "metric_scope": round_metrics.get("metric_scope", "student_vs_selected_pseudo_label"),
            "evaluation_chain": evaluation_chain,
            "organ_type_postprocess": postprocess_summary,
        },
        "three_way_comparison": comparison,
        "reliability_weights": em._student_manifest_weight_summary(ROUND_IDX),
        "success": success,
        "code_commit_sha": em._git_commit_sha(),
    }
    promotion = em.record_checkpoint_promotion(
        ROUND_IDX,
        status="promoted" if success else "competition_blocked",
        reason=None if success else "repaired_round2_verification_failed",
        mstep_result=mstep_result,
        manifest_path=manifest_path,
        summary_path=summary_path,
        inference_audit=inference_audit,
    )
    summary["checkpoint_promotion"] = promotion
    write_json(summary_path, summary)
    if not success or promotion.get("status") != "promoted":
        update_state("verification_failed", summary=summary)
        return 1

    update_state("complete", summary=summary)
    launch = launch_round3_screen(case_list) if args.launch_round3 else {"status": "not_requested"}
    print(json.dumps({
        "status": "success",
        "summary": str(summary_path),
        "promotion": promotion,
        "round3_launch": launch,
    }, indent=2, ensure_ascii=False))
    return 0 if launch.get("status") in {"screen_started", "already_running", "not_requested"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
