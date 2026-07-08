#!/usr/bin/env python3
"""Run the non-formal/bootstrap-only Round2 +10 experiment.

This script explores the separate "add new cases plus teacher cache" axis. It
is not the repaired formal EM Round2 path, which is implemented in
``run_em_training.py`` as ``em_student_vs_previous``.

This is the safe path for:
  1) reuse the promoted 10-case Round1 teacher cache/checkpoint,
  2) add teacher cache for the selected +10 cases,
  3) run Round2 E-step from the merged Round1 teacher cache,
  4) train only if material_update_audit finds new reliable teacher positives,
  5) run Round2 student inference/evaluation after a successful M-step.

Unlike ``run_em_training.py --start-round 2 --baseline-run-root``, this script
does not reuse an old Round2 E-step from the baseline root.  It creates a fresh
Round2 under --output-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))

import scripts.run_em_training as em
from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


DEFAULT_BASELINE = ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"
DEFAULT_CASE_PLAN = ROOT / "outputs" / "round2_plus10_case_plan_20260705"
DEFAULT_OUTPUT = ROOT / "outputs" / "em_round2_plus10_20260705"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def link_or_copy_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or "").strip()
        ]


def write_case_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty case list")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def baseline_case_ids(baseline: Path) -> set[str]:
    ids = {
        path.name
        for path in (baseline / "round1" / "estep" / "cases").iterdir()
        if path.is_dir()
    }
    if ids:
        return ids
    manifest = read_json(baseline / "round1" / "mstep" / "voxtell_prompt_student_manifest.json", {})
    return {
        str(row.get("case_id") or "")
        for row in manifest.get("items", [])
        if isinstance(row, dict) and row.get("case_id")
    }


def configure_em(output_root: Path, case_list: Path) -> None:
    em.OUTPUT_ROOT = output_root.resolve()
    em.CASE_LIST = case_list.resolve()
    em.LOG_FILE = em.OUTPUT_ROOT / "training.log"
    em.START_ROUND = 2
    em.NUM_ROUNDS = 2


def audit_and_link_reusable_cache(*, output_root: Path, new_case_list: Path, copy: bool = False) -> dict[str, Any]:
    output = output_root / "teacher_assets" / "round2_plus10" / "reusable_teacher_cache_audit.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "audit_and_link_reusable_teacher_cache.py"),
        "--case-list",
        str(new_case_list),
        "--target-cases-root",
        str(output_root / "round1" / "estep" / "cases"),
        "--output",
        str(output),
    ]
    if copy:
        cmd.append("--copy")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    audit = read_json(output, {})
    audit["return_code"] = proc.returncode
    if proc.returncode != 0:
        audit.setdefault("status", "failed")
        audit["reason"] = "reusable_teacher_cache_audit_failed"
        write_json(output, audit)
    return audit


def prepare_merged_round1_cache(*, baseline: Path, output_root: Path, case_list: Path, reuse_prior_cache: bool = True) -> dict[str, Any]:
    """Expose baseline Round1 artifacts and leave room for new-case caches."""
    baseline = baseline.resolve()
    output_root = output_root.resolve()
    round1 = output_root / "round1"
    round1_estep = round1 / "estep"
    round1_cases = round1_estep / "cases"
    round1_ann = round1_estep / "annotation_versions"
    round1_cases.mkdir(parents=True, exist_ok=True)
    round1_ann.mkdir(parents=True, exist_ok=True)

    # Checkpoint/manifest are immutable Round1 baseline assets.
    link_or_copy_tree(baseline / "round1" / "mstep", round1 / "mstep")
    if (baseline / "checkpoint_promotion_registry.json").exists():
        link_or_copy_file(
            baseline / "checkpoint_promotion_registry.json",
            output_root / "checkpoint_promotion_registry.json",
        )
    if (baseline / "evaluation_protocol.json").exists():
        link_or_copy_file(baseline / "evaluation_protocol.json", output_root / "evaluation_protocol.json")

    old_case_ids = baseline_case_ids(baseline)
    for case_id in sorted(old_case_ids):
        source_case = baseline / "round1" / "estep" / "cases" / case_id
        source_ann = baseline / "round1" / "estep" / "annotation_versions" / case_id
        if source_case.exists():
            link_or_copy_tree(source_case, round1_cases / case_id)
        if source_ann.exists():
            link_or_copy_tree(source_ann, round1_ann / case_id)

    all_rows = read_case_rows(case_list)
    new_rows = [row for row in all_rows if row["case_id"] not in old_case_ids]
    new_case_list = output_root / "teacher_assets" / "round2_plus10" / "new10_case_list.csv"
    write_case_rows(new_case_list, new_rows)
    reuse_audit = (
        audit_and_link_reusable_cache(output_root=output_root, new_case_list=new_case_list)
        if reuse_prior_cache and new_rows
        else {"status": "skipped", "reason": "reuse_prior_cache_disabled_or_no_new_cases"}
    )
    report = {
        "stage": "prepare_merged_round1_cache",
        "status": "success",
        "baseline": str(baseline),
        "output_root": str(output_root),
        "case_list": str(case_list),
        "baseline_case_count": len(old_case_ids),
        "new_case_count": len(new_rows),
        "new_case_ids": [row["case_id"] for row in new_rows],
        "new_case_list": str(new_case_list),
        "reusable_teacher_cache_audit": reuse_audit,
        "policy": "baseline Round1 case artifacts are linked/copied per case; new cases are materialized in the same Round1 E-step cache root",
    }
    write_json(output_root / "teacher_assets" / "round2_plus10" / "prepare_merged_round1_cache.json", report)
    return report


def build_new_teacher_cache(*, output_root: Path, new_case_list: Path, max_cases: int = 0) -> dict[str, Any]:
    configure_em(output_root, new_case_list)
    rows = read_case_rows(new_case_list)
    if max_cases:
        rows = rows[:max_cases]
        limited = output_root / "teacher_assets" / "round2_plus10" / f"new{len(rows)}_case_list_limited.csv"
        write_case_rows(limited, rows)
        new_case_list = limited
        configure_em(output_root, new_case_list)
    if not rows:
        result = {"stage": "build_new_teacher_cache", "status": "success", "reason": "no_new_cases"}
        write_json(output_root / "teacher_assets" / "round2_plus10" / "cache_manifest.json", result)
        return result

    organs = em.load_student_target_organs()
    result = run_multimodel_annotation_loop(
        case_list=new_case_list,
        output_folder=output_root / "round1" / "estep",
        models=list(em.ALL_TEACHERS),
        organs=organs,
        registry_path=ROOT / "configs" / "model_registry.yaml",
        checkpoint_map_models=False,
        enable_shapekit=em.ENABLE_SHAPEKIT,
        shapekit_root=ROOT / "third_party" / "ShapeKit-main",
        enable_critic=em.ENABLE_CRITIC,
        critic_backend=em.CRITIC_BACKEND,
        critic_base_url=em.VLLM_BASE_URL,
        critic_port=8000,
        dry_run=False,
        timeout_sec=em.INFER_TIMEOUT_SEC,
        device="cuda",
        perf_tracker_path=output_root / "organ_model_performance.json",
        resume=True,
        labelcritic_options=em.LABELCRITIC_OPTIONS,
        candidate_mode=em.CANDIDATE_MODE,
        teacher_inference_mode=em.TEACHER_INFERENCE_MODE,
        roi_margin_mm=em.ROI_MARGIN_MM,
    )
    cache = em._round_teacher_cache_dirs(1)
    manifest = {
        "stage": "build_new_teacher_cache",
        "status": "success" if result.get("status") == "success" else "failed",
        "result": result,
        "case_list": str(new_case_list),
        "new_case_ids": [row["case_id"] for row in rows],
        "teacher_cache_count": len(cache),
        "teacher_cache_keys": sorted(cache),
        "missing_required_teachers": sorted(set(em.ALL_TEACHERS) - set(cache)),
        "teacher_inference_rerun": True,
    }
    write_json(output_root / "teacher_assets" / "round2_plus10" / "cache_manifest.json", manifest)
    if manifest["status"] != "success":
        raise RuntimeError(f"New teacher cache failed: {result}")
    return manifest


def run_round2(*, output_root: Path, case_list: Path) -> dict[str, Any]:
    configure_em(output_root, case_list)
    os.environ.setdefault(
        "MEDAI_ROUND_REFERENCE_ROOT",
        str((output_root / "round1" / "estep" / "annotation_versions").resolve()),
    )
    em.ensure_current_student_backend_allowed()
    em.ensure_formal_teacher_pool_registered()
    em.ensure_formal_quality_gates()
    em.ensure_evaluation_protocol()

    round_idx = 2
    round_start = time.time()
    summary_path = output_root / "round2" / "round_summary.json"

    estep_result = em.run_estep(round_idx)
    if estep_result.get("status") != "success":
        summary = {"round": round_idx, "estep_status": estep_result.get("status"), "mstep_status": "skipped_estep_failed", "estep_result": estep_result}
        write_json(summary_path, summary)
        return summary

    label_scoring_dashboard = em.build_round_label_scoring_dashboard(round_idx)
    dataset_path = em.build_student_dataset(round_idx)
    manifest_doc = read_json(dataset_path, {})
    material_update = manifest_doc.get("material_update_audit") or read_json(output_root / "round2" / "estep" / "material_update_audit.json", {})
    estep_gate = em.formal_estep_gate(round_idx, estep_result, dataset_path)
    if estep_gate.get("status") != "success":
        summary = {
            "round": round_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estep_status": estep_result.get("status"),
            "estep_formal_gate": estep_gate,
            "mstep_status": "blocked_by_estep_gate",
            "material_update_audit": material_update,
            "label_scoring_dashboard": label_scoring_dashboard,
            "round_elapsed_hours": round((time.time() - round_start) / 3600, 2),
        }
        write_json(summary_path, summary)
        return summary

    if material_update.get("material_update_decision") == "no_material_update":
        mstep_result = em.run_student_mstep(round_idx, dataset_path, global_consolidation=True)
        summary = {
            "round": round_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estep_status": estep_result.get("status"),
            "estep_formal_gate": estep_gate,
            "mstep_status": mstep_result.get("status"),
            "mstep_training_status": "no_material_update",
            "student_backend": em.STUDENT_BACKEND,
            "mstep_type": "checkpoint_reuse",
            "metrics": {
                "status": "skipped",
                "reason": "no_material_update_reused_previous_promoted_checkpoint",
                "novelty_audit": mstep_result.get("novelty_audit"),
            },
            "material_update_audit": material_update,
            "label_scoring_dashboard": label_scoring_dashboard,
            "round_elapsed_hours": round((time.time() - round_start) / 3600, 2),
            "success": True,
        }
        try:
            summary["checkpoint_promotion"] = em.record_checkpoint_promotion(
                round_idx,
                status="promoted",
                reason="no_material_update_reused_previous_promoted_checkpoint",
                mstep_result=mstep_result,
                manifest_path=dataset_path,
                summary_path=summary_path,
            )
        except Exception as exc:
            summary["checkpoint_promotion_error"] = str(exc)
        write_json(summary_path, summary)
        return summary

    mstep_result = em.run_student_mstep(round_idx, dataset_path, global_consolidation=True)
    if mstep_result.get("status") != "success":
        summary = {
            "round": round_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estep_status": estep_result.get("status"),
            "estep_formal_gate": estep_gate,
            "mstep_status": "failed",
            "mstep_result": mstep_result,
            "material_update_audit": material_update,
            "label_scoring_dashboard": label_scoring_dashboard,
            "round_elapsed_hours": round((time.time() - round_start) / 3600, 2),
        }
        write_json(summary_path, summary)
        return summary

    em.save_student_predictions(round_idx)
    postprocess_summary = em.apply_round_organ_type_postprocess(round_idx)
    round_metrics = em.compute_round_metrics(round_idx)
    evaluation_chain = em.compute_round_evaluation_chain(round_idx)
    round_metrics["evaluation_chain"] = evaluation_chain

    summary = {
        "round": round_idx,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "estep_status": estep_result.get("status"),
        "estep_formal_gate": estep_gate,
        "mstep_status": mstep_result.get("status"),
        "student_backend": em.STUDENT_BACKEND,
        "mstep_type": "global_consolidation",
        "finetuned_checkpoint": mstep_result.get("finetuned_checkpoint"),
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
        "material_update_audit": material_update,
        "label_scoring_dashboard": label_scoring_dashboard,
        "round_elapsed_hours": round((time.time() - round_start) / 3600, 2),
    }
    summary["reliability_weights"] = em._student_manifest_weight_summary(round_idx)
    summary["success"] = bool(
        summary.get("estep_status") == "success"
        and (summary.get("estep_formal_gate") or {}).get("status") == "success"
        and summary.get("mstep_status") == "success"
        and bool(mstep_result.get("eligible_for_next_round_prompt_student", mstep_result.get("checkpoint_eligible_for_next_round", False)))
        and (mstep_result.get("retention_audit") or {}).get("status") == "passed"
    )
    try:
        summary["checkpoint_promotion"] = em.record_checkpoint_promotion(
            round_idx,
            status="promoted" if summary["success"] else "competition_blocked",
            reason=None if summary["success"] else "round_verification_failed",
            mstep_result=mstep_result,
            manifest_path=dataset_path,
            summary_path=summary_path,
        )
    except Exception as exc:
        summary["checkpoint_promotion_error"] = str(exc)
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--case-plan-dir", type=Path, default=DEFAULT_CASE_PLAN)
    parser.add_argument("--case-list", type=Path, default=None, help="Override case-plan case_list_round2_plus10.csv; useful for +7 fast reporting runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-case-plan", action="store_true")
    parser.add_argument("--skip-teacher-cache", action="store_true")
    parser.add_argument("--no-reuse-prior-cache", action="store_true")
    parser.add_argument("--max-new-cases", type=int, default=0, help="Debug only; 0 means all selected new cases.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    case_plan_dir = args.case_plan_dir.resolve()
    baseline = args.baseline_run_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "round2_plus10_formal_state.json"
    state: dict[str, Any] = {
        "stage": "round2_plus10_formal",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_run_root": str(baseline),
        "case_plan_dir": str(case_plan_dir),
        "output_root": str(output_root),
        "steps": [],
    }
    write_json(state_path, state)

    if not args.skip_case_plan:
        proc = subprocess.run(
            [sys.executable, "scripts/build_round2_plus10_case_plan.py", "--output-dir", str(case_plan_dir)],
            cwd=str(ROOT),
            check=False,
        )
        state["steps"].append({"name": "case_plan", "return_code": proc.returncode})
        write_json(state_path, state)
        if proc.returncode != 0:
            state["status"] = "failed_case_plan"
            write_json(state_path, state)
            return 2

    case_list = args.case_list.resolve() if args.case_list else case_plan_dir / "case_list_round2_plus10.csv"
    prepare = prepare_merged_round1_cache(
        baseline=baseline,
        output_root=output_root,
        case_list=case_list,
        reuse_prior_cache=not args.no_reuse_prior_cache,
    )
    state["steps"].append({"name": "prepare_merged_round1_cache", "result": prepare})
    write_json(state_path, state)

    if not args.skip_teacher_cache:
        cache_manifest = build_new_teacher_cache(
            output_root=output_root,
            new_case_list=Path(prepare["new_case_list"]),
            max_cases=args.max_new_cases,
        )
        state["steps"].append({"name": "build_new_teacher_cache", "result": cache_manifest})
        write_json(state_path, state)

    summary = run_round2(output_root=output_root, case_list=case_list)
    state["steps"].append({"name": "run_round2", "result": summary})
    state["status"] = "success" if summary.get("success") or summary.get("mstep_training_status") == "no_material_update" else "failed_or_blocked"
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0 if state["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
