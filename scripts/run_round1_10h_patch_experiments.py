#!/usr/bin/env python3
"""10-hour Round1 repair experiments for the automatic VoxTell-agent report.

This script turns an existing Round1 E-step into a compact, auditable result
package for the teacher-facing story: automated pseudo-label selection,
VoxTell-aligned prompt student distillation readiness, automatic quality gates,
prompt robustness, and negative prompt suppression. It never assumes expert
labels; all metrics are pseudo-label consistency or automatic contract checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

DEFAULT_ROUND_ROOT = ROOT / "outputs/stage4b_round1_50cases_20260611/round1"
DEFAULT_CASE_LIST = ROOT / "data_manifest/case_list_50_tumor.csv"
DEFAULT_TARGET_CONFIG = ROOT / "configs/student_3d_prompt_target_organs.json"
DEFAULT_VOXTELL_MODEL = ROOT / "checkpoints/VoxTell/voxtell_v1.1"
DEFAULT_TEXT_MODEL = ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"
PREFERRED_ORGANS = ["liver", "spleen", "pancreas", "kidney_left", "aorta"]
SAFE_NEGATIVE_SOURCES = {
    "nonmedical_absent_object",
    "out_of_scan_anatomy_with_coverage_evidence",
    "explicit_confirmed_absent_anatomy",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the 10-hour Round1 automatic VoxTell-agent experiment package.")
    ap.add_argument("--round-root", default=str(DEFAULT_ROUND_ROOT), help="Round root containing estep/ and mstep/.")
    ap.add_argument("--case-list", default=str(DEFAULT_CASE_LIST))
    ap.add_argument("--target-config", default=str(DEFAULT_TARGET_CONFIG))
    ap.add_argument("--model-dir", default=os.getenv("MEDAI_VOXTELL_MODEL_DIR", str(DEFAULT_VOXTELL_MODEL)))
    ap.add_argument("--text-encoding-model", default=os.getenv("MEDAI_TEXT_ENCODING_MODEL", str(DEFAULT_TEXT_MODEL)))
    ap.add_argument("--output-dir", default="", help="Defaults to <round-root>/mstep/round1_10h_patch_experiments.")
    ap.add_argument("--rebuild-manifest", action="store_true", help="Regenerate voxtell_prompt_student_manifest.json with current code.")
    ap.add_argument("--run-prompt-robustness", action="store_true", help="Run checkpoint inference for canonical vs variants.")
    ap.add_argument("--run-negative-suppression", action="store_true", help="Run checkpoint inference for sampled negative prompts.")
    ap.add_argument("--run-existing-gates", action="store_true", help="Run run_em_training sanity and quality gates if checkpoint exists.")
    ap.add_argument("--max-robustness-cases", type=int, default=1)
    ap.add_argument("--max-negative-cases", type=int, default=1)
    ap.add_argument("--max-negatives-per-source", type=int, default=3)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--device", default=os.getenv("MEDAI_DEVICE", "cuda"))
    return ap.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def rows_from_json(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    if isinstance(doc, dict):
        for key in ("items", "case_organ_scores", "records", "evaluations"):
            rows = doc.get(key)
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
    return []


def counter_dict(rows: list[dict[str, Any]], key: str, limit: int | None = None) -> dict[str, int]:
    c = Counter(str(row.get(key)) for row in rows if row.get(key) not in (None, ""))
    items = c.most_common(limit) if limit else sorted(c.items())
    return {str(k): int(v) for k, v in items}


def list_counter_dict(rows: list[dict[str, Any]], key: str, limit: int | None = None) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            c.update(str(x) for x in value if x not in (None, ""))
        elif value not in (None, ""):
            c[str(value)] += 1
    items = c.most_common(limit) if limit else sorted(c.items())
    return {str(k): int(v) for k, v in items}


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def pct(n: int, d: int) -> float:
    return round(float(n) / float(d), 6) if d else 0.0


def load_case_list(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_auto_quality(round_root: Path, case_list: Path, output_dir: Path) -> dict[str, Any]:
    estep = round_root / "estep"
    dashboards = estep / "dashboards"
    dataset_summary = read_json(dashboards / "auto_fine_label_dataset_summary.json", {}) or {}
    case_scores_doc = read_json(dashboards / "case_organ_label_scores.json", {}) or {}
    organ_dashboard_doc = read_json(dashboards / "organ_capability_dashboard.json", {}) or {}
    run_summary = read_json(estep / "run_summary.json", {}) or {}
    case_rows = rows_from_json(case_scores_doc)
    organ_rows = rows_from_json(organ_dashboard_doc)
    cases = load_case_list(case_list)

    reliability = [x for x in (safe_float(r.get("auto_fine_label_reliability_score")) for r in case_rows) if x is not None]
    train_weights = [x for x in (safe_float(r.get("training_weight")) for r in case_rows) if x is not None]
    selected = [r for r in case_rows if str(r.get("selection_status") or "") == "selected" or r.get("mask_path")]
    eligible = [r for r in case_rows if r.get("distillation_eligible") is True]
    excluded = [r for r in case_rows if r.get("distillation_eligible") is False]
    labelcritic_used = sum(1 for r in case_rows if r.get("labelcritic_used") is True)
    shapekit_fallback = sum(1 for r in case_rows if "shapekit_fallback" in set(r.get("quality_flags") or []) or r.get("shapekit_status") not in (None, "", "pass", "fusion_consensus"))

    summary = {
        "stage": "round1_auto_pseudo_label_quality_summary",
        "status": "success" if case_rows else "missing_case_organ_scores",
        "round_root": str(round_root),
        "estep_root": str(estep),
        "case_list": str(case_list),
        "num_cases_case_list": len(cases),
        "num_cases_summary": dataset_summary.get("num_cases") or run_summary.get("num_cases"),
        "target_organs": dataset_summary.get("target_organs"),
        "case_organ_score_rows": len(case_rows),
        "expected_case_organ_rows": dataset_summary.get("case_organ_score_expected_rows"),
        "formal_complete_50_cases": bool(dataset_summary.get("formal_complete_50_cases")),
        "num_selected_labels": dataset_summary.get("num_selected_labels") or len(selected),
        "num_distillation_eligible_rows": len(eligible),
        "num_distillation_excluded_rows": len(excluded),
        "grade_counts": dataset_summary.get("grade_counts") or counter_dict(case_rows, "grade"),
        "selection_status_counts": counter_dict(case_rows, "selection_status"),
        "selection_method_counts": counter_dict(case_rows, "selection_method", 20),
        "selected_model_counts": counter_dict(case_rows, "selected_model", 25),
        "source_model_counts": counter_dict(case_rows, "source_model", 25),
        "teacher_lineage_counts": list_counter_dict(case_rows, "teacher_lineage", 30),
        "candidate_model_counts": list_counter_dict(case_rows, "candidate_models", 30),
        "quality_status_counts": counter_dict(case_rows, "quality_status"),
        "quality_flag_counts": list_counter_dict(case_rows, "quality_flags", 30),
        "review_flag_counts": list_counter_dict(case_rows, "review_flags", 30),
        "labelcritic": {
            "used_count": labelcritic_used,
            "used_rate": pct(labelcritic_used, len(case_rows)),
            "uncertain_count": sum(1 for r in case_rows if r.get("labelcritic_uncertain") is True),
            "compare_used_count": sum(1 for r in case_rows if r.get("labelcritic_compare_used") is True),
            "grade_used_count": sum(1 for r in case_rows if r.get("labelcritic_grade_used") is True),
            "auto_grade_accept_count": sum(1 for r in case_rows if r.get("auto_grade_accept") is True),
        },
        "shapekit": {
            "status_counts": counter_dict(case_rows, "shapekit_status"),
            "fallback_or_nonpass_count": shapekit_fallback,
            "fallback_or_nonpass_rate": pct(shapekit_fallback, len(case_rows)),
        },
        "route_confidence_counts": counter_dict(case_rows, "route_confidence"),
        "reliability_score": {
            "mean": round(statistics.mean(reliability), 6) if reliability else None,
            "median": round(statistics.median(reliability), 6) if reliability else None,
            "min": round(min(reliability), 6) if reliability else None,
            "max": round(max(reliability), 6) if reliability else None,
        },
        "training_weight": {
            "mean": round(statistics.mean(train_weights), 6) if train_weights else None,
            "median": round(statistics.median(train_weights), 6) if train_weights else None,
            "num_weight_ge_0_5": sum(1 for x in train_weights if x >= 0.5),
            "num_weight_zero": sum(1 for x in train_weights if x == 0.0),
        },
        "organ_dashboard": {
            "num_organs": len(organ_rows),
            "status_counts": counter_dict(organ_rows, "status"),
            "weakest_organs_by_mean_reliability": sorted(
                [r for r in organ_rows if safe_float(r.get("mean_reliability_score")) is not None],
                key=lambda r: safe_float(r.get("mean_reliability_score")) or 0.0,
            )[:10],
            "strongest_organs_by_mean_reliability": sorted(
                [r for r in organ_rows if safe_float(r.get("mean_reliability_score")) is not None],
                key=lambda r: safe_float(r.get("mean_reliability_score")) or 0.0,
                reverse=True,
            )[:10],
        },
        "accuracy_warning": "Automatic pseudo-label quality only; no expert ground-truth accuracy is claimed.",
    }
    write_json(output_dir / "round1_auto_quality_summary.json", summary)
    return summary


def rebuild_manifest(round_root: Path, case_list: Path, target_config: Path, model_dir: Path, device: str) -> dict[str, Any]:
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    manifest_path = round_root / "mstep" / "voxtell_prompt_student_manifest.json"
    student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device=device)
    return student.build_training_manifest(
        cases_root=round_root / "estep" / "annotation_versions",
        output_manifest=manifest_path,
        case_list=case_list,
        require_images=True,
        student_prediction_root=None,
    )


def audit_manifest(round_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = round_root / "mstep" / "voxtell_prompt_student_manifest.json"
    doc = read_json(manifest_path, {}) or {}
    items = rows_from_json(doc)
    by_source = Counter(str(r.get("negative_source")) for r in items if r.get("supervision_type") == "negative")
    by_prompt = Counter(str(r.get("prompt_source")) for r in items if r.get("prompt_source") not in (None, ""))
    positive = [r for r in items if r.get("supervision_type") == "positive"]
    negative = [r for r in items if r.get("supervision_type") == "negative"]
    variant_positive = [r for r in positive if r.get("is_prompt_variant")]
    audit = {
        "stage": "round1_voxtell_manifest_audit",
        "status": "success" if items else "missing_manifest_items",
        "manifest_path": str(manifest_path),
        "num_items": len(items),
        "num_cases": doc.get("num_cases") or len({r.get("case_id") for r in items}),
        "num_positive_items": doc.get("num_positive_items") or len(positive),
        "num_negative_items": doc.get("num_negative_items") or len(negative),
        "grade_counts": doc.get("grade_counts") or counter_dict(items, "grade"),
        "prompt_variant_expansion_enabled": doc.get("prompt_variant_expansion_enabled"),
        "num_prompt_expanded_items": doc.get("num_prompt_expanded_items") or len(items),
        "num_canonical_prompt_items": doc.get("num_canonical_prompt_items") or sum(1 for r in items if not r.get("is_prompt_variant")),
        "num_prompt_variant_items": doc.get("num_prompt_variant_items") or sum(1 for r in items if r.get("is_prompt_variant")),
        "prompt_source_counts": dict(sorted(by_prompt.items())),
        "positive_prompt_variant_rate": pct(len(variant_positive), len(positive)),
        "negative_prompt_ratio": doc.get("negative_prompt_ratio"),
        "negative_quota_policy": doc.get("negative_quota_policy"),
        "negative_source_counts": dict(sorted(by_source.items())),
        "negative_source_counts_canonical": doc.get("negative_source_counts") or {},
        "negative_source_shortfalls": doc.get("negative_source_shortfalls") or {},
        "num_strong_training_items": doc.get("num_strong_training_items"),
        "num_distillation_eligible_items": doc.get("num_distillation_eligible_items"),
        "num_zero_weight_items": doc.get("num_zero_weight_items"),
        "fields_present": {
            "prompt_source": any("prompt_source" in r for r in items),
            "prompt_variant_index": any("prompt_variant_index" in r for r in items),
            "canonical_prompt": any("canonical_prompt" in r for r in items),
            "is_prompt_variant": any("is_prompt_variant" in r for r in items),
            "prompt_family_id": any("prompt_family_id" in r for r in items),
            "negative_source": any("negative_source" in r for r in negative),
        },
        "ready_for_teacher_report": bool(items and any(r.get("is_prompt_variant") for r in items) and negative),
        "accuracy_warning": "Manifest audit describes pseudo-label distillation data, not expert-label accuracy.",
    }
    write_json(output_dir / "round1_manifest_audit.json", audit)
    return audit


def checkpoint_dir(round_root: Path) -> Path:
    return round_root / "mstep" / "voxtell_finetuned_model"


def checkpoint_ready(model_dir: Path) -> bool:
    return (model_dir / "plans.json").exists() and (model_dir / "fold_0" / "checkpoint_final.pth").exists()




def sync_pilot_mstep_result(round_root: Path) -> dict[str, Any]:
    """Keep voxtell_prompt_mstep_result.json aligned when training was launched directly.

    Short 10h pilots produce a real checkpoint and can pass automatic gates, but
    they should not silently become a formal Round2 competition checkpoint unless
    a full M-step run explicitly marks them eligible.
    """
    mstep = round_root / "mstep"
    result_path = mstep / "voxtell_prompt_mstep_result.json"
    train = read_json(mstep / "voxtell_prompt_train_result.json", {}) or {}
    if not train:
        return read_json(result_path, {}) or {}
    manifest = read_json(mstep / "voxtell_prompt_student_manifest.json", {}) or {}
    sanity = read_json(mstep / "voxtell_student_sanity_check.json", {}) or {}
    quality = read_json(mstep / "voxtell_student_quality_gate.json", {}) or {}
    model = mstep / "voxtell_finetuned_model"
    full_items = int(manifest.get("num_items") or 0)
    trained_items = int(train.get("num_manifest_items") or 0)
    max_steps = int(train.get("max_steps") or 0)
    pilot_short = bool((full_items and trained_items and trained_items < full_items) or max_steps > 0)
    gates_success = sanity.get("status") == "success" and quality.get("status") == "success"
    result = {
        "stage": "voxtell_style_3d_prompt_mstep",
        "status": "success" if train.get("status") == "success" and checkpoint_ready(model) else "failed",
        "training_status": "completed_pilot_quality_gated" if gates_success and pilot_short else ("completed" if train.get("status") == "success" else "failed"),
        "student_backend": "voxtell_style_3d_prompt",
        "manifest_path": str(mstep / "voxtell_prompt_student_manifest.json"),
        "num_items": manifest.get("num_items"),
        "num_cases": manifest.get("num_cases"),
        "num_positive_items": manifest.get("num_positive_items"),
        "num_negative_items": manifest.get("num_negative_items"),
        "num_prompt_variant_items": manifest.get("num_prompt_variant_items"),
        "num_distillation_eligible_items": manifest.get("num_distillation_eligible_items"),
        "target_config": manifest.get("target_config"),
        "model_dir": train.get("model_dir"),
        "trainer_enabled": True,
        "pilot_short_training": pilot_short,
        "pilot_trained_manifest_items": trained_items,
        "pilot_max_steps": train.get("max_steps"),
        "pilot_epochs": train.get("epochs"),
        "pilot_freeze_encoder": train.get("freeze_encoder"),
        "mean_loss": train.get("mean_loss"),
        "last_loss": train.get("last_loss"),
        "mean_effective_loss_weight": train.get("mean_effective_loss_weight"),
        "effective_loss_weight_range": train.get("effective_loss_weight_range"),
        "finetuned_checkpoint": train.get("finetuned_checkpoint"),
        "inference_model_dir": train.get("inference_model_dir"),
        "inference_checkpoint": train.get("inference_checkpoint"),
        "sanity_check": sanity or None,
        "quality_gate": quality or None,
        "pilot_checkpoint_quality_gated": gates_success,
        "checkpoint_eligible_for_next_round": False if pilot_short else bool(gates_success),
        "eligible_for_next_round_prompt_student": False if pilot_short else bool(gates_success),
        "eligible_as_teacher_candidate": False if pilot_short else bool(gates_success),
        "checkpoint_origin": "project_distillation",
        "is_project_student": True,
        "formal_round2_recommendation": "Pilot checkpoint passed automatic gates but was trained on a capped subset; run full M-step before formal Round2 competition." if pilot_short else "Eligible if automatic gates passed.",
        "accuracy_warning": "Pseudo-consistency only; no expert ground-truth accuracy is claimed.",
    }
    write_json(result_path, result)
    return result

def collect_mstep_status(round_root: Path, output_dir: Path) -> dict[str, Any]:
    mstep = round_root / "mstep"
    result = read_json(mstep / "voxtell_prompt_mstep_result.json", {}) or {}
    sanity = read_json(mstep / "voxtell_student_sanity_check.json", None)
    quality = read_json(mstep / "voxtell_student_quality_gate.json", None)
    model = checkpoint_dir(round_root)
    manifest = read_json(mstep / "voxtell_prompt_student_manifest.json", {}) or {}
    status = {
        "stage": "round1_student_training_status",
        "status": "checkpoint_ready" if checkpoint_ready(model) else "checkpoint_missing_or_pending",
        "mstep_result_path": str(mstep / "voxtell_prompt_mstep_result.json"),
        "inference_model_dir": str(model),
        "checkpoint_exists": checkpoint_ready(model),
        "training_status": result.get("training_status"),
        "mstep_status": result.get("status"),
        "checkpoint_eligible_for_next_round": bool(result.get("checkpoint_eligible_for_next_round")),
        "eligible_for_next_round_prompt_student": bool(result.get("eligible_for_next_round_prompt_student", result.get("checkpoint_eligible_for_next_round"))),
        "eligible_as_teacher_candidate": bool(result.get("eligible_as_teacher_candidate", result.get("checkpoint_eligible_for_next_round"))),
        "num_items": manifest.get("num_items") or result.get("num_items"),
        "num_cases": manifest.get("num_cases") or result.get("num_cases"),
        "num_positive_items": manifest.get("num_positive_items"),
        "num_negative_items": manifest.get("num_negative_items"),
        "num_prompt_variant_items": manifest.get("num_prompt_variant_items"),
        "mean_effective_loss_weight": (result.get("train_result") or {}).get("mean_effective_loss_weight") or result.get("mean_effective_loss_weight"),
        "sanity_check": sanity or result.get("sanity_check"),
        "quality_gate": quality or result.get("quality_gate"),
        "accuracy_warning": "Training gates are automatic pseudo-consistency checks, not expert-label accuracy.",
    }
    write_json(output_dir / "round1_student_training_status.json", status)
    return status


def maybe_run_existing_gates(round_root: Path, case_list: Path, target_config: Path, model_dir: Path) -> dict[str, Any]:
    if not checkpoint_ready(model_dir):
        return {"status": "skipped", "reason": "checkpoint_missing"}
    os.environ["MEDAI_OUTPUT_ROOT"] = str(round_root.parent)
    os.environ["MEDAI_VOXTELL_MODEL_DIR"] = str(model_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("round1_patch_run_em_training", ROOT / "scripts" / "run_em_training.py")
    if spec is None or spec.loader is None:
        return {"status": "failed", "reason": "cannot_load_run_em_training"}
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)

    em.OUTPUT_ROOT = round_root.parent
    em.CASE_LIST = case_list
    em.PROMPT_TARGET_CONFIG = target_config
    sanity = em.run_voxtell_student_sanity_check(1, model_dir, round_root / "mstep" / "voxtell_prompt_student_manifest.json")
    quality = None
    if sanity.get("status") == "success":
        quality = em.run_voxtell_student_quality_gate(1, model_dir, round_root / "mstep" / "voxtell_prompt_student_manifest.json")
    return {"status": "completed", "sanity_check": sanity, "quality_gate": quality}


def dice(pred: Path, ref: Path) -> tuple[float | None, bool, bool, str | None]:
    try:
        import nibabel as nib
        import numpy as np
        pa = np.asanyarray(nib.load(str(pred)).dataobj) > 0
        ra = np.asanyarray(nib.load(str(ref)).dataobj) > 0
        if pa.shape != ra.shape:
            return None, bool(pa.sum()), bool(ra.sum()), "shape_mismatch"
        pred_has = bool(pa.sum())
        ref_has = bool(ra.sum())
        total = int(pa.sum()) + int(ra.sum())
        return (float(2 * int((pa & ra).sum()) / total) if total else 1.0), pred_has, ref_has, None
    except Exception as exc:
        return None, False, False, str(exc)


def select_prompt_robustness_jobs(manifest_path: Path, organs: list[str], max_cases: int) -> list[dict[str, Any]]:
    doc = read_json(manifest_path, {}) or {}
    items = rows_from_json(doc)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("supervision_type") != "positive":
            continue
        if str(item.get("organ")) not in organs:
            continue
        if not Path(str(item.get("image") or item.get("ct_path") or "")).exists():
            continue
        if not Path(str(item.get("mask") or item.get("mask_path") or "")).exists():
            continue
        grouped[(str(item.get("case_id")), str(item.get("organ")))].append(item)
    jobs = []
    used_cases: set[str] = set()
    for organ in organs:
        for (case_id, item_organ), rows in sorted(grouped.items()):
            if item_organ != organ:
                continue
            if len(used_cases) >= max_cases and case_id not in used_cases:
                continue
            prompts = []
            seen = set()
            for row in sorted(rows, key=lambda r: int(r.get("prompt_variant_index") or 0)):
                prompt = str(row.get("prompt") or "").strip()
                if prompt and prompt not in seen:
                    prompts.append(prompt)
                    seen.add(prompt)
                if len(prompts) >= 4:
                    break
            if len(prompts) >= 2:
                base = rows[0]
                jobs.append({"case_id": case_id, "organ": organ, "image": base.get("image") or base.get("ct_path"), "mask": base.get("mask") or base.get("mask_path"), "prompts": prompts})
                used_cases.add(case_id)
                break
    return jobs


def run_prompt_robustness(round_root: Path, target_config: Path, model_dir: Path, output_dir: Path, device: str, max_cases: int, timeout_sec: int) -> dict[str, Any]:
    manifest_path = round_root / "mstep" / "voxtell_prompt_student_manifest.json"
    out_root = output_dir / "prompt_robustness_outputs"
    if not checkpoint_ready(model_dir):
        result = {"stage": "round1_prompt_robustness_eval", "status": "skipped", "reason": "checkpoint_missing", "model_dir": str(model_dir)}
        write_json(output_dir / "round1_prompt_robustness_eval.json", result)
        return result
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    jobs = select_prompt_robustness_jobs(manifest_path, PREFERRED_ORGANS, max_cases)
    student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device=device)
    rows = []
    for job in jobs:
        case_out = out_root / job["case_id"] / job["organ"]
        case_out.mkdir(parents=True, exist_ok=True)
        ref_mask = Path(job["mask"])
        canonical_mask: Path | None = None
        for idx, prompt in enumerate(job["prompts"]):
            variant_dir = case_out / f"variant_{idx:02d}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            infer = student.segment(
                Path(job["image"]),
                variant_dir,
                prompts=[job["organ"]],
                prompt_overrides={job["organ"]: prompt},
                dry_run=False,
                timeout_sec=timeout_sec,
                prompt_batch_size=1,
            )
            pred = variant_dir / f"{job['organ']}.nii.gz"
            if idx == 0:
                canonical_mask = pred
            ref_dsc, pred_has, ref_has, reason = dice(pred, ref_mask)
            can_dsc = None
            can_reason = None
            if idx > 0 and canonical_mask and canonical_mask.exists() and pred.exists():
                can_dsc, _, _, can_reason = dice(pred, canonical_mask)
            rows.append({
                "case_id": job["case_id"],
                "organ": job["organ"],
                "prompt": prompt,
                "prompt_variant_index": idx,
                "prediction_mask": str(pred),
                "reference_mask": str(ref_mask),
                "dsc_vs_reference": ref_dsc,
                "dsc_vs_canonical_prediction": can_dsc,
                "prediction_nonempty": pred_has,
                "reference_nonempty": ref_has,
                "skip_reason": reason or can_reason,
                "inference_status": infer.get("status"),
            })
    variant_rows = [r for r in rows if r["prompt_variant_index"] > 0]
    consistency = [r["dsc_vs_canonical_prediction"] for r in variant_rows if isinstance(r.get("dsc_vs_canonical_prediction"), (int, float))]
    result = {
        "stage": "round1_prompt_robustness_eval",
        "status": "success" if rows else "no_jobs",
        "model_dir": str(model_dir),
        "num_jobs": len(jobs),
        "num_prompt_evaluations": len(rows),
        "variant_vs_canonical_mean_dsc": round(sum(consistency) / len(consistency), 6) if consistency else None,
        "variant_empty_rate": pct(sum(1 for r in variant_rows if not r.get("prediction_nonempty")), len(variant_rows)),
        "evaluations": rows,
        "accuracy_warning": "Prompt robustness is measured against selected pseudo-labels and canonical student outputs, not expert labels.",
    }
    write_json(output_dir / "round1_prompt_robustness_eval.json", result)
    return result


def select_negative_jobs(manifest_path: Path, max_cases: int, max_per_source: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = read_json(manifest_path, {}) or {}
    items = rows_from_json(doc)
    manifest_sources = sorted({
        str(item.get("negative_source") or "unknown")
        for item in items
        if item.get("supervision_type") == "negative" and not item.get("is_prompt_variant")
    })
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_cases_by_source: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if item.get("supervision_type") != "negative" or item.get("is_prompt_variant"):
            continue
        source = str(item.get("negative_source") or "unknown")
        if source not in SAFE_NEGATIVE_SOURCES:
            continue
        if item.get("zero_mask_role") != "negative_target_mask":
            continue
        if source != "nonmedical_absent_object" and not item.get("negative_evidence"):
            continue
        case_id = str(item.get("case_id") or "")
        organ = str(item.get("organ") or "")
        if not case_id or not organ or (source, case_id, organ) in seen:
            continue
        if len(used_cases_by_source[source]) >= max_cases and case_id not in used_cases_by_source[source]:
            continue
        if len(by_source[source]) >= max_per_source:
            continue
        image = Path(str(item.get("image") or item.get("ct_path") or ""))
        if not image.exists():
            continue
        by_source[source].append(item)
        used_cases_by_source[source].add(case_id)
        seen.add((source, case_id, organ))
    jobs = []
    for source in sorted(by_source):
        jobs.extend(by_source[source])
    audit = {
        "manifest_sources": manifest_sources,
        "allowed_sources": sorted(SAFE_NEGATIVE_SOURCES),
        "skipped_unsafe_sources": [source for source in manifest_sources if source not in SAFE_NEGATIVE_SOURCES],
        "sampled_sources": sorted(by_source),
        "missing_sampled_sources": [source for source in manifest_sources if source not in by_source],
        "max_cases_per_source": max_cases,
        "max_negatives_per_source": max_per_source,
    }
    return jobs, audit


def mask_stats(path: Path) -> dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
        arr = np.asanyarray(nib.load(str(path)).dataobj)
        return {"exists": True, "non_nan": not bool(np.isnan(arr).any()), "voxels": int((arr > 0).sum()), "empty": bool((arr > 0).sum() == 0), "shape": [int(x) for x in arr.shape[:3]]}
    except Exception as exc:
        return {"exists": path.exists(), "error": str(exc), "empty": None}


def run_negative_suppression(round_root: Path, target_config: Path, model_dir: Path, output_dir: Path, device: str, max_cases: int, max_per_source: int, timeout_sec: int) -> dict[str, Any]:
    manifest_path = round_root / "mstep" / "voxtell_prompt_student_manifest.json"
    out_root = output_dir / "negative_suppression_outputs"
    if not checkpoint_ready(model_dir):
        result = {"stage": "round1_negative_prompt_suppression_eval", "status": "skipped", "reason": "checkpoint_missing", "model_dir": str(model_dir)}
        write_json(output_dir / "round1_negative_prompt_suppression_eval.json", result)
        return result
    from cli_anything.medai.core.voxtell_student import VoxTellStudent

    jobs, sampling_audit = select_negative_jobs(manifest_path, max_cases, max_per_source)
    student = VoxTellStudent(model_dir=model_dir, target_config=target_config, device=device)
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job.get("case_id"))].append(job)
    for case_id, case_jobs in grouped.items():
        prompts = [str(j.get("organ")) for j in case_jobs]
        image = Path(str(case_jobs[0].get("image") or case_jobs[0].get("ct_path")))
        case_out = out_root / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        infer = student.segment(image, case_out, prompts=prompts, dry_run=False, timeout_sec=timeout_sec, prompt_batch_size=max(1, len(prompts)))
        for job in case_jobs:
            organ = str(job.get("organ"))
            pred = case_out / f"{organ}.nii.gz"
            stats = mask_stats(pred)
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "prompt": job.get("prompt"),
                "negative_source": job.get("negative_source"),
                "negative_reason": job.get("negative_reason"),
                "prediction_mask": str(pred),
                "prediction_empty": stats.get("empty"),
                "prediction_voxels": stats.get("voxels"),
                "non_nan": stats.get("non_nan"),
                "inference_status": infer.get("status"),
            })
    source_summary = {}
    for source in sorted({str(r.get("negative_source")) for r in rows}):
        sr = [r for r in rows if str(r.get("negative_source")) == source]
        source_summary[source] = {
            "n": len(sr),
            "empty_count": sum(1 for r in sr if r.get("prediction_empty") is True),
            "empty_rate": pct(sum(1 for r in sr if r.get("prediction_empty") is True), len(sr)),
            "mean_positive_voxels": round(sum(float(r.get("prediction_voxels") or 0) for r in sr) / len(sr), 3) if sr else 0.0,
        }
    result = {
        "stage": "round1_negative_prompt_suppression_eval",
        "status": "success" if rows else "no_negative_jobs",
        "model_dir": str(model_dir),
        "num_evaluations": len(rows),
        "sampling_audit": sampling_audit,
        "source_summary": source_summary,
        "overall_empty_rate": pct(sum(1 for r in rows if r.get("prediction_empty") is True), len(rows)),
        "evaluations": rows,
        "accuracy_warning": "Negative suppression checks whether the automatic agent avoids masks for negative prompts; no expert labels are used.",
    }
    write_json(output_dir / "round1_negative_prompt_suppression_eval.json", result)
    return result


def write_markdown_summary(output_dir: Path, auto_quality: dict[str, Any], manifest: dict[str, Any], mstep: dict[str, Any], robustness: dict[str, Any] | None, negative: dict[str, Any] | None, final: dict[str, Any]) -> None:
    lines = [
        "# Round1 10h Automatic VoxTell-Agent Patch Experiments",
        "",
        "## Teacher-facing conclusion order",
        "1. This is an automated medical segmentation agent; no expert manual labels or review are used.",
        "2. Teacher ensemble produces pseudo-labels, audited by LabelCritic and ShapeKit.",
        "3. Prompt variants and negative prompts are included in VoxTell-aligned student training data.",
        "4. A trained student checkpoint is reportable only after automatic sanity and quality gates pass.",
        "5. Student quality is reported as pseudo-label consistency, prompt robustness, and negative suppression.",
        "",
        "## Current results",
        f"- Cases: {auto_quality.get('num_cases_summary')} / case-organ rows: {auto_quality.get('case_organ_score_rows')} / selected labels: {auto_quality.get('num_selected_labels')}",
        f"- Grade counts: `{auto_quality.get('grade_counts')}`",
        f"- LabelCritic used rate: `{(auto_quality.get('labelcritic') or {}).get('used_rate')}`",
        f"- Manifest items: `{manifest.get('num_items')}`; positives: `{manifest.get('num_positive_items')}`; negatives: `{manifest.get('num_negative_items')}`",
        f"- Prompt variants: `{manifest.get('num_prompt_variant_items')}`; prompt source counts: `{manifest.get('prompt_source_counts')}`",
        f"- Negative source counts expanded: `{manifest.get('negative_source_counts')}`",
        f"- Negative source counts canonical: `{manifest.get('negative_source_counts_canonical')}`",
        f"- Checkpoint status: `{mstep.get('status')}`; prompt-student eligible for next round: `{mstep.get('eligible_for_next_round_prompt_student', mstep.get('checkpoint_eligible_for_next_round'))}`",
    ]
    if robustness:
        lines.append(f"- Prompt robustness status: `{robustness.get('status')}`; variant-vs-canonical mean DSC: `{robustness.get('variant_vs_canonical_mean_dsc')}`")
    if negative:
        lines.append(f"- Negative suppression status: `{negative.get('status')}`; overall empty rate: `{negative.get('overall_empty_rate')}`")
    lines.extend([
        "",
        "## Reporting guardrail",
        "All metrics are automatic pseudo-label consistency or I/O quality checks. Do not claim expert-label accuracy or official VoxTell full reproduction.",
        "",
        "## Next action",
        final.get("next_action", "Run/finish M-step training, then re-run this script with robustness and negative suppression enabled."),
        "",
    ])
    (output_dir / "round1_10h_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    round_root = Path(args.round_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else round_root / "mstep" / "round1_10h_patch_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_list = Path(args.case_list).resolve()
    target_config = Path(args.target_config).resolve()
    base_model_dir = Path(args.model_dir).resolve()
    report_model_dir = checkpoint_dir(round_root) if checkpoint_ready(checkpoint_dir(round_root)) else base_model_dir

    started = time.time()
    auto_quality = summarize_auto_quality(round_root, case_list, output_dir)
    manifest_doc = None
    if args.rebuild_manifest:
        manifest_doc = rebuild_manifest(round_root, case_list, target_config, base_model_dir, args.device)
    manifest_audit = audit_manifest(round_root, output_dir)

    gate_run = None
    if args.run_existing_gates:
        gate_run = maybe_run_existing_gates(round_root, case_list, target_config, checkpoint_dir(round_root))
    sync_pilot_mstep_result(round_root)
    mstep_status = collect_mstep_status(round_root, output_dir)

    robustness = None
    if args.run_prompt_robustness:
        robustness = run_prompt_robustness(round_root, target_config, checkpoint_dir(round_root), output_dir, args.device, args.max_robustness_cases, args.timeout_sec)
    else:
        robustness = read_json(output_dir / "round1_prompt_robustness_eval.json", None)

    negative = None
    if args.run_negative_suppression:
        negative = run_negative_suppression(round_root, target_config, checkpoint_dir(round_root), output_dir, args.device, args.max_negative_cases, args.max_negatives_per_source, args.timeout_sec)
    else:
        negative = read_json(output_dir / "round1_negative_prompt_suppression_eval.json", None)

    ready = {
        "auto_quality_summary": auto_quality.get("status") == "success",
        "manifest_prompt_variants": bool(manifest_audit.get("fields_present", {}).get("prompt_source") and manifest_audit.get("num_prompt_variant_items", 0)),
        "manifest_negative_sources": bool(manifest_audit.get("fields_present", {}).get("negative_source") and manifest_audit.get("num_negative_items", 0)),
        "checkpoint_ready": bool(mstep_status.get("checkpoint_exists")),
        "sanity_success": (mstep_status.get("sanity_check") or {}).get("status") == "success",
        "quality_gate_success": (mstep_status.get("quality_gate") or {}).get("status") == "success",
        "prompt_robustness_done": bool(robustness and robustness.get("status") == "success"),
        "negative_suppression_done": bool(negative and negative.get("status") == "success"),
    }
    if not ready["checkpoint_ready"]:
        next_action = "Run Round1 VoxTell M-step short training to produce voxtell_finetuned_model, then re-run gates and inference evaluations."
    elif not (ready["sanity_success"] and ready["quality_gate_success"]):
        next_action = "Run automatic sanity and mini pseudo-label consistency gates before using the checkpoint in any formal claim."
    elif not (ready["prompt_robustness_done"] and ready["negative_suppression_done"]):
        next_action = "Run prompt robustness and negative suppression inference checks for the teacher-facing Round1 report."
    else:
        next_action = "Round1 10-hour report package is complete for automatic-agent pseudo-consistency claims."

    final = {
        "stage": "round1_10h_patch_experiment_summary",
        "status": "complete" if all(ready.values()) else "partial_complete",
        "round_root": str(round_root),
        "output_dir": str(output_dir),
        "runtime_sec": round(time.time() - started, 3),
        "readiness": ready,
        "next_action": next_action,
        "artifacts": {
            "auto_quality_summary": str(output_dir / "round1_auto_quality_summary.json"),
            "manifest_audit": str(output_dir / "round1_manifest_audit.json"),
            "student_training_status": str(output_dir / "round1_student_training_status.json"),
            "prompt_robustness_eval": str(output_dir / "round1_prompt_robustness_eval.json"),
            "negative_suppression_eval": str(output_dir / "round1_negative_prompt_suppression_eval.json"),
            "summary_json": str(output_dir / "round1_10h_experiment_summary.json"),
            "summary_markdown": str(output_dir / "round1_10h_experiment_summary.md"),
        },
        "teacher_facing_claim": (
            "Automatic teacher ensemble + LabelCritic/ShapeKit pseudo-label selection and VoxTell-aligned "
            "student distillation audit. Results are pseudo-consistency and automatic quality-gate evidence, not expert accuracy."
        ),
        "accuracy_warning": "No expert manual labels or review are used; do not claim clinical/expert-label accuracy.",
    }
    if gate_run:
        final["gate_run"] = gate_run
    if manifest_doc is not None:
        final["rebuilt_manifest_items"] = manifest_doc.get("num_items")
    write_json(output_dir / "round1_10h_experiment_summary.json", final)
    write_markdown_summary(output_dir, auto_quality, manifest_audit, mstep_status, robustness, negative, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if final["status"] in {"complete", "partial_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
