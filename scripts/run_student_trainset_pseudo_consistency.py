#!/usr/bin/env python3
"""Audit full M-step VoxTell student consistency against selected pseudo labels.

This script intentionally evaluates only student-vs-selected-pseudo-label
consistency. It does not open or report any external reference labels.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.shapekit_runner import run_shapekit
from cli_anything.medai.core.voxtell_student import VoxTellStudent


KEY_ORGANS = {
    "pancreas",
    "aorta",
    "adrenal_gland_left",
    "adrenal_gland_right",
    "duodenum",
    "colon",
    "small_bowel",
    "spleen",
    "kidney_left",
    "kidney_right",
    "liver",
    "stomach",
    "gall_bladder",
}
SAFE_SHAPEKIT_STATUSES = {"success"}
FORBIDDEN_OUTPUT_TOKENS = {"GT", "metric_target=GT", "student_vs_gt", "teacher_vs_gt"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run trainset pseudo-consistency audit for a full M-step VoxTell student."
    )
    ap.add_argument("--case-list", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--student-model-dir", type=Path, required=True)
    ap.add_argument("--selected-pseudo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--postprocess-policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "configs/organ_taxonomy.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--prompt-batch-size", type=int, default=int(os.getenv("MEDAI_STUDENT_PROMPT_BATCH_SIZE", "16")))
    ap.add_argument("--shapekit-timeout-sec", type=int, default=int(os.getenv("MEDAI_STUDENT_SHAPEKIT_TIMEOUT_SEC", "1800")))
    ap.add_argument("--reuse-raw-root", type=Path, default=None)
    ap.add_argument("--reuse-shapekit-root", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--disable-negative-safe-postprocess", action="store_true")
    ap.add_argument("--positive-min-mean-dsc", type=float, default=float(os.getenv("MEDAI_TRAINSET_POSITIVE_MIN_MEAN_DSC", "0.60")))
    ap.add_argument("--max-oversegmentation-rate", type=float, default=float(os.getenv("MEDAI_TRAINSET_MAX_OVERSEG_RATE", "0.15")))
    ap.add_argument("--max-key-organ-median-volume-ratio", type=float, default=float(os.getenv("MEDAI_TRAINSET_MAX_KEY_VOLUME_RATIO", "1.5")))
    ap.add_argument("--negative-false-positive-voxel-threshold", type=int, default=int(os.getenv("MEDAI_NEGATIVE_FALSE_POSITIVE_VOXELS", "0")))
    return ap.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def read_cases(case_list: Path, max_cases: int = 0) -> list[dict[str, str]]:
    with case_list.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        ct_path = str(row.get("ct_path") or row.get("image") or row.get("ct_image") or "").strip()
        if case_id and ct_path:
            out.append({"case_id": case_id, "ct_path": str(Path(ct_path).resolve())})
    return out[:max_cases] if max_cases > 0 else out


def write_case_list(path: Path, cases: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        writer.writerows(cases)


def load_target_organs(target_config: Path) -> list[str]:
    doc = read_json(target_config, default={})
    raw = doc.get("target_organs") if isinstance(doc, dict) else doc
    organs = [
        str(item if isinstance(item, str) else item.get("organ") or item.get("id") or "").strip()
        for item in (raw or [])
    ]
    organs = [organ for organ in organs if organ]
    if len(organs) != 373:
        raise SystemExit(f"Expected 373 target organs, found {len(organs)} in {target_config}")
    if len(set(organs)) != len(organs):
        raise SystemExit("Target organ config contains duplicate target ids")
    return organs


def manifest_index(manifest_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    doc = read_json(manifest_path, default={}) or {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    priority = {"positive": 3, "negative": 2}
    for row in doc.get("items") or []:
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or row.get("canonical_organ") or "")
        if not case_id or not organ:
            continue
        key = (case_id, organ)
        current = index.get(key)
        row_kind = str(row.get("supervision_type") or "positive")
        current_kind = str(current.get("supervision_type") or "") if current else ""
        if current is None or priority.get(row_kind, 0) > priority.get(current_kind, 0):
            index[key] = row
    return index, doc


def _path_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(str(path)).exists()


def preflight(args: argparse.Namespace, cases: list[dict[str, str]], targets: list[str], manifest_doc: dict[str, Any]) -> dict[str, Any]:
    model_dir = args.student_model_dir.resolve()
    run_dir = model_dir.parent
    result = read_json(run_dir / "voxtell_prompt_train_result.json", default={}) or {}
    stability = read_json(run_dir / "training_stability_diagnosis.json", default={}) or {}
    exposure = read_json(run_dir / "organ_exposure_audit.json", default={}) or {}
    nonfinite = read_json(run_dir / "nonfinite_gradient_audit.json", default={}) or {}
    failures: list[str] = []
    if result.get("status") != "success":
        failures.append("mstep_result_not_success")
    if not (model_dir / "plans.json").exists() or not (model_dir / "fold_0" / "checkpoint_final.pth").exists():
        failures.append("student_checkpoint_missing")
    if stability.get("status") != "passed":
        failures.append("training_stability_not_passed")
    if exposure.get("status") != "passed":
        failures.append("organ_exposure_not_passed")
    if int(nonfinite.get("nonfinite_gradient_steps") or 0) != 0:
        failures.append("nonfinite_gradient_steps_nonzero")
    bad_manifest_rows = [
        idx for idx, row in enumerate(manifest_doc.get("items") or [])
        if float(row.get("training_weight") or 0.0) > 0.0
        and (row.get("training_eligible") is not True or not row.get("organ") or not row.get("case_id"))
    ]
    if bad_manifest_rows:
        failures.append("manifest_trainable_rows_invalid")
    missing_ct = [row for row in cases if not Path(row["ct_path"]).exists()]
    if missing_ct:
        failures.append("case_ct_missing")
    audit = {
        "stage": "student_trainset_pseudo_consistency_preflight",
        "status": "passed" if not failures else "failed",
        "failure_reasons": failures,
        "case_count": len(cases),
        "target_count": len(targets),
        "expected_prediction_rows": len(cases) * len(targets),
        "student_model_dir": str(model_dir),
        "checkpoint": str(model_dir / "fold_0" / "checkpoint_final.pth"),
        "mstep_result_status": result.get("status"),
        "mstep_steps": result.get("steps"),
        "mstep_stopped_early": result.get("stopped_early"),
        "training_stability_status": stability.get("status"),
        "organ_exposure_status": exposure.get("status"),
        "nonfinite_gradient_steps": nonfinite.get("nonfinite_gradient_steps"),
        "manifest_items": len(manifest_doc.get("items") or []),
        "invalid_manifest_row_examples": bad_manifest_rows[:20],
        "missing_ct_examples": missing_ct[:10],
        "metric_target_policy": "selected_pseudo_label",
    }
    return audit


def run_student_inference(args: argparse.Namespace, cases: list[dict[str, str]], targets: list[str], raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_root.mkdir(parents=True, exist_ok=True)
    student = VoxTellStudent(
        model_dir=args.student_model_dir.resolve(),
        target_config=args.target_config.resolve(),
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    success_cases = 0
    for case in cases:
        case_id = case["case_id"]
        case_out = raw_root / case_id
        result_path = case_out / "voxtell_student_result.json"
        existing = read_json(result_path, default={}) if result_path.exists() and not args.overwrite else {}
        existing_ok = (
            existing.get("status") in {"success", "partial_success"}
            and all((case_out / f"{organ}.nii.gz").exists() for organ in targets)
        )
        if existing_ok:
            result = existing
        else:
            if args.overwrite and case_out.exists():
                shutil.rmtree(case_out)
            case_out.mkdir(parents=True, exist_ok=True)
            result = student.segment(
                ct_image=case["ct_path"],
                prompts=targets,
                output_dir=case_out,
                dry_run=False,
                timeout_sec=1800,
                prompt_batch_size=args.prompt_batch_size,
            )
        if all((case_out / f"{organ}.nii.gz").exists() for organ in targets):
            success_cases += 1
        case_results.append({
            "case_id": case_id,
            "status": result.get("status"),
            "num_masks": result.get("num_masks"),
            "num_empty_masks": result.get("num_empty_masks"),
            "num_failed_organs": result.get("num_failed_organs"),
            "runtime_sec": result.get("runtime_sec"),
        })
        per_organ = result.get("per_organ_status") or {}
        for organ in targets:
            mask = case_out / f"{organ}.nii.gz"
            status = (per_organ.get(organ) or {}).get("status")
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "ct_path": case["ct_path"],
                "raw_student_mask": str(mask),
                "raw_prediction_exists": mask.exists(),
                "raw_student_status": status or ("success" if mask.exists() else "missing"),
            })
    summary = {
        "stage": "student_trainset_raw_inference",
        "status": "success" if success_cases == len(cases) else "partial_success",
        "case_count": len(cases),
        "case_success_count": success_cases,
        "target_count": len(targets),
        "prediction_rows": len(rows),
        "raw_root": str(raw_root),
        "case_results": case_results,
    }
    return rows, summary


def collect_existing_raw_predictions(raw_root: Path, cases: list[dict[str, str]], targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    success_cases = 0
    for case in cases:
        case_id = case["case_id"]
        case_ok = True
        for organ in targets:
            mask = raw_root / case_id / f"{organ}.nii.gz"
            exists = mask.exists()
            case_ok = case_ok and exists
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "ct_path": case["ct_path"],
                "raw_student_mask": str(mask),
                "raw_prediction_exists": exists,
                "raw_student_status": "success" if exists else "missing",
            })
        if case_ok:
            success_cases += 1
    return rows, {
        "stage": "student_trainset_raw_inference",
        "status": "success" if success_cases == len(cases) else "partial_success",
        "case_count": len(cases),
        "case_success_count": success_cases,
        "target_count": len(targets),
        "prediction_rows": len(rows),
        "raw_root": str(raw_root),
        "reused_existing_raw_root": True,
    }


def stage_for_shapekit(raw_root: Path, staged_root: Path, cases: list[dict[str, str]], targets: list[str]) -> int:
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    target_set = set(targets)
    for case in cases:
        case_id = case["case_id"]
        src_case = raw_root / case_id
        dst = staged_root / case_id / "segmentations"
        dst.mkdir(parents=True, exist_ok=True)
        for mask in sorted(src_case.glob("*.nii.gz")):
            organ = mask.name[:-7]
            if organ in target_set:
                shutil.copy2(mask, dst / mask.name)
                copied += 1
    return copied


def run_student_shapekit(args: argparse.Namespace, raw_root: Path, shapekit_root: Path, cases: list[dict[str, str]], targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    staged_root = args.output_dir / "student_predictions_shapekit_input"
    raw_output_root = args.output_dir / "student_predictions_shapekit_raw_output"
    flat_root = shapekit_root
    for path in [raw_output_root, flat_root]:
        if path.exists() and args.overwrite:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    staged = stage_for_shapekit(raw_root, staged_root, cases, targets)
    result = run_shapekit(
        ROOT / "third_party/ShapeKit-main",
        staged_root,
        raw_output_root,
        raw_output_root / "logs",
        cpu_count=2,
        dry_run=False,
        auto_config=True,
        timeout_sec=args.shapekit_timeout_sec,
    )
    rows: list[dict[str, Any]] = []
    processed = fallback = missing = 0
    for case in cases:
        case_id = case["case_id"]
        out_case = flat_root / case_id
        out_case.mkdir(parents=True, exist_ok=True)
        for organ in targets:
            raw_mask = raw_root / case_id / f"{organ}.nii.gz"
            sk_mask = raw_output_root / case_id / "segmentations" / f"{organ}.nii.gz"
            dst = out_case / f"{organ}.nii.gz"
            if result.get("status") == "success" and sk_mask.exists():
                shutil.copy2(sk_mask, dst)
                status = "success"
                reason = "student_mask_processed_by_shapekit"
                selected = "shapekit"
                processed += 1
            elif raw_mask.exists():
                shutil.copy2(raw_mask, dst)
                status = "fallback_original"
                reason = "shapekit_missing_for_target_raw_retained_for_audit"
                selected = "raw_fallback"
                fallback += 1
            else:
                status = "missing"
                reason = "raw_student_mask_missing"
                selected = "missing"
                missing += 1
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "raw_student_mask": str(raw_mask),
                "shapekit_mask": str(sk_mask) if sk_mask.exists() else "",
                "output_path": str(dst) if dst.exists() else "",
                "student_shapekit_status": status,
                "student_shapekit_reason": reason,
                "selected_for_student_candidate": selected,
            })
    summary = {
        "stage": "student_candidate_shapekit",
        "status": "success" if result.get("status") == "success" and processed > 0 else ("partial_success" if fallback else "failed"),
        "input_root": str(raw_root),
        "staged_root": str(staged_root),
        "raw_shapekit_output_root": str(raw_output_root),
        "flat_output_root": str(flat_root),
        "staged_masks": staged,
        "processed_by_shapekit": processed,
        "fallback_original": fallback,
        "missing": missing,
        "shapekit_result": result,
        "formal_policy": "Only ShapeKit-success student masks can become formal replacement candidates.",
    }
    return rows, summary


def collect_existing_shapekit_predictions(raw_root: Path, shapekit_root: Path, cases: list[dict[str, str]], targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    success = missing = 0
    for case in cases:
        case_id = case["case_id"]
        for organ in targets:
            raw_mask = raw_root / case_id / f"{organ}.nii.gz"
            sk_mask = shapekit_root / case_id / f"{organ}.nii.gz"
            if sk_mask.exists():
                status = "success"
                reason = "student_mask_reused_existing_shapekit"
                selected = "shapekit"
                success += 1
            else:
                status = "missing"
                reason = "reused_shapekit_mask_missing"
                selected = "missing"
                missing += 1
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "raw_student_mask": str(raw_mask),
                "shapekit_mask": str(sk_mask) if sk_mask.exists() else "",
                "output_path": str(sk_mask) if sk_mask.exists() else "",
                "student_shapekit_status": status,
                "student_shapekit_reason": reason,
                "selected_for_student_candidate": selected,
            })
    return rows, {
        "stage": "student_candidate_shapekit",
        "status": "success" if missing == 0 else "partial_success",
        "input_root": str(raw_root),
        "flat_output_root": str(shapekit_root),
        "processed_by_shapekit": success,
        "fallback_original": 0,
        "missing": missing,
        "reused_existing_shapekit_root": True,
        "formal_policy": "Only ShapeKit-success student masks can become formal replacement candidates.",
    }


def run_postprocess(args: argparse.Namespace, case_list_used: Path, shapekit_root: Path, post_root: Path) -> dict[str, Any]:
    if post_root.exists() and args.overwrite:
        shutil.rmtree(post_root)
    post_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/apply_organ_type_postprocess.py"),
        "--input-root", str(shapekit_root),
        "--output-root", str(post_root),
        "--case-list", str(case_list_used),
        "--policy", str(args.postprocess_policy.resolve()),
        "--taxonomy", str(args.taxonomy.resolve()),
        "--target-config", str(args.target_config.resolve()),
        "--parent-root", str(args.selected_pseudo_root.resolve()),
        "--overwrite",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    summary_path = post_root / "organ_type_postprocess_summary.json"
    summary = read_json(summary_path, default={}) or {}
    if proc.returncode != 0:
        summary.update({"status": "failed", "reason": f"postprocess_return_code_{proc.returncode}"})
    summary.update({
        "stage": "student_shape_then_organ_type_postprocess",
        "command": cmd,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-3000:],
        "input_root": str(shapekit_root),
        "output_root": str(post_root),
    })
    write_json(args.output_dir / "student_postprocess_summary.json", summary)
    return summary


def load_mask(path: Path | None) -> tuple[np.ndarray | None, str | None]:
    if path is None or not path.exists():
        return None, "missing"
    try:
        import nibabel as nib

        return np.asanyarray(nib.load(str(path)).dataobj) > 0, None
    except Exception as exc:
        return None, f"unreadable:{exc}"


def mask_volume(path: Path | None) -> int | None:
    mask, _ = load_mask(path)
    return int(mask.sum()) if mask is not None else None


NEGATIVE_PROVENANCE_FIELDS = [
    "fov_status",
    "fov_evidence",
    "coverage_evidence",
    "negative_source",
    "zero_mask_role",
    "negative_reason",
    "absence_confidence",
]


def is_confirmed_negative_absent(row: dict[str, Any]) -> bool:
    target_type = str(row.get("target_type") or "")
    supervision_type = str(row.get("supervision_type") or "")
    fov_status = str(row.get("fov_status") or "")
    zero_mask_role = str(row.get("zero_mask_role") or "")
    negative_source = str(row.get("negative_source") or "")
    if target_type not in {"negative_absent", "absent_negative"} or supervision_type != "negative":
        return False
    if zero_mask_role not in {"negative_absent_target_mask", "absent_negative_target_mask", "negative_target_mask"}:
        return False
    if fov_status not in {"out_of_fov", "confirmed_absent", "absent"}:
        return False
    return negative_source not in {"", "teacher_missing", "missing_teacher_output"}


def save_zero_like(mask_path: Path) -> None:
    import nibabel as nib

    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    zero = np.zeros(data.shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(zero, img.affine, img.header), str(mask_path))


def apply_negative_safe_postprocess(
    *,
    manifest: dict[tuple[str, str], dict[str, Any]],
    post_root: Path,
    cases: list[dict[str, str]],
    targets: list[str],
    threshold: int,
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suppressed = 0
    eligible = 0
    nonempty = 0
    for case in cases:
        case_id = case["case_id"]
        for organ in targets:
            mrow = manifest.get((case_id, organ), {})
            if not is_confirmed_negative_absent(mrow):
                continue
            eligible += 1
            mask_path = post_root / case_id / f"{organ}.nii.gz"
            volume = mask_volume(mask_path)
            if volume is None:
                continue
            should_suppress = volume > threshold
            if should_suppress:
                nonempty += 1
                if enabled:
                    save_zero_like(mask_path)
                    suppressed += 1
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "target_type": mrow.get("target_type"),
                "supervision_type": mrow.get("supervision_type"),
                "pre_suppression_postprocess_volume_voxels": volume,
                "negative_false_positive_before_suppression": should_suppress,
                "negative_safe_suppression_applied": bool(enabled and should_suppress),
                "suppression_reason": "confirmed_negative_absent_out_of_fov" if enabled and should_suppress else "",
                **{field: mrow.get(field) for field in NEGATIVE_PROVENANCE_FIELDS},
            })
    summary = {
        "stage": "negative_safe_postprocess",
        "status": "success",
        "enabled": enabled,
        "eligible_confirmed_negative_absent_rows": eligible,
        "pre_suppression_nonempty_negative_count": nonempty,
        "suppressed_negative_candidate_count": suppressed,
        "negative_false_positive_voxel_threshold": threshold,
        "policy": "Confirmed negative_absent/out-of-FOV student candidates are forced empty for formal Round2 safety; raw evidence remains in diagnosis artifacts.",
    }
    return rows, summary


def binary_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if pred.shape != ref.shape:
        return {
            "status": "shape_mismatch",
            "pseudo_consistency_dsc": None,
            "precision": None,
            "recall": None,
            "intersection_voxels": None,
            "student_volume_voxels": int(pred.sum()),
            "pseudo_label_volume_voxels": int(ref.sum()),
            "volume_ratio": None,
        }
    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())
    inter = int(np.logical_and(pred, ref).sum())
    total = pred_sum + ref_sum
    dsc = 1.0 if total == 0 else (2.0 * inter / total)
    precision = 1.0 if pred_sum == 0 else (inter / pred_sum)
    recall = 1.0 if ref_sum == 0 else (inter / ref_sum)
    volume_ratio = None if ref_sum == 0 else (pred_sum / ref_sum)
    return {
        "status": "measured",
        "pseudo_consistency_dsc": round(float(dsc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "intersection_voxels": inter,
        "student_volume_voxels": pred_sum,
        "pseudo_label_volume_voxels": ref_sum,
        "volume_ratio": round(float(volume_ratio), 6) if volume_ratio is not None else None,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _volume_bucket(volume: int | float | None) -> str:
    if volume is None:
        return "missing_volume"
    volume = float(volume)
    if volume <= 0:
        return "empty"
    if volume <= 10:
        return "tiny_noise_1_10_voxels"
    if volume <= 100:
        return "small_component_11_100_voxels"
    if volume <= 1000:
        return "moderate_component_101_1000_voxels"
    if volume <= 10000:
        return "large_component_1001_10000_voxels"
    return "large_anatomic_false_positive_gt_10000_voxels"


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None


def _diagnosis_reason(metric_row: dict[str, Any], post_row: dict[str, Any], pre_volume: float | None) -> str:
    organ = str(metric_row.get("organ") or "")
    target_type = str(metric_row.get("target_type") or "")
    fov_status = str(metric_row.get("fov_status") or "")
    if organ in {"pancreas_head", "pancreas_body", "pancreas_tail"} and target_type == "negative_absent":
        return "possible_bad_negative_label_abdominal_suborgan"
    status = str(post_row.get("status") or "")
    if status == "copied_no_containment_rule":
        return "postprocess_no_containment_rule_retained_mask"
    if status in {"warning_empty_parent_roi_copied_raw", "warning_missing_parent_roi_copied_raw"}:
        return "postprocess_parent_roi_unavailable_retained_mask"
    if pre_volume is not None and pre_volume > 10000:
        return "large_anatomic_false_positive"
    if fov_status == "out_of_fov":
        return "student_nonempty_on_confirmed_out_of_fov_negative"
    return "negative_false_positive"


def selected_reference_path(selected_root: Path, case_id: str, organ: str, manifest_row: dict[str, Any] | None) -> Path | None:
    if manifest_row:
        for key in ("mask", "mask_path", "final_mask"):
            value = manifest_row.get(key)
            if _path_exists(value):
                return Path(str(value))
    candidate = selected_root / case_id / "updated" / f"{organ}.nii.gz"
    return candidate if candidate.exists() else None


def compute_metrics(
    *,
    cases: list[dict[str, str]],
    targets: list[str],
    manifest: dict[tuple[str, str], dict[str, Any]],
    selected_root: Path,
    post_root: Path,
    shapekit_rows: list[dict[str, Any]],
    suppression_rows: list[dict[str, Any]],
    negative_fp_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    shapekit_status = {
        (str(row.get("case_id")), str(row.get("organ"))): row
        for row in shapekit_rows
    }
    suppression_status = {
        (str(row.get("case_id")), str(row.get("organ"))): row
        for row in suppression_rows
    }
    metric_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    organ_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        case_id = case["case_id"]
        for organ in targets:
            mrow = manifest.get((case_id, organ), {})
            supervision_type = str(mrow.get("supervision_type") or "withheld_uncertain")
            target_type = str(mrow.get("target_type") or "withheld_uncertain")
            training_weight = float(mrow.get("training_weight") or 0.0)
            ref_path = selected_reference_path(selected_root, case_id, organ, mrow)
            student_path = post_root / case_id / f"{organ}.nii.gz"
            pred, pred_err = load_mask(student_path)
            ref, ref_err = load_mask(ref_path)
            sk = shapekit_status.get((case_id, organ), {})
            row: dict[str, Any] = {
                "case_id": case_id,
                "organ": organ,
                "supervision_type": supervision_type,
                "target_type": target_type,
                "training_weight": training_weight,
                "student_mask": str(student_path),
                "selected_pseudo_label": str(ref_path) if ref_path else "",
                "student_shapekit_status": sk.get("student_shapekit_status", "missing"),
                "postprocess_status": "present" if student_path.exists() else "missing",
                "metric_family": "pseudo_consistency",
                "metric_scope": "student_trainset_vs_selected_pseudo_label",
                "metric_target": "selected_pseudo_label",
                "metric_subject": "student",
                "metric_interpretation": "pseudo_label_consistency",
                "label_role": "student_pseudo_candidate",
                "supervision_role": "selected_pseudo_label",
                "replacement_eligible": False,
            }
            row.update({field: mrow.get(field) for field in NEGATIVE_PROVENANCE_FIELDS})
            srow = suppression_status.get((case_id, organ), {})
            row.update({
                "negative_safe_suppression_applied": bool(srow.get("negative_safe_suppression_applied")),
                "suppression_reason": srow.get("suppression_reason") or "",
                "pre_suppression_postprocess_volume_voxels": srow.get("pre_suppression_postprocess_volume_voxels"),
                "negative_false_positive_before_suppression": bool(srow.get("negative_false_positive_before_suppression")),
            })
            if pred is None or ref is None:
                row.update({
                    "metric_status": "missing_student_prediction" if pred is None else "missing_selected_pseudo_label",
                    "missing_reason": pred_err or ref_err,
                    "pseudo_consistency_dsc": None,
                    "precision": None,
                    "recall": None,
                    "student_volume_voxels": None,
                    "pseudo_label_volume_voxels": None,
                    "volume_ratio": None,
                    "empty_student_prediction": pred is not None and int(pred.sum()) == 0,
                    "empty_selected_pseudo_label": ref is not None and int(ref.sum()) == 0,
                    "oversegmentation_flag": False,
                    "negative_false_positive": False,
                })
            else:
                metrics = binary_metrics(pred, ref)
                row.update(metrics)
                pred_sum = int(metrics.get("student_volume_voxels") or 0)
                ref_sum = int(metrics.get("pseudo_label_volume_voxels") or 0)
                recall = metrics.get("recall")
                precision = metrics.get("precision")
                volume_ratio = metrics.get("volume_ratio")
                overseg = bool(
                    recall is not None
                    and precision is not None
                    and volume_ratio is not None
                    and float(recall) >= 0.80
                    and float(precision) < 0.50
                    and float(volume_ratio) > 1.5
                )
                negative_fp = bool(
                    supervision_type == "negative"
                    and target_type in {"negative_absent", "absent_negative"}
                    and pred_sum > negative_fp_threshold
                )
                row.update({
                    "empty_student_prediction": pred_sum == 0,
                    "empty_selected_pseudo_label": ref_sum == 0,
                    "oversegmentation_flag": overseg,
                    "negative_false_positive": negative_fp,
                    "metric_status": metrics["status"],
                })
                row["replacement_eligible"] = bool(
                    sk.get("student_shapekit_status") in SAFE_SHAPEKIT_STATUSES
                    and row["metric_status"] == "measured"
                    and not overseg
                    and supervision_type == "positive"
                )
            metric_rows.append(row)
            if supervision_type == "positive" and training_weight > 0:
                positive_rows.append(row)
                organ_groups[organ].append(row)
            elif supervision_type == "negative" and target_type in {"negative_absent", "absent_negative"}:
                negative_rows.append(row)

    measured_positive = [r for r in positive_rows if r.get("pseudo_consistency_dsc") is not None]
    overseg_positive = [r for r in measured_positive if r.get("oversegmentation_flag")]
    negative_fp_rows = [r for r in negative_rows if r.get("negative_false_positive")]
    negative_pre_suppression_fp_rows = [
        r for r in negative_rows if r.get("negative_false_positive_before_suppression")
    ]
    mean_positive_dsc = (
        float(np.mean([float(r["pseudo_consistency_dsc"]) for r in measured_positive]))
        if measured_positive else None
    )
    overseg_rate = float(len(overseg_positive) / len(measured_positive)) if measured_positive else None
    key_ratios = [
        float(r["volume_ratio"])
        for r in measured_positive
        if r["organ"] in KEY_ORGANS and r.get("volume_ratio") is not None
    ]
    key_median_ratio = float(np.median(key_ratios)) if key_ratios else None
    organ_rows = []
    for organ, rows in sorted(organ_groups.items()):
        measured = [r for r in rows if r.get("pseudo_consistency_dsc") is not None]
        organ_rows.append({
            "organ": organ,
            "n": len(rows),
            "measured": len(measured),
            "mean_pseudo_consistency_dsc": round(float(np.mean([float(r["pseudo_consistency_dsc"]) for r in measured])), 6) if measured else None,
            "mean_precision": round(float(np.mean([float(r["precision"]) for r in measured if r.get("precision") is not None])), 6) if measured else None,
            "mean_recall": round(float(np.mean([float(r["recall"]) for r in measured if r.get("recall") is not None])), 6) if measured else None,
            "median_volume_ratio": round(float(np.median([float(r["volume_ratio"]) for r in measured if r.get("volume_ratio") is not None])), 6) if any(r.get("volume_ratio") is not None for r in measured) else None,
            "oversegmentation_count": sum(1 for r in measured if r.get("oversegmentation_flag")),
            "empty_prediction_count": sum(1 for r in measured if r.get("empty_student_prediction")),
            "shapekit_success_count": sum(1 for r in measured if r.get("student_shapekit_status") == "success"),
        })
    summary = {
        "stage": "student_trainset_pseudo_consistency_summary",
        "metric_target": "selected_pseudo_label",
        "metric_interpretation": "pseudo_label_consistency",
        "total_rows": len(metric_rows),
        "positive_trainable_rows": len(positive_rows),
        "positive_measured_rows": len(measured_positive),
        "negative_absent_rows": len(negative_rows),
        "negative_false_positive_count": len(negative_fp_rows),
        "negative_false_positive_before_suppression_count": len(negative_pre_suppression_fp_rows),
        "negative_safe_suppressed_count": sum(1 for r in negative_rows if r.get("negative_safe_suppression_applied")),
        "mean_positive_pseudo_consistency_dsc": round(mean_positive_dsc, 6) if mean_positive_dsc is not None else None,
        "oversegmentation_positive_count": len(overseg_positive),
        "oversegmentation_positive_rate": round(overseg_rate, 6) if overseg_rate is not None else None,
        "key_organ_median_volume_ratio": round(key_median_ratio, 6) if key_median_ratio is not None else None,
        "organ_summary": organ_rows,
    }
    overseg_audit = {
        "stage": "student_oversegmentation_audit",
        "status": "passed" if not overseg_positive else "failed",
        "rule": "recall >= 0.80 and precision < 0.50 and volume_ratio > 1.5",
        "oversegmentation_positive_count": len(overseg_positive),
        "oversegmentation_positive_rate": summary["oversegmentation_positive_rate"],
        "examples": overseg_positive[:50],
        "organ_counts": dict(Counter(str(r["organ"]) for r in overseg_positive)),
    }
    return metric_rows, summary, overseg_audit, {
        "organ_rows": organ_rows,
        "negative_false_positive_rows": negative_fp_rows,
        "negative_pre_suppression_false_positive_rows": negative_pre_suppression_fp_rows,
    }


def build_negative_false_positive_diagnosis(
    *,
    metric_rows: list[dict[str, Any]],
    raw_root: Path,
    shapekit_rows: list[dict[str, Any]],
    postprocess_rows: list[dict[str, Any]],
    suppression_rows: list[dict[str, Any]],
    threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shapekit_by_key = {(str(r.get("case_id")), str(r.get("organ"))): r for r in shapekit_rows}
    post_by_key = {
        (str(r.get("case_id")), str(r.get("canonical_organ") or r.get("organ"))): r
        for r in postprocess_rows
    }
    suppression_by_key = {(str(r.get("case_id")), str(r.get("organ"))): r for r in suppression_rows}
    diagnosis_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        if str(row.get("target_type") or "") not in {"negative_absent", "absent_negative"}:
            continue
        if str(row.get("supervision_type") or "") != "negative":
            continue
        key = (str(row.get("case_id")), str(row.get("organ")))
        srow = suppression_by_key.get(key, {})
        post_row = post_by_key.get(key, {})
        pre_volume = _float_or_none(srow.get("pre_suppression_postprocess_volume_voxels"))
        post_volume = _float_or_none(row.get("student_volume_voxels"))
        was_fp = bool(row.get("negative_false_positive")) or bool(srow.get("negative_false_positive_before_suppression"))
        if not was_fp:
            continue
        sk = shapekit_by_key.get(key, {})
        raw_volume = mask_volume(raw_root / key[0] / f"{key[1]}.nii.gz")
        shapekit_path = Path(str(sk.get("output_path") or "")) if sk.get("output_path") else None
        diagnosis_rows.append({
            "case_id": key[0],
            "organ": key[1],
            "target_type": row.get("target_type"),
            "supervision_type": row.get("supervision_type"),
            "raw_volume_voxels": raw_volume,
            "shapekit_volume_voxels": mask_volume(shapekit_path),
            "pre_suppression_postprocess_volume_voxels": pre_volume,
            "post_suppression_student_volume_voxels": post_volume,
            "negative_false_positive": bool(row.get("negative_false_positive")),
            "negative_false_positive_before_suppression": bool(srow.get("negative_false_positive_before_suppression")),
            "negative_safe_suppression_applied": bool(row.get("negative_safe_suppression_applied")),
            "suppression_reason": row.get("suppression_reason") or "",
            "volume_bucket": _volume_bucket(pre_volume if pre_volume is not None else post_volume),
            "diagnosis_reason": _diagnosis_reason(row, post_row, pre_volume if pre_volume is not None else post_volume),
            "student_shapekit_status": row.get("student_shapekit_status"),
            "postprocess_status": post_row.get("status") or row.get("postprocess_status"),
            "containment_enabled": post_row.get("containment_enabled"),
            "containment_source": post_row.get("containment_source"),
            "roi_status": post_row.get("roi_status"),
            "organ_group": post_row.get("organ_group"),
            "hierarchy_role": post_row.get("hierarchy_role"),
            "voxels_before_postprocess": post_row.get("voxels_before"),
            "voxels_after_postprocess": post_row.get("voxels_after"),
            "false_positive_voxels_removed": post_row.get("false_positive_voxels_removed"),
            **{field: row.get(field) for field in NEGATIVE_PROVENANCE_FIELDS},
        })
    volumes = [
        float(r["pre_suppression_postprocess_volume_voxels"])
        for r in diagnosis_rows
        if r.get("pre_suppression_postprocess_volume_voxels") is not None
    ]
    sorted_volumes = sorted(volumes)
    quantiles: dict[str, float] = {}
    for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        if sorted_volumes:
            quantiles[str(q)] = sorted_volumes[min(len(sorted_volumes) - 1, int(q * (len(sorted_volumes) - 1)))]
    summary = {
        "stage": "negative_absent_false_positive_diagnosis",
        "status": "passed",
        "metric_target": "selected_pseudo_label",
        "negative_false_positive_voxel_threshold": threshold,
        "rows": len(diagnosis_rows),
        "post_suppression_negative_false_positive_count": sum(1 for r in diagnosis_rows if r["negative_false_positive"]),
        "pre_suppression_negative_false_positive_count": sum(1 for r in diagnosis_rows if r["negative_false_positive_before_suppression"]),
        "suppressed_negative_candidate_count": sum(1 for r in diagnosis_rows if r["negative_safe_suppression_applied"]),
        "volume_quantiles": quantiles,
        "top_organs": Counter(str(r["organ"]) for r in diagnosis_rows).most_common(30),
        "top_cases": Counter(str(r["case_id"]) for r in diagnosis_rows).most_common(30),
        "diagnosis_reason_counts": Counter(str(r["diagnosis_reason"]) for r in diagnosis_rows).most_common(),
        "volume_bucket_counts": Counter(str(r["volume_bucket"]) for r in diagnosis_rows).most_common(),
        "postprocess_status_counts": Counter(str(r.get("postprocess_status") or "") for r in diagnosis_rows).most_common(),
        "fov_status_counts": Counter(str(r.get("fov_status") or "") for r in diagnosis_rows).most_common(),
        "negative_source_counts": Counter(str(r.get("negative_source") or "") for r in diagnosis_rows).most_common(),
        "notes": [
            "Counts describe pseudo-label consistency and confirmed-negative safety, not external-reference accuracy.",
            "Pre-suppression rows preserve raw audit evidence; post-suppression rows determine formal candidate safety.",
        ],
    }
    return diagnosis_rows, summary


def build_round2_gate(
    *,
    preflight: dict[str, Any],
    consistency_summary: dict[str, Any],
    overseg_audit: dict[str, Any],
    shapekit_summary: dict[str, Any],
    postprocess_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reasons: list[str] = []
    mean_dsc = consistency_summary.get("mean_positive_pseudo_consistency_dsc")
    overseg_rate = consistency_summary.get("oversegmentation_positive_rate")
    key_ratio = consistency_summary.get("key_organ_median_volume_ratio")
    if preflight.get("status") != "passed":
        reasons.append("preflight_failed")
    if shapekit_summary.get("status") != "success":
        reasons.append("student_shapekit_not_fully_successful")
    if postprocess_summary.get("status") != "success":
        reasons.append("student_postprocess_failed")
    if mean_dsc is None or float(mean_dsc) < args.positive_min_mean_dsc:
        reasons.append("positive_pseudo_consistency_below_threshold")
    if overseg_rate is None or float(overseg_rate) > args.max_oversegmentation_rate:
        reasons.append("systematic_oversegmentation")
    if key_ratio is not None and float(key_ratio) > args.max_key_organ_median_volume_ratio:
        reasons.append("key_organ_volume_ratio_too_high")
    if consistency_summary.get("negative_false_positive_count", 0) > 0:
        reasons.append("negative_absent_false_positive_detected")
    if overseg_audit.get("status") != "passed":
        reasons.append("oversegmentation_audit_failed")
    return {
        "stage": "round2_progression_gate_after_trainset_consistency",
        "status": "passed" if not reasons else "blocked",
        "round2_progression_allowed": not reasons,
        "block_reasons": reasons,
        "required_external_gate": "LabelCritic known-better benchmark must pass separately before formal Round2 replacement.",
        "metric_target": "selected_pseudo_label",
        "thresholds": {
            "positive_min_mean_dsc": args.positive_min_mean_dsc,
            "max_oversegmentation_rate": args.max_oversegmentation_rate,
            "max_key_organ_median_volume_ratio": args.max_key_organ_median_volume_ratio,
            "negative_false_positive_voxel_threshold": args.negative_false_positive_voxel_threshold,
        },
        "summary": {
            "mean_positive_pseudo_consistency_dsc": mean_dsc,
            "oversegmentation_positive_rate": overseg_rate,
            "key_organ_median_volume_ratio": key_ratio,
            "negative_false_positive_count": consistency_summary.get("negative_false_positive_count"),
            "negative_false_positive_before_suppression_count": consistency_summary.get("negative_false_positive_before_suppression_count"),
            "negative_safe_suppressed_count": consistency_summary.get("negative_safe_suppressed_count"),
            "shapekit_status": shapekit_summary.get("status"),
            "postprocess_status": postprocess_summary.get("status"),
        },
        "negative_absent_policy": {
            "formal_candidate_rule": "negative_absent/out-of-FOV student outputs are audit evidence only and are not eligible for formal replacement.",
            "forced_empty_policy": "confirmed negative_absent/out-of-FOV masks may be forced empty after audit capture to keep Round2 candidates fail-closed.",
            "threshold_interpretation": "threshold applies to post-suppression formal candidate volume; pre-suppression evidence is reported separately.",
        },
    }


def scan_output_for_forbidden_tokens(output_dir: Path) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".txt", ".log"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_OUTPUT_TOKENS:
            if token in text or token in path.name:
                hits.append({"path": str(path), "token": token})
    return {
        "stage": "forbidden_output_token_audit",
        "status": "passed" if not hits else "failed",
        "hits": hits[:50],
    }


def main() -> int:
    args = parse_args()
    start = time.time()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.case_list.resolve(), args.max_cases)
    case_list_used = args.output_dir / "case_list_used.csv"
    write_case_list(case_list_used, cases)
    targets = load_target_organs(args.target_config.resolve())
    manifest, manifest_doc = manifest_index(args.manifest.resolve())

    pre = preflight(args, cases, targets, manifest_doc)
    write_json(args.output_dir / "student_prediction_preflight.json", pre)
    if pre["status"] != "passed":
        gate = {
            "stage": "round2_progression_gate_after_trainset_consistency",
            "status": "blocked",
            "round2_progression_allowed": False,
            "block_reasons": pre["failure_reasons"],
        }
        write_json(args.output_dir / "round2_progression_gate_after_trainset_consistency.json", gate)
        print(json.dumps(gate, indent=2, ensure_ascii=False))
        return 2

    raw_root = args.reuse_raw_root.resolve() if args.reuse_raw_root else args.output_dir / "student_predictions_raw"
    if args.reuse_raw_root:
        raw_rows, raw_summary = collect_existing_raw_predictions(raw_root, cases, targets)
    else:
        raw_rows, raw_summary = run_student_inference(args, cases, targets, raw_root)
    write_json(args.output_dir / "student_prediction_raw_summary.json", raw_summary)

    shapekit_root = args.reuse_shapekit_root.resolve() if args.reuse_shapekit_root else args.output_dir / "student_predictions_shapekit"
    if args.reuse_shapekit_root:
        shapekit_rows, shapekit_summary = collect_existing_shapekit_predictions(raw_root, shapekit_root, cases, targets)
    else:
        shapekit_rows, shapekit_summary = run_student_shapekit(args, raw_root, shapekit_root, cases, targets)
    if not args.reuse_shapekit_root:
        write_json(shapekit_root / "student_shapekit_summary.json", shapekit_summary)
    write_json(args.output_dir / "student_shapekit_summary.json", shapekit_summary)
    write_csv(args.output_dir / "student_shapekit_status.csv", shapekit_rows)

    post_root = args.output_dir / "student_predictions_postprocessed"
    post_summary = run_postprocess(args, case_list_used, shapekit_root, post_root)
    suppression_rows, suppression_summary = apply_negative_safe_postprocess(
        manifest=manifest,
        post_root=post_root,
        cases=cases,
        targets=targets,
        threshold=args.negative_false_positive_voxel_threshold,
        enabled=not args.disable_negative_safe_postprocess,
    )
    write_csv(args.output_dir / "negative_safe_postprocess_suppression.csv", suppression_rows)
    write_json(args.output_dir / "negative_safe_postprocess_summary.json", suppression_summary)
    postprocess_rows = read_csv_rows(post_root / "student_containment_postprocess_per_mask.csv")

    metric_rows, consistency_summary, overseg_audit, extra = compute_metrics(
        cases=cases,
        targets=targets,
        manifest=manifest,
        selected_root=args.selected_pseudo_root.resolve(),
        post_root=post_root,
        shapekit_rows=shapekit_rows,
        suppression_rows=suppression_rows,
        negative_fp_threshold=args.negative_false_positive_voxel_threshold,
    )
    diagnosis_rows, diagnosis_summary = build_negative_false_positive_diagnosis(
        metric_rows=metric_rows,
        raw_root=raw_root,
        shapekit_rows=shapekit_rows,
        postprocess_rows=postprocess_rows,
        suppression_rows=suppression_rows,
        threshold=args.negative_false_positive_voxel_threshold,
    )
    metric_fields = [
        "case_id", "organ", "supervision_type", "target_type",
        "training_weight", "fov_status", "fov_evidence", "coverage_evidence",
        "negative_source", "zero_mask_role", "negative_reason", "absence_confidence",
        "student_shapekit_status", "postprocess_status",
        "pseudo_consistency_dsc", "precision", "recall",
        "student_volume_voxels", "pseudo_label_volume_voxels", "volume_ratio",
        "pre_suppression_postprocess_volume_voxels",
        "negative_false_positive_before_suppression",
        "negative_safe_suppression_applied", "suppression_reason",
        "empty_student_prediction", "empty_selected_pseudo_label",
        "oversegmentation_flag", "negative_false_positive", "replacement_eligible",
        "metric_status", "metric_target", "metric_family", "metric_scope",
        "metric_subject", "metric_interpretation", "student_mask", "selected_pseudo_label",
    ]
    write_csv(args.output_dir / "student_trainset_pseudo_consistency_full_mstep.csv", metric_rows, metric_fields)
    write_json(args.output_dir / "student_trainset_pseudo_consistency_summary.json", consistency_summary)
    write_json(args.output_dir / "student_oversegmentation_audit.json", overseg_audit)
    write_csv(args.output_dir / "student_organ_pseudo_consistency_summary.csv", extra["organ_rows"])
    write_csv(args.output_dir / "negative_absent_false_positive_rows.csv", extra["negative_false_positive_rows"])
    write_csv(args.output_dir / "negative_absent_pre_suppression_false_positive_rows.csv", extra["negative_pre_suppression_false_positive_rows"])
    write_csv(args.output_dir / "negative_absent_false_positive_diagnosis.csv", diagnosis_rows)
    write_json(args.output_dir / "negative_absent_false_positive_diagnosis.json", diagnosis_summary)

    prediction_manifest = {
        "stage": "student_prediction_manifest",
        "status": "success" if len(raw_rows) == len(cases) * len(targets) else "partial_success",
        "case_count": len(cases),
        "target_count": len(targets),
        "prediction_rows": len(raw_rows),
        "raw_root": str(raw_root),
        "shapekit_root": str(shapekit_root),
        "postprocessed_root": str(post_root),
        "metric_target": "selected_pseudo_label",
        "rows_csv": str(args.output_dir / "student_prediction_manifest.csv"),
    }
    manifest_rows = []
    shapekit_by_key = {(r["case_id"], r["organ"]): r for r in shapekit_rows}
    for row in raw_rows:
        sk = shapekit_by_key.get((row["case_id"], row["organ"]), {})
        manifest_rows.append({**row, **{
            "shapekit_mask": sk.get("output_path", ""),
            "postprocessed_mask": str(post_root / row["case_id"] / f"{row['organ']}.nii.gz"),
            "student_shapekit_status": sk.get("student_shapekit_status", "missing"),
        }})
    write_csv(args.output_dir / "student_prediction_manifest.csv", manifest_rows)
    write_json(args.output_dir / "student_prediction_manifest.json", prediction_manifest)

    gate = build_round2_gate(
        preflight=pre,
        consistency_summary=consistency_summary,
        overseg_audit=overseg_audit,
        shapekit_summary=shapekit_summary,
        postprocess_summary=post_summary,
        args=args,
    )
    gate["runtime_sec"] = round(time.time() - start, 3)
    write_json(args.output_dir / "round2_progression_gate_after_trainset_consistency.json", gate)
    token_audit = scan_output_for_forbidden_tokens(args.output_dir)
    write_json(args.output_dir / "forbidden_output_token_audit.json", token_audit)
    if token_audit["status"] != "passed":
        gate["status"] = "blocked"
        gate["round2_progression_allowed"] = False
        gate.setdefault("block_reasons", []).append("forbidden_output_token_detected")
        write_json(args.output_dir / "round2_progression_gate_after_trainset_consistency.json", gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0 if pre["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
