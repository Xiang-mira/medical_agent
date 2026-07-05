from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .auto_fine_label import grade_to_training_weight
from .auto_label_core import ACCEPTED_SCORING_SCHEMA_VERSIONS
from .continual_learning import (
    TRAINING_CONTRACT_VERSION,
    TrainingContractError,
    canonical_target_type,
    canonicalize_training_record,
)
from .json_utils import write_json
from .organ_taxonomy import normalize_canonical_id
from .subprocess_utils import subprocess_text


def _quality_status(review_flags: Any, quality_flags: Any) -> str:
    flags = set(review_flags or []) | set(quality_flags or [])
    if not flags:
        return "ok"
    if {"missing_candidate", "missing_final_mask"} & flags:
        return "missing"
    if any(str(flag).startswith("shapekit_") for flag in flags):
        return "postprocess_review"
    if "selection_fallback" in flags:
        return "selection_review"
    return "review"


def _labelcritic_decision_path(records: Any) -> str | None:
    for record in records or []:
        if isinstance(record, dict) and record.get("output_json"):
            return str(record["output_json"])
    return None


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _load_case_selection_index(root: Path, case_id: str) -> dict[str, dict[str, Any]]:
    """Load per-organ pseudo-label source metadata for one case when available."""
    candidates = [
        root / case_id / "selection_metadata.json",
        root.parent / "cases" / case_id / "pseudo_label_selection.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = doc.get("selected_organs") or []
        if not isinstance(rows, list):
            continue
        return {
            str(item.get("organ")): item
            for item in rows
            if isinstance(item, dict) and item.get("organ")
        }
    return {}


def build_training_manifest(updated_annotations_root: str | Path, output_manifest: str | Path, organs: list[str] | None = None) -> dict[str, Any]:
    root = Path(updated_annotations_root).resolve()
    out = Path(output_manifest).resolve()
    rows: list[dict[str, Any]] = []
    excluded_identity_rows: list[dict[str, Any]] = []
    excluded_training_rows: list[dict[str, Any]] = []
    if root.exists():
        for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            seg_dir = case_dir / "updated"
            if not seg_dir.exists() or not any(seg_dir.glob("*.nii.gz")):
                seg_dir = case_dir / "segmentations"
            selection_index = _load_case_selection_index(root, case_dir.name)
            for mask in sorted(seg_dir.glob("*.nii.gz")):
                organ = mask.name[:-7]
                if organs and organ not in organs:
                    continue
                meta = selection_index.get(organ, {})
                identity_status = str(meta.get("identity_status") or "legacy_unverified")
                requested_canonical_id = normalize_canonical_id(meta.get("requested_canonical_id"))
                resolved_canonical_id = normalize_canonical_id(meta.get("resolved_canonical_id"))
                organ_canonical_id = normalize_canonical_id(organ)
                if identity_status == "valid" and (
                    requested_canonical_id != organ_canonical_id or resolved_canonical_id != organ_canonical_id
                ):
                    identity_status = "identity_mismatch"
                    meta = {
                        **meta,
                        "identity_mismatch_reasons": [
                            *(meta.get("identity_mismatch_reasons", []) or []),
                            "manifest_organ_canonical_id_mismatch",
                        ],
                    }
                if identity_status != "valid":
                    excluded_identity_rows.append({
                        "case_id": case_dir.name,
                        "organ": organ,
                        "mask_path": str(mask.resolve()),
                        "identity_status": identity_status,
                        "identity_mismatch_reasons": meta.get("identity_mismatch_reasons", ["missing_identity_provenance"]),
                    })
                    continue
                grade_value = str(meta.get("grade") or "C").upper()
                training_weight_value = float(meta.get("training_weight", grade_to_training_weight(grade_value)) or 0.0)
                distillation_eligible_value = meta.get("distillation_eligible", training_weight_value > 0.0)
                scoring_schema_version = str(meta.get("scoring_schema_version") or "legacy")
                schema_supported = scoring_schema_version in ACCEPTED_SCORING_SCHEMA_VERSIONS
                if not schema_supported:
                    training_weight_value = 0.0
                    distillation_eligible_value = False
                try:
                    target_type_value = canonical_target_type(meta.get("target_type", "hard"))
                except TrainingContractError:
                    target_type_value = str(meta.get("target_type", "hard")).lower()
                probability_path_value = meta.get("probability_mask_path")
                probability_path_exists = bool(probability_path_value and Path(str(probability_path_value)).exists())
                if not schema_supported:
                    training_gate_decision = "exclude_unsupported_schema"
                    training_gate_policy = "Unsupported scoring schemas require rescoring before student training."
                elif grade_value in {"A", "B"} and target_type_value == "positive_hard" and training_weight_value > 0.0:
                    training_gate_decision = "include_hard_ab"
                    training_gate_policy = "A/B hard pseudo-labels are eligible for direct student training."
                elif grade_value in {"A", "B"}:
                    training_gate_decision = "exclude_ab_non_hard_or_zero_weight"
                    training_gate_policy = "A/B labels must be hard targets with positive training weight for direct student training."
                elif grade_value == "C" and target_type_value == "positive_soft" and training_weight_value > 0.0 and probability_path_exists:
                    training_gate_decision = "include_soft_c"
                    training_gate_policy = "C is eligible only as a soft target with an explicit probability mask."
                elif grade_value == "C" and target_type_value == "positive_soft":
                    training_gate_decision = "exclude_c_soft_missing_probability"
                    training_gate_policy = "C soft labels require an explicit probability mask before student training."
                elif grade_value == "C":
                    training_gate_decision = "exclude_or_review_c"
                    training_gate_policy = "C hard/provisional labels are audit/review only unless a soft target is present."
                else:
                    training_gate_decision = "exclude_d_or_zero_weight"
                    training_gate_policy = "D and zero-weight labels are excluded from student training."

                row: dict[str, Any] = {
                    "case_id": case_dir.name,
                    "dataset_type": "auto_fine_label_dataset",
                    "organ": organ,
                    "prompt": meta.get("prompt", organ.replace("_", " ")),
                    "mask_path": str(
                        Path(probability_path_value).resolve()
                        if target_type_value == "positive_soft" and probability_path_value
                        else mask.resolve()
                    ),
                    "mask": str(
                        Path(probability_path_value).resolve()
                        if target_type_value == "positive_soft" and probability_path_value
                        else mask.resolve()
                    ),
                    "ct_path": meta.get("ct_path", ""),
                    "image": meta.get("ct_path", ""),
                    "selected_model": meta.get("selected_model"),
                    "source_model": meta.get("source_model", meta.get("selected_model")),
                    "candidate_models": meta.get("candidate_models", []),
                    "candidate_count": meta.get("candidate_count"),
                    "comparison_candidate_models": meta.get("comparison_candidate_models", []),
                    "comparison_candidate_count": meta.get("comparison_candidate_count"),
                    "qc_rejected_candidates": meta.get("qc_rejected_candidates", []),
                    "candidate_qc_policy": meta.get("candidate_qc_policy"),
                    "selection_method": meta.get("selection_method"),
                    "selection_status": meta.get("selection_status"),
                    "comparison_input_stage": meta.get("comparison_input_stage"),
                    "fallback_reason": meta.get("fallback_reason"),
                    "selected_dice": meta.get("selected_dice"),
                    "selected_pseudo_consistency_dice": meta.get("selected_pseudo_consistency_dice", meta.get("selected_dice")),
                    "metric_family": meta.get("metric_family", "pseudo_consistency"),
                    "metric_scope": meta.get("metric_scope", "selected_pseudo_label_for_student_training"),
                    "accuracy_warning": meta.get("accuracy_warning", "Pseudo labels are not expert ground truth; do not report true accuracy from this manifest."),
                    "selected_candidate_qc_status": meta.get("selected_candidate_qc_status"),
                    "selected_candidate_qc_score": meta.get("selected_candidate_qc_score"),
                    "selected_candidate_qc_flags": meta.get("selected_candidate_qc_flags", []),
                    "selected_reference_quality_bucket": meta.get("selected_reference_quality_bucket"),
                    "labelcritic_records": meta.get("labelcritic_records", meta.get("critic_records", [])),
                    "labelcritic_decision_path": meta.get("labelcritic_decision_path") or _labelcritic_decision_path(meta.get("labelcritic_records", meta.get("critic_records", []))),
                    "label_critic_decision_path": meta.get("label_critic_decision_path") or meta.get("labelcritic_decision_path") or _labelcritic_decision_path(meta.get("labelcritic_records", meta.get("critic_records", []))),
                    "shapekit_status": meta.get("shapekit_status"),
                    "shapekit_reason": meta.get("shapekit_reason"),
                    "dataset_role": meta.get("dataset_role", "pseudo_label"),
                    "ground_truth_status": meta.get("ground_truth_status", "machine_generated_candidate"),
                    "label_maturity_level": meta.get("label_maturity_level"),
                    "auto_fine_label_status": meta.get("auto_fine_label_status", "auto_fine_label_candidate" if meta else "machine_label_candidate"),
                    "auto_fine_label_reliability_score": meta.get("auto_fine_label_reliability_score"),
                    "estimated_reliability": meta.get("estimated_reliability", meta.get("evidence_confidence")),
                    "evidence_confidence": meta.get("evidence_confidence"),
                    "evidence_scores": meta.get("evidence_scores", {}),
                    "missing_evidence": meta.get("missing_evidence", []),
                    "decision_status": meta.get("decision_status"),
                    "decision_reasons": meta.get("decision_reasons", []),
                    "target_type": target_type_value,
                    "training_gate_decision": training_gate_decision,
                    "training_gate_policy": training_gate_policy,
                    "probability_mask_path": probability_path_value,
                    "voxel_uncertainty_path": meta.get("voxel_uncertainty_path"),
                    "independent_family_count": meta.get("independent_family_count", 0),
                    "family_membership": meta.get("family_membership", {}),
                    "scoring_schema_version": scoring_schema_version,
                    "grade": grade_value,
                    "training_weight": training_weight_value,
                    "distillation_eligible": distillation_eligible_value,
                    "distillation_exclusion_reason": meta.get("distillation_exclusion_reason"),
                    "label_passport_path": meta.get("label_passport_path"),
                    "review_flags": meta.get("review_flags", []),
                    "quality_flags": meta.get("quality_flags", []),
                    "quality_status": meta.get("quality_status") or _quality_status(meta.get("review_flags", []), meta.get("quality_flags", [])),
                    "source_metadata_available": bool(meta),
                    "requested_canonical_id": meta.get("requested_canonical_id"),
                    "source_local_label": meta.get("source_local_label"),
                    "resolved_canonical_id": meta.get("resolved_canonical_id"),
                    "comparison_family": meta.get("comparison_family"),
                    "parent_ids": meta.get("parent_ids", []),
                    "mapping_type": meta.get("mapping_type"),
                    "mapping_source": meta.get("mapping_source"),
                    "identity_status": identity_status,
                }
                row = canonicalize_training_record(
                    row,
                    project_root=Path(__file__).resolve().parents[4],
                    strict_soft=True,
                )
                grade = str(row.get("grade") or "D").upper()
                training_weight = float(row.get("training_weight") or 0.0)
                include_gate = row.get("training_gate_decision") in {"include_hard_ab", "include_soft_c"}
                if (
                    (not include_gate)
                    or grade == "D"
                    or training_weight <= 0.0
                    or row.get("distillation_eligible") is False
                    or not row.get("training_eligible")
                ):
                    exclusion_reason = row.get("distillation_exclusion_reason")
                    if not exclusion_reason:
                        if row.get("scoring_schema_version") not in ACCEPTED_SCORING_SCHEMA_VERSIONS:
                            exclusion_reason = "unsupported_scoring_schema_requires_rescoring"
                        elif row.get("training_gate_decision") == "exclude_c_soft_missing_probability":
                            exclusion_reason = "soft_target_probability_mask_missing"
                        elif row.get("training_gate_decision") == "exclude_or_review_c":
                            exclusion_reason = "grade_C_requires_soft_probability_target"
                        elif row.get("training_gate_decision") == "exclude_ab_non_hard_or_zero_weight":
                            exclusion_reason = "grade_AB_requires_hard_positive_target"
                        elif grade == "D":
                            exclusion_reason = "grade_D_or_zero_weight"
                        elif training_weight <= 0.0:
                            exclusion_reason = "training_weight_zero"
                        elif not include_gate:
                            exclusion_reason = str(row.get("training_gate_decision") or "training_gate_excluded")
                        elif row.get("contract_failures"):
                            exclusion_reason = "training_contract:" + ",".join(row["contract_failures"])
                        else:
                            exclusion_reason = "distillation_eligible_false"
                    excluded_training_rows.append({
                        "case_id": case_dir.name,
                        "organ": organ,
                        "mask_path": str(mask.resolve()),
                        "grade": grade,
                        "training_weight": training_weight,
                        "target_type": row.get("target_type"),
                        "probability_mask_path": row.get("probability_mask_path"),
                        "training_gate_decision": row.get("training_gate_decision"),
                        "training_gate_policy": row.get("training_gate_policy"),
                        "distillation_eligible": row.get("distillation_eligible"),
                        "distillation_exclusion_reason": exclusion_reason,
                        "exclusion_category": exclusion_reason.split(":", 1)[0],
                        "identity_status": identity_status,
                        "selected_candidate_qc_status": row.get("selected_candidate_qc_status"),
                        "selected_candidate_qc_flags": row.get("selected_candidate_qc_flags", []),
                        "shapekit_status": row.get("shapekit_status"),
                    })
                    continue
                rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        fieldnames = sorted({k for row in rows for k in row.keys()}) or ["case_id", "organ", "mask_path"]
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: _csv_safe(v) for k, v in row.items()} for row in rows])
    else:
        write_json(out, rows)
    exclusion_path = out.with_name(out.stem + ".identity_exclusions.json")
    write_json(exclusion_path, excluded_identity_rows)
    training_exclusion_path = out.with_name(out.stem + ".training_exclusions.json")
    write_json(training_exclusion_path, excluded_training_rows)
    grade_counts = {grade: sum(1 for r in rows if r.get("grade") == grade) for grade in ["A", "B", "C", "D"]}
    target_type_counts: dict[str, int] = {}
    for r in rows:
        target_type_counts[str(r.get("target_type") or "hard")] = target_type_counts.get(str(r.get("target_type") or "hard"), 0) + 1
    exclusion_counts: dict[str, int] = {}
    for r in excluded_training_rows:
        key = str(r.get("distillation_exclusion_reason") or "unknown")
        exclusion_counts[key] = exclusion_counts.get(key, 0) + 1
    training_gate_summary = {
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "included_grade_counts": grade_counts,
        "included_target_type_counts": target_type_counts,
        "num_included": len(rows),
        "num_excluded_training": len(excluded_training_rows),
        "num_c_soft_included": sum(
            1
            for r in rows
            if r.get("grade") == "C"
            and r.get("target_type") == "positive_soft"
            and r.get("probability_mask_path")
        ),
        "num_c_missing_probability_excluded": sum(1 for r in excluded_training_rows if r.get("distillation_exclusion_reason") == "soft_target_probability_mask_missing"),
        "num_c_hard_or_provisional_excluded": sum(1 for r in excluded_training_rows if r.get("distillation_exclusion_reason") == "grade_C_requires_soft_probability_target"),
        "num_d_excluded": sum(1 for r in excluded_training_rows if r.get("grade") == "D"),
        "num_unsupported_schema_excluded": sum(1 for r in excluded_training_rows if r.get("distillation_exclusion_reason") == "unsupported_scoring_schema_requires_rescoring"),
        "exclusion_reason_counts": exclusion_counts,
    }
    return {
        "stage": "mstep_manifest",
        "status": "success",
        "updated_annotations_root": str(root),
        "output_manifest": str(out),
        "num_items": len(rows),
        "identity_exclusions": str(exclusion_path),
        "num_identity_exclusions": len(excluded_identity_rows),
        "training_exclusions": str(training_exclusion_path),
        "num_training_exclusions": len(excluded_training_rows),
        "training_gate_summary": training_gate_summary,
        "sample_items": rows[:20],
    }


def write_mstep_config(output_config: str | Path, training_manifest: str | Path, base_model: str = "nnunet_or_epai", notes: str | None = None) -> dict[str, Any]:
    out = Path(output_config).resolve()
    cfg = {
        "stage": "m_step_interface",
        "base_model": base_model,
        "training_manifest": str(Path(training_manifest).resolve()),
        "mode": "smoke_test_then_scale",
        "warning": "50 cases are for workflow debugging only; do not claim formal retraining performance from this small set.",
        "next_training_backend": ["nnUNet v2", "ePAI fine-tuning", "private model fine-tuning"],
        "notes": notes or "Replace this interface with the lab's real training script once the checkpoint/training code is available.",
    }
    write_json(out, cfg)
    return {"stage": "mstep_config", "status": "success", "output_config": str(out), "config": cfg}


# ---------------------------------------------------------------------------
# Real M-step: nnUNet v2 fine-tuning
# ---------------------------------------------------------------------------

def _prepare_nnunet_dataset(
    training_manifest: str | Path,
    dataset_root: Path,
    dataset_id: int = 999,
    dataset_name: str = "MedAI_EMLoop",
    ct_source_root: Path | None = None,
) -> dict[str, Any]:
    """Convert training_manifest into nnUNet raw dataset format."""
    manifest_path = Path(training_manifest).resolve()
    if manifest_path.suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    ds_folder = dataset_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_masks: dict[str, list[dict]] = {}
    for r in rows:
        # nnUNet consumes discrete labels only. Soft/provisional targets are
        # reserved for the VoxTell probability-target training path.
        if str(r.get("target_type") or "positive_hard") not in {"hard", "positive_hard"}:
            continue
        cid = r.get("case_id", "")
        if cid:
            case_masks.setdefault(cid, []).append(r)

    label_map: dict[str, int] = {"background": 0}
    label_idx = 1
    for masks in case_masks.values():
        for m in masks:
            organ = m.get("organ", "")
            if organ and organ not in label_map:
                label_map[organ] = label_idx
                label_idx += 1

    prepared_cases = []
    for cid, masks in sorted(case_masks.items()):
        ct_found = None
        if ct_source_root:
            for cand in [
                ct_source_root / cid / "ct.nii.gz",
                ct_source_root / "ImageTr" / cid / "ct.nii.gz",
                ct_source_root / "data" / "ImageTr" / cid / "ct.nii.gz",
            ]:
                if cand.exists():
                    ct_found = cand; break

        dst_img = images_tr / f"{cid}_0000.nii.gz"
        if ct_found and not dst_img.exists():
            shutil.copy2(ct_found, dst_img)

        dst_label = labels_tr / f"{cid}.nii.gz"
        if not dst_label.exists():
            try:
                import nibabel as nib
                import numpy as np
                ref_img = None; combined = None
                for m in masks:
                    mp = Path(m.get("mask_path", ""))
                    organ = m.get("organ", "")
                    if not mp.exists() or organ not in label_map:
                        continue
                    img = nib.load(str(mp))
                    arr = (np.asanyarray(img.dataobj) > 0).astype("uint8")
                    if combined is None:
                        combined = np.zeros_like(arr, dtype="uint8"); ref_img = img
                    combined[arr > 0] = label_map[organ]
                if combined is not None and ref_img is not None:
                    nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), str(dst_label))
            except Exception:
                pass

        prepared_cases.append({"case_id": cid, "image": str(dst_img), "label": str(dst_label), "ct_found": ct_found is not None, "label_built": dst_label.exists()})

    images_found = sum(1 for c in prepared_cases if c["ct_found"])
    if images_found == 0 and ct_source_root is None:
        return {"status": "failed", "reason": "ct_source_root not provided; imagesTr is empty. Pass --ct-source-root to include CT images.", "dataset_folder": str(ds_folder), "num_labels": len(label_map) - 1}
    ds_json = {"name": dataset_name, "channel_names": {"0": "CT"}, "labels": label_map, "numTraining": len(prepared_cases), "file_ending": ".nii.gz"}
    write_json(ds_folder / "dataset.json", ds_json)
    return {"status": "success", "dataset_folder": str(ds_folder), "dataset_id": dataset_id, "num_cases": len(prepared_cases), "num_labels": len(label_map) - 1, "label_map": label_map, "prepared_cases": prepared_cases[:20], "images_found": images_found}


def run_mstep_nnunet_training(
    training_manifest: str | Path,
    output_folder: str | Path,
    dataset_id: int = 999,
    dataset_name: str = "MedAI_EMLoop",
    ct_source_root: str | Path | None = None,
    trainer: str = "nnUNetTrainer_fast_50epochs",
    plans: str = "nnUNetPlans",
    configuration: str = "3d_fullres",
    folds: str = "all",
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Run M-step: prepare nnUNet dataset and launch training.

    50 cases = smoke test only (will overfit). Scale to 500+ for real training.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nnunet_raw = out / "nnUNet_raw"
    nnunet_preprocessed = out / "nnUNet_preprocessed"
    nnunet_results = out / "nnUNet_results"

    ct_root = Path(ct_source_root).resolve() if ct_source_root else None
    prep = _prepare_nnunet_dataset(training_manifest, nnunet_raw, dataset_id, dataset_name, ct_root)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["nnUNet_n_proc_DA"] = "2"

    plan_cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"]
    # nnUNetv2_train does not accept --num_epochs; epoch count is controlled by the
    # trainer class. Use nnUNetTrainer_fast (500 epochs) or a _Xepochs variant.
    train_cmd = ["nnUNetv2_train", str(dataset_id), configuration, folds, "-tr", trainer, "-p", plans, "--npz"]
    if pretrained_weights:
        train_cmd.extend(["-pretrained_weights", pretrained_weights])

    result: dict[str, Any] = {
        "stage": "m_step_nnunet_training", "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out), "dataset_id": dataset_id, "dataset_preparation": prep,
        "plan_command": plan_cmd, "train_command": train_cmd, "max_epochs_intended": max_epochs,
        "environment": {"nnUNet_raw": str(nnunet_raw), "nnUNet_preprocessed": str(nnunet_preprocessed), "nnUNet_results": str(nnunet_results)},
        "pretrained_weights": pretrained_weights,
        "50_case_warning": "50 cases are for debugging the EM loop only. The model WILL overfit. Scale to 500+ for real training.",
    }

    if dry_run:
        result["status"] = "dry_run"
        result["note"] = "Commands prepared but not executed. Run on a GPU server with nnUNet v2 installed."
        write_json(out / "mstep_training_plan.json", result)
        return result

    start = time.time()
    try:
        plan_proc = subprocess.run(plan_cmd, env=env, capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["plan_returncode"] = plan_proc.returncode
        result["plan_stdout_tail"] = plan_proc.stdout[-4000:]
        result["plan_stderr_tail"] = plan_proc.stderr[-4000:]
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "nnUNetv2_plan_and_preprocess not found. Install nnUNet v2."})
        write_json(out / "mstep_training_plan.json", result); return result
    except subprocess.TimeoutExpired:
        result.update({"status": "failed", "reason": f"Planning timed out after {timeout_sec}s"})
        write_json(out / "mstep_training_plan.json", result); return result

    if plan_proc.returncode != 0:
        result.update({"status": "failed", "reason": "nnUNetv2_plan_and_preprocess failed"})
        write_json(out / "mstep_training_plan.json", result); return result

    try:
        train_proc = subprocess.run(train_cmd, env=env, capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["train_returncode"] = train_proc.returncode
        result["train_stdout_tail"] = train_proc.stdout[-4000:]
        result["train_stderr_tail"] = train_proc.stderr[-4000:]
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "nnUNetv2_train not found"})
        write_json(out / "mstep_training_plan.json", result); return result
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "timeout", "reason": f"Training timed out after {timeout_sec}s"})
        train_proc = subprocess.CompletedProcess(train_cmd, -1,
            stdout=subprocess_text(exc.stdout),
            stderr=subprocess_text(exc.stderr) + f"\n[timeout after {timeout_sec}s]",
        )

    elapsed = time.time() - start
    result["runtime_sec"] = round(elapsed, 2)

    ckpt_folder = nnunet_results / f"Dataset{dataset_id:03d}_{dataset_name}" / f"{trainer}__{plans}__{configuration}" / f"fold_{folds}"
    has_checkpoint = ckpt_folder.exists() and any(ckpt_folder.glob("checkpoint_*.pth"))
    result["checkpoint_folder"] = str(ckpt_folder)
    result["has_checkpoint"] = has_checkpoint

    # Accept both clean exit (rc=0) and timeout (rc=-1) as long as a checkpoint exists.
    if has_checkpoint:
        result["status"] = "success"
        result["updated_model_path"] = str(ckpt_folder)
        result["next_step"] = f"Use the updated checkpoint for next inference round. Pass --nnunet-results {nnunet_results} to infer."
        if train_proc.returncode not in (0, -1):
            result["warning"] = f"Training exited with code {train_proc.returncode} but checkpoint exists — treating as success."
    else:
        result.update({"status": "failed", "reason": "Training completed but no checkpoint produced"})

    write_json(out / "mstep_training_result.json", result)

    return result


def update_registry_checkpoint(
    registry_path: str | Path,
    model_key: str,
    new_checkpoint_path: str | Path,
    new_dataset_json_path: str | Path | None = None,
) -> dict:
    """Update model_registry.yaml with a new checkpoint path after M-step training.

    Uses targeted line-by-line text replacement rather than yaml.dump() so that
    all YAML comments, formatting, and multi-line strings are preserved exactly.
    """
    import re
    import yaml
    reg_path = Path(registry_path).resolve()
    text = reg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    models = data.get("models", {})
    if model_key not in models:
        return {"status": "failed", "reason": f"model_key '{model_key}' not in registry"}
    old_ckpt = models[model_key].get("checkpoint_path", "")

    def _patch_field(src: str, mk: str, field: str, new_val: str) -> str:
        """Replace `field: <value>` under the model_key block, preserving all other lines."""
        lines = src.splitlines(keepends=True)
        in_block = False
        indent = ""
        result = []
        for line in lines:
            stripped = line.lstrip()
            if not in_block:
                if re.match(rf"^  {re.escape(mk)}\s*:", line):
                    in_block = True
                    indent = "    "
                result.append(line)
            else:
                cur_indent = len(line) - len(line.lstrip())
                if cur_indent < 4 and stripped and not stripped.startswith("#"):
                    in_block = False
                    result.append(line)
                elif re.match(rf"^    {re.escape(field)}\s*:", line):
                    result.append(f"    {field}: {new_val}\n")
                else:
                    result.append(line)
        return "".join(result)

    new_ckpt_str = str(new_checkpoint_path).replace("\\", "/")
    text = _patch_field(text, model_key, "checkpoint_path", new_ckpt_str)
    if new_dataset_json_path:
        new_dsj_str = str(new_dataset_json_path).replace("\\", "/")
        text = _patch_field(text, model_key, "dataset_json_path", new_dsj_str)
    reg_path.write_text(text, encoding="utf-8")
    return {
        "status": "success",
        "registry_path": str(reg_path),
        "model_key": model_key,
        "old_checkpoint_path": old_ckpt,
        "new_checkpoint_path": str(new_checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Conditional M-step backends for bundled public/foundation models
# ---------------------------------------------------------------------------

def run_mstep_totalsegmentator_public_nnunet_plan(
    training_manifest: str | Path,
    output_folder: str | Path,
    dataset_id: int = 2101,
    ct_source_root: str | Path | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Prepare a TotalSegmentator-style public nnUNet M-step plan.

    TotalSegmentator bundles a public nnUNet training recipe under
    resources/train_nnunet.md and resources/train_nnunet.sh. That recipe can be
    used as a reference for training a TotalSegmentator-style nnUNet backend.
    It does NOT fully reproduce released TotalSegmentator v2 because official
    v2 used additional non-public data.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ts_root = Path("third_party/TotalSegmentator-master").resolve()
    recipe_md = ts_root / "resources" / "train_nnunet.md"
    recipe_sh = ts_root / "resources" / "train_nnunet.sh"
    converter = ts_root / "resources" / "convert_dataset_to_nnunet.py"

    # For PanTS updated annotations we use the same nnUNet preparation logic as
    # the generic backend, but label this explicitly as TotalSegmentator-style.
    # Keep the generated folder names short because Windows can still hit the
    # classic MAX_PATH limit in deeply nested project directories.
    nnunet_result = run_mstep_nnunet_training(
        training_manifest=training_manifest,
        output_folder=out / "ts_nnunet",
        dataset_id=dataset_id,
        dataset_name="MedAI_TS",
        ct_source_root=ct_source_root,
        trainer="nnUNetTrainerNoMirroring",
        plans="nnUNetPlans",
        configuration="3d_fullres",
        folds="all",
        max_epochs=max_epochs,
        pretrained_weights=pretrained_weights,
        dry_run=True if dry_run else False,
        timeout_sec=timeout_sec,
    )
    result = {
        "stage": "mstep_totalsegmentator_public_nnunet",
        "status": nnunet_result.get("status"),
        "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out),
        "dataset_id": dataset_id,
        "backend": "totalseg_public_nnunet",
        "totalsegmentator_source_root": str(ts_root),
        "recipe_files": {
            "train_nnunet_md": str(recipe_md),
            "train_nnunet_sh": str(recipe_sh),
            "convert_dataset_to_nnunet": str(converter),
            "exists": {
                "train_nnunet_md": recipe_md.exists(),
                "train_nnunet_sh": recipe_sh.exists(),
                "convert_dataset_to_nnunet": converter.exists(),
            },
        },
        "nnunet_update_plan": nnunet_result,
        "limitations": [
            "This trains a TotalSegmentator-style nnUNet backend from updated annotations, not the released TotalSegmentator v2 model itself.",
            "Released TotalSegmentator v2 cannot be fully reproduced from this public recipe because its official training used additional non-public data.",
            "50 PanTS cases are for workflow smoke testing only and will overfit.",
        ],
    }
    write_json(out / "mstep_totalseg_public_nnunet_plan.json", result)
    return result


def _prepare_vista3d_datalist_from_manifest(
    training_manifest: str | Path,
    output_folder: Path,
    ct_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a minimal MONAI datalist for VISTA3D fine-tuning.

    We reuse the nnUNet dataset preparation to combine per-organ updated masks
    into one label map, then produce a MONAI Auto3DSeg-style datalist with
    imagesTr/labelsTr paths.
    """
    prep_root = output_folder / "vista3d_medai_dataset"
    prep = _prepare_nnunet_dataset(
        training_manifest=training_manifest,
        dataset_root=prep_root / "nnunet_raw_proxy",
        dataset_id=3101,
        dataset_name="MedAI_VISTA3D",
        ct_source_root=Path(ct_source_root).resolve() if ct_source_root else None,
    )
    ds_folder = Path(prep["dataset_folder"])
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    rows = []
    image_files = sorted(images_tr.glob("*_0000.nii.gz"))
    for idx, img in enumerate(image_files):
        cid = img.name[:-12]
        label = labels_tr / f"{cid}.nii.gz"
        if not label.exists():
            continue
        # MONAI datafold_read uses the requested fold for validation and all
        # other folds for training. Keep at least one validation case when possible.
        fold = 0 if idx == 0 else 1
        rows.append({"fold": fold, "image": str(img.relative_to(ds_folder)), "label": str(label.relative_to(ds_folder))})
    datalist = {"training": rows, "testing": []}
    datalist_path = output_folder / "vista3d_medai_datalist.json"
    write_json(datalist_path, datalist)
    return {"dataset_folder": str(ds_folder), "datalist_path": str(datalist_path), "num_training_items": len(rows), "nnunet_proxy_preparation": prep}


def run_mstep_vista3d_monai_finetune_plan(
    training_manifest: str | Path,
    output_folder: str | Path,
    vista_root: str | Path = "third_party/VISTA3D-Inference-Pipeline-master",
    ct_source_root: str | Path | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Prepare or launch VISTA3D MONAI bundle fine-tuning.

    The bundled VISTA3D pipeline includes configs/train.json,
    train_continual.json, multi_gpu_train.json and scripts/trainer.py. This
    backend creates a project-specific datalist and prepares the MONAI command.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vista = Path(vista_root).resolve()
    train_json = vista / "configs" / "train.json"
    continual_json = vista / "configs" / "train_continual.json"
    multi_gpu_json = vista / "configs" / "multi_gpu_train.json"
    trainer_py = vista / "scripts" / "trainer.py"
    prep = _prepare_vista3d_datalist_from_manifest(training_manifest, out, ct_source_root)

    cmd = [
        "python", "-m", "monai.bundle", "run",
        "--config_file", str(train_json),
        "--dataset_dir", prep["dataset_folder"],
        "--data_list_file_path", prep["datalist_path"],
        "--finetune", "True",
        "--epochs", str(max_epochs),
    ]
    if pretrained_weights:
        cmd.extend(["--finetune_model_path", str(pretrained_weights)])

    result: dict[str, Any] = {
        "stage": "mstep_vista3d_monai_finetune",
        "status": "dry_run" if dry_run else "prepared",
        "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out),
        "backend": "monai_bundle_finetune",
        "vista_root": str(vista),
        "recipe_files": {
            "train_json": str(train_json),
            "train_continual_json": str(continual_json),
            "multi_gpu_train_json": str(multi_gpu_json),
            "trainer_py": str(trainer_py),
            "exists": {
                "train_json": train_json.exists(),
                "train_continual_json": continual_json.exists(),
                "multi_gpu_train_json": multi_gpu_json.exists(),
                "trainer_py": trainer_py.exists(),
            },
        },
        "dataset_preparation": prep,
        "command": cmd,
        "pretrained_weights": pretrained_weights,
        "limitations": [
            "This fine-tunes a VISTA3D MONAI bundle checkpoint if one is provided; it does not reproduce the original foundation model from scratch.",
            "VISTA3D training/fine-tuning requires a MONAI bundle environment and sufficient GPU memory.",
            "50 PanTS cases are for workflow smoke testing only and will overfit.",
        ],
    }
    if dry_run:
        write_json(out / "mstep_vista3d_monai_plan.json", result)
        return result
    try:
        proc = subprocess.run(cmd, cwd=str(vista), capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["returncode"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        result["status"] = "success" if proc.returncode == 0 else "failed"
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "python/monai.bundle not found. Install MONAI bundle dependencies."})
    except subprocess.TimeoutExpired:
        result.update({"status": "timeout", "reason": f"VISTA3D fine-tuning timed out after {timeout_sec}s"})
    write_json(out / "mstep_vista3d_monai_result.json", result)
    return result

# ---------------------------------------------------------------------------
# Selected-model-aware M-step
# ---------------------------------------------------------------------------

def _filter_manifest_for_target_model(training_manifest: str | Path, output_manifest: str | Path, target_model: str) -> dict[str, Any]:
    """Copy a manifest and annotate each row with the selected M-step target model.

    The current run-loop produces final updated annotations. It may not always
    know which candidate originally won after ShapeKit/LabelCritic/human review,
    so the M-step target is explicitly supplied by the selected task model. This
    keeps the loop semantically clear: E-step primary model -> updated data ->
    M-step update of that same model family when trainable.
    """
    inp = Path(training_manifest).resolve()
    out = Path(output_manifest).resolve()
    if inp.suffix.lower() == ".csv":
        with inp.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(inp.read_text(encoding="utf-8")) if inp.exists() else []
    supports_soft = "voxtell" in str(target_model).lower() or "prompt_student" in str(target_model).lower()
    excluded_soft = [r for r in rows if str(r.get("target_type") or "hard") != "hard" and not supports_soft]
    if not supports_soft:
        rows = [r for r in rows if str(r.get("target_type") or "hard") == "hard"]
    for r in rows:
        if supports_soft and str(r.get("target_type") or "hard") == "soft" and r.get("probability_mask_path"):
            r["mask"] = r["probability_mask_path"]
            r["mask_path"] = r["probability_mask_path"]
        r["target_model"] = target_model
        r.setdefault("mstep_scope", "selected_model_family")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        fieldnames = sorted({k for r in rows for k in r.keys()}) or ["case_id", "organ", "mask_path", "target_model"]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    else:
        write_json(out, rows)
    return {"stage": "target_model_manifest", "status": "success", "input_manifest": str(inp), "output_manifest": str(out), "target_model": target_model, "num_items": len(rows), "num_soft_or_provisional_excluded": len(excluded_soft)}


def run_model_specific_mstep_update(
    training_manifest: str | Path,
    output_folder: str | Path,
    target_model: str,
    registry_path: str | Path = "configs/model_registry.yaml",
    ct_source_root: str | Path | None = None,
    dataset_id: int | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Selected-model-aware M-step.

    This is the corrected M-step semantics: update the selected model family
    whenever that family is trainable/nnUNet-compatible. TotalSegmentator and
    other non-trainable inference-only backends are explicitly rejected as M-step
    targets instead of silently training a generic model.
    """
    from .model_registry import load_registry

    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    models = registry.get("models", {})
    if target_model not in models:
        result = {"stage": "mstep_update", "status": "failed", "reason": f"target_model '{target_model}' not found in registry", "available_models": sorted(models.keys())}
        write_json(out / "mstep_update_result.json", result)
        return result

    entry = models[target_model]
    backend = entry.get("mstep_backend", "none")
    trainable = entry.get("trainable", "unknown")
    role = entry.get("mstep_role", "")
    reason = entry.get("mstep_reason", "")
    base = {
        "stage": "mstep_update",
        "target_model": target_model,
        "target_model_name": entry.get("name", target_model),
        "trainable": trainable,
        "mstep_backend": backend,
        "mstep_role": role,
        "mstep_reason": reason,
        "selected_model_semantics": "E-step primary model is the preferred M-step target when it is trainable. Non-trainable models remain E-step candidates/baselines only.",
        "warning_50_case": "50 cases are for debugging the loop only; do not claim formal model-improvement results from this small set.",
    }

    if backend in {"none", None} or str(trainable) in {"no", "unknown_template_only"}:
        result = {**base, "status": "not_trainable_in_current_project", "next_action": "Use this model as an E-step candidate only, or provide the original training script/checkpoint layout to enable model-specific M-step."}
        write_json(out / "mstep_update_result.json", result)
        return result
    if backend == "external_training_required":
        result = {**base, "status": "external_training_required", "next_action": "This model needs its own training recipe. The current project only wraps inference."}
        write_json(out / "mstep_update_result.json", result)
        return result

    dsid = dataset_id or int(entry.get("mstep_dataset_id") or 999)
    target_manifest = out / f"training_manifest__target_{target_model}.json"
    manifest_result = _filter_manifest_for_target_model(training_manifest, target_manifest, target_model)

    # Prefer explicit pretrained weights from user. Registry mstep_init_from is a
    # folder hint; we do not blindly pass it as -pretrained_weights unless it is a file.
    pretrained = pretrained_weights
    init_hint = entry.get("mstep_init_from")
    if not pretrained and init_hint:
        p = Path(str(init_hint))
        if p.exists() and p.is_file():
            pretrained = str(p)

    if backend == "totalseg_public_nnunet":
        training_result = run_mstep_totalsegmentator_public_nnunet_plan(
            training_manifest=target_manifest,
            output_folder=out / f"ts_update__{target_model}",
            dataset_id=dsid,
            ct_source_root=ct_source_root,
            max_epochs=max_epochs,
            pretrained_weights=pretrained,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
        result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
        write_json(out / "mstep_update_result.json", result)
        return result

    if backend == "monai_bundle_finetune":
        vista_root = entry.get("mstep_init_from") or entry.get("source_code_path") or entry.get("checkpoint_path") or "third_party/VISTA3D-Inference-Pipeline-master"
        training_result = run_mstep_vista3d_monai_finetune_plan(
            training_manifest=target_manifest,
            output_folder=out / f"vista3d_update__{target_model}",
            vista_root=vista_root,
            ct_source_root=ct_source_root,
            max_epochs=max_epochs,
            pretrained_weights=pretrained,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
        result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
        if not pretrained:
            result["pretrained_note"] = "VISTA3D fine-tuning normally needs --pretrained-weights pointing to a VISTA3D/MONAI checkpoint. Dry-run can be used without it."
        write_json(out / "mstep_update_result.json", result)
        return result

    if not str(backend).startswith("nnunetv2"):
        result = {**base, "status": "unsupported_mstep_backend", "next_action": f"Implement a backend adapter for {backend}."}
        write_json(out / "mstep_update_result.json", result)
        return result

    training_result = run_mstep_nnunet_training(
        training_manifest=target_manifest,
        output_folder=out / f"nnunet_update__{target_model}",
        dataset_id=dsid,
        dataset_name=f"MedAI_{target_model}",
        ct_source_root=ct_source_root,
        # Use nnUNetTrainer_fast (500 epochs + compile + cudnn.benchmark) for speed.
        # Fall back to the registry trainer only if it's a specialised variant.
        trainer=entry.get("trainer", "nnUNetTrainer").replace("nnUNetTrainer", "nnUNetTrainer_fast_50epochs"),
        plans=entry.get("plans", "nnUNetPlans"),
        configuration="3d_fullres",
        folds="all",
        max_epochs=max_epochs,
        pretrained_weights=pretrained,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
    )
    result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
    if init_hint and not pretrained:
        result["pretrained_note"] = "Registry contains an init_from folder hint, but no explicit checkpoint file was passed. Use --pretrained-weights if you want true fine-tuning from a specific .pth file."

    # Auto-update registry checkpoint_path so the next E-step uses the newly trained model.
    if training_result.get("status") == "success" and training_result.get("updated_model_path"):
        reg_update = update_registry_checkpoint(
            registry_path=registry_path,
            model_key=target_model,
            new_checkpoint_path=training_result["updated_model_path"],
        )
        result["registry_update"] = reg_update

    write_json(out / "mstep_update_result.json", result)
    return result
