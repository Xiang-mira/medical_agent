#!/usr/bin/env python3
"""Repair trainable C-soft targets from LabelCritic-abstained QC-pass candidates.

This is intentionally conservative: it does not pick a hard winner after a
LabelCritic abstention.  It only creates a soft probability target when the
E-step already recorded an expected-present positive-soft row and at least two
independent candidate masks passed objective QC, ShapeKit, identity checks, and
non-empty mask inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


HARD_QC_FLAGS = {
    "zero_volume_mask",
    "empty_mask",
    "geometry_mismatch",
    "shape_mismatch_ct",
    "affine_mismatch_ct",
    "orientation_mismatch_ct",
    "postprocess_failed",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mask_array(path: Path) -> tuple[np.ndarray | None, nib.Nifti1Image | None, str | None]:
    if not path.exists():
        return None, None, "mask_missing"
    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)
        binary = arr > 0
        if int(binary.sum()) <= 0:
            return None, img, "empty_mask"
        return binary.astype(np.float32), img, None
    except Exception as exc:
        return None, None, f"mask_unreadable:{exc}"


def candidate_is_safe(candidate: dict[str, Any]) -> tuple[bool, str | None]:
    model = str(candidate.get("model") or "")
    if model == "student_prev" or model.startswith("student_"):
        return False, "student_prediction_candidate_excluded"
    if str(candidate.get("candidate_qc_status") or "").lower() not in {"pass", "passed", "success", "ok"}:
        return False, "candidate_qc_not_pass"
    if str(candidate.get("candidate_shapekit_status") or "").lower() != "success":
        return False, "candidate_shapekit_not_success"
    if str(candidate.get("identity_status") or "valid").lower() not in {"valid", "ok", "success"}:
        return False, "candidate_identity_not_valid"
    flags = {str(flag).lower() for flag in (candidate.get("candidate_qc_flags") or [])}
    if flags & HARD_QC_FLAGS:
        return False, f"hard_qc_flag:{sorted(flags & HARD_QC_FLAGS)[0]}"
    if not candidate.get("prediction"):
        return False, "candidate_prediction_missing"
    return True, None


def row_is_soft_repair_candidate(row: dict[str, Any]) -> bool:
    if str(row.get("expected_presence") or "") != "expected_present":
        return False
    if str(row.get("target_type") or "").lower() not in {"positive_soft", "soft"}:
        return False
    if row.get("selected_model"):
        return False
    if not bool(row.get("labelcritic_compare_used")):
        return False
    if "automatic_abstention" not in {str(x) for x in (row.get("review_flags") or [])}:
        return False
    return True


def build_soft_consensus(
    *,
    meta_path: Path,
    row: dict[str, Any],
    min_candidates: int,
    threshold: float,
    write_files: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    case_dir = meta_path.parent
    case_id = str(row.get("case_id") or case_dir.name)
    organ = str(row.get("organ") or "")
    rejected: list[dict[str, str]] = []
    arrays: list[np.ndarray] = []
    image_ref: nib.Nifti1Image | None = None
    source_models: list[str] = []
    candidate_paths: list[str] = []

    for candidate in row.get("candidate_predictions") or []:
        if not isinstance(candidate, dict):
            continue
        ok, reason = candidate_is_safe(candidate)
        model = str(candidate.get("model") or "")
        if not ok:
            rejected.append({"model": model, "reason": str(reason)})
            continue
        path = Path(str(candidate.get("prediction"))).expanduser()
        arr, img, mask_reason = mask_array(path)
        if arr is None:
            rejected.append({"model": model, "reason": str(mask_reason)})
            continue
        if arrays and arr.shape != arrays[0].shape:
            rejected.append({"model": model, "reason": "candidate_shape_mismatch"})
            continue
        arrays.append(arr)
        image_ref = image_ref or img
        source_models.append(model)
        candidate_paths.append(str(path.resolve()))

    audit = {
        "case_id": case_id,
        "organ": organ,
        "eligible_candidate_count": len(arrays),
        "source_models": source_models,
        "rejected_candidates": rejected,
    }
    if len(arrays) < min_candidates or image_ref is None:
        audit["status"] = "skipped"
        audit["reason"] = "insufficient_safe_qc_pass_candidates"
        return None, audit

    prob = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    hard = (prob >= float(threshold)).astype(np.uint8)
    if int(hard.sum()) <= 0:
        audit["status"] = "skipped"
        audit["reason"] = "soft_consensus_threshold_empty"
        return None, audit

    soft_dir = case_dir / "soft_consensus_targets"
    prob_path = soft_dir / f"{organ}_probability.nii.gz"
    updated_dir = case_dir / "updated"
    hard_path = updated_dir / f"{organ}.nii.gz"
    if write_files:
        soft_dir.mkdir(parents=True, exist_ok=True)
        updated_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(prob, image_ref.affine, image_ref.header), str(prob_path))
        nib.save(nib.Nifti1Image(hard, image_ref.affine, image_ref.header), str(hard_path))

    selected = {
        **row,
        "case_id": case_id,
        "organ": organ,
        "selection_status": "selected",
        "selection_method": "soft_consensus_from_labelcritic_abstention",
        "selected_model": "soft_consensus_qc_pass_candidates",
        "source_model": "soft_consensus_qc_pass_candidates",
        "selected_provider": "soft_consensus_qc_pass_candidates",
        "selected_prediction": str(hard_path.resolve()),
        "final_mask": str(hard_path.resolve()),
        "mask_path": str(hard_path.resolve()),
        "probability_mask_path": str(prob_path.resolve()),
        "grade": "C",
        "target_type": "positive_soft",
        "legacy_target_type": row.get("target_type"),
        "training_weight": float(row.get("soft_consensus_training_weight") or 0.1),
        "distillation_eligible": True,
        "distillation_exclusion_reason": None,
        "student_training_priority": "C",
        "publication_status": "accepted_soft_consensus",
        "soft_consensus_source_models": source_models,
        "soft_consensus_candidate_paths": candidate_paths,
        "soft_consensus_probability_threshold": float(threshold),
        "soft_consensus_policy": (
            "C-soft target from multiple QC-pass candidates after LabelCritic abstention; "
            "no hard winner was promoted."
        ),
        "scoring_schema_version": row.get("scoring_schema_version") or "autolabel_core_v2",
        "ground_truth_status": row.get("ground_truth_status") or "soft_consensus_pseudo_label_not_expert_gt",
        "dataset_role": "pseudo_label",
        "selected_candidate_qc_status": "pass",
        "selected_candidate_qc_flags": [],
        "quality_status": "soft_consensus_reviewed",
    }
    audit.update({
        "status": "repaired",
        "probability_mask_path": str(prob_path.resolve()),
        "final_mask": str(hard_path.resolve()),
        "foreground_voxels": int(hard.sum()),
        "files_written": bool(write_files),
    })
    return selected, audit


def repair_soft_consensus_targets(
    ann_root: Path,
    *,
    apply: bool,
    min_candidates: int = 2,
    threshold: float = 0.5,
) -> dict[str, Any]:
    report = {
        "stage": "repair_soft_consensus_positive_targets",
        "annotation_root": str(ann_root.resolve()),
        "apply": apply,
        "min_candidates": min_candidates,
        "threshold": threshold,
        "repaired_count": 0,
        "audits": [],
    }
    for meta_path in sorted(ann_root.glob("*/selection_metadata.json")):
        doc = read_json(meta_path)
        if not isinstance(doc, dict):
            continue
        selected_organs = list(doc.get("selected_organs") or [])
        changed = False
        selected_by_organ = {
            str(item.get("organ") or ""): idx
            for idx, item in enumerate(selected_organs)
            if isinstance(item, dict)
        }
        for row in doc.get("selection_rows") or []:
            if not isinstance(row, dict) or not row_is_soft_repair_candidate(row):
                continue
            selected, audit = build_soft_consensus(
                meta_path=meta_path,
                row=row,
                min_candidates=min_candidates,
                threshold=threshold,
                write_files=apply,
            )
            report["audits"].append(audit)
            if selected is None:
                continue
            if apply:
                organ = str(selected["organ"])
                if organ in selected_by_organ:
                    selected_organs[selected_by_organ[organ]] = selected
                else:
                    selected_by_organ[organ] = len(selected_organs)
                    selected_organs.append(selected)
                row.update({
                    "selection_status": selected["selection_status"],
                    "selection_method": selected["selection_method"],
                    "selected_model": selected["selected_model"],
                    "source_model": selected["source_model"],
                    "selected_prediction": selected["selected_prediction"],
                    "final_mask": selected["final_mask"],
                    "mask_path": selected["mask_path"],
                    "probability_mask_path": selected["probability_mask_path"],
                    "grade": selected["grade"],
                    "target_type": selected["target_type"],
                    "training_weight": selected["training_weight"],
                    "distillation_eligible": True,
                    "distillation_exclusion_reason": None,
                    "publication_status": selected["publication_status"],
                    "soft_consensus_source_models": selected["soft_consensus_source_models"],
                })
                changed = True
                report["repaired_count"] += 1
        if apply and changed:
            backup = meta_path.with_name("selection_metadata.pre_soft_consensus_repair.json")
            if not backup.exists():
                backup.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
            doc["selected_organs"] = selected_organs
            doc["soft_consensus_repair"] = {
                "status": "applied",
                "policy": "C-soft probability targets only from multiple QC-pass LabelCritic-abstained candidates",
            }
            write_json(meta_path, doc)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-root", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-candidates", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    report = repair_soft_consensus_targets(
        Path(args.annotation_root).resolve(),
        apply=args.apply,
        min_candidates=args.min_candidates,
        threshold=args.threshold,
    )
    output = Path(args.output).resolve() if args.output else Path(args.annotation_root).resolve().parent / "soft_consensus_repair_report.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("repaired_count", 0) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
