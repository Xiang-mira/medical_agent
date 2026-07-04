#!/usr/bin/env python3
"""Conservative Round2 gate repair from completed E-step metadata.

This script does not run teacher inference, LabelCritic, ShapeKit, or training.
It only carries forward trustworthy previous-round selected masks for formal key
organs when the current Round2 metadata has no usable selected label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"
)
DEFAULT_ROUND1_SELECTED_ROOT = (
    ROOT / "outputs" / "formal_round1_final_20260627" / "round1" / "estep" / "annotation_versions"
)

POLICY_VERSION = "round2_key_organ_carry_forward_v1"
BACKUP_NAME = "carry_forward_v1"
FORMAL_KEY_ORGANS = [
    "liver",
    "spleen",
    "pancreas",
    "kidney_left",
    "kidney_right",
    "aorta",
    "adrenal_gland_left",
    "adrenal_gland_right",
    "stomach",
    "duodenum",
    "colon",
    "small_bowel",
    "bladder",
]
HARD_FLAGS = {
    "zero_volume_mask",
    "expected_present_zero_volume_mask",
    "geometry_mismatch",
    "shape_mismatch_ct",
    "affine_mismatch_ct",
    "orientation_mismatch_ct",
    "postprocess_failed",
    "many_connected_components",
}
OUT_OF_FOV_STATUSES = {
    "out_of_fov",
    "out_of_scan",
    "outside_fov",
    "outside_scan",
    "not_in_field_of_view",
    "not_visible",
    "unknown",
}


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_key_organs() -> list[str]:
    return list(FORMAL_KEY_ORGANS)


def _mask_name(organ: str) -> str:
    return f"{organ}.nii.gz"


def _has_usable_selected_label(selected: dict[str, dict[str, Any]], case_id: str, organ: str) -> bool:
    item = selected.get(organ)
    if not item:
        return False
    final_mask = item.get("final_mask") or item.get("mask_path") or item.get("mask") or item.get("selected_prediction")
    return bool(
        str(item.get("grade") or "D").upper() in {"A", "B", "C"}
        and (item.get("selected_model") or item.get("source_model"))
        and final_mask
    )


def _candidate_flags(candidate: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    for key in ("candidate_qc_flags", "quality_flags", "review_flags", "selected_candidate_qc_flags"):
        value = candidate.get(key)
        if isinstance(value, list):
            flags.update(str(x) for x in value)
    checks = candidate.get("candidate_qc_checks") or candidate.get("selected_candidate_qc_checks") or {}
    if isinstance(checks, dict):
        if int(checks.get("mask_voxels") or 1) <= 0:
            flags.add("zero_volume_mask")
        if checks.get("connected_components") is not None and int(checks.get("connected_components") or 0) > 20:
            flags.add("many_connected_components")
        status = str(checks.get("geometry_status") or "")
        if "mismatch" in status:
            flags.add("geometry_mismatch")
    return flags


def _candidate_is_trusted(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(candidate.get("model") or "") != "round_prev_selected":
        reasons.append("not_round_prev_selected")
    identity_status = str(candidate.get("identity_status") or "").lower()
    if identity_status and identity_status != "valid":
        reasons.append(f"identity_not_valid:{identity_status}")
    qc_status = str(candidate.get("candidate_qc_status") or "").lower()
    if qc_status == "fail":
        reasons.append("candidate_qc_failed")
    hard_flags = sorted(_candidate_flags(candidate) & HARD_FLAGS)
    if hard_flags:
        reasons.append("hard_quality_flags:" + ",".join(hard_flags))
    prediction = Path(str(candidate.get("prediction") or ""))
    if not prediction.is_file():
        reasons.append("candidate_prediction_missing")
    return not reasons, reasons


def _load_round1_selected_item(round1_selected_root: Path, case_id: str, organ: str) -> dict[str, Any]:
    meta = read_json(round1_selected_root / case_id / "selection_metadata.json", {})
    for item in meta.get("selected_organs", []) or []:
        if isinstance(item, dict) and str(item.get("organ") or "") == organ:
            return dict(item)
    for row in meta.get("selection_rows", []) or []:
        if isinstance(row, dict) and str(row.get("organ") or "") == organ:
            return dict(row)
    return {}


def _fallback_round1_candidate(round1_selected_root: Path, case_id: str, organ: str) -> dict[str, Any] | None:
    mask = round1_selected_root / case_id / "updated" / _mask_name(organ)
    if not mask.is_file():
        return None
    item = _load_round1_selected_item(round1_selected_root, case_id, organ)
    flags = set()
    for key in ("quality_flags", "review_flags", "selected_candidate_qc_flags"):
        value = item.get(key)
        if isinstance(value, list):
            flags.update(str(x) for x in value)
    return {
        "model": "round_prev_selected",
        "prediction": str(mask),
        "pre_shapekit_prediction": item.get("pre_shapekit_mask") or item.get("selected_pre_shapekit_prediction") or str(mask),
        "candidate_qc_status": item.get("selected_candidate_qc_status") or item.get("quality_status") or "pass",
        "candidate_qc_score": item.get("selected_candidate_qc_score", 1.0),
        "candidate_qc_flags": sorted(flags),
        "identity_status": item.get("identity_status") or "valid",
        "identity_mismatch_reasons": item.get("identity_mismatch_reasons") or [],
        "mask_sha256": item.get("final_mask_sha256") or item.get("selected_source_mask_sha256"),
        "candidate_source": "round1_selected_root",
    }


def _round_prev_candidate(
    row: dict[str, Any],
    round1_selected_root: Path,
    case_id: str,
    organ: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    candidates = row.get("candidate_predictions") or []
    rejected_reasons: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict) or str(candidate.get("model") or "") != "round_prev_selected":
                continue
            ok, reasons = _candidate_is_trusted(candidate)
            if ok:
                return dict(candidate), []
            rejected_reasons.extend(reasons)
    fallback = _fallback_round1_candidate(round1_selected_root, case_id, organ)
    if fallback:
        ok, reasons = _candidate_is_trusted(fallback)
        if ok:
            return fallback, []
        rejected_reasons.extend(reasons)
    return None, rejected_reasons or ["no_round_prev_selected_source"]


def validate_mask_against_ct(mask_path: Path, ct_path: Path) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {
        "mask_path": str(mask_path),
        "ct_path": str(ct_path),
        "status": "failed",
        "flags": [],
    }
    flags: list[str] = []
    if not mask_path.is_file():
        flags.append("missing_file")
    if not ct_path.is_file():
        flags.append("ct_missing")
    if flags:
        checks["flags"] = flags
        checks["reason"] = ",".join(flags)
        return False, checks
    try:
        mask_img = nib.load(str(mask_path))
        ct_img = nib.load(str(ct_path))
        checks["mask_shape"] = list(mask_img.shape[:3])
        checks["ct_shape"] = list(ct_img.shape[:3])
        checks["mask_spacing"] = [float(x) for x in mask_img.header.get_zooms()[:3]]
        checks["ct_spacing"] = [float(x) for x in ct_img.header.get_zooms()[:3]]
        if tuple(mask_img.shape[:3]) != tuple(ct_img.shape[:3]):
            flags.append("geometry_mismatch")
            flags.append("shape_mismatch_ct")
        if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-3):
            flags.append("geometry_mismatch")
            flags.append("affine_mismatch_ct")
        mask_arr = np.asanyarray(mask_img.dataobj) > 0
        voxels = int(mask_arr.sum())
        checks["mask_voxels"] = voxels
        if voxels <= 0:
            flags.append("zero_volume_mask")
            flags.append("expected_present_zero_volume_mask")
        try:
            from scipy.ndimage import label as scipy_label

            _, components = scipy_label(mask_arr)
            checks["connected_components"] = int(components)
            if int(components) > 20:
                flags.append("many_connected_components")
        except Exception as exc:
            checks["connected_components"] = None
            checks["connected_components_status"] = f"unavailable:{type(exc).__name__}"
        if voxels > 0:
            voxel_volume = float(abs(np.linalg.det(mask_img.affine[:3, :3])))
            if voxel_volume <= 0:
                voxel_volume = float(np.prod(mask_img.header.get_zooms()[:3]))
            checks["mask_volume_mm3"] = float(voxels * voxel_volume)
    except Exception as exc:
        flags.append("unreadable_mask")
        checks["exception"] = f"{type(exc).__name__}: {exc}"
    checks["flags"] = sorted(set(flags))
    hard_flags = sorted(set(flags) & (HARD_FLAGS | {"missing_file", "ct_missing", "unreadable_mask"}))
    checks["status"] = "success" if not hard_flags else "failed"
    checks["hard_flags"] = hard_flags
    return not hard_flags, checks


def _link_or_copy_mask(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _updated_metadata(
    row: dict[str, Any],
    candidate: dict[str, Any],
    final_mask: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    updated = dict(row)
    candidate_prediction = str(candidate.get("prediction") or final_mask)
    mask_sha = candidate.get("mask_sha256") or file_sha256(Path(candidate_prediction))
    updated.update({
        "selection_method": "round_prev_selected_carry_forward",
        "selection_status": "selected",
        "selected_model": "round_prev_selected",
        "source_model": "round_prev_selected",
        "selected_teacher": "round_prev_selected",
        "selected_prediction": str(final_mask),
        "selected_pre_shapekit_prediction": candidate.get("pre_shapekit_prediction") or candidate_prediction,
        "hard_mask_path": str(final_mask),
        "final_mask": str(final_mask),
        "mask": str(final_mask),
        "mask_path": str(final_mask),
        "grade": "B",
        "target_type": "positive_hard",
        "training_weight": 0.7,
        "distillation_eligible": True,
        "should_enter_student_training": True,
        "publication_status": "accepted_carry_forward",
        "carry_forward_policy_version": POLICY_VERSION,
        "carry_forward_timestamp": now,
        "carry_forward_source_mask": candidate_prediction,
        "carry_forward_source_model": "round_prev_selected",
        "carry_forward_validation": validation,
        "selected_candidate_qc_status": candidate.get("candidate_qc_status") or "pass",
        "selected_candidate_qc_score": candidate.get("candidate_qc_score", 1.0),
        "selected_candidate_qc_flags": [],
        "selected_candidate_qc_checks": validation,
        "selected_candidate_shapekit_status": candidate.get("candidate_shapekit_status") or "carried_forward",
        "selected_candidate_shapekit_reason": candidate.get("candidate_shapekit_reason"),
        "identity_status": "valid",
        "identity_mismatch_reasons": [],
        "quality_flags": [],
        "review_flags": [],
        "requires_manual_review": False,
        "distillation_exclusion_reason": None,
        "decision_status": "accepted_carry_forward",
        "decision_reasons": [
            "current Round2 row had no usable selected label",
            "previous-round selected mask passed geometry/nonempty/QC carry-forward checks",
        ],
        "selected_reason": "Conservative carry-forward from trusted Round1 selected pseudo label.",
        "selection_reason": "round_prev_selected_carry_forward_policy",
        "primary_selector": "round_prev_selected_carry_forward_policy",
        "scoring_schema_version": row.get("scoring_schema_version") or "autolabel_core_v2",
        "final_mask_sha256": mask_sha,
        "selected_source_mask_sha256": mask_sha,
        "labelcritic_formal_selection_used": False,
        "labelcritic_audit_only_not_used_for_formal_selection": True,
    })
    if "round_prev_selected" not in [str(x) for x in updated.get("candidate_models") or []]:
        models = list(updated.get("candidate_models") or [])
        models.insert(0, "round_prev_selected")
        updated["candidate_models"] = models
        updated["candidate_count"] = len(models)
    return updated


def _repair_case(
    metadata_path: Path,
    run_root: Path,
    round1_selected_root: Path,
    *,
    apply: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    doc = read_json(metadata_path, {})
    case_id = str(doc.get("case_id") or metadata_path.parent.name)
    selected_index = {
        str(item.get("organ") or ""): item
        for item in doc.get("selected_organs", []) or []
        if isinstance(item, dict) and item.get("organ")
    }
    rows = [row for row in doc.get("selection_rows", []) or [] if isinstance(row, dict)]
    selected_rows = [item for item in doc.get("selected_organs", []) or [] if isinstance(item, dict)]
    case_report: dict[str, Any] = {
        "case_id": case_id,
        "metadata_path": str(metadata_path),
        "repaired": [],
        "skipped": [],
    }
    changed = False
    row_updates: dict[str, dict[str, Any]] = {}
    for row in rows:
        organ = str(row.get("organ") or "")
        if organ not in FORMAL_KEY_ORGANS:
            continue
        if str(row.get("expected_presence") or "") != "expected_present":
            case_report["skipped"].append({
                "organ": organ,
                "reason": "expected_presence_not_expected_present",
                "expected_presence": row.get("expected_presence"),
            })
            continue
        fov_status = str(row.get("fov_status") or "").lower()
        if fov_status in OUT_OF_FOV_STATUSES or "out_of" in fov_status:
            case_report["skipped"].append({
                "organ": organ,
                "reason": "out_of_fov_or_unknown",
                "fov_status": row.get("fov_status"),
            })
            continue
        if _has_usable_selected_label(selected_index, case_id, organ):
            case_report["skipped"].append({"organ": organ, "reason": "already_has_usable_selected_label"})
            continue
        candidate, source_reasons = _round_prev_candidate(row, round1_selected_root, case_id, organ)
        if not candidate:
            case_report["skipped"].append({
                "organ": organ,
                "reason": "no_trusted_round_prev_selected_source",
                "details": source_reasons,
            })
            continue
        source_mask = Path(str(candidate.get("prediction") or ""))
        ct_path = Path(str(row.get("ct_path") or doc.get("ct_path") or ""))
        valid, validation = validate_mask_against_ct(source_mask, ct_path)
        if not valid:
            case_report["skipped"].append({
                "organ": organ,
                "reason": "mask_validation_failed",
                "validation": validation,
            })
            continue
        final_mask = metadata_path.parent / "updated" / _mask_name(organ)
        updated = _updated_metadata(row, candidate, final_mask, validation)
        row_updates[organ] = updated
        case_report["repaired"].append({
            "organ": organ,
            "source_mask": str(source_mask),
            "final_mask": str(final_mask),
            "source": candidate.get("candidate_source", "round2_candidate_predictions"),
            "mask_voxels": validation.get("mask_voxels"),
        })
        changed = True

    if changed and apply:
        backup = run_root / "round2" / "estep" / "repair_backups" / BACKUP_NAME / case_id / "selection_metadata.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(metadata_path, backup)
        for organ, updated in row_updates.items():
            source = Path(str(updated["carry_forward_source_mask"]))
            _link_or_copy_mask(source, Path(str(updated["final_mask"])))
        new_rows = [row_updates.get(str(row.get("organ") or ""), row) for row in rows]
        new_selected: list[dict[str, Any]] = []
        replaced = set()
        for item in selected_rows:
            organ = str(item.get("organ") or "")
            if organ in row_updates:
                new_selected.append(dict(row_updates[organ]))
                replaced.add(organ)
            else:
                new_selected.append(item)
        for organ, updated in row_updates.items():
            if organ not in replaced:
                new_selected.append(dict(updated))
        doc["selection_rows"] = new_rows
        doc["selected_organs"] = new_selected
        repair_history = list(doc.get("repair_history") or [])
        repair_history.append({
            "policy_version": POLICY_VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repaired_organs": sorted(row_updates),
            "backup": str(backup),
        })
        doc["repair_history"] = repair_history
        write_json(metadata_path, doc)

    return case_report, doc if changed and apply else None


def _coverage_from_metadata(run_root: Path, round_idx: int = 2) -> dict[str, dict[str, Any]]:
    ann_root = run_root / f"round{round_idx}" / "estep" / "annotation_versions"
    coverage = {organ: {"expected_present": 0, "usable": 0} for organ in FORMAL_KEY_ORGANS}
    for metadata_path in sorted(ann_root.glob("*/selection_metadata.json")):
        doc = read_json(metadata_path, {})
        selected = {
            str(item.get("organ") or ""): item
            for item in doc.get("selected_organs", []) or []
            if isinstance(item, dict) and item.get("organ")
        }
        for row in doc.get("selection_rows", []) or []:
            if not isinstance(row, dict):
                continue
            organ = str(row.get("organ") or "")
            if organ not in coverage or str(row.get("expected_presence") or "") != "expected_present":
                continue
            coverage[organ]["expected_present"] += 1
            if _has_usable_selected_label(selected, str(row.get("case_id") or ""), organ):
                coverage[organ]["usable"] += 1
    for organ, counts in coverage.items():
        expected = int(counts["expected_present"])
        usable = int(counts["usable"])
        counts["coverage_rate"] = round(float(usable / expected), 6) if expected else 1.0
    return coverage


def _unique_row_audit(run_root: Path, round_idx: int = 2) -> dict[str, Any]:
    ann_root = run_root / f"round{round_idx}" / "estep" / "annotation_versions"
    cases = []
    failures = []
    for metadata_path in sorted(ann_root.glob("*/selection_metadata.json")):
        doc = read_json(metadata_path, {})
        rows = [row for row in doc.get("selection_rows", []) or [] if isinstance(row, dict)]
        organs = [str(row.get("organ") or "") for row in rows]
        unique = len(set(organs))
        case = {
            "case_id": str(doc.get("case_id") or metadata_path.parent.name),
            "rows": len(rows),
            "unique_organs": unique,
        }
        cases.append(case)
        if len(rows) != 373 or unique != 373:
            failures.append(case)
    return {
        "status": "success" if not failures else "failed",
        "cases": cases,
        "failures": failures,
    }


def repair_round2_gate(
    run_root: Path = DEFAULT_RUN_ROOT,
    round1_selected_root: Path = DEFAULT_ROUND1_SELECTED_ROOT,
    *,
    apply: bool = False,
    round_idx: int = 2,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    round1_selected_root = round1_selected_root.resolve()
    ann_root = run_root / f"round{round_idx}" / "estep" / "annotation_versions"
    before = _coverage_from_metadata(run_root, round_idx)
    case_reports = []
    changed_docs = 0
    for metadata_path in sorted(ann_root.glob("*/selection_metadata.json")):
        case_report, changed_doc = _repair_case(
            metadata_path,
            run_root,
            round1_selected_root,
            apply=apply,
        )
        if changed_doc is not None:
            changed_docs += 1
        case_reports.append(case_report)
    after = _coverage_from_metadata(run_root, round_idx) if apply else before
    repaired = [
        {"case_id": case["case_id"], **item}
        for case in case_reports
        for item in case.get("repaired", [])
    ]
    skipped = [
        {"case_id": case["case_id"], **item}
        for case in case_reports
        for item in case.get("skipped", [])
    ]
    report = {
        "stage": "repair_round2_gate_from_existing_estep",
        "status": "success",
        "dry_run": not apply,
        "applied": apply,
        "policy_version": POLICY_VERSION,
        "run_root": str(run_root),
        "round": round_idx,
        "annotation_versions": str(ann_root),
        "round1_selected_root": str(round1_selected_root),
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "shapekit_rerun": False,
        "repaired_count": len(repaired),
        "changed_metadata_files": changed_docs,
        "repaired": repaired,
        "skipped": skipped,
        "coverage_before": before,
        "coverage_after": after,
        "unique_row_audit": _unique_row_audit(run_root, round_idx),
        "backup_root": str(run_root / f"round{round_idx}" / "estep" / "repair_backups" / BACKUP_NAME),
    }
    report_path = run_root / f"round{round_idx}" / "estep" / "carry_forward_repair_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--round1-selected-root", type=Path, default=DEFAULT_ROUND1_SELECTED_ROOT)
    parser.add_argument("--round", dest="round_idx", type=int, default=2)
    parser.add_argument("--apply", action="store_true", help="Apply the repair. Omit for dry-run report only.")
    args = parser.parse_args()

    report = repair_round2_gate(
        run_root=args.run_root,
        round1_selected_root=args.round1_selected_root,
        apply=bool(args.apply),
        round_idx=args.round_idx,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
