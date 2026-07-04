from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .organ_taxonomy import load_taxonomy, normalize_canonical_id
from .student_postprocess import (
    _component_filter,
    _component_policy,
    build_parent_roi,
    component_stats,
    containment_rule_for_organ,
    load_yaml,
    save_mask,
)


def roi_bounds(roi: np.ndarray) -> tuple[slice, slice, slice]:
    points = np.argwhere(roi > 0)
    if not len(points):
        raise ValueError("empty ROI")
    lo = points.min(axis=0)
    hi = points.max(axis=0) + 1
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))  # type: ignore[return-value]


def crop_image(image: nib.Nifti1Image, bounds: tuple[slice, slice, slice]) -> nib.Nifti1Image:
    data = np.asanyarray(image.dataobj)[bounds]
    start = np.array([s.start or 0 for s in bounds], dtype=float)
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(image.affine, start)
    header = image.header.copy()
    header.set_data_shape(data.shape)
    return nib.Nifti1Image(data, affine, header)


def paste_mask(
    cropped_mask: nib.Nifti1Image,
    reference: nib.Nifti1Image,
    bounds: tuple[slice, slice, slice],
) -> np.ndarray:
    expected_shape = tuple(s.stop - (s.start or 0) for s in bounds)
    if cropped_mask.shape[:3] != expected_shape:
        target_affine = crop_image(reference, bounds).affine
        cropped_mask = resample_from_to(cropped_mask, (expected_shape, target_affine), order=0)
    out = np.zeros(reference.shape[:3], dtype=bool)
    out[bounds] = np.asanyarray(cropped_mask.dataobj) > 0
    return out


def _strict_parent_containment(organ: str) -> bool:
    organ = normalize_canonical_id(organ)
    return organ.startswith("hepatic_segment_") or organ.startswith("liver_segment_")


def _candidate_parent_paths(root: Path, case_id: str, parent: str) -> list[Path]:
    return [
        root / case_id / "updated" / f"{parent}.nii.gz",
        root / case_id / f"{parent}.nii.gz",
        root / "cases" / case_id / "updated" / f"{parent}.nii.gz",
        root / "cases" / case_id / "final" / f"{parent}.nii.gz",
        root / "cases" / case_id / f"{parent}.nii.gz",
    ]


def reliable_parent(
    *,
    roots: list[tuple[str, Path]],
    case_id: str,
    parent: str,
    reference: nib.Nifti1Image,
    allowed_student_parents: set[str],
) -> tuple[Path | None, dict[str, Any]]:
    parent = normalize_canonical_id(parent)
    rejected: list[dict[str, str]] = []
    for source, root in roots:
        if source == "student_allowlisted" and parent not in allowed_student_parents:
            rejected.append({"source": source, "reason": "student_parent_not_allowlisted"})
            continue
        for path in _candidate_parent_paths(root, case_id, parent):
            if not path.is_file():
                continue
            try:
                image = nib.load(str(path))
                if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4):
                    image = resample_from_to(image, reference, order=0)
                arr = np.asanyarray(image.dataobj) > 0
            except Exception as exc:
                rejected.append({"source": source, "path": str(path), "reason": f"invalid_nifti:{exc}"})
                continue
            ratio = float(arr.mean())
            if not arr.any():
                rejected.append({"source": source, "path": str(path), "reason": "empty"})
                continue
            if ratio >= 0.95:
                rejected.append({"source": source, "path": str(path), "reason": "implausible_volume"})
                continue
            return path, {
                "status": "success",
                "source": source,
                "path": str(path),
                "voxel_count": int(arr.sum()),
                "image_fraction": ratio,
            }
    return None, {"status": "failed", "reason": "no_reliable_parent", "rejected": rejected}


def _read_allowlisted_organs(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("organs", doc if isinstance(doc, list) else [])
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return {
        normalize_canonical_id(str(row.get("organ") or ""))
        for row in rows
        if str(row.get("decision") or row.get("status") or "").lower() in {"allow", "allowed", "pass"}
    }


def cascade_case(
    *,
    case_id: str,
    ct_path: Path,
    output_root: Path,
    inference: Callable[[Path, str, Path], dict[str, Any]],
    teacher_parent_roots: list[Path],
    student_parent_root: Path | None,
    student_parent_allowlist: Path | None,
    taxonomy_path: Path,
    policy_path: Path,
    organs: Iterable[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run true parent-ROI inference for child organs.

    ``inference`` receives ``(cropped_ct_path, organ, cropped_output_dir)`` and
    must materialize ``cropped_output_dir/<organ>.nii.gz``.
    """
    taxonomy = load_taxonomy(taxonomy_path)
    policy = load_yaml(policy_path)
    reference = nib.load(str(ct_path))
    allowed_student_parents = _read_allowlisted_organs(student_parent_allowlist)
    roots = [("selected_teacher", p) for p in teacher_parent_roots]
    if student_parent_root is not None:
        roots.append(("student_allowlisted", student_parent_root))
    case_out = output_root / case_id
    crop_root = output_root / "_roi_work" / case_id
    roi_root = output_root / "parent_rois" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    prior_audit = case_out / "student_cascade_audit.json"
    requested_organs = {normalize_canonical_id(x) for x in organs}
    if prior_audit.is_file() and not dry_run:
        try:
            prior = json.loads(prior_audit.read_text(encoding="utf-8"))
            prior_rows = prior.get("rows", [])
            prior_organs = {str(row.get("organ") or "") for row in prior_rows}
            outputs_complete = all(
                row.get("status") == "blocked"
                or (row.get("status") == "success" and (case_out / f"{row.get('organ')}.nii.gz").is_file())
                for row in prior_rows
                if row.get("organ") in requested_organs
            )
            if requested_organs == prior_organs and outputs_complete:
                prior["status"] = "skipped_existing"
                return prior
        except Exception:
            pass
    rows: list[dict[str, Any]] = []

    for requested in requested_organs:
        organ = normalize_canonical_id(requested)
        rule = containment_rule_for_organ(organ, taxonomy, policy)
        row: dict[str, Any] = {
            "case_id": case_id,
            "organ": organ,
            "status": "blocked",
            "confidence": "unavailable",
            "containment_source": rule.source,
            "parents": list(rule.parents),
            "margin_mm": rule.margin_mm,
        }
        if not rule.enabled or not rule.parents:
            row["reason"] = "no_parent_roi_rule"
            rows.append(row)
            continue
        parent_paths: dict[str, Path] = {}
        parent_audit: dict[str, Any] = {}
        for parent in rule.parents:
            path, audit = reliable_parent(
                roots=roots,
                case_id=case_id,
                parent=parent,
                reference=reference,
                allowed_student_parents=allowed_student_parents,
            )
            parent_audit[parent] = audit
            if path is not None:
                parent_paths[parent] = path
        row["parent_audit"] = parent_audit
        if len(parent_paths) != len(rule.parents):
            row["reason"] = "missing_or_unreliable_parent"
            rows.append(row)
            continue
        roi, roi_audit = build_parent_roi(parent_paths, reference, rule.margin_mm)
        row["roi_audit"] = roi_audit
        if roi is None:
            row["reason"] = "empty_parent_roi"
            rows.append(row)
            continue
        bounds = roi_bounds(roi)
        row["crop_bounds_ijk"] = [[s.start, s.stop] for s in bounds]
        row["parent_sources"] = {p: parent_audit[p]["source"] for p in parent_paths}
        if dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue
        roi_root.mkdir(parents=True, exist_ok=True)
        save_mask(roi, reference, roi_root / f"{organ}_allowed_roi.nii.gz")
        organ_work = crop_root / organ
        organ_work.mkdir(parents=True, exist_ok=True)
        cropped_ct_path = organ_work / "ct_roi.nii.gz"
        nib.save(crop_image(reference, bounds), str(cropped_ct_path))
        infer_out = organ_work / "prediction"
        infer_out.mkdir(parents=True, exist_ok=True)
        infer_result = inference(cropped_ct_path, organ, infer_out)
        predicted_path = infer_out / f"{organ}.nii.gz"
        row["inference_status"] = infer_result.get("status")
        if not predicted_path.is_file():
            row["reason"] = "roi_inference_mask_missing"
            rows.append(row)
            continue
        full = paste_mask(nib.load(str(predicted_path)), reference, bounds)
        full &= roi
        if _strict_parent_containment(organ):
            parent_img = nib.load(str(next(iter(parent_paths.values()))))
            if parent_img.shape[:3] != reference.shape[:3] or not np.allclose(parent_img.affine, reference.affine, atol=1e-4):
                parent_img = resample_from_to(parent_img, reference, order=0)
            full &= np.asanyarray(parent_img.dataobj) > 0
        before = component_stats(full)
        filtered, filter_audit = _component_filter(full, **_component_policy(rule, policy))
        after = component_stats(filtered)
        destination = case_out / f"{organ}.nii.gz"
        save_mask(filtered, reference, destination)
        row.update({
            "status": "success",
            "output_path": str(destination),
            "voxels_before_component_filter": int(full.sum()),
            "voxels_after": int(filtered.sum()),
            "outside_roi_voxels": int(np.logical_and(filtered, ~roi).sum()),
            "component_count_before": before.get("component_count"),
            "component_count_after": after.get("component_count"),
            "component_filter": filter_audit,
        })
        rows.append(row)

    summary = {
        "stage": "parent_roi_cascaded_student_inference",
        "case_id": case_id,
        "ct_path": str(ct_path),
        "status": "dry_run" if dry_run else "success",
        "confidence_availability": "unavailable_binary_backend",
        "rows": rows,
        "success_count": sum(r["status"] == "success" for r in rows),
        "blocked_count": sum(r["status"] == "blocked" for r in rows),
    }
    (case_out / "student_cascade_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
