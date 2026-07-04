from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None  # type: ignore[assignment]

from .organ_taxonomy import load_taxonomy, normalize_canonical_id, taxonomy_entry


@dataclass(frozen=True)
class ContainmentRule:
    organ: str
    enabled: bool
    parents: tuple[str, ...]
    margin_mm: float
    source: str
    organ_group: str
    comparison_family: str | None = None
    hierarchy_role: str | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if yaml is None or not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def organ_group(organ: str) -> str:
    s = normalize_canonical_id(organ)
    if any(x in s for x in ["vein", "artery", "vessel", "duct", "cava", "aorta", "postcava"]):
        return "vessel_or_duct"
    if any(x in s for x in ["stomach", "colon", "duodenum", "bowel", "intestine", "rectum"]):
        return "hollow_gi"
    if any(x in s for x in ["rib", "femur", "bone", "clavicula", "scapula", "sternum", "vertebra"]):
        return "bone_or_fragmented_structure"
    if any(x in s for x in ["liver", "spleen", "kidney", "pancreas", "lung", "adrenal"]):
        return "core_large_organ"
    if any(x in s for x in ["bladder", "prostate", "uterus", "gonad"]):
        return "partial_fov"
    return "longtail_uncertain"


def _student_cfg(policy: dict[str, Any]) -> dict[str, Any]:
    return policy.get("student_containment", {}) if isinstance(policy, dict) else {}


def _rule_override(policy: dict[str, Any], organ: str) -> dict[str, Any]:
    rules = _student_cfg(policy).get("containment_rules", {}) or {}
    return dict(rules.get(normalize_canonical_id(organ), {}) or {})


def containment_rule_for_organ(
    organ: str,
    taxonomy: dict[str, Any],
    policy: dict[str, Any],
) -> ContainmentRule:
    organ_id = normalize_canonical_id(organ)
    cfg = _student_cfg(policy)
    entry = taxonomy_entry(taxonomy, organ_id) or {}
    group = organ_group(organ_id)
    if not cfg or cfg.get("enabled") is False:
        return ContainmentRule(
            organ=organ_id,
            enabled=False,
            parents=tuple(normalize_canonical_id(x) for x in entry.get("parent_ids", []) or []),
            margin_mm=0.0,
            source="student_containment_disabled",
            organ_group=group,
            comparison_family=entry.get("comparison_family"),
            hierarchy_role=entry.get("hierarchy_role"),
        )
    override = _rule_override(policy, organ_id)
    if override.get("enabled") is False:
        return ContainmentRule(
            organ=organ_id,
            enabled=False,
            parents=tuple(str(x) for x in override.get("parents", [])),
            margin_mm=float(override.get("margin_mm", 0.0)),
            source="explicit_disabled",
            organ_group=group,
            comparison_family=entry.get("comparison_family"),
            hierarchy_role=entry.get("hierarchy_role"),
        )

    if "parents" in override:
        parents = tuple(normalize_canonical_id(x) for x in override.get("parents", []) if normalize_canonical_id(x))
        enabled = bool(parents) if override.get("enabled") is None else bool(override.get("enabled"))
        margin = float(override.get("margin_mm", cfg.get("default_roi_margin_mm", 15.0)))
        return ContainmentRule(
            organ=organ_id,
            enabled=enabled,
            parents=parents,
            margin_mm=margin,
            source="explicit_override",
            organ_group=group,
            comparison_family=entry.get("comparison_family"),
            hierarchy_role=entry.get("hierarchy_role"),
        )

    auto = cfg.get("auto_rules", {}) or {}
    disabled_major = {normalize_canonical_id(x) for x in auto.get("disabled_major_organs", []) or []}
    hierarchy_role = str(entry.get("hierarchy_role") or "")
    comparison_family = str(entry.get("comparison_family") or "")
    parents = tuple(normalize_canonical_id(x) for x in entry.get("parent_ids", []) or [] if normalize_canonical_id(x))
    enabled_roles = set(auto.get("enabled_hierarchy_roles", ["child"]) or [])
    enabled_families = set(auto.get("enabled_comparison_families", ["vessel_or_small_structure"]) or [])
    enabled = False
    source = "no_rule"
    if organ_id in disabled_major or (hierarchy_role == "major" and not parents):
        enabled = False
        source = "major_roi_provider"
    elif auto.get("inherit_taxonomy_parent_ids", True) and parents and hierarchy_role in enabled_roles:
        enabled = True
        source = "taxonomy_parent"
    elif parents and comparison_family in enabled_families:
        enabled = True
        source = "taxonomy_family"
    margins = cfg.get("group_margins_mm", {}) or {}
    margin = float(margins.get(group, cfg.get("default_roi_margin_mm", 15.0)))
    if margin <= 0:
        if enabled:
            source = "zero_margin_group"
        enabled = False
    return ContainmentRule(
        organ=organ_id,
        enabled=enabled,
        parents=parents,
        margin_mm=margin,
        source=source,
        organ_group=group,
        comparison_family=comparison_family or None,
        hierarchy_role=hierarchy_role or None,
    )


def read_mask(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    return img, np.asanyarray(img.dataobj) > 0


def _resample_to_reference(img: nib.Nifti1Image, reference: nib.Nifti1Image) -> tuple[nib.Nifti1Image, bool]:
    if img.shape[:3] == reference.shape[:3] and np.allclose(img.affine, reference.affine, atol=1e-4):
        return img, False
    return resample_from_to(img, reference, order=0), True


def save_mask(array: np.ndarray, reference: nib.Nifti1Image, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image((array > 0).astype(np.uint8), reference.affine, header), str(out))


def dilate_roi_mm(mask: np.ndarray, spacing: Iterable[float], margin_mm: float) -> np.ndarray:
    mask = mask > 0
    if margin_mm <= 0 or not mask.any():
        return mask
    if ndi is None:
        # Conservative fallback: no dilation if scipy is unavailable.
        return mask
    sampling = tuple(float(x) for x in spacing)
    distance = ndi.distance_transform_edt(~mask, sampling=sampling)
    return np.logical_or(mask, distance <= float(margin_mm))


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    vox = int((mask > 0).sum())
    if vox == 0:
        return {"component_count": 0, "largest_component_voxels": 0, "largest_component_ratio": 0.0}
    if ndi is None:
        return {"component_count": None, "largest_component_voxels": None, "largest_component_ratio": None}
    lab, n = ndi.label(mask > 0)
    if n == 0:
        return {"component_count": 0, "largest_component_voxels": 0, "largest_component_ratio": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    return {"component_count": int(n), "largest_component_voxels": largest, "largest_component_ratio": float(largest / max(vox, 1))}


def _component_filter(mask: np.ndarray, keep_top_k: int | None, min_fraction: float, min_voxels: int) -> tuple[np.ndarray, dict[str, Any]]:
    mask = mask > 0
    vox = int(mask.sum())
    before = component_stats(mask)
    if vox == 0 or ndi is None:
        return mask, {**before, "components_before": before.get("component_count"), "components_after": before.get("component_count"), "component_removed_voxels": 0}
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask, {**before, "components_before": int(n), "components_after": int(n), "component_removed_voxels": 0}
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    keep: list[int] = []
    for rank, idx in enumerate(order):
        size = int(sizes[idx])
        if keep_top_k is not None and rank >= keep_top_k:
            continue
        if size < int(min_voxels) and size / max(vox, 1) < float(min_fraction):
            continue
        keep.append(int(idx + 1))
    if not keep and len(order):
        keep = [int(order[0] + 1)]
    out = np.isin(lab, keep)
    after = component_stats(out)
    return out, {
        "components_before": int(n),
        "components_after": after.get("component_count"),
        "component_removed_voxels": int(vox - out.sum()),
        "largest_component_ratio_after_filter": after.get("largest_component_ratio"),
    }


def _component_policy(rule: ContainmentRule, policy: dict[str, Any]) -> dict[str, Any]:
    filt = (_student_cfg(policy).get("component_filter", {}) or {})
    key = "sub_organ" if rule.hierarchy_role == "child" and rule.organ_group != "vessel_or_duct" else rule.organ_group
    spec = dict(filt.get(key, filt.get("default", {})) or {})
    return {
        "keep_top_k": spec.get("keep_top_k"),
        "min_fraction": float(spec.get("min_fraction", 0.002)),
        "min_voxels": int(spec.get("min_voxels", 64)),
    }


def candidate_parent_paths(parent_roots: list[Path], case_id: str, parent: str) -> list[Path]:
    paths: list[Path] = []
    for root in parent_roots:
        paths.extend([
            root / case_id / "updated" / f"{parent}.nii.gz",
            root / case_id / f"{parent}.nii.gz",
            root / "cases" / case_id / "updated" / f"{parent}.nii.gz",
            root / "cases" / case_id / f"{parent}.nii.gz",
            root / "cases" / case_id / "final" / f"{parent}.nii.gz",
        ])
    return paths


def find_parent_masks(parent_roots: list[Path], case_id: str, parents: Iterable[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for parent in parents:
        parent_id = normalize_canonical_id(parent)
        for path in candidate_parent_paths(parent_roots, case_id, parent_id):
            if path.exists():
                found[parent_id] = path
                break
    return found


def build_parent_roi(
    parent_paths: dict[str, Path],
    reference: nib.Nifti1Image,
    margin_mm: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    union = np.zeros(reference.shape[:3], dtype=bool)
    resampled: list[str] = []
    missing_or_empty: list[str] = []
    for parent, path in parent_paths.items():
        parent_img = nib.load(str(path))
        parent_img, did_resample = _resample_to_reference(parent_img, reference)
        arr = np.asanyarray(parent_img.dataobj) > 0
        if not arr.any():
            missing_or_empty.append(parent)
        union |= arr
        if did_resample:
            resampled.append(parent)
    if not union.any():
        return None, {"status": "empty_parent_roi", "resampled_parents": resampled, "empty_parents": missing_or_empty}
    spacing = reference.header.get_zooms()[:3]
    roi = dilate_roi_mm(union, spacing, margin_mm)
    return roi, {
        "status": "success",
        "parent_voxels": int(union.sum()),
        "roi_voxels": int(roi.sum()),
        "roi_margin_mm": float(margin_mm),
        "resampled_parents": resampled,
        "empty_parents": missing_or_empty,
    }


def process_mask(
    *,
    case_id: str,
    organ: str,
    student_mask_path: Path,
    output_mask_path: Path,
    parent_roots: list[Path],
    taxonomy: dict[str, Any],
    policy: dict[str, Any],
    roi_output_path: Path | None = None,
) -> dict[str, Any]:
    organ_id = normalize_canonical_id(organ)
    rule = containment_rule_for_organ(organ_id, taxonomy, policy)
    img, raw = read_mask(student_mask_path)
    before_stats = component_stats(raw)
    row: dict[str, Any] = {
        "case_id": case_id,
        "organ": organ_id,
        "input_path": str(student_mask_path),
        "output_path": str(output_mask_path),
        "containment_enabled": rule.enabled,
        "containment_source": rule.source,
        "parents": ";".join(rule.parents),
        "organ_group": rule.organ_group,
        "hierarchy_role": rule.hierarchy_role,
        "comparison_family": rule.comparison_family,
        "roi_margin_mm": rule.margin_mm,
        "voxels_before": int(raw.sum()),
        "connected_component_count_before": before_stats.get("component_count"),
        "largest_component_ratio_before": before_stats.get("largest_component_ratio"),
    }
    if not rule.enabled:
        save_mask(raw, img, output_mask_path)
        row.update({
            "status": "copied_no_containment_rule",
            "voxels_after": int(raw.sum()),
            "false_positive_voxels_removed": 0,
            "connected_component_count_after": before_stats.get("component_count"),
            "largest_component_ratio_after": before_stats.get("largest_component_ratio"),
        })
        return row

    parent_paths = find_parent_masks(parent_roots, case_id, rule.parents)
    missing = [p for p in rule.parents if p not in parent_paths]
    row["parent_paths"] = json.dumps({k: str(v) for k, v in parent_paths.items()}, ensure_ascii=False)
    if missing:
        save_mask(raw, img, output_mask_path)
        row.update({
            "status": "warning_missing_parent_roi_copied_raw",
            "missing_parents": ";".join(missing),
            "voxels_after": int(raw.sum()),
            "false_positive_voxels_removed": 0,
            "connected_component_count_after": before_stats.get("component_count"),
            "largest_component_ratio_after": before_stats.get("largest_component_ratio"),
        })
        return row

    roi, roi_stats = build_parent_roi(parent_paths, img, rule.margin_mm)
    row.update({f"roi_{k}": v for k, v in roi_stats.items()})
    if roi is None:
        save_mask(raw, img, output_mask_path)
        row.update({
            "status": "warning_empty_parent_roi_copied_raw",
            "voxels_after": int(raw.sum()),
            "false_positive_voxels_removed": 0,
            "connected_component_count_after": before_stats.get("component_count"),
            "largest_component_ratio_after": before_stats.get("largest_component_ratio"),
        })
        return row

    clipped = np.logical_and(raw, roi)
    outside_removed = int(raw.sum() - clipped.sum())
    cp = _component_policy(rule, policy)
    filtered, filter_stats = _component_filter(clipped, **cp)
    after_stats = component_stats(filtered)
    save_mask(filtered, img, output_mask_path)
    if roi_output_path is not None:
        save_mask(roi, img, roi_output_path)
    row.update({
        "status": "postprocessed" if outside_removed or filter_stats.get("component_removed_voxels") else "postprocessed_no_voxels_removed",
        "voxels_after": int(filtered.sum()),
        "false_positive_voxels_removed": int(raw.sum() - filtered.sum()),
        "roi_outside_voxels_removed": outside_removed,
        "connected_component_count_after": after_stats.get("component_count"),
        "largest_component_ratio_after": after_stats.get("largest_component_ratio"),
        **filter_stats,
    })
    return row


def _case_ids(case_list: Path | None, input_root: Path) -> list[str]:
    if case_list and case_list.exists():
        with case_list.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            col = "case_id" if "case_id" in rows[0] else next(iter(rows[0]))
            return [str(row.get(col) or "") for row in rows if row.get(col)]
    return sorted(p.name for p in input_root.iterdir() if p.is_dir())


def process_student_root(
    *,
    input_root: Path,
    output_root: Path,
    parent_roots: list[Path],
    taxonomy_path: Path,
    policy_path: Path,
    case_list: Path | None = None,
    organs: list[str] | None = None,
    max_cases: int = 0,
    overwrite: bool = False,
    write_roi_masks: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    policy = load_yaml(policy_path)
    wanted = {normalize_canonical_id(x) for x in organs or []}
    cases = _case_ids(case_list, input_root)
    if max_cases > 0:
        cases = cases[:max_cases]
    output_root.mkdir(parents=True, exist_ok=True)
    roi_root = output_root / "parent_rois"
    rows: list[dict[str, Any]] = []
    planned = 0
    processed = 0
    skipped_existing = 0
    for case_id in cases:
        in_case = input_root / case_id
        if not in_case.exists():
            continue
        out_case = output_root / case_id
        if not dry_run:
            out_case.mkdir(parents=True, exist_ok=True)
        for src in sorted(in_case.glob("*.nii.gz")):
            source_name = src.name[:-7]
            organ = normalize_canonical_id(source_name)
            # When an explicit target list is supplied, require the on-disk
            # canonical filename. Historical ``ct_*`` aliases must not be
            # normalized into duplicate formal candidates.
            if wanted and source_name not in wanted:
                continue
            dst = out_case / f"{organ}.nii.gz"
            if dst.exists() and not overwrite and not dry_run:
                skipped_existing += 1
                continue
            planned += 1
            rule = containment_rule_for_organ(organ, taxonomy, policy)
            if dry_run:
                rows.append({
                    "case_id": case_id,
                    "organ": organ,
                    "input_path": str(src),
                    "output_path": str(dst),
                    "containment_enabled": rule.enabled,
                    "containment_source": rule.source,
                    "parents": ";".join(rule.parents),
                    "roi_margin_mm": rule.margin_mm,
                    "status": "dry_run",
                })
                continue
            if not rule.enabled:
                # Preserve the raw prediction byte-for-byte. Avoid loading and
                # rewriting hundreds of large NIfTIs that have no containment
                # rule, and make it explicit that this is not cascaded inference.
                shutil.copy2(src, dst)
                rows.append({
                    "case_id": case_id,
                    "organ": organ,
                    "input_path": str(src),
                    "output_path": str(dst),
                    "containment_enabled": False,
                    "containment_source": rule.source,
                    "parents": ";".join(rule.parents),
                    "roi_margin_mm": rule.margin_mm,
                    "status": "copied_no_containment_rule",
                })
                processed += 1
                continue
            roi_path = roi_root / case_id / f"{organ}_allowed_roi.nii.gz" if write_roi_masks and rule.enabled else None
            rows.append(process_mask(
                case_id=case_id,
                organ=organ,
                student_mask_path=src,
                output_mask_path=dst,
                parent_roots=parent_roots,
                taxonomy=taxonomy,
                policy=policy,
                roi_output_path=roi_path,
            ))
            processed += 1
        for meta in ["voxtell_student_result.json", "voxtell_student_plan.json"]:
            src_meta = in_case / meta
            if src_meta.exists() and not dry_run:
                shutil.copy2(src_meta, out_case / meta)
    summary = {
        "stage": "student_containment_postprocess",
        "status": "dry_run" if dry_run else "success",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "parent_roots": [str(x) for x in parent_roots],
        "taxonomy": str(taxonomy_path),
        "policy": str(policy_path),
        "cases": len(cases),
        "planned_masks": planned,
        "processed_masks": processed,
        "skipped_existing": skipped_existing,
        "rows": len(rows),
        "outputs": {
            "per_mask_csv": str(output_root / "student_containment_postprocess_per_mask.csv"),
            "summary_json": str(output_root / "student_containment_postprocess_summary.json"),
            "roi_root": str(roi_root) if write_roi_masks else None,
        },
    }
    if rows:
        with (output_root / "student_containment_postprocess_per_mask.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    (output_root / "student_containment_postprocess_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
