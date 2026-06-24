from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_utils import read_json, write_json
from .paths import resolve_path


def _norm(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (text or "")).strip("_")


def _load_global_space(path: str | Path) -> dict[str, Any]:
    data = read_json(resolve_path(path), default=None)
    if not isinstance(data, dict) or "organ_to_id" not in data:
        raise FileNotFoundError(f"Missing global label space: {resolve_path(path)}")
    return data


def _load_alias_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = read_json(resolve_path(path), default={})
    return data if isinstance(data, dict) else {}


def _load_nifti():
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("segment-all requires nibabel and numpy to merge masks") from exc
    return nib, np


def _affine_close(a, b) -> bool:
    try:
        import numpy as np
        return bool(np.allclose(a, b, atol=1e-3))
    except Exception:
        return False


def _resample_mask_to_base(img, base_img):
    """Resample a binary mask to the base image grid with nearest neighbor."""
    try:
        from nibabel.processing import resample_from_to
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("nibabel.processing.resample_from_to is required for geometry resampling") from exc

    return resample_from_to(img, base_img, order=0)


def _reverse_supported_aliases(model_entry: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for local_name, global_name in (model_entry.get("supported_organ_aliases", {}) or {}).items():
        out[str(global_name)] = str(local_name)
    return out


def _find_local_name_for_global(
    global_organ: str,
    available_names: list[str],
    model_aliases: dict[str, Any],
    model_entry: dict[str, Any],
) -> tuple[str | None, str]:
    available_set = set(available_names)
    direct_candidates = [global_organ, global_organ.lower(), global_organ.upper()]
    reverse_supported = _reverse_supported_aliases(model_entry)
    if global_organ in reverse_supported:
        direct_candidates.insert(0, reverse_supported[global_organ])

    aliases_map = (model_aliases.get("local_to_global", {}) or {})
    mapping_types = model_aliases.get("mapping_types", {}) or {}
    approved_union_locals = [
        str(local_name) for local_name, mapped_global in aliases_map.items()
        if mapped_global == global_organ and mapping_types.get(local_name) == "approved_union"
    ]
    if approved_union_locals:
        return (global_organ, "approved_union") if global_organ in available_set else (None, "approved_union_incomplete")
    reverse_manual = {str(v): str(k) for k, v in aliases_map.items()}
    if global_organ in reverse_manual:
        direct_candidates.insert(0, reverse_manual[global_organ])

    for candidate in direct_candidates:
        if candidate in available_set:
            return candidate, "direct"

    global_norm = _norm(global_organ)
    for name in available_names:
        if _norm(name) == global_norm:
            return name, "normalized"

    for local_name, mapped_global in aliases_map.items():
        if mapped_global == global_organ and local_name in available_set:
            return local_name, "manual_alias"

    supported = model_entry.get("supported_organ_aliases", {}) or {}
    for local_name, mapped_global in supported.items():
        if mapped_global == global_organ and local_name in available_set:
            return local_name, "supported_alias"

    return None, "unresolved"


def _materialize_approved_union(seg_dir: Path, global_organ: str, model_aliases: dict[str, Any]) -> dict[str, Any] | None:
    aliases_map = model_aliases.get("local_to_global", {}) or {}
    mapping_types = model_aliases.get("mapping_types", {}) or {}
    local_names = [
        str(local_name) for local_name, mapped_global in aliases_map.items()
        if mapped_global == global_organ and mapping_types.get(local_name) == "approved_union"
    ]
    if not local_names:
        return None
    paths = [seg_dir / f"{name}.nii.gz" for name in local_names]
    missing = [name for name, path in zip(local_names, paths) if not path.exists()]
    if missing:
        return {"status": "incomplete", "local_names": local_names, "missing": missing}
    nib, np = _load_nifti()
    from nibabel.processing import resample_from_to

    base = nib.load(str(paths[0]))
    union = np.zeros(base.shape, dtype=np.uint8)
    for path in paths:
        image = nib.load(str(path))
        if image.shape != base.shape or not _affine_close(image.affine, base.affine):
            image = resample_from_to(image, base, order=0)
        union |= (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
    output = seg_dir / f"{global_organ}.nii.gz"
    nib.save(nib.Nifti1Image(union, base.affine, base.header), str(output))
    return {"status": "success", "local_names": local_names, "output": str(output)}


def merge_case_segmentations(
    case_root: str | Path,
    route_result: dict[str, Any],
    registry: dict[str, Any],
    global_label_space_path: str | Path = "configs/global_label_space.json",
    alias_config_path: str | Path | None = "configs/model_label_aliases.json",
    write_per_organ_masks: bool = True,
) -> dict[str, Any]:
    nib, np = _load_nifti()

    case_root = Path(case_root).resolve()
    per_model_root = case_root / "per_model"
    segmentations_root = case_root / "segmentations"
    segmentations_root.mkdir(parents=True, exist_ok=True)

    global_space = _load_global_space(global_label_space_path)
    alias_config = _load_alias_config(alias_config_path)
    organ_to_id = global_space.get("organ_to_id", {}) or {}
    ranked_candidates = route_result.get("ranked_candidates", {}) or {}

    chosen: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing_masks: list[dict[str, Any]] = []
    geometry_errors: list[dict[str, Any]] = []

    base_img = None
    base_shape = None
    base_affine = None
    unified = None

    for organ in route_result.get("requested_organs", []):
        candidates = ranked_candidates.get(organ, []) or []
        if not candidates:
            unresolved.append({"organ": organ, "reason": "No enabled routed candidates"})
            continue

        selected = None
        for rank, candidate in enumerate(candidates, start=1):
            model_key = candidate.get("model_key")
            if not model_key:
                continue
            model_dir = per_model_root / model_key
            seg_dir = model_dir / "segmentations"
            if not seg_dir.exists():
                missing_masks.append({"organ": organ, "model_key": model_key, "reason": "segmentations folder missing"})
                continue

            model_entry = (registry.get("models", {}) or {}).get(model_key, {})
            available_files = sorted(seg_dir.glob("*.nii.gz"))
            available_names = [p.name[:-7] for p in available_files]
            model_aliases = (alias_config.get("models", {}) or {}).get(model_key, {})
            skip_reason = (model_aliases.get("skip_global_organs", {}) or {}).get(organ)
            if skip_reason:
                skipped.append({
                    "organ": organ,
                    "model_key": model_key,
                    "rank": rank,
                    "reason": str(skip_reason),
                    "policy": "skip_unresolvable_coarse_label",
                })
                selected = "skipped"
                break
            union_status = _materialize_approved_union(seg_dir, organ, model_aliases)
            if union_status and union_status.get("status") != "success":
                missing_masks.append({
                    "organ": organ,
                    "model_key": model_key,
                    "reason": "Approved union is missing required components",
                    "union_status": union_status,
                })
                continue
            if union_status:
                available_files = sorted(seg_dir.glob("*.nii.gz"))
                available_names = [p.name[:-7] for p in available_files]
            local_name, match_mode = _find_local_name_for_global(organ, available_names, model_aliases, model_entry)
            if not local_name:
                missing_masks.append({
                    "organ": organ,
                    "model_key": model_key,
                    "reason": "No matching local mask name found",
                    "available_local_names_sample": available_names[:20],
                })
                continue

            mask_path = seg_dir / f"{local_name}.nii.gz"
            if not mask_path.exists():
                missing_masks.append({
                    "organ": organ,
                    "model_key": model_key,
                    "reason": "Resolved local mask path does not exist",
                    "local_name": local_name,
                })
                continue

            img = nib.load(str(mask_path))
            arr = np.asanyarray(img.dataobj)
            if int(arr.sum()) <= 0:
                missing_masks.append({
                    "organ": organ,
                    "model_key": model_key,
                    "reason": "Mask exists but is empty",
                    "local_name": local_name,
                })
                continue

            if unified is None:
                base_img = img
                base_shape = arr.shape
                base_affine = img.affine.copy()
                unified = np.zeros(base_shape, dtype="uint16")
            resampled = False
            original_shape = arr.shape
            if arr.shape != base_shape or not _affine_close(img.affine, base_affine):
                if base_img is None:
                    geometry_errors.append({
                        "organ": organ,
                        "model_key": model_key,
                        "local_name": local_name,
                        "shape": list(arr.shape),
                        "expected_shape": list(base_shape) if base_shape else None,
                        "reason": "Geometry mismatch before base image was initialized",
                    })
                    continue
                try:
                    img = _resample_mask_to_base(img, base_img)
                    arr = np.asanyarray(img.dataobj)
                    resampled = True
                except Exception as exc:
                    geometry_errors.append({
                        "organ": organ,
                        "model_key": model_key,
                        "local_name": local_name,
                        "shape": list(original_shape),
                        "expected_shape": list(base_shape),
                        "reason": f"Geometry mismatch and nearest-neighbor resampling failed: {exc}",
                    })
                    continue

            selected = {
                "organ": organ,
                "model_key": model_key,
                "rank": rank,
                "local_name": local_name,
                "mask_path": str(mask_path),
                "match_mode": match_mode,
                "voxels": int(arr.sum()),
                "resampled_to_base": resampled,
                "original_shape": list(original_shape),
                "merged_shape": list(arr.shape),
                "array": arr,
            }
            break

        if selected == "skipped":
            continue

        if not selected:
            unresolved.append({"organ": organ, "reason": "No non-empty compatible mask found"})
            continue

        organ_id = organ_to_id.get(organ)
        if organ_id is None:
            unresolved.append({"organ": organ, "reason": "Organ missing from global_label_space"})
            continue

        chosen[organ] = {k: v for k, v in selected.items() if k != "array"}
        mask_bool = selected["array"] > 0
        fresh = mask_bool & (unified == 0)
        overlap_voxels = int(mask_bool.sum() - fresh.sum())
        unified[fresh] = int(organ_id)
        chosen[organ]["written_voxels"] = int(fresh.sum())
        chosen[organ]["overlap_voxels_discarded"] = overlap_voxels

        if write_per_organ_masks and base_img is not None:
            organ_mask = fresh.astype("uint8")
            nib.save(
                nib.Nifti1Image(organ_mask, base_img.affine, base_img.header),
                str(segmentations_root / f"{organ}.nii.gz"),
            )

    if unified is None or base_img is None:
        report = {
            "status": "failed",
            "reason": "No mergeable masks were found",
            "case_root": str(case_root),
            "unresolved_organs": unresolved,
            "skipped_organs": skipped,
            "missing_masks": missing_masks,
            "geometry_errors": geometry_errors,
        }
        write_json(case_root / "merge_report.json", report)
        return report

    unified_path = case_root / "unified_labels.nii.gz"
    nib.save(nib.Nifti1Image(unified, base_img.affine, base_img.header), str(unified_path))

    coverage = {
        "requested_organs": len(route_result.get("requested_organs", [])),
        "merged_organs": len(chosen),
        "unresolved_organs": len(unresolved),
        "skipped_organs": len(skipped),
        "geometry_error_organs": len(geometry_errors),
        "total_labeled_voxels": int((unified > 0).sum()),
    }
    report = {
        "status": "success" if not unresolved and not geometry_errors else "partial_success",
        "case_root": str(case_root),
        "unified_labels": str(unified_path),
        "coverage_summary": coverage,
        "selected_organs": chosen,
        "unresolved_organs": unresolved,
        "skipped_organs": skipped,
        "missing_masks": missing_masks,
        "geometry_errors": geometry_errors,
        "global_label_space": str(resolve_path(global_label_space_path)),
        "alias_config": str(resolve_path(alias_config_path)) if alias_config_path else None,
    }
    write_json(case_root / "merge_report.json", report)
    return report


def build_runtime_alias_report(
    case_root: str | Path,
    route_result: dict[str, Any],
    registry: dict[str, Any],
    alias_config_path: str | Path | None = "configs/model_label_aliases.json",
) -> str:
    case_root = Path(case_root).resolve()
    per_model_root = case_root / "per_model"
    alias_config = _load_alias_config(alias_config_path)
    report: dict[str, Any] = {"models": {}}

    for model_key in route_result.get("selected_model_keys", []):
        seg_dir = per_model_root / model_key / "segmentations"
        available = sorted(p.name[:-7] for p in seg_dir.glob("*.nii.gz")) if seg_dir.exists() else []
        model_entry = (registry.get("models", {}) or {}).get(model_key, {})
        model_aliases = (alias_config.get("models", {}) or {}).get(model_key, {})
        resolved = {}
        unresolved = []
        skipped = []
        for organ, candidates in (route_result.get("ranked_candidates", {}) or {}).items():
            if not any(c.get("model_key") == model_key for c in candidates):
                continue
            skip_reason = (model_aliases.get("skip_global_organs", {}) or {}).get(organ)
            if skip_reason:
                skipped.append({
                    "organ": organ,
                    "reason": str(skip_reason),
                    "policy": "skip_unresolvable_coarse_label",
                })
                continue
            local_name, mode = _find_local_name_for_global(organ, available, model_aliases, model_entry)
            if local_name:
                resolved[organ] = {"local_name": local_name, "mode": mode}
            else:
                unresolved.append(organ)
        report["models"][model_key] = {
            "available_local_names": available,
            "resolved_organs": resolved,
            "unresolved_organs": unresolved,
            "skipped_organs": skipped,
        }

    return write_json(case_root / "runtime_alias_report.json", report)
