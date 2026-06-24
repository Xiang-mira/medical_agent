from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


HIERARCHICAL_PIPELINE_VERSION = "1.4"
MIN_CROSS_PARENT_SUPPORT_FRACTION = 0.95


@dataclass(frozen=True)
class BoundingBox:
    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def volume(self) -> int:
        return int(np.prod([max(0, b - a) for a, b in zip(self.start, self.stop)]))

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            tuple(min(a, b) for a, b in zip(self.start, other.start)),
            tuple(max(a, b) for a, b in zip(self.stop, other.stop)),
        )


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def mask_bbox(mask_path: str | Path, reference_path: str | Path, margin_mm: float = 20.0) -> BoundingBox | None:
    reference = nib.load(str(reference_path))
    mask = nib.load(str(mask_path))
    if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, atol=1e-4):
        mask = resample_from_to(mask, reference, order=0)
    array = np.asanyarray(mask.dataobj) > 0
    coords = np.argwhere(array)
    if coords.size == 0:
        return None
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    margin_voxels = np.ceil(float(margin_mm) / np.maximum(spacing, 1e-6)).astype(int)
    start = np.maximum(coords.min(axis=0) - margin_voxels, 0)
    stop = np.minimum(coords.max(axis=0) + 1 + margin_voxels, np.asarray(reference.shape[:3]))
    return BoundingBox(tuple(int(x) for x in start), tuple(int(x) for x in stop))


def crop_nifti(image_path: str | Path, bbox: BoundingBox, output_path: str | Path) -> dict[str, Any]:
    image = nib.load(str(image_path))
    slices = tuple(slice(a, b) for a, b in zip(bbox.start, bbox.stop))
    cropped = image.slicer[slices]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(cropped, str(output))
    return {
        "path": str(output.resolve()),
        "start": list(bbox.start),
        "stop": list(bbox.stop),
        "shape": list(cropped.shape),
        "affine": cropped.affine.tolist(),
    }


def restore_mask_to_full(
    cropped_mask_path: str | Path,
    cropped_image_path: str | Path,
    full_image_path: str | Path,
    bbox: BoundingBox,
    output_path: str | Path,
) -> dict[str, Any]:
    cropped_reference = nib.load(str(cropped_image_path))
    prediction = nib.load(str(cropped_mask_path))
    resampled = prediction.shape != cropped_reference.shape or not np.allclose(prediction.affine, cropped_reference.affine, atol=1e-4)
    if resampled:
        prediction = resample_from_to(prediction, cropped_reference, order=0)
    full_reference = nib.load(str(full_image_path))
    full = np.zeros(full_reference.shape[:3], dtype=np.uint8)
    values = (np.asanyarray(prediction.dataobj) > 0).astype(np.uint8)
    expected_shape = tuple(b - a for a, b in zip(bbox.start, bbox.stop))
    if values.shape[:3] != expected_shape:
        raise ValueError(f"ROI prediction shape {values.shape} does not match bbox shape {expected_shape}")
    full[tuple(slice(a, b) for a, b in zip(bbox.start, bbox.stop))] = values
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, full_reference.affine, full_reference.header), str(output))
    return {"path": str(output.resolve()), "voxels": int(full.sum()), "resampled_to_crop": resampled}


def _merge_parent_groups(
    groups: list[dict[str, Any]],
    full_volume: int,
    max_full_ratio: float,
    max_union_ratio: float,
) -> list[dict[str, Any]]:
    work = [dict(group) for group in groups]
    changed = True
    while changed and len(work) > 1:
        changed = False
        best: tuple[float, int, int, BoundingBox] | None = None
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                union = work[i]["bbox"].union(work[j]["bbox"])
                separate = work[i]["bbox"].volume + work[j]["bbox"].volume
                if union.volume > full_volume * max_full_ratio or union.volume > separate * max_union_ratio:
                    continue
                score = union.volume / max(separate, 1)
                if best is None or score < best[0]:
                    best = (score, i, j, union)
        if best is not None:
            _, i, j, union = best
            merged = {
                "bbox": union,
                "parents": sorted(set(work[i]["parents"] + work[j]["parents"])),
                "organs": sorted(set(work[i]["organs"] + work[j]["organs"])),
                "parent_masks": sorted(set(work[i]["parent_masks"] + work[j]["parent_masks"])),
                "organ_support_bboxes": {
                    **work[i].get("organ_support_bboxes", {}),
                    **work[j].get("organ_support_bboxes", {}),
                },
            }
            work = [g for index, g in enumerate(work) if index not in {i, j}] + [merged]
            changed = True
    return work


def plan_roi_tasks(
    *,
    ct_path: str | Path,
    child_routes: list[dict[str, Any]],
    parent_masks: dict[str, Path],
    margin_mm: float = 20.0,
    max_full_ratio: float = 0.60,
    max_union_ratio: float = 1.50,
    allow_cross_parent_merge: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image = nib.load(str(ct_path))
    full_volume = int(np.prod(image.shape[:3]))
    by_model: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    blocked: list[dict[str, Any]] = []
    for route in child_routes:
        organ = str(route["organ"])
        parents = tuple(str(x) for x in route.get("parent_ids", []))
        available = [parent_masks[p] for p in parents if p in parent_masks]
        missing = [p for p in parents if p not in parent_masks]
        if missing or not available:
            blocked.append({"organ": organ, "status": "blocked_by_parent", "missing_parents": missing or list(parents)})
            continue
        bbox: BoundingBox | None = None
        for parent_mask in available:
            current = mask_bbox(parent_mask, ct_path, margin_mm=margin_mm)
            if current is not None:
                bbox = current if bbox is None else bbox.union(current)
        if bbox is None:
            blocked.append({"organ": organ, "status": "blocked_by_parent", "reason": "parent_mask_empty", "parents": list(parents)})
            continue
        model = str(route["model"])
        model_groups = by_model.setdefault(model, {})
        group = model_groups.setdefault(parents, {
            "bbox": bbox,
            "parents": list(parents),
            "organs": [],
            "parent_masks": [str(p) for p in available],
            "organ_support_bboxes": {},
        })
        group["bbox"] = group["bbox"].union(bbox)
        group["organs"].append(organ)
        group["organ_support_bboxes"][organ] = bbox

    tasks: list[dict[str, Any]] = []
    for model, keyed_groups in sorted(by_model.items()):
        groups = list(keyed_groups.values())
        merged = _merge_parent_groups(groups, full_volume, max_full_ratio, max_union_ratio) if allow_cross_parent_merge else groups
        for index, group in enumerate(sorted(merged, key=lambda x: (x["bbox"].start, x["parents"])), start=1):
            bbox = group["bbox"]
            tasks.append({
                "task_id": f"{model}__roi_{index:03d}",
                "model": model,
                "organs": sorted(set(group["organs"])),
                "parents": group["parents"],
                "parent_masks": group["parent_masks"],
                "bbox": bbox,
                "crop_voxels": bbox.volume,
                "full_voxels": full_volume,
                "crop_ratio": round(bbox.volume / max(full_volume, 1), 6),
                "organ_support_bboxes": group.get("organ_support_bboxes", {}),
            })
    return tasks, blocked


def execute_roi_tasks(
    *,
    ct_path: Path,
    tasks: list[dict[str, Any]],
    work_root: Path,
    merged_model_dirs: dict[str, Path],
    run_model: Callable[[Path, Path, str, list[str], str], dict[str, Any]],
    alias_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    def clear_stale_provenance(segmentation_dir: Path, organ: str) -> None:
        provenance_path = segmentation_dir / "identity_provenance.json"
        if not provenance_path.exists():
            return
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            organs = provenance.get("organs", {}) or {}
            if organ in organs:
                organs.pop(organ, None)
                provenance["organs"] = organs
                provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            provenance_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    for task in tasks:
        bbox: BoundingBox = task["bbox"]
        task_id = str(task["task_id"])
        model = str(task["model"])
        crop_path = work_root / "inputs" / f"{task_id}.nii.gz"
        crop_meta = crop_nifti(ct_path, bbox, crop_path)
        raw_output = work_root / "raw" / task_id
        infer = run_model(crop_path, raw_output, model, list(task["organs"]), task_id)
        seg_dir = Path(str(infer.get("segmentation_output", raw_output / task_id / "segmentations")))
        restored: list[dict[str, Any]] = []
        for organ in task["organs"]:
            destination = merged_model_dirs[model] / f"{organ}.nii.gz"
            destination.unlink(missing_ok=True)
            clear_stale_provenance(merged_model_dirs[model], organ)
            source = seg_dir / f"{organ}.nii.gz"
            source_labels = [organ]
            mapping_type = "exact_synonym"
            if not source.exists() and alias_config:
                model_aliases = ((alias_config.get("models", {}) or {}).get(model, {}) or {})
                local_to_global = model_aliases.get("local_to_global", {}) or {}
                mapping_types = model_aliases.get("mapping_types", {}) or {}
                approved = [
                    str(local) for local, global_name in local_to_global.items()
                    if str(global_name) == organ
                    and str(mapping_types.get(local, "exact_synonym")) in {"exact_synonym", "approved_union"}
                    and (seg_dir / f"{local}.nii.gz").exists()
                ]
                if approved:
                    source_labels = approved
                    mapping_type = "approved_union" if len(approved) > 1 else str(mapping_types.get(approved[0], "exact_synonym"))
                    if len(approved) == 1:
                        source = seg_dir / f"{approved[0]}.nii.gz"
                    else:
                        crop_reference = nib.load(str(crop_path))
                        union = np.zeros(crop_reference.shape, dtype=np.uint8)
                        for local in approved:
                            local_image = nib.load(str(seg_dir / f"{local}.nii.gz"))
                            if local_image.shape != crop_reference.shape or not np.allclose(local_image.affine, crop_reference.affine, atol=1e-4):
                                local_image = resample_from_to(local_image, crop_reference, order=0)
                            union |= (np.asanyarray(local_image.dataobj) > 0).astype(np.uint8)
                        source = work_root / "approved_unions" / task_id / f"{organ}.nii.gz"
                        source.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(nib.Nifti1Image(union, crop_reference.affine, crop_reference.header), str(source))
            if not source.exists():
                continue
            restore_record = {
                "organ": organ,
                "source_local_labels": source_labels,
                "mapping_type": mapping_type,
                **restore_mask_to_full(source, crop_path, ct_path, bbox, destination),
            }
            support_bbox = (task.get("organ_support_bboxes", {}) or {}).get(organ)
            if support_bbox is not None and len(task.get("parents", [])) > 1:
                restored_image = nib.load(str(destination))
                restored_array = np.asanyarray(restored_image.dataobj) > 0
                support_slices = tuple(slice(a, b) for a, b in zip(support_bbox.start, support_bbox.stop))
                support_voxels = int(restored_array[support_slices].sum())
                total_voxels = int(restored_array.sum())
                support_fraction = support_voxels / total_voxels if total_voxels else 0.0
                restore_record["parent_support_fraction"] = round(float(support_fraction), 6)
                restore_record["parent_support_minimum"] = MIN_CROSS_PARENT_SUPPORT_FRACTION
                if support_fraction < MIN_CROSS_PARENT_SUPPORT_FRACTION:
                    restore_record["status"] = "rejected_parent_support"
                    destination.unlink(missing_ok=True)
                    restored.append(restore_record)
                    continue
                restore_record["status"] = "accepted_parent_support"
            else:
                restore_record["status"] = "restored"
            restored.append(restore_record)
            provenance_path = merged_model_dirs[model] / "identity_provenance.json"
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {"organs": {}}
            except Exception:
                provenance = {"organs": {}}
            provenance.setdefault("organs", {})[organ] = {
                "source_local_labels": source_labels,
                "resolved_canonical_id": organ,
                "mapping_type": mapping_type,
                "mapping_source": "hierarchical_roi_restore",
            }
            provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append({
            **{k: v for k, v in task.items() if k not in {"bbox", "organ_support_bboxes"}},
            "bbox": {"start": list(bbox.start), "stop": list(bbox.stop)},
            "organ_support_bboxes": {
                organ: {"start": list(support.start), "stop": list(support.stop)}
                for organ, support in (task.get("organ_support_bboxes", {}) or {}).items()
            },
            "crop": crop_meta,
            "inference": infer,
            "restored_masks": restored,
        })
    return results


def write_hierarchical_manifest(path: Path, payload: dict[str, Any]) -> None:
    serializable = dict(payload)
    ct_path = Path(payload["ct_path"])
    serializable["ct_fingerprint"] = _fingerprint(ct_path) if ct_path.is_file() else {"path": str(ct_path), "status": "missing_dry_run_input"}
    parent_paths = {
        str(mask)
        for section in (payload.get("roi_tasks", []), payload.get("backup_roi_tasks", []))
        for task in section
        for mask in task.get("parent_masks", [])
        if Path(str(mask)).is_file()
    }
    serializable["parent_mask_fingerprints"] = [_fingerprint(Path(mask)) for mask in sorted(parent_paths)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
