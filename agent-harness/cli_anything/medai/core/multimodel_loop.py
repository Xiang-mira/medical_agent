from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .json_utils import write_json
from .auto_fine_label import build_label_passport, passport_path_for_mask
from .label_verifier import verify_annotation
from .labelcritic_wrapper import run_labelcritic_compare
from .mstep_runner import build_training_manifest, write_mstep_config
from .model_registry import candidate_models_for_organs, load_registry, recommend_primary_models_for_organs
from .organ_model_performance import OrganModelPerformance
from .radthinking import build_reasoning_trace
from .registered_infer import run_registered_model
from .shapekit_runner import run_shapekit
from .target_space import validate_formal_373_target_space


def _read_case_list(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any((v or "").strip() for v in row.values()):
                continue
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _mask_path(seg_dir: Path, organ: str) -> Path:
    return seg_dir / f"{organ}.nii.gz"


def _load_model_label_aliases(root: Path) -> dict[str, Any]:
    path = root / "configs" / "model_label_aliases.json"
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_target_space_policy(root: Path, requested_organs: list[str]) -> dict[str, Any]:
    """Load the accepted 373-target policy so every run summary explains exclusions."""
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return {
            "status": "missing_target_config",
            "target_config": str(path),
            "requested_organs": len(requested_organs),
        }
    doc = json.loads(path.read_text(encoding="utf-8"))
    target_organs = [str(x) for x in doc.get("target_organs", [])]
    target_set = set(target_organs)
    requested_set = set(requested_organs)
    return {
        "status": "loaded",
        "target_config": str(path),
        "counts": doc.get("counts", {}),
        "accepted_current_exact_prompt_targets": len(target_organs),
        "requested_organs": len(requested_organs),
        "requested_is_full_accepted_target": requested_set == target_set,
        "requested_target_organs": sorted(requested_set & target_set),
        "requested_non_target_organs": sorted(requested_set - target_set),
        "target_organs_not_requested": sorted(target_set - requested_set),
        "policy_skipped_organs": doc.get("policy_skipped_organs", []),
        "no_enabled_route_organs": doc.get("no_enabled_route_organs", []),
        "ground_truth_status": "pseudo_label_candidate",
        "accuracy_warning": "Teacher outputs are pseudo-label candidates; this target policy does not prove true accuracy.",
    }


def _target_validation_for_run(root: Path, requested_organs: list[str]) -> dict[str, Any]:
    try:
        return validate_formal_373_target_space(
            root / "configs" / "student_3d_prompt_target_organs.json",
            requested_organs=requested_organs,
            require_full_target=len(requested_organs) == 373,
        )
    except Exception as exc:
        return {
            "stage": "formal_373_target_validation",
            "status": "failed",
            "reason": str(exc),
            "requested_organs": len(requested_organs),
        }


def _load_default_target_organs(root: Path) -> list[str]:
    """Load current accepted 373-organ target for mainline loop defaults."""
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    targets = [str(x) for x in doc.get("target_organs", [])]
    return targets or ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]


def _load_organ_prompts(root: Path) -> dict[str, str]:
    path = root / "configs" / "student_3d_prompt_target_organs.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (doc.get("organ_to_prompt", {}) or {}).items()}


def _candidate_mask_path(seg_dir: Path, organ: str, model_key: str, alias_config: dict[str, Any]) -> tuple[Path, str]:
    direct = _mask_path(seg_dir, organ)
    if direct.exists():
        return direct, "direct"

    local_to_global = (
        (alias_config.get("models", {}) or {})
        .get(model_key, {})
        .get("local_to_global", {})
        or {}
    )
    for local_name, global_name in local_to_global.items():
        if str(global_name) == organ:
            local_path = _mask_path(seg_dir, str(local_name))
            if local_path.exists():
                return local_path, f"local_alias:{local_name}"

    return direct, "missing"


def _copy_annotation(src: Path | None, dst_dir: Path, organ: str) -> str | None:
    if src and src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{organ}.nii.gz"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst.resolve())
    return None


def _copy_case_mask(src: Path, dst_case_root: Path, organ: str) -> Path | None:
    """Copy one selected organ mask into a case/segmentations layout."""
    if not src.exists():
        return None
    seg_dir = dst_case_root / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    dst = seg_dir / f"{organ}.nii.gz"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _resolve_preseeded_case_dir(seed_base: Path, case_id: str) -> Path | None:
    """Find a preseeded case mask directory across supported case layouts."""
    if "{case_id}" in str(seed_base):
        seed_base = Path(str(seed_base).replace("{case_id}", case_id))
    case_root = seed_base / case_id
    for candidate in (seed_base, case_root, case_root / "updated", case_root / "segmentations"):
        if candidate.exists() and any(candidate.glob("*.nii.gz")):
            return candidate
    return None


def _add_shapekit_calibration_masks(
    *,
    selected_case_root: Path,
    selected_organs: list[str],
    model_seg_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Add auxiliary masks required by ShapeKit but not by the final manifest.

    Some ShapeKit post-processors use the liver mask as an anatomical calibration
    standard for left/right reassignment.  We can provide that mask to ShapeKit
    without turning it into a selected pseudo-label for student training.
    """
    liver_dependent_families = {
        "femur": {"femur_left", "femur_right"},
        "kidney": {"kidney_left", "kidney_right"},
        "lung": {"lung_left", "lung_right"},
        "adrenal_gland": {"adrenal_gland_left", "adrenal_gland_right"},
    }
    requested = set(selected_organs)
    needs_liver = any(bool(requested & organs) for organs in liver_dependent_families.values())
    if not needs_liver:
        return []

    seg_dir = selected_case_root / "segmentations"
    liver_dst = seg_dir / "liver.nii.gz"
    if liver_dst.exists():
        return []

    for model_key, model_seg_dir in model_seg_dirs.items():
        liver_src = model_seg_dir / "liver.nii.gz"
        if liver_src.exists():
            copied = _copy_case_mask(liver_src, selected_case_root, "liver")
            if copied:
                return [{
                    "organ": "liver",
                    "source_model": model_key,
                    "mask": str(copied),
                    "reason": "ShapeKit calibration-only mask for liver-dependent post-processing",
                    "dataset_role": "shapekit_calibration_only",
                    "included_in_training_manifest": False,
                }]
    return []


def _prepare_candidate_shapekit_input(seg_dir: Path, input_root: Path, case_id: str) -> int:
    dst_seg = input_root / case_id / "segmentations"
    if dst_seg.exists():
        shutil.rmtree(dst_seg)
    dst_seg.mkdir(parents=True, exist_ok=True)
    count = 0
    for mask in sorted(seg_dir.glob("*.nii.gz")):
        shutil.copy2(mask, dst_seg / mask.name)
        count += 1
    return count


def _postprocess_candidate_models_with_shapekit(
    *,
    model_seg_dirs: dict[str, Path],
    case_refined: Path,
    shapekit_root: str | Path,
    case_id: str,
    enable_shapekit: bool,
    dry_run: bool,
    timeout_sec: int,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    processed_dirs: dict[str, Path] = {}
    reports: dict[str, dict[str, Any]] = {}
    if not enable_shapekit:
        for model_key, seg_dir in model_seg_dirs.items():
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "skipped_debug_only",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
            }
        return processed_dirs, reports
    if dry_run:
        for model_key, seg_dir in model_seg_dirs.items():
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "skipped_dry_run",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
            }
        return processed_dirs, reports

    for model_key, seg_dir in model_seg_dirs.items():
        input_root = case_refined / "candidate_shapekit_input" / model_key
        output_root = case_refined / "candidate_shapekit" / model_key
        copied = _prepare_candidate_shapekit_input(seg_dir, input_root, case_id)
        if copied == 0:
            processed_dirs[model_key] = seg_dir
            reports[model_key] = {
                "stage": "candidate_preselection_shapekit",
                "status": "failed",
                "reason": "candidate model produced no masks for ShapeKit input",
                "model": model_key,
                "raw_seg_dir": str(seg_dir),
                "processed_seg_dir": str(seg_dir),
                "fallback_used": True,
            }
            continue
        result = run_shapekit(
            shapekit_root,
            input_root,
            output_root,
            output_root / "logs",
            cpu_count=2,
            dry_run=False,
            auto_config=True,
            timeout_sec=min(timeout_sec, 900),
        )
        candidate_seg = output_root / case_id / "segmentations"
        has_processed_masks = candidate_seg.exists() and any(candidate_seg.glob("*.nii.gz"))
        use_processed = result.get("status") == "success" and has_processed_masks
        processed_dirs[model_key] = candidate_seg if use_processed else seg_dir
        reports[model_key] = {
            "stage": "candidate_preselection_shapekit",
            "status": "success" if use_processed else ("unsupported_target" if result.get("reason") == "No safe ShapeKit target organs detected" else "fallback_original"),
            "model": model_key,
            "raw_seg_dir": str(seg_dir),
            "processed_seg_dir": str(processed_dirs[model_key]),
            "fallback_used": not use_processed,
            "copied_masks": copied,
            "result": result,
            "reason": result.get("reason"),
        }
    return processed_dirs, reports


def _add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _pick_reference_fallback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback selection when LabelCritic is unavailable or inconclusive."""
    with_dice = [c for c in candidates if c.get("dice") is not None]
    if with_dice:
        return max(with_dice, key=lambda c: float(c.get("dice") or -1))
    return candidates[0]


def _candidate_qc_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    qc = candidate.get("candidate_qc") or {}
    return {
        "model": candidate.get("model"),
        "prediction": candidate.get("prediction"),
        "candidate_qc_status": qc.get("status", candidate.get("candidate_qc_status")),
        "candidate_qc_score": qc.get("score", candidate.get("candidate_qc_score")),
        "candidate_qc_flags": qc.get("flags", candidate.get("candidate_qc_flags", [])),
    }


def _compute_candidate_qc(
    *,
    ct: Path,
    mask: Path | None,
    organ: str,
    reference: Path | None = None,
) -> dict[str, Any]:
    """Run cheap structural QC after candidate ShapeKit and before LabelCritic."""
    flags: list[str] = []
    checks: dict[str, Any] = {
        "organ": organ,
        "ct_path": str(ct),
        "mask_path": str(mask) if mask else None,
        "reference_path": str(reference) if reference else None,
    }
    if not mask or not mask.exists():
        return {
            **checks,
            "status": "fail",
            "score": 0.0,
            "eligible_for_labelcritic": False,
            "flags": ["missing_mask"],
            "reason": "candidate mask is missing",
        }

    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        return {
            **checks,
            "status": "review",
            "score": 0.5,
            "eligible_for_labelcritic": True,
            "flags": ["candidate_qc_dependency_missing"],
            "reason": f"candidate QC dependency missing: {exc}",
        }

    try:
        mask_img = nib.load(str(mask))
        mask_arr = np.asarray(mask_img.get_fdata() > 0)
    except Exception as exc:
        return {
            **checks,
            "status": "fail",
            "score": 0.0,
            "eligible_for_labelcritic": False,
            "flags": ["unreadable_mask"],
            "reason": f"candidate mask is unreadable: {exc}",
        }

    mask_shape = tuple(int(x) for x in mask_img.shape[:3])
    checks["mask_shape"] = list(mask_shape)
    ct_img = None
    try:
        ct_img = nib.load(str(ct))
        ct_shape = tuple(int(x) for x in ct_img.shape[:3])
        checks["ct_shape"] = list(ct_shape)
        if mask_shape != ct_shape:
            flags.append("shape_mismatch_ct")
        if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-3):
            flags.append("affine_mismatch_ct")
    except Exception as exc:
        checks["ct_geometry_status"] = f"unreadable:{exc}"
        flags.append("ct_geometry_unavailable")

    voxels = int(mask_arr.sum())
    checks["mask_voxels"] = voxels
    if voxels == 0:
        flags.append("empty_mask")

    try:
        voxel_volume = float(abs(np.linalg.det(mask_img.affine[:3, :3])))
        if voxel_volume <= 0:
            voxel_volume = float(np.prod(mask_img.header.get_zooms()[:3]))
        checks["mask_volume_mm3"] = float(voxels * voxel_volume)
    except Exception:
        checks["mask_volume_mm3"] = None

    try:
        from scipy.ndimage import label as scipy_label

        _, component_count = scipy_label(mask_arr)
        checks["connected_components"] = int(component_count)
        if int(component_count) > 20:
            flags.append("many_connected_components")
    except Exception:
        checks["connected_components"] = None
        flags.append("connected_components_unavailable")

    if reference and reference.exists():
        try:
            ref_img = nib.load(str(reference))
            ref_arr = np.asarray(ref_img.get_fdata() > 0)
            ref_voxels = int(ref_arr.sum())
            checks["reference_voxels"] = ref_voxels
            if ref_voxels > 0 and voxels > 0:
                ratio = float(voxels / ref_voxels)
                checks["volume_ratio_to_reference"] = ratio
                if ratio < 0.25:
                    flags.append("volume_ratio_too_small_vs_reference")
                elif ratio > 4.0:
                    flags.append("volume_ratio_too_large_vs_reference")
        except Exception as exc:
            checks["reference_status"] = f"unreadable:{exc}"

    hard_fail_flags = {"missing_mask", "unreadable_mask", "shape_mismatch_ct", "empty_mask"}
    review_flags = {
        "affine_mismatch_ct",
        "ct_geometry_unavailable",
        "many_connected_components",
        "connected_components_unavailable",
        "volume_ratio_too_small_vs_reference",
        "volume_ratio_too_large_vs_reference",
    }
    if hard_fail_flags & set(flags):
        status = "fail"
        score = 0.0
        eligible = False
    elif review_flags & set(flags):
        status = "review"
        score = max(0.25, 1.0 - 0.15 * len(set(flags) & review_flags))
        eligible = True
    else:
        status = "pass"
        score = 1.0
        eligible = True

    return {
        **checks,
        "status": status,
        "score": float(score),
        "eligible_for_labelcritic": eligible,
        "flags": flags,
        "reason": "ok" if not flags else ";".join(flags),
    }


def _labelcritic_decision_path(records: list[dict[str, Any]] | None) -> str | None:
    for record in records or []:
        if record.get("output_json"):
            return str(record["output_json"])
    return None


def _quality_status(review_flags: list[str] | None, quality_flags: list[str] | None) -> str:
    flags = set(review_flags or []) | set(quality_flags or [])
    if not flags:
        return "ok"
    if {"missing_candidate", "missing_final_mask"} & flags:
        return "missing"
    if any(str(flag).startswith("candidate_qc_") for flag in flags):
        return "candidate_qc_review"
    if any(str(flag).startswith("shapekit_") for flag in flags):
        return "postprocess_review"
    if "selection_fallback" in flags:
        return "selection_review"
    return "review"


def _case_resume_state(
    *,
    case_out: Path,
    updated_root: Path,
    case_id: str,
    organs: list[str],
    enable_shapekit: bool,
) -> dict[str, Any]:
    pred_root = case_out / "raw_predictions"
    raw_ready = pred_root.exists() and any(pred_root.glob(f"*/{case_id}/segmentations/*.nii.gz"))
    meta_path = updated_root / case_id / "selection_metadata.json"
    updated_dir = updated_root / case_id / "updated"
    if not raw_ready:
        return {"complete": False, "raw_ready": False, "reason": "raw_predictions_missing"}
    if not meta_path.exists():
        return {"complete": False, "raw_ready": True, "reason": "selection_metadata_missing"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"complete": False, "raw_ready": True, "reason": f"selection_metadata_unreadable:{exc}"}
    selection_rows = meta.get("selection_rows") or []
    selected_organs = meta.get("selected_organs") or []
    seen = {str(row.get("organ")) for row in selection_rows if isinstance(row, dict) and row.get("organ")}
    missing_selection_rows = [organ for organ in organs if organ not in seen]
    if missing_selection_rows:
        return {
            "complete": False,
            "raw_ready": True,
            "reason": "selection_rows_incomplete",
            "missing_selection_rows": missing_selection_rows[:50],
        }
    pending = [
        item.get("organ")
        for item in selected_organs
        if isinstance(item, dict) and item.get("shapekit_status") == "pending"
    ]
    if pending:
        return {"complete": False, "raw_ready": True, "reason": "shapekit_metadata_pending", "pending_organs": pending[:50]}
    missing_final = [
        item.get("organ")
        for item in selected_organs
        if isinstance(item, dict)
        and item.get("organ")
        and not (updated_dir / f"{item['organ']}.nii.gz").exists()
    ]
    if missing_final:
        return {"complete": False, "raw_ready": True, "reason": "updated_masks_missing", "missing_final_masks": missing_final[:50]}
    if enable_shapekit:
        unknown_postprocess = [
            item.get("organ")
            for item in selected_organs
            if isinstance(item, dict)
            and item.get("shapekit_status") in {None, "", "pending"}
        ]
        if unknown_postprocess:
            return {"complete": False, "raw_ready": True, "reason": "shapekit_status_incomplete", "organs": unknown_postprocess[:50]}
    return {"complete": True, "raw_ready": True, "reason": "complete"}


def _build_gap_rows(
    *,
    case_id: str,
    ct: Path,
    organs: list[str],
    selection_rows: list[dict[str, Any]],
    selected_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_organ = {str(item.get("organ")): item for item in selected_metadata if item.get("organ")}
    selection_by_organ = {str(item.get("organ")): item for item in selection_rows if item.get("organ")}
    rows: list[dict[str, Any]] = []
    for organ in organs:
        selected = selected_by_organ.get(organ)
        selection = selection_by_organ.get(organ, {})
        if not selected:
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": "missing_final_pseudo_label",
                "reason": selection.get("reason") or "No selected final mask for requested organ",
                "candidate_count": selection.get("candidate_count", 0),
                "candidate_models": selection.get("candidate_models", []),
                "selection_method": selection.get("selection_method", "none"),
                "selection_status": selection.get("selection_status", "missing"),
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
            continue
        status = selected.get("shapekit_status")
        if status in {"unsupported_target", "fallback_original", "failed"}:
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": "shapekit_not_success",
                "reason": selected.get("shapekit_reason") or status,
                "candidate_count": selected.get("candidate_count"),
                "candidate_models": selected.get("candidate_models", []),
                "selection_method": selected.get("selection_method"),
                "selection_status": selected.get("selection_status"),
                "shapekit_status": status,
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
        qc_status = selected.get("selected_candidate_qc_status")
        if qc_status and qc_status != "pass":
            rows.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "gap_type": "candidate_qc_not_pass",
                "reason": ";".join(selected.get("selected_candidate_qc_flags", []) or []) or qc_status,
                "candidate_count": selected.get("candidate_count"),
                "candidate_models": selected.get("candidate_models", []),
                "comparison_candidate_models": selection.get("comparison_candidate_models", []),
                "selection_method": selected.get("selection_method"),
                "selection_status": selected.get("selection_status"),
                "candidate_qc_status": qc_status,
                "candidate_qc_flags": selected.get("selected_candidate_qc_flags", []),
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
            })
    return rows


def _summarize_preseeded_competition(
    *,
    preseeded_keys: list[str],
    selection_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize whether Round2+ injected sources actually joined selection."""
    if not preseeded_keys:
        return None

    per_source: dict[str, dict[str, Any]] = {}
    for key in preseeded_keys:
        candidate_entries = [
            row for row in selection_rows
            if key in (row.get("candidate_models") or [])
        ]
        missing_entries = [
            row for row in selection_rows
            if key not in (row.get("candidate_models") or [])
        ]
        selected_entries = [
            row for row in selection_rows
            if row.get("selected_model") == key
        ]
        per_source[key] = {
            "candidate_entries": len(candidate_entries),
            "missing_entries": len(missing_entries),
            "selected_entries": len(selected_entries),
            "missing_examples": [
                {
                    "case_id": row.get("case_id"),
                    "organ": row.get("organ"),
                    "candidate_models": row.get("candidate_models", []),
                }
                for row in missing_entries[:20]
            ],
        }

    return {
        "status": "computed",
        "preseeded_sources": preseeded_keys,
        "selection_entries": len(selection_rows),
        "per_source": per_source,
        "note": (
            "Round2+ preseeded sources are candidates, not automatic winners. "
            "Missing entries usually mean that source had no mask for that "
            "case-organ layout and should be reviewed before formal claims."
        ),
    }


def _select_candidate(
    *,
    ct: Path,
    organ: str,
    candidates: list[dict[str, Any]],
    out: Path,
    case_id: str,
    enable_critic: bool,
    critic_backend: str,
    critic_base_url: str,
    critic_port: int,
    timeout_sec: int,
    dry_run: bool,
    labelcritic_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the pseudo-label candidate for one organ.

    Teacher outputs are pseudo-label candidates. If multiple candidates exist,
    LabelCritic is the primary selector; Dice is only a fallback/metadata signal.
    """
    qc_rejected = [
        c for c in candidates
        if not bool(c.get("eligible_for_labelcritic", True))
    ]
    eligible_candidates = [
        c for c in candidates
        if bool(c.get("eligible_for_labelcritic", True))
    ]
    comparison_candidates = eligible_candidates or candidates
    qc_review_flags: list[str] = []
    qc_quality_flags: list[str] = []
    if qc_rejected:
        qc_review_flags.append("candidate_qc_rejected")
        qc_quality_flags.append("candidate_qc_rejected")
    if candidates and not eligible_candidates:
        qc_review_flags.append("all_candidates_failed_qc")
        qc_quality_flags.append("all_candidates_failed_qc")

    if not candidates:
        return None, {
            "selection_method": "none",
            "selection_status": "missing",
            "reason": "no candidate masks produced",
            "candidate_count": 0,
            "comparison_candidate_count": 0,
            "comparison_candidate_models": [],
            "qc_rejected_candidates": [],
            "selected_model": None,
            "selected_prediction": None,
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": None,
            "quality_flags": ["missing_candidate"],
            "review_flags": ["missing_candidate"],
        }

    if not eligible_candidates:
        selected = _pick_reference_fallback(candidates)
        return selected, {
            "selection_method": "candidate_qc_fallback",
            "selection_status": "fallback",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 0,
            "comparison_candidate_models": [],
            "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
            "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
            "selected_model": selected["model"],
            "selected_prediction": selected["prediction"],
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": "All candidates failed QC; selected only for review/fallback continuity",
            "quality_flags": ["fallback_selection", *qc_quality_flags],
            "review_flags": ["selection_fallback", *qc_review_flags],
        }

    if len(comparison_candidates) == 1:
        selected = comparison_candidates[0]
        return selected, {
            "selection_method": "single_teacher_default",
            "selection_status": "selected",
            "candidate_count": len(candidates),
            "comparison_candidate_count": 1,
            "comparison_candidate_models": [selected["model"]],
            "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
            "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
            "selected_model": selected["model"],
            "selected_prediction": selected["prediction"],
            "critic_records": [],
            "labelcritic_records": [],
            "fallback_reason": None,
            "quality_flags": ["single_candidate", *qc_quality_flags],
            "review_flags": qc_review_flags,
        }

    critic_records: list[dict[str, Any]] = []
    labelcritic_options = labelcritic_options or {}
    selected = comparison_candidates[0]
    selection_status = "selected"
    fallback_reason = None

    if enable_critic and not dry_run:
        for challenger in comparison_candidates[1:]:
            critic_out = out / "critic" / case_id / f"{organ}_{selected['model']}_vs_{challenger['model']}.json"
            critic = run_labelcritic_compare(
                ct,
                Path(selected["prediction"]),
                Path(challenger["prediction"]),
                organ,
                critic_out,
                backend=critic_backend,
                base_url=critic_base_url,
                port=critic_port,
                dry_run=False,
                timeout_sec=min(timeout_sec, 300),
                **labelcritic_options,
            )
            record = {
                "candidate_a": selected["model"],
                "candidate_b": challenger["model"],
                "output_json": str(critic_out),
                "status": critic.get("status"),
                "decision": critic.get("decision", {}),
            }
            critic_records.append(record)
            winner = (critic.get("decision", {}) or {}).get("winner")
            if critic.get("status") == "success" and winner == "b":
                selected = challenger
            elif critic.get("status") == "success" and winner == "a":
                continue
            else:
                fallback_reason = f"LabelCritic inconclusive for {selected['model']} vs {challenger['model']}"
                selected = _pick_reference_fallback(comparison_candidates)
                selection_status = "fallback"
                break
        method = "label_critic" if selection_status == "selected" else "label_critic_fallback"
    else:
        selected = _pick_reference_fallback(comparison_candidates)
        method = "critic_disabled_fallback"
        selection_status = "fallback"
        fallback_reason = "LabelCritic disabled, unavailable, or dry-run"

    return selected, {
        "selection_method": method,
        "selection_status": selection_status,
        "candidate_count": len(candidates),
        "comparison_candidate_count": len(comparison_candidates),
        "comparison_candidate_models": [c["model"] for c in comparison_candidates],
        "qc_rejected_candidates": [_candidate_qc_summary(c) for c in qc_rejected],
        "candidate_qc_policy": "hard_fail_candidates_excluded_before_labelcritic",
        "selected_model": selected["model"],
        "selected_prediction": selected["prediction"],
        "critic_records": critic_records,
        "labelcritic_records": critic_records,
        "fallback_reason": fallback_reason,
        "quality_flags": (["labelcritic_selected"] if method == "label_critic" else ["fallback_selection"]) + qc_quality_flags,
        "review_flags": ([] if selection_status == "selected" else ["selection_fallback"]) + qc_review_flags,
    }


def run_multimodel_annotation_loop(
    case_list: str | Path,
    output_folder: str | Path,
    models: list[str] | None = None,
    organs: list[str] | None = None,
    registry_path: str | Path = "configs/model_registry.yaml",
    checkpoint_map_models: bool = False,
    shapekit_root: str | Path = "third_party/ShapeKit-main",
    enable_shapekit: bool = True,
    enable_critic: bool = True,
    critic_backend: str = "labelcritic",
    critic_base_url: str = "http://localhost",
    critic_port: int = 8000,
    vlm_threshold: float = 0.5,
    accept_threshold: float = 0.8,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    device: str | None = None,
    perf_tracker_path: str | Path | None = None,
    resume: bool = True,
    preseeded_model_dirs: dict[str, Path] | None = None,
    labelcritic_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    preseeded_model_dirs: mapping of model_key -> base directory where
        per-case predictions already exist as one of:
        <base>/<case_id>/<organ>.nii.gz,
        <base>/<case_id>/updated/<organ>.nii.gz, or
        <base>/<case_id>/segmentations/<organ>.nii.gz.
        These models are injected directly into model_seg_dirs without running
        inference, allowing the previous round's selected pseudo labels and the
        previous student model to compete with teachers in the current E-step.
    """
    """Run the teacher-requested multi-model annotation refinement loop.

    Required case_list columns:
      case_id, ct_path, annotation_folder
    Optional columns:
      report_path, clinical_path, pathology_path
    """
    case_csv = Path(case_list).resolve()
    out = Path(output_folder).resolve()
    labelcritic_options = labelcritic_options or {}
    out.mkdir(parents=True, exist_ok=True)
    cases = _read_case_list(case_csv)
    registry = load_registry(registry_path)
    project_root = Path(__file__).resolve().parents[4]
    alias_config = _load_model_label_aliases(project_root)
    organ_prompts = _load_organ_prompts(project_root)
    if organs is None or not organs:
        organs = _load_default_target_organs(project_root)
    if models is None:
        models = ["mock_seg"] if dry_run else ["totalsegmentator"]
    target_validation = _target_validation_for_run(project_root, organs)
    target_blocking = target_validation.get("blocking", {}) if isinstance(target_validation, dict) else {}
    if target_blocking.get("requested_non_target_organs"):
        raise ValueError(
            "Requested organs include non-target organs for the formal 373-organ mainline: "
            + json.dumps(target_blocking.get("requested_non_target_organs"), ensure_ascii=False)
        )
    if len(organs) == 373 and target_validation.get("status") != "success":
        raise ValueError(
            "Formal 373-organ run failed target-space validation: "
            + json.dumps(target_validation.get("blocking", target_validation), ensure_ascii=False)
        )

    # Initialise organ-model performance tracker (None = disabled)
    tracker: OrganModelPerformance | None = (
        OrganModelPerformance(perf_tracker_path) if perf_tracker_path else None
    )

    dice_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    inference_results: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    all_gap_rows: list[dict[str, Any]] = []
    resume_rows: list[dict[str, Any]] = []
    updated_root = out / "annotation_versions"
    review_queue = out / "review_queue.jsonl"
    vlm_decisions = out / "vlm_decisions.jsonl"
    traces_jsonl = out / "patient_traces.jsonl"
    report_supervision_jsonl = out / "report_supervision.jsonl"

    # Reset append-only outputs for a clean run.
    for p in (review_queue, vlm_decisions, traces_jsonl, report_supervision_jsonl):
        if p.exists():
            p.unlink()

    for idx, case in enumerate(cases, start=1):
        case_id = case.get("case_id") or Path(case.get("ct_path", f"case_{idx}")).parent.name
        ct = Path(case.get("ct_path", "")).resolve()
        ref_dir = Path(case.get("annotation_folder", "")).resolve() if case.get("annotation_folder") else None
        case_out = out / "cases" / case_id
        case_raw = case_out / "raw_predictions"
        case_refined = case_out / "refined_predictions"
        case_updated = updated_root / case_id / "updated"
        case_updated.mkdir(parents=True, exist_ok=True)

        import time as _time
        _case_start = _time.time()

        # Resume only when the full E-step artifact set is complete.  Raw
        # predictions alone are not enough because a prior run may have stopped
        # before LabelCritic selection, ShapeKit, or manifest metadata.
        if resume and not dry_run:
            resume_state = _case_resume_state(
                case_out=case_out,
                updated_root=updated_root,
                case_id=case_id,
                organs=organs,
                enable_shapekit=enable_shapekit,
            )
            resume_rows.append({"case_id": case_id, **resume_state})
            if resume_state["complete"]:
                print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 已完整完成，跳过", flush=True)
                continue
            if resume_state.get("raw_ready"):
                print(
                    f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} raw 已存在但 {resume_state.get('reason')}，继续补齐后续阶段",
                    flush=True,
                )

        print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 开始推理...", flush=True)

        if not dry_run and not ct.exists():
            _append_jsonl(review_queue, {"case_id": case_id, "reason": "ct_path missing", "ct_path": str(ct)})
            continue

        # Optionally expand candidate models by organ mapping.
        case_models = list(models)
        if checkpoint_map_models:
            mapped = candidate_models_for_organs(registry, organs)
            for organ_models in mapped.values():
                for m in organ_models:
                    if m not in case_models:
                        case_models.append(m)

        model_seg_dirs: dict[str, Path] = {}

        # Inject preseeded predictions (e.g. student from previous round) directly
        # into model_seg_dirs without running inference.
        if preseeded_model_dirs:
            for seed_key, seed_base in preseeded_model_dirs.items():
                seed_seg = _resolve_preseeded_case_dir(Path(seed_base), case_id)
                if seed_seg:
                    model_seg_dirs[seed_key] = seed_seg
                    print(f"[{_time.strftime('%H:%M:%S')}]   [preseeded] {seed_key} ✓ ({sum(1 for _ in seed_seg.glob('*.nii.gz'))} masks)", flush=True)

        for model_idx, model_key in enumerate(case_models, start=1):
            print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key}...", flush=True)
            infer = run_registered_model(
                ct,
                case_raw / model_key,
                model_key,
                registry_path=registry_path,
                case_id=case_id,
                dry_run=dry_run,
                timeout_sec=timeout_sec,
                device=device,
                extra_context={"requested_organs": organs},
            )
            inference_results.append({"case_id": case_id, **infer})
            seg_dir = Path(infer.get("segmentation_output", case_raw / model_key / case_id / "segmentations"))
            if infer.get("status") in {"success", "dry_run"}:
                model_seg_dirs[model_key] = seg_dir
                print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key} ✓ ({infer.get('num_masks',0)} masks)", flush=True)
            else:
                print(f"[{_time.strftime('%H:%M:%S')}]   [{model_idx}/{len(case_models)}] {model_key} ✗ ({infer.get('status')})", flush=True)

        candidate_model_seg_dirs, candidate_shapekit_reports = _postprocess_candidate_models_with_shapekit(
            model_seg_dirs=model_seg_dirs,
            case_refined=case_refined,
            shapekit_root=shapekit_root,
            case_id=case_id,
            enable_shapekit=enable_shapekit,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )

        checked = accepted = low_dice = uncertain = updated = critic_count = 0
        selected_input_root = case_out / "selected_after_candidate_shapekit"
        selected_case_root = selected_input_root / case_id
        selected_seg_dir = selected_case_root / "segmentations"
        if selected_seg_dir.exists() and not dry_run:
            shutil.rmtree(selected_seg_dir)
        selected_metadata: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []

        for organ in organs:
            current_ref = _mask_path(ref_dir, organ) if ref_dir else None
            current_ref_exists = bool(current_ref and (not dry_run) and current_ref.exists())
            organ_rows: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []

            # Performance tracker: decide which models to run for this organ.
            # Preseeded models (e.g. student_prev) always participate regardless
            # of tracker state — they must compete to drive distillation.
            if tracker and not tracker.should_run_all(organ):
                top_models = tracker.get_top_k_models(organ, k=2)
                organ_model_seg_dirs = {
                    k: v for k, v in candidate_model_seg_dirs.items()
                    if k in top_models
                    or k.replace("_shapekit", "") in top_models
                    or k in (preseeded_model_dirs or {})
                }
                if not organ_model_seg_dirs:
                    organ_model_seg_dirs = candidate_model_seg_dirs  # fallback to all
            else:
                organ_model_seg_dirs = candidate_model_seg_dirs

            for model_key, seg_dir in organ_model_seg_dirs.items():
                raw_seg_dir = model_seg_dirs.get(model_key, seg_dir)
                raw_pred, raw_alias_match = _candidate_mask_path(raw_seg_dir, organ, model_key, alias_config)
                pred, alias_match = _candidate_mask_path(seg_dir, organ, model_key, alias_config)
                shapekit_report = candidate_shapekit_reports.get(model_key, {})
                candidate_shapekit_status = shapekit_report.get("status", "skipped_debug_only")
                candidate_shapekit_reason = shapekit_report.get("reason")
                if not pred.exists() and seg_dir != raw_seg_dir and raw_pred.exists():
                    pred = raw_pred
                    alias_match = f"post_shapekit_missing_fallback:{raw_alias_match}"
                    candidate_shapekit_status = "fallback_original"
                    candidate_shapekit_reason = "ShapeKit did not produce this organ mask; using raw candidate for LabelCritic comparison"
                pred_for_verify = pred if pred.exists() else None
                candidate_qc = _compute_candidate_qc(
                    ct=ct,
                    mask=pred_for_verify,
                    organ=organ,
                    reference=current_ref if current_ref_exists else None,
                )
                v = verify_annotation(current_ref if current_ref_exists else None, pred_for_verify, organ, dsc_replace_threshold=0.0, dsc_vlm_threshold=vlm_threshold)
                dice = v.get("dice")
                checked += 1
                row = {
                    "case_id": case_id,
                    "organ": organ,
                    "model": model_key,
                    "prediction": str(pred),
                    "pre_shapekit_prediction": str(raw_pred),
                    "reference": str(current_ref) if current_ref else "",
                    "reference_role": "prior_or_selected_pseudo_reference" if current_ref_exists else "none",
                    "metric_family": "pseudo_consistency",
                    "metric_scope": "candidate_vs_prior_or_selected_pseudo_reference",
                    "ground_truth_status": "pseudo_label_candidate",
                    "accuracy_warning": "Dice is pseudo-label consistency, not true expert-label accuracy.",
                    "dice": dice,
                    "pseudo_consistency_dice": dice,
                    "decision": v.get("decision"),
                    "status": v.get("status"),
                    "reason": v.get("reason"),
                    "reference_quality_bucket": v.get("quality_bucket"),
                    "candidate_exists": pred.exists(),
                    "alias_match": alias_match,
                    "candidate_shapekit_status": candidate_shapekit_status,
                    "candidate_shapekit_reason": candidate_shapekit_reason,
                    "candidate_shapekit_report": shapekit_report,
                    "candidate_qc": candidate_qc,
                    "candidate_qc_status": candidate_qc.get("status"),
                    "candidate_qc_score": candidate_qc.get("score"),
                    "candidate_qc_flags": candidate_qc.get("flags", []),
                    "eligible_for_labelcritic": candidate_qc.get("eligible_for_labelcritic", True),
                }
                dice_rows.append(row); organ_rows.append(row)
                if pred.exists():
                    candidates.append(row)

                # Update performance tracker with this (organ, model, pseudo-consistency Dice) observation.
                if tracker and dice is not None and not dry_run:
                    tracker.update(organ, model_key, float(dice))

            selected, selection = _select_candidate(
                ct=ct,
                organ=organ,
                candidates=candidates,
                out=out,
                case_id=case_id,
                enable_critic=enable_critic,
                critic_backend=critic_backend,
                critic_base_url=critic_base_url,
                critic_port=critic_port,
                timeout_sec=timeout_sec,
                dry_run=dry_run,
                labelcritic_options=labelcritic_options,
            )
            critic_count += len(selection.get("critic_records", []) or [])
            best_dice = selected.get("dice") if selected else None
            selected_reference_quality_bucket = selected.get("reference_quality_bucket") if selected else None
            empty_reference_nonempty_prediction = selected_reference_quality_bucket == "empty_reference_nonempty_prediction"
            if best_dice is not None and float(best_dice) < vlm_threshold and not empty_reference_nonempty_prediction:
                low_dice += 1

            selection_record = {
                "case_id": case_id,
                "ct_path": str(ct),
                "organ": organ,
                "prompt": organ_prompts.get(organ, organ.replace("_", " ")),
                "candidate_models": [c["model"] for c in candidates],
                "candidate_predictions": [
                    {
                        "model": c["model"],
                        "prediction": c["prediction"],
                        "pre_shapekit_prediction": c.get("pre_shapekit_prediction"),
                        "dice": c.get("dice"),
                        "status": c.get("status"),
                        "reason": c.get("reason"),
                        "reference_quality_bucket": c.get("reference_quality_bucket"),
                        "candidate_shapekit_status": c.get("candidate_shapekit_status"),
                        "candidate_shapekit_reason": c.get("candidate_shapekit_reason"),
                        "candidate_qc_status": c.get("candidate_qc_status"),
                        "candidate_qc_score": c.get("candidate_qc_score"),
                        "candidate_qc_flags": c.get("candidate_qc_flags", []),
                        "eligible_for_labelcritic": c.get("eligible_for_labelcritic", True),
                    }
                    for c in candidates
                ],
                "candidate_count": len(candidates),
                "reference": str(current_ref) if current_ref else "",
                "reference_role": "prior_or_selected_pseudo_reference" if current_ref_exists else "none",
                "selected_model": selection.get("selected_model"),
                "source_model": selection.get("selected_model"),
                "selected_prediction": selected.get("prediction") if selected else None,
                "selected_pre_shapekit_prediction": selected.get("pre_shapekit_prediction") if selected else None,
                "selected_candidate_shapekit_status": selected.get("candidate_shapekit_status") if selected else None,
                "selected_candidate_shapekit_reason": selected.get("candidate_shapekit_reason") if selected else None,
                "selected_candidate_qc_status": selected.get("candidate_qc_status") if selected else None,
                "selected_candidate_qc_score": selected.get("candidate_qc_score") if selected else None,
                "selected_candidate_qc_flags": selected.get("candidate_qc_flags", []) if selected else [],
                "selected_reference_quality_bucket": selected_reference_quality_bucket,
                "selected_dice": best_dice,
                "selected_pseudo_consistency_dice": best_dice,
                "metric_family": "pseudo_consistency",
                "metric_scope": "selected_candidate_vs_prior_or_selected_pseudo_reference",
                "accuracy_warning": "Selected Dice is pseudo-label consistency, not true expert-label accuracy.",
                "comparison_input_stage": "post_shapekit_candidate" if enable_shapekit and not dry_run else "raw_candidate",
                "dataset_type": "pseudo_label_dataset",
                "ground_truth_status": "pseudo_label_candidate",
                **selection,
            }
            selection_record["labelcritic_records"] = selection_record.get("critic_records", [])
            selection_record["labelcritic_decision_path"] = _labelcritic_decision_path(selection_record["labelcritic_records"])
            selection_record["label_critic_decision_path"] = selection_record["labelcritic_decision_path"]
            selection_rows.append(selection_record)
            all_selection_rows.append(selection_record)
            for critic_record in selection_record.get("labelcritic_records", []) or []:
                _append_jsonl(vlm_decisions, {
                    "case_id": case_id,
                    "ct_path": str(ct),
                    "organ": organ,
                    "selection_method": selection_record.get("selection_method"),
                    "selected_model_after_pairwise": selection_record.get("selected_model"),
                    **critic_record,
                })

            if not selected:
                _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, **selection_record})
                uncertain += 1
                continue

            copied = _copy_case_mask(Path(selected["prediction"]), selected_case_root, organ)
            if copied:
                review_flags: list[str] = list(selection_record.get("review_flags", []) or [])
                quality_flags: list[str] = list(selection_record.get("quality_flags", []) or [])
                selected_qc_status = selected.get("candidate_qc_status") if selected else None
                if selected_qc_status and selected_qc_status != "pass":
                    _add_unique(review_flags, f"candidate_qc_{selected_qc_status}")
                    _add_unique(quality_flags, f"candidate_qc_{selected_qc_status}")
                if empty_reference_nonempty_prediction:
                    _add_unique(review_flags, "empty_reference_nonempty_prediction")
                    _add_unique(quality_flags, "empty_reference_nonempty_prediction")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "prior pseudo/reference mask is empty but selected teacher prediction is non-empty; do not treat Dice=0 as true low-quality evidence",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                elif best_dice is not None and float(best_dice) < vlm_threshold:
                    _add_unique(review_flags, "low_pseudo_consistency_dice")
                    _add_unique(quality_flags, "low_pseudo_consistency_dice")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "selected pseudo-label has low pseudo-consistency Dice against available pseudo reference",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                updated += 1
                if best_dice is not None and float(best_dice) >= accept_threshold:
                    accepted += 1
                if selection.get("selection_status") != "selected":
                    uncertain += 1
                    _add_unique(review_flags, "selection_fallback")
                    _add_unique(quality_flags, "selection_fallback")
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "fallback pseudo-label selection requires review",
                        **selection_record,
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                if any(str(flag).startswith("candidate_qc_") for flag in review_flags):
                    _append_jsonl(review_queue, {
                        "case_id": case_id,
                        "organ": organ,
                        "reason": "candidate QC flagged selection or rejected competing candidates",
                        **selection_record,
                        "selected_candidate_qc_status": selected_qc_status,
                        "selected_candidate_qc_flags": selected.get("candidate_qc_flags", []) if selected else [],
                        "comparison_candidate_models": selection_record.get("comparison_candidate_models", []),
                        "qc_rejected_candidates": selection_record.get("qc_rejected_candidates", []),
                        "review_flags": review_flags,
                        "quality_flags": quality_flags,
                    })
                selected_metadata.append({
                    **selection_record,
                    "pre_shapekit_mask": str(copied),
                    "mask_path": None,
                    "mask": None,
                    "dataset_role": "pseudo_label",
                    "ground_truth_status": "pseudo_label_candidate",
                    "shapekit_status": selected.get("candidate_shapekit_status") if selected else ("skipped_dry_run" if dry_run else "skipped_debug_only"),
                    "shapekit_reason": selected.get("candidate_shapekit_reason") if selected else None,
                    "shapekit_report": selected.get("candidate_shapekit_report") if selected else None,
                    "shapekit_attempted": bool(enable_shapekit and not dry_run),
                    "final_mask": None,
                    "review_flags": review_flags,
                    "quality_flags": quality_flags,
                    "quality_status": _quality_status(review_flags, quality_flags),
                })

        shapekit_result: dict[str, Any] = {
            "stage": "candidate_preselection_shapekit",
            "status": "completed",
            "order": "ShapeKit candidates before LabelCritic selection",
            "candidate_reports": candidate_shapekit_reports,
        }

        case_updated.mkdir(parents=True, exist_ok=True)
        for meta in selected_metadata:
            organ = meta["organ"]
            pre_mask = Path(meta["pre_shapekit_mask"])
            final = _copy_annotation(pre_mask, case_updated, organ)
            meta["final_mask"] = final
            meta["mask_path"] = final
            meta["mask"] = final
            if enable_shapekit and not dry_run and meta.get("shapekit_status") != "success":
                shapekit_unsupported = meta.get("shapekit_status") == "unsupported_target"
                _add_unique(meta.setdefault("review_flags", []), "shapekit_fallback")
                _add_unique(meta.setdefault("quality_flags", []), "shapekit_fallback")
                if shapekit_unsupported:
                    _add_unique(meta.setdefault("review_flags", []), "shapekit_unsupported_target")
                    _add_unique(meta.setdefault("quality_flags", []), "shapekit_unsupported_target")
                _append_jsonl(review_queue, {
                    "case_id": case_id,
                    "organ": organ,
                    "reason": "Candidate ShapeKit fallback before LabelCritic selection",
                    "selected_model": meta.get("selected_model"),
                    "candidate_models": meta.get("candidate_models", []),
                    "selection_method": meta.get("selection_method"),
                    "selection_status": meta.get("selection_status"),
                    "shapekit_status": meta.get("shapekit_status"),
                    "shapekit_reason": meta.get("shapekit_reason"),
                    "shapekit_unsupported_target": shapekit_unsupported,
                    "review_flags": meta.get("review_flags", []),
                    "quality_flags": meta.get("quality_flags", []),
                })
            else:
                if not enable_shapekit:
                    meta["shapekit_status"] = "skipped_debug_only"
                elif dry_run:
                    meta["shapekit_status"] = "skipped_dry_run"
            meta["quality_status"] = _quality_status(meta.get("review_flags", []), meta.get("quality_flags", []))
            passport = build_label_passport({
                **meta,
                "case_id": case_id,
                "ct_path": str(ct),
                "mask_path": final,
            })
            meta.update({
                "label_maturity_level": passport["label_maturity_level"],
                "auto_fine_label_status": passport["auto_fine_label_status"],
                "auto_fine_label_reliability_score": passport["auto_fine_label_reliability_score"],
                "grade": passport["grade"],
                "training_weight": passport["training_weight"],
                "label_passport_path": str(passport_path_for_mask(final)) if final else None,
                "ground_truth_status": "machine_generated_candidate",
            })
            if final:
                write_json(passport_path_for_mask(final), passport)

        case_gap_rows = _build_gap_rows(
            case_id=case_id,
            ct=ct,
            organs=organs,
            selection_rows=selection_rows,
            selected_metadata=selected_metadata,
        )
        all_gap_rows.extend(case_gap_rows)

        write_json(updated_root / case_id / "selection_metadata.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "dataset_type": "pseudo_label_dataset",
            "ground_truth_status": "pseudo_label_candidate",
            "selected_organs": selected_metadata,
            "selection_rows": selection_rows,
            "gap_rows": case_gap_rows,
            "shapekit": shapekit_result,
        })
        write_json(updated_root / case_id / "shapekit_report.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "stage": "shapekit_report",
            "dataset_type": "pseudo_label_dataset",
            "ground_truth_status": "pseudo_label_candidate",
            "enable_shapekit": enable_shapekit,
            "result": shapekit_result,
            "selected_organs": [
                {
                    "organ": item.get("organ"),
                    "shapekit_status": item.get("shapekit_status"),
                    "shapekit_reason": item.get("shapekit_reason"),
                    "final_mask": item.get("final_mask"),
                    "review_flags": item.get("review_flags", []),
                    "quality_flags": item.get("quality_flags", []),
                }
                for item in selected_metadata
            ],
        })

        write_json(case_out / "pseudo_label_selection.json", {
            "case_id": case_id,
            "ct_path": str(ct),
            "dataset_type": "pseudo_label_dataset",
            "selection_rows": selection_rows,
            "selected_organs": selected_metadata,
            "gap_rows": case_gap_rows,
            "shapekit": shapekit_result,
        })

        # Report supervision: compare tumor mask against report if both are available.
        if not dry_run and case.get("report_path"):
            try:
                from .report_supervision import verify_tumor_with_report
                tumor_mask = case_updated / "pancreatic_lesion.nii.gz"
                if not tumor_mask.exists() and ref_dir:
                    tumor_mask = ref_dir / "pancreatic_lesion.nii.gz"
                if tumor_mask.exists():
                    report_decision = verify_tumor_with_report(Path(case["report_path"]).resolve(), tumor_mask, "pancreas", Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None)
                    _append_jsonl(report_supervision_jsonl, {"case_id": case_id, **report_decision})
            except Exception as exc:
                _append_jsonl(report_supervision_jsonl, {"case_id": case_id, "stage": "report_supervision", "status": "failed", "reason": str(exc)})

        # Reasoning trace grounded in available case paths — one entry per organ with updated mask.
        if not dry_run:
            for trace_organ in organs:
                organ_mask = case_updated / f"{trace_organ}.nii.gz"
                if not organ_mask.exists():
                    continue
                try:
                    trace = build_reasoning_trace(
                        patient_folder=None, scan_id=case_id, ct_image=ct,
                        current_mask=organ_mask,
                        previous_mask=None, organ=trace_organ,
                        report_path=Path(case["report_path"]).resolve() if case.get("report_path") else None,
                        clinical_path=Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None,
                        pathology_path=Path(case["pathology_path"]).resolve() if case.get("pathology_path") else None,
                        output_json=None,
                    )
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace": trace})
                except Exception as exc:
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace_status": "failed", "reason": str(exc)})

        round_rows.append({
            "case_id": case_id,
            "checked_masks": checked,
            "accepted_masks": accepted,
            "low_dice_masks": low_dice,
            "vlm_reviewed": critic_count,
            "updated_masks": updated,
            "remaining_uncertain": uncertain,
        })
        _case_elapsed = round(_time.time() - _case_start, 1)
        print(f"[{_time.strftime('%H:%M:%S')}] Case {idx}/{len(cases)}: {case_id} 完成 "
              f"(耗时{_case_elapsed}s, accepted={accepted}, updated={updated}, critic={critic_count})", flush=True)

    dice_csv = out / "dice_metrics.csv"
    round_csv = out / "round_metrics.csv"
    _write_csv(dice_csv, dice_rows, [
        "case_id", "organ", "model", "prediction", "reference", "reference_role",
        "metric_family", "metric_scope", "ground_truth_status", "accuracy_warning",
        "dice", "pseudo_consistency_dice", "decision", "status", "reason",
        "candidate_exists", "alias_match",
    ])
    _write_csv(round_csv, round_rows, ["case_id", "checked_masks", "accepted_masks", "low_dice_masks", "vlm_reviewed", "updated_masks", "remaining_uncertain"])
    write_json(out / "inference_results.json", inference_results)
    manifest = build_training_manifest(updated_root, out / "training_manifest.json", organs=organs)
    gap_report = {
        "stage": "pseudo_label_gap_report",
        "status": "success",
        "dataset_type": "pseudo_label_dataset",
        "ground_truth_status": "pseudo_label_candidate",
        "num_gap_rows": len(all_gap_rows),
        "gap_rows": all_gap_rows,
        "target_space_policy": _load_target_space_policy(project_root, organs),
        "formal_373_target_validation": target_validation,
        "note": "Rows here are missing selected pseudo labels, ShapeKit fallbacks, or target-policy exclusions; they are not silently dropped.",
    }
    write_json(out / "pseudo_label_gap_report.json", gap_report)
    shapekit_reports = [
        str((updated_root / case.get("case_id", "") / "shapekit_report.json").resolve())
        for case in cases
        if case.get("case_id") and (updated_root / case.get("case_id", "") / "shapekit_report.json").exists()
    ]
    write_json(out / "shapekit_report.json", {
        "stage": "shapekit_report",
        "status": "success",
        "dataset_type": "pseudo_label_dataset",
        "ground_truth_status": "pseudo_label_candidate",
        "enable_shapekit": enable_shapekit,
        "num_case_reports": len(shapekit_reports),
        "case_reports": shapekit_reports,
        "note": "Per-case ShapeKit reports live under annotation_versions/<case_id>/shapekit_report.json.",
    })
    _write_csv(
        out / "pseudo_label_gap_report.csv",
        all_gap_rows,
        ["case_id", "ct_path", "organ", "gap_type", "reason", "candidate_count", "candidate_models", "selection_method", "selection_status", "shapekit_status", "dataset_type", "ground_truth_status"],
    )
    # Selected-model-aware M-step routing: record which primary model should be
    # updated for each organ, instead of implying that a generic model is always
    # the M-step target.
    mstep_routing = recommend_primary_models_for_organs(registry, organs)
    write_json(out / "mstep_model_routing.json", mstep_routing)
    mcfg = write_mstep_config(
        out / "mstep_config.json",
        out / "training_manifest.json",
        base_model="selected_model_aware",
        notes="Use `mstep-update --target-model <primary_model>` for each trainable primary model in mstep_model_routing.json. TotalSegmentator is baseline-only in this project.",
    )

    summary = {
        "stage": "run_loop", "status": "success", "case_list": str(case_csv), "output_folder": str(out),
        "num_cases": len(cases), "models_requested": models, "organs": organs,
        "target_space_policy": _load_target_space_policy(project_root, organs),
        "formal_373_target_validation": target_validation,
        "dry_run": dry_run, "enable_shapekit": enable_shapekit, "enable_critic": enable_critic, "critic_backend": critic_backend, "critic_base_url": critic_base_url, "critic_port": critic_port,
        "labelcritic_options": labelcritic_options,
        "dice_metrics_csv": str(dice_csv), "round_metrics_csv": str(round_csv), "review_queue_jsonl": str(review_queue),
        "vlm_decisions_jsonl": str(vlm_decisions), "patient_traces_jsonl": str(traces_jsonl), "report_supervision_jsonl": str(report_supervision_jsonl),
        "training_manifest": manifest, "pseudo_label_gap_report": gap_report, "shapekit_report": str((out / "shapekit_report.json").resolve()), "mstep_config": mcfg, "mstep_model_routing": mstep_routing,
        "resume_audit": resume_rows,
        "round2_competition_audit": _summarize_preseeded_competition(
            preseeded_keys=sorted((preseeded_model_dirs or {}).keys()),
            selection_rows=all_selection_rows,
        ),
        "round_rows": round_rows,
        "total_updated": sum(int(r.get("updated_masks", 0) or 0) for r in round_rows),
        "total_labelcritic_decisions": sum(int(r.get("vlm_reviewed", 0) or 0) for r in round_rows),
    }
    write_json(out / "run_summary.json", summary)
    return summary
