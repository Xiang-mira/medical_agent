from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEAD_TARGETS = {
    "brain_ventricle",
    "cerebrospinal_fluid",
    "gray_matter",
    "white_matter",
}
EYEBALL_TARGETS = {"eyeball"}
FACE_TARGETS = {
    "face",
    "muscle_of_head",
    "scalp",
}

CENTRAL_AIRWAY_TARGETS = {"airway_tree", "airway_wall"}
PULMONARY_VASCULAR_TARGETS = {"lung_pulmonary_arteries", "lung_pulmonary_veins"}

HEAD_LANDMARKS = ("brain", "skull", "eyeball_left", "eyeball_right")
THORAX_LANDMARKS = ("lung_left", "lung_right", "heart", "aorta")
CENTRAL_AIRWAY_LANDMARKS = ("airway", "trachea", "bronchus", "lung_trachea_bronchia")


@dataclass(frozen=True)
class LandmarkEvidence:
    name: str
    path: str
    exists: bool
    readable: bool
    foreground_voxels: int
    geometry_match: bool | None
    reason: str


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_evidence(mask: Path, *, ct_path: Path | None = None, name: str | None = None) -> LandmarkEvidence:
    if not mask.exists():
        return LandmarkEvidence(name or mask.name.removesuffix(".nii.gz"), str(mask), False, False, 0, None, "missing")
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(mask))
        foreground = int((np.asanyarray(img.dataobj) != 0).sum())
        geometry_match = None
        if ct_path and ct_path.exists():
            ct = nib.load(str(ct_path))
            geometry_match = bool(
                tuple(img.shape[:3]) == tuple(ct.shape[:3])
                and np.allclose(img.header.get_zooms()[:3], ct.header.get_zooms()[:3], rtol=0, atol=1e-5)
                and np.allclose(img.affine, ct.affine, rtol=0, atol=1e-5)
            )
        return LandmarkEvidence(
            name or mask.name.removesuffix(".nii.gz"),
            str(mask),
            True,
            True,
            foreground,
            geometry_match,
            "positive" if foreground > 0 else "zero",
        )
    except Exception as exc:
        return LandmarkEvidence(
            name or mask.name.removesuffix(".nii.gz"),
            str(mask),
            True,
            False,
            0,
            None,
            f"unreadable:{type(exc).__name__}",
        )


def collect_landmark_evidence(ref_dir: Path, *, ct_path: Path | None = None) -> dict[str, dict[str, Any]]:
    roots = [ref_dir, ref_dir / "segmentations", ref_dir / "updated"]
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(set(HEAD_LANDMARKS + THORAX_LANDMARKS + CENTRAL_AIRWAY_LANDMARKS)):
        candidates = [root / f"{name}.nii.gz" for root in roots]
        existing = next((path for path in candidates if path.exists()), candidates[0])
        item = mask_evidence(existing, ct_path=ct_path, name=name)
        inventory[name] = {
            "name": item.name,
            "path": item.path,
            "exists": item.exists,
            "readable": item.readable,
            "foreground_voxels": item.foreground_voxels,
            "geometry_match": item.geometry_match,
            "reason": item.reason,
        }
    return inventory


def _positive(landmarks: dict[str, dict[str, Any]], name: str, threshold: int = 1) -> bool:
    row = landmarks.get(name) or {}
    geometry = row.get("geometry_match")
    return bool(row.get("readable")) and int(row.get("foreground_voxels") or 0) >= threshold and geometry is not False


def summarize_fov_evidence(landmarks: dict[str, dict[str, Any]], coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage = coverage or {}
    brain = int((landmarks.get("brain") or {}).get("foreground_voxels") or 0)
    skull = int((landmarks.get("skull") or {}).get("foreground_voxels") or 0)
    eye_l = int((landmarks.get("eyeball_left") or {}).get("foreground_voxels") or 0)
    eye_r = int((landmarks.get("eyeball_right") or {}).get("foreground_voxels") or 0)
    lung_l = int((landmarks.get("lung_left") or {}).get("foreground_voxels") or 0)
    lung_r = int((landmarks.get("lung_right") or {}).get("foreground_voxels") or 0)
    heart = int((landmarks.get("heart") or {}).get("foreground_voxels") or 0)
    aorta = int((landmarks.get("aorta") or {}).get("foreground_voxels") or 0)
    airway = max(int((landmarks.get(name) or {}).get("foreground_voxels") or 0) for name in CENTRAL_AIRWAY_LANDMARKS)

    brain_skull = brain > 10000 and skull > 1000
    bilateral_eyeballs = eye_l > 0 and eye_r > 0
    strong_head = bool(coverage.get("has_head_coverage")) or brain_skull
    complete_head = bool(coverage.get("has_head_coverage")) or (brain_skull and bilateral_eyeballs)
    thorax = bool(coverage.get("has_thorax_coverage")) or (
        lung_l > 10000 and lung_r > 10000 and heart > 10000 and aorta > 1000
    )
    partial_thorax = bool(coverage.get("has_partial_thorax_coverage")) or (
        (lung_l > 10000 or lung_r > 10000) and heart > 10000 and aorta > 1000
    )
    central_airway = lung_l > 10000 and lung_r > 10000 and airway > 0
    return {
        "head_evidence": strong_head,
        "brain_skull_evidence": brain_skull,
        "bilateral_eyeball_evidence": bilateral_eyeballs and skull > 1000,
        "complete_head_evidence": complete_head,
        "thorax_evidence": thorax,
        "partial_thorax_evidence": partial_thorax,
        "central_airway_evidence": central_airway,
        "brain_foreground_voxels": brain,
        "skull_foreground_voxels": skull,
        "eyeball_left_foreground_voxels": eye_l,
        "eyeball_right_foreground_voxels": eye_r,
        "lung_left_foreground_voxels": lung_l,
        "lung_right_foreground_voxels": lung_r,
        "heart_foreground_voxels": heart,
        "aorta_foreground_voxels": aorta,
        "central_airway_foreground_voxels": airway,
    }


def target_fov_eligibility(
    target: str,
    *,
    landmarks: dict[str, dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    fallback_fov_status: str = "unknown",
) -> dict[str, Any]:
    norm = str(target).strip()
    summary = summarize_fov_evidence(landmarks or {}, coverage or {})
    if norm in HEAD_TARGETS:
        eligible = bool(summary["brain_skull_evidence"] or coverage and coverage.get("has_head_coverage"))
        return {
            **summary,
            "target": norm,
            "target_group": "head",
            "fov_status": "fully_visible" if eligible else "out_of_fov",
            "eligible": eligible,
            "reason": "strong_head_evidence" if eligible else "head_evidence_missing",
        }
    if norm in EYEBALL_TARGETS:
        eligible = bool(summary["bilateral_eyeball_evidence"])
        return {
            **summary,
            "target": norm,
            "target_group": "eyeball",
            "fov_status": "fully_visible" if eligible else "out_of_fov",
            "eligible": eligible,
            "reason": "bilateral_eyeball_skull_evidence" if eligible else "bilateral_eyeball_skull_evidence_missing",
        }
    if norm in FACE_TARGETS:
        eligible = bool(summary["complete_head_evidence"])
        return {
            **summary,
            "target": norm,
            "target_group": "complete_head",
            "fov_status": "fully_visible" if eligible else "out_of_fov",
            "eligible": eligible,
            "reason": "complete_head_evidence" if eligible else "complete_head_evidence_missing",
        }
    if norm in CENTRAL_AIRWAY_TARGETS:
        eligible = bool(summary["central_airway_evidence"])
        return {
            **summary,
            "target": norm,
            "target_group": "central_airway",
            "fov_status": "fully_visible" if eligible else "out_of_fov",
            "eligible": eligible,
            "reason": "central_airway_evidence" if eligible else "central_airway_evidence_missing",
        }
    if norm in PULMONARY_VASCULAR_TARGETS:
        eligible = bool(summary["thorax_evidence"] or summary["partial_thorax_evidence"])
        return {
            **summary,
            "target": norm,
            "target_group": "pulmonary_vascular",
            "fov_status": "fully_visible" if summary["thorax_evidence"] else ("partially_visible" if eligible else "out_of_fov"),
            "eligible": eligible,
            "reason": "thorax_lung_heart_aorta_evidence" if eligible else "thorax_evidence_missing",
        }
    eligible = fallback_fov_status not in {"out_of_fov", "unknown"}
    return {
        **summary,
        "target": norm,
        "target_group": "generic",
        "fov_status": fallback_fov_status,
        "eligible": eligible,
        "reason": f"fallback_fov_status:{fallback_fov_status}",
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
