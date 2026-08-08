#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_ROOT = Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_MASK_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_SCAN_LIMIT = 2000
DEFAULT_MAX_DIRECTORIES_EXAMINED = 50000
DEFAULT_CONSECUTIVE_GEOMETRY_FAILURE_LIMIT = 20
DEFAULT_GEOMETRY_FAILURE_RATE_THRESHOLD = 0.90
DEFAULT_MIN_GEOMETRY_FAILURES_FOR_RATE = 20
TARGETS = (
    "cerebrospinal_fluid",
    "gray_matter",
    "white_matter",
    "eyeball",
    "face",
    "muscle_of_head",
    "scalp",
    "airway_tree",
    "airway_wall",
    "lung_pulmonary_arteries",
    "lung_pulmonary_veins",
    "brain_ventricle",
)
BRAIN_TARGETS = {"cerebrospinal_fluid", "gray_matter", "white_matter", "brain_ventricle"}
FACE_TARGETS = {"face", "muscle_of_head", "scalp"}
CENTRAL_AIRWAY_TARGETS = {"airway_tree", "airway_wall"}
PULMONARY_TARGETS = {"lung_pulmonary_arteries", "lung_pulmonary_veins"}
PRESEEDED_PULMONARY_CASES = ("BDMAP_00000424", "BDMAP_00078156")
ALL_LANDMARKS = (
    "brain",
    "skull",
    "eyeball_left",
    "eyeball_right",
    "lung_left",
    "lung_right",
    "heart",
    "aorta",
    "trachea",
    "bronchus",
    "airway",
    "lung_trachea_bronchia",
)
OUTPUT_TARGET_FIELDS = (
    "target_name",
    "case_id",
    "rank",
    "evidence_type",
    "landmark_voxels",
    "geometry_valid",
    "reason",
    "ct_path",
    "annotation_folder",
)
OUTPUT_UNION_FIELDS = ("case_id", "ct_path", "annotation_folder", "targets", "evidence_types")
GEOMETRY_FAILURE_FIELDS = ("status", "case_id", "ct_path", "exception_type", "reason")
FAILURE_STATUSES = {
    "DATA_ROOT_INVALID",
    "FAILED_NO_VALID_CASES_SCANNED",
    "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE",
    "RUNTIME_DEPENDENCY_MISSING",
}


class RuntimeEnvironmentError(RuntimeError):
    def __init__(self, status: str, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.status = status
        self.payload = {"status": status, "reason": message, **payload}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp, path)


def _case_ids_from_mask_root(mask_root: Path) -> list[str]:
    if not mask_root.exists():
        return []
    return sorted(path.name for path in mask_root.iterdir() if path.is_dir())


def _import_required_dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeEnvironmentError(
            "RUNTIME_DEPENDENCY_MISSING",
            f"RUNTIME_DEPENDENCY_MISSING dependency={name} python_executable={sys.executable}",
            dependency=name,
            python_executable=sys.executable,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        ) from exc


def validate_runtime_environment(*, image_root: Path, mask_root: Path, output_root: Path) -> dict[str, Any]:
    nib = _import_required_dependency("nibabel")
    np = _import_required_dependency("numpy")
    root_errors: list[str] = []
    if not image_root.is_dir():
        root_errors.append(f"image_root_missing_or_not_directory:{image_root}")
    if not mask_root.is_dir():
        root_errors.append(f"mask_root_missing_or_not_directory:{mask_root}")
    if root_errors:
        raise RuntimeEnvironmentError(
            "DATA_ROOT_INVALID",
            ";".join(root_errors),
            python_executable=sys.executable,
            image_root=str(image_root),
            mask_root=str(mask_root),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        probe = output_root / f".runtime_preflight_write_probe.{os.getpid()}"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        raise RuntimeEnvironmentError(
            "DATA_ROOT_INVALID",
            f"output_root_not_writable:{output_root}:{type(exc).__name__}:{exc}",
            python_executable=sys.executable,
            output_root=str(output_root),
        ) from exc

    return {
        "runtime_preflight": "READY",
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "nibabel_version": getattr(nib, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "output_root": str(output_root),
        "image_root_exists": True,
        "mask_root_exists": True,
        "output_root_writable": True,
    }


def _ct_geometry_failure(case_id: str, ct: Path, exc: Exception) -> dict[str, Any]:
    return {
        "status": "CT_GEOMETRY_FAILED",
        "case_id": case_id,
        "ct_path": str(ct),
        "exception_type": type(exc).__name__,
        "reason": str(exc),
    }


def _ct_geometry(ct: Path, case_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        nib = _import_required_dependency("nibabel")

        img = nib.load(str(ct))
        return {
            "shape": tuple(int(x) for x in img.shape[:3]),
            "spacing": tuple(float(x) for x in img.header.get_zooms()[:3]),
            "affine": img.affine.copy(),
        }, None
    except Exception:
        exc = sys.exc_info()[1]
        assert exc is not None
        if isinstance(exc, RuntimeEnvironmentError):
            raise exc
        return None, _ct_geometry_failure(case_id, ct, exc)


def _mask_geometry_matches(mask_img: Any, ct_geom: dict[str, Any]) -> bool:
    np = _import_required_dependency("numpy")

    return bool(
        tuple(int(x) for x in mask_img.shape[:3]) == tuple(ct_geom["shape"])
        and np.allclose(mask_img.header.get_zooms()[:3], ct_geom["spacing"], rtol=0, atol=1e-5)
        and np.allclose(mask_img.affine, ct_geom["affine"], rtol=0, atol=1e-5)
    )


def _mask_voxels(seg_dir: Path, name: str, ct_geom: dict[str, Any]) -> dict[str, Any]:
    path = seg_dir / f"{name}.nii.gz"
    row = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "foreground_voxels": 0,
        "geometry_match": False,
        "reason": "missing",
    }
    if not path.exists():
        return row
    try:
        nib = _import_required_dependency("nibabel")
        np = _import_required_dependency("numpy")

        img = nib.load(str(path))
        geometry = _mask_geometry_matches(img, ct_geom)
        foreground = int((np.asanyarray(img.dataobj) != 0).sum()) if geometry else 0
        row.update(
            {
                "readable": True,
                "foreground_voxels": foreground,
                "geometry_match": geometry,
                "reason": "positive" if foreground > 0 else ("geometry_mismatch" if not geometry else "zero"),
            }
        )
    except Exception as exc:
        if isinstance(exc, RuntimeEnvironmentError):
            raise
        row["reason"] = f"unreadable:{type(exc).__name__}"
    return row


def _selected_count(target_cases: dict[str, list[dict[str, Any]]], target: str) -> int:
    return len(target_cases.get(target, []))


def _targets_remaining(target_cases: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [target for target in TARGETS if _selected_count(target_cases, target) < 2]


def _all_targets_covered(target_cases: dict[str, list[dict[str, Any]]]) -> bool:
    return all(_selected_count(target_cases, target) >= 2 for target in TARGETS)


def _required_landmarks_for_remaining(target_cases: dict[str, list[dict[str, Any]]]) -> set[str]:
    remaining = set(_targets_remaining(target_cases))
    required: set[str] = set()
    if remaining & BRAIN_TARGETS:
        required.update(("brain", "skull"))
    if "eyeball" in remaining:
        required.update(("eyeball_left", "eyeball_right", "skull"))
    if remaining & FACE_TARGETS:
        required.update(("brain", "skull", "eyeball_left", "eyeball_right"))
    if remaining & CENTRAL_AIRWAY_TARGETS:
        required.update(("lung_left", "lung_right", "trachea", "bronchus", "airway", "lung_trachea_bronchia"))
    if remaining & PULMONARY_TARGETS:
        required.update(("lung_left", "lung_right", "heart", "aorta"))
    return required


def _voxels(landmarks: dict[str, dict[str, Any]], name: str) -> int:
    row = landmarks.get(name) or {}
    return int(row.get("foreground_voxels") or 0) if row.get("geometry_match") is True and row.get("readable") else 0


def _target_evidence(target: str, landmarks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = {name: _voxels(landmarks, name) for name in ALL_LANDMARKS}
    central_airway = max(values["trachea"], values["bronchus"], values["airway"], values["lung_trachea_bronchia"])
    if target in BRAIN_TARGETS:
        ok = values["brain"] > 10000 and values["skull"] > 1000
        names = ("brain", "skull")
        reason = "brain_skull_coverage" if ok else "brain_skull_coverage_missing"
    elif target == "eyeball":
        ok = values["eyeball_left"] > 0 and values["eyeball_right"] > 0 and values["skull"] > 1000
        names = ("eyeball_left", "eyeball_right", "skull")
        reason = "bilateral_eyeball_skull_coverage" if ok else "bilateral_eyeball_skull_coverage_missing"
    elif target in FACE_TARGETS:
        ok = (
            values["brain"] > 10000
            and values["skull"] > 1000
            and values["eyeball_left"] > 0
            and values["eyeball_right"] > 0
        )
        names = ("brain", "skull", "eyeball_left", "eyeball_right")
        reason = "complete_head_coverage" if ok else "complete_head_coverage_missing"
    elif target in CENTRAL_AIRWAY_TARGETS:
        ok = values["lung_left"] > 10000 and values["lung_right"] > 10000 and central_airway > 0
        names = ("lung_left", "lung_right", "trachea", "bronchus", "airway", "lung_trachea_bronchia")
        reason = "central_airway_coverage" if ok else "central_airway_coverage_missing"
    elif target in PULMONARY_TARGETS:
        ok = values["lung_left"] > 10000 and values["lung_right"] > 10000 and values["heart"] > 10000 and values["aorta"] > 1000
        names = ("lung_left", "lung_right", "heart", "aorta")
        reason = "pulmonary_vascular_fov_coverage" if ok else "pulmonary_vascular_fov_coverage_missing"
    else:
        ok = False
        names = ()
        reason = "unknown_target"
    return {
        "eligible": ok,
        "reason": reason,
        "landmark_voxels": {name: values[name] for name in names},
        "geometry_valid": all((landmarks.get(name) or {}).get("geometry_match") is True for name in names),
    }


def _add_case(
    target_cases: dict[str, list[dict[str, Any]]],
    target: str,
    case_id: str,
    *,
    evidence_type: str,
    evidence: dict[str, Any],
) -> bool:
    rows = target_cases.setdefault(target, [])
    if len(rows) >= 2 or any(row["case_id"] == case_id for row in rows):
        return False
    rows.append(
        {
            "target_name": target,
            "case_id": case_id,
            "rank": len(rows) + 1,
            "evidence_type": evidence_type,
            "landmark_voxels": json.dumps(evidence.get("landmark_voxels") or {}, sort_keys=True),
            "geometry_valid": bool(evidence.get("geometry_valid")),
            "reason": evidence.get("reason", ""),
            "ct_path": evidence.get("ct_path", ""),
            "annotation_folder": evidence.get("annotation_folder", ""),
        }
    )
    return True


def _initial_target_cases(preseed_pulmonary: bool, *, image_root: Path, mask_root: Path) -> dict[str, list[dict[str, Any]]]:
    target_cases = {target: [] for target in TARGETS}
    if not preseed_pulmonary:
        return target_cases
    for target in PULMONARY_TARGETS:
        for case_id in PRESEEDED_PULMONARY_CASES:
            evidence = {
                "landmark_voxels": {
                    "lung_left": "verified_positive",
                    "lung_right": "verified_positive",
                    "heart": "verified_positive",
                    "aorta": "verified_positive",
                },
                "geometry_valid": True,
                "reason": "preseeded_append_case_verified_lung_heart_aorta_geometry",
                "ct_path": str(image_root / case_id / "ct.nii.gz"),
                "annotation_folder": str(mask_root / case_id / "segmentations"),
            }
            _add_case(target_cases, target, case_id, evidence_type="preseeded_verified_pulmonary_landmarks", evidence=evidence)
    return target_cases


def _target_rows(target_cases: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for index, row in enumerate(target_cases.get(target, [])[:2], start=1):
            item = dict(row)
            item["rank"] = index
            rows.append(item)
    return rows


def _union_rows(target_cases: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for rows in target_cases.values():
        for row in rows[:2]:
            item = by_case.setdefault(
                row["case_id"],
                {"targets": set(), "evidence_types": set(), "ct_path": row.get("ct_path", ""), "annotation_folder": row.get("annotation_folder", "")},
            )
            item["targets"].add(row["target_name"])
            item["evidence_types"].add(row["evidence_type"])
            if not item.get("ct_path"):
                item["ct_path"] = row.get("ct_path", "")
            if not item.get("annotation_folder"):
                item["annotation_folder"] = row.get("annotation_folder", "")
    return [
        {
            "case_id": case_id,
            "ct_path": item.get("ct_path", ""),
            "annotation_folder": item.get("annotation_folder", ""),
            "targets": ";".join(sorted(item["targets"])),
            "evidence_types": ";".join(sorted(item["evidence_types"])),
        }
        for case_id, item in sorted(by_case.items())
    ]


def _persist_outputs(
    output_root: Path,
    *,
    target_cases: dict[str, list[dict[str, Any]]],
    checkpoint: dict[str, Any],
    geometry_failures: list[dict[str, Any]] | None = None,
    runtime_preflight: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    _write_csv(output_root / "target_case_coverage.csv", _target_rows(target_cases), OUTPUT_TARGET_FIELDS)
    _write_csv(output_root / "selected_case_union.csv", _union_rows(target_cases), OUTPUT_UNION_FIELDS)
    _write_csv(output_root / "ct_geometry_failures.csv", geometry_failures or [], GEOMETRY_FAILURE_FIELDS)
    _write_json(output_root / "search_checkpoint.json", checkpoint)
    if runtime_preflight is not None:
        _write_json(output_root / "runtime_preflight.json", runtime_preflight)
    if summary is not None:
        _write_json(output_root / "search_summary.json", summary)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _progress_line(
    scanned: int,
    scan_limit: int,
    target_cases: dict[str, list[dict[str, Any]]],
    started: float,
    *,
    directories_examined: int = 0,
    valid_case_candidates: int = 0,
    ct_geometry_failure_count: int = 0,
) -> str:
    counts = " ".join(f"{target}={min(_selected_count(target_cases, target), 2)}/2" for target in TARGETS)
    unique_count = len(_union_rows(target_cases))
    return (
        f"[FOV_SEARCH] scanned={scanned}/{scan_limit} directories_examined={directories_examined} "
        f"valid_case_candidates={valid_case_candidates} ct_geometry_failures={ct_geometry_failure_count} "
        f"unique_selected_cases={unique_count} elapsed_sec={int(time.time() - started)} {counts}"
    )


def search_fov_candidates(
    *,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    mask_root: Path = DEFAULT_MASK_ROOT,
    output_root: Path,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    max_directories_examined: int = DEFAULT_MAX_DIRECTORIES_EXAMINED,
    consecutive_geometry_failure_limit: int = DEFAULT_CONSECUTIVE_GEOMETRY_FAILURE_LIMIT,
    geometry_failure_rate_threshold: float = DEFAULT_GEOMETRY_FAILURE_RATE_THRESHOLD,
    min_geometry_failures_for_rate: int = DEFAULT_MIN_GEOMETRY_FAILURES_FOR_RATE,
    progress_every: int = 50,
    checkpoint_every: int = 10,
    resume: bool = True,
    preseed_pulmonary: bool = True,
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime_preflight = validate_runtime_environment(image_root=image_root, mask_root=mask_root, output_root=output_root)
    except RuntimeEnvironmentError as exc:
        runtime_preflight = {
            "runtime_preflight": "FAILED",
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            **exc.payload,
        }
        target_cases = _initial_target_cases(preseed_pulmonary, image_root=image_root, mask_root=mask_root)
        summary = {
            **runtime_preflight,
            "image_root": str(image_root),
            "mask_root": str(mask_root),
            "scan_limit": scan_limit,
            "max_directories_examined": max_directories_examined,
            "directories_examined": 0,
            "valid_case_candidates": 0,
            "scanned_count": 0,
            "missing_ct_count": 0,
            "missing_seg_count": 0,
            "ct_geometry_failure_count": 0,
            "runtime_failure_count": 1 if exc.status == "RUNTIME_DEPENDENCY_MISSING" else 0,
            "early_stopped": False,
            "all_targets_covered": False,
            "target_counts": {target: min(_selected_count(target_cases, target), 2) for target in TARGETS},
            "uncovered_targets": [target for target in TARGETS if _selected_count(target_cases, target) < 2],
            "selected_unique_case_count": len(_union_rows(target_cases)),
            "preseeded_cases": list(PRESEEDED_PULMONARY_CASES) if preseed_pulmonary else [],
            "resume_used": False,
            "next_mask_index": 0,
            "elapsed_sec": round(time.time() - started, 3),
        }
        checkpoint = {
            "version": 3,
            "image_root": str(image_root),
            "mask_root": str(mask_root),
            "scan_limit": scan_limit,
            "max_directories_examined": max_directories_examined,
            "next_mask_index": 0,
            "directories_examined": 0,
            "valid_case_candidates": 0,
            "scanned_count": 0,
            "last_case_id": "",
            "target_cases": target_cases,
            "target_counts": summary["target_counts"],
            "preseeded_cases": summary["preseeded_cases"],
        }
        _persist_outputs(
            output_root,
            target_cases=target_cases,
            checkpoint=checkpoint,
            geometry_failures=[],
            runtime_preflight=runtime_preflight,
            summary=summary,
        )
        return summary

    case_ids = _case_ids_from_mask_root(mask_root)
    checkpoint_path = output_root / "search_checkpoint.json"
    loaded = _load_checkpoint(checkpoint_path) if resume else None
    resume_used = bool(loaded)
    if loaded:
        target_cases = {
            target: [dict(row) for row in (loaded.get("target_cases", {}) or {}).get(target, [])][:2]
            for target in TARGETS
        }
        next_mask_index = int(loaded.get("next_mask_index") or 0)
        scanned_count = int(loaded.get("scanned_count") or 0)
        directories_examined = int(loaded.get("directories_examined") or next_mask_index)
        valid_case_candidates = int(loaded.get("valid_case_candidates") or scanned_count)
        missing_ct_count = int(loaded.get("missing_ct_count") or 0)
        missing_seg_count = int(loaded.get("missing_seg_count") or 0)
        ct_geometry_failures = [dict(row) for row in (loaded.get("ct_geometry_failures") or [])]
        ct_geometry_failure_count = int(loaded.get("ct_geometry_failure_count") or len(ct_geometry_failures))
    else:
        target_cases = _initial_target_cases(preseed_pulmonary, image_root=image_root, mask_root=mask_root)
        next_mask_index = 0
        scanned_count = 0
        directories_examined = 0
        valid_case_candidates = 0
        missing_ct_count = 0
        missing_seg_count = 0
        ct_geometry_failures: list[dict[str, Any]] = []
        ct_geometry_failure_count = 0

    early_stopped = False
    last_case_id = ""
    failure_status: str | None = None
    failure_reason = ""
    consecutive_geometry_failures = 0

    def persist_checkpoint() -> None:
        checkpoint = {
            "version": 3,
            "image_root": str(image_root),
            "mask_root": str(mask_root),
            "scan_limit": scan_limit,
            "max_directories_examined": max_directories_examined,
            "next_mask_index": next_mask_index,
            "directories_examined": directories_examined,
            "valid_case_candidates": valid_case_candidates,
            "scanned_count": scanned_count,
            "missing_ct_count": missing_ct_count,
            "missing_seg_count": missing_seg_count,
            "ct_geometry_failure_count": ct_geometry_failure_count,
            "ct_geometry_failures": ct_geometry_failures,
            "last_case_id": last_case_id,
            "target_cases": target_cases,
            "target_counts": {target: min(_selected_count(target_cases, target), 2) for target in TARGETS},
            "preseeded_cases": list(PRESEEDED_PULMONARY_CASES) if preseed_pulmonary else [],
        }
        _persist_outputs(
            output_root,
            target_cases=target_cases,
            checkpoint=checkpoint,
            geometry_failures=ct_geometry_failures,
            runtime_preflight=runtime_preflight,
        )

    for mask_index in range(next_mask_index, len(case_ids)):
        if scanned_count >= scan_limit:
            next_mask_index = mask_index
            break
        if directories_examined >= max_directories_examined:
            failure_status = "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE"
            failure_reason = (
                f"max_directories_examined reached before scan_limit: "
                f"directories_examined={directories_examined} scanned_count={scanned_count} scan_limit={scan_limit}"
            )
            next_mask_index = mask_index
            break
        if _all_targets_covered(target_cases):
            early_stopped = True
            next_mask_index = mask_index
            break

        case_id = case_ids[mask_index]
        directories_examined += 1
        ct = image_root / case_id / "ct.nii.gz"
        seg_dir = mask_root / case_id / "segmentations"
        next_mask_index = mask_index + 1
        if not ct.exists():
            missing_ct_count += 1
            if checkpoint_every <= 1 or (checkpoint_every > 0 and directories_examined % checkpoint_every == 0):
                persist_checkpoint()
            continue
        if not seg_dir.is_dir():
            missing_seg_count += 1
            if checkpoint_every <= 1 or (checkpoint_every > 0 and directories_examined % checkpoint_every == 0):
                persist_checkpoint()
            continue
        valid_case_candidates += 1
        ct_geom, ct_failure = _ct_geometry(ct, case_id)
        if ct_geom is None:
            if ct_failure is not None:
                ct_geometry_failure_count += 1
                consecutive_geometry_failures += 1
                ct_geometry_failures.append(ct_failure)
                if consecutive_geometry_failures >= consecutive_geometry_failure_limit:
                    failure_status = "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE"
                    failure_reason = (
                        f"consecutive CT geometry failures reached {consecutive_geometry_failures}; "
                        f"last_case_id={case_id}"
                    )
                    break
                if (
                    ct_geometry_failure_count >= min_geometry_failures_for_rate
                    and valid_case_candidates > 0
                    and (ct_geometry_failure_count / valid_case_candidates) > geometry_failure_rate_threshold
                ):
                    failure_status = "POSSIBLE_RUNTIME_OR_DATASET_CONFIGURATION_FAILURE"
                    failure_reason = (
                        "CT geometry failure rate exceeded threshold: "
                        f"{ct_geometry_failure_count}/{valid_case_candidates} > {geometry_failure_rate_threshold}"
                    )
                    break
            if checkpoint_every <= 1 or (checkpoint_every > 0 and directories_examined % checkpoint_every == 0):
                persist_checkpoint()
            continue

        consecutive_geometry_failures = 0
        scanned_count += 1
        last_case_id = case_id
        required = _required_landmarks_for_remaining(target_cases)
        landmarks: dict[str, dict[str, Any]] = {}
        for name in sorted(required):
            landmarks[name] = _mask_voxels(seg_dir, name, ct_geom)

        for target in _targets_remaining(target_cases):
            evidence = _target_evidence(target, landmarks)
            if evidence["eligible"]:
                evidence["ct_path"] = str(ct)
                evidence["annotation_folder"] = str(seg_dir)
                _add_case(target_cases, target, case_id, evidence_type="landmark_mask_scan", evidence=evidence)

        if checkpoint_every <= 1 or scanned_count % checkpoint_every == 0 or directories_examined % checkpoint_every == 0:
            persist_checkpoint()
        if progress_every > 0 and (scanned_count % progress_every == 0):
            print(
                _progress_line(
                    scanned_count,
                    scan_limit,
                    target_cases,
                    started,
                    directories_examined=directories_examined,
                    valid_case_candidates=valid_case_candidates,
                    ct_geometry_failure_count=ct_geometry_failure_count,
                ),
                flush=True,
            )

    if _all_targets_covered(target_cases):
        early_stopped = True

    target_counts = {target: min(_selected_count(target_cases, target), 2) for target in TARGETS}
    uncovered = [target for target, count in target_counts.items() if count < 2]
    if failure_status is not None:
        status = failure_status
    elif scanned_count == 0:
        status = "FAILED_NO_VALID_CASES_SCANNED"
        failure_reason = "no valid cases reached FOV analysis; check Python dependencies, roots, CT readability, and dataset layout"
    elif uncovered:
        status = "completed_with_uncovered_targets"
    else:
        status = "completed"
    summary = {
        "status": status,
        "failure_reason": failure_reason,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "scan_limit": scan_limit,
        "max_directories_examined": max_directories_examined,
        "directories_examined": directories_examined,
        "valid_case_candidates": valid_case_candidates,
        "scanned_count": scanned_count,
        "missing_ct_count": missing_ct_count,
        "missing_seg_count": missing_seg_count,
        "ct_geometry_failure_count": ct_geometry_failure_count,
        "runtime_failure_count": 0,
        "early_stopped": early_stopped,
        "all_targets_covered": not uncovered,
        "target_counts": target_counts,
        "uncovered_targets": uncovered,
        "selected_unique_case_count": len(_union_rows(target_cases)),
        "preseeded_cases": list(PRESEEDED_PULMONARY_CASES) if preseed_pulmonary else [],
        "resume_used": resume_used,
        "next_mask_index": next_mask_index,
        "runtime": runtime_preflight,
        "ct_geometry_failures": ct_geometry_failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    final_checkpoint = {
        "version": 3,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "scan_limit": scan_limit,
        "max_directories_examined": max_directories_examined,
        "next_mask_index": next_mask_index,
        "directories_examined": directories_examined,
        "valid_case_candidates": valid_case_candidates,
        "scanned_count": scanned_count,
        "missing_ct_count": missing_ct_count,
        "missing_seg_count": missing_seg_count,
        "ct_geometry_failure_count": ct_geometry_failure_count,
        "ct_geometry_failures": ct_geometry_failures,
        "last_case_id": last_case_id,
        "target_cases": target_cases,
        "target_counts": target_counts,
        "preseeded_cases": summary["preseeded_cases"],
    }
    _persist_outputs(
        output_root,
        target_cases=target_cases,
        checkpoint=final_checkpoint,
        geometry_failures=ct_geometry_failures,
        runtime_preflight=runtime_preflight,
        summary=summary,
    )
    print(
        _progress_line(
            scanned_count,
            scan_limit,
            target_cases,
            started,
            directories_examined=directories_examined,
            valid_case_candidates=valid_case_candidates,
            ct_geometry_failure_count=ct_geometry_failure_count,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only Task2 target-specific FOV candidate search over the first 2000 valid AbdomenAtlasPro cases.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scan-limit", default=DEFAULT_SCAN_LIMIT, type=int)
    parser.add_argument("--max-directories-examined", default=DEFAULT_MAX_DIRECTORIES_EXAMINED, type=int)
    parser.add_argument("--consecutive-geometry-failure-limit", default=DEFAULT_CONSECUTIVE_GEOMETRY_FAILURE_LIMIT, type=int)
    parser.add_argument("--geometry-failure-rate-threshold", default=DEFAULT_GEOMETRY_FAILURE_RATE_THRESHOLD, type=float)
    parser.add_argument("--min-geometry-failures-for-rate", default=DEFAULT_MIN_GEOMETRY_FAILURES_FOR_RATE, type=int)
    parser.add_argument("--progress-every", default=50, type=int)
    parser.add_argument("--checkpoint-every", default=10, type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-preseed", action="store_true")
    args = parser.parse_args()
    summary = search_fov_candidates(
        image_root=args.image_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        scan_limit=args.scan_limit,
        max_directories_examined=args.max_directories_examined,
        consecutive_geometry_failure_limit=args.consecutive_geometry_failure_limit,
        geometry_failure_rate_threshold=args.geometry_failure_rate_threshold,
        min_geometry_failures_for_rate=args.min_geometry_failures_for_rate,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        preseed_pulmonary=not args.no_preseed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 2 if summary.get("status") in FAILURE_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
