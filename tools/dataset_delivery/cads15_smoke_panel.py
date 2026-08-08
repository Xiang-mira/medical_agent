#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ, _load_case_presence_context  # noqa: E402
from tools.dataset_delivery.cads15_contract_audit import CADS15_TARGETS, DEFAULT_CONTRACT, contract_targets  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_fov import collect_landmark_evidence, target_fov_eligibility  # noqa: E402


DEFAULT_CASE_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)
CADS15_PANEL_CACHE_POLICY_VERSION = "target_fov_v2"

REGION_TERM_MAP = {
    "abdomen": {"abdomen", "abdominal", "abdomen_pelvis", "abdominopelvic", "multi_region"},
    "pelvis": {"pelvis", "pelvic", "abdomen_pelvis", "abdominopelvic", "multi_region"},
    "thorax": {"thorax", "chest", "lung", "cardiac", "multi_region"},
    "head_neck": {"head", "neck", "head_neck", "brain", "craniofacial", "multi_region"},
    "extremity": {"extremity", "extremities", "arms", "legs", "multi_region"},
}


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _annotation_folder(row: dict[str, str]) -> str:
    return row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _metadata_terms(value: Any) -> set[str]:
    terms: set[str] = set()
    if value is None:
        return terms
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    for item in items:
        text = str(item).strip().lower().replace("-", "_").replace("/", "_")
        if not text:
            continue
        for sep in (",", ";", "|"):
            text = text.replace(sep, " ")
        for token in text.split():
            if token:
                terms.add(token)
        terms.add(text)
    return terms


def _load_lightweight_case_metadata(row: dict[str, str], ct: Path, case_id: str, output_case_root: Path) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for key in (
        "scan_coverage",
        "coverage_regions",
        "body_region",
        "ct_region",
        "confirmed_absent_organs",
        "out_of_scan_organs",
        "negative_organs",
        "absent_organs",
    ):
        if row.get(key):
            doc[key] = row.get(key)
    for raw_path in (
        row.get("case_metadata_path"),
        row.get("scan_coverage_path"),
        row.get("metadata_path"),
    ):
        if raw_path:
            payload = _read_json(Path(raw_path))
            if payload:
                doc.update(payload)
    for path in [
        ct.parent / "scan_coverage.json",
        ct.parent / "case_metadata.json",
        ct.parent / f"{case_id}_scan_coverage.json",
        ct.parent / f"{case_id}_case_metadata.json",
        output_case_root / "scan_coverage.json",
        output_case_root / "case_metadata.json",
    ]:
        payload = _read_json(path)
        if payload:
            doc.update(payload)
    return doc


def _presence_from_lightweight_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    terms = set()
    for key in ("scan_coverage", "coverage_regions", "body_region", "ct_region"):
        terms |= _metadata_terms(doc.get(key))
    context = {
        "metadata": doc,
        "coverage_terms": sorted(terms),
        "coverage_evidence": [],
        "confirmed_absent_organs": [],
        "has_region_evidence": False,
        "has_abdomen_coverage": False,
        "has_pelvis_coverage": False,
        "has_thorax_coverage": False,
        "has_partial_thorax_coverage": False,
        "has_head_coverage": False,
        "has_extremity_coverage": False,
    }
    for region, region_terms in REGION_TERM_MAP.items():
        if terms & region_terms:
            key = "has_head_coverage" if region == "head_neck" else f"has_{region}_coverage"
            context[key] = True
    if terms - {"unknown", "multi", "region"}:
        context["has_region_evidence"] = True
        context["coverage_evidence"].append("lightweight_metadata")
    for key in ("confirmed_absent_organs", "out_of_scan_organs", "negative_organs", "absent_organs"):
        if doc.get(key):
            context["confirmed_absent_organs"].extend(sorted(_metadata_terms(doc.get(key))))
    context["confirmed_absent_organs"] = sorted(set(context["confirmed_absent_organs"]))
    return context


def _reference_candidates(ref_dir: Path, target: str) -> list[Path]:
    return [
        ref_dir / f"{target}.nii.gz",
        ref_dir / "segmentations" / f"{target}.nii.gz",
        ref_dir / "updated" / f"{target}.nii.gz",
    ]


def _load_reference_inventory(ref_dir: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    json_candidates = [
        ref_dir / "mask_inventory.json",
        ref_dir / "reference_mask_inventory.json",
        ref_dir / "deep_mask_audit.json",
        ref_dir / "mask_stats.json",
        ref_dir.parent / "mask_inventory.json",
    ]
    for path in json_candidates:
        data = _read_json(path)
        rows = data.get("rows") or data.get("masks") or data.get("items") or []
        if isinstance(rows, dict):
            rows = [{"target": key, **(value if isinstance(value, dict) else {"value": value})} for key, value in rows.items()]
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("target") or row.get("target_name") or row.get("label_name") or row.get("organ") or "").strip()
            if not name:
                continue
            inventory[name] = dict(row)
    csv_candidates = [
        ref_dir / "mask_inventory.csv",
        ref_dir / "reference_mask_inventory.csv",
        ref_dir / "deep_mask_audit.csv",
        ref_dir / "mask_stats.csv",
        ref_dir.parent / "mask_inventory.csv",
    ]
    for path in csv_candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("target") or row.get("target_name") or row.get("label_name") or row.get("organ") or "").strip()
                if name:
                    inventory[name] = dict(row)
    return inventory


def _inventory_positive(row: dict[str, Any] | None) -> bool | None:
    if not row:
        return None
    for key in ("foreground_voxels", "nonzero_voxels", "nonzero", "voxel_count"):
        if row.get(key) not in {None, ""}:
            try:
                return int(float(row[key])) > 0
            except Exception:
                pass
    for key in ("positive", "nonempty", "has_mask"):
        if row.get(key) not in {None, ""}:
            return str(row[key]).strip().lower() in {"1", "true", "yes", "positive", "valid"}
    return None


def _mask_positive(path: Path, *, inventory_row: dict[str, Any] | None = None, allow_voxel_stats: bool = True) -> dict[str, Any]:
    inv_positive = _inventory_positive(inventory_row)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "positive": False,
        "foreground_voxels": 0,
        "reason": "",
        "evidence_source": "reference_mask",
    }
    if inv_positive is not None:
        result["positive"] = bool(inv_positive)
        result["foreground_voxels"] = int(float(inventory_row.get("foreground_voxels") or inventory_row.get("nonzero_voxels") or 1)) if inv_positive else 0
        result["reason"] = "positive_reference_inventory" if inv_positive else "zero_reference_inventory"
        result["evidence_source"] = "reference_inventory"
        return result
    if not path.exists():
        result["reason"] = "missing_reference"
        return result
    try:
        if path.stat().st_size <= 0:
            result["reason"] = "empty_reference_file"
            return result
    except Exception as exc:
        result["reason"] = f"reference_stat_error:{type(exc).__name__}:{exc}"
        return result
    if not allow_voxel_stats:
        result["reason"] = "reference_voxel_stats_not_requested"
        return result
    try:
        import nibabel as nib
        import numpy as np

        arr = np.asanyarray(nib.load(str(path)).dataobj)
        voxels = int((arr != 0).sum())
        result["foreground_voxels"] = voxels
        result["positive"] = voxels > 0
        result["reason"] = "positive_reference" if voxels > 0 else "zero_reference"
    except Exception as exc:
        result["reason"] = f"reference_read_error:{type(exc).__name__}:{exc}"
    return result


def _target_positive_reference(
    ref_dir: Path,
    target: str,
    *,
    inventory: dict[str, dict[str, Any]],
    allow_reference_voxel_stats: bool,
) -> dict[str, Any]:
    for path in _reference_candidates(ref_dir, target):
        result = _mask_positive(
            path,
            inventory_row=inventory.get(target),
            allow_voxel_stats=allow_reference_voxel_stats,
        )
        if result["positive"] or result["exists"]:
            return result
    return _mask_positive(ref_dir / f"{target}.nii.gz", inventory_row=inventory.get(target), allow_voxel_stats=False)


def _historical_teacher_positive(
    historical_roots: list[Path],
    *,
    case_id: str,
    target: str,
    ct: Path,
) -> dict[str, Any]:
    candidates: list[Path] = []
    for root in historical_roots:
        if not root:
            continue
        candidates.extend([
            root / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz",
            root / "cases" / case_id / "hierarchical_predictions" / "cads553" / "segmentations" / f"{target}.nii.gz",
            root / "cases" / case_id / "hierarchical_predictions" / "cads557" / "segmentations" / f"{target}.nii.gz",
            root / "cases" / case_id / "hierarchical_predictions" / "cads559" / "segmentations" / f"{target}.nii.gz",
        ])
        candidates.extend(sorted((root / "cases" / case_id).glob(f"**/segmentations/{target}.nii.gz")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result = _mask_positive(path, allow_voxel_stats=True)
        if not result["positive"]:
            continue
        geometry_ok = True
        try:
            import nibabel as nib
            import numpy as np

            mask_img = nib.load(str(path))
            ct_img = nib.load(str(ct))
            geometry_ok = bool(
                tuple(mask_img.shape[:3]) == tuple(ct_img.shape[:3])
                and np.allclose(mask_img.header.get_zooms()[:3], ct_img.header.get_zooms()[:3], rtol=0, atol=1e-5)
                and np.allclose(mask_img.affine, ct_img.affine, rtol=0, atol=1e-5)
            )
        except Exception:
            geometry_ok = False
        if geometry_ok:
            return {
                **result,
                "evidence_source": "historical_teacher_output",
                "geometry_match": True,
                "reason": "historical_teacher_positive",
            }
    return {
        "path": "",
        "exists": False,
        "positive": False,
        "foreground_voxels": 0,
        "evidence_source": "historical_teacher_output",
        "geometry_match": None,
        "reason": "missing_historical_teacher_output",
    }


def _cache_config_fingerprint(
    *,
    targets: list[str],
    allow_heavy_ct_fov: bool,
    allow_reference_voxel_stats: bool,
    historical_roots: list[Path] | None,
) -> dict[str, Any]:
    return {
        "policy_version": CADS15_PANEL_CACHE_POLICY_VERSION,
        "targets": sorted(targets),
        "allow_heavy_ct_fov": bool(allow_heavy_ct_fov),
        "allow_reference_voxel_stats": bool(allow_reference_voxel_stats),
        "historical_teacher_roots": sorted(str(path.resolve()) for path in (historical_roots or [])),
    }


def _cache_valid(
    cached: dict[str, Any],
    *,
    ct: Path,
    ref_paths: dict[str, list[dict[str, Any]]],
    cache_config: dict[str, Any],
) -> bool:
    return (
        cached.get("processing_status") == "completed"
        and cached.get("ct_fingerprint") == _file_fingerprint(ct)
        and cached.get("reference_fingerprints") == ref_paths
        and cached.get("cache_config") == cache_config
    )


def _case_reference_fingerprints(ref_dir: Path, targets: list[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        target: [_file_fingerprint(path) for path in _reference_candidates(ref_dir, target)]
        for target in targets
    }


def _process_case(
    *,
    source_row: dict[str, str],
    index: int,
    targets: list[str],
    context_root: Path,
    allow_heavy_ct_fov: bool,
    allow_reference_voxel_stats: bool,
    historical_roots: list[Path] | None = None,
) -> dict[str, Any]:
    case_id = _case_id(source_row, index)
    ct = Path(_ct_path(source_row))
    ref_dir = Path(_annotation_folder(source_row))
    case_root = context_root / case_id
    cache_path = case_root / "presence_context.json"
    ref_fps = _case_reference_fingerprints(ref_dir, targets) if ref_dir.exists() else {}
    cache_config = _cache_config_fingerprint(
        targets=targets,
        allow_heavy_ct_fov=allow_heavy_ct_fov,
        allow_reference_voxel_stats=allow_reference_voxel_stats,
        historical_roots=historical_roots,
    )
    cached = _read_json(cache_path)
    if ct.exists() and ref_dir.exists() and _cache_valid(cached, ct=ct, ref_paths=ref_fps, cache_config=cache_config):
        return {**cached, "cache_status": "reused"}

    record: dict[str, Any] = {
        "case_id": case_id,
        "ct_path": str(ct),
        "annotation_folder": str(ref_dir),
        "ct_fingerprint": _file_fingerprint(ct),
        "reference_fingerprints": ref_fps,
        "cache_config": cache_config,
        "processing_status": "running",
        "started_at": _utc_timestamp(),
        "cache_status": "computed",
        "positive_targets": [],
        "targets": {},
        "error": "",
    }
    _atomic_write_json(cache_path, record)
    try:
        if not ct.exists():
            raise FileNotFoundError(f"ct_path missing: {ct}")
        if not ref_dir.exists():
            raise FileNotFoundError(f"annotation_folder missing: {ref_dir}")
        normalized = {"case_id": case_id, "ct_path": str(ct), "annotation_folder": str(ref_dir), **source_row}
        lightweight_doc = _load_lightweight_case_metadata(normalized, ct, case_id, case_root)
        context = _presence_from_lightweight_metadata(lightweight_doc)
        inventory = _load_reference_inventory(ref_dir)
        target_records: dict[str, Any] = {}
        positive_targets: list[str] = []
        needs_heavy_context = allow_heavy_ct_fov and not context.get("has_region_evidence")
        if needs_heavy_context:
            context = _load_case_presence_context(normalized, ct, case_id, case_root)
        landmarks = collect_landmark_evidence(ref_dir, ct_path=ct)
        for target in targets:
            ref = _target_positive_reference(
                ref_dir,
                target,
                inventory=inventory,
                allow_reference_voxel_stats=allow_reference_voxel_stats,
            )
            fallback_fov = _fov_status_for_organ(target, context)
            target_fov = target_fov_eligibility(
                target,
                landmarks=landmarks,
                coverage=context,
                fallback_fov_status=fallback_fov,
            )
            historical = _historical_teacher_positive(
                historical_roots or [],
                case_id=case_id,
                target=target,
                ct=ct,
            )
            if ref["positive"]:
                fov_status = "fully_visible"
                fov_evidence = ref.get("reason") or "positive_reference"
            else:
                fov_status = str(target_fov["fov_status"])
                fov_evidence = str(target_fov["reason"])
            fov_compatible = fov_status not in {"out_of_fov", "unknown"}
            evidence_source = (
                "canonical_reference_positive" if ref["positive"]
                else "historical_teacher_positive" if historical["positive"]
                else "target_specific_fov_compatible" if fov_compatible
                else "none"
            )
            eligible = bool(fov_compatible)
            if historical["positive"] and not fov_compatible:
                evidence_source = "historical_teacher_positive_rejected_by_current_fov"
            target_records[target] = {
                "target": target,
                "fov_status": fov_status,
                "fov_evidence": fov_evidence,
                "reference": ref,
                "historical_teacher": historical,
                "target_fov": target_fov,
                "eligible": eligible,
                "selection_evidence": evidence_source,
                "reason": (
                    "canonical_reference_positive"
                    if eligible and ref["positive"] else
                    "historical_teacher_positive"
                    if eligible and historical["positive"] else
                    "FOV_COMPATIBLE_NO_REFERENCE"
                    if eligible else
                    fov_status
                ),
            }
            if eligible:
                positive_targets.append(target)
        record.update({
            "processing_status": "completed",
            "completed_at": _utc_timestamp(),
            "presence_context": context,
            "targets": target_records,
            "positive_targets": sorted(positive_targets),
            "error": "",
        })
    except Exception as exc:
        record.update({
            "processing_status": "error",
            "completed_at": _utc_timestamp(),
            "error": f"{type(exc).__name__}: {exc}",
        })
    _atomic_write_json(cache_path, record)
    return record


def _write_progress(output_root: Path, completed: int, total: int, status: str, *, reused: int = 0) -> None:
    _atomic_write_json(output_root / "panel_progress.json", {
        "status": status,
        "panel_cases_completed": completed,
        "panel_cases_total": total,
        "panel_cases_reused_from_cache": reused,
        "updated_at": _utc_timestamp(),
    })


def build_smoke_panel(
    *,
    case_manifest: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    require_positive_reference: bool = True,
    allow_heavy_ct_fov: bool = False,
    allow_reference_voxel_stats: bool = True,
    historical_teacher_roots: list[Path] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_root / "panel_state.json", {
        "status": "PANEL_RUNNING",
        "started_at": _utc_timestamp(),
        "case_manifest": str(case_manifest),
    })
    target_contracts = {row["canonical_id"]: row for row in contract_targets(contract_path)}
    targets = [target for target in CADS15_TARGETS if target in target_contracts]
    rows = _read_csv(case_manifest)
    target_coverage: dict[str, list[str]] = {target: [] for target in targets}
    case_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    context_root = output_root / "panel_context"
    reused = 0
    _write_progress(output_root, 0, len(rows), "PANEL_RUNNING")
    try:
        for index, source_row in enumerate(rows):
            case_record = _process_case(
                source_row=source_row,
                index=index,
                targets=targets,
                context_root=context_root,
                allow_heavy_ct_fov=allow_heavy_ct_fov,
                allow_reference_voxel_stats=allow_reference_voxel_stats,
                historical_roots=historical_teacher_roots,
            )
            if case_record.get("cache_status") == "reused":
                reused += 1
            case_id = str(case_record.get("case_id") or _case_id(source_row, index))
            covered_targets = list(case_record.get("positive_targets") or [])
            for target in targets:
                target_record = ((case_record.get("targets") or {}).get(target) or {})
                ref = target_record.get("reference") or {}
                eligible = bool(target_record.get("eligible"))
                audit_rows.append({
                    "case_id": case_id,
                    "target": target,
                    "fov_status": target_record.get("fov_status", "unknown"),
                    "fov_evidence": target_record.get("fov_evidence", ""),
                    "reference_path": ref.get("path", ""),
                    "reference_positive": ref.get("positive", False),
                    "reference_foreground_voxels": ref.get("foreground_voxels", 0),
                    "historical_teacher_path": (target_record.get("historical_teacher") or {}).get("path", ""),
                    "historical_teacher_positive": (target_record.get("historical_teacher") or {}).get("positive", False),
                    "selection_evidence": target_record.get("selection_evidence", ""),
                    "eligible": eligible,
                    "reason": target_record.get("reason") or case_record.get("error") or "",
                })
                if eligible:
                    target_coverage[target].append(case_id)
            if covered_targets:
                case_records.append({
                    "case_id": case_id,
                    "ct_path": case_record.get("ct_path", ""),
                    "annotation_folder": case_record.get("annotation_folder", ""),
                    "targets": sorted(covered_targets),
                    "target_count": len(covered_targets),
                    "selection_reason": "target_specific_fov_compatible",
                })
            _write_progress(output_root, index + 1, len(rows), "PANEL_RUNNING", reused=reused)
    except KeyboardInterrupt:
        _atomic_write_json(output_root / "panel_state.json", {
            "status": "PANEL_FAILED",
            "reason": "KeyboardInterrupt",
            "updated_at": _utc_timestamp(),
        })
        raise

    uncovered = sorted(target for target, case_ids in target_coverage.items() if not case_ids)
    selected: list[dict[str, Any]] = []
    uncovered_remaining = set(targets)
    candidates = sorted(case_records, key=lambda item: item["case_id"])
    while uncovered_remaining:
        best = max(
            candidates,
            key=lambda item: (
                len(set(item["targets"]) & uncovered_remaining),
                -len(item["targets"]),
                "".join(chr(255 - ord(ch)) for ch in item["case_id"]),
            ),
            default=None,
        )
        if best is None or not (set(best["targets"]) & uncovered_remaining):
            break
        chosen_targets = sorted(set(best["targets"]) & uncovered_remaining)
        selected.append({
            "case_id": best["case_id"],
            "ct_path": best["ct_path"],
            "annotation_folder": best["annotation_folder"],
            "targets": chosen_targets,
            "selection_reason": "target_specific_fov_compatible",
        })
        uncovered_remaining -= set(chosen_targets)

    status = "READY_FOR_HPC_SMOKE" if not uncovered_remaining and not uncovered else "NO_FOV_COMPATIBLE_SMOKE_CASE"
    panel = {
        "status": status,
        "read_only": True,
        "case_manifest": str(case_manifest),
        "contract_path": str(contract_path),
        "targets": targets,
        "panel_cache_root": str(context_root),
        "allow_heavy_ct_fov": allow_heavy_ct_fov,
        "allow_reference_voxel_stats": allow_reference_voxel_stats,
        "historical_teacher_roots": [str(path) for path in (historical_teacher_roots or [])],
        "tie_break_rules": [
            "current target-specific FOV evidence is authoritative",
            "canonical target reference is strong selection evidence but is not required for Task2 missing-GT smoke",
            "historical Teacher output is selection evidence only and is rejected when current FOV is out-of-FOV",
            "FOV_COMPATIBLE_NO_REFERENCE is eligible for fresh smoke generation",
            "greedy select max uncovered targets",
            "case_id ascending tie-break",
        ],
        "cases": selected,
        "target_coverage": {target: sorted(case_ids) for target, case_ids in target_coverage.items()},
        "uncovered_targets": sorted(set(uncovered) | uncovered_remaining),
        "audit_rows": audit_rows,
    }
    write_json(output_root / "cads15_smoke_case_panel.json", panel)
    write_csv(
        output_root / "cads15_smoke_case_panel_rows.csv",
        audit_rows,
        [
            "case_id", "target", "fov_status", "fov_evidence", "reference_path",
            "reference_positive", "reference_foreground_voxels", "historical_teacher_path",
            "historical_teacher_positive", "selection_evidence", "eligible", "reason",
        ],
    )
    manifest_rows = [
        {
            "case_id": case["case_id"],
            "ct_path": case["ct_path"],
            "annotation_folder": case["annotation_folder"],
            "targets": ",".join(case["targets"]),
        }
        for case in selected
    ]
    write_csv(output_root / "cads15_selected_case_targets.csv", manifest_rows, ["case_id", "ct_path", "annotation_folder", "targets"])
    summary = {
        "status": status,
        "selected_case_count": len(selected),
        "covered_target_count": len(targets) - len(panel["uncovered_targets"]),
        "target_count": len(targets),
        "uncovered_targets": panel["uncovered_targets"],
        "panel_cases_total": len(rows),
        "panel_cases_reused_from_cache": reused,
    }
    write_json(output_root / "panel_summary.json", summary)
    lines = [
        "# CADS15 Smoke Case Panel",
        "",
        f"- Status: `{status}`",
        f"- Selected cases: `{len(selected)}`",
        f"- Covered targets: `{len(targets) - len(panel['uncovered_targets'])}/{len(targets)}`",
        f"- Reused case caches: `{reused}`",
        f"- Uncovered targets: `{', '.join(panel['uncovered_targets']) or 'none'}`",
        "",
    ]
    for case in selected:
        lines.append(f"- `{case['case_id']}`: `{', '.join(case['targets'])}`")
    (output_root / "cads15_smoke_case_panel.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_write_json(output_root / "panel_state.json", {
        "status": "PANEL_COMPLETED" if status == "READY_FOR_HPC_SMOKE" else "PANEL_FAILED_NO_FOV_CASE",
        "panel_status": status,
        "panel_cases_completed": len(rows),
        "panel_cases_total": len(rows),
        "panel_cases_reused_from_cache": reused,
        "completed_at": _utc_timestamp(),
        "uncovered_targets": panel["uncovered_targets"],
    })
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only CADS15 positive smoke case panel.")
    parser.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST, type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-missing-positive-reference", action="store_true", help="Test-only: allow in-FOV cases without positive reference.")
    parser.add_argument("--allow-heavy-ct-fov", action="store_true", help="Allow fallback CT voxel FOV inference when lightweight evidence is insufficient.")
    parser.add_argument("--no-reference-voxel-stats", action="store_true", help="Use only cached/inventory reference nonzero evidence.")
    parser.add_argument("--historical-teacher-root", action="append", default=[], type=Path, help="Optional previous run root used only as smoke case-selection evidence.")
    args = parser.parse_args()
    panel = build_smoke_panel(
        case_manifest=args.case_manifest.resolve(),
        output_root=args.output_root.resolve(),
        contract_path=args.contract.resolve(),
        require_positive_reference=not args.allow_missing_positive_reference,
        allow_heavy_ct_fov=bool(args.allow_heavy_ct_fov),
        allow_reference_voxel_stats=not bool(args.no_reference_voxel_stats),
        historical_teacher_roots=[path.resolve() for path in args.historical_teacher_root],
    )
    print(json.dumps({"status": panel["status"], "case_count": len(panel["cases"])}, indent=2))
    return 0 if panel["status"] == "READY_FOR_HPC_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
