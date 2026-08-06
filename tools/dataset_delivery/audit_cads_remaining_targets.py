#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_rows, write_csv, write_json  # noqa: E402


DEFAULT_TASK2_TARGETS = REPO_ROOT / "configs" / "dataset_delivery" / "task1" / "task2_generate_targets_23.csv"
DEFAULT_TAXONOMY = REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "model_registry.yaml"
DEFAULT_CLASS_CHECKPOINT_MAP = REPO_ROOT / "configs" / "class_checkpoint_map.parsed.csv"
DEFAULT_ALIAS_CONFIG = REPO_ROOT / "configs" / "model_label_aliases.json"
DEFAULT_COMPLETED_REPORT = REPO_ROOT / "reports" / "task2_cads_existing_audit.json"
DEFAULT_EXISTING_CADS_ROOT = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/"
    "formal_teacher_22_full_volume_20260730/results/cads"
)
DEFAULT_CASE_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)

CADS_COMPLETED_7_PROJECT_RECORD = [
    "cerebrospinal_fluid",
    "eyeball",
    "face",
    "gray_matter",
    "muscle_of_head",
    "scalp",
    "white_matter",
]

STALE_REPORT_CANDIDATE_REMAINING8 = [
    "brain",
    "trachea",
    "brainstem",
    "oral_cavity",
    "compact_bone",
    "spongy_bone",
    "blood",
    "larynx",
]

STATUS_FIELDS = [
    "target_name",
    "model_key",
    "model_group",
    "checkpoint",
    "dataset_task",
    "source_label",
    "canonical_target",
    "is_task2_generate",
    "taxonomy_valid",
    "has_100case_output",
    "existing_mask_count",
    "existing_valid_mask_count",
    "completion_status",
    "completion_source",
    "completion_verification_status",
    "latest_smoke_status",
    "latest_failure_reason",
    "failure_stage",
    "root_cause",
    "next_action",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_text(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_set_fingerprint(targets: list[str]) -> str:
    return _sha256_text(sorted({str(target).strip() for target in targets if str(target).strip()}))


def _load_taxonomy(path: Path) -> set[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("target_organs"), list):
        return {str(item).strip() for item in doc["target_organs"] if str(item).strip()}
    if isinstance(doc, dict) and isinstance(doc.get("organs"), dict):
        return {str(item).strip() for item in doc["organs"].keys() if str(item).strip()}
    if isinstance(doc, dict) and isinstance(doc.get("organ_to_id"), dict):
        return {str(item).strip() for item in doc["organ_to_id"].keys() if str(item).strip()}
    if isinstance(doc, list):
        return {str(item).strip() for item in doc if str(item).strip()}
    raise ValueError(f"Unsupported taxonomy format: {path}")


def load_task2_targets(path: Path) -> set[str]:
    rows = read_csv_rows(path)
    return {row.get("target_name", "").strip() for row in rows if row.get("target_name", "").strip()}


def _load_alias_config(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    return data if isinstance(data.get("models"), dict) else {"models": {}}


def _local_to_global_aliases(alias_config: dict[str, Any], model_key: str) -> dict[str, str]:
    model_alias = ((alias_config.get("models", {}) or {}).get(model_key, {}) or {})
    return {
        str(local).strip(): str(global_name).strip()
        for local, global_name in (model_alias.get("local_to_global", {}) or {}).items()
        if str(local).strip() and str(global_name).strip()
    }


def source_label_for_target(
    *,
    target: str,
    model_key: str,
    registry: dict[str, Any],
    alias_config: dict[str, Any],
) -> str:
    local_to_global = _local_to_global_aliases(alias_config, model_key)
    for local, global_name in local_to_global.items():
        if global_name == target:
            return local
    covered = {str(item).strip() for item in ((registry.get("models", {}) or {}).get(model_key, {}) or {}).get("covered_organs", [])}
    return target if target in covered or not covered else target


def _class_checkpoint_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get("organ", ""): row for row in read_csv_rows(path) if row.get("organ")}


def _checkpoint_path(entry: dict[str, Any]) -> str:
    value = str(entry.get("checkpoint_path") or "")
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())


def _dataset_task(entry: dict[str, Any]) -> str:
    dataset_id = str(entry.get("dataset_id") or "")
    name = str(entry.get("name") or "")
    if dataset_id:
        return f"Dataset{dataset_id}:{name}"
    return name


def compute_cads_targets(
    *,
    task2_targets: Path = DEFAULT_TASK2_TARGETS,
    registry_path: Path = DEFAULT_REGISTRY,
) -> list[str]:
    task2 = load_task2_targets(task2_targets)
    registry = load_registry(registry_path)
    cads_targets: list[str] = []
    for target in sorted(task2):
        candidates = candidate_models_for_organs(registry, [target]).get(target, [])
        if any(str(model).startswith("cads") for model in candidates):
            cads_targets.append(target)
    return cads_targets


def _load_completed_project_record(path: Path) -> list[str]:
    data = _read_json(path)
    completed = data.get("completed_organs")
    if isinstance(completed, list) and completed:
        return sorted({str(item).strip() for item in completed if str(item).strip()})
    return sorted(CADS_COMPLETED_7_PROJECT_RECORD)


def completed_targets_for_cads(cads_targets: list[str], completed_report: Path = DEFAULT_COMPLETED_REPORT) -> list[str]:
    project_record = set(_load_completed_project_record(completed_report))
    return sorted(set(cads_targets) & project_record)


def remaining_targets_for_cads(
    cads_targets: list[str],
    completed_targets: list[str],
) -> list[str]:
    return sorted(set(cads_targets) - set(completed_targets))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _mask_candidates(existing_root: Path, case_id: str, target: str) -> list[Path]:
    return [
        existing_root / case_id / "segmentations" / f"{target}.nii.gz",
        existing_root / case_id / f"{target}.nii.gz",
        existing_root / case_id / "updated" / f"{target}.nii.gz",
        existing_root / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz",
        existing_root / "standard_dataset" / case_id / "segmentations" / f"{target}.nii.gz",
        existing_root / "cases" / case_id / "selected_after_candidate_shapekit" / case_id / "segmentations" / f"{target}.nii.gz",
    ]


def find_target_mask(existing_root: Path, case_id: str, target: str) -> Path | None:
    return next((path for path in _mask_candidates(existing_root, case_id, target) if path.exists()), None)


def validate_nifti_mask(mask_path: Path, ct_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(mask_path),
        "exists": mask_path.exists(),
        "nonempty_file": False,
        "nifti_readable": False,
        "binary": False,
        "foreground_voxels": 0,
        "shape_matches_ct": None,
        "spacing_matches_ct": None,
        "affine_matches_ct": None,
        "valid": False,
        "reason": "",
    }
    if not mask_path.exists():
        result["reason"] = "missing_mask"
        return result
    try:
        stat = mask_path.stat()
        result["size_bytes"] = int(stat.st_size)
        result["mtime_ns"] = int(stat.st_mtime_ns)
        result["nonempty_file"] = stat.st_size > 0
        if stat.st_size <= 0:
            result["reason"] = "empty_file"
            return result
    except Exception as exc:
        result["reason"] = f"stat_failed:{type(exc).__name__}:{exc}"
        return result
    try:
        import nibabel as nib
        import numpy as np

        mask_img = nib.load(str(mask_path))
        arr = np.asanyarray(mask_img.dataobj)
        unique_values = sorted({int(value) for value in np.unique(arr)})
        foreground = int((arr != 0).sum())
        result.update({
            "nifti_readable": True,
            "labels": unique_values,
            "binary": set(unique_values).issubset({0, 1}),
            "foreground_voxels": foreground,
            "shape": list(mask_img.shape[:3]),
            "spacing": [float(x) for x in mask_img.header.get_zooms()[:3]],
        })
        if ct_path and ct_path.exists():
            ct_img = nib.load(str(ct_path))
            mask_spacing = tuple(float(x) for x in mask_img.header.get_zooms()[:3])
            ct_spacing = tuple(float(x) for x in ct_img.header.get_zooms()[:3])
            result.update({
                "ct_path": str(ct_path),
                "ct_shape": list(ct_img.shape[:3]),
                "ct_spacing": [float(x) for x in ct_spacing],
                "shape_matches_ct": tuple(mask_img.shape[:3]) == tuple(ct_img.shape[:3]),
                "spacing_matches_ct": bool(np.allclose(mask_spacing, ct_spacing, rtol=0, atol=1e-5)),
                "affine_matches_ct": bool(np.allclose(mask_img.affine, ct_img.affine, rtol=0, atol=1e-5)),
            })
        geometry_ok = all(
            value is not False
            for value in (result["shape_matches_ct"], result["spacing_matches_ct"], result["affine_matches_ct"])
        )
        result["valid"] = bool(result["binary"] and foreground > 0 and geometry_ok)
        if not result["binary"]:
            result["reason"] = "non_binary_mask"
        elif foreground <= 0:
            result["reason"] = "empty_mask"
        elif not geometry_ok:
            result["reason"] = "geometry_mismatch"
    except Exception as exc:
        result["reason"] = f"nifti_read_failed:{type(exc).__name__}:{exc}"
    return result


def _validate_existing_outputs(
    *,
    targets: list[str],
    case_manifest: Path,
    existing_root: Path,
) -> dict[str, dict[str, Any]]:
    out = {
        target: {
            "has_100case_output": False,
            "existing_mask_count": 0,
            "existing_valid_mask_count": 0,
            "sample_failures": [],
        }
        for target in targets
    }
    if not case_manifest.exists():
        for status in out.values():
            status["verification_status"] = "case_manifest_missing"
        return out
    if not existing_root.exists():
        for status in out.values():
            status["verification_status"] = "history_unmounted"
        return out
    rows = read_csv_rows(case_manifest)
    for index, row in enumerate(rows):
        case_id = _case_id(row, index)
        ct_value = _ct_path(row)
        ct_path = Path(ct_value) if ct_value else None
        for target in targets:
            path = find_target_mask(existing_root, case_id, target)
            if path is None:
                if len(out[target]["sample_failures"]) < 10:
                    out[target]["sample_failures"].append({"case_id": case_id, "reason": "missing_mask"})
                continue
            out[target]["existing_mask_count"] += 1
            validation = validate_nifti_mask(path, ct_path)
            if validation.get("valid"):
                out[target]["existing_valid_mask_count"] += 1
            elif len(out[target]["sample_failures"]) < 10:
                out[target]["sample_failures"].append({
                    "case_id": case_id,
                    "reason": validation.get("reason") or "invalid_mask",
                    "path": str(path),
                })
    expected_cases = len(rows)
    for target, status in out.items():
        status["expected_case_count"] = expected_cases
        status["has_100case_output"] = status["existing_mask_count"] == expected_cases and expected_cases > 0
        status["verification_status"] = (
            "validated_100case"
            if status["existing_valid_mask_count"] == expected_cases and expected_cases > 0
            else "incomplete_or_invalid"
        )
    return out


def _latest_smoke_status(smoke_roots: list[Path], target: str) -> tuple[str, str]:
    candidates: list[Path] = []
    for root in smoke_roots:
        if root.exists():
            candidates.extend(root.rglob("smoke_verdict.json"))
    if not candidates:
        return "not_found", ""
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        verdict = _read_json(path)
        target_rows = verdict.get("target_validation")
        if isinstance(target_rows, dict):
            row = target_rows.get(target)
            if isinstance(row, dict):
                return str(row.get("status") or verdict.get("status") or "unknown"), str(row.get("reason") or "")
        if isinstance(target_rows, list):
            for row in target_rows:
                if isinstance(row, dict) and row.get("target_name") == target:
                    return str(row.get("status") or verdict.get("status") or "unknown"), str(row.get("reason") or "")
        if target in set(verdict.get("requested_targets") or []):
            return str(verdict.get("status") or "unknown"), ";".join(str(x) for x in verdict.get("failures") or [])
    return "not_found", ""


def _root_cause_for_target(
    *,
    target: str,
    source_label: str,
    is_completed: bool,
    existing_valid_count: int,
) -> tuple[str, str, str]:
    if is_completed:
        return "completed_existing_100case", "completed", "no_action_completed"
    if existing_valid_count >= 100:
        return "already_completed_but_registry_stale", "existing_output_validation", "update_registry_or_completion_report"
    if source_label != target:
        return "export_name_mismatch", "post_export_split", "run_cads_remaining8_smoke_after_alias_export_fix"
    return "post_export_missing", "strict_delivery_smoke", "run_cads_remaining8_smoke"


def build_cads_status(
    *,
    output_root: Path,
    task2_targets: Path = DEFAULT_TASK2_TARGETS,
    taxonomy: Path = DEFAULT_TAXONOMY,
    registry_path: Path = DEFAULT_REGISTRY,
    class_checkpoint_map: Path = DEFAULT_CLASS_CHECKPOINT_MAP,
    alias_config_path: Path = DEFAULT_ALIAS_CONFIG,
    completed_report: Path = DEFAULT_COMPLETED_REPORT,
    case_manifest: Path = DEFAULT_CASE_MANIFEST,
    existing_cads_root: Path = DEFAULT_EXISTING_CADS_ROOT,
    smoke_roots: list[Path] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    alias_config = _load_alias_config(alias_config_path)
    taxonomy_names = _load_taxonomy(taxonomy)
    task2 = load_task2_targets(task2_targets)
    class_rows = _class_checkpoint_rows(class_checkpoint_map)
    cads_targets = compute_cads_targets(task2_targets=task2_targets, registry_path=registry_path)
    completed = completed_targets_for_cads(cads_targets, completed_report)
    remaining = remaining_targets_for_cads(cads_targets, completed)
    existing = _validate_existing_outputs(
        targets=cads_targets,
        case_manifest=case_manifest,
        existing_root=existing_cads_root,
    )
    smoke_roots = smoke_roots or [
        Path("/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373"),
        REPO_ROOT / "outputs",
    ]

    rows: list[dict[str, Any]] = []
    for target in cads_targets:
        candidates = candidate_models_for_organs(registry, [target]).get(target, [])
        model_key = next((model for model in candidates if str(model).startswith("cads")), candidates[0] if candidates else "")
        entry = (registry.get("models", {}) or {}).get(model_key, {}) if model_key else {}
        source_label = source_label_for_target(
            target=target,
            model_key=model_key,
            registry=registry,
            alias_config=alias_config,
        )
        smoke_status, smoke_failure = _latest_smoke_status(smoke_roots, target)
        is_completed = target in set(completed)
        existing_status = existing[target]
        root_cause, failure_stage, next_action = _root_cause_for_target(
            target=target,
            source_label=source_label,
            is_completed=is_completed,
            existing_valid_count=int(existing_status.get("existing_valid_mask_count") or 0),
        )
        row = {
            "target_name": target,
            "model_key": model_key,
            "model_group": "CADS",
            "checkpoint": _checkpoint_path(entry),
            "dataset_task": _dataset_task(entry),
            "source_label": source_label,
            "canonical_target": target,
            "is_task2_generate": str(target in task2).lower(),
            "taxonomy_valid": str(target in taxonomy_names).lower(),
            "has_100case_output": str(bool(existing_status.get("has_100case_output"))).lower(),
            "existing_mask_count": int(existing_status.get("existing_mask_count") or 0),
            "existing_valid_mask_count": int(existing_status.get("existing_valid_mask_count") or 0),
            "completion_status": "completed" if is_completed else "remaining",
            "completion_source": "reports/task2_cads_existing_audit.json" if is_completed else "",
            "completion_verification_status": existing_status.get("verification_status") or "",
            "latest_smoke_status": smoke_status,
            "latest_failure_reason": smoke_failure,
            "failure_stage": failure_stage,
            "root_cause": root_cause,
            "next_action": next_action,
            "_candidate_models": ";".join(candidates),
            "_checkpoint_map_models": (class_rows.get(source_label) or class_rows.get(target) or {}).get("candidate_models", ""),
        }
        rows.append(row)

    all_rows = [row for row in rows]
    completed_rows = [row for row in rows if row["target_name"] in set(completed)]
    remaining_rows = [row for row in rows if row["target_name"] in set(remaining)]
    for path, target_rows in (
        (output_root / "cads_all_targets.csv", all_rows),
        (output_root / "cads_completed_targets.csv", completed_rows),
        (output_root / "cads_remaining_targets.csv", remaining_rows),
    ):
        write_csv(path, target_rows, STATUS_FIELDS)

    stale_candidates_in_taxonomy = sorted(set(STALE_REPORT_CANDIDATE_REMAINING8) & taxonomy_names)
    stale_candidates_in_task2 = sorted(set(STALE_REPORT_CANDIDATE_REMAINING8) & task2)
    stale_candidates_outside_current_task2 = sorted(set(STALE_REPORT_CANDIDATE_REMAINING8) - task2)
    summary = {
        "status": "success",
        "read_only": True,
        "source_data_mutation": False,
        "canonical_config_mutation": False,
        "task2_targets_path": str(task2_targets),
        "taxonomy_path": str(taxonomy),
        "registry_path": str(registry_path),
        "alias_config_path": str(alias_config_path),
        "existing_cads_root": str(existing_cads_root),
        "case_manifest": str(case_manifest),
        "cads_all_targets": cads_targets,
        "cads_all_count": len(cads_targets),
        "cads_completed_targets": completed,
        "cads_completed_count": len(completed),
        "cads_remaining_targets": remaining,
        "cads_remaining_count": len(remaining),
        "cads_all_targets_sha256": target_set_fingerprint(cads_targets),
        "cads_remaining_targets_sha256": target_set_fingerprint(remaining),
        "stale_report_candidate_remaining8": STALE_REPORT_CANDIDATE_REMAINING8,
        "stale_candidates_in_taxonomy": stale_candidates_in_taxonomy,
        "stale_candidates_in_current_task2": stale_candidates_in_task2,
        "stale_candidates_outside_current_task2": stale_candidates_outside_current_task2,
        "rows": rows,
        "existing_output_validation": existing,
    }
    write_json(output_root / "cads_target_status.json", summary)
    write_markdown_report(output_root / "cads_target_status.md", summary)
    return summary


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CADS Task 2 Target Status",
        "",
        f"- Read only: `{summary['read_only']}`",
        f"- CADS all targets: `{summary['cads_all_count']}`",
        f"- Completed targets: `{summary['cads_completed_count']}`",
        f"- Remaining targets: `{summary['cads_remaining_count']}`",
        f"- Existing CADS root: `{summary['existing_cads_root']}`",
        "",
        "## Canonical CADS Targets",
        "",
        ", ".join(f"`{target}`" for target in summary["cads_all_targets"]),
        "",
        "## Completed",
        "",
        ", ".join(f"`{target}`" for target in summary["cads_completed_targets"]) or "`none`",
        "",
        "## Remaining",
        "",
        ", ".join(f"`{target}`" for target in summary["cads_remaining_targets"]) or "`none`",
        "",
        "## Stale Candidate List Check",
        "",
        "The old CADS remaining list is compared against the current canonical Task 2 target CSV.",
        "",
        f"- In current Task 2: `{', '.join(summary['stale_candidates_in_current_task2']) or 'none'}`",
        f"- Outside current Task 2: `{', '.join(summary['stale_candidates_outside_current_task2']) or 'none'}`",
        "",
        "## Per-target Status",
        "",
        "| target | model | source label | status | masks | valid | root cause | next action |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {target_name} | {model_key} | {source_label} | {completion_status} | "
            "{existing_mask_count} | {existing_valid_mask_count} | {root_cause} | {next_action} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_csv_dict_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get(key, ""): row for row in read_csv_rows(path) if row.get(key)}


def strict_delivery_failure_reasons(path: Path) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    if not path.exists():
        return failures
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            organ = str(row.get("organ") or "")
            reason = str(row.get("reason") or "")
            if organ and reason:
                failures.setdefault(organ, []).append(reason)
    return failures


def _find_inference_summary(run_out: Path, case_id: str, model_key: str) -> dict[str, Any]:
    candidates = [
        run_out / "cases" / case_id / "raw_predictions" / model_key / case_id / "inference_summary.json",
        run_out / "cases" / case_id / "hierarchical_predictions" / model_key / case_id / "inference_summary.json",
        run_out / "cases" / case_id / "hierarchical_predictions" / model_key / "inference_summary.json",
    ]
    for path in candidates:
        data = _read_json(path)
        if data:
            return data
    for path in sorted(run_out.rglob("inference_summary.json")):
        data = _read_json(path)
        if data.get("model_key") == model_key or data.get("model") == model_key:
            return data
    return {}


def _selected_mask_path(run_out: Path, case_id: str, target: str) -> Path:
    return next(
        (
            path
            for path in [
                run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz",
                run_out / "standard_dataset" / case_id / "segmentations" / f"{target}.nii.gz",
                run_out / "cases" / case_id / "selected_after_candidate_shapekit" / case_id / "segmentations" / f"{target}.nii.gz",
            ]
            if path.exists()
        ),
        run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz",
    )


def validate_cads_smoke(
    *,
    run_out: Path,
    output_root: Path,
    case_id: str,
    targets: list[str],
    target_model_map: dict[str, str],
    run_return_code: int = 0,
) -> dict[str, Any]:
    summary = _read_json(run_out / "run_summary.json")
    plan = _read_json(run_out / "annotation_versions" / case_id / "case_execution_plan.json")
    strict_reasons = strict_delivery_failure_reasons(run_out / "strict_delivery_failures.csv")
    plan_organs = set((plan.get("per_organ") or {}).keys())
    teacher_run_list = set(plan.get("teacher_run_list") or [])
    ct_path = Path(str(plan.get("ct_path") or ""))
    target_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for target in targets:
        model_key = target_model_map.get(target, "")
        inference = _find_inference_summary(run_out, case_id, model_key) if model_key else {}
        mask_path = _selected_mask_path(run_out, case_id, target)
        mask_validation = validate_nifti_mask(mask_path, ct_path)
        reasons: list[str] = []
        if target not in plan_organs:
            reasons.append("target_not_in_execution_plan")
        if model_key and model_key not in teacher_run_list:
            reasons.append("model_not_scheduled")
        if inference and inference.get("return_code") not in (0, None):
            reasons.append(f"model_return_code:{inference.get('return_code')}")
        if inference and inference.get("status") not in {"success", None}:
            reasons.append(f"inference_status:{inference.get('status')}")
        for bad in strict_reasons.get(target, []):
            reasons.append(f"strict_delivery_failure:{bad}")
        if not mask_validation.get("valid"):
            reasons.append(mask_validation.get("reason") or "invalid_mask")
        row = {
            "target_name": target,
            "model_key": model_key,
            "requested": "true",
            "in_execution_plan": str(target in plan_organs).lower(),
            "teacher_scheduled": str(model_key in teacher_run_list if model_key else False).lower(),
            "model_return_code": inference.get("return_code", ""),
            "inference_status": inference.get("status", ""),
            "mask_path": str(mask_path),
            "exists": str(mask_validation.get("exists")).lower(),
            "foreground_voxels": mask_validation.get("foreground_voxels", 0),
            "binary": str(mask_validation.get("binary")).lower(),
            "shape_matches_ct": str(mask_validation.get("shape_matches_ct")).lower(),
            "spacing_matches_ct": str(mask_validation.get("spacing_matches_ct")).lower(),
            "affine_matches_ct": str(mask_validation.get("affine_matches_ct")).lower(),
            "status": "passed" if not reasons else "failed",
            "reason": ";".join(reasons),
        }
        if reasons:
            failures.append(f"{target}:{';'.join(reasons)}")
        target_rows.append(row)

    if run_return_code != 0:
        failures.append(f"run_loop_return_code:{run_return_code}")
    if summary and summary.get("status") != "success":
        failures.append(f"run_summary_status:{summary.get('status')}")
    if int(summary.get("strict_delivery_failure_count") or 0) != 0:
        failures.append(f"strict_delivery_failure_count:{summary.get('strict_delivery_failure_count')}")

    verdict = {
        "status": "passed" if not failures else "failed",
        "case_id": case_id,
        "run_out": str(run_out),
        "run_return_code": run_return_code,
        "requested_targets": targets,
        "teacher_run_list": sorted(teacher_run_list),
        "passed_targets": [row["target_name"] for row in target_rows if row["status"] == "passed"],
        "failed_targets": [row["target_name"] for row in target_rows if row["status"] != "passed"],
        "target_validation": target_rows,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "target_validation.csv",
        target_rows,
        [
            "target_name",
            "model_key",
            "requested",
            "in_execution_plan",
            "teacher_scheduled",
            "model_return_code",
            "inference_status",
            "mask_path",
            "exists",
            "foreground_voxels",
            "binary",
            "shape_matches_ct",
            "spacing_matches_ct",
            "affine_matches_ct",
            "status",
            "reason",
        ],
    )
    write_json(output_root / "target_validation.json", {"rows": target_rows})
    write_json(output_root / "smoke_verdict.json", verdict)
    md = [
        "# CADS Remaining-target Strict-delivery Smoke",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Case: `{case_id}`",
        f"- Run output: `{run_out}`",
        f"- Passed targets: `{', '.join(verdict['passed_targets']) or 'none'}`",
        f"- Failed targets: `{', '.join(verdict['failed_targets']) or 'none'}`",
        "",
        "## Failures",
        "",
        *(f"- `{failure}`" for failure in failures),
    ]
    (output_root / "smoke_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only CADS Task 2 target status audit and smoke validator.")
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit", help="Recompute canonical CADS targets and remaining target status.")
    audit.add_argument("--output-root", required=True, type=Path)
    audit.add_argument("--task2-targets", default=DEFAULT_TASK2_TARGETS, type=Path)
    audit.add_argument("--taxonomy", default=DEFAULT_TAXONOMY, type=Path)
    audit.add_argument("--registry", default=DEFAULT_REGISTRY, type=Path)
    audit.add_argument("--class-checkpoint-map", default=DEFAULT_CLASS_CHECKPOINT_MAP, type=Path)
    audit.add_argument("--alias-config", default=DEFAULT_ALIAS_CONFIG, type=Path)
    audit.add_argument("--completed-report", default=DEFAULT_COMPLETED_REPORT, type=Path)
    audit.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST, type=Path)
    audit.add_argument("--existing-cads-root", default=DEFAULT_EXISTING_CADS_ROOT, type=Path)
    audit.add_argument("--smoke-root", action="append", type=Path, default=[])

    validate = sub.add_parser("validate-smoke", help="Validate a CADS remaining-target strict-delivery smoke run.")
    validate.add_argument("--run-out", required=True, type=Path)
    validate.add_argument("--output-root", required=True, type=Path)
    validate.add_argument("--case-id", required=True)
    validate.add_argument("--targets", required=True, help="Comma-separated canonical targets.")
    validate.add_argument("--target-model-map", required=True, type=Path)
    validate.add_argument("--run-return-code", type=int, default=0)

    args = parser.parse_args()
    if args.command == "audit":
        summary = build_cads_status(
            output_root=args.output_root,
            task2_targets=args.task2_targets,
            taxonomy=args.taxonomy,
            registry_path=args.registry,
            class_checkpoint_map=args.class_checkpoint_map,
            alias_config_path=args.alias_config,
            completed_report=args.completed_report,
            case_manifest=args.case_manifest,
            existing_cads_root=args.existing_cads_root,
            smoke_roots=args.smoke_root or None,
        )
        print(json.dumps({
            "status": summary["status"],
            "output_root": str(args.output_root),
            "cads_all_count": summary["cads_all_count"],
            "cads_completed_count": summary["cads_completed_count"],
            "cads_remaining_count": summary["cads_remaining_count"],
            "cads_remaining_targets": summary["cads_remaining_targets"],
        }, indent=2))
        return 0
    if args.command == "validate-smoke":
        target_model_map = _read_json(args.target_model_map)
        targets = [item.strip() for item in args.targets.replace(";", ",").split(",") if item.strip()]
        verdict = validate_cads_smoke(
            run_out=args.run_out,
            output_root=args.output_root,
            case_id=args.case_id,
            targets=targets,
            target_model_map={str(k): str(v) for k, v in target_model_map.items()},
            run_return_code=args.run_return_code,
        )
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["status"] == "passed" else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
