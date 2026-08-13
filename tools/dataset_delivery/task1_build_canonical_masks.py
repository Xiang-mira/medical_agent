#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "agent-harness"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.medai.core.target_space import canonical_target_name, validate_formal_373_target_space  # noqa: E402
from tools.dataset_delivery.delivery_lib import (  # noqa: E402
    DeliveryError,
    NIFTI_SUFFIX,
    read_alias_groups,
    read_csv_rows,
    read_rename_mapping,
    safe_label_name,
    sha256_file,
    sha256_file_if_exists,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_WORKSPACE = Path("/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812")
DEFAULT_MAPPING = REPO_ROOT / "configs/dataset_delivery/task1/organ_rename_mapping_373.csv"
DEFAULT_NON_RENAME = REPO_ROOT / "configs/dataset_delivery/task1/non_rename_decisions.csv"
DEFAULT_ALIAS_GROUPS = REPO_ROOT / "configs/dataset_delivery/task1/task1_alias_groups.csv"
DEFAULT_BOUNDARY = REPO_ROOT / "configs/dataset_delivery/task1/task_boundary_classification.csv"
DEFAULT_TARGET_CONFIG = REPO_ROOT / "configs/student_3d_prompt_target_organs.json"

AUDIT_FIELDS = [
    "case_id",
    "source_name",
    "source_path",
    "canonical_name",
    "action",
    "mapping_status",
    "mapping_source",
    "reason",
    "destination_path",
    "source_checksum",
    "destination_checksum",
    "collision_status",
    "selected_source_name",
    "all_source_aliases",
    "selection_policy",
]
COLLISION_FIELDS = [
    "case_id",
    "canonical_name",
    "collision_status",
    "source_names",
    "source_paths",
    "source_checksums",
    "destination_path",
    "reason",
]
OUTPUT_FIELDS = [
    "case_id",
    "canonical_target",
    "destination_path",
    "selected_source_path",
    "selected_source_name",
    "resolution_type",
    "all_source_aliases",
    "mapping_source",
    "selection_policy",
    "source_checksum",
    "destination_checksum",
]
UNMAPPED_FIELDS = ["case_id", "source_name", "source_path", "action", "reason", "source_checksum"]
COVERAGE_FIELDS = [
    "case_id",
    "raw_target_name",
    "canonical_target",
    "coverage_status",
    "source_label_names",
    "source_mask_path",
    "source_mask_sha256",
    "needs_task2_generation",
    "reason",
]
NON_ANATOMY_TOKENS = (
    "cancer",
    "lesion",
    "metastases",
    "metastasis",
    "nodule",
    "pdac",
    "pnet",
    "primaries",
    "primary",
    "tumor",
    "tumour",
)
OUTSIDE_373_SOURCE_NAMES = {
    "abdominal_tissue",
    "intermuscular_adipose_tissue",
    "lymph_nodes",
    "unknown_tissue",
    "visceral_adipose_tissue",
}
GRANULARITY_OR_OUTSIDE_373_SOURCE_NAMES = {
    "liver_vessels",
    "muscles",
    "nasal_cavity",
    "skeletal_muscle",
}


def _canonical_target_contract(target_config: Path) -> tuple[set[str], int]:
    raw_targets = _raw_target_names(target_config)
    return {canonical_target_name(str(item)) for item in raw_targets}, len(raw_targets)


def _raw_target_names(target_config: Path) -> list[str]:
    validation = validate_formal_373_target_space(target_config)
    if validation.get("status") != "success" and target_config.resolve() == DEFAULT_TARGET_CONFIG.resolve():
        raise DeliveryError(f"Formal target config failed validation: {target_config}")
    doc = json.loads(target_config.read_text(encoding="utf-8"))
    return [str(item) for item in doc.get("target_organs", [])]


def _workspace_paths(workspace_root: Path) -> dict[str, Path]:
    return {
        "original": workspace_root / "inputs" / "masks_original",
        "canonical": workspace_root / "inputs" / "masks_373_canonical",
        "manifests": workspace_root / "manifests",
    }


def _case_dirs(original_root: Path, case_manifest: Path | None) -> list[tuple[str, Path]]:
    if case_manifest:
        out = []
        for index, row in enumerate(read_csv_rows(case_manifest)):
            case_id = str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()
            raw = row.get("original_annotation_folder") or row.get("annotation_folder") or row.get("reference_mask_dir") or ""
            mask_dir = Path(raw) if raw else original_root / case_id / "segmentations"
            out.append((case_id, mask_dir))
        return out
    return [(case_dir.name, case_dir / "segmentations") for case_dir in sorted(original_root.iterdir()) if case_dir.is_dir()]


def _load_non_rename(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    if not path.exists():
        return out
    for row in read_csv_rows(path):
        source = safe_label_name(row.get("source_name", ""))
        normalized = dict(row)
        normalized["source_name"] = source
        normalized["target_name"] = safe_label_name(row.get("target_name", ""))
        out.setdefault(source, []).append(normalized)
    return out


def _mapping_indexes(mapping: Path) -> tuple[dict[str, Any], dict[str, list[Any]], dict[str, list[Any]], dict[tuple[str, str], int]]:
    confirmed: dict[str, Any] = {}
    pending: dict[str, list[Any]] = {}
    rejected: dict[str, list[Any]] = {}
    order: dict[tuple[str, str], int] = {}
    for index, row in enumerate(read_rename_mapping(mapping)):
        order[(row.source_name, row.target_name)] = index
        if row.status == "confirmed":
            confirmed[row.source_name] = row
        elif row.status == "pending_review":
            pending.setdefault(row.source_name, []).append(row)
        elif row.status == "rejected":
            rejected.setdefault(row.source_name, []).append(row)
    return confirmed, pending, rejected, order


def _non_rename_action(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    classifications = {str(row.get("classification") or "") for row in rows}
    reasons = ";".join(str(row.get("reason") or "") for row in rows if row.get("reason"))
    if "task2_generate" in classifications:
        return "NONRENAME_TASK2", "confirmed_non_rename", reasons or "boundary_classification_task2_generate"
    if "exclude" in classifications:
        return "EXCLUDED_NON_ANATOMY", "confirmed_non_rename", reasons or "boundary_classification_exclude"
    return "PENDING_MAPPING", "pending_non_rename_review", reasons or "boundary_review_pending"


def _is_non_anatomy(name: str) -> bool:
    parts = set(name.split("_"))
    return any(token in parts or token in name for token in NON_ANATOMY_TOKENS)


def _outside_373_action(name: str) -> tuple[str, str] | None:
    if name in OUTSIDE_373_SOURCE_NAMES:
        return "EXCLUDED_OUTSIDE_373", "source_label_outside_authoritative_373_target_set"
    if name in GRANULARITY_OR_OUTSIDE_373_SOURCE_NAMES:
        return "OUTSIDE_373_OR_GRANULARITY_MISMATCH", "source_label_not_confirmed_as_same_granularity_373_target"
    return None


def _audit_row(
    *,
    case_id: str,
    source_name: str,
    source_path: Path,
    canonical_name: str,
    action: str,
    mapping_status: str,
    mapping_source: str,
    reason: str,
    destination_path: Path | None,
    collision_status: str = "",
    selected_source_name: str = "",
    all_source_aliases: str = "",
    selection_policy: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_name": source_name,
        "source_path": str(source_path),
        "canonical_name": canonical_name,
        "action": action,
        "mapping_status": mapping_status,
        "mapping_source": mapping_source,
        "reason": reason,
        "destination_path": str(destination_path or ""),
        "source_checksum": sha256_file_if_exists(source_path),
        "destination_checksum": sha256_file_if_exists(destination_path) if destination_path else "",
        "collision_status": collision_status,
        "selected_source_name": selected_source_name,
        "all_source_aliases": all_source_aliases,
        "selection_policy": selection_policy,
    }


def _confirmed_alias_sources(alias_groups: dict[str, dict[str, Any]], canonical_name: str) -> set[str]:
    alias = alias_groups.get(canonical_name, {})
    return {
        safe_label_name(str(row.get("source_name", "")))
        for row in alias.get("rows", [])
        if str(row.get("status") or "").strip().lower() == "confirmed"
    }


def _confirmed_mapping_sources(confirmed: dict[str, Any], canonical_name: str) -> set[str]:
    return {
        source
        for source, row in confirmed.items()
        if canonical_target_name(str(row.target_name)) == canonical_name and row.status == "confirmed"
    }


def _alias_order(alias_groups: dict[str, dict[str, Any]], canonical_name: str) -> dict[str, int]:
    alias = alias_groups.get(canonical_name, {})
    return {
        safe_label_name(str(row.get("source_name", ""))): index
        for index, row in enumerate(alias.get("rows", []))
    }


def _select_canonical_entry(
    *,
    canonical_name: str,
    entries: list[tuple[Path, str, str, str, str, str]],
    alias_groups: dict[str, dict[str, Any]],
    mapping_order: dict[tuple[str, str], int],
) -> tuple[tuple[Path, str, str, str, str, str], str]:
    alias_rank = _alias_order(alias_groups, canonical_name)

    def rank(entry: tuple[Path, str, str, str, str, str]) -> tuple[int, int, int, str]:
        _source_path, source_name, action, _mapping_status, _mapping_source, _reason = entry
        canonical_source = 0 if action == "KEEP_CANONICAL" and source_name == canonical_name else 1
        alias_position = alias_rank.get(source_name, 9999)
        mapping_position = mapping_order.get((source_name, canonical_name), 9999)
        return canonical_source, alias_position, mapping_position, source_name

    selected = sorted(entries, key=rank)[0]
    if selected[2] == "KEEP_CANONICAL" and selected[1] == canonical_name and len(entries) > 1:
        policy = "canonical_source_over_confirmed_aliases"
    elif len(entries) > 1:
        policy = "confirmed_alias_priority_alias_groups_then_mapping_order"
    else:
        policy = "single_source"
    return selected, policy


def _plan_case(
    *,
    case_id: str,
    source_dir: Path,
    dest_dir: Path,
    target_set: set[str],
    confirmed: dict[str, Any],
    pending: dict[str, list[Any]],
    rejected: dict[str, list[Any]],
    mapping_order: dict[tuple[str, str], int],
    non_rename: dict[str, list[dict[str, str]]],
    alias_groups: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    if not source_dir.is_dir():
        unmapped_rows.append({"case_id": case_id, "source_name": "", "source_path": str(source_dir), "action": "UNMAPPED", "reason": "source_mask_dir_missing", "source_checksum": ""})
        return audit_rows, collision_rows, unmapped_rows

    files = sorted(path for path in source_dir.glob(f"*{NIFTI_SUFFIX}") if path.is_file())
    planned: dict[str, list[tuple[Path, str, str, str, str, str]]] = {}
    deferred_rows: list[dict[str, Any]] = []
    for source_path in files:
        source_name = safe_label_name(source_path.name)
        canonical_source = canonical_target_name(source_name)
        if canonical_source in target_set:
            canonical_name = canonical_source
            planned.setdefault(canonical_name, []).append((source_path, source_name, "KEEP_CANONICAL", "already_canonical", "formal_373_target_config", "source_already_matches_canonical_target"))
            continue
        if source_name in confirmed:
            row = confirmed[source_name]
            canonical_name = canonical_target_name(row.target_name)
            if canonical_name not in target_set:
                raise DeliveryError(f"confirmed mapping target is outside authoritative target contract: {source_name}->{canonical_name}")
            planned.setdefault(canonical_name, []).append((source_path, source_name, "RENAMED", row.status, "organ_rename_mapping_373.csv", row.reason or "confirmed_task1_rename"))
            continue
        if source_name in non_rename:
            action, status, reason = _non_rename_action(non_rename[source_name])
            deferred_rows.append(
                _audit_row(
                    case_id=case_id,
                    source_name=source_name,
                    source_path=source_path,
                    canonical_name="",
                    action=action,
                    mapping_status=status,
                    mapping_source="non_rename_decisions.csv",
                    reason=reason,
                    destination_path=None,
                )
            )
            continue
        if _is_non_anatomy(source_name):
            deferred_rows.append(
                _audit_row(
                    case_id=case_id,
                    source_name=source_name,
                    source_path=source_path,
                    canonical_name="",
                    action="EXCLUDED_NON_ANATOMY",
                    mapping_status="excluded",
                    mapping_source="non_anatomy_name_policy",
                    reason="non_anatomy_label_excluded_from_canonical_tree",
                    destination_path=None,
                )
            )
            continue
        outside_action = _outside_373_action(source_name)
        if outside_action:
            action, reason = outside_action
            row = _audit_row(
                case_id=case_id,
                source_name=source_name,
                source_path=source_path,
                canonical_name="",
                action=action,
                mapping_status="excluded",
                mapping_source="task1_outside_373_policy",
                reason=reason,
                destination_path=None,
            )
            deferred_rows.append(row)
            continue
        if source_name in pending:
            row = pending[source_name][0]
            canonical_name = canonical_target_name(row.target_name)
            audit = _audit_row(
                case_id=case_id,
                source_name=source_name,
                source_path=source_path,
                canonical_name=canonical_name,
                action="PENDING_MAPPING",
                mapping_status="pending_review",
                mapping_source="organ_rename_mapping_373.csv",
                reason=row.reason or "mapping_status_pending_review",
                destination_path=None,
            )
            deferred_rows.append(audit)
            unmapped_rows.append({"case_id": case_id, "source_name": source_name, "source_path": str(source_path), "action": "PENDING_MAPPING", "reason": audit["reason"], "source_checksum": audit["source_checksum"]})
            continue
        if source_name in rejected:
            row = rejected[source_name][0]
            canonical_name = canonical_target_name(row.target_name)
            audit = _audit_row(
                case_id=case_id,
                source_name=source_name,
                source_path=source_path,
                canonical_name=canonical_name,
                action="PENDING_MAPPING",
                mapping_status="rejected",
                mapping_source="organ_rename_mapping_373.csv",
                reason=row.reason or "mapping_status_rejected",
                destination_path=None,
            )
            deferred_rows.append(audit)
            unmapped_rows.append({"case_id": case_id, "source_name": source_name, "source_path": str(source_path), "action": "PENDING_MAPPING", "reason": audit["reason"], "source_checksum": audit["source_checksum"]})
            continue
        else:
            reason = "no_confirmed_task1_mapping_or_formal_target"
            action = "UNMAPPED"
            status = "unmapped"
        row = _audit_row(
            case_id=case_id,
            source_name=source_name,
            source_path=source_path,
            canonical_name="",
            action=action,
            mapping_status=status,
            mapping_source="task1_artifacts",
            reason=reason,
            destination_path=None,
        )
        deferred_rows.append(row)
        unmapped_rows.append({"case_id": case_id, "source_name": source_name, "source_path": str(source_path), "action": action, "reason": reason, "source_checksum": row["source_checksum"]})

    for canonical_name, entries in sorted(planned.items()):
        destination = dest_dir / f"{canonical_name}{NIFTI_SUFFIX}"
        selected_entry, selection_policy = _select_canonical_entry(
            canonical_name=canonical_name,
            entries=entries,
            alias_groups=alias_groups,
            mapping_order=mapping_order,
        )
        selected_source_name = selected_entry[1]
        all_source_aliases = ";".join(entry[1] for entry in entries)
        if len(entries) > 1:
            sources = {entry[1] for entry in entries}
            confirmed_aliases = _confirmed_alias_sources(alias_groups, canonical_name) | _confirmed_mapping_sources(confirmed, canonical_name)
            unapproved = {
                source
                for source_path, source, action, mapping_status, _mapping_source, _reason in entries
                if action != "KEEP_CANONICAL" and (mapping_status != "confirmed" or source not in confirmed_aliases)
            }
            if unapproved:
                reason = "multiple_sources_map_to_same_canonical_target_without_confirmed_alias_identity"
                collision_rows.append(
                    {
                        "case_id": case_id,
                        "canonical_name": canonical_name,
                        "collision_status": "UNAPPROVED_COLLISION",
                        "source_names": ";".join(sorted(sources)),
                        "source_paths": ";".join(str(entry[0]) for entry in entries),
                        "source_checksums": json.dumps({entry[1]: sha256_file(entry[0]) for entry in entries}, sort_keys=True),
                        "destination_path": str(destination),
                        "reason": reason,
                    }
                )
                for source_path, source_name, action, mapping_status, mapping_source, _reason in entries:
                    audit_rows.append(
                        _audit_row(
                            case_id=case_id,
                            source_name=source_name,
                            source_path=source_path,
                            canonical_name=canonical_name,
                            action="UNAPPROVED_COLLISION",
                            mapping_status=mapping_status,
                            mapping_source=mapping_source,
                            reason=reason,
                            destination_path=destination,
                            collision_status="UNAPPROVED_COLLISION",
                            selected_source_name=selected_source_name,
                            all_source_aliases=all_source_aliases,
                            selection_policy=selection_policy,
                        )
                    )
                continue
            for source_path, source_name, action, mapping_status, mapping_source, _reason in entries:
                if source_name == selected_source_name:
                    resolved_action = "CANONICAL_SOURCE_SELECTED" if action == "KEEP_CANONICAL" else "SELECTED_CONFIRMED_ALIAS"
                    reason = "selected_for_canonical_output"
                else:
                    resolved_action = "REDUNDANT_CONFIRMED_ALIAS"
                    reason = "confirmed_alias_not_selected_by_deterministic_precedence"
                audit_rows.append(
                    _audit_row(
                        case_id=case_id,
                        source_name=source_name,
                        source_path=source_path,
                        canonical_name=canonical_name,
                        action=resolved_action,
                        mapping_status=mapping_status,
                        mapping_source=mapping_source,
                        reason=reason,
                        destination_path=destination,
                        selected_source_name=selected_source_name,
                        all_source_aliases=all_source_aliases,
                        selection_policy=selection_policy,
                    )
                )
            continue
        source_path, source_name, action, mapping_status, mapping_source, reason = entries[0]
        resolved_action = "CANONICAL_SOURCE_SELECTED" if action == "KEEP_CANONICAL" else "CONFIRMED_RENAME" if action == "RENAMED" else action
        if resolved_action == "PENDING_MAPPING":
            unmapped_rows.append({"case_id": case_id, "source_name": source_name, "source_path": str(source_path), "action": resolved_action, "reason": reason, "source_checksum": sha256_file_if_exists(source_path)})
        audit_rows.append(
            _audit_row(
                case_id=case_id,
                source_name=source_name,
                source_path=source_path,
                canonical_name=canonical_name,
                action=resolved_action,
                mapping_status=mapping_status,
                mapping_source=mapping_source,
                reason=reason,
                destination_path=destination,
                selected_source_name=source_name,
                all_source_aliases=source_name,
                selection_policy=selection_policy,
            )
        )
    audit_rows.extend(deferred_rows)
    return audit_rows, collision_rows, unmapped_rows


def _copy_planned_rows(rows: list[dict[str, Any]], *, resume: bool) -> None:
    for row in rows:
        if row["action"] not in {"CANONICAL_SOURCE_SELECTED", "CONFIRMED_RENAME", "SELECTED_CONFIRMED_ALIAS"}:
            continue
        src = Path(str(row["source_path"]))
        dst = Path(str(row["destination_path"]))
        if not src.is_file():
            raise DeliveryError(f"source mask missing during apply: {src}")
        src_hash = sha256_file(src)
        if dst.exists():
            dst_hash = sha256_file(dst)
            if dst_hash == src_hash and resume:
                continue
            raise DeliveryError(f"destination exists with different content or resume disabled: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != src_hash:
            raise DeliveryError(f"checksum mismatch after copy: {src} -> {dst}")


def _output_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in audit_rows:
        if row["action"] not in {"CANONICAL_SOURCE_SELECTED", "CONFIRMED_RENAME", "SELECTED_CONFIRMED_ALIAS"}:
            continue
        resolution_type = "KEEP_CANONICAL" if row["action"] == "CANONICAL_SOURCE_SELECTED" else "CONFIRMED_RENAME"
        if row["selection_policy"] == "canonical_source_over_confirmed_aliases":
            resolution_type = "CANONICAL_SOURCE_OVER_ALIAS"
        elif row["selection_policy"] == "confirmed_alias_priority_alias_groups_then_mapping_order":
            resolution_type = "CONFIRMED_ALIAS_PRIORITY"
        rows.append(
            {
                "case_id": row["case_id"],
                "canonical_target": row["canonical_name"],
                "destination_path": row["destination_path"],
                "selected_source_path": row["source_path"],
                "selected_source_name": row["source_name"],
                "resolution_type": resolution_type,
                "all_source_aliases": row["all_source_aliases"],
                "mapping_source": row["mapping_source"],
                "selection_policy": row["selection_policy"],
                "source_checksum": row["source_checksum"],
                "destination_checksum": row["destination_checksum"],
            }
        )
    return rows


def _original_source_index(source_dir: Path) -> dict[str, Path]:
    if not source_dir.is_dir():
        return {}
    return {
        safe_label_name(path.name): path
        for path in sorted(source_dir.glob(f"*{NIFTI_SUFFIX}"))
        if path.is_file()
    }


def _coverage_gap_index(non_rename: dict[str, list[dict[str, str]]], target_set: set[str]) -> dict[str, set[str]]:
    gaps: dict[str, set[str]] = {}
    for source_name, rows in non_rename.items():
        for row in rows:
            if str(row.get("classification") or "") != "task2_generate":
                continue
            target = canonical_target_name(safe_label_name(row.get("target_name", "")))
            if target in target_set:
                gaps.setdefault(target, set()).add(source_name)
    return gaps


def audit_task1_source_coverage(
    *,
    workspace_root: Path,
    non_rename_decisions: Path = DEFAULT_NON_RENAME,
    target_config: Path = DEFAULT_TARGET_CONFIG,
    case_manifest: Path | None = None,
    expected_case_count: int = 103,
) -> dict[str, Any]:
    """Write a label-coverage manifest without inferring anatomical absence."""
    workspace_root = workspace_root.resolve()
    paths = _workspace_paths(workspace_root)
    raw_targets = _raw_target_names(target_config)
    target_set = {canonical_target_name(str(item)) for item in raw_targets}
    non_rename = _load_non_rename(non_rename_decisions)
    gap_sources_by_target = _coverage_gap_index(non_rename, target_set)
    case_dirs = _case_dirs(paths["original"], case_manifest)

    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for case_id, source_dir in case_dirs:
        original_sources = _original_source_index(source_dir)
        canonical_dir = paths["canonical"] / case_id / "segmentations"
        for raw_target in raw_targets:
            canonical_target = canonical_target_name(str(raw_target))
            canonical_path = canonical_dir / f"{canonical_target}{NIFTI_SUFFIX}"
            if canonical_path.is_file():
                status = "SOURCE_AVAILABLE"
                source_names = canonical_target
                source_paths = str(canonical_path)
                source_hashes = sha256_file(canonical_path)
                needs_task2 = "false"
                reason = "canonical_source_mask_available_after_task1_rename"
            else:
                gap_sources = sorted(
                    source_name
                    for source_name in gap_sources_by_target.get(canonical_target, set())
                    if source_name in original_sources
                )
                if gap_sources:
                    status = "GRANULARITY_GAP"
                    source_names = ";".join(gap_sources)
                    source_paths = ";".join(str(original_sources[source_name]) for source_name in gap_sources)
                    source_hashes = ";".join(sha256_file(original_sources[source_name]) for source_name in gap_sources)
                    needs_task2 = "true"
                    reason = "source_label_exists_but_task1_boundary_marks_target_as_task2_generate"
                else:
                    status = "TARGET_MISSING"
                    source_names = ""
                    source_paths = ""
                    source_hashes = ""
                    needs_task2 = "true"
                    reason = "no_same_granularity_canonical_source_label_available_do_not_infer_absent_or_out_of_fov"
            status_counts[status] = status_counts.get(status, 0) + 1
            rows.append(
                {
                    "case_id": case_id,
                    "raw_target_name": str(raw_target),
                    "canonical_target": canonical_target,
                    "coverage_status": status,
                    "source_label_names": source_names,
                    "source_mask_path": source_paths,
                    "source_mask_sha256": source_hashes,
                    "needs_task2_generation": needs_task2,
                    "reason": reason,
                }
            )

    summary = {
        "status": "READY" if len(case_dirs) == expected_case_count else "BLOCKED",
        "stage": "task1_373_source_coverage_audit",
        "mode": "coverage-audit",
        "created_at": utc_now(),
        "workspace_root": str(workspace_root),
        "original_mask_root": str(paths["original"]),
        "canonical_mask_root": str(paths["canonical"]),
        "target_config": str(target_config),
        "authoritative_raw_target_count": len(raw_targets),
        "effective_canonical_target_count": len(target_set),
        "case_count": len(case_dirs),
        "expected_case_count": expected_case_count,
        "case_count_ok": len(case_dirs) == expected_case_count,
        "coverage_row_count": len(rows),
        "expected_coverage_row_count": len(case_dirs) * len(raw_targets),
        "status_counts": status_counts,
        "source_available_count": status_counts.get("SOURCE_AVAILABLE", 0),
        "granularity_gap_count": status_counts.get("GRANULARITY_GAP", 0),
        "target_missing_count": status_counts.get("TARGET_MISSING", 0),
        "needs_task2_generation_count": sum(1 for row in rows if row["needs_task2_generation"] == "true"),
        "absence_inference_policy": "source_mask_absence_is_not_ABSENT_or_OUT_OF_FOV",
        "writes_masks": False,
    }
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["manifests"] / "task1_373_source_coverage.csv", rows, COVERAGE_FIELDS)
    write_json(paths["manifests"] / "task1_373_source_coverage_summary.json", summary)
    return summary


def build_task1_canonical_masks(
    *,
    workspace_root: Path,
    mapping: Path = DEFAULT_MAPPING,
    non_rename_decisions: Path = DEFAULT_NON_RENAME,
    alias_groups: Path = DEFAULT_ALIAS_GROUPS,
    target_config: Path = DEFAULT_TARGET_CONFIG,
    case_manifest: Path | None = None,
    mode: str = "dry-run",
    expected_case_count: int = 103,
    allow_unmapped: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    if mode not in {"dry-run", "apply", "validate"}:
        raise DeliveryError(f"unsupported mode: {mode}")
    workspace_root = workspace_root.resolve()
    paths = _workspace_paths(workspace_root)
    target_set, raw_target_count = _canonical_target_contract(target_config)
    confirmed, pending, rejected, mapping_order = _mapping_indexes(mapping)
    non_rename = _load_non_rename(non_rename_decisions)
    aliases = read_alias_groups(alias_groups)
    case_dirs = _case_dirs(paths["original"], case_manifest)

    audit_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    for case_id, source_dir in case_dirs:
        dest_dir = paths["canonical"] / case_id / "segmentations"
        rows, collisions, unmapped = _plan_case(
            case_id=case_id,
            source_dir=source_dir,
            dest_dir=dest_dir,
            target_set=target_set,
            confirmed=confirmed,
            pending=pending,
            rejected=rejected,
            mapping_order=mapping_order,
            non_rename=non_rename,
            alias_groups=aliases,
        )
        audit_rows.extend(rows)
        collision_rows.extend(collisions)
        unmapped_rows.extend(unmapped)

    if mode == "apply":
        if len(case_dirs) != expected_case_count:
            raise DeliveryError(f"Task1 canonical apply requires {expected_case_count} cases, got {len(case_dirs)}")
        if collision_rows:
            raise DeliveryError(f"Task1 canonical apply blocked by {len(collision_rows)} collision(s)")
        blocking_unmapped = [row for row in unmapped_rows if row.get("action") == "UNMAPPED"]
        if blocking_unmapped and not allow_unmapped:
            raise DeliveryError(f"Task1 canonical apply blocked by {len(blocking_unmapped)} unmapped source mask(s)")
        _copy_planned_rows(audit_rows, resume=resume)
        for row in audit_rows:
            if row["destination_path"]:
                row["destination_checksum"] = sha256_file_if_exists(Path(str(row["destination_path"])))

    if mode == "validate":
        missing = [
            row for row in audit_rows
            if row["action"] in {"CANONICAL_SOURCE_SELECTED", "CONFIRMED_RENAME", "SELECTED_CONFIRMED_ALIAS"}
            and (not row["destination_path"] or row["source_checksum"] != sha256_file_if_exists(Path(str(row["destination_path"]))))
        ]
        if missing:
            raise DeliveryError(f"Task1 canonical validation failed for {len(missing)} copied mask(s)")
        for row in audit_rows:
            if row["destination_path"]:
                row["destination_checksum"] = sha256_file_if_exists(Path(str(row["destination_path"])))

    output_rows = _output_rows(audit_rows)
    nonblocking_exclusion_rows = [
        row for row in audit_rows
        if row.get("action") in {"EXCLUDED_NON_ANATOMY", "EXCLUDED_OUTSIDE_373", "OUTSIDE_373_OR_GRANULARITY_MISMATCH", "NONRENAME_TASK2", "PENDING_MAPPING"}
    ]
    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status_counts[str(row["action"])] = status_counts.get(str(row["action"]), 0) + 1
    blocking_unmapped = [row for row in unmapped_rows if row.get("action") == "UNMAPPED"]
    summary = {
        "status": "READY" if not collision_rows and (allow_unmapped or not blocking_unmapped) else "BLOCKED",
        "stage": "task1_373_canonical_masks",
        "mode": mode,
        "created_at": utc_now(),
        "workspace_root": str(workspace_root),
        "original_mask_root": str(paths["original"]),
        "canonical_mask_root": str(paths["canonical"]),
        "target_config": str(target_config),
        "authoritative_raw_target_count": raw_target_count,
        "effective_canonical_target_count": len(target_set),
        "case_count": len(case_dirs),
        "expected_case_count": expected_case_count,
        "case_count_ok": len(case_dirs) == expected_case_count,
        "audit_row_count": len(audit_rows),
        "status_counts": status_counts,
        "source_mask_count": len(audit_rows),
        "confirmed_rename_count": status_counts.get("CONFIRMED_RENAME", 0),
        "keep_canonical_count": status_counts.get("CANONICAL_SOURCE_SELECTED", 0),
        "canonical_source_selected_count": status_counts.get("CANONICAL_SOURCE_SELECTED", 0),
        "selected_alias_count": status_counts.get("SELECTED_CONFIRMED_ALIAS", 0),
        "redundant_confirmed_alias_count": status_counts.get("REDUNDANT_CONFIRMED_ALIAS", 0),
        "nonrename_task2_count": status_counts.get("NONRENAME_TASK2", 0),
        "pending_mapping_count": status_counts.get("PENDING_MAPPING", 0),
        "excluded_non_anatomy_count": status_counts.get("EXCLUDED_NON_ANATOMY", 0),
        "excluded_outside_373_count": status_counts.get("EXCLUDED_OUTSIDE_373", 0),
        "outside_373_or_granularity_mismatch_count": status_counts.get("OUTSIDE_373_OR_GRANULARITY_MISMATCH", 0),
        "unmapped_count": status_counts.get("UNMAPPED", 0),
        "unapproved_collision_count": len(collision_rows),
        "collision_count": len(collision_rows),
        "mapping_artifacts": {
            "organ_rename_mapping_373": str(mapping),
            "non_rename_decisions": str(non_rename_decisions),
            "task1_alias_groups": str(alias_groups),
            "task_boundary_classification": str(DEFAULT_BOUNDARY),
        },
    }
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    write_csv(paths["manifests"] / "task1_373_rename_audit.csv", audit_rows, AUDIT_FIELDS)
    write_csv(paths["manifests"] / "task1_373_canonical_outputs.csv", output_rows, OUTPUT_FIELDS)
    write_csv(paths["manifests"] / "task1_373_excluded_source_labels.csv", nonblocking_exclusion_rows, AUDIT_FIELDS)
    write_json(paths["manifests"] / "task1_373_rename_summary.json", summary)
    write_csv(paths["manifests"] / "task1_373_collision_report.csv", collision_rows, COLLISION_FIELDS)
    write_csv(paths["manifests"] / "task1_373_unmapped.csv", unmapped_rows, UNMAPPED_FIELDS)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Task1 373-canonical source-mask tree for Formal Round1.")
    parser.add_argument("--workspace-root", default=DEFAULT_WORKSPACE, type=Path)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING, type=Path)
    parser.add_argument("--non-rename-decisions", default=DEFAULT_NON_RENAME, type=Path)
    parser.add_argument("--alias-groups", default=DEFAULT_ALIAS_GROUPS, type=Path)
    parser.add_argument("--target-config", default=DEFAULT_TARGET_CONFIG, type=Path)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--expected-case-count", default=103, type=int)
    parser.add_argument("--allow-unmapped", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true", default=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--coverage-audit", action="store_true")
    args = parser.parse_args()
    try:
        if args.coverage_audit:
            result = audit_task1_source_coverage(
                workspace_root=args.workspace_root,
                non_rename_decisions=args.non_rename_decisions,
                target_config=args.target_config,
                case_manifest=args.case_manifest,
                expected_case_count=args.expected_case_count,
            )
        else:
            selected_mode = "apply" if args.apply else ("validate" if args.validate else "dry-run")
            result = build_task1_canonical_masks(
                workspace_root=args.workspace_root,
                mapping=args.mapping,
                non_rename_decisions=args.non_rename_decisions,
                alias_groups=args.alias_groups,
                target_config=args.target_config,
                case_manifest=args.case_manifest,
                mode=selected_mode,
                expected_case_count=args.expected_case_count,
                allow_unmapped=args.allow_unmapped,
                resume=not args.no_resume,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "READY" else 2
    except Exception as exc:
        failure = {"status": "failed", "error": str(exc)}
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
