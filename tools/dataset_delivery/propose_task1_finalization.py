#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import (  # noqa: E402
    DeliveryError,
    NIFTI_SUFFIX,
    load_task2_targets,
    load_taxonomy_names,
    normalize_name,
    read_alias_groups,
    read_boundary_classification,
    read_csv_fieldnames,
    read_csv_rows,
    read_rename_mapping,
    safe_label_name,
    sha256_file,
    write_csv,
    write_json,
)


SEMANTIC_FIELDS = [
    "concept_group",
    "label_a",
    "label_b",
    "relationship",
    "same_anatomy",
    "same_laterality",
    "same_granularity",
    "evidence_type",
    "evidence_source",
    "evidence_reference",
    "decision",
    "notes",
]
ALLOWED_RELATIONSHIPS = {
    "exact_synonym",
    "token_order_equivalent",
    "parent_child",
    "aggregate_vs_sided",
    "whole_vs_substructure",
    "overlapping_not_equivalent",
    "ambiguous",
    "distinct",
}
ALLOWED_SEMANTIC_DECISIONS = {"confirmed_equivalent", "confirmed_not_equivalent", "pending_review"}
SEMANTIC_DUPLICATE_RELATIONSHIPS = {"exact_synonym", "token_order_equivalent"}
VOXEL_ROW_FIELDS = [
    "case_id",
    "alias_group",
    "target_name",
    "source_a",
    "source_b",
    "path_a",
    "path_b",
    "status",
    "reason",
    "target_already_exists",
    "shape_a",
    "shape_b",
    "spacing_a",
    "spacing_b",
    "orientation_a",
    "orientation_b",
    "affine_max_abs_diff",
    "foreground_voxels_a",
    "foreground_voxels_b",
    "empty_a",
    "empty_b",
    "exact_equal",
    "xor_voxel_count",
    "dice",
    "intersection",
    "union",
]


def _case_mask_dir(data_root: Path, row: dict[str, str]) -> Path:
    ref = row.get("reference_mask_dir") or row.get("mask_dir") or ""
    if ref:
        path = Path(ref)
        return path if path.is_absolute() else data_root / path
    return data_root / row["case_id"] / "segmentations"


def _json_hash(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_canonical(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        })
    return {"hash": _json_hash(rows), "files": rows}


def fingerprint_source_inventory(data_root: Path, case_manifest: Path) -> dict[str, Any]:
    rows = []
    seen: set[Path] = set()
    for case in read_csv_rows(case_manifest):
        mask_dir = _case_mask_dir(data_root, case)
        if not mask_dir.exists():
            continue
        for path in sorted(mask_dir.glob(f"*{NIFTI_SUFFIX}")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = path.stat()
            try:
                rel = path.resolve().relative_to(data_root.resolve())
            except Exception:
                rel = path
            rows.append({"path": str(rel), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "hash": _json_hash(rows),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "files": rows,
    }


def read_semantic_evidence(path: Path) -> list[dict[str, str]]:
    fieldnames = read_csv_fieldnames(path)
    missing = [field for field in SEMANTIC_FIELDS if field not in fieldnames]
    if missing:
        raise DeliveryError(f"Semantic evidence schema missing fields: {missing}")
    rows = []
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(read_csv_rows(path), start=2):
        clean = {field: row.get(field, "").strip() for field in SEMANTIC_FIELDS}
        clean["label_a"] = safe_label_name(clean["label_a"])
        clean["label_b"] = safe_label_name(clean["label_b"])
        if clean["relationship"] not in ALLOWED_RELATIONSHIPS:
            raise DeliveryError(f"Invalid semantic relationship at row {idx}: {clean['relationship']!r}")
        if clean["decision"] not in ALLOWED_SEMANTIC_DECISIONS:
            raise DeliveryError(f"Invalid semantic decision at row {idx}: {clean['decision']!r}")
        key = tuple(sorted((clean["label_a"], clean["label_b"])))
        if key in seen:
            raise DeliveryError(f"Duplicate semantic evidence pair at row {idx}: {key}")
        seen.add(key)
        rows.append(clean)
    return rows


def _same_side(a: str, b: str) -> bool:
    def side(value: str) -> str:
        parts = set(value.split("_"))
        if "left" in parts and "right" not in parts:
            return "left"
        if "right" in parts and "left" not in parts:
            return "right"
        return ""

    return side(a) == side(b)


def generated_taxonomy_candidates(taxonomy_names: list[str], evidence_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {tuple(sorted((row["label_a"], row["label_b"]))) for row in evidence_rows}
    candidates: list[dict[str, str]] = []
    for left, right in itertools.combinations(sorted(taxonomy_names), 2):
        if tuple(sorted((left, right))) in seen:
            continue
        if sorted(left.split("_")) != sorted(right.split("_")):
            continue
        if not _same_side(left, right):
            continue
        candidates.append({
            "concept_group": f"candidate_{left}__{right}",
            "label_a": left,
            "label_b": right,
            "relationship": "token_order_equivalent",
            "same_anatomy": "candidate",
            "same_laterality": "candidate",
            "same_granularity": "candidate",
            "evidence_type": "generated_candidate",
            "evidence_source": "normalized_token_order",
            "evidence_reference": "Generated by exact token multiset match; not auto-confirmed",
            "decision": "pending_review",
            "notes": "Generated candidate only; semantic equivalence requires review evidence",
        })
    return candidates


def build_taxonomy_semantic_audit(
    *,
    taxonomy: Path,
    mapping: Path,
    boundary_classification: Path,
    alias_groups: Path,
    task2_targets: Path,
    semantic_evidence: Path,
    output_root: Path,
) -> dict[str, Any]:
    taxonomy_names = load_taxonomy_names(taxonomy)
    taxonomy_set = set(taxonomy_names)
    evidence_rows = read_semantic_evidence(semantic_evidence)
    generated_rows = generated_taxonomy_candidates(taxonomy_names, evidence_rows)
    decision_rows = []
    for source, rows in (("configured_evidence", evidence_rows), ("generated_candidate", generated_rows)):
        for row in rows:
            enriched = dict(row)
            enriched["candidate_source"] = source
            enriched["label_a_in_taxonomy"] = str(row["label_a"] in taxonomy_set).lower()
            enriched["label_b_in_taxonomy"] = str(row["label_b"] in taxonomy_set).lower()
            enriched["task2_overlap"] = str(bool({row["label_a"], row["label_b"]} & load_task2_targets(task2_targets))).lower()
            decision_rows.append(enriched)

    relationship_counts = Counter(row["relationship"] for row in decision_rows)
    decision_counts = Counter(row["decision"] for row in decision_rows)
    groups: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        group = groups.setdefault(row["concept_group"], {"labels": set(), "relationships": set(), "decisions": set(), "rows": 0})
        group["labels"].update([row["label_a"], row["label_b"]])
        group["relationships"].add(row["relationship"])
        group["decisions"].add(row["decision"])
        group["rows"] += 1
    concept_rows = []
    confirmed_reduction = 0
    for group_name, group in sorted(groups.items()):
        labels = sorted(group["labels"])
        decisions = sorted(group["decisions"])
        if decisions == ["confirmed_equivalent"]:
            confirmed_reduction += max(0, len(labels) - 1)
        concept_rows.append({
            "concept_group": group_name,
            "labels": ";".join(labels),
            "label_count": len(labels),
            "relationships": ";".join(sorted(group["relationships"])),
            "decisions": ";".join(decisions),
            "row_count": group["rows"],
        })
    pending = decision_counts.get("pending_review", 0)
    summary = {
        "status": "success",
        "read_only": True,
        "label_entry_count": len(taxonomy_names),
        "mapping": str(mapping),
        "boundary_classification": str(boundary_classification),
        "alias_groups": str(alias_groups),
        "task2_targets": str(task2_targets),
        "semantic_evidence": str(semantic_evidence),
        "semantic_duplicate_candidate_count": sum(
            1 for row in decision_rows if row["relationship"] in SEMANTIC_DUPLICATE_RELATIONSHIPS or row["decision"] == "pending_review"
        ),
        "confirmed_exact_synonym_count": sum(
            1 for row in decision_rows
            if row["relationship"] == "exact_synonym" and row["decision"] == "confirmed_equivalent"
        ),
        "token_order_duplicate_count": sum(
            1 for row in decision_rows
            if row["relationship"] == "token_order_equivalent" and row["decision"] == "confirmed_equivalent"
        ),
        "parent_child_count": relationship_counts.get("parent_child", 0),
        "aggregate_vs_sided_count": relationship_counts.get("aggregate_vs_sided", 0),
        "whole_vs_substructure_count": relationship_counts.get("whole_vs_substructure", 0),
        "pending_count": pending,
        "confirmed_semantic_duplicate_groups": sum(
            1 for row in concept_rows if row["decisions"] == "confirmed_equivalent"
        ),
        "pending_semantic_duplicate_groups": sum(
            1 for row in concept_rows if "pending_review" in row["decisions"].split(";")
        ),
        "confirmed_distinct_relationships": decision_counts.get("confirmed_not_equivalent", 0),
        "unresolved_relationships": pending,
        "medical_concept_count": "not_determined" if pending else len(taxonomy_names) - confirmed_reduction,
        "medical_concept_count_lower_bound": len(taxonomy_names) - confirmed_reduction - pending,
        "medical_concept_count_upper_bound": len(taxonomy_names) - confirmed_reduction,
        "relationship_counts": dict(relationship_counts),
        "decision_counts": dict(decision_counts),
    }
    fields = SEMANTIC_FIELDS + ["candidate_source", "label_a_in_taxonomy", "label_b_in_taxonomy", "task2_overlap"]
    write_csv(output_root / "taxonomy_semantic_pair_candidates.csv", decision_rows, fields)
    write_csv(output_root / "taxonomy_semantic_decisions.csv", decision_rows, fields)
    write_csv(output_root / "taxonomy_concept_groups.csv", concept_rows, ["concept_group", "labels", "label_count", "relationships", "decisions", "row_count"])
    write_json(output_root / "taxonomy_semantic_audit.json", summary)
    write_taxonomy_semantic_md(output_root / "taxonomy_semantic_audit.md", summary)
    return {"summary": summary, "decision_rows": decision_rows, "concept_rows": concept_rows}


def write_taxonomy_semantic_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Taxonomy Semantic Audit",
        "",
        f"- label entry count: `{summary['label_entry_count']}`",
        f"- semantic duplicate candidates: `{summary['semantic_duplicate_candidate_count']}`",
        f"- confirmed exact synonyms: `{summary['confirmed_exact_synonym_count']}`",
        f"- confirmed token-order duplicates: `{summary['token_order_duplicate_count']}`",
        f"- parent-child: `{summary['parent_child_count']}`",
        f"- aggregate-vs-sided: `{summary['aggregate_vs_sided_count']}`",
        f"- pending: `{summary['pending_count']}`",
        f"- medical concept count: `{summary['medical_concept_count']}`",
        f"- medical concept lower bound: `{summary['medical_concept_count_lower_bound']}`",
        f"- medical concept upper bound: `{summary['medical_concept_count_upper_bound']}`",
        "",
        "Pending rows intentionally prevent reporting a single exact medical concept count.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_binary_mask(path: Path) -> dict[str, Any]:
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise DeliveryError("nibabel and numpy are required for voxel proposal audit") from exc
    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj) > 0
        spacing = tuple(float(x) for x in img.header.get_zooms()[: len(img.shape)])
        orientation = tuple(str(x) for x in nib.aff2axcodes(img.affine))
        return {
            "path": path,
            "ok": True,
            "shape": tuple(int(x) for x in img.shape),
            "spacing": spacing,
            "orientation": orientation,
            "affine": np.asarray(img.affine, dtype=float),
            "data": arr,
            "foreground_voxels": int(arr.sum()),
            "error": "",
        }
    except Exception as exc:
        return {
            "path": path,
            "ok": False,
            "shape": None,
            "spacing": None,
            "orientation": None,
            "affine": None,
            "data": None,
            "foreground_voxels": None,
            "error": str(exc),
        }


def _fmt_tuple(value: Any) -> str:
    if value is None:
        return ""
    return "x".join(str(x) for x in value)


def _fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.12g}"


def compare_binary_masks(mask_a: Path, mask_b: Path, *, affine_atol: float = 1e-3) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise DeliveryError("numpy is required for voxel proposal audit") from exc
    a = _load_binary_mask(mask_a)
    b = _load_binary_mask(mask_b)
    row: dict[str, Any] = {
        "shape_a": _fmt_tuple(a["shape"]),
        "shape_b": _fmt_tuple(b["shape"]),
        "spacing_a": _fmt_tuple(a["spacing"]),
        "spacing_b": _fmt_tuple(b["spacing"]),
        "orientation_a": _fmt_tuple(a["orientation"]),
        "orientation_b": _fmt_tuple(b["orientation"]),
        "foreground_voxels_a": a["foreground_voxels"] if a["foreground_voxels"] is not None else "",
        "foreground_voxels_b": b["foreground_voxels"] if b["foreground_voxels"] is not None else "",
        "empty_a": "",
        "empty_b": "",
        "exact_equal": "false",
        "xor_voxel_count": "",
        "dice": "",
        "intersection": "",
        "union": "",
        "affine_max_abs_diff": "",
    }
    if not a["ok"] or not b["ok"]:
        row.update({"status": "read_error", "reason": "; ".join(x for x in [a["error"], b["error"]] if x)})
        return row
    if a["shape"] != b["shape"]:
        row.update({"status": "geometry_mismatch", "reason": "shape_mismatch"})
        return row
    if a["spacing"] != b["spacing"]:
        row.update({"status": "geometry_mismatch", "reason": "spacing_mismatch"})
        return row
    if a["orientation"] != b["orientation"]:
        row.update({"status": "geometry_mismatch", "reason": "orientation_mismatch"})
        return row
    affine_diff = float(np.max(np.abs(a["affine"] - b["affine"])))
    row["affine_max_abs_diff"] = _fmt_float(affine_diff)
    if not bool(np.allclose(a["affine"], b["affine"], atol=affine_atol)):
        row.update({"status": "geometry_mismatch", "reason": "affine_mismatch"})
        return row
    arr_a = a["data"]
    arr_b = b["data"]
    fg_a = int(a["foreground_voxels"])
    fg_b = int(b["foreground_voxels"])
    row["empty_a"] = str(fg_a == 0).lower()
    row["empty_b"] = str(fg_b == 0).lower()
    xor_count = int(np.logical_xor(arr_a, arr_b).sum())
    intersection = int(np.logical_and(arr_a, arr_b).sum())
    union = int(np.logical_or(arr_a, arr_b).sum())
    dice = 1.0 if fg_a + fg_b == 0 else float(2 * intersection / (fg_a + fg_b))
    exact = bool(xor_count == 0 and arr_a.shape == arr_b.shape)
    row.update({
        "exact_equal": str(exact).lower(),
        "xor_voxel_count": xor_count,
        "dice": _fmt_float(dice),
        "intersection": intersection,
        "union": union,
    })
    if fg_a == 0 or fg_b == 0:
        row.update({"status": "empty_mask", "reason": "one_or_both_masks_empty"})
    elif exact and xor_count == 0 and dice == 1.0:
        row.update({"status": "exact_equal", "reason": "binary_masks_exact_equal"})
    else:
        row.update({"status": "voxel_mismatch", "reason": "binary_masks_not_exact_equal"})
    return row


def alias_semantic_decision(
    *,
    target: str,
    sources: list[str],
    mapping_rows: Any,
    boundary_rows: dict[tuple[str, str], dict[str, str]],
    task2_targets: set[str],
) -> tuple[str, str]:
    if target in task2_targets:
        return "confirmed_not_equivalent", "target_is_fixed_task2_generate"
    mapping_status = {(row.source_name, row.target_name): row.status for row in mapping_rows}
    for source in sources:
        if mapping_status.get((source, target)) != "confirmed":
            return "pending_review", f"mapping_not_confirmed:{source}->{target}"
        boundary = boundary_rows.get((source, target), {})
        if boundary.get("classification") != "task1_rename" or boundary.get("status") != "confirmed":
            return "pending_review", f"boundary_not_confirmed:{source}->{target}"
        required = (
            boundary.get("same_anatomy") == "true",
            boundary.get("same_laterality") == "true",
            boundary.get("same_granularity") == "true",
            boundary.get("requires_voxel_change") == "false",
        )
        if not all(required):
            return "pending_review", f"boundary_conditions_not_exact:{source}->{target}"
    return "confirmed_equivalent", "canonical_boundary_rows_confirm_same_anatomy_laterality_granularity"


def _priority_order(target: str, alias: dict[str, Any]) -> list[str]:
    rows = {safe_label_name(row.get("source_name", "")): row for row in alias.get("rows", [])}

    def rank(source: str) -> tuple[int, int, float, str]:
        row = rows.get(source, {})
        if row.get("preferred_source", "").strip().lower() in {"1", "true", "yes", "y"}:
            preferred = 0
        else:
            preferred = 1
        try:
            explicit = int(row.get("priority", "9999") or "9999")
        except ValueError:
            explicit = 9999
        lexical_distance = 1.0 - difflib.SequenceMatcher(None, source, target).ratio()
        return preferred, explicit, lexical_distance, source

    return sorted(alias["sources"], key=rank)


def audit_alias_voxel_equivalence(
    *,
    data_root: Path,
    case_manifest: Path,
    alias_groups: Path,
    mapping: Path,
    boundary_classification: Path,
    task2_targets: Path,
    output_root: Path,
    affine_atol: float = 1e-3,
) -> dict[str, Any]:
    aliases = read_alias_groups(alias_groups)
    mapping_rows = read_rename_mapping(mapping)
    boundary_rows = read_boundary_classification(boundary_classification)
    task2 = load_task2_targets(task2_targets)
    cases = read_csv_rows(case_manifest)
    pair_rows: list[dict[str, Any]] = []
    blocked_groups: list[dict[str, Any]] = []
    blocked_cases: list[dict[str, Any]] = []
    priority_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for target, alias in sorted(aliases.items()):
        sources = _priority_order(target, alias)
        alias_group = str(alias["alias_group"])
        semantic_decision, semantic_reason = alias_semantic_decision(
            target=target,
            sources=sources,
            mapping_rows=mapping_rows,
            boundary_rows=boundary_rows,
            task2_targets=task2,
        )
        exact_case_ids: list[str] = []
        mismatch_case_ids: list[str] = []
        error_case_ids: list[str] = []
        single_source_case_ids: list[str] = []
        missing_case_ids: list[str] = []
        target_exists_case_ids: list[str] = []
        multi_source_case_count = 0
        pair_count = 0

        for case in cases:
            case_id = case.get("case_id", "")
            mask_dir = _case_mask_dir(data_root, case)
            present = [source for source in sources if (mask_dir / f"{source}{NIFTI_SUFFIX}").exists()]
            target_exists = (mask_dir / f"{target}{NIFTI_SUFFIX}").exists()
            if target_exists:
                target_exists_case_ids.append(case_id)
                row = _base_pair_row(case_id, alias_group, target, "", "", "", "", target_exists)
                row.update({"status": "target_already_exists", "reason": "canonical_target_mask_exists"})
                pair_rows.append(row)
                status_counts[row["status"]] += 1
            if not present:
                missing_case_ids.append(case_id)
                row = _base_pair_row(case_id, alias_group, target, "", "", "", "", target_exists)
                row.update({"status": "missing_source", "reason": "no_alias_source_present"})
                pair_rows.append(row)
                status_counts[row["status"]] += 1
                continue
            if len(present) == 1:
                single_source_case_ids.append(case_id)
                source = present[0]
                row = _base_pair_row(case_id, alias_group, target, source, "", mask_dir / f"{source}{NIFTI_SUFFIX}", "", target_exists)
                row.update({"status": "single_source_only", "reason": "only_one_alias_source_present"})
                pair_rows.append(row)
                status_counts[row["status"]] += 1
                continue

            multi_source_case_count += 1
            case_statuses: list[str] = []
            for source_a, source_b in itertools.combinations(present, 2):
                path_a = mask_dir / f"{source_a}{NIFTI_SUFFIX}"
                path_b = mask_dir / f"{source_b}{NIFTI_SUFFIX}"
                row = _base_pair_row(case_id, alias_group, target, source_a, source_b, path_a, path_b, target_exists)
                row.update(compare_binary_masks(path_a, path_b, affine_atol=affine_atol))
                pair_rows.append(row)
                status_counts[row["status"]] += 1
                case_statuses.append(row["status"])
                pair_count += 1
            if case_statuses and all(status == "exact_equal" for status in case_statuses):
                exact_case_ids.append(case_id)
            elif any(status == "read_error" for status in case_statuses):
                error_case_ids.append(case_id)
            else:
                mismatch_case_ids.append(case_id)

        if semantic_decision != "confirmed_equivalent":
            group_decision = "pending_semantic_review"
        elif error_case_ids:
            group_decision = "blocked_alias_error"
        elif mismatch_case_ids:
            group_decision = "blocked_alias_mismatch"
        elif multi_source_case_count == 0:
            group_decision = "insufficient_overlap"
        else:
            group_decision = "source_priority_global"

        preferred = sources[0] if sources else ""
        priority_order = ";".join(sources)
        if group_decision == "source_priority_global":
            priority_rows.append({
                "alias_group": alias_group,
                "target_name": target,
                "case_id": "*",
                "preferred_source": preferred,
                "source_priority": priority_order,
                "decision": "use_global_source_priority",
                "reason": "all_multi_source_cases_semantic_and_voxel_exact",
                "evidence_case_count": multi_source_case_count,
                "blocked_case_count": 0,
            })
        elif semantic_decision == "confirmed_equivalent" and exact_case_ids:
            for case_id in exact_case_ids:
                priority_rows.append({
                    "alias_group": alias_group,
                    "target_name": target,
                    "case_id": case_id,
                    "preferred_source": preferred,
                    "source_priority": priority_order,
                    "decision": "use_case_source_priority",
                    "reason": "case_semantic_and_voxel_exact_but_group_has_blocked_cases",
                    "evidence_case_count": 1,
                    "blocked_case_count": len(set(mismatch_case_ids + error_case_ids)),
                })

        for case_id in sorted(set(mismatch_case_ids)):
            blocked_cases.append({
                "case_id": case_id,
                "alias_group": alias_group,
                "target_name": target,
                "decision": "blocked_alias_mismatch",
                "reason": "one_or_more_pairwise_alias_masks_not_exact_equal",
            })
        for case_id in sorted(set(error_case_ids)):
            blocked_cases.append({
                "case_id": case_id,
                "alias_group": alias_group,
                "target_name": target,
                "decision": "blocked_alias_error",
                "reason": "one_or_more_pairwise_alias_masks_unreadable",
            })
        if group_decision.startswith("blocked") or group_decision == "pending_semantic_review":
            blocked_groups.append({
                "alias_group": alias_group,
                "target_name": target,
                "decision": group_decision if group_decision.startswith("blocked") else "pending_semantic_review",
                "semantic_decision": semantic_decision,
                "semantic_reason": semantic_reason,
                "multi_source_case_count": multi_source_case_count,
                "exact_case_count": len(exact_case_ids),
                "mismatch_case_count": len(set(mismatch_case_ids)),
                "error_case_count": len(set(error_case_ids)),
            })
        group_summaries.append({
            "alias_group": alias_group,
            "target_name": target,
            "sources": sources,
            "semantic_decision": semantic_decision,
            "semantic_reason": semantic_reason,
            "decision": group_decision,
            "multi_source_case_count": multi_source_case_count,
            "pairwise_comparison_count": pair_count,
            "exact_case_count": len(exact_case_ids),
            "mismatch_case_count": len(set(mismatch_case_ids)),
            "error_case_count": len(set(error_case_ids)),
            "single_source_only_case_count": len(single_source_case_ids),
            "missing_source_case_count": len(missing_case_ids),
            "target_already_exists_case_count": len(target_exists_case_ids),
        })

    summary = {
        "status": "success",
        "alias_group_count": len(aliases),
        "case_count": len(cases),
        "row_count": len(pair_rows),
        "status_counts": dict(status_counts),
        "group_summaries": group_summaries,
        "global_priority_alias_groups": [
            row["alias_group"] for row in priority_rows
            if row["case_id"] == "*"
        ],
        "case_level_priority_alias_groups": sorted({
            row["alias_group"] for row in priority_rows
            if row["case_id"] != "*"
        }),
        "blocked_alias_groups": sorted({row["alias_group"] for row in blocked_groups if str(row["decision"]).startswith("blocked")}),
        "pending_alias_groups": sorted({row["alias_group"] for row in blocked_groups if row["decision"] == "pending_semantic_review"}),
    }
    write_csv(output_root / "alias_voxel_equivalence_rows.csv", pair_rows, VOXEL_ROW_FIELDS)
    write_csv(
        output_root / "blocked_alias_groups.csv",
        blocked_groups,
        ["alias_group", "target_name", "decision", "semantic_decision", "semantic_reason", "multi_source_case_count", "exact_case_count", "mismatch_case_count", "error_case_count"],
    )
    write_csv(output_root / "blocked_cases.csv", blocked_cases, ["case_id", "alias_group", "target_name", "decision", "reason"])
    write_csv(
        output_root / "proposed" / "proposed_source_priority.csv",
        priority_rows,
        ["alias_group", "target_name", "case_id", "preferred_source", "source_priority", "decision", "reason", "evidence_case_count", "blocked_case_count"],
    )
    write_json(output_root / "alias_voxel_equivalence_summary.json", summary)
    write_alias_voxel_md(output_root / "alias_voxel_equivalence_report.md", summary)
    return {
        "summary": summary,
        "pair_rows": pair_rows,
        "blocked_groups": blocked_groups,
        "blocked_cases": blocked_cases,
        "priority_rows": priority_rows,
    }


def _base_pair_row(
    case_id: str,
    alias_group: str,
    target: str,
    source_a: str,
    source_b: str,
    path_a: Path | str,
    path_b: Path | str,
    target_exists: bool,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "alias_group": alias_group,
        "target_name": target,
        "source_a": source_a,
        "source_b": source_b,
        "path_a": str(path_a) if path_a else "",
        "path_b": str(path_b) if path_b else "",
        "status": "",
        "reason": "",
        "target_already_exists": str(target_exists).lower(),
        "shape_a": "",
        "shape_b": "",
        "spacing_a": "",
        "spacing_b": "",
        "orientation_a": "",
        "orientation_b": "",
        "affine_max_abs_diff": "",
        "foreground_voxels_a": "",
        "foreground_voxels_b": "",
        "empty_a": "",
        "empty_b": "",
        "exact_equal": "",
        "xor_voxel_count": "",
        "dice": "",
        "intersection": "",
        "union": "",
    }


def write_alias_voxel_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 1 Alias Voxel Equivalence Audit",
        "",
        f"- alias groups: `{summary['alias_group_count']}`",
        f"- cases: `{summary['case_count']}`",
        f"- rows: `{summary['row_count']}`",
        "",
        "## Status Counts",
    ]
    for key, value in sorted(summary["status_counts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Group Decisions"])
    for row in summary["group_summaries"]:
        lines.append(
            f"- `{row['alias_group']}`: `{row['decision']}`; "
            f"exact={row['exact_case_count']} mismatch={row['mismatch_case_count']} error={row['error_case_count']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_coronary_evidence(repo_root: Path, *, max_matches: int = 80) -> list[dict[str, Any]]:
    terms = ["coronary_arteries", "coronary_artery", "coronary arteries", "coronary artery"]
    suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    roots = [repo_root / name for name in ("configs", "tools", "scripts", "tests", "docs")]
    for readme in repo_root.glob("README*"):
        roots.append(readme)
    matches: list[dict[str, Any]] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []
        for path in paths:
            if path.suffix.lower() not in suffixes or path.stat().st_size > 1_000_000:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line_no, line in enumerate(lines, start=1):
                normalized = line.lower()
                if any(term in normalized for term in terms):
                    matches.append({
                        "path": str(path.relative_to(repo_root)),
                        "line": line_no,
                        "text": line.strip()[:240],
                    })
                    if len(matches) >= max_matches:
                        return matches
    return matches


def coronary_proposed_decision(semantic_rows: list[dict[str, str]], repo_root: Path) -> dict[str, Any]:
    pair = {"coronary_artery", "coronary_arteries"}
    configured = [
        row for row in semantic_rows
        if {row["label_a"], row["label_b"]} == pair
    ]
    matches = collect_coronary_evidence(repo_root)
    confirmed = any(
        row["decision"] == "confirmed_equivalent"
        and row["same_anatomy"] == "true"
        and row["same_laterality"] == "true"
        and row["same_granularity"] == "true"
        and row["evidence_type"] in {"source_data_dictionary", "taxonomy_definition", "manual_data_dictionary_review"}
        for row in configured
    )
    if confirmed:
        return {
            "classification": "task1_rename",
            "status": "confirmed",
            "reason": "source_dictionary_confirms_singular_plural_scope",
            "evidence": configured,
            "repo_matches": matches,
        }
    return {
        "classification": "boundary_review",
        "status": "pending_review",
        "reason": "insufficient_evidence_singular_plural_scope",
        "evidence": configured,
        "repo_matches": matches,
    }


def write_proposed_configs(
    *,
    mapping: Path,
    boundary_classification: Path,
    alias_groups: Path,
    semantic_decision_rows: list[dict[str, Any]],
    coronary_decision: dict[str, Any],
    priority_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    proposed = output_root / "proposed"
    proposed.mkdir(parents=True, exist_ok=True)
    mapping_rows = read_csv_rows(mapping)
    for row in mapping_rows:
        if safe_label_name(row.get("source_name", "")) == "coronary_arteries" and safe_label_name(row.get("target_name", "")) == "coronary_artery":
            row["status"] = coronary_decision["status"]
            row["reason"] = coronary_decision["reason"]
            row["notes"] = "proposal_only; requires data dictionary proof before confirmed Task 1 rename"
    write_csv(proposed / "proposed_organ_rename_mapping_373.csv", mapping_rows, read_csv_fieldnames(mapping))

    boundary_rows = read_csv_rows(boundary_classification)
    for row in boundary_rows:
        if safe_label_name(row.get("source_name", "")) == "coronary_arteries" and safe_label_name(row.get("target_name", "")) == "coronary_artery":
            row.update({
                "same_anatomy": "unknown",
                "same_laterality": "unknown",
                "same_granularity": "unknown",
                "requires_voxel_change": "false",
                "classification": coronary_decision["classification"],
                "status": coronary_decision["status"],
                "reason": coronary_decision["reason"],
                "evidence": "repo search did not prove both labels denote the whole coronary arterial tree",
            })
    write_csv(proposed / "proposed_task_boundary_classification.csv", boundary_rows, read_csv_fieldnames(boundary_classification))

    priority_by_group = {row["alias_group"]: row for row in priority_rows if row["case_id"] == "*"}
    alias_rows = read_csv_rows(alias_groups)
    alias_fields = read_csv_fieldnames(alias_groups)
    proposed_alias_fields = alias_fields + ["proposed_priority_rank", "proposed_priority_scope", "proposal_decision"]
    for row in alias_rows:
        group = row.get("alias_group", "")
        source = safe_label_name(row.get("source_name", ""))
        priority = priority_by_group.get(group)
        if priority:
            order = str(priority["source_priority"]).split(";")
            row["proposed_priority_rank"] = str(order.index(source) + 1) if source in order else ""
            row["proposed_priority_scope"] = "*"
            row["proposal_decision"] = priority["decision"]
        else:
            row["proposed_priority_rank"] = ""
            row["proposed_priority_scope"] = ""
            row["proposal_decision"] = "no_global_priority"
    write_csv(proposed / "proposed_task1_alias_groups.csv", alias_rows, proposed_alias_fields)

    semantic_fields = SEMANTIC_FIELDS + ["candidate_source", "label_a_in_taxonomy", "label_b_in_taxonomy", "task2_overlap"]
    write_csv(proposed / "proposed_taxonomy_semantic_evidence.csv", semantic_decision_rows, semantic_fields)


def write_pending_semantic_review(
    *,
    output_root: Path,
    semantic_decision_rows: list[dict[str, Any]],
    coronary_decision: dict[str, Any],
    alias_result: dict[str, Any],
) -> None:
    rows = [
        {
            "scope": "taxonomy_semantic",
            "concept_group": row["concept_group"],
            "label_a": row["label_a"],
            "label_b": row["label_b"],
            "relationship": row["relationship"],
            "status": row["decision"],
            "reason": row["notes"],
        }
        for row in semantic_decision_rows
        if row["decision"] == "pending_review"
    ]
    if coronary_decision["status"] == "pending_review":
        rows.append({
            "scope": "mapping_boundary",
            "concept_group": "semantic_coronary_singular_plural",
            "label_a": "coronary_artery",
            "label_b": "coronary_arteries",
            "relationship": "ambiguous",
            "status": "pending_review",
            "reason": coronary_decision["reason"],
        })
    for row in alias_result["blocked_groups"]:
        if row["decision"] == "pending_semantic_review":
            rows.append({
                "scope": "alias_group",
                "concept_group": row["alias_group"],
                "label_a": row["target_name"],
                "label_b": "",
                "relationship": "ambiguous",
                "status": "pending_review",
                "reason": row["semantic_reason"],
            })
    write_csv(output_root / "pending_semantic_review.csv", rows, ["scope", "concept_group", "label_a", "label_b", "relationship", "status", "reason"])


def write_proposal_summary_md(path: Path, summary: dict[str, Any]) -> None:
    semantic = summary["taxonomy_semantic_audit"]
    alias = summary["alias_voxel_equivalence_summary"]
    focus = summary["focus_alias_summary"]
    lines = [
        "# Task 1 Semantic and Voxel Proposal Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- read_only: `{summary['read_only']}`",
        f"- apply: `{summary['apply']}`",
        f"- canonical_config_mutation: `{summary['safety']['canonical_config_mutation']}`",
        f"- source_data_mutation: `{summary['safety']['source_data_mutation']}`",
        "",
        "## Taxonomy Semantic Audit",
        f"- label entries: `{semantic['label_entry_count']}`",
        f"- semantic duplicate candidates: `{semantic['semantic_duplicate_candidate_count']}`",
        f"- confirmed exact synonyms: `{semantic['confirmed_exact_synonym_count']}`",
        f"- confirmed token-order duplicates: `{semantic['token_order_duplicate_count']}`",
        f"- parent-child: `{semantic['parent_child_count']}`",
        f"- aggregate-vs-sided: `{semantic['aggregate_vs_sided_count']}`",
        f"- pending: `{semantic['pending_count']}`",
        f"- medical concept count: `{semantic['medical_concept_count']}`",
        "",
        "## Focus Alias Findings",
        f"- celiac coexisting cases: `{focus['celiac']['coexisting_case_count']}`",
        f"- celiac exact/mismatch/error: `{focus['celiac']['exact_equal_case_count']}`/`{focus['celiac']['mismatch_case_count']}`/`{focus['celiac']['error_case_count']}`",
        f"- lung-lobe target-case groups: `{focus['lung_lobe']['target_case_group_count']}`",
        f"- lung-lobe exact/mismatch/error: `{focus['lung_lobe']['exact_equal_case_count']}`/`{focus['lung_lobe']['mismatch_case_count']}`/`{focus['lung_lobe']['error_case_count']}`",
        f"- brachiocephalic exact/mismatch/error: `{focus['brachiocephalic']['exact_equal_case_count']}`/`{focus['brachiocephalic']['mismatch_case_count']}`/`{focus['brachiocephalic']['error_case_count']}`",
        "",
        "## Alias Voxel Audit",
    ]
    for row in alias["group_summaries"]:
        lines.append(
            f"- `{row['alias_group']}`: `{row['decision']}`; "
            f"multi-source={row['multi_source_case_count']} exact={row['exact_case_count']} "
            f"mismatch={row['mismatch_case_count']} error={row['error_case_count']}"
        )
    lines.extend([
        "",
        "## Coronary",
        f"- proposed status: `{summary['coronary']['status']}`",
        f"- reason: `{summary['coronary']['reason']}`",
        f"- repo evidence matches: `{len(summary['coronary']['repo_matches'])}`",
        "",
        "## Source Priority",
        f"- global alias groups: `{';'.join(alias['global_priority_alias_groups'])}`",
        f"- case-level alias groups: `{';'.join(alias['case_level_priority_alias_groups'])}`",
        f"- blocked alias groups: `{';'.join(alias['blocked_alias_groups'])}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_focus_alias_summary(alias_summary: dict[str, Any]) -> dict[str, Any]:
    groups = alias_summary["group_summaries"]

    def add_rows(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "alias_groups": [row["alias_group"] for row in selected],
            "coexisting_case_count": sum(int(row["multi_source_case_count"]) for row in selected),
            "target_case_group_count": sum(int(row["multi_source_case_count"]) for row in selected),
            "exact_equal_case_count": sum(int(row["exact_case_count"]) for row in selected),
            "mismatch_case_count": sum(int(row["mismatch_case_count"]) for row in selected),
            "error_case_count": sum(int(row["error_case_count"]) for row in selected),
        }

    return {
        "celiac": add_rows([row for row in groups if row["alias_group"] == "alias_celiac_aa"]),
        "lung_lobe": add_rows([row for row in groups if str(row["alias_group"]).startswith("alias_lung_")]),
        "brachiocephalic": add_rows([row for row in groups if str(row["alias_group"]).startswith("alias_brachiocephalic_vein_")]),
    }


def run_proposal_audit(
    *,
    data_root: Path,
    case_manifest: Path,
    output_root: Path,
    mapping: Path,
    taxonomy: Path,
    boundary_classification: Path,
    alias_groups: Path,
    task2_targets: Path,
    semantic_evidence: Path,
    affine_atol: float = 1e-3,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    canonical_paths = [mapping, taxonomy, boundary_classification, alias_groups, task2_targets, semantic_evidence]
    canonical_before = fingerprint_canonical(canonical_paths)
    source_before = fingerprint_source_inventory(data_root, case_manifest)
    semantic_result = build_taxonomy_semantic_audit(
        taxonomy=taxonomy,
        mapping=mapping,
        boundary_classification=boundary_classification,
        alias_groups=alias_groups,
        task2_targets=task2_targets,
        semantic_evidence=semantic_evidence,
        output_root=output_root,
    )
    alias_result = audit_alias_voxel_equivalence(
        data_root=data_root,
        case_manifest=case_manifest,
        alias_groups=alias_groups,
        mapping=mapping,
        boundary_classification=boundary_classification,
        task2_targets=task2_targets,
        output_root=output_root,
        affine_atol=affine_atol,
    )
    coronary = coronary_proposed_decision(semantic_result["decision_rows"], repo_root)
    focus_alias_summary = build_focus_alias_summary(alias_result["summary"])
    write_proposed_configs(
        mapping=mapping,
        boundary_classification=boundary_classification,
        alias_groups=alias_groups,
        semantic_decision_rows=semantic_result["decision_rows"],
        coronary_decision=coronary,
        priority_rows=alias_result["priority_rows"],
        output_root=output_root,
    )
    write_pending_semantic_review(
        output_root=output_root,
        semantic_decision_rows=semantic_result["decision_rows"],
        coronary_decision=coronary,
        alias_result=alias_result,
    )
    canonical_after = fingerprint_canonical(canonical_paths)
    source_after = fingerprint_source_inventory(data_root, case_manifest)
    safety = {
        "canonical_config_mutation": canonical_before["hash"] != canonical_after["hash"],
        "source_data_mutation": source_before["hash"] != source_after["hash"],
        "source_inventory_before": {
            "hash": source_before["hash"],
            "file_count": source_before["file_count"],
            "total_bytes": source_before["total_bytes"],
        },
        "source_inventory_after": {
            "hash": source_after["hash"],
            "file_count": source_after["file_count"],
            "total_bytes": source_after["total_bytes"],
        },
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
    }
    summary = {
        "status": "failed" if safety["canonical_config_mutation"] or safety["source_data_mutation"] else "success",
        "read_only": True,
        "apply": False,
        "canonical_config_mutation": False,
        "source_data_mutation": False,
        "data_root": str(data_root),
        "case_manifest": str(case_manifest),
        "output_root": str(output_root),
        "taxonomy_semantic_audit": semantic_result["summary"],
        "alias_voxel_equivalence_summary": alias_result["summary"],
        "focus_alias_summary": focus_alias_summary,
        "coronary": coronary,
        "safety": safety,
        "expected_output_files": [
            "proposed/proposed_organ_rename_mapping_373.csv",
            "proposed/proposed_task_boundary_classification.csv",
            "proposed/proposed_task1_alias_groups.csv",
            "proposed/proposed_source_priority.csv",
            "proposed/proposed_taxonomy_semantic_evidence.csv",
            "alias_voxel_equivalence_rows.csv",
            "alias_voxel_equivalence_summary.json",
            "alias_voxel_equivalence_report.md",
            "blocked_alias_groups.csv",
            "blocked_cases.csv",
            "pending_semantic_review.csv",
            "taxonomy_semantic_pair_candidates.csv",
            "taxonomy_semantic_decisions.csv",
            "taxonomy_semantic_audit.json",
            "taxonomy_semantic_audit.md",
            "taxonomy_concept_groups.csv",
            "proposal_summary.json",
            "proposal_summary.md",
        ],
    }
    summary["canonical_config_mutation"] = bool(safety["canonical_config_mutation"])
    summary["source_data_mutation"] = bool(safety["source_data_mutation"])
    write_json(output_root / "proposal_summary.json", summary)
    write_proposal_summary_md(output_root / "proposal_summary.md", summary)
    if summary["status"] != "success":
        raise DeliveryError("Read-only proposal audit detected source or canonical config mutation")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Read-only Task 1 semantic and voxel proposal audit.")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--case-manifest", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--mapping", default=Path("configs/dataset_delivery/task1/organ_rename_mapping_373.csv"), type=Path)
    p.add_argument("--taxonomy", default=Path("configs/student_3d_prompt_target_organs.json"), type=Path)
    p.add_argument("--boundary-classification", default=Path("configs/dataset_delivery/task1/task_boundary_classification.csv"), type=Path)
    p.add_argument("--alias-groups", default=Path("configs/dataset_delivery/task1/task1_alias_groups.csv"), type=Path)
    p.add_argument("--task2-targets", default=Path("configs/dataset_delivery/task1/task2_generate_targets_23.csv"), type=Path)
    p.add_argument("--semantic-evidence", default=Path("configs/dataset_delivery/task1/taxonomy_semantic_evidence.csv"), type=Path)
    p.add_argument("--affine-atol", default=1e-3, type=float)
    args = p.parse_args()
    try:
        result = run_proposal_audit(
            data_root=args.data_root,
            case_manifest=args.case_manifest,
            output_root=args.output_root,
            mapping=args.mapping,
            taxonomy=args.taxonomy,
            boundary_classification=args.boundary_classification,
            alias_groups=args.alias_groups,
            task2_targets=args.task2_targets,
            semantic_evidence=args.semantic_evidence,
            affine_atol=args.affine_atol,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        summary_path = args.output_root / "proposal_summary.json"
        failure = {
            "status": "failed",
            "read_only": True,
            "apply": False,
            "canonical_config_mutation": False,
            "source_data_mutation": False,
            "error": str(exc),
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        if not summary_path.exists():
            write_json(summary_path, failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
