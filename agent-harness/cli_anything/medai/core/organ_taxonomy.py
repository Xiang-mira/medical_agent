from __future__ import annotations

import hashlib
import json
import re
import datetime
from pathlib import Path
from typing import Any


TAXONOMY_SCHEMA_VERSION = "1.0"
COMPARISON_FAMILIES = {
    "whole_organ",
    "sub_organ",
    "vessel_or_small_structure",
    "other_noncomparable",
}

_VESSEL_OR_SMALL = re.compile(
    r"(?:^|_)(?:aorta|artery|arterial|vein|vena|venous|vessel|duct|airway|bronchus|"
    r"canal|sinus|trunk|stent|nerve|cord)(?:_|$)"
)
_OTHER = re.compile(
    r"(?:^|_)(?:tumou?r|cancer|lesion|cyst|mass|implant|prosthetic|fluid|blood|fat|"
    r"adipose|content|cavity|body|skin|muscle|bones|bone|cartilage|tissue)(?:_|$)"
)
_ANATOMICAL_SUBORGAN = re.compile(
    r"(?:^|_)(?:segment_[1-9][0-9]*|head|body|tail)(?:_|$)"
)


def normalize_canonical_id(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def parse_binary_flag(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "yes", "true"}:
        return True
    if text in {"0", "0.0", "no", "false"}:
        return False
    return None


def parse_parent_ids(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parents: list[str] = []
    for item in re.split(r"[,;/+|]", raw):
        parent = normalize_canonical_id(item)
        if not parent:
            continue
        # The workbook has one synthetic bilateral parent. It is an ROI-only union.
        expanded = ["breast_left", "breast_right"] if parent == "breast" else [parent]
        for candidate in expanded:
            if candidate not in parents:
                parents.append(candidate)
    return parents


def infer_comparison_family(canonical_id: str, hierarchy_role: str) -> str:
    organ = normalize_canonical_id(canonical_id)
    if _VESSEL_OR_SMALL.search(organ):
        return "vessel_or_small_structure"
    # Anatomical child regions like pancreas_body must stay comparable only to
    # their own sub-organ family, even though tokens like "body" are otherwise
    # used by non-comparable labels such as body_trunc.
    if hierarchy_role == "child" and _ANATOMICAL_SUBORGAN.search(organ):
        return "sub_organ"
    if _OTHER.search(organ):
        return "other_noncomparable"
    if hierarchy_role == "child":
        return "sub_organ"
    return "whole_organ"


def workbook_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_taxonomy(records: list[dict[str, Any]], source_xlsx: str | Path) -> dict[str, Any]:
    organs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for record in records:
        canonical_id = normalize_canonical_id(record.get("organ") or record.get("canonical_id"))
        major = parse_binary_flag(record.get("major_organ"))
        if not canonical_id:
            continue
        if canonical_id in organs:
            errors.append({"type": "duplicate_canonical_id", "organ": canonical_id})
            continue
        if major is None:
            errors.append({"type": "invalid_major_organ", "organ": canonical_id, "value": record.get("major_organ")})
            continue
        role = "major" if major else "child"
        parents = parse_parent_ids(record.get("primary_organ_involved"))
        if role == "child" and not parents:
            errors.append({"type": "child_missing_parent", "organ": canonical_id})
        if role == "major" and parents:
            errors.append({"type": "major_has_parent", "organ": canonical_id, "parents": parents})
        candidates = [str(x) for x in record.get("candidate_models", []) if str(x)]
        organs[canonical_id] = {
            "canonical_id": canonical_id,
            "display_name": str(record.get("organ_display") or canonical_id),
            "hierarchy_role": role,
            "parent_ids": parents,
            "comparison_family": infer_comparison_family(canonical_id, role),
            "candidate_models": candidates,
            "primary_teacher": candidates[0] if candidates else None,
            "backup_teachers": candidates[1:],
        }

    for organ, entry in organs.items():
        for parent in entry["parent_ids"]:
            parent_entry = organs.get(parent)
            if parent_entry is None:
                errors.append({"type": "missing_parent", "organ": organ, "parent": parent})
            elif parent_entry["hierarchy_role"] != "major":
                errors.append({"type": "parent_not_major", "organ": organ, "parent": parent})

    source = Path(source_xlsx).resolve()
    source_mtime = datetime.datetime.fromtimestamp(source.stat().st_mtime, tz=datetime.timezone.utc).isoformat()
    return {
        "schema_version": TAXONOMY_SCHEMA_VERSION,
        "source_xlsx": str(source),
        "source_sha256": workbook_sha256(source),
        "source_modified_utc": source_mtime,
        "num_organs": len(organs),
        "comparison_families": sorted(COMPARISON_FAMILIES),
        "organs": dict(sorted(organs.items())),
        "validation": {"status": "success" if not errors else "failed", "errors": errors},
        "identity_policy": {
            "comparison_key": ["case_id", "canonical_id"],
            "parent_is_roi_only": True,
            "cross_canonical_comparison_allowed": False,
            "unknown_mapping_policy": "block",
        },
    }


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def taxonomy_entry(taxonomy: dict[str, Any], organ: str) -> dict[str, Any] | None:
    return (taxonomy.get("organs", {}) or {}).get(normalize_canonical_id(organ))


def topological_order_organs(
    taxonomy: dict[str, Any],
    organs: list[str],
    *,
    strict: bool = False,
) -> list[str]:
    """Return parents before children and reject cycles deterministically."""
    requested = [normalize_canonical_id(x) for x in organs]
    requested_set = set(requested)
    entries = taxonomy.get("organs", {}) or {}
    if strict:
        missing = sorted(x for x in requested if x not in entries)
        if missing:
            raise ValueError(f"Organs missing from taxonomy: {missing}")
    state: dict[str, int] = {}
    ordered: list[str] = []

    def visit(organ: str, stack: list[str]) -> None:
        marker = state.get(organ, 0)
        if marker == 2:
            return
        if marker == 1:
            cycle = " -> ".join([*stack, organ])
            raise ValueError(f"Taxonomy parent cycle detected: {cycle}")
        state[organ] = 1
        entry = entries.get(organ) or {}
        for parent in entry.get("parent_ids", []) or []:
            parent = normalize_canonical_id(parent)
            if parent in requested_set:
                visit(parent, [*stack, organ])
            elif strict and parent not in entries:
                raise ValueError(f"Taxonomy parent missing: {organ} -> {parent}")
        state[organ] = 2
        ordered.append(organ)

    for organ in requested:
        visit(organ, [])
    return ordered


def identity_contract(
    taxonomy: dict[str, Any],
    requested_organ: str,
    source_local_label: str,
    resolved_organ: str,
    *,
    mapping_type: str,
    mapping_source: str,
) -> dict[str, Any]:
    requested = normalize_canonical_id(requested_organ)
    resolved = normalize_canonical_id(resolved_organ)
    entry = taxonomy_entry(taxonomy, requested)
    resolved_entry = taxonomy_entry(taxonomy, resolved)
    reasons: list[str] = []
    if entry is None:
        reasons.append("requested_canonical_id_missing_from_taxonomy")
    if resolved_entry is None:
        reasons.append("resolved_canonical_id_missing_from_taxonomy")
    if requested != resolved:
        reasons.append("canonical_id_mismatch")
    if entry and resolved_entry and entry.get("comparison_family") != resolved_entry.get("comparison_family"):
        reasons.append("comparison_family_mismatch")
    if mapping_type in {"forbidden_coarsening", "unknown"}:
        reasons.append(f"mapping_type_{mapping_type}")
    return {
        "requested_canonical_id": requested,
        "source_local_label": source_local_label,
        "resolved_canonical_id": resolved,
        "comparison_family": entry.get("comparison_family") if entry else None,
        "parent_ids": list(entry.get("parent_ids", [])) if entry else [],
        "mapping_type": mapping_type,
        "mapping_source": mapping_source,
        "identity_status": "valid" if not reasons else "identity_mismatch",
        "identity_mismatch_reasons": reasons,
    }


def assert_same_identity(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str | None]:
    for item in (left, right):
        if item.get("identity_status") != "valid":
            return False, "candidate_identity_invalid"
    if left.get("requested_canonical_id") != right.get("requested_canonical_id"):
        return False, "canonical_id_mismatch"
    if left.get("comparison_family") != right.get("comparison_family"):
        return False, "comparison_family_mismatch"
    return True, None
