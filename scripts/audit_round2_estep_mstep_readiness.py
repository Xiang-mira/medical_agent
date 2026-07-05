#!/usr/bin/env python3
"""Audit archived Round2 E-step labels before any Round2 M-step.

This script does not train and does not run student inference.  It reads the
archived Round2 E-step metadata, classifies each case/organ label as usable or
not usable for a future M-step, and records whether there is any new reliable
positive supervision beyond Round1 replay.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVE_ROOT = ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"
DEFAULT_ARCHIVED_ROUND2 = (
    ROOT
    / "outputs"
    / "archived_bad_round2_round3_20260705"
    / "round2_bad_retrospective_quality_regression"
)
DEFAULT_SOURCE_ESTEP = DEFAULT_ARCHIVED_ROUND2 / "estep"
DEFAULT_CASE_LIST = ROOT / "outputs" / "formal_round1_final_20260627" / "case_list_10.csv"
DEFAULT_ROUND1_MANIFEST = (
    DEFAULT_ACTIVE_ROOT / "round1" / "mstep" / "voxtell_prompt_student_manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "round2_estep_mstep_readiness_20260705"
OLD_ACTIVE_ROUND2 = DEFAULT_ACTIVE_ROOT / "round2"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_case_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            str(row.get("case_id") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or "").strip()
        ]


def round1_replay_positive_keys(manifest_path: Path) -> set[tuple[str, str]]:
    doc = read_json(manifest_path, {})
    keys: set[tuple[str, str]] = set()
    for row in doc.get("items") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("supervision_type") or "").lower() != "positive":
            continue
        if row.get("distillation_eligible") is False:
            continue
        try:
            if float(row.get("training_weight") or 0.0) <= 0.0:
                continue
        except Exception:
            continue
        case_id = str(row.get("case_id") or "").strip()
        organ = str(row.get("organ") or "").strip()
        if case_id and organ:
            keys.add((case_id, organ))
    return keys


def remap_archived_path(path_value: Any, archived_round2: Path) -> tuple[str, bool]:
    raw = str(path_value or "").strip()
    if not raw:
        return "", False
    path = Path(raw)
    candidates = [path]
    try:
        relative = path.resolve().relative_to(OLD_ACTIVE_ROUND2.resolve())
        candidates.append(archived_round2.resolve() / relative)
    except Exception:
        text = raw
        marker = str(OLD_ACTIVE_ROUND2)
        if marker in text:
            candidates.append(Path(text.replace(marker, str(archived_round2), 1)))
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return str(candidate), True
    return str(candidates[-1]), False


def selected_rows_for_case(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per organ, preferring selected_organs because it has masks."""
    by_organ: dict[str, dict[str, Any]] = {}
    for row in metadata.get("selection_rows") or []:
        if isinstance(row, dict) and row.get("organ"):
            by_organ[str(row["organ"])] = dict(row)
    for row in metadata.get("selected_organs") or []:
        if isinstance(row, dict) and row.get("organ"):
            merged = {**by_organ.get(str(row["organ"]), {}), **dict(row)}
            by_organ[str(row["organ"])] = merged
    return [by_organ[key] for key in sorted(by_organ)]


def classify_row(
    row: dict[str, Any],
    *,
    expected_case_ids: set[str],
    replay_keys: set[tuple[str, str]],
    archived_round2: Path,
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "").strip()
    organ = str(row.get("organ") or "").strip()
    grade = str(row.get("grade") or "").upper()
    target_type = str(row.get("target_type") or "").lower()
    method = str(row.get("selection_method") or "")
    selected_model = str(row.get("selected_model") or row.get("model") or "")
    scoring_schema = str(row.get("scoring_schema_version") or "")
    quality_status = str(row.get("quality_status") or "")
    try:
        weight = float(row.get("training_weight") or 0.0)
    except Exception:
        weight = 0.0
    mask_path, mask_exists = remap_archived_path(
        row.get("final_mask") or row.get("mask_path") or row.get("mask"),
        archived_round2,
    )
    probability_path, probability_exists = remap_archived_path(
        row.get("probability_mask_path"),
        archived_round2,
    )

    decision = "exclude"
    reason = "not_a_trainable_round2_mstep_label"
    role = "excluded"
    is_new_positive = False

    if case_id not in expected_case_ids:
        reason = "case_not_in_formal_10case_list"
    elif target_type == "negative_absent":
        if weight > 0.0 and mask_exists:
            decision = "usable"
            role = "legal_negative"
            reason = "legal_negative_absent"
        else:
            reason = "negative_absent_missing_mask_or_weight"
    elif grade in {"A", "B"} and target_type in {"positive_hard", "hard"}:
        if not mask_exists or weight <= 0.0:
            reason = "positive_hard_missing_mask_or_weight"
        elif selected_model == "round_prev_selected" or method == "round_prev_selected_carry_forward":
            role = "historical_replay_reference"
            reason = (
                "round_prev_selected_is_not_new_round2_content"
                if (case_id, organ) not in replay_keys
                else "already_covered_by_round1_replay"
            )
        elif method == "geometric_teacher_consensus" and scoring_schema.startswith("autolabel_core_v3"):
            decision = "candidate"
            role = "new_positive_candidate"
            reason = "new_ab_hard_candidate_requires_final_pre_mstep_review"
            is_new_positive = True
        else:
            reason = "positive_hard_source_not_allowed_for_mstep"
    elif grade == "C" and target_type in {"positive_soft", "soft"}:
        if not probability_exists or weight <= 0.0:
            reason = "c_soft_missing_probability_or_weight"
        elif method == "soft_consensus_from_labelcritic_abstention":
            reason = "soft_consensus_from_abstention_not_reliable_for_mstep"
        elif not scoring_schema.startswith("autolabel_core_v3"):
            reason = "c_soft_requires_current_scoring_schema_review"
        else:
            decision = "candidate"
            role = "new_soft_positive_candidate"
            reason = "new_c_soft_candidate_requires_final_pre_mstep_review"
            is_new_positive = True
    elif target_type in {"rejected", "unresolved_visible", "partial_fov"}:
        reason = f"{target_type}_not_usable_for_mstep"

    return {
        "case_id": case_id,
        "organ": organ,
        "decision": decision,
        "role": role,
        "reason": reason,
        "is_new_positive_candidate": is_new_positive,
        "grade": grade,
        "target_type": target_type,
        "training_weight": weight,
        "selection_method": method,
        "selected_model": selected_model,
        "quality_status": quality_status,
        "scoring_schema_version": scoring_schema,
        "mask_path": mask_path,
        "mask_exists": mask_exists,
        "probability_mask_path": probability_path,
        "probability_mask_exists": probability_exists,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "organ",
        "decision",
        "role",
        "reason",
        "is_new_positive_candidate",
        "grade",
        "target_type",
        "training_weight",
        "selection_method",
        "selected_model",
        "quality_status",
        "scoring_schema_version",
        "mask_exists",
        "probability_mask_exists",
        "mask_path",
        "probability_mask_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit archived Round2 E-step labels before any M-step."
    )
    parser.add_argument("--source-estep", type=Path, default=DEFAULT_SOURCE_ESTEP)
    parser.add_argument("--archived-round2-root", type=Path, default=DEFAULT_ARCHIVED_ROUND2)
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--round1-manifest", type=Path, default=DEFAULT_ROUND1_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--active-summary",
        type=Path,
        default=DEFAULT_ACTIVE_ROOT / "round2_no_material_update_summary.json",
    )
    args = parser.parse_args()

    source_versions = args.source_estep.resolve() / "annotation_versions"
    if not source_versions.is_dir():
        raise SystemExit(f"Missing source annotation_versions: {source_versions}")
    expected_case_ids = read_case_ids(args.case_list.resolve())
    replay_keys = round1_replay_positive_keys(args.round1_manifest.resolve())
    rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()

    for metadata_path in sorted(source_versions.glob("*/selection_metadata.json")):
        metadata = read_json(metadata_path, {})
        case_id = str(metadata.get("case_id") or metadata_path.parent.name)
        seen_cases.add(case_id)
        for row in selected_rows_for_case(metadata):
            row = dict(row)
            row.setdefault("case_id", case_id)
            rows.append(
                classify_row(
                    row,
                    expected_case_ids=set(expected_case_ids),
                    replay_keys=replay_keys,
                    archived_round2=args.archived_round2_root.resolve(),
                )
            )

    role_counts = Counter(row["role"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows)
    decision_counts = Counter(row["decision"] for row in rows)
    current_new_positive_candidates = [
        row for row in rows if row["is_new_positive_candidate"]
    ]
    decision = (
        "material_update_candidate"
        if current_new_positive_candidates
        else "no_material_update"
    )
    max_steps = 0 if decision == "no_material_update" else None

    output_dir = args.output_dir.resolve()
    rows_csv = output_dir / "round2_estep_mstep_readiness_rows.csv"
    candidates_csv = output_dir / "round2_estep_new_positive_candidates.csv"
    write_csv(rows_csv, rows)
    write_csv(candidates_csv, current_new_positive_candidates)

    summary = {
        "stage": "round2_estep_mstep_readiness_audit",
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_estep": str(args.source_estep.resolve()),
        "case_list": str(args.case_list.resolve()),
        "round1_manifest": str(args.round1_manifest.resolve()),
        "expected_case_ids": expected_case_ids,
        "source_case_ids": sorted(seen_cases),
        "case_scope_status": (
            "passed" if set(seen_cases) == set(expected_case_ids) else "failed"
        ),
        "round1_replay_positive_count": len(replay_keys),
        "total_case_organ_rows": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "top_exclusion_reasons": reason_counts.most_common(20),
        "legal_negative_count": role_counts.get("legal_negative", 0),
        "historical_replay_reference_count": role_counts.get(
            "historical_replay_reference", 0
        ),
        "current_new_positive_candidate_count": len(current_new_positive_candidates),
        "decision": decision,
        "max_steps": max_steps,
        "mstep_allowed": False,
        "mstep_reason": (
            "No new reliable positive labels were found; keep reusing Round1 checkpoint."
            if decision == "no_material_update"
            else "New positive candidates require manual/final pre-M-step approval before training."
        ),
        "outputs": {
            "rows_csv": str(rows_csv),
            "new_positive_candidates_csv": str(candidates_csv),
            "summary_json": str(output_dir / "round2_estep_mstep_readiness_summary.json"),
        },
    }
    write_json(output_dir / "round2_estep_mstep_readiness_summary.json", summary)

    if args.active_summary.exists():
        active = read_json(args.active_summary, {})
        active["round2_estep_mstep_readiness_audit"] = {
            "summary": str(output_dir / "round2_estep_mstep_readiness_summary.json"),
            "rows_csv": str(rows_csv),
            "decision": decision,
            "max_steps": max_steps,
            "mstep_allowed": False,
        }
        write_json(args.active_summary, active)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
