#!/usr/bin/env python3
"""Split a full 373 manifest into core, single-teacher ablation and audit views.

The splitter is deliberately conservative:
- core positive labels come only from family-free geometric teacher consensus;
- reliable absent negatives are copied into core and ablation;
- single-teacher entries are included only in the ablation view and remain low
  weight;
- LabelCritic audit records are retained with zero training weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_items(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return {"stage": "full_case_373_manifest"}, [dict(row) for row in doc]
    items = doc.get("items") or doc.get("records") or doc.get("labels") or []
    return doc, [dict(row) for row in items]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _zero_training(row: dict[str, Any], reason: str) -> dict[str, Any]:
    out = dict(row)
    out["training_weight"] = 0.0
    out["distillation_eligible"] = False
    out["should_enter_student_training"] = False
    out["distillation_exclusion_reason"] = reason
    return out


def _is_reliable_negative(row: dict[str, Any]) -> bool:
    return (
        row.get("target_type") == "negative_absent"
        and row.get("selection_method") == "negative_absent"
        and row.get("fov_status") == "out_of_fov"
        and row.get("absence_confidence") == "high"
        and row.get("zero_mask_role") == "negative_absent_target_mask"
    )


def _is_geometric_consensus_positive(row: dict[str, Any]) -> bool:
    return (
        row.get("selection_method") in {"geometric_teacher_consensus", "near_identical_agreement"}
        and row.get("selection_status") == "selected"
        and row.get("winner_is_original_teacher") is True
        and row.get("target_type") in {"positive_hard", "hard"}
        and row.get("selected_model")
        and row.get("selected_prediction")
    )


def _single_teacher_qc_eligible(row: dict[str, Any]) -> bool:
    flags = set(row.get("review_flags") or []) | set(row.get("quality_flags") or []) | set(row.get("selected_candidate_qc_flags") or [])
    hard_flags = {
        "missing_candidate",
        "missing_final_mask",
        "identity_mismatch",
        "left_right_mismatch",
        "zero_volume_mask",
        "expected_present_zero_volume_mask",
        "geometry_mismatch",
        "shape_mismatch_ct",
        "affine_mismatch_ct",
        "orientation_mismatch_ct",
        "candidate_qc_fail",
        "mask_lineage_mismatch",
    }
    return (
        row.get("selection_method") in {"single_teacher_provisional", "single_teacher_default"}
        and row.get("selection_status") in {"provisional", "selected"}
        and row.get("fov_status") == "fully_visible"
        and row.get("identity_status") in {None, "", "valid"}
        and row.get("selected_candidate_qc_status") in {None, "pass"}
        and row.get("selected_model")
        and row.get("selected_prediction")
        and not (flags & hard_flags)
        and row.get("description_status") != "runtime_only_unverified"
    )


def split_manifest(input_path: Path, output_dir: Path) -> dict[str, Any]:
    source_doc, rows = _read_items(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    core: list[dict[str, Any]] = []
    single_teacher: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for row in rows:
        method = row.get("selection_method")
        if _is_geometric_consensus_positive(row):
            item = dict(row)
            item["selection_method"] = "geometric_teacher_consensus"
            item["legacy_selection_method"] = row.get("selection_method")
            item["family_role"] = "audit_only_not_used_for_selection_or_training_gate"
            item["should_enter_student_training"] = True
            item["distillation_eligible"] = True
            item["training_weight"] = max(_as_float(item.get("training_weight"), 1.0), 1.0)
            core.append(item)
            continue
        if _is_reliable_negative(row):
            item = dict(row)
            item["should_enter_student_training"] = _as_float(item.get("training_weight"), 0.0) > 0.0
            item["distillation_eligible"] = item["should_enter_student_training"]
            core.append(item)
            continue
        if method in {"label_critic_audit_only", "label_critic_inconclusive", "label_critic"}:
            audit.append(_zero_training(row, "labelcritic_audit_manifest_zero_weight"))
        elif _single_teacher_qc_eligible(row):
            item = dict(row)
            item["selection_method"] = "single_teacher_qc_eligible"
            item["training_weight"] = 0.1
            item["distillation_eligible"] = True
            item["should_enter_student_training"] = True
            item["single_teacher_ablation_only"] = True
            single_teacher.append(item)

    ablation = [*core, *single_teacher]

    def write(name: str, items: list[dict[str, Any]], role: str) -> Path:
        path = output_dir / name
        path.write_text(json.dumps({
            "stage": role,
            "source_manifest": str(input_path.resolve()),
            "source_stage": source_doc.get("stage"),
            "num_items": len(items),
            "items": items,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    core_path = write("consensus_core_manifest.json", core, "consensus_core_manifest")
    ablation_path = write("single_teacher_ablation_manifest.json", ablation, "single_teacher_ablation_manifest")
    audit_path = write("labelcritic_audit_manifest.json", audit, "labelcritic_audit_manifest")
    summary = {
        "stage": "labelcritic_repair_manifest_split",
        "source_manifest": str(input_path.resolve()),
        "input_items": len(rows),
        "consensus_core_items": len(core),
        "geometric_consensus_positive_items": sum(1 for row in core if row.get("selection_method") == "geometric_teacher_consensus"),
        "negative_absent_items": sum(1 for row in core if row.get("selection_method") == "negative_absent"),
        "single_teacher_ablation_extra_items": len(single_teacher),
        "labelcritic_audit_items": len(audit),
        "outputs": {
            "consensus_core_manifest": str(core_path.resolve()),
            "single_teacher_ablation_manifest": str(ablation_path.resolve()),
            "labelcritic_audit_manifest": str(audit_path.resolve()),
        },
    }
    (output_dir / "manifest_split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Full 373 manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for split manifests")
    args = parser.parse_args()
    print(json.dumps(split_manifest(Path(args.input), Path(args.output_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
