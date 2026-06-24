#!/usr/bin/env python3
"""Offline AutoLabelCore v3 rescoring; source E-step artifacts remain read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.auto_fine_label import build_label_passport
from cli_anything.medai.core.auto_label_core import (
    SCORING_SCHEMA_VERSION, _load_mask, anatomy_support, binary_dice,
    ct_mask_support, score_evidence_record,
)
from cli_anything.medai.core.organ_taxonomy import load_taxonomy, topological_order_organs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "source_type": "historical_pseudo_label",
            "version": "imported_annotation_v1", "sha256": sha256(path)}


def detail(score: Any, status: str, source: str, reason: str | None = None,
           source_provenance: Any = None) -> dict[str, Any]:
    return {"score": score, "status": status, "source": source,
            "reason": reason, "provenance": source_provenance}


def dice(left: Path, right: Path) -> tuple[float | None, str]:
    try:
        if left.resolve() == right.resolve():
            return None, "self_reference"
        left_image, left_mask = _load_mask(left)
        right_image, right_mask = _load_mask(right)
        import numpy as np
        if left_mask.shape != right_mask.shape or not np.allclose(
            left_image.affine, right_image.affine, atol=1e-3
        ):
            return None, "geometry_mismatch"
        return binary_dice(left_mask, right_mask), "success"
    except Exception as exc:
        return None, f"read_failed:{exc}"


def rescore(metadata: Path, case: dict[str, str], taxonomy: dict[str, Any],
            output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc = json.loads(metadata.read_text(encoding="utf-8"))
    case_id = str(doc["case_id"])
    ct_path = Path(case.get("ct_path") or doc["ct_path"]).resolve()
    annotation = Path(case["annotation_folder"]).resolve()
    selected = {str(row["organ"]): dict(row)
                for row in doc.get("selected_organs", [])}
    ordered = topological_order_organs(taxonomy, list(selected))
    masks = {
        organ: Path(row.get("final_mask") or row.get("mask_path") or row.get("mask")).resolve()
        for organ, row in selected.items()
        if row.get("final_mask") or row.get("mask_path") or row.get("mask")
    }
    rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    passport_dir = output / "passports" / case_id
    passport_dir.mkdir(parents=True, exist_ok=True)

    for organ in ordered:
        row = selected[organ]
        mask = masks.get(organ)
        if mask is None or not mask.exists():
            continue
        ct_result = ct_mask_support(ct_path, mask)
        parent_masks = {parent: masks[parent] for parent in row.get("parent_ids", [])
                        if parent in masks and masks[parent].exists()}
        anatomy = anatomy_support({
            "prediction": str(mask), "parent_ids": row.get("parent_ids", []),
            "candidate_qc_flags": row.get("selected_candidate_qc_flags", []),
            "identity_status": row.get("identity_status"),
        }, parent_masks, 0.95)

        reference = annotation / f"{organ}.nii.gz"
        reference_info = provenance(reference) if reference.exists() else None
        cross_score, cross_reason = (dice(reference, mask) if reference.exists()
                                     else (None, "historical_reference_unavailable"))
        cross_status = ("available" if cross_score is not None else
                        "failed" if reference.exists() else "unavailable")
        old_scores = row.get("evidence_scores") or {}
        family_score = old_scores.get("family_consensus")
        tta_score = old_scores.get("perturbation_stability")
        loo_score = old_scores.get("loo_model_reliability")
        evidence_details = {
            "family_consensus": detail(
                family_score, "available" if family_score is not None else "unavailable",
                "independent_teacher_families",
                None if family_score is not None else "single_teacher_family"),
            "ct_support": detail(
                ct_result.get("score"), "available" if ct_result.get("score") is not None else "failed",
                "ct_mask_support", ct_result.get("status")),
            "anatomy_plausibility": detail(
                anatomy.get("score"), "available" if anatomy.get("score") is not None else
                ("unavailable" if anatomy.get("status") == "missing_parent" else "failed"),
                "parent_containment", anatomy.get("status")),
            "perturbation_stability": detail(
                tta_score, "available" if tta_score is not None else "unavailable",
                "risk_triggered_tta", None if tta_score is not None else "not_cached_requires_risk_tta"),
            "cross_round_stability": detail(
                cross_score, cross_status, "historical_pseudo_label",
                cross_reason, reference_info),
            "loo_model_reliability": detail(
                loo_score, "available" if loo_score is not None else "unavailable",
                "leave_one_family_out",
                None if loo_score is not None else "insufficient_other_families"),
            "teacher_student_oof": detail(
                None, "unavailable", "verified_oof_student",
                "oof_student_not_available"),
        }
        scored = score_evidence_record({
            **row, "mask_path": str(mask), "evidence_details": evidence_details,
            "cross_round_stability_score": cross_score,
            "selected_pseudo_consistency_dice": cross_score,
            "reference_provenance": reference_info,
            "out_of_fold_verified": False,
        })
        updated = {
            **row, **scored, "case_id": case_id, "ct_path": str(ct_path),
            "mask_path": str(mask), "mask": str(mask), "final_mask": str(mask),
            "reference": str(reference) if reference.exists() else None,
            "reference_role": "historical_pseudo_label" if reference.exists() else "none",
            "reference_provenance": reference_info,
            "selected_pseudo_consistency_dice": cross_score,
            "family_consensus_dice": family_score,
            "teacher_student_oof_dice": None,
            "out_of_fold_verified": False,
            "distillation_eligible": float(scored["training_weight"]) > 0.0,
            "distillation_exclusion_reason": (
                None if float(scored["training_weight"]) > 0.0
                else "autolabel_core_v3_training_weight_zero"),
            "accuracy_warning": "Pseudo-consistency and evidence reliability are not expert accuracy.",
        }
        passport = build_label_passport(updated)
        passport_path = passport_dir / f"{organ}.label_passport.json"
        passport_path.write_text(json.dumps(passport, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        updated["label_passport_path"] = str(passport_path)
        rows.append(updated)
        changes.append({
            "case_id": case_id, "organ": organ,
            "old_grade": row.get("grade"), "new_grade": scored["grade"],
            "old_confidence": row.get("evidence_confidence"),
            "new_confidence": scored["evidence_confidence"],
            "grade_cap": scored.get("grade_cap"),
            "grade_cap_reason": scored.get("grade_cap_reason"),
        })
    return rows, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estep", required=True)
    parser.add_argument("--case-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--taxonomy", default=str(ROOT / "configs/organ_taxonomy.json"))
    parser.add_argument("--expected-cases", type=int, default=20)
    args = parser.parse_args()
    estep, output = Path(args.estep).resolve(), Path(args.output).resolve()
    if output == estep or estep in output.parents:
        raise SystemExit("Output must be outside the source E-step directory")
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.case_list).resolve().open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        cases = {str(row["case_id"]): row for row in csv.DictReader(handle)}
    metadata = sorted((estep / "annotation_versions").glob("*/selection_metadata.json"))
    if len(metadata) != args.expected_cases:
        raise SystemExit(f"Expected {args.expected_cases} cases, found {len(metadata)}")
    taxonomy = load_taxonomy(Path(args.taxonomy).resolve())
    all_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for path in metadata:
        case_id = path.parent.name
        if case_id not in cases:
            raise SystemExit(f"Case missing from case list: {case_id}")
        rows, changes = rescore(path, cases[case_id], taxonomy, output)
        all_rows.extend(rows)
        comparisons.extend(changes)

    eligible = [row for row in all_rows if row.get("distillation_eligible") is True
                and float(row.get("training_weight") or 0.0) > 0.0]
    excluded = [row for row in all_rows if row not in eligible]
    (output / "training_manifest.json").write_text(
        json.dumps(eligible, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "training_manifest.training_exclusions.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "v2_v3_comparison.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")

    case_ids = sorted({row["case_id"] for row in all_rows})
    index = {"stage": "standard_dataset_index_v3", "status": "success",
             "case_count": len(case_ids), "cases": []}
    for case_id in case_ids:
        rows = [row for row in all_rows if row["case_id"] == case_id]
        index["cases"].append({
            "case_id": case_id, "image": cases[case_id]["ct_path"],
            "source_segmentations": str(
                (estep / "annotation_versions" / case_id / "updated").resolve()),
            "num_masks": len(rows),
            "num_training_eligible": sum(
                row.get("distillation_eligible") is True for row in rows),
        })
    (output / "standard_dataset_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["case_id", "organ", "grade", "training_weight",
              "evidence_confidence", "selected_pseudo_consistency_dice",
              "family_consensus_dice", "teacher_student_oof_dice",
              "grade_cap", "grade_cap_reason"]
    with (output / "label_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields}
                          for row in all_rows])

    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in all_rows:
        for name, item in (row.get("evidence_details") or {}).items():
            statuses[f"{name}:{item.get('status')}"] += 1
            if item.get("reason"):
                reasons[f"{name}:{item.get('reason')}"] += 1
    report = {
        "status": "success", "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "source_estep": str(estep), "output": str(output),
        "num_cases": len(case_ids), "num_labels": len(all_rows),
        "grade_counts": dict(Counter(str(row.get("grade")) for row in all_rows)),
        "eligible_labels": len(eligible), "excluded_labels": len(excluded),
        "evidence_status_counts": dict(statuses),
        "evidence_reason_counts": dict(reasons), "source_unchanged": True,
    }
    (output / "evidence_coverage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

