#!/usr/bin/env python3
"""Regrade completed 373 repair artifacts and build the VoxTell M-step manifest."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.auto_fine_label import build_label_passport, compute_reliability
from cli_anything.medai.core.voxtell_student import VoxTellStudent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estep", default=str(ROOT / "outputs/round1_373_hierarchical_repair_20260620/estep"))
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--expected-cases", type=int, default=50)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    estep = Path(args.estep).resolve()
    metadata_paths = sorted(estep.glob("annotation_versions/*/selection_metadata.json"))
    if len(metadata_paths) != args.expected_cases and not args.allow_partial:
        raise SystemExit(f"Expected {args.expected_cases} completed cases, found {len(metadata_paths)}")

    grades: Counter[str] = Counter()
    regraded = 0
    identity_failures = []
    old_child_leaks = []
    for path in metadata_paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("selected_organs", []) or []:
            if row.get("identity_status") != "valid":
                identity_failures.append({"case_id": doc.get("case_id"), "organ": row.get("organ")})
            source = str(row.get("selected_pre_shapekit_prediction") or "")
            if "stage4b_round1_50cases_20260611" in source and (row.get("parent_ids") or []):
                old_child_leaks.append({"case_id": doc.get("case_id"), "organ": row.get("organ"), "source": source})
            reliability = compute_reliability(row)
            row.update(reliability)
            row["distillation_eligible"] = reliability["training_weight"] > 0.0
            row["distillation_exclusion_reason"] = None if row["distillation_eligible"] else "grade_D_or_zero_weight"
            row["student_training_priority"] = reliability["grade"]
            row["label_confidence"] = reliability["auto_fine_label_reliability_score"]
            passport_path = row.get("label_passport_path")
            if passport_path:
                Path(passport_path).write_text(json.dumps(build_label_passport(row), indent=2, ensure_ascii=False), encoding="utf-8")
            grades[reliability["grade"]] += 1
            regraded += 1
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    if identity_failures or old_child_leaks:
        raise SystemExit(json.dumps({"identity_failures": identity_failures[:20], "old_child_leaks": old_child_leaks[:20]}, indent=2))

    mstep = estep.parent / "mstep"
    mstep.mkdir(parents=True, exist_ok=True)
    student = VoxTellStudent(
        model_dir=ROOT / "checkpoints/VoxTell/voxtell_v1.1",
        target_config=ROOT / "configs/student_3d_prompt_target_organs.json",
        device="cpu",
    )
    manifest = student.build_training_manifest(
        cases_root=estep / "annotation_versions",
        output_manifest=mstep / "voxtell_prompt_student_manifest.json",
        case_list=Path(args.case_list).resolve(),
        require_images=True,
    )
    positive = [row for row in manifest["items"] if row.get("supervision_type") == "positive"]
    invalid_positive = [row for row in positive if row.get("grade") == "D" or float(row.get("training_weight") or 0.0) <= 0.0]
    report = {
        "status": "success" if not invalid_positive else "failed",
        "completed_cases": len(metadata_paths),
        "regraded_selected_masks": regraded,
        "grade_counts": dict(grades),
        "mstep_manifest": str((mstep / "voxtell_prompt_student_manifest.json").resolve()),
        "mstep_items": manifest.get("num_items"),
        "positive_items": len(positive),
        "invalid_positive_items": len(invalid_positive),
        "skipped_ineligible_positive": manifest.get("num_skipped_ineligible_positive"),
        "identity_failures": 0,
        "old_child_leaks": 0,
    }
    (mstep / "repair_mstep_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
