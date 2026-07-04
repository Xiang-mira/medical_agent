#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_taxonomy import load_taxonomy, normalize_canonical_id, taxonomy_entry
from cli_anything.medai.core.student_postprocess import containment_rule_for_organ, load_yaml, organ_group

HARD_BLOCK = {
    "abdominal_cavity",
    "muscle",
    "subcutaneous_adipose_tissue",
    "caudate_nucleus",
    "white_matter",
    "brain_ventricle",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fail-closed organ allowlist from 10-case raw/containment/cascade A/B.")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--containment-root", type=Path, required=True)
    p.add_argument("--cascade-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, action="append", required=True)
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--postprocessed-root", type=Path, required=True)
    p.add_argument("--taxonomy", type=Path, default=ROOT / "configs/organ_taxonomy.json")
    p.add_argument("--policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    p.add_argument("--expected-cases", type=int, default=10)
    return p.parse_args()


def case_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle) if r.get("case_id")]


def reference_path(
    roots: list[Path],
    case_id: str,
    organ: str,
    annotation_folder: str = "",
) -> tuple[Path | None, str]:
    if annotation_folder:
        gt_root = Path(annotation_folder)
        for path in [gt_root / f"{organ}.nii.gz", gt_root / "segmentations" / f"{organ}.nii.gz"]:
            if path.is_file():
                return path, "expert_gt"
    for root in roots:
        for path in [
            root / case_id / "updated" / f"{organ}.nii.gz",
            root / case_id / f"{organ}.nii.gz",
            root / "cases" / case_id / "updated" / f"{organ}.nii.gz",
            root / "cases" / case_id / "final" / f"{organ}.nii.gz",
            root / "cases" / case_id / f"{organ}.nii.gz",
        ]:
            if path.is_file():
                return path, "selected_teacher"
    return None, ""


def read(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    if reference is not None and (
        image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4)
    ):
        image = resample_from_to(image, reference, order=0)
    return image, np.asanyarray(image.dataobj) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def main() -> int:
    a = args()
    rows_by_case = {row["case_id"]: row for row in case_rows(a.case_list)}
    cases = list(rows_by_case)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.postprocessed_root.mkdir(parents=True, exist_ok=True)
    if len(cases) != a.expected_cases:
        raise SystemExit(f"expected exactly {a.expected_cases} cases, got {len(cases)}")
    taxonomy = load_taxonomy(a.taxonomy)
    policy = load_yaml(a.policy)
    organs = sorted({
        normalize_canonical_id(p.name[:-7])
        for case in cases
        for p in (a.raw_root / case).glob("*.nii.gz")
        if not p.name.startswith("ct_")
    })
    per_case: list[dict[str, Any]] = []
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for organ in organs:
        rule = containment_rule_for_organ(organ, taxonomy, policy)
        for case in cases:
            ref_path, reference_kind = reference_path(
                a.reference_root,
                case,
                organ,
                rows_by_case[case].get("annotation_folder", ""),
            )
            raw_path = a.raw_root / case / f"{organ}.nii.gz"
            containment_path = a.containment_root / case / f"{organ}.nii.gz"
            cascade_path = a.cascade_root / case / f"{organ}.nii.gz"
            row: dict[str, Any] = {
                "case_id": case,
                "organ": organ,
                "reference_path": str(ref_path or ""),
                "reference_kind": reference_kind,
                "raw_path": str(raw_path if raw_path.is_file() else ""),
                "containment_path": str(containment_path if containment_path.is_file() else ""),
                "cascade_path": str(cascade_path if cascade_path.is_file() else ""),
                "has_parent_rule": rule.enabled,
            }
            if ref_path is None or not raw_path.is_file():
                row["status"] = "missing_reference_or_raw"
                per_case.append(row)
                by_organ[organ].append(row)
                continue
            ref_img, ref = read(ref_path)
            _, raw = read(raw_path, ref_img)
            row.update({
                "status": "evaluated",
                "reference_nonempty": bool(ref.any()),
                "raw_nonempty": bool(raw.any()),
                "raw_dice": dice(raw, ref),
                "raw_volume_ratio": float(raw.sum() / ref.sum()) if ref.any() else None,
                "raw_false_positive_voxels": int(raw.sum()) if not ref.any() else 0,
            })
            for version, path in [("containment", containment_path), ("cascade", cascade_path)]:
                if not path.is_file():
                    row[f"{version}_available"] = False
                    continue
                _, pred = read(path, ref_img)
                row.update({
                    f"{version}_available": True,
                    f"{version}_nonempty": bool(pred.any()),
                    f"{version}_dice": dice(pred, ref),
                    f"{version}_volume_ratio": float(pred.sum() / ref.sum()) if ref.any() else None,
                    f"{version}_false_positive_voxels": int(pred.sum()) if not ref.any() else 0,
                })
                if version == "cascade" and rule.enabled:
                    roi_path = a.cascade_root / "parent_rois" / case / f"{organ}_allowed_roi.nii.gz"
                    if roi_path.is_file():
                        _, roi = read(roi_path, ref_img)
                        row["cascade_outside_roi_voxels"] = int(np.logical_and(pred, ~roi).sum())
                    else:
                        row["cascade_outside_roi_voxels"] = None
            per_case.append(row)
            by_organ[organ].append(row)

    decisions: list[dict[str, Any]] = []
    for organ in organs:
        rows = by_organ[organ]
        rule = containment_rule_for_organ(organ, taxonomy, policy)
        evaluated = [r for r in rows if r.get("status") == "evaluated"]
        ref_positive = [r for r in evaluated if r.get("reference_nonempty")]
        negative_rows = [r for r in evaluated if not r.get("reference_nonempty")]
        low, high = (0.2, 3.0) if organ_group(organ) == "vessel_or_duct" else (0.5, 1.8)

        def summarize(version: str) -> dict[str, Any]:
            metric_key = f"{version}_dice"
            available = [r for r in ref_positive if isinstance(r.get(metric_key), (int, float))]
            dscs = [float(r[metric_key]) for r in available]
            volumes = [
                float(r[f"{version}_volume_ratio"])
                for r in available
                if isinstance(r.get(f"{version}_volume_ratio"), (int, float))
            ]
            recall = (
                sum(bool(r.get(f"{version}_nonempty")) for r in ref_positive) / len(ref_positive)
                if ref_positive else 0.0
            )
            fp_zero = all(int(r.get(f"{version}_false_positive_voxels") or 0) == 0 for r in negative_rows)
            failures: list[str] = []
            if len(ref_positive) < 3:
                failures.append("insufficient_nonempty_reference_support")
            if len(available) != len(ref_positive):
                failures.append(f"{version}_missing")
            if not dscs or float(np.mean(dscs)) < 0.80:
                failures.append("mean_dice_below_0.80")
            if not dscs or min(dscs) < 0.60:
                failures.append("worst_case_dice_below_0.60")
            if recall < 0.80:
                failures.append("nonempty_recall_below_0.80")
            if not fp_zero:
                failures.append("false_positive_on_reference_empty_case")
            if not volumes or not low <= median(volumes) <= high:
                failures.append("median_volume_ratio_out_of_range")
            if version == "cascade" and any(r.get("cascade_outside_roi_voxels") not in {0} for r in available):
                failures.append("roi_leakage_or_missing_roi_audit")
            return {
                "version": version,
                "dscs": dscs,
                "volumes": volumes,
                "recall": recall,
                "fp_zero": fp_zero,
                "mean": float(np.mean(dscs)) if dscs else None,
                "worst": min(dscs) if dscs else None,
                "failures": failures,
            }

        summaries = {version: summarize(version) for version in ("raw", "containment", "cascade")}
        raw_summary = summaries["raw"]
        selected = raw_summary if not raw_summary["failures"] else None
        improvement_audit: dict[str, Any] = {}
        for version in ("containment", "cascade"):
            candidate = summaries[version]
            paired = [
                float(r[f"{version}_dice"]) - float(r["raw_dice"])
                for r in ref_positive
                if isinstance(r.get(f"{version}_dice"), (int, float))
                and isinstance(r.get("raw_dice"), (int, float))
            ]
            mean_delta = float(np.mean(paired)) if paired else None
            no_regression = bool(paired) and all(delta >= -0.02 for delta in paired)
            raw_fp = sum(int(r.get("raw_false_positive_voxels") or 0) for r in negative_rows)
            candidate_fp = sum(int(r.get(f"{version}_false_positive_voxels") or 0) for r in negative_rows)
            fp_improved = raw_fp > 0 and candidate_fp <= raw_fp * 0.5
            demonstrably_better = (
                mean_delta is not None
                and no_regression
                and (mean_delta >= 0.02 or (mean_delta >= -0.001 and fp_improved))
            )
            improvement_audit[version] = {
                "mean_dice_delta_vs_raw": mean_delta,
                "no_case_regression_over_0.02": no_regression,
                "false_positive_improved_at_least_50pct": fp_improved,
                "demonstrably_better": demonstrably_better,
            }
            if not candidate["failures"] and demonstrably_better:
                if selected is None or float(candidate["mean"] or 0.0) > float(selected["mean"] or 0.0):
                    selected = candidate

        if selected is None:
            diagnostic = max(
                summaries.values(),
                key=lambda item: float(item["mean"]) if item["mean"] is not None else -1.0,
            )
            reasons = list(diagnostic["failures"])
            if not reasons:
                reasons.append("no_eligible_non_regressing_variant")
            selected_version = ""
            metric_summary = diagnostic
        else:
            reasons = []
            selected_version = str(selected["version"])
            metric_summary = selected
        if organ in HARD_BLOCK:
            reasons.insert(0, "explicit_safety_block")
            selected_version = ""
        entry = taxonomy_entry(taxonomy, organ) or {}
        decisions.append({
            "organ": organ,
            "decision": "allow" if not reasons else "block",
            "round2_route": "student_competition" if not reasons else "teacher_or_fov_negative",
            "selected_version": selected_version if not reasons else "",
            "parents": list(rule.parents),
            "reference_nonempty_cases": len(ref_positive),
            "evaluated_cases": len(evaluated),
            "expert_gt_cases": sum(r.get("reference_kind") == "expert_gt" for r in evaluated),
            "selected_teacher_cases": sum(r.get("reference_kind") == "selected_teacher" for r in evaluated),
            "mean_dice": round(float(metric_summary["mean"]), 6) if metric_summary["mean"] is not None else None,
            "worst_case_dice": round(float(metric_summary["worst"]), 6) if metric_summary["worst"] is not None else None,
            "nonempty_recall": round(float(metric_summary["recall"]), 6),
            "median_volume_ratio": round(median(metric_summary["volumes"]), 6) if metric_summary["volumes"] else None,
            "reference_empty_false_positive_free": metric_summary["fp_zero"],
            "confidence": "unavailable_binary_backend",
            "hierarchy_role": entry.get("hierarchy_role"),
            "variant_metrics": {
                version: {
                    "mean_dice": summary["mean"],
                    "worst_case_dice": summary["worst"],
                    "failures": summary["failures"],
                }
                for version, summary in summaries.items()
            },
            "improvement_audit": improvement_audit,
            "reasons": reasons,
        })

    allowed = [d for d in decisions if d["decision"] == "allow"]
    for decision in allowed:
        organ = decision["organ"]
        source_root = {
            "raw": a.raw_root,
            "containment": a.containment_root,
            "cascade": a.cascade_root,
        }[decision["selected_version"]]
        for case in cases:
            src = source_root / case / f"{organ}.nii.gz"
            if not src.is_file():
                raise RuntimeError(f"allowlisted mask missing: {src}")
            dst = a.postprocessed_root / case / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    per_case_fields = sorted({k for row in per_case for k in row})
    with (a.output_dir / "student_ab_per_case_organ.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_case_fields)
        writer.writeheader()
        writer.writerows(per_case)
    decision_fields = [k for k in decisions[0] if k != "reasons"] + ["reasons"]
    with (a.output_dir / "student_round2_organ_allowlist.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_fields)
        writer.writeheader()
        for row in decisions:
            writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    report = {
        "schema_version": 2,
        "stage": "student_round2_organ_quality_gate",
        "status": "success" if len(decisions) == 373 else "failed",
        "metric_interpretation": "per-organ metrics use expert GT when available; selected-teacher metrics remain pseudo-label consistency",
        "reference_policy": "expert_gt_when_annotation_folder_has_exact_organ_else_selected_teacher",
        "confidence_availability": "unavailable_binary_backend",
        "routing_contract": {
            "target_organs": 373,
            "resolved_organs": len(decisions),
            "student_competition": len(allowed),
            "teacher_or_fov_negative": len(decisions) - len(allowed),
            "teacher_fallback_semantics": "Visible organs rejected by the student gate remain teacher-selected positive supervision for the next M-step.",
            "fov_negative_semantics": "Case-organ targets proven outside scan coverage use an aligned all-zero mask and remain trainable negative supervision.",
            "unresolved_visible_semantics": "Visible/partial targets without a reliable teacher are withheld with zero training weight; they are not false negative labels.",
        },
        "case_count": len(cases),
        "expected_case_count": a.expected_cases,
        "raw_root": str(a.raw_root.resolve()),
        "containment_root": str(a.containment_root.resolve()),
        "cascade_root": str(a.cascade_root.resolve()),
        "postprocessed_root": str(a.postprocessed_root.resolve()),
        "allowed_count": len(allowed),
        "blocked_count": len(decisions) - len(allowed),
        "organs": decisions,
    }
    (a.output_dir / "student_round2_organ_allowlist.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "organs"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
