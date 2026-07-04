#!/usr/bin/env python3
"""Pure cached geometric teacher-consensus reselect.

This script is deliberately offline: it reads existing teacher mask files from a
completed E-step cache and never calls model inference, ShapeKit, LabelCritic,
or the shared hierarchical runner.  Its purpose is to test the current
family-free geometric consensus rule on cached teacher masks and materialize a
full case×373 manifest plus the three approved split manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import (  # noqa: E402
    _CaseMaskCache,
    _geometric_teacher_consensus_selection,
)

DEFAULT_TEACHERS = [
    "cads551", "cads552", "cads553", "cads554", "cads555", "cads556",
    "cads557", "cads558", "cads559", "moose666", "moose888",
    "nnunet_private", "saros_nnunet", "atm", "airrc", "lvp", "daps",
    "epai_20250421", "vsmtrans", "vista3d", "unest", "totalsegmentator",
]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_cases(case_list: Path) -> list[dict[str, str]]:
    with case_list.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_targets(path: Path) -> list[str]:
    doc = read_json(path, {}) or {}
    targets = [str(x).strip() for x in doc.get("target_organs", []) if str(x).strip()]
    return list(dict.fromkeys(targets))


def prompt_map(path: Path) -> dict[str, str]:
    doc = read_json(path, {}) or {}
    out: dict[str, str] = {}
    for row in doc.get("mappings", []) or []:
        if isinstance(row, dict) and row.get("project_class"):
            out[str(row["project_class"])] = str(row.get("canonical_prompt") or row["project_class"]).strip()
    return out


def source_rows_by_case_organ(source_estep: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for meta_path in sorted((source_estep / "annotation_versions").glob("*/selection_metadata.json")):
        doc = read_json(meta_path, {}) or {}
        case_id = str(doc.get("case_id") or meta_path.parent.name)
        for row in doc.get("selection_rows", []) or []:
            if isinstance(row, dict) and row.get("organ"):
                rows[(case_id, str(row["organ"]))] = dict(row)
    return rows


def safe_mask_nonempty(path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj) > 0
        voxels = int(arr.sum())
        return voxels > 0, {
            "status": "ok",
            "voxels": voxels,
            "shape": [int(x) for x in arr.shape[:3]],
        }
    except Exception as exc:
        return False, {"status": "unreadable", "reason": str(exc)}


def collect_candidates(
    source_estep: Path,
    case_id: str,
    organ: str,
    teachers: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for teacher in teachers:
        mask = source_estep / "cases" / case_id / "hierarchical_predictions" / teacher / "segmentations" / f"{organ}.nii.gz"
        if not mask.exists():
            skipped.append({"model": teacher, "reason": "missing_cached_mask"})
            continue
        nonempty, stats = safe_mask_nonempty(mask)
        if not nonempty:
            skipped.append({"model": teacher, "reason": "zero_or_unreadable_mask", "mask": str(mask), "stats": stats})
            continue
        candidates.append({
            "model": teacher,
            "model_key": teacher,
            "prediction": str(mask.resolve()),
            "candidate_qc_status": "pass",
            "identity_status": "valid",
            "original_teacher_mask": True,
            "cached_only": True,
            "mask_stats": stats,
        })
    return candidates, skipped


def reliable_negative_from_source(
    source: dict[str, Any] | None,
    *,
    case_id: str,
    organ: str,
    ct_path: Path,
    source_estep: Path,
    prompt: str,
) -> dict[str, Any] | None:
    if not source:
        return None
    if not (
        source.get("target_type") == "negative_absent"
        and source.get("selection_method") == "negative_absent"
        and source.get("fov_status") == "out_of_fov"
        and source.get("absence_confidence") == "high"
        and source.get("zero_mask_role") == "negative_absent_target_mask"
    ):
        return None
    zero = source.get("final_mask") or source.get("selected_prediction") or source.get("mask_path")
    zero_path = Path(str(zero)) if zero else source_estep / "annotation_versions" / case_id / "updated" / "negative_targets" / "zero_mask.nii.gz"
    if not zero_path.is_absolute():
        zero_path = ROOT / zero_path
    return {
        **source,
        "case_id": case_id,
        "organ": organ,
        "ct_path": str(ct_path),
        "image": str(ct_path),
        "prompt": source.get("prompt") or prompt,
        "target_type": "negative_absent",
        "record_type": "negative_absent",
        "selection_method": "negative_absent",
        "selection_status": "selected",
        "supervision_type": "negative",
        "training_weight": float(source.get("training_weight") or 0.1),
        "distillation_eligible": float(source.get("training_weight") or 0.1) > 0,
        "should_enter_student_training": float(source.get("training_weight") or 0.1) > 0,
        "mask": str(zero_path),
        "mask_path": str(zero_path),
        "final_mask": str(zero_path),
        "selected_prediction": str(zero_path),
        "zero_mask_role": "negative_absent_target_mask",
        "absence_confidence": "high",
        "fov_status": "out_of_fov",
        "grade": source.get("grade") or "A",
        "scoring_schema_version": "autolabel_core_v3_absent_negative",
        "negative_source": source.get("negative_source") or "case_373_expected_absent",
    }


def unresolved_row(
    source: dict[str, Any] | None,
    *,
    case_id: str,
    organ: str,
    ct_path: Path,
    prompt: str,
    reason: str,
) -> dict[str, Any]:
    target_type = "unresolved_visible"
    if source and source.get("target_type") in {"partial_fov", "rejected", "unresolved_visible"}:
        target_type = str(source.get("target_type"))
    return {
        **(source or {}),
        "case_id": case_id,
        "organ": organ,
        "ct_path": str(ct_path),
        "image": str(ct_path),
        "prompt": (source or {}).get("prompt") or prompt,
        "target_type": target_type,
        "record_type": target_type,
        "selection_method": "pure_cached_geometric_consensus",
        "selection_status": "withheld",
        "training_weight": 0.0,
        "distillation_eligible": False,
        "should_enter_student_training": False,
        "distillation_exclusion_reason": reason,
        "zero_mask_role": "io_placeholder_not_training_target",
        "pure_cached_reselect_status": "withheld",
        "pure_cached_reselect_reason": reason,
        "accuracy_claim_allowed": False,
    }


def positive_row(
    *,
    case_id: str,
    organ: str,
    ct_path: Path,
    prompt: str,
    selected: dict[str, Any],
    evidence: dict[str, Any],
    candidate_count: int,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    mask = Path(str(selected["prediction"])).resolve()
    return {
        "case_id": case_id,
        "organ": organ,
        "ct_path": str(ct_path),
        "image": str(ct_path),
        "prompt": prompt,
        "target_type": "positive_hard",
        "record_type": "positive_hard",
        "supervision_type": "positive",
        "selection_method": "geometric_teacher_consensus",
        "legacy_selection_method": "pure_cached_geometric_teacher_consensus",
        "selection_status": "selected",
        "selected_model": selected.get("model"),
        "selected_prediction": str(mask),
        "mask": str(mask),
        "mask_path": str(mask),
        "final_mask": str(mask),
        "winner_is_original_teacher": True,
        "training_weight": 1.0,
        "distillation_eligible": True,
        "should_enter_student_training": True,
        "grade": "A",
        "scoring_schema_version": "autolabel_core_v2",
        "ground_truth_status": "geometric_consensus_pseudo_label_not_expert_accuracy",
        "accuracy_claim_allowed": False,
        "family_role": "audit_only_not_used_for_selection_or_training_gate",
        "pure_cached_reselect": True,
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "shapekit_rerun": False,
        "candidate_count": candidate_count,
        "skipped_cached_candidates": skipped,
        **evidence,
    }


def single_teacher_ablation_row(
    source: dict[str, Any] | None,
    *,
    case_id: str,
    organ: str,
    ct_path: Path,
    prompt: str,
    candidate: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Strict low-weight single-teacher ablation candidate.

    The full manifest keeps this zero-weight/provisional.  The approved splitter
    is responsible for copying it only into single_teacher_ablation_manifest and
    setting training_weight=0.1.  Core consensus remains unchanged.
    """
    source = source or {}
    if source.get("target_type") in {"partial_fov", "rejected", "negative_absent"}:
        return None
    if source.get("fov_status") not in {"fully_visible", None, ""}:
        return None
    flags = set(source.get("review_flags") or []) | set(source.get("quality_flags") or []) | set(source.get("selected_candidate_qc_flags") or [])
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
        "auto_grade_reject",
    }
    if flags & hard_flags:
        return None
    mask = Path(str(candidate["prediction"])).resolve()
    return {
        **source,
        "case_id": case_id,
        "organ": organ,
        "ct_path": str(ct_path),
        "image": str(ct_path),
        "prompt": source.get("prompt") or prompt,
        "target_type": "positive_hard",
        "record_type": "positive_hard",
        "supervision_type": "positive",
        "selection_method": "single_teacher_provisional",
        "legacy_selection_method": "pure_cached_single_teacher_ablation",
        "selection_status": "provisional",
        "selected_model": candidate.get("model"),
        "selected_prediction": str(mask),
        "mask": str(mask),
        "mask_path": str(mask),
        "final_mask": str(mask),
        "winner_is_original_teacher": True,
        "fov_status": "fully_visible",
        "identity_status": "valid",
        "selected_candidate_qc_status": "pass",
        "training_weight": 0.0,
        "distillation_eligible": False,
        "should_enter_student_training": False,
        "distillation_exclusion_reason": "single_teacher_ablation_only_zero_weight_in_full_manifest",
        "grade": "A",
        "scoring_schema_version": "autolabel_core_v2",
        "ground_truth_status": "single_teacher_ablation_pseudo_label_not_expert_accuracy",
        "accuracy_claim_allowed": False,
        "single_teacher_ablation_only": True,
        "pure_cached_reselect": True,
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "shapekit_rerun": False,
        "candidate_count": 1,
        "skipped_cached_candidates": skipped,
        "quality_flags": ["single_teacher_provisional"],
        "review_flags": ["single_teacher_no_pairwise_comparison"],
    }


def run_split_and_audit(full_manifest: Path, output_dir: Path) -> dict[str, Any]:
    audit_path = output_dir / "full_case_373_manifest_audit.json"
    split_dir = output_dir / "labelcritic_repair_split_manifests"
    audit_proc = subprocess.run([
        sys.executable, str(ROOT / "scripts/audit_full_case373_manifest.py"),
        "--manifest", str(full_manifest),
        "--output", str(audit_path),
    ], cwd=str(ROOT), check=False)
    split_proc = subprocess.run([
        sys.executable, str(ROOT / "scripts/split_labelcritic_repair_manifests.py"),
        "--input", str(full_manifest),
        "--output-dir", str(split_dir),
    ], cwd=str(ROOT), check=False)
    return {
        "audit_returncode": audit_proc.returncode,
        "split_returncode": split_proc.returncode,
        "audit": read_json(audit_path, {}),
        "split_summary": read_json(split_dir / "manifest_split_summary.json", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-estep", required=True, type=Path)
    ap.add_argument("--case-list", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--prompt-map", type=Path, default=ROOT / "configs/voxtell_official_prompt_map.json")
    ap.add_argument("--teachers", default=",".join(DEFAULT_TEACHERS))
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    source_estep = args.source_estep.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.target_config)
    prompts = prompt_map(args.prompt_map)
    teachers = [x.strip() for x in args.teachers.replace(";", ",").split(",") if x.strip()]
    cases = read_cases(args.case_list.resolve())
    source_rows = source_rows_by_case_organ(source_estep)
    cache = _CaseMaskCache(max_arrays=128)

    full_items: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    candidate_hist: Counter[int] = Counter()
    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        ct_path = Path(str(case.get("ct_path") or ""))
        if not ct_path.is_absolute():
            ct_path = ROOT / ct_path
        case_positive = 0
        case_negative = 0
        case_status_counts: Counter[str] = Counter()
        for organ in targets:
            prompt = prompts.get(organ) or organ.replace("_", " ")
            source = source_rows.get((case_id, organ))
            candidates, skipped = collect_candidates(source_estep, case_id, organ, teachers)
            candidate_hist[len(candidates)] += 1
            selected = None
            evidence: dict[str, Any] = {}
            if len(candidates) >= 2:
                selected, evidence = _geometric_teacher_consensus_selection(
                    candidates,
                    near_identical_dice=args.threshold,
                    mask_cache=cache,
                )
            else:
                evidence = {
                    "primary_selector": "complete_link_geometric_teacher_consensus",
                    "geometric_consensus_threshold": args.threshold,
                    "geometric_consensus_status": "insufficient_teacher_models",
                    "geometric_consensus_policy": "family_free_original_teacher_complete_link",
                    "family_role": "audit_only_not_used_for_selection_or_training_gate",
                }
            if selected is not None:
                row = positive_row(
                    case_id=case_id,
                    organ=organ,
                    ct_path=ct_path,
                    prompt=prompt,
                    selected=selected,
                    evidence=evidence,
                    candidate_count=len(candidates),
                    skipped=skipped[:20],
                )
                case_positive += 1
                status_counts["geometric_teacher_consensus"] += 1
                case_status_counts["geometric_teacher_consensus"] += 1
            elif len(candidates) == 1:
                single = single_teacher_ablation_row(
                    source,
                    case_id=case_id,
                    organ=organ,
                    ct_path=ct_path,
                    prompt=prompt,
                    candidate=candidates[0],
                    skipped=skipped[:20],
                )
                if single is not None:
                    row = single
                    status_counts["single_teacher_provisional"] += 1
                    case_status_counts["single_teacher_provisional"] += 1
                else:
                    neg = reliable_negative_from_source(
                        source,
                        case_id=case_id,
                        organ=organ,
                        ct_path=ct_path,
                        source_estep=source_estep,
                        prompt=prompt,
                    )
                    if neg is not None:
                        row = neg
                        case_negative += 1
                        status_counts["negative_absent"] += 1
                        case_status_counts["negative_absent"] += 1
                    else:
                        reason = str(evidence.get("geometric_consensus_status") or "single_teacher_not_ablation_eligible")
                        row = unresolved_row(
                            source,
                            case_id=case_id,
                            organ=organ,
                            ct_path=ct_path,
                            prompt=prompt,
                            reason=reason,
                        )
                        row.update({
                            "candidate_count": len(candidates),
                            "skipped_cached_candidates": skipped[:20],
                            **evidence,
                        })
                        status_counts[row["target_type"]] += 1
                        case_status_counts[row["target_type"]] += 1
            else:
                neg = reliable_negative_from_source(
                    source,
                    case_id=case_id,
                    organ=organ,
                    ct_path=ct_path,
                    source_estep=source_estep,
                    prompt=prompt,
                )
                if neg is not None:
                    row = neg
                    case_negative += 1
                    status_counts["negative_absent"] += 1
                    case_status_counts["negative_absent"] += 1
                else:
                    reason = str(evidence.get("geometric_consensus_status") or "no_cached_geometric_consensus")
                    row = unresolved_row(
                        source,
                        case_id=case_id,
                        organ=organ,
                        ct_path=ct_path,
                        prompt=prompt,
                        reason=reason,
                    )
                    row.update({
                        "candidate_count": len(candidates),
                        "skipped_cached_candidates": skipped[:20],
                        **evidence,
                    })
                    status_counts[row["target_type"]] += 1
                    case_status_counts[row["target_type"]] += 1
            full_items.append(row)
        case_summaries.append({
            "case_id": case_id,
            "geometric_consensus_positive_items": case_positive,
            "negative_absent_items": case_negative,
            "status_counts": dict(case_status_counts),
        })

    full_manifest = {
        "stage": "pure_cached_geometric_reselect_full_case_373_manifest",
        "status": "success" if len(full_items) == len(cases) * len(targets) else "failed",
        "source_estep": str(source_estep),
        "case_list": str(args.case_list.resolve()),
        "target_config": str(args.target_config.resolve()),
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "shapekit_rerun": False,
        "num_cases": len(cases),
        "num_classes": len(targets),
        "expected_targets": len(cases) * len(targets),
        "actual_targets": len(full_items),
        "geometric_consensus_threshold": args.threshold,
        "teachers_requested": teachers,
        "selection_method": "geometric_teacher_consensus",
        "accuracy_claim_allowed": False,
        "target_type_counts": dict(Counter(str(r.get("target_type")) for r in full_items)),
        "selection_status_counts": dict(status_counts),
        "candidate_count_histogram": {str(k): v for k, v in sorted(candidate_hist.items())},
        "case_summaries": case_summaries,
        "items": full_items,
    }
    full_path = out / "full_case_373_manifest.json"
    write_json(full_path, full_manifest)
    post = run_split_and_audit(full_path, out)
    summary = {
        "stage": "pure_cached_geometric_reselect",
        "status": "success" if full_manifest["status"] == "success" and post["audit_returncode"] == 0 and post["split_returncode"] == 0 else "failed",
        "output_dir": str(out),
        "full_manifest": str(full_path),
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "shapekit_rerun": False,
        "num_cases": len(cases),
        "num_classes": len(targets),
        "expected_targets": full_manifest["expected_targets"],
        "actual_targets": full_manifest["actual_targets"],
        "geometric_consensus_positive_items": status_counts.get("geometric_teacher_consensus", 0),
        "geometric_consensus_positive_organs": len({str(r.get("organ")) for r in full_items if r.get("selection_method") == "geometric_teacher_consensus"}),
        "geometric_consensus_positive_cases": len({str(r.get("case_id")) for r in full_items if r.get("selection_method") == "geometric_teacher_consensus"}),
        "negative_absent_items": status_counts.get("negative_absent", 0),
        "target_type_counts": full_manifest["target_type_counts"],
        "candidate_count_histogram": full_manifest["candidate_count_histogram"],
        "case_summaries": case_summaries,
        "postprocess": post,
    }
    write_json(out / "pure_cached_geometric_reselect_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
