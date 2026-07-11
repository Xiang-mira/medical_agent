#!/usr/bin/env python3
"""Repair reused Round1 373-target metadata with legal absent negatives.

This does not rerun teacher inference.  It revisits reused case-level
``selection_metadata.json`` files, applies the current FOV/absence policy, and
turns out-of-FOV unresolved targets into trainable all-zero ``negative_absent``
rows while leaving expected-present missing targets withheld.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import (  # noqa: E402
    _load_case_presence_context,
    _materialize_case_373_targets,
)
from cli_anything.medai.core.continual_learning import TRAINING_CONTRACT_VERSION  # noqa: E402
from cli_anything.medai.core.voxtell_student import VoxTellStudent  # noqa: E402


DEFAULT_TARGET_CONFIG = ROOT / "configs" / "student_3d_prompt_target_organs.json"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_case_list(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id") or "").strip()
            ct_path = str(row.get("ct_path") or "").strip()
            if case_id and ct_path:
                out[case_id] = ct_path
    return out


def load_targets(path: Path) -> list[str]:
    doc = read_json(path, {})
    return [
        str(item).strip()
        for item in doc.get("target_organs", [])
        if str(item).strip()
    ]


def positive_landmark_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = {
        str(row.get("organ") or "")
        for row in rows
        if str(row.get("supervision_type") or "").lower() == "positive"
        and str(row.get("target_type") or "").lower() in {"hard", "soft", "positive_hard", "positive_soft"}
        and float(row.get("training_weight") or 0.0) > 0.0
    }
    context: dict[str, Any] = {}
    evidence: list[str] = []
    if positives & {"lung_left", "lung_right", "heart", "pericardium"}:
        context["has_partial_thorax_coverage"] = True
        evidence.append("reused_positive_landmark_partial_only:thorax")
    if positives & {"bladder", "prostate", "uterus", "femur_left", "femur_right"}:
        context["has_pelvis_coverage"] = True
        evidence.append("reused_positive_landmark:pelvis")
    if positives & {"brain", "skull", "eyeball_left", "eyeball_right"}:
        context["has_head_coverage"] = True
        evidence.append("reused_positive_landmark:head_neck")
    if positives & {"humerus_left", "humerus_right", "tibia_left", "tibia_right"}:
        context["has_extremity_coverage"] = True
        evidence.append("reused_positive_landmark:extremity")
    if evidence:
        context["coverage_evidence"] = evidence
        context["has_region_evidence"] = True
    return context


def merge_presence_context(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    evidence = list(out.get("coverage_evidence") or [])
    evidence.extend(item for item in extra.get("coverage_evidence", []) if item not in evidence)
    for key, value in extra.items():
        if key == "coverage_evidence":
            continue
        if isinstance(value, bool):
            out[key] = bool(out.get(key)) or value
        else:
            out[key] = value
    out["coverage_evidence"] = evidence
    out["has_region_evidence"] = bool(out.get("has_region_evidence")) or any(
        bool(out.get(key))
        for key in (
            "has_abdomen_coverage", "has_pelvis_coverage",
            "has_thorax_coverage", "has_partial_thorax_coverage",
            "has_head_coverage", "has_extremity_coverage",
        )
    )
    return out


def rebuild_full_case_manifest(estep_dir: Path, targets: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted((estep_dir / "annotation_versions").glob("*/selection_metadata.json")):
        doc = read_json(meta_path, {})
        case_rows = doc.get("selection_rows") or []
        if isinstance(case_rows, list):
            rows.extend(row for row in case_rows if isinstance(row, dict))
    case_counts = Counter(str(row.get("case_id") or "") for row in rows)
    target_type_counts = Counter(str(row.get("target_type") or "") for row in rows)
    status = "passed" if case_counts and all(count == len(targets) for count in case_counts.values()) else "failed"
    payload = {
        "stage": "full_case_373_estep_manifest",
        "status": status,
        "summary": {
            "stage": "full_case_373_estep_manifest",
            "status": status,
            "num_cases": len(case_counts),
            "num_classes": len(targets),
            "expected_targets": len(case_counts) * len(targets),
            "manifest_targets": len(rows),
            "target_type_counts": dict(target_type_counts),
            "absent_negative_targets": int(target_type_counts.get("negative_absent", 0)),
            "withheld_uncertain_targets": int(target_type_counts.get("withheld_uncertain", 0)),
            "repaired_reused_absent_negative_policy": True,
        },
        "items": rows,
    }
    write_json(estep_dir / "full_case_373_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--case-list", type=Path, default=ROOT / "data_manifest" / "case_list_25_tumor.csv")
    ap.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    ap.add_argument("--write-mstep-manifest", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    estep_dir = output_root / "round1" / "estep"
    annotation_root = estep_dir / "annotation_versions"
    targets = load_targets(args.target_config.resolve())
    case_ct = read_case_list(args.case_list.resolve())
    repaired_cases: list[dict[str, Any]] = []

    for meta_path in sorted(annotation_root.glob("*/selection_metadata.json")):
        case_id = meta_path.parent.name
        doc = read_json(meta_path, {})
        if str(doc.get("reuse_role") or "") != "reused_round1_selected_pseudo_label":
            continue
        ct_raw = case_ct.get(case_id)
        if not ct_raw:
            repaired_cases.append({"case_id": case_id, "status": "failed", "reason": "ct_path_missing"})
            continue
        ct = Path(ct_raw).resolve()
        rows = [dict(row) for row in doc.get("selection_rows", []) if isinstance(row, dict)]
        selected = [dict(row) for row in doc.get("selected_organs", []) if isinstance(row, dict)]
        # Reused metadata may have been produced by an older FOV policy.  Clear
        # cached FOV labels so current anatomy/coverage logic is authoritative.
        for row in [*rows, *selected]:
            row.pop("fov_status", None)
            row.pop("expected_presence", None)
        base_context = _load_case_presence_context(
            {"case_id": case_id, "ct_path": str(ct), "dataset": "PanTS"},
            ct,
            case_id,
            meta_path.parent,
        )
        presence_context = merge_presence_context(base_context, positive_landmark_context(rows))
        before = Counter(str(row.get("target_type") or "") for row in rows)
        summary = _materialize_case_373_targets(
            case_id=case_id,
            ct=ct,
            organs=targets,
            case_updated=meta_path.parent / "updated",
            selection_rows=rows,
            selected_metadata=selected,
            presence_context=presence_context,
        )
        after = Counter(str(row.get("target_type") or "") for row in rows)
        doc.update({
            "ct_path": str(ct),
            "case_presence_context": presence_context,
            "selection_rows": rows,
            "selected_organs": selected,
            "target_count": len(targets),
            "reused_absent_negative_repair": {
                "status": "passed",
                "before_target_type_counts": dict(before),
                "after_target_type_counts": dict(after),
                "materialize_summary": summary,
            },
            "positive_mask_count": sum(
                1 for row in rows
                if str(row.get("supervision_type") or "").lower() == "positive"
                and float(row.get("training_weight") or 0.0) > 0.0
            ),
            "negative_absent_count": int(after.get("negative_absent", 0)),
            "withheld_uncertain_count": int(after.get("withheld_uncertain", 0)),
        })
        write_json(meta_path, doc)
        repaired_cases.append({
            "case_id": case_id,
            "status": "passed",
            "before_target_type_counts": dict(before),
            "after_target_type_counts": dict(after),
            "summary": summary,
        })

    full_manifest = rebuild_full_case_manifest(estep_dir, targets)
    mstep_manifest: dict[str, Any] | None = None
    if args.write_mstep_manifest:
        student = VoxTellStudent(
            model_dir=ROOT / "checkpoints" / "VoxTell" / "voxtell_v1.1",
            target_config=args.target_config.resolve(),
            device="cuda",
        )
        mstep_manifest = student.build_training_manifest(
            cases_root=annotation_root,
            output_manifest=output_root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json",
            case_list=args.case_list.resolve(),
            require_images=True,
        )
        mstep_manifest["training_contract_version"] = TRAINING_CONTRACT_VERSION
        mstep_manifest["run_spec_path"] = str((output_root / "round1" / "run_spec.json").resolve())
        write_json(
            output_root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json",
            mstep_manifest,
        )
    payload = {
        "stage": "repair_reused_round1_373_absent_negatives",
        "status": "passed" if repaired_cases else "no_reused_cases_found",
        "output_root": str(output_root),
        "case_list": str(args.case_list.resolve()),
        "target_config": str(args.target_config.resolve()),
        "num_repaired_cases": len(repaired_cases),
        "repaired_cases": repaired_cases,
        "full_case_373_manifest_status": full_manifest.get("status"),
        "mstep_manifest": {
            "path": str(output_root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json"),
            "num_items": mstep_manifest.get("num_items") if mstep_manifest else None,
            "num_trainable_positive_items": mstep_manifest.get("num_trainable_positive_items") if mstep_manifest else None,
            "num_negative_items": mstep_manifest.get("num_negative_items") if mstep_manifest else None,
        } if args.write_mstep_manifest else None,
    }
    write_json(estep_dir / "reused_absent_negative_repair_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if repaired_cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
