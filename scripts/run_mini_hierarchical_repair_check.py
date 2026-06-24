#!/usr/bin/env python3
"""Mini hierarchical ROI repair check before full 373-organ repair.

Runs a small E-step repair on 1-3 cases and a focused organ subset. It writes to
an explicit output directory and never mutates the old Round1 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


DEFAULT_ORGANS = [
    "liver",
    "pancreas",
    "kidney",
    "kidney_left",
    "kidney_right",
    "liver_segment_1",
    "liver_segment_2",
    "liver_segment_3",
    "liver_segment_4",
    "liver_segment_5",
    "liver_segment_6",
    "liver_segment_7",
    "liver_segment_8",
    "pancreas_head",
    "pancreas_body",
    "pancreas_tail",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audits/mini_hierarchical_repair"))
    ap.add_argument("--old-round1-estep", default=str(ROOT / "outputs/round1/estep"))
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--models", default="vsmtrans,cads551,atlasnet,nnunet_private,totalsegmentator,vista3d")
    ap.add_argument("--critic-backend", default="stub", choices=["stub", "labelcritic"])
    ap.add_argument("--enable-shapekit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    return ap.parse_args()


def write_case_subset(case_list: Path, out_csv: Path, num_cases: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with case_list.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: v for k, v in row.items()})
            if len(rows) >= num_cases:
                break
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def preseed_major_cache(old_estep: Path) -> dict[str, Path]:
    cases_root = old_estep / "cases"
    if not cases_root.exists():
        return {}
    teachers: set[str] = set()
    for case_dir in cases_root.iterdir():
        if not case_dir.is_dir():
            continue
        raw = case_dir / "raw_predictions"
        for teacher_dir in raw.glob("*"):
            case_seg = teacher_dir / case_dir.name / "segmentations"
            if case_seg.exists() and any(case_seg.glob("*.nii.gz")):
                teachers.add(teacher_dir.name)
    return {
        teacher: cases_root / "{case_id}" / "raw_predictions" / teacher
        for teacher in sorted(teachers)
    }


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def summarize(out: Path) -> dict[str, Any]:
    estep = out / "estep"
    manifest_path = estep / "training_manifest.json"
    manifest = read_json(manifest_path, [])
    items = manifest if isinstance(manifest, list) else manifest.get("items", [])
    grade_counts = Counter(str(item.get("grade", "missing")) for item in items if isinstance(item, dict))
    focus = [
        item for item in items if isinstance(item, dict)
        and item.get("organ") in {"liver", "pancreas", "kidney", "kidney_left", "kidney_right"}
    ]

    child_scope_failures: list[dict[str, Any]] = []
    fresh_major_full_volume: list[dict[str, Any]] = []
    blocked = []
    fusion_rows = []
    for plan_path in (estep / "cases").glob("*/hierarchical_inference_plan.json"):
        plan = read_json(plan_path, {})
        fresh_major_full_volume.extend([
            {"case_id": plan.get("case_id"), **row}
            for row in plan.get("major_resolution", []) or []
            if row.get("inference_scope") == "major_full_volume"
        ])
        for result in plan.get("roi_tasks", []) or []:
            inference = result.get("inference", {}) or {}
            if inference.get("inference_scope") != "child_roi":
                child_scope_failures.append({"case_id": plan.get("case_id"), "scope": inference.get("inference_scope")})
        blocked.extend([{"case_id": plan.get("case_id"), **b} for b in plan.get("blocked", []) or []])
    for meta_path in (estep / "annotation_versions").glob("*/selection_metadata.json"):
        meta = read_json(meta_path, {})
        for row in meta.get("selection_rows", []) or []:
            if "fusion_consensus" in (row.get("candidate_models") or []):
                fusion_rows.append({
                    "case_id": meta.get("case_id"),
                    "organ": row.get("organ"),
                    "selected_model": row.get("selected_model"),
                    "candidate_models": row.get("candidate_models"),
                })

    status = "success"
    failures: list[str] = []
    if child_scope_failures:
        status = "failed"
        failures.append("child inference outside child_roi")
    if fresh_major_full_volume:
        status = "failed"
        failures.append("fresh full-volume major inference used despite repair cache policy")
    for item in focus:
        if item.get("organ") in {"liver", "pancreas", "kidney"} and item.get("grade") not in {"A", "B", None}:
            failures.append(f"{item.get('case_id')}/{item.get('organ')} grade={item.get('grade')}")
    if failures:
        status = "failed"

    summary = {
        "stage": "mini_hierarchical_repair_check",
        "status": status,
        "output_dir": str(out),
        "manifest": str(manifest_path),
        "num_manifest_items": len(items),
        "grade_counts": dict(grade_counts),
        "focus_whole_organ_items": focus,
        "child_scope_failures": child_scope_failures,
        "fresh_major_full_volume": fresh_major_full_volume,
        "blocked": blocked,
        "fusion_rows": fusion_rows,
        "failures": failures,
    }
    (out / "mini_repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset = out / "case_list_subset.csv"
    write_case_subset(Path(args.case_list).resolve(), subset, max(1, min(args.num_cases, 3)))
    organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()]
    models = [x.strip() for x in args.models.replace(";", ",").split(",") if x.strip()]
    preseeded = preseed_major_cache(Path(args.old_round1_estep).resolve())

    result = run_multimodel_annotation_loop(
        case_list=subset,
        output_folder=out / "estep",
        models=models,
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=args.enable_shapekit,
        enable_critic=True,
        critic_backend=args.critic_backend,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        device="cuda",
        resume=False,
        preseeded_model_dirs=preseeded or None,
        preseeded_parent_only=True,
        teacher_inference_mode="hierarchical_roi",
        roi_margin_mm=20.0,
        enable_fusion=True,
        fusion_method="weighted_vote",
        enable_auto_arbitration=True,
        candidate_mode="route_pruned_with_competition",
    )
    summary = summarize(out)
    summary["estep_status"] = result.get("status")
    print(json.dumps({k: summary.get(k) for k in ["status", "estep_status", "grade_counts", "failures"]}, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" and result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
