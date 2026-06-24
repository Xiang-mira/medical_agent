#!/usr/bin/env python3
"""Run a small real-teacher E-step check.

This validates that at least one real teacher wrapper can run on a real CT,
participate in candidate selection, pass through ShapeKit, and produce an
auditable manifest. The default uses ePAI plus the synthetic mock teacher so the
multi-candidate/LabelCritic-record path is exercised without running the full
teacher pool.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Small real-teacher pipeline check.")
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audit_21_models/real_teacher_subset_check"))
    ap.add_argument("--num-cases", type=int, default=1)
    ap.add_argument("--models", default="epai_20250421,mock_seg")
    ap.add_argument("--organs", default="liver,pancreas,spleen,kidney_left,kidney_right")
    ap.add_argument("--critic-backend", default="stub", choices=["stub", "labelcritic"])
    ap.add_argument("--enable-shapekit", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--timeout-sec", type=int, default=420)
    return ap.parse_args()


def write_subset_case_list(src: Path, dst: Path, num_cases: int) -> None:
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("case_id") and row.get("ct_path")]
    rows = rows[:num_cases]
    if not rows:
        raise SystemExit(f"No cases found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    case_subset = out / "case_list_subset.csv"
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    write_subset_case_list(Path(args.case_list).resolve(), case_subset, args.num_cases)

    estep_dir = out / "estep"
    result = run_multimodel_annotation_loop(
        case_list=case_subset,
        output_folder=estep_dir,
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
        teacher_inference_mode="hierarchical_roi",
    )

    manifest_path = estep_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    vlm_path = estep_dir / "vlm_decisions.jsonl"
    vlm_count = len(vlm_path.read_text(encoding="utf-8").splitlines()) if vlm_path.exists() else 0
    selected_models = sorted({str(r.get("selected_model")) for r in manifest})
    candidate_model_sets = sorted({tuple(r.get("candidate_models", [])) for r in manifest})
    shapekit_statuses = sorted({str(r.get("shapekit_status")) for r in manifest})
    summary = {
        "stage": "real_teacher_subset_check",
        "status": "success" if result.get("status") == "success" and len(manifest) > 0 else "failed",
        "models": models,
        "organs": organs,
        "estep": {
            "status": result.get("status"),
            "total_updated": result.get("total_updated"),
            "total_labelcritic_decisions": result.get("total_labelcritic_decisions"),
        },
        "manifest_items": len(manifest),
        "selected_models": selected_models,
        "candidate_model_sets": [list(x) for x in candidate_model_sets],
        "shapekit_statuses": shapekit_statuses,
        "vlm_decision_lines": vlm_count,
        "all_items_have_source_metadata": all(bool(r.get("source_metadata_available")) for r in manifest),
        "accuracy_warning": "This check validates pipeline wiring, not true segmentation accuracy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_teacher_subset_check_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
