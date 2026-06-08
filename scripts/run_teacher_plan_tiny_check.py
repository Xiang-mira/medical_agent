#!/usr/bin/env python3
"""Tiny end-to-end check for the teacher-meeting pipeline.

This is a lightweight regression check, not a scientific experiment. It runs
real CT inputs through the project pipeline with the smoke-test `mock_seg`
teacher, ShapeKit enabled by default, VoxTell manifest/trainer dry-run, student
inference dry-run, and failure mining.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
from cli_anything.medai.core.voxtell_student import VoxTellStudent


DEFAULT_ORGANS = ["liver", "pancreas", "spleen", "kidney_left", "kidney_right"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a tiny real-CT pipeline check.")
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audit_21_models/teacher_plan_tiny_check"))
    ap.add_argument("--num-cases", type=int, default=2)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--enable-shapekit", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--critic-backend", default="stub", choices=["stub", "labelcritic"])
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--trainer-max-items", type=int, default=4)
    return ap.parse_args()


def write_subset_case_list(src: Path, dst: Path, num_cases: int) -> list[dict[str, str]]:
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("case_id") and row.get("ct_path")]
    subset = rows[:num_cases]
    if not subset:
        raise SystemExit(f"No cases found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subset[0].keys()))
        writer.writeheader()
        writer.writerows(subset)
    return subset


def run_cmd(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    return {
        "command": cmd,
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    case_subset = out / "case_list_subset.csv"
    cases = write_subset_case_list(Path(args.case_list).resolve(), case_subset, args.num_cases)

    estep_dir = out / "estep"
    estep = run_multimodel_annotation_loop(
        case_list=case_subset,
        output_folder=estep_dir,
        models=["mock_seg"],
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=args.enable_shapekit,
        enable_critic=True,
        critic_backend=args.critic_backend,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        resume=False,
    )

    voxtell_manifest = out / "voxtell_prompt_manifest.json"
    student = VoxTellStudent(
        model_dir=ROOT / "checkpoints/VoxTell/voxtell_v1.1",
        target_config=ROOT / "configs/student_3d_prompt_target_organs.json",
    )
    manifest = student.build_training_manifest(
        cases_root=estep_dir / "annotation_versions",
        output_manifest=voxtell_manifest,
        case_list=case_subset,
        require_images=True,
    )

    train = run_cmd([
        sys.executable,
        "scripts/train_voxtell_prompt_student.py",
        "--manifest", str(voxtell_manifest),
        "--model-dir", str(ROOT / "checkpoints/VoxTell/voxtell_v1.1"),
        "--text-encoding-model", str(ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"),
        "--output-dir", str(out / "voxtell_train_dryrun"),
        "--dry-run",
        "--max-items", str(args.trainer_max_items),
    ], ROOT)

    infer_prompts = ",".join(organs[:2])
    infer = run_cmd([
        sys.executable,
        "scripts/run_student_infer_then_round2.py",
        "--round", "1",
        "--case-list", str(case_subset),
        "--output-root", str(out / "student_infer"),
        "--dry-run",
        "--max-cases", str(len(cases)),
        "--prompts", infer_prompts,
        "--timeout-sec", "5",
    ], ROOT)

    failure = run_cmd([
        sys.executable,
        "scripts/mine_student_failure_cases.py",
        "--student-rounds", "1",
        "--output-root", str(out),
        "--case-list", str(case_subset),
        "--organs", infer_prompts,
        "--reference-root", str(estep_dir / "annotation_versions"),
        "--student-root-template", str(out / "student_infer/round{round}/student_predictions"),
        "--dice-threshold", "0.5",
    ], ROOT)

    manifest_rows = json.loads(voxtell_manifest.read_text(encoding="utf-8")).get("items", [])
    summary = {
        "stage": "teacher_plan_tiny_check",
        "status": "success" if estep.get("status") == "success" and train["return_code"] == 0 and infer["return_code"] == 0 and failure["return_code"] == 0 else "failed",
        "num_cases": len(cases),
        "organs": organs,
        "estep": {
            "status": estep.get("status"),
            "total_updated": estep.get("total_updated"),
            "total_labelcritic_decisions": estep.get("total_labelcritic_decisions"),
        },
        "manifest": {
            "num_items": manifest.get("num_items"),
            "num_cases": manifest.get("num_cases"),
            "num_items_missing_image": manifest.get("num_items_missing_image"),
            "all_items_have_source_metadata": all(bool(r.get("source_metadata_available")) for r in manifest_rows),
            "shapekit_statuses": sorted({str(r.get("shapekit_status")) for r in manifest_rows}),
        },
        "train_dryrun": train,
        "student_infer_dryrun": infer,
        "failure_mining": failure,
        "accuracy_warning": "This is a smoke/regression check. It does not measure true accuracy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiny_check_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(out),
        "estep_total_updated": summary["estep"]["total_updated"],
        "manifest_items": summary["manifest"]["num_items"],
        "all_items_have_source_metadata": summary["manifest"]["all_items_have_source_metadata"],
        "shapekit_statuses": summary["manifest"]["shapekit_statuses"],
    }, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
