#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("case_id") and row.get("ct_path")]
    if not rows:
        raise SystemExit(f"No usable rows in case list: {path}")
    return rows


def write_single_case_csv(rows: list[dict[str, str]], output_csv: Path, index: int) -> dict[str, str]:
    if index < 0 or index >= len(rows):
        raise SystemExit(f"case-index {index} out of range for {len(rows)} rows")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = rows[index]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def load_target_organs(path: Path) -> list[str]:
    doc = read_json(path, {})
    organs = [str(x) for x in doc.get("target_organs", [])]
    if len(organs) != 373:
        raise SystemExit(f"Target config must contain 373 exact prompt targets: {path}")
    return organs


def env_for_run(output_root: Path, case_list: Path) -> None:
    os.environ["MEDAI_OUTPUT_ROOT"] = str(output_root)
    os.environ["MEDAI_CASE_LIST"] = str(case_list)


def cmd_teacher_case(args: argparse.Namespace) -> int:
    rows = read_case_rows(args.case_list)
    case = write_single_case_csv(rows, args.output_dir / "manifests" / f"case_{args.case_index:02d}.csv", args.case_index)
    organs = load_target_organs(args.target_config)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    result = run_multimodel_annotation_loop(
        case_list=args.output_dir / "manifests" / f"case_{args.case_index:02d}.csv",
        output_folder=args.output_dir,
        models=models,
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=True,
        enable_critic=False,
        critic_backend="stub",
        dry_run=False,
        timeout_sec=args.timeout_sec,
        device="cuda",
        resume=False,
        teacher_inference_mode="hierarchical_roi",
    )
    raw_root = args.output_dir / "cases" / case["case_id"] / "raw_predictions"
    per_model = {}
    failures = []
    for model in models:
        seg_dir = raw_root / model / case["case_id"] / "segmentations"
        masks = sorted(seg_dir.glob("*.nii.gz")) if seg_dir.exists() else []
        per_model[model] = {"segmentations_dir": str(seg_dir), "mask_count": len(masks)}
        if not masks:
            failures.append(f"missing_raw_masks:{model}")
    summary = {
        "stage": "teacher_case",
        "status": "success" if not failures and result.get("status") == "success" else "failed",
        "case_id": case["case_id"],
        "output_dir": str(args.output_dir),
        "raw_predictions_root": str(raw_root),
        "models": per_model,
        "loop_status": result.get("status"),
        "failures": failures,
    }
    write_json(args.output_dir / "teacher_case_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def cmd_teacher_merge_audit(args: argparse.Namespace) -> int:
    rows = read_case_rows(args.case_list)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    failures = []
    cases = []
    for row in rows:
        case_id = row["case_id"]
        per_model = {}
        for model in models:
            seg_dir = args.teacher_root / "cases" / case_id / "raw_predictions" / model / case_id / "segmentations"
            masks = sorted(seg_dir.glob("*.nii.gz")) if seg_dir.exists() else []
            per_model[model] = {"segmentations_dir": str(seg_dir), "mask_count": len(masks)}
            if not masks:
                failures.append(f"{case_id}:{model}:missing_raw_masks")
        cases.append({"case_id": case_id, "models": per_model})
    summary = {
        "stage": "teacher_merge_and_audit",
        "status": "success" if not failures else "failed",
        "teacher_root": str(args.teacher_root),
        "case_count": len(rows),
        "cases": cases,
        "failures": failures,
    }
    write_json(args.teacher_root / "teacher_merge_audit.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def _load_run_em_training(output_root: Path, case_list: Path):
    env_for_run(output_root, case_list)
    import scripts.run_em_training as em

    em.OUTPUT_ROOT = output_root
    em.CASE_LIST = case_list
    return em


def cmd_build_round_manifest(args: argparse.Namespace) -> int:
    case_list = Path(os.environ["MEDAI_CASE_LIST"]).resolve()
    output_root = args.output_root.resolve()
    em = _load_run_em_training(output_root, case_list)
    manifest_path = em.build_3d_prompt_student_dataset(args.round)
    summary = {
        "stage": "build_round_manifest",
        "status": "success" if manifest_path.exists() else "failed",
        "round": args.round,
        "manifest_path": str(manifest_path),
        "material_update_audit": str(output_root / f"round{args.round}" / "estep" / "material_update_audit.json"),
        "novelty_audit": str(output_root / f"round{args.round}" / "mstep" / "novelty_audit.json"),
    }
    write_json(output_root / f"round{args.round}" / "mstep" / "build_manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def cmd_student_mstep(args: argparse.Namespace) -> int:
    case_list = Path(os.environ["MEDAI_CASE_LIST"]).resolve()
    output_root = args.output_root.resolve()
    em = _load_run_em_training(output_root, case_list)
    manifest_path = output_root / f"round{args.round}" / "mstep" / "voxtell_prompt_student_manifest.json"
    result = em.run_prompt_student_mstep(args.round, manifest_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 2


def cmd_student_inference_case(args: argparse.Namespace) -> int:
    rows = read_case_rows(args.case_list)
    subset_csv = args.output_root / "manifests" / f"student_round{args.round}_case_{args.case_index:02d}.csv"
    write_single_case_csv(rows, subset_csv, args.case_index)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_student_infer_then_round2.py"),
        "--round",
        str(args.round),
        "--case-list",
        str(subset_csv),
        "--output-root",
        str(args.output_root),
        "--target-config",
        str(args.target_config),
        "--device",
        "cuda",
    ]
    proc = os.spawnvpe(os.P_WAIT, sys.executable, command, os.environ.copy())
    return int(proc)


def cmd_prepare_round2(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    selected_root = output_root / "round1" / "estep" / "annotation_versions"
    student_root = output_root / "round1" / "student_predictions_postprocessed"
    if not student_root.exists():
        student_root = output_root / "round1" / "student_predictions"
    failures = []
    if not selected_root.exists():
        failures.append("missing_round1_selected")
    if not student_root.exists():
        failures.append("missing_round1_student_predictions")
    summary = {
        "stage": "prepare_round2_em_student_vs_previous",
        "status": "success" if not failures else "failed",
        "selected_root": str(selected_root),
        "student_root": str(student_root),
        "candidate_mode": "em_student_vs_previous",
        "failures": failures,
        "teacher_rerun": False,
    }
    write_json(output_root / "round2" / "prepare_round2_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def cmd_round2_competition(args: argparse.Namespace) -> int:
    rows = read_case_rows(args.case_list)
    organs = load_target_organs(args.target_config)
    output_root = args.output_root.resolve()
    estep = output_root / "round2" / "estep"
    subset_csv = output_root / "manifests" / "round2_full_case_list.csv"
    with subset_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = run_multimodel_annotation_loop(
        case_list=subset_csv,
        output_folder=estep,
        models=[],
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=True,
        enable_critic=True,
        critic_backend="labelcritic",
        critic_base_url="http://127.0.0.1",
        critic_port=int(os.environ.get("LABELCRITIC_PORT", "8000")),
        dry_run=False,
        timeout_sec=3600,
        device="cuda",
        resume=False,
        teacher_inference_mode="hierarchical_roi",
        candidate_mode="em_student_vs_previous",
        reuse_preseeded_only=True,
        preseeded_model_dirs={
            "round_prev_selected": output_root / "round1" / "estep" / "annotation_versions",
            "student_prev": (output_root / "round1" / "student_predictions_postprocessed") if (output_root / "round1" / "student_predictions_postprocessed").exists() else (output_root / "round1" / "student_predictions"),
        },
    )
    summary = {
        "stage": "round2_competition",
        "status": "success" if result.get("status") == "success" else "failed",
        "estep_status": result.get("status"),
        "output_dir": str(estep),
        "round2_competition_audit": result.get("round2_competition_audit"),
    }
    write_json(output_root / "round2" / "round2_competition_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def cmd_round2_material_gate(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    case_list = Path(os.environ["MEDAI_CASE_LIST"]).resolve()
    em = _load_run_em_training(output_root, case_list)
    manifest_path = em.build_3d_prompt_student_dataset(2)
    manifest = read_json(manifest_path, {})
    material = manifest.get("material_update_audit") or read_json(output_root / "round2" / "estep" / "material_update_audit.json", {})
    decision = str(material.get("material_update_decision") or "")
    summary = {
        "stage": "round2_material_update_gate",
        "status": "success" if decision in {"material_update", "no_material_update"} else "failed",
        "fail_closed": True,
        "decision": decision,
        "manifest_path": str(manifest_path),
        "material_update_audit": material,
        "checkpoint_reuse_allowed": decision == "no_material_update",
    }
    write_json(output_root / "round2" / "round2_material_update_gate.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def cmd_final_audit(args: argparse.Namespace) -> int:
    rows = read_case_rows(args.case_list)
    output_root = args.output_root.resolve()
    required = [
        output_root / "round1" / "estep" / "training_manifest.json",
        output_root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json",
        output_root / "round1" / "mstep" / "voxtell_prompt_mstep_result.json",
        output_root / "round2" / "estep" / "training_manifest.json",
        output_root / "round2" / "mstep" / "voxtell_prompt_mstep_result.json",
        output_root / "round2" / "round2_material_update_gate.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    inference_cases = []
    for round_idx in (1, 2):
        root = output_root / f"round{round_idx}" / "student_predictions"
        inference_cases.append({"round": round_idx, "case_dirs": len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0})
    summary = {
        "stage": "final_train10_audit",
        "status": "success" if not missing and all(row["case_dirs"] == len(rows) for row in inference_cases) else "failed",
        "case_count": len(rows),
        "missing": missing,
        "inference_cases": inference_cases,
    }
    write_json(output_root / "final_train10_audit.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Formal Train10 Round1+Round2 pipeline helper CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("teacher-case")
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--target-config", type=Path, required=True)
    p.add_argument("--case-index", type=int, required=True)
    p.add_argument("--timeout-sec", type=int, default=3600)

    p = sub.add_parser("teacher-merge-audit")
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--models", required=True)

    p = sub.add_parser("build-round-manifest")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)

    p = sub.add_parser("student-mstep")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)

    p = sub.add_parser("student-inference-case")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target-config", type=Path, required=True)
    p.add_argument("--case-index", type=int, required=True)

    p = sub.add_parser("prepare-round2")
    p.add_argument("--output-root", type=Path, required=True)

    p = sub.add_parser("round2-competition")
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--target-config", type=Path, required=True)

    p = sub.add_parser("round2-material-gate")
    p.add_argument("--output-root", type=Path, required=True)

    p = sub.add_parser("final-audit")
    p.add_argument("--case-list", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "teacher-case":
        return cmd_teacher_case(args)
    if args.cmd == "teacher-merge-audit":
        return cmd_teacher_merge_audit(args)
    if args.cmd == "build-round-manifest":
        return cmd_build_round_manifest(args)
    if args.cmd == "student-mstep":
        return cmd_student_mstep(args)
    if args.cmd == "student-inference-case":
        return cmd_student_inference_case(args)
    if args.cmd == "prepare-round2":
        return cmd_prepare_round2(args)
    if args.cmd == "round2-competition":
        return cmd_round2_competition(args)
    if args.cmd == "round2-material-gate":
        return cmd_round2_material_gate(args)
    if args.cmd == "final-audit":
        return cmd_final_audit(args)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
