#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
from cli_anything.medai.core.voxtell_student import VoxTellStudent


def write_subset_case_list(src: Path, dst: Path, num_cases: int) -> list[dict[str, str]]:
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("case_id") and r.get("ct_path") and Path(r["ct_path"]).exists()]
    rows = rows[:num_cases]
    if not rows:
        raise SystemExit(f"No usable cases in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_cmd(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec)
    return {
        "command": cmd,
        "return_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def normalize_manifest_for_smoke(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    items = doc.get("items", [])
    for item in items:
        item.setdefault("prompt_source", "canonical_template")
        item.setdefault("validation_status", "passed")
        item["scoring_schema_version"] = "autolabel_core_v2"
        item["grade"] = "A"
        item["evidence_confidence"] = 0.99
        item["training_weight"] = 1.0
        item["target_type"] = "hard"
        item["distillation_eligible"] = True
        item["training_gate_decision"] = "include_hard_A_smoke_override"
        item["human_review_status"] = "accepted"
        item["selected_provider"] = item.get("selected_provider") or item.get("selected_model") or item.get("source_model") or "mock_seg"
        item["selection_reason"] = item.get("selection_reason") or item.get("selection_method") or "round_smoke_selected_mock"
        item["smoke_override_applied"] = True
        item["smoke_override_reason"] = "mock_seg E-step does not produce full AutoLabelCore v2 scores; override is for plumbing smoke only"
    doc["items"] = items
    doc["num_distillation_eligible_items"] = sum(1 for i in items if i.get("distillation_eligible") and float(i.get("training_weight") or 0) > 0)
    doc["smoke_override_applied"] = True
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def mask_summary(path: Path, ct: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "shape_ok": False, "voxels": None}
    mask_img = nib.load(str(path))
    ct_img = nib.load(str(ct))
    arr = np.asanyarray(mask_img.dataobj) > 0
    return {"exists": True, "shape_ok": mask_img.shape[:3] == ct_img.shape[:3], "voxels": int(arr.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description="Small Round0 -> M-step -> Student prediction smoke.")
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/smoke_voxtell/round0_mstep_prediction"))
    ap.add_argument("--num-cases", type=int, default=1)
    ap.add_argument("--organs", default="liver,spleen")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    subset_csv = out / "case_list_subset.csv"
    cases = write_subset_case_list(Path(args.case_list).resolve(), subset_csv, args.num_cases)

    estep_dir = out / "round0_estep"
    estep = run_multimodel_annotation_loop(
        case_list=subset_csv,
        output_folder=estep_dir,
        models=["mock_seg"],
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=False,
        enable_critic=True,
        critic_backend="stub",
        dry_run=False,
        timeout_sec=180,
        resume=False,
        teacher_inference_mode="full_volume",
        candidate_mode="formal_full_legacy",
    )

    manifest_path = out / "round0_prompt_student_manifest.json"
    student = VoxTellStudent(
        model_dir=ROOT / "checkpoints/VoxTell/voxtell_v1.1",
        target_config=ROOT / "configs/student_3d_prompt_target_organs.json",
        device=args.device,
    )
    manifest = student.build_training_manifest(
        cases_root=estep_dir / "annotation_versions",
        output_manifest=manifest_path,
        case_list=subset_csv,
        require_images=True,
    )
    manifest = normalize_manifest_for_smoke(manifest_path)

    train_out = out / "round0_mstep"
    train = run_cmd([
        sys.executable,
        "scripts/train_voxtell_prompt_student.py",
        "--manifest", str(manifest_path),
        "--model-dir", str(ROOT / "checkpoints/VoxTell/voxtell_v1.1"),
        "--text-encoding-model", str(ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"),
        "--output-dir", str(train_out),
        "--epochs", "1",
        "--max-steps", str(args.max_steps),
        "--max-items", str(max(1, len(organs))),
        "--freeze-encoder",
        "--device", args.device,
    ], timeout_sec=args.timeout_sec)
    train_result_path = train_out / "voxtell_prompt_train_result.json"
    train_result = json.loads(train_result_path.read_text(encoding="utf-8")) if train_result_path.exists() else {}
    model_dir = Path(train_result.get("inference_model_dir") or train_out / "voxtell_finetuned_model")

    pred_root = out / "round1_student_predictions"
    infer_results = []
    for case in cases:
        case_out = pred_root / case["case_id"]
        infer = student.__class__(
            model_dir=model_dir,
            target_config=ROOT / "configs/student_3d_prompt_target_organs.json",
            text_encoding_model=ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B",
            device=args.device,
        ).segment(case["ct_path"], case_out, prompts=organs, dry_run=False, timeout_sec=args.timeout_sec, prompt_batch_size=max(1, len(organs)))
        infer_results.append({
            "case_id": case["case_id"],
            "status": infer.get("status"),
            "per_organ": {organ: mask_summary(case_out / f"{organ}.nii.gz", Path(case["ct_path"])) for organ in organs},
            "result_path": str(case_out / "voxtell_student_result.json"),
        })

    loss_history_path = train_out / "loss_history.json"
    history = json.loads(loss_history_path.read_text(encoding="utf-8")).get("history", []) if loss_history_path.exists() else []
    loss_decreased = bool(history and history[-1].get("loss", 1e9) < history[0].get("loss", -1e9))
    masks_ok = all(
        item["status"] in {"success", "partial_success"}
        and all(v["exists"] and v["shape_ok"] for v in item["per_organ"].values())
        for item in infer_results
    )
    pass_checks = bool(
        estep.get("status") == "success"
        and manifest.get("num_distillation_eligible_items", 0) > 0
        and train.get("return_code") == 0
        and train_result.get("status") == "success"
        and (model_dir / "plans.json").exists()
        and (model_dir / "fold_0" / "checkpoint_final.pth").exists()
        and masks_ok
    )
    summary = {
        "stage": "round0_mstep_student_prediction_smoke",
        "status": "success" if pass_checks else "failed",
        "output_dir": str(out),
        "cases": [c["case_id"] for c in cases],
        "organs": organs,
        "estep": {"status": estep.get("status"), "total_updated": estep.get("total_updated"), "models": ["mock_seg"]},
        "manifest": {"path": str(manifest_path), "num_items": manifest.get("num_items"), "num_distillation_eligible_items": manifest.get("num_distillation_eligible_items"), "smoke_override_applied": True},
        "mstep": {"return_code": train.get("return_code"), "status": train_result.get("status"), "steps": train_result.get("steps"), "mean_loss": train_result.get("mean_loss"), "last_loss": train_result.get("last_loss"), "loss_decreased": loss_decreased, "inference_model_dir": str(model_dir)},
        "student_prediction": infer_results,
        "accuracy_warning": "Smoke validates plumbing only. mock_seg and smoke_override are not scientific training evidence.",
    }
    (out / "round_smoke_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if pass_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
