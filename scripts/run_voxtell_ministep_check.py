#!/usr/bin/env python3
"""Run a minimal real VoxTell fine-tuning and inference check."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal real VoxTell train/infer check.")
    ap.add_argument("--manifest", default=str(ROOT / "outputs/audit_21_models/real_subset_e2e/voxtell_prompt_manifest.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audit_21_models/voxtell_ministep_check"))
    ap.add_argument("--model-dir", default=str(ROOT / "checkpoints/VoxTell/voxtell_v1.1"))
    ap.add_argument("--text-encoding-model", default=str(ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"))
    ap.add_argument("--ct-image", default=str(ROOT / "data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz"))
    ap.add_argument("--case-id", default="PanTS_00000026")
    ap.add_argument("--prompt", default="liver")
    ap.add_argument("--max-items", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=900)
    return ap.parse_args()


def run_cmd(cmd: list[str], timeout_sec: int | None = None) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False, timeout=timeout_sec)
    return {
        "command": cmd,
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    train_out = out / "train"
    infer_out = out / "student_predictions" / args.case_id
    out.mkdir(parents=True, exist_ok=True)

    train = run_cmd([
        sys.executable,
        "scripts/train_voxtell_prompt_student.py",
        "--manifest", str(Path(args.manifest).resolve()),
        "--model-dir", str(Path(args.model_dir).resolve()),
        "--text-encoding-model", str(Path(args.text_encoding_model).resolve()),
        "--output-dir", str(train_out),
        "--epochs", "1",
        "--max-steps", str(args.max_steps),
        "--max-items", str(args.max_items),
        "--freeze-encoder",
        "--device", "cuda",
    ], timeout_sec=args.timeout_sec)

    train_result_path = train_out / "voxtell_prompt_train_result.json"
    train_result = json.loads(train_result_path.read_text(encoding="utf-8")) if train_result_path.exists() else {}
    inference_model_dir = train_result.get("inference_model_dir") or str(train_out / "voxtell_finetuned_model")

    infer = run_cmd([
        sys.executable,
        "run_medai_cli.py",
        "--json",
        "voxtell-student-segment",
        "--ct-image", str(Path(args.ct_image).resolve()),
        "--output-folder", str(infer_out),
        "--model-dir", str(inference_model_dir),
        "--target-config", str(ROOT / "configs/student_3d_prompt_target_organs.json"),
        "--text-encoding-model", str(Path(args.text_encoding_model).resolve()),
        "--prompts", args.prompt,
        "--timeout-sec", str(args.timeout_sec),
    ], timeout_sec=args.timeout_sec)

    mask = infer_out / f"{args.prompt}.nii.gz"
    mask_ok = False
    mask_voxels = None
    shape_ok = False
    if mask.exists():
        ct_img = nib.load(str(args.ct_image))
        mask_img = nib.load(str(mask))
        arr = np.asanyarray(mask_img.dataobj) > 0
        mask_voxels = int(arr.sum())
        shape_ok = ct_img.shape[:3] == mask_img.shape[:3]
        mask_ok = shape_ok

    summary = {
        "stage": "voxtell_ministep_check",
        "status": "success" if train.get("return_code") == 0 and infer.get("return_code") == 0 and mask_ok else "failed",
        "train": {
            "return_code": train.get("return_code"),
            "result_path": str(train_result_path),
            "status": train_result.get("status"),
            "steps": train_result.get("steps"),
            "mean_loss": train_result.get("mean_loss"),
            "inference_model_dir": inference_model_dir,
        },
        "inference": {
            "return_code": infer.get("return_code"),
            "mask": str(mask),
            "mask_exists": mask.exists(),
            "shape_ok": shape_ok,
            "mask_voxels": mask_voxels,
        },
        "accuracy_warning": "This validates train/infer plumbing, not model quality.",
    }
    (out / "voxtell_ministep_check_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
