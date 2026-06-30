#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.voxtell_official_predictor import OfficialVoxTellPretrainedAdapter


def mask_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "voxels": None, "shape": None, "affine": None}
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj) > 0
    return {
        "exists": True,
        "voxels": int(arr.sum()),
        "shape": [int(x) for x in img.shape[:3]],
        "spacing": [float(x) for x in img.header.get_zooms()[:3]],
        "affine": np.asarray(img.affine).round(6).tolist(),
    }


def dice(a: Path, b: Path) -> float | None:
    if not a.exists() or not b.exists():
        return None
    aa = np.asanyarray(nib.load(str(a)).dataobj) > 0
    bb = np.asanyarray(nib.load(str(b)).dataobj) > 0
    denom = int(aa.sum() + bb.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(aa, bb).sum() / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description="Official pretrained VoxTell baseline_only smoke.")
    ap.add_argument("--ct-image", default=str(ROOT / "data/PanTS/ImageTr/PanTS_00000026/ct.nii.gz"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/smoke_voxtell/baseline_official_pretrained"))
    ap.add_argument("--model-dir", default=str(ROOT / "checkpoints/VoxTell/voxtell_v1.1"))
    ap.add_argument("--target-config", default=str(ROOT / "configs/student_3d_prompt_target_organs.json"))
    ap.add_argument("--text-encoding-model", default=str(ROOT / "checkpoints/Qwen/Qwen3-Embedding-4B"))
    ap.add_argument("--prompts", default="liver,spleen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout-sec", type=int, default=1800)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    organs = [x.strip() for x in args.prompts.split(",") if x.strip()]
    adapter = OfficialVoxTellPretrainedAdapter(
        model_dir=args.model_dir,
        target_config=args.target_config,
        text_encoding_model=args.text_encoding_model,
        mode="baseline_only",
        device=args.device,
    )
    result = adapter.segment(
        ct_image=args.ct_image,
        output_dir=out / "masks",
        prompts=organs,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        prompt_batch_size=max(1, len(organs)),
    )
    ct_img = nib.load(str(args.ct_image))
    masks = {organ: Path(result.get("expected_masks", {}).get(organ, out / "masks" / f"{organ}.nii.gz")) for organ in organs}
    stats = {organ: mask_stats(path) for organ, path in masks.items()}
    pairwise = {}
    if len(organs) >= 2:
        pairwise[f"{organs[0]}_vs_{organs[1]}_dice"] = dice(masks[organs[0]], masks[organs[1]])
    no_manifest_files = not any((out / "masks").glob("*manifest*.json"))
    pass_checks = bool(
        result.get("official_voxtell_mode") == "baseline_only"
        and result.get("allow_selection_by_autolabelcore") is False
        and result.get("used_for_selected_pseudo_label") is False
        and result.get("used_for_training_manifest") is False
        and result.get("eligible_for_next_round_prompt_student") is False
        and all(s["exists"] and s["shape"] == list(ct_img.shape[:3]) for s in stats.values())
        and no_manifest_files
    )
    summary = {
        "stage": "official_voxtell_pretrained_baseline_smoke",
        "status": "success" if pass_checks else "failed",
        "ct_image": str(Path(args.ct_image).resolve()),
        "output_dir": str(out),
        "organs": organs,
        "adapter_result_status": result.get("status"),
        "baseline_audit": {
            "provider": result.get("provider"),
            "model_role": result.get("model_role"),
            "official_voxtell_mode": result.get("official_voxtell_mode"),
            "allow_selection_by_autolabelcore": result.get("allow_selection_by_autolabelcore"),
            "active_as_teacher_candidate": result.get("active_as_teacher_candidate"),
            "used_for_selected_pseudo_label": result.get("used_for_selected_pseudo_label"),
            "used_for_training_manifest": result.get("used_for_training_manifest"),
            "eligible_for_next_round_prompt_student": result.get("eligible_for_next_round_prompt_student"),
            "is_project_student": result.get("is_project_student"),
            "prompt_conditioned": result.get("prompt_conditioned"),
            "uses_official_voxtell_predictor": result.get("uses_official_voxtell_predictor"),
        },
        "mask_stats": stats,
        "pairwise": pairwise,
        "no_manifest_files_in_baseline_output": no_manifest_files,
        "checks_passed": pass_checks,
        "raw_result_path": str(out / "masks" / "voxtell_student_result.json"),
    }
    (out / "baseline_smoke_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if pass_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
