from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotation_manager import AnnotationManager
from .shapekit_runner import run_shapekit
from .totalseg_runner import run_totalsegmentator
from .vlm_label_expert import run_vlm_label_expert


def _compute_dsc(mask_a: Path | None, mask_b: Path | None) -> float | None:
    if mask_a is None or mask_b is None:
        return None
    if not mask_a.exists() or not mask_b.exists():
        return None
    try:
        import nibabel as nib
        import numpy as np
        a = np.asanyarray(nib.load(str(mask_a)).dataobj) > 0
        b = np.asanyarray(nib.load(str(mask_b)).dataobj) > 0
        intersection = int(np.logical_and(a, b).sum())
        union = int(a.sum() + b.sum())
        return round(float(2 * intersection / union), 4) if union > 0 else 1.0
    except Exception:
        return None


def _find_seg_dir(root: Path, case_id: str) -> Path | None:
    for cand in [
        root / case_id / "segmentations",
        root / "segmentations",
        root,
    ]:
        if cand.exists() and any(cand.glob("*.nii.gz")):
            return cand
    return None


def run_em_loop(
    case_id: str,
    ct_image: str | Path,
    annotation_folder: str | Path,
    output_folder: str | Path,
    organs: list[str],
    num_rounds: int = 2,
    vlm_backend: str = "ollama",
    vlm_model: str = "qwen2.5vl:7b",
    dsc_replace_threshold: float = 0.0,
    dsc_vlm_threshold: float = 0.5,
    fast: bool = True,
    device: str | None = None,
    postprocess: str = "shapekit",
    shapekit_root: str | Path = "third_party/ShapeKit-main",
    ct_source_root: str | Path | None = None,
    enable_mstep_training: bool = False,
    mstep_max_epochs: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Mini EM loop for annotation refinement.

    E-step : Label Verifier + VLM Label Expert compare reference vs prediction.
    M-step : Annotation update (GPU retraining stubbed — not available on CPU).

    Rounds:
      Round 0 : TotalSegmentator inference → E-step → annotation update
      Round 1+: Re-evaluate metrics on updated annotations (M-step stub)
    """
    ct_path = Path(ct_image).resolve()
    ann_folder = Path(annotation_folder).resolve()
    out_dir = Path(output_folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "stage": "em_loop",
        "case_id": case_id,
        "ct_image": str(ct_path),
        "annotation_folder": str(ann_folder),
        "output_folder": str(out_dir),
        "organs": organs,
        "num_rounds": num_rounds,
        "vlm_backend": vlm_backend,
        "vlm_model": vlm_model,
        "dry_run": dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    manager = AnnotationManager(out_dir / "annotations", case_id)

    # Save reference labels as raw (never overwritten once copied)
    raw_saved: dict[str, str | None] = {}
    for organ in organs:
        src = ann_folder / f"{organ}.nii.gz"
        dst = manager.save_raw(organ, src)
        raw_saved[organ] = str(dst) if src.exists() else None
    result["raw_annotations_saved"] = raw_saved

    rounds_log: list[dict] = []

    for round_idx in range(num_rounds):
        round_result: dict[str, Any] = {
            "round": round_idx,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── Inference ────────────────────────────────────────────────────────
        pred_dir = out_dir / f"predictions_round_{round_idx:03d}"
        if dry_run:
            infer = {
                "stage": "infer", "status": "dry_run",
                "segmentation_output": str(pred_dir / case_id / "segmentations"),
            }
        else:
            infer = run_totalsegmentator(
                str(ct_path), str(pred_dir),
                case_id=f"{case_id}_r{round_idx}",
                fast=fast, device=device,
            )
        round_result["inference"] = infer

        # Optional ShapeKit post-processing (anatomy-aware refinement)
        post_result = None
        refined_dir = out_dir / f"refined_round_{round_idx:03d}"
        if postprocess == "shapekit" and not dry_run and infer.get("status") == "success":
            post_result = run_shapekit(
                str(Path(shapekit_root).resolve()),
                str(pred_dir),
                str(refined_dir),
                str(out_dir / f"shapekit_logs_round_{round_idx:03d}"),
                dry_run=False,
            )
            round_result["shapekit"] = post_result
            # Use refined output as source for E-step if ShapeKit succeeded
            seg_source_dir = refined_dir if post_result.get("status") == "success" else pred_dir
        else:
            seg_source_dir = pred_dir

        # Locate segmentation masks
        seg_dir = _find_seg_dir(seg_source_dir, f"{case_id}_r{round_idx}") \
                  or _find_seg_dir(seg_source_dir, case_id)

        # ── E-step ───────────────────────────────────────────────────────────
        e_step: list[dict] = []
        for organ in organs:
            pred_mask = seg_dir / f"{organ}.nii.gz" if seg_dir else None
            if pred_mask and pred_mask.exists():
                manager.save_prediction(organ, pred_mask, round_idx)

            ann_a = manager.get_current(organ)   # currently accepted annotation
            ann_b = manager.get_prediction(organ, round_idx)  # this round's prediction

            if dry_run:
                expert = {
                    "stage": "vlm_label_expert", "status": "dry_run",
                    "decision": "keep_annotation_a", "winner": "A",
                    "confidence": 1.0, "reason": "dry-run: skipping VLM",
                    "vlm_called": False,
                }
            else:
                expert = run_vlm_label_expert(
                    ct_image=ct_path,
                    annotation_a=ann_a,
                    annotation_b=ann_b,
                    organ=organ,
                    output_folder=out_dir / "vlm_outputs" / f"round_{round_idx:03d}",
                    vlm_backend=vlm_backend,
                    vlm_model=vlm_model,
                    case_id=case_id,
                    dsc_replace_threshold=dsc_replace_threshold,
                    dsc_vlm_threshold=dsc_vlm_threshold,
                )

            winner = expert.get("winner", "A")
            manager.log_decision(organ, round_idx, expert)
            updated = manager.apply_update(
                organ, winner, round_idx,
                candidate_a=ann_a,
                candidate_b=ann_b,
            )

            dice_vs_raw = _compute_dsc(manager.get_raw(organ), updated) if not dry_run else None
            e_step.append({
                "organ": organ,
                "winner": winner,
                "decision": expert.get("decision"),
                "confidence": expert.get("confidence"),
                "reason": expert.get("reason", ""),
                "updated_path": str(updated) if updated else None,
                "dice_updated_vs_raw": dice_vs_raw,
            })

        round_result["e_step"] = e_step

        # ── M-step (nnUNet v2 fine-tuning or dry-run stub) ────────────────
        organs_flipped = [r["organ"] for r in e_step if r["winner"] == "B"]
        raw_weight   = max(0.0, round(1.0 - round_idx * 0.3, 1))
        pseudo_weight = min(1.0, round(round_idx * 0.3, 1))

        if enable_mstep_training and len(organs_flipped) > 0 and not dry_run:
            from .mstep_runner import build_training_manifest, run_mstep_nnunet_training
            mstep_out = out_dir / f"mstep_round_{round_idx:03d}"
            manifest = build_training_manifest(
                out_dir / "annotations", mstep_out / "training_manifest.json", organs=organs,
            )
            mstep_result = run_mstep_nnunet_training(
                training_manifest=mstep_out / "training_manifest.json",
                output_folder=mstep_out,
                ct_source_root=ct_source_root,
                max_epochs=mstep_max_epochs,
                dry_run=False,
            )
            round_result["m_step"] = {
                "status": mstep_result.get("status", "failed"),
                "organs_annotation_updated": organs_flipped,
                "training_result": mstep_result,
                "data_annealing_plan": {
                    "round": round_idx,
                    "raw_verified_weight": raw_weight,
                    "pseudo_high_confidence_weight": pseudo_weight,
                    "human_reviewed_weight": 1.0 if round_idx >= 2 else 0.0,
                },
            }
        else:
            round_result["m_step"] = {
                "status": "dry_run" if dry_run else ("skipped_no_updates" if len(organs_flipped) == 0 else "training_disabled"),
                "note": (
                    "M-step training interface is ready (run_mstep_nnunet_training). "
                    "Enable with --enable-mstep-training on a GPU server with nnUNet v2 installed. "
                    "50 cases will overfit; scale to 500+ for real performance."
                ),
                "organs_annotation_updated": organs_flipped,
                "would_retrain": len(organs_flipped) > 0,
                "data_annealing_plan": {
                    "round": round_idx,
                    "raw_verified_weight": raw_weight,
                    "pseudo_high_confidence_weight": pseudo_weight,
                    "human_reviewed_weight": 1.0 if round_idx >= 2 else 0.0,
                    "note": "Planned data mix for this round.",
                },
            }

        round_result["completed_at"] = datetime.now(timezone.utc).isoformat()
        rounds_log.append(round_result)

    # ── Final metrics ─────────────────────────────────────────────────────────
    final_metrics: dict[str, Any] = {}
    for organ in organs:
        raw_path = manager.get_raw(organ)
        current_path = manager.get_current(organ)
        # DSC between the final accepted annotation and the original reference
        dsc = _compute_dsc(raw_path, current_path) if not dry_run else None
        final_metrics[organ] = {
            "dice_final_vs_raw": dsc,
            "final_annotation": str(current_path) if current_path else None,
        }

    result["rounds"] = rounds_log
    result["final_metrics"] = final_metrics
    result["annotation_summary"] = manager.summary()
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "dry_run" if dry_run else "success"

    # Persist rounds_metrics.json
    metrics_path = out_dir / "rounds_metrics.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    result["rounds_metrics_json"] = str(metrics_path)
    return result
