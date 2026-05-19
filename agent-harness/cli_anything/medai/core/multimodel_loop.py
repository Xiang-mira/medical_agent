from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .adapter import normalize_totalseg_to_shapekit
from .json_utils import write_json
from .label_verifier import verify_annotation
from .labelcritic_wrapper import run_labelcritic_compare
from .mstep_runner import build_training_manifest, write_mstep_config
from .model_registry import candidate_models_for_organs, load_registry, recommend_primary_models_for_organs
from .radthinking import build_reasoning_trace
from .registered_infer import run_registered_model
from .shapekit_runner import run_shapekit


def _read_case_list(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any((v or "").strip() for v in row.values()):
                continue
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def _mask_path(seg_dir: Path, organ: str) -> Path:
    return seg_dir / f"{organ}.nii.gz"


def _copy_annotation(src: Path | None, dst_dir: Path, organ: str) -> str | None:
    if src and src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{organ}.nii.gz"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst.resolve())
    return None


def run_multimodel_annotation_loop(
    case_list: str | Path,
    output_folder: str | Path,
    models: list[str] | None = None,
    organs: list[str] | None = None,
    registry_path: str | Path = "configs/model_registry.yaml",
    checkpoint_map_models: bool = False,
    shapekit_root: str | Path = "third_party/ShapeKit-main",
    enable_shapekit: bool = True,
    enable_critic: bool = True,
    critic_backend: str = "labelcritic",
    critic_base_url: str = "http://localhost",
    critic_port: int = 8000,
    vlm_threshold: float = 0.5,
    accept_threshold: float = 0.8,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the teacher-requested multi-model annotation refinement loop.

    Required case_list columns:
      case_id, ct_path, annotation_folder
    Optional columns:
      report_path, clinical_path, pathology_path
    """
    case_csv = Path(case_list).resolve()
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = _read_case_list(case_csv)
    registry = load_registry(registry_path)
    if organs is None or not organs:
        organs = ["pancreas", "liver", "spleen", "kidney_left", "kidney_right", "colon", "duodenum", "stomach", "aorta", "postcava"]
    if models is None or not models:
        models = ["mock_seg"] if dry_run else ["totalsegmentator"]

    dice_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    inference_results: list[dict[str, Any]] = []
    updated_root = out / "annotation_versions"
    review_queue = out / "review_queue.jsonl"
    vlm_decisions = out / "vlm_decisions.jsonl"
    traces_jsonl = out / "patient_traces.jsonl"
    report_supervision_jsonl = out / "report_supervision.jsonl"

    # Reset append-only outputs for a clean run.
    for p in (review_queue, vlm_decisions, traces_jsonl, report_supervision_jsonl):
        if p.exists():
            p.unlink()

    for idx, case in enumerate(cases, start=1):
        case_id = case.get("case_id") or Path(case.get("ct_path", f"case_{idx}")).parent.name
        ct = Path(case.get("ct_path", "")).resolve()
        ref_dir = Path(case.get("annotation_folder", "")).resolve() if case.get("annotation_folder") else None
        case_out = out / "cases" / case_id
        case_raw = case_out / "raw_predictions"
        case_refined = case_out / "refined_predictions"
        case_updated = updated_root / case_id / "updated"
        case_updated.mkdir(parents=True, exist_ok=True)

        if not ct.exists() and not dry_run:
            _append_jsonl(review_queue, {"case_id": case_id, "reason": "ct_path missing", "ct_path": str(ct)})
            continue

        # Optionally expand candidate models by organ mapping.
        case_models = list(models)
        if checkpoint_map_models:
            mapped = candidate_models_for_organs(registry, organs)
            for organ_models in mapped.values():
                for m in organ_models:
                    if m not in case_models:
                        case_models.append(m)

        model_seg_dirs: dict[str, Path] = {}
        for model_key in case_models:
            infer = run_registered_model(ct, case_raw / model_key, model_key, registry_path=registry_path, case_id=case_id, dry_run=dry_run, timeout_sec=timeout_sec, device=device)
            inference_results.append({"case_id": case_id, **infer})
            seg_dir = Path(infer.get("segmentation_output", case_raw / model_key / case_id / "segmentations"))
            if infer.get("status") in {"success", "dry_run"}:
                model_seg_dirs[model_key] = seg_dir
                if not dry_run:
                    normalize_totalseg_to_shapekit(seg_dir)
                if enable_shapekit and infer.get("status") == "success" and not dry_run:
                    refined_root = case_refined / model_key
                    post = run_shapekit(shapekit_root, case_raw / model_key, refined_root, refined_root / "logs", cpu_count=2, dry_run=False, auto_config=True, timeout_sec=min(timeout_sec, 900))
                    refined_seg = refined_root / case_id / "segmentations"
                    if post.get("status") == "success" and refined_seg.exists():
                        model_seg_dirs[f"{model_key}_shapekit"] = refined_seg

        checked = accepted = low_dice = uncertain = updated = critic_count = 0
        for organ in organs:
            current_ref = _mask_path(ref_dir, organ) if ref_dir else None
            best_model = None
            best_dice = -1.0
            best_pred: Path | None = None
            organ_rows: list[dict[str, Any]] = []

            for model_key, seg_dir in model_seg_dirs.items():
                pred = _mask_path(seg_dir, organ)
                v = verify_annotation(current_ref if current_ref and current_ref.exists() else None, pred if pred.exists() else None, organ, dsc_replace_threshold=0.0, dsc_vlm_threshold=vlm_threshold)
                dice = v.get("dice")
                checked += 1
                if dice is not None and dice > best_dice:
                    best_dice = float(dice); best_model = model_key; best_pred = pred
                row = {"case_id": case_id, "organ": organ, "model": model_key, "prediction": str(pred), "reference": str(current_ref) if current_ref else "", "dice": dice, "decision": v.get("decision"), "status": v.get("status"), "reason": v.get("reason")}
                dice_rows.append(row); organ_rows.append(row)

            if not organ_rows:
                _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, "reason": "no candidate masks produced"})
                uncertain += 1
                continue

            if best_dice >= accept_threshold and best_pred and best_pred.exists():
                _copy_annotation(best_pred, case_updated, organ); accepted += 1; updated += 1
            elif best_dice >= 0 and best_dice < vlm_threshold:
                low_dice += 1
                if enable_critic and len([r for r in organ_rows if Path(r["prediction"]).exists()]) >= 2 and not dry_run:
                    # Sort by DICE descending so LabelCritic compares the two highest-DICE candidates.
                    ranked = sorted(
                        [r for r in organ_rows if Path(r["prediction"]).exists() and r.get("dice") is not None],
                        key=lambda r: float(r["dice"]) if r["dice"] is not None else -1.0,
                        reverse=True,
                    )
                    if len(ranked) < 2:
                        ranked = [r for r in organ_rows if Path(r["prediction"]).exists()]
                    existing = [Path(ranked[0]["prediction"]), Path(ranked[1]["prediction"])]
                    critic_out = out / "critic" / case_id / f"{organ}.json"
                    critic = run_labelcritic_compare(ct, existing[0], existing[1], organ, critic_out, backend=critic_backend, base_url=critic_base_url, port=critic_port, dry_run=False, timeout_sec=min(timeout_sec, 300))
                    _append_jsonl(vlm_decisions, {"case_id": case_id, "organ": organ, **critic.get("decision", {}), "output_json": str(critic_out)})
                    critic_count += 1
                    winner = critic.get("decision", {}).get("winner")
                    chosen = existing[0] if winner == "a" else existing[1] if winner == "b" else best_pred
                    if chosen and chosen.exists():
                        _copy_annotation(chosen, case_updated, organ); updated += 1
                    else:
                        uncertain += 1
                else:
                    _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, "reason": "low dice or dry-run; needs LabelCritic/manual review", "best_model": best_model, "best_dice": best_dice})
                    uncertain += 1
            elif best_pred and best_pred.exists():
                # Moderate quality: keep the best candidate but mark as uncertain if below accept threshold.
                _copy_annotation(best_pred, case_updated, organ); updated += 1
                uncertain += 1
                _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, "reason": "moderate dice; copied best candidate but recommend sanity check", "best_model": best_model, "best_dice": best_dice})
            else:
                uncertain += 1
                _append_jsonl(review_queue, {"case_id": case_id, "organ": organ, "reason": "no readable prediction", "best_model": best_model})

        # Report supervision: compare tumor mask against report if both are available.
        if not dry_run and case.get("report_path"):
            try:
                from .report_supervision import verify_tumor_with_report
                tumor_mask = case_updated / "pancreatic_lesion.nii.gz"
                if not tumor_mask.exists() and ref_dir:
                    tumor_mask = ref_dir / "pancreatic_lesion.nii.gz"
                if tumor_mask.exists():
                    report_decision = verify_tumor_with_report(Path(case["report_path"]).resolve(), tumor_mask, "pancreas", Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None)
                    _append_jsonl(report_supervision_jsonl, {"case_id": case_id, **report_decision})
            except Exception as exc:
                _append_jsonl(report_supervision_jsonl, {"case_id": case_id, "stage": "report_supervision", "status": "failed", "reason": str(exc)})

        # Reasoning trace grounded in available case paths — one entry per organ with updated mask.
        if not dry_run:
            for trace_organ in organs:
                organ_mask = case_updated / f"{trace_organ}.nii.gz"
                if not organ_mask.exists():
                    continue
                try:
                    trace = build_reasoning_trace(
                        patient_folder=None, scan_id=case_id, ct_image=ct,
                        current_mask=organ_mask,
                        previous_mask=None, organ=trace_organ,
                        report_path=Path(case["report_path"]).resolve() if case.get("report_path") else None,
                        clinical_path=Path(case["clinical_path"]).resolve() if case.get("clinical_path") else None,
                        pathology_path=Path(case["pathology_path"]).resolve() if case.get("pathology_path") else None,
                        output_json=None,
                    )
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace": trace})
                except Exception as exc:
                    _append_jsonl(traces_jsonl, {"case_id": case_id, "organ": trace_organ, "trace_status": "failed", "reason": str(exc)})

        round_rows.append({
            "case_id": case_id,
            "checked_masks": checked,
            "accepted_masks": accepted,
            "low_dice_masks": low_dice,
            "vlm_reviewed": critic_count,
            "updated_masks": updated,
            "remaining_uncertain": uncertain,
        })

    dice_csv = out / "dice_metrics.csv"
    round_csv = out / "round_metrics.csv"
    _write_csv(dice_csv, dice_rows, ["case_id", "organ", "model", "prediction", "reference", "dice", "decision", "status", "reason"])
    _write_csv(round_csv, round_rows, ["case_id", "checked_masks", "accepted_masks", "low_dice_masks", "vlm_reviewed", "updated_masks", "remaining_uncertain"])
    write_json(out / "inference_results.json", inference_results)
    manifest = build_training_manifest(updated_root, out / "training_manifest.json", organs=organs)
    # Selected-model-aware M-step routing: record which primary model should be
    # updated for each organ, instead of implying that a generic model is always
    # the M-step target.
    mstep_routing = recommend_primary_models_for_organs(registry, organs)
    write_json(out / "mstep_model_routing.json", mstep_routing)
    mcfg = write_mstep_config(
        out / "mstep_config.json",
        out / "training_manifest.json",
        base_model="selected_model_aware",
        notes="Use `mstep-update --target-model <primary_model>` for each trainable primary model in mstep_model_routing.json. TotalSegmentator is baseline-only in this project.",
    )

    summary = {
        "stage": "run_loop", "status": "success", "case_list": str(case_csv), "output_folder": str(out),
        "num_cases": len(cases), "models_requested": models, "organs": organs,
        "dry_run": dry_run, "enable_shapekit": enable_shapekit, "enable_critic": enable_critic, "critic_backend": critic_backend, "critic_base_url": critic_base_url, "critic_port": critic_port,
        "dice_metrics_csv": str(dice_csv), "round_metrics_csv": str(round_csv), "review_queue_jsonl": str(review_queue),
        "vlm_decisions_jsonl": str(vlm_decisions), "patient_traces_jsonl": str(traces_jsonl), "report_supervision_jsonl": str(report_supervision_jsonl),
        "training_manifest": manifest, "mstep_config": mcfg, "mstep_model_routing": mstep_routing,
        "round_rows": round_rows,
    }
    write_json(out / "run_summary.json", summary)
    return summary
