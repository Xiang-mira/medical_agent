"""Build per-case aggregated dataset samples in the format the teacher specified.

Each sample contains: case_id, ct_path, organ, original_mask, candidate_masks,
dice_scores, vlm_decision, final_annotation, reasoning_trace.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .json_utils import write_json


def _read_dice_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def build_case_samples(
    run_output_folder: str | Path,
    organs: list[str] | None = None,
    output_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-case data from a run-loop output into the teacher-expected format."""
    run = Path(run_output_folder).resolve()
    if organs is None:
        organs = ["pancreas", "liver", "spleen", "kidney_left", "kidney_right",
                  "colon", "duodenum", "stomach", "aorta", "postcava"]

    dice_rows = _read_dice_csv(run / "dice_metrics.csv")
    vlm_decisions = _read_jsonl(run / "vlm_decisions.jsonl")
    traces = _read_jsonl(run / "patient_traces.jsonl")
    review_items = _read_jsonl(run / "review_queue.jsonl")
    inference_results = []
    if (run / "inference_results.json").exists():
        try:
            inference_results = json.loads((run / "inference_results.json").read_text(encoding="utf-8"))
        except Exception:
            inference_results = []
    inference_ct_by_case = {}
    for item in inference_results:
        cid = item.get("case_id", "")
        if cid and item.get("image"):
            inference_ct_by_case[cid] = item.get("image")
        elif cid and item.get("ct_image"):
            inference_ct_by_case[cid] = item.get("ct_image")

    # Index by case_id
    dice_by_case: dict[str, list[dict]] = {}
    for r in dice_rows:
        cid = r.get("case_id", "")
        if cid:
            dice_by_case.setdefault(cid, []).append(r)

    vlm_by_case: dict[str, list[dict]] = {}
    for r in vlm_decisions:
        cid = r.get("case_id", "")
        if cid:
            vlm_by_case.setdefault(cid, []).append(r)

    trace_by_case: dict[str, dict] = {}
    for r in traces:
        cid = r.get("case_id", "")
        if cid:
            trace_by_case[cid] = r.get("trace", r)

    # Build samples
    samples = []
    case_ids = sorted(set(r.get("case_id", "") for r in dice_rows if r.get("case_id")))

    for cid in case_ids:
        case_dice = dice_by_case.get(cid, [])
        case_vlm = vlm_by_case.get(cid, [])
        case_trace = trace_by_case.get(cid, {})

        for organ in organs:
            organ_dice = [r for r in case_dice if r.get("organ") == organ]
            organ_vlm = [r for r in case_vlm if r.get("organ") == organ]

            if not organ_dice:
                continue

            # Build candidate_masks and dice_scores
            candidate_masks: dict[str, str] = {}
            dice_scores: dict[str, float | None] = {}
            best_model = None
            best_dice = -1.0
            reference = ""

            for r in organ_dice:
                model = r.get("model", "unknown")
                pred = r.get("prediction", "")
                dice = None
                try:
                    dice = float(r.get("dice", ""))
                except (ValueError, TypeError):
                    pass
                candidate_masks[model] = pred
                dice_scores[model] = dice
                reference = r.get("reference", reference)
                if dice is not None and dice > best_dice:
                    best_dice = dice
                    best_model = model

            # VLM decision
            vlm_decision = None
            if organ_vlm:
                vd = organ_vlm[0]
                vlm_decision = {
                    "selected": vd.get("winner", "uncertain"),
                    "reason": vd.get("reason", ""),
                }

            # Final annotation
            updated_dir = run / "annotation_versions" / cid / "updated"
            final_mask = updated_dir / f"{organ}.nii.gz"

            # Reasoning trace (extract organ-relevant part)
            trace_data = case_trace.get("radthinking_trace", case_trace) if case_trace else {}

            sample = {
                "case_id": cid,
                "ct_path": inference_ct_by_case.get(cid),
                "organ": organ,
                "original_mask": reference,
                "candidate_masks": candidate_masks,
                "dice_scores": dice_scores,
                "best_candidate": {"model": best_model, "dice": best_dice if best_dice >= 0 else None},
                "vlm_decision": vlm_decision,
                "final_annotation": str(final_mask) if final_mask.exists() else None,
                "reasoning_trace": {
                    "step1_observations": trace_data.get("observation", {}),
                    "step2_temporal": trace_data.get("temporal_comparison", {}),
                    "step3_clinical_context": trace_data.get("clinical_context", {}),
                    "step4_conclusion": trace_data.get("diagnostic_conclusion", {}),
                    "narrative": trace_data.get("narrative", {}),
                    "reasoning_complexity": trace_data.get("complexity_level_prototype", "PERCEPTUAL") if isinstance(trace_data, dict) else "PERCEPTUAL",
                },
            }
            samples.append(sample)

    if output_jsonl:
        out = Path(output_jsonl).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return {
        "stage": "case_sample_builder",
        "status": "success",
        "num_samples": len(samples),
        "num_cases": len(case_ids),
        "organs": organs,
        "output_jsonl": str(output_jsonl) if output_jsonl else None,
        "sample_preview": samples[:3],
    }


def build_convergence_table(
    round_metrics_csvs: list[str | Path],
) -> dict[str, Any]:
    """Build the teacher-expected loop convergence table across multiple rounds.

    Each CSV is the round_metrics.csv from one EM iteration.
    """
    rows = []
    for idx, csv_path in enumerate(round_metrics_csvs):
        p = Path(csv_path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            total_checked = total_low = total_vlm = total_updated = total_uncertain = 0
            for r in reader:
                total_checked += int(r.get("checked_masks", 0))
                total_low += int(r.get("low_dice_masks", 0))
                total_vlm += int(r.get("vlm_reviewed", 0))
                total_updated += int(r.get("updated_masks", 0))
                total_uncertain += int(r.get("remaining_uncertain", 0))
            rows.append({
                "loop": idx + 1,
                "checked_masks": total_checked,
                "dice_below_threshold": total_low,
                "vlm_reviewed": total_vlm,
                "updated_masks": total_updated,
                "remaining_uncertain": total_uncertain,
            })

    return {
        "stage": "convergence_table",
        "status": "success",
        "num_rounds": len(rows),
        "table": rows,
        "note": "Teacher expects: low_dice and uncertain counts should decrease across loops.",
    }
