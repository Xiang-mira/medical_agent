#!/usr/bin/env python3
"""Project-side driver for one unmodified official LabelCritic comparison.

The vendor source remains byte-identical to the pinned upstream commit. This
driver stages project inputs, invokes the official AP projection, supplies a
project extension description through the official Python API, and serializes
the A/B/uncertain result without exposing teacher identities to the VLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "LabelCritic-main"


def _verify_vendor_lock() -> dict:
    lock = json.loads((ROOT / "configs" / "labelcritic_vendor_lock.json").read_text())
    mismatches = []
    for relative, expected in lock["files"].items():
        path = VENDOR / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"LabelCritic vendor lock failed: {mismatches}")
    return {"commit": lock["commit"], "verified_files": len(lock["files"])}


def _render_description(entry: dict) -> str:
    def joined(key: str) -> str:
        value = entry.get(key)
        if isinstance(value, list):
            return "; ".join(str(item) for item in value)
        return str(value or "")

    lines = [
            f"Target: {entry['canonical_name']}.",
            f"Expected CT body regions: {joined('expected_body_regions')}.",
            f"CT location: {joined('ct_location')}.",
            f"CT appearance: {joined('ct_appearance')}.",
            f"Morphology: {joined('morphology')}.",
            f"Adjacent structures: {joined('adjacent_structures')}.",
            f"Continuity: {joined('continuity_prior')}.",
            f"Symmetry/laterality: {joined('symmetry_prior')}.",
            f"Partial field of view: {joined('partial_fov_behavior')}.",
            f"Anatomical constraints: {joined('anatomical_constraints')}.",
            f"Common failure modes: {joined('common_failure_modes')}.",
            f"Reject or penalize: {joined('penalty_rules')}.",
        ]
    context = entry.get("candidate_context")
    if isinstance(context, list) and context:
        lines.append(
            "Anonymous objective candidate summaries (no teacher identity, prior "
            "ranking, historical Dice, or training weight): "
            + json.dumps(context, ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


def _structured_assessment(
    decision: dict,
    candidate_context: list[dict],
) -> dict:
    """Serialize the required nine-field audit contract without inventing prose."""
    winner = decision.get("winner")
    selected_index = 0 if winner == "a" else 1 if winner == "b" else None
    selected = (
        candidate_context[selected_index]
        if selected_index is not None and selected_index < len(candidate_context)
        else {}
    )
    objective = {
        "location": {
            "centroid_ras_mm": selected.get("centroid_ras_mm"),
            "fov_status": selected.get("fov_status"),
        },
        "oversegmentation": {
            "volume_mm3": selected.get("volume_mm3"),
            "boundary_contacts": selected.get("boundary_contacts"),
        },
        "undersegmentation": {
            "foreground_voxel_count": selected.get("foreground_voxel_count"),
            "truncation_suspected": selected.get("truncation_suspected"),
        },
        "continuity": {
            "connected_components": selected.get("connected_components"),
        },
        "laterality": {
            "centroid_laterality": selected.get("centroid_laterality"),
        },
        "partial_fov": {
            "fov_status": selected.get("fov_status"),
            "truncation_suspected": selected.get("truncation_suspected"),
        },
    }
    return {
        "identity": {
            "assessment": "anonymous_pairwise_target_identity",
            "source": "official_labelcritic_pairwise_vote",
        },
        **{
            key: {
                "assessment": "objective_summary_recorded",
                "source": "deterministic_mask_qc",
                "evidence": value,
            }
            for key, value in objective.items()
        },
        "gross_error": {
            "assessment": (
                "no_decisive_preference" if winner == "uncertain"
                else "selected_candidate_preferred_by_official_pairwise"
            ),
            "source": "official_labelcritic_pairwise_vote",
        },
        "final_recommendation": {
            "candidate_id": selected.get("candidate_id"),
            "winner": winner,
            "source": "official_labelcritic_dual_confirmation",
        },
        "vlm_free_text_rationale_available": False,
        "rationale_policy": (
            "The pinned official API returns votes, not preserved free-text rationale; "
            "objective QC evidence is recorded separately and never fabricated as VLM prose."
        ),
    }


def _read_decision(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {
            "winner": "uncertain",
            "reason": "official LabelCritic produced no comparison row",
            "parse_status": "missing_csv",
        }
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {
            "winner": "uncertain",
            "reason": "official DSC/presence gate skipped comparison",
            "parse_status": "no_comparison_rows",
        }
    row = rows[-1]
    answer = str(row.get("answer") or "").strip()
    randomized_best = str(row.get("label") or "").strip()
    if answer in {"", "0.5"}:
        winner = "uncertain"
    elif answer == randomized_best:
        winner = "b"
    elif answer in {"1", "2"} and randomized_best in {"1", "2"}:
        winner = "a"
    else:
        winner = "uncertain"
    return {
        "winner": winner,
        "parse_status": "official_csv",
        "display_order_answer": answer,
        "randomized_reference_position": randomized_best,
        "dual_confirmation_answer_1": row.get("answer_1"),
        "dual_confirmation_answer_2": row.get("answer_2"),
        "reason": (
            "Official LabelCritic dual-confirmed pairwise comparison."
            if winner != "uncertain"
            else "Official LabelCritic abstained or order confirmation was inconclusive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask-a-dir", required=True)
    parser.add_argument("--mask-b-dir", required=True)
    parser.add_argument("--organ", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--description-json", required=True)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    args = parser.parse_args()
    vendor_lock = _verify_vendor_lock()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    projection_dir = work_dir / "official_projection"
    csv_path = work_dir / "official_result.csv"
    description_doc = json.loads(Path(args.description_json).read_text())
    description_entry = (
        description_doc.get("organ_ct_appearance", {}).get(args.organ)
        if "organ_ct_appearance" in description_doc
        else description_doc
    )
    candidate_context = (
        list(description_entry.get("candidate_context") or [])
        if isinstance(description_entry, dict) else []
    )

    projection_cmd = [
        sys.executable,
        str(VENDOR / "ProjectDatasetFlex_single.py"),
        "--ct_good",
        str(Path(args.ct).resolve()),
        "--mask_good",
        str(Path(args.mask_a_dir).resolve()),
        "--ct_bad",
        str(Path(args.ct).resolve()),
        "--mask_bad",
        str(Path(args.mask_b_dir).resolve()),
        "--organ",
        args.organ,
        "--output_dir",
        str(projection_dir),
        "--axis",
        "1",
        "--device",
        "cpu",
        "--num_processes",
        "2",
    ]
    projection = subprocess.run(
        projection_cmd,
        cwd=VENDOR,
        capture_output=True,
        text=True,
        check=False,
    )
    if projection.returncode != 0:
        result = {
            "stage": "official_labelcritic_pair",
            "status": "failed",
            "reason": "official_projection_failed",
            "projection_command": projection_cmd,
            "projection_stdout_tail": projection.stdout[-4000:],
            "projection_stderr_tail": projection.stderr[-4000:],
        }
    else:
        sys.path.insert(0, str(VENDOR))
        import ErrorDetector as ed

        descriptions = dict(ed.DescriptionsED)
        if args.organ not in descriptions:
            entry = description_entry
            if not isinstance(entry, dict) or not entry.get("formal_selection_eligible"):
                result = {
                    "stage": "official_labelcritic_pair",
                    "status": "blocked",
                    "reason": "description_not_formally_eligible",
                    "organ": args.organ,
                }
                Path(args.output_json).write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps(result, indent=2))
                raise SystemExit(2)
            descriptions[args.organ] = _render_description(entry)

        base_url = f"{args.base_url.rstrip('/')}:{args.port}/v1"
        ed.SystematicComparisonLMDeploySepFigures(
            pth=str(projection_dir / args.organ),
            base_url=base_url,
            size=512,
            organ=args.organ,
            organ_descriptions=descriptions,
            save_memory=True,
            solid_overlay="auto",
            multi_image_prompt_2=(
                "auto" if args.organ in ed.DescriptionsED else True
            ),
            dual_confirmation=True,
            conservative_dual=False,
            dice_check=True,
            dice_th=args.dice_threshold,
            csv_file=str(csv_path),
            restart=True,
            examples=0,
        )
        decision = _read_decision(csv_path)
        structured_assessment = _structured_assessment(decision, candidate_context)
        result = {
            "stage": "official_labelcritic_pair",
            "status": "success",
            "organ": args.organ,
            "ct": str(Path(args.ct).resolve()),
            "mask_a_dir": str(Path(args.mask_a_dir).resolve()),
            "mask_b_dir": str(Path(args.mask_b_dir).resolve()),
            "projection_backend": "official_labelcritic_ap_axis_1",
            "centered_slice_fallback_used": False,
            "dice_gate_enabled": True,
            "dual_confirmation_enabled": True,
            "order_reversal_enabled": True,
            "order_reversal_backend": "official_dual_confirmation_y1_y2_and_y2_y1",
            "candidate_identity_exposed_to_vlm": False,
            "vendor_lock": vendor_lock,
            "description_source": (
                "official_labelcritic_seed"
                if args.organ in ed.DescriptionsED
                else "project_class_agnostic_extension"
            ),
            "decision": decision,
            "candidate_context": candidate_context,
            "structured_assessment": structured_assessment,
            "official_csv": str(csv_path),
            "projection_command": projection_cmd,
        }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
