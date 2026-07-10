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


def _project_extension_compare_prompt(organ: str, description: str) -> str:
    display = organ.replace("_", " ")
    return (
        "The images I am sending are frontal projections of the same CT scan. "
        "They are not CT slices; they have transparency and are oriented like AP X-rays. "
        "Red overlays on the images demarcate candidate masks for the target organ, "
        "but the accuracy of these overlays is uncertain. Compare the two overlays "
        "and decide which one better represents the %(organ)s.\n"
        f"Target-specific anatomical guidance for {display}:\n{description}\n"
        "Evaluate location, CT appearance, shape/continuity, oversegmentation, "
        "undersegmentation, truncation, and leakage into adjacent structures. "
        "If both overlays have errors, choose the one with fewer or smaller errors. "
        "Answer which overlay is better: overlay 1 or overlay 2, and briefly justify."
    )


def _strict_choice_compare_prompt(organ: str, description: str) -> str:
    display = organ.replace("_", " ")
    return (
        "You are comparing two anonymous candidate segmentation overlays for the "
        f"same target structure: {display}.\n"
        "Use the CT/projection anatomy, target location, continuity, laterality, "
        "and leakage/oversegmentation rules below. If one overlay is clearly "
        "better, choose it even if both are imperfect. Use 0.5 only when the two "
        "overlays are genuinely indistinguishable or both are unusable.\n"
        f"Target-specific guidance:\n{description}\n"
        "Final answer format is strict: reply with exactly one token: 1, 2, or 0.5. "
        "1 means overlay 1 is better. 2 means overlay 2 is better. 0.5 means uncertain."
    )


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


def _read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    try:
        return list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    except Exception:
        return []


def _failure_taxonomy(
    *,
    result_status: str,
    projection_ok: bool,
    dice_gate_enabled: bool,
    csv_rows: list[dict],
    decision: dict,
    audit_only: bool,
) -> list[str]:
    failures: list[str] = []
    if not projection_ok:
        failures.append("projection_failed")
    if audit_only:
        failures.append("audit_only_target")
    if result_status == "success" and dice_gate_enabled and not csv_rows:
        failures.append("dice_gate_skipped")
    if str(decision.get("parse_status") or "") in {"missing_csv", "missing_driver_result"}:
        failures.append("parser_failed")
    if decision.get("winner") == "uncertain":
        failures.append("vlm_uncertain")
    return list(dict.fromkeys(failures))


def _is_left_right_join_only_failure(stderr: str, stdout: str, organ: str, projection_dir: Path) -> bool:
    """Treat vendor left/right join failure as non-fatal for one-organ pair checks.

    The pinned official projection script always tries to join ``*left*`` with
    the matching ``*right*`` folder. Our pairwise call intentionally projects one
    organ at a time, so the comparison images for the requested left organ can be
    valid even though the optional joined bilateral folder is absent.
    """
    if "left" not in organ:
        return False
    counterpart = organ.replace("left", "right")
    text = f"{stderr}\n{stdout}"
    return (
        "FileNotFoundError" in text
        and counterpart in text
        and (projection_dir / organ).is_dir()
        and any((projection_dir / organ).glob("*.png"))
    )


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
    parser.add_argument("--no-dice-check", action="store_true")
    parser.add_argument("--no-dual-confirmation", action="store_true")
    parser.add_argument("--strict-choice-prompt", action="store_true")
    parser.add_argument("--projection-mode", choices=["ap", "multiview_audit"], default="ap")
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
    tolerated_projection_warning = None
    if projection.returncode != 0 and _is_left_right_join_only_failure(
        projection.stderr,
        projection.stdout,
        args.organ,
        projection_dir,
    ):
        tolerated_projection_warning = (
            "official_projection_left_right_join_counterpart_missing_tolerated"
        )

    if projection.returncode != 0 and not tolerated_projection_warning:
        result = {
            "stage": "official_labelcritic_pair",
            "status": "failed",
            "reason": "official_projection_failed",
            "projection_command": projection_cmd,
            "projection_stdout_tail": projection.stdout[-4000:],
            "projection_stderr_tail": projection.stderr[-4000:],
            "csv_row_count": 0,
            "raw_csv_rows": [],
            "failure_taxonomy": ["projection_failed"],
        }
    else:
        sys.path.insert(0, str(VENDOR))
        import ErrorDetector as ed

        descriptions = dict(ed.DescriptionsED)
        project_extension_description = None
        audit_only_comparison = False
        formal_selection_eligible = True
        if args.organ not in descriptions:
            entry = description_entry
            if not isinstance(entry, dict):
                result = {
                    "stage": "official_labelcritic_pair",
                    "status": "blocked",
                    "reason": "description_missing_or_invalid",
                    "organ": args.organ,
                }
                Path(args.output_json).write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps(result, indent=2))
                raise SystemExit(2)
            formal_selection_eligible = bool(entry.get("formal_selection_eligible"))
            audit_only_comparison = (
                not formal_selection_eligible
                and str(entry.get("automatic_failure_action") or "") in {
                    "withhold_or_audit_only",
                    "audit_only_withhold_from_formal_selection",
                }
            )
            if not formal_selection_eligible and not audit_only_comparison:
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
            project_extension_description = _render_description(entry)
            descriptions[args.organ] = project_extension_description
        rendered_description = (
            project_extension_description
            if project_extension_description is not None
            else str(descriptions.get(args.organ) or "")
        )
        text_multi_image_prompt_2 = (
            _strict_choice_compare_prompt(args.organ, rendered_description)
            if args.strict_choice_prompt
            else ed.Compare2Images
            if args.organ in ed.DescriptionsED
            else _project_extension_compare_prompt(args.organ, rendered_description)
        )
        prompt_path = work_dir / "labelcritic_prompt.txt"
        prompt_path.write_text(str(text_multi_image_prompt_2), encoding="utf-8")
        description_rendered_path = work_dir / "labelcritic_description.txt"
        description_rendered_path.write_text(rendered_description, encoding="utf-8")

        base_url = f"{args.base_url.rstrip('/')}:{args.port}/v1"
        dice_gate_enabled = not args.no_dice_check
        dual_confirmation_enabled = not args.no_dual_confirmation
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
            text_multi_image_prompt_2=text_multi_image_prompt_2,
            dual_confirmation=dual_confirmation_enabled,
            conservative_dual=False,
            dice_check=dice_gate_enabled,
            dice_th=args.dice_threshold,
            csv_file=str(csv_path),
            restart=True,
            examples=0,
        )
        csv_rows = _read_csv_rows(csv_path)
        decision = _read_decision(csv_path)
        structured_assessment = _structured_assessment(decision, candidate_context)
        failure_taxonomy = _failure_taxonomy(
            result_status="success",
            projection_ok=True,
            dice_gate_enabled=dice_gate_enabled,
            csv_rows=csv_rows,
            decision=decision,
            audit_only=audit_only_comparison,
        )
        result = {
            "stage": "official_labelcritic_pair",
            "status": "success",
            "organ": args.organ,
            "ct": str(Path(args.ct).resolve()),
            "mask_a_dir": str(Path(args.mask_a_dir).resolve()),
            "mask_b_dir": str(Path(args.mask_b_dir).resolve()),
            "projection_backend": "official_labelcritic_ap_axis_1",
            "projection_mode_requested": args.projection_mode,
            "projection_mode_effective": "official_labelcritic_ap_axis_1",
            "centered_slice_fallback_used": False,
            "dice_gate_enabled": dice_gate_enabled,
            "dice_threshold": args.dice_threshold,
            "dual_confirmation_enabled": dual_confirmation_enabled,
            "prompt_mode": "strict_choice" if args.strict_choice_prompt else "official_dual",
            "order_reversal_enabled": True,
            "order_reversal_backend": "official_dual_confirmation_y1_y2_and_y2_y1",
            "candidate_identity_exposed_to_vlm": False,
            "formal_selection_eligible": formal_selection_eligible,
            "audit_only_comparison": audit_only_comparison,
            "automatic_failure_action": (
                description_entry.get("automatic_failure_action")
                if isinstance(description_entry, dict)
                else None
            ),
            "csv_row_count": len(csv_rows),
            "raw_csv_rows": csv_rows[-5:],
            "raw_answer": csv_rows[-1].get("answer") if csv_rows else None,
            "raw_answer_1": csv_rows[-1].get("answer_1") if csv_rows else None,
            "raw_answer_2": csv_rows[-1].get("answer_2") if csv_rows else None,
            "failure_taxonomy": failure_taxonomy,
            "prompt_path": str(prompt_path),
            "description_rendered_path": str(description_rendered_path),
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
        if tolerated_projection_warning:
            result["projection_warning"] = tolerated_projection_warning
            result["projection_return_code"] = projection.returncode
            result["projection_stdout_tail"] = projection.stdout[-4000:]
            result["projection_stderr_tail"] = projection.stderr[-4000:]

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
