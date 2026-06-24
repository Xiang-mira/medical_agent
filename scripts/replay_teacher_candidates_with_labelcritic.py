#!/usr/bin/env python3
"""Replay existing teacher raw predictions through real candidate selection.

This is useful after an expensive teacher subset has already produced
`raw_predictions/<model>/<case>/segmentations`. The script creates a lightweight
preseeded layout and reruns the shared E-step selection path:

existing teacher masks -> candidate collection -> LabelCritic -> ShapeKit
-> auditable manifest.

It does not train or rerun teacher models.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop


def parse_args() -> argparse.Namespace:
    default_base = ROOT / "outputs/audit_21_models/small_loop_2case_multiteacher_10organ_shapekit"
    ap = argparse.ArgumentParser(description="Replay existing teacher candidates through LabelCritic.")
    ap.add_argument("--source-estep", default=str(default_base / "estep"))
    ap.add_argument("--case-list", default=str(default_base / "case_list_subset.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audit_21_models/replay_teacher_candidates_real_labelcritic"))
    ap.add_argument(
        "--models",
        default="epai_20250421,vsmtrans,cads551,cads552,cads553,cads554,nnunet_private",
        help="Comma-separated existing raw_prediction model keys to replay.",
    )
    ap.add_argument(
        "--organs",
        default="liver,pancreas,spleen,kidney_left,kidney_right,aorta,gall_bladder,vertebrae_L1,femur_left,esophagus",
    )
    ap.add_argument("--critic-backend", default="labelcritic", choices=["stub", "labelcritic"])
    ap.add_argument("--critic-base-url", default="http://localhost")
    ap.add_argument("--critic-port", type=int, default=8000)
    ap.add_argument("--timeout-sec", type=int, default=600)
    ap.add_argument("--max-critic-decisions", type=int, default=0, help="Optional cap for quick debugging; 0 means no cap.")
    ap.add_argument("--labelcritic-no-dice-check", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--labelcritic-no-dual-confirmation", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--labelcritic-simple-prompt-ablation", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--labelcritic-conservative-dual", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--labelcritic-skip-organ-presence-gate", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--labelcritic-strict-choice-prompt", action="store_true", default=False, help="Diagnostic LabelCritic mode.")
    ap.add_argument("--no-shapekit", action="store_true", help="Debug only; formal checks should keep ShapeKit on.")
    return ap.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _case_ids_from_case_list(case_list: Path) -> list[str]:
    import csv

    with case_list.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["case_id"].strip() for row in csv.DictReader(f) if row.get("case_id")]


def build_preseeded_layout(
    *,
    source_estep: Path,
    output_dir: Path,
    case_ids: list[str],
    models: list[str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Create <model>/<case>/segmentations symlinks/copies for preseeded input."""
    seed_root = output_dir / "preseeded_teachers"
    if seed_root.exists():
        shutil.rmtree(seed_root)
    seed_root.mkdir(parents=True, exist_ok=True)

    preseeded: dict[str, Path] = {}
    model_stats: dict[str, Any] = {}
    for model in models:
        model_base = seed_root / model
        copied_cases = 0
        copied_masks = 0
        missing_cases: list[str] = []
        for case_id in case_ids:
            src = source_estep / "cases" / case_id / "raw_predictions" / model / case_id / "segmentations"
            dst = model_base / case_id / "segmentations"
            if not src.exists() or not any(src.glob("*.nii.gz")):
                missing_cases.append(case_id)
                continue
            dst.mkdir(parents=True, exist_ok=True)
            copied_cases += 1
            for mask in src.glob("*.nii.gz"):
                target = dst / mask.name
                try:
                    target.symlink_to(mask.resolve())
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(mask, target)
                copied_masks += 1
        if copied_cases:
            preseeded[model] = model_base
        model_stats[model] = {
            "preseeded_cases": copied_cases,
            "preseeded_masks": copied_masks,
            "missing_cases": missing_cases,
        }
    return preseeded, {"seed_root": str(seed_root), "models": model_stats}


def cap_organs_by_expected_critic_count(
    *,
    organs: list[str],
    case_ids: list[str],
    preseeded: dict[str, Path],
    max_critic_decisions: int,
) -> tuple[list[str], dict[str, Any]]:
    """Limit replay scope for quick LabelCritic debugging.

    The shared E-step compares candidate masks pairwise only when more than one
    model has an organ mask. This helper estimates those multi-candidate
    comparisons from the preseeded folders and truncates the organ list before
    running the loop.
    """
    if max_critic_decisions <= 0:
        return organs, {"applied": False, "reason": "max_critic_decisions <= 0"}

    kept: list[str] = []
    estimated = 0
    per_organ: list[dict[str, Any]] = []
    for organ in organs:
        organ_estimated = 0
        for case_id in case_ids:
            candidate_count = 0
            for model_base in preseeded.values():
                if any((
                    (model_base / case_id / f"{organ}.nii.gz").exists(),
                    (model_base / case_id / "updated" / f"{organ}.nii.gz").exists(),
                    (model_base / case_id / "segmentations" / f"{organ}.nii.gz").exists(),
                )):
                    candidate_count += 1
            if candidate_count > 1:
                organ_estimated += candidate_count - 1
        if organ_estimated == 0:
            continue
        if kept and estimated + organ_estimated > max_critic_decisions:
            break
        kept.append(organ)
        estimated += organ_estimated
        per_organ.append({"organ": organ, "estimated_pairwise_decisions": organ_estimated})
        if estimated >= max_critic_decisions:
            break

    if not kept:
        kept = organs[:1]
    return kept, {
        "applied": True,
        "requested_max_critic_decisions": max_critic_decisions,
        "original_organs": organs,
        "kept_organs": kept,
        "estimated_pairwise_decisions": estimated,
        "per_organ": per_organ,
    }


def main() -> int:
    args = parse_args()
    source_estep = Path(args.source_estep).resolve()
    case_list = Path(args.case_list).resolve()
    out = Path(args.output_dir).resolve()
    models = [x.strip() for x in args.models.replace(";", ",").split(",") if x.strip()]
    organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()]
    case_ids = _case_ids_from_case_list(case_list)

    preseeded, preseed_summary = build_preseeded_layout(
        source_estep=source_estep,
        output_dir=out,
        case_ids=case_ids,
        models=models,
    )
    if not preseeded:
        raise SystemExit(f"No preseeded teacher masks found in {source_estep}")

    organs, organ_cap_summary = cap_organs_by_expected_critic_count(
        organs=organs,
        case_ids=case_ids,
        preseeded=preseeded,
        max_critic_decisions=args.max_critic_decisions,
    )

    # Do not run any heavy model; all candidates come from the existing raw
    # prediction folders. The caller can include mock or real teachers in
    # preseeded models if desired.
    result = run_multimodel_annotation_loop(
        case_list=case_list,
        output_folder=out / "estep",
        models=[],
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=not args.no_shapekit,
        enable_critic=True,
        critic_backend=args.critic_backend,
        critic_base_url=args.critic_base_url,
        critic_port=args.critic_port,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        device="cuda",
        resume=False,
        teacher_inference_mode="hierarchical_roi",
        preseeded_model_dirs=preseeded,
        labelcritic_options={
            "no_dice_check": args.labelcritic_no_dice_check,
            "no_dual_confirmation": args.labelcritic_no_dual_confirmation,
            "simple_prompt_ablation": args.labelcritic_simple_prompt_ablation,
            "conservative_dual": args.labelcritic_conservative_dual,
            "skip_organ_presence_gate": args.labelcritic_skip_organ_presence_gate,
            "strict_choice_prompt": args.labelcritic_strict_choice_prompt,
        },
    )

    manifest_path = out / "estep/training_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else []
    multi_candidate_items = [
        item for item in manifest
        if len(item.get("candidate_models") or []) > 1
    ]
    selected_models = sorted({str(item.get("selected_model")) for item in manifest})
    shapekit_statuses = sorted({str(item.get("shapekit_status")) for item in manifest})
    status = "success"
    failures: list[str] = []
    if result.get("status") != "success":
        failures.append(f"E-step status is {result.get('status')}")
    if not manifest:
        failures.append("training_manifest.json is empty or missing")
    if not multi_candidate_items:
        failures.append("no multi-candidate manifest items were produced")
    if args.critic_backend == "labelcritic" and int(result.get("total_labelcritic_decisions") or 0) <= 0:
        failures.append("real LabelCritic produced no decisions")
    if not args.no_shapekit and "success" not in shapekit_statuses:
        failures.append("ShapeKit did not succeed for any manifest item")
    if failures:
        status = "failed"

    summary = {
        "stage": "replay_teacher_candidates_with_labelcritic",
        "status": status,
        "failures": failures,
        "source_estep": str(source_estep),
        "case_list": str(case_list),
        "output_dir": str(out),
        "preseed_summary": preseed_summary,
        "organ_cap_summary": organ_cap_summary,
        "models": models,
        "organs": organs,
        "critic_backend": args.critic_backend,
        "labelcritic_options": {
            "no_dice_check": args.labelcritic_no_dice_check,
            "no_dual_confirmation": args.labelcritic_no_dual_confirmation,
            "simple_prompt_ablation": args.labelcritic_simple_prompt_ablation,
            "conservative_dual": args.labelcritic_conservative_dual,
            "skip_organ_presence_gate": args.labelcritic_skip_organ_presence_gate,
            "strict_choice_prompt": args.labelcritic_strict_choice_prompt,
        },
        "enable_shapekit": not args.no_shapekit,
        "estep_status": result.get("status"),
        "total_updated": result.get("total_updated"),
        "total_labelcritic_decisions": result.get("total_labelcritic_decisions"),
        "manifest_items": len(manifest),
        "multi_candidate_items": len(multi_candidate_items),
        "selected_models": selected_models,
        "shapekit_statuses": shapekit_statuses,
        "vlm_decision_lines": _line_count(out / "estep/vlm_decisions.jsonl"),
        "review_queue_lines": _line_count(out / "estep/review_queue.jsonl"),
        "run_summary": str(out / "estep/run_summary.json"),
        "training_manifest": str(manifest_path),
        "accuracy_warning": "This replays pseudo-label candidate selection; it is not true expert-label accuracy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_teacher_candidates_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
