#!/usr/bin/env python3
"""Formal 50-case x 373-target hierarchical Round1 repair."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop
from cli_anything.medai.core.target_space import validate_formal_373_target_space


def preseed_roots(old_estep: Path) -> dict[str, Path]:
    cases_root = old_estep / "cases"
    models: set[str] = set()
    for case_dir in cases_root.iterdir() if cases_root.exists() else []:
        raw = case_dir / "raw_predictions"
        if raw.exists():
            models.update(path.name for path in raw.iterdir() if path.is_dir())
    return {model: cases_root / "{case_id}" / "raw_predictions" / model for model in sorted(models)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--old-estep", default=str(ROOT / "outputs/stage4b_round1_50cases_20260611/round1/estep"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/round1_373_hierarchical_repair_20260620/estep"))
    ap.add_argument("--critic-backend", choices=["stub", "labelcritic"], default="labelcritic")
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-shapekit", action="store_true")
    args = ap.parse_args()

    target_config = ROOT / "configs/student_3d_prompt_target_organs.json"
    target_doc = json.loads(target_config.read_text(encoding="utf-8"))
    organs = [str(x) for x in target_doc["target_organs"]]
    validation = validate_formal_373_target_space(target_config=target_config, requested_organs=organs, require_full_target=True)
    if validation.get("status") != "success":
        raise SystemExit(json.dumps(validation, indent=2, ensure_ascii=False))

    routing_path = ROOT / "outputs/audit_21_models/routing_373_audit.json"
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    if routing.get("status") != "success":
        raise SystemExit("373 routing audit is not successful")
    routed_models = sorted({
        str(candidate["model_key"])
        for entry in routing["per_organ"].values()
        for candidate in entry.get("candidates", [])
        if candidate.get("resolvable")
    })

    case_list = Path(args.case_list).resolve()
    with case_list.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))[: max(1, min(args.num_cases, 50))]
    run_case_list = case_list
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(cases) != 50:
        run_case_list = output.parent / f"case_list_{len(cases)}.csv"
        with run_case_list.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)

    old_estep = Path(args.old_estep).resolve()
    preseeded = preseed_roots(old_estep)
    models = [model for model in routed_models if model in preseeded]
    missing_cached_route = [
        organ for organ, entry in routing["per_organ"].items()
        if not any(
            candidate.get("resolvable") and candidate.get("model_key") in models
            for candidate in entry.get("candidates", [])
        )
    ]
    if missing_cached_route:
        raise SystemExit(f"Repair cache has no routed model for targets: {missing_cached_route}")
    config = {
        "stage": "full_373_hierarchical_repair",
        "case_list": str(run_case_list),
        "num_cases": len(cases),
        "num_organs": len(organs),
        "models": models,
        "excluded_uncached_routed_models": sorted(set(routed_models) - set(models)),
        "old_estep": str(old_estep),
        "preseeded_models": sorted(preseeded),
        "preseeded_parent_only": True,
        "old_child_policy": "quarantined_from_candidates_and_mstep",
        "teacher_inference_mode": "hierarchical_roi",
        "critic_backend": args.critic_backend,
        "formal_experiment_eligible": args.critic_backend == "labelcritic",
        "formal_exclusion_reason": None if args.critic_backend == "labelcritic" else "stub LabelCritic is debug-only and cannot produce formal manifests",
        "auto_arbitration": args.critic_backend == "labelcritic",
        "enable_shapekit": not args.no_shapekit,
        "fusion_mode": "disabled_formal_mainline",
    }
    (output.parent / "repair_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    result = run_multimodel_annotation_loop(
        case_list=run_case_list,
        output_folder=output,
        models=models,
        organs=organs,
        registry_path=ROOT / "configs/model_registry.yaml",
        checkpoint_map_models=False,
        shapekit_root=ROOT / "third_party/ShapeKit-main",
        enable_shapekit=not args.no_shapekit,
        enable_critic=True,
        critic_backend=args.critic_backend,
        dry_run=False,
        timeout_sec=args.timeout_sec,
        device=args.device,
        resume=True,
        preseeded_model_dirs=preseeded,
        preseeded_parent_only=True,
        teacher_inference_mode="hierarchical_roi",
        roi_margin_mm=20.0,
        enable_fusion=False,
        fusion_method="weighted_vote",
        enable_auto_arbitration=args.critic_backend == "labelcritic",
        candidate_mode="route_pruned_with_competition",
    )
    print(json.dumps({
        "status": result.get("status"),
        "output": str(output),
        "num_cases": len(cases),
        "num_organs": len(organs),
        "training_manifest": str(output / "training_manifest.json"),
    }, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
