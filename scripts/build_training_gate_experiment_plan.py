#!/usr/bin/env python3
"""Write a CPU-only plan for pseudo-label quality-gating experiments.

This script does not train, run inference, call Qwen, load embeddings, import
torch, or touch CUDA. It records the intended manifest filters and metrics so
the GPU experiments can be launched later in a controlled window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "all_pseudo_labels",
        "purpose": "Baseline showing what happens without confidence quality gating.",
        "manifest_filter_rule": "Include all pseudo-labels with supported schema and positive training_weight; record this as a risky baseline only.",
        "expected_included_grades": ["A", "B", "C"],
        "target_type_policy": "hard A/B and any legacy positive-weight C allowed only for the baseline manifest copy.",
        "training_weight_policy": "Use recorded training_weight; do not change source manifest.",
    },
    {
        "name": "only_A_B",
        "purpose": "Primary high-quality pseudo-label student training condition.",
        "manifest_filter_rule": "Include grade A/B hard targets only.",
        "expected_included_grades": ["A", "B"],
        "target_type_policy": "hard only",
        "training_weight_policy": "A=1.0, B=0.5 from AutoLabelCore config.",
    },
    {
        "name": "A_B_plus_soft_C",
        "purpose": "Test whether uncertainty-aware soft C adds useful signal without corrupting hard supervision.",
        "manifest_filter_rule": "Include A/B hard plus C only when target_type=soft and probability_mask_path exists.",
        "expected_included_grades": ["A", "B", "C"],
        "target_type_policy": "A/B hard; C soft probability masks only.",
        "training_weight_policy": "A=1.0, B=0.5, C soft=0.1.",
    },
    {
        "name": "A_only",
        "purpose": "Upper-precision, lower-coverage ablation.",
        "manifest_filter_rule": "Include grade A hard targets only.",
        "expected_included_grades": ["A"],
        "target_type_policy": "hard only",
        "training_weight_policy": "A=1.0.",
    },
]


METRICS = [
    "student_pseudo_consistency_dice",
    "student_dice_when_expert_reference_available",
    "convergence_speed",
    "empty_mask_rate",
    "wrong_organ_or_laterality_error_count",
    "C_D_failure_mining_count",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Student Training Quality-Gate Experiment Plan",
        "",
        "This is a plan only. It does not run training, Qwen, embeddings, inference, or CUDA.",
        "",
        "## Experiments",
    ]
    for exp in report["experiments"]:
        lines.extend([
            f"### {exp['name']}",
            exp["purpose"],
            f"- Filter: `{exp['manifest_filter_rule']}`",
            f"- Target policy: `{exp['target_type_policy']}`",
            f"- Weight policy: `{exp['training_weight_policy']}`",
            f"- Output dir: `{exp['output_dir']}`",
            "",
        ])
    lines.append("## Metrics")
    for metric in report["metrics_to_compare"]:
        lines.append(f"- `{metric}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CPU-only student quality-gate experiment plan.")
    ap.add_argument("--manifest", default="", help="Optional source manifest path for provenance only.")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audits/training_gate_experiments"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    experiments = []
    for exp in EXPERIMENTS:
        experiments.append({
            **exp,
            "source_manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "output_dir": str(out / exp["name"]),
            "execution_status": "plan_only_not_run_gpu_training_or_embedding",
        })
    report = {
        "stage": "training_gate_experiment_plan",
        "status": "success",
        "gpu_policy": "do_not_run_until_gpu_is_free_and_explicitly_requested",
        "experiments": experiments,
        "metrics_to_compare": METRICS,
    }
    write_json(out / "training_gate_experiment_plan.json", report)
    write_markdown(out / "training_gate_experiment_plan.md", report)
    print(json.dumps({"status": "success", "output_dir": str(out), "num_experiments": len(experiments)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
