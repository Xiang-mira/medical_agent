#!/usr/bin/env python3
"""CPU-only prompt governance report for the 373-organ prompt system.

This script reorganizes existing prompt artifacts into a human-readable taxonomy
and writes a free-form prompt safety contract. It does not call Qwen, embedding
models, inference, training, torch, CUDA, or any network service.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.prompt_governance import (
    FREEFORM_PROMPT_SAFETY_CONTRACT,
    PROMPT_TAXONOMY,
    classify_prompt,
)

DEFAULT_TARGET_CONFIG = ROOT / "configs/student_3d_prompt_target_organs.json"

EMBEDDING_SANITY_CHECK_TEMPLATE = [
    {
        "organ": "liver",
        "prompt_a_role": "original_liver",
        "prompt_b_role": "expanded_liver",
        "prompt_a": "segment the liver",
        "prompt_b": "segment the liver, a large organ in the right upper abdomen inferior to the diaphragm",
        "expected_distance": "near",
        "expected_similarity_order": "highest_similarity_pair_for_liver",
    },
    {
        "organ": "liver",
        "prompt_a_role": "original_liver",
        "prompt_b_role": "negative_pancreas",
        "prompt_a": "segment the liver",
        "prompt_b": "segment the pancreas",
        "expected_distance": "far",
        "expected_similarity_order": "lower_than_liver_expansion_pair",
    },
    {
        "organ": "liver",
        "prompt_a_role": "expanded_liver",
        "prompt_b_role": "negative_pancreas",
        "prompt_a": "segment the liver, a large organ in the right upper abdomen inferior to the diaphragm",
        "prompt_b": "segment the pancreas",
        "expected_distance": "far",
        "expected_similarity_order": "lower_than_liver_expansion_pair",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_prompt_taxonomy_rows(config: dict[str, Any], max_examples_per_subclass: int = 0) -> list[dict[str, Any]]:
    variants = config.get("prompt_variants") or {}
    organ_to_prompt = config.get("organ_to_prompt") or {}
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    for organ in config.get("target_organs", []):
        canonical = str(organ_to_prompt.get(organ) or f"segment the {str(organ).replace('_', ' ')}")
        prompts = [canonical, *[str(x) for x in variants.get(organ, [])]]
        seen: set[str] = set()
        for prompt in prompts:
            if prompt in seen:
                continue
            seen.add(prompt)
            level, subclass = classify_prompt(prompt, canonical, str(organ))
            key = (level, subclass)
            counts[key] = counts.get(key, 0) + 1
            if max_examples_per_subclass <= 0 or counts[key] <= max_examples_per_subclass:
                rows.append({
                    "organ": organ,
                    "prompt": prompt,
                    "prompt_level": level,
                    "prompt_subclass": subclass,
                    "prompt_source": "existing_config",
                    "validation_status": "fixed_prompt_assumed_valid_pending_human_spot_check",
                    "full_subclass_count": counts[key],
                })
    return rows


def expected_subclass_keys() -> list[str]:
    return [f"{level}::{subclass}" for level, info in PROMPT_TAXONOMY.items() for subclass in info["subclasses"]]


def build_summary(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    subclass_counts: dict[str, int] = {}
    for row in rows:
        counts[row["prompt_level"]] = max(counts.get(row["prompt_level"], 0), int(row.get("full_subclass_count") or 1))
        key = f"{row['prompt_level']}::{row['prompt_subclass']}"
        subclass_counts[key] = max(subclass_counts.get(key, 0), int(row.get("full_subclass_count") or 1))
    missing_subclasses = [key for key in expected_subclass_keys() if key not in subclass_counts]
    total_sampled = sum(subclass_counts.values()) or 1
    overrepresented_subclasses = {key: count for key, count in subclass_counts.items() if count / total_sampled >= 0.40}
    return {
        "stage": "prompt_governance_report",
        "status": "success",
        "target_count": len(config.get("target_organs", [])),
        "taxonomy": PROMPT_TAXONOMY,
        "prompt_counts_by_level": counts,
        "prompt_counts_by_subclass": subclass_counts,
        "missing_prompt_subclasses": missing_subclasses,
        "overrepresented_prompt_subclasses": overrepresented_subclasses,
        "freeform_prompt_safety_contract": FREEFORM_PROMPT_SAFETY_CONTRACT,
        "embedding_sanity_check_status": "plan_only_not_run_cpu_gpu_or_qwen",
        "embedding_sanity_check_required_pairs": EMBEDDING_SANITY_CHECK_TEMPLATE,
        "report_warning": "This report reorganizes existing fixed prompts and safety rules only; it does not generate or accept new free-form prompts. Embedding sanity checks are plan-only until a GPU/embedding-safe window.",
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Prompt System Governance Report", "",
        "Our prompt system is organized into three user-oriented levels: AI researcher prompts, clinical expert prompts, and natural user prompts.",
        "Free-form prompts are accepted only after identity, laterality, anatomy, and CT scan-range validation.", "",
        "## Three-Level Taxonomy",
    ]
    for level, info in PROMPT_TAXONOMY.items():
        lines.append(f"### {level}")
        lines.append(info["purpose"])
        for subclass, desc in info["subclasses"].items():
            example = next((r for r in rows if r["prompt_level"] == level and r["prompt_subclass"] == subclass), None)
            suffix = f" Example: `{example['prompt']}`" if example else ""
            lines.append(f"- `{subclass}`: {desc}{suffix}")
        lines.append("")
    lines.extend([
        "## Free-Form Prompt Safety Boundary",
        "1. Input organ token, canonical name, region, CT appearance, synonyms, laterality, allowed adjacent anatomy, and scan range.",
        "2. Qwen may generate a JSON prompt only in an explicit replay/generation step.",
        "3. Validate JSON schema, organ identity, laterality, introduced anatomy, and CT scan range.",
        "4. On any failure, fallback to fixed prompt and save `prompt_source`, `validation_status`, and `fallback_reason`.", "",
        "## Embedding Sanity Check Plan",
        "This report does not run Qwen embedding. It writes the required pairs for a later GPU/embedding-safe window.",
    ])
    for pair in EMBEDDING_SANITY_CHECK_TEMPLATE:
        lines.append(f"- {pair['prompt_a_role']} vs {pair['prompt_b_role']}: expected `{pair['expected_distance']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build CPU-only prompt taxonomy and free-form prompt safety report.")
    ap.add_argument("--target-config", default=str(DEFAULT_TARGET_CONFIG))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/audits/prompt_governance"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(Path(args.target_config).resolve())
    out = Path(args.output_dir).resolve()
    rows = build_prompt_taxonomy_rows(config)
    summary = build_summary(config, rows)
    write_json(out / "prompt_system_taxonomy.json", {"taxonomy": PROMPT_TAXONOMY, "examples": rows})
    write_csv(out / "prompt_system_taxonomy_examples.csv", rows)
    write_json(out / "freeform_prompt_safety_contract.json", FREEFORM_PROMPT_SAFETY_CONTRACT)
    write_json(out / "prompt_embedding_sanity_check_plan.json", EMBEDDING_SANITY_CHECK_TEMPLATE)
    write_csv(out / "prompt_embedding_sanity_check_plan.csv", EMBEDDING_SANITY_CHECK_TEMPLATE)
    write_json(out / "summary.json", summary)
    write_markdown(out / "prompt_governance_report.md", summary, rows)
    print(json.dumps({"status": "success", "output_dir": str(out), "target_count": summary["target_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
