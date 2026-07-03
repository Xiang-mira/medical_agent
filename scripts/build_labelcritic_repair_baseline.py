#!/usr/bin/env python3
"""Freeze the pre-repair evidence boundary for the LabelCritic/373 project."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = (
        ROOT
        / "outputs"
        / "labelcritic_teacher_audit_PanTS_00000145_20260703"
        / "summary.json"
    )
    experiment_doc = json.loads(experiment.read_text()) if experiment.exists() else {}
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = [
        ROOT / "checkpoints/VoxTell/voxtell_v1.1/fold_0/checkpoint_final.pth",
        ROOT / "checkpoints/VoxTell/voxtell_v1.1/plans.json",
        ROOT / "checkpoints/VoxTell/embeddings/voxtell_v1.1/text_embeddings.npz",
        ROOT / "configs/organ_ct_appearance_373.json",
        ROOT / "configs/student_3d_prompt_target_organs.json",
    ]
    report = {
        "stage": "labelcritic_373_repair_baseline",
        "status": "frozen",
        "git_status_return_code": status.returncode,
        "dirty_paths": status.stdout.splitlines(),
        "asset_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in tracked
        },
        "legacy_single_mask_experiment": {
            "status": "adapter_diagnostic_only",
            "path": str(experiment),
            "num_organs": experiment_doc.get("num_organs"),
            "num_candidates": experiment_doc.get("num_candidates"),
            "grade_counts": experiment_doc.get("grade_counts"),
            "known_invalidity": (
                "Used a project single-mask three-bucket prompt and centered-slice "
                "fallback rather than the official pairwise AP-projection protocol."
            ),
        },
        "known_issues": [
            "single-mask centered-slice fallback",
            "27/30 legacy grades collapsed to 0.65",
            "363 descriptions were project-generated and require validation",
            "short PanTS scans can be overclassified as thorax",
            "family_balanced_consensus can obscure teacher provenance",
            "complete_case_373 means record completeness, not visibility",
        ],
        "evidence_policy": {
            "historical_pseudo": "auxiliary_only",
            "official_benchmark": "required_for_accuracy_claim",
            "human_review": "not_used",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
