#!/usr/bin/env python3
"""Build deterministic mask-corruption tasks for 373-class LabelCritic audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def mutations(mask: np.ndarray) -> dict[str, np.ndarray]:
    src = mask > 0
    shape = src.shape
    shifted = np.roll(src, max(1, shape[0] // 6), axis=0)
    shifted[: max(1, shape[0] // 6)] = False
    fragmented = src.copy()
    fragmented[:, :, shape[2] // 3: shape[2] // 3 + max(1, shape[2] // 8)] = False
    truncated = src.copy()
    truncated[:, :, shape[2] // 2:] = False
    over = src.copy()
    try:
        from scipy.ndimage import binary_dilation
        over = binary_dilation(src, iterations=max(1, min(shape) // 20))
    except Exception:
        pass
    rng = np.random.default_rng(373)
    false_positive = src | (rng.random(shape) < 0.002)
    return {
        "shifted": shifted,
        "middle_missing": fragmented,
        "superior_inferior_truncated": truncated,
        "oversegmented": over,
        "random_false_positive": false_positive,
        "empty": np.zeros(shape, dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--organ", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = nib.load(args.mask)
    source = np.asanyarray(image.dataobj) > 0
    tasks = []
    for kind, array in mutations(source).items():
        path = output / f"{args.organ}__{kind}.nii.gz"
        nib.save(nib.Nifti1Image(array.astype(np.uint8), image.affine, image.header), path)
        tasks.append({
            "case_id": args.case_id,
            "organ": args.organ,
            "ct": str(Path(args.ct).resolve()),
            "candidate_a": str(Path(args.mask).resolve()),
            "candidate_b": str(path),
            "corruption": kind,
            "expected": "a",
            "required_checks": ["dual_confirmation", "order_reversal"],
            "failure_action": "abstain_and_runtime_only_unverified",
        })
    result = {
        "stage": "labelcritic_373_class_agnostic_regression_tasks",
        "status": "tasks_built",
        "human_judgment_required": False,
        "tasks": tasks,
        "note": "Task construction is not validation. A class is promoted only after official pairwise execution passes.",
    }
    path = output / "regression_tasks.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
