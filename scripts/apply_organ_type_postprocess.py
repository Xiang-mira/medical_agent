#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))
from cli_anything.medai.core.student_postprocess import process_student_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply conservative organ-type-aware postprocess to binary NIfTI masks.")
    ap.add_argument("--input-root", type=Path, required=True, help="case_id/*.nii.gz masks")
    ap.add_argument("--output-root", type=Path, required=True, help="postprocessed case_id/*.nii.gz masks")
    ap.add_argument("--case-list", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "configs/organ_taxonomy.json")
    ap.add_argument(
        "--parent-root", type=Path, action="append", default=[],
        help="Teacher/selected-mask root used to construct anatomical parent ROIs; repeatable.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def organ_group(organ: str) -> str:
    s = organ.lower()
    if any(x in s for x in ["vein", "artery", "vessel", "duct", "cava", "aorta", "postcava"]):
        return "vessel_or_duct"
    if any(x in s for x in ["stomach", "colon", "duodenum", "bowel", "intestine", "rectum"]):
        return "hollow_gi"
    if any(x in s for x in ["rib", "femur", "bone", "clavicula", "scapula", "sternum", "vertebra"]):
        return "bone_or_fragmented_structure"
    if any(x in s for x in ["liver", "spleen", "kidney", "pancreas", "lung", "adrenal"]):
        return "core_large_organ"
    if any(x in s for x in ["bladder", "prostate", "uterus", "gonad"]):
        return "partial_fov"
    return "longtail_uncertain"


def case_ids(case_list: Path | None, input_root: Path) -> list[str]:
    if case_list and case_list.exists():
        with open(case_list, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            col = "case_id" if "case_id" in rows[0] else next(iter(rows[0]))
            return [str(row.get(col) or "") for row in rows if row.get(col)]
    return sorted(p.name for p in input_root.iterdir() if p.is_dir())


def _component_filter(mask: np.ndarray, *, keep_top_k: int | None, min_fraction: float, min_voxels: int) -> tuple[np.ndarray, dict[str, Any]]:
    vox = int(mask.sum())
    if vox == 0 or ndi is None:
        return mask, {"components_before": None if ndi is None else 0, "components_after": None if ndi is None else 0, "removed_voxels": 0}
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask, {"components_before": int(n), "components_after": int(n), "removed_voxels": 0}
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    keep: list[int] = []
    for rank, idx in enumerate(order):
        size = int(sizes[idx])
        if keep_top_k is not None and rank >= keep_top_k:
            continue
        if size < min_voxels and size / max(vox, 1) < min_fraction:
            continue
        keep.append(int(idx + 1))
    if not keep and len(order):
        keep = [int(order[0] + 1)]
    out = np.isin(lab, keep)
    return out, {"components_before": int(n), "components_after": len(keep), "removed_voxels": int(vox - out.sum())}


def postprocess(mask: np.ndarray, group: str) -> tuple[np.ndarray, dict[str, Any]]:
    mask = mask > 0
    if group == "core_large_organ":
        return _component_filter(mask, keep_top_k=2, min_fraction=0.01, min_voxels=128)
    if group == "hollow_gi":
        return _component_filter(mask, keep_top_k=5, min_fraction=0.003, min_voxels=64)
    if group == "vessel_or_duct":
        # Preserve thin, possibly multi-component anatomy; only remove tiny isolated blobs.
        return _component_filter(mask, keep_top_k=None, min_fraction=0.001, min_voxels=16)
    if group == "bone_or_fragmented_structure":
        return _component_filter(mask, keep_top_k=None, min_fraction=0.0005, min_voxels=32)
    return _component_filter(mask, keep_top_k=8, min_fraction=0.002, min_voxels=64)


def main() -> int:
    args = parse_args()
    parent_roots = [p.resolve() for p in args.parent_root]
    if not parent_roots:
        parent_roots = [args.input_root.resolve()]
    manifest = process_student_root(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        parent_roots=parent_roots,
        taxonomy_path=args.taxonomy.resolve(),
        policy_path=args.policy.resolve(),
        case_list=args.case_list.resolve() if args.case_list else None,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_cases=args.max_cases,
        write_roi_masks=True,
    )
    manifest["entrypoint"] = "apply_organ_type_postprocess.py"
    manifest["description"] = "Anatomy-aware parent-ROI containment followed by organ-specific component filtering."
    (args.output_root / "organ_type_postprocess_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
