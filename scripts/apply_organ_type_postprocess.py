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

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply conservative organ-type-aware postprocess to binary NIfTI masks.")
    ap.add_argument("--input-root", type=Path, required=True, help="case_id/*.nii.gz masks")
    ap.add_argument("--output-root", type=Path, required=True, help="postprocessed case_id/*.nii.gz masks")
    ap.add_argument("--case-list", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=ROOT / "configs/organ_postprocess_policy.yaml")
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
    policy = load_yaml(args.policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    processed = 0
    copied_meta = 0
    for case_id in case_ids(args.case_list, args.input_root):
        in_case = args.input_root / case_id
        if not in_case.exists():
            continue
        out_case = args.output_root / case_id
        out_case.mkdir(parents=True, exist_ok=True)
        for src in sorted(in_case.glob("*.nii.gz")):
            dst = out_case / src.name
            if dst.exists() and not args.overwrite:
                continue
            organ = src.name[:-7]
            group = organ_group(organ)
            img = sitk.ReadImage(str(src))
            arr = sitk.GetArrayFromImage(img)
            pp, stats = postprocess(arr, group)
            out = sitk.GetImageFromArray(pp.astype(np.uint8))
            out.CopyInformation(img)
            sitk.WriteImage(out, str(dst))
            processed += 1
            rows.append({
                "case_id": case_id,
                "organ": organ,
                "organ_group": group,
                "input_path": str(src),
                "output_path": str(dst),
                "voxels_before": int((arr > 0).sum()),
                "voxels_after": int(pp.sum()),
                **stats,
            })
        for meta in ["voxtell_student_result.json"]:
            src_meta = in_case / meta
            if src_meta.exists():
                shutil.copy2(src_meta, out_case / meta)
                copied_meta += 1
    manifest = {
        "status": "success",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "policy": str(args.policy),
        "policy_version": policy.get("version"),
        "processed_masks": processed,
        "copied_metadata_files": copied_meta,
        "description": "Conservative organ-type postprocess: hollow GI keeps top-k components; vessel/duct preserves thin structures and removes only tiny isolated blobs; bones allow multiple components.",
    }
    (args.output_root / "organ_type_postprocess_summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with open(args.output_root / "organ_type_postprocess_per_mask.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
