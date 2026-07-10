#!/usr/bin/env python3
"""Synthetic LabelCritic sanity benchmark for pseudo-label masks.

This benchmark does not use GT.  It treats an already selected pseudo-label as
the "known better" member of a pair and compares it against controlled mask
corruptions to test LabelCritic plumbing, prompt quality, and VLM decisiveness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.labelcritic_wrapper import run_labelcritic_compare  # noqa: E402


DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "aorta",
    "adrenal_gland_left", "adrenal_gland_right", "stomach", "duodenum",
    "colon", "small_bowel", "bladder", "lung_left", "lung_right",
    "gall_bladder", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "vertebrae_L1", "vertebrae_T12", "rib_left_1", "rib_right_1",
    "femur_left", "femur_right", "skin", "bones", "fat", "muscle",
    "veins", "subcutaneous_adipose_tissue",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    doc = read_json(path)
    rows = doc.get("items") if isinstance(doc, dict) else doc
    return [row for row in rows if isinstance(row, dict)]


def nonempty_mask(path: Path) -> bool:
    try:
        return bool(np.asanyarray(nib.load(str(path)).dataobj).sum() > 0)
    except Exception:
        return False


def save_like(ref_img: nib.Nifti1Image, arr: np.ndarray, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), ref_img.affine, ref_img.header), str(out))
    return out


def corrupt(mask_path: Path, kind: str, out: Path) -> Path:
    img = nib.load(str(mask_path))
    arr = (np.asanyarray(img.dataobj) > 0).astype(np.uint8)
    if kind == "shift":
        bad = np.roll(arr, shift=max(3, arr.shape[0] // 20), axis=0)
    elif kind == "dilation":
        bad = arr.copy()
        for axis in range(3):
            bad = np.maximum(bad, np.roll(bad, 1, axis=axis))
            bad = np.maximum(bad, np.roll(bad, -1, axis=axis))
    elif kind == "erosion":
        bad = arr.copy()
        for axis in range(3):
            bad = np.minimum(bad, np.roll(bad, 1, axis=axis))
            bad = np.minimum(bad, np.roll(bad, -1, axis=axis))
    elif kind == "partial_delete":
        bad = arr.copy()
        coords = np.argwhere(bad > 0)
        if coords.size:
            cutoff = np.median(coords[:, 0])
            bad[: int(cutoff), :, :] = 0
    elif kind == "random_blob":
        bad = np.zeros_like(arr, dtype=np.uint8)
        center = tuple(max(1, s // 2) for s in arr.shape)
        radius = max(2, min(arr.shape) // 12)
        slices = tuple(slice(max(0, c - radius), min(s, c + radius)) for c, s in zip(center, arr.shape))
        bad[slices] = 1
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return save_like(img, bad, out)


def candidate_rows(rows: list[dict[str, Any]], organs: set[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        organ = str(row.get("organ") or "")
        if organ not in organs:
            continue
        if str(row.get("supervision_type") or "").lower() != "positive":
            continue
        if row.get("distillation_eligible") is False:
            continue
        mask = Path(str(row.get("mask_path") or row.get("mask") or ""))
        ct = Path(str(row.get("ct_path") or row.get("image") or ""))
        if mask.exists() and ct.exists() and nonempty_mask(mask):
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--organs", nargs="*", default=DEFAULT_ORGANS)
    ap.add_argument("--max-per-organ", type=int, default=1)
    ap.add_argument("--corruptions", nargs="*", default=["shift", "dilation", "erosion", "random_blob", "partial_delete"])
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = candidate_rows(load_manifest_rows(args.manifest), set(args.organs))
    selected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for row in rows:
        organ = str(row.get("organ"))
        if seen.get(organ, 0) >= args.max_per_organ:
            continue
        seen[organ] = seen.get(organ, 0) + 1
        selected.append(row)

    results: list[dict[str, Any]] = []
    for row in selected:
        organ = str(row["organ"])
        case_id = str(row.get("case_id") or "unknown")
        good = Path(str(row.get("mask_path") or row.get("mask"))).resolve()
        ct = Path(str(row.get("ct_path") or row.get("image"))).resolve()
        for kind in args.corruptions:
            bad = corrupt(good, kind, args.output_dir / "corruptions" / case_id / organ / f"{kind}.nii.gz")
            if args.dry_run:
                results.append({"case_id": case_id, "organ": organ, "corruption": kind, "status": "dry_run"})
                continue
            ab = run_labelcritic_compare(
                ct, good, bad, organ,
                args.output_dir / "comparisons" / case_id / organ / f"{kind}_good_vs_bad.json",
                backend="labelcritic", base_url=args.base_url, port=args.port,
                timeout_sec=args.timeout_sec, no_dice_check=True,
                candidate_context=[
                    {"candidate_id": "known_better_selected_pseudo_label", "model": "selected_pseudo_label"},
                    {"candidate_id": f"controlled_corruption_{kind}", "model": "synthetic_corruption"},
                ],
            )
            ba = run_labelcritic_compare(
                ct, bad, good, organ,
                args.output_dir / "comparisons" / case_id / organ / f"{kind}_bad_vs_good.json",
                backend="labelcritic", base_url=args.base_url, port=args.port,
                timeout_sec=args.timeout_sec, no_dice_check=True,
                candidate_context=[
                    {"candidate_id": f"controlled_corruption_{kind}", "model": "synthetic_corruption"},
                    {"candidate_id": "known_better_selected_pseudo_label", "model": "selected_pseudo_label"},
                ],
            )
            ab_w = (ab.get("decision") or {}).get("winner")
            ba_w = (ba.get("decision") or {}).get("winner")
            consistent = ab_w == "uncertain" or ba_w == "uncertain" or (ab_w == "a" and ba_w == "b") or (ab_w == "b" and ba_w == "a")
            known_better = ab_w == "a" and ba_w == "b"
            results.append({
                "case_id": case_id,
                "organ": organ,
                "corruption": kind,
                "ab_status": ab.get("status"),
                "ba_status": ba.get("status"),
                "ab_winner": ab_w,
                "ba_winner": ba_w,
                "ab_ba_consistent": consistent,
                "known_better_picked": known_better,
                "ab_failure_taxonomy": ab.get("failure_taxonomy", []),
                "ba_failure_taxonomy": ba.get("failure_taxonomy", []),
            })

    total = len(results)
    non_dry = [r for r in results if r.get("status") != "dry_run"]
    summary = {
        "stage": "labelcritic_synthetic_benchmark",
        "status": "success",
        "manifest": str(args.manifest),
        "num_pairs": total,
        "num_organs": len({r.get("organ") for r in results}),
        "projection_success_rate": None,
        "parser_success_rate": None,
        "decisive_rate": None,
        "ab_ba_consistency_rate": None,
        "known_better_pick_rate": None,
        "results": results,
        "interpretation": "Synthetic pseudo-label corruption benchmark; not GT accuracy.",
    }
    if non_dry:
        summary.update({
            "projection_success_rate": sum(r["ab_status"] == "success" and r["ba_status"] == "success" for r in non_dry) / len(non_dry),
            "parser_success_rate": sum(bool(r["ab_winner"]) and bool(r["ba_winner"]) for r in non_dry) / len(non_dry),
            "decisive_rate": sum(r["ab_winner"] != "uncertain" and r["ba_winner"] != "uncertain" for r in non_dry) / len(non_dry),
            "ab_ba_consistency_rate": sum(bool(r["ab_ba_consistent"]) for r in non_dry) / len(non_dry),
            "known_better_pick_rate": sum(bool(r["known_better_picked"]) for r in non_dry) / len(non_dry),
        })
    write_json(args.output_dir / "labelcritic_synthetic_benchmark_summary.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
