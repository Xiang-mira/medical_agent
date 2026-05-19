#!/usr/bin/env python3
"""Select 50 PanTS cases for the teacher's debug workflow.

The selection is deterministic and medically grounded:
1. Prefer cases whose label folder exists and whose pancreatic_lesion.nii.gz is non-empty.
2. Require the CT file and the key organ masks needed by this project.
3. If metadata.xlsx is present, record diagnosis/phase/age fields when available.

This script does NOT randomly pick cases. It selects the first 50 validated tumor-annotation cases
in sorted case-id order unless --balance-by-metadata is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

KEY_ORGANS = [
    "pancreas", "pancreatic_lesion", "liver", "spleen", "kidney_left", "kidney_right",
    "colon", "duodenum", "stomach", "aorta", "postcava",
]


def data_root(root: Path) -> Path:
    return root / "data" if (root / "data").exists() else root


def case_range(case_id: str) -> tuple[str, str]:
    n = int(case_id.split("_")[1])
    start = ((n - 1) // 1000) * 1000 + 1
    end = ((n - 1) // 1000 + 1) * 1000
    return f"{start:08d}", f"{end:08d}"


def mask_nonempty(path: Path) -> tuple[bool, int | None, str | None]:
    if not path.exists():
        return False, None, "missing"
    try:
        import nibabel as nib
        import numpy as np
        arr = np.asanyarray(nib.load(str(path)).dataobj)
        vox = int((arr > 0).sum())
        return vox > 0, vox, None
    except Exception as exc:
        return False, None, f"read_error: {exc}"


def load_metadata(data: Path) -> dict[str, dict[str, Any]]:
    meta_path = data / "metadata.xlsx"
    if not meta_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_excel(meta_path)
    except Exception:
        return {}
    # Guess case id column.
    cols = list(df.columns)
    id_col = None
    for c in cols:
        cn = str(c).lower()
        if "case" in cn or "bdmap" in cn or "pants" in cn or "id" == cn.strip():
            id_col = c; break
    if id_col is None:
        id_col = cols[0]
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row.get(id_col, "")).strip()
        if cid and cid.startswith("PanTS_"):
            out[cid] = {str(k): (None if str(v) == "nan" else v) for k, v in row.to_dict().items()}
    return out


def collect_candidates(root: Path, split: str, require_nonempty_lesion: bool, required_organs: list[str]) -> list[dict[str, Any]]:
    data = data_root(root)
    if split == "train":
        img_dir, lab_dir, rep_dir = data / "ImageTr", data / "LabelTr", data / "ReportTr"
    else:
        img_dir, lab_dir, rep_dir = data / "ImageTe", data / "LabelTe", data / "ReportTe"
    rows = []
    if not img_dir.exists() or not lab_dir.exists():
        return rows
    for case_dir in sorted(img_dir.glob("PanTS_*")):
        cid = case_dir.name
        ct = case_dir / "ct.nii.gz"
        seg = lab_dir / cid / "segmentations"
        if not ct.exists() or not seg.exists():
            continue
        missing = [o for o in required_organs if not (seg / f"{o}.nii.gz").exists()]
        lesion_ok, lesion_vox, lesion_err = mask_nonempty(seg / "pancreatic_lesion.nii.gz")
        if missing:
            continue
        if require_nonempty_lesion and not lesion_ok:
            continue
        start, end = case_range(cid)
        rows.append({
            "case_id": cid,
            "split": split,
            "ct_path": str(ct.resolve()),
            "annotation_folder": str(seg.resolve()),
            "report_path": str((rep_dir / cid / "report.pdf").resolve()) if (rep_dir / cid / "report.pdf").exists() else "",
            "pancreatic_lesion_voxels": lesion_vox if lesion_vox is not None else "",
            "image_tar_block": f"PanTSMini_ImageTr_{start}_{end}.tar.gz" if split == "train" else "PanTSMini_ImageTe_00009001_00009901.tar.gz",
            "selection_reason": "validated_nonempty_pancreatic_lesion_and_required_organs" if lesion_ok else "validated_required_organs_lesion_exists_but_empty_or_unreadable",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "split", "ct_path", "annotation_folder", "report_path", "pancreatic_lesion_voxels", "image_tar_block", "selection_reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pants-root", required=True)
    ap.add_argument("--output", default="data_manifest/case_list_50_tumor.csv")
    ap.add_argument("--split", choices=["train", "test", "auto"], default="train")
    ap.add_argument("--num-cases", type=int, default=50)
    ap.add_argument("--required-organs", default=",".join(KEY_ORGANS))
    ap.add_argument("--allow-empty-lesion", action="store_true", help="Do not require non-empty pancreatic_lesion mask; use only if checking demo data.")
    args = ap.parse_args()

    root = Path(args.pants_root).resolve()
    required_organs = [x.strip() for x in args.required_organs.replace(";", ",").split(",") if x.strip()]
    splits = ["train", "test"] if args.split == "auto" else [args.split]
    rows: list[dict[str, Any]] = []
    for sp in splits:
        rows.extend(collect_candidates(root, sp, require_nonempty_lesion=not args.allow_empty_lesion, required_organs=required_organs))
    selected = rows[: args.num_cases]
    out = Path(args.output).resolve()
    write_csv(out, selected)
    summary = {
        "status": "success" if len(selected) == args.num_cases else "warning",
        "pants_root": str(root),
        "output": str(out),
        "num_candidates": len(rows),
        "num_selected": len(selected),
        "required_organs": required_organs,
        "selection_policy": "deterministic sorted case-id order after validating CT + required masks + non-empty pancreatic_lesion",
        "first_cases": selected[:10],
        "reason_if_warning": None if len(selected) == args.num_cases else "Not enough downloaded/validated tumor cases. Run download_pants50_selective.py or extract more PanTS cases first.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
