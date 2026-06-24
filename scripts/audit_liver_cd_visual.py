#!/usr/bin/env python3
"""Audit and visualize old Round1 liver C/D pseudo-label cases.

This script is intentionally read-only with respect to E-step artifacts. It
writes review CSV/JSON/PNG overlays under an audit output directory and does not
modify training manifests or selected masks.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


DIAGNOSES = {
    "mask_genuinely_bad",
    "mapping_or_identity_error",
    "labelcritic_error_or_unsupported_prompt",
    "abcd_formula_too_strict",
    "shapekit_fallback_over_penalized",
    "zero_volume_or_missing_misclassified",
    "reference_or_pseudo_consistency_misleading",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def mask_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {"status": "missing_file", "voxels": None}
    p = Path(path)
    if not p.exists():
        return {"status": "missing_file", "path": str(p), "voxels": None}
    try:
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj) > 0
        voxels = int(arr.sum())
        return {
            "status": "zero_volume_mask" if voxels == 0 else "nonzero_mask",
            "path": str(p),
            "shape": list(img.shape[:3]),
            "voxels": voxels,
        }
    except Exception as exc:
        return {"status": "unreadable_mask", "path": str(p), "error": str(exc), "voxels": None}


def infer_diagnosis(row: dict[str, Any], selected_stats: dict[str, Any]) -> str:
    flags = set(row.get("review_flags") or []) | set(row.get("quality_flags") or [])
    identity = row.get("identity_status")
    shapekit = str(row.get("shapekit_status") or row.get("selected_candidate_shapekit_status") or "")
    labelcritic_status = str(row.get("labelcritic_status") or "")
    labelcritic_records = row.get("labelcritic_records") or []
    if identity not in {None, "", "valid", "legacy_unverified"}:
        return "mapping_or_identity_error"
    if selected_stats.get("status") in {"missing_file", "unreadable_mask", "zero_volume_mask"}:
        return "zero_volume_or_missing_misclassified"
    if "empty_reference_nonempty_prediction" in flags or row.get("selected_reference_quality_bucket") == "empty_reference_nonempty_prediction":
        return "reference_or_pseudo_consistency_misleading"
    if shapekit in {"fallback_original", "unsupported_target", "unsupported_target_skipped_by_policy", "failed"}:
        return "shapekit_fallback_over_penalized"
    if labelcritic_status in {"uncertain", "unsupported"} or any((r.get("decision") or {}).get("winner") == "uncertain" for r in labelcritic_records if isinstance(r, dict)):
        return "labelcritic_error_or_unsupported_prompt"
    qc = row.get("selected_candidate_qc_status")
    if qc in {"pass", "review"} and row.get("grade") in {"C", "D"}:
        return "abcd_formula_too_strict"
    return "mask_genuinely_bad"


def normalize_ct(ct_path: str | None) -> tuple[np.ndarray, Any] | tuple[None, None]:
    if not ct_path or not Path(ct_path).exists():
        return None, None
    try:
        img = nib.load(str(ct_path))
        arr = np.asanyarray(img.dataobj).astype("float32")
        lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99]) if np.isfinite(arr).any() else (float(arr.min()), float(arr.max()))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        return arr, img
    except Exception:
        return None, None


def load_mask_like(path: str | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    try:
        arr = np.asanyarray(nib.load(str(path)).dataobj) > 0
        if arr.shape[:3] != shape[:3]:
            return None
        return arr
    except Exception:
        return None


def overlay_png(ct: np.ndarray, mask: np.ndarray | None, out_png: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if mask is not None and mask.any():
        coords = np.argwhere(mask)
        z = int(np.median(coords[:, 2]))
        y = int(np.median(coords[:, 1]))
        x = int(np.median(coords[:, 0]))
    else:
        x, y, z = [s // 2 for s in ct.shape[:3]]
    planes = [
        ("axial", ct[:, :, z].T, None if mask is None else mask[:, :, z].T),
        ("coronal", ct[:, y, :].T, None if mask is None else mask[:, y, :].T),
        ("sagittal", ct[x, :, :].T, None if mask is None else mask[x, :, :].T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img, m) in zip(axes, planes):
        ax.imshow(img, cmap="gray", origin="lower")
        if m is not None:
            ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.45, origin="lower")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def iter_liver_rows(estep: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted((estep / "annotation_versions").glob("*/selection_metadata.json")):
        meta = read_json(meta_path, {}) or {}
        case_id = str(meta.get("case_id") or meta_path.parent.name)
        for row in meta.get("selected_organs", []) or []:
            if str(row.get("organ")) != "liver":
                continue
            grade = str(row.get("grade") or "")
            weight = float(row.get("training_weight") or 0.0)
            if grade in {"C", "D"} or weight <= 0.1:
                rows.append({"case_id": case_id, "meta_path": str(meta_path), "ct_path": meta.get("ct_path"), **row})
    manifest_path = estep / "training_manifest.json"
    manifest = read_json(manifest_path, [])
    manifest_items = manifest if isinstance(manifest, list) else (manifest.get("items", []) if isinstance(manifest, dict) else [])
    for row in manifest_items or []:
        if not isinstance(row, dict) or str(row.get("organ")) != "liver":
            continue
        grade = row.get("grade")
        weight = row.get("training_weight")
        if grade is None and weight is None:
            continue
        try:
            low_weight = float(weight or 0.0) <= 0.1
        except Exception:
            low_weight = False
        if str(grade or "") in {"C", "D"} or low_weight:
            rows.append({
                "case_id": str(row.get("case_id") or ""),
                "meta_path": str(manifest_path),
                "ct_path": row.get("ct_path") or row.get("image"),
                **row,
            })
    # Modern E-step manifests repeat the selected-organ record. Prefer the
    # richer selection_metadata row and keep only one record per case/organ.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_id") or ""), str(row.get("organ") or "liver"))
        previous = deduped.get(key)
        if previous is None or len(row.get("candidate_predictions") or []) > len(previous.get("candidate_predictions") or []):
            deduped[key] = row
    return list(deduped.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estep", default="outputs/round1/estep")
    ap.add_argument("--output-dir", default="outputs/audits/liver_cd_visual_audit")
    ap.add_argument("--max-cases", type=int, default=20)
    args = ap.parse_args()

    estep = Path(args.estep).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_rows = iter_liver_rows(estep)[: max(args.max_cases, 1)]
    audit_rows: list[dict[str, Any]] = []
    diagnosis_counts: Counter[str] = Counter()
    for row in raw_rows:
        selected_mask = row.get("mask_path") or row.get("final_mask") or row.get("selected_prediction")
        selected_stats = mask_stats(selected_mask)
        diagnosis = infer_diagnosis(row, selected_stats)
        diagnosis_counts[diagnosis] += 1
        case_id = str(row["case_id"])
        ct_path = row.get("ct_path")
        ct, _ = normalize_ct(ct_path)
        overlay = ""
        candidate_overlays: list[str] = []
        if ct is not None:
            m = load_mask_like(selected_mask, ct.shape)
            overlay_path = out / "overlays" / f"{case_id}_liver_selected.png"
            overlay_png(ct, m, overlay_path, f"{case_id} liver selected {row.get('grade')}")
            overlay = str(overlay_path)
        candidate_models = row.get("candidate_models") or []
        candidate_paths = [
            c.get("prediction")
            for c in (row.get("candidate_predictions") or [])
            if isinstance(c, dict)
        ]
        if ct is not None:
            ranked_candidates = sorted(
                (c for c in (row.get("candidate_predictions") or []) if isinstance(c, dict)),
                key=lambda c: float(c.get("dice") or -1.0),
                reverse=True,
            )
            for rank, candidate in enumerate(ranked_candidates[:3], start=1):
                candidate_path = candidate.get("prediction")
                candidate_mask = load_mask_like(candidate_path, ct.shape)
                if candidate_mask is None:
                    continue
                model = str(candidate.get("model") or f"candidate_{rank}").replace("/", "_")
                candidate_overlay_path = out / "overlays" / f"{case_id}_liver_candidate{rank}_{model}.png"
                overlay_png(
                    ct,
                    candidate_mask,
                    candidate_overlay_path,
                    f"{case_id} liver candidate {rank}: {model} dice={candidate.get('dice')}",
                )
                candidate_overlays.append(str(candidate_overlay_path))
        candidate_qc = [
            {
                "model": c.get("model"),
                "status": c.get("candidate_qc_status"),
                "flags": c.get("candidate_qc_flags") or [],
                "dice": c.get("dice"),
            }
            for c in (row.get("candidate_predictions") or [])
            if isinstance(c, dict)
        ]
        audit_rows.append({
            "case_id": case_id,
            "organ": "liver",
            "grade": row.get("grade"),
            "training_weight": row.get("training_weight"),
            "selected_model": row.get("selected_model") or row.get("source_model"),
            "selected_mask": selected_mask,
            "selected_mask_status": selected_stats.get("status"),
            "selected_mask_voxels": selected_stats.get("voxels"),
            "candidate_models": ";".join(str(x) for x in candidate_models),
            "candidate_paths": ";".join(str(x) for x in candidate_paths[:8]),
            "candidate_qc": json.dumps(candidate_qc, ensure_ascii=False),
            "candidate_qc_flags": ";".join(str(x) for x in row.get("selected_candidate_qc_flags", []) or []),
            "shapekit_status": row.get("shapekit_status") or row.get("selected_candidate_shapekit_status"),
            "labelcritic_status": row.get("labelcritic_status"),
            "identity_status": row.get("identity_status"),
            "mapping_type": row.get("mapping_type"),
            "mapping_source": row.get("mapping_source"),
            "selected_dice": row.get("selected_dice"),
            "selected_reference_quality_bucket": row.get("selected_reference_quality_bucket"),
            "review_flags": ";".join(str(x) for x in row.get("review_flags", []) or []),
            "quality_flags": ";".join(str(x) for x in row.get("quality_flags", []) or []),
            "suspected_failure_type": diagnosis,
            "overlay_png": overlay,
            "candidate_overlay_pngs": ";".join(candidate_overlays),
            "meta_path": row.get("meta_path"),
        })

    csv_path = out / "liver_cd_visual_audit.csv"
    json_path = out / "liver_cd_visual_audit.json"
    fieldnames = list(audit_rows[0].keys()) if audit_rows else [
        "case_id", "organ", "grade", "training_weight", "selected_model", "suspected_failure_type"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "stage": "liver_cd_visual_audit",
        "status": "success",
        "estep": str(estep),
        "output_dir": str(out),
        "num_cases": len(audit_rows),
        "diagnosis_counts": dict(diagnosis_counts),
        "csv": str(csv_path),
        "rows": audit_rows,
        "allowed_diagnoses": sorted(DIAGNOSES),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "num_cases", "diagnosis_counts", "csv"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
