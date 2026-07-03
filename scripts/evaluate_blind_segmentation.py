#!/usr/bin/env python3
"""CPU-only blind segmentation evaluation.

Prediction files are fingerprinted before any reference path is opened.  The
inference case list must contain only case_id and ct_path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", required=True)
    ap.add_argument("--prediction-root", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reference-kind", default="historical_pseudo")
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    return ap.parse_args()


def metric_target_from_reference_kind(reference_kind: str) -> tuple[str, str, str]:
    if reference_kind == "expert_gt":
        return ("GT", "prediction_vs_expert_gt", "real_gt_segmentation_performance")
    if reference_kind in {"all_zero", "all-zero", "all_zero_target", "absent_negative"}:
        return ("all-zero target", "prediction_vs_all_zero_target", "negative_absence_quality")
    return ("pseudo-label", "prediction_vs_held_out_pseudo_label", "pseudo_label_consistency")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prediction_dir(root: Path, case_id: str) -> Path:
    candidates = [
        root / case_id,
        root / case_id / "segmentations",
        root / "annotation_versions" / case_id / "updated",
        root / "cases" / case_id / "segmentations",
        root / case_id / "updated",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No prediction directory for {case_id} below {root}")


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj) > 0


def surface_metrics(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    if not pred.any() and not ref.any():
        return {"hd95_mm": 0.0, "assd_mm": 0.0, "nsd_1mm": 1.0, "nsd_3mm": 1.0}
    if not pred.any() or not ref.any():
        return {"hd95_mm": float("inf"), "assd_mm": float("inf"), "nsd_1mm": 0.0, "nsd_3mm": 0.0}
    ps = np.logical_xor(pred, binary_erosion(pred))
    rs = np.logical_xor(ref, binary_erosion(ref))
    pr = distance_transform_edt(~rs, sampling=spacing)[ps]
    rp = distance_transform_edt(~ps, sampling=spacing)[rs]
    both = np.concatenate([pr, rp])
    denom = max(1, int(ps.sum() + rs.sum()))
    return {
        "hd95_mm": float(np.percentile(both, 95)),
        "assd_mm": float(both.mean()),
        "nsd_1mm": float(((pr <= 1).sum() + (rp <= 1).sum()) / denom),
        "nsd_3mm": float(((pr <= 3).sum() + (rp <= 3).sum()) / denom),
    }


def metric_row(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float | int | bool]:
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    pred_n, ref_n = int(pred.sum()), int(ref.sum())
    union = tp + fp + fn
    result: dict[str, float | int | bool] = {
        "pred_present": bool(pred_n),
        "reference_present": bool(ref_n),
        "dice": float(2 * tp / max(1, pred_n + ref_n)) if pred_n or ref_n else 1.0,
        "iou": float(tp / max(1, union)) if union else 1.0,
        "precision": float(tp / max(1, pred_n)) if pred_n else float(ref_n == 0),
        "recall": float(tp / max(1, ref_n)) if ref_n else float(pred_n == 0),
        "relative_volume_error": float((pred_n - ref_n) / ref_n) if ref_n else (float("inf") if pred_n else 0.0),
        "pred_voxels": pred_n,
        "reference_voxels": ref_n,
        "false_positive_voxels": fp,
        "connected_components": int(label(pred)[1]) if pred_n else 0,
    }
    result.update(surface_metrics(pred, ref, spacing))
    result["catastrophic_failure"] = bool(result["dice"] < 0.2)
    return result


def bootstrap_ci(rows: list[dict], key: str, samples: int) -> list[float] | None:
    cases = sorted({r["case_id"] for r in rows if np.isfinite(float(r[key]))})
    if not cases:
        return None
    by_case = {case: [float(r[key]) for r in rows if r["case_id"] == case and np.isfinite(float(r[key]))] for case in cases}
    rng = np.random.default_rng(20260628)
    values = []
    for _ in range(samples):
        sampled = rng.choice(cases, len(cases), replace=True)
        values.append(float(np.mean([x for case in sampled for x in by_case[case]])))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> int:
    args = parse_args()
    case_list = Path(args.case_list).resolve()
    pred_root = Path(args.prediction_root).resolve()
    ref_root = Path(args.reference_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with case_list.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) - {"case_id", "ct_path"}:
            raise SystemExit("Blind case list may contain only case_id and ct_path")
        cases = list(reader)

    # Seal predictions before references are accessed.
    sealed = []
    pred_dirs = {}
    for case in cases:
        case_id = case["case_id"]
        directory = prediction_dir(pred_root, case_id)
        pred_dirs[case_id] = directory
        for path in sorted(directory.glob("*.nii.gz")):
            sealed.append({"case_id": case_id, "organ": path.name[:-7], "path": str(path), "sha256": sha256(path)})
    seal = {"status": "sealed", "case_list": str(case_list), "prediction_root": str(pred_root), "predictions": sealed}
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2))

    rows = []
    geometry_errors = []
    metric_target, metric_comparison, metric_interpretation = metric_target_from_reference_kind(args.reference_kind)
    for case in cases:
        case_id = case["case_id"]
        ct = nib.load(case["ct_path"])
        refs = ref_root / case_id / "segmentations"
        for ref_path in sorted(refs.glob("*.nii.gz")):
            organ = ref_path.name[:-7]
            pred_path = pred_dirs[case_id] / ref_path.name
            if not pred_path.exists():
                continue
            pred_img, pred = load_binary(pred_path)
            ref_img, ref = load_binary(ref_path)
            prediction_geometry_ok = (
                pred_img.shape[:3] == ct.shape[:3]
                and np.allclose(pred_img.affine, ct.affine, atol=1e-3)
            )
            reference_shape_ok = ref_img.shape[:3] == ct.shape[:3]
            if not prediction_geometry_ok or not reference_shape_ok:
                geometry_errors.append({"case_id": case_id, "organ": organ, "prediction": str(pred_path)})
                continue
            reference_geometry_status = (
                "pass" if np.allclose(ref_img.affine, ct.affine, atol=1e-3)
                else "affine_mismatch_index_aligned"
            )
            row = {
                "case_id": case_id, "organ": organ, "geometry_status": "pass",
                "reference_geometry_status": reference_geometry_status,
                "ct_path": case["ct_path"], "prediction_path": str(pred_path),
                "reference_path": str(ref_path),
                "reference_kind": args.reference_kind,
                "metric_target": metric_target,
                "metric_subject": "student",
                "metric_comparison": metric_comparison,
                "metric_interpretation": metric_interpretation,
            }
            row.update(metric_row(pred, ref, tuple(float(x) for x in ct.header.get_zooms()[:3])))
            rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with (out / "per_organ_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    finite_hd = [r["hd95_mm"] for r in rows if np.isfinite(r["hd95_mm"])]
    present_tp = sum(r["pred_present"] and r["reference_present"] for r in rows)
    present_fn = sum((not r["pred_present"]) and r["reference_present"] for r in rows)
    absent_tn = sum((not r["pred_present"]) and (not r["reference_present"]) for r in rows)
    absent_fp = sum(r["pred_present"] and (not r["reference_present"]) for r in rows)
    summary = {
        "status": "success" if rows and not geometry_errors else "failed",
        "reference_kind": args.reference_kind,
        "metric_target": metric_target,
        "metric_subject": "student",
        "metric_comparison": metric_comparison,
        "metric_interpretation": metric_interpretation,
        "accuracy_wording": "held-out reference agreement" if args.reference_kind != "expert_gt" else "held-out expert-ground-truth accuracy",
        "cases": len(cases),
        "case_organ_pairs": len(rows),
        "macro_dice": float(np.mean([r["dice"] for r in rows])) if rows else None,
        "median_dice": float(np.median([r["dice"] for r in rows])) if rows else None,
        "macro_iou": float(np.mean([r["iou"] for r in rows])) if rows else None,
        "macro_precision": float(np.mean([r["precision"] for r in rows])) if rows else None,
        "macro_recall": float(np.mean([r["recall"] for r in rows])) if rows else None,
        "macro_nsd_1mm": float(np.mean([r["nsd_1mm"] for r in rows])) if rows else None,
        "macro_nsd_3mm": float(np.mean([r["nsd_3mm"] for r in rows])) if rows else None,
        "macro_hd95_mm": float(np.mean(finite_hd)) if finite_hd else None,
        "catastrophic_failure_rate": float(np.mean([r["catastrophic_failure"] for r in rows])) if rows else None,
        "presence_sensitivity": present_tp / max(1, present_tp + present_fn),
        "presence_specificity": absent_tn / max(1, absent_tn + absent_fp),
        "false_positive_voxels": int(sum(int(r["false_positive_voxels"]) for r in rows)),
        "dice_bootstrap_95ci": bootstrap_ci(rows, "dice", args.bootstrap_samples),
        "nsd_3mm_bootstrap_95ci": bootstrap_ci(rows, "nsd_3mm", args.bootstrap_samples),
        "geometry_errors": geometry_errors,
        "reference_header_anomalies": [
            {"case_id": r["case_id"], "organ": r["organ"]}
            for r in rows if r["reference_geometry_status"] != "pass"
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    core_organs = ["liver", "pancreas", "kidney_left", "kidney_right", "spleen", "aorta"]
    for case in cases:
        case_id = case["case_id"]
        case_rows = [r for r in rows if r["case_id"] == case_id]
        finite_hd_case = [r["hd95_mm"] for r in case_rows if np.isfinite(r["hd95_mm"])]
        case_summary = {
            "case_id": case_id,
            "macro_dice": float(np.mean([r["dice"] for r in case_rows])) if case_rows else None,
            "median_dice": float(np.median([r["dice"] for r in case_rows])) if case_rows else None,
            "worst_dice": sorted(
                [{"organ": r["organ"], "dice": r["dice"]} for r in case_rows],
                key=lambda x: x["dice"],
            )[:10],
            "largest_hd95": sorted(
                [{"organ": r["organ"], "hd95_mm": r["hd95_mm"]} for r in case_rows if np.isfinite(r["hd95_mm"])],
                key=lambda x: x["hd95_mm"], reverse=True,
            )[:10],
            "vr_abnormal": [
                {"organ": r["organ"], "relative_volume_error": r["relative_volume_error"]}
                for r in case_rows
                if not np.isfinite(r["relative_volume_error"]) or abs(r["relative_volume_error"]) > 1.0
            ],
            "macro_hd95_mm": float(np.mean(finite_hd_case)) if finite_hd_case else None,
        }
        case_out = out / "cases" / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / "summary.json").write_text(json.dumps(case_summary, indent=2, allow_nan=True))

        chosen = []
        by_organ = {r["organ"]: r for r in case_rows}
        for organ in core_organs:
            if organ in by_organ:
                chosen.append(by_organ[organ])
        chosen.extend(r for r in sorted(case_rows, key=lambda x: x["dice"]) if r not in chosen)
        chosen = chosen[:10]
        if chosen:
            ct = np.asanyarray(nib.as_closest_canonical(nib.load(case["ct_path"])).dataobj)
            fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
            for ax, row in zip(axes.flat, chosen):
                pred = np.asanyarray(nib.as_closest_canonical(nib.load(row["prediction_path"])).dataobj) > 0
                ref = np.asanyarray(nib.as_closest_canonical(nib.load(row["reference_path"])).dataobj) > 0
                z = int(np.argmax(np.count_nonzero(pred | ref, axis=(0, 1))))
                ax.imshow(np.clip(ct[:, :, z], -150, 250).T, cmap="gray", origin="lower", vmin=-150, vmax=250)
                if ref[:, :, z].any():
                    ax.contour(ref[:, :, z].T, [0.5], colors=["cyan"], linewidths=1.1)
                if pred[:, :, z].any():
                    ax.contour(pred[:, :, z].T, [0.5], colors=["magenta"], linewidths=1.0)
                ax.set_title(f"{row['organ']} D={row['dice']:.2f}")
                ax.axis("off")
            for ax in axes.flat[len(chosen):]:
                ax.axis("off")
            fig.suptitle(f"{case_id}: cyan reference, magenta prediction")
            fig.savefig(case_out / "core_and_worst_overlays.png", dpi=160)
            plt.close(fig)

    # Compact heatmap for automated inspection.
    organs = sorted({r["organ"] for r in rows})
    case_ids = [c["case_id"] for c in cases]
    matrix = np.full((len(organs), len(case_ids)), np.nan)
    for row in rows:
        matrix[organs.index(row["organ"]), case_ids.index(row["case_id"])] = row["dice"]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(organs))), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(case_ids)), [x[-3:] for x in case_ids], rotation=45)
    ax.set_yticks(range(len(organs)), organs)
    ax.set_title("Blind held-out Dice agreement")
    fig.colorbar(image, ax=ax, label="Dice")
    fig.savefig(out / "dice_heatmap.png", dpi=180)
    plt.close(fig)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
