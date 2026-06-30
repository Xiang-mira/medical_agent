#!/usr/bin/env python3
"""Build auditable old-vs-current CT segmentation figures for the weekly report."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "outputs/round1_best6_all_organs_20260622/masks"
NEW_ROOT = ROOT / "outputs/formal_round1_final_20260627/round1/estep/annotation_versions"
DATA_ROOT = ROOT / "data/PanTS"
OUT = ROOT / "outputs/formal_round1_final_20260627/weekly_report_visuals"
CASES = ["PanTS_00000031", "PanTS_00000162", "PanTS_00000423"]
GT_ORGANS = [
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "bladder",
    "celiac_artery", "colon", "common_bile_duct", "duodenum", "femur_left",
    "femur_right", "gall_bladder", "kidney_left", "kidney_right", "liver",
    "lung_left", "lung_right", "pancreas", "pancreas_body", "pancreas_head",
    "pancreas_tail", "pancreatic_duct", "postcava", "prostate", "spleen",
    "stomach", "superior_mesenteric_artery", "veins",
]


def load(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asarray(image.dataobj), image


def canonical_masks(folder: Path, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    if not folder.exists():
        return masks
    for path in sorted(folder.glob("*.nii.gz")):
        organ = path.name.removesuffix(".nii.gz")
        if organ.startswith("ct_segment_") or organ in masks or organ not in set(GT_ORGANS):
            continue
        try:
            arr, _ = load(path)
        except Exception:
            continue
        if arr.shape != shape:
            continue
        mask = arr > 0
        if mask.any():
            masks[organ] = mask
    return masks


def color_for(name: str) -> np.ndarray:
    digest = hashlib.sha256(name.encode()).digest()
    return np.array([40 + digest[0] % 200, 40 + digest[1] % 200, 40 + digest[2] % 200]) / 255


def normalize_ct(ct_slice: np.ndarray) -> np.ndarray:
    lo, hi = -150.0, 250.0
    return np.clip((ct_slice - lo) / (hi - lo), 0, 1)


def rgba_overlay(masks: dict[str, np.ndarray], z: int) -> tuple[np.ndarray, int]:
    shape = next(iter(masks.values())).shape[:2]
    rgb = np.zeros((*shape, 3), dtype=float)
    alpha = np.zeros(shape, dtype=float)
    active = 0
    for organ, mask in masks.items():
        plane = mask[:, :, z]
        if not plane.any():
            continue
        active += 1
        c = color_for(organ)
        rgb[plane] = rgb[plane] * 0.35 + c * 0.65
        alpha[plane] = 0.50
    return np.dstack([rgb, alpha]), active


def pick_slices(masks_a: dict[str, np.ndarray], masks_b: dict[str, np.ndarray], depth: int) -> list[int]:
    union = np.zeros((next(iter(masks_a.values())).shape[:2] + (depth,)), dtype=bool)
    for mask in list(masks_a.values()) + list(masks_b.values()):
        union |= mask
    area = union.sum(axis=(0, 1))
    valid = np.flatnonzero(area > 0)
    if len(valid) < 3:
        return [depth // 4, depth // 2, 3 * depth // 4]
    weights = np.cumsum(area[valid])
    return [int(valid[np.searchsorted(weights, q * weights[-1])]) for q in (0.25, 0.5, 0.75)]


def plot_all_organs(case: str, ct: np.ndarray, old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> None:
    slices = pick_slices(old, new, ct.shape[2])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, (label, masks) in enumerate((("Previous run", old), ("Current E-step", new))):
        for col, z in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(normalize_ct(ct[:, :, z]).T, cmap="gray", origin="lower")
            overlay, active = rgba_overlay(masks, z)
            ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
            ax.set_title(f"{label} | axial z={z} | {active} visible masks", fontsize=11)
            ax.axis("off")
    fig.suptitle(
        f"{case}: all 27 GT-organ predictions\n"
        f"previous={len(old)} organs, current={len(new)} organs; identical CT and slice locations",
        fontsize=15,
    )
    fig.savefig(OUT / f"{case}_all_organs_previous_vs_current.png", dpi=180)
    plt.close(fig)


def metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = int(pred.sum())
    g = int(gt.sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else float(g == 0)
    recall = tp / (tp + fn) if (tp + fn) else float(p == 0)
    volume_ratio = p / g if g else float("nan")
    hd95 = float("nan")
    nsd3 = float("nan")
    if False and p and g:
        pred_surface = pred ^ binary_erosion(pred)
        gt_surface = gt ^ binary_erosion(gt)
        d_gt = distance_transform_edt(~gt_surface, sampling=spacing)
        d_pred = distance_transform_edt(~pred_surface, sampling=spacing)
        distances = np.concatenate([d_gt[pred_surface], d_pred[gt_surface]])
        hd95 = float(np.percentile(distances, 95))
        nsd3 = float((distances <= 3.0).mean())
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "volume_ratio": volume_ratio, "hd95_mm": hd95, "nsd_3mm": nsd3,
    }


def plot_organ(case: str, organ: str, ct: np.ndarray, gt: np.ndarray,
               old: np.ndarray, new: np.ndarray, old_m: dict[str, float],
               new_m: dict[str, float]) -> None:
    disagreement = gt | old | new
    score = disagreement.sum(axis=(0, 1))
    z = int(np.argmax(score))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    panels = [
        ("Expert GT", gt, None),
        ("Previous run", old, old_m),
        ("Current E-step", new, new_m),
    ]
    for ax, (label, mask, m) in zip(axes, panels):
        ax.imshow(normalize_ct(ct[:, :, z]).T, cmap="gray", origin="lower")
        ax.contour(gt[:, :, z].T, levels=[0.5], colors=["lime"], linewidths=1.5)
        if label != "Expert GT":
            ax.imshow(np.ma.masked_where(~mask[:, :, z].T, mask[:, :, z].T),
                      cmap="Reds" if label == "Previous run" else "Blues",
                      alpha=0.42, origin="lower", vmin=0, vmax=1)
            ax.contour(mask[:, :, z].T, levels=[0.5],
                       colors=["red" if label == "Previous run" else "cyan"],
                       linewidths=1.2)
        subtitle = label
        if m:
            subtitle += (
                f"\nDice {m['dice']:.3f} | IoU {m['iou']:.3f} | "
                f"P/R {m['precision']:.3f}/{m['recall']:.3f} | Vol {m['volume_ratio']:.2f}x"
            )
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{case} — {organ} — axial z={z}\n"
        "green contour=expert GT; red=previous prediction; cyan=current prediction",
        fontsize=14,
    )
    fig.savefig(OUT / f"{case}_{organ}_previous_vs_current_gt.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cached: dict[str, tuple[np.ndarray, nib.Nifti1Image, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for case in CASES:
        ct, ct_img = load(DATA_ROOT / "ImageTr" / case / "ct.nii.gz")
        old = canonical_masks(OLD_ROOT / case, ct.shape)
        new = canonical_masks(NEW_ROOT / case / "updated", ct.shape)
        if not old or not new:
            raise RuntimeError(f"Missing masks for {case}: old={len(old)}, new={len(new)}")
        cached[case] = (ct, ct_img, old, new)
        plot_all_organs(case, ct, old, new)
        gt_dir = DATA_ROOT / "LabelTr" / case / "segmentations"
        spacing = tuple(float(x) for x in ct_img.header.get_zooms()[:3])
        for organ in GT_ORGANS:
            gt_path = gt_dir / f"{organ}.nii.gz"
            if not gt_path.exists() or organ not in old or organ not in new:
                continue
            gt, _ = load(gt_path)
            gt = gt > 0
            om = metrics(old[organ], gt, spacing)
            nm = metrics(new[organ], gt, spacing)
            rows.append({
                "case_id": case, "organ": organ,
                **{f"previous_{k}": v for k, v in om.items()},
                **{f"current_{k}": v for k, v in nm.items()},
                "delta_dice": nm["dice"] - om["dice"],
                "delta_iou": nm["iou"] - om["iou"],
                "delta_nsd_3mm": nm["nsd_3mm"] - om["nsd_3mm"],
            })
    csv_path = OUT / "previous_vs_current_gt_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = rows
    improvements = sorted(eligible, key=lambda r: float(r["delta_dice"]), reverse=True)[:3]
    regressions = sorted(eligible, key=lambda r: float(r["delta_dice"]))[:2]
    chosen = improvements + regressions
    for row in chosen:
        case, organ = str(row["case_id"]), str(row["organ"])
        ct, ct_img, old, new = cached[case]
        gt, _ = load(DATA_ROOT / "LabelTr" / case / "segmentations" / f"{organ}.nii.gz")
        spacing = tuple(float(x) for x in ct_img.header.get_zooms()[:3])
        plot_organ(case, organ, ct, gt > 0, old[organ], new[organ],
                   metrics(old[organ], gt > 0, spacing),
                   metrics(new[organ], gt > 0, spacing))
    print(f"Wrote {len(rows)} paired metric rows to {csv_path}")
    print("Selected organ panels:")
    for row in chosen:
        print(row["case_id"], row["organ"], f"delta_dice={float(row['delta_dice']):.3f}")


if __name__ == "__main__":
    main()
