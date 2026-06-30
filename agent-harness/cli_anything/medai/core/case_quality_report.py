from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def build_case_quality_report(
    *,
    case_id: str,
    ct_path: str | Path,
    selected: list[dict[str, Any]],
    gap_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write an automatic, non-interactive case audit and overlay sheet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selected:
        qc = item.get("selected_candidate_qc_checks") or {}
        rows.append({
            "organ": item.get("organ"),
            "grade": item.get("grade"),
            "publication_status": item.get("publication_status"),
            "distillation_eligible": item.get("distillation_eligible"),
            "reference_dice": qc.get("reference_dice", item.get("selected_pseudo_consistency_dice")),
            "hd95_mm": qc.get("hd95_mm"),
            "volume_ratio": qc.get("volume_ratio_to_reference"),
            "connected_components": qc.get("connected_components"),
            "qc_flags": item.get("selected_candidate_qc_flags", []),
            "final_mask": item.get("final_mask"),
            "audit_mask_path": item.get("audit_mask_path"),
            "reference": item.get("reference"),
            "shapekit_status": item.get("shapekit_status"),
        })
    dice_rows = [r for r in rows if isinstance(r.get("reference_dice"), (int, float))]
    hd_rows = [r for r in rows if isinstance(r.get("hd95_mm"), (int, float))]
    report = {
        "case_id": case_id,
        "metric_warning": "Reference metrics are agreement metrics, not expert accuracy.",
        "selected_candidates": len(rows),
        "published_labels": sum(r["publication_status"] != "rejected_but_recorded" for r in rows),
        "training_eligible_labels": sum(bool(r["distillation_eligible"]) for r in rows),
        "rejected_labels": sum(r["publication_status"] == "rejected_but_recorded" for r in rows),
        "mean_reference_dice": float(np.mean([r["reference_dice"] for r in dice_rows])) if dice_rows else None,
        "median_reference_dice": float(np.median([r["reference_dice"] for r in dice_rows])) if dice_rows else None,
        "lowest_dice": sorted(dice_rows, key=lambda r: r["reference_dice"])[:10],
        "largest_hd95": sorted(hd_rows, key=lambda r: r["hd95_mm"], reverse=True)[:10],
        "abnormal_volume_ratio": [
            r for r in rows
            if isinstance(r.get("volume_ratio"), (int, float))
            and (r["volume_ratio"] < 0.25 or r["volume_ratio"] > 4.0)
        ],
        "d_or_rejected": [r for r in rows if r["grade"] == "D" or r["publication_status"] == "rejected_but_recorded"],
        "shapekit_failures": [r for r in rows if r["shapekit_status"] in {"failed", "postprocess_failed", "fallback_original"}],
        "gap_summary": gap_summary,
    }
    (out / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # One fully automatic overlay page: core structures followed by worst Dice.
    try:
        import matplotlib.pyplot as plt

        core = ["liver", "pancreas", "kidney_left", "kidney_right", "spleen", "aorta"]
        by_organ = {r["organ"]: r for r in rows}
        chosen = [by_organ[x] for x in core if x in by_organ]
        chosen.extend(r for r in sorted(dice_rows, key=lambda r: r["reference_dice"]) if r not in chosen)
        chosen = chosen[:10]
        ct = np.asanyarray(nib.as_closest_canonical(nib.load(str(ct_path))).dataobj)
        fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
        for ax, row in zip(axes.flat, chosen):
            mask_path = row.get("final_mask") or row.get("audit_mask_path")
            ref_path = row.get("reference")
            if not mask_path or not Path(mask_path).exists():
                ax.axis("off"); continue
            pred = np.asanyarray(nib.as_closest_canonical(nib.load(mask_path)).dataobj) > 0
            ref = (
                np.asanyarray(nib.as_closest_canonical(nib.load(ref_path)).dataobj) > 0
                if ref_path and Path(ref_path).exists() else np.zeros_like(pred)
            )
            z = int(np.argmax(np.count_nonzero(pred | ref, axis=(0, 1))))
            ax.imshow(np.clip(ct[:, :, z], -150, 250).T, cmap="gray", origin="lower", vmin=-150, vmax=250)
            if ref[:, :, z].any(): ax.contour(ref[:, :, z].T, [0.5], colors=["cyan"], linewidths=1)
            if pred[:, :, z].any(): ax.contour(pred[:, :, z].T, [0.5], colors=["magenta"], linewidths=1)
            dice = row.get("reference_dice")
            ax.set_title(f"{row['organ']} D={dice:.2f}" if isinstance(dice, (int, float)) else str(row["organ"]))
            ax.axis("off")
        for ax in axes.flat[len(chosen):]: ax.axis("off")
        fig.suptitle(f"{case_id}: core and lowest-agreement masks")
        fig.savefig(out / "core_and_worst_overlays.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        report["visualization_error"] = f"{type(exc).__name__}: {exc}"
        (out / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report
