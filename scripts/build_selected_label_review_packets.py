#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HIGH_RISK_DECISIONS = {
    "low_confidence_selected",
    "low_winner_margin",
    "labelcritic_warning",
    "laterality_risk",
    "parent_child_conflict",
    "empty_or_near_empty_risk",
    "high_teacher_disagreement",
    "student_selected_for_training",
}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def iter_manifest_items(path: Path) -> list[dict[str, Any]]:
    doc = read_json(path, {})
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    return [x for x in doc.get("items", []) if isinstance(x, dict)]


def risk_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    conf = item.get("evidence_confidence", item.get("confidence", item.get("C")))
    try:
        if conf is not None and float(conf) < 0.75:
            reasons.append("low_confidence_selected")
    except Exception:
        pass
    margin = item.get("winner_margin") or item.get("selection_margin")
    try:
        if margin is not None and float(margin) < 0.05:
            reasons.append("low_winner_margin")
    except Exception:
        pass
    flags = item.get("selected_candidate_qc_flags") or item.get("candidate_qc_flags") or []
    flags_text = " ".join(str(x) for x in flags).lower()
    if any(k in flags_text for k in ("laterality", "left", "right")):
        reasons.append("laterality_risk")
    if any(k in flags_text for k in ("empty", "zero", "near_empty")):
        reasons.append("empty_or_near_empty_risk")
    if any(k in flags_text for k in ("parent", "child", "conflict")):
        reasons.append("parent_child_conflict")
    labelcritic = str(item.get("labelcritic_summary") or item.get("labelcritic_compare_reason") or "").lower()
    if any(k in labelcritic for k in ("warning", "disagree", "uncertain", "ambiguous")):
        reasons.append("labelcritic_warning")
    provider = str(item.get("selected_provider") or item.get("source_model") or item.get("teacher") or "")
    if provider == "student_prev" or item.get("is_project_student") is True:
        reasons.append("student_selected_for_training")
    if int(item.get("candidate_count") or item.get("selection_candidate_count") or 0) >= 3:
        reasons.append("high_teacher_disagreement")
    return sorted(set(reasons))


def maybe_write_overlay(item: dict[str, Any], packet_dir: Path) -> str | None:
    image = item.get("image") or item.get("ct_path") or item.get("image_path")
    mask = item.get("selected_mask_path") or item.get("mask_path") or item.get("mask")
    if not image or not mask:
        return None
    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
        img = np.asanyarray(nib.load(str(image)).dataobj)
        m = np.asanyarray(nib.load(str(mask)).dataobj) > 0
        if img.ndim != 3 or m.shape != img.shape:
            return None
        z = int(np.argmax(m.sum(axis=(0, 1)))) if m.any() else img.shape[2] // 2
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img[:, :, z].T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(~m[:, :, z].T, m[:, :, z].T), cmap="autumn", alpha=0.45, origin="lower")
        ax.axis("off")
        out = packet_dir / "overlay.png"
        fig.tight_layout(pad=0)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return str(out)
    except Exception:
        return None


def build_review_packets(manifest: Path, output_dir: Path, high_risk_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = iter_manifest_items(manifest)
    queue: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        organ = str(item.get("organ") or item.get("organ_name") or "unknown")
        case_id = str(item.get("case_id") or f"item{idx:05d}")
        risks = risk_reasons(item)
        if high_risk_only and not risks:
            continue
        packet_dir = output_dir / case_id / organ
        packet_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "case_id": case_id,
            "organ": organ,
            "selected_provider": item.get("selected_provider") or item.get("source_model") or item.get("teacher"),
            "selected_mask_path": item.get("selected_mask_path") or item.get("mask_path") or item.get("mask"),
            "top_candidates": item.get("top_candidates") or item.get("candidate_models") or [],
            "confidence": item.get("evidence_confidence", item.get("confidence", item.get("C"))),
            "grade": item.get("grade"),
            "labelcritic_summary": item.get("labelcritic_summary") or item.get("labelcritic_compare_reason"),
            "risk_reasons": risks,
            "review_packet": str(packet_dir / "review_packet.json"),
            "overlay_png": None,
            "human_review_status": item.get("human_review_status") or "pending",
            "human_review_reason": item.get("human_review_reason"),
            "reviewer": item.get("reviewer"),
            "review_time": item.get("review_time"),
            "affects_training_manifest": bool(risks),
        }
        row["overlay_png"] = maybe_write_overlay(item, packet_dir)
        (packet_dir / "review_packet.json").write_text(json.dumps({"item": item, "review": row}, indent=2, ensure_ascii=False), encoding="utf-8")
        queue.append(row)
    (output_dir / "review_queue.json").write_text(json.dumps({"items": queue}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "review_queue.jsonl").open("w", encoding="utf-8") as f:
        for row in queue:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "stage": "selected_label_human_review_packets",
        "status": "success",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "num_input_items": len(items),
        "num_review_items": len(queue),
        "num_high_risk_items": sum(1 for row in queue if row["risk_reasons"]),
        "formal_gate_policy": "high-risk pending/rejected/needs_manual_fix labels are blocked from formal M-step manifest",
    }
    (output_dir / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build human review packets for selected pseudo-label manifest items.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--high-risk-only", action="store_true")
    args = ap.parse_args()
    summary = build_review_packets(Path(args.manifest), Path(args.output_dir), args.high_risk_only)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
