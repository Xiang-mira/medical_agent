#!/usr/bin/env python3
"""Audit LabelCritic projection artifacts against VLM decisions.

This diagnostic checks whether failed LabelCritic comparisons are caused by
missing/weak overlay visibility in the generated PNG projections. It does not
judge segmentation correctness or true accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit LabelCritic projection quality.")
    ap.add_argument("--critic-root", required=True, help="Directory containing LabelCritic JSON files.")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--red-threshold", type=float, default=0.001, help="Minimum red-pixel fraction considered visible.")
    return ap.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_row(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _red_fraction(path: Path | None) -> float | None:
    if not path or not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert("RGB"))
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    mask = (red > 120) & (red > green + 30) & (red > blue + 30)
    return float(mask.mean())


def _artifact_dir(csv_path: Path, organ: str) -> Path | None:
    # .../results/<run_id>/<organ>.csv -> .../comparison_results/<run_id>/<organ>/<organ>
    try:
        run_id = csv_path.parent.name
        labelcritic_dir = csv_path.parents[2]
    except IndexError:
        return None
    artifact_dir = labelcritic_dir / "comparison_results" / run_id / organ / organ
    return artifact_dir if artifact_dir.exists() else None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _projection_paths(artifact_dir: Path | None, organ: str) -> dict[str, Path | None]:
    if not artifact_dir:
        return {"y1": None, "y2": None, "composite": None, "best1": None}
    return {
        "y1": _first_existing([
            artifact_dir / f"case001_overlay_window_bone_axis_1_{organ}_y1.png",
            artifact_dir / f"case001_overlay_window_organs_axis_1_{organ}_y1.png",
            artifact_dir / f"case001_overlay_window_skeleton_axis_1_{organ}_y1.png",
        ]),
        "y2": _first_existing([
            artifact_dir / f"case001_overlay_window_bone_axis_1_{organ}_y2.png",
            artifact_dir / f"case001_overlay_window_organs_axis_1_{organ}_y2.png",
            artifact_dir / f"case001_overlay_window_skeleton_axis_1_{organ}_y2.png",
        ]),
        "composite": _first_existing([
            artifact_dir / f"case001_composite_image_2_figs_axis_1_{organ}.png",
            artifact_dir / f"case001_composite_image_2_figs_axis_1_{organ}_skeleton.png",
        ]),
        "best1": artifact_dir / f"case001_best1_composite_image_2_figs_axis_1_{organ}.png",
    }


def main() -> int:
    args = parse_args()
    critic_root = Path(args.critic_root).resolve()
    rows: list[dict[str, Any]] = []

    for json_path in sorted(critic_root.glob("**/*.json")):
        data = _read_json(json_path)
        organ = str(data.get("organ") or json_path.name.split("_")[0])
        decision = data.get("decision") or {}
        options = data.get("labelcritic_options") or {}
        csv_path = Path(options.get("csv_path", ""))
        csv_data = _csv_row(csv_path)
        artifact_dir = _artifact_dir(csv_path, organ) if csv_path else None
        paths = _projection_paths(artifact_dir, organ)
        y1_red = _red_fraction(paths["y1"])
        y2_red = _red_fraction(paths["y2"])
        composite_red = _red_fraction(paths["composite"])
        visible_y1 = y1_red is not None and y1_red >= args.red_threshold
        visible_y2 = y2_red is not None and y2_red >= args.red_threshold
        rows.append({
            "json_path": str(json_path),
            "case_id": json_path.relative_to(critic_root).parts[0] if len(json_path.relative_to(critic_root).parts) > 1 else "",
            "organ": organ,
            "status": data.get("status"),
            "winner": decision.get("winner"),
            "parse_status": decision.get("parse_status"),
            "csv_answer": csv_data.get("answer"),
            "csv_answer_1": csv_data.get("answer_1"),
            "csv_answer_2": csv_data.get("answer_2"),
            "y1_path": str(paths["y1"]) if paths["y1"] else "",
            "y2_path": str(paths["y2"]) if paths["y2"] else "",
            "composite_path": str(paths["composite"]) if paths["composite"] else "",
            "y1_red_fraction": y1_red,
            "y2_red_fraction": y2_red,
            "composite_red_fraction": composite_red,
            "visible_y1": visible_y1,
            "visible_y2": visible_y2,
            "visibility_issue": not (visible_y1 and visible_y2),
            "strict_choice_prompt": bool(options.get("strict_choice_prompt")),
            "skip_organ_presence_gate": bool(options.get("skip_organ_presence_gate")),
            "no_dice_check": bool(options.get("no_dice_check")),
        })

    out_json = Path(args.output_json).resolve()
    out_csv = Path(args.output_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter, defaultdict
    summary = {
        "status": "success",
        "critic_root": str(critic_root),
        "num_items": len(rows),
        "winner_counts": dict(Counter(str(r["winner"]) for r in rows)),
        "parse_status_counts": dict(Counter(str(r["parse_status"]) for r in rows)),
        "visibility_issue_count": sum(1 for r in rows if r["visibility_issue"]),
        "by_organ": {},
        "accuracy_warning": "Projection visibility audit only; not true segmentation accuracy.",
    }
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_organ[str(row["organ"])].append(row)
    for organ, organ_rows in by_organ.items():
        summary["by_organ"][organ] = {
            "n": len(organ_rows),
            "winner_counts": dict(Counter(str(r["winner"]) for r in organ_rows)),
            "visibility_issue_count": sum(1 for r in organ_rows if r["visibility_issue"]),
            "mean_y1_red_fraction": float(np.mean([r["y1_red_fraction"] for r in organ_rows if r["y1_red_fraction"] is not None])) if any(r["y1_red_fraction"] is not None for r in organ_rows) else None,
            "mean_y2_red_fraction": float(np.mean([r["y2_red_fraction"] for r in organ_rows if r["y2_red_fraction"] is not None])) if any(r["y2_red_fraction"] is not None for r in organ_rows) else None,
        }

    out_json.write_text(json.dumps({"summary": summary, "items": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
