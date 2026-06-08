#!/usr/bin/env python3
"""Combine VoxTell prompt/student manifests from multiple E-step runs.

This keeps staged PanTS pilots auditable: each stage can run independently, then
the student M-step receives one manifest with deterministic de-duplication and
the same quality/weight metadata used by the per-stage manifests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.target_space import validate_formal_373_target_space


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build one VoxTell manifest from multiple staged manifests.")
    ap.add_argument("--manifest", action="append", required=True, help="Input manifest JSON. Repeat for each stage.")
    ap.add_argument("--output", required=True, help="Combined manifest JSON to write.")
    ap.add_argument(
        "--target-config",
        default=str(ROOT / "configs/student_3d_prompt_target_organs.json"),
        help="Formal 373-organ target config.",
    )
    ap.add_argument("--require-existing-image", action="store_true")
    ap.add_argument("--require-existing-mask", action="store_true")
    return ap.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return {"stage": "raw_training_manifest", "status": "success", "items": doc}
    if not isinstance(doc, dict):
        raise TypeError(f"Unsupported manifest format: {path}")
    items = doc.get("items")
    if isinstance(items, list):
        return doc
    raise ValueError(f"Manifest has no items list: {path}")


def item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("case_id") or ""),
        str(item.get("organ") or ""),
        str(item.get("mask") or item.get("mask_path") or ""),
    )


def main() -> int:
    args = parse_args()
    target_config = Path(args.target_config).resolve()
    inputs = [Path(p).resolve() for p in args.manifest]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for manifest_path in inputs:
        doc = read_manifest(manifest_path)
        source_items = doc.get("items", [])
        added = 0
        duplicate = 0
        for raw in source_items:
            if not isinstance(raw, dict):
                skipped.append({"manifest": str(manifest_path), "reason": "non_object_item"})
                continue
            item = dict(raw)
            item["image"] = item.get("image") or item.get("ct_path")
            item["ct_path"] = item.get("ct_path") or item.get("image")
            item["mask"] = item.get("mask") or item.get("mask_path")
            item["mask_path"] = item.get("mask_path") or item.get("mask")
            item["source_manifest"] = str(manifest_path)
            item["training_weight"] = float(item.get("training_weight", 0.0) or 0.0)

            if not item.get("case_id") or not item.get("organ") or not item.get("prompt"):
                skipped.append({"manifest": str(manifest_path), "reason": "missing_case_organ_or_prompt", "item": item_key(item)})
                continue
            if args.require_existing_image and (not item.get("image") or not Path(str(item["image"])).exists()):
                skipped.append({"manifest": str(manifest_path), "reason": "missing_image", "item": item_key(item)})
                continue
            if args.require_existing_mask and (not item.get("mask") or not Path(str(item["mask"])).exists()):
                skipped.append({"manifest": str(manifest_path), "reason": "missing_mask", "item": item_key(item)})
                continue

            key = item_key(item)
            if key in seen:
                duplicate += 1
                skipped.append({"manifest": str(manifest_path), "reason": "duplicate_item", "item": key})
                continue
            seen.add(key)
            rows.append(item)
            added += 1

        source_summaries.append({
            "manifest": str(manifest_path),
            "stage": doc.get("stage"),
            "status": doc.get("status"),
            "source_items": len(source_items),
            "added_items": added,
            "duplicate_items": duplicate,
        })

    organs = sorted({str(r["organ"]) for r in rows})
    validation = validate_formal_373_target_space(
        target_config,
        requested_organs=organs,
        require_full_target=len(organs) == 373,
    )
    combined = {
        "stage": "combined_voxtell_3d_prompt_training_manifest",
        "status": "success" if not skipped else "success_with_skips",
        "student_backend": "voxtell_style_3d_prompt",
        "target_config": str(target_config),
        "formal_373_target_validation": validation,
        "source_manifests": source_summaries,
        "num_items": len(rows),
        "num_cases": len({r["case_id"] for r in rows}),
        "num_organs": len(organs),
        "organs": organs,
        "num_items_missing_image": sum(1 for r in rows if not r.get("image")),
        "num_items_missing_mask": sum(1 for r in rows if not r.get("mask")),
        "grade_counts": {grade: sum(1 for r in rows if r.get("grade") == grade) for grade in ["A", "B", "C", "D"]},
        "num_strong_training_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) >= 0.5),
        "num_zero_weight_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) == 0.0),
        "num_empty_reference_items": sum(
            1 for r in rows if r.get("selected_reference_quality_bucket") == "empty_reference_nonempty_prediction"
        ),
        "skipped_items": skipped,
        "items": rows,
        "accuracy_warning": "Pseudo-consistency only; this is not expert-label accuracy.",
    }
    output.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: combined[k] for k in [
        "status",
        "num_items",
        "num_cases",
        "num_organs",
        "grade_counts",
        "num_strong_training_items",
        "num_zero_weight_items",
        "num_empty_reference_items",
        "num_items_missing_image",
        "num_items_missing_mask",
    ]}, indent=2, ensure_ascii=False))
    return 0 if combined["num_items"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
