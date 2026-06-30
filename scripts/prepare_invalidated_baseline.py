#!/usr/bin/env python3
"""Archive stale Round 1 artifacts before a clean rerun.

This script never deletes previous outputs. It moves the requested source
directories under a dedicated invalidated_baseline root so old E-step/M-step
summaries and checkpoints cannot be mistaken for the fresh formal rerun.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        default=[
            str(ROOT / "outputs" / "round1"),
            str(ROOT / "outputs" / "round1_373_hierarchical_repair_20260620"),
        ],
        help="Existing output roots to quarantine under invalidated_baseline.",
    )
    ap.add_argument(
        "--invalidated-root",
        default=str(ROOT / "outputs" / "invalidated_baseline"),
    )
    ap.add_argument(
        "--tag",
        default="round1_old_estep_labels",
        help="Human-readable suffix for the invalidated batch folder.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    invalidated_root = Path(args.invalidated_root).resolve()
    invalidated_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = invalidated_root / f"{stamp}_{args.tag}"
    batch_root.mkdir(parents=True, exist_ok=True)

    archived = []
    missing = []
    for raw in args.sources:
        source = Path(raw).resolve()
        if not source.exists():
            missing.append(str(source))
            continue
        destination = batch_root / source.name
        if destination.exists():
            destination = batch_root / f"{source.name}__{len(archived)+1}"
        shutil.move(str(source), str(destination))
        archived.append({
            "source": str(source),
            "destination": str(destination),
        })

    report = {
        "stage": "prepare_invalidated_baseline",
        "status": "success",
        "invalidated_root": str(invalidated_root),
        "batch_root": str(batch_root),
        "archived": archived,
        "missing": missing,
    }
    (batch_root / "invalidated_baseline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
