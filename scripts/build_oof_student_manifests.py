#!/usr/bin/env python3
"""Build deterministic five-fold OOF student manifests with leakage provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.auto_label_core import SCORING_SCHEMA_VERSION, stable_case_fold


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return {}, [row for row in value if isinstance(row, dict)]
    return {key: item for key, item in value.items() if key != "items"}, [
        row for row in value.get("items", []) if isinstance(row, dict)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds != 5:
        raise SystemExit("Formal OOF policy requires exactly five folds")
    source = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, rows = load_rows(source)
    case_ids = sorted({str(row.get("case_id")) for row in rows if row.get("case_id")})
    if not case_ids:
        raise SystemExit("Manifest contains no case IDs")

    summary = {
        "status": "success", "folds": args.folds,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "source_manifest": str(source), "source_manifest_hash": canonical_hash(rows),
        "case_folds": {case_id: stable_case_fold(case_id, args.folds)
                       for case_id in case_ids},
        "folds_detail": [],
    }
    for held_out in range(args.folds):
        heldout_cases = [case_id for case_id in case_ids
                         if stable_case_fold(case_id, args.folds) == held_out]
        training_cases = [case_id for case_id in case_ids
                          if stable_case_fold(case_id, args.folds) != held_out]
        training = [row for row in rows if str(row.get("case_id")) in training_cases]
        validation = [row for row in rows if str(row.get("case_id")) in heldout_cases]
        if {str(row.get("case_id")) for row in training} & set(heldout_cases):
            raise SystemExit(f"OOF leakage detected in fold {held_out}")
        fold_dir = output / f"fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_doc = {**base, "split": "train", "held_out_fold": held_out,
                     "num_items": len(training), "items": training}
        heldout_doc = {**base, "split": "heldout", "held_out_fold": held_out,
                       "num_items": len(validation), "items": validation}
        manifest_hash = canonical_hash(training)
        provenance = {
            "scoring_schema_version": SCORING_SCHEMA_VERSION,
            "student_held_out_fold": held_out,
            "student_training_folds": [fold for fold in range(args.folds)
                                       if fold != held_out],
            "student_training_case_ids": training_cases,
            "student_heldout_case_ids": heldout_cases,
            "training_manifest_hash": manifest_hash,
            "source_manifest_hash": summary["source_manifest_hash"],
            "out_of_fold_policy": "stable_case_hash_5fold",
        }
        (fold_dir / "train_manifest.json").write_text(
            json.dumps(train_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        (fold_dir / "heldout_manifest.json").write_text(
            json.dumps(heldout_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        (fold_dir / "oof_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["folds_detail"].append({
            "fold": held_out, "training_cases": len(training_cases),
            "heldout_cases": len(heldout_cases), "training_items": len(training),
            "heldout_items": len(validation), "training_manifest_hash": manifest_hash,
        })
    (output / "oof_split_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

