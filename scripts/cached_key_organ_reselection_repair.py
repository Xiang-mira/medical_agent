#!/usr/bin/env python3
"""Fail-closed cached key-organ reselection repair gate.

This script does not run teacher inference and does not use GT. It records
whether LabelCritic benchmark evidence is strong enough to permit a later
cached reselection pass. If not, it emits after-repair diagnostics that keep the
current pseudo-label state unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/em_round1_25case_pseudo_label_20260709"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--benchmark-summary", type=Path, required=True)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    round_root = args.output_root / f"round{args.round}"
    estep = round_root / "estep"
    out_dir = estep / "repair"
    benchmark = read_json(args.benchmark_summary, {})
    gate = read_json(estep / "formal_gate.json", {})
    key_diag = read_json(estep / "key_organ_coverage_diagnosis.json", {})
    lc_diag = read_json(estep / "labelcritic_diagnosis/labelcritic_failure_taxonomy.json", {})
    benchmark_passed = benchmark.get("status") == "passed" or any(
        bool(row.get("pass")) for row in benchmark.get("rows", []) if isinstance(row, dict)
    )

    if not benchmark_passed:
        rows = []
        for row in key_diag.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            rows.append({
                **row,
                "repair_action": "kept_current_state",
                "repair_reason": "labelcritic_known_better_benchmark_failed",
            })
        payload = {
            "stage": "cached_key_organ_reselection_repair",
            "status": "blocked",
            "blocker": "labelcritic_known_better_benchmark_failed",
            "benchmark_summary": str(args.benchmark_summary),
            "formal_round2_gate_remains": "failed",
            "teacher_inference_rerun": False,
            "pseudo_label_mutation": False,
            "current_formal_gate_status": gate.get("status"),
            "current_formal_gate_reason": gate.get("reason"),
            "current_failed_key_organs": gate.get("cohort_failed_key_organs", []),
            "current_key_organ_failure_reason_counts": key_diag.get("failure_reason_counts", {}),
            "current_labelcritic_failure_counts": lc_diag.get("failure_counts", {}),
            "policy": "Expected-present missing masks remain withheld and are never converted to negative_absent.",
            "rows": rows,
        }
        write_json(out_dir / "key_organ_coverage_diagnosis_after_repair.json", payload)
        write_csv(out_dir / "key_organ_coverage_diagnosis_after_repair.csv", rows)
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, ensure_ascii=False))
        return 2

    payload = {
        "stage": "cached_key_organ_reselection_repair",
        "status": "ready_for_cached_reselection",
        "benchmark_summary": str(args.benchmark_summary),
        "teacher_inference_rerun": False,
        "pseudo_label_mutation": False,
        "next_action": "Run cached reselection with the passing LabelCritic matrix configuration; keep fail-closed replacement gates.",
    }
    write_json(out_dir / "key_organ_coverage_diagnosis_after_repair.json", payload)
    write_csv(out_dir / "key_organ_coverage_diagnosis_after_repair.csv", [])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
