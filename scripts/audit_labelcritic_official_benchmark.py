#!/usr/bin/env python3
"""Audit availability of the official LabelCritic benchmark without substitutes.

PanTS and historical pseudo labels are deliberately rejected as benchmark
sources.  This command is an acquisition/identity gate; evaluation may start
only after a licensed manifest with per-file SHA256 values is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        return {
            "stage": "official_labelcritic_benchmark",
            "status": "blocked_data_unavailable",
            "reason": "No licensed AtlasBench/JHHBench manifest with checksums was provided.",
            "forbidden_substitutes": ["PanTS", "historical_pseudo", "teacher_consensus"],
            "accuracy_claim_allowed": False,
        }
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_name = str(doc.get("benchmark_name") or "")
    license_text = str(doc.get("license") or doc.get("license_url") or "")
    files = doc.get("files") or []
    forbidden = any(token in benchmark_name.lower() for token in ("pants", "pseudo"))
    rows = []
    root = Path(doc.get("data_root") or manifest_path.parent)
    for item in files:
        path = root / str(item.get("path") or "")
        expected = str(item.get("sha256") or "")
        actual = sha256(path) if path.is_file() else None
        rows.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual,
                     "status": "match" if expected and actual == expected else "invalid"})
    ready = bool(
        benchmark_name in {"AtlasBench", "JHHBench", "AtlasBench/JHHBench"}
        and license_text and files and all(row["status"] == "match" for row in rows)
        and not forbidden
    )
    return {
        "stage": "official_labelcritic_benchmark",
        "status": "ready" if ready else "blocked_invalid_or_unverified_data",
        "benchmark_name": benchmark_name,
        "license": license_text,
        "manifest": str(manifest_path.resolve()),
        "files": rows,
        "model_calibration_target": "Qwen2-VL-7B",
        "paper_72b_metrics_inherited": False,
        "required_metrics": [
            "overall_accuracy", "per_organ_accuracy", "decisive_coverage",
            "abstention_rate", "dual_confirmation_consistency",
            "order_reversal_consistency", "error_types",
        ],
        "thresholds": {
            "overall_accuracy_min": 0.80,
            "wilson_95ci_lower_gt": 0.50,
            "decisive_coverage_min": 0.50,
            "per_organ_accuracy_min_when_n_ge_20": 0.65,
        },
        "accuracy_claim_allowed": ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "labelcritic_373_repair_20260703" / "stage8_benchmark_gate.json"),
    )
    args = parser.parse_args()
    result = audit(Path(args.manifest).resolve() if args.manifest else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] in {"ready", "blocked_data_unavailable"} else 2)


if __name__ == "__main__":
    main()
