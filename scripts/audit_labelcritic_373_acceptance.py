#!/usr/bin/env python3
"""Combine repair artifacts into a fail-closed experiment-resumption gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default=str(ROOT / "outputs" / "labelcritic_373_repair_20260703"),
    )
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    vendor = read(artifact_dir / "stage1_vendor_audit.json")
    descriptions = read(artifact_dir / "stage2_description_audit.json")
    benchmark = read(artifact_dir / "stage8_benchmark_gate.json")
    case_one = read(artifact_dir / "stage6_case_373_audit.json")
    case_twenty = read(artifact_dir / "stage6_20case_7460_audit.json")
    colon_regression = read(artifact_dir / "stage9_colon_regression" / "results.json")
    regression_registry = read(ROOT / "configs" / "labelcritic_373_regression_status.json")
    colon_fail_closed = (
        colon_regression.get("status") == "passed"
        or (
            colon_regression.get("status") == "failed"
            and (regression_registry.get("classes", {}).get("colon", {})).get("failure_action")
            == "automatic_abstention"
        )
    )
    checks = {
        "vendor_hash_passed": vendor.get("status") == "passed",
        "descriptions_373_schema_passed": (
            descriptions.get("status") in {"passed", "success"}
            and descriptions.get("target_count") == 373
            and descriptions.get("entry_count") == 373
        ),
        "official_benchmark_or_explicit_audit_only": benchmark.get("status") in {
            "ready", "blocked_data_unavailable"
        },
        "case_373_runtime_audit_passed": case_one.get("status") == "passed",
        "case_20x373_7460_audit_passed": case_twenty.get("status") == "passed",
        "class_agnostic_regression_fail_closed": colon_fail_closed,
        "baseline_regression_passed": read(
            artifact_dir / "stage10_baseline_regression.json"
        ).get("status") == "passed",
    }
    benchmark_ready = benchmark.get("status") == "ready"
    regression_passed = colon_regression.get("status") == "passed"
    result = {
        "stage": "final_experiment_resumption_gate",
        "checks": checks,
        "labelcritic_mode": (
            "selection_enabled" if benchmark_ready and regression_passed else "audit_only"
        ),
        "official_benchmark_ready": benchmark_ready,
        "class_agnostic_corruption_regression_passed": regression_passed,
        "mstep_resume_allowed": all(checks.values()),
        "blocked_experiments": (
            [] if all(checks.values())
            else [
                "25_step_mstep", "lambda_scan", "100_step_mstep",
                "300_step_mstep", "new_estep",
            ]
        ),
        "next_allowed_experiment": (
            "obtain_official_benchmark_data_or_keep_labelcritic_audit_only"
            if not regression_passed
            else "baseline_regression_25_step"
            if not checks["baseline_regression_passed"]
            else "none"
        ),
    }
    output = artifact_dir / "final_acceptance_gate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
