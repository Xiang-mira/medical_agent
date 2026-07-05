#!/usr/bin/env python3
"""Audit and remove the invalid mixed-scope Round2 inference artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def case_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [str(row.get("case_id") or "").strip() for row in csv.DictReader(handle) if row.get("case_id")]


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--invalid-case-list", type=Path, required=True)
    parser.add_argument("--correct-case-list", type=Path, required=True)
    parser.add_argument("--screen-name", required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--superseded-dir", action="append", type=Path, default=[])
    parser.add_argument("--min-free-gib", type=float, default=15.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.output_root.resolve()
    round_dir = root / "round2"
    predictions = round_dir / "student_predictions"
    checkpoint = args.checkpoint.resolve()
    checkpoint_mtime = checkpoint.stat().st_mtime
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    audit_dir = round_dir / "invalid_50case_inference_audit" / timestamp
    result_archive = audit_dir / "per_case_results"

    invalid_ids = case_ids(args.invalid_case_list.resolve())
    correct_ids = case_ids(args.correct_case_list.resolve())
    per_case: list[dict] = []
    classifications = {"new_repaired_checkpoint": [], "stale_or_unknown_checkpoint": [], "missing": []}
    for case_id in invalid_ids:
        result_path = predictions / case_id / "voxtell_student_result.json"
        if not result_path.exists():
            classifications["missing"].append(case_id)
            per_case.append({"case_id": case_id, "classification": "missing", "result_path": str(result_path)})
            continue
        stat = result_path.stat()
        classification = (
            "new_repaired_checkpoint"
            if stat.st_mtime > checkpoint_mtime
            else "stale_or_unknown_checkpoint"
        )
        classifications[classification].append(case_id)
        doc = read_json(result_path)
        per_case.append({
            "case_id": case_id,
            "classification": classification,
            "result_path": str(result_path),
            "result_mtime": stat.st_mtime,
            "result_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "status": doc.get("status"),
            "num_prompts": doc.get("num_prompts"),
            "num_masks": doc.get("num_masks"),
            "num_empty_masks": doc.get("num_empty_masks"),
            "num_failed_organs": doc.get("num_failed_organs"),
        })
        if args.apply:
            result_archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_path, result_archive / f"{case_id}.json")

    binary_inventory: list[dict] = []
    binary_suffixes = {".pth", ".pt", ".ckpt", ".npz"}
    for directory in args.superseded_dir:
        directory = directory.resolve()
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in binary_suffixes:
                continue
            binary_inventory.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "deleted": bool(args.apply),
            })

    audit = {
        "stage": "invalid_50case_inference_audit",
        "status": "invalid_scope_interrupted",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "screen_name": args.screen_name,
        "process_command": "python scripts/run_repaired_round2_continuation.py",
        "reason": "MEDAI_CASE_LIST was omitted; default 50-case list mixed stale and repaired checkpoint outputs.",
        "invalid_case_list": str(args.invalid_case_list.resolve()),
        "invalid_case_list_sha256": sha256_file(args.invalid_case_list.resolve()),
        "correct_case_list": str(args.correct_case_list.resolve()),
        "correct_case_list_sha256": sha256_file(args.correct_case_list.resolve()),
        "invalid_case_ids": invalid_ids,
        "correct_case_ids": correct_ids,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_mtime": checkpoint_mtime,
        "log_path": str(args.log_path.resolve()),
        "classifications": classifications,
        "counts": {key: len(value) for key, value in classifications.items()},
        "per_case": per_case,
        "superseded_binary_inventory": binary_inventory,
        "apply": args.apply,
        "free_bytes_before_cleanup": available_bytes(root),
    }

    if args.apply:
        audit_dir.mkdir(parents=True, exist_ok=True)
        if args.log_path.exists():
            shutil.copy2(args.log_path, audit_dir / args.log_path.name)
        old_summary = predictions / "student_inference_summary.json"
        if old_summary.exists():
            shutil.copy2(old_summary, audit_dir / old_summary.name)
        for path in binary_inventory:
            Path(path["path"]).unlink(missing_ok=True)
        for removable in [
            predictions,
            round_dir / "student_predictions_postprocessed",
            round_dir / "student_predictions_targeted_postprocessed",
            round_dir / "student_predictions_round3_competition",
        ]:
            if removable.exists():
                shutil.rmtree(removable)
        state_path = round_dir / "repaired_round2_continuation_state.json"
        write_json(state_path, {
            "stage": "repaired_round2_continuation",
            "status": "invalid_scope_interrupted",
            "timestamp": audit["timestamp"],
            "audit": str(audit_dir / "invalid_50case_inference_audit.json"),
        })
        audit["free_bytes_after_cleanup"] = available_bytes(root)
        audit["minimum_free_bytes"] = int(args.min_free_gib * 1024**3)
        audit["disk_gate_status"] = (
            "passed"
            if audit["free_bytes_after_cleanup"] >= audit["minimum_free_bytes"]
            else "failed"
        )

    write_json(audit_dir / "invalid_50case_inference_audit.json", audit)
    print(json.dumps({
        "status": audit.get("status"),
        "audit": str(audit_dir / "invalid_50case_inference_audit.json"),
        "counts": audit["counts"],
        "disk_gate_status": audit.get("disk_gate_status", "dry_run"),
        "free_gib": round(float(audit.get("free_bytes_after_cleanup", audit["free_bytes_before_cleanup"])) / 1024**3, 2),
    }, indent=2))
    return 0 if not args.apply or audit.get("disk_gate_status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
