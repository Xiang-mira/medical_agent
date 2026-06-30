#!/usr/bin/env python3
"""Wait for a quality-gated M-step, then execute the blind-10 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_model(model_dir: Path, result: dict, output: Path) -> Path:
    files = [
        {
            "path": str(path.relative_to(model_dir)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    ]
    manifest = output / "frozen_student_model.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "status": "frozen",
        "quality_gate_status": result.get("status"),
        "eligible_for_next_round_prompt_student": result.get(
            "eligible_for_next_round_prompt_student"
        ),
        "model_dir": str(model_dir),
        "files": files,
    }, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--output-root")
    ap.add_argument("--case-list")
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    result_path = root / "round1/mstep/voxtell_prompt_mstep_result.json"
    output = Path(args.output_root).resolve() if args.output_root else root / "blind10_batch1"
    protocol = Path(__file__).with_name("run_blind10_protocol.py")
    while True:
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except Exception:
                result = {}
            ready = bool(
                result.get("status") == "success"
                and result.get("eligible_for_next_round_prompt_student")
                and result.get("inference_model_dir")
            )
            failed = str(result.get("status")) == "failed" or str(result.get("training_status", "")).startswith("failed")
            if ready:
                model_dir = Path(result["inference_model_dir"]).resolve()
                freeze_model(model_dir, result, output)
                model = str(model_dir)
                case_list_args = (
                    ["--case-list", str(Path(args.case_list).resolve())]
                    if args.case_list else []
                )
                subprocess.run([
                    sys.executable, str(protocol), "student",
                    "--student-model-dir", model, "--output-root", str(output),
                    *case_list_args,
                ], check=True)
                subprocess.run([
                    sys.executable, str(protocol), "agent", "--output-root", str(output),
                    *case_list_args,
                ], check=True)
                subprocess.run([
                    sys.executable, str(protocol), "evaluate", "--output-root", str(output),
                    *case_list_args,
                ], check=True)
                return 0
            if failed:
                (output / "blind10_blocked.json").parent.mkdir(parents=True, exist_ok=True)
                (output / "blind10_blocked.json").write_text(json.dumps({
                    "status": "blocked", "reason": "mstep_not_quality_gated", "mstep_result": result,
                }, indent=2))
                return 2
        time.sleep(max(10, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
