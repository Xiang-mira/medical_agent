#!/usr/bin/env python3
"""Stop the active 20-case run at the case-10 boundary, then finish Round 1 and Blind-5."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def case_ids(case_list: Path) -> list[str]:
    with case_list.open(encoding="utf-8-sig", newline="") as handle:
        return [row["case_id"] for row in csv.DictReader(handle)]


def complete_cases(run_root: Path, expected: list[str]) -> set[str]:
    estep = run_root / "round1" / "estep"
    complete: set[str] = set()
    for case_id in expected:
        metadata = estep / "annotation_versions" / case_id / "selection_metadata.json"
        updated = estep / "annotation_versions" / case_id / "updated"
        plan = estep / "cases" / case_id / "hierarchical_inference_plan.json"
        if metadata.is_file() and plan.is_file() and updated.is_dir() and any(updated.glob("*.nii.gz")):
            complete.add(case_id)
    return complete


def process_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    env = os.environ.copy()
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
    return env


def process_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def stop_at_boundary(pid: int, screen_name: str, log) -> None:
    log.write(f"Case-10 boundary reached; stopping screen session {screen_name}.\n")
    log.flush()
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    for _ in range(60):
        if not process_exists(pid):
            return
        time.sleep(0.5)
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not process_exists(pid):
            return
        time.sleep(0.5)
    os.kill(pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--screen-name", default="formal_round1_quality_v3")
    parser.add_argument("--train-case-list", required=True)
    parser.add_argument("--blind-case-list", required=True)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    train_case_list = Path(args.train_case_list).resolve()
    blind_case_list = Path(args.blind_case_list).resolve()
    expected = case_ids(train_case_list)
    if len(expected) != 10:
        raise SystemExit(f"Expected exactly 10 training cases, found {len(expected)}")
    if len(case_ids(blind_case_list)) != 5:
        raise SystemExit("Blind case list must contain exactly 5 cases")

    state_path = run_root / "case10_continuation_state.json"
    log_path = run_root / "case10_continuation.log"
    with log_path.open("a", buffering=1) as log:
        env = process_environment(args.training_pid)
        env["MEDAI_CASE_LIST"] = str(train_case_list)
        env["MEDAI_OUTPUT_ROOT"] = str(run_root)
        env["PYTHONUNBUFFERED"] = "1"
        state_path.write_text(json.dumps({
            "status": "waiting_for_case10",
            "training_pid": args.training_pid,
            "train_case_list": str(train_case_list),
            "blind_case_list": str(blind_case_list),
            "expected_training_cases": expected,
        }, indent=2))

        while True:
            done = complete_cases(run_root, expected)
            if len(done) == len(expected):
                break
            if not process_exists(args.training_pid):
                raise SystemExit(
                    f"Training process {args.training_pid} exited before case 10; "
                    f"completed {len(done)}/10"
                )
            time.sleep(max(0.5, args.poll_sec))

        stop_at_boundary(args.training_pid, args.screen_name, log)
        state_path.write_text(json.dumps({
            "status": "running_10_case_continuation",
            "completed_cases": sorted(complete_cases(run_root, expected)),
        }, indent=2))

        training_log = run_root / "formal_round1_quality_v3.log"
        with training_log.open("a") as output:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_em_training.py")],
                cwd=str(ROOT),
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            state_path.write_text(json.dumps({
                "status": "training_continuation_failed",
                "return_code": result.returncode,
            }, indent=2))
            return result.returncode

        state_path.write_text(json.dumps({"status": "waiting_for_quality_gated_mstep"}, indent=2))
        blind_output = run_root / "blind5_batch1"
        watcher = subprocess.run([
            sys.executable,
            str(ROOT / "scripts/watch_and_run_blind10.py"),
            "--run-root", str(run_root),
            "--output-root", str(blind_output),
            "--case-list", str(blind_case_list),
            "--poll-sec", "10",
        ], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        state_path.write_text(json.dumps({
            "status": "complete" if watcher.returncode == 0 else "blind5_failed",
            "return_code": watcher.returncode,
            "blind_output": str(blind_output),
        }, indent=2))
        return watcher.returncode


if __name__ == "__main__":
    raise SystemExit(main())
