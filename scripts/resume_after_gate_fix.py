#!/usr/bin/env python3
"""Resume cached Round 1 at the formal gate, then run the quality-gated Blind-5."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_state(path: Path, status: str, **extra) -> None:
    path.write_text(json.dumps({"status": status, **extra}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-case-list", required=True)
    parser.add_argument("--blind-case-list", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    train_case_list = Path(args.train_case_list).resolve()
    blind_case_list = Path(args.blind_case_list).resolve()
    state = run_root / "case10_continuation_state.json"
    env = os.environ.copy()
    env.update({
        "MEDAI_CASE_LIST": str(train_case_list),
        "MEDAI_OUTPUT_ROOT": str(run_root),
        "PYTHONUNBUFFERED": "1",
    })

    write_state(state, "rerunning_formal_gate_then_mstep")
    training_log = run_root / "formal_round1_quality_v3.log"
    with training_log.open("a") as output:
        training = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_em_training.py")],
            cwd=str(ROOT),
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if training.returncode != 0:
        write_state(state, "mstep_resume_failed", return_code=training.returncode)
        return training.returncode

    write_state(state, "starting_quality_gated_blind5")
    blind_output = run_root / "blind5_batch1"
    watcher = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/watch_and_run_blind10.py"),
        "--run-root", str(run_root),
        "--output-root", str(blind_output),
        "--case-list", str(blind_case_list),
        "--poll-sec", "10",
    ], cwd=str(ROOT), env=env, check=False)
    write_state(
        state,
        "complete" if watcher.returncode == 0 else "blind5_failed",
        return_code=watcher.returncode,
        blind_output=str(blind_output),
    )
    return watcher.returncode


if __name__ == "__main__":
    raise SystemExit(main())
