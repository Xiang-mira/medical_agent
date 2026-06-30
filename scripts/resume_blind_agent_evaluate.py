#!/usr/bin/env python3
"""Resume Blind evaluation from the Agent phase without rerunning Student inference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_state(path: Path, status: str, **extra) -> None:
    path.write_text(json.dumps({"status": status, **extra}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--case-list", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    case_list = Path(args.case_list).resolve()
    output = run_root / "blind5_batch1"
    state = run_root / "case10_continuation_state.json"
    protocol = ROOT / "scripts" / "run_blind10_protocol.py"

    write_state(state, "running_blind_agent2", case_list=str(case_list))
    agent = subprocess.run([
        sys.executable, str(protocol), "agent",
        "--output-root", str(output),
        "--case-list", str(case_list),
    ], cwd=str(ROOT), check=False)
    if agent.returncode != 0:
        write_state(state, "blind_agent2_failed", return_code=agent.returncode)
        return agent.returncode

    write_state(state, "evaluating_blind2", case_list=str(case_list))
    evaluation = subprocess.run([
        sys.executable, str(protocol), "evaluate",
        "--output-root", str(output),
        "--case-list", str(case_list),
    ], cwd=str(ROOT), check=False)
    write_state(
        state,
        "complete" if evaluation.returncode == 0 else "blind2_evaluation_failed",
        return_code=evaluation.returncode,
        blind_output=str(output),
    )
    return evaluation.returncode


if __name__ == "__main__":
    raise SystemExit(main())
