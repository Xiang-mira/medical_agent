#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import run_teacher_case


def main() -> int:
    p = argparse.ArgumentParser(description="Run one fixed-manifest case through the existing Teacher run-loop.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gap-file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--case-index", type=int, default=None)
    p.add_argument("--code-root", type=Path, default=Path("."))
    p.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    p.add_argument("--teacher-entry", type=Path)
    p.add_argument("--registry", type=Path)
    p.add_argument("--models", default=os.environ.get("MEDAI_TEACHER_MODELS", "epai_20250421,vsmtrans"))
    p.add_argument("--target-config", type=Path)
    p.add_argument("--timeout-sec", type=int, default=int(os.environ.get("MEDAI_TEACHER_TIMEOUT_SEC", "1800")))
    p.add_argument("--device", default=os.environ.get("MEDAI_DEVICE"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    case_index = args.case_index
    if case_index is None:
        case_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    payload = vars(args)
    payload["case_index"] = case_index
    print(json.dumps(run_teacher_case(**payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

