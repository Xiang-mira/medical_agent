#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import run_preflight


def main() -> int:
    p = argparse.ArgumentParser(description="Run minimum preflight checks before 100-case Teacher inference.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gap-file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--rename-mapping", type=Path)
    p.add_argument("--teacher-entry", type=Path, default=Path("run_medai_cli.py"))
    p.add_argument("--registry", type=Path, default=Path("configs/model_registry.yaml"))
    p.add_argument("--output-root", type=Path)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--mask-root", type=Path)
    p.add_argument("--code-root", type=Path)
    args = p.parse_args()
    print(json.dumps(run_preflight(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

