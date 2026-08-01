#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import package_dataset_overlay


def main() -> int:
    p = argparse.ArgumentParser(description="Package validated generated labels into a dataset overlay directory.")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--validation-report", required=True, type=Path)
    p.add_argument("--delivery-dir", required=True, type=Path)
    p.add_argument("--no-fail-on-conflict", action="store_true")
    args = p.parse_args()
    print(json.dumps(package_dataset_overlay(args.run_dir, args.manifest, args.validation_report, args.delivery_dir, fail_on_conflict=not args.no_fail_on_conflict), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

