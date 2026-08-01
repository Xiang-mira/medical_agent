#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import validate_100case_manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Validate the fixed 100-case manifest.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--no-exists-check", action="store_true")
    args = p.parse_args()
    print(json.dumps(validate_100case_manifest(args.manifest, check_exists=not args.no_exists_check, report=args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

