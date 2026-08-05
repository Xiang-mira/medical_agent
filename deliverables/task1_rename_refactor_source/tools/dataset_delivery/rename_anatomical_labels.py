#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import apply_rename


def main() -> int:
    p = argparse.ArgumentParser(description="Safely rename existing anatomical NIfTI mask files.")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--mapping-file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False)
    mode.add_argument("--apply", action="store_true", default=False)
    args = p.parse_args()
    print(json.dumps(apply_rename(args.data_root, args.mapping_file, args.taxonomy, args.report, apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

