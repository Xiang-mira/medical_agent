#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import validate_gap_resolution


def main() -> int:
    p = argparse.ArgumentParser(description="Validate rename/generate/manual_review organ gap resolution table.")
    p.add_argument("--gap-file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--rename-mapping", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    print(json.dumps(validate_gap_resolution(args.gap_file, args.taxonomy, args.rename_mapping, args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

