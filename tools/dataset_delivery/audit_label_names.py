#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import audit_label_names


def main() -> int:
    p = argparse.ArgumentParser(description="Audit dataset mask names against the canonical 373 target list.")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit_label_names(args.data_root, args.taxonomy, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

