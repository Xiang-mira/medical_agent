#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import validate_generated_masks


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated NIfTI masks against CT geometry and delivery rules.")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gap-file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(validate_generated_masks(args.run_dir, args.manifest, args.gap_file, args.taxonomy, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

