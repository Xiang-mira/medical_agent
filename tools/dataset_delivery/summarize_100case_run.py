#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import summarize_run


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize validated, failed, and incomplete 100-case Teacher outputs.")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(summarize_run(args.run_dir, args.manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

