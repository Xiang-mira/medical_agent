#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.resource_guard import inspect_gpu_resources


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative GPU test resource gate.")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    args = parser.parse_args()
    result = inspect_gpu_resources(args.min_free_mib, args.max_utilization, args.samples, args.interval_sec)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
