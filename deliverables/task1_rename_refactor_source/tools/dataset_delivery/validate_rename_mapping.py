#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.dataset_delivery.delivery_lib import validate_rename_mapping


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a confirmed anatomical-label rename mapping.")
    p.add_argument("--mapping", "--mapping-file", dest="mapping_file", required=True, type=Path)
    p.add_argument("--taxonomy", required=True, type=Path)
    p.add_argument("--task2-targets", type=Path)
    p.add_argument("--alias-groups", type=Path)
    p.add_argument("--boundary-classification", type=Path)
    p.add_argument("--standalone-dir", type=Path)
    p.add_argument("--report", "--output-json", dest="output_json", type=Path)
    p.add_argument("--output-md", type=Path)
    args = p.parse_args()
    boundary = args.boundary_classification
    if boundary is None:
        sibling = args.mapping_file.parent / "task_boundary_classification.csv"
        boundary = sibling if sibling.exists() else None
    try:
        result = validate_rename_mapping(
            args.mapping_file,
            args.taxonomy,
            args.output_json,
            task2_targets=args.task2_targets,
            alias_groups=args.alias_groups,
            boundary_classification=boundary,
            output_md=args.output_md,
            standalone_dir=args.standalone_dir,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "error": str(exc)}
        print(json.dumps(failure, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
