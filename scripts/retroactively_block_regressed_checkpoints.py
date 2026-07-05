#!/usr/bin/env python3
"""Fail-closed migration for historically promoted but regressed checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = PROJECT_ROOT / "agent-harness"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from cli_anything.medai.core.continual_learning import (  # noqa: E402
    retroactively_block_rounds,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round", type=int, action="append", dest="rounds", required=True)
    parser.add_argument("--reason", default="retrospective_quality_regression")
    parser.add_argument("--evidence-json", type=Path)
    args = parser.parse_args()

    registry_path = args.output_root / "checkpoint_promotion_registry.json"
    if not registry_path.is_file():
        raise SystemExit(f"promotion registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    evidence = (
        json.loads(args.evidence_json.read_text(encoding="utf-8"))
        if args.evidence_json
        else {}
    )
    updated, changed = retroactively_block_rounds(
        registry,
        args.rounds,
        reason=args.reason,
        evidence=evidence,
    )
    write_json(registry_path, updated)
    audit = {
        "stage": "retrospective_checkpoint_block",
        "status": "success",
        "registry_path": str(registry_path.resolve()),
        "requested_rounds": sorted(set(args.rounds)),
        "changed_rounds": changed,
        "reason": args.reason,
        "evidence_path": str(args.evidence_json.resolve()) if args.evidence_json else None,
    }
    write_json(args.output_root / "retrospective_checkpoint_block_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
