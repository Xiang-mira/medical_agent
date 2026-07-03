#!/usr/bin/env python3
"""Verify that the vendored LabelCritic source matches the pinned upstream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(lock_path: Path, vendor_root: Path) -> dict:
    lock = json.loads(lock_path.read_text())
    rows = []
    for relative_path, expected in sorted(lock["files"].items()):
        path = vendor_root / relative_path
        actual = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": relative_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "match" if actual == expected else "mismatch",
            }
        )
    passed = all(row["status"] == "match" for row in rows)
    return {
        "stage": "labelcritic_vendor_lock_audit",
        "status": "passed" if passed else "failed",
        "repository": lock["repository"],
        "commit": lock["commit"],
        "policy": lock["policy"],
        "vendor_root": str(vendor_root.resolve()),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        default=str(ROOT / "configs" / "labelcritic_vendor_lock.json"),
    )
    parser.add_argument(
        "--vendor-root",
        default=str(ROOT / "third_party" / "LabelCritic-main"),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(Path(args.lock), Path(args.vendor_root))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
