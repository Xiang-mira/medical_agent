#!/usr/bin/env python3
"""Verify a downloaded HF private migration repo against its TSV manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--skip-sha256", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8"), delimiter="\t"))
    failures: list[dict[str, str]] = []
    checked = 0
    checked_bytes = 0
    for row in rows:
        if row.get("destination") != "hf_private":
            continue
        repo_path = row["repo_path"]
        expected_size = int(row.get("size_bytes") or 0)
        expected_sha = row.get("sha256") or ""
        path = args.root / repo_path
        if not path.exists():
            failures.append({"repo_path": repo_path, "reason": "missing"})
            continue
        size = path.stat().st_size
        if size != expected_size:
            failures.append({"repo_path": repo_path, "reason": f"size_mismatch:{size}!={expected_size}"})
            continue
        if expected_sha and not args.skip_sha256:
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                failures.append({"repo_path": repo_path, "reason": "sha256_mismatch"})
                continue
        checked += 1
        checked_bytes += size

    payload = {
        "status": "passed" if not failures else "failed",
        "checked_files": checked,
        "checked_bytes": checked_bytes,
        "failure_count": len(failures),
        "failures": failures[:200],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
