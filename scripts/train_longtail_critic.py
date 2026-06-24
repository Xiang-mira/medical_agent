#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.longtail_critic import build_pairwise_dataset, train_pairwise_ranker


def main() -> int:
    ap = argparse.ArgumentParser(description="Build synthetic-corruption pairs and train the LongTailCritic ranker.")
    ap.add_argument("--passports", required=True, help="JSON list or JSONL of AutoLabelCore passports")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    source = Path(args.passports)
    if source.suffix == ".jsonl":
        passports = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        doc = json.loads(source.read_text())
        passports = doc.get("items", doc) if isinstance(doc, dict) else doc
    out = Path(args.output_dir)
    dataset = build_pairwise_dataset(passports, out / "longtail_critic_pairs.json")
    result = train_pairwise_ranker(out / "longtail_critic_pairs.json", out / "longtail_critic_model.json") if dataset["num_pairs"] else {"status": "failed", "reason": "no pairs"}
    print(json.dumps({"dataset": dataset, "training": result}, indent=2))
    return 0 if result.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
