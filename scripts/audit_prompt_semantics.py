#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit prompt banks/manifests for dataset-definition semantic hazards.")
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=ROOT / "configs/prompt_semantic_policy.yaml")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def load_policy(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def flatten_prompts_from_target_config(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    banks = doc.get("organ_prompt_bank", {}) if isinstance(doc, dict) else {}
    variants = doc.get("prompt_variants", {}) if isinstance(doc, dict) else {}
    for organ, bank in banks.items():
        if isinstance(bank, dict):
            rows.append({"organ": organ, "prompt_source": "canonical", "prompt": bank.get("canonical_prompt", "")})
            prompts = bank.get("prompts", {})
            if isinstance(prompts, dict):
                for source, vals in prompts.items():
                    for prompt in vals or []:
                        rows.append({"organ": organ, "prompt_source": source, "prompt": prompt})
        for prompt in variants.get(organ, []) or []:
            rows.append({"organ": organ, "prompt_source": "variant", "prompt": prompt})
    return rows


def flatten_prompts_from_manifest(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in doc.get("items", []) or []:
        rows.append({
            "case_id": item.get("case_id", ""),
            "organ": item.get("organ", ""),
            "prompt_source": item.get("prompt_source", ""),
            "prompt": item.get("prompt") or item.get("prompt_text") or "",
            "grade": item.get("grade", ""),
            "target_type": item.get("target_type", ""),
        })
    return rows


def main() -> int:
    args = parse_args()
    policy = load_policy(args.policy)
    banned = list(policy.get("default_banned_terms", []) or [])
    rows = flatten_prompts_from_manifest(args.manifest) if args.manifest else flatten_prompts_from_target_config(args.target_config)
    out_rows = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        low = prompt.lower()
        hits = [term for term in banned if str(term).lower() in low]
        replacement = ""
        for bad, repl in (policy.get("safe_replacements", {}) or {}).items():
            if str(bad).lower() in low:
                replacement = str(repl)
                if bad not in hits:
                    hits.append(str(bad))
        if hits:
            out = dict(row)
            out["semantic_flags"] = ";".join(hits)
            out["recommended_replacement"] = replacement
            out_rows.append(out)
    df = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summary = {
        "audited_prompts": len(rows),
        "flagged_prompts": len(out_rows),
        "flagged_organs": sorted(df["organ"].dropna().unique().tolist()) if not df.empty else [],
        "output": str(args.output),
    }
    (args.output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
