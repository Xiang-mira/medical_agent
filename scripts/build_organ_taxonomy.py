#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.model_registry import parse_checkpoint_map
from cli_anything.medai.core.organ_taxonomy import build_taxonomy
from cli_anything.medai.core.organ_router import route_organs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict organ taxonomy from the checkpoint workbook.")
    parser.add_argument("--workbook", default=str(ROOT / "configs/class_checkpoint_map_updates.xlsx"))
    parser.add_argument("--output", default=str(ROOT / "configs/organ_taxonomy.json"))
    args = parser.parse_args()
    parsed = parse_checkpoint_map(args.workbook)
    if parsed.get("status") != "success":
        raise SystemExit(json.dumps(parsed, indent=2))
    taxonomy = build_taxonomy(parsed["organs"], args.workbook)
    routed = route_organs(list(taxonomy["organs"]))
    branch_map = yaml.safe_load((ROOT / "configs/teacher_branch_map.yaml").read_text(encoding="utf-8")) or {}
    for organ, entry in taxonomy["organs"].items():
        runnable = [
            str(item.get("model_key"))
            for item in (routed.get("ranked_candidates", {}) or {}).get(organ, [])
            if item.get("model_key")
        ]
        if runnable:
            branch = branch_map.get(organ, {}) or {}
            branch_primary = branch.get("teacher_model")
            if branch_primary in runnable:
                runnable = [str(branch_primary), *[model for model in runnable if model != branch_primary]]
            entry["candidate_models"] = runnable
            entry["primary_teacher"] = runnable[0]
            entry["backup_teachers"] = runnable[1:]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "num_organs": taxonomy["num_organs"], "validation": taxonomy["validation"]}, indent=2))
    return 0 if taxonomy["validation"]["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
