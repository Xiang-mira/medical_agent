#!/usr/bin/env python3
"""Read-only-input audit and baseline snapshot for AutoLabelCore rollout."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.auto_label_core import SCORING_SCHEMA_VERSION, load_autolabel_config
from cli_anything.medai.core.model_registry import load_registry
from cli_anything.medai.core.target_space import validate_formal_373_target_space


def _rows(path: Path | None) -> list[dict]:
    if not path or not path.exists(): return []
    if path.suffix == ".jsonl": return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    doc = json.loads(path.read_text())
    return doc.get("items", doc.get("selections", doc)) if isinstance(doc, dict) else doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", help="Optional legacy/new selection JSON or JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--registry", default=str(ROOT / "configs" / "model_registry.yaml"))
    args = ap.parse_args()
    config = load_autolabel_config()
    registry = load_registry(args.registry)
    targets = validate_formal_373_target_space(require_full_target=False)
    rows = _rows(Path(args.selection).resolve() if args.selection else None)
    required_lineage = {"evidence_family", "architecture_lineage", "training_data_lineage", "is_student"}
    lineage_errors = {
        key: sorted(required_lineage - set(entry))
        for key, entry in registry.get("models", {}).items()
        if required_lineage - set(entry)
    }
    snapshot = {
        "stage": "autolabel_core_baseline_and_contract_audit", "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success" if targets["status"] == "success" and not lineage_errors else "failed",
        "scoring_schema_version": SCORING_SCHEMA_VERSION, "config_schema_version": config["schema_version"],
        "formal_373_target_validation": targets, "registry_lineage_errors": lineage_errors,
        "legacy_input_is_accuracy": False, "num_selection_rows": len(rows),
        "grade_counts": dict(Counter(str(r.get("grade") or "missing") for r in rows)),
        "training_weight_counts": dict(Counter(str(r.get("training_weight")) for r in rows)),
        "route_counts": dict(Counter(str(r.get("selected_model") or "missing") for r in rows)),
        "accuracy_warning": "Baseline values are process/pseudo-consistency state, not expert accuracy.",
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": snapshot["status"], "output": str(out)}, indent=2))
    return 0 if snapshot["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
