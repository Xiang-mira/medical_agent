#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.organ_router import route_organs
from cli_anything.medai.core.organ_taxonomy import load_taxonomy, normalize_canonical_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact organ identities across taxonomy, routes, branches and aliases.")
    parser.add_argument("--output", default=str(ROOT / "configs/organ_identity_mapping_audit.json"))
    args = parser.parse_args()
    taxonomy = load_taxonomy(ROOT / "configs/organ_taxonomy.json")
    targets = json.loads((ROOT / "configs/student_3d_prompt_target_organs.json").read_text())["target_organs"]
    branches = yaml.safe_load((ROOT / "configs/teacher_branch_map.yaml").read_text()) or {}
    aliases = json.loads((ROOT / "configs/model_label_aliases.json").read_text())
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    taxonomy_organs = taxonomy.get("organs", {}) or {}
    missing_taxonomy = sorted(target for target in targets if normalize_canonical_id(target) not in taxonomy_organs)
    if missing_taxonomy:
        errors.append({"type": "formal_targets_missing_taxonomy", "organs": missing_taxonomy})

    routes = route_organs(targets)
    if routes.get("missing_organs"):
        errors.append({"type": "formal_targets_missing_route", "organs": routes["missing_organs"]})

    missing_explicit_branch = sorted(set(targets) - set(branches))
    if missing_explicit_branch:
        warnings.append({
            "type": "formal_targets_use_workbook_route_without_branch_override",
            "count": len(missing_explicit_branch),
            "organs": missing_explicit_branch,
        })

    forbidden = {(normalize_canonical_id(a), normalize_canonical_id(b)) for a, b in aliases.get("forbidden_coarsening", [])}
    by_global: dict[tuple[str, str], list[str]] = defaultdict(list)
    for model, model_aliases in (aliases.get("models", {}) or {}).items():
        mapping_types = model_aliases.get("mapping_types", {}) or {}
        for local, global_name in (model_aliases.get("local_to_global", {}) or {}).items():
            local_id, global_id = normalize_canonical_id(local), normalize_canonical_id(global_name)
            mapping_type = str(mapping_types.get(local, "exact_synonym"))
            if (local_id, global_id) in forbidden or mapping_type == "forbidden_coarsening":
                errors.append({"type": "forbidden_alias_active", "model": model, "local": local, "global": global_name})
            if global_id not in taxonomy_organs:
                errors.append({"type": "alias_target_missing_taxonomy", "model": model, "local": local, "global": global_name})
            by_global[(model, global_id)].append(str(local))
        for (alias_model, global_id), locals_ in list(by_global.items()):
            if alias_model != model or len(locals_) < 2:
                continue
            if not all(mapping_types.get(local) == "approved_union" for local in locals_):
                errors.append({"type": "unapproved_many_to_one_alias", "model": model, "global": global_id, "locals": locals_})

    critical = {
        "liver": ("major", "whole_organ", []),
        "liver_segment_1": ("child", "sub_organ", ["liver"]),
        "pancreas": ("major", "whole_organ", []),
        "pancreas_head": ("child", "sub_organ", ["pancreas"]),
        "liver_portal_vein": ("child", "vessel_or_small_structure", ["liver"]),
    }
    for organ, expected in critical.items():
        entry = taxonomy_organs.get(organ, {})
        actual = (entry.get("hierarchy_role"), entry.get("comparison_family"), entry.get("parent_ids", []))
        if actual != expected:
            errors.append({"type": "critical_taxonomy_mismatch", "organ": organ, "expected": expected, "actual": actual})

    report = {
        "stage": "organ_identity_mapping_audit",
        "status": "success" if not errors else "failed",
        "formal_targets": len(targets),
        "taxonomy_organs": len(taxonomy_organs),
        "workbook_sha256": taxonomy.get("source_sha256"),
        "missing_taxonomy": missing_taxonomy,
        "missing_routes": routes.get("missing_organs", []),
        "errors": errors,
        "warnings": warnings,
        "policy": {
            "comparison_requires_exact_canonical_id": True,
            "parent_relationship_is_roi_only": True,
            "formal_default_teacher_fallback": False,
        },
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "formal_targets": len(targets), "errors": len(errors), "warnings": len(warnings)}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
