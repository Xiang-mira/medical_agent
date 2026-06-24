#!/usr/bin/env python3
"""Read-only inventory of old Round1 masks reusable by 373-organ repair."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import _resolve_preseeded_case_dir
from cli_anything.medai.core.organ_taxonomy import normalize_canonical_id, taxonomy_entry


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preseed_roots(old_estep: Path) -> dict[str, Path]:
    cases_root = old_estep / "cases"
    models: set[str] = set()
    for case_dir in cases_root.iterdir() if cases_root.exists() else []:
        raw = case_dir / "raw_predictions"
        if raw.exists():
            models.update(path.name for path in raw.iterdir() if path.is_dir())
    return {model: cases_root / "{case_id}" / "raw_predictions" / model for model in sorted(models)}


def candidate_components(
    seg_dir: Path,
    organ: str,
    model: str,
    alias_config: dict[str, Any],
) -> tuple[list[Path], str]:
    files = {normalize_canonical_id(path.name[:-7]): path for path in seg_dir.glob("*.nii.gz")}
    direct = files.get(normalize_canonical_id(organ))
    if direct:
        return [direct], "direct"
    aliases = ((alias_config.get("models", {}) or {}).get(model, {}) or {})
    local_to_global = aliases.get("local_to_global", {}) or {}
    mapping_types = aliases.get("mapping_types", {}) or {}
    union_names = [
        str(local) for local, resolved in local_to_global.items()
        if normalize_canonical_id(resolved) == normalize_canonical_id(organ)
        and mapping_types.get(local) == "approved_union"
    ]
    if union_names:
        paths = [files.get(normalize_canonical_id(local)) for local in union_names]
        if all(paths):
            return [path for path in paths if path], "approved_union"
    for local, resolved in local_to_global.items():
        if normalize_canonical_id(resolved) == normalize_canonical_id(organ):
            path = files.get(normalize_canonical_id(local))
            if path:
                return [path], "declared_alias"
    return [], "missing"


def usable(paths: list[Path]) -> bool:
    if not paths:
        return False
    try:
        import nibabel as nib
        import numpy as np

        return all(int((np.asanyarray(nib.load(str(path)).dataobj) > 0).sum()) > 10 for path in paths)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-estep", default=str(ROOT / "outputs/round1/estep"))
    ap.add_argument("--case-list", default=str(ROOT / "data_manifest/case_list_50_tumor.csv"))
    ap.add_argument("--output", default=str(ROOT / "outputs/audits/round1_repair_reuse_audit.json"))
    args = ap.parse_args()

    old_estep = Path(args.old_estep).resolve()
    with Path(args.case_list).resolve().open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    targets = read_json(ROOT / "configs/student_3d_prompt_target_organs.json")["target_organs"]
    taxonomy = read_json(ROOT / "configs/organ_taxonomy.json")
    aliases = read_json(ROOT / "configs/model_label_aliases.json")
    routing = read_json(ROOT / "outputs/audit_21_models/routing_373_audit.json")
    major_targets = sorted(
        organ for organ in targets
        if (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "major"
    )
    child_targets = {
        organ for organ in targets
        if (taxonomy_entry(taxonomy, organ) or {}).get("hierarchy_role") == "child"
    }
    roots = preseed_roots(old_estep)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    quarantined_child_files = 0
    mode_counts: Counter[str] = Counter()

    for case in cases:
        case_id = str(case.get("case_id") or "")
        resolved_dirs = {
            model: directory
            for model, base in roots.items()
            if (directory := _resolve_preseeded_case_dir(base, case_id)) is not None
        }
        for directory in resolved_dirs.values():
            quarantined_child_files += sum(
                1 for path in directory.glob("*.nii.gz")
                if any(normalize_canonical_id(path.name[:-7]) == normalize_canonical_id(child) for child in child_targets)
            )
        for organ in major_targets:
            route_candidates = [
                str(item.get("model_key"))
                for item in ((routing.get("per_organ", {}) or {}).get(organ, {}) or {}).get("candidates", [])
                if item.get("resolvable")
            ]
            selected: dict[str, Any] | None = None
            for model in route_candidates:
                directory = resolved_dirs.get(model)
                if directory is None:
                    continue
                components, mode = candidate_components(directory, organ, model, aliases)
                if usable(components):
                    selected = {
                        "model": model,
                        "mode": mode,
                        "components": [str(path) for path in components],
                    }
                    mode_counts[mode] += 1
                    break
            row = {"case_id": case_id, "organ": organ, "reusable": selected is not None, **(selected or {})}
            rows.append(row)
            if selected is None:
                missing.append({"case_id": case_id, "organ": organ, "route_candidates": route_candidates})

    report = {
        "stage": "round1_repair_reuse_audit",
        "status": "success" if rows else "failed",
        "old_estep": str(old_estep),
        "case_list": str(Path(args.case_list).resolve()),
        "num_cases": len(cases),
        "num_models_in_old_cache": len(roots),
        "old_cache_models": sorted(roots),
        "num_major_targets": len(major_targets),
        "expected_case_major_pairs": len(cases) * len(major_targets),
        "reusable_case_major_pairs": sum(1 for row in rows if row["reusable"]),
        "missing_case_major_pairs": len(missing),
        "cases_with_all_major_targets_reusable": sum(
            1 for case in cases
            if not any(item["case_id"] == case.get("case_id") for item in missing)
        ),
        "reuse_mode_counts": dict(mode_counts),
        "quarantined_old_child_files": quarantined_child_files,
        "old_child_policy": "parent-cache-only; old child masks must not enter repair candidates or M-step",
        "missing": missing,
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in {"rows", "missing"}}, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(output), "missing_sample": missing[:10]}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
