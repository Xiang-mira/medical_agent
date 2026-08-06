#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ, _load_case_presence_context  # noqa: E402
from tools.dataset_delivery.cads15_contract_audit import CADS15_TARGETS, DEFAULT_CONTRACT, contract_targets  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402


DEFAULT_CASE_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _annotation_folder(row: dict[str, str]) -> str:
    return row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""


def _mask_positive(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "exists": path.exists(), "positive": False, "foreground_voxels": 0, "reason": ""}
    if not path.exists():
        result["reason"] = "missing_reference"
        return result
    try:
        if path.stat().st_size <= 0:
            result["reason"] = "empty_reference_file"
            return result
        import nibabel as nib
        import numpy as np

        arr = np.asanyarray(nib.load(str(path)).dataobj)
        voxels = int((arr != 0).sum())
        result["foreground_voxels"] = voxels
        result["positive"] = voxels > 0
        result["reason"] = "positive_reference" if voxels > 0 else "zero_reference"
    except Exception as exc:
        result["reason"] = f"reference_read_error:{type(exc).__name__}:{exc}"
    return result


def _reference_candidates(ref_dir: Path, target: str) -> list[Path]:
    return [
        ref_dir / f"{target}.nii.gz",
        ref_dir / "segmentations" / f"{target}.nii.gz",
        ref_dir / "updated" / f"{target}.nii.gz",
    ]


def _target_positive_reference(ref_dir: Path, target: str) -> dict[str, Any]:
    checks = [_mask_positive(path) for path in _reference_candidates(ref_dir, target)]
    positive = next((item for item in checks if item["positive"]), None)
    return positive or checks[0]


def build_smoke_panel(
    *,
    case_manifest: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    require_positive_reference: bool = True,
) -> dict[str, Any]:
    target_contracts = {row["canonical_id"]: row for row in contract_targets(contract_path)}
    targets = [target for target in CADS15_TARGETS if target in target_contracts]
    rows = _read_csv(case_manifest)
    target_coverage: dict[str, list[str]] = {target: [] for target in targets}
    case_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    context_root = output_root / "case_presence_context"
    for index, source_row in enumerate(rows):
        case_id = _case_id(source_row, index)
        ct = Path(_ct_path(source_row))
        ref_dir = Path(_annotation_folder(source_row))
        if not ct.exists() or not ref_dir.exists():
            continue
        normalized = {"case_id": case_id, "ct_path": str(ct), "annotation_folder": str(ref_dir)}
        context = _load_case_presence_context(normalized, ct, case_id, context_root / case_id)
        covered_targets: list[str] = []
        per_target: dict[str, Any] = {}
        for target in targets:
            fov_status = _fov_status_for_organ(target, context)
            ref = _target_positive_reference(ref_dir, target)
            eligible = fov_status not in {"out_of_fov", "unknown"} and (ref["positive"] or not require_positive_reference)
            per_target[target] = {
                "target": target,
                "fov_status": fov_status,
                "reference": ref,
                "eligible": eligible,
            }
            audit_rows.append({
                "case_id": case_id,
                "target": target,
                "fov_status": fov_status,
                "reference_path": ref["path"],
                "reference_positive": ref["positive"],
                "reference_foreground_voxels": ref["foreground_voxels"],
                "eligible": eligible,
                "reason": "target_in_fov_and_positive_reference" if eligible else ref["reason"] or fov_status,
            })
            if eligible:
                covered_targets.append(target)
                target_coverage[target].append(case_id)
        if covered_targets:
            case_records.append({
                "case_id": case_id,
                "ct_path": str(ct),
                "annotation_folder": str(ref_dir),
                "targets": sorted(covered_targets),
                "target_count": len(covered_targets),
                "per_target": per_target,
            })

    uncovered = sorted(target for target, case_ids in target_coverage.items() if not case_ids)
    selected: list[dict[str, Any]] = []
    uncovered_remaining = set(targets)
    candidates = sorted(case_records, key=lambda item: item["case_id"])
    while uncovered_remaining:
        best = max(
            candidates,
            key=lambda item: (
                len(set(item["targets"]) & uncovered_remaining),
                -len(item["targets"]),
                "".join(chr(255 - ord(ch)) for ch in item["case_id"]),
            ),
            default=None,
        )
        if best is None or not (set(best["targets"]) & uncovered_remaining):
            break
        chosen_targets = sorted(set(best["targets"]) & uncovered_remaining)
        selected.append({
            "case_id": best["case_id"],
            "ct_path": best["ct_path"],
            "annotation_folder": best["annotation_folder"],
            "targets": chosen_targets,
            "selection_reason": "target_in_fov_and_positive_reference",
        })
        uncovered_remaining -= set(chosen_targets)

    status = "READY_FOR_HPC_SMOKE" if not uncovered_remaining and not uncovered else "BLOCKED_NO_POSITIVE_SMOKE_CASE"
    panel = {
        "status": status,
        "read_only": True,
        "case_manifest": str(case_manifest),
        "contract_path": str(contract_path),
        "targets": targets,
        "tie_break_rules": [
            "eligible requires fov_status not in out_of_fov/unknown",
            "eligible requires positive reference mask unless disabled for tests",
            "greedy select max uncovered targets",
            "case_id ascending tie-break",
        ],
        "cases": selected,
        "target_coverage": {target: sorted(case_ids) for target, case_ids in target_coverage.items()},
        "uncovered_targets": sorted(set(uncovered) | uncovered_remaining),
        "audit_rows": audit_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "cads15_smoke_case_panel.json", panel)
    write_csv(
        output_root / "cads15_smoke_case_panel_rows.csv",
        audit_rows,
        [
            "case_id", "target", "fov_status", "reference_path", "reference_positive",
            "reference_foreground_voxels", "eligible", "reason",
        ],
    )
    manifest_rows = [
        {
            "case_id": case["case_id"],
            "ct_path": case["ct_path"],
            "annotation_folder": case["annotation_folder"],
            "targets": ",".join(case["targets"]),
        }
        for case in selected
    ]
    write_csv(output_root / "cads15_selected_case_targets.csv", manifest_rows, ["case_id", "ct_path", "annotation_folder", "targets"])
    lines = [
        "# CADS15 Smoke Case Panel",
        "",
        f"- Status: `{status}`",
        f"- Selected cases: `{len(selected)}`",
        f"- Covered targets: `{len(targets) - len(panel['uncovered_targets'])}/{len(targets)}`",
        f"- Uncovered targets: `{', '.join(panel['uncovered_targets']) or 'none'}`",
        "",
    ]
    for case in selected:
        lines.append(f"- `{case['case_id']}`: `{', '.join(case['targets'])}`")
    (output_root / "cads15_smoke_case_panel.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only CADS15 positive smoke case panel.")
    parser.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST, type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--allow-missing-positive-reference", action="store_true", help="Test-only: allow in-FOV cases without positive reference.")
    args = parser.parse_args()
    panel = build_smoke_panel(
        case_manifest=args.case_manifest.resolve(),
        output_root=args.output_root.resolve(),
        contract_path=args.contract.resolve(),
        require_positive_reference=not args.allow_missing_positive_reference,
    )
    print(json.dumps({"status": panel["status"], "case_count": len(panel["cases"])}, indent=2))
    return 0 if panel["status"] == "READY_FOR_HPC_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
