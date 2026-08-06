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

from tools.dataset_delivery.cads15_formal_launcher import CADS15_MODEL_TARGETS, DEFAULT_CASE_MANIFEST  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_smoke_validator import validate_mask  # noqa: E402


TARGET_TO_MODEL = {
    target: model
    for model, targets in CADS15_MODEL_TARGETS.items()
    for target in targets
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _final_row(run_out: Path, case_id: str, target: str) -> dict[str, Any]:
    doc = _read_json(run_out / "final_delivery_status.json")
    for row in doc.get("rows") or []:
        if row.get("case_id") == case_id and row.get("organ") == target:
            return dict(row)
    return {}


def build_status_matrix(
    *,
    output_root: Path,
    case_manifest: Path = DEFAULT_CASE_MANIFEST,
) -> dict[str, Any]:
    cases = _read_manifest(case_manifest)
    targets = sorted(TARGET_TO_MODEL)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = _case_id(case, index)
        ct = Path(_ct_path(case))
        for target in targets:
            model = TARGET_TO_MODEL[target]
            run_out = output_root / "cases" / case_id / model / "run_loop"
            mask_path = run_out / "annotation_versions" / case_id / "updated" / f"{target}.nii.gz"
            final = _final_row(run_out, case_id, target)
            validation = validate_mask(mask_path, ct if ct.exists() else None)
            final_status = str(final.get("final_status") or "")
            if final_status not in {"delivered", "delivered_for_review", "out_of_fov", "confirmed_absent", "failed"}:
                if validation["valid"]:
                    final_status = "delivered_for_review"
                else:
                    final_status = "failed"
            expected_presence = "expected_present" if final_status not in {"out_of_fov", "confirmed_absent"} else final_status
            failure_reason = ""
            if final_status in {"delivered", "delivered_for_review"} and not validation["valid"]:
                final_status = "failed"
                failure_reason = validation["reason"]
            elif final_status == "failed":
                failure_reason = validation["reason"] or ";".join(final.get("hard_errors") or [])
            rows.append({
                "case_id": case_id,
                "canonical_target": target,
                "model": model,
                "FOV status": final.get("fov_status", ""),
                "fov_status": final.get("fov_status", ""),
                "expected_presence": expected_presence,
                "inference_status": _read_json(run_out / "run_summary.json").get("status", ""),
                "output_path": str(mask_path),
                "nonzero_voxels": validation.get("foreground_voxels", 0),
                "NIfTI validation": validation.get("reason"),
                "nifti_validation": validation.get("reason"),
                "delivery_status": final_status,
                "QC status": final.get("delivery_status", ""),
                "qc_status": final.get("delivery_status", ""),
                "manual_review_flag": str(final_status == "delivered_for_review").lower(),
                "failure_reason": failure_reason,
            })
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["delivery_status"]] = counts.get(row["delivery_status"], 0) + 1
    status = "success" if not any(row["delivery_status"] == "failed" for row in rows) else "failed"
    report = {
        "status": status,
        "case_count": len(cases),
        "target_count": len(targets),
        "row_count": len(rows),
        "expected_row_count": len(cases) * len(targets),
        "delivery_status_counts": counts,
        "rows": rows,
    }
    write_csv(
        output_root / "cads15_100case_status_matrix.csv",
        rows,
        [
            "case_id", "canonical_target", "model", "fov_status", "expected_presence",
            "inference_status", "output_path", "nonzero_voxels", "nifti_validation",
            "delivery_status", "qc_status", "manual_review_flag", "failure_reason",
        ],
    )
    write_json(output_root / "cads15_100case_status_matrix.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CADS15 formal 100-case status matrix.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST, type=Path)
    args = parser.parse_args()
    report = build_status_matrix(output_root=args.output_root.resolve(), case_manifest=args.case_manifest.resolve())
    print(json.dumps({"status": report["status"], "row_count": report["row_count"]}, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
