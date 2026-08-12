#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import NIFTI_SUFFIX, sha256_file, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_smoke_validator import validate_mask  # noqa: E402


TERMINAL_STATES = {
    "SUCCESS",
    "FAILED",
    "PENDING",
    "ABSENT",
    "COMPLETED_NO_NONZERO",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_merge_rows(path: Path, row_key: tuple[str, str], row: dict[str, Any]) -> None:
    doc = _read_json(path)
    rows = [item for item in (doc.get("rows") or []) if isinstance(item, dict)]
    merged = []
    replaced = False
    for existing in rows:
        key = (str(existing.get("case_id") or ""), str(existing.get("organ") or existing.get("target") or ""))
        if key == row_key:
            merged.append(row)
            replaced = True
        else:
            merged.append(existing)
    if not replaced:
        merged.append(row)
    doc.update({"stage": "final_delivery_status", "status": "success", "rows": merged})
    write_json(path, doc)


def _run_out(output_root: Path, case_id: str, group: str) -> Path:
    candidates = [
        output_root / "cases" / case_id / group / "run_loop",
        output_root / group / "cases" / case_id / "run_loop",
        output_root / "cases" / case_id / "run_loop",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _candidate_id(case_id: str, target: str, teacher: str, raw_path: Path) -> str:
    payload = f"{case_id}|{target}|{teacher}|{raw_path.resolve()}"
    return "cand_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _infer_teacher(path: Path, *, models: list[str], group: str) -> str:
    parts = set(path.parts)
    for model in models:
        if model in parts:
            return model
    if group in parts:
        return group
    return models[0] if len(models) == 1 else group


def _raw_roots(run_out: Path) -> list[Path]:
    roots = [run_out / "raw_predictions", run_out]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.exists() and root not in seen:
            out.append(root)
            seen.add(root)
    return out


def discover_raw_candidates(
    *,
    output_root: Path,
    case_id: str,
    group: str,
    target: str,
    ct_path: Path,
    models: list[str],
) -> list[dict[str, Any]]:
    run_out = _run_out(output_root, case_id, group)
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in _raw_roots(run_out):
        for path in sorted(root.glob(f"**/segmentations/{target}{NIFTI_SUFFIX}")):
            if "annotation_versions" in path.parts or "selected_after_candidate_shapekit" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            validation = validate_mask(path, ct_path if ct_path.exists() else None)
            teacher = _infer_teacher(path, models=models, group=group)
            rows.append(
                {
                    "case_id": case_id,
                    "target": target,
                    "teacher": teacher,
                    "teacher_family": teacher,
                    "model": teacher,
                    "checkpoint": "",
                    "source_label": target,
                    "canonical_label": target,
                    "raw_mask_path": str(path),
                    "raw_mask_sha256": sha256_file(path) if path.exists() else "",
                    "raw_validation": validation,
                    "valid": bool(validation.get("valid")),
                    "nonzero_voxels": int(validation.get("foreground_voxels") or 0),
                    "candidate_id": _candidate_id(case_id, target, teacher, path),
                    "selection_result": "not_selected",
                    "selection_reason": "candidate_preserved_for_selection",
                }
            )
    return rows


def _task_completed_without_nonzero(run_out: Path, target: str) -> bool:
    for summary_path in sorted((run_out / "raw_predictions").glob("**/inference_summary.json")):
        summary = _read_json(summary_path)
        if summary.get("return_code") == 0 and str(summary.get("status") or "") in {"success", "completed"}:
            return True
    summary = _read_json(run_out / "run_summary.json")
    return (
        str(summary.get("status") or "") in {"success", "completed"}
        and target in {str(x).removesuffix(NIFTI_SUFFIX) for x in summary.get("requested_organs", []) or []}
    )


def _copy_idempotent(src: Path, dst: Path, *, backup_root: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src_hash = sha256_file(src)
        dst_hash = sha256_file(dst)
        if src_hash == dst_hash:
            return {"action": "kept_existing_identical", "backup_path": ""}
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_path = backup_root / f"{dst.name}.{dst_hash[:12]}.bak"
        if not backup_path.exists():
            shutil.copy2(dst, backup_path)
        shutil.copy2(src, dst)
        return {"action": "overwrote_after_backup", "backup_path": str(backup_path)}
    shutil.copy2(src, dst)
    return {"action": "copied", "backup_path": ""}


def recover_case_group(
    *,
    output_root: Path,
    case_id: str,
    group: str,
    targets: list[str],
    ct_path: Path,
    models: list[str],
    apply: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    run_out = _run_out(output_root, case_id, group)
    all_candidate_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    recovered = 0
    for target in targets:
        candidates = discover_raw_candidates(
            output_root=output_root,
            case_id=case_id,
            group=group,
            target=target,
            ct_path=ct_path,
            models=models,
        )
        valid_candidates = [row for row in candidates if row["valid"]]
        status = "PENDING"
        reason = "raw_teacher_output_not_found"
        if valid_candidates:
            status = "SUCCESS"
            reason = "valid_raw_candidate_recovered"
        elif candidates and any(int(row.get("nonzero_voxels") or 0) == 0 for row in candidates):
            status = "COMPLETED_NO_NONZERO"
            reason = "teacher_completed_but_mask_empty_or_invalid"
        elif _task_completed_without_nonzero(run_out, target):
            status = "ABSENT"
            reason = "teacher_completed_no_mask_materialized"
        target_row = {
            "case_id": case_id,
            "target": target,
            "group": group,
            "status": status,
            "reason": reason,
            "candidate_count": len(valid_candidates),
            "candidate_ids": [row["candidate_id"] for row in valid_candidates],
            "teacher_names": [row["teacher"] for row in valid_candidates],
        }
        if apply and valid_candidates:
            candidate_dir = run_out / "candidates" / case_id / target
            for row in valid_candidates:
                raw_path = Path(row["raw_mask_path"])
                staged = candidate_dir / f"{row['candidate_id']}{NIFTI_SUFFIX}"
                copy_result = _copy_idempotent(raw_path, staged, backup_root=run_out / "backups" / case_id / target)
                row.update(
                    {
                        "candidate_path": str(staged),
                        "candidate_valid": True,
                        "stage_action": copy_result["action"],
                        "backup_path": copy_result["backup_path"],
                    }
                )
            if len(valid_candidates) == 1:
                selected = valid_candidates[0]
                source = Path(str(selected.get("candidate_path") or selected["raw_mask_path"]))
                selected_path = run_out / "selected_after_candidate_shapekit" / case_id / "segmentations" / f"{target}{NIFTI_SUFFIX}"
                final_path = run_out / "annotation_versions" / case_id / "updated" / f"{target}{NIFTI_SUFFIX}"
                selected_copy = _copy_idempotent(source, selected_path, backup_root=run_out / "backups" / case_id / target)
                final_copy = _copy_idempotent(source, final_path, backup_root=run_out / "backups" / case_id / target)
                selected.update(
                    {
                        "selection_result": "selected",
                        "selection_reason": "single_valid_recovered_raw_candidate",
                        "selected_after_candidate_shapekit": str(selected_path),
                        "annotation_version": str(final_path),
                        "selected_copy_action": selected_copy["action"],
                        "final_copy_action": final_copy["action"],
                    }
                )
                recovered += 1
                final_row = {
                    "case_id": case_id,
                    "organ": target,
                    "final_status": "delivered_for_review",
                    "delivery_status": "delivered_for_review",
                    "fov_status": "fully_visible",
                    "mask_path": str(final_path),
                    "selection_status": "selected",
                    "selection_method": "single_valid_recovered_raw_candidate",
                    "candidate_count": 1,
                    "candidate_ids": [selected["candidate_id"]],
                    "candidate_models": [selected["model"]],
                    "selected_candidate": selected["candidate_id"],
                    "selected_model": selected["model"],
                    "raw_mask_path": selected["raw_mask_path"],
                    "raw_mask_sha256": selected["raw_mask_sha256"],
                    "shapekit_status": "recovered_existing_raw_candidate",
                    "provenance": "task2_recovery",
                }
                _write_json_merge_rows(run_out / "final_delivery_status.json", (case_id, target), final_row)
                target_row.update({"selected_candidate": selected["candidate_id"], "mask_path": str(final_path)})
            else:
                target_row["reason"] = "multiple_valid_candidates_require_labelcritic"
        all_candidate_rows.extend(candidates)
        target_rows.append(target_row)
    if apply:
        write_json(
            run_out / "candidate_recovery.json",
            {
                "stage": "task2_candidate_recovery",
                "status": "success",
                "case_id": case_id,
                "group": group,
                "targets": targets,
                "terminal_states": sorted(TERMINAL_STATES),
                "recovered_target_count": recovered,
                "target_rows": target_rows,
                "candidate_rows": all_candidate_rows,
            },
        )
        write_csv(
            run_out / "candidate_recovery.csv",
            all_candidate_rows,
            [
                "case_id", "target", "candidate_id", "teacher", "teacher_family",
                "model", "checkpoint", "source_label", "canonical_label",
                "raw_mask_path", "raw_mask_sha256", "valid", "nonzero_voxels",
                "candidate_path", "selection_result", "selection_reason",
            ],
        )
        case_meta = run_out / "annotation_versions" / case_id / "selection_metadata.json"
        selected_organs = []
        selection_rows = []
        for target in target_rows:
            selected = next(
                (row for row in all_candidate_rows if row["target"] == target["target"] and row.get("selection_result") == "selected"),
                None,
            )
            selection_rows.append(
                {
                    "case_id": case_id,
                    "organ": target["target"],
                    "candidate_count": target["candidate_count"],
                    "candidate_ids": target["candidate_ids"],
                    "candidate_models": target["teacher_names"],
                    "selection_status": "selected" if selected else "review_required",
                    "selection_method": selected.get("selection_reason") if selected else target["reason"],
                    "selected_candidate": selected.get("candidate_id") if selected else None,
                    "selected_model": selected.get("model") if selected else None,
                    "selected_prediction": selected.get("annotation_version") if selected else None,
                    "labelcritic_called": False,
                    "labelcritic_skipped_reason": "single_candidate_recovery" if selected else "requires_labelcritic_or_no_candidate",
                    "candidate_predictions": [row for row in all_candidate_rows if row["target"] == target["target"]],
                }
            )
            if selected:
                selected_organs.append(
                    {
                        "case_id": case_id,
                        "organ": target["target"],
                        "selected_candidate": selected["candidate_id"],
                        "selected_model": selected["model"],
                        "final_mask": selected["annotation_version"],
                        "mask_path": selected["annotation_version"],
                        "delivery_status": "delivered_for_review",
                        "selection_status": "selected",
                        "selection_method": "single_valid_recovered_raw_candidate",
                        "candidate_count": 1,
                        "candidate_models": [selected["model"]],
                        "shapekit_status": "recovered_existing_raw_candidate",
                    }
                )
        write_json(case_meta, {"case_id": case_id, "ct_path": str(ct_path), "selection_rows": selection_rows, "selected_organs": selected_organs})
        existing_summary = _read_json(run_out / "run_summary.json")
        existing_summary.update(
            {
                "status": "success" if recovered else existing_summary.get("status", "partial"),
                "candidate_recovery_status": "success",
                "candidate_recovery_recovered_target_count": recovered,
                "teacher_inference_models": sorted(set(existing_summary.get("teacher_inference_models", []) or models)),
                "teacher_inference_count": int(existing_summary.get("teacher_inference_count") or 1),
                "inference_success_count": int(existing_summary.get("inference_success_count") or (1 if all_candidate_rows else 0)),
            }
        )
        write_json(run_out / "run_summary.json", existing_summary)
    return {
        "status": "SUCCESS" if recovered else "PENDING",
        "case_id": case_id,
        "group": group,
        "targets": targets,
        "recovered_target_count": recovered,
        "target_rows": target_rows,
        "candidate_count": len([row for row in all_candidate_rows if row["valid"]]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover completed raw Teacher outputs into formal Task 2 candidates/delivery.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--targets", required=True, help="Comma-separated canonical targets.")
    parser.add_argument("--ct-path", required=True, type=Path)
    parser.add_argument("--models", default="", help="Comma-separated teacher/model keys.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = recover_case_group(
        output_root=args.output_root,
        case_id=args.case_id,
        group=args.group,
        targets=[x.strip() for x in args.targets.split(",") if x.strip()],
        ct_path=args.ct_path,
        models=[x.strip() for x in args.models.split(",") if x.strip()] or [args.group],
        apply=bool(args.apply),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"SUCCESS", "PENDING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
