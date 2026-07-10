#!/usr/bin/env python3
"""Prepare the formal 25-case Round1 run with 10-case pseudo-label reuse.

The generated case list intentionally contains only ``case_id`` and ``ct_path``.
PanTS ``annotation_folder`` paths are not copied into the formal run input,
because the mainline experiment is pseudo-label supervision only.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REUSED_CASE_IDS = [
    "PanTS_00000031",
    "PanTS_00000086",
    "PanTS_00000100",
    "PanTS_00000145",
    "PanTS_00000162",
    "PanTS_00000246",
    "PanTS_00000270",
    "PanTS_00000363",
    "PanTS_00000423",
    "PanTS_00000449",
]

EXPECTED_NEW_CASE_IDS = [
    "PanTS_00000026",
    "PanTS_00000029",
    "PanTS_00000035",
    "PanTS_00000047",
    "PanTS_00000049",
    "PanTS_00000074",
    "PanTS_00000224",
    "PanTS_00000368",
    "PanTS_00000416",
    "PanTS_00000451",
    "PanTS_00000465",
    "PanTS_00000482",
    "PanTS_00000485",
    "PanTS_00000488",
    "PanTS_00000554",
]

PER_CASE_ESTEP_DIRS = ["cases", "annotation_versions", "critic", "autolabel_core", "standard_dataset"]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
            if str(row.get("case_id") or "").strip()
        ]


def write_pseudo_only_case_list(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"case_id": row["case_id"], "ct_path": row["ct_path"]})


def link_or_copy_tree(source: Path, destination: Path, *, copy: bool) -> str:
    if destination.exists() or destination.is_symlink():
        return "already_present"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, destination)
        return "copied"
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
        return "symlinked"
    except OSError:
        shutil.copytree(source, destination)
        return "copied_after_symlink_failed"


def target_organs() -> list[str]:
    doc = read_json(ROOT / "configs" / "student_3d_prompt_target_organs.json", {})
    return [str(x).strip() for x in doc.get("target_organs", []) if str(x).strip()]


def audit_reused_case(baseline_estep: Path, case_id: str, targets: set[str]) -> dict[str, Any]:
    ann_dir = baseline_estep / "annotation_versions" / case_id
    case_dir = baseline_estep / "cases" / case_id
    meta_path = ann_dir / "selection_metadata.json"
    updated_dir = ann_dir / "updated"
    meta = read_json(meta_path, {})
    rows = [row for row in (meta.get("selection_rows") or []) if isinstance(row, dict)]
    row_organs = {str(row.get("organ") or "") for row in rows if row.get("organ")}
    selected = [row for row in (meta.get("selected_organs") or []) if isinstance(row, dict)]
    missing_selected_masks = []
    for row in selected:
        organ = str(row.get("organ") or "")
        if (
            organ
            and row.get("publication_status") != "rejected_but_recorded"
            and not (updated_dir / f"{organ}.nii.gz").is_file()
        ):
            missing_selected_masks.append(organ)
    hierarchy = read_json(case_dir / "hierarchical_inference_plan.json", {})
    status = "passed"
    reasons: list[str] = []
    if not meta_path.is_file():
        reasons.append("selection_metadata_missing")
    if not updated_dir.is_dir():
        reasons.append("updated_dir_missing")
    if len(row_organs) != len(targets) or row_organs != targets:
        reasons.append("selection_rows_do_not_cover_373_targets")
    if missing_selected_masks:
        reasons.append("selected_mask_missing")
    if hierarchy.get("teacher_inference_mode") != "hierarchical_roi":
        reasons.append("hierarchical_roi_manifest_missing_or_stale")
    if reasons:
        status = "failed"
    return {
        "case_id": case_id,
        "status": status,
        "reasons": reasons,
        "selection_metadata": str(meta_path),
        "updated_dir": str(updated_dir),
        "selection_row_count": len(rows),
        "target_count": len(targets),
        "missing_selected_masks": missing_selected_masks[:50],
        "reuse_role": "reused_round1_selected_pseudo_label",
    }


def _sanitize_reuse_row(row: dict[str, Any], *, case_id: str, organ: str) -> dict[str, Any]:
    out = dict(row)
    legacy_gt = out.pop("ground_truth_status", None)
    if legacy_gt is not None:
        out["legacy_ground_truth_status"] = legacy_gt
    out["accuracy_claim_allowed"] = False
    out["accuracy_warning"] = "Pseudo-label supervision only; not true accuracy."
    out["case_id"] = case_id
    out["organ"] = organ
    out.setdefault("ct_path", out.get("image"))
    out["label_source_role"] = "reused_round1_selected_pseudo_label"
    out["metric_target"] = "selected_pseudo_label"
    out["metric_interpretation"] = "pseudo_label_consistency"
    return out


def materialize_manifest_reuse_case(
    *,
    manifest_rows: list[dict[str, Any]],
    output_estep: Path,
    case_id: str,
    targets: set[str],
    copy: bool,
) -> dict[str, Any]:
    case_rows = [
        row for row in manifest_rows
        if isinstance(row, dict) and str(row.get("case_id") or "") == case_id
    ]
    by_organ = {str(row.get("organ") or ""): row for row in case_rows if row.get("organ")}
    ann_dir = output_estep / "annotation_versions" / case_id
    updated_dir = ann_dir / "updated"
    updated_dir.mkdir(parents=True, exist_ok=True)
    selected_organs: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    linked_positive = 0
    missing_positive_masks: list[str] = []
    for organ in sorted(targets):
        source_row = by_organ.get(organ) or {"case_id": case_id, "organ": organ}
        row = _sanitize_reuse_row(source_row, case_id=case_id, organ=organ)
        source_mask_raw = row.get("final_mask") or row.get("mask_path") or row.get("mask")
        source_mask = Path(str(source_mask_raw)).expanduser() if source_mask_raw else None
        is_positive = (
            str(row.get("supervision_type") or "").lower() == "positive"
            and str(row.get("target_type") or "").lower() in {"positive_hard", "positive_soft", "hard", "soft"}
            and bool(source_mask and source_mask.exists())
            and float(row.get("training_weight") or 0.0) > 0.0
        )
        if is_positive:
            dst = updated_dir / f"{organ}.nii.gz"
            if not dst.exists() and not dst.is_symlink():
                if copy:
                    shutil.copy2(source_mask, dst)
                else:
                    try:
                        dst.symlink_to(source_mask.resolve())
                    except OSError:
                        shutil.copy2(source_mask, dst)
            row.update({
                "final_mask": str(dst),
                "mask_path": str(dst),
                "mask": str(dst),
                "label_role": "selected_pseudo_label",
                "supervision_role": "selected_pseudo_label",
                "supervision_type": "positive",
                "target_type": "positive_hard" if str(row.get("target_type") or "").lower() != "positive_soft" else "positive_soft",
                "publication_status": "selected",
                "selection_status": "selected",
                "distillation_eligible": True,
                "training_weight": float(row.get("training_weight") or 1.0),
            })
            linked_positive += 1
        else:
            if str(source_row.get("supervision_type") or "").lower() == "positive" and source_mask_raw and not (source_mask and source_mask.exists()):
                missing_positive_masks.append(organ)
            row.update({
                "final_mask": "",
                "mask_path": "",
                "mask": "",
                "label_role": "withheld_uncertain",
                "supervision_role": "withheld_uncertain",
                "supervision_type": "withheld_uncertain",
                "legacy_target_type": row.get("target_type"),
                "target_type": "withheld_uncertain",
                "publication_status": "rejected_but_recorded",
                "selection_status": "withheld_uncertain",
                "distillation_eligible": False,
                "training_weight": 0.0,
                "grade": row.get("grade") or "D",
            })
        row.setdefault("candidate_models", [])
        row.setdefault("candidate_predictions", [])
        selected_organs.append(dict(row))
        selection_rows.append(dict(row))
    meta = {
        "case_id": case_id,
        "quality_contract_version": "estep_quality_contract_v3",
        "fov_policy_version": "fov_appearance_regions_v4",
        "reuse_role": "reused_round1_selected_pseudo_label",
        "source_manifest": "outputs/em_round_pure_cached_10case_formal_lite_20260703/round1/estep/full_case_373_manifest.json",
        "accuracy_claim_allowed": False,
        "metric_policy": "selected pseudo labels are student supervision, not GT accuracy",
        "selection_rows": selection_rows,
        "selected_organs": selected_organs,
        "target_count": len(targets),
        "positive_mask_count": linked_positive,
        "withheld_uncertain_count": len(targets) - linked_positive,
    }
    (ann_dir / "selection_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": case_id,
        "status": "passed" if len(selection_rows) == len(targets) and not missing_positive_masks else "failed",
        "reasons": ([] if len(selection_rows) == len(targets) else ["selection_rows_do_not_cover_373_targets"]) + (["positive_mask_missing"] if missing_positive_masks else []),
        "selection_metadata": str(ann_dir / "selection_metadata.json"),
        "updated_dir": str(updated_dir),
        "selection_row_count": len(selection_rows),
        "target_count": len(targets),
        "positive_mask_count": linked_positive,
        "withheld_uncertain_count": len(targets) - linked_positive,
        "missing_selected_masks": missing_positive_masks[:50],
        "reuse_role": "reused_round1_selected_pseudo_label",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-case-list", type=Path, default=ROOT / "data_manifest" / "case_list_50_tumor.csv")
    ap.add_argument("--output-case-list", type=Path, default=ROOT / "data_manifest" / "case_list_25_tumor.csv")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--baseline-estep", type=Path, default=ROOT / "outputs" / "formal_round1_final_20260627" / "round1" / "estep")
    ap.add_argument("--full-373-manifest", type=Path, default=ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703" / "round1" / "estep" / "full_case_373_manifest.json")
    ap.add_argument("--num-cases", type=int, default=25)
    ap.add_argument("--copy", action="store_true", help="Copy reusable artifacts instead of symlinking them.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    baseline_estep = args.baseline_estep.resolve()
    rows = read_case_rows(args.source_case_list)
    selected_rows = rows[: args.num_cases]
    case_ids = [row["case_id"] for row in selected_rows]
    reused = [case_id for case_id in case_ids if case_id in set(REUSED_CASE_IDS)]
    new = [case_id for case_id in case_ids if case_id not in set(REUSED_CASE_IDS)]
    failures: list[str] = []
    if len(selected_rows) != args.num_cases:
        failures.append("source_case_list_too_short")
    if sorted(reused) != sorted(REUSED_CASE_IDS):
        failures.append("reused_case_set_mismatch")
    if sorted(new) != sorted(EXPECTED_NEW_CASE_IDS):
        failures.append("new_case_set_mismatch")
    missing_ct = [row["case_id"] for row in selected_rows if not Path(row["ct_path"]).is_file()]
    if missing_ct:
        failures.append("ct_path_missing")

    write_pseudo_only_case_list(args.output_case_list, selected_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    round1_estep = output_root / "round1" / "estep"
    round1_estep.mkdir(parents=True, exist_ok=True)

    targets = set(target_organs())
    full_manifest = read_json(args.full_373_manifest, {})
    full_manifest_rows = [
        row for row in (full_manifest.get("items") or [])
        if isinstance(row, dict)
    ]
    reuse_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, str]] = []
    for case_id in reused:
        audit = audit_reused_case(baseline_estep, case_id, targets)
        use_manifest_repair = audit["status"] != "passed" and bool(full_manifest_rows)
        for dirname in PER_CASE_ESTEP_DIRS:
            if dirname == "annotation_versions" and use_manifest_repair:
                continue
            source = baseline_estep / dirname / case_id
            if not source.exists():
                continue
            destination = round1_estep / dirname / case_id
            action = link_or_copy_tree(source, destination, copy=args.copy)
            link_rows.append({
                "case_id": case_id,
                "artifact_group": dirname,
                "source": str(source),
                "destination": str(destination),
                "action": action,
            })
        if use_manifest_repair:
            audit = materialize_manifest_reuse_case(
                manifest_rows=full_manifest_rows,
                output_estep=round1_estep,
                case_id=case_id,
                targets=targets,
                copy=args.copy,
            )
            audit["repair_source"] = str(args.full_373_manifest.resolve())
        reuse_rows.append(audit)
        if audit["status"] != "passed":
            failures.append(f"reused_case_failed:{case_id}")

    report = {
        "stage": "prepare_round1_25case_reuse",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "source_case_list": str(args.source_case_list.resolve()),
        "output_case_list": str(args.output_case_list.resolve()),
        "output_root": str(output_root),
        "baseline_estep": str(baseline_estep),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "reused_case_ids": reused,
        "new_case_ids": new,
        "case_list_policy": "pseudo-only case list: annotation_folder/GT references intentionally omitted",
        "reuse_policy": "Only prior Round1 teacher-derived selected pseudo labels and teacher cache artifacts are reused; no old M-step checkpoint is reused.",
        "target_count": len(targets),
        "reused_case_audit": reuse_rows,
        "linked_artifacts": link_rows,
    }
    report_path = output_root / "teacher_assets" / "round1_25case_reuse" / "prepare_round1_25case_reuse.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
