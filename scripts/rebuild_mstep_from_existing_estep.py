#!/usr/bin/env python3
"""Replay metadata from an existing E-step and rebuild M-step without inference."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-harness"))
sys.path.insert(0, str(ROOT / "scripts"))

from cli_anything.medai.core.multimodel_loop import (
    _load_case_presence_context,
    _materialize_case_373_targets,
)
from cli_anything.medai.core.voxtell_student import VoxTellStudent


def link_masks(source: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for mask in source.glob("*.nii.gz"):
        target = destination / mask.name
        if target.exists():
            count += 1
            continue
        try:
            target.symlink_to(mask.resolve())
        except OSError:
            shutil.copy2(mask, target)
        count += 1
    return count


def link_tree(source: Path, destination: Path) -> bool:
    """Expose an existing artifact tree without rerunning inference.

    Metadata replay is meant to be read-only with respect to the original
    E-step.  Prefer a symlink for large teacher-cache directories; fall back to
    copytree only on filesystems that disallow directory symlinks.
    """
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return True
    try:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reuse completed E-step masks/metadata; do not run teachers, LabelCritic or GPU inference."
    )
    ap.add_argument("--source-estep", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--case-list", type=Path)
    ap.add_argument(
        "--case-ids",
        nargs="*",
        default=[],
        help="Optional exact case IDs to replay; omitted means every completed source case.",
    )
    ap.add_argument(
        "--ct-root",
        type=Path,
        default=ROOT / "data" / "PanTS" / "ImageTr",
        help="Case-folder CT root used when a requested case has no source E-step metadata.",
    )
    ap.add_argument("--target-config", type=Path, default=ROOT / "configs/student_3d_prompt_target_organs.json")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints/VoxTell/voxtell_v1.1")
    ap.add_argument(
        "--absent-negative-training-weight",
        type=float,
        default=float(os.getenv("MEDAI_NEGATIVE_ABSENT_TRAINING_WEIGHT", "0.1")),
        help="Loss/sampling weight for scan-coverage-proven all-zero targets.",
    )
    args = ap.parse_args()
    if not 0.0 <= args.absent_negative_training_weight <= 1.0:
        raise SystemExit("--absent-negative-training-weight must be in [0, 1]")

    source_versions = args.source_estep.resolve() / "annotation_versions"
    source_cases = args.source_estep.resolve() / "cases"
    replay_estep = args.output_root.resolve() / "estep"
    replay_versions = replay_estep / "annotation_versions"
    replay_cases = replay_estep / "cases"
    mstep = args.output_root.resolve() / "mstep"
    replay_versions.mkdir(parents=True, exist_ok=True)
    linked_cases_cache = link_tree(source_cases, replay_cases)
    target_doc = json.loads(args.target_config.read_text(encoding="utf-8"))
    organs = list(target_doc["target_organs"])
    cases = []
    full_items = []
    processed_case_ids: set[str] = set()
    for metadata_path in sorted(source_versions.glob("*/selection_metadata.json")):
        doc = json.loads(metadata_path.read_text(encoding="utf-8"))
        case_id = str(doc.get("case_id") or metadata_path.parent.name)
        if args.case_ids and case_id not in set(args.case_ids):
            continue
        ct = Path(str(doc.get("ct_path") or ""))
        source_updated = metadata_path.parent / "updated"
        if not source_updated.exists():
            source_updated = metadata_path.parent
        replay_case = replay_versions / case_id
        replay_updated = replay_case / "updated"
        linked = link_masks(source_updated, replay_updated)
        current_presence_context = _load_case_presence_context(
            {
                "case_id": case_id,
                "dataset_name": "PanTS" if "pants" in f"{case_id} {ct}".lower() else "",
            },
            ct,
            case_id,
            replay_case,
        )
        selection_rows = [dict(x) for x in doc.get("selection_rows", []) if isinstance(x, dict)]
        selected = [dict(x) for x in doc.get("selected_organs", []) if isinstance(x, dict)]
        summary = _materialize_case_373_targets(
            case_id=case_id,
            ct=ct,
            organs=organs,
            case_updated=replay_updated,
            selection_rows=selection_rows,
            selected_metadata=selected,
            presence_context=current_presence_context,
            negative_absent_training_weight=args.absent_negative_training_weight,
        )
        for row in [*selection_rows, *selected]:
            row["evidence_revalidated_under_current_adapter"] = False
            if row.get("target_type") in {"positive_hard", "positive_soft"}:
                row["training_weight"] = 0.0
                row["distillation_eligible"] = False
                row["should_enter_student_training"] = False
                row["training_block_reason"] = (
                    "metadata_replay_did_not_rerun_current_labelcritic_adapter"
                )
        replay_doc = {
            **doc,
            "metadata_replay": {
                "source_selection_metadata": str(metadata_path),
                "teacher_inference_rerun": False,
                "labelcritic_rerun": False,
                "linked_existing_masks": linked,
            },
            "case_373_target_summary": summary,
            "case_presence_context": current_presence_context,
            "selection_rows": selection_rows,
            "selected_organs": selected,
        }
        replay_case.mkdir(parents=True, exist_ok=True)
        (replay_case / "selection_metadata.json").write_text(
            json.dumps(replay_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cases.append({
            "case_id": case_id,
            "linked_masks": linked,
            "absent_added": summary.get("absent_negative_added", 0),
            "review_gap_missing": summary.get("review_gap_missing", 0),
        })
        full_items.extend(selection_rows)
        processed_case_ids.add(case_id)

    # Contract-only cases are legitimate: no teacher output means unresolved,
    # never absent by itself. FOV may still establish semantic negatives.
    for case_id in args.case_ids:
        if case_id in processed_case_ids:
            continue
        ct = (args.ct_root / case_id / "ct.nii.gz").resolve()
        if not ct.is_file():
            raise FileNotFoundError(f"Requested contract case has no CT: {ct}")
        replay_case = replay_versions / case_id
        replay_updated = replay_case / "updated"
        replay_updated.mkdir(parents=True, exist_ok=True)
        current_presence_context = _load_case_presence_context(
            {"case_id": case_id, "dataset_name": "PanTS"},
            ct,
            case_id,
            replay_case,
        )
        selection_rows: list[dict] = []
        selected: list[dict] = []
        summary = _materialize_case_373_targets(
            case_id=case_id,
            ct=ct,
            organs=organs,
            case_updated=replay_updated,
            selection_rows=selection_rows,
            selected_metadata=selected,
            presence_context=current_presence_context,
            negative_absent_training_weight=args.absent_negative_training_weight,
        )
        for row in selection_rows:
            row["contract_only_no_teacher_evidence"] = True
        replay_doc = {
            "case_id": case_id,
            "ct_path": str(ct),
            "case_presence_context": current_presence_context,
            "metadata_replay": {
                "source_selection_metadata": None,
                "teacher_inference_rerun": False,
                "labelcritic_rerun": False,
                "contract_only_no_teacher_evidence": True,
            },
            "case_373_target_summary": summary,
            "selection_rows": selection_rows,
            "selected_organs": selected,
        }
        (replay_case / "selection_metadata.json").write_text(
            json.dumps(replay_doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cases.append({
            "case_id": case_id,
            "linked_masks": 0,
            "absent_added": summary.get("absent_negative_added", 0),
            "review_gap_missing": summary.get("review_gap_missing", 0),
            "contract_only_no_teacher_evidence": True,
        })
        full_items.extend(selection_rows)
        processed_case_ids.add(case_id)

    student = VoxTellStudent(
        model_dir=args.model_dir,
        target_config=args.target_config,
        device="cpu",
    )
    manifest_path = mstep / "voxtell_prompt_student_manifest.json"
    manifest = student.build_training_manifest(
        cases_root=replay_versions,
        output_manifest=manifest_path,
        case_list=args.case_list.resolve() if args.case_list else None,
        require_images=True,
    )
    report = {
        "stage": "rebuild_mstep_from_existing_estep",
        "status": "success",
        "teacher_inference_rerun": False,
        "labelcritic_rerun": False,
        "source_estep": str(args.source_estep.resolve()),
        "replay_estep": str(replay_estep),
        "linked_existing_cases_cache": linked_cases_cache,
        "source_cases_cache": str(source_cases),
        "replay_cases_cache": str(replay_cases),
        "manifest": str(manifest_path),
        "num_cases": len(cases),
        "absent_negative_targets": manifest.get("absent_negative_targets", 0),
        "absent_negative_training_weight": args.absent_negative_training_weight,
        "positive_items": manifest.get("num_positive_items", 0),
        "negative_items": manifest.get("num_negative_items", 0),
        "cases": cases,
    }
    mstep.mkdir(parents=True, exist_ok=True)
    full_manifest = {
        "stage": "full_case_373_estep_manifest",
        "status": "success" if len(full_items) == len(cases) * len(organs) else "failed",
        "num_cases": len(cases),
        "num_classes": len(organs),
        "expected_targets": len(cases) * len(organs),
        "actual_targets": len(full_items),
        "items": full_items,
    }
    (replay_estep / "full_case_373_manifest.json").write_text(
        json.dumps(full_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (mstep / "existing_estep_replay_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
