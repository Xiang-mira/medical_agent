from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .auto_label_core import ACCEPTED_SCORING_SCHEMA_VERSIONS
from .json_utils import read_json, write_json
from .paths import resolve_path

OFFICIAL_ENCODER_MODE = "official_voxtell_nnunet_encoder_baseline"
LEGACY_ENCODER_MODE = "official_voxtell_nnunet_encoder_finetune"
LEGACY_OFFICIAL_MODE = "official_voxtell_finetune"
VOXTELL_NNUNET_TRAINER = "VoxTellTrainer_noMirroring"


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).resolve()
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    doc = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(doc, dict):
        rows = doc.get("items") or doc.get("rows") or []
    else:
        rows = doc
    return [r for r in rows if isinstance(r, dict)]


def _load_target_organs(target_config: str | Path) -> list[str]:
    doc = read_json(resolve_path(target_config), default={})
    organs = doc.get("target_organs") if isinstance(doc, dict) else None
    if not isinstance(organs, list):
        raise FileNotFoundError(f"Missing target_organs in target config: {target_config}")
    return [str(o) for o in organs]


def _is_supported_hard_ab(row: dict[str, Any]) -> tuple[bool, str | None]:
    grade = str(row.get("grade") or "D").upper()
    target_type = str(row.get("target_type") or "hard").lower()
    schema = str(row.get("scoring_schema_version") or "legacy")
    weight = float(row.get("training_weight") or 0.0)
    if schema not in ACCEPTED_SCORING_SCHEMA_VERSIONS:
        return False, "unsupported_scoring_schema_requires_rescoring"
    if grade not in {"A", "B"}:
        return False, f"grade_{grade}_excluded_from_official_nnunet_encoder_baseline"
    if target_type != "hard":
        return False, "official_nnunet_encoder_baseline_requires_hard_label"
    if weight <= 0.0 or row.get("distillation_eligible") is False:
        return False, row.get("distillation_exclusion_reason") or "training_weight_zero_or_ineligible"
    mask = row.get("mask") or row.get("mask_path")
    image = row.get("image") or row.get("ct_path")
    if not mask or not Path(str(mask)).exists():
        return False, "mask_missing"
    if not image or not Path(str(image)).exists():
        return False, "image_missing"
    return True, None


def convert_manifest_to_nnunet_dataset(
    manifest_path: str | Path,
    output_root: str | Path,
    dataset_id: int,
    dataset_name: str,
    target_config: str | Path,
    dry_run: bool = False,
    nnunet_raw: str | Path | None = None,
) -> dict[str, Any]:
    """Convert the project prompt-student manifest to official VoxTell nnU-Net raw format.

    nnU-Net labels must be consecutive (0, 1, 2, ...). The project frozen
    target IDs are preserved in label_mapping.json, but labelsTr/dataset.json
    use compressed nnU-Net IDs.
    """
    import nibabel as nib
    import numpy as np

    rows = _load_manifest(manifest_path)
    target_organs = _load_target_organs(target_config)
    frozen_label_map = {organ: idx for idx, organ in enumerate(target_organs, start=1)}
    target_order = {organ: idx for idx, organ in enumerate(target_organs)}
    root = Path(output_root).resolve()
    nnunet_raw_path = Path(nnunet_raw).resolve() if nnunet_raw else root / "nnUNet_raw"
    ds_folder = nnunet_raw_path / f"Dataset{int(dataset_id):03d}_{dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    audits = root / "converter_audit"
    if not dry_run:
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)
        audits.mkdir(parents=True, exist_ok=True)

    case_rows: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    included_organs: set[str] = set()
    for row in rows:
        organ = str(row.get("organ") or "").strip()
        case_id = str(row.get("case_id") or "").strip()
        if not organ or not case_id:
            exclusions.append({**row, "reason": "missing_case_id_or_organ"})
            continue
        if organ not in frozen_label_map:
            exclusions.append({**row, "reason": "organ_not_in_target_config"})
            continue
        ok, reason = _is_supported_hard_ab(row)
        if not ok:
            exclusions.append({**row, "reason": reason})
            continue
        case_rows.setdefault(case_id, []).append(row)
        included_organs.add(organ)

    ordered_included_organs = [organ for organ in target_organs if organ in included_organs]
    nnunet_label_map = {organ: idx for idx, organ in enumerate(ordered_included_organs, start=1)}
    label_records = [
        {
            "organ": organ,
            "nnunet_label_id": nnunet_label_map[organ],
            "project_frozen_target_id": frozen_label_map[organ],
            "target_config_order": target_order[organ],
        }
        for organ in ordered_included_organs
    ]

    prepared_cases: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    labels_used: dict[str, int] = {"background": 0}

    grade_rank = {"B": 1, "A": 2}
    for case_id, masks in sorted(case_rows.items()):
        masks = sorted(
            masks,
            key=lambda r: (
                grade_rank.get(str(r.get("grade") or "D").upper(), 0),
                float(r.get("training_weight") or 0.0),
                -frozen_label_map[str(r.get("organ"))],
            ),
        )
        image_src = Path(str(masks[0].get("image") or masks[0].get("ct_path"))).resolve()
        image_dst = images_tr / f"{case_id}_0000.nii.gz"
        label_dst = labels_tr / f"{case_id}.nii.gz"
        if dry_run:
            prepared_cases.append({"case_id": case_id, "image": str(image_dst), "label": str(label_dst), "num_masks": len(masks), "dry_run": True})
            for m in masks:
                organ = str(m["organ"])
                labels_used[organ] = nnunet_label_map[organ]
            continue
        if not image_dst.exists():
            shutil.copy2(image_src, image_dst)
        combined = None
        ref_img = None
        for m in masks:
            organ = str(m["organ"])
            nnunet_label_id = nnunet_label_map[organ]
            frozen_label_id = frozen_label_map[organ]
            img = nib.load(str(Path(str(m.get("mask") or m.get("mask_path"))).resolve()))
            arr = np.asarray(img.dataobj) > 0
            if combined is None:
                combined = np.zeros(arr.shape, dtype=np.uint16)
                ref_img = img
            if combined.shape != arr.shape:
                exclusions.append({**m, "reason": "mask_shape_mismatch"})
                continue
            overlap = arr & (combined > 0)
            if bool(overlap.any()):
                old_ids, counts = np.unique(combined[overlap], return_counts=True)
                reverse_nnunet = {v: k for k, v in nnunet_label_map.items()}
                for old_id, count in zip(old_ids.tolist(), counts.tolist()):
                    old_organ = reverse_nnunet.get(int(old_id), str(old_id))
                    conflicts.append({
                        "case_id": case_id,
                        "new_organ": organ,
                        "new_nnunet_label_id": nnunet_label_id,
                        "new_project_frozen_target_id": frozen_label_id,
                        "overwritten_organ": old_organ,
                        "overwritten_nnunet_label_id": int(old_id),
                        "overwritten_project_frozen_target_id": frozen_label_map.get(old_organ),
                        "overlap_voxels": int(count),
                        "resolution": "higher_grade_then_training_weight_then_lower_frozen_target_id_priority",
                    })
            combined[arr] = nnunet_label_id
            labels_used[organ] = nnunet_label_id
        if combined is not None and ref_img is not None:
            nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), str(label_dst))
        prepared_cases.append({"case_id": case_id, "image": str(image_dst), "label": str(label_dst), "num_masks": len(masks), "label_built": label_dst.exists()})

    dataset_json = {
        "name": dataset_name,
        "channel_names": {"0": "CT"},
        "labels": dict(sorted(labels_used.items(), key=lambda kv: kv[1])),
        "numTraining": len(prepared_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    label_mapping = {
        "source": "configs/student_3d_prompt_target_organs.json target_organs order",
        "labels": label_records,
        "nnunet_label_ids": dict(sorted(nnunet_label_map.items(), key=lambda kv: kv[1])),
        "project_frozen_target_ids": {organ: frozen_label_map[organ] for organ in ordered_included_organs},
        "full_project_frozen_target_ids": frozen_label_map,
    }
    audit = {
        "stage": "voxtell_manifest_to_nnunet_dataset",
        "status": "dry_run" if dry_run else "success",
        "training_mode": OFFICIAL_ENCODER_MODE,
        "converter_role": "baseline_only",
        "baseline_only": True,
        "prompt_conditioned": False,
        "main_mstep_allowed": False,
        "eligible_for_next_round_prompt_student": False,
        "input_manifest_type": "prompt_level_selected_pseudo_labels",
        "output_label_format": "nnunet_multiclass",
        "not_used_for_prompt_student_training": True,
        "input_manifest": str(Path(manifest_path).resolve()),
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name,
        "dataset_folder": str(ds_folder),
        "nnUNet_raw": str(nnunet_raw_path),
        "num_input_rows": len(rows),
        "num_cases": len(prepared_cases),
        "num_labels": max(len(labels_used) - 1, 0),
        "num_exclusions": len(exclusions),
        "num_overlap_conflicts": len(conflicts),
        "included_policy": "A/B hard pseudo-labels only for explicit official nnU-Net encoder baseline; C soft/hard, D, unsupported schema excluded.",
        "label_id_policy": "dataset.json and labelsTr use consecutive nnU-Net IDs; project frozen target IDs are preserved in label_mapping.json.",
        "overlap_policy": "Higher grade, then higher training_weight, then lower frozen target id wins.",
        "prepared_cases": prepared_cases[:50],
    }
    if not dry_run:
        write_json(ds_folder / "dataset.json", dataset_json)
        write_json(ds_folder / "label_mapping.json", label_mapping)
        write_json(root / "converter_audit.json", audit)
        write_json(root / "training_exclusions.json", exclusions)
        conflicts_csv = root / "overlap_conflicts.csv"
        with conflicts_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "case_id", "new_organ", "new_nnunet_label_id", "new_project_frozen_target_id",
                "overwritten_organ", "overwritten_nnunet_label_id", "overwritten_project_frozen_target_id",
                "overlap_voxels", "resolution",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(conflicts)
        audit.update({
            "dataset_json": str(ds_folder / "dataset.json"),
            "label_mapping": str(ds_folder / "label_mapping.json"),
            "training_exclusions": str(root / "training_exclusions.json"),
            "overlap_conflicts": str(conflicts_csv),
        })
    else:
        audit.update({"dataset_json_plan": dataset_json, "label_mapping_plan": label_mapping, "training_exclusions_preview": exclusions[:50], "overlap_conflicts_preview": conflicts[:50]})
    return audit


def official_encoder_transfer_preflight(
    nnunet_raw: str | Path,
    nnunet_preprocessed: str | Path,
    nnunet_results: str | Path,
    dataset_id: int | str,
    dataset_name: str,
    pretrained_checkpoint: str | Path,
    trainer: str = VOXTELL_NNUNET_TRAINER,
) -> dict[str, Any]:
    ds_folder = Path(nnunet_raw) / f"Dataset{int(dataset_id):03d}_{dataset_name}"
    checks: dict[str, Any] = {
        "voxtell_finetune": shutil.which("voxtell-finetune"),
        "nnUNetv2_plan_and_preprocess": shutil.which("nnUNetv2_plan_and_preprocess"),
        "nnUNet_raw": str(nnunet_raw),
        "nnUNet_preprocessed": str(nnunet_preprocessed),
        "nnUNet_results": str(nnunet_results),
        "dataset_folder": str(ds_folder),
        "dataset_json_exists": (ds_folder / "dataset.json").exists(),
        "imagesTr_exists": (ds_folder / "imagesTr").exists(),
        "labelsTr_exists": (ds_folder / "labelsTr").exists(),
        "pretrained_checkpoint_exists": Path(pretrained_checkpoint).exists(),
        "trainer": trainer,
    }
    errors: list[str] = []
    if checks["voxtell_finetune"] is None:
        errors.append("voxtell-finetune_not_found")
    if checks["nnUNetv2_plan_and_preprocess"] is None:
        errors.append("nnUNetv2_plan_and_preprocess_not_found")
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not str(checks[key]):
            errors.append(f"{key}_not_set")
    for key in ("dataset_json_exists", "imagesTr_exists", "labelsTr_exists", "pretrained_checkpoint_exists"):
        if not checks[key]:
            errors.append(key.replace("_exists", "_missing"))
    import_code = (
        "import voxtell.training.run_finetuning as r; "
        "from voxtell.training import VoxTellTrainer, VoxTellTrainer_noMirroring; "
        "print(r, VoxTellTrainer, VoxTellTrainer_noMirroring)"
    )
    import_env = os.environ.copy()
    import_env.update({
        "nnUNet_raw": str(nnunet_raw),
        "nnUNet_preprocessed": str(nnunet_preprocessed),
        "nnUNet_results": str(nnunet_results),
    })
    proc = subprocess.run([sys.executable, "-c", import_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=import_env)
    checks["training_import_return_code"] = proc.returncode
    checks["training_import_stdout"] = (proc.stdout or "")[-1000:]
    checks["training_import_stderr"] = (proc.stderr or "")[-1000:]
    if proc.returncode != 0:
        errors.append("voxtell_training_import_failed")
    checks["status"] = "passed" if not errors else "failed"
    checks["errors"] = errors
    return checks


def build_official_encoder_transfer_commands(
    dataset_id: int,
    configuration: str,
    fold: str,
    pretrained_checkpoint: str | Path,
    trainer: str = VOXTELL_NNUNET_TRAINER,
    plans: str = "nnUNetPlans",
) -> dict[str, list[str]]:
    return {
        "preprocess_command": ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
        "train_command": [
            "voxtell-finetune",
            str(dataset_id),
            configuration,
            str(fold),
            "-tr",
            trainer,
            "-p",
            plans,
            "-pretrained_weights",
            str(pretrained_checkpoint),
        ],
    }
