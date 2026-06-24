"""VoxTell-style 3D prompt-based student wrapper.

This is the new student direction: 3D CT in, text prompts in, one 3D mask per
prompt out. It intentionally does not use VISTA3D label IDs or a 127-class
student target space.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import csv
from pathlib import Path
from typing import Any

from .auto_fine_label import grade_to_training_weight
from .auto_label_core import ACCEPTED_SCORING_SCHEMA_VERSIONS
from .json_utils import read_json, write_json
from .organ_prompt_bank import (
    flatten_prompt_bank_entry,
    prompt_category_for,
    prompt_record_for,
    select_balanced_prompt_variants,
    select_prompt_for_organ,
)
from .paths import resolve_path
from .target_space import validate_formal_373_target_space


DEFAULT_TARGETS = "configs/student_3d_prompt_target_organs.json"
DEFAULT_SANITY_PROMPTS = ["liver", "spleen", "pancreas", "kidney_left", "aorta"]
NONMEDICAL_NEGATIVE_PROMPTS = [
    ("negative_nonmedical_cat", "segment the cat"),
    ("negative_nonmedical_dog", "segment the dog"),
    ("negative_nonmedical_car", "segment the car"),
    ("negative_nonmedical_tree", "segment the tree"),
]
OUT_OF_SCAN_ANATOMY_NEGATIVE_PROMPTS = [
    ("negative_out_of_scan_head", "segment the head"),
    ("negative_out_of_scan_brain", "segment the brain"),
    ("negative_out_of_scan_skull", "segment the skull"),
]
ABDOMEN_PELVIS_COVERAGE_TERMS = {
    "abdomen",
    "abdominal",
    "abdomen_pelvis",
    "abdomen-pelvis",
    "abdominopelvic",
    "pelvis",
    "pelvic",
}
HEAD_COVERAGE_TERMS = {"head", "brain", "cranial", "skull", "neck", "head_neck", "head-neck"}


def load_prompt_targets(target_config: str | Path = DEFAULT_TARGETS) -> dict[str, Any]:
    path = resolve_path(target_config)
    data = read_json(path, default={})
    if not isinstance(data, dict) or "target_organs" not in data:
        raise FileNotFoundError(f"Missing 3D prompt student target config: {path}")
    return data


def load_case_image_lookup(case_list: str | Path | None = None) -> dict[str, str]:
    """Load case_id -> CT path from the project case-list CSV format."""
    if not case_list:
        return {}
    path = Path(case_list).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Case list not found: {path}")
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            case_id = (row.get("case_id") or "").strip()
            ct_path = (row.get("ct_path") or row.get("image") or row.get("ct_image") or "").strip()
            if case_id and ct_path:
                lookup[case_id] = str(Path(ct_path).resolve())
    return lookup


def _find_mask_dir(case_dir: Path) -> Path | None:
    for name in ("segmentations", "updated"):
        candidate = case_dir / name
        if candidate.exists() and any(candidate.glob("*.nii.gz")):
            return candidate
    direct_masks = [
        p for p in case_dir.glob("*.nii.gz")
        if p.name not in {"image.nii.gz", "ct.nii.gz", "zero_mask.nii.gz", "combined_labels.nii.gz"}
    ] if case_dir.exists() else []
    if direct_masks:
        return case_dir
    return None


def _find_case_image(case_dir: Path) -> str | None:
    for name in ("image.nii.gz", "ct.nii.gz"):
        candidate = case_dir / name
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _load_selection_index(cases_root: Path, case_id: str) -> dict[str, dict[str, Any]]:
    """Load source/quality metadata written by the E-step selection loop."""
    candidates = [
        cases_root / case_id / "selection_metadata.json",
        cases_root.parent / "cases" / case_id / "pseudo_label_selection.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        selected = doc.get("selected_organs") or []
        if not isinstance(selected, list):
            continue
        return {
            str(item.get("organ")): item
            for item in selected
            if isinstance(item, dict) and item.get("organ")
        }
    return {}


def _load_case_label_mapping(case_dir: Path) -> dict[str, dict[str, Any]]:
    path = case_dir / "label_mapping.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = doc.get("mappings") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        organ = str(row.get("canonical_organ_name") or row.get("organ") or "").strip()
        if organ:
            out[organ] = row
    return out


def _safe_prompt_name(prompt: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in prompt)
    return safe_name.replace(" ", "_")


def _input_stem_for_voxtell(path: Path) -> str:
    input_filename = path.stem
    if input_filename.endswith(".nii"):
        input_filename = input_filename[:-4]
    return input_filename


def _voxtell_suffix(path: Path) -> str:
    if path.suffix == ".gz" and path.stem.endswith(".nii"):
        return ".nii.gz"
    return path.suffix


def _is_voxtell_model_dir(path: Path) -> bool:
    return (path / "plans.json").exists() and (path / "fold_0" / "checkpoint_final.pth").exists()


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0 or size >= len(items):
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _mask_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "mask_path": str(path), "mask_voxels": None, "empty_mask": None}
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(path))
        arr = np.asarray(img.get_fdata() > 0)
        voxels = int(arr.sum())
        bbox = None
        if voxels > 0:
            coords = np.argwhere(arr)
            bbox = {
                "min": [int(x) for x in coords.min(axis=0)],
                "max": [int(x) for x in coords.max(axis=0)],
            }
        spacing = [float(x) for x in img.header.get_zooms()[:3]]
        try:
            voxel_volume = float(abs(np.linalg.det(img.affine[:3, :3])))
        except Exception:
            voxel_volume = float(np.prod(spacing))
        return {
            "status": "success",
            "mask_path": str(path),
            "shape": [int(x) for x in img.shape[:3]],
            "spacing": spacing,
            "orientation_note": "VoxTell expects correctly reorientable RAS metadata via NibabelIOWithReorient.",
            "mask_voxels": voxels,
            "empty_mask": voxels == 0,
            "bbox": bbox,
            "mask_volume_mm3": round(float(voxels * voxel_volume), 6),
        }
    except Exception as exc:
        return {
            "status": "unreadable",
            "mask_path": str(path),
            "mask_voxels": None,
            "empty_mask": None,
            "reason": str(exc),
        }


def _quality_status(review_flags: Any, quality_flags: Any) -> str:
    flags = set(review_flags or []) | set(quality_flags or [])
    if not flags:
        return "ok"
    if {"missing_candidate", "missing_final_mask"} & flags:
        return "missing"
    if any(str(flag).startswith("shapekit_") for flag in flags):
        return "postprocess_review"
    if "selection_fallback" in flags:
        return "selection_review"
    return "review"


def _labelcritic_decision_path(records: Any) -> str | None:
    for record in records or []:
        if isinstance(record, dict) and record.get("output_json"):
            return str(record["output_json"])
    return None


class VoxTellStudent:
    """3D prompt-based student using a VoxTell-compatible model directory."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        gpu: int = 0,
        target_config: str | Path = DEFAULT_TARGETS,
        text_encoding_model: str | Path | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        self.gpu = gpu
        self.target_config = resolve_path(target_config)
        self.text_encoding_model = str(
            text_encoding_model
            or os.getenv("MEDAI_TEXT_ENCODING_MODEL")
            or resolve_path("checkpoints/Qwen/Qwen3-Embedding-4B")
        )
        self.python_executable = str(python_executable or sys.executable)

    def _target_doc(self) -> dict[str, Any]:
        return load_prompt_targets(self.target_config)

    def _select_organs(self, prompts: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
        doc = self._target_doc()
        prompt_sampling = os.getenv("MEDAI_PROMPT_SAMPLING", "canonical").strip().lower() or "canonical"
        if prompts:
            organs = [p for p in prompts if p in set(doc.get("target_organs", []))]
            unknown = [p for p in prompts if p not in set(doc.get("target_organs", []))]
            if unknown:
                raise ValueError(f"Unknown or non-target organs for VoxTell student: {unknown[:20]}")
        else:
            organs = list(doc.get("target_organs", []))
        return organs, {
            organ: select_prompt_for_organ(doc, organ, mode=prompt_sampling, seed=str(self.target_config))
            for organ in organs
        }

    def segment(
        self,
        ct_image: str | Path,
        output_dir: str | Path,
        prompts: list[str] | None = None,
        dry_run: bool = False,
        timeout_sec: int = 1800,
        prompt_batch_size: int = 16,
        prompt_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Segment a 3D CT volume with text prompts.

        Parameters
        ----------
        ct_image:
            Path to a 3D CT NIfTI.
        output_dir:
            Directory for per-organ masks.
        prompts:
            Optional list of global organ names. If omitted, all 373 configured
            target organs are requested.
        prompt_overrides:
            Optional organ-key to free-text prompt mapping. This keeps project
            outputs named by organ while sending the variant text to VoxTell.
        dry_run:
            Build the command and expected output contract without running.
        """
        ct = Path(ct_image).resolve()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        organs, organ_to_prompt = self._select_organs(prompts)
        if prompt_overrides:
            for organ, prompt_text in prompt_overrides.items():
                if organ in organ_to_prompt and str(prompt_text).strip():
                    organ_to_prompt[organ] = str(prompt_text).strip()
        prompt_texts = [organ_to_prompt[o] for o in organs]

        def command_for(batch_organs: list[str]) -> list[str]:
            batch_prompt_texts = [organ_to_prompt[o] for o in batch_organs]
            return [
                self.python_executable,
                "-m",
                "voxtell.inference.predict_from_raw_data",
                "--input",
                str(ct),
                "--output",
                str(out),
                "--model",
                str(self.model_dir),
                "--prompts",
                *batch_prompt_texts,
                "--device",
                self.device,
                "--gpu",
                str(self.gpu),
                "--text-encoding-model",
                self.text_encoding_model,
            ]

        command = [
            self.python_executable,
            "-m",
            "voxtell.inference.predict_from_raw_data",
            "--input",
            str(ct),
            "--output",
            str(out),
            "--model",
            str(self.model_dir),
            "--prompts",
            *prompt_texts,
            "--device",
            self.device,
            "--gpu",
            str(self.gpu),
            "--text-encoding-model",
            self.text_encoding_model,
        ]
        expected_masks = {organ: str(out / f"{organ}.nii.gz") for organ in organs}
        official_output_masks = {
            organ: str(out / f"{_input_stem_for_voxtell(ct)}_{_safe_prompt_name(organ_to_prompt[organ])}{_voxtell_suffix(ct)}")
            for organ in organs
        }
        result: dict[str, Any] = {
            "stage": "voxtell_3d_prompt_student_inference",
            "status": "dry_run" if dry_run else "pending",
            "ct_image": str(ct),
            "output_dir": str(out),
            "model_dir": str(self.model_dir),
            "text_encoding_model": self.text_encoding_model,
            "target_config": str(self.target_config),
            "num_prompts": len(organs),
            "organs": organs,
            "organ_to_prompt": organ_to_prompt,
            "prompt_sampling": os.getenv("MEDAI_PROMPT_SAMPLING", "canonical"),
            "expected_masks": expected_masks,
            "official_output_masks": official_output_masks,
            "command": command,
            "prompt_batch_size": prompt_batch_size,
            "prompt_overrides": prompt_overrides or {},
            "num_batches": len(_chunked(organs, prompt_batch_size)),
            "batch_commands": [command_for(batch) for batch in _chunked(organs, prompt_batch_size)],
            "formal_373_target_validation": validate_formal_373_target_space(
                self.target_config,
                requested_organs=organs,
                require_full_target=len(organs) == 373,
            ),
            "note": "3D prompt-based student; no VISTA3D label-id mapping is used.",
            "io_contract": {
                "official_input": "3D NIfTI + free-text prompt list + VoxTell model dir + Qwen3 text encoder",
                "official_output": "one binary mask per prompt, named <input_stem>_<prompt>.nii.gz",
                "project_output": "one binary mask per organ, standardized to <organ>.nii.gz",
                "combined_multilabel_policy": "not used formally because overlapping prompt masks can overwrite earlier labels",
            },
        }

        if dry_run:
            write_json(out / "voxtell_student_plan.json", result)
            return result

        if not ct.exists():
            result.update({"status": "failed", "reason": f"CT not found: {ct}"})
            write_json(out / "voxtell_student_result.json", result)
            return result
        if not self.model_dir.exists():
            result.update({"status": "failed", "reason": f"VoxTell model dir not found: {self.model_dir}"})
            write_json(out / "voxtell_student_result.json", result)
            return result
        if not _is_voxtell_model_dir(self.model_dir):
            result.update({
                "status": "failed",
                "reason": f"Invalid VoxTell model dir: expected plans.json and fold_0/checkpoint_final.pth under {self.model_dir}",
            })
            write_json(out / "voxtell_student_result.json", result)
            return result

        start = time.time()
        standardized_masks: dict[str, str] = {}
        batch_results: list[dict[str, Any]] = []
        per_organ_status: dict[str, dict[str, Any]] = {}
        stdout_tail = ""
        stderr_tail = ""

        for batch_idx, batch_organs in enumerate(_chunked(organs, prompt_batch_size), start=1):
            batch_command = command_for(batch_organs)
            batch_start = time.time()
            try:
                proc = subprocess.run(
                    batch_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=timeout_sec,
                )
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(
                    batch_command,
                    124,
                    stdout=exc.stdout or "",
                    stderr=(exc.stderr or "") + f"\n[VoxTellStudent] Timeout after {timeout_sec}s.",
                )
                timed_out = True
            stdout_tail = ((stdout_tail + "\n" + (proc.stdout or ""))[-4000:])
            stderr_tail = ((stderr_tail + "\n" + (proc.stderr or ""))[-4000:])
            batch_status = "timed_out" if timed_out else ("success" if proc.returncode == 0 else "failed")
            batch_results.append({
                "batch_index": batch_idx,
                "organs": batch_organs,
                "status": batch_status,
                "return_code": proc.returncode,
                "runtime_sec": round(time.time() - batch_start, 3),
                "command": batch_command,
                "stdout_tail": (proc.stdout or "")[-1200:],
                "stderr_tail": (proc.stderr or "")[-1200:],
            })
            if proc.returncode != 0 or timed_out:
                for organ in batch_organs:
                    per_organ_status[organ] = {
                        "status": batch_status,
                        "official_mask": official_output_masks[organ],
                        "standardized_mask": expected_masks[organ],
                        "empty_mask": None,
                    }
                continue

            for organ in batch_organs:
                official = Path(official_output_masks[organ])
                standardized = out / f"{organ}.nii.gz"
                if official.exists():
                    if official.resolve() != standardized.resolve():
                        shutil.copy2(official, standardized)
                    standardized_masks[organ] = str(standardized)
                    stats = _mask_stats(standardized)
                    per_organ_status[organ] = {
                        **stats,
                        "status": "empty" if stats.get("empty_mask") else stats.get("status"),
                        "official_mask": str(official),
                        "standardized_mask": str(standardized),
                        "prompt": organ_to_prompt[organ],
                        "retry_recommended": bool(stats.get("empty_mask")),
                        "retry_reason": "empty_mask_check_prompt_orientation_spacing_threshold" if stats.get("empty_mask") else None,
                    }
                else:
                    per_organ_status[organ] = {
                        "status": "failed",
                        "official_mask": str(official),
                        "standardized_mask": str(standardized),
                        "prompt": organ_to_prompt[organ],
                        "empty_mask": None,
                        "reason": "official output mask missing after successful VoxTell command",
                    }

        elapsed = time.time() - start
        masks = sorted(out.glob("*.nii.gz"))
        empty_organs = [organ for organ, item in per_organ_status.items() if item.get("empty_mask") is True]
        failed_organs = [organ for organ, item in per_organ_status.items() if item.get("status") in {"failed", "timed_out", "missing", "unreadable"}]
        successful_nonempty = [organ for organ, item in per_organ_status.items() if item.get("status") == "success"]
        if failed_organs and not standardized_masks:
            status = "failed"
        elif failed_organs or empty_organs or len(standardized_masks) < len(organs):
            status = "partial_success"
        else:
            status = "success"
        result.update({
            "status": status,
            "return_code": 0 if not failed_organs else 1,
            "runtime_sec": round(elapsed, 3),
            "num_masks": len(standardized_masks),
            "num_success_nonempty": len(successful_nonempty),
            "num_empty_masks": len(empty_organs),
            "num_failed_organs": len(failed_organs),
            "empty_organs": empty_organs,
            "failed_organs": failed_organs,
            "retry_queue": [
                {
                    "organ": organ,
                    "prompt": organ_to_prompt[organ],
                    "reason": per_organ_status[organ].get("retry_reason") or per_organ_status[organ].get("reason") or "failed_or_empty_voxtell_output",
                }
                for organ in empty_organs + failed_organs
            ],
            "standardized_masks": standardized_masks,
            "per_organ_status": per_organ_status,
            "batch_results": batch_results,
            "sample_masks": [str(p) for p in masks[:20]],
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        })
        write_json(out / "voxtell_student_result.json", result)
        return result

    def build_training_manifest(
        self,
        cases_root: str | Path,
        output_manifest: str | Path,
        image_lookup: dict[str, str] | None = None,
        case_list: str | Path | None = None,
        require_images: bool = False,
        student_prediction_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build a prompt/mask manifest from merged teacher outputs.

        This is a dataset description for the future VoxTell-style M-step
        trainer. It does not train yet.
        """
        cases = Path(cases_root).resolve()
        out = Path(output_manifest).resolve()
        combined_image_lookup = dict(image_lookup or {})
        combined_image_lookup.update(load_case_image_lookup(case_list))
        doc = self._target_doc()
        target_organs = set(doc.get("target_organs", []))
        organ_to_prompt = doc.get("organ_to_prompt", {}) or {}
        rows: list[dict[str, Any]] = []
        skipped_missing_image: list[dict[str, Any]] = []
        skipped_ineligible_positive: list[dict[str, Any]] = []
        max_negative_ratio = float(os.getenv("MEDAI_NEGATIVE_PROMPT_RATIO", "0.25"))
        expand_setting = os.getenv("MEDAI_EXPAND_PROMPT_VARIANTS")
        if expand_setting is None:
            prompt_variant_mode = os.getenv("MEDAI_PROMPT_VARIANT_MODE", "category_balanced").strip().lower()
        elif expand_setting.strip().lower() in {"1", "true", "yes"}:
            prompt_variant_mode = "all"
        else:
            prompt_variant_mode = "canonical"
        negative_source_counts: dict[str, int] = {}
        zero_mask_targets: list[dict[str, Any]] = []
        negative_quota_policy = _negative_quota_policy()
        negative_source_shortfalls: dict[str, int] = {}
        ignored_student_prediction_root = Path(student_prediction_root).resolve() if student_prediction_root else None

        if _find_mask_dir(cases):
            case_dirs = [cases]
        else:
            case_dirs = sorted(p for p in cases.iterdir() if p.is_dir()) if cases.exists() else []

        for case_dir in case_dirs:
            case_id = case_dir.name
            seg_dir = _find_mask_dir(case_dir)
            image = combined_image_lookup.get(case_id) or _find_case_image(case_dir)
            selection_index = _load_selection_index(cases, case_id)
            label_mapping_index = _load_case_label_mapping(case_dir)
            if not image and selection_index:
                image = next((str(m.get("ct_path")) for m in selection_index.values() if m.get("ct_path")), None)
            if require_images and not image:
                skipped_missing_image.append({"case_id": case_id, "reason": "No CT image path found for case"})
                continue
            positive_organs: set[str] = set()
            positive_count = 0
            for mask in sorted(seg_dir.glob("*.nii.gz")) if seg_dir and seg_dir.exists() else []:
                organ = mask.name[:-7]
                if organ not in target_organs:
                    continue
                meta = selection_index.get(organ, {})
                if organ in label_mapping_index:
                    meta = {**label_mapping_index[organ], **meta}
                canonical_prompt = str(organ_to_prompt.get(organ, organ.replace("_", " ")))
                prompt_variants = _prompt_variants(organ, canonical_prompt, doc)
                item = _manifest_base_item(
                    case_id=case_id, image=image, organ=organ, prompt=canonical_prompt,
                    prompt_variants=prompt_variants, doc=doc, meta=meta,
                )
                item.update({
                    "mask": str(meta.get("probability_mask_path") or mask) if meta.get("target_type") == "soft" else str(mask),
                    "mask_path": str(meta.get("probability_mask_path") or mask) if meta.get("target_type") == "soft" else str(mask),
                    "supervision_type": "positive",
                    "distillation_role": "positive",
                    "negative_reason": None,
                })
                scoring_schema_version = str(meta.get("scoring_schema_version") or "legacy")
                grade = str(item.get("grade") or "D").upper()
                target_type = str(item.get("target_type") or "hard").lower()
                training_weight = float(item.get("training_weight") or 0.0)
                schema_supported = scoring_schema_version in ACCEPTED_SCORING_SCHEMA_VERSIONS
                soft_probability_missing = (
                    target_type == "soft"
                    and not (meta.get("probability_mask_path") and Path(str(meta.get("probability_mask_path"))).exists())
                )
                c_without_soft_target = grade == "C" and target_type != "soft"
                if (
                    not schema_supported
                    or grade == "D"
                    or training_weight <= 0.0
                    or item.get("distillation_eligible") is False
                    or c_without_soft_target
                    or soft_probability_missing
                ):
                    reason = "unsupported_scoring_schema_requires_rescoring"
                    if schema_supported:
                        if c_without_soft_target:
                            reason = "grade_C_requires_soft_probability_target"
                        elif soft_probability_missing:
                            reason = "soft_target_probability_mask_missing"
                        else:
                            reason = item.get("distillation_exclusion_reason") or "grade_D_or_zero_weight"
                    skipped_ineligible_positive.append({
                        "case_id": case_id,
                        "organ": organ,
                        "grade": item.get("grade"),
                        "training_weight": item.get("training_weight"),
                        "target_type": item.get("target_type"),
                        "scoring_schema_version": scoring_schema_version,
                        "reason": reason,
                        "exclusion_category": reason.split(":", 1)[0],
                        "identity_status": item.get("identity_status"),
                        "selected_candidate_qc_status": item.get("selected_candidate_qc_status"),
                        "selected_candidate_qc_flags": item.get("selected_candidate_qc_flags", []),
                        "shapekit_status": item.get("shapekit_status"),
                    })
                    continue
                rows.extend(_expand_prompt_manifest_item(item, prompt_variant_mode, doc))
                positive_organs.add(organ)
                if float(item.get("training_weight") or 0.0) > 0.0:
                    positive_count += 1

            max_negative = int(max(0, round(positive_count * max_negative_ratio)))
            zero_mask = _zero_mask_path_for_case(case_dir, image)
            if zero_mask and max_negative > 0:
                zero_mask_targets.append({"case_id": case_id, "zero_mask": zero_mask, "image": image})
                negative_pools = _negative_prompt_pools(
                    case_dir=case_dir,
                    target_organs=target_organs,
                    positive_organs=positive_organs,
                    selection_index=selection_index,
                    organ_to_prompt=organ_to_prompt,
                )
                negative_candidates, shortfalls = _select_negative_prompt_candidates(negative_pools, max_negative, negative_quota_policy)
                for source, count in shortfalls.items():
                    negative_source_shortfalls[source] = negative_source_shortfalls.get(source, 0) + count
                for neg in negative_candidates:
                    organ = neg["organ"]
                    canonical_prompt = neg["prompt"]
                    prompt_variants = [neg["prompt"]]
                    item = _manifest_base_item(
                        case_id=case_id, image=image, organ=organ, prompt=canonical_prompt,
                        prompt_variants=prompt_variants, doc=doc, meta={},
                    )
                    item.update({
                        "mask": zero_mask,
                        "mask_path": zero_mask,
                        "grade": "D",
                        "training_weight": 0.1,
                        "distillation_eligible": True,
                        "distillation_exclusion_reason": None,
                        "student_training_priority": "negative",
                        "supervision_type": "negative",
                        "distillation_role": "negative",
                        "negative_reason": neg["negative_reason"],
                        "negative_source": neg["negative_source"],
                        "negative_evidence": neg.get("negative_evidence"),
                        "negative_prompt_category": neg.get("negative_prompt_category"),
                        "zero_mask_role": "negative_target_mask",
                        "source_quality": "allowed_negative_prompt",
                    })
                    negative_source_counts[neg["negative_source"]] = negative_source_counts.get(neg["negative_source"], 0) + 1
                    rows.extend(_expand_prompt_manifest_item(item, prompt_variant_mode, doc))

        manifest = {
            "stage": "voxtell_3d_prompt_training_manifest",
            "status": "success",
            "student_backend": "voxtell_style_3d_prompt",
            "target_config": str(self.target_config),
            "formal_373_target_validation": validate_formal_373_target_space(
                self.target_config,
                requested_organs=list(target_organs),
                require_full_target=True,
            ),
            "cases_root": str(cases),
            "case_list": str(Path(case_list).resolve()) if case_list else None,
            "student_prediction_root": str(ignored_student_prediction_root) if ignored_student_prediction_root else None,
            "num_items": len(rows),
            "num_cases": len({r["case_id"] for r in rows}),
            "num_items_missing_image": sum(1 for r in rows if not r.get("image")),
            "grade_counts": {grade: sum(1 for r in rows if r.get("grade") == grade) for grade in ["A", "B", "C", "D"]},
            "num_positive_items": sum(1 for r in rows if r.get("supervision_type") == "positive"),
            "num_negative_items": sum(1 for r in rows if r.get("supervision_type") == "negative"),
            "negative_prompt_ratio": max_negative_ratio,
            "negative_quota_policy": negative_quota_policy,
            "negative_source_counts": negative_source_counts,
            "negative_source_shortfalls": negative_source_shortfalls,
            "negative_prompt_policy": {
                "rule": "Do not sample arbitrary absent target organs as negatives.",
                "allowed_sources": [
                    "nonmedical_absent_object",
                    "out_of_scan_anatomy_with_coverage_evidence",
                    "explicit_confirmed_absent_anatomy",
                ],
                "zero_mask_note": "A zero mask is only a negative target when attached to an allowed negative prompt. Empty organ outputs by themselves are not treated as negative supervision.",
                "student_prediction_root_note": "Previous student empty/failed outputs are recorded for traceability only and are not used as negative evidence.",
            },
            "prompt_variant_mode": prompt_variant_mode,
            "prompt_variant_expansion_enabled": prompt_variant_mode == "all",
            "prompt_variant_sampling_policy": (
                "canonical plus one deterministic hash-selected prompt per category"
                if prompt_variant_mode == "category_balanced"
                else prompt_variant_mode
            ),
            "num_prompt_expanded_items": len(rows),
            "num_canonical_prompt_items": sum(1 for r in rows if not r.get("is_prompt_variant")),
            "num_prompt_variant_items": sum(1 for r in rows if r.get("is_prompt_variant")),
            "num_strong_training_items": sum(1 for r in rows if r.get("supervision_type") == "positive" and float(r.get("training_weight") or 0.0) >= 0.5),
            "num_distillation_eligible_items": sum(1 for r in rows if r.get("distillation_eligible") is not False and float(r.get("training_weight") or 0.0) > 0.0),
            "num_zero_weight_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) == 0.0),
            "zero_mask_targets": zero_mask_targets,
            "skipped_missing_image": skipped_missing_image,
            "skipped_ineligible_positive": skipped_ineligible_positive,
            "num_skipped_ineligible_positive": len(skipped_ineligible_positive),
            "items": rows,
            "training_note": (
                "VoxTell-aligned pseudo-label distillation student manifest. "
                "This is not a reproduction of official large-scale VoxTell CT/MRI/PET training."
            ),
        }
        write_json(out, manifest)
        return manifest



def _prompt_variant_source(prompt: str, idx: int, canonical_prompt: str, doc: dict[str, Any], organ: str) -> str:
    if idx == 0 or prompt == canonical_prompt:
        return "canonical"
    bank_entry = (doc.get("organ_prompt_bank", {}) or {}).get(organ)
    if isinstance(bank_entry, dict):
        category = prompt_category_for(bank_entry, prompt)
        if category != "unknown":
            return category
    configured = doc.get("prompt_variants", {}) or {}
    configured_prompts = {str(p).strip() for p in (configured.get(organ) or []) if str(p).strip()}
    return "configured_variant" if prompt in configured_prompts else "template_variant"


def _expand_prompt_manifest_item(item: dict[str, Any], mode: str | bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [str(p).strip() for p in item.get("prompt_variants", []) if str(p).strip()]
    canonical = str(item.get("prompt") or item.get("canonical_prompt") or "").strip()
    if canonical and canonical not in variants:
        variants.insert(0, canonical)
    if isinstance(mode, bool):
        mode = "all" if mode else "canonical"
    if mode == "all":
        prompts = variants
    elif mode == "category_balanced" and item.get("supervision_type", "positive") == "positive":
        bank_entry = (doc.get("organ_prompt_bank", {}) or {}).get(str(item.get("organ") or ""))
        if isinstance(bank_entry, dict):
            sampling_seed = os.getenv("MEDAI_PROMPT_SEED", "0")
            family_seed = f"{sampling_seed}:{item.get('case_id')}:{item.get('organ')}:{item.get('round', '')}"
            prompts = select_balanced_prompt_variants(bank_entry, seed=family_seed)
        else:
            configured = (doc.get("prompt_variants", {}) or {}).get(str(item.get("organ") or "")) or []
            prompts = variants if configured else ([canonical] if canonical else variants[:1])
    else:
        prompts = [canonical] if canonical else variants[:1]
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        row = dict(item)
        row["prompt"] = prompt
        row["prompt_text"] = prompt
        row["canonical_prompt"] = canonical or prompt
        row["prompt_variant_index"] = idx
        row["is_prompt_variant"] = idx > 0
        row["prompt_source"] = _prompt_variant_source(prompt, idx, canonical or prompt, doc, str(item.get("organ") or ""))
        row["prompt_type"] = row["prompt_source"]
        bank_entry = (doc.get("organ_prompt_bank", {}) or {}).get(str(item.get("organ") or ""))
        record = prompt_record_for(bank_entry, prompt) if isinstance(bank_entry, dict) else None
        row["prompt_provenance"] = record
        row["prompt_family_id"] = f"{row.get('case_id')}:{row.get('organ')}:{row.get('supervision_type', 'positive')}"
        row.setdefault("negative_source", None)
        rows.append(row)
    return rows


def _prompt_variants(organ: str, canonical_prompt: str, doc: dict[str, Any]) -> list[str]:
    bank_entry = (doc.get("organ_prompt_bank", {}) or {}).get(organ)
    if isinstance(bank_entry, dict):
        variants = flatten_prompt_bank_entry(bank_entry)
        if variants:
            return variants
    configured = doc.get("prompt_variants", {}) or {}
    raw = configured.get(organ) or []
    variants: list[str] = []
    for prompt in [canonical_prompt, *raw, f"segment the {canonical_prompt}", f"3D mask of {canonical_prompt}"]:
        prompt = str(prompt or "").strip()
        if prompt and prompt not in variants:
            variants.append(prompt)
    return variants


def _negative_quota_policy() -> dict[str, float]:
    default = {
        "nonmedical_absent_object": 0.60,
        "out_of_scan_anatomy_with_coverage_evidence": 0.30,
        "explicit_confirmed_absent_anatomy": 0.10,
    }
    raw = os.getenv("MEDAI_NEGATIVE_SOURCE_QUOTAS", "").strip()
    if not raw:
        return default
    parsed: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip() or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        if key not in default:
            continue
        try:
            parsed[key] = max(0.0, float(val))
        except Exception:
            continue
    total = sum(parsed.values())
    if total <= 0:
        return default
    return {key: parsed.get(key, 0.0) / total for key in default}


def _quota_counts(max_negative: int, policy: dict[str, float]) -> dict[str, int]:
    sources = list(policy.keys())
    raw = {src: max_negative * float(policy.get(src, 0.0)) for src in sources}
    counts = {src: int(raw[src]) for src in sources}
    remaining = max_negative - sum(counts.values())
    for src in sorted(sources, key=lambda k: raw[k] - counts[k], reverse=True):
        if remaining <= 0:
            break
        counts[src] += 1
        remaining -= 1
    return counts


def _case_level_metadata(case_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("selection_metadata.json", "scan_coverage.json", "case_metadata.json"):
        path = case_dir / name
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict):
            out.update(doc)
    return out


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _metadata_terms(doc: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for key in keys:
        for value in _as_text_list(doc.get(key)):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized:
                terms.add(normalized)
            expanded = normalized
            for sep in (",", ";", "|", "/", "\\"):
                expanded = expanded.replace(sep, "_")
            for token in expanded.split("_"):
                if token:
                    terms.add(token)
    return terms


def _confirmed_absent_organs(doc: dict[str, Any]) -> list[str]:
    keys = (
        "confirmed_absent_organs",
        "out_of_scan_organs",
        "negative_organs",
        "absent_organs",
    )
    organs: list[str] = []
    for key in keys:
        raw_organs: list[str] = []
        for raw in _as_text_list(doc.get(key)):
            parts = [raw]
            for sep in (",", ";", "|"):
                parts = [piece for part in parts for piece in part.split(sep)]
            raw_organs.extend(parts)
        for organ in raw_organs:
            organ = organ.strip().lower().replace(" ", "_")
            if organ and organ not in organs:
                organs.append(organ)
    return organs


def _coverage_excludes_head(doc: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    coverage_terms = _metadata_terms(doc, ("scan_coverage", "coverage_regions", "body_region", "ct_region"))
    if not coverage_terms:
        return False, {"type": "missing_scan_coverage"}
    has_abdomen_pelvis = bool(coverage_terms & ABDOMEN_PELVIS_COVERAGE_TERMS)
    has_head = bool(coverage_terms & HEAD_COVERAGE_TERMS)
    excludes = has_abdomen_pelvis and not has_head
    return excludes, {
        "type": "scan_coverage_metadata",
        "coverage_terms": sorted(coverage_terms),
        "rule": "abdomen/pelvis coverage without head/brain/cranial/head-neck terms",
    }


def _negative_prompt_pools(
    *,
    case_dir: Path,
    target_organs: set[str],
    positive_organs: set[str],
    selection_index: dict[str, dict[str, Any]],
    organ_to_prompt: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    del target_organs
    del selection_index
    doc = _case_level_metadata(case_dir)
    prompt_lookup = organ_to_prompt or {}
    pools: dict[str, list[dict[str, Any]]] = {
        "nonmedical_absent_object": [
            {
                "organ": organ,
                "prompt": prompt,
                "negative_reason": "nonmedical_object_not_present_in_medical_ct",
                "negative_source": "nonmedical_absent_object",
                "negative_prompt_category": "nonmedical_absent_object",
                "negative_evidence": {
                    "type": "domain_prior",
                    "statement": "The prompt names a nonmedical object outside the CT anatomy label space.",
                },
            }
            for organ, prompt in NONMEDICAL_NEGATIVE_PROMPTS
        ],
        "out_of_scan_anatomy_with_coverage_evidence": [],
        "explicit_confirmed_absent_anatomy": [],
    }

    excludes_head, coverage_evidence = _coverage_excludes_head(doc)
    if excludes_head:
        pools["out_of_scan_anatomy_with_coverage_evidence"] = [
            {
                "organ": organ,
                "prompt": prompt,
                "negative_reason": "out_of_scan_by_scan_coverage",
                "negative_source": "out_of_scan_anatomy_with_coverage_evidence",
                "negative_prompt_category": "out_of_scan_anatomy",
                "negative_evidence": coverage_evidence,
            }
            for organ, prompt in OUT_OF_SCAN_ANATOMY_NEGATIVE_PROMPTS
        ]

    explicit_rows: list[dict[str, Any]] = []
    for organ in _confirmed_absent_organs(doc):
        if organ in positive_organs:
            continue
        display = organ.replace("_", " ")
        prompt = prompt_lookup.get(organ) or f"segment the {display}"
        explicit_rows.append({
            "organ": organ,
            "prompt": prompt,
            "negative_reason": "explicitly_confirmed_absent_for_case",
            "negative_source": "explicit_confirmed_absent_anatomy",
            "negative_prompt_category": "confirmed_absent_anatomy",
            "negative_evidence": {
                "type": "case_metadata",
                "fields": ["confirmed_absent_organs", "out_of_scan_organs", "negative_organs", "absent_organs"],
            },
        })
    pools["explicit_confirmed_absent_anatomy"] = explicit_rows
    return pools


def _select_negative_prompt_candidates(
    pools: dict[str, list[dict[str, Any]]],
    max_negative: int,
    policy: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = _quota_counts(max_negative, policy)
    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    used: set[str] = set()
    for source in policy:
        wanted = counts.get(source, 0)
        available = [item for item in pools.get(source, []) if item["organ"] not in used]
        take = available[:wanted]
        selected.extend(take)
        used.update(item["organ"] for item in take)
        if len(take) < wanted:
            shortfalls[source] = wanted - len(take)
    if len(selected) < max_negative:
        for source in policy:
            for item in pools.get(source, []):
                if len(selected) >= max_negative:
                    break
                if item["organ"] in used:
                    continue
                selected.append(item)
                used.add(item["organ"])
            if len(selected) >= max_negative:
                break
    return selected, shortfalls


def _zero_mask_path_for_case(case_dir: Path, image: str | None) -> str | None:
    if not image:
        return None
    out = case_dir / "negative_targets" / "zero_mask.nii.gz"
    if out.exists():
        return str(out)
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(image))
        zero = np.zeros(img.shape[:3], dtype=np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(zero, img.affine, img.header), str(out))
        return str(out)
    except Exception:
        return None


def _manifest_base_item(
    *,
    case_id: str,
    image: str | None,
    organ: str,
    prompt: str,
    prompt_variants: list[str],
    doc: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "dataset_type": "auto_fine_label_dataset",
        "image": image or meta.get("ct_path"),
        "ct_path": image or meta.get("ct_path"),
        "organ": organ,
        "canonical_organ_name": organ,
        "prompt": prompt,
        "prompt_text": prompt,
        "prompt_variants": prompt_variants,
        "prompt_type": "canonical",
        "teacher_target_id": meta.get("teacher_target_id"),
        "teacher_output_name": meta.get("teacher_output_name") or meta.get("source_output_name") or meta.get("organ"),
        "teacher_output_file": meta.get("teacher_output_file"),
        "student_target_id": doc.get("organ_to_student_id", {}).get(organ),
        "target_mapping_policy": "teacher output/name -> canonical organ name -> student target id; teacher IDs are never reused as student IDs",
        "selected_model": meta.get("selected_model"),
        "source_model": meta.get("source_model", meta.get("selected_model")),
        "candidate_models": meta.get("candidate_models", []),
        "teacher_lineage": meta.get("teacher_lineage") or meta.get("candidate_models", []),
        "candidate_count": meta.get("candidate_count"),
        "comparison_candidate_models": meta.get("comparison_candidate_models", []),
        "comparison_candidate_count": meta.get("comparison_candidate_count"),
        "qc_rejected_candidates": meta.get("qc_rejected_candidates", []),
        "candidate_qc_policy": meta.get("candidate_qc_policy"),
        "selection_method": meta.get("selection_method"),
        "selection_status": meta.get("selection_status"),
        "comparison_input_stage": meta.get("comparison_input_stage"),
        "fallback_reason": meta.get("fallback_reason"),
        "selected_dice": meta.get("selected_dice"),
        "selected_pseudo_consistency_dice": meta.get("selected_pseudo_consistency_dice", meta.get("selected_dice")),
        "metric_family": meta.get("metric_family", "pseudo_consistency"),
        "metric_scope": meta.get("metric_scope", "selected_pseudo_label_for_student_training"),
        "accuracy_warning": meta.get("accuracy_warning", "Pseudo labels are not expert ground truth; do not report true accuracy from this manifest."),
        "selected_candidate_qc_status": meta.get("selected_candidate_qc_status"),
        "selected_candidate_qc_score": meta.get("selected_candidate_qc_score"),
        "selected_candidate_qc_flags": meta.get("selected_candidate_qc_flags", []),
        "selected_reference_quality_bucket": meta.get("selected_reference_quality_bucket"),
        "labelcritic_records": meta.get("labelcritic_records", meta.get("critic_records", [])),
        "labelcritic_decision_path": meta.get("labelcritic_decision_path") or _labelcritic_decision_path(meta.get("labelcritic_records", meta.get("critic_records", []))),
        "label_critic_decision_path": meta.get("label_critic_decision_path") or meta.get("labelcritic_decision_path") or _labelcritic_decision_path(meta.get("labelcritic_records", meta.get("critic_records", []))),
        "shapekit_status": meta.get("shapekit_status"),
        "shapekit_reason": meta.get("shapekit_reason"),
        "dataset_role": meta.get("dataset_role", "pseudo_label"),
        "ground_truth_status": meta.get("ground_truth_status", "machine_generated_candidate"),
        "label_maturity_level": meta.get("label_maturity_level"),
        "auto_fine_label_status": meta.get("auto_fine_label_status", "auto_fine_label_candidate" if meta else "machine_label_candidate"),
        "auto_fine_label_reliability_score": meta.get("auto_fine_label_reliability_score"),
        "estimated_reliability": meta.get("estimated_reliability", meta.get("evidence_confidence")),
        "evidence_confidence": meta.get("evidence_confidence"),
        "evidence_scores": meta.get("evidence_scores", {}),
        "missing_evidence": meta.get("missing_evidence", []),
        "decision_status": meta.get("decision_status"),
        "decision_reasons": meta.get("decision_reasons", []),
        "target_type": meta.get("target_type", "hard"),
        "probability_mask_path": meta.get("probability_mask_path"),
        "voxel_uncertainty_path": meta.get("voxel_uncertainty_path"),
        "independent_family_count": meta.get("independent_family_count", 0),
        "family_membership": meta.get("family_membership", {}),
        "scoring_schema_version": meta.get("scoring_schema_version", "legacy"),
        "grade": meta.get("grade", "D"),
        "training_weight": float(meta.get("training_weight", grade_to_training_weight(meta.get("grade", "D")))),
        "distillation_eligible": meta.get("distillation_eligible", float(meta.get("training_weight", grade_to_training_weight(meta.get("grade", "D"))) or 0.0) > 0.0),
        "distillation_exclusion_reason": meta.get("distillation_exclusion_reason"),
        "student_training_priority": meta.get("student_training_priority", meta.get("grade", "D")),
        "route_confidence": meta.get("route_confidence"),
        "label_confidence": meta.get("label_confidence", meta.get("auto_fine_label_reliability_score")),
        "label_passport_path": meta.get("label_passport_path"),
        "review_flags": meta.get("review_flags", []),
        "quality_flags": meta.get("quality_flags", []),
        "quality_status": meta.get("quality_status") or _quality_status(meta.get("review_flags", []), meta.get("quality_flags", [])),
        "source_metadata_available": bool(meta),
        "distillation_source": meta.get("selected_model") or meta.get("source_model"),
        "source_quality": meta.get("quality_status") or meta.get("grade", "unknown"),
    }

def find_voxtell_executable() -> str | None:
    return shutil.which("voxtell-predict")
