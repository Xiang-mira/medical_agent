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
from .json_utils import read_json, write_json
from .paths import resolve_path
from .target_space import validate_formal_373_target_space


DEFAULT_TARGETS = "configs/student_3d_prompt_target_organs.json"


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
    if case_dir.exists() and any(case_dir.glob("*.nii.gz")):
        return case_dir
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
        organ_to_prompt = doc.get("organ_to_prompt", {}) or {}
        if prompts:
            organs = [p for p in prompts if p in set(doc.get("target_organs", []))]
            unknown = [p for p in prompts if p not in set(doc.get("target_organs", []))]
            if unknown:
                raise ValueError(f"Unknown or non-target organs for VoxTell student: {unknown[:20]}")
        else:
            organs = list(doc.get("target_organs", []))
        return organs, {organ: str(organ_to_prompt.get(organ, organ.replace("_", " "))) for organ in organs}

    def segment(
        self,
        ct_image: str | Path,
        output_dir: str | Path,
        prompts: list[str] | None = None,
        dry_run: bool = False,
        timeout_sec: int = 1800,
        prompt_batch_size: int = 16,
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
        dry_run:
            Build the command and expected output contract without running.
        """
        ct = Path(ct_image).resolve()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        organs, organ_to_prompt = self._select_organs(prompts)
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
            "expected_masks": expected_masks,
            "official_output_masks": official_output_masks,
            "command": command,
            "prompt_batch_size": prompt_batch_size,
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

        if _find_mask_dir(cases):
            case_dirs = [cases]
        else:
            case_dirs = sorted(p for p in cases.iterdir() if p.is_dir()) if cases.exists() else []

        for case_dir in case_dirs:
            case_id = case_dir.name
            seg_dir = _find_mask_dir(case_dir)
            image = combined_image_lookup.get(case_id)
            selection_index = _load_selection_index(cases, case_id)
            if not image and selection_index:
                image = next((str(m.get("ct_path")) for m in selection_index.values() if m.get("ct_path")), None)
            if require_images and not image:
                skipped_missing_image.append({"case_id": case_id, "reason": "No CT image path found for case"})
                continue
            for mask in sorted(seg_dir.glob("*.nii.gz")) if seg_dir and seg_dir.exists() else []:
                organ = mask.name[:-7]
                if organ not in target_organs:
                    continue
                meta = selection_index.get(organ, {})
                rows.append({
                    "case_id": case_id,
                    "dataset_type": "auto_fine_label_dataset",
                    "image": image or meta.get("ct_path"),
                    "ct_path": image or meta.get("ct_path"),
                    "mask": str(mask),
                    "mask_path": str(mask),
                    "organ": organ,
                    "prompt": organ_to_prompt.get(organ, organ.replace("_", " ")),
                    "student_target_id": doc.get("organ_to_student_id", {}).get(organ),
                    "selected_model": meta.get("selected_model"),
                    "source_model": meta.get("source_model", meta.get("selected_model")),
                    "candidate_models": meta.get("candidate_models", []),
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
                    "grade": meta.get("grade", "D"),
                    "training_weight": float(meta.get("training_weight", grade_to_training_weight(meta.get("grade", "D")))),
                    "label_passport_path": meta.get("label_passport_path"),
                    "review_flags": meta.get("review_flags", []),
                    "quality_flags": meta.get("quality_flags", []),
                    "quality_status": meta.get("quality_status") or _quality_status(meta.get("review_flags", []), meta.get("quality_flags", [])),
                    "source_metadata_available": bool(meta),
                })

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
            "num_items": len(rows),
            "num_cases": len({r["case_id"] for r in rows}),
            "num_items_missing_image": sum(1 for r in rows if not r.get("image")),
            "grade_counts": {grade: sum(1 for r in rows if r.get("grade") == grade) for grade in ["A", "B", "C", "D"]},
            "num_strong_training_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) >= 0.5),
            "num_zero_weight_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) == 0.0),
            "skipped_missing_image": skipped_missing_image,
            "items": rows,
            "training_note": (
                "Official VoxTell currently exposes inference; this manifest is "
                "the input contract for the project-specific 3D prompt M-step trainer."
            ),
        }
        write_json(out, manifest)
        return manifest


def find_voxtell_executable() -> str | None:
    return shutil.which("voxtell-predict")
