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
    filter_semantically_risky_prompts,
    flatten_prompt_bank_entry,
    prompt_category_for,
    prompt_record_for,
    select_balanced_prompt_variants,
    select_prompt_for_organ,
)
from .paths import resolve_path
from .subprocess_utils import subprocess_text
from .target_space import validate_formal_373_target_space


DEFAULT_TARGETS = "configs/student_3d_prompt_target_organs.json"
DEFAULT_OFFICIAL_PROMPT_MAP = "configs/voxtell_official_prompt_map.json"
DEFAULT_OFFICIAL_EMBEDDING_BANK = "checkpoints/VoxTell/embeddings/voxtell_v1.1/text_embeddings.npz"
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
VOXTELL_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "VoxTell"
VOXTELL_BACKENDS = {"official_python_api", "official_cli"}


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


def _as_positive_target_type(value: Any) -> str:
    raw = str(value or "hard").strip().lower()
    if raw in {"positive_hard", "hard"}:
        return "positive_hard"
    if raw in {"positive_soft", "soft"}:
        return "positive_soft"
    return raw


def _is_positive_supervision(row: dict[str, Any]) -> bool:
    return str(row.get("supervision_type") or "positive").lower() == "positive"


def _is_negative_absent(row: dict[str, Any]) -> bool:
    return str(row.get("target_type") or "").strip().lower() in {"negative_absent", "absent_negative"}


def _manifest_row_mask_path(row: dict[str, Any]) -> Path | None:
    raw = row.get("mask_path") or row.get("mask") or row.get("final_mask")
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser().resolve()
    except Exception:
        return Path(str(raw))


def _mask_has_foreground(path: Path | None) -> tuple[bool, str | None]:
    if path is None:
        return False, "mask_missing"
    if not path.exists():
        return False, "mask_missing"
    try:
        import nibabel as nib
        import numpy as np

        arr = np.asanyarray(nib.load(str(path)).dataobj)
        return bool((arr > 0).sum() > 0), None
    except Exception as exc:
        return False, f"mask_unreadable:{exc}"


def _positive_replay_candidate_reason(row: dict[str, Any]) -> str | None:
    """Return None for rows that may be considered for historical replay.

    This is intentionally stricter than ordinary manifest inclusion. Historical
    replay must never promote student predictions, GT rows, empty masks, failed
    QC labels, or absent-negative all-zero masks into the positive memory.
    """
    if not isinstance(row, dict):
        return "not_a_manifest_row"
    if not _is_positive_supervision(row):
        return "not_positive_supervision"
    target_type = _as_positive_target_type(row.get("target_type"))
    if target_type not in {"positive_hard", "positive_soft"}:
        return f"not_positive_target_type:{target_type}"
    if _is_negative_absent(row):
        return "negative_absent_not_replay_positive"
    grade = str(row.get("grade") or "D").upper()
    if grade not in {"A", "B", "C"}:
        return f"grade_not_trainable:{grade}"
    if grade == "C" and target_type != "positive_soft":
        return "grade_C_requires_soft_positive_target"
    try:
        if float(row.get("training_weight") or 0.0) <= 0.0:
            return "nonpositive_training_weight"
    except Exception:
        return "invalid_training_weight"
    if row.get("distillation_eligible") is False:
        return "distillation_ineligible"
    model_tokens = " ".join(
        str(row.get(key) or "")
        for key in ("selected_model", "source_model", "distillation_source", "selected_provider")
    ).lower()
    if "student" in model_tokens:
        return "student_prediction_not_replay_memory"
    dataset_role = str(row.get("dataset_role") or "").lower()
    gt_status = str(row.get("ground_truth_status") or "").lower()
    gt_tokens = {"gt", "ground_truth", "expert_ground_truth", "manual_ground_truth", "human_ground_truth"}
    if dataset_role in gt_tokens or gt_status in gt_tokens or "ground_truth" in dataset_role:
        return "ground_truth_not_replay_memory"
    qc_status = str(row.get("selected_candidate_qc_status") or row.get("candidate_qc_status") or "").lower()
    if qc_status and qc_status not in {"pass", "passed", "success", "ok"}:
        return f"candidate_qc_not_pass:{qc_status}"
    quality_status = str(row.get("quality_status") or "").lower()
    if quality_status in {"failed", "fail", "rejected", "qc_failed"}:
        return f"quality_status_not_pass:{quality_status}"
    flags = {str(flag).lower() for flag in (row.get("quality_flags") or [])}
    flags |= {str(flag).lower() for flag in (row.get("review_flags") or [])}
    flags |= {str(flag).lower() for flag in (row.get("selected_candidate_qc_flags") or [])}
    hard_fail_flags = {
        "zero_volume_mask",
        "empty_mask",
        "geometry_mismatch",
        "shape_mismatch_ct",
        "affine_mismatch_ct",
        "orientation_mismatch_ct",
        "candidate_qc_fail",
        "postprocess_failed",
    }
    if flags & hard_fail_flags:
        return f"hard_quality_flag:{sorted(flags & hard_fail_flags)[0]}"
    return None


def _is_replay_safe_positive(row: dict[str, Any]) -> tuple[bool, str | None]:
    reason = _positive_replay_candidate_reason(row)
    if reason:
        return False, reason
    nonempty, mask_reason = _mask_has_foreground(_manifest_row_mask_path(row))
    if not nonempty:
        return False, mask_reason or "empty_mask_not_replay_memory"
    return True, None


def _is_trainable_positive_row(row: dict[str, Any], *, require_nonempty: bool = False) -> bool:
    if _positive_replay_candidate_reason(row) is not None:
        return False
    if require_nonempty:
        nonempty, _ = _mask_has_foreground(_manifest_row_mask_path(row))
        return bool(nonempty)
    return True


def _normalize_replay_positive_item(
    row: dict[str, Any],
    *,
    manifest_path: Path,
    replay_index: int,
) -> dict[str, Any]:
    item = dict(row)
    target_type = _as_positive_target_type(item.get("target_type"))
    item["target_type"] = target_type
    item["legacy_target_type"] = row.get("legacy_target_type") or row.get("target_type")
    item["supervision_type"] = "positive"
    item["distillation_role"] = "positive"
    item["historical_replay"] = True
    item["cumulative_manifest_role"] = "historical_replay_positive"
    item["replay_source"] = "promoted_history"
    item["replay_source_manifest"] = str(manifest_path)
    item["replay_index"] = replay_index
    item["training_gate_decision"] = (
        "include_soft_c"
        if str(item.get("grade") or "").upper() == "C" or target_type == "positive_soft"
        else "include_hard_ab"
    )
    item["training_gate_policy"] = (
        "Promoted historical trainable positive replay restored by the cumulative manifest."
    )
    item["negative_reason"] = None
    item["negative_source"] = None
    return item


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


def _voxtell_vendor_audit() -> dict[str, Any]:
    audit: dict[str, Any] = {
        "official_repo": "https://github.com/MIC-DKFZ/VoxTell",
        "vendor_root": str(VOXTELL_VENDOR_ROOT),
        "vendor_policy": "Do not patch third_party/VoxTell/voxtell/*; keep project adaptation in VoxTellStudent.",
    }
    if not VOXTELL_VENDOR_ROOT.exists():
        audit.update({"status": "missing_vendor_root", "dirty": None})
        return audit
    try:
        commit = subprocess.run(
            ["git", "-C", str(VOXTELL_VENDOR_ROOT), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        audit["commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        status = subprocess.run(
            ["git", "-C", str(VOXTELL_VENDOR_ROOT), "status", "--short"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        dirty_lines = [line for line in (status.stdout or "").splitlines() if line.strip()]
        audit.update({
            "status": "ok" if status.returncode == 0 else "git_status_failed",
            "dirty": bool(dirty_lines),
            "dirty_files": dirty_lines,
        })
    except Exception as exc:
        audit.update({"status": "audit_failed", "dirty": None, "reason": str(exc)})
    return audit


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
        backend: str | None = None,
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
        self.backend = (backend or os.getenv("MEDAI_VOXTELL_BACKEND") or "official_python_api").strip().lower()
        if self.backend not in VOXTELL_BACKENDS:
            raise ValueError(f"Unsupported VoxTell backend: {self.backend}. Expected one of {sorted(VOXTELL_BACKENDS)}")

    def _target_doc(self) -> dict[str, Any]:
        return load_prompt_targets(self.target_config)

    def _select_organs(self, prompts: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
        doc = self._target_doc()
        prompt_sampling = os.getenv("MEDAI_PROMPT_SAMPLING", "canonical").strip().lower() or "canonical"
        prompt_map_path = resolve_path(os.getenv("MEDAI_VOXTELL_OFFICIAL_PROMPT_MAP", DEFAULT_OFFICIAL_PROMPT_MAP))
        prompt_map_doc = read_json(prompt_map_path, default={}) or {}
        official_prompt_map = {
            str(row.get("project_class")): row
            for row in prompt_map_doc.get("mappings", [])
            if isinstance(row, dict) and row.get("project_class")
        }
        if prompts:
            organs = [p for p in prompts if p in set(doc.get("target_organs", []))]
            unknown = [p for p in prompts if p not in set(doc.get("target_organs", []))]
            if unknown:
                raise ValueError(f"Unknown or non-target organs for VoxTell student: {unknown[:20]}")
        else:
            organs = list(doc.get("target_organs", []))
        selected = {}
        for organ in organs:
            official = official_prompt_map.get(organ, {})
            if prompt_sampling == "canonical" and official.get("canonical_prompt"):
                selected[organ] = str(official["canonical_prompt"])
            else:
                selected[organ] = select_prompt_for_organ(
                    doc, organ, mode=prompt_sampling, seed=str(self.target_config)
                )
        return organs, selected

    def segment(
        self,
        ct_image: str | Path,
        output_dir: str | Path,
        prompts: list[str] | None = None,
        dry_run: bool = False,
        timeout_sec: int = 1800,
        prompt_batch_size: int = 16,
        prompt_overrides: dict[str, str] | None = None,
        save_probability_outputs: bool | None = None,
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
        probability_dir = out / "probability_masks"
        if save_probability_outputs is None:
            save_probability_outputs = os.getenv("MEDAI_SAVE_STUDENT_PROBABILITY", "1").strip().lower() in {"1", "true", "yes", "on"}
        if save_probability_outputs:
            probability_dir.mkdir(parents=True, exist_ok=True)
        organs, organ_to_prompt = self._select_organs(prompts)
        if prompt_overrides:
            for organ, prompt_text in prompt_overrides.items():
                if organ in organ_to_prompt and str(prompt_text).strip():
                    organ_to_prompt[organ] = str(prompt_text).strip()
        prompt_texts = [organ_to_prompt[o] for o in organs]

        def command_for(batch_organs: list[str]) -> list[str]:
            batch_prompt_texts = [organ_to_prompt[o] for o in batch_organs]
            if self.backend == "official_cli":
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
                ]
            return [
                "official_python_api",
                "voxtell.inference.predictor.VoxTellPredictor",
                "predict_single_image",
                "--input",
                str(ct),
                "--output",
                str(out),
                "--model",
                str(self.model_dir),
                "--prompts",
                *batch_prompt_texts,
                "--device",
                f"{self.device}:{self.gpu}" if self.device == "cuda" else self.device,
                "--text-encoding-model",
                self.text_encoding_model,
            ]

        command = command_for(organs)
        expected_masks = {organ: str(out / f"{organ}.nii.gz") for organ in organs}
        official_output_masks = {
            organ: str(out / f"{_input_stem_for_voxtell(ct)}_{_safe_prompt_name(organ_to_prompt[organ])}{_voxtell_suffix(ct)}")
            for organ in organs
        }
        official_path_to_organs: dict[str, list[str]] = {}
        for organ, path in official_output_masks.items():
            official_path_to_organs.setdefault(path, []).append(organ)
        official_output_name_collisions = {
            path: aliases
            for path, aliases in official_path_to_organs.items()
            if len(aliases) > 1
        }
        result: dict[str, Any] = {
            "stage": "voxtell_3d_prompt_student_inference",
            "status": "dry_run" if dry_run else "pending",
            "ct_image": str(ct),
            "output_dir": str(out),
            "probability_output_dir": str(probability_dir),
            "save_probability_outputs_requested": bool(save_probability_outputs),
            "probability_outputs_available": False,
            "probability_output_reason": "VoxTell adapter currently receives binary segmentations from the vendor API/CLI; probability/logit tensors are not exposed.",
            "model_dir": str(self.model_dir),
            "text_encoding_model": self.text_encoding_model,
            "target_config": str(self.target_config),
            "backend": self.backend,
            "voxtell_source_mode": "official_vendor_via_project_adapter",
            "vendor_audit": _voxtell_vendor_audit(),
            "num_prompts": len(organs),
            "organs": organs,
            "organ_to_prompt": organ_to_prompt,
            "prompt_sampling": os.getenv("MEDAI_PROMPT_SAMPLING", "canonical"),
            "expected_masks": expected_masks,
            "official_output_masks": official_output_masks,
            "official_output_name_collisions": official_output_name_collisions,
            "official_output_name_collision_count": len(official_output_name_collisions),
            "command": command,
            "official_api_or_cli_command": command,
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
                "probability_output": "optional probability_masks/<organ>.nii.gz when backend exposes logits/probabilities; currently recorded as unavailable for binary-only VoxTell outputs",
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

        api_context: dict[str, Any] | None = None
        if self.backend == "official_python_api":
            try:
                import torch
                if VOXTELL_VENDOR_ROOT.exists() and str(VOXTELL_VENDOR_ROOT) not in sys.path:
                    sys.path.insert(0, str(VOXTELL_VENDOR_ROOT))
                from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
                from voxtell.inference.predictor import VoxTellPredictor

                if self.device == "cuda":
                    device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device("cpu")
                reader_writer = NibabelIOWithReorient()
                img, props = reader_writer.read_images([str(ct)])
                embedding_bank_path = resolve_path(
                    os.getenv("MEDAI_VOXTELL_EMBEDDING_BANK", DEFAULT_OFFICIAL_EMBEDDING_BANK)
                )
                predictor = VoxTellPredictor(
                    model_dir=str(self.model_dir),
                    device=device,
                    text_encoding_model=self.text_encoding_model,
                    embedding_bank=str(embedding_bank_path) if embedding_bank_path.exists() else None,
                    use_precomputed_embeddings=embedding_bank_path.exists(),
                )
                api_context = {
                    "device": str(device),
                    "img": img,
                    "props": props,
                    "predictor": predictor,
                    "reader_writer": reader_writer,
                    "input_filename": _input_stem_for_voxtell(ct),
                    "suffix": _voxtell_suffix(ct),
                    "embedding_bank": str(embedding_bank_path) if embedding_bank_path.exists() else None,
                }
            except Exception as exc:
                result.update({
                    "status": "failed",
                    "reason": f"Official VoxTell Python API initialization failed: {exc}",
                    "return_code": 1,
                })
                write_json(out / "voxtell_student_result.json", result)
                return result

        for batch_idx, batch_organs in enumerate(_chunked(organs, prompt_batch_size), start=1):
            batch_command = command_for(batch_organs)
            batch_prompt_texts = [organ_to_prompt[o] for o in batch_organs]
            batch_start = time.time()
            proc_stdout = ""
            proc_stderr = ""
            return_code = 0
            timed_out = False
            if self.backend == "official_cli":
                try:
                    proc = subprocess.run(
                        batch_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=timeout_sec,
                    )
                    return_code = proc.returncode
                    proc_stdout = proc.stdout or ""
                    proc_stderr = proc.stderr or ""
                except subprocess.TimeoutExpired as exc:
                    return_code = 124
                    proc_stdout = subprocess_text(exc.stdout)
                    proc_stderr = subprocess_text(exc.stderr) + f"\n[VoxTellStudent] Timeout after {timeout_sec}s."
                    timed_out = True
            else:
                try:
                    assert api_context is not None
                    segmentations = api_context["predictor"].predict_single_image(api_context["img"], batch_prompt_texts)
                    for i, prompt_text in enumerate(batch_prompt_texts):
                        official_path = (
                            out
                            / (
                                f"{api_context['input_filename']}_"
                                f"{_safe_prompt_name(prompt_text)}"
                                f"{api_context['suffix']}"
                            )
                        )
                        api_context["reader_writer"].write_seg(
                            segmentations[i],
                            str(official_path),
                            api_context["props"],
                        )
                except Exception as exc:
                    return_code = 1
                    proc_stderr = f"Official VoxTell Python API prediction failed: {exc}"

            stdout_tail = ((stdout_tail + "\n" + proc_stdout)[-4000:])
            stderr_tail = ((stderr_tail + "\n" + proc_stderr)[-4000:])
            batch_status = "timed_out" if timed_out else ("success" if return_code == 0 else "failed")
            batch_results.append({
                "batch_index": batch_idx,
                "organs": batch_organs,
                "status": batch_status,
                "return_code": return_code,
                "runtime_sec": round(time.time() - batch_start, 3),
                "backend": self.backend,
                "command": batch_command,
                "stdout_tail": proc_stdout[-1200:],
                "stderr_tail": proc_stderr[-1200:],
            })
            if return_code != 0 or timed_out:
                for organ in batch_organs:
                    per_organ_status[organ] = {
                        "status": batch_status,
                        "official_mask": official_output_masks[organ],
                        "standardized_mask": expected_masks[organ],
                        "prompt": organ_to_prompt[organ],
                        "backend": self.backend,
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
                        "backend": self.backend,
                        "retry_recommended": bool(stats.get("empty_mask")),
                        "retry_reason": "empty_mask_check_prompt_orientation_spacing_threshold" if stats.get("empty_mask") else None,
                    }
                else:
                    per_organ_status[organ] = {
                        "status": "failed",
                        "official_mask": str(official),
                        "standardized_mask": str(standardized),
                        "prompt": organ_to_prompt[organ],
                        "backend": self.backend,
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
        historical_replay_manifests: list[str | Path] | None = None,
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
        prompt_map_path = resolve_path(os.getenv("MEDAI_VOXTELL_OFFICIAL_PROMPT_MAP", DEFAULT_OFFICIAL_PROMPT_MAP))
        prompt_map_doc = read_json(prompt_map_path, default={}) or {}
        official_prompt_map = {
            str(row.get("project_class")): row
            for row in prompt_map_doc.get("mappings", [])
            if isinstance(row, dict) and row.get("project_class")
        }
        rows: list[dict[str, Any]] = []
        skipped_missing_image: list[dict[str, Any]] = []
        skipped_ineligible_positive: list[dict[str, Any]] = []
        max_negative_ratio = float(os.getenv("MEDAI_NEGATIVE_PROMPT_RATIO", "0.25"))
        expand_setting = os.getenv("MEDAI_EXPAND_PROMPT_VARIANTS")
        if expand_setting is None:
            prompt_variant_mode = os.getenv("MEDAI_PROMPT_VARIANT_MODE", "canonical").strip().lower()
        elif expand_setting.strip().lower() in {"1", "true", "yes"}:
            prompt_variant_mode = "all"
        else:
            prompt_variant_mode = "canonical"
        negative_source_counts: dict[str, int] = {}
        zero_mask_targets: list[dict[str, Any]] = []
        negative_quota_policy = _negative_quota_policy()
        negative_source_shortfalls: dict[str, int] = {}
        ignored_student_prediction_root = Path(student_prediction_root).resolve() if student_prediction_root else None
        replay_manifest_paths = [
            Path(path).expanduser().resolve()
            for path in (historical_replay_manifests or [])
            if path
        ]

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
                official_mapping = official_prompt_map.get(organ, {})
                canonical_prompt = str(
                    official_mapping.get("canonical_prompt")
                    or organ_to_prompt.get(organ, organ.replace("_", " "))
                )
                # The formal VoxTell path uses exactly the official bank string
                # when mapped, otherwise the canonical anatomical term.  CT
                # descriptions and project-authored rewrites belong only to
                # LabelCritic and must never leak into VoxTell encoding.
                prompt_variants = [canonical_prompt]
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
                    "official_prompt_mapping": official_mapping,
                    "official_prompt_map_version": prompt_map_doc.get("schema_version"),
                })
                scoring_schema_version = str(meta.get("scoring_schema_version") or "legacy")
                grade = str(item.get("grade") or "D").upper()
                target_type = str(item.get("target_type") or "hard").lower()
                target_type = {
                    "positive_hard": "hard",
                    "positive_soft": "soft",
                    "negative_absent": "absent_negative",
                }.get(target_type, target_type)
                training_weight = float(item.get("training_weight") or 0.0)
                schema_supported = scoring_schema_version in ACCEPTED_SCORING_SCHEMA_VERSIONS
                soft_probability_missing = (
                    target_type == "soft"
                    and not (meta.get("probability_mask_path") and Path(str(meta.get("probability_mask_path"))).exists())
                )
                c_without_soft_target = grade == "C" and target_type != "soft"
                if not schema_supported:
                    training_gate_decision = "exclude_unsupported_schema"
                    training_gate_policy = "Unsupported scoring schemas require rescoring before student training."
                elif grade in {"A", "B"} and target_type == "hard" and training_weight > 0.0:
                    training_gate_decision = "include_hard_ab"
                    training_gate_policy = "A/B hard pseudo-labels are eligible for direct student training."
                elif grade in {"A", "B"}:
                    training_gate_decision = "exclude_ab_non_hard_or_zero_weight"
                    training_gate_policy = "A/B labels must be hard targets with positive training weight for direct student training."
                elif grade == "C" and target_type == "soft" and training_weight > 0.0 and not soft_probability_missing:
                    training_gate_decision = "include_soft_c"
                    training_gate_policy = "C is eligible only as a soft target with an explicit probability mask."
                elif grade == "C" and target_type == "soft":
                    training_gate_decision = "exclude_c_soft_missing_probability"
                    training_gate_policy = "C soft labels require an explicit probability mask before student training."
                elif grade == "C":
                    training_gate_decision = "exclude_or_review_c"
                    training_gate_policy = "C hard/provisional labels are audit/review only unless a soft target is present."
                else:
                    training_gate_decision = "exclude_d_or_zero_weight"
                    training_gate_policy = "D and zero-weight labels are excluded from student training."
                item["training_gate_decision"] = training_gate_decision
                item["training_gate_policy"] = training_gate_policy
                if training_gate_decision not in {"include_hard_ab", "include_soft_c"} or item.get("distillation_eligible") is False:
                    reason = "legacy_requires_autolabel_core_v2_rescoring" if scoring_schema_version in {"", "legacy", "none", "null"} else "unsupported_scoring_schema_requires_rescoring"
                    if schema_supported:
                        if c_without_soft_target:
                            reason = "grade_C_requires_soft_probability_target"
                        elif soft_probability_missing:
                            reason = "soft_target_probability_mask_missing"
                        elif training_gate_decision == "exclude_ab_non_hard_or_zero_weight":
                            reason = "grade_AB_requires_hard_positive_target"
                        else:
                            reason = item.get("distillation_exclusion_reason") or training_gate_decision or "grade_D_or_zero_weight"
                    skipped_ineligible_positive.append({
                        "case_id": case_id,
                        "organ": organ,
                        "grade": item.get("grade"),
                        "training_weight": item.get("training_weight"),
                        "target_type": item.get("target_type"),
                        "scoring_schema_version": scoring_schema_version,
                        "training_gate_decision": training_gate_decision,
                        "training_gate_policy": training_gate_policy,
                        "probability_mask_path": item.get("probability_mask_path"),
                        "distillation_eligible": item.get("distillation_eligible"),
                        "reason": reason,
                        "exclusion_category": reason.split(":", 1)[0],
                        "identity_status": item.get("identity_status"),
                        "selected_candidate_qc_status": item.get("selected_candidate_qc_status"),
                        "selected_candidate_qc_flags": item.get("selected_candidate_qc_flags", []),
                        "shapekit_status": item.get("shapekit_status"),
                    })
                    continue
                item["legacy_target_type"] = item.get("target_type")
                item["target_type"] = (
                    "positive_soft" if target_type == "soft" else "positive_hard"
                )
                rows.extend(_expand_prompt_manifest_item(item, prompt_variant_mode, doc))
                positive_organs.add(organ)
                if float(item.get("training_weight") or 0.0) > 0.0:
                    positive_count += 1

            absent_negative_count = 0
            for organ, meta in sorted(selection_index.items()):
                if organ not in target_organs or str(meta.get("target_type")) not in {"absent_negative", "negative_absent"}:
                    continue
                zero_mask = meta.get("mask_path") or meta.get("mask") or meta.get("final_mask") or _zero_mask_path_for_case(case_dir, image)
                if not zero_mask or not Path(str(zero_mask)).exists():
                    skipped_ineligible_positive.append({
                        "case_id": case_id,
                        "organ": organ,
                        "grade": meta.get("grade", "A"),
                        "training_weight": meta.get("training_weight"),
                        "target_type": "negative_absent",
                        "reason": "absent_negative_zero_mask_missing",
                        "exclusion_category": "absent_negative_zero_mask_missing",
                    })
                    continue
                official_mapping = official_prompt_map.get(organ, {})
                canonical_prompt = str(
                    official_mapping.get("canonical_prompt")
                    or organ_to_prompt.get(organ, organ.replace("_", " "))
                )
                prompt_variants = [canonical_prompt]
                item = _manifest_base_item(
                    case_id=case_id, image=image, organ=organ, prompt=canonical_prompt,
                    prompt_variants=prompt_variants, doc=doc, meta=meta,
                )
                item.update({
                    "mask": str(zero_mask),
                    "mask_path": str(zero_mask),
                    "grade": meta.get("grade", "A"),
                    "grade_scope": "absence_target",
                    "training_weight": float(meta.get("training_weight", os.getenv("MEDAI_NEGATIVE_ABSENT_TRAINING_WEIGHT", "0.1")) or 0.0),
                    "distillation_eligible": bool(meta.get("distillation_eligible", True)),
                    "student_training_priority": "negative_absent",
                    "supervision_type": "negative",
                    "distillation_role": "negative",
                    "target_type": "negative_absent",
                    "negative_reason": meta.get("negative_reason") or "out_of_scan_by_scan_coverage",
                    "negative_source": meta.get("negative_source") or "case_373_expected_absent",
                    "zero_mask_role": meta.get("zero_mask_role") or "negative_absent_target_mask",
                    "source_quality": "valid_absent_negative",
                    "training_gate_decision": "include_absent_negative",
                    "training_gate_policy": "Absent-negative all-zero masks are valid negative supervision targets, separate from A/B/C/D positive quality grades.",
                    "official_prompt_mapping": official_mapping,
                    "official_prompt_map_version": prompt_map_doc.get("schema_version"),
                })
                rows.extend(_expand_prompt_manifest_item(item, prompt_variant_mode, doc))
                absent_negative_count += 1
                negative_source_counts[item["negative_source"]] = negative_source_counts.get(item["negative_source"], 0) + 1
            if absent_negative_count:
                zero_mask_targets.append({"case_id": case_id, "source": "case_373_absent_negative", "count": absent_negative_count, "image": image})

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

        current_trainable_positive_keys = {
            (str(r.get("case_id") or ""), str(r.get("organ") or ""))
            for r in rows
            if _is_trainable_positive_row(r, require_nonempty=False)
        }
        current_trainable_positive_organs = {
            str(r.get("organ") or "")
            for r in rows
            if _is_trainable_positive_row(r, require_nonempty=False)
        }
        replay_positive_rows: list[dict[str, Any]] = []
        replay_excluded_rows: list[dict[str, Any]] = []
        historical_trainable_positive_organs: set[str] = set()
        historical_replay_source_manifests: list[str] = []
        seen_replay_keys = set(current_trainable_positive_keys)
        for manifest_idx, replay_manifest in enumerate(replay_manifest_paths):
            if not replay_manifest.exists():
                replay_excluded_rows.append({
                    "manifest": str(replay_manifest),
                    "reason": "historical_manifest_missing",
                })
                continue
            try:
                replay_doc = json.loads(replay_manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                replay_excluded_rows.append({
                    "manifest": str(replay_manifest),
                    "reason": f"historical_manifest_unreadable:{exc}",
                })
                continue
            historical_replay_source_manifests.append(str(replay_manifest))
            replay_items = replay_doc.get("items") if isinstance(replay_doc, dict) else replay_doc
            if not isinstance(replay_items, list):
                replay_excluded_rows.append({
                    "manifest": str(replay_manifest),
                    "reason": "historical_manifest_items_missing",
                })
                continue
            for item_idx, replay_item in enumerate(replay_items):
                if not isinstance(replay_item, dict):
                    continue
                organ = str(replay_item.get("organ") or "")
                if organ:
                    candidate_reason = _positive_replay_candidate_reason(replay_item)
                    if candidate_reason is None:
                        historical_trainable_positive_organs.add(organ)
                safe, reason = _is_replay_safe_positive(replay_item)
                key = (str(replay_item.get("case_id") or ""), organ)
                if not safe:
                    replay_excluded_rows.append({
                        "manifest": str(replay_manifest),
                        "index": item_idx,
                        "case_id": key[0],
                        "organ": organ,
                        "reason": reason,
                    })
                    continue
                if not key[0] or not key[1]:
                    replay_excluded_rows.append({
                        "manifest": str(replay_manifest),
                        "index": item_idx,
                        "case_id": key[0],
                        "organ": organ,
                        "reason": "missing_case_or_organ",
                    })
                    continue
                if key in seen_replay_keys:
                    continue
                replay_positive_rows.append(_normalize_replay_positive_item(
                    replay_item,
                    manifest_path=replay_manifest,
                    replay_index=item_idx + manifest_idx * 1_000_000,
                ))
                seen_replay_keys.add(key)

        rows.extend(replay_positive_rows)
        final_trainable_positive_organs = {
            str(r.get("organ") or "")
            for r in rows
            if _is_trainable_positive_row(r, require_nonempty=False)
        }
        recovered_historical_organs = sorted(
            organ for organ in historical_trainable_positive_organs
            if organ in final_trainable_positive_organs
        )
        missing_historical_organs = sorted(historical_trainable_positive_organs - final_trainable_positive_organs)
        positive_rows = [r for r in rows if _is_positive_supervision(r) and not _is_negative_absent(r)]
        negative_rows = [r for r in rows if str(r.get("supervision_type") or "").lower() == "negative"]
        trainable_positive_rows = [r for r in positive_rows if _is_trainable_positive_row(r, require_nonempty=False)]
        positive_grade_counts = {
            grade: sum(1 for r in positive_rows if str(r.get("grade") or "").upper() == grade)
            for grade in ["A", "B", "C", "D"]
        }
        included_positive_grade_counts = {
            grade: sum(1 for r in trainable_positive_rows if str(r.get("grade") or "").upper() == grade)
            for grade in ["A", "B", "C", "D"]
        }
        all_grade_counts = {
            grade: sum(1 for r in rows if str(r.get("grade") or "").upper() == grade)
            for grade in ["A", "B", "C", "D"]
        }

        manifest = {
            "stage": "voxtell_3d_prompt_training_manifest",
            "status": "success",
            "student_backend": "voxtell_style_3d_prompt",
            "target_config": str(self.target_config),
            "official_prompt_map": str(prompt_map_path),
            "official_prompt_map_version": prompt_map_doc.get("schema_version"),
            "official_prompt_exact_match_count": prompt_map_doc.get("official_exact_match_count"),
            "official_prompt_project_extension_count": prompt_map_doc.get("project_extension_count"),
            "formal_373_target_validation": validate_formal_373_target_space(
                self.target_config,
                requested_organs=list(target_organs),
                require_full_target=True,
            ),
            "cases_root": str(cases),
            "case_list": str(Path(case_list).resolve()) if case_list else None,
            "student_prediction_root": str(ignored_student_prediction_root) if ignored_student_prediction_root else None,
            "historical_replay_manifests": historical_replay_source_manifests,
            "num_items": len(rows),
            "num_cases": len({r["case_id"] for r in rows}),
            "num_classes": len(target_organs),
            "expected_targets": len(case_dirs) * len(target_organs),
            "manifest_targets": len(rows),
            "candidate_pseudo_targets": len(positive_rows),
            "absent_negative_targets": sum(1 for r in rows if str(r.get("target_type")) in {"absent_negative", "negative_absent"}),
            "all_zero_masks": len({str(r.get("mask_path") or r.get("mask") or "") for r in rows if str(r.get("target_type")) in {"absent_negative", "negative_absent"} and (r.get("mask_path") or r.get("mask"))}),
            "num_items_missing_image": sum(1 for r in rows if not r.get("image")),
            "grade_counts": positive_grade_counts,
            "positive_grade_counts": positive_grade_counts,
            "all_grade_counts": all_grade_counts,
            "num_positive_items": len(positive_rows),
            "num_trainable_positive_items": len(trainable_positive_rows),
            "num_negative_items": len(negative_rows),
            "negative_prompt_ratio": max_negative_ratio,
            "negative_prompt_ratio_note": "Candidate-manifest compatibility field only; final training positive/negative mix is controlled at runtime by scripts/train_voxtell_prompt_student.py --pos-neg-ratio.",
            "negative_quota_policy": negative_quota_policy,
            "negative_source_counts": negative_source_counts,
            "negative_source_shortfalls": negative_source_shortfalls,
            "negative_prompt_policy": {
                "rule": "This manifest records positive/negative candidate pools. Runtime training samples from these pools with --pos-neg-ratio; do not interpret manifest row counts as the final positive/negative training ratio.",
                "allowed_sources": [
                    "nonmedical_absent_object",
                    "out_of_scan_anatomy_with_coverage_evidence",
                    "explicit_confirmed_absent_anatomy",
                    "case_373_expected_absent",
                ],
                "zero_mask_note": "All-zero masks are trainable only for target_type=negative_absent when scan coverage/FOV proves the organ is absent; unresolved/partial zeros are I/O placeholders with weight 0.",
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
            "num_distillation_eligible_positive_items": sum(1 for r in trainable_positive_rows if r.get("distillation_eligible") is not False and float(r.get("training_weight") or 0.0) > 0.0),
            "num_zero_weight_items": sum(1 for r in rows if float(r.get("training_weight") or 0.0) == 0.0),
            "target_type_counts": {
                target_type: sum(
                    1 for r in rows
                    if str(r.get("target_type") or "positive_hard") == target_type
                )
                for target_type in [
                    "positive_hard", "positive_soft", "negative_absent",
                    "unresolved_visible", "partial_fov", "rejected",
                ]
            },
            "training_gate_summary": {
                "num_included": len(rows),
                "num_included_positive": len(positive_rows),
                "num_trainable_positive": len(trainable_positive_rows),
                "included_grade_counts": included_positive_grade_counts,
                "positive_grade_counts": positive_grade_counts,
                "all_grade_counts": all_grade_counts,
                "ab_nonzero": int(included_positive_grade_counts.get("A", 0) or 0) + int(included_positive_grade_counts.get("B", 0) or 0),
                "negative_absent_excluded_from_positive_quota": True,
                "num_excluded_positive": len(skipped_ineligible_positive),
                "num_c_soft_included": sum(1 for r in rows if r.get("grade") == "C" and r.get("target_type") == "soft" and r.get("probability_mask_path")),
                "num_c_missing_probability_excluded": sum(1 for r in skipped_ineligible_positive if r.get("reason") == "soft_target_probability_mask_missing"),
                "num_c_hard_or_provisional_excluded": sum(1 for r in skipped_ineligible_positive if r.get("reason") == "grade_C_requires_soft_probability_target"),
                "num_d_excluded": sum(1 for r in skipped_ineligible_positive if r.get("grade") == "D"),
                "num_unsupported_schema_excluded": sum(1 for r in skipped_ineligible_positive if r.get("reason") == "unsupported_scoring_schema_requires_rescoring"),
            },
            "cumulative_manifest_summary": {
                "enabled": bool(replay_manifest_paths),
                "policy": "current_trainable_positive + promoted_historical_replay_positive + current_legal_negative",
                "current_trainable_positive_organs": sorted(current_trainable_positive_organs),
                "historical_trainable_positive_organs": sorted(historical_trainable_positive_organs),
                "recovered_historical_trainable_positive_organs": recovered_historical_organs,
                "missing_historical_trainable_positive_organs": missing_historical_organs,
                "num_replay_positive_items": len(replay_positive_rows),
                "num_replay_excluded_items": len(replay_excluded_rows),
                "replay_excluded_rows": replay_excluded_rows[:200],
                "negative_absent_is_positive": False,
                "student_prediction_gt_empty_qc_fail_replay_blocked": True,
            },
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
    return filter_semantically_risky_prompts(variants)


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
        "selected_provider": meta.get("selected_provider") or meta.get("selected_model") or meta.get("source_model"),
        "selection_reason": meta.get("selection_reason") or meta.get("selection_method") or meta.get("fallback_reason"),
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
        "human_review_status": meta.get("human_review_status", "pending"),
        "human_review_reason": meta.get("human_review_reason"),
        "review_packet": meta.get("review_packet"),
        "affects_training_manifest": meta.get("affects_training_manifest"),
        "quality_flags": meta.get("quality_flags", []),
        "quality_status": meta.get("quality_status") or _quality_status(meta.get("review_flags", []), meta.get("quality_flags", [])),
        "source_metadata_available": bool(meta),
        "distillation_source": meta.get("selected_model") or meta.get("source_model"),
        "source_quality": meta.get("quality_status") or meta.get("grade", "unknown"),
    }

def find_voxtell_executable() -> str | None:
    return shutil.which("voxtell-predict")
