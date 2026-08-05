#!/usr/bin/env python3
"""Project-specific fine-tuning entry for the VoxTell-style 3D prompt student.

The official VoxTell release currently provides inference code and pretrained
weights, but not a project fine-tuning script. This trainer fills that gap for
our EM loop: it reads prompt/mask pairs from
`voxtell_prompt_student_manifest.json`, loads a VoxTell checkpoint, and
fine-tunes the 3D prompt segmentation network with frozen text embeddings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pydoc
import random
import shutil
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VOXTELL_ROOT = ROOT / "third_party" / "VoxTell"
sys.path.insert(0, str(VOXTELL_ROOT))
sys.path.insert(0, str(ROOT / "agent-harness"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch._dynamo import OptimizedModule
from transformers import AutoModel, AutoTokenizer

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from cli_anything.medai.core.continual_learning import (
    TRAINING_CONTRACT_VERSION,
    sha256_file as contract_sha256_file,
)


SAFE_NEGATIVE_SOURCES = {
    "nonmedical_absent_object",
    "out_of_scan_anatomy_with_coverage_evidence",
    "explicit_confirmed_absent_anatomy",
    "case_373_expected_absent",
}
SAFE_ZERO_MASK_ROLES = {"negative_target_mask", "absent_negative_target_mask", "negative_absent_target_mask"}
PAPER_ALIGNED_PROFILE = "paper_aligned"
QUALITY_WEIGHTED_PROFILE = "quality_weighted_ablation"
OFFICIAL_VOXTELL_COMMIT = "ec517b79a19aa59b25789c878d808790326e9651"
ACCEPTED_AUTOLABEL_SCHEMAS = {"autolabel_core_v2", "autolabel_core_v3"}
NEGATIVE_REASON_MAP = {
    "case_373_expected_absent": "absent_in_scan",
    "out_of_scan_anatomy_with_coverage_evidence": "absent_in_scan",
    "explicit_confirmed_absent_anatomy": "absent_in_scan",
    "nonmedical_absent_object": "wrong_prompt",
}
DERIVED_CROP_NEGATIVE_EXACT_EXCLUSIONS = {
    "abdominal_cavity",
    "body",
    "torso",
    "trunk",
    "skin",
    "subcutaneous_tissue",
}
DERIVED_CROP_NEGATIVE_KEYWORD_EXCLUSIONS = (
    "cavity",
    "body_region",
    "whole_body",
)


def parse_args() -> argparse.Namespace:
    env = os.environ
    ap = argparse.ArgumentParser(description="Fine-tune VoxTell-style 3D prompt student on prompt/mask manifest.")
    ap.add_argument("--manifest", default=env.get("MEDAI_PROMPT_STUDENT_MANIFEST"), help="Prompt/mask manifest JSON.")
    ap.add_argument("--model-dir", default=env.get("MEDAI_VOXTELL_MODEL_DIR"), help="VoxTell model dir with plans.json and fold_0/checkpoint_final.pth.")
    ap.add_argument("--output-dir", default=env.get("MEDAI_PROMPT_STUDENT_OUTPUT_DIR", str(ROOT / "outputs/voxtell_prompt_mstep")))
    ap.add_argument("--text-encoding-model", default=env.get("MEDAI_TEXT_ENCODING_MODEL", "Qwen/Qwen3-Embedding-4B"))
    ap.add_argument("--embedding-cache", default=None, help="Optional torch cache for prompt embeddings.")
    ap.add_argument("--official-embedding-bank", default=env.get("MEDAI_VOXTELL_EMBEDDING_BANK"))
    ap.add_argument("--training-profile", choices=[PAPER_ALIGNED_PROFILE, QUALITY_WEIGHTED_PROFILE],
                    default=env.get("MEDAI_VOXTELL_TRAINING_PROFILE", PAPER_ALIGNED_PROFILE))
    ap.add_argument("--device", default=env.get("MEDAI_DEVICE", "cuda"))
    ap.add_argument("--epochs", type=int, default=int(env.get("MEDAI_MSTEP_EPOCHS", "1")))
    ap.add_argument("--max-steps", type=int, default=int(env.get("MEDAI_MAX_STEPS", "0")), help="0 means one pass over manifest per epoch.")
    ap.add_argument("--max-items", type=int, default=int(env.get("MEDAI_MAX_ITEMS", "0")), help="Optional subset for smoke tests.")
    ap.add_argument("--learning-rate", type=float, default=float(env.get("MEDAI_MSTEP_LR", "1e-4")))
    ap.add_argument("--weight-decay", type=float, default=float(env.get("MEDAI_WEIGHT_DECAY", "3e-5")))
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default=env.get("MEDAI_OPTIMIZER", "sgd"))
    ap.add_argument(
        "--amp-mode",
        choices=["auto", "off"],
        default=env.get("MEDAI_AMP_MODE", "auto"),
        help="auto enables CUDA AMP; off uses full precision and disables GradScaler.",
    )
    ap.add_argument("--poly-power", type=float, default=float(env.get("MEDAI_POLY_POWER", "0.9")))
    ap.add_argument("--deep-supervision", action="store_true", default=env.get("MEDAI_DEEP_SUPERVISION", "1").lower() not in {"0", "false", "no"})
    ap.add_argument("--foreground-prob", type=float, default=float(env.get("MEDAI_FOREGROUND_PROB", "0.85")))
    ap.add_argument("--batch-size", type=int, default=int(env.get("MEDAI_MSTEP_BATCH_SIZE", "2")))
    ap.add_argument("--pos-neg-ratio", default=env.get("MEDAI_POS_NEG_RATIO", "2:1"), help="Runtime positive:negative sampling ratio, e.g. 2:1, 1:1, 10:1, or 2.0.")
    ap.add_argument(
        "--absent-derived-negative-ratio",
        default=env.get("MEDAI_ABSENT_DERIVED_NEGATIVE_RATIO", "1:1"),
        help="Within negative steps, manifest absent/wrong-prompt versus derived empty-crop ratio.",
    )
    ap.add_argument("--sampling-log-interval", type=int, default=int(env.get("MEDAI_SAMPLING_LOG_INTERVAL", "10")), help="Steps per sampling-stat log window.")
    ap.add_argument("--seed", type=int, default=int(env.get("MEDAI_SEED", "42")))
    ap.add_argument("--save-every", type=int, default=int(env.get("MEDAI_SAVE_EVERY", "0")), help="0 disables intermediate checkpoints.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and write a training plan without loading Qwen/model weights.")
    ap.add_argument("--ddp", action="store_true", default=os.getenv("WORLD_SIZE", "1") not in {"", "1"}, help="Enable single-node torch.distributed training when launched with torchrun.")
    ap.add_argument("--local-rank", "--local_rank", type=int, default=int(env.get("LOCAL_RANK", "0")), help="Local rank provided by torchrun.")
    ap.add_argument(
        "--run-spec",
        default=env.get("MEDAI_RUN_SPEC"),
        help="RunSpec JSON. Required when MEDAI_FORMAL_STATE_MACHINE=1.",
    )
    ap.add_argument("--freeze-encoder", action="store_true", help="Only train prompt projection/decoder layers.")
    ap.add_argument(
        "--trainable-scope",
        choices=["all_decoder", "prompt_path"],
        default=env.get("MEDAI_TRAINABLE_SCOPE", "all_decoder"),
        help="prompt_path freezes the spatial decoder and only adapts text/image prompt fusion layers.",
    )
    ap.add_argument(
        "--bce-pos-weight-cap",
        type=float,
        default=float(env.get("MEDAI_BCE_POS_WEIGHT_CAP", "100")),
        help="Cap for per-patch foreground-balanced BCE; <=1 restores unweighted BCE.",
    )
    ap.add_argument(
        "--official-retention-weight",
        type=float,
        default=float(env.get("MEDAI_OFFICIAL_RETENTION_WEIGHT", "0")),
        help="Functional distillation weight against a frozen official VoxTell copy; 0 disables it.",
    )
    ap.add_argument(
        "--grad-clip-norm",
        type=float,
        default=float(env.get("MEDAI_GRAD_CLIP_NORM", "1.0")),
        help="Maximum gradient norm after AMP unscale; <=0 disables clipping.",
    )
    return ap.parse_args()


def init_distributed_if_needed(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "0") or 0)
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)) or 0)
    enabled = bool(args.ddp or world_size > 1)
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA devices")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return {"enabled": enabled, "world_size": world_size, "rank": rank, "local_rank": local_rank, "is_rank0": rank == 0}


def cleanup_distributed(ddp: dict[str, Any]) -> None:
    if ddp.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


def ddp_mean(value: float, device: torch.device, ddp: dict[str, Any]) -> float:
    if not ddp.get("enabled"):
        return float(value)
    tensor = torch.tensor([float(value)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= max(int(ddp.get("world_size") or 1), 1)
    return float(tensor.item())


def maybe_wrap_ddp(module: nn.Module, ddp: dict[str, Any]) -> nn.Module:
    if not ddp.get("enabled"):
        return module
    return DistributedDataParallel(module, device_ids=[int(ddp["local_rank"])], output_device=int(ddp["local_rank"]), find_unused_parameters=False)


def model_state_dict(module: nn.Module) -> dict[str, Any]:
    return module.module.state_dict() if isinstance(module, DistributedDataParallel) else module.state_dict()


def load_model_state_dict(module: nn.Module, state: dict[str, Any], *, strict: bool = True) -> None:
    target = module.module if isinstance(module, DistributedDataParallel) else module
    target.load_state_dict(state, strict=strict)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def outbound_write_audit() -> dict[str, Any]:
    explicit = {}
    for key in ("MEDAI_PUSH_TO_HUB", "MEDAI_UPLOAD_ARTIFACTS", "MEDAI_GIT_PUSH"):
        value = os.environ.get(key)
        if value and value.lower() not in {"0", "false", "no", "off"}:
            explicit[key] = value
    return {
        "status": "failed" if explicit else "passed",
        "policy": "official_sources_read_only_no_project_uploads",
        "forbidden_write_flags": explicit,
        "external_uploads_performed": False,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _manifest_image_path(row: dict[str, Any]) -> Path | None:
    value = row.get("image") or row.get("ct_path")
    return Path(str(value)) if value else None


def _manifest_mask_path(row: dict[str, Any]) -> Path | None:
    value = row.get("mask") or row.get("mask_path")
    return Path(str(value)) if value else None


@lru_cache(maxsize=8192)
def _nifti_audit_meta_cached(path_text: str, inspect_foreground: bool) -> tuple[dict[str, Any], str | None]:
    path = Path(path_text)
    if not path.exists():
        return {}, "missing"
    try:
        import nibabel as nib
        img = nib.load(str(path))
        shape = tuple(int(x) for x in img.shape[:3])
        zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
        affine = np.asarray(img.affine, dtype=float)
        try:
            orientation = "".join(nib.orientations.aff2axcodes(affine))
        except Exception:
            orientation = None
        meta: dict[str, Any] = {
            "shape": list(shape),
            "spacing": [round(x, 6) for x in zooms],
            "orientation": orientation,
            "affine": [[round(float(v), 6) for v in row] for row in affine.tolist()],
        }
        if inspect_foreground:
            data = np.asanyarray(img.dataobj)
            finite = bool(np.isfinite(data).all())
            foreground = int(np.count_nonzero(data > 0))
            voxel_count = int(np.prod(shape)) if shape else 0
            meta.update({
                "finite": finite,
                "foreground_voxels": foreground,
                "voxel_count": voxel_count,
                "foreground_voxel_ratio": (
                    float(foreground / voxel_count) if voxel_count else None
                ),
            })
        return meta, None
    except Exception as exc:
        return {}, f"read_error:{exc}"


def _nifti_audit_meta(path: Path | None, inspect_foreground: bool = False) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, "missing_path"
    return _nifti_audit_meta_cached(str(path.resolve()), bool(inspect_foreground))


def _alignment_fail_reasons(
    image_meta: dict[str, Any],
    mask_meta: dict[str, Any],
    row: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if image_meta.get("shape") and mask_meta.get("shape") and image_meta["shape"] != mask_meta["shape"]:
        reasons.append("ct_mask_shape_mismatch")
    if image_meta.get("spacing") and mask_meta.get("spacing") and image_meta["spacing"] != mask_meta["spacing"]:
        reasons.append("ct_mask_spacing_mismatch")
    if image_meta.get("orientation") and mask_meta.get("orientation") and image_meta["orientation"] != mask_meta["orientation"]:
        reasons.append("ct_mask_orientation_mismatch")
    if image_meta.get("affine") and mask_meta.get("affine"):
        try:
            if not np.allclose(np.asarray(image_meta["affine"]), np.asarray(mask_meta["affine"]), atol=1e-3):
                reasons.append("ct_mask_affine_mismatch")
        except Exception:
            reasons.append("ct_mask_affine_unreadable")
    target_type = str(row.get("target_type") or "").lower()
    supervision_type = str(row.get("supervision_type") or "positive").lower()
    if "foreground_voxels" in mask_meta:
        foreground = int(mask_meta.get("foreground_voxels") or 0)
        if supervision_type == "positive" and target_type not in {"negative_absent", "absent_negative"} and foreground <= 0:
            reasons.append("positive_mask_empty")
        if target_type in {"negative_absent", "absent_negative"} and foreground > 0:
            reasons.append("negative_absent_mask_nonzero")
    if supervision_type == "negative":
        source = str(row.get("negative_source") or "")
        role = str(row.get("zero_mask_role") or "")
        if source not in SAFE_NEGATIVE_SOURCES:
            reasons.append("negative_source_not_safe")
        if role not in SAFE_ZERO_MASK_ROLES:
            reasons.append("zero_mask_role_not_safe")
    if not row.get("organ") or not (row.get("prompt") or row.get("prompt_text")):
        reasons.append("prompt_organ_mapping_missing")
    return reasons


def build_prompt_mask_ct_alignment_audit(rows: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    """Audit CT/mask/prompt contracts for rows that can enter training."""
    strict_scan = os.getenv("MEDAI_ALIGNMENT_AUDIT_STRICT_SCAN", "1").lower() not in {"0", "false", "no"}
    alignment_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    foreground_ratios: list[float] = []
    positive_empty = 0
    negative_nonzero = 0
    for idx, row in enumerate(rows):
        image_path = _manifest_image_path(row)
        mask_path = _manifest_mask_path(row)
        image_meta, image_error = _nifti_audit_meta(image_path, inspect_foreground=False)
        mask_meta, mask_error = _nifti_audit_meta(mask_path, inspect_foreground=strict_scan)
        reasons: list[str] = []
        if image_error:
            reasons.append(f"image_{image_error}")
        if mask_error:
            reasons.append(f"mask_{mask_error}")
        if not image_error and not mask_error:
            reasons.extend(_alignment_fail_reasons(image_meta, mask_meta, row))
        if "positive_mask_empty" in reasons:
            positive_empty += 1
        if "negative_absent_mask_nonzero" in reasons:
            negative_nonzero += 1
        if mask_meta.get("foreground_voxel_ratio") is not None:
            foreground_ratios.append(float(mask_meta["foreground_voxel_ratio"]))
        audit_row = {
            "row_index": idx,
            "case_id": row.get("case_id"),
            "organ": row.get("organ"),
            "supervision_type": row.get("supervision_type", "positive"),
            "target_type": row.get("target_type"),
            "image_path": str(image_path) if image_path else "",
            "mask_path": str(mask_path) if mask_path else "",
            "image_shape": image_meta.get("shape"),
            "mask_shape": mask_meta.get("shape"),
            "image_spacing": image_meta.get("spacing"),
            "mask_spacing": mask_meta.get("spacing"),
            "image_orientation": image_meta.get("orientation"),
            "mask_orientation": mask_meta.get("orientation"),
            "mask_foreground_voxels": mask_meta.get("foreground_voxels"),
            "mask_foreground_voxel_ratio": mask_meta.get("foreground_voxel_ratio"),
            "status": "failed" if reasons else "passed",
            "reasons": reasons,
        }
        alignment_rows.append(audit_row)
        if reasons:
            failures.append(audit_row)
    positive = [r for r in rows if str(r.get("supervision_type") or "positive") == "positive"]
    negative = [r for r in rows if str(r.get("supervision_type") or "") == "negative"]
    return {
        "stage": "prompt_mask_ct_alignment_audit",
        "status": "passed" if not failures else "failed",
        "manifest": str(manifest_path),
        "strict_mask_foreground_scan": strict_scan,
        "policy": "CT is the real input image; masks are project-generated selected pseudo labels used as student supervision.",
        "items": len(rows),
        "positive_items": len(positive),
        "negative_absent_items": len(negative),
        "num_alignment_failures": len(failures),
        "alignment_failure_examples": failures[:20],
        "positive_empty_masks": positive_empty,
        "negative_absent_nonzero_masks": negative_nonzero,
        "mean_mask_foreground_voxel_ratio": (
            float(sum(foreground_ratios) / len(foreground_ratios))
            if foreground_ratios else None
        ),
        "positive_negative_ratio": (len(positive) / max(1, len(negative))) if negative else None,
        "rows_csv": "prompt_mask_ct_alignment_audit.csv",
        "_rows": alignment_rows,
    }


def build_prompt_organ_mapping_audit(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("prompt_text")
        organ = row.get("organ") or row.get("canonical_organ")
        mask_path = row.get("mask_path") or row.get("mask")
        status = "passed" if organ and prompt and mask_path else "failed"
        audit_row = {
            "row_index": idx,
            "case_id": row.get("case_id"),
            "organ": organ,
            "canonical_organ_name": row.get("canonical_organ_name") or organ,
            "student_target_id": row.get("student_target_id"),
            "prompt": prompt,
            "prompt_type": row.get("prompt_type"),
            "supervision_type": row.get("supervision_type", "positive"),
            "target_type": row.get("target_type"),
            "label_role": row.get("label_role"),
            "supervision_role": row.get("supervision_role"),
            "mask_path": mask_path,
            "training_weight": row.get("training_weight"),
            "status": status,
        }
        mapping_rows.append(audit_row)
        if status == "failed":
            failures.append(audit_row)
    audit = {
        "stage": "prompt_organ_mapping_audit",
        "status": "passed" if not failures else "failed",
        "items": len(mapping_rows),
        "unique_organs": len({str(r.get("organ")) for r in mapping_rows if r.get("organ")}),
        "unique_prompts": len({str(r.get("prompt")) for r in mapping_rows if r.get("prompt")}),
        "missing_mapping_examples": failures[:20],
    }
    return audit, mapping_rows


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _positive_organs_from_history_row(row: dict[str, Any]) -> list[str]:
    organs = row.get("positive_foreground_organs") or row.get("positive_organs")
    if isinstance(organs, list):
        return [str(x) for x in organs if str(x)]
    organ = str(row.get("organ") or "")
    if organ and str(row.get("sample_kind") or "") != "negative":
        return [organ]
    return []


def _negative_organs_from_history_row(row: dict[str, Any]) -> list[str]:
    organ = row.get("negative_organ")
    if organ:
        return [str(organ)]
    if str(row.get("sample_kind") or "") == "negative" and row.get("organ"):
        return [str(row["organ"])]
    return []


def build_sampling_gradient_audits(
    *,
    training_rows: list[dict[str, Any]],
    loss_history: list[dict[str, Any]],
    positive_organ_sample_counts: dict[str, int] | None = None,
    negative_organ_sample_counts: dict[str, int] | None = None,
    positive_organ_finite_gradient_mass: dict[str, float] | None = None,
    positive_organ_nonfinite_gradient_steps: dict[str, int] | None = None,
    legacy_organ_sample_counts: dict[str, int] | None = None,
    legacy_gradient_shares: dict[str, float] | None = None,
    min_positive_exposure_count: int | None = None,
    nonfinite_gradient_step_rate_threshold: float | None = None,
) -> dict[str, Any]:
    """Build robust M-step sampling and gradient audits.

    Exposure is counted independently from finite gradient mass so a single NaN
    gradient norm cannot make all organs appear underrepresented.
    """
    positive_organ_sample_counts = dict(positive_organ_sample_counts or {})
    negative_organ_sample_counts = dict(negative_organ_sample_counts or {})
    positive_organ_finite_gradient_mass = dict(positive_organ_finite_gradient_mass or {})
    positive_organ_nonfinite_gradient_steps = dict(positive_organ_nonfinite_gradient_steps or {})
    legacy_organ_sample_counts = dict(legacy_organ_sample_counts or {})
    legacy_gradient_shares = dict(legacy_gradient_shares or {})
    min_positive_exposure_count = (
        int(os.getenv("MEDAI_MIN_POSITIVE_ORGAN_EXPOSURE", "1"))
        if min_positive_exposure_count is None else int(min_positive_exposure_count)
    )
    nonfinite_gradient_step_rate_threshold = (
        float(os.getenv("MEDAI_NONFINITE_GRADIENT_STEP_RATE_FAIL", "0.01"))
        if nonfinite_gradient_step_rate_threshold is None
        else float(nonfinite_gradient_step_rate_threshold)
    )

    # Offline recompute path: reconstruct exposure from new loss history fields.
    if not positive_organ_sample_counts and loss_history:
        for row in loss_history:
            for organ in set(_positive_organs_from_history_row(row)):
                positive_organ_sample_counts[organ] = positive_organ_sample_counts.get(organ, 0) + 1
                if _is_finite_number(row.get("grad_norm_before_clip")):
                    positive_organ_finite_gradient_mass[organ] = (
                        positive_organ_finite_gradient_mass.get(organ, 0.0)
                        + float(row.get("grad_norm_before_clip") or 0.0)
                        / max(1, len(set(_positive_organs_from_history_row(row))))
                    )
                else:
                    positive_organ_nonfinite_gradient_steps[organ] = (
                        positive_organ_nonfinite_gradient_steps.get(organ, 0) + 1
                    )
            for organ in set(_negative_organs_from_history_row(row)):
                negative_organ_sample_counts[organ] = negative_organ_sample_counts.get(organ, 0) + 1

    all_positive_organs = sorted(
        {
            str(row.get("organ") or "")
            for row in training_rows
            if row.get("supervision_type", "positive") == "positive"
            and str(row.get("organ") or "")
        }
    )
    positive_exposure_total = sum(positive_organ_sample_counts.values())
    expected_gradient_organs = (
        all_positive_organs
        if positive_exposure_total >= len(all_positive_organs)
        else all_positive_organs
    )
    finite_gradient_mass_total = sum(
        float(value)
        for value in positive_organ_finite_gradient_mass.values()
        if _is_finite_number(value)
    )
    nonfinite_gradient_steps = sum(
        1 for row in loss_history
        if not _is_finite_number(row.get("grad_norm_before_clip"))
    )
    optimizer_steps_skipped_nonfinite_grad = sum(
        1 for row in loss_history
        if bool(row.get("optimizer_step_skipped_nonfinite_grad"))
    )
    nonfinite_gradient_step_rate = (
        float(nonfinite_gradient_steps / len(loss_history)) if loss_history else 0.0
    )
    minimum_gradient_share = (
        0.1 / len(expected_gradient_organs)
        if expected_gradient_organs else 0.0
    )
    gradient_shares = {
        organ: (
            float(positive_organ_finite_gradient_mass.get(organ, 0.0))
            / finite_gradient_mass_total
            if finite_gradient_mass_total > 0 else None
        )
        for organ in all_positive_organs
    }
    exposure_rows: list[dict[str, Any]] = []
    underexposed_organs: list[str] = []
    underrepresented_organs: list[str] = []
    issue_counts: Counter[str] = Counter()
    for organ in all_positive_organs:
        exposure_count = int(positive_organ_sample_counts.get(organ, 0))
        legacy_count = int(legacy_organ_sample_counts.get(organ, 0))
        nonfinite_count = int(positive_organ_nonfinite_gradient_steps.get(organ, 0))
        finite_mass = float(positive_organ_finite_gradient_mass.get(organ, 0.0))
        share = gradient_shares.get(organ)
        issues: list[str] = []
        if exposure_count <= 0 and legacy_count > 0:
            issues.append("paper_aligned_metadata_missing_organ_in_loss_history")
        elif exposure_count <= 0:
            issues.append("organ_never_sampled")
            underexposed_organs.append(organ)
        elif exposure_count < min_positive_exposure_count:
            issues.append("organ_sampled_but_below_minimum_exposure")
            underexposed_organs.append(organ)
        if exposure_count > 0 and finite_mass <= 0 and nonfinite_count > 0:
            issues.append("organ_sampled_but_gradient_nan")
        if finite_gradient_mass_total > 0 and share is not None and share < minimum_gradient_share:
            issues.append("organ_gradient_share_below_policy_minimum")
            underrepresented_organs.append(organ)
        if not issues:
            issues.append("ok")
        for issue in issues:
            issue_counts[issue] += 1
        exposure_rows.append({
            "organ": organ,
            "positive_sample_count": exposure_count,
            "negative_sample_count": int(negative_organ_sample_counts.get(organ, 0)),
            "legacy_sample_count": legacy_count,
            "finite_gradient_mass": finite_mass,
            "gradient_share": share,
            "nonfinite_gradient_steps": nonfinite_count,
            "issues": issues,
        })

    legacy_all_zero_shares = bool(
        legacy_gradient_shares
        and all(float(v or 0.0) == 0.0 for v in legacy_gradient_shares.values())
        and any(int(v or 0) > 0 for k, v in legacy_organ_sample_counts.items() if k in all_positive_organs)
    )
    failure_reasons: list[str] = []
    if underexposed_organs:
        failure_reasons.append("organ_exposure_below_policy_minimum")
    if finite_gradient_mass_total <= 0:
        failure_reasons.append("gradient_mass_nonfinite_or_insufficient")
    if nonfinite_gradient_step_rate > nonfinite_gradient_step_rate_threshold:
        failure_reasons.append("nonfinite_gradient_step_rate_above_policy")
    if underrepresented_organs:
        failure_reasons.append("organ_gradient_share_below_policy_minimum")
    if legacy_all_zero_shares:
        issue_counts["all_gradient_shares_zero_due_to_nan_total"] += 1

    organ_exposure_audit = {
        "status": "passed" if not underexposed_organs else "failed",
        "expected_organs": expected_gradient_organs,
        "min_positive_exposure_count": min_positive_exposure_count,
        "underexposed_organs": underexposed_organs,
        "positive_organ_sample_counts": positive_organ_sample_counts,
        "negative_organ_sample_counts": negative_organ_sample_counts,
    }
    nonfinite_gradient_audit = {
        "status": (
            "passed"
            if nonfinite_gradient_step_rate <= nonfinite_gradient_step_rate_threshold
            and finite_gradient_mass_total > 0
            else "failed"
        ),
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "optimizer_steps_skipped_nonfinite_grad": optimizer_steps_skipped_nonfinite_grad,
        "total_steps": len(loss_history),
        "nonfinite_gradient_step_rate": nonfinite_gradient_step_rate,
        "fail_threshold": nonfinite_gradient_step_rate_threshold,
        "finite_gradient_mass_total": finite_gradient_mass_total,
        "positive_organ_nonfinite_gradient_steps": positive_organ_nonfinite_gradient_steps,
    }
    organ_gradient_audit = {
        "status": "passed" if not failure_reasons else "failed",
        "failure_reasons": failure_reasons,
        "positive_steps": positive_exposure_total,
        "expected_organs": expected_gradient_organs,
        "minimum_gradient_share": minimum_gradient_share,
        "underrepresented_organs": underrepresented_organs,
        "sample_counts": positive_organ_sample_counts,
        "negative_sample_counts": negative_organ_sample_counts,
        "gradient_shares": gradient_shares,
        "positive_organ_finite_gradient_mass": positive_organ_finite_gradient_mass,
        "positive_organ_nonfinite_gradient_steps": positive_organ_nonfinite_gradient_steps,
        "finite_gradient_mass_total": finite_gradient_mass_total,
        "nonfinite_gradient_step_rate": nonfinite_gradient_step_rate,
        "legacy_all_zero_gradient_share_bug_detected": legacy_all_zero_shares,
    }
    diagnosis = {
        "stage": "student_sampling_gradient_diagnosis",
        "status": "passed" if not failure_reasons else "failed",
        "failure_reasons": failure_reasons,
        "issue_counts": dict(issue_counts),
        "all_gradient_shares_zero_due_to_nan_total": legacy_all_zero_shares,
        "paper_aligned_metadata_missing_organ_in_loss_history": any(
            "paper_aligned_metadata_missing_organ_in_loss_history" in row["issues"]
            for row in exposure_rows
        ),
        "organ_rows": exposure_rows,
    }
    return {
        "organ_gradient_audit": organ_gradient_audit,
        "organ_exposure_audit": organ_exposure_audit,
        "nonfinite_gradient_audit": nonfinite_gradient_audit,
        "student_sampling_gradient_diagnosis": diagnosis,
        "organ_rows": exposure_rows,
    }


def build_training_stability_diagnosis(
    loss_history: list[dict[str, Any]],
    *,
    nonfinite_gradient_step_rate_threshold: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = (
        float(os.getenv("MEDAI_NONFINITE_GRADIENT_STEP_RATE_FAIL", "0.01"))
        if nonfinite_gradient_step_rate_threshold is None
        else float(nonfinite_gradient_step_rate_threshold)
    )
    total = len(loss_history)
    nonfinite_gradient_rows = [
        row for row in loss_history
        if not _is_finite_number(row.get("grad_norm_before_clip"))
        or bool(row.get("optimizer_step_skipped_nonfinite_grad"))
    ]
    nonfinite_component_rows = [
        row for row in loss_history
        if not bool(row.get("loss_component_finite", True))
    ]
    finite_component_count = total - len(nonfinite_component_rows)
    nonfinite_rate = float(len(nonfinite_gradient_rows) / total) if total else 0.0
    component_finite_rate = float(finite_component_count / total) if total else 0.0
    failure_reasons: list[str] = []
    if nonfinite_rate > threshold:
        failure_reasons.append("nonfinite_gradient_step_rate_above_policy")
    if nonfinite_component_rows:
        failure_reasons.append("loss_component_nonfinite")
    diagnostic_rows = []
    for row in loss_history:
        reasons: list[str] = []
        if not _is_finite_number(row.get("grad_norm_before_clip")):
            reasons.append("nonfinite_grad_norm")
        if bool(row.get("optimizer_step_skipped_nonfinite_grad")):
            reasons.append("optimizer_step_skipped_nonfinite_grad")
        if not bool(row.get("loss_component_finite", True)):
            reasons.append("loss_component_nonfinite")
        if reasons:
            diagnostic_rows.append({
                "step": row.get("step"),
                "case_id": row.get("case_id"),
                "organ": row.get("organ"),
                "positive_organs": row.get("positive_organs"),
                "negative_organ": row.get("negative_organ"),
                "loss": row.get("loss"),
                "task_loss": row.get("task_loss"),
                "bce_component": row.get("bce_component"),
                "dice_component": row.get("dice_component"),
                "grad_norm_before_clip": row.get("grad_norm_before_clip"),
                "amp_enabled": row.get("amp_enabled"),
                "grad_scaler_scale_before": row.get("grad_scaler_scale_before"),
                "grad_scaler_scale_after": row.get("grad_scaler_scale_after"),
                "nonfinite_param_count": row.get("nonfinite_param_count"),
                "nonfinite_param_names": row.get("nonfinite_param_names"),
                "logit_min": row.get("logit_min"),
                "logit_max": row.get("logit_max"),
                "target_min": row.get("target_min"),
                "target_max": row.get("target_max"),
                "reasons": reasons,
            })
    return {
        "stage": "training_stability_diagnosis",
        "status": "passed" if not failure_reasons else "failed",
        "failure_reasons": failure_reasons,
        "total_steps": total,
        "nonfinite_gradient_steps": len(nonfinite_gradient_rows),
        "nonfinite_gradient_step_rate": nonfinite_rate,
        "nonfinite_gradient_step_rate_threshold": threshold,
        "loss_component_finite_steps": finite_component_count,
        "loss_component_finite_rate": component_finite_rate,
        "loss_component_finite_rate_required": 1.0,
        "optimizer_steps_performed": sum(1 for row in loss_history if bool(row.get("optimizer_step_performed"))),
        "optimizer_steps_skipped_nonfinite_grad": sum(
            1 for row in loss_history
            if bool(row.get("optimizer_step_skipped_nonfinite_grad"))
        ),
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostic_examples": diagnostic_rows[:20],
    }, diagnostic_rows


def diagnose_loss_curve(loss_history: list[dict[str, Any]]) -> dict[str, Any]:
    finite_losses = [
        float(row["loss"]) for row in loss_history
        if _is_finite_number(row.get("loss"))
    ]
    if not finite_losses:
        status = "insufficient_steps"
        recommendation = "No finite training loss was recorded; inspect trainer invocation and input contracts before advancing EM."
        first_mean = last_mean = None
    else:
        window = max(1, len(finite_losses) // 5)
        first_mean = float(np.mean(finite_losses[:window]))
        last_mean = float(np.mean(finite_losses[-window:]))
        if len(finite_losses) < 5:
            status = "insufficient_steps"
            recommendation = "Run more M-step iterations before judging convergence."
        elif last_mean <= first_mean * 0.98:
            status = "decreasing"
            recommendation = "Loss is decreasing; keep this setting if stability, alignment, and trainset pseudo-consistency also pass."
        elif last_mean <= first_mean * 1.02:
            status = "stable"
            recommendation = "Loss is stable but not clearly improving; inspect pseudo-consistency and consider scheduler or LR sweep."
        else:
            status = "increasing"
            recommendation = "Loss increased; first check prompt/mask/CT alignment, then retry a lower learning rate."
    return {
        "stage": "loss_curve_diagnosis",
        "status": status,
        "num_finite_losses": len(finite_losses),
        "first_loss_window_mean": first_mean,
        "last_loss_window_mean": last_mean,
        "negative_loss_interpretation": "Allowed when nnU-Net Dice component is negative; BCE/Dice component finiteness is audited separately.",
        "recommendation": recommendation,
    }


def write_loss_curve_artifacts(output_dir: Path, loss_history: list[dict[str, Any]]) -> None:
    fields = [
        "step", "loss", "task_loss", "bce_component", "dice_component",
        "retention_loss", "learning_rate", "grad_norm_before_clip",
        "finite_grad_norm", "loss_component_finite", "optimizer_step_performed",
        "optimizer_step_skipped_nonfinite_grad", "amp_enabled",
        "grad_scaler_scale_before", "grad_scaler_scale_after",
        "logit_min", "logit_max", "target_min", "target_max",
        "case_id", "organ", "prompt", "sample_kind", "positive_organs",
        "negative_organ", "target_type", "negative_source",
        "negative_source_class", "fov_status", "foreground_voxel_ratio",
        "all_zero_target", "batch_unit_count",
    ]
    path = output_dir / "student_training_loss_curve.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in loss_history:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(row.get("step") or i + 1) for i, row in enumerate(loss_history)]
        values = [
            float(row.get("loss")) if _is_finite_number(row.get("loss")) else float("nan")
            for row in loss_history
        ]
        bce_values = [
            float(row.get("bce_component")) if _is_finite_number(row.get("bce_component")) else float("nan")
            for row in loss_history
        ]
        dice_values = [
            float(row.get("dice_component")) if _is_finite_number(row.get("dice_component")) else float("nan")
            for row in loss_history
        ]
        plt.figure(figsize=(9, 4))
        plt.plot(steps, values, linewidth=1.2, label="total")
        plt.plot(steps, bce_values, linewidth=0.9, label="bce")
        plt.plot(steps, dice_values, linewidth=0.9, label="dice")
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "student_training_loss_curve.png", dpi=160)
        plt.close()
    except Exception as exc:
        write_json(output_dir / "student_training_loss_curve_plot_status.json", {
            "status": "skipped_plot_error",
            "error": str(exc),
        })


def build_training_sample_distribution_audit(
    *,
    loss_history: list[dict[str, Any]],
    sampling_history: list[dict[str, Any]],
    pos_neg_ratio: tuple[int, int],
) -> dict[str, Any]:
    window_positive = sum(int(row.get("batch_positive_count") or 0) for row in sampling_history)
    window_negative = sum(int(row.get("batch_negative_count") or 0) for row in sampling_history)
    positive_units = window_positive or sum(len(row.get("positive_organs") or []) for row in loss_history)
    negative_units = window_negative or sum(1 for row in loss_history if row.get("negative_organ"))
    bad_negative_sources = [
        row for row in loss_history
        if row.get("negative_organ")
        and str(row.get("negative_source") or row.get("negative_source_class") or "") in {"", "teacher_missing", "missing_teacher_output"}
    ]
    positive_organ_counts: Counter[str] = Counter()
    negative_organ_counts: Counter[str] = Counter()
    for row in loss_history:
        positive_organ_counts.update(str(x) for x in (row.get("positive_organs") or []) if str(x))
        if row.get("negative_organ"):
            negative_organ_counts.update([str(row["negative_organ"])])
    return {
        "stage": "training_sample_distribution_audit",
        "status": "passed" if loss_history and not bad_negative_sources and positive_units > 0 else "failed",
        "configured_pos_neg_ratio": f"{pos_neg_ratio[0]}:{pos_neg_ratio[1]}",
        "sampled_positive_prompt_units": positive_units,
        "sampled_negative_prompt_units": negative_units,
        "actual_pos_neg_ratio": (float(positive_units) / float(negative_units)) if negative_units else None,
        "positive_organ_exposure": dict(positive_organ_counts),
        "negative_organ_exposure": dict(negative_organ_counts),
        "sampling_history_windows": len(sampling_history),
        "all_zero_target_count": sum(int(row.get("all_zero_target") or 0) for row in loss_history),
        "mean_runtime_foreground_voxel_ratio": (
            float(sum(float(row.get("foreground_voxel_ratio") or 0.0) for row in loss_history) / len(loss_history))
            if loss_history else None
        ),
        "negative_source_policy": "Negatives must be confirmed absent/out-of-FOV zero masks or legal derived empty-crop negatives, not teacher-missing positives.",
        "bad_negative_source_examples": bad_negative_sources[:20],
    }


def _nonfinite_gradient_param_names(module: nn.Module, limit: int = 20) -> tuple[int, list[str]]:
    names: list[str] = []
    count = 0
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        try:
            if not bool(torch.isfinite(grad).all().item()):
                count += 1
                if len(names) < limit:
                    names.append(name)
        except Exception:
            count += 1
            if len(names) < limit:
                names.append(name)
    return count, names


TEXT_ENCODER_POLICY = {
    "text_encoder": "Qwen/Qwen3-Embedding-4B by default, or --text-encoding-model override",
    "trainability": "frozen",
    "implementation": "prompt embeddings are precomputed under torch.inference_mode(); Qwen parameters are set requires_grad=False and are not passed to the optimizer",
    "token_flow": "tokenize wrapped instruction/query text -> Qwen hidden states -> last_token_pool over last_hidden_state -> pooled embedding",
    "embedding_shape": "training stores one tensor per prompt with shape (1, 1, 2560) for the current Qwen3-Embedding-4B setup",
    "fusion": "VoxTell projects pooled text embeddings to query_dim and uses them as transformer decoder queries with cross-attention over projected CT bottleneck features; resulting mask embeddings condition the U-Net decoder via einsum fusion at multiple scales",
}


def _text_encoder_cache_meta(prompts: list[str], text_model_name: str) -> dict[str, Any]:
    prompt_hash = hashlib.sha1(json.dumps(sorted(prompts), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "format_version": 2,
        "text_model_name": str(text_model_name),
        "num_prompts": len(prompts),
        "prompt_hash": prompt_hash,
        "policy": TEXT_ENCODER_POLICY,
    }


def write_voxtell_model_dir(
    source_model_dir: Path,
    output_dir: Path,
    network: nn.Module,
    step: int,
    manifest_path: Path,
    *,
    rank0_write: bool = True,
) -> Path:
    """Write an inference-compatible VoxTell model directory."""
    model_out = output_dir / "voxtell_finetuned_model"
    fold_out = model_out / "fold_0"
    if not rank0_write:
        return model_out
    fold_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_dir / "plans.json", model_out / "plans.json")
    info_path = source_model_dir / "fold_0" / "INFO.txt"
    if info_path.exists():
        shutil.copy2(info_path, fold_out / "INFO.txt")
    torch.save(
        {
            "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": model_state_dict(network),
            "source_model_dir": str(source_model_dir),
            "manifest": str(manifest_path),
            "step": step,
        },
        fold_out / "checkpoint_final.pth",
    )
    return model_out


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _legacy_prompt_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    if "prompt_variant_index" in item or "canonical_prompt" in item:
        return [item]
    canonical = str(item.get("prompt") or "").strip()
    allow_variants = os.getenv("MEDAI_ALLOW_EXPERIMENTAL_PROMPT_VARIANTS", "0").strip().lower() in {"1", "true", "yes"}
    variants = (
        [str(p).strip() for p in item.get("prompt_variants", []) if str(p).strip()]
        if allow_variants else ([canonical] if canonical else [])
    )
    if canonical and canonical not in variants:
        variants.insert(0, canonical)
    if not variants and canonical:
        variants = [canonical]
    rows: list[dict[str, Any]] = []
    configured = set(variants[1:])
    for idx, prompt in enumerate(variants):
        row = dict(item)
        row["prompt"] = prompt
        row["prompt_text"] = prompt
        row["canonical_prompt"] = canonical or prompt
        row["prompt_variant_index"] = idx
        row["is_prompt_variant"] = idx > 0
        row["prompt_source"] = "canonical" if idx == 0 else ("configured_variant" if prompt in configured else "template_variant")
        row["prompt_family_id"] = f"{row.get('case_id')}:{row.get('organ')}:{row.get('supervision_type', 'positive')}"
        row.setdefault("negative_source", None)
        rows.append(row)
    return rows


def compute_sampling_weight(item: dict[str, Any]) -> float:
    base = max(0.0, float(item.get("training_weight", 1.0) or 0.0))
    grade = str(item.get("grade") or item.get("student_training_priority") or "C").upper()
    grade_mult = {"A": 1.6, "B": 1.3, "C": 1.0, "D": 0.55}.get(grade, 1.0)
    priority = str(item.get("student_training_priority") or "").lower()
    if priority in {"a", "high", "strong"}:
        grade_mult = max(grade_mult, 1.6)
    elif priority in {"b", "medium"}:
        grade_mult = max(grade_mult, 1.3)
    elif priority == "negative":
        grade_mult = min(grade_mult, 0.8)

    route_conf = _as_float(item.get("route_confidence"), None)
    label_conf = _as_float(item.get("label_confidence") or item.get("auto_fine_label_reliability_score"), None)
    confidence_mult = 1.0
    for conf in (route_conf, label_conf):
        if conf is not None:
            confidence_mult *= 0.75 + 0.5 * max(0.0, min(1.0, conf))

    lineage = item.get("teacher_lineage") or item.get("candidate_models") or []
    lineage_count = len(lineage) if isinstance(lineage, list) else 1
    lineage_mult = min(1.25, 1.0 + 0.05 * max(0, lineage_count - 1))

    flags = {str(x) for x in (item.get("review_flags") or [])} | {str(x) for x in (item.get("quality_flags") or [])}
    quality_mult = 0.7 if flags & {"missing_candidate", "missing_final_mask", "selection_fallback"} else 1.0
    if item.get("supervision_type") == "negative":
        quality_mult *= 0.75
    if item.get("is_prompt_variant"):
        quality_mult *= 0.9
    return max(0.0, base * grade_mult * confidence_mult * lineage_mult * quality_mult)


def canonical_negative_reason(item: dict[str, Any], default: str = "absent_in_crop") -> str:
    raw_reason = str(item.get("negative_reason") or "")
    if raw_reason in {"absent_in_scan", "absent_in_crop", "wrong_prompt"}:
        return raw_reason
    source = str(item.get("negative_source") or "")
    return NEGATIVE_REASON_MAP.get(source, default)


def is_allowed_negative_item(item: dict[str, Any]) -> bool:
    if item.get("supervision_type") != "negative":
        return True
    source = str(item.get("negative_source") or "")
    if source not in SAFE_NEGATIVE_SOURCES:
        return False
    if item.get("zero_mask_role") not in SAFE_ZERO_MASK_ROLES:
        return False
    if source in {"case_373_expected_absent", "nonmedical_absent_object"}:
        return True
    if str(item.get("target_type") or "") in {"absent_negative", "negative_absent"}:
        return True
    if not item.get("negative_evidence"):
        return False
    return True


def is_allowed_positive_item(item: dict[str, Any]) -> bool:
    if item.get("supervision_type", "positive") != "positive":
        return True
    if str(item.get("scoring_schema_version") or "legacy") not in ACCEPTED_AUTOLABEL_SCHEMAS:
        return False
    grade = str(item.get("grade") or "D").upper()
    target_type = str(item.get("target_type") or "hard").lower()
    target_type = {"positive_hard": "hard", "positive_soft": "soft"}.get(target_type, target_type)
    if grade in {"A", "B"}:
        return target_type == "hard"
    if grade == "C":
        probability_path = item.get("probability_mask_path") or item.get("mask")
        return target_type == "soft" and bool(probability_path) and Path(str(probability_path)).exists()
    return False


def load_manifest(
    path: Path,
    max_items: int = 0,
    training_profile: str = QUALITY_WEIGHTED_PROFILE,
) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for raw in doc.get("items", []):
        for item in _legacy_prompt_rows(raw):
            image = item.get("image")
            mask = item.get("mask")
            prompt = item.get("prompt")
            training_weight = float(item.get("training_weight", 1.0) or 0.0)
            if not prompt:
                raise ValueError("Prompt-level manifest item missing prompt. This breaks prompt-conditioned Student training.")
            if not image or not mask:
                continue
            if training_weight <= 0.0:
                continue
            if not is_allowed_positive_item(item):
                continue
            if not is_allowed_negative_item(item):
                continue
            if training_profile == PAPER_ALIGNED_PROFILE:
                if item.get("supervision_type", "positive") == "positive":
                    if str(item.get("grade") or "D").upper() not in {"A", "B"}:
                        continue
                    if str(item.get("target_type") or "hard").lower() not in {"hard", "positive_hard"}:
                        continue
                else:
                    if str(item.get("target_type") or "") not in {"absent_negative", "negative_absent"}:
                        continue
                    if str(item.get("negative_source") or "") == "nonmedical_absent_object":
                        continue
            if Path(image).exists() and Path(mask).exists():
                sampling_weight = compute_sampling_weight({**item, "training_weight": training_weight})
                normalized = {**item, "training_weight": training_weight}
                normalized["sampling_weight"] = round(float(sampling_weight), 6)
                normalized["effective_loss_weight"] = round(float(sampling_weight), 6)
                normalized["sampling_repeat"] = 1
                normalized.setdefault("supervision_type", "positive")
                normalized.setdefault("distillation_role", normalized["supervision_type"])
                if normalized.get("supervision_type") == "negative":
                    normalized["negative_reason"] = canonical_negative_reason(normalized)
                rows.append(dict(normalized))
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def parse_pos_neg_ratio(value: str | float | int) -> tuple[int, int]:
    text = str(value).strip()
    if not text:
        raise ValueError("pos_neg_ratio must not be empty")
    if ":" in text:
        left, right = text.split(":", 1)
        pos = float(left.strip())
        neg = float(right.strip())
    else:
        pos = float(text)
        neg = 1.0
    if pos <= 0 or neg < 0:
        raise ValueError(f"Invalid positive:negative ratio: {value}")
    if neg == 0:
        return (max(1, int(round(pos))), 0)
    scale = 10
    pos_i = max(1, int(round(pos * scale)))
    neg_i = max(1, int(round(neg * scale)))
    import math
    div = math.gcd(pos_i, neg_i)
    return pos_i // div, neg_i // div


def build_sample_pools(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in rows if r.get("supervision_type", "positive") == "positive"]
    manifest_negative = [r for r in rows if r.get("supervision_type") == "negative"]
    derived_weight = float(os.getenv("MEDAI_DERIVED_NEGATIVE_LOSS_WEIGHT", "0.1"))
    derived_crop_negative = [
        dict(
            r,
            supervision_type="negative",
            negative_reason="absent_in_crop",
            derived_negative_from_positive=True,
            training_weight=derived_weight,
            sampling_weight=1.0,
            effective_loss_weight=derived_weight,
        )
        for r in positive
        if can_derive_crop_negative(r)
    ]
    positive_by_organ: dict[str, list[dict[str, Any]]] = {}
    for row in positive:
        positive_by_organ.setdefault(str(row.get("organ") or ""), []).append(row)
    protected = {
        item.strip()
        for item in os.getenv("MEDAI_PROTECTED_ORGANS", "").split(",")
        if item.strip()
    }
    novelty_organs = {
        item.strip()
        for item in os.getenv("MEDAI_NOVELTY_ORGANS", "").split(",")
        if item.strip()
    }
    return {
        "positive": positive,
        "positive_by_organ": positive_by_organ,
        "protected_organs": sorted(protected & set(positive_by_organ)),
        "novelty_organs": sorted(novelty_organs & set(positive_by_organ)),
        "negative": manifest_negative + derived_crop_negative,
        "manifest_negative": manifest_negative,
        "semantic_negative": manifest_negative,
        "derived_crop_negative": derived_crop_negative,
    }


def build_paper_case_pools(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse expanded prompt rows into image-centered prompt families."""
    families: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or "")
        kind = str(row.get("supervision_type") or "positive")
        if not case_id or not organ:
            continue
        key = (case_id, organ, kind)
        family = families.setdefault(key, {
            **row,
            "canonical_prompt": str(row.get("canonical_prompt") or row.get("prompt") or organ.replace("_", " ")),
            "approved_variants": [],
        })
        prompt = str(row.get("prompt") or "").strip()
        if prompt and prompt != family["canonical_prompt"] and prompt not in family["approved_variants"]:
            family["approved_variants"].append(prompt)
    cases: dict[str, dict[str, Any]] = {}
    for (case_id, _organ, kind), family in families.items():
        case = cases.setdefault(case_id, {
            "case_id": case_id, "image": family.get("image"), "positive": [], "negative": [],
        })
        case[kind].append(family)
    return {
        key: case for key, case in cases.items()
        if case.get("image") and len(case["positive"]) >= 2 and len(case["negative"]) >= 1
    }


def build_paper_sampler_contract_audit(
    training_rows: list[dict[str, Any]],
    paper_case_pools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify every trainable positive organ can enter a paper-aligned unit."""
    positive_by_case: dict[str, set[str]] = {}
    negative_by_case: dict[str, int] = {}
    for row in training_rows:
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("organ") or "")
        if not case_id or not organ:
            continue
        if row.get("supervision_type", "positive") == "positive":
            positive_by_case.setdefault(case_id, set()).add(organ)
        elif row.get("supervision_type") == "negative":
            negative_by_case[case_id] = negative_by_case.get(case_id, 0) + 1

    trainable_positive_organs = sorted(
        {organ for organs in positive_by_case.values() for organ in organs}
    )
    sampleable_positive_organs = sorted(
        {
            str(row.get("organ") or "")
            for case in paper_case_pools.values()
            for row in case.get("positive", [])
            if str(row.get("organ") or "")
        }
    )
    unsampleable_organs = sorted(
        set(trainable_positive_organs) - set(sampleable_positive_organs)
    )
    case_rows = []
    for case_id in sorted(set(positive_by_case) | set(negative_by_case)):
        pos_organs = sorted(positive_by_case.get(case_id, set()))
        neg_count = int(negative_by_case.get(case_id, 0))
        eligible = case_id in paper_case_pools
        reasons = []
        if len(pos_organs) < 2:
            reasons.append("fewer_than_two_trainable_positive_organs")
        if neg_count < 1:
            reasons.append("missing_manifest_absent_negative")
        if not reasons:
            reasons.append("ok")
        case_rows.append({
            "case_id": case_id,
            "trainable_positive_organ_count": len(pos_organs),
            "trainable_positive_organs": pos_organs,
            "manifest_negative_count": neg_count,
            "paper_aligned_eligible": eligible,
            "issues": reasons,
        })

    failures = []
    if not paper_case_pools and trainable_positive_organs:
        failures.append("no_paper_aligned_eligible_cases")
    if unsampleable_organs:
        failures.append("trainable_positive_organs_not_sampleable_by_paper_profile")
    return {
        "stage": "paper_aligned_sampler_contract_audit",
        "status": "passed" if not failures else "failed",
        "failure_reasons": failures,
        "policy": (
            "Paper-aligned M-step samples 2 positive prompts and 1 manifest "
            "absent-negative prompt from the same CT. Every trainable positive "
            "organ must appear in at least one eligible training case, otherwise "
            "the training audit would fail only after a long run."
        ),
        "trainable_positive_organs": trainable_positive_organs,
        "sampleable_positive_organs": sampleable_positive_organs,
        "unsampleable_positive_organs": unsampleable_organs,
        "eligible_case_count": len(paper_case_pools),
        "case_rows": case_rows,
    }


def deterministic_case_split(case_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(set(case_ids))
    if len(ordered) <= 1:
        return {"training": ordered, "validation": []}
    validation = [
        case_id for case_id in ordered
        if int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16) % 10 == 0
    ]
    if len(ordered) > 1 and not validation:
        validation = [ordered[-1]]
    validation_set = set(validation)
    training = [case_id for case_id in ordered if case_id not in validation_set]
    return {"training": training, "validation": validation}


def select_family_prompt(family: dict[str, Any]) -> tuple[str, str]:
    canonical = str(family["canonical_prompt"])
    variants = [str(x) for x in family.get("approved_variants", []) if str(x).strip()]
    if not variants or random.random() < 0.25:
        return canonical, "canonical"
    return random.choice(variants), "rewrite"


def load_official_embedding_bank(path: Path | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing", "path": str(path) if path else None}
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        embeddings = data["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[1] != 2560:
        raise ValueError(f"Unexpected official embedding bank shape: {embeddings.shape}")
    bank = {
        str(label).lower(): torch.from_numpy(np.asarray(embeddings[i], dtype=np.float32)).view(1, 1, -1)
        for i, label in enumerate(labels)
    }
    return bank, {
        "status": "loaded",
        "path": str(path),
        "num_prompts": len(bank),
        "embedding_shape": list(embeddings.shape),
        "sha256": sha256_file(path),
    }


def can_derive_crop_negative(item: dict[str, Any]) -> bool:
    """Whether it is safe to synthesize an absent-in-crop negative from a positive mask.

    Global/container targets such as ``abdominal_cavity`` can occupy nearly every
    valid crop.  Treating them as derived crop-negative sources causes the
    runtime sampler to repeatedly fail while searching for an empty crop.  Whole
    scan absent negatives should still enter through manifest negative rows.
    """
    organ = str(item.get("organ") or item.get("canonical_organ") or "").strip().lower()
    if organ in DERIVED_CROP_NEGATIVE_EXACT_EXCLUSIONS:
        return False
    return not any(token in organ for token in DERIVED_CROP_NEGATIVE_KEYWORD_EXCLUSIONS)


def _weighted_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose uniformly; quality is applied exactly once in the task loss."""
    if not rows:
        raise ValueError("Cannot sample from an empty pool")
    return dict(random.choice(rows))


def _balanced_positive_choice(
    step: int,
    pools: dict[str, Any],
    pos_neg_ratio: tuple[int, int],
) -> dict[str, Any]:
    by_organ = pools.get("positive_by_organ") or {}
    if not by_organ:
        return _weighted_choice(pools["positive"])
    all_organs = sorted(organ for organ, rows in by_organ.items() if organ and rows)
    protected = [organ for organ in pools.get("protected_organs", []) if organ in by_organ]
    novelty = [organ for organ in pools.get("novelty_organs", []) if organ in by_organ]
    positive_ordinal = sum(
        choose_sample_kind(previous, pos_neg_ratio, pools) == "positive"
        for previous in range(step + 1)
    ) - 1
    # Even positive samples protect configured anchor organs; odd samples
    # balance the complete target space.
    if protected and novelty:
        # Preserve the 50% protected/all policy while assigning half of the
        # all-organ half to organs carrying materially changed evidence.
        selector = positive_ordinal % 4
        organ_pool = protected if selector in {0, 2} else (
            novelty if selector == 1 else all_organs
        )
    else:
        organ_pool = protected if protected and positive_ordinal % 2 == 0 else all_organs
    organ = organ_pool[(positive_ordinal // (4 if novelty else (2 if protected else 1))) % len(organ_pool)]
    return dict(random.choice(by_organ[organ]))


def choose_sample_kind(step: int, pos_neg_ratio: tuple[int, int], pools: dict[str, list[dict[str, Any]]]) -> str:
    pos_n, neg_n = pos_neg_ratio
    if not pools.get("negative") or neg_n <= 0:
        return "positive"
    if not pools.get("positive"):
        return "negative"
    cycle = ["positive"] * pos_n + ["negative"] * neg_n
    return cycle[step % len(cycle)]


def sample_training_item(
    step: int,
    pools: dict[str, list[dict[str, Any]]],
    pos_neg_ratio: tuple[int, int],
    absent_derived_ratio: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    kind = choose_sample_kind(step, pos_neg_ratio, pools)
    negative_source_class = None
    if kind == "negative":
        absent_n, derived_n = absent_derived_ratio
        source_cycle = (
            ["semantic_negative"] * absent_n
            + ["derived_crop_negative"] * derived_n
        )
        negative_ordinal = sum(
            choose_sample_kind(previous, pos_neg_ratio, pools) == "negative"
            for previous in range(step + 1)
        ) - 1
        preferred = source_cycle[negative_ordinal % len(source_cycle)] if source_cycle else "semantic_negative"
        fallback = "derived_crop_negative" if preferred == "semantic_negative" else "semantic_negative"
        selected_pool = pools.get(preferred) or pools.get(fallback) or pools["negative"]
        negative_source_class = preferred if pools.get(preferred) else fallback
        item = _weighted_choice(selected_pool)
    else:
        item = _balanced_positive_choice(step, pools, pos_neg_ratio)
    item["runtime_sample_kind"] = kind
    if kind == "negative":
        item["negative_reason"] = canonical_negative_reason(item, "absent_in_crop")
        item["negative_source_class"] = negative_source_class
    return item


def build_voxtell_network(model_dir: Path, deep_supervision: bool = False) -> nn.Module:
    plans = load_json(join(str(model_dir), "plans.json"))
    arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    arch_kwargs = dict(**arch_kwargs)
    for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    network = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=deep_supervision,
    )
    checkpoint = torch.load(model_dir / "fold_0" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    state = checkpoint.get("network_weights", checkpoint)
    if not isinstance(network, OptimizedModule):
        network.load_state_dict(state, strict=True)
    else:
        network._orig_mod.load_state_dict(state, strict=True)
    return network


def load_patch_size(model_dir: Path) -> tuple[int, int, int]:
    plans = load_json(join(str(model_dir), "plans.json"))
    return tuple(int(x) for x in plans["configurations"]["3d_fullres"]["patch_size"])


def read_image(path: Path) -> np.ndarray:
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        arr, _ = NibabelIOWithReorient().read_images([str(path)])
        return arr.astype(np.float32)
    except Exception:
        import nibabel as nib
        arr = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
        return arr[None] if arr.ndim == 3 else arr


def read_mask(path: Path, preserve_probabilities: bool = False) -> np.ndarray:
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        arr, _ = NibabelIOWithReorient().read_images([str(path)])
        arr = arr[0] if arr.ndim == 4 else arr
    except Exception:
        import nibabel as nib
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    arr = np.asarray(arr, dtype=np.float32)
    if preserve_probabilities:
        if not np.isfinite(arr).all() or float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
            raise ValueError(f"Soft target must contain finite probabilities in [0,1]: {path}")
        return arr
    return (arr > 0).astype(np.float32)


def bbox_to_slices(bbox: list[list[int]] | tuple[tuple[int, int], ...]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(b[0]), int(b[1])) for b in bbox)  # type: ignore[return-value]


def pad_to_shape(image: np.ndarray, mask: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    spatial = image.shape[1:]
    pad_width = [(0, 0)]
    mask_pad = []
    for dim, target in zip(spatial, patch_size):
        needed = max(0, target - int(dim))
        before = needed // 2
        after = needed - before
        pad_width.append((before, after))
        mask_pad.append((before, after))
    if any(a or b for a, b in pad_width[1:]):
        image = np.pad(image, pad_width, mode="constant")
        mask = np.pad(mask, mask_pad, mode="constant")
    return image, mask


def choose_patch_start(
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_prob: float,
    foreground_flat: np.ndarray | None = None,
) -> tuple[int, int, int]:
    shape = mask.shape
    max_start = [max(0, int(s) - int(p)) for s, p in zip(shape, patch_size)]
    foreground_flat = foreground_flat if foreground_flat is not None else np.flatnonzero(mask)
    use_fg = random.random() < foreground_prob and foreground_flat.size > 0
    if use_fg:
        center = np.unravel_index(int(foreground_flat[random.randrange(foreground_flat.size)]), mask.shape)
        start = []
        for c, p, m in zip(center, patch_size, max_start):
            lo = max(0, int(c) - int(p) // 2)
            start.append(min(lo, m))
        return tuple(start)  # type: ignore[return-value]
    return tuple(random.randint(0, m) if m > 0 else 0 for m in max_start)  # type: ignore[return-value]


def patch_slices(start: tuple[int, int, int], patch_size: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(s), int(s) + int(p)) for s, p in zip(start, patch_size))  # type: ignore[return-value]


def choose_negative_patch_start(mask: np.ndarray, patch_size: tuple[int, int, int], max_attempts: int = 64) -> tuple[int, int, int]:
    max_start = [max(0, int(s) - int(p)) for s, p in zip(mask.shape, patch_size)]
    if np.flatnonzero(mask).size == 0:
        return tuple(random.randint(0, m) if m > 0 else 0 for m in max_start)  # type: ignore[return-value]
    for _ in range(max_attempts):
        start = tuple(random.randint(0, m) if m > 0 else 0 for m in max_start)  # type: ignore[assignment]
        if float(mask[patch_slices(start, patch_size)].sum()) <= 0.0:
            return start  # type: ignore[return-value]
    raise ValueError("Could not sample a target-absent negative crop for this organ")


@lru_cache(maxsize=max(1, int(os.getenv("MEDAI_IMAGE_CACHE_SIZE", "16"))))
def _cached_preprocessed_image(path: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[int, ...]]:
    image = read_image(Path(path))
    original_shape = tuple(int(x) for x in image.shape[1:])
    image, _, bbox = crop_to_nonzero(image, None)
    image = ZScoreNormalization(intensityproperties={}).run(image, None)
    bbox_key = tuple((int(pair[0]), int(pair[1])) for pair in bbox)
    return image, bbox_key, original_shape


@lru_cache(maxsize=max(1, int(os.getenv("MEDAI_MASK_CACHE_SIZE", "64"))))
def _cached_cropped_mask(path: str, bbox: tuple[tuple[int, int], ...], original_shape: tuple[int, ...], target_type: str = "hard") -> tuple[np.ndarray, np.ndarray]:
    mask = read_mask(Path(path), preserve_probabilities=target_type == "soft")
    if tuple(mask.shape) != tuple(original_shape):
        raise ValueError(f"Image/mask shape mismatch: {original_shape} vs {mask.shape}")
    mask = mask[bbox_to_slices(bbox)]
    return mask, np.flatnonzero(mask)


def load_training_patch_with_metadata(
    item: dict[str, Any],
    patch_size: tuple[int, int, int],
    foreground_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image, bbox, original_shape = _cached_preprocessed_image(str(Path(item["image"]).resolve()))
    target_type = {
        "positive_hard": "hard",
        "positive_soft": "soft",
    }.get(str(item.get("target_type") or "hard"), str(item.get("target_type") or "hard"))
    mask, foreground_flat = _cached_cropped_mask(str(Path(item["mask"]).resolve()), bbox, original_shape, target_type)
    image, mask = pad_to_shape(image, mask, patch_size)
    if tuple(mask.shape) != tuple(_cached_cropped_mask(str(Path(item["mask"]).resolve()), bbox, original_shape, target_type)[0].shape):
        foreground_flat = np.flatnonzero(mask)

    sample_kind = str(item.get("runtime_sample_kind") or item.get("supervision_type") or "positive")
    if sample_kind == "positive":
        if foreground_flat.size <= 0:
            raise ValueError("Positive sample has no target foreground")
        start = choose_patch_start(mask, patch_size, 1.0, foreground_flat)
    elif item.get("derived_negative_from_positive"):
        start = choose_negative_patch_start(mask, patch_size)
    else:
        start = choose_patch_start(mask, patch_size, foreground_prob, foreground_flat)

    sx, sy, sz = start
    px, py, pz = patch_size
    image_patch = image[:, sx:sx + px, sy:sy + py, sz:sz + pz]
    mask_patch = mask[sx:sx + px, sy:sy + py, sz:sz + pz]
    if sample_kind == "negative" and item.get("derived_negative_from_positive") and float(mask_patch.sum()) > 0.0:
        raise ValueError("Derived negative crop unexpectedly contains target foreground")

    foreground_voxels = float(mask_patch.sum())
    voxel_count = int(np.prod(mask_patch.shape)) if mask_patch.size else 0
    meta = {
        "case_id": str(item.get("case_id") or ""),
        "organ": str(item.get("organ") or ""),
        "prompt": str(item.get("prompt") or ""),
        "sample_kind": sample_kind,
        "negative_reason": canonical_negative_reason(item) if sample_kind == "negative" else None,
        "negative_source_class": item.get("negative_source_class") if sample_kind == "negative" else None,
        "negative_source": item.get("negative_source") if sample_kind == "negative" else None,
        "target_type": item.get("target_type"),
        "fov_status": item.get("fov_status"),
        "fov_evidence": item.get("fov_evidence", []),
        "foreground_voxel_count": foreground_voxels,
        "foreground_voxel_ratio": foreground_voxels / float(voxel_count) if voxel_count else 0.0,
        "all_zero_target": foreground_voxels <= 0.0,
        "crop_start": [int(sx), int(sy), int(sz)],
    }
    return torch.from_numpy(image_patch[None].astype(np.float32)), torch.from_numpy(mask_patch[None, None].astype(np.float32)), meta


def load_training_patch(item: dict[str, Any], patch_size: tuple[int, int, int], foreground_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    image, target, _meta = load_training_patch_with_metadata(item, patch_size, foreground_prob)
    return image, target


def load_paper_training_unit(
    case: dict[str, Any],
    patch_size: tuple[int, int, int],
    foreground_prob: float,
    preferred_organ: str | None = None,
    max_attempts: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, Any]]:
    """One image patch queried by two present and one volume-absent prompt."""
    preferred = [
        row for row in case["positive"]
        if str(row.get("organ") or "") == str(preferred_organ or "")
    ]
    if preferred:
        first = random.choice(preferred)
        alternatives = [row for row in case["positive"] if row is not first]
        positives = [first, random.choice(alternatives)]
    else:
        positives = random.sample(case["positive"], 2)
    negative = random.choice(case["negative"])
    image, bbox, original_shape = _cached_preprocessed_image(str(Path(case["image"]).resolve()))
    masks = [
        _cached_cropped_mask(str(Path(x["mask"]).resolve()), bbox, original_shape, "hard")[0]
        for x in positives
    ]
    negative_mask = _cached_cropped_mask(
        str(Path(negative["mask"]).resolve()), bbox, original_shape, "hard"
    )[0]
    if any(float(mask.sum()) <= 0 for mask in masks):
        raise ValueError("Paper-aligned positive mask is empty in the volume")
    original_image = image
    image, masks[0] = pad_to_shape(original_image, masks[0], patch_size)
    _, masks[1] = pad_to_shape(original_image, masks[1], patch_size)
    _, negative_mask = pad_to_shape(original_image, negative_mask, patch_size)
    chosen = None
    foreground_selected = random.random() < foreground_prob
    for _ in range(max_attempts):
        if foreground_selected:
            start = choose_patch_start(masks[0], patch_size, 1.0)
        else:
            start = choose_patch_start(masks[0], patch_size, 0.0)
        ps = patch_slices(start, patch_size)
        # The preferred positive prompt must contribute foreground.  Requiring
        # both positives in the same 192^3 patch over-penalizes distant organ
        # pairs and starves the organ-balanced sampler.
        if float(masks[0][ps].sum()) > 0:
            chosen = ps
            break
    if chosen is None:
        raise ValueError("Could not find one 192^3 patch containing the preferred positive target")
    if float(negative_mask.sum()) != 0:
        raise ValueError("Volume-absent negative mask is not all-zero")
    prompt_pairs = [select_family_prompt(x) for x in positives] + [select_family_prompt(negative)]
    targets = np.stack([masks[0][chosen], masks[1][chosen], negative_mask[chosen]], axis=0)
    foreground_positive_organs = [
        organ
        for organ, mask in zip([x["organ"] for x in positives], targets[:2], strict=False)
        if float(mask.sum()) > 0.0
    ]
    return (
        torch.from_numpy(np.ascontiguousarray(image[(slice(None), *chosen)])).float(),
        torch.from_numpy(np.ascontiguousarray(targets)).float(),
        [x[0] for x in prompt_pairs],
        {
            "case_id": case["case_id"],
            "sample_kind": "paper_aligned_2_positive_1_negative",
            "positive_organs": [x["organ"] for x in positives],
            "positive_foreground_organs": foreground_positive_organs,
            "negative_organ": negative["organ"],
            "prompt_kinds": [x[1] for x in prompt_pairs],
            "foreground_oversample_selected": foreground_selected,
            "all_zero_negative": True,
        },
    )


@torch.inference_mode()
def build_prompt_embeddings(
    prompts: list[str],
    text_model_name: str,
    device: torch.device,
    cache_path: Path | None,
    official_bank: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    expected_meta = _text_encoder_cache_meta(prompts, text_model_name)
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and "embeddings" in cached:
            cached_meta = cached.get("metadata") or {}
            cached_embeddings = cached.get("embeddings") or {}
            if (
                cached_meta.get("format_version") == expected_meta["format_version"]
                and cached_meta.get("text_model_name") == expected_meta["text_model_name"]
                and cached_meta.get("prompt_hash") == expected_meta["prompt_hash"]
                and all(prompt in cached_embeddings for prompt in prompts)
            ):
                return {prompt: cached_embeddings[prompt].float() for prompt in prompts}
        elif isinstance(cached, dict) and all(prompt in cached for prompt in prompts):
            # Legacy cache format lacked text-model metadata, so only reuse when
            # the caller explicitly accepts old cache files.
            if os.getenv("MEDAI_ALLOW_LEGACY_PROMPT_CACHE", "0").lower() in {"1", "true", "yes"}:
                return {prompt: cached[prompt].float() for prompt in prompts}

    official_bank = official_bank or {}
    out = {
        prompt: official_bank[prompt.lower()].float()
        for prompt in prompts if prompt.lower() in official_bank
    }
    missing = [prompt for prompt in prompts if prompt not in out]
    if not missing:
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"metadata": expected_meta, "embeddings": out}, cache_path)
        return out
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side="left", local_files_only=True)
    text_backbone = AutoModel.from_pretrained(text_model_name, local_files_only=True).eval().to(device)
    text_backbone.requires_grad_(False)
    for prompt in missing:
        wrapped = wrap_with_instruction([prompt])
        tokens = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        encoded = text_backbone(**tokens)
        embedding = last_token_pool(encoded.last_hidden_state, tokens["attention_mask"]).view(1, 1, -1)
        out[prompt] = embedding.detach().cpu().float()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": expected_meta, "embeddings": out}, cache_path)
    del text_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out



def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    reduce_dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=reduce_dims)
    denom = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _resize_target_like(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if tuple(target.shape[2:]) == tuple(pred.shape[2:]):
        return target
    return F.interpolate(target.float(), size=pred.shape[2:], mode="nearest")


def _prediction_tensors(outputs: Any) -> list[torch.Tensor]:
    preds = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    tensors: list[torch.Tensor] = []
    for pred in preds:
        if isinstance(pred, dict):
            pred = pred.get("logits") or pred.get("seg") or pred.get("prediction")
        if isinstance(pred, torch.Tensor):
            tensors.append(pred)
    if not tensors:
        raise ValueError("VoxTell network returned no tensor outputs")
    return tensors


def _deep_supervision_weights(num_outputs: int) -> np.ndarray:
    raw = np.asarray([1 / (2 ** i) for i in range(num_outputs)], dtype=np.float64)
    return raw / raw.sum()


def _tensor_range(value: Any) -> tuple[float | None, float | None]:
    try:
        tensors = _prediction_tensors(value) if not isinstance(value, torch.Tensor) else [value]
        mins = [float(t.detach().float().min().cpu()) for t in tensors]
        maxs = [float(t.detach().float().max().cpu()) for t in tensors]
        return min(mins), max(maxs)
    except Exception:
        return None, None


def _tensor_is_finite(value: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(value.detach()).all().item())
    except Exception:
        return False


def _foreground_balanced_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight_cap: float) -> torch.Tensor:
    positive = target.sum()
    if positive <= 0 or pos_weight_cap <= 1:
        return F.binary_cross_entropy_with_logits(pred, target)
    negative = target.numel() - positive
    pos_weight = torch.clamp(negative / positive.clamp_min(1.0), min=1.0, max=float(pos_weight_cap))
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)


def voxtell_supervision_loss(
    outputs: Any,
    target: torch.Tensor,
    training_weight: float,
    bce_pos_weight_cap: float = 100.0,
) -> torch.Tensor:
    return voxtell_supervision_loss_components(
        outputs, target, training_weight, bce_pos_weight_cap
    )["loss"]


def voxtell_supervision_loss_components(
    outputs: Any,
    target: torch.Tensor,
    training_weight: float,
    bce_pos_weight_cap: float = 100.0,
) -> dict[str, torch.Tensor]:
    preds = _prediction_tensors(outputs)
    total = None
    total_bce = None
    total_dice = None
    total_w = 0.0
    for idx, pred in enumerate(preds):
        stage_target = _resize_target_like(target, pred)
        bce = _foreground_balanced_bce(pred, stage_target, bce_pos_weight_cap)
        dice = dice_loss_with_logits(pred, stage_target)
        stage_w = 0.5 ** idx
        loss = stage_w * (bce + dice)
        total = loss if total is None else total + loss
        total_bce = stage_w * bce if total_bce is None else total_bce + stage_w * bce
        total_dice = stage_w * dice if total_dice is None else total_dice + stage_w * dice
        total_w += stage_w
    if total is None or total_w <= 0:
        raise ValueError("VoxTell network outputs were not tensors")
    weight = float(training_weight or 0.0)
    return {
        "loss": (total / total_w) * weight,
        "bce_component": (total_bce / total_w) * weight,
        "dice_component": (total_dice / total_w) * weight,
    }


def official_retention_loss(student_outputs: Any, official_outputs: Any) -> torch.Tensor:
    """Foreground-aware functional retention against the previous checkpoint."""
    students = list(student_outputs) if isinstance(student_outputs, (list, tuple)) else [student_outputs]
    officials = list(official_outputs) if isinstance(official_outputs, (list, tuple)) else [official_outputs]
    if len(students) != len(officials):
        raise ValueError(
            f"Retention output-stage mismatch: student={len(students)}, official={len(officials)}"
        )
    total = None
    total_w = 0.0
    for idx, (student, official) in enumerate(zip(students, officials)):
        if not isinstance(student, torch.Tensor) or not isinstance(official, torch.Tensor):
            raise ValueError("Retention outputs must be tensors")
        if tuple(student.shape) != tuple(official.shape):
            raise ValueError(
                f"Retention output-shape mismatch at stage {idx}: "
                f"student={tuple(student.shape)}, official={tuple(official.shape)}"
            )
        stage_w = 0.5 ** idx
        teacher_probability = torch.sigmoid(official.detach().float())
        student_probability = torch.sigmoid(student.float())
        foreground_weight = 1.0 + 4.0 * teacher_probability
        probability_match = (
            foreground_weight
            * (student_probability - teacher_probability).square()
        ).mean()
        # Bernoulli KL has zero value and zero gradient at identical logits.
        # Plain BCE has zero gradient there but nonzero entropy, which made the
        # previous "retention contribution" audit look strong while exerting
        # essentially no constraint.
        cross_entropy = F.binary_cross_entropy_with_logits(
            student.float(), teacher_probability
        )
        teacher_entropy = F.binary_cross_entropy_with_logits(
            official.detach().float(), teacher_probability
        )
        calibration_kl = (cross_entropy - teacher_entropy).clamp_min(0.0)
        stage_loss = 0.5 * probability_match + 0.5 * calibration_kl
        total = stage_w * stage_loss if total is None else total + stage_w * stage_loss
        total_w += stage_w
    if total is None or total_w <= 0:
        raise ValueError("Official retention received no output tensors")
    return total / total_w


def foreground_probability_dice(outputs: Any, target: torch.Tensor) -> float:
    """Foreground-aware soft Dice for checkpoint/retention auditing."""
    prediction = list(outputs)[0] if isinstance(outputs, (list, tuple)) else outputs
    if not isinstance(prediction, torch.Tensor):
        raise ValueError("Dice audit requires tensor output")
    resized_target = _resize_target_like(target, prediction).float()
    probability = torch.sigmoid(prediction.detach().float())
    numerator = 2.0 * (probability * resized_target).sum()
    denominator = probability.sum() + resized_target.sum()
    return float(((numerator + 1e-5) / (denominator + 1e-5)).cpu())


def paper_aligned_supervision_loss(outputs: Any, target: torch.Tensor) -> torch.Tensor:
    """Paper A.3: Dice+BCE at all five decoder scales with normalized nnU-Net weights."""
    return paper_aligned_supervision_loss_components(outputs, target)["loss"]


def paper_aligned_supervision_loss_components(outputs: Any, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return total loss plus BCE and negative-Dice components for diagnostics."""
    preds = _prediction_tensors(outputs)
    targets = [_resize_target_like(target, pred) for pred in preds]
    loss = DC_and_BCE_loss(
        {},
        {"batch_dice": False, "do_bg": True, "smooth": 1e-5, "ddp": False},
        weight_ce=1,
        weight_dice=1,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    weights = _deep_supervision_weights(len(preds))
    total = None
    total_bce = None
    total_dice = None
    for i, pred in enumerate(preds):
        stage_target = targets[i].float()
        bce = loss.ce(pred, stage_target)
        dice = loss.dc(pred, stage_target)
        stage_total = float(weights[i]) * (bce + dice)
        total = stage_total if total is None else total + stage_total
        total_bce = float(weights[i]) * bce if total_bce is None else total_bce + float(weights[i]) * bce
        total_dice = float(weights[i]) * dice if total_dice is None else total_dice + float(weights[i]) * dice
    if total is None or total_bce is None or total_dice is None:
        raise ValueError("Paper-aligned loss received no tensor outputs")
    return {
        "loss": total,
        "bce_component": total_bce,
        "dice_component": total_dice,
    }


def build_optimizer(params, args: argparse.Namespace):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.SGD(params, lr=args.learning_rate, momentum=0.99, nesterov=True, weight_decay=args.weight_decay)


def poly_lr(step: int, max_steps: int, base_lr: float, power: float) -> float:
    if max_steps <= 0:
        return base_lr
    progress = min(max(step, 0), max_steps) / float(max_steps)
    return float(base_lr * ((1.0 - progress) ** power))


def set_optimizer_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for group in optim.param_groups:
        group["lr"] = lr

def set_trainable_params(network: nn.Module, freeze_encoder: bool, trainable_scope: str) -> None:
    prompt_prefixes = (
        "project_bottleneck_embed.",
        "project_text_embed.",
        "project_to_decoder_channels.",
        "transformer_decoder.",
    )
    for name, param in network.named_parameters():
        if trainable_scope == "prompt_path":
            param.requires_grad = name.startswith(prompt_prefixes)
        elif freeze_encoder:
            param.requires_grad = not name.startswith("encoder.")


def mask_locality_shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shuffle mask groups while keeping prompt variants cache-local."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("image")), str(row.get("mask")))
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    random.shuffle(keys)
    ordered: list[dict[str, Any]] = []
    for key in keys:
        group = groups[key]
        random.shuffle(group)
        ordered.extend(group)
    return ordered


def main() -> int:
    args = parse_args()
    ddp = init_distributed_if_needed(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.manifest:
        raise SystemExit("--manifest or MEDAI_PROMPT_STUDENT_MANIFEST is required")
    if not args.model_dir:
        raise SystemExit("--model-dir or MEDAI_VOXTELL_MODEL_DIR is required")

    manifest_path = Path(args.manifest).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "voxtell_prompt_train_result.json"

    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal_state_machine = os.getenv("MEDAI_FORMAL_STATE_MACHINE", "0").lower() in {
        "1", "true", "yes",
    }
    run_spec_path = Path(args.run_spec).resolve() if args.run_spec else None
    run_spec_sha256 = (
        contract_sha256_file(run_spec_path)
        if run_spec_path and run_spec_path.is_file()
        else None
    )
    if formal_state_machine:
        if not run_spec_sha256:
            raise SystemExit("Formal M-step requires a readable --run-spec")
        if manifest_document.get("training_contract_version") != TRAINING_CONTRACT_VERSION:
            raise SystemExit(
                "Formal M-step manifest does not carry the current training contract"
            )
        invalid_contract_rows = [
            index
            for index, row in enumerate(manifest_document.get("items") or [])
            if float(row.get("training_weight") or 0.0) > 0.0
            and row.get("distillation_eligible") is not False
            if row.get("contract_version") != TRAINING_CONTRACT_VERSION
            or row.get("training_eligible") is not True
        ]
        if invalid_contract_rows:
            raise SystemExit(
                "Formal M-step contains noncanonical/ineligible manifest rows: "
                + ",".join(map(str, invalid_contract_rows[:20]))
            )

    rows = load_manifest(manifest_path, max_items=args.max_items, training_profile=args.training_profile)
    case_split = deterministic_case_split([str(row.get("case_id") or "") for row in rows if row.get("case_id")])
    training_cases = set(case_split["training"])
    training_rows = [row for row in rows if str(row.get("case_id") or "") in training_cases]
    validation_rows = [row for row in rows if str(row.get("case_id") or "") in set(case_split["validation"])]
    validation_positive_rows = [
        row
        for row in validation_rows
        if row.get("supervision_type", "positive") == "positive"
    ]
    retention_sample_pools = (
        build_sample_pools(validation_positive_rows)
        if validation_positive_rows
        else None
    )
    sample_pools = build_sample_pools(training_rows)
    if args.training_profile == PAPER_ALIGNED_PROFILE:
        # Paper-aligned training never uses derived crop negatives, but keeps
        # the same organ-stratification metadata as every other backend.
        sample_pools["negative"] = list(sample_pools["manifest_negative"])
        sample_pools["derived_crop_negative"] = []
    paper_case_pools = build_paper_case_pools(training_rows) if args.training_profile == PAPER_ALIGNED_PROFILE else {}
    paper_sampler_contract_audit = (
        build_paper_sampler_contract_audit(training_rows, paper_case_pools)
        if args.training_profile == PAPER_ALIGNED_PROFILE
        else {
            "stage": "paper_aligned_sampler_contract_audit",
            "status": "not_applicable",
            "reason": f"training_profile={args.training_profile}",
        }
    )
    write_json(output_dir / "paper_sampler_contract_audit.json", paper_sampler_contract_audit)
    try:
        pos_neg_ratio = parse_pos_neg_ratio(args.pos_neg_ratio)
        pos_neg_ratio_error = None
    except Exception as exc:
        pos_neg_ratio = (2, 1)
        pos_neg_ratio_error = str(exc)
    try:
        absent_derived_ratio = parse_pos_neg_ratio(args.absent_derived_negative_ratio)
        absent_derived_ratio_error = None
    except Exception as exc:
        absent_derived_ratio = (1, 1)
        absent_derived_ratio_error = str(exc)
    model_files_ok = (model_dir / "plans.json").exists() and (model_dir / "fold_0" / "checkpoint_final.pth").exists()
    bank_path = Path(args.official_embedding_bank).resolve() if args.official_embedding_bank else None
    official_bank, official_bank_audit = load_official_embedding_bank(bank_path)
    validation_errors: list[str] = []
    if pos_neg_ratio_error:
        validation_errors.append(f"Invalid --pos-neg-ratio: {pos_neg_ratio_error}")
    if absent_derived_ratio_error:
        validation_errors.append(
            f"Invalid --absent-derived-negative-ratio: {absent_derived_ratio_error}"
        )
    if not rows:
        validation_errors.append("No valid manifest items with existing image/mask/prompt paths")
    if args.training_profile == PAPER_ALIGNED_PROFILE and not paper_case_pools:
        validation_errors.append("No case can form strict 2-positive + 1 volume-absent-negative training units")
    if args.training_profile == PAPER_ALIGNED_PROFILE and paper_sampler_contract_audit.get("status") != "passed":
        missing = ",".join(paper_sampler_contract_audit.get("unsampleable_positive_organs", [])[:20])
        validation_errors.append(
            "Paper-aligned sampler cannot expose all trainable positive organs; "
            f"unsampleable_positive_organs={missing}"
        )
    if args.training_profile == PAPER_ALIGNED_PROFILE and not case_split["training"]:
        validation_errors.append("No training cases remain after deterministic held-out split")
    if args.training_profile == QUALITY_WEIGHTED_PROFILE and not training_rows:
        validation_errors.append("No quality-weighted training rows remain after held-out split")
    if args.training_profile == QUALITY_WEIGHTED_PROFILE and args.batch_size != 1:
        validation_errors.append(
            "quality_weighted_ablation currently supports only --batch-size 1; "
            "refusing to silently ignore the requested batch size"
        )
    if args.official_retention_weight < 0:
        validation_errors.append("--official-retention-weight must be >= 0")
    if args.official_retention_weight > 0 and not validation_positive_rows:
        validation_errors.append(
            "Retention requires positive anchors from the frozen validation split"
        )
    if not model_files_ok:
        validation_errors.append("Missing plans.json or fold_0/checkpoint_final.pth in model_dir")
    if outbound_write_audit()["status"] != "passed":
        validation_errors.append("Outbound project-upload flags are forbidden for formal training")
    if args.training_profile == PAPER_ALIGNED_PROFILE:
        if official_bank_audit.get("status") != "loaded":
            validation_errors.append("paper_aligned requires the published VoxTell embedding bank")
        if args.optimizer != "sgd":
            validation_errors.append("paper_aligned requires SGD")
        if not args.deep_supervision:
            validation_errors.append("paper_aligned requires deep supervision")
        if args.freeze_encoder or args.trainable_scope != "all_decoder":
            validation_errors.append("paper_aligned requires the full VoxTell network to be trainable")
    plan = {
        "stage": "train_voxtell_prompt_student",
        "status": "dry_run" if args.dry_run else "pending",
        "training_mode": "project_voxtell_prompt_distillation_student",
        "legacy_mode": "project_distillation_experimental",
        "canonical_training_backend": "project_voxtell_prompt_distillation_student",
        "trainer": "project_voxtell_prompt_distillation_student",
        "is_official_voxtell_encoder_transfer": False,
        "is_prompt_conditioned_student": True,
        "training_profile": args.training_profile,
        "student_inference_backend": "voxtell_prompt_api",
        "is_project_distillation": True,
        "uses_official_voxtell_model": True,
        "uses_official_checkpoint_initialization": True,
        "uses_project_manifest": True,
        "training_profile": args.training_profile,
        "uses_autolabelcore_confidence": args.training_profile == QUALITY_WEIGHTED_PROFILE,
        "uses_abcd_training_weight": args.training_profile == QUALITY_WEIGHTED_PROFILE,
        "official_prompt_training_pipeline_available": False,
        "negative_prompt_sampling": "per_image_2_positive_1_volume_absent_negative" if args.training_profile == PAPER_ALIGNED_PROFILE else "runtime_pool_sampler",
        "training_provenance_warning": "Project prompt-conditioned distillation trainer using official VoxTell components; not official voxtell-finetune.",
        "formal_state_machine": formal_state_machine,
        "run_spec_path": str(run_spec_path) if run_spec_path else None,
        "run_spec_sha256": run_spec_sha256,
        "training_contract_version": manifest_document.get("training_contract_version"),
        "manifest": str(manifest_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "num_manifest_items": len(rows),
        "num_training_items": len(training_rows),
        "num_validation_items": len(validation_rows),
        "candidate_pool_positive_count": len(sample_pools["positive"]),
        "candidate_pool_negative_count": len(sample_pools["negative"]),
        "candidate_pool_manifest_negative_count": len(sample_pools["manifest_negative"]),
        "candidate_pool_derived_crop_negative_count": len(sample_pools["derived_crop_negative"]),
        "paper_aligned_eligible_cases": len(paper_case_pools),
        "case_split": case_split,
        "pos_neg_ratio": args.pos_neg_ratio,
        "pos_neg_ratio_parsed": list(pos_neg_ratio),
        "absent_derived_negative_ratio": args.absent_derived_negative_ratio,
        "absent_derived_negative_ratio_parsed": list(absent_derived_ratio),
        "sampling_log_interval": args.sampling_log_interval,
        "training_weight_policy": "equal_weight_AB_hard_targets" if args.training_profile == PAPER_ALIGNED_PROFILE else "quality_weighted_candidate_pool",
        "loss_mode": "nnunet_dice_bce_five_scale_deep_supervision" if args.training_profile == PAPER_ALIGNED_PROFILE else "weighted_dice_plus_bce_deep_supervision",
        "optimizer": args.optimizer,
        "poly_power": args.poly_power,
        "deep_supervision": args.deep_supervision,
        "candidate_pool_positive_manifest_items": sum(1 for r in rows if r.get("supervision_type", "positive") == "positive"),
        "candidate_pool_negative_manifest_items": sum(1 for r in rows if r.get("supervision_type") == "negative"),
        "num_prompt_variant_items": sum(1 for r in rows if r.get("is_prompt_variant")),
        "mean_sampling_weight": float(np.mean([r.get("sampling_weight", 0.0) for r in rows])) if rows else 0.0,
        "mean_effective_loss_weight": float(np.mean([r.get("effective_loss_weight", r.get("training_weight", 0.0)) for r in rows])) if rows else 0.0,
        "effective_loss_weight_range": [
            float(np.min([r.get("effective_loss_weight", r.get("training_weight", 0.0)) for r in rows])) if rows else 0.0,
            float(np.max([r.get("effective_loss_weight", r.get("training_weight", 0.0)) for r in rows])) if rows else 0.0,
        ],
        "model_files_ok": model_files_ok,
        "manifest_items_ok": bool(rows),
        "validation_status": "ok" if not validation_errors else "failed",
        "validation_errors": validation_errors,
        "text_encoding_model": args.text_encoding_model,
        "text_encoder_policy": TEXT_ENCODER_POLICY,
        "prompt_embedding_cache": str(Path(args.embedding_cache).resolve()) if args.embedding_cache else str(output_dir / "prompt_embeddings.pt"),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "freeze_encoder": args.freeze_encoder,
        "trainable_scope": args.trainable_scope,
        "bce_pos_weight_cap": args.bce_pos_weight_cap,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size if args.training_profile == PAPER_ALIGNED_PROFILE else 1,
        "official_retention_weight": args.official_retention_weight,
        "grad_clip_norm": args.grad_clip_norm,
        "foreground_oversample_probability": args.foreground_prob,
        "prompt_name_sampling": {"canonical": 0.25, "rewrite": 0.75},
        "official_voxtell_commit_required": OFFICIAL_VOXTELL_COMMIT,
        "outbound_write_audit": outbound_write_audit(),
        "official_embedding_bank": official_bank_audit,
        "paper_sampler_contract_audit": str(output_dir / "paper_sampler_contract_audit.json"),
        "amp_mode": args.amp_mode,
        "ddp": ddp,
        "ddp_contract": {
            "launcher": "torchrun",
            "single_node_only": True,
            "uses_local_rank": True,
            "rank0_checkpoint_writes": True,
            "destroy_process_group_on_exit": True,
        },
    }
    prompt_mapping_audit, prompt_mapping_rows = build_prompt_organ_mapping_audit(rows)
    prompt_mapping_audit["csv"] = str(output_dir / "prompt_organ_mapping_audit.csv")
    write_json(output_dir / "prompt_organ_mapping_audit.json", prompt_mapping_audit)
    write_csv(output_dir / "prompt_organ_mapping_audit.csv", prompt_mapping_rows)
    alignment_audit = build_prompt_mask_ct_alignment_audit(rows, manifest_path)
    alignment_rows = alignment_audit.pop("_rows")
    write_json(output_dir / "prompt_mask_ct_alignment_audit.json", alignment_audit)
    write_csv(output_dir / "prompt_mask_ct_alignment_audit.csv", alignment_rows)
    if prompt_mapping_audit.get("status") != "passed":
        validation_errors.append("Prompt/organ/mask mapping audit failed")
    if alignment_audit.get("status") != "passed":
        validation_errors.append("Prompt-mask-CT alignment audit failed")
    plan.update({
        "validation_status": "ok" if not validation_errors else "failed",
        "validation_errors": validation_errors,
        "prompt_mask_ct_alignment_audit": str(output_dir / "prompt_mask_ct_alignment_audit.json"),
        "prompt_organ_mapping_audit": str(output_dir / "prompt_organ_mapping_audit.json"),
    })
    if args.dry_run:
        write_json(output_dir / "case_split.json", case_split)
        plan["expected_inference_model_dir"] = str(output_dir / "voxtell_finetuned_model")
        plan["expected_inference_contract"] = {
            "plans": str(output_dir / "voxtell_finetuned_model" / "plans.json"),
            "checkpoint": str(output_dir / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth"),
        }
        write_json(output_dir / "voxtell_prompt_train_plan.json", plan)
        write_json(result_path, plan)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if validation_errors:
        plan.update({"status": "failed", "reason": validation_errors[0]})
        write_json(result_path, plan)
        return 2
    if not model_files_ok:
        plan.update({"status": "failed", "reason": "Missing plans.json or fold_0/checkpoint_final.pth in model_dir"})
        write_json(result_path, plan)
        return 2

    write_json(output_dir / "case_split.json", case_split)
    if ddp["enabled"]:
        device = torch.device(f"cuda:{ddp['local_rank']}")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    patch_size = load_patch_size(model_dir)
    prompts = sorted(
        {str(r["prompt"]) for r in training_rows}
        | {str(r["prompt"]) for r in validation_positive_rows}
    )
    cache_path = Path(args.embedding_cache).resolve() if args.embedding_cache else output_dir / "prompt_embeddings.pt"
    started = time.time()
    embeddings = build_prompt_embeddings(prompts, args.text_encoding_model, device, cache_path, official_bank)
    network = build_voxtell_network(model_dir, deep_supervision=args.deep_supervision)
    set_trainable_params(network, args.freeze_encoder, args.trainable_scope)
    network.to(device)
    network.train()
    official_network = None
    if args.official_retention_weight > 0:
        official_network = build_voxtell_network(
            model_dir,
            deep_supervision=args.deep_supervision,
        )
        official_network.requires_grad_(False)
        official_network.to(device)
        official_network.eval()

    network = maybe_wrap_ddp(network, ddp)

    trainable_parameters = [p for p in network.parameters() if p.requires_grad]
    optim = build_optimizer(trainable_parameters, args)
    amp_enabled = device.type == "cuda" and args.amp_mode == "auto"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    losses: list[float] = []
    task_losses: list[float] = []
    retention_losses: list[float] = []
    retention_contribution_ratios: list[float] = []
    retention_gradient_contribution_ratios: list[float] = []
    task_foreground_dices: list[float] = []
    retention_anchor_foreground_dices: list[float] = []
    effective_retention_weights: list[float] = []
    loss_history: list[dict[str, float | int]] = []
    organ_gradient_mass: dict[str, float] = {}
    organ_sample_counts: dict[str, int] = {}
    positive_organ_sample_counts: dict[str, int] = {}
    negative_organ_sample_counts: dict[str, int] = {}
    positive_organ_finite_gradient_mass: dict[str, float] = {}
    positive_organ_nonfinite_gradient_steps: dict[str, int] = {}
    optimizer_steps_skipped_nonfinite_grad = 0
    total_steps = 0
    max_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(training_rows))
    sampling_history: list[dict[str, Any]] = []
    sampling_window: list[dict[str, Any]] = []
    skipped_attempts = 0
    max_attempts = max_steps * 20
    attempts = 0
    retention_task_ema: float | None = None
    retention_loss_ema: float | None = None
    best_checkpoint = output_dir / "checkpoint_best.pth"
    best_task_score = float("-inf")
    best_anchor_score: float | None = None
    retention_anchor_baseline = 1.0
    checkpoint_patience = int(os.getenv("MEDAI_CHECKPOINT_PATIENCE", "3"))
    checkpoint_eval_interval = int(os.getenv("MEDAI_CHECKPOINT_EVAL_INTERVAL", "100"))
    checkpoint_bad_windows = 0
    stopped_early = False
    gradient_calibrated_retention_weight: float | None = None
    paper_cases = list(paper_case_pools.values())
    paper_cases_by_organ: dict[str, list[dict[str, Any]]] = {}
    for case in paper_cases:
        for organ in {
            str(row.get("organ") or "")
            for row in case.get("positive", [])
            if str(row.get("organ") or "")
        }:
            paper_cases_by_organ.setdefault(organ, []).append(case)
    paper_organs = sorted(paper_cases_by_organ)
    protected_paper_organs = [
        organ for organ in sample_pools.get("protected_organs", [])
        if organ in paper_cases_by_organ
    ]
    novelty_paper_organs = [
        organ for organ in sample_pools.get("novelty_organs", [])
        if organ in paper_cases_by_organ
    ]
    while total_steps < max_steps and attempts < max_attempts:
        attempts += 1
        if args.training_profile == PAPER_ALIGNED_PROFILE:
            try:
                units = []
                for batch_index in range(args.batch_size):
                    ordinal = total_steps * args.batch_size + batch_index
                    if protected_paper_organs and novelty_paper_organs:
                        selector = ordinal % 4
                        organ_pool = (
                            protected_paper_organs
                            if selector in {0, 2}
                            else (
                                novelty_paper_organs
                                if selector == 1 else paper_organs
                            )
                        )
                    else:
                        organ_pool = (
                            protected_paper_organs
                            if protected_paper_organs and ordinal % 2 == 0
                            else paper_organs
                        )
                    preferred_organ = organ_pool[ordinal % len(organ_pool)]
                    case = random.choice(paper_cases_by_organ[preferred_organ])
                    units.append(
                        load_paper_training_unit(
                            case,
                            patch_size,
                            args.foreground_prob,
                            preferred_organ=preferred_organ,
                        )
                    )
            except Exception as exc:
                skipped_attempts += 1
                print(f"[warn] skip paper-aligned unit: {exc}", flush=True)
                continue
            image = torch.stack([x[0] for x in units], dim=0).to(device, non_blocking=True)
            target = torch.stack([x[1] for x in units], dim=0).to(device, non_blocking=True)
            text_embedding = torch.cat([
                torch.cat([embeddings[prompt] for prompt in unit[2]], dim=1)
                for unit in units
            ], dim=0).to(device, non_blocking=True)
            sample_meta = {
                "sample_kind": "paper_aligned_2_positive_1_negative",
                "batch_units": [x[3] for x in units],
                "foreground_voxel_ratio": float(target[:, :2].mean().item()),
                "all_zero_target": False,
                "negative_reason": "absent_in_scan",
            }
            item = {"effective_loss_weight": 1.0, "training_weight": 1.0}
        else:
            item = sample_training_item(
                total_steps,
                sample_pools,
                pos_neg_ratio,
                absent_derived_ratio,
            )
            try:
                image, target, sample_meta = load_training_patch_with_metadata(item, patch_size, args.foreground_prob)
            except Exception as exc:
                skipped_attempts += 1
                print(f"[warn] skip {item.get('case_id')} {item.get('organ')}: {exc}", flush=True)
                continue
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            text_embedding = embeddings[str(item["prompt"])].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        set_optimizer_lr(optim, poly_lr(total_steps, max_steps, args.learning_rate, args.poly_power))
        with torch.autocast(device.type, enabled=amp_enabled):
            logits = network(image, text_embedding)
            if args.training_profile == PAPER_ALIGNED_PROFILE:
                task_components = paper_aligned_supervision_loss_components(logits, target)
            else:
                task_components = voxtell_supervision_loss_components(
                    logits,
                    target,
                    float(item.get("effective_loss_weight", item.get("training_weight", 1.0)) or 0.0),
                    args.bce_pos_weight_cap,
                )
            task_loss = task_components["loss"]
            bce_component = task_components["bce_component"]
            dice_component = task_components["dice_component"]
            if official_network is not None:
                # Retention uses an independent, organ-balanced positive anchor
                # rather than merely reusing the current task patch.
                try:
                    if retention_sample_pools is None:
                        raise RuntimeError("retention validation anchor pool is empty")
                    anchor_item = _balanced_positive_choice(
                        total_steps + 1, retention_sample_pools, pos_neg_ratio
                    )
                    anchor_item["runtime_sample_kind"] = "positive"
                    anchor_image, _anchor_target, anchor_meta = load_training_patch_with_metadata(
                        anchor_item, patch_size, args.foreground_prob
                    )
                    anchor_image = anchor_image.to(device, non_blocking=True)
                    anchor_embedding = embeddings[str(anchor_item["prompt"])].to(
                        device, non_blocking=True
                    )
                    student_anchor_logits = network(anchor_image, anchor_embedding)
                except Exception as exc:
                    raise RuntimeError(
                        f"Retention validation anchor failed: {exc}"
                    ) from exc
                with torch.no_grad():
                    official_logits = official_network(anchor_image, anchor_embedding)
                retention_loss = official_retention_loss(
                    student_anchor_logits, official_logits
                )
                retention_anchor_foreground_dices.append(
                    foreground_probability_dice(
                        student_anchor_logits,
                        torch.sigmoid(
                            (
                                list(official_logits)[0]
                                if isinstance(official_logits, (list, tuple))
                                else official_logits
                            ).detach().float()
                        ),
                    )
                )
                task_scalar = float(task_loss.detach().float().cpu())
                retention_scalar = float(retention_loss.detach().float().cpu())
                retention_task_ema = (
                    task_scalar
                    if retention_task_ema is None
                    else 0.95 * retention_task_ema + 0.05 * task_scalar
                )
                retention_loss_ema = (
                    retention_scalar
                    if retention_loss_ema is None
                    else 0.95 * retention_loss_ema + 0.05 * retention_scalar
                )
                effective_retention_weight = min(
                    10.0,
                    max(
                        0.01,
                        float(args.official_retention_weight)
                        * retention_task_ema
                        / max(retention_loss_ema, 1e-8),
                    ),
                )
                if (
                    retention_scalar > 0
                    and (
                        gradient_calibrated_retention_weight is None
                        or (total_steps + 1) % checkpoint_eval_interval == 0
                    )
                ):
                    task_gradients = torch.autograd.grad(
                        task_loss,
                        trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    retention_gradients = torch.autograd.grad(
                        retention_loss,
                        trainable_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    task_gradient_norm = math.sqrt(
                        sum(
                            float(gradient.detach().float().square().sum().cpu())
                            for gradient in task_gradients
                            if gradient is not None
                        )
                    )
                    raw_retention_gradient_norm = math.sqrt(
                        sum(
                            float(gradient.detach().float().square().sum().cpu())
                            for gradient in retention_gradients
                            if gradient is not None
                        )
                    )
                    if task_gradient_norm > 0 and raw_retention_gradient_norm > 0:
                        target_fraction = min(
                            0.30, max(0.15, float(args.official_retention_weight))
                        )
                        gradient_calibrated_retention_weight = min(
                            10.0,
                            max(
                                0.01,
                                target_fraction
                                / (1.0 - target_fraction)
                                * task_gradient_norm
                                / raw_retention_gradient_norm,
                            ),
                        )
                        weighted_retention_norm = (
                            gradient_calibrated_retention_weight
                            * raw_retention_gradient_norm
                        )
                        retention_gradient_contribution_ratios.append(
                            weighted_retention_norm
                            / (task_gradient_norm + weighted_retention_norm)
                        )
                if gradient_calibrated_retention_weight is not None:
                    effective_retention_weight = gradient_calibrated_retention_weight
            else:
                retention_loss = task_loss.new_zeros(())
                effective_retention_weight = 0.0
                anchor_meta = {}
            loss = task_loss + effective_retention_weight * retention_loss
            task_foreground_dices.append(
                foreground_probability_dice(logits, target)
            )
        logit_min, logit_max = _tensor_range(logits)
        target_min, target_max = _tensor_range(target)
        scale_before = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else None
        loss_component_finite = all(
            _tensor_is_finite(component)
            for component in (loss, task_loss, bce_component, dice_component, retention_loss)
        )
        scaled_backward = False
        nonfinite_param_count = 0
        nonfinite_param_names: list[str] = []
        if loss_component_finite:
            scaler.scale(loss).backward()
            scaled_backward = True
            scaler.unscale_(optim)
            nonfinite_param_count, nonfinite_param_names = _nonfinite_gradient_param_names(network)
            if args.grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    (p for p in network.parameters() if p.requires_grad),
                    max_norm=float(args.grad_clip_norm),
                ))
            else:
                grad_norm = 0.0
        else:
            grad_norm = float("nan")
        positive_sampled_organs: list[str] = []
        negative_sampled_organs: list[str] = []
        if sample_meta.get("batch_units"):
            for unit in sample_meta["batch_units"]:
                positive_sampled_organs.extend(
                    str(organ)
                    for organ in (
                        unit.get("positive_foreground_organs")
                        or unit.get("positive_organs", [])
                    )
                    if str(organ)
                )
                if unit.get("negative_organ"):
                    negative_sampled_organs.append(str(unit["negative_organ"]))
        elif sample_meta.get("organ"):
            if sample_meta.get("sample_kind") == "negative":
                negative_sampled_organs = [str(sample_meta["organ"])]
            else:
                positive_sampled_organs = [str(sample_meta["organ"])]
        unique_positive_organs = set(positive_sampled_organs)
        unique_negative_organs = set(negative_sampled_organs)
        finite_grad_norm = _is_finite_number(grad_norm) and nonfinite_param_count == 0
        for organ in unique_positive_organs:
            fraction = 1.0 / max(1, len(unique_positive_organs))
            positive_organ_sample_counts[organ] = positive_organ_sample_counts.get(organ, 0) + 1
            organ_sample_counts[organ] = organ_sample_counts.get(organ, 0) + 1
            if finite_grad_norm:
                positive_organ_finite_gradient_mass[organ] = (
                    positive_organ_finite_gradient_mass.get(organ, 0.0)
                    + float(grad_norm) * fraction
                )
                organ_gradient_mass[organ] = (
                    organ_gradient_mass.get(organ, 0.0) + float(grad_norm) * fraction
                )
            else:
                positive_organ_nonfinite_gradient_steps[organ] = (
                    positive_organ_nonfinite_gradient_steps.get(organ, 0) + 1
                )
        for organ in unique_negative_organs:
            negative_organ_sample_counts[organ] = negative_organ_sample_counts.get(organ, 0) + 1
            organ_sample_counts[organ] = organ_sample_counts.get(organ, 0) + 1
        optimizer_step_performed = bool(loss_component_finite and finite_grad_norm)
        if optimizer_step_performed:
            scaler.step(optim)
        else:
            optimizer_steps_skipped_nonfinite_grad += 1
            optim.zero_grad(set_to_none=True)
        if scaled_backward:
            scaler.update()
        scale_after = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else None

        total_steps += 1
        loss_value = float(loss.detach().cpu())
        task_loss_value = float(task_loss.detach().cpu())
        bce_component_value = float(bce_component.detach().cpu())
        dice_component_value = float(dice_component.detach().cpu())
        retention_loss_value = float(retention_loss.detach().cpu())
        retention_contribution = (
            effective_retention_weight * retention_loss_value
            / max(task_loss_value, 1e-8)
        )
        losses.append(loss_value)
        task_losses.append(task_loss_value)
        retention_losses.append(retention_loss_value)
        effective_retention_weights.append(effective_retention_weight)
        retention_contribution_ratios.append(retention_contribution)
        sampling_window.append(sample_meta)
        loss_history.append({
            "step": total_steps,
            "loss": loss_value,
            "task_loss": task_loss_value,
            "bce_component": bce_component_value,
            "dice_component": dice_component_value,
            "retention_loss": retention_loss_value,
            "grad_norm_before_clip": grad_norm,
            "learning_rate": float(optim.param_groups[0]["lr"]),
            "amp_mode": args.amp_mode,
            "amp_enabled": amp_enabled,
            "amp_scale": scale_before,
            "grad_scaler_scale_before": scale_before,
            "grad_scaler_scale_after": scale_after,
            "loss_component_finite": bool(loss_component_finite),
            "nonfinite_param_count": int(nonfinite_param_count),
            "nonfinite_param_names": nonfinite_param_names,
            "logit_min": logit_min,
            "logit_max": logit_max,
            "target_min": target_min,
            "target_max": target_max,
            "case_id": str(sample_meta.get("case_id") or ""),
            "organ": str(sample_meta.get("organ") or ""),
            "prompt": str(sample_meta.get("prompt") or ""),
            "sample_kind": sample_meta["sample_kind"],
            "target_type": sample_meta.get("target_type"),
            "negative_source": sample_meta.get("negative_source"),
            "negative_source_class": sample_meta.get("negative_source_class"),
            "fov_status": sample_meta.get("fov_status"),
            "fov_evidence": sample_meta.get("fov_evidence", []),
            "foreground_voxel_ratio": float(sample_meta["foreground_voxel_ratio"]),
            "all_zero_target": int(bool(sample_meta["all_zero_target"])),
            "positive_organs": sorted(unique_positive_organs),
            "positive_foreground_organs": sorted(unique_positive_organs),
            "negative_organ": sorted(unique_negative_organs)[0] if unique_negative_organs else None,
            "batch_unit_count": len(sample_meta.get("batch_units", [])),
            "finite_grad_norm": bool(finite_grad_norm),
            "optimizer_step_performed": optimizer_step_performed,
            "optimizer_step_skipped_nonfinite_grad": not optimizer_step_performed,
        })
        log_interval = max(1, int(args.sampling_log_interval))
        if total_steps % log_interval == 0 or total_steps == max_steps:
            recent = [x for x in losses[-log_interval:] if np.isfinite(x)]
            if args.training_profile == PAPER_ALIGNED_PROFILE:
                unit_count = sum(len(m.get("batch_units", [])) for m in sampling_window)
                pos_count, neg_count = 2 * unit_count, unit_count
            else:
                pos_count = sum(1 for m in sampling_window if m.get("sample_kind") == "positive")
                neg_count = sum(1 for m in sampling_window if m.get("sample_kind") == "negative")
            reason_counts: dict[str, int] = {"absent_in_scan": 0, "absent_in_crop": 0, "wrong_prompt": 0}
            source_counts: dict[str, int] = {
                "semantic_negative": 0,
                "derived_crop_negative": 0,
            }
            for m in sampling_window:
                reason = m.get("negative_reason")
                if reason:
                    reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
                source = m.get("negative_source_class")
                if source:
                    source_counts[str(source)] = source_counts.get(str(source), 0) + 1
            positive_organ_window_counts: Counter[str] = Counter()
            for m in sampling_window:
                for unit in m.get("batch_units", []) or []:
                    positive_organ_window_counts.update(
                        str(organ)
                        for organ in (
                            unit.get("positive_foreground_organs")
                            or unit.get("positive_organs", [])
                        )
                        if str(organ)
                    )
                if m.get("organ") and m.get("sample_kind") != "negative":
                    positive_organ_window_counts.update([str(m["organ"])])
            stat = {
                "step": total_steps,
                "loss": float(np.mean(recent)) if recent else None,
                "batch_positive_count": pos_count,
                "batch_negative_count": neg_count,
                "configured_pos_neg_ratio": f"{pos_neg_ratio[0]}:{pos_neg_ratio[1]}",
                "actual_pos_neg_ratio": (float(pos_count) / float(neg_count)) if neg_count else None,
                "positive_negative_ratio": (float(pos_count) / float(neg_count)) if neg_count else None,
                "foreground_voxel_ratio": float(np.mean([m.get("foreground_voxel_ratio", 0.0) for m in sampling_window])) if sampling_window else 0.0,
                "all_zero_target_count": sum(1 for m in sampling_window if m.get("all_zero_target")),
                "negative_reason_counts": reason_counts,
                "negative_source_class_counts": source_counts,
                "positive_organ_window_counts": dict(positive_organ_window_counts),
                "mean_task_loss": float(np.mean(task_losses[-log_interval:])),
                "mean_retention_loss": float(np.mean(retention_losses[-log_interval:])),
            }
            sampling_history.append(stat)
            print(json.dumps(stat, ensure_ascii=False), flush=True)
            sampling_window = []
        if (
            checkpoint_eval_interval > 0
            and total_steps % checkpoint_eval_interval == 0
        ):
            window_task = float(
                np.mean(task_foreground_dices[-checkpoint_eval_interval:])
            )
            window_retention = (
                float(
                    np.mean(
                        retention_anchor_foreground_dices[
                            -checkpoint_eval_interval:
                        ]
                    )
                )
                if retention_anchor_foreground_dices
                else 1.0
            )
            retention_ok = (
                official_network is None
                or window_retention >= retention_anchor_baseline - 0.002
            )
            if retention_ok and window_task > best_task_score:
                best_task_score = window_task
                best_anchor_score = window_retention
                checkpoint_bad_windows = 0
                if ddp["is_rank0"]:
                    torch.save(
                        {
                            "eligible_for_next_round_prompt_student": False,
                            "eligible_as_teacher_candidate": False,
                            "network_weights": model_state_dict(network),
                            "step": total_steps,
                            "selection": {
                                "new_supervision_foreground_dice": window_task,
                                "anchor_foreground_dice": window_retention,
                                "retention_anchor_baseline": retention_anchor_baseline,
                            },
                        },
                        best_checkpoint,
                    )
            else:
                checkpoint_bad_windows += 1
            if checkpoint_bad_windows >= checkpoint_patience:
                stopped_early = True
                break
        if args.save_every > 0 and total_steps % args.save_every == 0:
            # Rotate one recovery checkpoint; 3D checkpoints are ~1.7 GB
            # and retaining every interval can exhaust the experiment disk.
            if ddp["is_rank0"]:
                torch.save({"eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": model_state_dict(network), "step": total_steps}, output_dir / "checkpoint_latest.pth")
    if total_steps < max_steps and not stopped_early:
        raise RuntimeError(f"Runtime sampler produced only {total_steps}/{max_steps} steps after {attempts} attempts")

    best_candidate_found = best_checkpoint.exists()
    if best_candidate_found:
        best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
        load_model_state_dict(network, best_payload["network_weights"], strict=True)
    elif official_network is not None:
        # Fail safe: never publish the last step when no rolling candidate
        # satisfied the anchor constraint.
        load_model_state_dict(network, official_network.state_dict(), strict=True)
    final_ckpt = output_dir / "model_finetune.pth"
    if ddp["is_rank0"]:
        torch.save({
        "training_mode": "project_voxtell_prompt_distillation_student",
        "legacy_mode": "project_distillation_experimental",
        "canonical_training_backend": "project_voxtell_prompt_distillation_student",
        "trainer": "project_voxtell_prompt_distillation_student",
        "is_official_voxtell_encoder_transfer": False,
        "is_prompt_conditioned_student": True,
        "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": model_state_dict(network),
        "optimizer_state": optim.state_dict(),
        "source_model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "step": total_steps,
        "patch_size": patch_size,
    }, final_ckpt)
    inference_model_dir = write_voxtell_model_dir(model_dir, output_dir, network, total_steps, manifest_path, rank0_write=ddp["is_rank0"])
    if ddp["is_rank0"]:
        best_checkpoint.unlink(missing_ok=True)
    finite_losses = [x for x in losses if np.isfinite(x)]
    write_json(output_dir / "loss_history.json", {"steps": total_steps, "history": loss_history, "sampling_history": sampling_history, "skipped_sampling_attempts": skipped_attempts})
    write_loss_curve_artifacts(output_dir, loss_history)
    loss_curve_diagnosis = diagnose_loss_curve(loss_history)
    loss_curve_diagnosis.update({
        "learning_rate": args.learning_rate,
        "scheduler": "poly",
        "optimizer": args.optimizer,
        "amp_mode": args.amp_mode,
    })
    write_json(output_dir / "loss_curve_diagnosis.json", loss_curve_diagnosis)
    training_stability_diagnosis, training_stability_rows = build_training_stability_diagnosis(loss_history)
    write_json(output_dir / "training_stability_diagnosis.json", training_stability_diagnosis)
    write_csv(output_dir / "training_stability_diagnosis.csv", training_stability_rows)
    sample_distribution_audit = build_training_sample_distribution_audit(
        loss_history=loss_history,
        sampling_history=sampling_history,
        pos_neg_ratio=pos_neg_ratio,
    )
    write_json(output_dir / "training_sample_distribution_audit.json", sample_distribution_audit)
    provenance = {
        "statement": "Official VoxTell v1.1 full-model fine-tuning with a paper-aligned prompt-conditioned M-step, integrated into a project-specific quality-gated EM framework.",
        "training_profile": args.training_profile,
        "official_vendor_commit_required": OFFICIAL_VOXTELL_COMMIT,
        "source_checkpoint_sha256": sha256_file(model_dir / "fold_0" / "checkpoint_final.pth"),
        "plans_sha256": sha256_file(model_dir / "plans.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "official_embedding_bank": official_bank_audit,
        "outbound_write_audit": outbound_write_audit(),
        "external_uploads_performed": False,
        "official_retention": {
            "enabled": official_network is not None,
            "target_fraction": args.official_retention_weight,
            "mean_effective_weight": (
                float(np.mean(effective_retention_weights))
                if effective_retention_weights else 0.0
            ),
            "mean_contribution_ratio": (
                float(np.mean(retention_contribution_ratios))
                if retention_contribution_ratios else 0.0
            ),
            "mean_gradient_contribution_ratio": (
                float(np.mean(retention_gradient_contribution_ratios))
                if retention_gradient_contribution_ratios else 0.0
            ),
            "loss": "multi-scale foreground-weighted probability matching plus Bernoulli KL against frozen previous checkpoint",
            "anchor_policy": "frozen validation split, organ-balanced positive anchors",
        },
        "paper_alignment": {
            "prompts_per_image": {"positive": 2, "negative": 1},
            "foreground_oversample_probability": args.foreground_prob,
            "canonical_probability": 0.25,
            "rewrite_probability": 0.75,
            "deep_supervision_weights": [1, 0.5, 0.25, 0.125, 0.0625],
            "loss": "nnUNet DC_and_BCE_loss" if args.training_profile == PAPER_ALIGNED_PROFILE else "project quality-weighted Dice+BCE",
            "patch_size": list(patch_size),
        },
        "known_differences": [
            "project-specific pseudo-labeled dataset",
            "single-node hardware rather than the paper's 64xA100 final run",
            "project-specific quality-gated EM outer loop",
        ],
    }
    audit_bundle = build_sampling_gradient_audits(
        training_rows=training_rows,
        loss_history=loss_history,
        positive_organ_sample_counts=positive_organ_sample_counts,
        negative_organ_sample_counts=negative_organ_sample_counts,
        positive_organ_finite_gradient_mass=positive_organ_finite_gradient_mass,
        positive_organ_nonfinite_gradient_steps=positive_organ_nonfinite_gradient_steps,
    )
    organ_gradient_audit = audit_bundle["organ_gradient_audit"]
    organ_exposure_audit = audit_bundle["organ_exposure_audit"]
    nonfinite_gradient_audit = audit_bundle["nonfinite_gradient_audit"]
    student_sampling_gradient_diagnosis = audit_bundle["student_sampling_gradient_diagnosis"]
    write_json(output_dir / "training_provenance.json", provenance)
    write_json(output_dir / "sampling_audit.json", {
        "training_profile": args.training_profile,
        "eligible_image_centered_cases": len(paper_case_pools),
        "sampling_history": sampling_history,
        "skipped_sampling_attempts": skipped_attempts,
        "optimizer_steps_skipped_nonfinite_grad": optimizer_steps_skipped_nonfinite_grad,
        "derived_crop_negatives_allowed": args.training_profile == QUALITY_WEIGHTED_PROFILE,
        "organ_gradient_audit": organ_gradient_audit,
        "organ_exposure_audit": organ_exposure_audit,
        "nonfinite_gradient_audit": nonfinite_gradient_audit,
        "student_sampling_gradient_diagnosis": student_sampling_gradient_diagnosis,
    })
    write_json(output_dir / "organ_exposure_audit.json", organ_exposure_audit)
    write_json(output_dir / "nonfinite_gradient_audit.json", nonfinite_gradient_audit)
    write_json(output_dir / "student_sampling_gradient_diagnosis.json", student_sampling_gradient_diagnosis)
    write_csv(output_dir / "organ_exposure_audit.csv", audit_bundle["organ_rows"])
    write_csv(output_dir / "student_sampling_gradient_diagnosis.csv", audit_bundle["organ_rows"])
    result = {
        **plan,
        "status": "success",
        "device": str(device),
        "patch_size": list(patch_size),
        "num_prompts": len(prompts),
        "optimizer": args.optimizer,
        "loss_mode": "nnunet_dice_bce_five_scale_deep_supervision" if args.training_profile == PAPER_ALIGNED_PROFILE else "weighted_dice_plus_bce_deep_supervision",
        "training_provenance": str(output_dir / "training_provenance.json"),
        "sampling_audit": str(output_dir / "sampling_audit.json"),
        "deep_supervision": args.deep_supervision,
        "poly_power": args.poly_power,
        "steps": total_steps,
        "stopped_early": stopped_early,
        "best_candidate_found": best_candidate_found,
        "checkpoint_selection": {
            "evaluation_interval": checkpoint_eval_interval,
            "patience": checkpoint_patience,
            "best_task_objective": best_task_score if math.isfinite(best_task_score) else None,
            "retention_anchor_baseline": retention_anchor_baseline,
            "best_new_supervision_foreground_dice": (
                best_task_score if math.isfinite(best_task_score) else None
            ),
            "best_anchor_foreground_dice": best_anchor_score,
        },
        "sampling_history": sampling_history,
        "skipped_sampling_attempts": skipped_attempts,
        "optimizer_steps_skipped_nonfinite_grad": optimizer_steps_skipped_nonfinite_grad,
        "mean_loss": float(np.mean(finite_losses)) if finite_losses else None,
        "mean_task_loss": float(np.mean(task_losses)) if task_losses else None,
        "mean_retention_loss": float(np.mean(retention_losses)) if retention_losses else None,
        "mean_effective_retention_weight": (
            float(np.mean(effective_retention_weights))
            if effective_retention_weights else 0.0
        ),
        "mean_retention_contribution_ratio": (
            float(np.mean(retention_contribution_ratios))
            if retention_contribution_ratios else 0.0
        ),
        "mean_retention_gradient_contribution_ratio": (
            float(np.mean(retention_gradient_contribution_ratios))
            if retention_gradient_contribution_ratios else 0.0
        ),
        "mean_task_foreground_dice": (
            float(np.mean(task_foreground_dices))
            if task_foreground_dices else None
        ),
        "mean_retention_anchor_foreground_dice": (
            float(np.mean(retention_anchor_foreground_dices))
            if retention_anchor_foreground_dices else None
        ),
        "best_retention_anchor_foreground_dice": best_anchor_score,
        "organ_gradient_audit": organ_gradient_audit,
        "organ_exposure_audit": organ_exposure_audit,
        "nonfinite_gradient_audit": nonfinite_gradient_audit,
        "student_sampling_gradient_diagnosis": str(output_dir / "student_sampling_gradient_diagnosis.json"),
        "training_stability_diagnosis": str(output_dir / "training_stability_diagnosis.json"),
        "loss_curve_diagnosis": str(output_dir / "loss_curve_diagnosis.json"),
        "training_sample_distribution_audit": str(output_dir / "training_sample_distribution_audit.json"),
        "student_training_loss_curve": str(output_dir / "student_training_loss_curve.csv"),
        "last_loss": finite_losses[-1] if finite_losses else None,
        "finetuned_checkpoint": str(final_ckpt),
        "inference_model_dir": str(inference_model_dir),
        "inference_checkpoint": str(inference_model_dir / "fold_0" / "checkpoint_final.pth"),
        "runtime_sec": round(time.time() - started, 3),
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
