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
import hashlib
import json
import os
import pydoc
import random
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VOXTELL_ROOT = ROOT / "third_party" / "VoxTell"
sys.path.insert(0, str(VOXTELL_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from torch import nn
from torch._dynamo import OptimizedModule
from transformers import AutoModel, AutoTokenizer

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


SAFE_NEGATIVE_SOURCES = {
    "nonmedical_absent_object",
    "out_of_scan_anatomy_with_coverage_evidence",
    "explicit_confirmed_absent_anatomy",
    "case_373_expected_absent",
}
SAFE_ZERO_MASK_ROLES = {"negative_target_mask", "absent_negative_target_mask"}
ACCEPTED_AUTOLABEL_SCHEMAS = {"autolabel_core_v2", "autolabel_core_v3"}
NEGATIVE_REASON_MAP = {
    "case_373_expected_absent": "absent_in_scan",
    "out_of_scan_anatomy_with_coverage_evidence": "absent_in_scan",
    "explicit_confirmed_absent_anatomy": "absent_in_scan",
    "nonmedical_absent_object": "wrong_prompt",
}


def parse_args() -> argparse.Namespace:
    env = os.environ
    ap = argparse.ArgumentParser(description="Fine-tune VoxTell-style 3D prompt student on prompt/mask manifest.")
    ap.add_argument("--manifest", default=env.get("MEDAI_PROMPT_STUDENT_MANIFEST"), help="Prompt/mask manifest JSON.")
    ap.add_argument("--model-dir", default=env.get("MEDAI_VOXTELL_MODEL_DIR"), help="VoxTell model dir with plans.json and fold_0/checkpoint_final.pth.")
    ap.add_argument("--output-dir", default=env.get("MEDAI_PROMPT_STUDENT_OUTPUT_DIR", str(ROOT / "outputs/voxtell_prompt_mstep")))
    ap.add_argument("--text-encoding-model", default=env.get("MEDAI_TEXT_ENCODING_MODEL", "Qwen/Qwen3-Embedding-4B"))
    ap.add_argument("--embedding-cache", default=None, help="Optional torch cache for prompt embeddings.")
    ap.add_argument("--device", default=env.get("MEDAI_DEVICE", "cuda"))
    ap.add_argument("--epochs", type=int, default=int(env.get("MEDAI_MSTEP_EPOCHS", "1")))
    ap.add_argument("--max-steps", type=int, default=int(env.get("MEDAI_MAX_STEPS", "0")), help="0 means one pass over manifest per epoch.")
    ap.add_argument("--max-items", type=int, default=int(env.get("MEDAI_MAX_ITEMS", "0")), help="Optional subset for smoke tests.")
    ap.add_argument("--learning-rate", type=float, default=float(env.get("MEDAI_MSTEP_LR", "1e-4")))
    ap.add_argument("--weight-decay", type=float, default=float(env.get("MEDAI_WEIGHT_DECAY", "3e-5")))
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default=env.get("MEDAI_OPTIMIZER", "sgd"))
    ap.add_argument("--poly-power", type=float, default=float(env.get("MEDAI_POLY_POWER", "0.9")))
    ap.add_argument("--deep-supervision", action="store_true", default=env.get("MEDAI_DEEP_SUPERVISION", "1").lower() not in {"0", "false", "no"})
    ap.add_argument("--foreground-prob", type=float, default=float(env.get("MEDAI_FOREGROUND_PROB", "0.7")))
    ap.add_argument("--pos-neg-ratio", default=env.get("MEDAI_POS_NEG_RATIO", "2:1"), help="Runtime positive:negative sampling ratio, e.g. 2:1, 1:1, 10:1, or 2.0.")
    ap.add_argument("--sampling-log-interval", type=int, default=int(env.get("MEDAI_SAMPLING_LOG_INTERVAL", "10")), help="Steps per sampling-stat log window.")
    ap.add_argument("--seed", type=int, default=int(env.get("MEDAI_SEED", "42")))
    ap.add_argument("--save-every", type=int, default=int(env.get("MEDAI_SAVE_EVERY", "0")), help="0 disables intermediate checkpoints.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and write a training plan without loading Qwen/model weights.")
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
    return ap.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
) -> Path:
    """Write an inference-compatible VoxTell model directory."""
    model_out = output_dir / "voxtell_finetuned_model"
    fold_out = model_out / "fold_0"
    fold_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_dir / "plans.json", model_out / "plans.json")
    info_path = source_model_dir / "fold_0" / "INFO.txt"
    if info_path.exists():
        shutil.copy2(info_path, fold_out / "INFO.txt")
    torch.save(
        {
            "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": network.state_dict(),
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
    variants = [str(p).strip() for p in item.get("prompt_variants", []) if str(p).strip()]
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
    if str(item.get("target_type") or "") == "absent_negative":
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
    if grade in {"A", "B"}:
        return target_type == "hard"
    if grade == "C":
        probability_path = item.get("probability_mask_path") or item.get("mask")
        return target_type == "soft" and bool(probability_path) and Path(str(probability_path)).exists()
    return False


def load_manifest(path: Path, max_items: int = 0) -> list[dict[str, Any]]:
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


def build_sample_pools(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive = [r for r in rows if r.get("supervision_type", "positive") == "positive"]
    manifest_negative = [r for r in rows if r.get("supervision_type") == "negative"]
    derived_crop_negative = [dict(r, supervision_type="negative", negative_reason="absent_in_crop", derived_negative_from_positive=True) for r in positive]
    return {
        "positive": positive,
        "negative": manifest_negative + derived_crop_negative,
        "manifest_negative": manifest_negative,
        "derived_crop_negative": derived_crop_negative,
    }


def _weighted_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot sample from an empty pool")
    weights = [max(0.0, float(r.get("sampling_weight", r.get("training_weight", 1.0)) or 0.0)) for r in rows]
    if sum(weights) <= 0:
        return dict(random.choice(rows))
    return dict(random.choices(rows, weights=weights, k=1)[0])


def choose_sample_kind(step: int, pos_neg_ratio: tuple[int, int], pools: dict[str, list[dict[str, Any]]]) -> str:
    pos_n, neg_n = pos_neg_ratio
    if not pools.get("negative") or neg_n <= 0:
        return "positive"
    if not pools.get("positive"):
        return "negative"
    cycle = ["positive"] * pos_n + ["negative"] * neg_n
    return cycle[step % len(cycle)]


def sample_training_item(step: int, pools: dict[str, list[dict[str, Any]]], pos_neg_ratio: tuple[int, int]) -> dict[str, Any]:
    kind = choose_sample_kind(step, pos_neg_ratio, pools)
    item = _weighted_choice(pools[kind])
    item["runtime_sample_kind"] = kind
    if kind == "negative":
        item["negative_reason"] = canonical_negative_reason(item, "absent_in_crop")
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
    target_type = str(item.get("target_type") or "hard")
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
        "sample_kind": sample_kind,
        "negative_reason": canonical_negative_reason(item) if sample_kind == "negative" else None,
        "foreground_voxel_count": foreground_voxels,
        "foreground_voxel_ratio": foreground_voxels / float(voxel_count) if voxel_count else 0.0,
        "all_zero_target": foreground_voxels <= 0.0,
        "crop_start": [int(sx), int(sy), int(sz)],
    }
    return torch.from_numpy(image_patch[None].astype(np.float32)), torch.from_numpy(mask_patch[None, None].astype(np.float32)), meta


def load_training_patch(item: dict[str, Any], patch_size: tuple[int, int, int], foreground_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    image, target, _meta = load_training_patch_with_metadata(item, patch_size, foreground_prob)
    return image, target


@torch.inference_mode()
def build_prompt_embeddings(prompts: list[str], text_model_name: str, device: torch.device, cache_path: Path | None) -> dict[str, torch.Tensor]:
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

    tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side="left")
    text_backbone = AutoModel.from_pretrained(text_model_name).eval().to(device)
    text_backbone.requires_grad_(False)
    out: dict[str, torch.Tensor] = {}
    for prompt in prompts:
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
    preds = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    if not preds:
        raise ValueError("VoxTell network returned no outputs")
    total = None
    total_w = 0.0
    for idx, pred in enumerate(preds):
        if isinstance(pred, dict):
            pred = pred.get("logits") or pred.get("seg") or pred.get("prediction")
        if pred is None:
            continue
        stage_target = _resize_target_like(target, pred)
        bce = _foreground_balanced_bce(pred, stage_target, bce_pos_weight_cap)
        dice = dice_loss_with_logits(pred, stage_target)
        stage_w = 0.5 ** idx
        loss = stage_w * (bce + dice)
        total = loss if total is None else total + loss
        total_w += stage_w
    if total is None or total_w <= 0:
        raise ValueError("VoxTell network outputs were not tensors")
    return (total / total_w) * float(training_weight or 0.0)


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

    rows = load_manifest(manifest_path, max_items=args.max_items)
    sample_pools = build_sample_pools(rows)
    try:
        pos_neg_ratio = parse_pos_neg_ratio(args.pos_neg_ratio)
        pos_neg_ratio_error = None
    except Exception as exc:
        pos_neg_ratio = (2, 1)
        pos_neg_ratio_error = str(exc)
    model_files_ok = (model_dir / "plans.json").exists() and (model_dir / "fold_0" / "checkpoint_final.pth").exists()
    validation_errors: list[str] = []
    if pos_neg_ratio_error:
        validation_errors.append(f"Invalid --pos-neg-ratio: {pos_neg_ratio_error}")
    if not rows:
        validation_errors.append("No valid manifest items with existing image/mask/prompt paths")
    if not model_files_ok:
        validation_errors.append("Missing plans.json or fold_0/checkpoint_final.pth in model_dir")
    plan = {
        "stage": "train_voxtell_prompt_student",
        "status": "dry_run" if args.dry_run else "pending",
        "training_mode": "project_voxtell_prompt_distillation_student",
        "legacy_mode": "project_distillation_experimental",
        "canonical_training_backend": "project_voxtell_prompt_distillation_student",
        "trainer": "project_voxtell_prompt_distillation_student",
        "is_official_voxtell_encoder_transfer": False,
        "is_prompt_conditioned_student": True,
        "student_inference_backend": "voxtell_prompt_api",
        "is_project_distillation": True,
        "uses_official_voxtell_model": True,
        "uses_official_checkpoint_initialization": True,
        "uses_project_manifest": True,
        "uses_autolabelcore_confidence": True,
        "uses_abcd_training_weight": True,
        "official_prompt_training_pipeline_available": False,
        "negative_prompt_sampling": "runtime_pool_sampler",
        "training_provenance_warning": "Project prompt-conditioned distillation trainer using official VoxTell components; not official voxtell-finetune.",
        "manifest": str(manifest_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "num_manifest_items": len(rows),
        "candidate_pool_positive_count": len(sample_pools["positive"]),
        "candidate_pool_negative_count": len(sample_pools["negative"]),
        "candidate_pool_manifest_negative_count": len(sample_pools["manifest_negative"]),
        "candidate_pool_derived_crop_negative_count": len(sample_pools["derived_crop_negative"]),
        "pos_neg_ratio": args.pos_neg_ratio,
        "pos_neg_ratio_parsed": list(pos_neg_ratio),
        "sampling_log_interval": args.sampling_log_interval,
        "training_weight_policy": "Items with training_weight <= 0 are skipped; prompt variants are kept as candidate-pool rows; runtime sampler uses sampling_weight within positive/negative pools, while --pos-neg-ratio controls the actual positive/negative training mix.",
        "loss_mode": "weighted_dice_plus_bce_deep_supervision",
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
    }
    if args.dry_run:
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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    patch_size = load_patch_size(model_dir)
    prompts = sorted({str(r["prompt"]) for r in rows})
    cache_path = Path(args.embedding_cache).resolve() if args.embedding_cache else output_dir / "prompt_embeddings.pt"

    started = time.time()
    embeddings = build_prompt_embeddings(prompts, args.text_encoding_model, device, cache_path)
    network = build_voxtell_network(model_dir, deep_supervision=args.deep_supervision)
    set_trainable_params(network, args.freeze_encoder, args.trainable_scope)
    network.to(device)
    network.train()

    optim = build_optimizer((p for p in network.parameters() if p.requires_grad), args)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    losses: list[float] = []
    loss_history: list[dict[str, float | int]] = []
    total_steps = 0
    max_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(rows))
    sampling_history: list[dict[str, Any]] = []
    sampling_window: list[dict[str, Any]] = []
    skipped_attempts = 0
    max_attempts = max_steps * 20
    attempts = 0
    while total_steps < max_steps and attempts < max_attempts:
        attempts += 1
        item = sample_training_item(total_steps, sample_pools, pos_neg_ratio)
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
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = network(image, text_embedding)
            loss = voxtell_supervision_loss(
                logits,
                target,
                float(item.get("effective_loss_weight", item.get("training_weight", 1.0)) or 0.0),
                args.bce_pos_weight_cap,
            )
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total_steps += 1
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        sampling_window.append(sample_meta)
        loss_history.append({
            "step": total_steps,
            "loss": loss_value,
            "learning_rate": float(optim.param_groups[0]["lr"]),
            "sample_kind": sample_meta["sample_kind"],
            "foreground_voxel_ratio": float(sample_meta["foreground_voxel_ratio"]),
            "all_zero_target": int(bool(sample_meta["all_zero_target"])),
        })
        log_interval = max(1, int(args.sampling_log_interval))
        if total_steps % log_interval == 0 or total_steps == max_steps:
            recent = [x for x in losses[-log_interval:] if np.isfinite(x)]
            pos_count = sum(1 for m in sampling_window if m.get("sample_kind") == "positive")
            neg_count = sum(1 for m in sampling_window if m.get("sample_kind") == "negative")
            reason_counts: dict[str, int] = {"absent_in_scan": 0, "absent_in_crop": 0, "wrong_prompt": 0}
            for m in sampling_window:
                reason = m.get("negative_reason")
                if reason:
                    reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
            stat = {
                "step": total_steps,
                "loss": float(np.mean(recent)) if recent else None,
                "batch_positive_count": pos_count,
                "batch_negative_count": neg_count,
                "positive_negative_ratio": (float(pos_count) / float(neg_count)) if neg_count else None,
                "foreground_voxel_ratio": float(np.mean([m.get("foreground_voxel_ratio", 0.0) for m in sampling_window])) if sampling_window else 0.0,
                "all_zero_target_count": sum(1 for m in sampling_window if m.get("all_zero_target")),
                "negative_reason_counts": reason_counts,
            }
            sampling_history.append(stat)
            print(json.dumps(stat, ensure_ascii=False), flush=True)
            sampling_window = []
        if args.save_every > 0 and total_steps % args.save_every == 0:
            # Rotate one recovery checkpoint; 3D checkpoints are ~1.7 GB
            # and retaining every interval can exhaust the experiment disk.
            torch.save({"eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": network.state_dict(), "step": total_steps}, output_dir / "checkpoint_latest.pth")
    if total_steps < max_steps:
        raise RuntimeError(f"Runtime sampler produced only {total_steps}/{max_steps} steps after {attempts} attempts")

    final_ckpt = output_dir / "model_finetune.pth"
    torch.save({
        "training_mode": "project_voxtell_prompt_distillation_student",
        "legacy_mode": "project_distillation_experimental",
        "canonical_training_backend": "project_voxtell_prompt_distillation_student",
        "trainer": "project_voxtell_prompt_distillation_student",
        "is_official_voxtell_encoder_transfer": False,
        "is_prompt_conditioned_student": True,
        "eligible_for_next_round_prompt_student": False,
        "eligible_as_teacher_candidate": False,
        "network_weights": network.state_dict(),
        "optimizer_state": optim.state_dict(),
        "source_model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "step": total_steps,
        "patch_size": patch_size,
    }, final_ckpt)
    inference_model_dir = write_voxtell_model_dir(model_dir, output_dir, network, total_steps, manifest_path)
    finite_losses = [x for x in losses if np.isfinite(x)]
    write_json(output_dir / "loss_history.json", {"steps": total_steps, "history": loss_history, "sampling_history": sampling_history, "skipped_sampling_attempts": skipped_attempts})
    result = {
        **plan,
        "status": "success",
        "device": str(device),
        "patch_size": list(patch_size),
        "num_prompts": len(prompts),
        "optimizer": args.optimizer,
        "loss_mode": "weighted_dice_plus_bce_deep_supervision",
        "deep_supervision": args.deep_supervision,
        "poly_power": args.poly_power,
        "steps": total_steps,
        "sampling_history": sampling_history,
        "skipped_sampling_attempts": skipped_attempts,
        "mean_loss": float(np.mean(finite_losses)) if finite_losses else None,
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
