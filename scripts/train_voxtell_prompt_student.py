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
import json
import os
import pydoc
import random
import shutil
import sys
import time
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
    ap.add_argument("--learning-rate", type=float, default=float(env.get("MEDAI_MSTEP_LR", "5e-5")))
    ap.add_argument("--weight-decay", type=float, default=float(env.get("MEDAI_WEIGHT_DECAY", "1e-5")))
    ap.add_argument("--foreground-prob", type=float, default=float(env.get("MEDAI_FOREGROUND_PROB", "0.7")))
    ap.add_argument("--seed", type=int, default=int(env.get("MEDAI_SEED", "42")))
    ap.add_argument("--save-every", type=int, default=int(env.get("MEDAI_SAVE_EVERY", "0")), help="0 disables intermediate checkpoints.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and write a training plan without loading Qwen/model weights.")
    ap.add_argument("--freeze-encoder", action="store_true", help="Only train prompt projection/decoder layers.")
    return ap.parse_args()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
            "network_weights": network.state_dict(),
            "source_model_dir": str(source_model_dir),
            "manifest": str(manifest_path),
            "step": step,
        },
        fold_out / "checkpoint_final.pth",
    )
    return model_out


def load_manifest(path: Path, max_items: int = 0) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in doc.get("items", []):
        image = item.get("image")
        mask = item.get("mask")
        prompt = item.get("prompt")
        training_weight = float(item.get("training_weight", 1.0) or 0.0)
        if not image or not mask or not prompt:
            continue
        if training_weight <= 0.0:
            continue
        if Path(image).exists() and Path(mask).exists():
            rows.append({**item, "training_weight": training_weight})
    if max_items > 0:
        rows = rows[:max_items]
    return rows


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


def read_mask(path: Path) -> np.ndarray:
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        arr, _ = NibabelIOWithReorient().read_images([str(path)])
        arr = arr[0] if arr.ndim == 4 else arr
    except Exception:
        import nibabel as nib
        arr = np.asanyarray(nib.load(str(path)).dataobj)
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


def choose_patch_start(mask: np.ndarray, patch_size: tuple[int, int, int], foreground_prob: float) -> tuple[int, int, int]:
    shape = mask.shape
    max_start = [max(0, int(s) - int(p)) for s, p in zip(shape, patch_size)]
    use_fg = random.random() < foreground_prob and bool(mask.sum() > 0)
    if use_fg:
        coords = np.argwhere(mask > 0)
        center = coords[random.randrange(len(coords))]
        start = []
        for c, p, m in zip(center, patch_size, max_start):
            lo = max(0, int(c) - int(p) // 2)
            start.append(min(lo, m))
        return tuple(start)  # type: ignore[return-value]
    return tuple(random.randint(0, m) if m > 0 else 0 for m in max_start)  # type: ignore[return-value]


def load_training_patch(item: dict[str, Any], patch_size: tuple[int, int, int], foreground_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    image = read_image(Path(item["image"]))
    mask = read_mask(Path(item["mask"]))
    if image.shape[1:] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch for {item['case_id']} {item['organ']}: {image.shape[1:]} vs {mask.shape}")

    image, _, bbox = crop_to_nonzero(image, None)
    mask = mask[bbox_to_slices(bbox)]
    image = ZScoreNormalization(intensityproperties={}).run(image, None)
    image, mask = pad_to_shape(image, mask, patch_size)

    sx, sy, sz = choose_patch_start(mask, patch_size, foreground_prob)
    px, py, pz = patch_size
    image_patch = image[:, sx:sx + px, sy:sy + py, sz:sz + pz]
    mask_patch = mask[sx:sx + px, sy:sy + py, sz:sz + pz]
    return torch.from_numpy(image_patch[None].astype(np.float32)), torch.from_numpy(mask_patch[None, None].astype(np.float32))


@torch.inference_mode()
def build_prompt_embeddings(prompts: list[str], text_model_name: str, device: torch.device, cache_path: Path | None) -> dict[str, torch.Tensor]:
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if all(prompt in cached for prompt in prompts):
            return {prompt: cached[prompt].float() for prompt in prompts}

    tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side="left")
    text_backbone = AutoModel.from_pretrained(text_model_name).eval().to(device)
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
        torch.save(out, cache_path)
    del text_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def set_trainable_params(network: nn.Module, freeze_encoder: bool) -> None:
    if not freeze_encoder:
        return
    for name, param in network.named_parameters():
        param.requires_grad = not name.startswith("encoder.")


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
    model_files_ok = (model_dir / "plans.json").exists() and (model_dir / "fold_0" / "checkpoint_final.pth").exists()
    validation_errors: list[str] = []
    if not rows:
        validation_errors.append("No valid manifest items with existing image/mask/prompt paths")
    if not model_files_ok:
        validation_errors.append("Missing plans.json or fold_0/checkpoint_final.pth in model_dir")
    plan = {
        "stage": "train_voxtell_prompt_student",
        "status": "dry_run" if args.dry_run else "pending",
        "manifest": str(manifest_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "num_manifest_items": len(rows),
        "training_weight_policy": "Items with training_weight <= 0 are skipped; A/B/C map to strong/lower/weak supervision.",
        "model_files_ok": model_files_ok,
        "manifest_items_ok": bool(rows),
        "validation_status": "ok" if not validation_errors else "failed",
        "validation_errors": validation_errors,
        "text_encoding_model": args.text_encoding_model,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "freeze_encoder": args.freeze_encoder,
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
    if not rows:
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
    network = build_voxtell_network(model_dir, deep_supervision=False)
    set_trainable_params(network, args.freeze_encoder)
    network.to(device)
    network.train()

    optim = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    losses: list[float] = []
    total_steps = 0
    max_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(rows)
    while total_steps < max_steps:
        random.shuffle(rows)
        for item in rows:
            if total_steps >= max_steps:
                break
            try:
                image, target = load_training_patch(item, patch_size, args.foreground_prob)
            except Exception as exc:
                losses.append(float("nan"))
                print(f"[warn] skip {item.get('case_id')} {item.get('organ')}: {exc}", flush=True)
                continue
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            text_embedding = embeddings[str(item["prompt"])].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = network(image, text_embedding)
                raw_loss = F.binary_cross_entropy_with_logits(logits, target)
                loss = raw_loss * float(item.get("training_weight", 1.0) or 0.0)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_steps += 1
            losses.append(float(loss.detach().cpu()))
            if total_steps % 10 == 0:
                recent = [x for x in losses[-10:] if np.isfinite(x)]
                print(f"step={total_steps} loss={np.mean(recent):.5f}", flush=True)
            if args.save_every > 0 and total_steps % args.save_every == 0:
                torch.save({"network_weights": network.state_dict(), "step": total_steps}, output_dir / f"checkpoint_step_{total_steps}.pth")

    final_ckpt = output_dir / "model_finetune.pth"
    torch.save({
        "network_weights": network.state_dict(),
        "optimizer_state": optim.state_dict(),
        "source_model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "step": total_steps,
        "patch_size": patch_size,
    }, final_ckpt)
    inference_model_dir = write_voxtell_model_dir(model_dir, output_dir, network, total_steps, manifest_path)
    finite_losses = [x for x in losses if np.isfinite(x)]
    result = {
        **plan,
        "status": "success",
        "device": str(device),
        "patch_size": list(patch_size),
        "num_prompts": len(prompts),
        "steps": total_steps,
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
