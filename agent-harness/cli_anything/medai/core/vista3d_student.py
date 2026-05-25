"""VISTA3D-based Promptable Unified Student Model.

Architecture:
  Input CT + text prompt ("segment pancreas")
        ↓
  text_to_label_id()  [teacher_branch_map.py]
        ↓
  label_prompt = [4]  (pancreas class ID)
        ↓
  VISTA3D inference   (SwinUNETR + point_head)
        ↓
  pancreas mask (3D NIfTI)

This replaces nnUNetv2 as the unified M-step training target.

Key design decisions (from teacher feedback):
1. VISTA3D is the unified student — no new anatomical router needed.
2. Language prompt = text → label_id mapping (not open-vocabulary VLM).
3. M-step = VISTA3D class-level continual fine-tuning via train_continual.json.
4. Default: freeze SwinUNETR backbone, only update point_head / class embedding.
5. Periodic global consolidation: small LR, partial backbone unfreeze.
6. Multiple organs can be segmented in one forward pass via label_prompt list.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .teacher_branch_map import text_to_label_id, texts_to_label_ids, get_teacher_for_organ
from .json_utils import write_json


class VISTA3DStudent:
    """VISTA3D-based unified segmentation student with language prompt interface.

    Parameters
    ----------
    vista3d_root : path to VISTA3D-Inference-Pipeline-master directory
    model_path : path to VISTA3D model checkpoint (.pt file)
    device : "cuda" or "cpu"
    """

    def __init__(
        self,
        vista3d_root: str | Path,
        model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        self.vista3d_root = Path(vista3d_root).resolve()
        self.model_path = Path(model_path).resolve() if model_path else self.vista3d_root / "models" / "model.pt"
        self.device = device
        self._inference_config = self.vista3d_root / "configs" / "inference.json"
        self._train_config = self.vista3d_root / "configs" / "train.json"
        self._continual_config = self.vista3d_root / "configs" / "train_continual.json"

    # ── Inference ─────────────────────────────────────────────────────────────

    def segment(
        self,
        ct_image: str | Path,
        prompts: list[str],
        output_dir: str | Path,
        dry_run: bool = False,
        timeout_sec: int = 600,
    ) -> dict[str, Any]:
        """Segment organs from a CT using language prompts.

        Parameters
        ----------
        ct_image : path to CT NIfTI file
        prompts : list of organ names or natural language prompts
                  e.g. ["pancreas", "liver"] or ["segment pancreas", "segment liver"]
        output_dir : directory to write per-organ mask NIfTI files
        dry_run : build command without running VISTA3D

        Returns
        -------
        dict with status, organ_masks paths, label_ids used
        """
        ct = Path(ct_image).resolve()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # text → label_id mapping (language prompt interface)
        organ_label_ids = texts_to_label_ids(prompts)
        unknown = [p for p in prompts if text_to_label_id(p) is None]

        result: dict[str, Any] = {
            "stage": "vista3d_student_inference",
            "ct_image": str(ct),
            "prompts": prompts,
            "organ_label_ids": organ_label_ids,
            "unknown_prompts": unknown,
            "output_dir": str(out),
            "model_path": str(self.model_path),
            "device": self.device,
        }

        if not organ_label_ids:
            result.update({"status": "failed", "reason": "No prompts could be mapped to VISTA3D label IDs"})
            return result

        if not ct.exists() and not dry_run:
            result.update({"status": "failed", "reason": f"CT not found: {ct}"})
            return result

        # Build VISTA3D inference command using monai.bundle run + inference.json.
        # We use an override config to:
        #   1. Remove the batch-mode Lambda postprocessing (labels2onehot.seperate_class)
        #   2. Override checkpointloader to load the fine-tuned model
        label_ids = sorted(set(organ_label_ids.values()))
        input_dict = json.dumps({"image": str(ct), "label_prompt": label_ids})

        # Write a per-call override config to strip the batch-only Lambda transform
        infer_override = {
            "postprocessing": {
                "_target_": "Compose",
                "transforms": [
                    {"_target_": "monai.apps.vista3d.transforms.VistaPostTransformd", "keys": "pred"},
                    {"_target_": "Invertd", "keys": "pred",
                     "transform": "$copy.deepcopy(@preprocessing)",
                     "orig_keys": "@image_key", "nearest_interp": True, "to_tensor": True},
                    {"_target_": "Lambdad", "func": "$lambda x: torch.nan_to_num(x, nan=255)", "keys": "pred"},
                    {"_target_": "SaveImaged", "keys": "pred", "resample": False,
                     "data_root_dir": "@input_dir", "output_dir": "@output_dir",
                     "output_ext": "@output_ext", "output_dtype": "@output_dtype",
                     "output_postfix": "@output_postfix", "separate_folder": "@separate_folder"},
                ],
            }
        }
        override_path = out / "_infer_override.json"
        with open(override_path, "w") as _f:
            json.dump(infer_override, _f)

        import os as _os
        env = {**_os.environ, "VISTA3D_OUTPUT_DIR": str(out)}

        command = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", f"['{self._inference_config}','{override_path}']",
            "--bundle_root", str(self.vista3d_root),
            "--input_dict", input_dict,
            "--input_dir", str(ct.parent),
            "--output_dir", str(out),
            "--output_postfix", "seg",
            "--separate_folder", "False",
            "--checkpointloader#load_path", str(self.model_path),
            "--device", self.device,
        ]

        result["command"] = command

        if dry_run:
            result.update({
                "status": "dry_run",
                "note": "VISTA3D command prepared but not executed.",
                "organ_masks": {organ: str(out / f"{organ}.nii.gz") for organ in organ_label_ids},
            })
            return result

        start = time.time()
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True,
                check=False, timeout=timeout_sec,
                cwd=str(self.vista3d_root),
                env=env,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(command, 124, stdout="", stderr=f"Timeout after {timeout_sec}s")
            timed_out = True

        elapsed = time.time() - start

        # VISTA3D outputs a combined integer label map (one .nii.gz per input).
        # Split it into per-organ binary masks named <organ>.nii.gz.
        organ_masks: dict[str, str] = {}
        if not timed_out and proc.returncode == 0:
            combined_files = sorted(out.rglob("*.nii.gz"))
            if combined_files:
                try:
                    import nibabel as nib
                    import numpy as np
                    combined_img = nib.load(str(combined_files[0]))
                    combined_arr = combined_img.get_fdata().astype("int16")
                    for organ, lid in organ_label_ids.items():
                        mask = (combined_arr == lid).astype("uint8")
                        if mask.sum() == 0:
                            continue
                        mask_path = out / f"{organ}.nii.gz"
                        nib.save(nib.Nifti1Image(mask, combined_img.affine), str(mask_path))
                        organ_masks[organ] = str(mask_path)
                except Exception as e:
                    pass  # fall through to failed status

        result.update({
            "status": "timed_out" if timed_out else ("success" if organ_masks else "failed"),
            "return_code": proc.returncode,
            "runtime_sec": round(elapsed, 2),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "organ_masks": organ_masks,
            "num_segmented": len(organ_masks),
        })
        return result

    # ── M-step: continual fine-tuning ─────────────────────────────────────────

    def continual_finetune(
        self,
        pseudo_label_dir: str | Path,
        ct_dir: str | Path,
        target_organs: list[str],
        output_dir: str | Path,
        learning_rate: float = 5e-5,
        n_train_samples: int = 50,
        max_epochs: int = 50,
        freeze_backbone: bool = True,
        global_consolidation: bool = False,
        dry_run: bool = False,
        timeout_sec: int = 7200,
        prebuilt_datalist_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """M-step: continual fine-tuning of VISTA3D on pseudo-labels.

        Only updates point_head / class_embedding for target organs.
        Backbone (SwinUNETR) is frozen by default.

        Parameters
        ----------
        pseudo_label_dir : directory with per-case pseudo-label NIfTI files
        ct_dir : directory with per-case CT NIfTI files
        target_organs : list of organ names to fine-tune
        output_dir : where to save the fine-tuned model checkpoint
        freeze_backbone : if True, only update point_head (default)
        global_consolidation : if True, partially unfreeze backbone with low LR
        """
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # Map target organs to VISTA3D label IDs
        organ_label_ids = texts_to_label_ids(target_organs)
        if not organ_label_ids:
            return {"status": "failed", "reason": "No target organs mapped to VISTA3D label IDs"}

        # Build label_mappings for train_continual.json
        # Format: [[source_label_id, target_label_id], ...]
        label_mappings = [[lid, lid] for lid in sorted(set(organ_label_ids.values()))]

        # Build datalist JSON for VISTA3D (or reuse a prebuilt one)
        datalist_path = out / "continual_datalist.json"
        if prebuilt_datalist_path and Path(prebuilt_datalist_path).exists():
            import shutil as _shutil
            _shutil.copy2(str(prebuilt_datalist_path), str(datalist_path))
            with open(datalist_path) as _f:
                datalist = json.load(_f)
        else:
            datalist = self._build_datalist(pseudo_label_dir, ct_dir, organ_label_ids)
            write_json(datalist_path, datalist)

        n_train = min(n_train_samples, len(datalist.get("training", [])))
        n_val = max(1, len(datalist.get("validation", [])))
        train_items = datalist.get("training", [])[:n_train]
        val_items = datalist.get("validation", [])[:n_val]
        ckpt_dir = out / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Override config for this fine-tuning run.
        # train_continual.json is a patch layer on top of train.json.
        # train.json has the `run` key (required by monai.bundle run).
        # We must pass train.json first, then train_continual.json as the patch.
        # We also override train_datalist / val_datalist directly so that
        # datafold_read (which expects fold-split JSON) is bypassed.
        config_overrides = {
            "bundle_root": str(self.vista3d_root),
            "ckpt_dir": str(ckpt_dir),
            "output_dir": str(out / "eval"),
            "finetune": True,
            "finetune_model_path": str(self.model_path),
            "label_mappings": {"default": label_mappings},
            "learning_rate": learning_rate,
            "n_train_samples": n_train,
            "n_val_samples": n_val,
            # Bypass datafold_read by directly providing list expressions
            "train_datalist": train_items,
            "val_datalist": val_items,
            "data_list_file_path": str(datalist_path),
            "device": self.device,
            "epochs": max_epochs,
            # Reduce patch size to fit in 79GB GPU (default 128^3 causes OOM)
            "patch_size": [96, 96, 96],
            # Disable TensorBoard — not installed in this environment
            "use_tensorboard": False,
            # Always save checkpoint regardless of metric improvement.
            # save_key_metric=True tries to delete the previous best file first,
            # which raises FileNotFoundError when the file doesn't exist yet.
            "validate#handlers#3#save_key_metric": False,
            "validate#handlers#3#save_final": True,
            "validate#handlers#3#key_metric_filename": "model_finetune.pt",
        }

        # Backbone freeze: only train point_head / class_embed
        if freeze_backbone and not global_consolidation:
            config_overrides["optimizer"] = {
                "_target_": "torch.optim.AdamW",
                "lr": learning_rate,
                "params": "$[p for n, p in @network.named_parameters() if 'point_head' in n or 'class_embed' in n]",
            }

        config_path = out / "continual_config_override.json"
        write_json(config_path, config_overrides)

        # Pass train.json first (has `run` key), then train_continual.json (patch),
        # then our override. monai.bundle merges configs left-to-right.
        command = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", f"['{self._train_config}','{self._continual_config}','{config_path}']",
            "--bundle_root", str(self.vista3d_root),
        ]

        result: dict[str, Any] = {
            "stage": "vista3d_continual_finetune",
            "target_organs": target_organs,
            "organ_label_ids": organ_label_ids,
            "label_mappings": label_mappings,
            "freeze_backbone": freeze_backbone,
            "global_consolidation": global_consolidation,
            "learning_rate": config_overrides["learning_rate"],
            "n_train_samples": config_overrides["n_train_samples"],
            "max_epochs": max_epochs,
            "command": command,
            "output_dir": str(out),
            "datalist_path": str(datalist_path),
        }

        if dry_run:
            result.update({"status": "dry_run", "note": "Fine-tuning command prepared but not executed."})
            write_json(out / "continual_finetune_plan.json", result)
            return result

        start = time.time()
        try:
            train_env = {**os.environ,
                         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            proc = subprocess.run(
                command, capture_output=True, text=True,
                check=False, timeout=timeout_sec,
                cwd=str(self.vista3d_root),
                env=train_env,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(command, 124, stdout="", stderr=f"Timeout after {timeout_sec}s")
            timed_out = True

        elapsed = time.time() - start

        # If training crashed (return_code != 0), never report success even if
        # a stale checkpoint file exists from a previous run.
        if not timed_out and proc.returncode != 0:
            result.update({
                "status": "failed",
                "return_code": proc.returncode,
                "runtime_sec": round(elapsed, 2),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "finetuned_checkpoint": None,
                "has_checkpoint": False,
            })
            write_json(out / "continual_finetune_result.json", result)
            return result

        # Check for output checkpoint — VISTA3D saves to ckpt_dir/model_finetune.pt
        # With save_final=True it may also save as model_final_iteration=N.pt
        finetuned_ckpt = out / "model_finetune.pt"
        if not finetuned_ckpt.exists():
            import shutil as _shutil
            # Try exact name first
            for cand in [
                out / "checkpoints" / "model_finetune.pt",
            ]:
                if cand.exists():
                    _shutil.copy2(str(cand), str(finetuned_ckpt))
                    break
            # Try any checkpoint saved by save_final (model_final_iteration=N.pt)
            if not finetuned_ckpt.exists():
                ckpt_dir = out / "checkpoints"
                finals = sorted(ckpt_dir.glob("model_final_iteration=*.pt")) if ckpt_dir.exists() else []
                if finals:
                    _shutil.copy2(str(finals[-1]), str(finetuned_ckpt))
        has_checkpoint = finetuned_ckpt.exists()

        result.update({
            "status": "timed_out" if timed_out else ("success" if has_checkpoint else "failed"),
            "return_code": proc.returncode,
            "runtime_sec": round(elapsed, 2),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "finetuned_checkpoint": str(finetuned_ckpt) if has_checkpoint else None,
            "has_checkpoint": has_checkpoint,
        })

        if has_checkpoint:
            result["next_step"] = f"Update model_path to {finetuned_ckpt} for next inference round."

        write_json(out / "continual_finetune_result.json", result)
        return result

    def _build_datalist(
        self,
        pseudo_label_dir: str | Path,
        ct_dir: str | Path,
        organ_label_ids: dict[str, int],
    ) -> dict[str, Any]:
        """Build VISTA3D-compatible datalist from pseudo-label directory."""
        label_dir = Path(pseudo_label_dir).resolve()
        ct_root = Path(ct_dir).resolve()
        training: list[dict[str, str]] = []

        for case_dir in sorted(label_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            # Find CT
            ct_path = None
            for cand in [
                ct_root / case_id / "ct.nii.gz",
                ct_root / "ImageTr" / case_id / "ct.nii.gz",
                ct_root / case_id / "image.nii.gz",
            ]:
                if cand.exists():
                    ct_path = cand
                    break
            if ct_path is None:
                continue
            # Find pseudo-label (combined or per-organ)
            combined = case_dir / "combined_label.nii.gz"
            if not combined.exists():
                # Try to find any organ mask
                masks = list(case_dir.glob("*.nii.gz"))
                if not masks:
                    continue
                combined = masks[0]
            training.append({"image": str(ct_path), "label": str(combined)})

        return {"training": training, "validation": training[:max(1, len(training) // 5)]}
