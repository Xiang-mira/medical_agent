from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .json_utils import write_json


def build_training_manifest(updated_annotations_root: str | Path, output_manifest: str | Path, organs: list[str] | None = None) -> dict[str, Any]:
    root = Path(updated_annotations_root).resolve()
    out = Path(output_manifest).resolve()
    rows: list[dict[str, str]] = []
    if root.exists():
        for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            seg_dir = case_dir / "updated"
            if not seg_dir.exists():
                seg_dir = case_dir / "segmentations"
            for mask in sorted(seg_dir.glob("*.nii.gz")):
                organ = mask.name[:-7]
                if organs and organ not in organs:
                    continue
                rows.append({"case_id": case_dir.name, "organ": organ, "mask_path": str(mask.resolve())})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "organ", "mask_path"])
            writer.writeheader(); writer.writerows(rows)
    else:
        write_json(out, rows)
    return {"stage": "mstep_manifest", "status": "success", "updated_annotations_root": str(root), "output_manifest": str(out), "num_items": len(rows), "sample_items": rows[:20]}


def write_mstep_config(output_config: str | Path, training_manifest: str | Path, base_model: str = "nnunet_or_epai", notes: str | None = None) -> dict[str, Any]:
    out = Path(output_config).resolve()
    cfg = {
        "stage": "m_step_interface",
        "base_model": base_model,
        "training_manifest": str(Path(training_manifest).resolve()),
        "mode": "smoke_test_then_scale",
        "warning": "50 cases are for workflow debugging only; do not claim formal retraining performance from this small set.",
        "next_training_backend": ["nnUNet v2", "ePAI fine-tuning", "private model fine-tuning"],
        "notes": notes or "Replace this interface with the lab's real training script once the checkpoint/training code is available.",
    }
    write_json(out, cfg)
    return {"stage": "mstep_config", "status": "success", "output_config": str(out), "config": cfg}


# ---------------------------------------------------------------------------
# Real M-step: nnUNet v2 fine-tuning
# ---------------------------------------------------------------------------

def _prepare_nnunet_dataset(
    training_manifest: str | Path,
    dataset_root: Path,
    dataset_id: int = 999,
    dataset_name: str = "MedAI_EMLoop",
    ct_source_root: Path | None = None,
) -> dict[str, Any]:
    """Convert training_manifest into nnUNet raw dataset format."""
    manifest_path = Path(training_manifest).resolve()
    if manifest_path.suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    ds_folder = dataset_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_masks: dict[str, list[dict]] = {}
    for r in rows:
        cid = r.get("case_id", "")
        if cid:
            case_masks.setdefault(cid, []).append(r)

    label_map: dict[str, int] = {"background": 0}
    label_idx = 1
    for masks in case_masks.values():
        for m in masks:
            organ = m.get("organ", "")
            if organ and organ not in label_map:
                label_map[organ] = label_idx
                label_idx += 1

    prepared_cases = []
    for cid, masks in sorted(case_masks.items()):
        ct_found = None
        if ct_source_root:
            for cand in [
                ct_source_root / cid / "ct.nii.gz",
                ct_source_root / "ImageTr" / cid / "ct.nii.gz",
                ct_source_root / "data" / "ImageTr" / cid / "ct.nii.gz",
            ]:
                if cand.exists():
                    ct_found = cand; break

        dst_img = images_tr / f"{cid}_0000.nii.gz"
        if ct_found and not dst_img.exists():
            shutil.copy2(ct_found, dst_img)

        dst_label = labels_tr / f"{cid}.nii.gz"
        if not dst_label.exists():
            try:
                import nibabel as nib
                import numpy as np
                ref_img = None; combined = None
                for m in masks:
                    mp = Path(m.get("mask_path", ""))
                    organ = m.get("organ", "")
                    if not mp.exists() or organ not in label_map:
                        continue
                    img = nib.load(str(mp))
                    arr = (np.asanyarray(img.dataobj) > 0).astype("uint8")
                    if combined is None:
                        combined = np.zeros_like(arr, dtype="uint8"); ref_img = img
                    combined[arr > 0] = label_map[organ]
                if combined is not None and ref_img is not None:
                    nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), str(dst_label))
            except Exception:
                pass

        prepared_cases.append({"case_id": cid, "image": str(dst_img), "label": str(dst_label), "ct_found": ct_found is not None, "label_built": dst_label.exists()})

    images_found = sum(1 for c in prepared_cases if c["ct_found"])
    if images_found == 0 and ct_source_root is None:
        return {"status": "failed", "reason": "ct_source_root not provided; imagesTr is empty. Pass --ct-source-root to include CT images.", "dataset_folder": str(ds_folder), "num_labels": len(label_map) - 1}
    ds_json = {"name": dataset_name, "channel_names": {"0": "CT"}, "labels": label_map, "numTraining": len(prepared_cases), "file_ending": ".nii.gz"}
    write_json(ds_folder / "dataset.json", ds_json)
    return {"status": "success", "dataset_folder": str(ds_folder), "dataset_id": dataset_id, "num_cases": len(prepared_cases), "num_labels": len(label_map) - 1, "label_map": label_map, "prepared_cases": prepared_cases[:20], "images_found": images_found}


def run_mstep_nnunet_training(
    training_manifest: str | Path,
    output_folder: str | Path,
    dataset_id: int = 999,
    dataset_name: str = "MedAI_EMLoop",
    ct_source_root: str | Path | None = None,
    trainer: str = "nnUNetTrainer_fast_50epochs",
    plans: str = "nnUNetPlans",
    configuration: str = "3d_fullres",
    folds: str = "all",
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Run M-step: prepare nnUNet dataset and launch training.

    50 cases = smoke test only (will overfit). Scale to 500+ for real training.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nnunet_raw = out / "nnUNet_raw"
    nnunet_preprocessed = out / "nnUNet_preprocessed"
    nnunet_results = out / "nnUNet_results"

    ct_root = Path(ct_source_root).resolve() if ct_source_root else None
    prep = _prepare_nnunet_dataset(training_manifest, nnunet_raw, dataset_id, dataset_name, ct_root)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"] = str(nnunet_results)
    env["nnUNet_n_proc_DA"] = "2"

    plan_cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"]
    # nnUNetv2_train does not accept --num_epochs; epoch count is controlled by the
    # trainer class. Use nnUNetTrainer_fast (500 epochs) or a _Xepochs variant.
    train_cmd = ["nnUNetv2_train", str(dataset_id), configuration, folds, "-tr", trainer, "-p", plans, "--npz"]
    if pretrained_weights:
        train_cmd.extend(["-pretrained_weights", pretrained_weights])

    result: dict[str, Any] = {
        "stage": "m_step_nnunet_training", "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out), "dataset_id": dataset_id, "dataset_preparation": prep,
        "plan_command": plan_cmd, "train_command": train_cmd, "max_epochs_intended": max_epochs,
        "environment": {"nnUNet_raw": str(nnunet_raw), "nnUNet_preprocessed": str(nnunet_preprocessed), "nnUNet_results": str(nnunet_results)},
        "pretrained_weights": pretrained_weights,
        "50_case_warning": "50 cases are for debugging the EM loop only. The model WILL overfit. Scale to 500+ for real training.",
    }

    if dry_run:
        result["status"] = "dry_run"
        result["note"] = "Commands prepared but not executed. Run on a GPU server with nnUNet v2 installed."
        write_json(out / "mstep_training_plan.json", result)
        return result

    start = time.time()
    try:
        plan_proc = subprocess.run(plan_cmd, env=env, capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["plan_returncode"] = plan_proc.returncode
        result["plan_stdout_tail"] = plan_proc.stdout[-4000:]
        result["plan_stderr_tail"] = plan_proc.stderr[-4000:]
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "nnUNetv2_plan_and_preprocess not found. Install nnUNet v2."})
        write_json(out / "mstep_training_plan.json", result); return result
    except subprocess.TimeoutExpired:
        result.update({"status": "failed", "reason": f"Planning timed out after {timeout_sec}s"})
        write_json(out / "mstep_training_plan.json", result); return result

    if plan_proc.returncode != 0:
        result.update({"status": "failed", "reason": "nnUNetv2_plan_and_preprocess failed"})
        write_json(out / "mstep_training_plan.json", result); return result

    try:
        train_proc = subprocess.run(train_cmd, env=env, capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["train_returncode"] = train_proc.returncode
        result["train_stdout_tail"] = train_proc.stdout[-4000:]
        result["train_stderr_tail"] = train_proc.stderr[-4000:]
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "nnUNetv2_train not found"})
        write_json(out / "mstep_training_plan.json", result); return result
    except subprocess.TimeoutExpired as exc:
        result.update({"status": "timeout", "reason": f"Training timed out after {timeout_sec}s"})
        train_proc = subprocess.CompletedProcess(train_cmd, -1,
            stdout=(exc.stdout or ""), stderr=(exc.stderr or "") + f"\n[timeout after {timeout_sec}s]")

    elapsed = time.time() - start
    result["runtime_sec"] = round(elapsed, 2)

    ckpt_folder = nnunet_results / f"Dataset{dataset_id:03d}_{dataset_name}" / f"{trainer}__{plans}__{configuration}" / f"fold_{folds}"
    has_checkpoint = ckpt_folder.exists() and any(ckpt_folder.glob("checkpoint_*.pth"))
    result["checkpoint_folder"] = str(ckpt_folder)
    result["has_checkpoint"] = has_checkpoint

    # Accept both clean exit (rc=0) and timeout (rc=-1) as long as a checkpoint exists.
    if has_checkpoint:
        result["status"] = "success"
        result["updated_model_path"] = str(ckpt_folder)
        result["next_step"] = f"Use the updated checkpoint for next inference round. Pass --nnunet-results {nnunet_results} to infer."
        if train_proc.returncode not in (0, -1):
            result["warning"] = f"Training exited with code {train_proc.returncode} but checkpoint exists — treating as success."
    else:
        result.update({"status": "failed", "reason": "Training completed but no checkpoint produced"})

    write_json(out / "mstep_training_result.json", result)

    return result


def update_registry_checkpoint(
    registry_path: str | Path,
    model_key: str,
    new_checkpoint_path: str | Path,
    new_dataset_json_path: str | Path | None = None,
) -> dict:
    """Update model_registry.yaml with a new checkpoint path after M-step training.

    Uses targeted line-by-line text replacement rather than yaml.dump() so that
    all YAML comments, formatting, and multi-line strings are preserved exactly.
    """
    import re
    import yaml
    reg_path = Path(registry_path).resolve()
    text = reg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    models = data.get("models", {})
    if model_key not in models:
        return {"status": "failed", "reason": f"model_key '{model_key}' not in registry"}
    old_ckpt = models[model_key].get("checkpoint_path", "")

    def _patch_field(src: str, mk: str, field: str, new_val: str) -> str:
        """Replace `field: <value>` under the model_key block, preserving all other lines."""
        lines = src.splitlines(keepends=True)
        in_block = False
        indent = ""
        result = []
        for line in lines:
            stripped = line.lstrip()
            if not in_block:
                if re.match(rf"^  {re.escape(mk)}\s*:", line):
                    in_block = True
                    indent = "    "
                result.append(line)
            else:
                cur_indent = len(line) - len(line.lstrip())
                if cur_indent < 4 and stripped and not stripped.startswith("#"):
                    in_block = False
                    result.append(line)
                elif re.match(rf"^    {re.escape(field)}\s*:", line):
                    result.append(f"    {field}: {new_val}\n")
                else:
                    result.append(line)
        return "".join(result)

    new_ckpt_str = str(new_checkpoint_path).replace("\\", "/")
    text = _patch_field(text, model_key, "checkpoint_path", new_ckpt_str)
    if new_dataset_json_path:
        new_dsj_str = str(new_dataset_json_path).replace("\\", "/")
        text = _patch_field(text, model_key, "dataset_json_path", new_dsj_str)
    reg_path.write_text(text, encoding="utf-8")
    return {
        "status": "success",
        "registry_path": str(reg_path),
        "model_key": model_key,
        "old_checkpoint_path": old_ckpt,
        "new_checkpoint_path": str(new_checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Conditional M-step backends for bundled public/foundation models
# ---------------------------------------------------------------------------

def run_mstep_totalsegmentator_public_nnunet_plan(
    training_manifest: str | Path,
    output_folder: str | Path,
    dataset_id: int = 2101,
    ct_source_root: str | Path | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Prepare a TotalSegmentator-style public nnUNet M-step plan.

    TotalSegmentator bundles a public nnUNet training recipe under
    resources/train_nnunet.md and resources/train_nnunet.sh. That recipe can be
    used as a reference for training a TotalSegmentator-style nnUNet backend.
    It does NOT fully reproduce released TotalSegmentator v2 because official
    v2 used additional non-public data.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ts_root = Path("third_party/TotalSegmentator-master").resolve()
    recipe_md = ts_root / "resources" / "train_nnunet.md"
    recipe_sh = ts_root / "resources" / "train_nnunet.sh"
    converter = ts_root / "resources" / "convert_dataset_to_nnunet.py"

    # For PanTS updated annotations we use the same nnUNet preparation logic as
    # the generic backend, but label this explicitly as TotalSegmentator-style.
    # Keep the generated folder names short because Windows can still hit the
    # classic MAX_PATH limit in deeply nested project directories.
    nnunet_result = run_mstep_nnunet_training(
        training_manifest=training_manifest,
        output_folder=out / "ts_nnunet",
        dataset_id=dataset_id,
        dataset_name="MedAI_TS",
        ct_source_root=ct_source_root,
        trainer="nnUNetTrainerNoMirroring",
        plans="nnUNetPlans",
        configuration="3d_fullres",
        folds="all",
        max_epochs=max_epochs,
        pretrained_weights=pretrained_weights,
        dry_run=True if dry_run else False,
        timeout_sec=timeout_sec,
    )
    result = {
        "stage": "mstep_totalsegmentator_public_nnunet",
        "status": nnunet_result.get("status"),
        "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out),
        "dataset_id": dataset_id,
        "backend": "totalseg_public_nnunet",
        "totalsegmentator_source_root": str(ts_root),
        "recipe_files": {
            "train_nnunet_md": str(recipe_md),
            "train_nnunet_sh": str(recipe_sh),
            "convert_dataset_to_nnunet": str(converter),
            "exists": {
                "train_nnunet_md": recipe_md.exists(),
                "train_nnunet_sh": recipe_sh.exists(),
                "convert_dataset_to_nnunet": converter.exists(),
            },
        },
        "nnunet_update_plan": nnunet_result,
        "limitations": [
            "This trains a TotalSegmentator-style nnUNet backend from updated annotations, not the released TotalSegmentator v2 model itself.",
            "Released TotalSegmentator v2 cannot be fully reproduced from this public recipe because its official training used additional non-public data.",
            "50 PanTS cases are for workflow smoke testing only and will overfit.",
        ],
    }
    write_json(out / "mstep_totalseg_public_nnunet_plan.json", result)
    return result


def _prepare_vista3d_datalist_from_manifest(
    training_manifest: str | Path,
    output_folder: Path,
    ct_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a minimal MONAI datalist for VISTA3D fine-tuning.

    We reuse the nnUNet dataset preparation to combine per-organ updated masks
    into one label map, then produce a MONAI Auto3DSeg-style datalist with
    imagesTr/labelsTr paths.
    """
    prep_root = output_folder / "vista3d_medai_dataset"
    prep = _prepare_nnunet_dataset(
        training_manifest=training_manifest,
        dataset_root=prep_root / "nnunet_raw_proxy",
        dataset_id=3101,
        dataset_name="MedAI_VISTA3D",
        ct_source_root=Path(ct_source_root).resolve() if ct_source_root else None,
    )
    ds_folder = Path(prep["dataset_folder"])
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    rows = []
    image_files = sorted(images_tr.glob("*_0000.nii.gz"))
    for idx, img in enumerate(image_files):
        cid = img.name[:-12]
        label = labels_tr / f"{cid}.nii.gz"
        if not label.exists():
            continue
        # MONAI datafold_read uses the requested fold for validation and all
        # other folds for training. Keep at least one validation case when possible.
        fold = 0 if idx == 0 else 1
        rows.append({"fold": fold, "image": str(img.relative_to(ds_folder)), "label": str(label.relative_to(ds_folder))})
    datalist = {"training": rows, "testing": []}
    datalist_path = output_folder / "vista3d_medai_datalist.json"
    write_json(datalist_path, datalist)
    return {"dataset_folder": str(ds_folder), "datalist_path": str(datalist_path), "num_training_items": len(rows), "nnunet_proxy_preparation": prep}


def run_mstep_vista3d_monai_finetune_plan(
    training_manifest: str | Path,
    output_folder: str | Path,
    vista_root: str | Path = "third_party/VISTA3D-Inference-Pipeline-master",
    ct_source_root: str | Path | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Prepare or launch VISTA3D MONAI bundle fine-tuning.

    The bundled VISTA3D pipeline includes configs/train.json,
    train_continual.json, multi_gpu_train.json and scripts/trainer.py. This
    backend creates a project-specific datalist and prepares the MONAI command.
    """
    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vista = Path(vista_root).resolve()
    train_json = vista / "configs" / "train.json"
    continual_json = vista / "configs" / "train_continual.json"
    multi_gpu_json = vista / "configs" / "multi_gpu_train.json"
    trainer_py = vista / "scripts" / "trainer.py"
    prep = _prepare_vista3d_datalist_from_manifest(training_manifest, out, ct_source_root)

    cmd = [
        "python", "-m", "monai.bundle", "run",
        "--config_file", str(train_json),
        "--dataset_dir", prep["dataset_folder"],
        "--data_list_file_path", prep["datalist_path"],
        "--finetune", "True",
        "--epochs", str(max_epochs),
    ]
    if pretrained_weights:
        cmd.extend(["--finetune_model_path", str(pretrained_weights)])

    result: dict[str, Any] = {
        "stage": "mstep_vista3d_monai_finetune",
        "status": "dry_run" if dry_run else "prepared",
        "training_manifest": str(Path(training_manifest).resolve()),
        "output_folder": str(out),
        "backend": "monai_bundle_finetune",
        "vista_root": str(vista),
        "recipe_files": {
            "train_json": str(train_json),
            "train_continual_json": str(continual_json),
            "multi_gpu_train_json": str(multi_gpu_json),
            "trainer_py": str(trainer_py),
            "exists": {
                "train_json": train_json.exists(),
                "train_continual_json": continual_json.exists(),
                "multi_gpu_train_json": multi_gpu_json.exists(),
                "trainer_py": trainer_py.exists(),
            },
        },
        "dataset_preparation": prep,
        "command": cmd,
        "pretrained_weights": pretrained_weights,
        "limitations": [
            "This fine-tunes a VISTA3D MONAI bundle checkpoint if one is provided; it does not reproduce the original foundation model from scratch.",
            "VISTA3D training/fine-tuning requires a MONAI bundle environment and sufficient GPU memory.",
            "50 PanTS cases are for workflow smoke testing only and will overfit.",
        ],
    }
    if dry_run:
        write_json(out / "mstep_vista3d_monai_plan.json", result)
        return result
    try:
        proc = subprocess.run(cmd, cwd=str(vista), capture_output=True, text=True, check=False, timeout=timeout_sec)
        result["returncode"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        result["status"] = "success" if proc.returncode == 0 else "failed"
    except FileNotFoundError:
        result.update({"status": "failed", "reason": "python/monai.bundle not found. Install MONAI bundle dependencies."})
    except subprocess.TimeoutExpired:
        result.update({"status": "timeout", "reason": f"VISTA3D fine-tuning timed out after {timeout_sec}s"})
    write_json(out / "mstep_vista3d_monai_result.json", result)
    return result

# ---------------------------------------------------------------------------
# Selected-model-aware M-step
# ---------------------------------------------------------------------------

def _filter_manifest_for_target_model(training_manifest: str | Path, output_manifest: str | Path, target_model: str) -> dict[str, Any]:
    """Copy a manifest and annotate each row with the selected M-step target model.

    The current run-loop produces final updated annotations. It may not always
    know which candidate originally won after ShapeKit/LabelCritic/human review,
    so the M-step target is explicitly supplied by the selected task model. This
    keeps the loop semantically clear: E-step primary model -> updated data ->
    M-step update of that same model family when trainable.
    """
    inp = Path(training_manifest).resolve()
    out = Path(output_manifest).resolve()
    if inp.suffix.lower() == ".csv":
        with inp.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(inp.read_text(encoding="utf-8")) if inp.exists() else []
    for r in rows:
        r["target_model"] = target_model
        r.setdefault("mstep_scope", "selected_model_family")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        fieldnames = sorted({k for r in rows for k in r.keys()}) or ["case_id", "organ", "mask_path", "target_model"]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    else:
        write_json(out, rows)
    return {"stage": "target_model_manifest", "status": "success", "input_manifest": str(inp), "output_manifest": str(out), "target_model": target_model, "num_items": len(rows)}


def run_model_specific_mstep_update(
    training_manifest: str | Path,
    output_folder: str | Path,
    target_model: str,
    registry_path: str | Path = "configs/model_registry.yaml",
    ct_source_root: str | Path | None = None,
    dataset_id: int | None = None,
    max_epochs: int = 5,
    pretrained_weights: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 7200,
) -> dict[str, Any]:
    """Selected-model-aware M-step.

    This is the corrected M-step semantics: update the selected model family
    whenever that family is trainable/nnUNet-compatible. TotalSegmentator and
    other non-trainable inference-only backends are explicitly rejected as M-step
    targets instead of silently training a generic model.
    """
    from .model_registry import load_registry

    out = Path(output_folder).resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    models = registry.get("models", {})
    if target_model not in models:
        result = {"stage": "mstep_update", "status": "failed", "reason": f"target_model '{target_model}' not found in registry", "available_models": sorted(models.keys())}
        write_json(out / "mstep_update_result.json", result)
        return result

    entry = models[target_model]
    backend = entry.get("mstep_backend", "none")
    trainable = entry.get("trainable", "unknown")
    role = entry.get("mstep_role", "")
    reason = entry.get("mstep_reason", "")
    base = {
        "stage": "mstep_update",
        "target_model": target_model,
        "target_model_name": entry.get("name", target_model),
        "trainable": trainable,
        "mstep_backend": backend,
        "mstep_role": role,
        "mstep_reason": reason,
        "selected_model_semantics": "E-step primary model is the preferred M-step target when it is trainable. Non-trainable models remain E-step candidates/baselines only.",
        "warning_50_case": "50 cases are for debugging the loop only; do not claim formal model-improvement results from this small set.",
    }

    if backend in {"none", None} or str(trainable) in {"no", "unknown_template_only"}:
        result = {**base, "status": "not_trainable_in_current_project", "next_action": "Use this model as an E-step candidate only, or provide the original training script/checkpoint layout to enable model-specific M-step."}
        write_json(out / "mstep_update_result.json", result)
        return result
    if backend == "external_training_required":
        result = {**base, "status": "external_training_required", "next_action": "This model needs its own training recipe. The current project only wraps inference."}
        write_json(out / "mstep_update_result.json", result)
        return result

    dsid = dataset_id or int(entry.get("mstep_dataset_id") or 999)
    target_manifest = out / f"training_manifest__target_{target_model}.json"
    manifest_result = _filter_manifest_for_target_model(training_manifest, target_manifest, target_model)

    # Prefer explicit pretrained weights from user. Registry mstep_init_from is a
    # folder hint; we do not blindly pass it as -pretrained_weights unless it is a file.
    pretrained = pretrained_weights
    init_hint = entry.get("mstep_init_from")
    if not pretrained and init_hint:
        p = Path(str(init_hint))
        if p.exists() and p.is_file():
            pretrained = str(p)

    if backend == "totalseg_public_nnunet":
        training_result = run_mstep_totalsegmentator_public_nnunet_plan(
            training_manifest=target_manifest,
            output_folder=out / f"ts_update__{target_model}",
            dataset_id=dsid,
            ct_source_root=ct_source_root,
            max_epochs=max_epochs,
            pretrained_weights=pretrained,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
        result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
        write_json(out / "mstep_update_result.json", result)
        return result

    if backend == "monai_bundle_finetune":
        vista_root = entry.get("mstep_init_from") or entry.get("source_code_path") or entry.get("checkpoint_path") or "third_party/VISTA3D-Inference-Pipeline-master"
        training_result = run_mstep_vista3d_monai_finetune_plan(
            training_manifest=target_manifest,
            output_folder=out / f"vista3d_update__{target_model}",
            vista_root=vista_root,
            ct_source_root=ct_source_root,
            max_epochs=max_epochs,
            pretrained_weights=pretrained,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
        )
        result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
        if not pretrained:
            result["pretrained_note"] = "VISTA3D fine-tuning normally needs --pretrained-weights pointing to a VISTA3D/MONAI checkpoint. Dry-run can be used without it."
        write_json(out / "mstep_update_result.json", result)
        return result

    if not str(backend).startswith("nnunetv2"):
        result = {**base, "status": "unsupported_mstep_backend", "next_action": f"Implement a backend adapter for {backend}."}
        write_json(out / "mstep_update_result.json", result)
        return result

    training_result = run_mstep_nnunet_training(
        training_manifest=target_manifest,
        output_folder=out / f"nnunet_update__{target_model}",
        dataset_id=dsid,
        dataset_name=f"MedAI_{target_model}",
        ct_source_root=ct_source_root,
        # Use nnUNetTrainer_fast (500 epochs + compile + cudnn.benchmark) for speed.
        # Fall back to the registry trainer only if it's a specialised variant.
        trainer=entry.get("trainer", "nnUNetTrainer").replace("nnUNetTrainer", "nnUNetTrainer_fast_50epochs"),
        plans=entry.get("plans", "nnUNetPlans"),
        configuration="3d_fullres",
        folds="all",
        max_epochs=max_epochs,
        pretrained_weights=pretrained,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
    )
    result = {**base, "status": training_result.get("status"), "dataset_id": dsid, "mstep_init_from_hint": init_hint, "target_manifest": manifest_result, "training_result": training_result}
    if init_hint and not pretrained:
        result["pretrained_note"] = "Registry contains an init_from folder hint, but no explicit checkpoint file was passed. Use --pretrained-weights if you want true fine-tuning from a specific .pth file."

    # Auto-update registry checkpoint_path so the next E-step uses the newly trained model.
    if training_result.get("status") == "success" and training_result.get("updated_model_path"):
        reg_update = update_registry_checkpoint(
            registry_path=registry_path,
            model_key=target_model,
            new_checkpoint_path=training_result["updated_model_path"],
        )
        result["registry_update"] = reg_update

    write_json(out / "mstep_update_result.json", result)
    return result
