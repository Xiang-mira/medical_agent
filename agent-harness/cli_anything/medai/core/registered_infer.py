from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .model_registry import get_model_entry, load_registry
from .totalseg_runner import run_totalsegmentator


def _q(path: str | Path) -> str:
    text = str(path)
    # Double quotes work in Windows cmd and POSIX shells for local paths.
    if text.startswith('"') and text.endswith('"'):
        return text
    return '"' + text.replace('"', r'\"') + '"'


def _mask_summary(seg_out: Path) -> tuple[int, list[str]]:
    masks = sorted([p.name for p in seg_out.glob("*.nii.gz")]) if seg_out.exists() else []
    return len(masks), masks[:80]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def run_registered_model(
    image_path: str | Path,
    output_folder: str | Path,
    model_key: str,
    registry_path: str | Path = "configs/model_registry.yaml",
    case_id: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    fast: bool = True,
    device: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one model through the registry and normalize the expected output folder.

    All model templates are expected to write organ masks into:
        <output_folder>/<case_id>/segmentations/*.nii.gz
    """
    image = Path(image_path).resolve()
    registry_file = Path(registry_path).resolve()
    registry = load_registry(registry_file)
    entry = get_model_entry(registry, model_key)
    if case_id is None:
        case_id = image.parent.name or image.stem

    case_out = Path(output_folder).resolve() / case_id
    seg_out = case_out / "segmentations"

    # Skip models whose checkpoint is known-corrupted or otherwise unavailable.
    model_status = entry.get("status", "")
    if model_status in ("checkpoint_corrupted", "unavailable"):
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "status": "skipped", "reason": f"model status is '{model_status}' — checkpoint not usable",
            "case_id": case_id,
        }

    checkpoint_path_value = entry.get("checkpoint_path", "")
    if checkpoint_path_value:
        checkpoint_path = Path(str(checkpoint_path_value)).resolve()
        if _is_relative_to(case_out, checkpoint_path):
            return {
                "stage": "infer", "backend": "registered", "model_key": model_key,
                "status": "failed", "reason": "Refusing to write inference output inside checkpoint_path. Use a separate outputs/ folder so linked teacher checkpoints remain read-only.",
                "case_id": case_id, "output_folder": str(case_out), "checkpoint_path": str(checkpoint_path),
            }
    seg_out.mkdir(parents=True, exist_ok=True)

    runner = entry.get("runner", "command_template")
    if runner == "builtin_totalsegmentator":
        # TotalSegmentator uses "gpu"/"cpu" not "cuda"
        ts_device = "gpu" if device and device.startswith("cuda") else (device or "gpu")
        return run_totalsegmentator(
            str(image), str(Path(output_folder).resolve()), case_id=case_id,
            fast=fast, task=None, roi_preset="none", roi_subset=None,
            device=ts_device, dry_run=dry_run, timeout_sec=timeout_sec,
        ) | {"model_key": model_key, "registry_path": str(registry_file)}

    template = entry.get("command_template")
    if not template:
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "status": "failed", "reason": "registry entry has no command_template",
            "registry_path": str(registry_file), "case_id": case_id,
        }

    checkpoint_path = checkpoint_path_value
    # Registry file is configs/model_registry.yaml; checkpoint paths like
    # "checkpoints/..." are relative to the project root (one level up).
    project_root = registry_file.parent.parent

    def _resolve_from_registry(p: str) -> Path:
        """Resolve a path relative to the project root (parent of configs/)."""
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (project_root / pp).resolve()

    context = {
        "image": _q(image),
        "output": _q(seg_out),
        "case_output": _q(case_out),
        "output_folder": _q(Path(output_folder).resolve()),
        "case_id": case_id,
        "checkpoint_path": _q(_resolve_from_registry(checkpoint_path)) if checkpoint_path else "",
        "checkpoint_root": _q(_resolve_from_registry(registry.get("checkpoint_root", "checkpoints"))),
        "model_key": model_key,
        "device": device or "",
    }
    # Expose registry entry fields to command_template. Path-like fields are quoted and
    # resolved relative to the project working directory so templates can stay compact.
    for k, v in entry.items():
        if k in context or isinstance(v, (dict, list)) or v is None:
            continue
        if isinstance(v, bool):
            context[k] = str(v).lower()
        elif isinstance(v, (int, float)):
            context[k] = str(v)
        else:
            sv = str(v)
            if k.endswith(("_path", "_root", "_json", "_folder")) or "/" in sv or "\\" in sv:
                context[k] = _q(_resolve_from_registry(sv))
            else:
                context[k] = sv
    if extra_context:
        context.update(extra_context)
    try:
        command = template.format(**context)
    except Exception as exc:
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "status": "failed", "reason": f"cannot render command_template: {exc}",
            "template": template, "case_id": case_id,
        }

    if dry_run:
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "model_name": entry.get("name", model_key), "status": "dry_run",
            "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
            "command": command, "registry_path": str(registry_file), "private_checkpoint": entry.get("private_checkpoint"),
            "notes": entry.get("notes"),
        }

    # Auto-apply ePAI all-organ patch before inference so the model outputs 25 classes
    # instead of only pancreas+tumor (the default filtering the teacher turned off).
    if model_key == "epai_20250421":
        try:
            patch_script = Path(__file__).resolve().parents[4] / "scripts" / "patch_epai_enable_all_organs.py"
            if patch_script.exists():
                subprocess.run([sys.executable, str(patch_script)], capture_output=True, check=False, timeout=30)
        except Exception:
            pass

    start = time.time()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, shell=True, timeout=timeout_sec)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(shlex.split(command), 124, stdout=exc.stdout or "", stderr=(exc.stderr or "") + f"\n[registered_infer] Timeout after {timeout_sec}s.")
        timed_out = True
    elapsed = time.time() - start
    num_masks, sample_masks = _mask_summary(seg_out)
    status = "timed_out" if timed_out else ("success" if completed.returncode == 0 and num_masks > 0 else "failed")
    return {
        "stage": "infer", "backend": "registered", "model_key": model_key,
        "model_name": entry.get("name", model_key), "status": status,
        "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
        "command": command, "registry_path": str(registry_file), "return_code": completed.returncode,
        "timed_out": timed_out, "timeout_sec": timeout_sec, "runtime_sec": round(elapsed, 3),
        "num_masks": num_masks, "sample_masks": sample_masks,
        "stdout_tail": (completed.stdout or "")[-4000:], "stderr_tail": (completed.stderr or "")[-4000:],
        "private_checkpoint": entry.get("private_checkpoint"),
    }
