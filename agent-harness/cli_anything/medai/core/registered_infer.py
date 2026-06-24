from __future__ import annotations

import datetime
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .model_registry import get_model_entry, load_registry
from .totalseg_runner import run_totalseg_with_contract, run_totalsegmentator

_PREDICT_SCRIPTS = (
    "nnunetv2_predict_and_split.py",
    "atlasnet_predict_and_split.py",
    "vista3d_predict_and_split.py",
    "unest_predict_and_split.py",
)

_LICENSED_TOTALSEG_TASKS: frozenset[str] = frozenset({
    "heartchambers_highres", "appendicular_bones", "appendicular_bones_mr",
    "tissue_types", "tissue_types_mr", "tissue_4_types", "face", "face_mr",
    "brain_structures", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr",
    "coronary_arteries", "coronary_arteries_LEGACY", "aortic_sinuses",
    "vertebrae_body",
})


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


def _write_run_meta(per_model_dir: Path, model_key: str, recipe: str, result: dict) -> None:
    run_meta = {
        "model_key": model_key,
        "recipe": recipe,
        "status": result.get("status", "unknown"),
        "duration_sec": result.get("runtime_sec"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "return_code": result.get("return_code"),
    }
    try:
        (per_model_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    except Exception:
        pass


def _write_inference_summary(case_out: Path, result: dict[str, Any], seg_out: Path | None = None) -> None:
    """Write a uniform per-model inference summary for audit automation."""
    summary = dict(result)
    if seg_out is not None:
        num_masks, sample_masks = _mask_summary(seg_out)
        summary["segmentation_output"] = str(seg_out)
        summary["num_masks"] = num_masks
        summary["sample_masks"] = sample_masks
    try:
        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / "inference_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _totalseg_subtasks_for_context(subtask_config: dict | None, extra_context: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return official TotalSegmentator subtasks safe for the current request.

    TotalSegmentator has per-task license constraints.  When the caller provides
    requested organs, we run only the subtasks that can produce those organs and
    skip academic-license tasks unless explicitly allowed.
    """
    config_subtasks = (subtask_config or {}).get("subtasks", {}) or {}
    allow_licensed = bool(extra_context.get("allow_licensed_totalseg"))
    requested_organs = {str(x) for x in (extra_context.get("requested_organs") or []) if str(x)}
    requested_subtasks = []
    if extra_context.get("subtasks"):
        requested_subtasks.extend(str(x) for x in extra_context.get("subtasks") or [] if str(x))
    if extra_context.get("subtask"):
        requested_subtasks.append(str(extra_context["subtask"]))

    if not requested_subtasks and requested_organs and config_subtasks:
        for task_name, task_entry in config_subtasks.items():
            task_organs = {str(x) for x in (task_entry.get("organs") or [])}
            if requested_organs & task_organs:
                requested_subtasks.append(str(task_entry.get("task") or task_name))

    if not requested_subtasks:
        requested_subtasks = list(config_subtasks.keys()) or ["total"]

    selected: list[str] = []
    skipped: list[dict[str, Any]] = []
    for task_name in dict.fromkeys(requested_subtasks):
        task_entry = config_subtasks.get(task_name, {})
        license_required = bool(task_entry.get("license_required")) or task_name in _LICENSED_TOTALSEG_TASKS
        if license_required and not allow_licensed:
            skipped.append({
                "subtask": task_name,
                "reason": "academic license required; skipped by default",
            })
            continue
        selected.append(task_name)

    return selected, skipped


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
    """Run one model through the registry and write the per_model output contract.

    Output contract (per model_key):
        outputs/<case>/per_model/<model_key>/combined_labels.nii.gz
        outputs/<case>/per_model/<model_key>/local_labels.json
        outputs/<case>/per_model/<model_key>/run_meta.json
    Legacy segmentation masks also written to:
        outputs/<case>/segmentations/*.nii.gz
    """
    image = Path(image_path).resolve()
    registry_file = Path(registry_path).resolve()
    registry = load_registry(registry_file)
    entry = get_model_entry(registry, model_key)
    if case_id is None:
        case_id = image.parent.name or image.stem

    output_root = Path(output_folder).resolve()
    extra_context = extra_context or {}
    case_output_override = extra_context.get("case_output_override")
    segmentation_output_override = extra_context.get("segmentation_output_override")
    case_out = Path(case_output_override).resolve() if case_output_override else output_root / case_id
    seg_out = Path(segmentation_output_override).resolve() if segmentation_output_override else case_out / "segmentations"

    # Skip models explicitly disabled in the registry.
    if entry.get("enabled") is False:
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "status": "skipped", "reason": "model is disabled (enabled=false in registry)",
            "case_id": case_id,
        }

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

    # Per-model output contract directory.
    per_model_dir = case_out / "per_model" / model_key
    per_model_dir.mkdir(parents=True, exist_ok=True)

    recipe = entry.get("recipe", "")
    runner = entry.get("runner", "command_template")

    if runner == "builtin_totalsegmentator":
        # Load subtask config so only request-relevant official subtasks are run.
        subtask_config_path = registry_file.parent / "totalseg_subtask_organs.json"
        subtask_config: dict | None = None
        if subtask_config_path.exists():
            try:
                subtask_config = json.loads(subtask_config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        selected_subtasks, skipped_subtasks = _totalseg_subtasks_for_context(subtask_config, extra_context)
        if not selected_subtasks:
            return {
                "stage": "infer",
                "backend": "TotalSegmentator",
                "model_key": model_key,
                "status": "skipped",
                "case_id": case_id,
                "reason": "No TotalSegmentator subtasks remain after license/request filtering.",
                "skipped_subtasks": skipped_subtasks,
                "registry_path": str(registry_file),
            }
        ts_device = "gpu" if device and device.startswith("cuda") else (device or "gpu")
        result = run_totalseg_with_contract(
            image, per_model_dir, case_out,
            subtask_config=subtask_config,
            subtasks=selected_subtasks,
            fast=fast, device=ts_device,
            dry_run=dry_run, timeout_sec=None,
            case_id=case_id,
        )
        result["model_key"] = model_key
        result["registry_path"] = str(registry_file)
        result["selected_subtasks"] = selected_subtasks
        result["skipped_subtasks"] = skipped_subtasks
        if not dry_run:
            _write_run_meta(per_model_dir, model_key, recipe, result)
        _write_inference_summary(case_out, result, Path(result.get("segmentation_output", seg_out)))
        return result

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
        "output_folder": _q(output_root),
        "per_model_dir": _q(per_model_dir),
        "case_id": case_id,
        "checkpoint_path": _q(_resolve_from_registry(checkpoint_path)) if checkpoint_path else "",
        "checkpoint_root": _q(_resolve_from_registry(registry.get("checkpoint_root", "checkpoints"))),
        "model_key": model_key,
        "device": device or "",
        "subtask": extra_context.get("subtask") or "",
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
    context.update(extra_context)
    try:
        command = template.format(**context)
    except Exception as exc:
        return {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "status": "failed", "reason": f"cannot render command_template: {exc}",
            "template": template, "case_id": case_id,
        }

    # Inject --per-model-dir for predict scripts that support it but whose templates
    # don't reference {per_model_dir} explicitly (keeps registry templates compact).
    if any(s in command for s in _PREDICT_SCRIPTS) and "--per-model-dir" not in command:
        command += f" --per-model-dir {_q(per_model_dir)}"

    if dry_run:
        result = {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "model_name": entry.get("name", model_key), "status": "dry_run",
            "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
            "per_model_dir": str(per_model_dir),
            "command": command, "registry_path": str(registry_file), "private_checkpoint": entry.get("private_checkpoint"),
            "notes": entry.get("notes"),
        }
        _write_inference_summary(case_out, result, seg_out)
        return result

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
    result = {
        "stage": "infer", "backend": "registered", "model_key": model_key,
        "model_name": entry.get("name", model_key), "status": status,
        "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
        "per_model_dir": str(per_model_dir),
        "command": command, "registry_path": str(registry_file), "return_code": completed.returncode,
        "timed_out": timed_out, "timeout_sec": timeout_sec, "runtime_sec": round(elapsed, 3),
        "num_masks": num_masks, "sample_masks": sample_masks,
        "stdout_tail": (completed.stdout or "")[-4000:], "stderr_tail": (completed.stderr or "")[-4000:],
        "private_checkpoint": entry.get("private_checkpoint"),
    }
    _write_run_meta(per_model_dir, model_key, recipe, result)
    _write_inference_summary(case_out, result, seg_out)
    return result
