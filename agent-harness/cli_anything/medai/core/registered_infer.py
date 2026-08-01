from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .backend_capabilities import profile_runtime_policy
from .model_registry import get_model_entry, load_registry
from .subprocess_utils import subprocess_text
from .totalseg_runner import run_totalseg_with_contract, run_totalsegmentator
from .voxtell_official_predictor import OfficialVoxTellPretrainedAdapter

_PREDICT_SCRIPTS = (
    "nnunetv2_predict_and_split.py",
    "atlasnet_predict_and_split.py",
    "vista3d_predict_and_split.py",
    "unest_predict_and_split.py",
)

_AUXILIARY_MASK_NAMES = {
    "image.nii.gz",
    "zero_mask.nii.gz",
    "combined_labels.nii.gz",
}

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


def _expected_output_organs(entry: dict[str, Any], extra_context: dict[str, Any]) -> list[str]:
    requested = [
        str(x).strip()
        for x in (extra_context.get("requested_organs") or [])
        if str(x).strip()
    ]
    if not requested:
        return []
    covered = {str(x).strip() for x in (entry.get("covered_organs") or []) if str(x).strip()}
    if not covered:
        return requested
    return [organ for organ in requested if organ in covered]


def _classify_mask_outputs(seg_out: Path, expected_organs: list[str]) -> dict[str, Any]:
    names = sorted([p.name for p in seg_out.glob("*.nii.gz")]) if seg_out.exists() else []
    expected_names = [f"{organ}.nii.gz" for organ in expected_organs]
    expected_set = set(expected_names)
    auxiliary = [name for name in names if name in _AUXILIARY_MASK_NAMES]
    expected_present = [name for name in expected_names if name in names]
    expected_missing = [name for name in expected_names if name not in names]
    unexpected_formal = [
        name for name in names
        if name not in _AUXILIARY_MASK_NAMES and (not expected_set or name not in expected_set)
    ]
    formal_masks = expected_present if expected_names else [
        name for name in names if name not in _AUXILIARY_MASK_NAMES
    ]
    return {
        "num_masks": len(names),
        "sample_masks": names[:80],
        "expected_outputs": expected_names,
        "expected_present": expected_present,
        "missing_expected_outputs": expected_missing,
        "auxiliary_outputs": auxiliary,
        "unexpected_formal_outputs": unexpected_formal,
        "formal_mask_count": len(formal_masks),
        "formal_masks": formal_masks[:80],
    }


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


def _resolve_registry_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (project_root / path).resolve()


def _quote_command_value(value: str | Path) -> str:
    text = str(value)
    if "/" in text or "\\" in text:
        return _q(text)
    return text


def _official_voxtell_model_dir(project_root: Path, extra_context: dict[str, Any]) -> Path:
    explicit = extra_context.get("model_dir") or extra_context.get("official_voxtell_model_dir")
    env_value = os.getenv("MEDAI_VOXTELL_MODEL_DIR")
    chosen = explicit or env_value or (project_root / "checkpoints" / "VoxTell" / "voxtell_v1.1")
    return Path(str(chosen)).resolve()


def _official_voxtell_target_config(project_root: Path, extra_context: dict[str, Any]) -> Path:
    explicit = extra_context.get("target_config") or extra_context.get("prompt_target_config")
    chosen = explicit or os.getenv("MEDAI_PROMPT_TARGET_CONFIG") or (project_root / "configs" / "student_3d_prompt_target_organs.json")
    return Path(str(chosen)).resolve()


def _run_official_voxtell_pretrained(
    *,
    image: Path,
    output_root: Path,
    case_out: Path,
    seg_out: Path,
    per_model_dir: Path,
    model_key: str,
    case_id: str,
    dry_run: bool,
    timeout_sec: int,
    device: str | None,
    extra_context: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    profile_name = os.getenv("MEDAI_EXPERIMENT_PROFILE", "")
    runtime_policy = profile_runtime_policy(profile_name)
    voxtell_policy = runtime_policy.get("official_voxtell_pretrained") or {}
    mode = str(voxtell_policy.get("official_voxtell_mode") or "baseline_only")
    requested_organs = [
        str(organ).strip()
        for organ in (extra_context.get("requested_organs") or [])
        if str(organ).strip()
    ]
    prompt_overrides = extra_context.get("prompt_overrides")
    text_encoding_model = extra_context.get("text_encoding_model") or os.getenv("MEDAI_TEXT_ENCODING_MODEL")
    model_dir = _official_voxtell_model_dir(project_root, extra_context)
    target_config = _official_voxtell_target_config(project_root, extra_context)
    try:
        target_doc = json.loads(target_config.read_text(encoding="utf-8"))
        supported_targets = {str(item) for item in target_doc.get("target_organs", [])}
    except Exception:
        supported_targets = set()
    supported_organs = [organ for organ in requested_organs if organ in supported_targets]
    skipped_organs = [organ for organ in requested_organs if organ not in supported_targets]
    if requested_organs and not supported_organs:
        result = {
            "stage": "infer",
            "status": "skipped_unsupported_targets",
            "backend": "official_voxtell_pretrained",
            "model_key": model_key,
            "case_id": case_id,
            "image": str(image),
            "segmentation_output": str(seg_out),
            "requested_organs": requested_organs,
            "supported_organs": [],
            "skipped_unsupported_organs": skipped_organs,
            "target_config": str(target_config),
            "reason": "None of the requested hierarchy/dependency organs belong to the formal VoxTell target space.",
        }
        _write_run_meta(per_model_dir, model_key, "official_voxtell_pretrained_adapter", result)
        _write_inference_summary(case_out, result, seg_out)
        return result
    prompts = supported_organs or None

    adapter = OfficialVoxTellPretrainedAdapter(
        model_dir=model_dir,
        target_config=target_config,
        mode=mode,
        device=device or "cuda",
        text_encoding_model=text_encoding_model,
    )
    result = adapter.segment(
        ct_image=image,
        output_dir=seg_out,
        prompts=prompts,
        dry_run=dry_run,
        timeout_sec=timeout_sec,
        prompt_batch_size=max(1, len(supported_organs)) if supported_organs else 16,
        prompt_overrides=prompt_overrides if isinstance(prompt_overrides, dict) else None,
    )
    result.update({
        "stage": "infer",
        "backend": "official_voxtell_pretrained",
        "model_key": model_key,
        "case_id": case_id,
        "image": str(image),
        "segmentation_output": str(seg_out),
        "per_model_dir": str(per_model_dir),
        "registry_path": str((project_root / "configs" / "model_registry.yaml").resolve()),
        "output_folder": str(output_root),
        "requested_organs": requested_organs,
        "supported_organs": supported_organs,
        "skipped_unsupported_organs": skipped_organs,
        "target_config": str(target_config),
        "model_dir": str(model_dir),
        "text_encoding_model": str(text_encoding_model) if text_encoding_model else None,
        "experiment_profile": runtime_policy.get("experiment_profile"),
    })
    _write_run_meta(per_model_dir, model_key, "official_voxtell_pretrained_adapter", result)
    _write_inference_summary(case_out, result, seg_out)
    return result


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
    if case_id is None:
        case_id = image.parent.name or image.stem

    output_root = Path(output_folder).resolve()
    extra_context = extra_context or {}
    case_output_override = extra_context.get("case_output_override")
    segmentation_output_override = extra_context.get("segmentation_output_override")
    case_out = Path(case_output_override).resolve() if case_output_override else output_root / case_id
    seg_out = Path(segmentation_output_override).resolve() if segmentation_output_override else case_out / "segmentations"
    seg_out.mkdir(parents=True, exist_ok=True)

    # Per-model output contract directory.
    per_model_dir = case_out / "per_model" / model_key
    per_model_dir.mkdir(parents=True, exist_ok=True)

    # project root relative to configs/model_registry.yaml
    project_root = registry_file.parent.parent

    if model_key == "official_voxtell_pretrained":
        return _run_official_voxtell_pretrained(
            image=image,
            output_root=output_root,
            case_out=case_out,
            seg_out=seg_out,
            per_model_dir=per_model_dir,
            model_key=model_key,
            case_id=case_id,
            dry_run=dry_run,
            timeout_sec=timeout_sec,
            device=device,
            extra_context=extra_context,
            project_root=project_root,
        )

    entry = get_model_entry(registry, model_key)

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
        checkpoint_path_raw = Path(str(checkpoint_path_value))
        checkpoint_path = (
            checkpoint_path_raw
            if checkpoint_path_raw.is_absolute()
            else (project_root / checkpoint_path_raw)
        ).resolve()
        if _is_relative_to(case_out, checkpoint_path):
            return {
                "stage": "infer", "backend": "registered", "model_key": model_key,
                "status": "failed", "reason": "Refusing to write inference output inside checkpoint_path. Use a separate outputs/ folder so linked teacher checkpoints remain read-only.",
                "case_id": case_id, "output_folder": str(case_out), "checkpoint_path": str(checkpoint_path),
            }

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

    def _resolve_from_registry(p: str) -> Path:
        """Resolve a path relative to the project root (parent of configs/)."""
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (project_root / pp).resolve()

    python_executable = (
        entry.get("python_executable")
        or extra_context.get("python_executable")
        or os.getenv("MEDAI_PYTHON_EXECUTABLE")
        or sys.executable
    )
    predict_executable = (
        entry.get("predict_executable")
        or extra_context.get("predict_executable")
        or os.getenv("MEDAI_NNUNETV2_PREDICT")
        or "nnUNetv2_predict"
    )
    sitecustomize_path = (
        entry.get("sitecustomize_path")
        or extra_context.get("sitecustomize_path")
        or os.getenv("MEDAI_NNUNET_COMPAT_SITECUSTOMIZE")
        or ""
    )

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
        "python_executable": _quote_command_value(python_executable),
        "predict_executable": _quote_command_value(predict_executable),
        "sitecustomize_path": _quote_command_value(sitecustomize_path) if sitecustomize_path else "",
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

    stripped = command.lstrip()
    leading_ws = command[:len(command) - len(stripped)]
    if stripped.startswith("python "):
        command = f"{leading_ws}{context['python_executable']} {stripped[len('python '):]}"

    # Inject --per-model-dir for predict scripts that support it but whose templates
    # don't reference {per_model_dir} explicitly (keeps registry templates compact).
    if any(s in command for s in _PREDICT_SCRIPTS) and "--per-model-dir" not in command:
        command += f" --per-model-dir {_q(per_model_dir)}"

    if dry_run:
        expected_organs = _expected_output_organs(entry, extra_context)
        result = {
            "stage": "infer", "backend": "registered", "model_key": model_key,
            "model_name": entry.get("name", model_key), "status": "dry_run",
            "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
            "per_model_dir": str(per_model_dir),
            "command": command, "registry_path": str(registry_file), "private_checkpoint": entry.get("private_checkpoint"),
            "expected_outputs": [f"{organ}.nii.gz" for organ in expected_organs],
            "expected_organs": expected_organs,
            "resolved_python": str(python_executable),
            "resolved_predict_executable": str(predict_executable),
            "resolved_sitecustomize_path": str(sitecustomize_path) if sitecustomize_path else None,
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
    env = os.environ.copy()
    env_overrides = entry.get("environment_overrides") or {}
    if isinstance(env_overrides, dict):
        for key, value in env_overrides.items():
            env[str(key)] = str(value)
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, shell=True, timeout=timeout_sec, env=env)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            shlex.split(command),
            124,
            stdout=subprocess_text(exc.stdout),
            stderr=subprocess_text(exc.stderr) + f"\n[registered_infer] Timeout after {timeout_sec}s.",
        )
        timed_out = True
    elapsed = time.time() - start
    expected_organs = _expected_output_organs(entry, extra_context)
    output_classification = _classify_mask_outputs(seg_out, expected_organs)
    if timed_out:
        status = "timed_out"
        failure_reason = "registered_teacher_timed_out"
    elif completed.returncode != 0:
        status = "failed"
        failure_reason = "registered_teacher_returned_nonzero"
    elif expected_organs and output_classification["missing_expected_outputs"]:
        status = "failed"
        failure_reason = "expected_mask_missing"
    elif output_classification["formal_mask_count"] <= 0:
        status = "failed"
        failure_reason = "teacher_executed_but_no_expected_masks"
    else:
        status = "success"
        failure_reason = None
    result = {
        "stage": "infer", "backend": "registered", "model_key": model_key,
        "model_name": entry.get("name", model_key), "status": status,
        "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out),
        "per_model_dir": str(per_model_dir),
        "command": command, "registry_path": str(registry_file), "return_code": completed.returncode,
        "timed_out": timed_out, "timeout_sec": timeout_sec, "runtime_sec": round(elapsed, 3),
        **output_classification,
        "failure_reason": failure_reason,
        "resolved_python": str(python_executable),
        "resolved_predict_executable": str(predict_executable),
        "resolved_sitecustomize_path": str(sitecustomize_path) if sitecustomize_path else None,
        "stdout_tail": (completed.stdout or "")[-4000:], "stderr_tail": (completed.stderr or "")[-4000:],
        "private_checkpoint": entry.get("private_checkpoint"),
    }
    _write_run_meta(per_model_dir, model_key, recipe, result)
    _write_inference_summary(case_out, result, seg_out)
    return result
