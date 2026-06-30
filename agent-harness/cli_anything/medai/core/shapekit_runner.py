from __future__ import annotations

import subprocess
import time
import shutil
import tempfile
import os
import signal
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .data_checker import check_case_folder
from .subprocess_utils import subprocess_text

_SHAPEKIT_TARGET_REQUIREMENTS = {
    "adrenal_gland": ["adrenal_gland_left", "adrenal_gland_right"],
    "aorta": ["aorta"], "bladder": ["bladder"], "colon": ["colon"], "duodenum": ["duodenum"],
    "femur": ["femur_left", "femur_right"], "intestine": ["intestine"],
    "kidney": ["kidney_left", "kidney_right"], "liver": ["liver"], "lung": ["lung_left", "lung_right"],
    "pancreas": ["pancreas"], "postcava": ["postcava"], "prostate": ["prostate"],
    "spleen": ["spleen"], "stomach": ["stomach"], "vertebrae": ["vertebrae_"],
}


def _mask_names(input_folder: Path) -> set[str]:
    names = set()
    for p in input_folder.glob("*/segmentations/*.nii.gz"):
        names.add(p.name[:-7])
    return names


def _case_mask_names(input_folder: Path) -> dict[str, set[str]]:
    cases: dict[str, set[str]] = {}
    for case in sorted(p for p in input_folder.iterdir() if p.is_dir()):
        seg = case / "segmentations"
        cases[case.name] = {p.name[:-7] for p in seg.glob("*.nii.gz")} if seg.exists() else set()
    return cases


def _resolve_affine_reference(config_ref: str, input_folder: Path, auto_config: bool) -> dict[str, Any]:
    case_masks = _case_mask_names(input_folder)
    if not case_masks:
        return {"status": "failed", "reason": "No ShapeKit cases detected for affine reference selection."}
    ref_stem = config_ref.replace(".nii.gz", "").replace(".nii", "")
    missing_default = [case for case, masks in case_masks.items() if ref_stem not in masks]
    if not missing_default:
        return {
            "status": "success",
            "reference_file_name": config_ref,
            "reference_source": "config_default",
            "missing_default_reference_cases": [],
        }
    if not auto_config:
        return {
            "status": "failed",
            "reason": f"ShapeKit requires affine reference mask '{config_ref}' in each case.",
            "missing_default_reference_cases": missing_default,
        }

    common_masks: set[str] | None = None
    for masks in case_masks.values():
        common_masks = set(masks) if common_masks is None else common_masks & masks
    common_sorted = sorted(common_masks or set())
    if not common_sorted:
        return {
            "status": "failed",
            "reason": (
                f"ShapeKit requires affine reference mask '{config_ref}' in each case, "
                "and no alternate mask exists in every case."
            ),
            "missing_default_reference_cases": missing_default,
        }
    alternate = common_sorted[0]
    return {
        "status": "success",
        "reference_file_name": f"{alternate}.nii.gz",
        "reference_source": "auto_selected_common_mask",
        "missing_default_reference_cases": missing_default,
        "available_common_masks": common_sorted[:80],
    }


def _derive_safe_targets(input_folder: Path) -> list[str]:
    names = _mask_names(input_folder)
    out = []
    for target, reqs in _SHAPEKIT_TARGET_REQUIREMENTS.items():
        if target == "vertebrae":
            if any(x.startswith("vertebrae_") for x in names):
                out.append(target)
        elif all(r in names for r in reqs):
            out.append(target)
    return out


def _prepare_config(root: Path, input_folder: Path, auto_config: bool) -> dict[str, Any]:
    config_path = root / "config.yaml"
    if not config_path.exists():
        return {"status": "failed", "reason": "ShapeKit config.yaml not found", "config_path": str(config_path)}
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    mask_names = sorted(_mask_names(input_folder))
    ref = config.get("affine_reference_file_name", "liver.nii.gz")
    reference_info = _resolve_affine_reference(str(ref), input_folder, auto_config)
    if reference_info.get("status") == "failed":
        return {
            "status": "failed",
            "reason": reference_info.get("reason"),
            "available_masks": mask_names[:80],
            "suggested_fix": "include a common affine reference mask or enable ShapeKit auto_config",
            "reference_check": reference_info,
        }
    safe_targets = _derive_safe_targets(input_folder)
    if not safe_targets:
        return {"status": "failed", "reason": "No safe ShapeKit target organs detected", "available_masks": mask_names[:80]}
    original_targets = config.get("target_organs", [])
    original_ref = config.get("affine_reference_file_name", "liver.nii.gz")
    if auto_config:
        config["target_organs"] = safe_targets
        config["affine_reference_file_name"] = reference_info["reference_file_name"]
        config["if_save_combined_label"] = True
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "status": "success",
        "auto_config": auto_config,
        "target_organs_original": original_targets,
        "target_organs_used": safe_targets,
        "affine_reference_original": original_ref,
        "affine_reference_used": reference_info["reference_file_name"],
        "affine_reference_source": reference_info.get("reference_source"),
        "missing_default_reference_cases": reference_info.get("missing_default_reference_cases", []),
        "available_common_reference_masks": reference_info.get("available_common_masks", []),
        "available_masks": mask_names[:80],
        "backup_text": text,
        "config_path": str(config_path),
    }


def _restore_config(info: dict[str, Any]) -> None:
    if info.get("backup_text") and info.get("config_path"):
        Path(info["config_path"]).write_text(info["backup_text"], encoding="utf-8")


def _summarize_output(out: Path) -> dict:
    cases = [p for p in out.iterdir() if p.is_dir()] if out.exists() else []
    summaries = []
    total = 0
    for case in sorted(cases):
        seg = case / "segmentations"
        masks = sorted([p.name for p in seg.glob("*.nii.gz")]) if seg.exists() else []
        total += len(masks)
        summaries.append({"case_id": case.name, "has_segmentations": seg.exists(), "num_masks": len(masks), "has_combined_labels": (case / "combined_labels.nii.gz").exists(), "sample_masks": masks[:30]})
    return {"num_cases": len(cases), "num_masks_total": total, "cases": summaries}


def _run_command_with_timeout(cmd: list[str], *, cwd: Path, timeout_sec: int) -> tuple[subprocess.CompletedProcess, bool]:
    """Run a subprocess and reliably terminate its process group on timeout.

    ShapeKit can spawn worker processes.  subprocess.run(timeout=...) only
    times out the direct child and may leave workers behind on some failure
    paths.  Use a new process group on POSIX so the wrapper can clean up the
    whole tree, and normalize stdout/stderr to text because TimeoutExpired may
    carry bytes even when text=True was requested.
    """
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr or ""), False
    except subprocess.TimeoutExpired as exc:
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
        else:
            proc.kill()
            stdout, stderr = proc.communicate()
        stdout_text = subprocess_text(stdout) or subprocess_text(exc.stdout)
        stderr_text = subprocess_text(stderr) or subprocess_text(exc.stderr)
        stderr_text += f"\n[ShapeKit wrapper] Timeout after {timeout_sec} seconds; process group terminated."
        return subprocess.CompletedProcess(cmd, 124, stdout_text, stderr_text), True


def run_shapekit(shapekit_root: str | Path, input_folder: str | Path, output_folder: str | Path, log_folder: str | Path, cpu_count: int = 2, continue_prediction: bool = False, csv: str | None = None, tqdm_ncols: int = 100, dry_run: bool = False, auto_config: bool = True, timeout_sec: int = 120) -> dict:
    root, inp, out, logs = Path(shapekit_root).resolve(), Path(input_folder).resolve(), Path(output_folder).resolve(), Path(log_folder).resolve()
    precheck = check_case_folder(inp)
    if not root.exists() or not (root / "main.py").exists():
        return {"stage": "postprocess", "tool": "ShapeKit", "status": "failed", "reason": "invalid ShapeKit root; main.py not found", "shapekit_root": str(root), "input_check": precheck}
    runtime_tmp = None
    runtime_root = root
    # Do not mutate the teacher-provided ShapeKit source tree.  ShapeKit reads
    # config.yaml at import time, so the safest wrapper strategy is to create a
    # small temporary runtime copy, auto-configure that copy, and run main.py there.
    if auto_config and not dry_run:
        runtime_tmp = tempfile.TemporaryDirectory(prefix="medai_shapekit_")
        runtime_root = Path(runtime_tmp.name) / "ShapeKit-main"
        shutil.copytree(
            root,
            runtime_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

    config_info = _prepare_config(runtime_root, inp, auto_config if not dry_run else False)
    clean_config = {k: v for k, v in config_info.items() if k != "backup_text"}
    if config_info.get("status") == "failed":
        if runtime_tmp is not None:
            runtime_tmp.cleanup()
        return {"stage": "postprocess", "tool": "ShapeKit", "status": "failed", "reason": config_info.get("reason"), "input_folder": str(inp), "input_check": precheck, "config_check": clean_config}
    cmd = ["python", "-W", "ignore", "main.py", "--input_folder", str(inp), "--output_folder", str(out), "--cpu_count", str(max(1, cpu_count)), "--log_folder", str(logs), "--tqdm_ncols", str(tqdm_ncols)]
    if continue_prediction: cmd.append("--continue_prediction")
    if csv: cmd += ["--csv", str(Path(csv).resolve())]
    if dry_run:
        if runtime_tmp is not None:
            runtime_tmp.cleanup()
        return {"stage": "postprocess", "tool": "ShapeKit", "status": "dry_run", "command": cmd, "input_check": precheck, "config_check": clean_config, "runtime_root": str(runtime_root)}
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        completed, timed_out = _run_command_with_timeout(cmd, cwd=runtime_root, timeout_sec=timeout_sec)
    finally:
        if runtime_tmp is not None:
            runtime_tmp.cleanup()
        else:
            _restore_config(config_info)
    elapsed = time.time() - start
    output_summary = _summarize_output(out)
    stdout_tail = str(completed.stdout)[-4000:]; stderr_tail = str(completed.stderr)[-6000:]
    crash = any(x in stdout_tail or x in stderr_tail for x in ["Traceback", "[CRASH]", "KeyError", "FileNotFoundError", "RuntimeError"])
    has_outputs = output_summary["num_cases"] > 0 and output_summary["num_masks_total"] > 0
    status, reason = "success", None
    if timed_out:
        status, reason = "failed", f"ShapeKit timed out after {timeout_sec} seconds"
    elif completed.returncode != 0:
        status, reason = "failed", f"ShapeKit returned non-zero code {completed.returncode}"
    elif crash:
        status, reason = "failed", "ShapeKit printed traceback/crash output; treated as failed even if return code is 0"
    elif not has_outputs:
        status, reason = "failed", "ShapeKit produced no output masks"
    return {"stage": "postprocess", "tool": "ShapeKit", "status": status, "reason": reason, "input_folder": str(inp), "output_folder": str(out), "log_folder": str(logs), "command": cmd, "runtime_root": str(runtime_root), "return_code": completed.returncode, "runtime_sec": round(elapsed, 3), "timeout_sec": timeout_sec, "timed_out": timed_out, "input_check": precheck, "config_check": clean_config, "output_summary": output_summary, "stdout_tail": stdout_tail, "stderr_tail": stderr_tail, "debug_log": str(logs / "debug.log"), "postprocessing_log": str(logs / "postprocessing.log")}
