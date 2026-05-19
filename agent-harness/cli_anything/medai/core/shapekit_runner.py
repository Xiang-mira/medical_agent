from __future__ import annotations

import subprocess
import time
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from .data_checker import check_case_folder

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
    ref_stem = ref.replace(".nii.gz", "").replace(".nii", "")
    if ref_stem not in mask_names:
        return {"status": "failed", "reason": f"ShapeKit requires affine reference mask '{ref}' in each case.", "available_masks": mask_names[:80], "suggested_fix": "include liver.nii.gz or change ShapeKit affine_reference_file_name"}
    safe_targets = _derive_safe_targets(input_folder)
    if not safe_targets:
        return {"status": "failed", "reason": "No safe ShapeKit target organs detected", "available_masks": mask_names[:80]}
    original_targets = config.get("target_organs", [])
    if auto_config:
        config["target_organs"] = safe_targets
        config["if_save_combined_label"] = True
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"status": "success", "auto_config": auto_config, "target_organs_original": original_targets, "target_organs_used": safe_targets, "available_masks": mask_names[:80], "backup_text": text, "config_path": str(config_path)}


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
    timed_out = False
    try:
        completed = subprocess.run(cmd, cwd=str(runtime_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        completed = subprocess.CompletedProcess(cmd, returncode=124, stdout=(exc.stdout or ""), stderr=(exc.stderr or "") + f"\n[ShapeKit wrapper] Timeout after {timeout_sec} seconds.")
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
