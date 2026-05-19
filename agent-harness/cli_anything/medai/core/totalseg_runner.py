from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .presets import parse_roi_subset


def find_totalseg_executable() -> str | None:
    return shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")


def build_totalseg_command(image_path: str | Path, output_dir: str | Path, fast: bool = True, task: str | None = None, roi_preset: str = "shapekit_abdomen", roi_subset: str | None = None, device: str | None = None, statistics: bool = False, preview: bool = False) -> list[str]:
    exe = find_totalseg_executable() or "TotalSegmentator"
    cmd = [exe, "-i", str(image_path), "-o", str(output_dir)]
    if fast:
        cmd.append("--fast")
        # Automatically switch to the fast-compatible preset when --fast is used
        # and no explicit override was given, to avoid KeyError on unsupported organs.
        if not roi_subset and roi_preset == "shapekit_abdomen":
            roi_preset = "shapekit_abdomen_fast"
    if task:
        cmd += ["--task", task]
    rois = parse_roi_subset(roi_preset, roi_subset)
    if rois:
        cmd += ["--roi_subset", *rois]
    if device:
        cmd += ["--device", device]
    if statistics:
        cmd.append("--statistics")
    if preview:
        cmd.append("--preview")
    return cmd


def run_totalsegmentator(image_path: str, output_folder: str, case_id: str | None = None, fast: bool = True, task: str | None = None, roi_preset: str = "shapekit_abdomen", roi_subset: str | None = None, device: str | None = None, statistics: bool = False, preview: bool = False, dry_run: bool = False, timeout_sec: int = 600) -> dict:
    image = Path(image_path).resolve()
    if case_id is None:
        case_id = image.parent.name or image.stem
    case_out = Path(output_folder).resolve() / case_id
    seg_out = case_out / "segmentations"
    cmd = build_totalseg_command(image, seg_out, fast, task, roi_preset, roi_subset, device, statistics, preview)
    if dry_run:
        return {"stage": "infer", "backend": "TotalSegmentator", "status": "dry_run", "case_id": case_id, "command": cmd, "segmentation_output": str(seg_out)}
    seg_out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    timed_out = False
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        completed = subprocess.CompletedProcess(cmd, returncode=124, stdout=(exc.stdout or ""), stderr=(exc.stderr or "") + f"\n[totalseg_runner] Timeout after {timeout_sec}s.")
    elapsed = time.time() - start
    masks = sorted([p.name for p in seg_out.glob("*.nii.gz")]) if seg_out.exists() else []
    status = "timed_out" if timed_out else ("success" if completed.returncode == 0 and masks else "failed")
    return {"stage": "infer", "backend": "TotalSegmentator", "status": status, "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out), "command": cmd, "return_code": completed.returncode, "timed_out": timed_out, "timeout_sec": timeout_sec, "runtime_sec": round(elapsed, 3), "num_masks": len(masks), "sample_masks": masks[:50], "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}


def run_custom_inference(image_path: str, output_folder: str, model_command: str, case_id: str | None = None, dry_run: bool = False) -> dict:
    image = Path(image_path).resolve()
    if case_id is None:
        case_id = image.parent.name or image.stem
    case_out = Path(output_folder).resolve() / case_id
    seg_out = case_out / "segmentations"
    seg_out.mkdir(parents=True, exist_ok=True)
    command_str = model_command.format(image=str(image), output=str(seg_out), case_output=str(case_out), case_id=case_id)
    if dry_run:
        return {"stage": "infer", "backend": "custom", "status": "dry_run", "case_id": case_id, "command": command_str, "segmentation_output": str(seg_out)}
    start = time.time()
    # Use shell=True for custom command templates because users often pass
    # Windows-style paths (e.g., third_party\mock_model\mock_seg_infer.py).
    # shlex.split() uses POSIX escaping by default and can silently strip
    # backslashes on Windows, causing the command to fail even though the
    # template looks correct. The command string is provided by the user, so
    # this wrapper treats it as a trusted local command.
    completed = subprocess.run(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, shell=True)
    elapsed = time.time() - start
    masks = sorted([p.name for p in seg_out.glob("*.nii.gz")])
    return {"stage": "infer", "backend": "custom", "status": "success" if completed.returncode == 0 and masks else "failed", "case_id": case_id, "image": str(image), "segmentation_output": str(seg_out), "command": command_str, "return_code": completed.returncode, "runtime_sec": round(elapsed, 3), "num_masks": len(masks), "sample_masks": masks[:50], "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
