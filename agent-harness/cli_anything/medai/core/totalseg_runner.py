from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .presets import parse_roi_subset

# Tasks that raise ValueError when --fast is passed (per TotalSegmentator python_api.py).
# These tasks use high-resolution models and cannot be run in fast mode.
_NO_FAST_TASKS: frozenset[str] = frozenset({
    "head_glands_cavities", "head_muscles", "oculomotor_muscles", "liver_segments",
    "liver_segments_mr", "appendicular_bones", "appendicular_bones_mr", "brain_structures",
    "coronary_arteries", "coronary_arteries_LEGACY", "lung_vessels_LEGACY", "cerebral_bleed",
    "hip_implant", "pleural_pericard_effusion", "liver_vessels", "lung_nodules",
    "kidney_cysts", "breasts", "ventricle_parts", "liver_lesions", "liver_lesions_mr",
    "craniofacial_structures", "abdominal_muscles", "trunk_cavities", "brain_aneurysm",
    "vertebrae_body", "tissue_types", "tissue_types_mr", "tissue_4_types", "face",
    "face_mr", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "aortic_sinuses",
})

# Tasks that require an academic license (free for non-commercial use, register at
# https://backend.totalsegmentator.com/license-academic/ then run: totalseg_set_license -l <key>)
LICENSED_TASKS: frozenset[str] = frozenset({
    "heartchambers_highres", "appendicular_bones", "appendicular_bones_mr",
    "tissue_types", "tissue_types_mr", "tissue_4_types", "face", "face_mr",
    "brain_structures", "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr",
    "coronary_arteries", "coronary_arteries_LEGACY", "aortic_sinuses", "vertebrae_body",
})


def find_totalseg_executable() -> str | None:
    return shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")


def build_totalseg_command(image_path: str | Path, output_dir: str | Path, fast: bool = True, task: str | None = None, roi_preset: str = "shapekit_abdomen", roi_subset: str | None = None, device: str | None = None, statistics: bool = False, preview: bool = False) -> list[str]:
    exe = find_totalseg_executable() or "TotalSegmentator"
    cmd = [exe, "-i", str(image_path), "-o", str(output_dir)]
    task_no_fast = task in _NO_FAST_TASKS if task else False
    if fast and not task_no_fast:
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


def run_totalseg_with_contract(
    image_path: str | Path,
    per_model_dir: Path,
    case_out: Path,
    subtask_config: dict | None = None,
    task: str | None = None,
    subtasks: list[str] | None = None,
    fast: bool = True,
    device: str | None = None,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    case_id: str = "unknown",
) -> dict:
    """Run TotalSegmentator across all subtasks and write the per_model output contract.

    Outputs:
        per_model_dir/combined_labels.nii.gz  — integer label map (local IDs, 1-indexed)
        per_model_dir/local_labels.json       — {organ_name: int_id}
    """
    image = Path(image_path).resolve()
    if subtasks:
        subtasks_to_run = list(dict.fromkeys(str(x) for x in subtasks if x))
    elif task:
        subtasks_to_run = [task]
    else:
        subtasks_to_run = list((subtask_config or {}).get("subtasks", {}).keys()) or ["total"]

    if dry_run:
        cmds = []
        for st in subtasks_to_run:
            cmds.append(build_totalseg_command(image, case_out / "segmentations" / f"totalseg_{st}", fast, st, "none", None, device))
        return {
            "stage": "infer", "backend": "TotalSegmentator",
            "status": "dry_run", "case_id": case_id,
            "subtasks": subtasks_to_run, "per_model_dir": str(per_model_dir),
            "commands": [str(c) for c in cmds],
        }

    per_model_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.time()
    all_organ_masks: dict[str, Path] = {}
    subtask_results: dict[str, dict] = {}
    direct_seg_dir = case_out / "segmentations"
    direct_seg_dir.mkdir(parents=True, exist_ok=True)

    for subtask in subtasks_to_run:
        st_seg_dir = case_out / "segmentations" / f"totalseg_{subtask}"
        st_seg_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_totalseg_command(image, st_seg_dir, fast, subtask, "none", None, device)
        timed_out = False
        st_start = time.time()
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            completed = subprocess.CompletedProcess(cmd, 124, stdout=(exc.stdout or ""), stderr=(exc.stderr or "") + f"\n[totalseg_runner] Subtask {subtask} timeout after {timeout_sec}s.")
        st_elapsed = time.time() - st_start
        st_masks = sorted(st_seg_dir.glob("*.nii.gz")) if st_seg_dir.exists() else []
        st_status = "timed_out" if timed_out else ("success" if completed.returncode == 0 and st_masks else "failed")
        subtask_results[subtask] = {"status": st_status, "runtime_sec": round(st_elapsed, 3), "return_code": completed.returncode, "num_masks": len(st_masks)}
        for mask_path in st_masks:
            name = mask_path.name
            organ_name = name[:-7] if name.endswith(".nii.gz") else (name[:-4] if name.endswith(".nii") else name)
            all_organ_masks[organ_name] = mask_path

    elapsed = time.time() - start_total

    if not all_organ_masks:
        return {"stage": "infer", "backend": "TotalSegmentator", "status": "failed", "case_id": case_id, "reason": "No organ masks produced by any subtask", "subtask_results": subtask_results, "runtime_sec": round(elapsed, 3)}

    sorted_organs = sorted(all_organ_masks)
    local_labels = {organ: i + 1 for i, organ in enumerate(sorted_organs)}

    try:
        import numpy as np
        import nibabel as nib
        ref_img = nib.load(str(next(iter(all_organ_masks.values()))))
        combined_arr = np.zeros(ref_img.shape, dtype=np.int16)
        for organ in sorted_organs:
            src_mask = all_organ_masks[organ]
            dst_mask = direct_seg_dir / f"{organ}.nii.gz"
            if src_mask.resolve() != dst_mask.resolve():
                shutil.copy2(src_mask, dst_mask)
            mask_arr = np.asanyarray(nib.load(str(src_mask)).dataobj).astype(bool)
            combined_arr[mask_arr] = local_labels[organ]
        nib.save(nib.Nifti1Image(combined_arr, ref_img.affine, ref_img.header), str(per_model_dir / "combined_labels.nii.gz"))
        (per_model_dir / "local_labels.json").write_text(json.dumps(local_labels, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        return {"stage": "infer", "backend": "TotalSegmentator", "status": "failed", "case_id": case_id, "reason": f"label map merge failed: {exc}", "subtask_results": subtask_results, "runtime_sec": round(elapsed, 3)}

    masks = sorted(p.name for p in direct_seg_dir.glob("*.nii.gz"))
    return {
        "stage": "infer",
        "backend": "TotalSegmentator",
        "status": "success",
        "case_id": case_id,
        "segmentation_output": str(direct_seg_dir),
        "per_model_dir": str(per_model_dir),
        "num_organs": len(local_labels),
        "num_masks": len(masks),
        "sample_masks": masks[:50],
        "runtime_sec": round(elapsed, 3),
        "subtasks": subtasks_to_run,
        "subtask_results": subtask_results,
    }


def run_custom_inference(image_path: str, output_folder: str, model_command: str, case_id: str | None = None, dry_run: bool = False) -> dict:
    image = Path(image_path).resolve()
    if case_id is None:
        case_id = image.parent.name or image.stem
    case_out = Path(output_folder).resolve() / case_id
    seg_out = case_out / "segmentations"
    command_str = model_command.format(image=str(image), output=str(seg_out), case_output=str(case_out), case_id=case_id)
    if dry_run:
        return {"stage": "infer", "backend": "custom", "status": "dry_run", "case_id": case_id, "command": command_str, "segmentation_output": str(seg_out)}
    seg_out.mkdir(parents=True, exist_ok=True)
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
