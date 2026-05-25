from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _mask_folder(mask: str | Path | None, organ: str, work_root: Path, label: str) -> Path | None:
    """Return a folder containing <organ>.nii.gz for LabelCritic ProjectDatasetFlex_single.py.

    LabelCritic expects a mask directory, not necessarily a single file.  This
    helper accepts either an organ-wise folder or a single .nii/.nii.gz file and
    materializes the minimal folder format in a temporary workspace.
    """
    if mask is None:
        return None
    p = Path(mask).resolve()
    if not p.exists():
        return None
    if p.is_dir():
        return p
    out = work_root / label
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{organ}.nii.gz"
    if not dst.exists():
        shutil.copy2(p, dst)
    return out


def build_labelcritic_projection(
    ct_image: str | Path,
    mask_a: str | Path | None,
    mask_b: str | Path | None,
    output_folder: str | Path,
    organ: str,
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    axis: int = 1,
    device: str = "cpu",
    num_processes: int = 2,
    timeout_sec: int = 900,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Use the teacher-provided LabelCritic projection code for 3D->2D images.

    This replaces the earlier simple mask-centered/average-style projection.
    LabelCritic's ProjectDatasetFlex_single.py internally calls projection.py,
    which windows CT volumes separately for organ/soft-tissue and bone/skeleton
    views and creates composite overlays for candidate-mask comparison.
    """
    ct = Path(ct_image).resolve()
    out = Path(output_folder).resolve()
    lc_root = Path(labelcritic_root).resolve()
    script = lc_root / "ProjectDatasetFlex_single.py"
    out.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "stage": "labelcritic_projection_builder",
        "projection_backend": "labelcritic",
        "organ": organ,
        "ct_image": str(ct),
        "output_folder": str(out),
        "labelcritic_root": str(lc_root),
        "script": str(script),
        "axis": axis,
        "device": device,
        "num_processes": num_processes,
        "teacher_alignment": "Uses LabelCritic ProjectDatasetFlex_single.py + projection.py rather than a naive average projection; CT windows include organ/soft-tissue and bone/skeleton views.",
    }

    if not ct.exists():
        result.update({"status": "failed", "reason": f"CT image not found: {ct}"})
        return result
    if not script.exists():
        result.update({"status": "failed", "reason": f"LabelCritic projection script not found: {script}"})
        return result
    if mask_a is None or mask_b is None:
        result.update({"status": "failed", "reason": "LabelCritic projection requires both mask_a and mask_b."})
        return result

    with tempfile.TemporaryDirectory(prefix="medai_labelcritic_proj_") as td:
        temp_root = Path(td)
        mask_a_folder = _mask_folder(mask_a, organ, temp_root, "candidate_a")
        mask_b_folder = _mask_folder(mask_b, organ, temp_root, "candidate_b")
        result["mask_a_folder"] = str(mask_a_folder) if mask_a_folder else None
        result["mask_b_folder"] = str(mask_b_folder) if mask_b_folder else None
        if mask_a_folder is None or mask_b_folder is None:
            result.update({"status": "failed", "reason": "One or both candidate masks are missing."})
            return result

        command = [
            "python", str(script),
            "--ct_good", str(ct),
            "--mask_good", str(mask_a_folder),
            "--ct_bad", str(ct),
            "--mask_bad", str(mask_b_folder),
            "--output_dir", str(out),
            "--organ", organ,
            "--device", device,
            "--num_processes", str(num_processes),
            "--axis", str(axis),
        ]
        result["command"] = command

        if dry_run:
            result.update({"status": "dry_run", "saved_projections": [], "note": "Command prepared but not executed."})
            return result

        proc = subprocess.run(
            command,
            cwd=str(lc_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        result["returncode"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        pngs = sorted(str(p.resolve()) for p in out.rglob("*.png"))
        csvs = sorted(str(p.resolve()) for p in out.rglob("*.csv"))
        result["saved_projections"] = pngs
        result["saved_csvs"] = csvs

        # Bug fix: copy mask staging folders to a persistent location inside `out`
        # before the TemporaryDirectory context exits, so result paths remain valid.
        for label, src_folder in [("candidate_a", mask_a_folder), ("candidate_b", mask_b_folder)]:
            if src_folder and src_folder.exists():
                persistent = out / f"{label}_staging"
                if not persistent.exists():
                    shutil.copytree(src_folder, persistent)
                result[f"mask_{label.split('_')[1]}_folder"] = str(persistent)
        if proc.returncode != 0:
            result.update({"status": "failed", "reason": "LabelCritic projection command failed."})
        else:
            result.update({"status": "success" if pngs else "warning", "reason": "No PNG projection was produced." if not pngs else "LabelCritic projection completed."})
        return result
