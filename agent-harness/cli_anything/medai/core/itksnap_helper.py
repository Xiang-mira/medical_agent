"""ITK-SNAP integration helper.

Teacher asked: 'download ITK-SNAP, drag CT and mask in, check rendering quality.'

This generates ready-to-copy ITK-SNAP command lines from the review queue,
so the user can quickly open uncertain cases for manual inspection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .json_utils import write_json


def _read_review_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def generate_itksnap_commands(
    review_queue_jsonl: str | Path,
    ct_root: str | Path | None = None,
    annotation_root: str | Path | None = None,
    output_script: str | Path | None = None,
    max_cases: int = 10,
) -> dict[str, Any]:
    """Generate ITK-SNAP CLI commands for uncertain cases in the review queue.

    ITK-SNAP command format:
        itksnap -g ct.nii.gz -s segmentation.nii.gz
    or for overlay comparison:
        itksnap -g ct.nii.gz -s mask_a.nii.gz -o mask_b.nii.gz
    """
    queue = _read_review_queue(Path(review_queue_jsonl))
    commands: list[dict[str, Any]] = []

    for item in queue[:max_cases]:
        case_id = item.get("case_id", "unknown")
        organ = item.get("organ", "")
        reason = item.get("reason", "")

        # Try to locate CT
        ct_path = None
        if ct_root:
            for cand in [
                Path(ct_root) / case_id / "ct.nii.gz",
                Path(ct_root) / "ImageTr" / case_id / "ct.nii.gz",
            ]:
                if cand.exists():
                    ct_path = str(cand.resolve())
                    break

        # Try to locate mask
        mask_path = None
        if annotation_root:
            for cand in [
                Path(annotation_root) / case_id / "updated" / f"{organ}.nii.gz",
                Path(annotation_root) / case_id / "segmentations" / f"{organ}.nii.gz",
            ]:
                if cand.exists():
                    mask_path = str(cand.resolve())
                    break

        cmd_parts = ["itksnap"]
        if ct_path:
            cmd_parts.extend(["-g", ct_path])
        if mask_path:
            cmd_parts.extend(["-s", mask_path])

        cmd_str = " ".join(cmd_parts) if len(cmd_parts) > 1 else None

        commands.append({
            "case_id": case_id,
            "organ": organ,
            "reason": reason,
            "itksnap_command": cmd_str,
            "ct_path": ct_path,
            "mask_path": mask_path,
            "note": "Open in ITK-SNAP to visually verify mask quality." if cmd_str else "CT or mask not found; locate manually.",
        })

    # Write shell script
    script_lines = [
        "#!/bin/bash",
        "# ITK-SNAP review commands for uncertain cases",
        f"# Generated from: {review_queue_jsonl}",
        f"# Total uncertain items: {len(queue)}, showing first {max_cases}",
        "",
    ]
    for c in commands:
        if c["itksnap_command"]:
            script_lines.append(f"# Case: {c['case_id']} | Organ: {c['organ']} | Reason: {c['reason']}")
            script_lines.append(c["itksnap_command"])
            script_lines.append("")

    if output_script:
        out = Path(output_script).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(script_lines), encoding="utf-8")

    return {
        "stage": "itksnap_helper",
        "status": "success",
        "review_queue": str(review_queue_jsonl),
        "num_uncertain": len(queue),
        "num_commands_generated": len([c for c in commands if c["itksnap_command"]]),
        "output_script": str(output_script) if output_script else None,
        "commands": commands,
    }
