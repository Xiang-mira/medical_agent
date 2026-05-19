from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .projection_builder import build_projection
from .json_utils import write_json


def _prepare_mask_folder(mask: Path, organ: str, dst_root: Path, label: str) -> Path:
    """LabelCritic expects a folder of organ masks. Accept either folder or file."""
    if mask.is_dir():
        return mask.resolve()
    out = dst_root / label
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{organ}.nii.gz"
    if mask.exists() and not dst.exists():
        shutil.copy2(mask, dst)
    return out.resolve()


def _parse_labelcritic_log(log_path: Path, mask1_folder: Path | None = None, mask2_folder: Path | None = None) -> dict[str, Any]:
    """Parse LabelCritic's comparison_summary.log.

    The teacher-supplied CompareOrgan.py writes a line like:
        Better: /path/to/mask1_or_mask2
    not always a natural-language phrase such as "mask1 is better".  The earlier
    parser only looked for words like "mask1 better" and could miss successful
    LabelCritic decisions.  This parser checks the explicit Better path first,
    then falls back to phrase-based parsing.
    """
    if not log_path.exists():
        return {"winner": "uncertain", "confidence": 0.5, "reason": "LabelCritic log was not generated", "parse_status": "missing_log"}
    text = log_path.read_text(encoding="utf-8", errors="ignore")[-12000:]
    lower = text.lower()

    import re
    better_lines = re.findall(r"Better:\s*(.+)", text, flags=re.I)
    if better_lines:
        best = better_lines[-1].strip().strip('"\'')
        best_lower = best.lower()
        if mask1_folder and str(mask1_folder).lower() in best_lower:
            return {"winner": "a", "confidence": 0.75, "reason": text, "parse_status": "better_path_parse"}
        if mask2_folder and str(mask2_folder).lower() in best_lower:
            return {"winner": "b", "confidence": 0.75, "reason": text, "parse_status": "better_path_parse"}
        if "mask1" in best_lower:
            return {"winner": "a", "confidence": 0.7, "reason": text, "parse_status": "better_line_mask1"}
        if "mask2" in best_lower:
            return {"winner": "b", "confidence": 0.7, "reason": text, "parse_status": "better_line_mask2"}

    winner = "uncertain"
    if any(k in lower for k in ["mask1 is better", "winner: mask1", "selected: mask1", "answer: 1"]):
        winner = "a"
    if any(k in lower for k in ["mask2 is better", "winner: mask2", "selected: mask2", "answer: 2"]):
        winner = "b"
    return {"winner": winner, "confidence": 0.5 if winner == "uncertain" else 0.7, "reason": text, "parse_status": "heuristic_log_parse"}


def _normalize_labelcritic_base_url(base_url: str, port: int) -> tuple[str, int]:
    """Normalize LabelCritic host for CompareOrgan.py / RunAPI_single.py.

    RunAPI_single.py constructs `f"{base_url}:{port}/v1"`. Therefore `base_url`
    must be a host without `/v1` and normally without the port.  Users often pass
    OpenAI-compatible values such as `http://localhost:8000/v1`; this helper
    converts them to (`http://localhost`, 8000) to avoid malformed URLs like
    `http://localhost:8000/v1:8000/v1`.
    """
    import re
    url = (base_url or "http://localhost").rstrip("/")
    url = re.sub(r"/v1/?$", "", url)
    m = re.match(r"^(https?://[^/:]+):(\d+)$", url)
    if m:
        return m.group(1), int(m.group(2))
    return url, int(port)


def run_labelcritic_compare(
    ct_image: str | Path,
    mask_a: str | Path,
    mask_b: str | Path,
    organ: str,
    output_json: str | Path,
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    backend: str = "labelcritic",
    base_url: str = "http://localhost",
    port: int = 8000,
    dry_run: bool = False,
    strict_alignment: bool = False,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    ct = Path(ct_image).resolve()
    a = Path(mask_a).resolve()
    b = Path(mask_b).resolve()
    out_json = Path(output_json).resolve()
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / f"labelcritic_{organ}"
    work_dir.mkdir(parents=True, exist_ok=True)

    lc_root = Path(labelcritic_root).resolve()
    script = lc_root / "CompareOrgan.py"
    normalized_base_url, normalized_port = _normalize_labelcritic_base_url(base_url, port)

    mask1_folder = _prepare_mask_folder(a, organ, work_dir, "mask1")
    mask2_folder = _prepare_mask_folder(b, organ, work_dir, "mask2")
    log_file = work_dir / "comparison_summary.log"
    command = [
        "python", str(script),
        "--ct", str(ct),
        "--mask1", str(mask1_folder),
        "--mask2", str(mask2_folder),
        "--organ", organ,
        "--port", str(normalized_port),
        "--log_file", str(log_file),
        "--base_url", normalized_base_url,
    ]

    if dry_run or backend == "stub":
        # Build projections for dry-run/stub so reviewers can inspect the images.
        proj = build_projection(
            ct,
            a if a.is_file() else a / f"{organ}.nii.gz",
            b if b.is_file() else b / f"{organ}.nii.gz",
            work_dir / "projections",
            organ=organ, views=["axial", "coronal"], strict_alignment=strict_alignment,
            projection_backend="auto", labelcritic_root=labelcritic_root,
            axis=1, device="cpu", num_processes=2, dry_run=dry_run,
        )
        decision = {
            "winner": "uncertain",
            "confidence": 0.5,
            "reason": "dry-run/stub backend: LabelCritic command prepared; run with --backend labelcritic and a VLM server for automatic A/B selection.",
            "parse_status": "stub",
        }
        result = {
            "stage": "labelcritic", "status": "dry_run" if dry_run else "stub",
            "backend": backend, "organ": organ, "ct_image": str(ct),
            "mask_a": str(a), "mask_b": str(b), "output_json": str(out_json),
            "projection": proj, "command": command, "normalized_base_url": normalized_base_url, "normalized_port": normalized_port, "decision": decision,
        }
        write_json(out_json, result)
        return result

    if not script.exists():
        result = {"stage": "labelcritic", "status": "failed", "reason": "CompareOrgan.py not found", "labelcritic_root": str(lc_root), "command": command, "projection": None}
        write_json(out_json, result)
        return result

    # Build projections only when we will actually run the comparison.
    proj = build_projection(
        ct,
        a if a.is_file() else a / f"{organ}.nii.gz",
        b if b.is_file() else b / f"{organ}.nii.gz",
        work_dir / "projections",
        organ=organ, views=["axial", "coronal"], strict_alignment=strict_alignment,
        projection_backend="auto", labelcritic_root=labelcritic_root,
        axis=1, device="cpu", num_processes=2, dry_run=False,
    )

    start = time.time()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec, cwd=str(lc_root))
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, 124, stdout=exc.stdout or "", stderr=(exc.stderr or "") + f"\n[labelcritic] Timeout after {timeout_sec}s")
        timed_out = True
    elapsed = time.time() - start
    decision = _parse_labelcritic_log(log_file, mask1_folder, mask2_folder)
    result = {
        "stage": "labelcritic", "status": "timed_out" if timed_out else ("success" if completed.returncode == 0 else "failed"),
        "backend": backend, "organ": organ, "ct_image": str(ct), "mask_a": str(a), "mask_b": str(b),
        "output_json": str(out_json), "projection": proj, "command": command,
        "return_code": completed.returncode, "runtime_sec": round(elapsed, 3),
        "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
        "log_file": str(log_file), "normalized_base_url": normalized_base_url, "normalized_port": normalized_port, "decision": decision,
    }
    write_json(out_json, result)
    return result
