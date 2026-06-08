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
        prepared = mask.resolve()
        _ensure_left_right_projection_companion(prepared, organ)
        return prepared
    out = dst_root / label
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{organ}.nii.gz"
    if mask.exists() and not dst.exists():
        shutil.copy2(mask, dst)
    _ensure_left_right_projection_companion(out, organ)
    return out.resolve()


def _ensure_left_right_projection_companion(mask_dir: Path, organ: str) -> dict[str, Any] | None:
    """Add a projection-only companion mask for LabelCritic left/right joins.

    ProjectDatasetFlex_single.py tries to merge any `*left*` organ projection
    with its `*right*` counterpart. Pairwise LabelCritic calls compare one organ
    at a time, so the right counterpart often is not present. Adding a copied
    companion prevents projection from failing; it is used only inside the
    temporary LabelCritic work folder and never becomes a training target.
    """
    organ_name = str(organ).removesuffix(".nii.gz")
    if "left" not in organ_name:
        return None
    src = mask_dir / f"{organ_name}.nii.gz"
    companion_name = organ_name.replace("left", "right")
    dst = mask_dir / f"{companion_name}.nii.gz"
    if not src.exists() or dst.exists():
        return None
    shutil.copy2(src, dst)
    return {
        "status": "created",
        "source": str(src),
        "companion": str(dst),
        "reason": "projection_only_companion_for_labelcritic_left_right_join",
    }


def _parse_labelcritic_log(
    log_path: Path,
    mask1_folder: Path | None = None,
    mask2_folder: Path | None = None,
    run_id: str | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Parse LabelCritic's comparison_summary.log.

    Uses run_id to scope the search to the current run's log block, avoiding
    stale results from earlier runs in the same append-only log file.
    Falls back to tail-read when run_id is not provided.
    """
    if not log_path.exists():
        return {"winner": "uncertain", "confidence": 0.5, "reason": "LabelCritic log was not generated", "parse_status": "missing_log"}

    import re
    csv_rows = 0
    if csv_path and csv_path.exists():
        csv_rows = max(0, len(csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1)
    full_text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Scope to the current run's block using run_id
    if run_id:
        pattern = rf"Run ID: {re.escape(run_id)}\n(.*?)(?=\n\[|\Z)"
        m = re.search(pattern, full_text, re.S)
        text = m.group(0) if m else full_text[-12000:]
    else:
        text = full_text[-12000:]

    lower = text.lower()

    better_lines = re.findall(r"Better:\s*(.+)", text, flags=re.I)
    if better_lines:
        best = better_lines[-1].strip().strip('"\'')

        # Bug fix: handle explicit "uncertain" written by patched CompareOrgan.py
        if best.lower() == "uncertain":
            if csv_path and csv_path.exists() and csv_rows == 0:
                return {
                    "winner": "uncertain",
                    "confidence": 0.0,
                    "reason": text,
                    "parse_status": "no_comparison_rows",
                    "csv_path": str(csv_path),
                    "csv_rows": csv_rows,
                }
            return {"winner": "uncertain", "confidence": 0.0, "reason": text, "parse_status": "vlm_undecided"}

        # Bug fix: normalize paths with resolve() before comparison
        try:
            best_resolved = str(Path(best).resolve()).lower()
        except Exception:
            best_resolved = best.lower()

        if mask1_folder:
            try:
                m1_resolved = str(Path(mask1_folder).resolve()).lower()
            except Exception:
                m1_resolved = str(mask1_folder).lower()
            if m1_resolved == best_resolved or m1_resolved in best_resolved:
                return {"winner": "a", "confidence": 0.75, "reason": text, "parse_status": "better_path_parse"}

        if mask2_folder:
            try:
                m2_resolved = str(Path(mask2_folder).resolve()).lower()
            except Exception:
                m2_resolved = str(mask2_folder).lower()
            if m2_resolved == best_resolved or m2_resolved in best_resolved:
                return {"winner": "b", "confidence": 0.75, "reason": text, "parse_status": "better_path_parse"}

        if "mask1" in best.lower():
            return {"winner": "a", "confidence": 0.7, "reason": text, "parse_status": "better_line_mask1"}
        if "mask2" in best.lower():
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
    no_dice_check: bool = False,
    no_dual_confirmation: bool = False,
    simple_prompt_ablation: bool = False,
    conservative_dual: bool = False,
    skip_organ_presence_gate: bool = False,
    strict_choice_prompt: bool = False,
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

    # Generate a stable run_id so _parse_labelcritic_log can scope to this run
    import uuid as _uuid
    run_id = _uuid.uuid4().hex[:8]

    mask1_folder = _prepare_mask_folder(a, organ, work_dir, "mask1")
    mask2_folder = _prepare_mask_folder(b, organ, work_dir, "mask2")
    log_file = work_dir / "comparison_summary.log"
    csv_path = work_dir / "results" / run_id / f"{organ}.csv"
    command = [
        "python", str(script),
        "--ct", str(ct),
        "--mask1", str(mask1_folder),
        "--mask2", str(mask2_folder),
        "--organ", organ,
        "--port", str(normalized_port),
        "--log_file", str(log_file),
        "--base_url", normalized_base_url,
        "--run_id", run_id,
        "--base_output", str(work_dir / "comparison_results"),
        "--base_csv", str(work_dir / "results"),
    ]
    if no_dice_check:
        command.append("--no_dice_check")
    if no_dual_confirmation:
        command.append("--no_dual_confirmation")
    if simple_prompt_ablation:
        command.append("--simple_prompt_ablation")
    if conservative_dual:
        command.append("--conservative_dual")
    if skip_organ_presence_gate:
        command.append("--skip_organ_presence_gate")
    if strict_choice_prompt:
        command.append("--strict_choice_prompt")
    labelcritic_options = {
        "no_dice_check": no_dice_check,
        "no_dual_confirmation": no_dual_confirmation,
        "simple_prompt_ablation": simple_prompt_ablation,
        "conservative_dual": conservative_dual,
        "skip_organ_presence_gate": skip_organ_presence_gate,
        "strict_choice_prompt": strict_choice_prompt,
        "run_id": run_id,
        "csv_path": str(csv_path),
    }

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
            "labelcritic_options": labelcritic_options,
        }
        write_json(out_json, result)
        return result

    if not script.exists():
        result = {"stage": "labelcritic", "status": "failed", "reason": "CompareOrgan.py not found", "labelcritic_root": str(lc_root), "command": command, "projection": None}
        write_json(out_json, result)
        return result

    # Bug fix: do NOT call build_projection here — CompareOrgan.py calls
    # ProjectDatasetFlex_single.py internally which runs the projection itself.
    # Calling it here would double the I/O and compute cost.
    proj = {"status": "skipped", "reason": "projection handled internally by CompareOrgan.py → ProjectDatasetFlex_single.py"}

    start = time.time()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec, cwd=str(lc_root))
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, 124, stdout=exc.stdout or "", stderr=(exc.stderr or "") + f"\n[labelcritic] Timeout after {timeout_sec}s")
        timed_out = True
    elapsed = time.time() - start
    decision = _parse_labelcritic_log(log_file, mask1_folder, mask2_folder, run_id=run_id, csv_path=csv_path)
    result = {
        "stage": "labelcritic", "status": "timed_out" if timed_out else ("success" if completed.returncode == 0 else "failed"),
        "backend": backend, "organ": organ, "ct_image": str(ct), "mask_a": str(a), "mask_b": str(b),
        "output_json": str(out_json), "projection": proj, "command": command,
        "return_code": completed.returncode, "runtime_sec": round(elapsed, 3),
        "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
        "log_file": str(log_file), "normalized_base_url": normalized_base_url, "normalized_port": normalized_port, "decision": decision,
        "labelcritic_options": labelcritic_options,
    }
    write_json(out_json, result)
    return result
