from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .projection_builder import build_projection
from .json_utils import write_json
from .subprocess_utils import subprocess_text


def _file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        p = Path(path)
        st = p.stat()
        return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except Exception:
        return None


def _prompt_target_config() -> Path:
    return Path(os.getenv("MEDAI_PROMPT_TARGET_CONFIG", "configs/student_3d_prompt_target_organs.json")).resolve()


def _same_file_fingerprint(src: Path, dst: Path) -> bool:
    src_fp = _file_fingerprint(src)
    dst_fp = _file_fingerprint(dst)
    return bool(src_fp and dst_fp and src_fp.get("size") == dst_fp.get("size") and src_fp.get("mtime_ns") == dst_fp.get("mtime_ns"))


def _compare_cache_payload(ct_image: str | Path, mask_a: str | Path, mask_b: str | Path, organ: str, options: dict[str, Any]) -> dict[str, Any]:
    return {
        "ct": _file_fingerprint(ct_image),
        "mask_a": _file_fingerprint(mask_a),
        "mask_b": _file_fingerprint(mask_b),
        "organ": str(organ),
        "options": {k: options.get(k) for k in sorted(options) if k not in {"run_id", "csv_path"}},
    }


def _compare_cache_key(ct_image: str | Path, mask_a: str | Path, mask_b: str | Path, organ: str, options: dict[str, Any]) -> str:
    payload = _compare_cache_payload(ct_image, mask_a, mask_b, organ, options)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _invert_compare_decision(decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(decision or {})
    if out.get("winner") == "a":
        out["winner"] = "b"
    elif out.get("winner") == "b":
        out["winner"] = "a"
    out["cache_orientation"] = "inverted"
    return out


def _load_cached_compare(cache_json: Path, *, invert: bool, output_json: Path) -> dict[str, Any] | None:
    if not cache_json.exists():
        return None
    try:
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get("status") not in {"success", "stub", "dry_run"}:
        return None
    result = dict(cached)
    if invert:
        result["decision"] = _invert_compare_decision(result.get("decision", {}))
        result["mask_a"], result["mask_b"] = result.get("mask_b"), result.get("mask_a")
    result["output_json"] = str(output_json)
    result["cache_status"] = "reused_labelcritic_compare_fingerprint"
    try:
        write_json(output_json, result)
    except Exception:
        pass
    return result


def _write_compare_cache(cache_json: Path, result: dict[str, Any], cache_key: str, payload: dict[str, Any]) -> None:
    try:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_json, {**result, "compare_cache_key": cache_key, "compare_cache_payload": payload})
    except Exception:
        pass


def _projection_cache_key(ct_image: str | Path, mask_file: str | Path, organ: str) -> str:
    payload = {
        "ct": _file_fingerprint(ct_image),
        "mask": _file_fingerprint(mask_file),
        "organ": str(organ),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_cached_projection(work_dir: Path, expected_key: str) -> dict[str, Any] | None:
    manifest = work_dir / "projection_manifest.json"
    if not manifest.exists():
        return None
    try:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get("projection_cache_key") != expected_key:
        return None
    pngs = [str(p) for p in cached.get("saved_projections", []) if str(p).endswith(".png") and Path(p).exists()]
    if not pngs:
        return None
    cached["saved_projections"] = pngs
    cached["cache_status"] = "reused_projection_cache"
    return cached


def _write_projection_cache_manifest(work_dir: Path, cache_key: str, proj: dict[str, Any]) -> None:
    try:
        manifest = {**proj, "projection_cache_key": cache_key}
        write_json(work_dir / "projection_manifest.json", manifest)
    except Exception:
        pass


def _prepare_mask_folder(mask: Path, organ: str, dst_root: Path, label: str) -> Path:
    """LabelCritic expects a folder of organ masks. Accept either folder or file."""
    out = dst_root / label
    out.mkdir(parents=True, exist_ok=True)
    if mask.is_dir():
        src = mask / f"{organ}.nii.gz"
        if src.exists():
            dst = out / f"{organ}.nii.gz"
            if not _same_file_fingerprint(src, dst):
                shutil.copy2(src, dst)
        _ensure_left_right_projection_companion(out, organ)
        return out.resolve()
    dst = out / f"{organ}.nii.gz"
    if mask.exists() and not _same_file_fingerprint(mask, dst):
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
    if not src.exists():
        return None
    if not _same_file_fingerprint(src, dst):
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

    # Scope to the current run's block using run_id. Never parse an append-only
    # log tail from an older LabelCritic call when the current run_id is absent.
    if run_id:
        pattern = rf"Run ID:\s*{re.escape(run_id)}\b(.*?)(?=\nRun ID:\s*[0-9A-Za-z_-]+\b|\Z)"
        m = re.search(pattern, full_text, re.S)
        if not m:
            return {
                "winner": "uncertain",
                "confidence": 0.0,
                "reason": f"LabelCritic log exists but run_id={run_id} was not found; refusing to parse stale log tail.",
                "parse_status": "run_id_not_found",
                "csv_path": str(csv_path) if csv_path else None,
                "csv_rows": csv_rows,
            }
        text = m.group(0)
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


def _resolve_vlm_model(base_url: str, port: int, model: str | None) -> str | None:
    """Return the VLM model id to call: explicit override, else first id served."""
    if model:
        return model
    host, p = _normalize_labelcritic_base_url(base_url, port)
    try:
        import requests as req_lib

        resp = req_lib.get(f"{host}:{p}/v1/models", timeout=10, proxies={"http": None, "https": None})
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0].get("id")
    except Exception:
        return None
    return None


def _call_vlm_grade(base_url: str, port: int, model: str, prompt: str, image_paths: list[str], timeout: int) -> str:
    """Single OpenAI-compatible (vLLM) chat call with images; returns raw content."""
    import base64

    import requests as req_lib

    host, p = _normalize_labelcritic_base_url(base_url, port)
    content: list[dict[str, Any]] = []
    for ip in image_paths:
        b64 = base64.b64encode(Path(ip).read_bytes()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 400, "temperature": 0}
    resp = req_lib.post(
        f"{host}:{p}/v1/chat/completions",
        json=payload, timeout=timeout, proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_grade_response(raw: str, accept_grade: float) -> dict[str, Any]:
    """Extract normalized grade semantics from a VLM absolute-quality reply."""
    import re

    label_to_grade = {
        "good": 0.9,
        "acceptable": 0.65,
        "bad": 0.1,
        "reject": 0.0,
    }

    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            grade = parsed.get("grade")
            grade_label = parsed.get("grade_label") or parsed.get("quality") or parsed.get("bucket")
            hard_failure_reason = parsed.get("hard_failure_reason")
            if isinstance(grade, str) and grade.strip().lower() in label_to_grade:
                grade_label = grade.strip().lower()
                grade = label_to_grade[grade_label]
            elif grade_label is not None:
                grade_label = str(grade_label).strip().lower()
                if grade_label in label_to_grade:
                    grade = label_to_grade[grade_label] if grade is None else grade
            grade = float(grade)
            grade = min(1.0, max(0.0, grade))
            if not grade_label:
                if grade >= 0.85:
                    grade_label = "good"
                elif grade >= accept_grade:
                    grade_label = "acceptable"
                else:
                    grade_label = "bad"
            accept = parsed.get("accept")
            if not isinstance(accept, bool):
                accept = grade >= accept_grade
            return {
                "grade": round(grade, 4),
                "grade_label": grade_label,
                "accept": bool(accept),
                "reason": str(parsed.get("reason", ""))[:300],
                "hard_failure_reason": str(hard_failure_reason)[:300] if hard_failure_reason else None,
                "parse_status": "success",
            }
    except Exception:
        pass
    return {
        "grade": None,
        "grade_label": None,
        "accept": None,
        "reason": (raw or "")[:300],
        "hard_failure_reason": "unparseable_grade_response",
        "parse_status": "unparseable",
    }


GRADE_PROMPT_TEMPLATE = """You are a medical imaging QA expert. The image shows a CT slice with a single predicted {organ} segmentation highlighted in color.

Grade how well the highlighted region matches the true {organ} in this slice:
- good: accurate location, shape and boundaries.
- acceptable: correct location with usable but imperfect boundaries.
- bad: mostly wrong, grossly mislocated, empty, or clearly a different organ.

Also provide a numeric grade:
- 0.9 for good
- 0.65 for acceptable
- 0.1 for bad

Localising small or elongated structures from one slice is hard: if you are unsure whether the location is correct but plausible, prefer acceptable over bad. Only use bad when the highlighted region is clearly not the {organ}, clearly empty, or grossly implausible. Set accept=true only for good/acceptable.

Respond with JSON only, no other text:
{{"grade_label": "<good|acceptable|bad>", "grade": <0.0-1.0>, "accept": <true|false>, "reason": "<brief anatomical reasoning>", "hard_failure_reason": "<optional: empty|wrong_organ|gross_mislocation|none>"}}"""


def run_labelcritic_grade_batch(
    jobs: list[dict[str, Any]],
    *,
    base_url: str = "http://localhost",
    port: int = 8000,
    vlm_model: str | None = None,
    accept_grade: float = 0.5,
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    projection_backend: str = "auto",
    dry_run: bool = False,
    timeout_sec: int = 300,
    concurrency: int = 2,
) -> list[dict[str, Any]]:
    """Batch/queue LC-2 grade calls with shared projection prep and bounded VLM concurrency."""
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending: list[dict[str, Any]] = []

    for idx, job in enumerate(jobs):
        out_json = Path(job["output_json"]).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if out_json.exists():
            try:
                cached = json.loads(out_json.read_text(encoding="utf-8"))
                if cached.get("status") in {"success", "skipped", "dry_run"}:
                    cached["cache_status"] = "reused_labelcritic_grade"
                    results[idx] = cached
                    continue
            except Exception:
                pass
        organ = str(job["organ"])
        mask = Path(job["mask"])
        ct_image = job["ct_image"]
        base = {
            "stage": "labelcritic_grade", "organ": organ, "mask": str(mask),
            "ct_image": str(ct_image), "output_json": str(out_json),
            "accept_grade": float(job.get("accept_grade", accept_grade)),
            "batch_status": "queued_grade",
        }
        if dry_run:
            result = {**base, "status": "dry_run", "grade": None, "accept": None, "reason": "dry-run: VLM grade skipped"}
            write_json(out_json, result)
            results[idx] = result
            continue
        mask_file = mask / f"{organ}.nii.gz" if mask.is_dir() else mask
        if not mask_file.exists():
            result = {**base, "status": "skipped", "grade": None, "accept": None, "reason": "mask file missing"}
            write_json(out_json, result)
            results[idx] = result
            continue
        try:
            import nibabel as nib
            import numpy as np
            if int((np.asanyarray(nib.load(str(mask_file)).dataobj) > 0).sum()) == 0:
                result = {**base, "status": "skipped", "grade": None, "accept": None, "reason": "zero-volume mask: no positive voxels; structural QC handles this without VLM", "parse_status": "zero_volume_mask"}
                write_json(out_json, result)
                results[idx] = result
                continue
        except Exception:
            pass
        pending.append({"index": idx, "base": base, "ct_image": ct_image, "mask_file": mask_file, "organ": organ, "output_json": out_json, "accept_grade": float(job.get("accept_grade", accept_grade))})

    if not pending:
        return [r for r in results if r is not None]

    model = _resolve_vlm_model(base_url, port, vlm_model)
    if not model:
        for item in pending:
            result = {**item["base"], "status": "skipped", "grade": None, "accept": None, "reason": "no VLM model/server available"}
            write_json(item["output_json"], result)
            results[item["index"]] = result
        return [r for r in results if r is not None]

    prepared: list[dict[str, Any]] = []
    for item in pending:
        cache_key = _projection_cache_key(item["ct_image"], item["mask_file"], item["organ"])
        work_dir = item["output_json"].parent / "grade_projection_cache" / f"{item['organ']}_{cache_key}"
        proj = _load_cached_projection(work_dir, cache_key)
        if proj is None:
            try:
                proj = build_projection(
                    item["ct_image"], item["mask_file"], None, work_dir / "projections", organ=item["organ"],
                    projection_backend=projection_backend, labelcritic_root=labelcritic_root,
                )
                _write_projection_cache_manifest(work_dir, cache_key, proj)
            except Exception as exc:
                result = {**item["base"], "status": "failed", "grade": None, "accept": None, "reason": f"projection error: {exc}"}
                write_json(item["output_json"], result)
                results[item["index"]] = result
                continue
        pngs = [p for p in (proj.get("saved_projections") or []) if str(p).endswith(".png") and Path(p).exists()][:2]
        if not pngs:
            result = {**item["base"], "status": "skipped", "grade": None, "accept": None, "reason": "no projection image produced", "projection": proj}
            write_json(item["output_json"], result)
            results[item["index"]] = result
            continue
        prepared.append({**item, "projection": proj, "pngs": pngs})

    def _grade_one(item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        prompt = GRADE_PROMPT_TEMPLATE.format(organ=item["organ"])
        try:
            raw = _call_vlm_grade(base_url, port, model, prompt, item["pngs"], timeout=min(timeout_sec, 300))
            parsed = _parse_grade_response(raw, item["accept_grade"])
            status = "success" if parsed.get("parse_status") == "success" else "failed"
            result = {**item["base"], "status": status, "vlm_model": model, "raw_response": raw[:1000], **parsed, "batch_status": "batched_grade"}
        except Exception as exc:
            result = {**item["base"], "status": "failed", "grade": None, "accept": None, "reason": f"VLM call failed: {exc}", "batch_status": "batched_grade"}
        return item["index"], result

    max_workers = max(1, int(concurrency or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_grade_one, item) for item in prepared]
        for future in as_completed(futures):
            idx, result = future.result()
            write_json(Path(result["output_json"]), result)
            results[idx] = result

    return [r for r in results if r is not None]


def run_labelcritic_grade(
    ct_image: str | Path,
    mask: str | Path,
    organ: str,
    output_json: str | Path,
    base_url: str = "http://localhost",
    port: int = 8000,
    vlm_model: str | None = None,
    accept_grade: float = 0.5,
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    projection_backend: str = "auto",
    dry_run: bool = False,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """Absolute single-mask quality grade via the VLM expert.

    Returns a dict with status and (on success) ``grade`` (0-1), ``accept``
    (bool), ``reason``. Degrades gracefully to status ``dry_run`` / ``skipped``
    / ``failed`` with ``grade=None, accept=None`` whenever projection or the VLM
    server is unavailable, so callers can treat "no verdict" as non-blocking.
    """
    out_json = Path(output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
            if cached.get("status") in {"success", "skipped", "dry_run"}:
                cached["cache_status"] = "reused_labelcritic_grade"
                return cached
        except Exception:
            pass
    m = Path(mask)
    base: dict[str, Any] = {
        "stage": "labelcritic_grade", "organ": organ, "mask": str(m),
        "ct_image": str(ct_image), "output_json": str(out_json),
        "accept_grade": accept_grade,
    }

    def _finish(extra: dict[str, Any]) -> dict[str, Any]:
        result = {**base, **extra}
        write_json(out_json, result)
        return result

    if dry_run:
        return _finish({"status": "dry_run", "grade": None, "accept": None, "reason": "dry-run: VLM grade skipped"})

    mask_file = m / f"{organ}.nii.gz" if m.is_dir() else m
    if not mask_file.exists():
        return _finish({"status": "skipped", "grade": None, "accept": None, "reason": "mask file missing"})

    # Deterministic short-circuit: a zero-volume mask has no positive voxels.
    # Do not call it "empty" (the file exists) and do not let the VLM grade a
    # blank overlay; structural QC / expected-anatomy policy handles it.
    try:
        import nibabel as nib
        import numpy as np

        if int((np.asanyarray(nib.load(str(mask_file)).dataobj) > 0).sum()) == 0:
            return _finish({"status": "skipped", "grade": None, "accept": None,
                            "reason": "zero-volume mask: no positive voxels; structural QC handles this without VLM", "parse_status": "zero_volume_mask"})
    except Exception:
        pass

    model = _resolve_vlm_model(base_url, port, vlm_model)
    if not model:
        return _finish({"status": "skipped", "grade": None, "accept": None, "reason": "no VLM model/server available"})

    cache_key = _projection_cache_key(ct_image, mask_file, organ)
    work_dir = out_json.parent / "grade_projection_cache" / f"{organ}_{cache_key}"
    proj = _load_cached_projection(work_dir, cache_key)
    if proj is None:
        try:
            proj = build_projection(
                ct_image, mask_file, None, work_dir / "projections", organ=organ,
                projection_backend=projection_backend, labelcritic_root=labelcritic_root,
            )
            _write_projection_cache_manifest(work_dir, cache_key, proj)
        except Exception as exc:
            return _finish({"status": "failed", "grade": None, "accept": None, "reason": f"projection error: {exc}"})

    pngs = [p for p in (proj.get("saved_projections") or []) if str(p).endswith(".png") and Path(p).exists()][:2]
    if not pngs:
        return _finish({"status": "skipped", "grade": None, "accept": None, "reason": "no projection image produced", "projection": proj})

    prompt = GRADE_PROMPT_TEMPLATE.format(organ=organ)
    try:
        raw = _call_vlm_grade(base_url, port, model, prompt, pngs, timeout=min(timeout_sec, 300))
    except Exception as exc:
        return _finish({"status": "failed", "grade": None, "accept": None, "reason": f"VLM call failed: {exc}"})

    parsed = _parse_grade_response(raw, accept_grade)
    status = "success" if parsed.get("parse_status") == "success" else "failed"
    return _finish({"status": status, "vlm_model": model, "raw_response": raw[:1000], **parsed})


def run_labelcritic_compare_batch(
    jobs: list[dict[str, Any]],
    *,
    labelcritic_root: str | Path = "third_party/LabelCritic-main",
    backend: str = "labelcritic",
    base_url: str = "http://localhost",
    port: int = 8000,
    dry_run: bool = False,
    strict_alignment: bool = False,
    timeout_sec: int = 900,
) -> list[dict[str, Any]]:
    """Batch multiple LabelCritic compare jobs through one CompareOrgan.py process.

    Each job accepts the same core fields as run_labelcritic_compare: ct_image,
    mask_a, mask_b, organ, output_json, plus LabelCritic option booleans. Cached
    jobs are returned immediately and omitted from the manifest.
    """
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    pending: list[dict[str, Any]] = []
    lc_root = Path(labelcritic_root).resolve()
    script = lc_root / "CompareOrgan.py"
    normalized_base_url, normalized_port = _normalize_labelcritic_base_url(base_url, port)

    for idx, job in enumerate(jobs):
        ct = Path(job["ct_image"]).resolve()
        a = Path(job["mask_a"]).resolve()
        b = Path(job["mask_b"]).resolve()
        organ = str(job["organ"])
        out_json = Path(job["output_json"]).resolve()
        out_dir = out_json.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_json.exists():
            try:
                cached = json.loads(out_json.read_text(encoding="utf-8"))
                if cached.get("status") in {"success", "stub", "dry_run"}:
                    cached["cache_status"] = "reused_labelcritic_compare"
                    results[idx] = cached
                    continue
            except Exception:
                pass
        preliminary_options = {
            "backend": backend,
            "strict_alignment": strict_alignment,
            "no_dice_check": bool(job.get("no_dice_check", False)),
            "no_dual_confirmation": bool(job.get("no_dual_confirmation", False)),
            "simple_prompt_ablation": bool(job.get("simple_prompt_ablation", False)),
            "conservative_dual": bool(job.get("conservative_dual", False)),
            "skip_organ_presence_gate": bool(job.get("skip_organ_presence_gate", False)),
            "strict_choice_prompt": bool(job.get("strict_choice_prompt", False)),
            "prompt_target_config": _file_fingerprint(_prompt_target_config()),
        }
        compare_cache_root = out_dir / "compare_cache"
        compare_key = _compare_cache_key(ct, a, b, organ, preliminary_options)
        reverse_compare_key = _compare_cache_key(ct, b, a, organ, preliminary_options)
        cached_compare = _load_cached_compare(compare_cache_root / f"{compare_key}.json", invert=False, output_json=out_json)
        if cached_compare is not None:
            results[idx] = cached_compare
            continue
        cached_compare = _load_cached_compare(compare_cache_root / f"{reverse_compare_key}.json", invert=True, output_json=out_json)
        if cached_compare is not None:
            results[idx] = cached_compare
            continue

        work_dir = out_dir / "compare_work" / compare_key
        work_dir.mkdir(parents=True, exist_ok=True)
        run_id = compare_key[:8]
        mask1_folder = _prepare_mask_folder(a, organ, work_dir, "mask1")
        mask2_folder = _prepare_mask_folder(b, organ, work_dir, "mask2")
        pending.append({
            "index": idx,
            "ct": str(ct),
            "mask1": str(mask1_folder),
            "mask2": str(mask2_folder),
            "organ": organ,
            "base_url": normalized_base_url,
            "port": normalized_port,
            "base_output": str(work_dir / "comparison_results"),
            "base_csv": str(work_dir / "results"),
            "log_file": str(work_dir / "comparison_summary.log"),
            "run_id": run_id,
            "output_json": str(out_json),
            "mask_a": str(a),
            "mask_b": str(b),
            "compare_key": compare_key,
            "compare_cache_root": str(compare_cache_root),
            "preliminary_options": preliminary_options,
            "no_dice_check": bool(job.get("no_dice_check", False)),
            "no_dual_confirmation": bool(job.get("no_dual_confirmation", False)),
            "simple_prompt_ablation": bool(job.get("simple_prompt_ablation", False)),
            "conservative_dual": bool(job.get("conservative_dual", False)),
            "skip_organ_presence_gate": bool(job.get("skip_organ_presence_gate", False)),
            "strict_choice_prompt": bool(job.get("strict_choice_prompt", False)),
        })

    if pending and (dry_run or backend == "stub"):
        for item in pending:
            out_json = Path(item["output_json"])
            decision = {"winner": "uncertain", "confidence": 0.5, "reason": "batch stub backend", "parse_status": "stub"}
            result = {
                "stage": "labelcritic", "status": "dry_run" if dry_run else "stub",
                "backend": backend, "organ": item["organ"], "ct_image": item["ct"],
                "mask_a": item["mask_a"], "mask_b": item["mask_b"], "output_json": str(out_json),
                "projection": {"status": "skipped", "reason": "batch stub"},
                "command": None, "normalized_base_url": normalized_base_url, "normalized_port": normalized_port,
                "decision": decision, "labelcritic_options": item["preliminary_options"],
                "batch_status": "batched_stub",
            }
            _write_compare_cache(Path(item["compare_cache_root"]) / f"{item['compare_key']}.json", result, item["compare_key"], _compare_cache_payload(item["ct"], item["mask_a"], item["mask_b"], item["organ"], item["preliminary_options"]))
            write_json(out_json, result)
            results[item["index"]] = result
        return [r for r in results if r is not None]

    if pending:
        if not script.exists():
            for item in pending:
                result = {"stage": "labelcritic", "status": "failed", "reason": "CompareOrgan.py not found", "labelcritic_root": str(lc_root), "projection": None}
                write_json(Path(item["output_json"]), result)
                results[item["index"]] = result
            return [r for r in results if r is not None]
        batch_root = Path(pending[0]["output_json"]).parent / "compare_batch"
        batch_root.mkdir(parents=True, exist_ok=True)
        manifest = batch_root / f"batch_{int(time.time() * 1000)}.json"
        manifest_items = [{k: v for k, v in item.items() if k in {
            "ct", "mask1", "mask2", "organ", "base_url", "port", "base_output", "base_csv",
            "log_file", "run_id", "no_dice_check", "no_dual_confirmation", "simple_prompt_ablation",
            "conservative_dual", "skip_organ_presence_gate", "strict_choice_prompt"
        }} for item in pending]
        write_json(manifest, {"items": manifest_items, "output_json": str(batch_root / "batch_result.json")})
        command = ["python", str(script), "--batch_manifest", str(manifest), "--base_url", normalized_base_url, "--port", str(normalized_port)]
        start = time.time()
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout_sec, cwd=str(lc_root))
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(
                command,
                124,
                stdout=subprocess_text(exc.stdout),
                stderr=subprocess_text(exc.stderr) + f"\n[labelcritic batch] Timeout after {timeout_sec}s",
            )
            timed_out = True
        elapsed = time.time() - start
        for item in pending:
            out_json = Path(item["output_json"])
            log_file = Path(item["log_file"])
            csv_path = Path(item["base_csv"]) / item["run_id"] / f"{item['organ']}.csv"
            decision = _parse_labelcritic_log(log_file, Path(item["mask1"]), Path(item["mask2"]), run_id=item["run_id"], csv_path=csv_path)
            result = {
                "stage": "labelcritic", "status": "timed_out" if timed_out else ("success" if completed.returncode == 0 else "failed"),
                "backend": backend, "organ": item["organ"], "ct_image": item["ct"],
                "mask_a": item["mask_a"], "mask_b": item["mask_b"], "output_json": str(out_json),
                "projection": {"status": "skipped", "reason": "projection handled by batched CompareOrgan.py"},
                "command": command, "return_code": completed.returncode, "runtime_sec": round(elapsed, 3),
                "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
                "log_file": str(log_file), "normalized_base_url": normalized_base_url, "normalized_port": normalized_port,
                "decision": decision, "labelcritic_options": item["preliminary_options"], "batch_status": "batched_compare",
            }
            if result.get("status") in {"success", "stub", "dry_run"}:
                _write_compare_cache(Path(item["compare_cache_root"]) / f"{item['compare_key']}.json", result, item["compare_key"], _compare_cache_payload(item["ct"], item["mask_a"], item["mask_b"], item["organ"], item["preliminary_options"]))
            write_json(out_json, result)
            results[item["index"]] = result

    return [r for r in results if r is not None]


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
    if out_json.exists():
        try:
            cached = json.loads(out_json.read_text(encoding="utf-8"))
            if cached.get("status") in {"success", "stub", "dry_run"}:
                cached["cache_status"] = "reused_labelcritic_compare"
                return cached
        except Exception:
            pass
    preliminary_options = {
        "backend": backend,
        "strict_alignment": strict_alignment,
        "no_dice_check": no_dice_check,
        "no_dual_confirmation": no_dual_confirmation,
        "simple_prompt_ablation": simple_prompt_ablation,
        "conservative_dual": conservative_dual,
        "skip_organ_presence_gate": skip_organ_presence_gate,
        "strict_choice_prompt": strict_choice_prompt,
        "prompt_target_config": _file_fingerprint(_prompt_target_config()),
    }
    compare_cache_root = out_dir / "compare_cache"
    compare_key = _compare_cache_key(ct, a, b, organ, preliminary_options)
    reverse_compare_key = _compare_cache_key(ct, b, a, organ, preliminary_options)
    cached_compare = _load_cached_compare(compare_cache_root / f"{compare_key}.json", invert=False, output_json=out_json)
    if cached_compare is not None:
        return cached_compare
    cached_compare = _load_cached_compare(compare_cache_root / f"{reverse_compare_key}.json", invert=True, output_json=out_json)
    if cached_compare is not None:
        return cached_compare

    work_dir = out_dir / "compare_work" / compare_key
    work_dir.mkdir(parents=True, exist_ok=True)

    lc_root = Path(labelcritic_root).resolve()
    script = lc_root / "CompareOrgan.py"
    normalized_base_url, normalized_port = _normalize_labelcritic_base_url(base_url, port)

    # Stable run_id/work folder lets interrupted compare artifacts be inspected and reused.
    run_id = compare_key[:8]

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
        _write_compare_cache(compare_cache_root / f"{compare_key}.json", result, compare_key, _compare_cache_payload(ct, a, b, organ, preliminary_options))
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
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=subprocess_text(exc.stdout),
            stderr=subprocess_text(exc.stderr) + f"\n[labelcritic] Timeout after {timeout_sec}s",
        )
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
    if result.get("status") in {"success", "stub", "dry_run"}:
        _write_compare_cache(compare_cache_root / f"{compare_key}.json", result, compare_key, _compare_cache_payload(ct, a, b, organ, preliminary_options))
    write_json(out_json, result)
    return result
