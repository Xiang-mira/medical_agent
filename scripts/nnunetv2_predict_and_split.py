#!/usr/bin/env python3
"""Run an nnUNet v2 checkpoint on one CT and split the combined label map into organ-wise masks.

This wrapper is designed for the teacher-provided checkpoint folders exported from Google Drive:
CADS_series, MOOSE_series, nnUNet_private, and VSmTrans.

It never bundles private weights. It assumes the checkpoint folder already exists locally or on the
server and only standardizes I/O for the medai CLI.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import datetime
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.runtime_resolver import (  # noqa: E402
    resolve_nnunet_predict_from_modelfolder,
    resolve_nnunet_predictor,
)


ORGAN_ALIASES = {
    "cerebrospinal_fluid": "csf",
    "celiac_aa_celiac_artery": "celiac_aa",
    "common_iliac_artery_left": "iliac_artery_left",
    "common_iliac_artery_right": "iliac_artery_right",
    "common_iliac_vein_left": "iliac_vena_left",
    "common_iliac_vein_right": "iliac_vena_right",
    "compact_bone": "compact bone",
    "eyeball": "eye balls",
    "gland_structure": "glands",
    "gray_matter": "gray matter",
    "muscle_of_head": "head muscles",
    "inferior_vena_cava": "postcava",
    "small_intestine": "intestine",
    "spongy_bone": "spongy bone",
    "white_matter": "white matter",
}


def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _worker_count(explicit: int | None, *, diagnostic: bool, default: int | None) -> int | None:
    if explicit is not None:
        return max(1, int(explicit))
    if diagnostic:
        return 1
    return default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _tail_file(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_stage(stage_log: Path, status_path: Path, stage: str, **payload) -> None:
    event = {"timestamp": _utc_now(), "stage": stage, **payload}
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    with stage_log.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
    status = dict(event)
    status["last_stage"] = stage
    _write_json(status_path, status)


def _write_log_header(path: Path, stream_name: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[medai nnunet] stream={stream_name}",
        f"[medai nnunet] start_time={metadata.get('start_time')}",
        f"[medai nnunet] command={metadata.get('command_text')}",
        f"[medai nnunet] python_executable={metadata.get('python_executable')}",
        f"[medai nnunet] predict_executable={metadata.get('predict_executable')}",
        f"[medai nnunet] nnUNet_results={metadata.get('nnUNet_results')}",
        f"[medai nnunet] workdir={metadata.get('workdir')}",
        f"[medai nnunet] CUDA_VISIBLE_DEVICES={metadata.get('CUDA_VISIBLE_DEVICES')}",
        f"[medai nnunet] input_dir={metadata.get('input_dir')}",
        f"[medai nnunet] combined_output_dir={metadata.get('combined_output_dir')}",
        f"[medai nnunet] per_model_output_dir={metadata.get('per_model_output_dir')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_log_footer(
    path: Path,
    *,
    end_time: str,
    return_code: int,
    raw_child_return_code: int | None = None,
    effective_inference_status: str | None = None,
) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[medai nnunet] end_time={end_time}\n")
        handle.write(f"[medai nnunet] return_code={return_code}\n")
        if raw_child_return_code is not None:
            handle.write(f"[medai nnunet] raw_child_return_code={raw_child_return_code}\n")
        if effective_inference_status:
            handle.write(f"[medai nnunet] effective_inference_status={effective_inference_status}\n")
        handle.flush()


def _stream_pipe_to_log(pipe, log_path: Path) -> None:
    try:
        with log_path.open("a", encoding="utf-8", buffering=1, errors="replace") as handle:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                handle.write(line)
                handle.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _label_value_ids(value: object) -> list[int]:
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().lstrip("-").isdigit()):
        try:
            return [int(value)]
        except Exception:
            try:
                return [int(float(value))]
            except Exception:
                return []
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            ids.extend(_label_value_ids(item))
        return ids
    return []


def _requested_label_names(requested_organs: list[str] | None) -> set[str] | None:
    if not requested_organs:
        return None
    names: set[str] = set()
    reverse_aliases = {v: k for k, v in ORGAN_ALIASES.items()}
    for organ in requested_organs:
        name = str(organ).strip()
        if not name:
            continue
        names.add(name)
        names.add(ORGAN_ALIASES.get(name, name))
        names.add(reverse_aliases.get(name, name))
    return names


def _canonical_output_name(label_name: str, requested_organs: list[str] | None) -> str:
    raw_name = str(label_name).strip()
    if not raw_name:
        return raw_name
    if requested_organs:
        for organ in requested_organs:
            requested = str(organ).strip()
            if not requested:
                continue
            if raw_name == requested or raw_name == ORGAN_ALIASES.get(requested, requested):
                return requested
    reverse_aliases = {v: k for k, v in ORGAN_ALIASES.items()}
    return reverse_aliases.get(raw_name, raw_name)


def _dataset_label_sets(dataset_json: Path, requested_organs: list[str] | None) -> tuple[set[int], dict[str, list[int]]]:
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    raw_labels = data.get("labels") or {}
    legal_ids = {0}
    target_names = _requested_label_names(requested_organs)
    target_ids: dict[str, list[int]] = {}
    for raw_name, raw_value in raw_labels.items():
        name = str(raw_name).strip()
        ids = [int(v) for v in _label_value_ids(raw_value)]
        legal_ids.update(ids)
        if name.lower() == "background":
            continue
        nonzero_ids = [v for v in ids if v != 0]
        if target_names is None or name in target_names:
            target_ids[name] = nonzero_ids
    return legal_ids, target_ids


def _validate_combined_output(
    label_map: Path,
    input_image: Path,
    dataset_json: Path,
    requested_organs: list[str] | None,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "path": str(label_map),
        "valid": False,
        "reasons": [],
        "checks": {},
    }

    if not label_map.exists():
        validation["reasons"].append("combined_output_missing")
        validation["checks"]["exists"] = False
        return validation
    validation["checks"]["exists"] = True

    try:
        stat = label_map.stat()
    except Exception as exc:
        validation["reasons"].append(f"combined_output_stat_failed: {exc}")
        validation["checks"]["stat_readable"] = False
        return validation
    validation["size_bytes"] = int(stat.st_size)
    validation["mtime_ns"] = int(stat.st_mtime_ns)
    validation["checks"]["nonempty"] = stat.st_size > 0
    if stat.st_size <= 0:
        validation["reasons"].append("combined_output_empty_file")
        return validation

    try:
        import nibabel as nib
        import numpy as np

        pred_img = nib.load(str(label_map))
        input_img = nib.load(str(input_image))
        arr = np.asanyarray(pred_img.dataobj)
        unique_values = np.unique(arr)

        pred_shape = tuple(int(v) for v in pred_img.shape[:3])
        input_shape = tuple(int(v) for v in input_img.shape[:3])
        pred_spacing = tuple(float(v) for v in pred_img.header.get_zooms()[:3])
        input_spacing = tuple(float(v) for v in input_img.header.get_zooms()[:3])
        shape_matches = pred_shape == input_shape
        affine_matches = bool(np.allclose(pred_img.affine, input_img.affine, rtol=0, atol=1e-5))
        spacing_matches = bool(np.allclose(pred_spacing, input_spacing, rtol=0, atol=1e-5))

        validation["checks"]["nifti_loadable"] = True
        validation["shape"] = list(pred_shape)
        validation["input_shape"] = list(input_shape)
        validation["spacing"] = list(pred_spacing)
        validation["input_spacing"] = list(input_spacing)
        validation["checks"]["shape_matches_input"] = shape_matches
        validation["checks"]["affine_matches_input"] = affine_matches
        validation["checks"]["spacing_matches_input"] = spacing_matches
        if not shape_matches:
            validation["reasons"].append("shape_mismatch")
        if not affine_matches:
            validation["reasons"].append("affine_mismatch")
        if not spacing_matches:
            validation["reasons"].append("spacing_mismatch")

        legal_ids, target_ids = _dataset_label_sets(dataset_json, requested_organs)
        finite_integer_labels = bool(
            np.all(np.isfinite(unique_values))
            and np.all(np.equal(unique_values, np.round(unique_values)))
        )
        unique_ids = sorted({int(v) for v in unique_values}) if finite_integer_labels else []
        illegal_ids = sorted(set(unique_ids) - set(legal_ids)) if finite_integer_labels else []
        validation["labels_present"] = unique_ids if finite_integer_labels else [str(v) for v in unique_values.tolist()]
        validation["legal_label_ids"] = sorted(legal_ids)
        validation["checks"]["labels_are_integer"] = finite_integer_labels
        validation["checks"]["labels_are_legal"] = finite_integer_labels and not illegal_ids
        validation["illegal_label_ids"] = illegal_ids
        if not finite_integer_labels:
            validation["reasons"].append("labels_are_not_integer")
        if illegal_ids:
            validation["reasons"].append("labels_outside_dataset_json")

        target_voxels: dict[str, int] = {}
        for name, ids in sorted(target_ids.items()):
            if not ids:
                target_voxels[name] = 0
            elif len(ids) == 1:
                target_voxels[name] = int((arr == ids[0]).sum())
            else:
                target_voxels[name] = int(np.isin(arr, ids).sum())
        expected_target_nonzero = bool(target_voxels) and any(v > 0 for v in target_voxels.values())
        validation["expected_target_label_ids"] = {name: ids for name, ids in sorted(target_ids.items())}
        validation["expected_target_voxels"] = target_voxels
        validation["checks"]["expected_target_nonzero"] = expected_target_nonzero
        if not expected_target_nonzero:
            validation["reasons"].append("expected_target_missing_or_empty")

        validation["valid"] = bool(
            validation["checks"].get("exists")
            and validation["checks"].get("nonempty")
            and validation["checks"].get("nifti_loadable")
            and shape_matches
            and affine_matches
            and spacing_matches
            and validation["checks"].get("labels_are_legal")
            and expected_target_nonzero
        )
        return validation
    except Exception as exc:
        validation["checks"]["nifti_loadable"] = False
        validation["reasons"].append(f"nifti_load_failed: {exc}")
        return validation


def _combined_output_split_safe(validation: dict[str, Any] | None) -> bool:
    if not validation:
        return False
    checks = validation.get("checks") or {}
    required = (
        "exists",
        "nonempty",
        "nifti_loadable",
        "shape_matches_input",
        "affine_matches_input",
        "spacing_matches_input",
        "labels_are_legal",
    )
    return all(bool(checks.get(name)) for name in required)


def _validate_first_combined_output(
    combined_dir: Path,
    input_image: Path,
    dataset_json: Path,
    requested_organs: list[str] | None,
) -> tuple[Path | None, dict[str, Any]]:
    candidates = sorted(combined_dir.glob("*.nii.gz"))
    if not candidates:
        return None, {"valid": False, "reasons": ["combined_output_missing"], "checks": {"exists": False}}
    fallback: tuple[Path, dict[str, Any]] | None = None
    for candidate in candidates:
        validation = _validate_combined_output(candidate, input_image, dataset_json, requested_organs)
        if fallback is None:
            fallback = (candidate, validation)
        if validation.get("valid"):
            return candidate, validation
    assert fallback is not None
    return fallback


def _termination_signal(return_code: int | None) -> int | None:
    if return_code is None or return_code >= 0:
        return None
    return -int(return_code)


def _process_group_id(proc: subprocess.Popen) -> int | None:
    try:
        return os.getpgid(proc.pid)
    except Exception:
        return None


def _append_optional_stage(stage_log: Path | None, status_path: Path | None, stage: str, **payload) -> None:
    if stage_log is not None and status_path is not None:
        _append_stage(stage_log, status_path, stage, **payload)


def _terminate_process_group(
    proc: subprocess.Popen,
    *,
    stage_log: Path | None,
    status_path: Path | None,
    reason: str,
    termination_grace_sec: float,
) -> dict[str, Any]:
    pgid = _process_group_id(proc)
    payload = {"reason": reason, "child_pid": proc.pid, "process_group_id": pgid}
    _append_optional_stage(stage_log, status_path, "process_group_termination_started", **payload)

    sent_sigterm = False
    sent_sigkill = False
    wait_error: str | None = None
    try:
        if proc.poll() is None:
            try:
                if pgid is not None and pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
                sent_sigterm = True
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=max(0.1, float(termination_grace_sec)))
            except subprocess.TimeoutExpired:
                try:
                    if pgid is not None and pgid != os.getpgrp():
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        proc.kill()
                    sent_sigkill = True
                except ProcessLookupError:
                    pass
                proc.wait(timeout=max(0.1, float(termination_grace_sec)))
        else:
            proc.wait(timeout=0)
    except Exception as exc:
        wait_error = str(exc)

    raw_return_code = proc.returncode
    result = {
        **payload,
        "sent_sigterm": sent_sigterm,
        "sent_sigkill": sent_sigkill,
        "raw_child_return_code": raw_return_code,
        "termination_signal": _termination_signal(raw_return_code),
        "wait_error": wait_error,
    }
    _append_optional_stage(stage_log, status_path, "process_group_terminated", **result)
    return result


def _run_streaming_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None,
    stdout_log: Path,
    stderr_log: Path,
    metadata: dict,
    stage_log: Path | None = None,
    status_path: Path | None = None,
    combined_dir: Path | None = None,
    input_image: Path | None = None,
    dataset_json: Path | None = None,
    requested_organs: list[str] | None = None,
    combined_output_stable_sec: float = 30.0,
    post_export_shutdown_grace_sec: float | None = 300.0,
    process_termination_grace_sec: float = 30.0,
    poll_interval_sec: float = 5.0,
    prediction_timeout_sec: float | None = None,
) -> dict:
    """Run a subprocess while streaming logs and optionally recovering from nnUNet post-export hangs."""
    command_text = shlex.join([str(part) for part in cmd])
    run_meta = {**metadata, "command": [str(part) for part in cmd], "command_text": command_text}
    _write_log_header(stdout_log, "stdout", run_meta)
    _write_log_header(stderr_log, "stderr", run_meta)
    proc: subprocess.Popen[str] | None = None
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
    )
    child_pid = proc.pid
    process_group_id = _process_group_id(proc)
    threads: list[threading.Thread] = []
    assert proc.stdout is not None
    assert proc.stderr is not None
    for pipe, path in ((proc.stdout, stdout_log), (proc.stderr, stderr_log)):
        thread = threading.Thread(target=_stream_pipe_to_log, args=(pipe, path), daemon=True)
        thread.start()
        threads.append(thread)

    start_monotonic = time.monotonic()
    observed_stats: dict[Path, tuple[int, int, float]] = {}
    validation_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    detected_paths: set[Path] = set()
    combined_validation: dict[str, Any] = {}
    combined_valid = False
    combined_path: Path | None = None
    valid_output_at: float | None = None
    forced_cleanup = False
    forced_cleanup_reason: str | None = None
    timeout_cleanup = False
    cleanup_result: dict[str, Any] = {}
    return_code: int | None = None
    raw_child_return_code: int | None = None
    effective_status = "subprocess_running"

    try:
        while True:
            now = time.monotonic()
            polled_return_code = proc.poll()

            if (
                polled_return_code is None
                and combined_dir is not None
                and input_image is not None
                and dataset_json is not None
                and not combined_valid
            ):
                for candidate in sorted(combined_dir.glob("*.nii.gz")):
                    try:
                        stat = candidate.stat()
                    except Exception:
                        continue
                    if stat.st_size <= 0:
                        continue
                    if candidate not in detected_paths:
                        detected_paths.add(candidate)
                        _append_optional_stage(
                            stage_log,
                            status_path,
                            "combined_output_detected",
                            combined_label=str(candidate),
                            size_bytes=int(stat.st_size),
                            mtime_ns=int(stat.st_mtime_ns),
                        )
                    previous = observed_stats.get(candidate)
                    if previous and previous[0] == stat.st_size and previous[1] == stat.st_mtime_ns:
                        stable_since = previous[2]
                    else:
                        stable_since = now
                        observed_stats[candidate] = (int(stat.st_size), int(stat.st_mtime_ns), stable_since)
                    stable_for = now - stable_since
                    if stable_for < max(0.0, float(combined_output_stable_sec)):
                        continue
                    cache_key = (str(candidate), int(stat.st_size), int(stat.st_mtime_ns))
                    validation = validation_cache.get(cache_key)
                    if validation is None:
                        validation = _validate_combined_output(candidate, input_image, dataset_json, requested_organs)
                        validation["stable_for_sec"] = round(stable_for, 3)
                        validation["stable_size_bytes"] = int(stat.st_size)
                        validation["stable_mtime_ns"] = int(stat.st_mtime_ns)
                        validation_cache[cache_key] = validation
                    combined_validation = validation
                    combined_path = candidate
                    if validation.get("valid"):
                        combined_valid = True
                        valid_output_at = now
                        _append_optional_stage(
                            stage_log,
                            status_path,
                            "combined_output_validated",
                            combined_label=str(candidate),
                            combined_output_validation=validation,
                        )
                        break

            if polled_return_code is not None:
                raw_child_return_code = int(polled_return_code)
                return_code = raw_child_return_code
                effective_status = "completed" if return_code == 0 else "failed"
                break

            if (
                combined_valid
                and valid_output_at is not None
                and post_export_shutdown_grace_sec is not None
                and now - valid_output_at >= max(0.0, float(post_export_shutdown_grace_sec))
            ):
                forced_cleanup = True
                forced_cleanup_reason = "post_prediction_process_shutdown_hang"
                _append_optional_stage(
                    stage_log,
                    status_path,
                    "post_prediction_process_shutdown_hang",
                    combined_label=str(combined_path) if combined_path else None,
                )
                _append_optional_stage(
                    stage_log,
                    status_path,
                    "post_export_shutdown_hang_detected",
                    reason=forced_cleanup_reason,
                    combined_label=str(combined_path) if combined_path else None,
                    grace_sec=float(post_export_shutdown_grace_sec),
                )
                cleanup_result = _terminate_process_group(
                    proc,
                    stage_log=stage_log,
                    status_path=status_path,
                    reason=forced_cleanup_reason,
                    termination_grace_sec=process_termination_grace_sec,
                )
                raw_child_return_code = proc.returncode
                return_code = 0
                effective_status = "valid_output_completed_after_forced_process_cleanup"
                break

            if prediction_timeout_sec is not None and now - start_monotonic >= float(prediction_timeout_sec):
                timeout_cleanup = True
                forced_cleanup_reason = "prediction_timeout"
                cleanup_result = _terminate_process_group(
                    proc,
                    stage_log=stage_log,
                    status_path=status_path,
                    reason=forced_cleanup_reason,
                    termination_grace_sec=process_termination_grace_sec,
                )
                raw_child_return_code = proc.returncode
                return_code = 124
                effective_status = "timed_out"
                break

            time.sleep(max(0.05, float(poll_interval_sec)))

        if (
            combined_dir is not None
            and input_image is not None
            and dataset_json is not None
            and return_code == 0
            and not combined_validation
        ):
            combined_path, combined_validation = _validate_first_combined_output(
                combined_dir,
                input_image,
                dataset_json,
                requested_organs,
            )
            if combined_path is not None and combined_path not in detected_paths and combined_path.exists():
                try:
                    stat = combined_path.stat()
                    _append_optional_stage(
                        stage_log,
                        status_path,
                        "combined_output_detected",
                        combined_label=str(combined_path),
                        size_bytes=int(stat.st_size),
                        mtime_ns=int(stat.st_mtime_ns),
                    )
                except Exception:
                    _append_optional_stage(stage_log, status_path, "combined_output_detected", combined_label=str(combined_path))
            if combined_validation.get("valid"):
                combined_valid = True
                _append_optional_stage(
                    stage_log,
                    status_path,
                    "combined_output_validated",
                    combined_label=str(combined_path) if combined_path else None,
                    combined_output_validation=combined_validation,
                )
    finally:
        if proc is not None and proc.poll() is None:
            cleanup_result = _terminate_process_group(
                proc,
                stage_log=stage_log,
                status_path=status_path,
                reason="wrapper_cleanup_finally",
                termination_grace_sec=process_termination_grace_sec,
            )
            raw_child_return_code = proc.returncode
            return_code = return_code if return_code is not None else 124
            effective_status = effective_status if effective_status != "subprocess_running" else "aborted_during_cleanup"
        for pipe in (proc.stdout, proc.stderr) if proc is not None else ():
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=5)

    assert return_code is not None
    if raw_child_return_code is None:
        raw_child_return_code = proc.returncode if proc is not None else return_code
    end_time = _utc_now()
    _append_log_footer(
        stdout_log,
        end_time=end_time,
        return_code=return_code,
        raw_child_return_code=raw_child_return_code,
        effective_inference_status=effective_status,
    )
    _append_log_footer(
        stderr_log,
        end_time=end_time,
        return_code=return_code,
        raw_child_return_code=raw_child_return_code,
        effective_inference_status=effective_status,
    )
    return {
        "return_code": return_code,
        "raw_child_return_code": raw_child_return_code,
        "termination_signal": _termination_signal(raw_child_return_code),
        "child_pid": child_pid,
        "process_group_id": process_group_id,
        "end_time": end_time,
        "command_text": command_text,
        "effective_inference_status": effective_status,
        "forced_process_cleanup": forced_cleanup,
        "forced_cleanup_reason": forced_cleanup_reason,
        "timed_out": timeout_cleanup,
        "combined_output_valid": bool(combined_valid),
        "combined_output_validation": combined_validation,
        "combined_label": str(combined_path) if combined_path else None,
        "cleanup": cleanup_result,
    }


def _start_gpu_monitor(path: Path, *, interval_sec: float, enabled: bool) -> tuple[threading.Event | None, threading.Thread | None]:
    if not enabled:
        return None, None
    stop_event = threading.Event()

    def monitor() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "timestamp,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib\n",
                encoding="utf-8",
            )
        while not stop_event.is_set():
            timestamp = _utc_now()
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=20,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    with path.open("a", encoding="utf-8", buffering=1) as handle:
                        for line in proc.stdout.strip().splitlines():
                            handle.write(f"{timestamp},{line}\n")
                            handle.flush()
            except Exception:
                pass
            stop_event.wait(max(1.0, float(interval_sec)))

    thread = threading.Thread(target=monitor, name="medai-nnunet-gpu-monitor", daemon=True)
    thread.start()
    return stop_event, thread


def _stop_gpu_monitor(stop_event: threading.Event | None, thread: threading.Thread | None) -> None:
    if stop_event is None:
        return
    stop_event.set()
    if thread is not None:
        thread.join(timeout=5)


def _coerce_label_id(value, label_arr=None) -> int | None:
    """Coerce a dataset.json label value to a single representative int id.

    nnUNet dataset.json labels are usually ``{name: int}``. For region-based
    datasets the value can be a list/tuple of ids (e.g. ``[1, 2]``). In that case
    we pick a representative id: if a combined label array is available we choose
    the first id that actually occurs in the array, otherwise the first id.
    """
    # Scalar int / float / numeric-string
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().lstrip("-").isdigit()):
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return None
    # Region-based: list/tuple of ids
    if isinstance(value, (list, tuple)) and value:
        ids: list[int] = []
        for v in value:
            try:
                ids.append(int(v))
            except Exception:
                try:
                    ids.append(int(float(v)))
                except Exception:
                    continue
        if not ids:
            return None
        if label_arr is not None:
            present = [i for i in ids if i != 0]
            for i in present:
                try:
                    import numpy as _np
                    if bool((_np.asarray(label_arr) == i).any()):
                        return i
                except Exception:
                    break
            for i in present:
                return i
        for i in ids:
            if i != 0:
                return i
        return ids[0]
    return None


def load_labels(dataset_json: Path, label_arr=None) -> dict[str, int]:
    """Read ``labels`` from a nnUNet dataset.json into ``{name: int_id}``.

    Tolerates both ``{name: id}`` and region-based ``{name: [ids]}`` formats.
    """
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    raw = data.get("labels") or {}
    labels: dict[str, int] = {}
    for name, value in raw.items():
        key = str(name).strip()
        if key.lower() == "background":
            continue
        ivalue = _coerce_label_id(value, label_arr=label_arr)
        if ivalue is None or ivalue == 0:
            continue
        labels[key] = ivalue
    return labels


def dump_local_labels(dataset_json: Path, out_path: Path, label_arr=None) -> dict[str, int]:
    """Write ``{local_label_name: int_id}`` to ``out_path`` and return the dict."""
    labels = load_labels(dataset_json, label_arr=label_arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    return labels


def split_labelmap(label_map: Path, dataset_json: Path, seg_dir: Path, requested_organs: list[str] | None = None) -> dict:
    try:
        import numpy as np
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel and numpy are required to split nnUNet label maps. Install requirements.txt first.") from exc

    labels = load_labels(dataset_json)
    if requested_organs:
        requested = _requested_label_names(requested_organs) or set()
        labels = {k: v for k, v in labels.items() if k in requested}
    img = nib.load(str(label_map))
    arr = np.asanyarray(img.dataobj)
    seg_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for organ, value in sorted(labels.items()):
        mask = (arr == value).astype("uint8")
        if int(mask.sum()) == 0:
            continue
        output_organ = _canonical_output_name(organ, requested_organs)
        out = seg_dir / f"{output_organ}.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out))
        written.append({
            "organ": output_organ,
            "source_label": organ,
            "label_value": value,
            "path": str(out),
            "voxels": int(mask.sum()),
        })
    return {"num_written": len(written), "written_masks": written, "labels_available": labels}


def validate_target_masks(seg_dir: Path, image: Path, expected_organs: list[str]) -> dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        return {"valid": False, "reason": f"nibabel_numpy_unavailable: {exc}", "masks": []}

    input_img = nib.load(str(image))
    input_shape = tuple(int(v) for v in input_img.shape[:3])
    input_spacing = tuple(float(v) for v in input_img.header.get_zooms()[:3])
    rows: list[dict[str, Any]] = []
    for organ in expected_organs:
        mask_path = seg_dir / f"{organ}.nii.gz"
        row: dict[str, Any] = {"organ": organ, "path": str(mask_path), "exists": mask_path.exists()}
        if not mask_path.exists():
            row["valid"] = False
            row["reason"] = "target_mask_missing"
            rows.append(row)
            continue
        try:
            mask_img = nib.load(str(mask_path))
            arr = np.asanyarray(mask_img.dataobj)
            unique_values = sorted({int(v) for v in np.unique(arr)})
            nonzero = int((arr != 0).sum())
            shape = tuple(int(v) for v in mask_img.shape[:3])
            spacing = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
            row.update({
                "shape": list(shape),
                "input_shape": list(input_shape),
                "spacing": list(spacing),
                "input_spacing": list(input_spacing),
                "labels": unique_values,
                "nonzero": nonzero,
                "binary": set(unique_values).issubset({0, 1}),
                "shape_matches_input": shape == input_shape,
                "spacing_matches_input": bool(np.allclose(spacing, input_spacing, rtol=0, atol=1e-5)),
                "affine_matches_input": bool(np.allclose(mask_img.affine, input_img.affine, rtol=0, atol=1e-5)),
            })
            row["valid"] = bool(
                row["binary"]
                and nonzero > 0
                and row["shape_matches_input"]
                and row["spacing_matches_input"]
                and row["affine_matches_input"]
            )
        except Exception as exc:
            row["valid"] = False
            row["reason"] = f"target_mask_load_failed: {exc}"
        rows.append(row)
    return {
        "valid": all(bool(row.get("valid")) for row in rows) if rows else True,
        "expected_organs": expected_organs,
        "masks": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input CT .nii.gz")
    ap.add_argument("--output", required=True, help="Case output folder. segmentations/*.nii.gz will be written here.")
    ap.add_argument("--dataset-id", required=True, help="nnUNet dataset id, e.g. 551 or 1339")
    ap.add_argument("--nnunet-results", required=True, help="Folder used as nnUNet_results")
    ap.add_argument("--dataset-json", required=True, help="dataset.json containing label map")
    ap.add_argument("--model-folder", default=None, help="Optional trained model folder. Use this for ePAI qchen76_2025_0421, which is launched with nnUNetv2_predict_from_modelfolder -m.")
    ap.add_argument("--workdir", default=None, help="Optional working directory for model-specific entrypoints. ePAI expects csv_header.csv in its train/binary folder.")
    ap.add_argument("--use-python-api", action="store_true", help="Use Python API directly instead of subprocess (for models trained with ePAI nnunetv2 but needing standard predict).")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    ap.add_argument("--predict-executable", default=None, help="Stable nnUNetv2_predict executable path/name. Defaults to NNUNETV2_PREDICT_EXECUTABLE, MEDAI_NNUNETV2_PREDICT, PATH lookup, then the HPC wrapper path.")
    ap.add_argument("--predict-from-modelfolder-executable", default=None, help="Stable nnUNetv2_predict_from_modelfolder executable path/name.")
    ap.add_argument("--sitecustomize-path", default=os.getenv("MEDAI_NNUNET_COMPAT_SITECUSTOMIZE"), help="Optional compatibility sitecustomize.py injected only into the nnUNet subprocess.")
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--device", default=None, help="Optional CUDA_VISIBLE_DEVICES value or cpu")
    ap.add_argument("--organs", default=None, help="Optional comma-separated organ names to split")
    ap.add_argument("--output-label-mode", choices=["all_organs", "pancreas_only"], default="all_organs", help="For ePAI native code compatibility; this wrapper keeps all combined labels unless --organs restricts splitting.")
    ap.add_argument("--per-model-dir", default=None, help="Optional directory where the unified per-model contract artifacts (combined_labels.nii.gz, local_labels.json) are written. Defaults to --output.")
    ap.add_argument("--diagnostic", action="store_true", help="Enable smoke/debug diagnostics: retain stage metadata and use conservative nnUNet worker counts.")
    ap.add_argument("--keep-workdir", action="store_true", help="Keep the prepared input and combined nnUNet output workdir under --output for timeout inspection.")
    ap.add_argument("--workdir-root", default=None, help="Optional root for retained diagnostic workdirs. Defaults to <output>/diagnostic_workdirs.")
    ap.add_argument("--preprocess-workers", type=int, default=None, help="Optional nnUNet preprocessing worker count; diagnostic mode defaults this to 1.")
    ap.add_argument("--export-workers", type=int, default=None, help="Optional nnUNet segmentation export worker count; diagnostic mode defaults this to 1.")
    ap.add_argument("--gpu-monitor", action="store_true", help="Diagnostic only: sample nvidia-smi to gpu_monitor.csv while prediction runs.")
    ap.add_argument("--gpu-monitor-interval-sec", type=float, default=60.0)
    ap.add_argument("--combined-output-stable-sec", type=float, default=_env_float("MEDAI_NNUNET_COMBINED_OUTPUT_STABLE_SEC", 30.0), help="Seconds a combined nnUNet output must keep the same size/mtime before post-export hang recovery can use it.")
    ap.add_argument("--post-export-shutdown-grace-sec", type=float, default=_env_float("MEDAI_NNUNET_POST_EXPORT_SHUTDOWN_GRACE_SEC", 300.0), help="Seconds to wait after a valid combined output before treating a still-running predictor as a post-export shutdown hang.")
    ap.add_argument("--process-termination-grace-sec", type=float, default=_env_float("MEDAI_NNUNET_PROCESS_TERMINATION_GRACE_SEC", 30.0), help="Seconds to wait after SIGTERM before SIGKILL during predictor process-group cleanup.")
    ap.add_argument("--post-export-poll-interval-sec", type=float, default=_env_float("MEDAI_NNUNET_POST_EXPORT_POLL_INTERVAL_SEC", 5.0), help="Polling interval for combined-output completion and child process state.")
    ap.add_argument("--prediction-timeout-sec", type=float, default=_env_float("MEDAI_NNUNET_PREDICTION_TIMEOUT_SEC", None), help="Optional internal predictor timeout. On timeout the predictor process group is cleaned up and the wrapper fails with code 124.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / "segmentations"
    dataset_json = Path(args.dataset_json).resolve()
    nnunet_results = Path(args.nnunet_results).resolve()
    model_folder = Path(args.model_folder).resolve() if args.model_folder else None
    workdir = Path(args.workdir).resolve() if args.workdir else None
    per_model_dir = Path(args.per_model_dir).resolve() if args.per_model_dir else output
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)
    per_model_seg_dir = per_model_dir / "segmentations"

    case_id = image.parent.name if image.name == "ct.nii.gz" else image.name.replace(".nii.gz", "").replace("_0000", "")
    requested_organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()] if args.organs else None
    predict_executable = resolve_nnunet_predictor(explicit=args.predict_executable, require_exists=not args.dry_run)
    predict_from_modelfolder_executable = resolve_nnunet_predict_from_modelfolder(
        explicit=args.predict_from_modelfolder_executable,
        require_exists=bool(model_folder and not args.dry_run),
    )

    if args.dry_run:
        if model_folder:
            command = [
                str(predict_from_modelfolder_executable), "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", "<input_csv>", "--output_csv", "<output_csv>",
                "--continue_prediction", "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
        else:
            command = [
                str(predict_executable), "-d", str(args.dataset_id), "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
                "-tr", args.trainer, "-c", args.configuration, "-f", str(args.folds), "-p", args.plans, "-chk", args.checkpoint_name, "--continue_prediction",
            ]
        print(json.dumps({"status": "dry_run", "command": command, "output": str(output), "seg_dir": str(seg_dir), "output_label_mode": args.output_label_mode}, indent=2))
        return 0

    if not image.exists():
        raise FileNotFoundError(image)
    if not dataset_json.exists():
        raise FileNotFoundError(dataset_json)
    if not nnunet_results.exists():
        raise FileNotFoundError(nnunet_results)
    if model_folder and not model_folder.exists():
        raise FileNotFoundError(model_folder)
    if workdir and not workdir.exists():
        raise FileNotFoundError(workdir)

    stdout_log = output / "nnunet_stdout.log"
    stderr_log = output / "nnunet_stderr.log"
    stage_log = output / "nnunet_stage_events.jsonl"
    status_path = output / "nnunet_status.json"
    metadata_path = output / "nnunet_run_metadata.json"
    gpu_monitor_path = output / "gpu_monitor.csv"
    stage_log.write_text("", encoding="utf-8")

    retain_workdir = bool(args.keep_workdir or args.diagnostic)
    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    if retain_workdir:
        retained_root = Path(args.workdir_root).resolve() if args.workdir_root else output / "diagnostic_workdirs"
        retained_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="medai_nnunet_", dir=str(retained_root))).resolve()
    else:
        tmp_holder = tempfile.TemporaryDirectory(prefix="medai_nnunet_")
        tmp = Path(tmp_holder.name).resolve()
    retained_workdir = str(tmp) if retain_workdir else None

    try:
        input_dir = tmp / "input"
        combined_dir = tmp / "combined"
        input_dir.mkdir(); combined_dir.mkdir()
        prepared = input_dir / f"{case_id}_0000.nii.gz"
        _append_stage(
            stage_log,
            status_path,
            "prepare_input_started",
            image=str(image),
            prepared_input=str(prepared),
            retained_workdir=retained_workdir,
        )
        shutil.copy2(image, prepared)
        _append_stage(stage_log, status_path, "prepare_input_completed", input_dir=str(input_dir), prepared_input=str(prepared))

        env = os.environ.copy()
        env["EPAI_OUTPUT_LABEL_MODE"] = args.output_label_mode
        env["nnUNet_results"] = str(nnunet_results)
        env.setdefault("nnUNet_raw", str(tmp / "nnUNet_raw"))
        env.setdefault("nnUNet_preprocessed", str(tmp / "nnUNet_preprocessed"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        if args.device:
            if args.device.lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env["CUDA_VISIBLE_DEVICES"] = args.device
        # If workdir contains its own nnunetv2 package, prepend it to PYTHONPATH so
        # the subprocess uses the correct architecture (e.g. VSmTrans vs ePAI).
        if workdir:
            vsm_nnunet = workdir / "nnunetv2"
            if not vsm_nnunet.exists():
                # workdir may be the nnUNet root (contains nnunetv2/ subdir)
                vsm_nnunet = workdir
            if (vsm_nnunet / "__init__.py").exists() or (vsm_nnunet / "nnunetv2").exists():
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(workdir) + (":" + existing if existing else "")
        if args.sitecustomize_path:
            sitecustomize = Path(args.sitecustomize_path).resolve()
            if not sitecustomize.exists():
                raise FileNotFoundError(sitecustomize)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(sitecustomize.parent) + (":" + existing if existing else "")

        metadata_doc = {
            "stage": "nnunet_wrapper",
            "status": "running",
            "start_time": _utc_now(),
            "python_executable": sys.executable,
            "predict_executable": str(predict_from_modelfolder_executable) if model_folder else str(predict_executable),
            "nnUNet_results": str(nnunet_results),
            "workdir": str(workdir) if workdir else None,
            "temporary_workdir": str(tmp),
            "retained_workdir": retained_workdir,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "input_dir": str(input_dir),
            "combined_output_dir": str(combined_dir),
            "per_model_output_dir": str(per_model_dir),
            "output": str(output),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stage_log": str(stage_log),
            "gpu_monitor_csv": str(gpu_monitor_path) if (args.gpu_monitor or _truthy(os.getenv("MEDAI_NNUNET_GPU_MONITOR"))) else None,
            "diagnostic": bool(args.diagnostic),
            "keep_workdir": bool(retain_workdir),
            "preprocess_workers": _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=None),
            "export_workers": _worker_count(args.export_workers, diagnostic=args.diagnostic, default=None),
        }
        _write_json(metadata_path, metadata_doc)

        gpu_stop, gpu_thread = _start_gpu_monitor(
            gpu_monitor_path,
            interval_sec=args.gpu_monitor_interval_sec,
            enabled=bool(args.gpu_monitor or _truthy(os.getenv("MEDAI_NNUNET_GPU_MONITOR"))),
        )
        proc_return_code: int
        proc_end_time: str
        proc_end_time = ""
        proc_return_code = 1
        proc_info: dict[str, Any] = {
            "return_code": 1,
            "raw_child_return_code": None,
            "termination_signal": None,
            "effective_inference_status": "not_started",
            "forced_process_cleanup": False,
            "forced_cleanup_reason": None,
            "combined_output_valid": False,
            "combined_output_validation": {},
            "combined_label": None,
        }
        subprocess_monitor_kwargs = {
            "stage_log": stage_log,
            "status_path": status_path,
            "combined_dir": combined_dir,
            "input_image": image,
            "dataset_json": dataset_json,
            "requested_organs": requested_organs,
            "combined_output_stable_sec": float(args.combined_output_stable_sec),
            "post_export_shutdown_grace_sec": args.post_export_shutdown_grace_sec,
            "process_termination_grace_sec": float(args.process_termination_grace_sec),
            "poll_interval_sec": float(args.post_export_poll_interval_sec),
            "prediction_timeout_sec": args.prediction_timeout_sec,
        }
        if model_folder:
            input_csv = output / "epai_input.csv"
            output_csv = output / "epai_output.csv"
            input_csv.write_text(f"Original ID,BDMAP ID\n{case_id},{case_id}\n", encoding="utf-8")
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=3)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=3)
            cmd = [
                str(predict_from_modelfolder_executable), "-i", str(input_dir), "-o", str(combined_dir),
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", str(input_csv), "--output_csv", str(output_csv),
                "--continue_prediction", "-npp", str(npp), "-nps", str(nps), "-num_parts", "1", "-part_id", "0",
                "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
            if args.save_probabilities:
                cmd.append("--save_probabilities")
            if args.device and args.device.lower() == "cpu":
                cmd.extend(["-device", "cpu"])
            metadata_doc.update({
                "command": [str(part) for part in cmd],
                "command_text": shlex.join([str(part) for part in cmd]),
                "predict_executable": str(predict_from_modelfolder_executable),
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command=metadata_doc["command_text"], child_backend="subprocess")
            proc_info = _run_streaming_subprocess(
                cmd,
                env=env,
                cwd=workdir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                metadata=metadata_doc,
                **subprocess_monitor_kwargs,
            )
            proc_return_code = int(proc_info["return_code"])
            proc_end_time = str(proc_info["end_time"])
        elif args.use_python_api:
            # Use Python API directly — needed when the model was trained with ePAI's nnunetv2
            # but the ePAI predict entrypoint requires a CSV (incompatible with standard usage).
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=2)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=2)
            metadata_doc.update({
                "command": ["python_api"],
                "command_text": "python_api",
                "predict_executable": "python_api",
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _write_log_header(stdout_log, "stdout", metadata_doc)
            _write_log_header(stderr_log, "stderr", metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command="python_api", child_backend="python_api")
            with stdout_log.open("a", encoding="utf-8", buffering=1) as stdout_handle, stderr_log.open("a", encoding="utf-8", buffering=1) as stderr_handle:
                with redirect_stdout(stdout_handle), redirect_stderr(stderr_handle):
                    try:
                        import sys as _sys
                        if workdir:
                            _sys.path.insert(0, str(workdir))
                        else:
                            _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ePAI-main" / "train"))
                        import torch as _torch
                        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
                        from batchgenerators.utilities.file_and_folder_operations import join as _join

                        _device = _torch.device("cpu" if (args.device and args.device.lower() == "cpu") else "cuda")
                        predictor = nnUNetPredictor(
                            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                            perform_everything_on_device=True, device=_device,
                            verbose=False, verbose_preprocessing=False, allow_tqdm=True
                        )
                        # model folder = nnunet_results / DatasetXXX_name / trainer__plans__config
                        _trainer_folder = Path(str(nnunet_results)) / f"Dataset{args.dataset_id:03d}_{args.dataset_id}" if False else None
                        # Find the correct trainer folder under nnunet_results
                        import glob as _glob
                        _pattern = str(nnunet_results / f"Dataset{int(args.dataset_id):03d}_*" / f"{args.trainer}__{args.plans}__{args.configuration}")
                        _matches = _glob.glob(_pattern)
                        if not _matches:
                            raise FileNotFoundError(f"No trainer folder found matching: {_pattern}")
                        _trainer_folder = Path(_matches[0])
                        predictor.initialize_from_trained_model_folder(
                            str(_trainer_folder),
                            use_folds=(args.folds,),
                            checkpoint_name=args.checkpoint_name
                        )
                        _input_files = sorted([str(f) for f in input_dir.glob("*.nii.gz")])
                        _list_of_lists = [[f] for f in _input_files]
                        _output_files = [str(combined_dir / Path(f).name.replace("_0000.nii.gz", ".nii.gz")) for f in _input_files]
                        combined_dir.mkdir(parents=True, exist_ok=True)
                        _data_iter = predictor._internal_get_data_iterator_from_lists_of_filenames(
                            _list_of_lists, None, _output_files, num_processes=npp
                        )
                        predictor.predict_from_data_iterator(_data_iter, save_probabilities=False, num_processes_segmentation_export=nps)
                        print("Python API inference completed.", flush=True)
                        proc_return_code = 0
                    except Exception:
                        import traceback as _tb
                        print(_tb.format_exc(), file=sys.stderr, flush=True)
                        proc_return_code = 1
            proc_end_time = _utc_now()
            proc_info = {
                **proc_info,
                "return_code": proc_return_code,
                "raw_child_return_code": proc_return_code,
                "termination_signal": None,
                "end_time": proc_end_time,
                "effective_inference_status": "completed" if proc_return_code == 0 else "failed",
            }
            _append_log_footer(stdout_log, end_time=proc_end_time, return_code=proc_return_code)
            _append_log_footer(stderr_log, end_time=proc_end_time, return_code=proc_return_code)
        else:
            cmd = [
                str(predict_executable), "-d", str(args.dataset_id), "-i", str(input_dir), "-o", str(combined_dir),
                "-tr", args.trainer, "-c", args.configuration, "-f", str(args.folds), "-p", args.plans, "-chk", args.checkpoint_name, "--continue_prediction",
            ]
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=None)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=None)
            if npp is not None:
                cmd.extend(["-npp", str(npp)])
            if nps is not None:
                cmd.extend(["-nps", str(nps)])
            if args.save_probabilities:
                cmd.append("--save_probabilities")
            metadata_doc.update({
                "command": [str(part) for part in cmd],
                "command_text": shlex.join([str(part) for part in cmd]),
                "predict_executable": str(predict_executable),
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command=metadata_doc["command_text"], child_backend="subprocess")
            proc_info = _run_streaming_subprocess(
                cmd,
                env=env,
                cwd=workdir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                metadata=metadata_doc,
                **subprocess_monitor_kwargs,
            )
            proc_return_code = int(proc_info["return_code"])
            proc_end_time = str(proc_info["end_time"])
        _stop_gpu_monitor(gpu_stop, gpu_thread)

        raw_child_return_code = proc_info.get("raw_child_return_code")
        if raw_child_return_code is None:
            raw_child_return_code = proc_return_code
        prediction_status_fields = {
            "raw_child_return_code": raw_child_return_code,
            "termination_signal": proc_info.get("termination_signal"),
            "effective_inference_status": proc_info.get("effective_inference_status"),
            "forced_process_cleanup": bool(proc_info.get("forced_process_cleanup")),
            "forced_cleanup_reason": proc_info.get("forced_cleanup_reason"),
            "combined_output_valid": bool(proc_info.get("combined_output_valid")),
            "combined_output_validation": proc_info.get("combined_output_validation") or {},
        }
        _append_stage(
            stage_log,
            status_path,
            "predictor_completed",
            return_code=proc_return_code,
            end_time=proc_end_time,
            **prediction_status_fields,
        )
        metadata_doc.update({
            "status": "predictor_completed",
            "end_time": proc_end_time,
            "return_code": proc_return_code,
            **prediction_status_fields,
        })
        _write_json(metadata_path, metadata_doc)
        if proc_return_code != 0:
            failure = {
                "status": "failed",
                "return_code": proc_return_code,
                **prediction_status_fields,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stderr_tail": _tail_file(stderr_log),
                "stage_log": str(stage_log),
                "run_metadata": str(metadata_path),
                "last_stage": "predictor_completed",
                "retained_workdir": retained_workdir,
            }
            _write_json(per_model_dir / "inference_summary.json", failure)
            if per_model_dir != output:
                _write_json(output / "inference_summary.json", failure)
            print(json.dumps(failure, indent=2))
            return proc_return_code

        combined_validation_path: Path | None = Path(str(proc_info["combined_label"])) if proc_info.get("combined_label") else None
        combined_output_validation = prediction_status_fields["combined_output_validation"]
        if not _combined_output_split_safe(combined_output_validation):
            combined_validation_path, combined_output_validation = _validate_first_combined_output(
                combined_dir,
                image,
                dataset_json,
                requested_organs,
            )
            prediction_status_fields["combined_output_validation"] = combined_output_validation
            prediction_status_fields["combined_output_valid"] = bool(combined_output_validation.get("valid"))
            if combined_validation_path is not None:
                proc_info["combined_label"] = str(combined_validation_path)
                try:
                    stat = combined_validation_path.stat()
                    _append_stage(
                        stage_log,
                        status_path,
                        "combined_output_detected",
                        combined_label=str(combined_validation_path),
                        size_bytes=int(stat.st_size),
                        mtime_ns=int(stat.st_mtime_ns),
                    )
                except Exception:
                    _append_stage(stage_log, status_path, "combined_output_detected", combined_label=str(combined_validation_path))
                if combined_output_validation.get("valid"):
                    _append_stage(
                        stage_log,
                        status_path,
                        "combined_output_validated",
                        combined_label=str(combined_validation_path),
                        combined_output_validation=combined_output_validation,
                    )
            metadata_doc.update(prediction_status_fields)
            _write_json(metadata_path, metadata_doc)
        if not _combined_output_split_safe(combined_output_validation):
            failure = {
                "status": "failed",
                "return_code": 2,
                "reason": "combined_output_invalid",
                "combined_dir": str(combined_dir),
                **prediction_status_fields,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stage_log": str(stage_log),
                "run_metadata": str(metadata_path),
                "last_stage": "predictor_completed",
                "retained_workdir": retained_workdir,
            }
            _write_json(per_model_dir / "inference_summary.json", failure)
            if per_model_dir != output:
                _write_json(output / "inference_summary.json", failure)
            print(json.dumps(failure, indent=2))
            return 2

        _append_stage(stage_log, status_path, "split_started", combined_output_dir=str(combined_dir), per_model_output_dir=str(per_model_dir))
        candidates = sorted(combined_dir.glob("*.nii.gz"))
        if not candidates:
            failure = {
                "status": "failed",
                "return_code": 2,
                "reason": "nnUNet produced no combined label map",
                "combined_dir": str(combined_dir),
                **prediction_status_fields,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stage_log": str(stage_log),
                "run_metadata": str(metadata_path),
                "last_stage": "split_started",
                "retained_workdir": retained_workdir,
            }
            _write_json(per_model_dir / "inference_summary.json", failure)
            if per_model_dir != output:
                _write_json(output / "inference_summary.json", failure)
            print(json.dumps(failure, indent=2))
            return 2

        # Python API outputs a combined label map named after the case (e.g. PanTS_00000002.nii.gz)
        # Standard subprocess outputs a combined label map too. Either way, take the first .nii.gz.
        combined = combined_validation_path if combined_validation_path in candidates else candidates[0]
        combined_out = per_model_dir / "combined_labels.nii.gz"
        shutil.copy2(combined, combined_out)
        # Also keep a copy at --output for backward compatibility when per-model-dir
        # differs from output.
        if combined_out != output / "combined_labels.nii.gz":
            shutil.copy2(combined, output / "combined_labels.nii.gz")
        # Dump local_labels.json (name -> int id), region-aware via the actual array.
        try:
            import numpy as _np
            import nibabel as _nib
            _label_arr = _np.asanyarray(_nib.load(str(combined)).dataobj)
        except Exception:
            _label_arr = None
        local_labels = dump_local_labels(dataset_json, per_model_dir / "local_labels.json", label_arr=_label_arr)
        split = split_labelmap(combined, dataset_json, per_model_seg_dir, requested_organs=requested_organs)
        if per_model_seg_dir != seg_dir:
            seg_dir.mkdir(parents=True, exist_ok=True)
            for mask_path in per_model_seg_dir.glob("*.nii.gz"):
                shutil.copy2(mask_path, seg_dir / mask_path.name)
        _append_stage(stage_log, status_path, "split_completed", combined_label=str(combined_out), num_written=split.get("num_written"))
        expected_mask_organs = requested_organs or [str(item.get("organ")) for item in split.get("written_masks", []) if item.get("organ")]
        target_mask_validation = validate_target_masks(per_model_seg_dir, image, expected_mask_organs)
        validation_status_fields = {
            **prediction_status_fields,
            "target_mask_validation": target_mask_validation,
        }
        _append_stage(
            stage_log,
            status_path,
            "validation_completed",
            combined_label=str(combined_out),
            segmentation_output=str(per_model_seg_dir),
            **validation_status_fields,
        )
        summary = {
            "status": "success", "case_id": case_id,
            "combined_label": str(combined_out),
            "local_labels": str(per_model_dir / "local_labels.json"),
            "num_local_labels": len(local_labels),
            "segmentation_output": str(per_model_seg_dir),
            "legacy_segmentation_output": str(seg_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stage_log": str(stage_log),
            "run_metadata": str(metadata_path),
            "gpu_monitor_csv": str(gpu_monitor_path) if gpu_monitor_path.exists() else None,
            "workdir": str(tmp),
            "retained_workdir": retained_workdir,
            "last_stage": "validation_completed",
            "return_code": 0,
            **validation_status_fields,
            **split,
        }
        (per_model_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if per_model_dir != output:
            (output / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metadata_doc.update({"status": "success", "last_stage": "validation_completed", "return_code": 0, **validation_status_fields})
        _write_json(metadata_path, metadata_doc)
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        try:
            _stop_gpu_monitor(locals().get("gpu_stop"), locals().get("gpu_thread"))
        except Exception:
            pass
        if tmp_holder is not None:
            tmp_holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
