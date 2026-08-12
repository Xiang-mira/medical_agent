#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "agent-harness"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry  # noqa: E402
from cli_anything.medai.core.continual_learning import TRAINING_CONTRACT_VERSION, canonicalize_training_record  # noqa: E402
from cli_anything.medai.core.multimodel_loop import _select_candidate  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_rows, utc_now, write_csv, write_json  # noqa: E402


FULL373_GROUP = "full373"
FULL373_ROOT_NAME = "full_373_multiteacher_round1"
TERMINAL_CANDIDATE_STATES = {
    "SUCCESS",
    "ABSENT",
    "OUT_OF_FOV",
    "COMPLETED_NO_NONZERO",
    "FAILED_FINAL",
}
NON_TERMINAL_CANDIDATE_STATES = {
    "READY",
    "CLAIMED",
    "QUEUED",
    "RUNNING",
    "RETRY_PENDING",
    "BACKPRESSURED",
}
VALID_TERMINAL_TARGET_STATES = {
    "SELECTED",
    "ABSENT",
    "VALID_SINGLE_TEACHER_ACCEPTED",
    "SELECTED_PSEUDO_LABEL",
    "ABSENT_NEGATIVE",
    "NEGATIVE_ABSENT",
    "WITHHELD_UNCERTAIN",
    "UNRESOLVED_REVIEW",
    "PARTIAL_FOV",
}
NON_TERMINAL_TARGET_STATES = {
    "READY",
    "CLAIMED",
    "RUNNING",
    "WAITING_FOR_CANDIDATES",
    "CANDIDATES_READY",
    "WAITING_FOR_LABELCRITIC",
    "LABELCRITIC_RUNNING",
    "RETRY_PENDING",
}


def _norm(text: str) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_targets(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    targets = [str(item).strip() for item in doc.get("target_organs", []) if str(item).strip()]
    if not targets:
        raise ValueError(f"No target_organs found in {path}")
    return targets


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _queue_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "queues"
    return {
        "root": root,
        "candidate_states": root / "candidate_states",
        "candidate_claims": root / "candidate_claims",
        "case_target_states": root / "case_target_states",
        "case_target_claims": root / "case_target_claims",
        "selection_results": root / "selection_results",
        "events": root / "events.jsonl",
        "telemetry": root / "telemetry.json",
    }


def _state_path(root: Path, case_id: str, target: str, teacher: str | None = None, *, kind: str) -> Path:
    safe_case = re_safe(case_id)
    safe_target = re_safe(target)
    if kind == "candidate":
        return _queue_paths(root)["candidate_states"] / safe_case / safe_target / f"{re_safe(teacher or '')}.json"
    if kind == "target":
        return _queue_paths(root)["case_target_states"] / safe_case / f"{safe_target}.json"
    if kind == "selection":
        return _queue_paths(root)["selection_results"] / safe_case / f"{safe_target}.json"
    raise ValueError(f"unknown state kind: {kind}")


def re_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_") or "na"


def _append_event(output_root: Path, payload: dict[str, Any]) -> None:
    path = _queue_paths(output_root)["events"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), **payload}, ensure_ascii=False) + "\n")


def _claim_path(output_root: Path, *, claim_kind: str, claim_key: str) -> Path:
    if claim_kind == "candidate":
        return _queue_paths(output_root)["candidate_claims"] / f"{re_safe(claim_key)}.json"
    if claim_kind == "case_target":
        return _queue_paths(output_root)["case_target_claims"] / f"{re_safe(claim_key)}.json"
    raise ValueError(f"unknown claim kind: {claim_kind}")


def claim_work(output_root: Path, *, claim_kind: str, claim_key: str, worker_id: str, lease_sec: int = 7200) -> dict[str, Any]:
    path = _claim_path(output_root, claim_kind=claim_kind, claim_key=claim_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if path.exists():
        existing = _read_json(path, {})
        try:
            created = float(existing.get("created_time") or 0.0)
        except Exception:
            created = 0.0
        if created and now - created < lease_sec:
            return {"status": "BUSY", "claim_path": str(path), "claim": existing}
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    payload = {
        "status": "CLAIMED",
        "claim_kind": claim_kind,
        "claim_key": claim_key,
        "worker_id": worker_id,
        "claim_id": f"{re_safe(worker_id)}_{_sha(str(now), 10)}",
        "created_at": utc_now(),
        "created_time": now,
        "lease_sec": int(lease_sec),
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {"status": "BUSY", "claim_path": str(path), "claim": _read_json(path, {})}
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {"status": "CLAIMED", "claim_path": str(path), "claim": payload}


def release_claim(output_root: Path, *, claim_kind: str, claim_key: str) -> None:
    try:
        _claim_path(output_root, claim_kind=claim_kind, claim_key=claim_key).unlink()
    except FileNotFoundError:
        pass


def _cache_rows_from_formal_status(root: Path) -> list[dict[str, Any]]:
    path = root / "task2_formal_case_target_status.json"
    doc = _read_json(path, {})
    rows = doc.get("rows") if isinstance(doc, dict) else None
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        mask = str(row.get("mask_path") or row.get("final_mask") or "")
        if not mask or not Path(mask).is_file():
            continue
        out.append(
            {
                "source": "task2_formal_case_target_status",
                "source_manifest": str(path),
                "case_id": str(row.get("case_id") or ""),
                "target": _norm(str(row.get("target_name") or row.get("organ") or row.get("target") or "")),
                "teacher": str(row.get("selected_model") or row.get("source_model") or row.get("model") or row.get("teacher") or ""),
                "mask_path": mask,
                "validation_status": str(row.get("final_status") or row.get("validation_status") or ""),
                "provenance": row,
            }
        )
    return out


def _cache_rows_from_selection_metadata(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "annotation_versions").glob("*/selection_metadata.json")):
        doc = _read_json(path, {})
        case_id = str(doc.get("case_id") or path.parent.name)
        for row in doc.get("selected_organs") or []:
            if not isinstance(row, dict):
                continue
            mask = str(row.get("final_mask") or row.get("mask_path") or "")
            if not mask or not Path(mask).is_file():
                continue
            out.append(
                {
                    "source": "run_loop_selection_metadata",
                    "source_manifest": str(path),
                    "case_id": case_id,
                    "target": _norm(str(row.get("organ") or row.get("target") or "")),
                    "teacher": str(row.get("selected_model") or row.get("source_model") or ""),
                    "mask_path": mask,
                    "validation_status": str(row.get("selection_status") or ""),
                    "provenance": row,
                }
            )
    return out


def audit_candidate_cache(
    *,
    cache_roots: list[Path],
    case_ids: set[str],
    routes: dict[str, list[str]],
) -> dict[str, Any]:
    usable = []
    rejected = []
    route_pairs = {(case_id, target, teacher) for case_id in case_ids for target, teachers in routes.items() for teacher in teachers}
    for root in cache_roots:
        if not root or not root.exists():
            continue
        rows = [*_cache_rows_from_formal_status(root), *_cache_rows_from_selection_metadata(root)]
        for row in rows:
            key = (str(row["case_id"]), str(row["target"]), str(row["teacher"]))
            if key in route_pairs:
                usable.append({**row, "cache_status": "REUSED_VALID_CANDIDATE"})
            else:
                rejected.append({**row, "cache_status": "REJECTED_CACHE_MISMATCH"})
    return {
        "status": "success",
        "cache_roots": [str(path) for path in cache_roots],
        "reusable_candidate_count": len(usable),
        "rejected_candidate_count": len(rejected),
        "reusable_candidates": usable[:1000],
        "rejected_candidates_sample": rejected[:100],
    }


def build_full_round1_scope(
    *,
    case_manifest: Path,
    registry_path: Path,
    target_config: Path,
    output_root: Path,
    cache_roots: list[Path] | None = None,
    expected_case_count: int = 103,
    file_stem: str = "full_round1_scope",
) -> dict[str, Any]:
    cases = read_csv_rows(case_manifest)
    case_ids = [str(row.get("case_id") or row.get("id") or "").strip() for row in cases]
    targets = _load_targets(target_config)
    registry = load_registry(registry_path)
    raw_routes = candidate_models_for_organs(registry, targets)
    routes = {_norm(target): list(raw_routes.get(_norm(target), [])) for target in targets}
    disabled_routes = []
    not_ready_routes = []
    for target, teachers in routes.items():
        retained = []
        for teacher in teachers:
            entry = (registry.get("models") or {}).get(teacher) or {}
            if entry.get("enabled") is False:
                disabled_routes.append({"target": target, "teacher": teacher, "reason": "registry_enabled_false"})
                continue
            status = str(entry.get("status") or "")
            if status in {"disabled", "not_ready", "template"}:
                not_ready_routes.append({"target": target, "teacher": teacher, "status": status})
                continue
            retained.append(teacher)
        routes[target] = retained
    unroutable = [target for target in [_norm(item) for item in targets] if not routes.get(target)]
    task_rows = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("id") or "").strip()
        for target, teachers in routes.items():
            for teacher in teachers:
                entry = (registry.get("models") or {}).get(teacher) or {}
                task_rows.append(
                    {
                        "candidate_id": f"cand_{_sha(case_id + '|' + target + '|' + teacher)}",
                        "case_id": case_id,
                        "target": target,
                        "teacher": teacher,
                        "teacher_family": str(entry.get("evidence_family") or entry.get("architecture_lineage") or teacher),
                        "checkpoint": str(entry.get("checkpoint_path") or entry.get("source_code_path") or ""),
                    }
                )
    cache = audit_candidate_cache(cache_roots=cache_roots or [], case_ids=set(case_ids), routes=routes)
    pair_count = sum(len(value) for value in routes.values())
    scope = {
        "stage": "full_373_multiteacher_round1_scope",
        "status": "READY" if len(cases) == expected_case_count and len(set(case_ids)) == expected_case_count and len(targets) == 373 and not unroutable else "BLOCKED",
        "authoritative_routing_source": "cli_anything.medai.core.model_registry.candidate_models_for_organs; precedence: organ_router.route_organs(configs/organ_routing_from_xlsx.json + configs/routing_token_to_model.json), fallback registry.organ_to_models, fallback registry covered_organs/aliases",
        "case_manifest": str(case_manifest),
        "target_config": str(target_config),
        "registry_path": str(registry_path),
        "case_count": len(cases),
        "unique_case_count": len(set(case_ids)),
        "canonical_target_count": len(targets),
        "enabled_teacher_count": len({teacher for teachers in routes.values() for teacher in teachers}),
        "target_teacher_pairs": pair_count,
        "total_logical_candidate_tasks": len(cases) * pair_count,
        "expected_logical_candidate_tasks_for_103_cases": expected_case_count * pair_count,
        "single_teacher_target_count": sum(1 for teachers in routes.values() if len(teachers) == 1),
        "multi_teacher_target_count": sum(1 for teachers in routes.values() if len(teachers) > 1),
        "unroutable_target_count": len(unroutable),
        "unroutable_targets": unroutable,
        "disabled_teacher_routes": disabled_routes,
        "not_ready_teacher_routes": not_ready_routes,
        "routes": routes,
        "task_rows": task_rows,
        "cached_reusable_candidate_count": cache["reusable_candidate_count"],
        "new_inference_candidate_count": max(0, len(cases) * pair_count - int(cache["reusable_candidate_count"])),
        "cache_audit": cache,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / f"{file_stem}.json", scope)
    write_csv(
        output_root / f"{file_stem}.csv",
        task_rows,
        ["candidate_id", "case_id", "target", "teacher", "teacher_family", "checkpoint"],
    )
    return scope


def _write_case_manifest(path: Path, row: dict[str, str]) -> None:
    fieldnames = list(row.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [row], fieldnames)


def _write_array_sbatch(path: Path, *, python: Path, output_root: Path, task_manifest: Path, state_root: Path | None = None) -> None:
    state_arg = f"  --state-root {shlex.quote(str(state_root))} \\\n" if state_root else ""
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=task2_full373_multiteacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output={output_root / 'slurm' / 'full373_%A_%a.out'}
#SBATCH --error={output_root / 'slurm' / 'full373_%A_%a.err'}

set -euo pipefail
unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS
export RUNTIME_NO_GIT=1
export SKIP_GIT_SYNC=1
export GIT_TERMINAL_PROMPT=0
cd {shlex.quote(str(REPO_ROOT))}
{shlex.quote(str(python))} tools/dataset_delivery/task2_full373_round1_launcher.py \\
  --execute-task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --task-manifest {shlex.quote(str(task_manifest))} \\
{state_arg}  --worker-id "${{SLURM_JOB_ID:-local}}_${{SLURM_ARRAY_TASK_ID:-0}}" \\
  --output-root {shlex.quote(str(output_root))}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def build_submission_manifest(
    *,
    case_manifest: Path,
    output_root: Path,
    registry_path: Path,
    target_config: Path,
    python: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    unest_python_executable: Path,
    state_root: Path | None = None,
    cache_roots: list[Path] | None = None,
    expected_case_count: int = 103,
) -> dict[str, Any]:
    input_rows = read_csv_rows(case_manifest)
    scope = build_full_round1_scope(
        case_manifest=case_manifest,
        registry_path=registry_path,
        target_config=target_config,
        output_root=output_root,
        cache_roots=cache_roots or [],
        expected_case_count=len(input_rows) if input_rows else expected_case_count,
        file_stem="full_round1_submission_scope",
    )
    if scope["status"] != "READY":
        raise RuntimeError(f"Full 373 Round1 scope blocked: {scope.get('unroutable_targets')}")
    case_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(input_rows):
        case_id = str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()
        case_by_id[case_id] = row
    rows = []
    for index, candidate in enumerate(scope.get("task_rows") or []):
        case_id = str(candidate.get("case_id") or "")
        row = case_by_id.get(case_id) or {}
        rows.append(
            {
                "task_index": index,
                "case_id": case_id,
                "target": str(candidate.get("target") or ""),
                "teacher": str(candidate.get("teacher") or ""),
                "candidate_id": str(candidate.get("candidate_id") or f"cand_{_sha(case_id + '|' + str(candidate.get('target')) + '|' + str(candidate.get('teacher')))}"),
                "ct_path": row.get("ct_path") or row.get("image_path") or "",
                "annotation_folder": row.get("annotation_folder") or row.get("reference_mask_dir") or "",
                "registry_path": str(registry_path),
                "target_config": str(target_config),
                "checkpoint_root": str(checkpoint_root),
                "nnunet_predict_executable": str(nnunet_predict_executable),
                "unest_python_executable": str(unest_python_executable),
                "python": str(python),
            }
        )
    slurm_root = output_root / "slurm"
    batch_key = _sha("|".join(row["candidate_id"] for row in rows), 12)
    task_manifest = slurm_root / f"full373_task_manifest_{batch_key}.csv"
    write_csv(task_manifest, rows, list(rows[0].keys()) if rows else ["task_index", "case_id", "target", "teacher", "candidate_id"])
    sbatch = slurm_root / "full373_multiteacher_array.sbatch"
    _write_array_sbatch(sbatch, python=python, output_root=output_root, task_manifest=task_manifest, state_root=state_root)
    summary = {
        "status": "READY",
        "stage": "full_373_multiteacher_round1_submission_manifest",
        "created_at": utc_now(),
        "scope": str(output_root / "full_round1_submission_scope.json"),
        "scope_status": scope,
        "task_count": len(rows),
        "scientific_task_unit": "case_id x canonical_target x eligible_teacher",
        "scheduler_array_unit": "case_id x canonical_target x eligible_teacher candidate shard",
        "groups": {
            FULL373_GROUP: {
                "task_count": len(rows),
                "task_manifest": str(task_manifest),
                "sbatch_file": str(sbatch),
            }
        },
    }
    write_json(output_root / "formal_task2_submission_manifest.json", summary)
    return summary


def _selection_metadata_path(case_output: Path, case_id: str) -> Path:
    return case_output / "annotation_versions" / case_id / "selection_metadata.json"


def _candidate_from_single_teacher_run(case_output: Path, *, case_id: str, target: str, teacher: str) -> dict[str, Any]:
    meta = _read_json(_selection_metadata_path(case_output, case_id), {})
    selection_rows = list(meta.get("selection_rows") or [])
    selected_rows = list(meta.get("selected_organs") or [])
    target_rows = [row for row in selection_rows if _norm(str(row.get("organ") or row.get("target") or "")) == target]
    selected_target_rows = [row for row in selected_rows if _norm(str(row.get("organ") or row.get("target") or "")) == target]
    candidate: dict[str, Any] = {}
    for row in target_rows:
        for pred in row.get("candidate_predictions") or []:
            if str(pred.get("model") or "") == teacher:
                candidate = dict(pred)
                break
        if candidate:
            break
    selected = selected_target_rows[0] if selected_target_rows else {}
    prediction = (
        candidate.get("candidate_cleaned_prediction")
        or candidate.get("candidate_raw_prediction")
        or candidate.get("prediction")
        or selected.get("pre_shapekit_mask")
        or selected.get("final_mask")
        or selected.get("mask_path")
    )
    path = Path(str(prediction or ""))
    exists = bool(prediction and path.is_file())
    qc_status = str(candidate.get("candidate_qc_status") or selected.get("selected_candidate_qc_status") or selected.get("candidate_qc_status") or "")
    target_type = str(selected.get("target_type") or "")
    if exists and qc_status != "fail":
        state = "SUCCESS"
    elif target_type in {"absent_negative", "negative_absent"}:
        state = "ABSENT"
    elif target_type in {"out_of_fov", "partial_fov"}:
        state = "OUT_OF_FOV"
    elif exists:
        state = "FAILED_FINAL"
    else:
        state = "COMPLETED_NO_NONZERO"
    return {
        "status": state,
        "candidate_exists": exists,
        "case_id": case_id,
        "target": target,
        "teacher": teacher,
        "model": teacher,
        "prediction": str(path) if exists else "",
        "candidate_cleaned_prediction": str(candidate.get("candidate_cleaned_prediction") or ""),
        "candidate_raw_prediction": str(candidate.get("candidate_raw_prediction") or ""),
        "candidate_id": candidate.get("candidate_id") or f"cand_{_sha(case_id + '|' + target + '|' + teacher)}",
        "candidate_qc_status": qc_status,
        "candidate_qc_score": candidate.get("candidate_qc_score") or selected.get("selected_candidate_qc_score"),
        "candidate_qc_flags": candidate.get("candidate_qc_flags") or selected.get("selected_candidate_qc_flags") or [],
        "candidate_qc": candidate.get("candidate_qc") or selected.get("selected_candidate_qc_checks") or {},
        "candidate_shapekit_status": candidate.get("candidate_shapekit_status") or selected.get("shapekit_status"),
        "candidate_shapekit_reason": candidate.get("candidate_shapekit_reason") or selected.get("shapekit_reason"),
        "eligible_for_labelcritic": bool(candidate.get("eligible_for_labelcritic", exists)),
        "source_run_loop": str(case_output),
        "selection_metadata": str(_selection_metadata_path(case_output, case_id)),
        "published_at": utc_now(),
    }


def load_candidate_state(output_root: Path, *, case_id: str, target: str, teacher: str) -> dict[str, Any]:
    return _read_json(_state_path(output_root, case_id, target, teacher, kind="candidate"), {})


def publish_candidate_state(output_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    case_id = str(state["case_id"])
    target = _norm(str(state["target"]))
    teacher = str(state["teacher"])
    path = _state_path(output_root, case_id, target, teacher, kind="candidate")
    atomic_write_json(path, state)
    _append_event(output_root, {"event": "candidate_state", "case_id": case_id, "target": target, "teacher": teacher, "status": state.get("status")})
    recompute_case_target_readiness(output_root, case_id=case_id, target=target)
    return state


def recompute_case_target_readiness(output_root: Path, *, case_id: str, target: str) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_submission_scope.json", {}) or _read_json(output_root / "full_round1_scope.json", {})
    routes = scope.get("routes") or {}
    teachers = [str(item) for item in routes.get(_norm(target), [])]
    candidate_states = [load_candidate_state(output_root, case_id=case_id, target=_norm(target), teacher=teacher) for teacher in teachers]
    observed = [state for state in candidate_states if state]
    statuses = [str(state.get("status") or "READY") for state in candidate_states]
    all_terminal = bool(teachers) and len(observed) == len(teachers) and all(status in TERMINAL_CANDIDATE_STATES for status in statuses)
    if all_terminal:
        valid = [
            state for state in observed
            if state.get("status") == "SUCCESS"
            and state.get("candidate_exists")
            and state.get("prediction")
            and Path(str(state.get("prediction"))).is_file()
        ]
        status = "CANDIDATES_READY" if valid else "ABSENT"
    else:
        status = "WAITING_FOR_CANDIDATES"
    target_state = {
        "status": status,
        "case_id": case_id,
        "target": _norm(target),
        "eligible_teachers": teachers,
        "candidate_statuses": {teacher: str(state.get("status") or "READY") for teacher, state in zip(teachers, candidate_states)},
        "candidate_state_paths": [
            str(_state_path(output_root, case_id, _norm(target), teacher, kind="candidate")) for teacher in teachers
        ],
        "candidate_count": len(valid) if all_terminal else 0,
        "updated_at": utc_now(),
    }
    existing = _read_json(_state_path(output_root, case_id, _norm(target), kind="target"), {})
    if str(existing.get("status") or "") in VALID_TERMINAL_TARGET_STATES:
        return existing
    atomic_write_json(_state_path(output_root, case_id, _norm(target), kind="target"), target_state)
    _append_event(output_root, {"event": "case_target_readiness", "case_id": case_id, "target": _norm(target), "status": status})
    return target_state


def execute_task_index(task_index: int, task_manifest: Path, output_root: Path, *, worker_id: str = "", state_root: Path | None = None) -> dict[str, Any]:
    rows = read_csv_rows(task_manifest)
    row = next((item for item in rows if int(item.get("task_index") or -1) == int(task_index)), None)
    if row is None:
        raise IndexError(f"task index not found: {task_index}")
    case_id = str(row["case_id"])
    target = _norm(str(row.get("target") or ""))
    teacher = str(row.get("teacher") or "")
    candidate_id = str(row.get("candidate_id") or f"cand_{_sha(case_id + '|' + target + '|' + teacher)}")
    existing = load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher)
    if existing.get("status") in TERMINAL_CANDIDATE_STATES:
        return {"status": "REUSED_TERMINAL_CANDIDATE", "candidate": existing}
    claim = claim_work(output_root, claim_kind="candidate", claim_key=candidate_id, worker_id=worker_id or f"pid_{os.getpid()}")
    if claim["status"] != "CLAIMED":
        return {"status": "CLAIM_BUSY", "candidate_id": candidate_id, "claim": claim}
    case_csv = output_root / "case_manifests" / f"{case_id}.csv"
    try:
        _write_case_manifest(
            case_csv,
            {
                "case_id": case_id,
                "ct_path": str(row.get("ct_path") or ""),
                "annotation_folder": str(row.get("annotation_folder") or ""),
            },
        )
        case_output = output_root / "candidate_runs" / case_id / target / teacher
        command = [
            str(row.get("python") or sys.executable),
            "run_medai_cli.py",
            "--json",
            "run-loop",
            "--case-list", str(case_csv),
            "--models", teacher,
            "--organs", target,
            "--target-config", str(row.get("target_config") or REPO_ROOT / "configs/student_3d_prompt_target_organs.json"),
            "--registry", str(row.get("registry_path") or REPO_ROOT / "configs/model_registry.yaml"),
            "--output", str(case_output),
            "--checkpoint-root", str(row.get("checkpoint_root") or ""),
            "--nnunet-predict-executable", str(row.get("nnunet_predict_executable") or ""),
            "--unest-python-executable", str(row.get("unest_python_executable") or ""),
            "--enable-shapekit",
            "--no-enable-critic",
            "--teacher-inference-mode", os.getenv("MEDAI_TEACHER_INFERENCE_MODE", "hierarchical_roi"),
            "--roi-margin-mm", os.getenv("MEDAI_ROI_MARGIN_MM", "20"),
            "--timeout-sec", os.getenv("MEDAI_INFER_TIMEOUT_SEC", "3600"),
            "--no-use-annotation-folder-reference",
            "--log-file", str(case_output / "run_loop.log"),
        ]
        env = os.environ.copy()
        env.update({"RUNTIME_NO_GIT": "1", "SKIP_GIT_SYNC": "1", "GIT_TERMINAL_PROMPT": "0"})
        if state_root:
            env["STATE_ROOT"] = str(state_root)
        for key in ("DISPLAY", "GITHUB_TOKEN", "GH_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"):
            env.pop(key, None)
        task_state_path = output_root / "task_states" / f"{candidate_id}.json"
        state = {"status": "RUNNING", "case_id": case_id, "target": target, "teacher": teacher, "candidate_id": candidate_id, "task_index": task_index, "command": command, "started_at": utc_now()}
        atomic_write_json(task_state_path, state)
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        candidate_state = _candidate_from_single_teacher_run(case_output, case_id=case_id, target=target, teacher=teacher)
        candidate_state["ct_path"] = str(row.get("ct_path") or "")
        if proc.returncode != 0 and candidate_state["status"] not in {"SUCCESS", "ABSENT", "OUT_OF_FOV"}:
            candidate_state["status"] = "FAILED_FINAL"
            candidate_state["failure_reason"] = proc.stderr[-1000:] or proc.stdout[-1000:] or "candidate_worker_failed"
        candidate_state.update({"return_code": int(proc.returncode), "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:], "finished_at": utc_now()})
        publish_candidate_state(output_root, candidate_state)
        final = {
            **state,
            "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
            "candidate_status": candidate_state["status"],
            "return_code": int(proc.returncode),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "finished_at": utc_now(),
            "case_output": str(case_output),
        }
        atomic_write_json(task_state_path, final)
        return final
    finally:
        release_claim(output_root, claim_kind="candidate", claim_key=candidate_id)


def _case_rows_from_scope(scope: dict[str, Any]) -> list[str]:
    rows = scope.get("task_rows") or []
    return sorted({str(row.get("case_id") or "") for row in rows if isinstance(row, dict) and row.get("case_id")})


def _target_rows_from_scope(scope: dict[str, Any]) -> list[str]:
    routes = scope.get("routes") or {}
    if routes:
        return sorted(str(target) for target in routes)
    return sorted({str(row.get("target") or "") for row in scope.get("task_rows") or [] if isinstance(row, dict) and row.get("target")})


def _load_target_config_prompts(path: Path) -> tuple[list[str], dict[str, str]]:
    doc = _read_json(path, {})
    targets = [str(item) for item in doc.get("target_organs") or []]
    prompts = {str(k): str(v) for k, v in (doc.get("organ_to_prompt") or {}).items()}
    return targets, prompts


def _candidate_for_selection(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(state.get("teacher") or state.get("model") or ""),
        "prediction": str(state.get("prediction") or ""),
        "candidate_id": str(state.get("candidate_id") or ""),
        "candidate_exists": bool(state.get("candidate_exists")),
        "eligible_for_labelcritic": bool(state.get("eligible_for_labelcritic", True)),
        "candidate_qc_status": state.get("candidate_qc_status") or "pass",
        "candidate_qc_score": state.get("candidate_qc_score"),
        "candidate_qc_flags": state.get("candidate_qc_flags") or [],
        "candidate_qc": state.get("candidate_qc") or {},
        "candidate_shapekit_status": state.get("candidate_shapekit_status"),
        "candidate_shapekit_reason": state.get("candidate_shapekit_reason"),
        "candidate_raw_prediction": state.get("candidate_raw_prediction"),
        "candidate_cleaned_prediction": state.get("candidate_cleaned_prediction"),
        "em_candidate_role": "teacher_candidate",
        "candidate_source_role": "teacher_pseudo_candidate",
    }


def _terminal_from_selection(selection: dict[str, Any], selected: dict[str, Any] | None, candidate_count: int) -> str:
    method = str(selection.get("selection_method") or "")
    status = str(selection.get("selection_status") or "")
    if selected and status == "selected" and method == "label_critic":
        return "SELECTED"
    if selected and method in {"single_teacher_provisional", "single_teacher_default"}:
        return "VALID_SINGLE_TEACHER_ACCEPTED"
    if selected and status in {"selected", "provisional"}:
        return "SELECTED"
    if candidate_count <= 0:
        return "ABSENT"
    return "UNRESOLVED_REVIEW"


def claim_next_case_target(output_root: Path, *, worker_id: str, labelcritic_ready: bool, lease_sec: int = 1800) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    ready_states: list[dict[str, Any]] = []
    for case_id in _case_rows_from_scope(scope):
        for target in _target_rows_from_scope(scope):
            state = recompute_case_target_readiness(output_root, case_id=case_id, target=target)
            if state.get("status") == "CANDIDATES_READY":
                ready_states.append(state)
    if not ready_states:
        return {"status": "NO_READY_CASE_TARGETS"}
    if not labelcritic_ready:
        for state in ready_states:
            state["status"] = "WAITING_FOR_LABELCRITIC"
            atomic_write_json(_state_path(output_root, state["case_id"], state["target"], kind="target"), state)
        return {"status": "WAITING_FOR_LABELCRITIC", "queue_depth": len(ready_states)}
    ready_states.sort(key=lambda row: (str(row.get("case_id") or ""), str(row.get("target") or "")))
    for state in ready_states:
        claim_key = f"{state['case_id']}|{state['target']}"
        claim = claim_work(output_root, claim_kind="case_target", claim_key=claim_key, worker_id=worker_id, lease_sec=lease_sec)
        if claim["status"] == "CLAIMED":
            state["status"] = "LABELCRITIC_RUNNING"
            state["claim"] = claim["claim"]
            atomic_write_json(_state_path(output_root, state["case_id"], state["target"], kind="target"), state)
            return {"status": "CLAIMED", "case_target": state, "claim": claim}
    return {"status": "NO_CLAIMABLE_CASE_TARGETS", "queue_depth": len(ready_states)}


def select_case_target(
    output_root: Path,
    *,
    case_id: str,
    target: str,
    critic_base_url: str,
    critic_port: int,
    worker_id: str = "",
    target_config: Path | None = None,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    target = _norm(target)
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    teachers = [str(item) for item in (scope.get("routes") or {}).get(target, [])]
    candidate_states = [load_candidate_state(output_root, case_id=case_id, target=target, teacher=teacher) for teacher in teachers]
    if not teachers or any(str(state.get("status") or "") not in TERMINAL_CANDIDATE_STATES for state in candidate_states):
        return {"status": "WAITING_FOR_CANDIDATES", "case_id": case_id, "target": target}
    candidates = [
        _candidate_for_selection(state)
        for state in candidate_states
        if state.get("status") == "SUCCESS" and state.get("candidate_exists") and state.get("prediction") and Path(str(state.get("prediction"))).is_file()
    ]
    ct_path = ""
    for state in candidate_states:
        source_meta = _read_json(Path(str(state.get("selection_metadata") or "")), {})
        ct_path = str(source_meta.get("ct_path") or state.get("ct_path") or ct_path)
    ct = Path(ct_path) if ct_path else Path("missing_ct.nii.gz")
    selection_dir = output_root / "selection_runs" / re_safe(case_id)
    selected: dict[str, Any] | None = None
    try:
        selected, selection = _select_candidate(
            ct=ct,
            organ=target,
            candidates=candidates,
            out=selection_dir,
            case_id=case_id,
            enable_critic=bool(len(candidates) > 1),
            critic_backend="labelcritic",
            critic_base_url=critic_base_url,
            critic_port=int(critic_port),
            timeout_sec=int(timeout_sec),
            dry_run=False,
            strict_labelcritic_selection=True,
            formal_72b_selection_ready=True,
        )
    except Exception as exc:
        state = {
            "status": "RETRY_PENDING",
            "case_id": case_id,
            "target": target,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "updated_at": utc_now(),
        }
        atomic_write_json(_state_path(output_root, case_id, target, kind="target"), state)
        release_claim(output_root, claim_kind="case_target", claim_key=f"{case_id}|{target}")
        return state
    terminal_state = _terminal_from_selection(selection, selected, len(candidates))
    final_mask = ""
    if selected and selected.get("prediction") and Path(str(selected["prediction"])).is_file():
        final_dir = output_root / "annotation_versions" / case_id / "updated"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{target}.nii.gz"
        tmp = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
        shutil.copy2(str(selected["prediction"]), str(tmp))
        tmp.replace(final_path)
        final_mask = str(final_path)
    result = {
        "status": terminal_state,
        "case_id": case_id,
        "ct_path": ct_path,
        "target": target,
        "organ": target,
        "selected_model": (selected or {}).get("model") or selection.get("selected_model"),
        "selected_candidate_id": (selected or {}).get("candidate_id"),
        "selected_prediction": (selected or {}).get("prediction") or selection.get("selected_prediction"),
        "final_mask": final_mask,
        "mask_path": final_mask,
        "selection": selection,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "teacher_names": [candidate["model"] for candidate in candidates],
        "labelcritic_compare_used": bool(selection.get("labelcritic_records") or selection.get("critic_records")),
        "labelcritic_records": selection.get("labelcritic_records") or selection.get("critic_records") or [],
        "updated_at": utc_now(),
    }
    atomic_write_json(_state_path(output_root, case_id, target, kind="selection"), result)
    atomic_write_json(_state_path(output_root, case_id, target, kind="target"), result)
    _append_event(output_root, {"event": "case_target_selected", "case_id": case_id, "target": target, "status": terminal_state, "selected_model": result.get("selected_model")})
    release_claim(output_root, claim_kind="case_target", claim_key=f"{case_id}|{target}")
    return result


def run_labelcritic_selection_worker(
    output_root: Path,
    *,
    worker_id: str,
    critic_base_url: str,
    critic_port: int,
    target_config: Path | None = None,
    poll_sec: int = 30,
    max_idle_sec: int = 300,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    started = time.time()
    idle_since: float | None = None
    completed = 0
    wait_labelcritic = 0
    while True:
        claim = claim_next_case_target(output_root, worker_id=worker_id, labelcritic_ready=True)
        if claim["status"] == "CLAIMED":
            idle_since = None
            case_target = claim["case_target"]
            result = select_case_target(
                output_root,
                case_id=str(case_target["case_id"]),
                target=str(case_target["target"]),
                critic_base_url=critic_base_url,
                critic_port=critic_port,
                worker_id=worker_id,
                target_config=target_config,
                timeout_sec=timeout_sec,
            )
            completed += 1 if result.get("status") in VALID_TERMINAL_TARGET_STATES else 0
            continue
        if claim["status"] == "WAITING_FOR_LABELCRITIC":
            wait_labelcritic += 1
        if idle_since is None:
            idle_since = time.time()
        if time.time() - idle_since >= max_idle_sec:
            telemetry = build_estep_telemetry(output_root)
            return {
                "status": "IDLE_EXIT",
                "worker_id": worker_id,
                "completed": completed,
                "wait_labelcritic_count": wait_labelcritic,
                "runtime_sec": round(time.time() - started, 3),
                "telemetry": telemetry,
            }
        time.sleep(max(1, int(poll_sec)))


def build_estep_telemetry(output_root: Path) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {}) or _read_json(output_root / "full_round1_submission_scope.json", {})
    candidate_total = int(scope.get("total_logical_candidate_tasks") or len(scope.get("task_rows") or []))
    case_target_total = int(scope.get("case_count") or 0) * int(scope.get("canonical_target_count") or 0)
    candidate_counts: dict[str, int] = {}
    for path in _queue_paths(output_root)["candidate_states"].glob("*/*/*.json"):
        status = str(_read_json(path, {}).get("status") or "PENDING")
        candidate_counts[status] = candidate_counts.get(status, 0) + 1
    target_counts: dict[str, int] = {}
    for path in _queue_paths(output_root)["case_target_states"].glob("*/*.json"):
        status = str(_read_json(path, {}).get("status") or "WAITING_FOR_CANDIDATES")
        target_counts[status] = target_counts.get(status, 0) + 1
    terminal_target_count = sum(target_counts.get(state, 0) for state in VALID_TERMINAL_TARGET_STATES)
    queue_depth = target_counts.get("CANDIDATES_READY", 0) + target_counts.get("WAITING_FOR_LABELCRITIC", 0)
    telemetry = {
        "status": "READY",
        "updated_at": utc_now(),
        "teacher_candidate": {
            "total": candidate_total,
            "success": candidate_counts.get("SUCCESS", 0),
            "running": candidate_counts.get("RUNNING", 0) + candidate_counts.get("CLAIMED", 0),
            "pending": max(0, candidate_total - sum(candidate_counts.values())),
            "retry": candidate_counts.get("RETRY_PENDING", 0) + candidate_counts.get("BACKPRESSURED", 0),
            "terminal": sum(candidate_counts.get(state, 0) for state in TERMINAL_CANDIDATE_STATES),
            "counts": candidate_counts,
        },
        "case_target": {
            "total": case_target_total,
            "waiting_candidates": max(0, case_target_total - sum(target_counts.values())) + target_counts.get("WAITING_FOR_CANDIDATES", 0),
            "candidates_ready": target_counts.get("CANDIDATES_READY", 0),
            "labelcritic_running": target_counts.get("LABELCRITIC_RUNNING", 0),
            "selected": target_counts.get("SELECTED", 0),
            "single_teacher_accepted": target_counts.get("VALID_SINGLE_TEACHER_ACCEPTED", 0),
            "absent": target_counts.get("ABSENT", 0) + target_counts.get("ABSENT_NEGATIVE", 0) + target_counts.get("NEGATIVE_ABSENT", 0),
            "failed": target_counts.get("FAILED_FINAL", 0),
            "terminal": terminal_target_count,
            "counts": target_counts,
        },
        "labelcritic": {
            "queue_depth": queue_depth,
            "running_requests": target_counts.get("LABELCRITIC_RUNNING", 0),
            "completed": target_counts.get("SELECTED", 0),
            "retry": target_counts.get("RETRY_PENDING", 0),
        },
    }
    atomic_write_json(_queue_paths(output_root)["telemetry"], telemetry)
    return telemetry


def _selection_training_item(selection: dict[str, Any], *, target_config: Path | None = None) -> dict[str, Any] | None:
    if not selection.get("final_mask") or not Path(str(selection.get("final_mask"))).is_file():
        return None
    targets, prompts = _load_target_config_prompts(target_config or (REPO_ROOT / "configs/student_3d_prompt_target_organs.json"))
    target = str(selection.get("target") or selection.get("organ") or "")
    try:
        target_id = targets.index(target)
    except ValueError:
        target_id = -1
    item = {
        "case_id": str(selection.get("case_id") or ""),
        "image": str(selection.get("ct_path") or ""),
        "ct_path": str(selection.get("ct_path") or ""),
        "organ": target,
        "canonical_organ": target,
        "requested_canonical_id": target,
        "resolved_canonical_id": target,
        "prompt": prompts.get(target, target.replace("_", " ")),
        "mask": str(selection.get("final_mask")),
        "mask_path": str(selection.get("final_mask")),
        "supervision_type": "positive",
        "target_type": "positive_hard",
        "label_role": "selected_pseudo_label",
        "supervision_role": "selected_pseudo_label",
        "distillation_role": "positive",
        "dataset_role": "pseudo_label",
        "source_model": str(selection.get("selected_model") or ""),
        "selected_model": str(selection.get("selected_model") or ""),
        "origin_provider": str(selection.get("selected_model") or ""),
        "ground_truth_status": "selected_pseudo_label_not_expert_gt",
        "scoring_schema_version": "autolabel_core_v3",
        "grade": "A",
        "training_weight": 1.0,
        "distillation_eligible": True,
        "training_eligible": True,
        "student_target_id": target_id,
        "source_stage": "full_373_multiteacher_round1_estep",
    }
    return canonicalize_training_record(item, round_index=1, project_root=REPO_ROOT, strict_soft=True)


def aggregate_full373_estep(output_root: Path, *, expected_cases: int = 103, expected_targets: int = 373) -> dict[str, Any]:
    scope = _read_json(output_root / "full_round1_scope.json", {})
    if int(scope.get("case_count") or 0) > 0:
        expected_cases = int(scope.get("case_count") or expected_cases)
    if int(scope.get("canonical_target_count") or 0) > 0:
        expected_targets = int(scope.get("canonical_target_count") or expected_targets)
    cases = sorted({
        str(row.get("case_id") or "")
        for row in (scope.get("task_rows") or [])
        if isinstance(row, dict) and row.get("case_id")
    })
    if not cases:
        manifest_rows: list[dict[str, str]] = []
        for task_manifest in sorted((output_root / "slurm").glob("full373_task_manifest_*.csv")):
            manifest_rows.extend(read_csv_rows(task_manifest))
        cases = sorted({str(row.get("case_id") or "") for row in manifest_rows if row.get("case_id")})
    telemetry = build_estep_telemetry(output_root)
    state_items = []
    training_items = []
    pending = []
    failed = []
    state_files = list(_queue_paths(output_root)["case_target_states"].glob("*/*.json"))
    if state_files:
        for case_id in cases:
            for target in _target_rows_from_scope(scope):
                state = _read_json(_state_path(output_root, case_id, target, kind="target"), {})
                status = str(state.get("status") or "WAITING_FOR_CANDIDATES")
                if status in VALID_TERMINAL_TARGET_STATES:
                    state_items.append({"case_id": case_id, "organ": target, "terminal_state": status, **state})
                    item = _selection_training_item(state)
                    if item:
                        training_items.append(item)
                elif status in NON_TERMINAL_TARGET_STATES:
                    pending.append({"case_id": case_id, "organ": target, "reason": status})
                else:
                    failed.append({"case_id": case_id, "organ": target, "reason": status})
        manifest_targets = len(state_items)
        expected_total = expected_cases * expected_targets
        status = "PASSED" if len(cases) == expected_cases and manifest_targets == expected_total and not pending and not failed else "RUNNING"
        if failed and not pending:
            status = "FAILED"
        trainable = [
            item for item in training_items
            if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0
        ]
        training_manifest = {
            "version": "full_373_multiteacher_round1_voxtell_manifest_v1",
            "stage": "round1_mstep_manifest",
            "status": "success" if status == "PASSED" else "pending",
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "source_formal_root": str(output_root),
            "num_items": len(training_items),
            "num_cases": len({str(item.get("case_id") or "") for item in training_items}),
            "num_distillation_eligible_items": len(trainable),
            "items": training_items,
        }
        if status == "PASSED":
            write_json(output_root / "training_manifest.json", training_manifest)
        report = {
            "stage": "full_373_multiteacher_round1_estep_gate",
            "status": status,
            "case_count": len(cases),
            "expected_case_count": expected_cases,
            "canonical_target_count": expected_targets,
            "expected_targets": expected_total,
            "manifest_targets": manifest_targets,
            "complete_case_373": manifest_targets == expected_total,
            "pending_cases": pending[:100],
            "failed_cases": failed[:100],
            "allowed_terminal_states": sorted(VALID_TERMINAL_TARGET_STATES),
            "training_manifest": str(output_root / "training_manifest.json") if training_items else "",
            "num_training_items": len(training_items),
            "num_distillation_eligible_items": len(trainable),
            "telemetry": telemetry,
        }
        write_json(output_root / "full_case_373_manifest.json", {"stage": "full_case_373_estep_manifest", "status": "success" if status == "PASSED" else status.lower(), "items": state_items})
        write_json(output_root / "full_373_estep_status.json", report)
        return report

    items = []
    training_items = []
    pending = []
    failed = []
    for case_id in cases:
        case_root = output_root / "run_loop_cases" / case_id
        manifest = _read_json(case_root / "full_case_373_manifest.json", {})
        summary = _read_json(case_root / "case_373_target_summary.json", {})
        task_state = _read_json(output_root / "task_states" / f"{case_id}.json", {})
        if task_state.get("status") == "FAILED":
            failed.append({"case_id": case_id, "reason": "run_loop_task_failed", "task_state": task_state})
            continue
        if manifest.get("status") != "success" or not summary.get("complete_case_373"):
            pending.append({"case_id": case_id, "reason": "full_case_373_not_complete", "manifest_status": manifest.get("status"), "complete_case_373": summary.get("complete_case_373")})
            continue
        items.extend(manifest.get("items") or [])
        tm = _read_json(case_root / "training_manifest.json", {})
        training_items.extend(tm.get("items") or [])
    manifest_targets = len(items)
    expected_total = expected_cases * expected_targets
    status = "PASSED" if len(cases) == expected_cases and manifest_targets == expected_total and not pending and not failed else "RUNNING"
    if failed and not pending:
        status = "FAILED"
    trainable = [
        item for item in training_items
        if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0
    ]
    training_manifest = {
        "version": "full_373_multiteacher_round1_voxtell_manifest_v1",
        "stage": "round1_mstep_manifest",
        "status": "success" if training_items else "pending",
        "source_formal_root": str(output_root),
        "num_items": len(training_items),
        "num_cases": len({str(item.get("case_id") or "") for item in training_items}),
        "num_distillation_eligible_items": len(trainable),
        "items": training_items,
    }
    if training_items:
        write_json(output_root / "training_manifest.json", training_manifest)
    report = {
        "stage": "full_373_multiteacher_round1_estep_gate",
        "status": status,
        "case_count": len(cases),
        "expected_case_count": expected_cases,
        "canonical_target_count": expected_targets,
        "expected_targets": expected_total,
        "manifest_targets": manifest_targets,
        "complete_case_373": manifest_targets == expected_total,
        "pending_cases": pending[:100],
        "failed_cases": failed[:100],
        "allowed_terminal_states": sorted(VALID_TERMINAL_TARGET_STATES),
        "training_manifest": str(output_root / "training_manifest.json") if training_items else "",
        "num_training_items": len(training_items),
        "num_distillation_eligible_items": len(trainable),
    }
    write_json(output_root / "full_373_estep_status.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Full 373 multi-Teacher Round1 launcher.")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs/model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs/student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--checkpoint-root", default=REPO_ROOT / "checkpoints", type=Path)
    parser.add_argument("--nnunet-predict-executable", default=Path("nnUNetv2_predict"), type=Path)
    parser.add_argument("--unest-python-executable", default=Path(sys.executable), type=Path)
    parser.add_argument("--cache-root", action="append", default=[], type=Path)
    parser.add_argument("--execute-task-index", type=int)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--selection-worker", action="store_true")
    parser.add_argument("--critic-base-url", default=os.getenv("LABELCRITIC_BASE_URL", "http://localhost"))
    parser.add_argument("--critic-port", default=int(os.getenv("LABELCRITIC_PORT", "8000")), type=int)
    parser.add_argument("--selection-poll-sec", default=int(os.getenv("LABELCRITIC_SELECTION_POLL_SEC", "30")), type=int)
    parser.add_argument("--selection-max-idle-sec", default=int(os.getenv("LABELCRITIC_SELECTION_MAX_IDLE_SEC", "300")), type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.execute_task_index is not None:
        if not args.task_manifest:
            raise SystemExit("--task-manifest is required with --execute-task-index")
        print(json.dumps(execute_task_index(args.execute_task_index, args.task_manifest.resolve(), output_root, worker_id=args.worker_id, state_root=args.state_root), indent=2, default=str))
        return 0
    if args.selection_worker:
        print(json.dumps(run_labelcritic_selection_worker(
            output_root,
            worker_id=args.worker_id or f"selection_{os.getpid()}",
            critic_base_url=args.critic_base_url,
            critic_port=args.critic_port,
            target_config=args.target_config,
            poll_sec=args.selection_poll_sec,
            max_idle_sec=args.selection_max_idle_sec,
        ), indent=2, default=str))
        return 0
    if args.aggregate:
        report = aggregate_full373_estep(output_root)
        print(json.dumps({"status": report["status"], "manifest_targets": report["manifest_targets"], "expected_targets": report["expected_targets"]}, indent=2))
        return 0 if report["status"] in {"PASSED", "RUNNING"} else 2
    if not args.case_manifest:
        raise SystemExit("--case-manifest is required")
    summary = build_submission_manifest(
        case_manifest=args.case_manifest.resolve(),
        output_root=output_root,
        registry_path=args.registry.resolve(),
        target_config=args.target_config.resolve(),
        python=args.python,
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        unest_python_executable=args.unest_python_executable,
        state_root=args.state_root,
        cache_roots=[path.resolve() for path in args.cache_root],
    )
    print(json.dumps({"status": summary["status"], "task_count": summary["task_count"], "scope": summary["scope"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
