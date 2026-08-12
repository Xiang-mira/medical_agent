#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.runtime_resolver import DEFAULT_HPC_CHECKPOINT_ROOT, DEFAULT_NNUNETV2_PREDICT, DEFAULT_UNEST_PYTHON  # noqa: E402
from tools.dataset_delivery.cads15_slurm_resources import preflight_sbatch_script, profile_manifest, resolve_gpu_profile  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_rows, sha256_file, write_csv, write_json  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import (  # noqa: E402
    DEFAULT_BASE_MANIFEST,
    DEFAULT_OUTPUT_MANIFEST,
    FORMAL_APPEND_CASE_IDS,
    FORMAL_CASE_COUNT,
    FORMAL_GROUP_MODELS,
    FORMAL_MODEL_KEYS,
    FORMAL_MODEL_TARGETS,
    FORMAL_TARGETS,
    FORMAL_TARGET_TO_GROUP,
    _annotation_folder,
    _case_id,
    _ct_path,
    validate_formal_manifest,
)
from tools.dataset_delivery.task2_formal_validator import VALID_COMPLETION_STATUSES, validate_case_group_run  # noqa: E402
from tools.dataset_delivery.task2_preflight import build_preflight  # noqa: E402
from tools.dataset_delivery.task2_recovery import recover_case_group  # noqa: E402
from tools.dataset_delivery.task2_smoke_launcher import _command_for_group  # noqa: E402


DEFAULT_CODE_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent")
DEFAULT_PYTHON = Path("/home/xhan74/envs/medical_agent/bin/python")
DEFAULT_TARGET_CONFIG = REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json"
DEFAULT_MODEL_REGISTRY = REPO_ROOT / "configs" / "model_registry.yaml"
DEFAULT_FORMAL_OUTPUT_ROOT = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/formal_task2_22targets_103cases_$(date +%Y%m%d_%H%M%S)"
)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_commit(repo: Path) -> str:
    rc, out, _ = _run_git(["rev-parse", "HEAD"], repo)
    return out if rc == 0 else ""


def _tracked_clean(repo: Path) -> bool:
    rc, out, _ = _run_git(["status", "--porcelain", "--untracked-files=no"], repo)
    return rc == 0 and not out.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_groups(groups: list[str] | None) -> list[str]:
    if groups is None:
        return list(FORMAL_MODEL_TARGETS)
    unknown = sorted(set(groups) - set(FORMAL_MODEL_TARGETS))
    if unknown:
        raise ValueError(f"unknown formal groups: {unknown}")
    return [group for group in FORMAL_MODEL_TARGETS if group in groups]


def _task_state_path(output_root: Path, case_id: str, group: str) -> Path:
    return output_root / "tasks" / case_id / group / "task_state.json"


def _group_root(output_root: Path, group: str) -> Path:
    return output_root / group


def _group_task_manifest(output_root: Path, group: str) -> Path:
    return _group_root(output_root, group) / f"{group}_task_manifest.csv"


def _group_run_out(output_root: Path, case_id: str, group: str) -> Path:
    return output_root / "cases" / case_id / group / "run_loop"


def _task_completed(output_root: Path, case_id: str, group: str, ct_path: Path) -> bool:
    targets = FORMAL_MODEL_TARGETS[group]
    for target in targets:
        row = validate_case_group_run(
            output_root=output_root,
            case_id=case_id,
            group=group,
            target=target,
            ct_path=ct_path,
        )
        if row["final_status"] not in VALID_COMPLETION_STATUSES:
            return False
    return True


def _task_failed(output_root: Path, case_id: str, group: str, ct_path: Path) -> bool:
    return not _task_completed(output_root, case_id, group, ct_path)


def _write_array_sbatch(
    path: Path,
    *,
    code_root: Path,
    python: Path,
    task_manifest: Path,
    output_root: Path,
    group: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
) -> None:
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=task2_{group}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={output_root / 'slurm' / (group + '_%A_%a.out')}
#SBATCH --error={output_root / 'slurm' / (group + '_%A_%a.err')}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {code_root}
{python} tools/dataset_delivery/task2_formal_launcher.py \\
  --execute-task-index "$SLURM_ARRAY_TASK_ID" \\
  --task-manifest {task_manifest} \\
  --output-root {output_root}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _formal_model_keys() -> list[str]:
    return [model for models in FORMAL_GROUP_MODELS.values() for model in models]


def execute_task_index(
    *,
    task_index: int,
    task_manifest: Path,
    output_root: Path,
    code_root: Path = DEFAULT_CODE_ROOT,
    python: Path = DEFAULT_PYTHON,
    checkpoint_root: Path = DEFAULT_HPC_CHECKPOINT_ROOT,
    nnunet_predict_executable: Path = DEFAULT_NNUNETV2_PREDICT,
    unest_python_executable: Path | None = DEFAULT_UNEST_PYTHON,
    timeout_sec: int = 14400,
) -> dict[str, Any]:
    rows = read_csv_rows(task_manifest)
    row = next((item for item in rows if int(item.get("task_index") or -1) == task_index), None)
    if row is None:
        raise IndexError(f"task index not found: {task_index}")
    case_id = str(row["case_id"])
    group = str(row["model_group"])
    targets = [target for target in str(row["targets"]).split(",") if target]
    ct_path = Path(str(row.get("ct_path") or ""))
    task_root = output_root / "tasks" / case_id / group
    run_out = _group_run_out(output_root, case_id, group)
    case_csv = task_root / "selected_case_manifest.csv"
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(case_csv, [{"case_id": case_id, "ct_path": str(ct_path), "annotation_folder": str(row.get("annotation_folder") or "")}], ["case_id", "ct_path", "annotation_folder"])
    if _task_completed(output_root, case_id, group, ct_path):
        state = {
            "status": "completed",
            "case_id": case_id,
            "group": group,
            "targets": targets,
            "task_index": task_index,
            "run_out": str(run_out),
            "return_code": 0,
            "resume_reason": "existing_valid_result",
        }
        write_json(task_root / "task_state.json", state)
        return state
    recovery = recover_case_group(
        output_root=output_root,
        case_id=case_id,
        group=group,
        targets=targets,
        ct_path=ct_path,
        models=[item for item in str(row.get("models") or "").split(",") if item],
        apply=True,
    )
    if recovery.get("recovered_target_count") and _task_completed(output_root, case_id, group, ct_path):
        state = {
            "status": "completed",
            "case_id": case_id,
            "group": group,
            "targets": targets,
            "task_index": task_index,
            "run_out": str(run_out),
            "return_code": 0,
            "resume_reason": "recovered_existing_raw_teacher_output",
            "candidate_recovery": recovery,
        }
        write_json(task_root / "task_state.json", state)
        return state
    registry_path = Path(str(row.get("registry_path") or REPO_ROOT / "configs" / "model_registry.yaml"))
    command = _command_for_group(
        group=group,
        python=python,
        case_csv=case_csv,
        registry=registry_path,
        run_out=run_out,
        timeout_sec=timeout_sec,
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=nnunet_predict_executable,
        unest_python_executable=unest_python_executable or DEFAULT_UNEST_PYTHON,
        enable_shapekit=True,
        enable_critic=True,
    )
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "command.txt").write_text(" ".join(str(part) for part in command) + "\n", encoding="utf-8")
    write_json(task_root / "task_state.json", {
        "status": "running",
        "case_id": case_id,
        "group": group,
        "targets": targets,
        "task_index": task_index,
        "run_out": str(run_out),
    })
    proc = subprocess.run(command, cwd=code_root, text=True, check=False)
    final_state = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "case_id": case_id,
        "group": group,
        "targets": targets,
        "task_index": task_index,
        "return_code": int(proc.returncode),
        "run_out": str(run_out),
    }
    validation_rows = [
        validate_case_group_run(
            output_root=output_root,
            case_id=case_id,
            group=group,
            target=target,
            ct_path=ct_path,
        )
        for target in targets
    ]
    final_state["validation_status"] = "PASSED" if all(row["final_status"] in VALID_COMPLETION_STATUSES for row in validation_rows) else "VALIDATION_FAILED"
    final_state["validation_rows"] = validation_rows
    if proc.returncode == 0 and final_state["validation_status"] == "PASSED":
        final_state["status"] = "completed"
    else:
        final_state["status"] = "failed"
    write_json(task_root / "task_state.json", final_state)
    return final_state


def build_formal_plan(
    *,
    output_root: Path,
    case_manifest: Path,
    base_manifest: Path = DEFAULT_BASE_MANIFEST,
    code_root: Path = DEFAULT_CODE_ROOT,
    python: Path = DEFAULT_PYTHON,
    registry_path: Path = DEFAULT_MODEL_REGISTRY,
    target_config: Path = DEFAULT_TARGET_CONFIG,
    checkpoint_root: Path = DEFAULT_HPC_CHECKPOINT_ROOT,
    nnunet_predict_executable: Path = DEFAULT_NNUNETV2_PREDICT,
    unest_python_executable: Path | None = DEFAULT_UNEST_PYTHON,
    expected_case_count: int = FORMAL_CASE_COUNT,
    groups: list[str] | None = None,
    case_id: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    allow_dirty_tracked: bool = False,
    gpu_partition: str | None = None,
    gpu_gres: str | None = None,
    gpu_cpus_per_task: int | None = None,
    gpu_mem: str | None = None,
    gpu_time_limit: str | None = None,
    run_slurm_test_only: bool = True,
) -> dict[str, Any]:
    groups = _resolve_groups(groups)
    case_manifest = case_manifest.resolve()
    output_root = output_root.resolve()
    task_manifest_report = output_root / "preflight" / "task2_formal_manifest_validation.json"
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, reason: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "reason": reason})

    add_check("canonical_code_root_exists", code_root.exists(), str(code_root))
    add_check("tracked_worktree_clean", allow_dirty_tracked or _tracked_clean(code_root), "git status --porcelain --untracked-files=no")
    add_check("case_manifest_exists", case_manifest.exists(), str(case_manifest))
    add_check("target_config_exists", target_config.exists(), str(target_config))
    add_check("registry_exists", registry_path.exists(), str(registry_path))
    add_check("checkpoint_root_exists", checkpoint_root.exists(), str(checkpoint_root))
    add_check("predictor_executable", nnunet_predict_executable.exists() and os.access(nnunet_predict_executable, os.X_OK), str(nnunet_predict_executable))
    if unest_python_executable is not None:
        add_check("unest_python_executable", unest_python_executable.exists() and os.access(unest_python_executable, os.X_OK), str(unest_python_executable))
    add_check("output_root_policy", dry_run or resume or not output_root.exists(), "new output root unless resume/dry-run")
    add_check("formal_case_count", True, f"expected {expected_case_count}")
    add_check("formal_target_count", len(FORMAL_TARGETS) == 22, str(len(FORMAL_TARGETS)))

    try:
        manifest_audit = validate_formal_manifest(
            manifest=case_manifest,
            base_manifest=base_manifest,
            report=None,
        )
        manifest_audit["final_count"] = manifest_audit.get("rows", 0)
        manifest_audit["manifest_sha256"] = sha256_file(case_manifest)
        add_check("case_manifest_validation", True, "")
    except Exception as exc:
        add_check("case_manifest_validation", False, str(exc))
        manifest_audit = {
            "status": "failed",
            "errors": [str(exc)],
            "manifest_sha256": "",
            "base_count": 0,
            "append_count": 0,
            "final_count": 0,
        }

    preflight_json = output_root / "preflight" / "task2_formal_preflight.json"
    try:
        preflight = build_preflight(
            models=_formal_model_keys(),
            case_list=case_manifest,
            output_root=output_root,
            registry_path=registry_path,
            target_config=target_config,
            checkpoint_root_arg=checkpoint_root,
            json_output=preflight_json,
            canonical_code_root=code_root,
            formal_mode=True,
            outer_python=python,
            nnunet_predict_executable=str(nnunet_predict_executable),
            unest_python_executable=str(unest_python_executable) if unest_python_executable else None,
            run_predictor_help=False,
            require_cuda=False,
            resume=resume,
            allow_dirty_tracked=allow_dirty_tracked,
        )
    except Exception as exc:
        preflight = {
            "status": "BLOCKED",
            "blocked_count": 1,
            "blocked_checks": [{"name": "formal_runtime_preflight_exception", "ok": False, "error": f"{type(exc).__name__}: {exc}"}],
            "checks": [],
            "runtime_manifest": {},
        }

    blocked_checks = [check for check in checks if not check["ok"]]
    if manifest_audit.get("status") not in {"READY", "success"}:
        blocked_checks.append({"name": "formal_manifest", "ok": False, "reason": "; ".join(str(error.get("type")) for error in manifest_audit.get("errors", []))})
    if preflight["status"] != "READY":
        blocked_checks.extend(preflight["blocked_checks"])

    if any(not check["ok"] for check in blocked_checks):
        summary = {
            "status": "BLOCKED",
            "dry_run": dry_run,
            "resume": resume,
            "retry_failed": retry_failed,
            "formal_task2_22targets_103cases": True,
            "case_count": manifest_audit.get("final_count", 0),
            "expected_case_count": expected_case_count,
            "task_count": 0,
            "model_group_count": len(groups),
            "target_count": len(FORMAL_TARGETS),
            "manifest_sha256": manifest_audit.get("manifest_sha256", ""),
            "case_manifest": str(case_manifest),
            "checks": checks,
            "blocked_checks": blocked_checks,
            "preflight": preflight,
            "submission_manifest": "",
            "task_registry": "",
            "git_commit": _git_commit(code_root),
        }
        write_json(output_root / "formal_task2_preflight.json", summary)
        write_json(task_manifest_report, manifest_audit)
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(task_manifest_report, manifest_audit)
    case_rows = read_csv_rows(case_manifest)
    manifest_sha256 = sha256_file(case_manifest)
    target_config_sha256 = sha256_file(target_config)
    registry_sha256 = sha256_file(registry_path)
    python_sha256 = sha256_file(python.resolve()) if python.exists() else ""
    predictor_sha256 = sha256_file(nnunet_predict_executable.resolve()) if nnunet_predict_executable.exists() else ""
    unest_python_sha256 = ""
    if unest_python_executable and unest_python_executable.exists():
        unest_python_sha256 = sha256_file(unest_python_executable.resolve())
    tasks_by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}
    selected_rows = case_rows
    if case_id:
        selected_rows = [row for index, row in enumerate(case_rows) if _case_id(row, index) == case_id]

    for index, row in enumerate(selected_rows):
        cid = _case_id(row, index)
        ct_path = _ct_path(row)
        ann = _annotation_folder(row)
        for group in groups:
            group_targets = FORMAL_MODEL_TARGETS[group]
            group_state = _task_state_path(output_root, cid, group)
            if resume and _task_completed(output_root, cid, group, Path(ct_path)):
                continue
            if retry_failed and not _task_failed(output_root, cid, group, Path(ct_path)):
                continue
            tasks_by_group[group].append({
                "task_index": len(tasks_by_group[group]),
                "case_id": cid,
                "model_group": group,
                "models": ",".join(FORMAL_GROUP_MODELS[group]),
                "targets": ",".join(group_targets),
                "ct_path": ct_path,
                "annotation_folder": ann,
                "task_state_path": str(group_state),
                "case_manifest_sha256": manifest_sha256,
                "target_config_sha256": target_config_sha256,
                "model_registry_sha256": registry_sha256,
                "registry_path": str(registry_path),
                "runtime_python": str(python.resolve()),
                "runtime_python_sha256": python_sha256,
                "nnunet_predict_executable": str(nnunet_predict_executable.resolve()),
                "nnunet_predict_sha256": predictor_sha256,
                "unest_python_executable": str(unest_python_executable.resolve()) if unest_python_executable else "",
                "unest_python_sha256": unest_python_sha256,
                "status": "planned",
            })

    gpu_profile = resolve_gpu_profile(
        partition=gpu_partition,
        gres=gpu_gres,
        cpus_per_task=gpu_cpus_per_task,
        memory=gpu_mem,
        time_limit=gpu_time_limit,
    )
    group_summaries: dict[str, Any] = {}
    total_task_count = 0
    for group, task_rows in tasks_by_group.items():
        group_root = _group_root(output_root, group)
        manifest_path = _group_task_manifest(output_root, group)
        group_root.mkdir(parents=True, exist_ok=True)
        write_csv(
            manifest_path,
            task_rows,
            [
                "task_index", "case_id", "model_group", "models", "targets", "ct_path", "annotation_folder",
                "task_state_path", "case_manifest_sha256", "target_config_sha256", "model_registry_sha256",
                "registry_path", "runtime_python", "runtime_python_sha256", "nnunet_predict_executable", "nnunet_predict_sha256",
                "unest_python_executable", "unest_python_sha256", "status",
            ],
        )
        sbatch_path = output_root / "slurm" / f"{group}_task2_array.sbatch"
        _write_array_sbatch(
            sbatch_path,
            code_root=code_root,
            python=python,
            task_manifest=manifest_path,
            output_root=output_root,
            group=group,
            partition=gpu_profile.partition,
            gres=gpu_profile.gres,
            cpus_per_task=gpu_profile.cpus_per_task,
            mem=gpu_profile.memory,
            time_limit=gpu_profile.time_limit,
        )
        resource_preflight = preflight_sbatch_script(
            profile=gpu_profile,
            sbatch_file=sbatch_path,
            run_sbatch_test_only=run_slurm_test_only,
        )
        group_summary = {
            "group": group,
            "models": list(FORMAL_GROUP_MODELS[group]),
            "targets": list(FORMAL_MODEL_TARGETS[group]),
            "task_count": len(task_rows),
            "task_manifest": str(manifest_path),
            "sbatch_file": str(sbatch_path),
            "resources": profile_manifest(gpu_profile),
            "resource_preflight": resource_preflight,
        }
        group_summaries[group] = group_summary
        total_task_count += len(task_rows)

    summary = {
        "status": "READY",
        "dry_run": dry_run,
        "resume": resume,
        "retry_failed": retry_failed,
        "formal_task2_22targets_103cases": True,
        "case_count": len(case_rows),
        "expected_case_count": expected_case_count,
        "task_count": total_task_count,
        "model_group_count": len(groups),
        "target_count": len(FORMAL_TARGETS),
        "manifest_sha256": manifest_sha256,
        "target_config_sha256": target_config_sha256,
        "model_registry_sha256": registry_sha256,
        "runtime_python_sha256": python_sha256,
        "nnunet_predict_sha256": predictor_sha256,
        "unest_python_sha256": unest_python_sha256,
        "case_manifest": str(case_manifest),
        "target_config": str(target_config),
        "registry_path": str(registry_path),
        "git_commit": _git_commit(code_root),
        "checks": checks,
        "blocked_checks": blocked_checks,
        "preflight": preflight,
        "manifest_audit": manifest_audit,
        "groups": group_summaries,
    }
    write_json(output_root / "formal_task2_preflight.json", summary)
    write_json(output_root / "formal_task2_submission_manifest.json", summary)
    write_json(output_root / "formal_task2_runtime_manifest.json", preflight.get("runtime_manifest") or {})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Task 2 formal 103-case x 22-target launch manifests.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-manifest", default=DEFAULT_OUTPUT_MANIFEST, type=Path)
    parser.add_argument("--base-manifest", default=DEFAULT_BASE_MANIFEST, type=Path)
    parser.add_argument("--code-root", default=DEFAULT_CODE_ROOT, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path)
    parser.add_argument("--registry", default=DEFAULT_MODEL_REGISTRY, type=Path)
    parser.add_argument("--target-config", default=DEFAULT_TARGET_CONFIG, type=Path)
    parser.add_argument("--checkpoint-root", default=DEFAULT_HPC_CHECKPOINT_ROOT, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=DEFAULT_NNUNETV2_PREDICT, type=Path)
    parser.add_argument("--unest-python-executable", default=DEFAULT_UNEST_PYTHON, type=Path)
    parser.add_argument("--expected-case-count", default=FORMAL_CASE_COUNT, type=int)
    parser.add_argument("--groups", default="")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tracked", action="store_true")
    parser.add_argument("--execute-task-index", default=None)
    parser.add_argument("--task-manifest", default=None, type=Path)
    parser.add_argument("--timeout-sec", default=14400, type=int)
    parser.add_argument("--gpu-partition", default=None)
    parser.add_argument("--gpu-gres", default=None)
    parser.add_argument("--gpu-cpus-per-task", default=None, type=int)
    parser.add_argument("--gpu-mem", default=None)
    parser.add_argument("--gpu-time-limit", default=None)
    parser.add_argument("--no-slurm-test-only", action="store_true")
    args = parser.parse_args()
    if args.execute_task_index is not None:
        if args.task_manifest is None:
            raise SystemExit("--task-manifest is required with --execute-task-index")
        state = execute_task_index(
            task_index=int(args.execute_task_index),
            task_manifest=args.task_manifest.resolve(),
            output_root=args.output_root.resolve(),
            code_root=args.code_root.resolve(),
            python=args.python,
            checkpoint_root=args.checkpoint_root,
            nnunet_predict_executable=args.nnunet_predict_executable,
            unest_python_executable=args.unest_python_executable,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps({"status": state["status"], "case_id": state["case_id"], "group": state["group"]}, indent=2))
        return 0 if state["status"] == "completed" else 1
    groups = [item.strip() for item in args.groups.replace(";", ",").split(",") if item.strip()] or None
    summary = build_formal_plan(
        output_root=args.output_root.resolve(),
        case_manifest=args.case_manifest.resolve(),
        base_manifest=args.base_manifest.resolve(),
        code_root=args.code_root.resolve(),
        python=args.python,
        registry_path=args.registry.resolve(),
        target_config=args.target_config.resolve(),
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        unest_python_executable=args.unest_python_executable.resolve() if args.unest_python_executable else None,
        expected_case_count=args.expected_case_count,
        groups=groups,
        case_id=args.case_id or None,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        dry_run=bool(args.dry_run),
        allow_dirty_tracked=bool(args.allow_dirty_tracked),
        gpu_partition=args.gpu_partition,
        gpu_gres=args.gpu_gres,
        gpu_cpus_per_task=args.gpu_cpus_per_task,
        gpu_mem=args.gpu_mem,
        gpu_time_limit=args.gpu_time_limit,
        run_slurm_test_only=not bool(args.no_slurm_test_only),
    )
    print(json.dumps({"status": summary["status"], "task_count": summary["task_count"]}, indent=2))
    return 0 if summary["status"] == "READY" or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
