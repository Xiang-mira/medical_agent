#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.runtime_resolver import DEFAULT_HPC_CHECKPOINT_ROOT, DEFAULT_NNUNETV2_PREDICT  # noqa: E402
from tools.dataset_delivery.cads15_contract_audit import DEFAULT_CONTRACT, contract_targets  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402


DEFAULT_CODE_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent")
DEFAULT_PYTHON = Path("/home/xhan74/envs/medical_agent/bin/python")
CADS_MODEL_ORDER = ["cads553", "cads557", "cads559"]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_commit(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _tracked_worktree_clean(repo: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode == 0 and not proc.stdout.strip()


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _write_case_manifest(path: Path, case: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path", "annotation_folder"])
        writer.writeheader()
        writer.writerow({
            "case_id": case["case_id"],
            "ct_path": case["ct_path"],
            "annotation_folder": case["annotation_folder"],
        })


def _target_model_map(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, str]:
    return {
        str(row.get("canonical_id")): str(row.get("primary_model"))
        for row in contract_targets(contract_path)
        if row.get("canonical_id") and row.get("primary_model")
    }


def _models_for_targets(targets: list[str], contract_path: Path = DEFAULT_CONTRACT) -> list[str]:
    model_by_target = _target_model_map(contract_path)
    return [model for model in CADS_MODEL_ORDER if any(model_by_target.get(target) == model for target in targets)]


def _command_for_case(
    *,
    python: Path,
    case_csv: Path,
    models: list[str],
    targets: list[str],
    registry: Path,
    run_out: Path,
    timeout_sec: int,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
) -> list[str]:
    return [
        str(python), "run_medai_cli.py", "--json", "run-loop",
        "--case-list", str(case_csv),
        "--models", ",".join(models),
        "--organs", ",".join(targets),
        "--registry", str(registry),
        "--output", str(run_out),
        "--checkpoint-root", str(checkpoint_root),
        "--nnunet-predict-executable", str(nnunet_predict_executable),
        "--timeout-sec", str(timeout_sec),
        "--teacher-inference-mode", "full_volume",
        "--no-enable-shapekit",
        "--debug-allow-no-shapekit",
        "--no-enable-critic",
        "--strict-delivery-targets",
        "--log-file", str(run_out / "run_loop.log"),
    ]


def _write_sbatch(
    *,
    path: Path,
    code_root: Path,
    python: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    run_out: Path,
    command: list[str],
    expected_commit: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command_text = _shell_join(command)
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=cads15_smoke
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={path.parent / (path.stem + '_%j.out')}
#SBATCH --error={path.parent / (path.stem + '_%j.err')}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {shlex.quote(str(code_root))}
mkdir -p {shlex.quote(str(run_out))}
echo "$SLURM_JOB_ID" > {shlex.quote(str(run_out.parent / 'slurm_job_id.txt'))}
echo {shlex.quote(expected_commit)} > {shlex.quote(str(run_out.parent / 'expected_git_commit.txt'))}
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" != {shlex.quote(expected_commit)} ]; then
  echo "Commit mismatch: expected {expected_commit}, got $current_commit" >&2
  touch {shlex.quote(str(run_out.parent / 'TASK_FAILED'))}
  exit 12
fi
export MEDAI_CHECKPOINT_ROOT={shlex.quote(str(checkpoint_root))}
export NNUNETV2_PREDICT_EXECUTABLE={shlex.quote(str(nnunet_predict_executable))}
export MEDAI_NNUNETV2_PREDICT={shlex.quote(str(nnunet_predict_executable))}
export nnUNet_raw={shlex.quote(str(run_out.parent / 'nnUNet_raw'))}
export nnUNet_preprocessed={shlex.quote(str(run_out.parent / 'nnUNet_preprocessed'))}
export nnUNet_results={shlex.quote(str(checkpoint_root))}
export PATH={shlex.quote(str(nnunet_predict_executable.parent))}:$PATH
export PYTHONUNBUFFERED=1
env | sort > {shlex.quote(str(run_out.parent / 'runtime_env.txt'))}
{shlex.quote(str(python))} --version > {shlex.quote(str(run_out.parent / 'python_version.txt'))} 2>&1 || true
set +e
{command_text}
run_rc=$?
set -e
if [ "$run_rc" -eq 0 ]; then
  touch {shlex.quote(str(run_out.parent / 'TASK_COMPLETED'))}
else
  touch {shlex.quote(str(run_out.parent / 'TASK_FAILED'))}
fi
exit "$run_rc"
"""
    path.write_text(content, encoding="utf-8")


def _write_panel_sbatch(
    *,
    path: Path,
    code_root: Path,
    python: Path,
    case_manifest: Path,
    smoke_root: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    expected_commit: str,
    partition: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    allow_heavy_ct_fov: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heavy_arg = " --allow-heavy-ct-fov" if allow_heavy_ct_fov else ""
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=cads15_panel
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={path.parent / 'cads15_panel_%j.out'}
#SBATCH --error={path.parent / 'cads15_panel_%j.err'}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {shlex.quote(str(code_root))}
mkdir -p {shlex.quote(str(smoke_root / 'preflight'))}
echo "$SLURM_JOB_ID" > {shlex.quote(str(smoke_root / 'panel_job_id.txt'))}
printf '%s\\n' '{{"status":"PANEL_RUNNING","panel_job_id":"'$SLURM_JOB_ID'"}}' > {shlex.quote(str(smoke_root / 'panel_state.json'))}
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" != {shlex.quote(expected_commit)} ]; then
  echo "Commit mismatch: expected {expected_commit}, got $current_commit" >&2
  echo 12 > {shlex.quote(str(smoke_root / 'panel_return_code.txt'))}
  touch {shlex.quote(str(smoke_root / 'PANEL_FAILED'))}
  exit 12
fi
set +e
{shlex.quote(str(python))} tools/dataset_delivery/cads15_contract_audit.py \\
  --output-root {shlex.quote(str(smoke_root / 'preflight' / 'route_audit'))} \\
  --checkpoint-root {shlex.quote(str(checkpoint_root))} \\
  --nnunet-predict-executable {shlex.quote(str(nnunet_predict_executable))}
audit_rc=$?
if [ "$audit_rc" -ne 0 ]; then
  echo "$audit_rc" > {shlex.quote(str(smoke_root / 'panel_return_code.txt'))}
  touch {shlex.quote(str(smoke_root / 'PANEL_FAILED'))}
  exit "$audit_rc"
fi
{shlex.quote(str(python))} tools/dataset_delivery/cads15_smoke_panel.py \\
  --case-manifest {shlex.quote(str(case_manifest))} \\
  --output-root {shlex.quote(str(smoke_root / 'preflight'))}{heavy_arg}
panel_rc=$?
echo "$panel_rc" > {shlex.quote(str(smoke_root / 'panel_return_code.txt'))}
if [ "$panel_rc" -eq 0 ]; then
  touch {shlex.quote(str(smoke_root / 'PANEL_COMPLETED'))}
else
  touch {shlex.quote(str(smoke_root / 'PANEL_FAILED'))}
fi
exit "$panel_rc"
"""
    path.write_text(content, encoding="utf-8")


def _write_gpu_panel_sbatch(
    *,
    path: Path,
    code_root: Path,
    python: Path,
    smoke_root: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    expected_commit: str,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    timeout_sec: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=cads15_gpu_smoke
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={path.parent / 'cads15_gpu_%j.out'}
#SBATCH --error={path.parent / 'cads15_gpu_%j.err'}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {shlex.quote(str(code_root))}
mkdir -p {shlex.quote(str(smoke_root))}
echo "$SLURM_JOB_ID" > {shlex.quote(str(smoke_root / 'gpu_job_ids.txt'))}
printf '%s\\n' '{{"status":"GPU_SMOKE_RUNNING","gpu_job_id":"'$SLURM_JOB_ID'"}}' > {shlex.quote(str(smoke_root / 'gpu_state.json'))}
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" != {shlex.quote(expected_commit)} ]; then
  echo "Commit mismatch: expected {expected_commit}, got $current_commit" >&2
  touch {shlex.quote(str(smoke_root / 'GPU_SMOKE_FAILED'))}
  exit 12
fi
export MEDAI_CHECKPOINT_ROOT={shlex.quote(str(checkpoint_root))}
export NNUNETV2_PREDICT_EXECUTABLE={shlex.quote(str(nnunet_predict_executable))}
export MEDAI_NNUNETV2_PREDICT={shlex.quote(str(nnunet_predict_executable))}
export nnUNet_raw={shlex.quote(str(smoke_root / 'nnUNet_raw'))}
export nnUNet_preprocessed={shlex.quote(str(smoke_root / 'nnUNet_preprocessed'))}
export nnUNet_results={shlex.quote(str(checkpoint_root))}
export PATH={shlex.quote(str(nnunet_predict_executable.parent))}:$PATH
export PYTHONUNBUFFERED=1
set +e
{shlex.quote(str(python))} tools/dataset_delivery/cads15_smoke_launcher.py \\
  --execute-panel \\
  --smoke-root {shlex.quote(str(smoke_root))} \\
  --panel-json {shlex.quote(str(smoke_root / 'preflight' / 'cads15_smoke_case_panel.json'))} \\
  --code-root {shlex.quote(str(code_root))} \\
  --python {shlex.quote(str(python))} \\
  --checkpoint-root {shlex.quote(str(checkpoint_root))} \\
  --nnunet-predict-executable {shlex.quote(str(nnunet_predict_executable))} \\
  --timeout-sec {timeout_sec}
gpu_rc=$?
if [ "$gpu_rc" -eq 0 ]; then
  touch {shlex.quote(str(smoke_root / 'GPU_SMOKE_COMPLETED'))}
else
  touch {shlex.quote(str(smoke_root / 'GPU_SMOKE_FAILED'))}
fi
exit "$gpu_rc"
"""
    path.write_text(content, encoding="utf-8")


def prepare_cads15_smoke(
    *,
    smoke_root: Path,
    panel_json: Path,
    code_root: Path,
    python: Path,
    registry: Path,
    contract_path: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    timeout_sec: int,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
) -> dict[str, Any]:
    smoke_root.mkdir(parents=True, exist_ok=True)
    panel = _read_json(panel_json)
    if panel.get("status") != "READY_FOR_HPC_SMOKE":
        raise ValueError(f"CADS15 panel is not ready: {panel.get('status')}")
    expected_commit = _git_commit(code_root)
    rows: list[dict[str, Any]] = []
    for case in panel.get("cases") or []:
        case_id = str(case["case_id"])
        targets = [str(target) for target in case.get("targets") or []]
        models = _models_for_targets(targets, contract_path)
        if not models:
            raise ValueError(f"case {case_id} has no CADS models for targets {targets}")
        case_root = smoke_root / "cads15" / case_id
        run_out = case_root / "run_loop"
        case_csv = case_root / "selected_case_manifest.csv"
        _write_case_manifest(case_csv, case)
        command = _command_for_case(
            python=python,
            case_csv=case_csv,
            models=models,
            targets=targets,
            registry=registry,
            run_out=run_out,
            timeout_sec=timeout_sec,
            checkpoint_root=checkpoint_root,
            nnunet_predict_executable=nnunet_predict_executable,
        )
        (case_root / "command.txt").write_text(_shell_join(command) + "\n", encoding="utf-8")
        (case_root / "git_commit.txt").write_text(expected_commit + "\n", encoding="utf-8")
        sbatch_file = smoke_root / "slurm" / f"cads15_{case_id}.sbatch"
        _write_sbatch(
            path=sbatch_file,
            code_root=code_root,
            python=python,
            checkpoint_root=checkpoint_root,
            nnunet_predict_executable=nnunet_predict_executable,
            run_out=run_out,
            command=command,
            expected_commit=expected_commit,
            partition=partition,
            gres=gres,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time_limit=time_limit,
        )
        rows.append({
            "group": "cads15",
            "case_id": case_id,
            "targets": ",".join(targets),
            "models": ",".join(models),
            "run_out": str(run_out),
            "case_manifest": str(case_csv),
            "command_txt": str(case_root / "command.txt"),
            "sbatch_file": str(sbatch_file),
        })
    write_csv(
        smoke_root / "cads15_sbatch_manifest.csv",
        rows,
        ["group", "case_id", "targets", "models", "run_out", "case_manifest", "command_txt", "sbatch_file"],
    )
    summary = {
        "status": "READY",
        "read_only_pre_submit": True,
        "launch_scope": "cads15_positive_smoke_panel",
        "formal_100case_array": False,
        "smoke_root": str(smoke_root),
        "panel_json": str(panel_json),
        "expected_commit": expected_commit,
        "cases": rows,
        "target_coverage": panel.get("target_coverage", {}),
    }
    write_json(smoke_root / "cads15_prepare_summary.json", summary)
    return summary


def prepare_cads15_orchestration(
    *,
    smoke_root: Path,
    case_manifest: Path,
    code_root: Path,
    python: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    timeout_sec: int,
    panel_partition: str,
    panel_cpus_per_task: int,
    panel_mem: str,
    panel_time_limit: str,
    gpu_partition: str,
    gpu_gres: str,
    gpu_cpus_per_task: int,
    gpu_mem: str,
    gpu_time_limit: str,
    allow_heavy_ct_fov: bool,
    require_clean_tracked: bool = True,
) -> dict[str, Any]:
    smoke_root.mkdir(parents=True, exist_ok=False)
    expected_commit = _git_commit(code_root)
    if require_clean_tracked and not _tracked_worktree_clean(code_root):
        raise RuntimeError("tracked worktree is not clean; refusing CADS15 smoke submission")
    if not case_manifest.exists():
        raise FileNotFoundError(f"case manifest not found: {case_manifest}")
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint root not found: {checkpoint_root}")
    if not nnunet_predict_executable.exists() or not os.access(nnunet_predict_executable, os.X_OK):
        raise FileNotFoundError(f"nnUNet predictor not executable: {nnunet_predict_executable}")
    slurm_root = smoke_root / "slurm"
    panel_sbatch = slurm_root / "cads15_panel_prepare.sbatch"
    gpu_sbatch = slurm_root / "cads15_gpu_smoke.sbatch"
    _write_panel_sbatch(
        path=panel_sbatch,
        code_root=code_root,
        python=python,
        case_manifest=case_manifest,
        smoke_root=smoke_root,
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=nnunet_predict_executable,
        expected_commit=expected_commit,
        partition=panel_partition,
        cpus_per_task=panel_cpus_per_task,
        mem=panel_mem,
        time_limit=panel_time_limit,
        allow_heavy_ct_fov=allow_heavy_ct_fov,
    )
    _write_gpu_panel_sbatch(
        path=gpu_sbatch,
        code_root=code_root,
        python=python,
        smoke_root=smoke_root,
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=nnunet_predict_executable,
        expected_commit=expected_commit,
        partition=gpu_partition,
        gres=gpu_gres,
        cpus_per_task=gpu_cpus_per_task,
        mem=gpu_mem,
        time_limit=gpu_time_limit,
        timeout_sec=timeout_sec,
    )
    manifest = {
        "status": "NOT_SUBMITTED",
        "workflow": "cads15_smoke",
        "smoke_root": str(smoke_root),
        "expected_commit": expected_commit,
        "case_manifest": str(case_manifest),
        "python": str(python),
        "checkpoint_root": str(checkpoint_root),
        "nnunet_predict_executable": str(nnunet_predict_executable),
        "panel": {
            "sbatch_file": str(panel_sbatch),
            "job_id": "",
            "state": "NOT_SUBMITTED",
        },
        "gpu": {
            "sbatch_file": str(gpu_sbatch),
            "job_ids": [],
            "dependency": "",
            "state": "NOT_SUBMITTED",
        },
        "formal_100case_array": False,
    }
    _atomic_write_json(smoke_root / "submission_manifest.json", manifest)
    _atomic_write_json(smoke_root / "panel_state.json", {"status": "NOT_SUBMITTED"})
    _atomic_write_json(smoke_root / "gpu_state.json", {"status": "NOT_SUBMITTED"})
    return manifest


def execute_panel_smoke(
    *,
    smoke_root: Path,
    panel_json: Path,
    code_root: Path,
    python: Path,
    registry: Path,
    contract_path: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    panel = _read_json(panel_json)
    if panel.get("status") != "READY_FOR_HPC_SMOKE":
        raise ValueError(f"CADS15 panel is not ready: {panel.get('status')}")
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    cases = list(panel.get("cases") or [])
    _atomic_write_json(smoke_root / "gpu_progress.json", {
        "status": "GPU_SMOKE_RUNNING",
        "cases_completed": 0,
        "cases_total": len(cases),
    })
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        targets = [str(target) for target in case.get("targets") or []]
        models = _models_for_targets(targets, contract_path)
        case_root = smoke_root / "cads15" / case_id
        run_out = case_root / "run_loop"
        case_csv = case_root / "selected_case_manifest.csv"
        _write_case_manifest(case_csv, case)
        command = _command_for_case(
            python=python,
            case_csv=case_csv,
            models=models,
            targets=targets,
            registry=registry,
            run_out=run_out,
            timeout_sec=timeout_sec,
            checkpoint_root=checkpoint_root,
            nnunet_predict_executable=nnunet_predict_executable,
        )
        (case_root / "command.txt").write_text(_shell_join(command) + "\n", encoding="utf-8")
        (case_root / "git_commit.txt").write_text(_git_commit(code_root) + "\n", encoding="utf-8")
        proc = subprocess.run(command, cwd=code_root, text=True, check=False)
        row = {
            "group": "cads15",
            "case_id": case_id,
            "targets": ",".join(targets),
            "models": ",".join(models),
            "run_out": str(run_out),
            "case_manifest": str(case_csv),
            "command_txt": str(case_root / "command.txt"),
            "return_code": int(proc.returncode),
            "status": "success" if proc.returncode == 0 else "failed",
        }
        rows.append(row)
        if proc.returncode != 0:
            failed.append(row)
        write_csv(
            smoke_root / "cads15_gpu_manifest.csv",
            rows,
            ["group", "case_id", "targets", "models", "run_out", "case_manifest", "command_txt", "return_code", "status"],
        )
        _atomic_write_json(smoke_root / "gpu_progress.json", {
            "status": "GPU_SMOKE_RUNNING",
            "cases_completed": index,
            "cases_total": len(cases),
            "failed_cases": [row["case_id"] for row in failed],
        })
    summary = {
        "status": "GPU_SMOKE_COMPLETED" if not failed else "GPU_SMOKE_FAILED",
        "cases_total": len(cases),
        "cases_completed": len(rows),
        "failed_cases": failed,
        "rows": rows,
    }
    _atomic_write_json(smoke_root / "gpu_state.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or execute CADS15 strict-delivery smoke jobs.")
    parser.add_argument("--prepare-orchestration", action="store_true", help="Generate CPU panel and dependent GPU smoke sbatch files without reading CT data.")
    parser.add_argument("--execute-panel", action="store_true", help="Run CADS15 smoke cases from an already prepared panel inside a GPU Slurm job.")
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--panel-json", default=None, type=Path)
    parser.add_argument("--case-manifest", default=None, type=Path)
    parser.add_argument("--code-root", default=DEFAULT_CODE_ROOT, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--checkpoint-root", default=DEFAULT_HPC_CHECKPOINT_ROOT, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=DEFAULT_NNUNETV2_PREDICT, type=Path)
    parser.add_argument("--timeout-sec", default=14400, type=int)
    parser.add_argument("--partition", default="gpu", help="Legacy single-stage sbatch partition.")
    parser.add_argument("--gres", default="gpu:t4:1", help="Legacy single-stage sbatch gres.")
    parser.add_argument("--cpus-per-task", default=8, type=int, help="Legacy single-stage sbatch CPU count.")
    parser.add_argument("--mem", default="64G", help="Legacy single-stage sbatch memory.")
    parser.add_argument("--time-limit", default="06:00:00", help="Legacy single-stage sbatch time limit.")
    parser.add_argument("--panel-partition", default="shared")
    parser.add_argument("--panel-cpus-per-task", default=8, type=int)
    parser.add_argument("--panel-mem", default="32G")
    parser.add_argument("--panel-time-limit", default="02:00:00")
    parser.add_argument("--gpu-partition", default="gpu")
    parser.add_argument("--gpu-gres", default="gpu:t4:1")
    parser.add_argument("--gpu-cpus-per-task", default=8, type=int)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--gpu-time-limit", default="06:00:00")
    parser.add_argument("--allow-heavy-ct-fov", action="store_true")
    parser.add_argument("--allow-dirty-tracked", action="store_true")
    args = parser.parse_args()
    if args.prepare_orchestration:
        if not args.case_manifest:
            raise SystemExit("--case-manifest is required with --prepare-orchestration")
        summary = prepare_cads15_orchestration(
            smoke_root=args.smoke_root.resolve(),
            case_manifest=args.case_manifest.resolve(),
            code_root=args.code_root.resolve(),
            python=args.python,
            checkpoint_root=args.checkpoint_root,
            nnunet_predict_executable=args.nnunet_predict_executable,
            timeout_sec=args.timeout_sec,
            panel_partition=args.panel_partition,
            panel_cpus_per_task=args.panel_cpus_per_task,
            panel_mem=args.panel_mem,
            panel_time_limit=args.panel_time_limit,
            gpu_partition=args.gpu_partition,
            gpu_gres=args.gpu_gres,
            gpu_cpus_per_task=args.gpu_cpus_per_task,
            gpu_mem=args.gpu_mem,
            gpu_time_limit=args.gpu_time_limit,
            allow_heavy_ct_fov=bool(args.allow_heavy_ct_fov),
            require_clean_tracked=not bool(args.allow_dirty_tracked),
        )
        print(json.dumps({"status": summary["status"], "smoke_root": summary["smoke_root"]}, indent=2))
        return 0
    if args.execute_panel:
        if not args.panel_json:
            raise SystemExit("--panel-json is required with --execute-panel")
        summary = execute_panel_smoke(
            smoke_root=args.smoke_root.resolve(),
            panel_json=args.panel_json.resolve(),
            code_root=args.code_root.resolve(),
            python=args.python,
            registry=args.registry.resolve(),
            contract_path=args.contract.resolve(),
            checkpoint_root=args.checkpoint_root,
            nnunet_predict_executable=args.nnunet_predict_executable,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps({"status": summary["status"], "failed_cases": [row["case_id"] for row in summary["failed_cases"]]}, indent=2))
        return 0 if summary["status"] == "GPU_SMOKE_COMPLETED" else 1
    if not args.panel_json:
        raise SystemExit("--panel-json is required unless --prepare-orchestration is used")
    summary = prepare_cads15_smoke(
        smoke_root=args.smoke_root.resolve(),
        panel_json=args.panel_json.resolve(),
        code_root=args.code_root.resolve(),
        python=args.python,
        registry=args.registry.resolve(),
        contract_path=args.contract.resolve(),
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        timeout_sec=args.timeout_sec,
        partition=args.partition,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time_limit,
    )
    print(json.dumps({"status": summary["status"], "case_count": len(summary["cases"])}, indent=2))
    return 0 if summary["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
