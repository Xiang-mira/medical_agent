#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CADS15 strict-delivery smoke sbatch files from a positive panel.")
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--panel-json", required=True, type=Path)
    parser.add_argument("--code-root", default=DEFAULT_CODE_ROOT, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--checkpoint-root", default=DEFAULT_HPC_CHECKPOINT_ROOT, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=DEFAULT_NNUNETV2_PREDICT, type=Path)
    parser.add_argument("--timeout-sec", default=14400, type=int)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:t4:1")
    parser.add_argument("--cpus-per-task", default=8, type=int)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time-limit", default="06:00:00")
    args = parser.parse_args()
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
