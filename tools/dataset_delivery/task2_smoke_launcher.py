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

from cli_anything.medai.core.runtime_resolver import (  # noqa: E402
    DEFAULT_CANONICAL_CODE_ROOT,
    DEFAULT_HPC_CHECKPOINT_ROOT,
    DEFAULT_NNUNETV2_PREDICT,
    DEFAULT_UNEST_PYTHON,
)
from tools.dataset_delivery.delivery_lib import write_json  # noqa: E402
from tools.dataset_delivery.task2_preflight import build_preflight  # noqa: E402
from tools.dataset_delivery.task2_smoke_validator import SMOKE_SPECS, parse_groups  # noqa: E402


DEFAULT_OUTER_PYTHON = Path("/home/xhan74/envs/medical_agent/bin/python")


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _read_manifest_case(case_manifest: Path, case_id: str) -> dict[str, str]:
    with case_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if (item.get("case_id") or item.get("id")) == case_id), None)
    if row is None:
        raise ValueError(f"case_id not found in manifest: {case_id}")
    ct_path = row.get("ct_path") or row.get("image_path") or ""
    ref_dir = row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""
    if not ct_path:
        raise ValueError(f"case {case_id} is missing ct_path/image_path")
    if not ref_dir:
        raise ValueError(f"case {case_id} is missing annotation_folder/reference_mask_dir")
    return {"case_id": case_id, "ct_path": ct_path, "annotation_folder": ref_dir}


def _write_case_manifest(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path", "annotation_folder"])
        writer.writeheader()
        writer.writerow(row)


def _command_for_group(
    *,
    group: str,
    python: Path,
    case_csv: Path,
    registry: Path,
    run_out: Path,
    timeout_sec: int,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    unest_python_executable: Path,
) -> list[str]:
    spec = SMOKE_SPECS[group]
    command = [
        str(python), "run_medai_cli.py", "--json", "run-loop",
        "--case-list", str(case_csv),
        "--models", ",".join(spec["models"]),
        "--organs", ",".join(spec["targets"]),
        "--registry", str(registry),
        "--output", str(run_out),
        "--checkpoint-root", str(checkpoint_root),
        "--nnunet-predict-executable", str(nnunet_predict_executable),
        "--unest-python-executable", str(unest_python_executable),
        "--timeout-sec", str(timeout_sec),
        "--teacher-inference-mode", "hierarchical_roi" if group in {"atm", "unest"} else "full_volume",
        "--no-enable-shapekit",
        "--debug-allow-no-shapekit",
        "--no-enable-critic",
        "--strict-delivery-targets",
    ]
    if group == "atm":
        command.extend(["--strict-delivery-fov-override-organs", "airway_tree"])
    command.extend(["--log-file", str(run_out / "run_loop.log")])
    return command


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_sbatch(
    *,
    path: Path,
    group: str,
    code_root: Path,
    python: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    unest_python_executable: Path,
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
    nnunet_bin = nnunet_predict_executable.parent
    unest_bin = unest_python_executable.parent
    command_text = _shell_join(command)
    cuda_probe = ""
    if group == "unest":
        cuda_probe = (
            f"if ! {shlex.quote(str(unest_python_executable))} -c "
            f"{shlex.quote('import torch; raise SystemExit(0 if torch.cuda.is_available() else 13)')} "
            f"> {shlex.quote(str(run_out.parent / 'unest_cuda_probe.txt'))} 2>&1; then\n"
            f"  touch {shlex.quote(str(run_out.parent / 'TASK_FAILED'))}\n"
            "  exit 13\n"
            "fi\n"
        )
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=task2_{group}_smoke
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={path.parent / (group + '_%j.out')}
#SBATCH --error={path.parent / (group + '_%j.err')}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {shlex.quote(str(code_root))}
mkdir -p {shlex.quote(str(run_out))}
echo "$SLURM_JOB_ID" > {shlex.quote(str(run_out.parent / 'slurm_job_id.txt'))}
echo "$SLURM_JOB_ID" > {shlex.quote(str(run_out.parent / 'TASK_STARTED'))}
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" != {shlex.quote(expected_commit)} ]; then
  echo "Commit mismatch: expected {expected_commit}, got $current_commit" >&2
  touch {shlex.quote(str(run_out.parent / 'TASK_FAILED'))}
  exit 12
fi
export MEDAI_CHECKPOINT_ROOT={shlex.quote(str(checkpoint_root))}
export NNUNETV2_PREDICT_EXECUTABLE={shlex.quote(str(nnunet_predict_executable))}
export MEDAI_NNUNETV2_PREDICT={shlex.quote(str(nnunet_predict_executable))}
export UNEST_PYTHON_EXECUTABLE={shlex.quote(str(unest_python_executable))}
export MEDAI_UNEST_PYTHON={shlex.quote(str(unest_python_executable))}
export nnUNet_raw={shlex.quote(str(run_out.parent / 'nnUNet_raw'))}
export nnUNet_preprocessed={shlex.quote(str(run_out.parent / 'nnUNet_preprocessed'))}
export nnUNet_results={shlex.quote(str(checkpoint_root))}
export PATH={shlex.quote(str(nnunet_bin))}:{shlex.quote(str(unest_bin))}:$PATH
export PYTHONUNBUFFERED=1
env | sort > {shlex.quote(str(run_out.parent / 'runtime_env.txt'))}
{shlex.quote(str(python))} --version > {shlex.quote(str(run_out.parent / 'outer_python_version.txt'))} 2>&1 || true
{shlex.quote(str(unest_python_executable))} --version > {shlex.quote(str(run_out.parent / 'unest_python_version.txt'))} 2>&1 || true
{cuda_probe.rstrip()}
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


def prepare_smokes(
    *,
    smoke_root: Path,
    groups: list[str],
    case_manifest: Path,
    case_id: str,
    code_root: Path,
    python: Path,
    registry: Path,
    target_config: Path,
    checkpoint_root: Path,
    nnunet_predict_executable: Path,
    unest_python_executable: Path,
    canonical_code_root: Path,
    formal_mode: bool,
    timeout_sec: int,
    partition: str,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    skip_predictor_help: bool,
) -> dict[str, Any]:
    smoke_root.mkdir(parents=True, exist_ok=True)
    expected_commit = _git_commit()
    selected_case = _read_manifest_case(case_manifest, case_id)
    group_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for group in groups:
        group_root = smoke_root / group
        case_csv = group_root / "selected_case_manifest.csv"
        run_out = group_root / "run_loop"
        preflight_json = group_root / "preflight" / "task2_preflight.json"
        slurm_file = smoke_root / "slurm" / f"{group}_smoke.sbatch"
        _write_case_manifest(case_csv, selected_case)
        preflight = build_preflight(
            models=SMOKE_SPECS[group]["models"],
            case_list=case_csv,
            output_root=run_out,
            registry_path=registry,
            target_config=target_config,
            checkpoint_root_arg=checkpoint_root,
            json_output=preflight_json,
            canonical_code_root=canonical_code_root,
            formal_mode=formal_mode,
            outer_python=python,
            nnunet_predict_executable=str(nnunet_predict_executable),
            unest_python_executable=str(unest_python_executable),
            run_predictor_help=not skip_predictor_help,
        )
        command = _command_for_group(
            group=group,
            python=python,
            case_csv=case_csv,
            registry=registry,
            run_out=run_out,
            timeout_sec=timeout_sec,
            checkpoint_root=checkpoint_root,
            nnunet_predict_executable=nnunet_predict_executable,
            unest_python_executable=unest_python_executable,
        )
        (group_root / "command.txt").write_text(_shell_join(command) + "\n", encoding="utf-8")
        (group_root / "git_commit.txt").write_text(expected_commit + "\n", encoding="utf-8")
        _write_sbatch(
            path=slurm_file,
            group=group,
            code_root=code_root,
            python=python,
            checkpoint_root=checkpoint_root,
            nnunet_predict_executable=nnunet_predict_executable,
            unest_python_executable=unest_python_executable,
            run_out=run_out,
            command=command,
            expected_commit=expected_commit,
            partition=partition,
            gres=gres,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time_limit=time_limit,
        )
        row = {
            "group": group,
            "status": preflight["status"],
            "case_id": selected_case["case_id"],
            "targets": SMOKE_SPECS[group]["targets"],
            "models": SMOKE_SPECS[group]["models"],
            "preflight_json": str(preflight_json),
            "sbatch_file": str(slurm_file),
            "command_txt": str(group_root / "command.txt"),
            "run_out": str(run_out),
        }
        group_rows.append(row)
        if preflight["status"] != "READY":
            blocked.append({"group": group, "blocked_checks": preflight.get("blocked_checks", [])})
    summary = {
        "status": "READY" if not blocked else "BLOCKED",
        "read_only_pre_submit": True,
        "formal_mode": formal_mode,
        "smoke_root": str(smoke_root),
        "code_root": str(code_root),
        "expected_commit": expected_commit,
        "case_manifest": str(case_manifest),
        "selected_case": selected_case,
        "checkpoint_root": str(checkpoint_root),
        "nnunet_predict_executable": str(nnunet_predict_executable),
        "unest_python_executable": str(unest_python_executable),
        "groups": group_rows,
        "blocked": blocked,
    }
    write_json(smoke_root / "prepare_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Task 2 Teacher strict-delivery smoke sbatch files.")
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--groups", default="atm,cads,unest")
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--case-id", default="BDMAP_00000120")
    parser.add_argument("--code-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--python", default=DEFAULT_OUTER_PYTHON, type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--checkpoint-root", default=DEFAULT_HPC_CHECKPOINT_ROOT, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=DEFAULT_NNUNETV2_PREDICT, type=Path)
    parser.add_argument("--unest-python-executable", default=DEFAULT_UNEST_PYTHON, type=Path)
    parser.add_argument("--canonical-code-root", default=DEFAULT_CANONICAL_CODE_ROOT, type=Path)
    parser.add_argument("--formal-mode", action="store_true")
    parser.add_argument("--timeout-sec", default=14400, type=int)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:t4:1")
    parser.add_argument("--cpus-per-task", default=8, type=int)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time-limit", default="06:00:00")
    parser.add_argument("--skip-predictor-help", action="store_true")
    args = parser.parse_args()
    summary = prepare_smokes(
        smoke_root=args.smoke_root.resolve(),
        groups=parse_groups(args.groups),
        case_manifest=args.case_manifest.resolve(),
        case_id=args.case_id,
        code_root=args.code_root.resolve(),
        python=args.python,
        registry=args.registry.resolve(),
        target_config=args.target_config.resolve(),
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        unest_python_executable=args.unest_python_executable,
        canonical_code_root=args.canonical_code_root,
        formal_mode=bool(args.formal_mode),
        timeout_sec=args.timeout_sec,
        partition=args.partition,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time_limit,
        skip_predictor_help=bool(args.skip_predictor_help),
    )
    print(json.dumps({"status": summary["status"], "smoke_root": summary["smoke_root"]}, indent=2))
    return 0 if summary["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
