#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
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

from cli_anything.medai.core.runtime_resolver import DEFAULT_HPC_CHECKPOINT_ROOT, DEFAULT_NNUNETV2_PREDICT  # noqa: E402
from tools.dataset_delivery.cads15_smoke_launcher import _command_for_case  # noqa: E402
from tools.dataset_delivery.cads15_contract_audit import DEFAULT_CONTRACT, audit_contract, contract_targets  # noqa: E402
from tools.dataset_delivery.delivery_lib import write_csv, write_json  # noqa: E402


DEFAULT_CODE_ROOT = Path("/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent")
DEFAULT_PYTHON = Path("/home/xhan74/envs/medical_agent/bin/python")
DEFAULT_CASE_MANIFEST = Path(
    "/projects/bodymaps/users/xhan74/medical_agent/outputs/"
    "dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"
)

CADS15_MODEL_TARGETS = {
    "cads553": [
        "common_iliac_artery_left",
        "common_iliac_artery_right",
        "common_iliac_vein_left",
        "common_iliac_vein_right",
        "face",
    ],
    "cads557": [
        "blood",
        "cerebrospinal_fluid",
        "compact_bone",
        "eyeball",
        "gray_matter",
        "muscle_of_head",
        "scalp",
        "spongy_bone",
        "white_matter",
    ],
    "cads559": ["gland_structure"],
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _tracked_clean(repo: Path) -> bool:
    proc = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode == 0 and not proc.stdout.strip()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_id(row: dict[str, str], index: int) -> str:
    return row.get("case_id") or row.get("id") or f"case_{index:03d}"


def _ct_path(row: dict[str, str]) -> str:
    return row.get("ct_path") or row.get("image_path") or ""


def _annotation_folder(row: dict[str, str]) -> str:
    return row.get("annotation_folder") or row.get("reference_mask_dir") or row.get("mask_dir") or ""


def smoke_passed(smoke_root: Path) -> bool:
    status = _read_json(smoke_root / "task2_smoke_verdict.json")
    if status.get("status") == "passed":
        summary = status.get("cads15_summary") or {}
        return (
            summary.get("CADS15_SMOKE_STATUS") == "PASSED"
            and int(summary.get("TARGETS_WITH_POSITIVE_SMOKE") or 0) == 15
            and int(summary.get("TARGETS_FAILED") or 0) == 0
            and int(summary.get("STRICT_DELIVERY_FAILURE_COUNT") or 0) == 0
        )
    state = _read_json(smoke_root / "smoke_status.json")
    return state.get("status") == "PASSED"


def _task_state_path(output_root: Path, case_id: str, model: str) -> Path:
    return output_root / "tasks" / case_id / model / "task_state.json"


def _task_completed(output_root: Path, case_id: str, model: str, *, contract_hash: str) -> bool:
    state = _read_json(_task_state_path(output_root, case_id, model))
    return state.get("status") == "completed" and state.get("contract_sha256") == contract_hash


def _task_failed(output_root: Path, case_id: str, model: str) -> bool:
    state = _read_json(_task_state_path(output_root, case_id, model))
    return state.get("status") == "failed"


def _write_array_sbatch(path: Path, *, code_root: Path, python: Path, task_manifest: Path, output_root: Path) -> None:
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name=cads15_formal
#SBATCH --partition=${{GPU_PARTITION:-gpu}}
#SBATCH --gres=${{GPU_GRES:-gpu:t4:1}}
#SBATCH --cpus-per-task=${{GPU_CPUS_PER_TASK:-8}}
#SBATCH --mem=${{GPU_MEM:-64G}}
#SBATCH --time=${{GPU_TIME_LIMIT:-06:00:00}}
#SBATCH --output={output_root / 'slurm' / 'cads15_formal_%A_%a.out'}
#SBATCH --error={output_root / 'slurm' / 'cads15_formal_%A_%a.err'}

set -euo pipefail
if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi
cd {code_root}
{python} tools/dataset_delivery/cads15_formal_launcher.py \\
  --execute-task-index "$SLURM_ARRAY_TASK_ID" \\
  --task-manifest {task_manifest} \\
  --output-root {output_root}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_case_manifest(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "ct_path", "annotation_folder"])
        writer.writeheader()
        writer.writerow({
            "case_id": row["case_id"],
            "ct_path": row["ct_path"],
            "annotation_folder": row.get("annotation_folder", ""),
        })


def execute_task_index(
    *,
    task_index: int,
    task_manifest: Path,
    output_root: Path,
    code_root: Path = DEFAULT_CODE_ROOT,
    python: Path = DEFAULT_PYTHON,
    checkpoint_root: Path = DEFAULT_HPC_CHECKPOINT_ROOT,
    nnunet_predict_executable: Path = DEFAULT_NNUNETV2_PREDICT,
    timeout_sec: int = 14400,
) -> dict[str, Any]:
    rows = _read_manifest(task_manifest)
    row = next((item for item in rows if int(item.get("task_index") or -1) == task_index), None)
    if row is None:
        raise IndexError(f"task index not found: {task_index}")
    case_id = str(row["case_id"])
    model = str(row["model"])
    targets = [target for target in str(row["targets"]).split(",") if target]
    task_root = output_root / "tasks" / case_id / model
    run_out = output_root / "cases" / case_id / model / "run_loop"
    case_csv = task_root / "selected_case_manifest.csv"
    _write_case_manifest(case_csv, row)
    command = _command_for_case(
        python=python,
        case_csv=case_csv,
        models=[model],
        targets=targets,
        registry=REPO_ROOT / "configs" / "model_registry.yaml",
        run_out=run_out,
        timeout_sec=timeout_sec,
        checkpoint_root=checkpoint_root,
        nnunet_predict_executable=nnunet_predict_executable,
    )
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "command.txt").write_text(" ".join(str(part) for part in command) + "\n", encoding="utf-8")
    write_json(task_root / "task_state.json", {
        "status": "running",
        "case_id": case_id,
        "model": model,
        "targets": targets,
        "contract_sha256": row.get("contract_sha256"),
        "run_out": str(run_out),
    })
    proc = subprocess.run(command, cwd=code_root, text=True, check=False)
    state = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "case_id": case_id,
        "model": model,
        "targets": targets,
        "contract_sha256": row.get("contract_sha256"),
        "return_code": int(proc.returncode),
        "run_out": str(run_out),
    }
    write_json(task_root / "task_state.json", state)
    return state


def build_formal_plan(
    *,
    output_root: Path,
    case_manifest: Path,
    smoke_root: Path,
    code_root: Path = DEFAULT_CODE_ROOT,
    python: Path = DEFAULT_PYTHON,
    contract_path: Path = DEFAULT_CONTRACT,
    checkpoint_root: Path = DEFAULT_HPC_CHECKPOINT_ROOT,
    nnunet_predict_executable: Path = DEFAULT_NNUNETV2_PREDICT,
    expected_case_count: int = 100,
    models: list[str] | None = None,
    case_id: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    allow_dirty_tracked: bool = False,
) -> dict[str, Any]:
    models = models or list(CADS15_MODEL_TARGETS)
    contract_hash = _sha256_file(contract_path)
    checks: list[dict[str, Any]] = []
    def add_check(name: str, ok: bool, reason: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "reason": reason})

    add_check("canonical_code_root_exists", code_root.exists(), str(code_root))
    add_check("tracked_worktree_clean", allow_dirty_tracked or _tracked_clean(code_root), "git status --porcelain --untracked-files=no")
    add_check("smoke_passed", smoke_passed(smoke_root), str(smoke_root))
    add_check("case_manifest_exists", case_manifest.exists(), str(case_manifest))
    add_check("checkpoint_root_exists", checkpoint_root.exists(), str(checkpoint_root))
    add_check("predictor_executable", nnunet_predict_executable.exists() and os.access(nnunet_predict_executable, os.X_OK), str(nnunet_predict_executable))
    add_check("output_root_policy", dry_run or resume or not output_root.exists(), "new output root unless resume/dry-run")
    rows = _read_manifest(case_manifest) if case_manifest.exists() else []
    case_ids = [_case_id(row, index) for index, row in enumerate(rows)]
    add_check("case_count", len(rows) == expected_case_count, f"found {len(rows)}, expected {expected_case_count}")
    add_check("case_ids_unique", len(case_ids) == len(set(case_ids)), "")
    missing_ct = [
        _case_id(row, index)
        for index, row in enumerate(rows)
        if not _ct_path(row) or not Path(_ct_path(row)).exists()
    ]
    add_check("ct_paths_exist", not missing_ct, ",".join(missing_ct[:10]))
    route_report = audit_contract(
        contract_path=contract_path,
        checkpoint_root_arg=checkpoint_root,
        predictor_arg=str(nnunet_predict_executable),
        require_runtime_files=True,
    )
    add_check("contract_static_verified", route_report.get("status") == "READY_FOR_HPC_SMOKE", "")
    blocked_checks = [check for check in checks if not check["ok"]]
    if any(check["name"] == "output_root_policy" for check in blocked_checks):
        return {
            "status": "BLOCKED",
            "dry_run": dry_run,
            "resume": resume,
            "retry_failed": retry_failed,
            "formal_100case_array": True,
            "task_granularity": "case_x_model",
            "case_count": len(rows),
            "task_count": 0,
            "max_expected_tasks": expected_case_count * len(CADS15_MODEL_TARGETS),
            "models": models,
            "contract_sha256": contract_hash,
            "head_commit": _git_commit(code_root),
            "checks": checks,
            "blocked_checks": blocked_checks,
            "task_manifest": "",
            "sbatch_file": "",
        }
    selected_rows = rows
    if case_id:
        selected_rows = [row for index, row in enumerate(rows) if _case_id(row, index) == case_id]
    task_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        cid = _case_id(row, index)
        for model in models:
            if model not in CADS15_MODEL_TARGETS:
                continue
            if resume and _task_completed(output_root, cid, model, contract_hash=contract_hash):
                continue
            if retry_failed and not _task_failed(output_root, cid, model):
                continue
            task_rows.append({
                "task_index": len(task_rows),
                "case_id": cid,
                "model": model,
                "targets": ",".join(CADS15_MODEL_TARGETS[model]),
                "ct_path": _ct_path(row),
                "annotation_folder": _annotation_folder(row),
                "contract_sha256": contract_hash,
                "output_root": str(output_root),
                "status": "planned",
            })
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "cads15_model_task_manifest.csv",
        task_rows,
        ["task_index", "case_id", "model", "targets", "ct_path", "annotation_folder", "contract_sha256", "output_root", "status"],
    )
    _write_array_sbatch(
        output_root / "slurm" / "cads15_formal_array.sbatch",
        code_root=code_root,
        python=python,
        task_manifest=output_root / "cads15_model_task_manifest.csv",
        output_root=output_root,
    )
    summary = {
        "status": "READY" if not blocked_checks else "BLOCKED",
        "dry_run": dry_run,
        "resume": resume,
        "retry_failed": retry_failed,
        "formal_100case_array": True,
        "task_granularity": "case_x_model",
        "case_count": len(rows),
        "task_count": len(task_rows),
        "max_expected_tasks": expected_case_count * len(CADS15_MODEL_TARGETS),
        "models": models,
        "contract_sha256": contract_hash,
        "head_commit": _git_commit(code_root),
        "checks": checks,
        "blocked_checks": blocked_checks,
        "task_manifest": str(output_root / "cads15_model_task_manifest.csv"),
        "sbatch_file": str(output_root / "slurm" / "cads15_formal_array.sbatch"),
    }
    write_json(output_root / "cads15_formal_preflight.json", summary)
    write_json(output_root / "submission_manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CADS15 formal 100-case case-by-model launch manifests.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-manifest", default=DEFAULT_CASE_MANIFEST, type=Path)
    parser.add_argument("--smoke-root", required=False, type=Path)
    parser.add_argument("--code-root", default=DEFAULT_CODE_ROOT, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--checkpoint-root", default=DEFAULT_HPC_CHECKPOINT_ROOT, type=Path)
    parser.add_argument("--nnunet-predict-executable", default=DEFAULT_NNUNETV2_PREDICT, type=Path)
    parser.add_argument("--expected-case-count", default=100, type=int)
    parser.add_argument("--models", default="")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty-tracked", action="store_true")
    parser.add_argument("--execute-task-index", default=None)
    parser.add_argument("--task-manifest", default=None, type=Path)
    parser.add_argument("--timeout-sec", default=14400, type=int)
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
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps({"status": state["status"], "case_id": state["case_id"], "model": state["model"]}, indent=2))
        return 0 if state["status"] == "completed" else 1
    smoke_root = args.smoke_root
    if smoke_root is None:
        raise SystemExit("--smoke-root is required unless --execute-task-index is used")
    models = [item.strip() for item in args.models.replace(";", ",").split(",") if item.strip()] or None
    summary = build_formal_plan(
        output_root=args.output_root.resolve(),
        case_manifest=args.case_manifest.resolve(),
        smoke_root=smoke_root.resolve(),
        code_root=args.code_root.resolve(),
        python=args.python,
        contract_path=args.contract.resolve(),
        checkpoint_root=args.checkpoint_root,
        nnunet_predict_executable=args.nnunet_predict_executable,
        expected_case_count=args.expected_case_count,
        models=models,
        case_id=args.case_id or None,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        dry_run=bool(args.dry_run),
        allow_dirty_tracked=bool(args.allow_dirty_tracked),
    )
    print(json.dumps({"status": summary["status"], "task_count": summary["task_count"]}, indent=2))
    return 0 if summary["status"] == "READY" or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
