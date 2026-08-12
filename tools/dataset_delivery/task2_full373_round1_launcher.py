#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "agent-harness"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.medai.core.model_registry import candidate_models_for_organs, load_registry  # noqa: E402
from tools.dataset_delivery.delivery_lib import read_csv_rows, utc_now, write_csv, write_json  # noqa: E402


FULL373_GROUP = "full373"
FULL373_ROOT_NAME = "full_373_multiteacher_round1"
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


def _norm(text: str) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


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


def _write_array_sbatch(path: Path, *, python: Path, output_root: Path, task_manifest: Path) -> None:
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
    rows = []
    for index, row in enumerate(input_rows):
        case_id = str(row.get("case_id") or row.get("id") or f"case_{index:03d}").strip()
        rows.append(
            {
                "task_index": index,
                "case_id": case_id,
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
    batch_key = _sha("|".join(row["case_id"] for row in rows), 12)
    task_manifest = slurm_root / f"full373_task_manifest_{batch_key}.csv"
    write_csv(task_manifest, rows, list(rows[0].keys()) if rows else ["task_index", "case_id"])
    sbatch = slurm_root / "full373_multiteacher_array.sbatch"
    _write_array_sbatch(sbatch, python=python, output_root=output_root, task_manifest=task_manifest)
    summary = {
        "status": "READY",
        "stage": "full_373_multiteacher_round1_submission_manifest",
        "created_at": utc_now(),
        "scope": str(output_root / "full_round1_submission_scope.json"),
        "scope_status": scope,
        "task_count": len(rows),
        "scientific_task_unit": "case_id x canonical_target x eligible_teacher",
        "scheduler_array_unit": "case_id full run-loop shard",
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


def execute_task_index(task_index: int, task_manifest: Path, output_root: Path) -> dict[str, Any]:
    rows = read_csv_rows(task_manifest)
    row = next((item for item in rows if int(item.get("task_index") or -1) == int(task_index)), None)
    if row is None:
        raise IndexError(f"task index not found: {task_index}")
    case_id = str(row["case_id"])
    case_csv = output_root / "case_manifests" / f"{case_id}.csv"
    _write_case_manifest(
        case_csv,
        {
            "case_id": case_id,
            "ct_path": str(row.get("ct_path") or ""),
            "annotation_folder": str(row.get("annotation_folder") or ""),
        },
    )
    case_output = output_root / "run_loop_cases" / case_id
    endpoint = os.getenv("LABELCRITIC_BASE_URL", "http://localhost")
    port = os.getenv("LABELCRITIC_PORT", "8000")
    command = [
        str(row.get("python") or sys.executable),
        "run_medai_cli.py",
        "--json",
        "run-loop",
        "--case-list", str(case_csv),
        "--models", "",
        "--organs", "student_373",
        "--target-config", str(row.get("target_config") or REPO_ROOT / "configs/student_3d_prompt_target_organs.json"),
        "--registry", str(row.get("registry_path") or REPO_ROOT / "configs/model_registry.yaml"),
        "--output", str(case_output),
        "--checkpoint-root", str(row.get("checkpoint_root") or ""),
        "--nnunet-predict-executable", str(row.get("nnunet_predict_executable") or ""),
        "--unest-python-executable", str(row.get("unest_python_executable") or ""),
        "--enable-shapekit",
        "--enable-critic",
        "--critic-backend", "labelcritic",
        "--critic-base-url", endpoint,
        "--critic-port", str(port),
        "--critic-vlm-model", os.getenv("LABELCRITIC_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct-AWQ"),
        "--teacher-inference-mode", os.getenv("MEDAI_TEACHER_INFERENCE_MODE", "hierarchical_roi"),
        "--roi-margin-mm", os.getenv("MEDAI_ROI_MARGIN_MM", "20"),
        "--timeout-sec", os.getenv("MEDAI_INFER_TIMEOUT_SEC", "3600"),
        "--no-use-annotation-folder-reference",
        "--log-file", str(case_output / "run_loop.log"),
    ]
    env = os.environ.copy()
    env.update({"RUNTIME_NO_GIT": "1", "SKIP_GIT_SYNC": "1", "GIT_TERMINAL_PROMPT": "0"})
    for key in ("DISPLAY", "GITHUB_TOKEN", "GH_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"):
        env.pop(key, None)
    state = {"status": "RUNNING", "case_id": case_id, "task_index": task_index, "command": command, "started_at": utc_now()}
    write_json(output_root / "task_states" / f"{case_id}.json", state)
    proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    final = {
        **state,
        "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
        "return_code": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "finished_at": utc_now(),
        "case_output": str(case_output),
    }
    write_json(output_root / "task_states" / f"{case_id}.json", final)
    return final


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
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.execute_task_index is not None:
        if not args.task_manifest:
            raise SystemExit("--task-manifest is required with --execute-task-index")
        print(json.dumps(execute_task_index(args.execute_task_index, args.task_manifest.resolve(), output_root), indent=2))
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
        cache_roots=[path.resolve() for path in args.cache_root],
    )
    print(json.dumps({"status": summary["status"], "task_count": summary["task_count"], "scope": summary["scope"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
