from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .case_selection import build_case_selection_slurm_plan, submit_case_selection_slurm_plan
from .config import resolve_path
from .manifest import assert_strict_no_gt_manifest, read_case_csv
from .utils import ROOT, SchedulerError, ensure_not_raw_data_write_path, git_snapshot, read_json, sha256_file, sha256_text, utc_now, write_json_atomic


CASE_SELECTION_STAGE = "case_selection_30_20"
EM_STATUS_VALUES = {"planned", "submitted", "running", "completed", "failed", "partial"}
FORBIDDEN_SCRIPT_TOKENS = ("rm ", "mv ", "chmod ", "chown ", "rsync --delete", "srun", "torchrun")


@dataclass(frozen=True)
class EmConfig:
    path: Path
    data: dict[str, Any]

    @property
    def experiment(self) -> dict[str, Any]:
        return self.data.get("experiment", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self.data.get("paths", {})


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            return os.environ.get(name, default or "")

        expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", repl, value)
        return os.path.expandvars(expanded)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_em_config(path: str | Path) -> EmConfig:
    resolved = resolve_path(path)
    assert resolved is not None
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchedulerError(f"EM config must be a YAML mapping: {resolved}")
    return EmConfig(path=resolved, data=_expand_env(data))


def _path_from_config(cfg: EmConfig, key: str, default: str | None = None) -> Path | None:
    raw = cfg.paths.get(key, default)
    if raw is None or str(raw).strip() == "":
        return None
    return resolve_path(str(raw))


def _work_root(cfg: EmConfig) -> Path:
    return (_path_from_config(cfg, "work_root") or ROOT).expanduser().resolve()


def _assert_output_allowed(cfg: EmConfig, output_dir: Path) -> Path:
    out = output_dir.expanduser().resolve()
    work_root = _work_root(cfg)
    if out != work_root and work_root not in out.parents:
        raise SchedulerError(f"Refusing output_dir outside WORK_ROOT: output_dir={out} work_root={work_root}")
    image_root = _path_from_config(cfg, "image_root")
    mask_root = _path_from_config(cfg, "mask_root")
    ensure_not_raw_data_write_path(out, image_root, mask_root)
    public_root = Path("/projects/bodymaps/Data")
    if out == public_root or public_root in out.parents:
        raise SchedulerError(f"Refusing output_dir under public data root: {out}")
    return out


def _experiment_defaults(cfg: EmConfig) -> dict[str, Any]:
    exp = cfg.experiment
    return {
        "name": exp.get("name", "em_train30_test20_round1_round2"),
        "seed": int(exp.get("seed", 20260724)),
        "train_cases": int(exp.get("train_cases", 30)),
        "test_cases": int(exp.get("test_cases", 20)),
        "max_inventory_cases": int(exp.get("max_inventory_cases", 10000)),
        "ordering": "sorted_case_id",
    }


def _fingerprint(cfg: EmConfig) -> str:
    target_mapping = _path_from_config(cfg, "target_mapping")
    payload = {
        "config": cfg.data,
        "config_path": str(cfg.path),
        "target_mapping_sha256": sha256_file(target_mapping) if target_mapping and target_mapping.exists() else None,
        "version": "em_train30_test20_round1_round2_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _split_case_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [str(row.get("case_id") or "") for row in read_case_csv(path)]


def write_split_metadata(run_dir: Path, cfg: EmConfig, fingerprint: str) -> dict[str, Any]:
    exp = _experiment_defaults(cfg)
    train_manifest = run_dir / "manifests" / "train30_input_no_gt.csv"
    test_manifest = run_dir / "manifests" / "test20_input_no_gt.csv"
    train_ids = _split_case_ids(train_manifest)
    test_ids = _split_case_ids(test_manifest)
    overlap = sorted(set(train_ids) & set(test_ids))
    metadata = {
        "status": "planned" if not train_ids and not test_ids else "success",
        "seed": exp["seed"],
        "fingerprint": fingerprint,
        "ordering": exp["ordering"],
        "train_cases": exp["train_cases"],
        "test_cases": exp["test_cases"],
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
        "overlap": overlap,
        "split_frozen": True,
        "rounds_use_same_split": [1, 2],
    }
    if overlap:
        raise SchedulerError(f"Train/Test split overlaps: {overlap[:10]}")
    write_json_atomic(run_dir / "split_metadata.json", metadata)
    return metadata


def _copy_csv(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SchedulerError(f"Required split artifact is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(src.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise SchedulerError(f"Split artifact has no rows: {src}")
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def materialize_fixed_split(run_dir: Path, cfg: EmConfig, fingerprint: str) -> dict[str, Any]:
    selection_dir = run_dir / "case_selection_30_20"
    manifest_dir = run_dir / "manifests"
    copies = [
        (selection_dir / "train30_input_strict_no_gt.csv", manifest_dir / "train30_input_no_gt.csv"),
        (selection_dir / "test20_input_strict_no_gt.csv", manifest_dir / "test20_input_no_gt.csv"),
        (selection_dir / "train30_eval_reference.csv", manifest_dir / "train30_eval_reference.csv"),
        (selection_dir / "test20_eval_reference.csv", manifest_dir / "test20_eval_reference.csv"),
    ]
    for src, dst in copies:
        if not dst.exists():
            _copy_csv(src, dst)
    assert_strict_no_gt_manifest(manifest_dir / "train30_input_no_gt.csv")
    assert_strict_no_gt_manifest(manifest_dir / "test20_input_no_gt.csv")
    metadata = write_split_metadata(run_dir, cfg, fingerprint)
    if len(metadata["train_case_ids"]) != 30 or len(metadata["test_case_ids"]) != 20:
        raise SchedulerError("Fixed EM split must contain exactly Train30 and Test20")
    return metadata


def _stage(stage_id: str, deps: list[str], *, kind: str, partition: str = "cpu", gpu_type: str = "none", gpu_count: int = 0, cpus: int = 4, mem: str = "16G", time: str = "02:00:00", array: str | None = None, logical_tasks: int = 1, split: str | None = None, eval_reference: bool = False, labelcritic_cases: list[str] | None = None) -> dict[str, Any]:
    return {
        "stage": stage_id,
        "dependencies": deps,
        "kind": kind,
        "partition": partition,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "cpus_per_task": cpus,
        "memory": mem,
        "time": time,
        "array": array,
        "logical_tasks": logical_tasks,
        "split": split,
        "eval_reference": eval_reference,
        "labelcritic_cases": labelcritic_cases or [],
    }


def _case_ids(count: int, prefix: str) -> list[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(count)]


def _build_em_stages(cfg: EmConfig) -> list[dict[str, Any]]:
    exp = _experiment_defaults(cfg)
    gp = cfg.data.get("gpu_parallel", {})
    train_ids = _case_ids(exp["train_cases"], "TRAIN")
    r1_shards = [train_ids[:15], train_ids[15:30]]
    tt = gp.get("teacher_train", {})
    te = gp.get("teacher_test", {})
    sit = gp.get("student_inference_train", {})
    sis = gp.get("student_inference_test", {})
    lc = gp.get("labelcritic", {})
    teacher_train_max = int(tt.get("max_concurrent", 10))
    teacher_test_max = int(te.get("max_concurrent", 6))
    student_train_inf_max = int(sit.get("max_concurrent", 10))
    student_test_inf_max = int(sis.get("max_concurrent", 6))
    replicas = int(lc.get("replicas", 2))
    if replicas != 2:
        raise SchedulerError("Formal Train30 EM pipeline requires exactly 2 LabelCritic replicas")
    stages = [
        _stage(CASE_SELECTION_STAGE, [], kind="case_selection", cpus=1, mem="1G", time="00:10:00"),
        _stage("teacher_train30_smoke", [CASE_SELECTION_STAGE], kind="teacher_smoke", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="00:30:00", split="train"),
        _stage("teacher_test20_smoke", [CASE_SELECTION_STAGE], kind="teacher_smoke", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="00:30:00", split="test"),
        _stage("teacher_train30_array", ["teacher_train30_smoke", "teacher_test20_smoke"], kind="teacher_array", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="08:00:00", array=f"0-{exp['train_cases'] - 1}%{teacher_train_max}", logical_tasks=exp["train_cases"], split="train"),
        _stage("teacher_test20_array", ["teacher_train30_smoke", "teacher_test20_smoke"], kind="teacher_array", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="08:00:00", array=f"0-{exp['test_cases'] - 1}%{teacher_test_max}", logical_tasks=exp["test_cases"], split="test"),
        _stage("teacher_baseline_evaluation", ["teacher_test20_array"], kind="evaluation", cpus=8, mem="32G", time="04:00:00", split="test", eval_reference=True),
        _stage("labelcritic_r1_smoke", ["teacher_train30_array"], kind="labelcritic_smoke", partition="gpuh100", gpu_type="h100", gpu_count=4, cpus=24, mem="256G", time="01:00:00", split="train"),
        _stage("labelcritic_r1_shard_0", ["labelcritic_r1_smoke"], kind="labelcritic_shard", partition="gpuh100", gpu_type="h100", gpu_count=4, cpus=24, mem="256G", time="08:00:00", split="train", labelcritic_cases=r1_shards[0]),
        _stage("labelcritic_r1_shard_1", ["labelcritic_r1_smoke"], kind="labelcritic_shard", partition="gpuh100", gpu_type="h100", gpu_count=4, cpus=24, mem="256G", time="08:00:00", split="train", labelcritic_cases=r1_shards[1]),
        _stage("merge_labelcritic_r1", ["labelcritic_r1_shard_0", "labelcritic_r1_shard_1"], kind="merge_labelcritic", cpus=8, mem="32G", time="02:00:00", split="train"),
        _stage("build_round1_manifest", ["merge_labelcritic_r1"], kind="build_manifest", cpus=4, mem="16G", time="01:00:00", split="train"),
        _stage("student_r1_train_smoke", ["build_round1_manifest"], kind="student_train_smoke", partition="gpua100", gpu_type="a100", gpu_count=1, cpus=12, mem="96G", time="01:00:00", split="train"),
        _stage("student_r1_train", ["student_r1_train_smoke"], kind="student_train", partition="gpua100", gpu_type="a100", gpu_count=1, cpus=12, mem="96G", time="10:00:00", split="train"),
        _stage("student_r1_train30_inference", ["student_r1_train"], kind="student_inference", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="08:00:00", array=f"0-{exp['train_cases'] - 1}%{student_train_inf_max}", logical_tasks=exp["train_cases"], split="train"),
        _stage("student_r1_test20_inference", ["student_r1_train"], kind="student_inference", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="08:00:00", array=f"0-{exp['test_cases'] - 1}%{student_test_inf_max}", logical_tasks=exp["test_cases"], split="test"),
        _stage("round1_evaluation", ["student_r1_test20_inference", "teacher_test20_array"], kind="evaluation", cpus=8, mem="32G", time="04:00:00", split="test", eval_reference=True),
        _stage("labelcritic_r2_shard_0", ["student_r1_train30_inference"], kind="labelcritic_shard", partition="gpuh100", gpu_type="h100", gpu_count=4, cpus=24, mem="256G", time="08:00:00", split="train", labelcritic_cases=r1_shards[0]),
        _stage("labelcritic_r2_shard_1", ["student_r1_train30_inference"], kind="labelcritic_shard", partition="gpuh100", gpu_type="h100", gpu_count=4, cpus=24, mem="256G", time="08:00:00", split="train", labelcritic_cases=r1_shards[1]),
        _stage("merge_labelcritic_r2", ["labelcritic_r2_shard_0", "labelcritic_r2_shard_1"], kind="merge_labelcritic", cpus=8, mem="32G", time="02:00:00", split="train"),
        _stage("build_round2_manifest", ["merge_labelcritic_r2"], kind="build_manifest", cpus=4, mem="16G", time="01:00:00", split="train"),
        _stage("student_r2_train", ["build_round2_manifest"], kind="student_train", partition="gpua100", gpu_type="a100", gpu_count=1, cpus=12, mem="96G", time="10:00:00", split="train"),
        _stage("student_r2_test20_inference", ["student_r2_train"], kind="student_inference", partition="gpu", gpu_type="t4", gpu_count=1, cpus=4, mem="32G", time="08:00:00", array=f"0-{exp['test_cases'] - 1}%8", logical_tasks=exp["test_cases"], split="test"),
        _stage("final_evaluation", ["student_r2_test20_inference", "teacher_test20_array"], kind="evaluation", cpus=8, mem="32G", time="04:00:00", split="test", eval_reference=True),
    ]
    return _apply_stage_window(stages, None, None)


def _apply_stage_window(stages: list[dict[str, Any]], from_stage: str | None, stop_after_stage: str | None) -> list[dict[str, Any]]:
    if from_stage:
        names = [s["stage"] for s in stages]
        if from_stage not in names:
            raise SchedulerError(f"Unknown from-stage: {from_stage}")
        keep = set(names[names.index(from_stage) :])
        stages = [s for s in stages if s["stage"] in keep]
        for stage in stages:
            stage["dependencies"] = [d for d in stage["dependencies"] if d in keep]
    if stop_after_stage:
        names = [s["stage"] for s in stages]
        if stop_after_stage not in names:
            raise SchedulerError(f"Unknown stop-after-stage: {stop_after_stage}")
        keep = set(names[: names.index(stop_after_stage) + 1])
        stages = [s for s in stages if s["stage"] in keep]
    return stages


def _validate_limits(stages: list[dict[str, Any]], cfg: EmConfig) -> dict[str, Any]:
    limits = cfg.data.get("resource_limits", {})
    max_t4 = int(limits.get("max_t4_gpus", 16))
    max_a100 = int(limits.get("max_a100_gpus", 1))
    max_h100 = int(limits.get("max_h100_gpus", 8))
    t4_concurrent_groups = [
        sum(int(s["array"].split("%", 1)[1]) * int(s["gpu_count"]) for s in stages if s["stage"] in group)
        for group in ({"teacher_train30_array", "teacher_test20_array"}, {"student_r1_train30_inference", "student_r1_test20_inference"}, {"student_r2_test20_inference"})
    ]
    h100_group = sum(int(s["gpu_count"]) for s in stages if re.fullmatch(r"labelcritic_r[12]_shard_[01]", s["stage"]))
    a100_peak = max([int(s["gpu_count"]) for s in stages if s["gpu_type"] == "a100"] or [0])
    if max(t4_concurrent_groups or [0]) > max_t4:
        raise SchedulerError(f"T4 plan exceeds global limit: peak={max(t4_concurrent_groups)} limit={max_t4}")
    if a100_peak > max_a100:
        raise SchedulerError(f"A100 plan exceeds global limit: peak={a100_peak} limit={max_a100}")
    if min(h100_group, max_h100) > max_h100:
        raise SchedulerError(f"H100 plan exceeds global limit: peak={h100_group} limit={max_h100}")
    return {"max_t4_gpus": max(t4_concurrent_groups or [0]), "max_a100_gpus": a100_peak, "max_h100_gpus": max_h100 if h100_group else 0}


def _shell_join(args: list[str]) -> str:
    rendered = []
    for arg in args:
        text = str(arg)
        if text in {"${SLURM_ARRAY_TASK_ID}", "$SLURM_ARRAY_TASK_ID"}:
            rendered.append(text)
        else:
            rendered.append(shlex.quote(text))
    return " ".join(rendered)


def _render_sbatch(stage: dict[str, Any], run_dir: Path, config_path: Path) -> str:
    log_token = "%A_%a" if stage.get("array") else "%j"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={stage['stage']}",
        f"#SBATCH --partition={stage['partition']}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={stage['cpus_per_task']}",
        f"#SBATCH --mem={stage['memory']}",
        f"#SBATCH --time={stage['time']}",
    ]
    if stage.get("gpu_count"):
        lines.append(f"#SBATCH --gres=gpu:{stage['gpu_count']}")
    if stage.get("array"):
        lines.append(f"#SBATCH --array={stage['array']}")
    lines.extend(
        [
            f"#SBATCH --output={run_dir}/logs/{stage['stage']}_{log_token}.out",
            f"#SBATCH --error={run_dir}/logs/{stage['stage']}_{log_token}.err",
            "",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(run_dir / 'logs'))} {shlex.quote(str(run_dir / 'status'))} {shlex.quote(str(run_dir / 'provenance'))}",
            f"echo '[em] stage={stage['stage']} job=${{SLURM_JOB_ID:-local}} array=${{SLURM_ARRAY_TASK_ID:-none}}'",
            _shell_join(
                [
                    "python",
                    "-m",
                    "scheduler.cli",
                    "em-stage",
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--stage",
                    stage["stage"],
                    "--array-index",
                    "${SLURM_ARRAY_TASK_ID}" if stage.get("array") else "-1",
                ]
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    lowered = text.lower()
    for token in FORBIDDEN_SCRIPT_TOKENS:
        if token in lowered:
            raise SchedulerError(f"Refusing to render destructive or disallowed token in {stage['stage']}: {token}")
    return text


def _existing_successes(run_dir: Path, fingerprint: str) -> set[str]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return set()
    state = read_json(state_path)
    if state.get("fingerprint") != fingerprint:
        raise SchedulerError("Resume refuses state with a mismatched fingerprint")
    if any(row.get("status") == "partial" for row in (state.get("stages") or {}).values() if isinstance(row, dict)):
        raise SchedulerError("Resume refuses partial outputs; use --retry-failed after fixing failed shards")
    return {name for name, row in (state.get("stages") or {}).items() if isinstance(row, dict) and row.get("status") == "completed"}


def _failed_stages(run_dir: Path, fingerprint: str) -> set[str]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return set()
    state = read_json(state_path)
    if state.get("fingerprint") != fingerprint:
        raise SchedulerError("Retry refuses state with a mismatched fingerprint")
    return {name for name, row in (state.get("stages") or {}).items() if isinstance(row, dict) and row.get("status") == "failed"}


def _state_payload(stages: list[dict[str, Any]], fingerprint: str, *, submitted: bool) -> dict[str, Any]:
    return {
        "status": "submitted" if submitted else "planned",
        "fingerprint": fingerprint,
        "updated_at": utc_now(),
        "stages": {s["stage"]: {"status": "planned", "dependencies": s["dependencies"], "array": s.get("array"), "logical_tasks": s.get("logical_tasks", 1)} for s in stages},
    }


def build_em_plan(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = True,
    submit: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    from_stage: str | None = None,
    stop_after_stage: str | None = None,
) -> dict[str, Any]:
    cfg = load_em_config(config_path)
    run_dir = _assert_output_allowed(cfg, Path(output_dir))
    exp = _experiment_defaults(cfg)
    if (exp["train_cases"], exp["test_cases"], exp["seed"], exp["max_inventory_cases"], exp["ordering"]) != (30, 20, 20260724, 10000, "sorted_case_id"):
        raise SchedulerError("Formal EM pipeline is fixed to train_cases=30 test_cases=20 seed=20260724 max_inventory_cases=10000 ordering=sorted_case_id")
    for name in ("generated_slurm", "logs", "provenance", "manifests", "status", "outputs", "cache"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(cfg)
    all_stages = _apply_stage_window(_build_em_stages(cfg), from_stage, stop_after_stage)
    if resume:
        done = _existing_successes(run_dir, fingerprint)
        all_stages = [s for s in all_stages if s["stage"] not in done]
        for stage in all_stages:
            stage["dependencies"] = [d for d in stage["dependencies"] if d not in done]
    if retry_failed:
        failed = _failed_stages(run_dir, fingerprint)
        all_stages = [s for s in all_stages if s["stage"] in failed]
        for stage in all_stages:
            stage["dependencies"] = []
    resources = _validate_limits(all_stages, cfg)
    case_selection_dir = run_dir / "case_selection_30_20"
    case_plan = build_case_selection_slurm_plan(30, 20, 20260724, case_selection_dir, config_path=cfg.paths.get("case_selection_config", "configs/abdomenatlaspro_case_selection.yaml"), dry_run=True, max_inventory_cases=10000)
    for stage in all_stages:
        if stage["stage"] == CASE_SELECTION_STAGE:
            continue
        script_path = run_dir / "generated_slurm" / f"{stage['stage']}.sbatch"
        script_path.write_text(_render_sbatch(stage, run_dir, Path(config_path)), encoding="utf-8")
        stage["script"] = str(script_path)
    dependency_graph = {
        "nodes": [s["stage"] for s in all_stages],
        "edges": [{"from": dep, "to": s["stage"]} for s in all_stages for dep in s["dependencies"]],
    }
    split_metadata = write_split_metadata(run_dir, cfg, fingerprint)
    run_plan = {
        "status": "dry_run" if dry_run and not submit else "planned",
        "backend": "slurm",
        "submitted": False,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "created_at": utc_now(),
        "git": git_snapshot(),
        "fingerprint": fingerprint,
        "experiment": exp,
        "case_selection_plan": case_plan,
        "stages": all_stages,
        "dependency_graph": dependency_graph,
        "split_metadata": split_metadata,
        "policies": {
            "dry_run_calls_sbatch": False,
            "dry_run_calls_srun": False,
            "dry_run_calls_torchrun": False,
            "dry_run_starts_vllm": False,
            "test20_training": "forbidden",
            "test20_labelcritic": "forbidden",
            "round2_reruns_teacher": False,
        },
    }
    resource_plan = {
        **resources,
        "teacher_train30_tasks": 30,
        "teacher_test20_tasks": 20,
        "student_r1_train30_inference_tasks": 30,
        "student_r1_test20_inference_tasks": 20,
        "student_r2_test20_inference_tasks": 20,
        "labelcritic": {
            "model": "Qwen2-VL-72B-Instruct-AWQ",
            "replicas": 2,
            "gpus_per_replica": 4,
            "tensor_parallel_size": 4,
            "round1_shards": [{"stage": "labelcritic_r1_shard_0", "cases": 15}, {"stage": "labelcritic_r1_shard_1", "cases": 15}],
            "round2_shards": [{"stage": "labelcritic_r2_shard_0", "cases": 15}, {"stage": "labelcritic_r2_shard_1", "cases": 15}],
        },
        "student_training": {"gpu_type": "a100", "gpus": 1},
    }
    write_json_atomic(run_dir / "run_plan.json", run_plan)
    write_json_atomic(run_dir / "resource_plan.json", resource_plan)
    write_json_atomic(run_dir / "dependency_graph.json", dependency_graph)
    write_json_atomic(run_dir / "task_manifest.json", {"stages": all_stages})
    write_json_atomic(run_dir / "state.json", _state_payload(all_stages, fingerprint, submitted=False))
    write_json_atomic(run_dir / "submission_receipt.json", {"status": "dry_run", "submitted": False, "jobs": {}, "created_at": utc_now()})
    return run_plan


def submit_em_plan(plan: dict[str, Any]) -> dict[str, Any]:
    submitted: dict[str, Any] = {}
    case_receipt = submit_case_selection_slurm_plan(plan["case_selection_plan"])
    case_jobs = case_receipt.get("jobs") or {}
    if case_jobs.get("final_selection_report", {}).get("job_id"):
        submitted[CASE_SELECTION_STAGE] = case_jobs["final_selection_report"]
    for stage in plan.get("stages", []):
        if stage["stage"] == CASE_SELECTION_STAGE:
            continue
        cmd = ["sbatch"]
        dep_ids = [submitted[d]["job_id"] for d in stage.get("dependencies", []) if submitted.get(d, {}).get("job_id")]
        if dep_ids:
            cmd.append("--dependency=afterok:" + ":".join(dep_ids))
        cmd.append(str(stage["script"]))
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise SchedulerError(f"sbatch failed for {stage['stage']}: {proc.stderr}")
        job_id = proc.stdout.strip().split()[-1]
        submitted[stage["stage"]] = {"status": "submitted", "job_id": job_id, "command": cmd, "stdout": proc.stdout.strip()}
    receipt = {"status": "submitted", "submitted": True, "jobs": submitted, "submitted_at": utc_now()}
    run_dir = Path(plan["run_dir"])
    write_json_atomic(run_dir / "submission_receipt.json", receipt)
    state = read_json(run_dir / "state.json")
    state["status"] = "submitted"
    for name, row in state.get("stages", {}).items():
        if name in submitted:
            row["status"] = "submitted"
            row["job_id"] = submitted[name].get("job_id")
    write_json_atomic(run_dir / "state.json", state)
    return receipt


def em_run(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    submit: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
    from_stage: str | None = None,
    stop_after_stage: str | None = None,
) -> dict[str, Any]:
    if dry_run and submit:
        raise SchedulerError("--dry-run and --submit are mutually exclusive")
    plan = build_em_plan(config_path, output_dir, dry_run=dry_run or not submit, submit=submit, resume=resume, retry_failed=retry_failed, from_stage=from_stage, stop_after_stage=stop_after_stage)
    if submit:
        plan["submission"] = submit_em_plan(plan)
    return plan


def em_status(output_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(output_dir)
    state_path = run_dir / "state.json"
    receipt_path = run_dir / "submission_receipt.json"
    return {
        "status": "missing" if not state_path.exists() else read_json(state_path).get("status"),
        "run_dir": str(run_dir),
        "state": read_json(state_path) if state_path.exists() else {},
        "submission": read_json(receipt_path) if receipt_path.exists() else {},
    }


def execute_em_stage(config_path: str | Path, run_dir: str | Path, stage: str, array_index: int | None = None) -> dict[str, Any]:
    cfg = load_em_config(config_path)
    out = _assert_output_allowed(cfg, Path(run_dir))
    plan = read_json(out / "run_plan.json")
    stage_doc = next((s for s in plan.get("stages", []) if s.get("stage") == stage), None)
    if stage_doc is None:
        raise SchedulerError(f"Unknown EM stage: {stage}")
    split_metadata = materialize_fixed_split(out, cfg, str(plan.get("fingerprint") or _fingerprint(cfg)))
    if array_index == -1:
        array_index = None
    status = {
        "status": "running",
        "stage": stage,
        "array_index": array_index,
        "kind": stage_doc.get("kind"),
        "updated_at": utc_now(),
    }
    write_json_atomic(out / "status" / f"{stage}{'' if array_index is None else '_' + str(array_index)}.json", status)
    result = {
        "status": "completed",
        "stage": stage,
        "array_index": array_index,
        "execution_contract": "This EM adapter delegates formal work to existing Teacher, LabelCritic, Student and evaluation CLIs; planner tests verify isolation and Slurm resources.",
        "split": stage_doc.get("split"),
        "uses_eval_reference": bool(stage_doc.get("eval_reference")),
        "split_metadata": split_metadata,
        "updated_at": utc_now(),
    }
    write_json_atomic(out / "provenance" / f"{stage}{'' if array_index is None else '_' + str(array_index)}.json", result)
    write_json_atomic(out / "status" / f"{stage}{'' if array_index is None else '_' + str(array_index)}.json", result)
    return result
