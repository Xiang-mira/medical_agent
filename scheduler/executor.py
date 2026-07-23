from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import SchedulerConfig, load_config, resolve_path
from .manifest import assert_strict_no_gt_manifest, case_id_from_row, read_case_csv
from .preflight import run_preflight
from .state import status_path, write_status
from .utils import SchedulerError, ensure_not_raw_data_write_path, read_json, sha256_file, utc_now, write_json_atomic


GT_ENV_KEYS = {
    "MASK_ROOT",
    "GT_ROOT",
    "LABEL_ROOT",
    "LABEL_PATH",
    "EVAL_REFERENCE",
    "PILOT_TRAIN_EVAL_CASE_LIST",
    "PILOT_TEST_EVAL_CASE_LIST",
}


def _task_from_plan(plan: dict[str, Any], task_name: str) -> dict[str, Any]:
    for task in plan.get("tasks", []):
        if task.get("task_name") == task_name:
            return task
    raise SchedulerError(f"Task not found in plan: {task_name}")


def _load_context(run_dir: Path, task_name: str) -> tuple[dict[str, Any], dict[str, Any], SchedulerConfig]:
    plan = read_json(run_dir / "plan.json")
    task = _task_from_plan(plan, task_name)
    cfg = load_config(run_dir / "config_snapshot.yaml")
    return plan, task, cfg


def _attempt(run_dir: Path, task_id: str, array_index: int | None) -> int:
    path = status_path(run_dir, task_id, array_index)
    if not path.exists():
        return 1
    try:
        return int(read_json(path).get("attempt") or 0) + 1
    except Exception:
        return 1


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.time() - started
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "return_code": proc.returncode,
                    "elapsed_seconds": elapsed,
                    "stdout_tail": proc.stdout[-8000:],
                    "stderr_tail": proc.stderr[-8000:],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise SchedulerError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr[-2000:]}")
    return {"return_code": proc.returncode, "elapsed_seconds": elapsed, "stdout_tail": proc.stdout[-2000:]}


def _env_for_task(task: dict[str, Any], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    policy = str(task.get("gt_access_policy") or "none")
    if policy != "eval_reference_only":
        for key in list(env):
            if key in GT_ENV_KEYS or key.upper() in GT_ENV_KEYS:
                env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def _write_one_row_manifest(rows: list[dict[str, str]], index: int, out: Path) -> dict[str, str]:
    if index < 0 or index >= len(rows):
        raise SchedulerError(f"Array index {index} outside manifest rows 0..{len(rows)-1}")
    row = rows[index]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    out = list(command)
    if option not in out:
        out.extend([option, value])
        return out
    idx = out.index(option)
    if idx + 1 >= len(out):
        raise SchedulerError(f"Malformed command option {option}: {command}")
    out[idx + 1] = value
    return out


def _manifest_for_array_task(task_name: str, cfg: SchedulerConfig) -> Path:
    paths = cfg.paths
    if task_name == "teacher_train_array":
        key = "pilot_train_input_case_list"
    elif task_name in {"teacher_test_array", "student_test_array"}:
        key = "pilot_test_input_case_list"
    else:
        raise SchedulerError(f"No manifest mapping for array task {task_name}")
    path = resolve_path(paths.get(key))
    if path is None:
        raise SchedulerError(f"Missing configured manifest for {task_name}: {key}")
    return path


def _stage_cases(cfg: SchedulerConfig, split: str) -> list[dict[str, str]]:
    key = "pilot_train_input_case_list" if split == "train" else "pilot_test_input_case_list"
    path = resolve_path(cfg.paths.get(key))
    if path is None:
        raise SchedulerError(f"Missing case list path: {key}")
    return read_case_csv(path)


def _target_prompts(cfg: SchedulerConfig) -> dict[str, str]:
    path = resolve_path(cfg.paths.get("pilot_target_config") or cfg.paths.get("target_config"))
    if path is None:
        raise SchedulerError("Missing target config")
    doc = read_json(path)
    return {str(k): str(v) for k, v in (doc.get("organ_to_prompt") or {}).items()}


def _run_array_task(run_dir: Path, task: dict[str, Any], cfg: SchedulerConfig, array_index: int) -> dict[str, Any]:
    rows = assert_strict_no_gt_manifest(_manifest_for_array_task(task["task_name"], cfg))["case_ids"]
    original_rows = read_case_csv(_manifest_for_array_task(task["task_name"], cfg))
    row = _write_one_row_manifest(
        original_rows,
        array_index,
        run_dir / "manifests" / task["task_name"] / f"case_{array_index:04d}.csv",
    )
    case_id = case_id_from_row(row)
    command = list(task["command"])
    env = _env_for_task(task, {"MEDAI_SCHEDULER_RUN_DIR": str(run_dir), "MEDAI_SCHEDULER_CASE_ID": case_id})
    if task["task_name"].startswith("teacher_"):
        command = _replace_option(command, "--case-list", str(run_dir / "manifests" / task["task_name"] / f"case_{array_index:04d}.csv"))
        output_root = run_dir / "outputs" / task["task_name"].replace("_array", "")
        command = _replace_option(command, "--output", str(output_root))
    elif task["task_name"] == "student_test_array":
        ct_path = str(row.get("ct_path") or row.get("image") or "")
        if not ct_path:
            raise SchedulerError(f"Student test row has no ct_path/image: {row}")
        output_dir = run_dir / "outputs" / "student_test" / case_id
        command = [str(output_dir if arg == "$ARRAY_OUTPUT_DIR" else ct_path if arg == "$ARRAY_CT_PATH" else arg) for arg in command]
    result = _run_subprocess(command, cwd=Path.cwd(), env=env, log_path=run_dir / "logs" / f"{task['task_name']}_{array_index:04d}_subprocess.json")
    result.update({"case_id": case_id, "array_index": array_index, "manifest_row_count": len(rows)})
    return result


def _audit_cases(run_dir: Path, cfg: SchedulerConfig, *, split: str, root_name: str) -> dict[str, Any]:
    cases = _stage_cases(cfg, split)
    root = run_dir / "outputs" / root_name
    rows = []
    missing = []
    for row in cases:
        case_id = case_id_from_row(row)
        candidates = [root / case_id, root / "annotation_versions" / case_id, root / "student_test" / case_id]
        found = next((p for p in candidates if p.exists()), None)
        file_count = len(list(found.rglob("*"))) if found else 0
        rows.append({"case_id": case_id, "found": bool(found), "path": str(found) if found else "", "file_count": file_count})
        if not found or file_count == 0:
            missing.append(case_id)
    report = {"status": "success" if not missing else "failed", "root": str(root), "cases": rows, "missing_cases": missing}
    if missing:
        raise SchedulerError(f"Audit failed for {root_name}; missing cases: {missing[:10]}")
    return report


def _candidate_paths(stage_root: Path, case_id: str, organ: str) -> list[Path]:
    patterns = [
        stage_root / case_id / "per_model",
        stage_root / "annotation_versions" / case_id,
        stage_root / case_id,
    ]
    paths: list[Path] = []
    for base in patterns:
        if base.exists():
            paths.extend(sorted(base.rglob(f"{organ}.nii.gz")))
            paths.extend(sorted(base.rglob(f"_{organ}.nii.gz")))
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _build_labelcritic_manifest(run_dir: Path, cfg: SchedulerConfig) -> dict[str, Any]:
    cases = _stage_cases(cfg, "train")
    prompts = _target_prompts(cfg)
    out = run_dir / "manifests" / "labelcritic_candidates.jsonl"
    rows = []
    stage_root = run_dir / "outputs" / "teacher_train"
    for case in cases:
        case_id = case_id_from_row(case)
        ct_path = str(case.get("ct_path") or case.get("image") or "")
        for organ in prompts:
            candidates = _candidate_paths(stage_root, case_id, organ)
            if len(candidates) < 2:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "organ": organ,
                    "ct_path": ct_path,
                    "candidate_a_path": str(candidates[0]),
                    "candidate_b_path": str(candidates[1]),
                    "candidate_a_source": candidates[0].parent.parent.name,
                    "candidate_b_source": candidates[1].parent.parent.name,
                    "prompt_path": "",
                    "swap_group_id": f"{case_id}:{organ}",
                    "expected_answer": None,
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "success" if rows else "failed", "path": str(out), "rows": len(rows)}
    if not rows:
        raise SchedulerError("No LabelCritic candidate pairs found after teacher_train_audit")
    return report


def _run_labelcritic_batch(run_dir: Path, task: dict[str, Any], cfg: SchedulerConfig) -> dict[str, Any]:
    manifest = run_dir / "manifests" / "labelcritic_candidates.jsonl"
    if not manifest.exists():
        raise SchedulerError(f"Missing LabelCritic candidate manifest: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SchedulerError("LabelCritic candidate manifest is empty")
    prompts = _target_prompts(cfg)
    out_dir = run_dir / "outputs" / "labelcritic"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    student_items = []
    env = _env_for_task(task, {"NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"})
    for idx, row in enumerate(rows):
        decision_path = out_dir / f"decision_{idx:06d}.json"
        command = [
            sys.executable,
            "run_medai_cli.py",
            "--json",
            "critic",
            "--ct",
            row["ct_path"],
            "--mask-a",
            row["candidate_a_path"],
            "--mask-b",
            row["candidate_b_path"],
            "--organ",
            row["organ"],
            "--output",
            str(decision_path),
            "--backend",
            "labelcritic",
        ]
        _run_subprocess(command, cwd=Path.cwd(), env=env, log_path=run_dir / "logs" / f"labelcritic_{idx:06d}.json")
        decision = read_json(decision_path) if decision_path.exists() else {}
        choice = str(decision.get("parsed_choice") or decision.get("choice") or decision.get("winner") or "").upper()
        selected = row["candidate_a_path"] if choice in {"A", "1", "MASK_A"} else row["candidate_b_path"] if choice in {"B", "2", "MASK_B"} else ""
        result = {**row, "decision_path": str(decision_path), "parsed_choice": choice or "invalid", "status": "success" if selected else "invalid"}
        result_rows.append(result)
        if selected:
            student_items.append(
                {
                    "case_id": row["case_id"],
                    "organ": row["organ"],
                    "image": row["ct_path"],
                    "ct_path": row["ct_path"],
                    "mask": selected,
                    "mask_path": selected,
                    "prompt": prompts.get(row["organ"], row["organ"].replace("_", " ")),
                    "prompt_text": prompts.get(row["organ"], row["organ"].replace("_", " ")),
                    "supervision_type": "positive",
                    "sample_kind": "labelcritic_selected_pseudo_label",
                    "source": "labelcritic",
                    "labelcritic_decision_path": str(decision_path),
                }
            )
    results_jsonl = out_dir / "labelcritic_results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_doc = {
        "version": "scheduler_voxtell_prompt_student_manifest_v1",
        "stage": "pseudo_label_manifest",
        "target_count": len(prompts),
        "items": student_items,
    }
    write_json_atomic(run_dir / "manifests" / "voxtell_prompt_student_manifest.json", manifest_doc)
    if not student_items:
        raise SchedulerError("LabelCritic produced no valid selected pseudo labels")
    return {"status": "success", "results": len(result_rows), "student_manifest_items": len(student_items), "results_jsonl": str(results_jsonl)}


def _audit_student_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifests" / "voxtell_prompt_student_manifest.json"
    if not path.exists():
        raise SchedulerError(f"Missing Student manifest: {path}")
    doc = read_json(path)
    items = doc.get("items") or []
    if not items:
        raise SchedulerError(f"Student manifest has no items: {path}")
    missing = [idx for idx, item in enumerate(items) if not item.get("image") or not item.get("mask") or not item.get("prompt")]
    if missing:
        raise SchedulerError(f"Student manifest has malformed items: {missing[:10]}")
    return {"status": "success", "path": str(path), "items": len(items), "sha256": sha256_file(path)}


def _audit_checkpoint(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "checkpoints" / "student"
    candidates = [
        root / "voxtell_prompt_train_result.json",
        root / "model_finetune.pth",
        root / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth",
    ]
    existing = [str(p) for p in candidates if p.exists()]
    if not existing:
        raise SchedulerError(f"No Student checkpoint/training result found under {root}")
    return {"status": "success", "root": str(root), "existing": existing}


def _find_mask(folder: Path, organ: str) -> Path | None:
    for name in (f"{organ}.nii.gz", f"_{organ}.nii.gz"):
        path = folder / name
        if path.exists():
            return path
    return None


def _mask_folder_from_eval_row(row: dict[str, str], cfg: SchedulerConfig) -> Path:
    for key in ("annotation_folder", "segmentation_folder", "segmentations", "label_folder", "gt_folder"):
        value = row.get(key)
        if value:
            return Path(value)
    mask_root = resolve_path(cfg.paths.get("mask_root"))
    if mask_root is None:
        raise SchedulerError("Eval manifest has no annotation folder and config has no mask_root")
    return mask_root / case_id_from_row(row) / "segmentations"


def _prediction_folder(root: Path, case_id: str) -> Path:
    candidates = [
        root / case_id,
        root / "annotation_versions" / case_id / "updated",
        root / "annotation_versions" / case_id,
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def _dice_precision_recall(pred_path: Path, gt_path: Path) -> dict[str, float]:
    import nibabel as nib
    import numpy as np

    pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
    gt = np.asanyarray(nib.load(str(gt_path)).dataobj) > 0
    if pred.shape != gt.shape:
        raise SchedulerError(f"Shape mismatch for metric: {pred_path} vs {gt_path}")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    dice_den = 2 * tp + fp + fn
    return {
        "dice": 1.0 if dice_den == 0 else float(2 * tp / dice_den),
        "precision": 1.0 if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": 1.0 if tp + fn == 0 else float(tp / (tp + fn)),
    }


def _run_metrics(run_dir: Path, cfg: SchedulerConfig) -> dict[str, Any]:
    eval_manifest = resolve_path(cfg.paths.get("pilot_test_eval_case_list"))
    if eval_manifest is None or not eval_manifest.exists():
        raise SchedulerError(f"Missing test eval-reference manifest: {eval_manifest}")
    rows = read_case_csv(eval_manifest)
    target_config = resolve_path(cfg.paths.get("pilot_target_config") or cfg.paths.get("target_config"))
    if target_config is None:
        raise SchedulerError("Missing metric target config")
    targets = [str(x) for x in read_json(target_config).get("target_organs", [])]
    metric_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        case_id = case_id_from_row(row)
        gt_folder = _mask_folder_from_eval_row(row, cfg)
        for source, pred_root in (
            ("teacher", run_dir / "outputs" / "teacher_test"),
            ("student", run_dir / "outputs" / "student_test"),
        ):
            pred_folder = _prediction_folder(pred_root, case_id)
            for organ in targets:
                pred = _find_mask(pred_folder, organ)
                gt = _find_mask(gt_folder, organ)
                if not pred or not gt:
                    missing.append({"case_id": case_id, "source": source, "organ": organ, "reason": "missing_prediction_or_gt"})
                    continue
                values = _dice_precision_recall(pred, gt)
                metric_rows.append({"case_id": case_id, "source": source, "organ": organ, **values})
    if not metric_rows:
        raise SchedulerError("No metric rows computed; check prediction and eval-reference paths")
    csv_path = run_dir / "summary" / "metrics_long.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["case_id", "source", "organ", "dice", "precision", "recall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    means: dict[str, dict[str, float]] = {}
    for source in ("teacher", "student"):
        rows_for_source = [r for r in metric_rows if r["source"] == source]
        if rows_for_source:
            means[source] = {
                "mean_dice": sum(float(r["dice"]) for r in rows_for_source) / len(rows_for_source),
                "mean_precision": sum(float(r["precision"]) for r in rows_for_source) / len(rows_for_source),
                "mean_recall": sum(float(r["recall"]) for r in rows_for_source) / len(rows_for_source),
                "rows": len(rows_for_source),
            }
    out = run_dir / "summary" / "metrics_summary.json"
    report = {
        "status": "success",
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": sha256_file(eval_manifest),
        "cases": len(rows),
        "target_count": len(targets),
        "metrics_long_csv": str(csv_path),
        "means": means,
        "missing_count": len(missing),
        "missing_examples": missing[:50],
        "note": "GT is intentionally accessed only in metrics_cpu.",
    }
    write_json_atomic(out, report)
    return report


def _final_report(run_dir: Path) -> dict[str, Any]:
    statuses = []
    for path in sorted((run_dir / "status").glob("*.json")):
        try:
            statuses.append(read_json(path))
        except Exception as exc:
            statuses.append({"path": str(path), "status": "invalid", "error": str(exc)})
    report = {
        "status": "success" if all(s.get("status") == "success" for s in statuses) else "incomplete_or_failed",
        "run_dir": str(run_dir),
        "statuses": statuses,
        "generated_at": utc_now(),
        "statement": "This report distinguishes full 373 target space from the pilot338 subset and preserves resource tier labels.",
    }
    write_json_atomic(run_dir / "summary" / "final_report.json", report)
    md = run_dir / "summary" / "final_report.md"
    md.write_text(
        "# Scheduler Final Report\n\n"
        f"- Status: {report['status']}\n"
        f"- Run dir: `{run_dir}`\n"
        "- Full target space: 373 target anatomical structures\n"
        "- Current pilot: 338 direct-match target subset\n",
        encoding="utf-8",
    )
    return report


def execute_task(run_dir: Path, task_name: str, array_index: int | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    plan, task, cfg = _load_context(run_dir, task_name)
    image_root = resolve_path(cfg.paths.get("image_root"))
    mask_root = resolve_path(cfg.paths.get("mask_root"))
    for out_root in (run_dir / "outputs", run_dir / "checkpoints", run_dir / "summary", run_dir / "logs"):
        ensure_not_raw_data_write_path(out_root, image_root, mask_root)
    attempt = _attempt(run_dir, task_name, array_index)
    write_status(run_dir, task_name, "running", array_index=array_index, attempt=attempt, started_at=utc_now())
    try:
        if task.get("array"):
            if array_index is None:
                array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
            result = _run_array_task(run_dir, task, cfg, array_index)
        elif task_name == "pilot_preflight_cpu":
            result = run_preflight(cfg, require_hpc_paths=True)
            write_json_atomic(run_dir / "summary" / "preflight.json", result)
        elif task.get("task_type") == "resource_selection":
            result = {"status": "success", "selected_profile": task.get("resource_profile"), "policy": "configured tier; use select-resources after smoke to update alternatives"}
            write_json_atomic(run_dir / "summary" / f"{task_name}.json", result)
        elif task_name == "teacher_train_audit":
            result = _audit_cases(run_dir, cfg, split="train", root_name="teacher_train")
            write_json_atomic(run_dir / "summary" / "teacher_train_audit.json", result)
        elif task_name == "teacher_test_array":
            raise SchedulerError("Unexpected non-array teacher_test_array execution")
        elif task_name == "teacher_train_array":
            raise SchedulerError("Unexpected non-array teacher_train_array execution")
        elif task_name == "teacher_test_resource_selection":
            result = {"status": "success", "selected_profile": task.get("resource_profile")}
        elif task_name == "labelcritic_candidate_manifest":
            result = _build_labelcritic_manifest(run_dir, cfg)
            write_json_atomic(run_dir / "summary" / "labelcritic_candidate_manifest.json", result)
        elif task_name == "labelcritic_batch":
            result = _run_labelcritic_batch(run_dir, task, cfg)
            write_json_atomic(run_dir / "summary" / "labelcritic_batch.json", result)
        elif task_name == "pseudo_label_manifest_audit":
            result = _audit_student_manifest(run_dir)
            write_json_atomic(run_dir / "summary" / "pseudo_label_manifest_audit.json", result)
        elif task_name == "student_train":
            result = _run_subprocess(task["command"], cwd=Path.cwd(), env=_env_for_task(task), log_path=run_dir / "logs" / "student_train_subprocess.json")
        elif task_name == "student_checkpoint_audit":
            result = _audit_checkpoint(run_dir)
            write_json_atomic(run_dir / "summary" / "student_checkpoint_audit.json", result)
        elif task_name == "test_prediction_audit":
            teacher = _audit_cases(run_dir, cfg, split="test", root_name="teacher_test")
            student = _audit_cases(run_dir, cfg, split="test", root_name="student_test")
            result = {"status": "success", "teacher": teacher, "student": student}
            write_json_atomic(run_dir / "summary" / "test_prediction_audit.json", result)
        elif task_name == "metrics_cpu":
            result = _run_metrics(run_dir, cfg)
        elif task_name == "final_report_cpu":
            result = _final_report(run_dir)
        else:
            if task.get("command"):
                result = _run_subprocess(task["command"], cwd=Path.cwd(), env=_env_for_task(task), log_path=run_dir / "logs" / f"{task_name}_subprocess.json")
            else:
                raise SchedulerError(f"No execution implementation for task {task_name}")
        write_status(run_dir, task_name, "success", array_index=array_index, attempt=attempt, finished_at=utc_now(), result=result)
        return {"status": "success", "task": task_name, "array_index": array_index, "result": result}
    except Exception as exc:
        write_status(run_dir, task_name, "failed", array_index=array_index, attempt=attempt, finished_at=utc_now(), error=str(exc))
        if isinstance(exc, SchedulerError):
            raise
        raise SchedulerError(str(exc)) from exc
