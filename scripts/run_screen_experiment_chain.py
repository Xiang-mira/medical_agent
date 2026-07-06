#!/usr/bin/env python3
"""Screen-friendly experiment chain runner with conservative auto-repair.

This orchestrator is meant for long medical-agent experiments where one stage
should automatically continue to the next, but known recoverable failures
should be repaired before retrying instead of killing the whole screen session.

Default mode builds a Round2 +10-case plan, runs a Round2 preflight, and then
launches the repaired Round2 entry.  For other experiment chains, pass a JSON
plan via --experiments-json or write a template with --write-template.

Safety policy:
  * No broad process killing; only vLLM is restarted when the failure indicates
    LabelCritic/vLLM is offline.
  * Dirty round directories are archived by rename, never deleted.
  * Unknown unrepaired failures are recorded and, by default, the runner moves
    to the next experiment instead of stopping the whole chain.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_BASELINE_RUN_ROOT = ROOT / "outputs" / "em_round_pure_cached_10case_formal_lite_20260703"
DEFAULT_CASE_PLAN_DIR = ROOT / "outputs" / "round2_plus10_case_plan_20260705"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "em_round2_plus10_20260705"
DEFAULT_CHAIN_ROOT = ROOT / "outputs" / "screen_experiment_chains" / "round2_plus10_20260705"
DEFAULT_QWEN_VLM_MODEL = ROOT / "checkpoints" / "Qwen" / "Qwen2-VL-7B-Instruct"


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def tail_text(path: Path, max_chars: int = 12000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-max_chars:]


def is_vllm_online(base_url: str = "http://localhost:8000") -> bool:
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def restart_vllm(screen_name: str = "vllm_server", *, model: Path = DEFAULT_QWEN_VLM_MODEL) -> dict[str, Any]:
    if is_vllm_online():
        return {"status": "already_online", "screen_name": screen_name}
    cmd = (
        "cd {root} && "
        "{python} -m vllm.entrypoints.openai.api_server "
        "--model {model} --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.4"
    ).format(
        root=shlex.quote(str(ROOT)),
        python=shlex.quote(PYTHON),
        model=shlex.quote(str(model)),
    )
    subprocess.run(["screen", "-dmS", screen_name, "bash", "-lc", cmd], cwd=str(ROOT), check=False)
    for idx in range(60):
        time.sleep(5)
        if is_vllm_online():
            return {"status": "restarted", "screen_name": screen_name, "waited_sec": (idx + 1) * 5}
    return {"status": "failed", "screen_name": screen_name, "waited_sec": 300}


def archive_dirty_round(output_root: Path, round_name: str = "round2") -> dict[str, Any]:
    target = output_root / round_name
    if not target.exists() and not target.is_symlink():
        return {"status": "not_needed", "path": str(target)}
    mstep = target / "mstep"
    checkpoint_markers = [
        mstep / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth",
        mstep / "model_finetune.pth",
        mstep / "voxtell_prompt_mstep_result.json",
    ]
    if any(path.exists() for path in checkpoint_markers):
        return {
            "status": "refused",
            "reason": "round_has_mstep_or_checkpoint_artifacts",
            "path": str(target),
        }
    archive = output_root / f"{round_name}_aborted_auto_{utc_stamp()}"
    target.rename(archive)
    return {"status": "archived", "from": str(target), "to": str(archive)}


def classify_failure(log_tail: str, status_doc: dict[str, Any] | None = None) -> str:
    text = (log_tail + "\n" + json.dumps(status_doc or {}, ensure_ascii=False)).lower()
    if "no_material_update" in text or "new_reliable_positive_count\": 0" in text:
        return "no_material_update"
    if "labelcritic_vllm_offline" in text or "vllm" in text and "offline" in text:
        return "vllm_offline"
    if "server unavailable" in text and "labelcritic" in text:
        return "vllm_offline"
    if "cuda out of memory" in text or "outofmemoryerror" in text:
        return "cuda_oom"
    if "round2_not_clean" in text or "round2 clean" in text and "failed" in text:
        return "dirty_round"
    if "case_list" in text and ("missing" in text or "unreadable" in text):
        return "case_list_missing"
    if "prompt semantic audit" in text and "review" in text:
        return "prompt_semantic_review"
    return "unknown"


@dataclass
class Stage:
    name: str
    cmd: list[str]
    env: dict[str, str]
    cwd: Path
    success_files: list[Path]
    status_files: list[Path]


def format_command(template: list[str], mapping: dict[str, str]) -> list[str]:
    return [part.format(**mapping) for part in template]


def default_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    case_plan_dir = Path(args.case_plan_dir).resolve()
    output_root = Path(args.output_root).resolve()
    baseline = Path(args.baseline_run_root).resolve()
    case_list = case_plan_dir / "case_list_round2_plus10.csv"
    preflight = case_plan_dir / "round2_preflight.json"
    stages: list[dict[str, Any]] = [
        {
            "name": "build_round2_plus10_case_plan",
            "cmd": [
                PYTHON,
                "scripts/build_round2_plus10_case_plan.py",
                "--output-dir",
                str(case_plan_dir),
            ],
            "success_files": [str(case_list)],
            "status_files": [str(case_plan_dir / "round2_plus10_case_plan_summary.json")],
        },
    ]
    if args.teacher_cache_cmd:
        stages.append({
            "name": "build_new10_teacher_cache",
            "cmd": args.teacher_cache_cmd,
            "shell": True,
            "success_files": [],
            "status_files": [str(case_plan_dir / "teacher_assets" / "round2_plus10" / "cache_manifest.json")],
        })
    else:
        stages.append({
            "name": "teacher_cache_instruction_gate",
            "cmd": [
                PYTHON,
                "-c",
                (
                    "import json, pathlib; "
                    f"p=pathlib.Path({str(case_plan_dir / 'teacher_cache_required.json')!r}); "
                    "p.parent.mkdir(parents=True, exist_ok=True); "
                    "p.write_text(json.dumps({'status':'needs_teacher_cache_command',"
                    "'reason':'pass --teacher-cache-cmd to build/cache the selected new 10 cases before formal Round2'},"
                    "indent=2)+'\\n'); "
                    "raise SystemExit(0)"
                ),
            ],
            "success_files": [str(case_plan_dir / "teacher_cache_required.json")],
            "status_files": [str(case_plan_dir / "teacher_cache_required.json")],
            "continue_after_success": False,
            "success_means_needs_manual": True,
        })
    stages.extend([
        {
            "name": "round2_preflight",
            "cmd": [
                PYTHON,
                "scripts/run_em_training.py",
                "--dry-run",
                "--rounds",
                "2",
                "--start-round",
                "2",
                "--baseline-run-root",
                str(baseline),
                "--case-list",
                str(case_list),
                "--output-root",
                str(output_root),
                "--voxtell-mstep-mode",
                "project_prompt_student",
                "--preflight-output",
                str(preflight),
            ],
            "success_files": [str(preflight)],
            "status_files": [str(preflight)],
        },
        {
            "name": "round2_estep_mstep_student_inference",
            "cmd": [
                PYTHON,
                "scripts/run_em_training.py",
                "--rounds",
                "2",
                "--start-round",
                "2",
                "--baseline-run-root",
                str(baseline),
                "--case-list",
                str(case_list),
                "--output-root",
                str(output_root),
                "--voxtell-mstep-mode",
                "project_prompt_student",
            ],
            "success_files": [
                str(output_root / "round2" / "round_summary.json"),
                str(output_root / "round2" / "estep" / "material_update_audit.json"),
            ],
            "status_files": [
                str(output_root / "round2" / "round_summary.json"),
                str(output_root / "round2" / "estep" / "material_update_audit.json"),
                str(output_root / "round2" / "mstep" / "voxtell_prompt_mstep_result.json"),
            ],
        },
    ])
    return [{"name": "round2_plus10", "stages": stages, "env": {}}]


def load_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.experiments_json:
        doc = read_json(Path(args.experiments_json).resolve(), {})
        experiments = doc.get("experiments") if isinstance(doc, dict) else None
        if not isinstance(experiments, list):
            raise SystemExit(f"Invalid experiments JSON: {args.experiments_json}")
        return experiments
    return default_experiments(args)


def material_update_terminal_status(output_root: Path) -> dict[str, Any] | None:
    audit = read_json(output_root / "round2" / "estep" / "material_update_audit.json", {})
    if not isinstance(audit, dict) or not audit:
        return None
    if audit.get("material_update_decision") == "no_material_update":
        return {
            "status": "complete_no_material_update",
            "reason": "No new reliable teacher positives; M-step/student inference should not run.",
            "material_update_audit": audit,
        }
    return None


def run_stage(
    *,
    experiment_name: str,
    stage_doc: dict[str, Any],
    chain_root: Path,
    base_env: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_name = str(stage_doc["name"])
    stage_root = chain_root / experiment_name / stage_name
    log_path = stage_root / "stage.log"
    stage_state = stage_root / "stage_state.json"
    stage_root.mkdir(parents=True, exist_ok=True)

    if stage_doc.get("success_means_needs_manual"):
        # Still run the command to write the instruction artifact, then stop the
        # current experiment gracefully.
        pass

    env = os.environ.copy()
    env.update(base_env)
    env.update({str(k): str(v) for k, v in (stage_doc.get("env") or {}).items()})
    env.setdefault("PYTHONPATH", f"{ROOT / 'agent-harness'}:{env.get('PYTHONPATH', '')}")
    env.setdefault("MEDAI_EXPERIMENT_PROFILE", "advisor_aligned_default")

    cmd = stage_doc.get("cmd")
    if not cmd:
        raise ValueError(f"Stage {stage_name} has no cmd")
    shell = bool(stage_doc.get("shell"))
    if shell:
        printable = cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)
        run_cmd: str | list[str] = printable
    else:
        run_cmd = [str(x) for x in cmd]
        printable = " ".join(shlex.quote(x) for x in run_cmd)

    max_attempts = int(stage_doc.get("max_attempts") or args.max_retries + 1)
    repairs: list[dict[str, Any]] = []
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n==== {utc_stamp()} {experiment_name}/{stage_name} attempt {attempt}/{max_attempts} ====\n")
            log.write(f"cmd: {printable}\n")
            log.flush()
            proc = subprocess.Popen(
                run_cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=shell,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = proc.wait()
        elapsed = round(time.time() - started, 3)

        status_docs = [
            read_json(Path(path), {})
            for path in stage_doc.get("status_files", [])
            if Path(path).exists()
        ]
        status_doc = next((doc for doc in status_docs if isinstance(doc, dict) and doc), {})
        success_files = [Path(path) for path in stage_doc.get("success_files", [])]
        success_files_ok = all(path.exists() for path in success_files)
        terminal_no_update = material_update_terminal_status(Path(args.output_root).resolve())
        if return_code == 0 and success_files_ok:
            payload = {
                "status": "success",
                "stage": stage_name,
                "attempt": attempt,
                "elapsed_sec": elapsed,
                "return_code": return_code,
                "log": str(log_path),
                "repairs": repairs,
                "status_doc": status_doc,
            }
            if stage_doc.get("success_means_needs_manual"):
                payload["status"] = "needs_manual_input"
                payload["reason"] = "stage_success_requested_manual_stop"
            write_json(stage_state, payload)
            return payload
        if terminal_no_update:
            payload = {
                **terminal_no_update,
                "stage": stage_name,
                "attempt": attempt,
                "elapsed_sec": elapsed,
                "return_code": return_code,
                "log": str(log_path),
                "repairs": repairs,
            }
            write_json(stage_state, payload)
            return payload

        failure_kind = classify_failure(tail_text(log_path), status_doc)
        repair: dict[str, Any] = {
            "attempt": attempt,
            "failure_kind": failure_kind,
            "time": utc_stamp(),
        }
        if failure_kind == "vllm_offline":
            repair["action"] = "restart_vllm"
            repair["result"] = restart_vllm()
        elif failure_kind == "cuda_oom":
            repair["action"] = "reduce_parallelism_and_retry"
            env["MEDAI_PROMPT_BATCH_SIZE"] = "1"
            env["MEDAI_MSTEP_BATCH_SIZE"] = "1"
            env["MEDAI_ORGAN_WORKER_COUNT"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            repair["env_updates"] = {
                key: env[key]
                for key in [
                    "MEDAI_PROMPT_BATCH_SIZE",
                    "MEDAI_MSTEP_BATCH_SIZE",
                    "MEDAI_ORGAN_WORKER_COUNT",
                    "PYTORCH_CUDA_ALLOC_CONF",
                ]
            }
        elif failure_kind == "dirty_round" and args.archive_dirty_round:
            repair["action"] = "archive_dirty_round"
            repair["result"] = archive_dirty_round(Path(args.output_root).resolve(), "round2")
        elif failure_kind == "case_list_missing":
            repair["action"] = "rebuild_case_plan_on_next_attempt"
            # The default chain has a dedicated case-plan stage; for custom
            # chains this records the issue and retries unchanged.
        elif failure_kind == "prompt_semantic_review":
            repair["action"] = "no_auto_override"
            repair["reason"] = "Prompt semantic review can change dataset definition; manual review required."
            repairs.append(repair)
            break
        else:
            repair["action"] = "none_available"
            repairs.append(repair)
            break
        repairs.append(repair)
        write_json(stage_state, {
            "status": "retrying",
            "stage": stage_name,
            "attempt": attempt,
            "return_code": return_code,
            "failure_kind": failure_kind,
            "repair": repair,
            "log": str(log_path),
        })
        if attempt < max_attempts:
            time.sleep(int(args.retry_sleep_sec))

    payload = {
        "status": "failed",
        "stage": stage_name,
        "attempts": attempt,
        "return_code": return_code if "return_code" in locals() else None,
        "failure_kind": failure_kind if "failure_kind" in locals() else "unknown",
        "log": str(log_path),
        "repairs": repairs,
        "tail": tail_text(log_path, 4000),
    }
    write_json(stage_state, payload)
    return payload


def write_template(path: Path) -> None:
    template = {
        "experiments": [
            {
                "name": "example_experiment",
                "env": {"MEDAI_ORGAN_WORKER_COUNT": "4"},
                "stages": [
                    {
                        "name": "preflight",
                        "cmd": [PYTHON, "scripts/run_em_training.py", "--dry-run"],
                        "success_files": [],
                        "status_files": [],
                    },
                    {
                        "name": "run",
                        "cmd": [PYTHON, "scripts/run_em_training.py", "--rounds", "2"],
                        "success_files": ["outputs/example/round2/round_summary.json"],
                        "status_files": ["outputs/example/round2/round_summary.json"],
                        "max_attempts": 3,
                    },
                ],
            }
        ]
    }
    write_json(path, template)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-root", default=str(DEFAULT_CHAIN_ROOT))
    parser.add_argument("--baseline-run-root", default=str(DEFAULT_BASELINE_RUN_ROOT))
    parser.add_argument("--case-plan-dir", default=str(DEFAULT_CASE_PLAN_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--teacher-cache-cmd",
        default="",
        help=(
            "Optional shell command that builds teacher cache for the selected new 10 cases. "
            "If omitted, the default Round2+10 chain writes an instruction artifact and stops before Round2."
        ),
    )
    parser.add_argument("--experiments-json", default="", help="Custom JSON experiment chain.")
    parser.add_argument("--write-template", default="", help="Write a custom experiments-json template and exit.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries after the first attempt for each stage.")
    parser.add_argument("--retry-sleep-sec", type=int, default=60)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archive-dirty-round", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_template:
        write_template(Path(args.write_template).resolve())
        print(f"Wrote template: {Path(args.write_template).resolve()}")
        return 0

    chain_root = Path(args.chain_root).resolve()
    chain_root.mkdir(parents=True, exist_ok=True)
    state_path = chain_root / "chain_state.json"
    experiments = load_experiments(args)
    chain_state: dict[str, Any] = {
        "stage": "screen_experiment_chain",
        "status": "running",
        "started_at": utc_stamp(),
        "chain_root": str(chain_root),
        "experiments_total": len(experiments),
        "experiments": [],
    }
    write_json(state_path, chain_state)

    overall_failed = False
    for experiment in experiments:
        name = str(experiment.get("name") or f"experiment_{len(chain_state['experiments']) + 1}")
        exp_state: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_at": utc_stamp(),
            "stages": [],
        }
        chain_state["experiments"].append(exp_state)
        write_json(state_path, chain_state)
        base_env = {str(k): str(v) for k, v in (experiment.get("env") or {}).items()}
        for stage_doc in experiment.get("stages") or []:
            result = run_stage(
                experiment_name=name,
                stage_doc=stage_doc,
                chain_root=chain_root,
                base_env=base_env,
                args=args,
            )
            exp_state["stages"].append(result)
            write_json(state_path, chain_state)
            if result["status"] in {"success"}:
                continue
            if result["status"] in {"complete_no_material_update", "needs_manual_input"}:
                exp_state["status"] = result["status"]
                exp_state["finished_at"] = utc_stamp()
                break
            exp_state["status"] = "failed"
            exp_state["finished_at"] = utc_stamp()
            overall_failed = True
            if not args.continue_on_failure:
                chain_state["status"] = "failed"
                chain_state["finished_at"] = utc_stamp()
                write_json(state_path, chain_state)
                return 2
            break
        else:
            exp_state["status"] = "success"
            exp_state["finished_at"] = utc_stamp()
        write_json(state_path, chain_state)

    chain_state["status"] = "completed_with_failures" if overall_failed else "success"
    chain_state["finished_at"] = utc_stamp()
    write_json(state_path, chain_state)
    print(json.dumps(chain_state, indent=2, ensure_ascii=False))
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
