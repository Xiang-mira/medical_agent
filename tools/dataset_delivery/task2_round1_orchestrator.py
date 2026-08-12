#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from tools.dataset_delivery.delivery_lib import read_csv_rows, utc_now, write_json  # noqa: E402
from tools.dataset_delivery.task2_formal_manifest import FORMAL_CASE_COUNT, build_formal_manifest, validate_formal_manifest  # noqa: E402


LABELCRITIC_MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "OOM", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
ACTIVE_STATES = {"PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
SUCCESS_STATES = {"COMPLETED"}
ONE_BY_ONE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return {
            "command": command,
            "return_code": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"command": command, "return_code": 127, "stdout": "", "stderr": str(exc), "ok": False}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "return_code": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "timeout", "ok": False}


def normalize_labelcritic_endpoint(base_url: str, port: int) -> tuple[str, int]:
    base = re.sub(r"/v1/?$", "", (base_url or "http://localhost").rstrip("/"))
    match = re.match(r"^(https?://[^/:]+):(\d+)$", base)
    if match:
        return match.group(1), int(match.group(2))
    return base, int(port)


def _urlopen_json(url: str, *, timeout: int = 20, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def labelcritic_health(base_url: str, port: int, *, expected_model: str = LABELCRITIC_MODEL_ID) -> dict[str, Any]:
    base, resolved_port = normalize_labelcritic_endpoint(base_url, port)
    try:
        urllib.request.urlopen(f"{base}:{resolved_port}/health", timeout=10).read()
        models_doc = _urlopen_json(f"{base}:{resolved_port}/v1/models", timeout=20)
        models = [str(item.get("id") or "") for item in models_doc.get("data", []) if isinstance(item, dict)]
        return {
            "status": "READY" if expected_model in models else "MODEL_MISMATCH",
            "base_url": base,
            "port": resolved_port,
            "expected_model": expected_model,
            "served_models": models,
            "failure_reason": "" if expected_model in models else f"expected {expected_model}, served={models}",
        }
    except Exception as exc:
        return {
            "status": "OFFLINE",
            "base_url": base,
            "port": resolved_port,
            "expected_model": expected_model,
            "served_models": [],
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def labelcritic_runtime_preflight(base_url: str, port: int, *, expected_model: str = LABELCRITIC_MODEL_ID) -> dict[str, Any]:
    health = labelcritic_health(base_url, port, expected_model=expected_model)
    if health["status"] != "READY":
        return {"status": "FAILED", "stage": "health", "health": health, "failure_reason": health.get("failure_reason", "")}
    base, resolved_port = normalize_labelcritic_endpoint(base_url, port)
    image_url = f"data:image/png;base64,{ONE_BY_ONE_PNG}"
    payload = {
        "model": expected_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are LabelCritic. Compare mask1 and mask2. This is a runtime parse smoke. Reply exactly: MASK1"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        doc = _urlopen_json(f"{base}:{resolved_port}/v1/chat/completions", timeout=90, payload=payload)
        text = str((((doc.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
        parsed = "MASK1" in text.upper() or "1" == text.strip()
        return {
            "status": "PASSED" if parsed else "FAILED",
            "stage": "multi_image_parse_request",
            "health": health,
            "response_text": text[:500],
            "failure_reason": "" if parsed else "runtime response did not parse as MASK1",
        }
    except Exception as exc:
        return {"status": "FAILED", "stage": "multi_image_request", "health": health, "failure_reason": f"{type(exc).__name__}: {exc}"}


def slurm_job_state(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {"state": "UNKNOWN", "job_id": job_id, "source": "none"}
    squeue = _run(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
    if squeue["ok"] and squeue["stdout"].strip():
        return {"state": squeue["stdout"].splitlines()[0].strip(), "job_id": job_id, "source": "squeue"}
    sacct = _run(["sacct", "-n", "-j", str(job_id), "--format=State", "-P"])
    if sacct["ok"] and sacct["stdout"].strip():
        state = sacct["stdout"].splitlines()[0].split("|", 1)[0].strip().split()[0]
        return {"state": state, "job_id": job_id, "source": "sacct"}
    return {"state": "UNKNOWN", "job_id": job_id, "source": "unknown", "squeue": squeue, "sacct": sacct}


def find_labelcritic_job_by_name() -> str:
    result = _run(["squeue", "-h", "-n", "labelcritic_72b_service", "-o", "%i %T"])
    if not result["ok"]:
        return ""
    for line in result["stdout"].splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ACTIVE_STATES:
            return parts[0]
    return ""


def _state_paths(state_root: Path) -> dict[str, Path]:
    root = state_root / "round1_orchestrated"
    return {
        "root": root,
        "state": root / "state.json",
        "events": root / "events.jsonl",
        "failures": root / "failures.jsonl",
        "last_failure": root / "last_failure.json",
        "controller_sbatch": root / "controller.sbatch",
        "controller_job": root / "controller_job_id.txt",
        "mstep_sbatch": root / "round1_mstep_student.sbatch",
        "mstep_job": root / "round1_mstep_job_id.txt",
        "mstep_manifest": root / "round1" / "mstep" / "voxtell_prompt_student_manifest.json",
        "mstep_output": root / "round1" / "mstep",
        "final": root / "round1_final_status.json",
    }


def _attempts_root(state_root: Path) -> Path:
    return state_root / "round1_orchestrated_attempts"


def archive_current_attempt(state_root: Path, *, reason: str) -> dict[str, Any]:
    paths = _state_paths(state_root)
    root = paths["root"]
    if not root.exists():
        return {"status": "NO_CURRENT_ATTEMPT", "reason": reason}
    attempts = _attempts_root(state_root)
    attempts.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in attempts.glob("attempt_*"):
        match = re.search(r"attempt_(\d+)$", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    destination = attempts / f"attempt_{(max(numbers) if numbers else 0) + 1:03d}"
    root.rename(destination)
    record = {
        "status": "ARCHIVED",
        "archived_attempt": str(destination),
        "reason": reason,
        "archived_at": utc_now(),
    }
    _write_json(destination / "attempt_archive.json", record)
    return record


def _load_state(state_root: Path) -> dict[str, Any]:
    paths = _state_paths(state_root)
    state = _read_json(paths["state"], {})
    return state if isinstance(state, dict) else {}


def _save_state(state_root: Path, **updates: Any) -> dict[str, Any]:
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    state.update(updates)
    state["updated_at"] = utc_now()
    _write_json(paths["state"], state)
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    with paths["events"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": state["updated_at"], **updates}, ensure_ascii=False) + "\n")
    return state


def _tail_text(value: Any, limit: int = 4000) -> str:
    text = str(value or "")
    return text[-limit:]


def _compact_details(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = str(key)
            if key in {"stdout", "stderr", "stdout_tail", "stderr_tail"}:
                compact[safe_key] = _tail_text(item)
            else:
                compact[safe_key] = _compact_details(item)
        return compact
    if isinstance(value, (list, tuple)):
        return [_compact_details(item) for item in value[:100]]
    if isinstance(value, set):
        return sorted(_compact_details(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def log_failure(state_root: Path, *, stage: str, failure_reason: str, details: Any | None = None) -> dict[str, Any]:
    paths = _state_paths(state_root)
    record = {
        "time": utc_now(),
        "stage": stage,
        "failure_reason": str(failure_reason or "unknown_failure"),
        "git_commit": _git_commit(),
        "details": _compact_details(details or {}),
    }
    paths["failures"].parent.mkdir(parents=True, exist_ok=True)
    with paths["failures"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(paths["last_failure"], record)
    return record


def _git_commit() -> str:
    result = _run(["git", "rev-parse", "HEAD"])
    return result["stdout"] if result["ok"] else ""


def run_static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    branch = _run(["git", "branch", "--show-current"])
    add("git_main_clean", status["ok"] and not status["stdout"] and branch["stdout"] == "main", {"status": status, "branch": branch["stdout"]})
    if not args.case_manifest.exists():
        try:
            generated = build_formal_manifest(
                base_manifest=args.base_manifest.resolve(),
                output_manifest=args.case_manifest.resolve(),
                audit_json=paths["root"] / "cases_103_manifest_build_audit.json",
            )
            add("cases_103_manifest_generated", generated.get("status") == "READY", generated)
        except Exception as exc:
            add("cases_103_manifest_generated", False, f"{type(exc).__name__}: {exc}")
    try:
        manifest = validate_formal_manifest(manifest=args.case_manifest.resolve(), base_manifest=args.base_manifest.resolve())
        add("cases_103_manifest", manifest.get("rows") == FORMAL_CASE_COUNT and manifest.get("unique_case_count") == FORMAL_CASE_COUNT, manifest)
    except Exception as exc:
        add("cases_103_manifest", False, f"{type(exc).__name__}: {exc}")
    add("personal_workspace_writable", os.access(args.workspace_root.resolve(), os.W_OK) if args.workspace_root.exists() else os.access(args.workspace_root.parent.resolve(), os.W_OK), str(args.workspace_root))
    public_root = Path("/projects/bodymaps/Data")
    add("public_data_not_output_root", public_root not in args.state_root.resolve().parents and args.state_root.resolve() != public_root, str(args.state_root))
    if public_root.exists():
        add("public_data_not_writable_by_user", not os.access(public_root, os.W_OK), str(public_root))
    for path_name, path in {
        "checkpoint_root": args.checkpoint_root,
        "nnunet_predict": args.nnunet_predict_executable,
        "unest_python": args.unest_python_executable,
        "resource_profiles": REPO_ROOT / "configs" / "resource_profiles.yaml",
        "target_config": args.target_config,
        "registry": args.registry,
    }.items():
        add(path_name, path.exists(), str(path))
    compile_result = _run([
        str(args.python), "-m", "py_compile",
        "tools/dataset_delivery/task2_round1_orchestrator.py",
        "tools/dataset_delivery/task2_formal_launcher.py",
        "tools/dataset_delivery/task2_dynamic_gpu_submitter.py",
        "tools/dataset_delivery/formal_round1_preflight.py",
    ])
    add("py_compile_orchestration", compile_result["ok"], compile_result)
    for script in [
        "scripts/task2/submit_task2_formal_103cases.sh",
        "scripts/task2/submit_labelcritic_72b_service.sh",
        "scripts/task2/check_task2_formal_103cases.sh",
        "scripts/task2/submit_task2_round1_orchestrated.sh",
        "scripts/task2/check_task2_round1_orchestrated.sh",
    ]:
        result = _run(["bash", "-n", script])
        add(f"bash_n:{script}", result["ok"], result)
    if args.run_static_tests:
        tests = [
            "tests/dataset_delivery/test_task2_round1_orchestrator.py",
            "tests/dataset_delivery/test_round1_production_hardening.py",
            "tests/dataset_delivery/test_task2_runtime_repair.py",
            "tests/dataset_delivery/test_task2_formal_production.py",
            "tests/dataset_delivery/test_task2_dynamic_gpu_submitter.py",
        ]
        result = _run([str(args.python), "-m", "pytest", "-q", *tests], timeout=int(args.static_tests_timeout_sec))
        add("task1_task2_scheduler_regression_tests", result["ok"], result)
    controller = render_controller_sbatch(args, paths["controller_sbatch"])
    add("controller_sbatch_rendered", paths["controller_sbatch"].exists(), controller)
    result = _run(["bash", "-n", str(paths["controller_sbatch"])])
    add("controller_bash_n", result["ok"], result)
    if not args.skip_sbatch_test_only:
        result = _run(["sbatch", "--test-only", str(paths["controller_sbatch"])])
        add("controller_sbatch_test_only", result["ok"], result)
        labelcritic_preview = render_labelcritic_preview_sbatch(args, paths["root"] / "labelcritic_72b_service.preview.sbatch")
        add("labelcritic_sbatch_preview_rendered", Path(labelcritic_preview["path"]).exists(), labelcritic_preview)
        result = _run(["bash", "-n", labelcritic_preview["path"]])
        add("labelcritic_sbatch_preview_bash_n", result["ok"], result)
        result = _run(["sbatch", "--test-only", labelcritic_preview["path"]])
        add("labelcritic_sbatch_preview_test_only", result["ok"], result)
    report = {
        "status": "PASSED" if all(check["ok"] for check in checks) else "FAILED",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "checks": checks,
    }
    _write_json(paths["root"] / "static_preflight.json", report)
    return report


def render_labelcritic_preview_sbatch(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=labelcritic_72b_service",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={args.labelcritic_partition}",
        f"#SBATCH --gres={args.labelcritic_gres}",
        f"#SBATCH --cpus-per-task={os.getenv('LABELCRITIC_CPUS', '16')}",
        f"#SBATCH --mem={os.getenv('LABELCRITIC_MEM', '192G')}",
        f"#SBATCH --time={os.getenv('LABELCRITIC_TIME', '08:00:00')}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export LABELCRITIC_MODEL_ID={shlex.quote(LABELCRITIC_MODEL_ID)}",
        f"export LABELCRITIC_TENSOR_PARALLEL_SIZE={int(args.labelcritic_tensor_parallel_size)}",
        "true",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path), "profile": {"partition": args.labelcritic_partition, "gres": args.labelcritic_gres, "tp": int(args.labelcritic_tensor_parallel_size)}}


def render_controller_sbatch(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    command = [
        str(args.python),
        "tools/dataset_delivery/task2_round1_orchestrator.py",
        "controller",
        "--state-root", str(args.state_root),
        "--workspace-root", str(args.workspace_root),
        "--case-manifest", str(args.case_manifest),
        "--base-manifest", str(args.base_manifest),
        "--python", str(args.python),
        "--checkpoint-root", str(args.checkpoint_root),
        "--nnunet-predict-executable", str(args.nnunet_predict_executable),
        "--unest-python-executable", str(args.unest_python_executable),
        "--registry", str(args.registry),
        "--target-config", str(args.target_config),
        "--gpu-target-workers", str(args.gpu_target_workers),
        "--gpu-overrequest-workers", str(args.gpu_overrequest_workers),
        "--gpu-profile-specs", args.gpu_profile_specs,
        "--poll-sec", str(args.poll_sec),
    ]
    env_exports = {
        "CODE_ROOT": str(REPO_ROOT),
        "STATE_ROOT": str(args.state_root),
        "WORKSPACE_ROOT": str(args.workspace_root),
        "LABELCRITIC_PARTITION": args.labelcritic_partition,
        "LABELCRITIC_GRES": args.labelcritic_gres,
        "LABELCRITIC_TENSOR_PARALLEL_SIZE": str(args.labelcritic_tensor_parallel_size),
        "LABELCRITIC_PORT": str(args.labelcritic_port),
        "LABELCRITIC_MODEL_ID": LABELCRITIC_MODEL_ID,
    }
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_round1_controller",
        f"#SBATCH --partition={args.controller_partition}",
        f"#SBATCH --cpus-per-task={args.controller_cpus}",
        f"#SBATCH --mem={args.controller_mem}",
        f"#SBATCH --time={args.controller_time}",
        f"#SBATCH --output={_state_paths(args.state_root)['root'] / 'controller_%j.out'}",
        f"#SBATCH --error={_state_paths(args.state_root)['root'] / 'controller_%j.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        *[f"export {key}={shlex.quote(value)}" for key, value in env_exports.items()],
        " ".join(shlex.quote(part) for part in command),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return {"path": str(path), "command": command, "env": env_exports}


def submit_controller(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    state = _load_state(args.state_root.resolve())
    if state.get("terminal_state") == "ROUND1_FAILED" and (getattr(args, "retry_failed", False) or getattr(args, "new_attempt", False)):
        archive_current_attempt(args.state_root.resolve(), reason="retry_failed_or_new_attempt")
    elif state.get("terminal_state") in {"ROUND1_PASSED", "ROUND1_FAILED"}:
        return {"status": "ROUND1_ALREADY_TERMINAL", "terminal_state": state.get("terminal_state"), "state_root": str(paths["root"])}
    elif getattr(args, "new_attempt", False) and paths["root"].exists():
        existing = str(state.get("controller_job_id") or "").strip()
        if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
            return {"status": "CONTROLLER_ALREADY_ACTIVE", "controller_job_id": existing, "state_root": str(paths["root"])}
        archive_current_attempt(args.state_root.resolve(), reason="explicit_new_attempt")
    preflight = run_static_preflight(args)
    paths = _state_paths(args.state_root.resolve())
    if preflight["status"] != "PASSED":
        failure = log_failure(args.state_root.resolve(), stage="static_preflight", failure_reason="static_preflight_failed", details=preflight)
        _save_state(args.state_root.resolve(), terminal_state="ROUND1_FAILED", stage="static_preflight", failure_reason="static_preflight_failed", static_preflight=preflight)
        raise SystemExit(json.dumps({"status": "STATIC_PREFLIGHT_FAILED", "report": str(paths["root"] / "static_preflight.json"), "last_failure": failure}, indent=2))
    state = _load_state(args.state_root.resolve())
    existing = str(state.get("controller_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in ACTIVE_STATES:
        return {"status": "CONTROLLER_ALREADY_ACTIVE", "controller_job_id": existing, "state_root": str(paths["root"])}
    result = _run(["sbatch", "--parsable", str(paths["controller_sbatch"])])
    if not result["ok"]:
        log_failure(args.state_root.resolve(), stage="controller_submit", failure_reason=result["stderr"] or "controller_submit_failed", details=result)
        _save_state(args.state_root.resolve(), terminal_state="ROUND1_FAILED", stage="controller_submit", failure_reason=result["stderr"], controller_submit=result)
        raise SystemExit(result["stderr"])
    job_id = result["stdout"].splitlines()[-1].strip()
    paths["controller_job"].write_text(job_id + "\n", encoding="utf-8")
    _save_state(args.state_root.resolve(), status="CONTROLLER_SUBMITTED", controller_job_id=job_id, git_commit=_git_commit(), static_preflight_path=str(paths["root"] / "static_preflight.json"))
    return {"status": "CONTROLLER_SUBMITTED", "controller_job_id": job_id, "state_root": str(paths["root"])}


def _service_paths(state_root: Path) -> dict[str, Path]:
    root = Path(os.getenv("LABELCRITIC_SERVICE_ROOT", str(state_root / "labelcritic_72b_service"))).expanduser()
    return {
        "root": root,
        "job": root / "job_id.txt",
        "base": root / "base_url.txt",
        "port": root / "port.txt",
        "endpoint": root / "endpoint.url",
    }


def ensure_labelcritic_service(state_root: Path) -> dict[str, Any]:
    service = _service_paths(state_root)
    env_base = os.getenv("LABELCRITIC_BASE_URL", "").strip()
    if env_base:
        env_port = int(os.getenv("LABELCRITIC_PORT", "8000"))
        health = labelcritic_health(env_base, env_port)
        if health["status"] == "READY":
            service["root"].mkdir(parents=True, exist_ok=True)
            service["base"].write_text(health["base_url"] + "\n", encoding="utf-8")
            service["port"].write_text(str(health["port"]) + "\n", encoding="utf-8")
            service["endpoint"].write_text(f"{health['base_url']}:{health['port']}\n", encoding="utf-8")
            return {"status": "REUSED_HEALTHY", "base_url": health["base_url"], "port": health["port"], "health": health, "source": "LABELCRITIC_BASE_URL"}
    base = service["base"].read_text(encoding="utf-8").strip() if service["base"].exists() else ""
    port = int(service["port"].read_text(encoding="utf-8").strip()) if service["port"].exists() else int(os.getenv("LABELCRITIC_PORT", "8000"))
    if base:
        health = labelcritic_health(base, port)
        if health["status"] == "READY":
            return {"status": "REUSED_HEALTHY", "base_url": health["base_url"], "port": health["port"], "health": health}
    job_id = service["job"].read_text(encoding="utf-8").strip() if service["job"].exists() else ""
    job_id = os.getenv("LABELCRITIC_JOB_ID", "").strip() or job_id
    if job_id and slurm_job_state(job_id).get("state") in ACTIVE_STATES:
        service["root"].mkdir(parents=True, exist_ok=True)
        service["job"].write_text(job_id + "\n", encoding="utf-8")
        return {"status": "REUSED_ACTIVE_JOB", "job_id": job_id}
    named = find_labelcritic_job_by_name()
    if named:
        service["root"].mkdir(parents=True, exist_ok=True)
        service["job"].write_text(named + "\n", encoding="utf-8")
        return {"status": "REUSED_ACTIVE_JOB", "job_id": named}
    result = _run(["bash", "scripts/task2/submit_labelcritic_72b_service.sh"], env=os.environ.copy())
    if not result["ok"]:
        log_failure(state_root, stage="labelcritic_submit", failure_reason=result["stderr"] or result["stdout"] or "labelcritic_submit_failed", details=result)
        return {"status": "SUBMIT_FAILED", "failure_reason": result["stderr"], "submit": result}
    job_id_match = re.search(r"LABELCRITIC_JOB_ID=([^\s]+)", result["stdout"])
    job_id = job_id_match.group(1) if job_id_match else (service["job"].read_text(encoding="utf-8").strip() if service["job"].exists() else "")
    return {"status": "SUBMITTED", "job_id": job_id, "submit": result}


def wait_for_labelcritic_runtime(state_root: Path, *, poll_sec: int) -> dict[str, Any]:
    while True:
        service = ensure_labelcritic_service(state_root)
        _save_state(state_root, stage="labelcritic_wait", labelcritic=service)
        if service["status"] in {"REUSED_HEALTHY"}:
            runtime = labelcritic_runtime_preflight(service["base_url"], int(service["port"]))
            _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
            if runtime["status"] == "PASSED":
                return {"status": "PASSED", "base_url": service["base_url"], "port": int(service["port"]), "runtime": runtime}
            log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
            return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime}
        job_id = str(service.get("job_id") or "")
        state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
        if state.get("state") in TERMINAL_FAILURE_STATES:
            log_failure(state_root, stage="labelcritic_job", failure_reason=f"labelcritic_job_terminal:{state.get('state')}", details=state)
            return {"status": "FAILED", "failure_reason": f"labelcritic_job_terminal:{state.get('state')}", "job": state}
        base_file = _service_paths(state_root)["base"]
        port_file = _service_paths(state_root)["port"]
        if base_file.exists() and port_file.exists():
            base = base_file.read_text(encoding="utf-8").strip()
            port = int(port_file.read_text(encoding="utf-8").strip())
            health = labelcritic_health(base, port)
            if health["status"] == "READY":
                runtime = labelcritic_runtime_preflight(base, port)
                _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
                if runtime["status"] == "PASSED":
                    return {"status": "PASSED", "base_url": base, "port": port, "runtime": runtime}
                log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
                return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime}
        time.sleep(max(5, poll_sec))


def poll_labelcritic_runtime(state_root: Path) -> dict[str, Any]:
    service = ensure_labelcritic_service(state_root)
    _save_state(state_root, stage="labelcritic_poll", labelcritic=service)
    if service["status"] == "SUBMIT_FAILED":
        return {"status": "FAILED", "failure_reason": service.get("failure_reason", "labelcritic_submit_failed"), "service": service}
    if service["status"] == "REUSED_HEALTHY":
        runtime = labelcritic_runtime_preflight(service["base_url"], int(service["port"]))
        _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
        if runtime["status"] == "PASSED":
            return {"status": "PASSED", "base_url": service["base_url"], "port": int(service["port"]), "runtime": runtime, "service": service}
        log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
        return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime, "service": service}
    job_id = str(service.get("job_id") or "")
    state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
    if state.get("state") in TERMINAL_FAILURE_STATES:
        log_failure(state_root, stage="labelcritic_job", failure_reason=f"labelcritic_job_terminal:{state.get('state')}", details=state)
        return {"status": "FAILED", "failure_reason": f"labelcritic_job_terminal:{state.get('state')}", "job": state, "service": service}
    base_file = _service_paths(state_root)["base"]
    port_file = _service_paths(state_root)["port"]
    if base_file.exists() and port_file.exists():
        base = base_file.read_text(encoding="utf-8").strip()
        port = int(port_file.read_text(encoding="utf-8").strip())
        health = labelcritic_health(base, port)
        if health["status"] == "READY":
            runtime = labelcritic_runtime_preflight(base, port)
            _save_state(state_root, stage="labelcritic_runtime_preflight", labelcritic_runtime_preflight=runtime)
            if runtime["status"] == "PASSED":
                return {"status": "PASSED", "base_url": base, "port": port, "runtime": runtime, "service": service}
            log_failure(state_root, stage="labelcritic_runtime_preflight", failure_reason=runtime.get("failure_reason", "runtime_preflight_failed"), details=runtime)
            return {"status": "FAILED", "failure_reason": runtime.get("failure_reason", "runtime_preflight_failed"), "runtime": runtime, "service": service}
    return {"status": "WAITING", "job": state, "service": service}


def _labelcritic_endpoint_hint(state_root: Path, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    if labelcritic and labelcritic.get("base_url") and labelcritic.get("port"):
        return {"base_url": str(labelcritic["base_url"]), "port": int(labelcritic["port"]), "source": "runtime"}
    service = _service_paths(state_root)
    if service["base"].exists() and service["port"].exists():
        return {
            "base_url": service["base"].read_text(encoding="utf-8").strip(),
            "port": int(service["port"].read_text(encoding="utf-8").strip()),
            "source": "service_files",
        }
    return {
        "base_url": os.getenv("LABELCRITIC_BASE_URL", "http://localhost"),
        "port": int(os.getenv("LABELCRITIC_PORT", "8000")),
        "source": "env_or_default",
    }


def submit_estep(args: argparse.Namespace, labelcritic: dict[str, Any] | None = None) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    state = _load_state(state_root)
    formal_root = Path(str(state.get("formal_root") or (_state_paths(state_root)["root"] / "formal_task2_round1"))).resolve()
    formal_summary = _read_json(formal_root / "task2_formal_summary.json", {})
    if state.get("e_step_status") in {"SUBMITTED", "PASSED"} and (
        (formal_root / "slurm" / "submitted_jobs.csv").exists()
        or formal_summary.get("status") == "PASSED"
    ):
        return {"status": "REUSED", "formal_root": str(formal_root), "e_step_status": state.get("e_step_status")}
    endpoint = _labelcritic_endpoint_hint(state_root, labelcritic)
    service_root = _service_paths(state_root)["root"]
    env = os.environ.copy()
    env.update({
        "DRY_RUN": "0",
        "RESUME": "1",
        "STAGE_WORKSPACE": "1",
        "DYNAMIC_GPU_SCHEDULER": "1",
        "GPU_TARGET_WORKERS": str(args.gpu_target_workers),
        "GPU_OVERREQUEST_WORKERS": str(args.gpu_overrequest_workers),
        "GPU_PROFILE_SPECS": args.gpu_profile_specs,
        "FORMAL_OUT_ROOT": str(formal_root),
        "STATE_ROOT": str(state_root),
        "WORKSPACE_ROOT": str(args.workspace_root),
        "CASE_MANIFEST": str(args.case_manifest),
        "BASE_MANIFEST": str(args.base_manifest),
        "LABELCRITIC_SERVICE_ROOT": str(service_root),
        "LABELCRITIC_BASE_URL": str(endpoint["base_url"]),
        "LABELCRITIC_PORT": str(endpoint["port"]),
        "LABELCRITIC_MODEL_ID": LABELCRITIC_MODEL_ID,
        "REQUIRE_LABELCRITIC_HEALTH": "0",
        "WAIT_LABELCRITIC_SEC": str(os.getenv("WAIT_LABELCRITIC_SEC", "14400")),
        "MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC": str(os.getenv("MEDAI_LABELCRITIC_ENDPOINT_WAIT_SEC", os.getenv("WAIT_LABELCRITIC_SEC", "14400"))),
    })
    result = _run(["bash", "scripts/task2/submit_task2_formal_103cases.sh"], env=env, timeout=None)
    if not result["ok"]:
        log_failure(state_root, stage="e_step_submit", failure_reason=result["stderr"] or result["stdout"] or "e_step_submit_failed", details=result)
        return {"status": "FAILED", "failure_reason": result["stderr"] or result["stdout"], "submit": result, "formal_root": str(formal_root)}
    _save_state(state_root, e_step_status="SUBMITTED", formal_root=str(formal_root), e_step_submit=result)
    return {"status": "SUBMITTED", "formal_root": str(formal_root), "submit": result}


def check_estep(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_state(args.state_root.resolve())
    formal_root = Path(str(state.get("formal_root") or ""))
    if not formal_root.exists():
        return {"status": "PENDING", "reason": "formal_root_missing"}
    env = os.environ.copy()
    env.update({"FORMAL_ROOT": str(formal_root), "CASE_MANIFEST": str(args.case_manifest), "BASE_MANIFEST": str(args.base_manifest), "STATE_ROOT": str(args.state_root)})
    result = _run(["bash", "scripts/task2/check_task2_formal_103cases.sh"], env=env, timeout=600)
    if result["ok"]:
        return {"status": "PASSED", "formal_root": str(formal_root), "check": result}
    jobs_csv = formal_root / "slurm" / "submitted_jobs.csv"
    active = []
    failed = []
    if jobs_csv.exists():
        for row in read_csv_rows(jobs_csv):
            job_id = str(row.get("job_id") or "")
            state = slurm_job_state(job_id)
            if state.get("state") in ACTIVE_STATES or state.get("state") == "UNKNOWN":
                active.append(state)
            if state.get("state") in TERMINAL_FAILURE_STATES:
                failed.append(state)
    if failed and not active:
        log_failure(args.state_root.resolve(), stage="e_step", failure_reason="e_step_jobs_terminal_failed", details={"failed_jobs": failed, "check": result})
        return {"status": "FAILED", "failure_reason": "e_step_jobs_terminal_failed", "failed_jobs": failed, "check": result}
    return {"status": "RUNNING", "active_jobs": active, "check": result}


def build_mstep_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from cli_anything.medai.core.continual_learning import TRAINING_CONTRACT_VERSION, canonicalize_training_record

    paths = _state_paths(args.state_root.resolve())
    if paths["mstep_manifest"].exists():
        doc = _read_json(paths["mstep_manifest"], {})
        if doc.get("items"):
            return {"status": "REUSED", "manifest": str(paths["mstep_manifest"]), "num_items": len(doc.get("items") or [])}
    state = _load_state(args.state_root.resolve())
    formal_root = Path(str(state.get("formal_root") or ""))
    rows_doc = _read_json(formal_root / "task2_formal_case_target_status.json", {})
    case_rows = {str(row.get("case_id") or ""): row for row in read_csv_rows(args.case_manifest)}
    target_doc = _read_json(args.target_config, {})
    targets = list(target_doc.get("target_organs") or [])
    organ_to_prompt = target_doc.get("organ_to_prompt") or {}
    items = []
    for row in rows_doc.get("rows") or []:
        if row.get("final_status") != "generated_valid_mask":
            continue
        case_id = str(row.get("case_id") or "")
        organ = str(row.get("target_name") or "")
        mask = str(row.get("mask_path") or "")
        ct = str((case_rows.get(case_id) or {}).get("ct_path") or "")
        if not case_id or not organ or not mask or not ct or organ not in targets:
            continue
        provider = str(row.get("selected_model") or row.get("source_model") or row.get("group") or row.get("target_group") or "task2_teacher")
        item = {
            "case_id": case_id,
            "image": ct,
            "ct_path": ct,
            "organ": organ,
            "canonical_organ": organ,
            "requested_canonical_id": organ,
            "resolved_canonical_id": organ,
            "prompt": organ_to_prompt.get(organ, organ.replace("_", " ")),
            "mask": mask,
            "mask_path": mask,
            "supervision_type": "positive",
            "target_type": "positive_hard",
            "label_role": "selected_pseudo_label",
            "supervision_role": "selected_pseudo_label",
            "distillation_role": "positive",
            "dataset_role": "pseudo_label",
            "source_model": provider,
            "selected_model": provider,
            "origin_provider": provider,
            "ground_truth_status": "selected_pseudo_label_not_expert_gt",
            "scoring_schema_version": "autolabel_core_v3",
            "grade": "A",
            "training_weight": 1.0,
            "distillation_eligible": True,
            "training_eligible": True,
            "student_target_id": targets.index(organ),
            "source_stage": "task2_formal_103case_round1_estep",
        }
        items.append(canonicalize_training_record(item, round_index=1, project_root=REPO_ROOT, strict_soft=True))
    manifest = {
        "version": "task2_formal_103_round1_voxtell_manifest_v1",
        "stage": "round1_mstep_manifest",
        "status": "success" if items else "failed",
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "source_formal_root": str(formal_root),
        "target_config": str(args.target_config),
        "num_items": len(items),
        "num_cases": len({item.get("case_id") for item in items}),
        "num_distillation_eligible_items": sum(1 for item in items if item.get("distillation_eligible") is not False and float(item.get("training_weight") or 0.0) > 0.0),
        "num_trainable_positive_items": sum(1 for item in items if item.get("training_eligible") is True and item.get("supervision_type") == "positive"),
        "novelty_audit": {
            "stage": "continual_learning_novelty",
            "status": "success",
            "decision": "full_update",
            "max_steps": int(os.getenv("MEDAI_MAX_STEPS", "2000")),
            "weighted_change_ratio": 1.0,
            "added_keys": [],
            "changed_keys": [],
            "removed_keys": [],
        },
        "items": items,
    }
    _write_json(paths["mstep_manifest"], manifest)
    return {"status": manifest["status"].upper(), "manifest": str(paths["mstep_manifest"]), "num_items": len(items)}


def submit_mstep(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    paths = _state_paths(state_root)
    state = _load_state(state_root)
    completed = _read_json(paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json", {})
    if state.get("mstep_status") == "PASSED" and completed.get("status") == "success":
        return {"status": "REUSED", "result": str(paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json")}
    existing = str(state.get("mstep_job_id") or "").strip()
    if existing and slurm_job_state(existing).get("state") in (ACTIVE_STATES | SUCCESS_STATES):
        return {"status": "REUSED", "job_id": existing}
    manifest = build_mstep_manifest(args)
    if manifest["status"] not in {"SUCCESS", "REUSED"}:
        log_failure(state_root, stage="m_step_manifest", failure_reason="mstep_manifest_failed", details=manifest)
        return {"status": "FAILED", "failure_reason": "mstep_manifest_failed", "manifest": manifest}
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=task2_round1_mstep_student",
        f"#SBATCH --partition={os.getenv('MSTEP_PARTITION', 'gpua100')}",
        f"#SBATCH --gres={os.getenv('MSTEP_GRES', 'gpu:A100:1')}",
        f"#SBATCH --cpus-per-task={os.getenv('MSTEP_CPUS', '12')}",
        f"#SBATCH --mem={os.getenv('MSTEP_MEM', '96G')}",
        f"#SBATCH --time={os.getenv('MSTEP_TIME', '10:00:00')}",
        f"#SBATCH --output={paths['root'] / 'mstep_%j.out'}",
        f"#SBATCH --error={paths['root'] / 'mstep_%j.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export MEDAI_OUTPUT_ROOT={shlex.quote(str(paths['root']))}",
        f"export MEDAI_VOXTELL_MODEL_DIR={shlex.quote(str(os.getenv('MEDAI_VOXTELL_MODEL_DIR', '/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/VoxTell/voxtell_v1.1')))}",
        "export MEDAI_STUDENT_BACKEND=voxtell_style_3d_prompt",
        "export MEDAI_VOXTELL_MSTEP_MODE=project_voxtell_prompt_distillation_student",
        "export MEDAI_VOXTELL_TRAINING_PROFILE=quality_weighted_ablation",
        "export MEDAI_MSTEP_BATCH_SIZE=1",
        "export MEDAI_FORMAL_STATE_MACHINE=1",
        f"{shlex.quote(str(args.python))} - <<'PY'",
        "from pathlib import Path",
        "from scripts import run_em_training as em",
        f"result = em.run_prompt_student_mstep(1, Path({str(paths['mstep_manifest'])!r}))",
        "raise SystemExit(0 if result.get('status') == 'success' else 2)",
        "PY",
        "",
    ]
    paths["mstep_sbatch"].write_text("\n".join(lines), encoding="utf-8")
    paths["mstep_sbatch"].chmod(0o755)
    for command in (["bash", "-n", str(paths["mstep_sbatch"])], ["sbatch", "--test-only", str(paths["mstep_sbatch"])]):
        result = _run(command)
        if not result["ok"]:
            log_failure(state_root, stage="m_step_sbatch_preflight", failure_reason="mstep_sbatch_preflight_failed", details=result)
            return {"status": "FAILED", "failure_reason": "mstep_sbatch_preflight_failed", "preflight": result}
    result = _run(["sbatch", "--parsable", str(paths["mstep_sbatch"])])
    if not result["ok"]:
        log_failure(state_root, stage="m_step_submit", failure_reason=result["stderr"] or "mstep_submit_failed", details=result)
        return {"status": "FAILED", "failure_reason": result["stderr"], "submit": result}
    job_id = result["stdout"].splitlines()[-1].strip()
    paths["mstep_job"].write_text(job_id + "\n", encoding="utf-8")
    _save_state(state_root, mstep_status="SUBMITTED", mstep_job_id=job_id, mstep_manifest=manifest)
    return {"status": "SUBMITTED", "job_id": job_id, "manifest": manifest}


def check_mstep(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    result_json = paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    result = _read_json(result_json, {})
    if result.get("status") == "success":
        checkpoint = paths["mstep_output"] / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth"
        if checkpoint.exists():
            return {"status": "PASSED", "result": str(result_json), "checkpoint": str(checkpoint), "failure_reason": ""}
        payload = {"status": "FAILED", "result": str(result_json), "checkpoint": str(checkpoint), "failure_reason": "checkpoint_missing"}
        log_failure(args.state_root.resolve(), stage="m_step", failure_reason="checkpoint_missing", details=payload)
        return payload
    state = _load_state(args.state_root.resolve())
    job_id = str(state.get("mstep_job_id") or "")
    job_state = slurm_job_state(job_id) if job_id else {"state": "UNKNOWN"}
    if job_state.get("state") in TERMINAL_FAILURE_STATES:
        log_failure(args.state_root.resolve(), stage="m_step", failure_reason=f"mstep_job_terminal:{job_state.get('state')}", details=job_state)
        return {"status": "FAILED", "failure_reason": f"mstep_job_terminal:{job_state.get('state')}", "job": job_state}
    return {"status": "RUNNING", "job": job_state}


def run_round1_final_validator(args: argparse.Namespace) -> dict[str, Any]:
    paths = _state_paths(args.state_root.resolve())
    state = _load_state(args.state_root.resolve())
    mstep_result_path = paths["root"] / "round1" / "mstep" / "voxtell_prompt_mstep_result.json"
    mstep_result = _read_json(mstep_result_path, {})
    checkpoint = Path(str(mstep_result.get("inference_checkpoint") or paths["root"] / "round1" / "mstep" / "voxtell_finetuned_model" / "fold_0" / "checkpoint_final.pth"))
    formal_root = Path(str(state.get("formal_root") or ""))
    formal_status = _read_json(formal_root / "task2_formal_summary.json", {})
    formal_validator_status = formal_status.get("status") or formal_status.get("validation_status")
    checks = [
        {"name": "e_step_status_passed", "ok": state.get("e_step_status") == "PASSED", "detail": state.get("e_step_status")},
        {"name": "formal_validator_passed", "ok": bool(formal_validator_status in {"PASSED", "passed", "success"}), "detail": formal_validator_status or "missing_task2_formal_summary"},
        {"name": "mstep_result_success", "ok": mstep_result.get("status") == "success", "detail": str(mstep_result_path)},
        {"name": "mstep_checkpoint_exists", "ok": checkpoint.is_file(), "detail": str(checkpoint)},
        {
            "name": "student_checkpoint_eligible",
            "ok": bool(mstep_result.get("eligible_for_next_round_prompt_student") or mstep_result.get("checkpoint_eligible_for_next_round")),
            "detail": mstep_result.get("training_status"),
        },
    ]
    report = {
        "status": "PASSED" if all(check["ok"] for check in checks) else "FAILED",
        "terminal_state": "ROUND1_PASSED" if all(check["ok"] for check in checks) else "ROUND1_FAILED",
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "state_root": str(paths["root"]),
        "formal_root": str(formal_root),
        "mstep_result": str(mstep_result_path),
        "checks": checks,
        "failure_reason": "" if all(check["ok"] for check in checks) else ",".join(check["name"] for check in checks if not check["ok"]),
    }
    _write_json(paths["final"], report)
    if report["status"] != "PASSED":
        log_failure(args.state_root.resolve(), stage="final_validator", failure_reason=report["failure_reason"], details=report)
    return report


def _controller_main(args: argparse.Namespace) -> int:
    state_root = args.state_root.resolve()
    current = _load_state(state_root)
    if current.get("terminal_state") == "ROUND1_PASSED":
        return 0
    if current.get("terminal_state") == "ROUND1_FAILED":
        return 2
    _save_state(state_root, status="CONTROLLER_RUNNING", terminal_state="", stage="start", git_commit=_git_commit(), started_at=utc_now())
    labelcritic = ensure_labelcritic_service(state_root)
    if labelcritic["status"] == "SUBMIT_FAILED":
        log_failure(state_root, stage="labelcritic_submit", failure_reason=labelcritic.get("failure_reason", "labelcritic_submit_failed"), details=labelcritic)
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="labelcritic_submit", failure_reason=labelcritic.get("failure_reason"), labelcritic=labelcritic)
        return 2
    _save_state(state_root, stage="labelcritic_submitted_or_reused", labelcritic=labelcritic)
    estep_submit = submit_estep(args, labelcritic)
    if estep_submit["status"] == "FAILED":
        log_failure(state_root, stage="e_step_submit", failure_reason=estep_submit.get("failure_reason", "e_step_submit_failed"), details=estep_submit)
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="e_step_submit", failure_reason=estep_submit.get("failure_reason"), e_step=estep_submit)
        return 2
    labelcritic_gate: dict[str, Any] = {"status": "WAITING"}
    while True:
        if labelcritic_gate.get("status") != "PASSED":
            labelcritic_gate = poll_labelcritic_runtime(state_root)
            if labelcritic_gate["status"] == "FAILED":
                log_failure(state_root, stage="labelcritic", failure_reason=labelcritic_gate.get("failure_reason", "labelcritic_failed"), details=labelcritic_gate)
                _save_state(state_root, terminal_state="ROUND1_FAILED", stage="labelcritic", failure_reason=labelcritic_gate.get("failure_reason"), labelcritic=labelcritic_gate)
                return 2
            if labelcritic_gate["status"] == "PASSED":
                _save_state(state_root, stage="labelcritic_ready", labelcritic=labelcritic_gate)
        estep = check_estep(args)
        wait_stage = "e_step_wait" if labelcritic_gate.get("status") == "PASSED" else "e_step_and_labelcritic_wait"
        _save_state(state_root, stage=wait_stage, e_step=estep, e_step_status=estep["status"], labelcritic=labelcritic_gate)
        if estep["status"] == "PASSED":
            if labelcritic_gate.get("status") == "PASSED":
                break
            time.sleep(max(10, args.poll_sec))
            continue
        if estep["status"] == "FAILED":
            log_failure(state_root, stage="e_step", failure_reason=estep.get("failure_reason", "e_step_failed"), details=estep)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="e_step", failure_reason=estep.get("failure_reason"), e_step=estep)
            return 2
        time.sleep(max(10, args.poll_sec))
    mstep_submit = submit_mstep(args)
    if mstep_submit["status"] == "FAILED":
        log_failure(state_root, stage="m_step_submit", failure_reason=mstep_submit.get("failure_reason", "m_step_submit_failed"), details=mstep_submit)
        _save_state(state_root, terminal_state="ROUND1_FAILED", stage="m_step_submit", failure_reason=mstep_submit.get("failure_reason"), m_step=mstep_submit)
        return 2
    while True:
        mstep = check_mstep(args)
        _save_state(state_root, stage="m_step_wait", m_step=mstep, mstep_status=mstep["status"])
        if mstep["status"] == "PASSED":
            final = run_round1_final_validator(args)
            if final["status"] == "PASSED":
                _save_state(state_root, terminal_state="ROUND1_PASSED", stage="complete", final=final, finished_at=utc_now())
                return 0
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="final_validator", failure_reason=final.get("failure_reason"), final=final, finished_at=utc_now())
            return 2
        if mstep["status"] == "FAILED":
            log_failure(state_root, stage="m_step", failure_reason=mstep.get("failure_reason", "m_step_failed"), details=mstep)
            _save_state(state_root, terminal_state="ROUND1_FAILED", stage="m_step", failure_reason=mstep.get("failure_reason"), m_step=mstep)
            return 2
        time.sleep(max(10, args.poll_sec))


def controller(args: argparse.Namespace) -> int:
    try:
        return _controller_main(args)
    except Exception as exc:
        state_root = args.state_root.resolve()
        failure = log_failure(
            state_root,
            stage="controller_unhandled_exception",
            failure_reason=f"{type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__, "exception": str(exc)},
        )
        _save_state(
            state_root,
            terminal_state="ROUND1_FAILED",
            stage="controller_unhandled_exception",
            failure_reason=failure["failure_reason"],
            last_failure=str(_state_paths(state_root)["last_failure"]),
        )
        return 2


def status(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_state(args.state_root.resolve())
    paths = _state_paths(args.state_root.resolve())
    last_failure = _read_json(paths["last_failure"], {})
    return {
        "status": state.get("terminal_state") or state.get("stage") or state.get("status") or "NOT_STARTED",
        "state_root": str(paths["root"]),
        "state_json": str(paths["state"]),
        "events_log": str(paths["events"]),
        "failure_log": str(paths["failures"]),
        "last_failure_json": str(paths["last_failure"]),
        "controller_job_id": state.get("controller_job_id"),
        "labelcritic": state.get("labelcritic"),
        "e_step_status": state.get("e_step_status"),
        "mstep_status": state.get("mstep_status"),
        "failure_reason": state.get("failure_reason", "") or last_failure.get("failure_reason", ""),
        "last_failure": last_failure,
        "formal_root": state.get("formal_root", ""),
        "updated_at": state.get("updated_at", ""),
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", default=os.getenv("STATE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state"), type=Path)
    parser.add_argument("--workspace-root", default=os.getenv("WORKSPACE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812"), type=Path)
    parser.add_argument("--case-manifest", default=os.getenv("CASE_MANIFEST", "/projects/bodymaps/users/xhan74/medical_agent/workspaces/abdomenatlaspro_103_round1_20260812/manifests/cases_103_manifest.csv"), type=Path)
    parser.add_argument("--base-manifest", default=os.getenv("BASE_MANIFEST", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/generated_labels_100cases_work/cases_100_manifest.csv"), type=Path)
    parser.add_argument("--python", default=os.getenv("PYTHON", "/home/xhan74/envs/medical_agent/bin/python"), type=Path)
    parser.add_argument("--checkpoint-root", default=os.getenv("CHECKPOINT_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints"), type=Path)
    parser.add_argument("--nnunet-predict-executable", default=os.getenv("NNUNETV2_PREDICT_EXECUTABLE", "/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict"), type=Path)
    parser.add_argument("--unest-python-executable", default=os.getenv("UNEST_PYTHON_EXECUTABLE", "/home/xhan74/envs/medical_agent_train_py311/bin/python"), type=Path)
    parser.add_argument("--registry", default=REPO_ROOT / "configs" / "model_registry.yaml", type=Path)
    parser.add_argument("--target-config", default=REPO_ROOT / "configs" / "student_3d_prompt_target_organs.json", type=Path)
    parser.add_argument("--gpu-target-workers", default=int(os.getenv("GPU_TARGET_WORKERS", "0")), type=int)
    parser.add_argument("--gpu-overrequest-workers", default=int(os.getenv("GPU_OVERREQUEST_WORKERS", "0")), type=int)
    parser.add_argument("--gpu-profile-specs", default=os.getenv("GPU_PROFILE_SPECS", "auto"))
    parser.add_argument("--poll-sec", default=int(os.getenv("ROUND1_ORCH_POLL_SEC", "60")), type=int)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unattended Task2 103-case Round1 orchestrator.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    submit_p = sub.add_parser("submit-controller")
    add_common(submit_p)
    submit_p.add_argument("--controller-partition", default=os.getenv("CONTROLLER_PARTITION", "cpu"))
    submit_p.add_argument("--controller-cpus", default=int(os.getenv("CONTROLLER_CPUS", "2")), type=int)
    submit_p.add_argument("--controller-mem", default=os.getenv("CONTROLLER_MEM", "8G"))
    submit_p.add_argument("--controller-time", default=os.getenv("CONTROLLER_TIME", "48:00:00"))
    submit_p.add_argument("--labelcritic-partition", default=os.getenv("LABELCRITIC_PARTITION", "gpuh100"))
    submit_p.add_argument("--labelcritic-gres", default=os.getenv("LABELCRITIC_GRES", "gpu:H100:2"))
    submit_p.add_argument("--labelcritic-tensor-parallel-size", default=int(os.getenv("LABELCRITIC_TENSOR_PARALLEL_SIZE", "2")), type=int)
    submit_p.add_argument("--labelcritic-port", default=int(os.getenv("LABELCRITIC_PORT", "8000")), type=int)
    submit_p.add_argument("--skip-sbatch-test-only", action="store_true")
    submit_p.add_argument("--run-static-tests", default=os.getenv("RUN_STATIC_PREFLIGHT_TESTS", "1").lower() not in {"0", "false", "no"}, action=argparse.BooleanOptionalAction)
    submit_p.add_argument("--static-tests-timeout-sec", default=int(os.getenv("STATIC_PREFLIGHT_TESTS_TIMEOUT_SEC", "900")), type=int)
    submit_p.add_argument("--retry-failed", default=os.getenv("RETRY_FAILED", "1").lower() not in {"0", "false", "no"}, action=argparse.BooleanOptionalAction)
    submit_p.add_argument("--new-attempt", default=os.getenv("NEW_ATTEMPT", "0").lower() in {"1", "true", "yes"}, action=argparse.BooleanOptionalAction)
    controller_p = sub.add_parser("controller")
    add_common(controller_p)
    status_p = sub.add_parser("status")
    status_p.add_argument("--state-root", default=os.getenv("STATE_ROOT", "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state"), type=Path)
    args = ap.parse_args()
    if args.cmd == "submit-controller":
        print(json.dumps(submit_controller(args), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "controller":
        return controller(args)
    if args.cmd == "status":
        print(json.dumps(status(args), indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
