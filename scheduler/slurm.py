from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .registry import enforce_student_single_gpu
from .utils import SchedulerError


DIRECTIVE_KEYS = [
    "job-name",
    "account",
    "partition",
    "qos",
    "nodes",
    "ntasks",
    "ntasks-per-node",
    "cpus-per-task",
    "mem",
    "time",
    "gres",
    "constraint",
    "array",
    "output",
    "error",
]
SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:%=,+@{}-]+$")


def _directive(key: str, value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value)
    if not SAFE_VALUE_RE.match(text):
        raise SchedulerError(f"Unsafe SBATCH value for {key}: {text!r}")
    return f"#SBATCH --{key}={text}"


def profile_to_directives(task: dict[str, Any], profile: dict[str, Any], *, run_dir: Path) -> list[str]:
    if task.get("task_type") == "student_train":
        enforce_student_single_gpu(profile)
    directives: dict[str, Any] = {
        "job-name": task.get("task_name"),
        "account": profile.get("account"),
        "partition": profile.get("partition"),
        "qos": profile.get("qos"),
        "nodes": profile.get("nodes", 1),
        "ntasks": profile.get("ntasks", 1),
        "ntasks-per-node": profile.get("ntasks_per_node"),
        "cpus-per-task": profile.get("cpus_per_task"),
        "mem": f"{profile['memory_gb']}G" if profile.get("memory_gb") else None,
        "time": profile.get("walltime"),
        "constraint": profile.get("constraint"),
        "output": run_dir / "logs" / f"{task.get('task_name')}_%A_%a.out" if task.get("array") else run_dir / "logs" / f"{task.get('task_name')}_%j.out",
        "error": run_dir / "logs" / f"{task.get('task_name')}_%A_%a.err" if task.get("array") else run_dir / "logs" / f"{task.get('task_name')}_%j.err",
    }
    if task.get("array"):
        directives["array"] = task["array"]
    if profile.get("gpu_count"):
        directives["gres"] = f"gpu:{int(profile['gpu_count'])}"
    elif task.get("requires_gpu"):
        raise SchedulerError(f"Task {task.get('task_name')} requires GPU but profile has gpu_count=0")
    return [line for key in DIRECTIVE_KEYS if (line := _directive(key, directives.get(key)))]


def shell_join(args: list[str]) -> str:
    rendered = []
    for arg in args:
        text = str(arg)
        if re.fullmatch(r"(\$[A-Z0-9_]+|\$\{[A-Za-z0-9_:-]+\})(/[A-Za-z0-9_./-]+)?", text):
            rendered.append(text)
        else:
            rendered.append(shlex.quote(text))
    return " ".join(rendered)


def render_task_script(task: dict[str, Any], profile: dict[str, Any], command: list[str], *, run_dir: Path) -> str:
    directives = profile_to_directives(task, profile, run_dir=run_dir)
    rendered_command = command
    if profile.get("launcher") == "torchrun" or task.get("launcher") == "torchrun":
        nproc = int(profile.get("gpu_count") or 1)
        rendered_command = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", *command]
    body = [
        "#!/bin/bash",
        *directives,
        "",
        "set -euo pipefail",
        'echo "[scheduler] task=' + str(task.get("task_name")) + ' job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-none}"',
        'echo "[scheduler] host=$(hostname) cwd=$(pwd)"',
        'echo "[scheduler] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true",
        f"mkdir -p {shlex.quote(str(run_dir / 'logs'))} {shlex.quote(str(run_dir / 'status'))}",
        shell_join(rendered_command),
        "",
    ]
    return "\n".join(body)


def render_labelcritic_script(task: dict[str, Any], profile: dict[str, Any], client_command: list[str], *, run_dir: Path) -> str:
    directives = profile_to_directives(task, profile, run_dir=run_dir)
    model_dir = shlex.quote(str(profile.get("model_dir", "")))
    model_id = shlex.quote(str(profile.get("model_id", "")))
    tp = int(profile.get("tensor_parallel_size") or profile.get("gpu_count") or 1)
    port = int(profile.get("port") or 8000)
    container = str(profile.get("container") or "")
    gpu_memory_utilization = profile.get("gpu_memory_utilization")
    max_model_len = profile.get("max_model_len")
    if not container:
        raise SchedulerError("LabelCritic profile requires a vLLM container path")
    extra_args = []
    if gpu_memory_utilization:
        extra_args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if max_model_len:
        extra_args.extend(["--max-model-len", str(max_model_len)])
    launch = (
        f"apptainer exec --nv {shlex.quote(container)} python -m vllm.entrypoints.openai.api_server "
        f"--model {model_dir} --served-model-name {model_id} --tensor-parallel-size {tp} --host 127.0.0.1 --port {port} "
        f"{shell_join(extra_args)}"
    )
    body = [
        "#!/bin/bash",
        *directives,
        "",
        "set -euo pipefail",
        "VLLM_PID=",
        "cleanup() {",
        '  if [[ -n "${VLLM_PID:-}" ]]; then kill "${VLLM_PID}" >/dev/null 2>&1 || true; fi',
        "}",
        "trap cleanup EXIT",
        "export NO_PROXY=127.0.0.1,localhost,::1",
        "export no_proxy=127.0.0.1,localhost,::1",
        'echo "[scheduler] LabelCritic tier=' + str(profile.get("tier", task.get("resource_profile"))) + ' job=${SLURM_JOB_ID:-local}"',
        'echo "[scheduler] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true",
        f"{launch} > {shlex.quote(str(run_dir / 'logs'))}/labelcritic_vllm_${{SLURM_JOB_ID:-local}}.log 2>&1 &",
        "VLLM_PID=$!",
        f"for i in $(seq 1 120); do curl --noproxy '*' -fsS http://127.0.0.1:{port}/health >/dev/null && break; sleep 5; done",
        f"curl --noproxy '*' -fsS http://127.0.0.1:{port}/health >/dev/null",
        shell_join(client_command),
        "",
    ]
    return "\n".join(body)
