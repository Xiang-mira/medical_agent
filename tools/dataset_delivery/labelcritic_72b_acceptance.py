#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import utc_now  # noqa: E402
from tools.dataset_delivery.labelcritic_service_contract import (  # noqa: E402
    apptainer_exec_base,
    validate_gpu_topology,
    validate_local_model,
    validate_model_endpoint,
)
from tools.dataset_delivery.task2_round1_orchestrator import _git_commit  # noqa: E402


MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
MODEL_DIR = "/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ"
CONTAINER = "/home/xhan74/containers/vllm-openai-v0.19.1.sif"
ACCEPTANCE_ROOT = "/projects/bodymaps/users/xhan74/medical_agent/outputs/dataset_delivery_373/runtime_state/labelcritic_72b_acceptance"
ONE_BY_ONE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _run(command: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        return {"command": command, "return_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "ok": proc.returncode == 0}
    except Exception as exc:
        return {"command": command, "return_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "ok": False}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_bytes()[-limit:]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _acceptance_spec_from_submit_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "labelcritic_72b_acceptance_spec_v1",
        "container": str(args.container),
        "model_dir": str(args.model_dir),
        "model_id": str(args.model_id),
        "vllm_python": str(args.vllm_python),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": str(args.gpu_memory_utilization),
        "startup_timeout_sec": int(args.startup_timeout_sec),
        "stability_sec": int(args.stability_sec),
        "partition": str(args.partition),
        "gres": str(args.gres),
        "cpus": int(args.cpus),
        "mem": str(args.mem),
        "time_limit": str(args.time_limit),
    }


def _acceptance_spec_from_runtime(validation_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    spec = _read_json(validation_root.parent / "acceptance_spec.json", {})
    if isinstance(spec, dict) and spec:
        return spec
    return _acceptance_spec_from_submit_args(args)


def _acceptance_spec_hash(spec: dict[str, Any]) -> str:
    normalized = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _acceptance_spec_paths(acceptance_root: Path) -> dict[str, Path]:
    return {
        "spec_json": acceptance_root / "acceptance_spec.json",
        "spec_hash": acceptance_root / "acceptance_spec_hash.txt",
        "authoritative_root": acceptance_root / "authoritative",
    }


def _write_authoritative_acceptance_marker(validation_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    spec_hash = str(report.get("acceptance_spec_hash") or "").strip()
    if not spec_hash or report.get("final_status") != "PASS":
        return {"status": "SKIPPED", "reason": "missing_spec_hash_or_non_pass"}
    marker = validation_root.parent / "authoritative" / f"{spec_hash}.json"
    payload = {
        "schema_version": "labelcritic_72b_acceptance_authoritative_v1",
        "timestamp": utc_now(),
        "acceptance_spec_hash": spec_hash,
        "final_status": str(report.get("final_status") or ""),
        "job_id": str(report.get("job_id") or ""),
        "validation_root": str(validation_root),
        "report": str(validation_root / "acceptance_report.json"),
        "acceptance_report": report,
    }
    _write_json(marker, payload)
    return {"status": "READY", "path": str(marker), "acceptance_spec_hash": spec_hash}


def _gpu_memory_snapshot() -> dict[str, Any]:
    result = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], timeout=30)
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "memory_total_mb": int(float(parts[1])), "memory_used_mb": int(float(parts[2]))})
    return {"status": "PASS" if result["ok"] and rows else "FAIL", "rows": rows, "raw": result}


def _chat_request(base_url: str, port: int, model_id: str, root: Path) -> dict[str, Any]:
    image_url = f"data:image/png;base64,{ONE_BY_ONE_PNG}"
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are LabelCritic. This is a formal runtime acceptance request with two mask images. Reply exactly: MASK1"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    _write_json(root / "request.json", payload)
    req = urllib.request.Request(f"{base_url}:{port}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as response:
        doc = json.loads(response.read().decode("utf-8"))
    _write_json(root / "response.json", doc)
    text = str((((doc.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
    parser_ok = "MASK1" in text.upper() or text.strip() == "1"
    return {
        "status": "PASS" if parser_ok else "FAIL",
        "http_success": True,
        "response_non_empty": bool(text.strip()),
        "parser_ok": parser_ok,
        "schema_ok": isinstance(doc.get("choices"), list),
        "required_fields_present": bool(doc.get("choices")),
        "not_mock": True,
        "not_fallback": True,
        "response_text": text[:500],
        "failure_reason": "" if parser_ok else "response_parser_failed",
    }


def _finish(root: Path, report: dict[str, Any]) -> int:
    _write_json(root / "acceptance_report.json", report)
    if report.get("final_status") == "PASS":
        _write_authoritative_acceptance_marker(root, report)
    md = [
        "# LabelCritic 72B Acceptance",
        "",
        f"- final_status: {report.get('final_status')}",
        f"- failed_gate: {report.get('failed_gate')}",
        f"- failure_reason: {report.get('failure_reason')}",
        f"- job_id: {report.get('job_id')}",
        f"- node: {report.get('node')}",
    ]
    (root / "acceptance_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0 if report.get("final_status") == "PASS" else 2


def run_job(args: argparse.Namespace) -> int:
    root = args.validation_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = _acceptance_spec_from_runtime(root, args)
    spec_hash = _acceptance_spec_hash(spec)
    stdout_log = root / "vllm.stdout.log"
    stderr_log = root / "vllm.stderr.log"
    report: dict[str, Any] = {
        "schema_version": "labelcritic_72b_acceptance_v1",
        "timestamp": utc_now(),
        "git_commit": _git_commit(),
        "acceptance_spec_hash": spec_hash,
        "acceptance_spec": spec,
        "job_id": os.getenv("SLURM_JOB_ID", ""),
        "node": os.getenv("SLURMD_NODENAME", os.getenv("HOSTNAME", "")),
        "model": args.model_id,
        "model_path": str(args.model_dir),
        "container": str(args.container),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gate_1_model_files": {"status": "FAIL"},
        "gate_2_h100_tp": {"status": "FAIL"},
        "gate_3_vllm_stability": {"status": "FAIL"},
        "gate_4_real_request": {"status": "FAIL"},
        "final_status": "FAIL",
        "failed_gate": "",
        "failure_reason": "",
    }
    process: subprocess.Popen[str] | None = None
    try:
        gate1 = validate_local_model(args.container, args.model_dir, args.vllm_python)
        report["gate_1_model_files"] = gate1
        if gate1.get("status") != "PASS":
            report.update({"failed_gate": "GATE_1_MODEL_FILES", "failure_reason": gate1.get("failure_reason", "model_files_failed")})
            return _finish(root, report)

        (root / "gpu_snapshot_before.txt").write_text(json.dumps(_gpu_memory_snapshot(), indent=2) + "\n", encoding="utf-8")
        gate2 = validate_gpu_topology(args.container, args.vllm_python, expected_count=2, expected_type="H100", tp=args.tensor_parallel_size)
        report["gate_2_h100_tp"] = gate2
        if gate2.get("status") != "PASS":
            report.update({"failed_gate": "GATE_2_H100_TP", "failure_reason": gate2.get("failure_reason", "gpu_topology_failed")})
            return _finish(root, report)

        port = _free_port()
        base_url = "http://127.0.0.1"
        command = [
            *apptainer_exec_base(args.container, args.vllm_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(args.model_dir),
            "--served-model-name",
            args.model_id,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
        ]
        report["gate_3_vllm_stability"]["command"] = command
        started = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=REPO_ROOT, text=True, stdout=out, stderr=err)
            deadline = time.time() + int(args.startup_timeout_sec)
            ready = False
            endpoint = {"status": "FAIL", "failure_reason": "startup_timeout"}
            while time.time() < deadline:
                if process.poll() is not None:
                    endpoint = {"status": "FAIL", "failure_reason": f"vllm_exited_before_ready:{process.returncode}"}
                    break
                endpoint = validate_model_endpoint(base_url, port, args.model_id, timeout=10)
                if endpoint.get("status") == "PASS":
                    ready = True
                    break
                time.sleep(10)
            if not ready:
                report["gate_3_vllm_stability"] = {"status": "FAIL", "endpoint": endpoint, "stderr_tail": _tail(stderr_log), "stdout_tail": _tail(stdout_log)}
                report.update({"failed_gate": "GATE_3_VLLM_STABILITY", "failure_reason": endpoint.get("failure_reason", "vllm_not_ready")})
                return _finish(root, report)
            loaded_snapshot = _gpu_memory_snapshot()
            (root / "gpu_snapshot_loaded.txt").write_text(json.dumps(loaded_snapshot, indent=2) + "\n", encoding="utf-8")
            stability_deadline = time.time() + int(args.stability_sec)
            alive_checks = 0
            while time.time() < stability_deadline:
                if process.poll() is not None:
                    report["gate_3_vllm_stability"] = {"status": "FAIL", "failure_reason": f"vllm_died_during_stability:{process.returncode}", "stderr_tail": _tail(stderr_log)}
                    report.update({"failed_gate": "GATE_3_VLLM_STABILITY", "failure_reason": "vllm_died_during_stability"})
                    return _finish(root, report)
                endpoint = validate_model_endpoint(base_url, port, args.model_id, timeout=10)
                if endpoint.get("status") != "PASS":
                    report["gate_3_vllm_stability"] = {"status": "FAIL", "endpoint": endpoint}
                    report.update({"failed_gate": "GATE_3_VLLM_STABILITY", "failure_reason": endpoint.get("failure_reason", "http_failed_during_stability")})
                    return _finish(root, report)
                alive_checks += 1
                time.sleep(10)
            report["gate_3_vllm_stability"] = {
                "status": "PASS",
                "startup_time_sec": round(time.time() - started - int(args.stability_sec), 3),
                "stability_sec": int(args.stability_sec),
                "alive_checks": alive_checks,
                "endpoint": {"base_url": base_url, "port": port},
                "gpu_memory_loaded": loaded_snapshot,
            }

            try:
                gate4 = _chat_request(base_url, port, args.model_id, root)
            except Exception as exc:
                gate4 = {"status": "FAIL", "failure_reason": f"{type(exc).__name__}: {exc}", "http_success": False, "parser_ok": False, "schema_ok": False}
            report["gate_4_real_request"] = gate4
            if gate4.get("status") != "PASS":
                report.update({"failed_gate": "GATE_4_REAL_REQUEST", "failure_reason": gate4.get("failure_reason", "real_request_failed")})
                return _finish(root, report)
        report.update({"final_status": "PASS", "failed_gate": "", "failure_reason": ""})
        return _finish(root, report)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


def render_sbatch(args: argparse.Namespace, validation_root: Path) -> Path:
    path = validation_root / "labelcritic_72b_acceptance.sbatch"
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=labelcritic_72b_acceptance",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --gres={args.gres}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={validation_root / 'slurm.out'}",
        f"#SBATCH --error={validation_root / 'slurm.err'}",
        "#SBATCH --export=ALL",
        "",
        "set -euo pipefail",
        "unset DISPLAY GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS",
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"{shlex.quote(sys.executable)} tools/dataset_delivery/labelcritic_72b_acceptance.py run-job "
        f"--validation-root {shlex.quote(str(validation_root))} "
        f"--container {shlex.quote(str(args.container))} "
        f"--model-dir {shlex.quote(str(args.model_dir))} "
        f"--model-id {shlex.quote(args.model_id)} "
        f"--vllm-python {shlex.quote(args.vllm_python)} "
        f"--tensor-parallel-size {args.tensor_parallel_size} "
        f"--max-model-len {args.max_model_len} "
        f"--gpu-memory-utilization {shlex.quote(str(args.gpu_memory_utilization))} "
        f"--startup-timeout-sec {args.startup_timeout_sec} "
        f"--stability-sec {args.stability_sec}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def submit(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    acceptance_root = Path(args.acceptance_root).expanduser()
    validation_root = acceptance_root / timestamp
    validation_root.mkdir(parents=True, exist_ok=True)
    spec = _acceptance_spec_from_submit_args(args)
    spec_hash = _acceptance_spec_hash(spec)
    spec_paths = _acceptance_spec_paths(acceptance_root)
    _write_json(spec_paths["spec_json"], spec)
    spec_paths["spec_hash"].write_text(spec_hash + "\n", encoding="utf-8")
    sbatch = render_sbatch(args, validation_root)
    for command in (["bash", "-n", str(sbatch)], ["sbatch", "--test-only", str(sbatch)]):
        result = _run(command, timeout=120)
        if not result["ok"]:
            print(json.dumps({"status": "FAIL", "stage": "sbatch_preflight", "command": command, "result": result, "validation_root": str(validation_root)}, indent=2))
            return 2
    result = _run(["sbatch", "--parsable", str(sbatch)], timeout=120)
    if not result["ok"]:
        print(json.dumps({"status": "FAIL", "stage": "sbatch_submit", "result": result, "validation_root": str(validation_root)}, indent=2))
        return 2
    job_id = result["stdout"].strip().splitlines()[-1]
    (validation_root / "job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    deadline = time.time() + int(args.wait_timeout_sec)
    report_path = validation_root / "acceptance_report.json"
    while time.time() < deadline:
        if report_path.exists():
            report = _read_json(report_path, {})
            print_acceptance_summary(report, validation_root, job_id)
            return 0 if report.get("final_status") == "PASS" else 2
        time.sleep(30)
    print_acceptance_summary({"final_status": "FAIL", "failure_reason": "acceptance_wait_timeout", "job_id": job_id}, validation_root, job_id)
    return 2


def print_acceptance_summary(report: dict[str, Any], root: Path, job_id: str = "") -> None:
    print("LABELCRITIC_72B_ACCEPTANCE")
    print(f"job_id={report.get('job_id') or job_id}")
    print(f"node={report.get('node', '')}")
    for key, label in (
        ("gate_1_model_files", "GATE_1_MODEL_FILES"),
        ("gate_2_h100_tp", "GATE_2_H100_TP"),
        ("gate_3_vllm_stability", "GATE_3_VLLM_STABILITY"),
        ("gate_4_real_request", "GATE_4_REAL_REQUEST"),
    ):
        print(f"{label}={(report.get(key) or {}).get('status', 'FAIL')}")
    print(f"FINAL={report.get('final_status', 'FAIL')}")
    if report.get("failure_reason"):
        print(f"failure_reason={report.get('failure_reason')}")
    print(f"report={root / 'acceptance_report.json'}")
    print(f"stdout={root / 'slurm.out'}")
    print(f"stderr={root / 'slurm.err'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelCritic 72B 2xH100 full acceptance harness.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    submit_p = sub.add_parser("submit")
    submit_p.add_argument("--acceptance-root", default=os.getenv("LABELCRITIC_ACCEPTANCE_ROOT", ACCEPTANCE_ROOT))
    submit_p.add_argument("--partition", default=os.getenv("LABELCRITIC_PARTITION", "gpuh100"))
    submit_p.add_argument("--gres", default=os.getenv("LABELCRITIC_GRES", "gpu:H100:2"))
    submit_p.add_argument("--cpus", default=int(os.getenv("LABELCRITIC_CPUS", "16")), type=int)
    submit_p.add_argument("--mem", default=os.getenv("LABELCRITIC_MEM", "192G"))
    submit_p.add_argument("--time-limit", default=os.getenv("LABELCRITIC_ACCEPTANCE_TIME", "04:00:00"))
    submit_p.add_argument("--container", default=os.getenv("VLLM_CONTAINER", CONTAINER), type=Path)
    submit_p.add_argument("--model-dir", default=os.getenv("LABELCRITIC_MODEL_DIR", MODEL_DIR), type=Path)
    submit_p.add_argument("--model-id", default=os.getenv("LABELCRITIC_MODEL_ID", MODEL_ID))
    submit_p.add_argument("--vllm-python", default=os.getenv("VLLM_PYTHON", "python3"))
    submit_p.add_argument("--tensor-parallel-size", default=int(os.getenv("LABELCRITIC_TENSOR_PARALLEL_SIZE", "2")), type=int)
    submit_p.add_argument("--max-model-len", default=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")), type=int)
    submit_p.add_argument("--gpu-memory-utilization", default=os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.88"))
    submit_p.add_argument("--startup-timeout-sec", default=int(os.getenv("LABELCRITIC_ACCEPTANCE_STARTUP_TIMEOUT_SEC", "1800")), type=int)
    submit_p.add_argument("--stability-sec", default=int(os.getenv("LABELCRITIC_ACCEPTANCE_STABILITY_SEC", "60")), type=int)
    submit_p.add_argument("--wait-timeout-sec", default=int(os.getenv("LABELCRITIC_ACCEPTANCE_WAIT_TIMEOUT_SEC", "14400")), type=int)
    run_p = sub.add_parser("run-job")
    run_p.add_argument("--validation-root", required=True, type=Path)
    run_p.add_argument("--container", required=True, type=Path)
    run_p.add_argument("--model-dir", required=True, type=Path)
    run_p.add_argument("--model-id", default=MODEL_ID)
    run_p.add_argument("--vllm-python", default="python3")
    run_p.add_argument("--tensor-parallel-size", default=2, type=int)
    run_p.add_argument("--max-model-len", default=8192, type=int)
    run_p.add_argument("--gpu-memory-utilization", default="0.88")
    run_p.add_argument("--startup-timeout-sec", default=1800, type=int)
    run_p.add_argument("--stability-sec", default=60, type=int)
    args = parser.parse_args()
    return run_job(args) if args.cmd == "run-job" else submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
