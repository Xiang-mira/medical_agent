#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import utc_now


LABELCRITIC_LAUNCHER_VERSION = "labelcritic_vllm_python3_spec_v1"
DEFAULT_PROJECT_BIND = "/projects/bodymaps/users/xhan74/medical_agent:/projects/bodymaps/users/xhan74/medical_agent"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run(command: list[str], *, timeout: int = 120, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout, env=env)
        return {
            "command": command,
            "return_code": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except Exception as exc:
        return {"command": command, "return_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "ok": False}


def project_bind_spec() -> str:
    return os.getenv("LABELCRITIC_PROJECT_BIND", DEFAULT_PROJECT_BIND)


def apptainer_exec_base(container: str | Path, python_executable: str, *, nv: bool = True) -> list[str]:
    command = ["apptainer", "exec"]
    if nv:
        command.append("--nv")
    bind = project_bind_spec()
    if bind:
        command.extend(["--bind", bind])
    command.extend([str(container), str(python_executable)])
    return command


def service_spec_from_env(*, generated_script: str = "") -> dict[str, Any]:
    gres = os.getenv("LABELCRITIC_GRES", "gpu:H100:2")
    gpu_count = 2
    try:
        gpu_count = int(str(gres).rsplit(":", 1)[-1])
    except Exception:
        pass
    return {
        "schema_version": "labelcritic_service_spec_v1",
        "container_path": os.getenv("VLLM_CONTAINER", "/home/xhan74/containers/vllm-openai-v0.19.1.sif"),
        "VLLM_PYTHON": os.getenv("VLLM_PYTHON", "python3"),
        "model_path": os.getenv("LABELCRITIC_MODEL_DIR", "/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ"),
        "served_model_name": os.getenv("LABELCRITIC_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct-AWQ"),
        "project_bind": project_bind_spec(),
        "tensor_parallel_size": int(os.getenv("LABELCRITIC_TENSOR_PARALLEL_SIZE", "2")),
        "partition": os.getenv("LABELCRITIC_PARTITION", "gpuh100"),
        "gres": gres,
        "gpu_count": gpu_count,
        "cpus": int(os.getenv("LABELCRITIC_CPUS", "16")),
        "memory": os.getenv("LABELCRITIC_MEM", "192G"),
        "walltime": os.getenv("LABELCRITIC_TIME", "08:00:00"),
        "port": int(os.getenv("LABELCRITIC_PORT", "8000")),
        "gpu_memory_utilization": os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.88"),
        "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
        "launcher_version": LABELCRITIC_LAUNCHER_VERSION,
        "generated_script": generated_script,
    }


def service_spec_hash(spec: dict[str, Any]) -> str:
    normalized = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_local_model(container: str | Path, model_path: str | Path, python_executable: str = "python3", *, timeout: int = 300) -> dict[str, Any]:
    script = r"""
import glob, json, os
from pathlib import Path
model = Path(os.environ["LABELCRITIC_MODEL_DIR"])
report = {
  "status": "PASS",
  "model_path": str(model),
  "dir": model.is_dir(),
  "config_json": (model / "config.json").is_file(),
  "processor_ok": False,
  "tokenizer_ok": False,
  "config_ok": False,
  "config_model_type": "",
  "quantization_method": "",
  "shard_count": 0,
  "total_weight_bytes": 0,
  "missing_files": [],
}
if not report["dir"]:
  report["status"] = "FAIL"; report["missing_files"].append(str(model))
if not report["config_json"]:
  report["status"] = "FAIL"; report["missing_files"].append(str(model / "config.json"))
try:
  from transformers import AutoConfig, AutoProcessor, AutoTokenizer
  cfg = AutoConfig.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
  report["config_ok"] = True
  report["config_model_type"] = str(getattr(cfg, "model_type", ""))
  q = getattr(cfg, "quantization_config", None) or {}
  if hasattr(q, "to_dict"):
    q = q.to_dict()
  report["quantization_method"] = str((q or {}).get("quant_method") or (q or {}).get("quantization_method") or "")
  AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
  report["processor_ok"] = True
  AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
  report["tokenizer_ok"] = True
except Exception as exc:
  report["status"] = "FAIL"; report["failure_reason"] = f"{type(exc).__name__}: {exc}"
index_files = list(model.glob("*.index.json"))
shards = set()
for pattern in ("*.safetensors", "*.bin", "*.pt"):
  shards.update(str(Path(path).name) for path in glob.glob(str(model / pattern)))
for index in index_files:
  try:
    doc = json.loads(index.read_text(encoding="utf-8"))
    for shard in set((doc.get("weight_map") or {}).values()):
      p = model / shard
      if not p.is_file():
        report["missing_files"].append(str(p)); report["status"] = "FAIL"
      else:
        shards.add(shard)
  except Exception as exc:
    report["status"] = "FAIL"; report["failure_reason"] = f"malformed_weight_index:{type(exc).__name__}: {exc}"
for shard in sorted(shards):
  p = model / shard
  try:
    size = p.stat().st_size
  except Exception:
    size = 0
  if size <= 0:
    report["missing_files"].append(str(p)); report["status"] = "FAIL"
  report["total_weight_bytes"] += size
report["shard_count"] = len(shards)
if report["shard_count"] <= 0:
  report["status"] = "FAIL"; report["failure_reason"] = report.get("failure_reason") or "missing_weight_shards"
if report["quantization_method"].lower() != "awq":
  report["status"] = "FAIL"; report["failure_reason"] = report.get("failure_reason") or "awq_quantization_config_missing"
print(json.dumps(report))
"""
    env = os.environ.copy()
    env["LABELCRITIC_MODEL_DIR"] = str(model_path)
    result = _run([*apptainer_exec_base(container, python_executable, nv=False), "-c", script], timeout=timeout, env=env)
    if not result["ok"]:
        return {"status": "FAIL", "failure_reason": "model_files_preflight_command_failed", "command": result}
    try:
        return json.loads(result["stdout"].splitlines()[-1])
    except Exception as exc:
        return {"status": "FAIL", "failure_reason": f"model_files_preflight_parse_failed:{type(exc).__name__}", "command": result}


def validate_gpu_topology(container: str | Path, python_executable: str = "python3", *, expected_count: int = 2, expected_type: str = "H100", tp: int = 2, timeout: int = 120) -> dict[str, Any]:
    script = r"""
import json, os, subprocess
report = {"status": "PASS", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "torch_cuda_available": False, "torch_cuda_device_count": 0, "gpu_names": [], "gpu_memory_mb": [], "tp": int(os.environ.get("LABELCRITIC_TENSOR_PARALLEL_SIZE", "0") or 0)}
try:
  import torch
  report["torch_cuda_available"] = bool(torch.cuda.is_available())
  report["torch_cuda_device_count"] = int(torch.cuda.device_count())
  report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
  for i in range(torch.cuda.device_count()):
    report["gpu_memory_mb"].append(int(torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)))
except Exception as exc:
  report["status"] = "FAIL"; report["failure_reason"] = f"torch_cuda_probe_failed:{type(exc).__name__}: {exc}"
if report["torch_cuda_device_count"] != int(os.environ["EXPECTED_GPU_COUNT"]):
  report["status"] = "FAIL"; report["failure_reason"] = "wrong_gpu_count"
if any(os.environ["EXPECTED_GPU_TYPE"] not in name for name in report["gpu_names"]):
  report["status"] = "FAIL"; report["failure_reason"] = report.get("failure_reason") or "wrong_gpu_type"
if report["tp"] != int(os.environ["EXPECTED_TP"]):
  report["status"] = "FAIL"; report["failure_reason"] = report.get("failure_reason") or "tp_mismatch"
print(json.dumps(report))
"""
    env = os.environ.copy()
    env.update({"EXPECTED_GPU_COUNT": str(expected_count), "EXPECTED_GPU_TYPE": expected_type, "EXPECTED_TP": str(tp), "LABELCRITIC_TENSOR_PARALLEL_SIZE": str(tp)})
    result = _run([*apptainer_exec_base(container, python_executable), "-c", script], timeout=timeout, env=env)
    if not result["ok"]:
        return {"status": "FAIL", "failure_reason": "gpu_topology_command_failed", "command": result}
    try:
        return json.loads(result["stdout"].splitlines()[-1])
    except Exception as exc:
        return {"status": "FAIL", "failure_reason": f"gpu_topology_parse_failed:{type(exc).__name__}", "command": result}


def validate_model_endpoint(base_url: str, port: int, expected_model: str, *, timeout: int = 20) -> dict[str, Any]:
    url_base = str(base_url).rstrip("/")
    if url_base.endswith("/v1"):
        url_base = url_base[:-3].rstrip("/")
    try:
        urllib.request.urlopen(f"{url_base}:{int(port)}/health", timeout=timeout).read()
        with urllib.request.urlopen(f"{url_base}:{int(port)}/v1/models", timeout=timeout) as response:
            doc = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("id") or "") for item in doc.get("data", []) if isinstance(item, dict)]
        ok = expected_model in models
        return {"status": "PASS" if ok else "FAIL", "served_models": models, "expected_model": expected_model, "failure_reason": "" if ok else "wrong_model"}
    except Exception as exc:
        return {"status": "FAIL", "served_models": [], "expected_model": expected_model, "failure_reason": f"{type(exc).__name__}: {exc}"}


def write_service_spec(service_root: Path, *, generated_script: str = "") -> dict[str, Any]:
    spec = service_spec_from_env(generated_script=generated_script)
    digest = service_spec_hash(spec)
    payload = {"status": "READY", "spec": spec, "service_spec_hash": digest, "timestamp": utc_now()}
    _write_json(service_root / "labelcritic_service_spec.json", payload)
    (service_root / "service_spec_hash.txt").write_text(digest + "\n", encoding="utf-8")
    return payload


def static_preflight(service_root: Path, *, generated_script: Path) -> dict[str, Any]:
    spec_payload = write_service_spec(service_root, generated_script=str(generated_script))
    spec = spec_payload["spec"]
    container = Path(str(spec["container_path"]))
    model = Path(str(spec["model_path"]))
    vllm_python = str(spec["VLLM_PYTHON"])
    report: dict[str, Any] = {
        "status": "PASSED",
        "container": str(container),
        "python_executable": vllm_python,
        "python_version": "",
        "vllm_version": "",
        "api_server_module_found": False,
        "model_path": str(model),
        "generated_script": str(generated_script),
        "service_spec_hash": spec_payload["service_spec_hash"],
        "project_bind": spec.get("project_bind", ""),
        "timestamp": utc_now(),
        "failure_reason": "",
        "checks": {},
    }

    def fail(reason: str, key: str = "", details: Any | None = None) -> dict[str, Any]:
        report["status"] = "FAILED"
        report["failure_reason"] = reason
        if key:
            report["checks"][key] = details if details is not None else {"status": "FAILED"}
        _write_json(service_root / "labelcritic_service_preflight.json", report)
        return report

    if shutil.which("apptainer") is None:
        return fail("apptainer_not_found", "apptainer", {"status": "FAILED"})
    report["checks"]["apptainer"] = {"status": "PASSED", "path": shutil.which("apptainer")}
    if not container.is_file() or not os.access(container, os.R_OK):
        return fail("container_missing_or_not_readable", "container", {"status": "FAILED", "path": str(container)})
    report["checks"]["container"] = {"status": "PASSED"}
    if not model.exists() or not os.access(model, os.R_OK):
        return fail("model_path_missing_or_not_readable", "model_path", {"status": "FAILED", "path": str(model)})
    report["checks"]["model_path"] = {"status": "PASSED"}
    if generated_script.exists():
        text = generated_script.read_text(encoding="utf-8")
        if " python -m vllm.entrypoints.openai.api_server" in text:
            return fail("generated_sbatch_uses_bare_python", "generated_script")
        if '"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server' not in text and f"{vllm_python} -m vllm.entrypoints.openai.api_server" not in text:
            return fail("generated_sbatch_missing_configured_vllm_python", "generated_script")
    version = _run([*apptainer_exec_base(container, vllm_python), "--version"])
    report["checks"]["python_version"] = version
    if not version["ok"]:
        return fail("configured_vllm_python_missing", "python_version", version)
    report["python_version"] = version["stdout"] or version["stderr"]
    probe = _run(
        [
            *apptainer_exec_base(container, vllm_python),
            "-c",
            "import importlib.util, json, vllm; print(json.dumps({'vllm_version': getattr(vllm, '__version__', ''), 'api_server_module_found': importlib.util.find_spec('vllm.entrypoints.openai.api_server') is not None}))",
        ]
    )
    report["checks"]["vllm_import"] = probe
    if not probe["ok"]:
        return fail("import_vllm_failed", "vllm_import", probe)
    try:
        doc = json.loads(probe["stdout"].splitlines()[-1])
    except Exception as exc:
        return fail(f"vllm_probe_parse_failed:{type(exc).__name__}", "vllm_import", probe)
    report["vllm_version"] = str(doc.get("vllm_version") or "")
    report["api_server_module_found"] = bool(doc.get("api_server_module_found"))
    if not report["api_server_module_found"]:
        return fail("api_server_module_missing", "api_server_module", probe)
    model_files = validate_local_model(container, model, vllm_python)
    report["checks"]["model_files"] = model_files
    if model_files.get("status") != "PASS":
        return fail(str(model_files.get("failure_reason") or "model_files_preflight_failed"), "model_files", model_files)
    bash_n = _run(["bash", "-n", str(generated_script)])
    report["checks"]["bash_n"] = bash_n
    if not bash_n["ok"]:
        return fail("generated_sbatch_shell_invalid", "bash_n", bash_n)
    _write_json(service_root / "labelcritic_service_preflight.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelCritic 72B service spec and static preflight.")
    sub = parser.add_subparsers(dest="command", required=True)
    spec_p = sub.add_parser("write-spec")
    spec_p.add_argument("--service-root", required=True, type=Path)
    spec_p.add_argument("--generated-script", default="")
    pre_p = sub.add_parser("preflight")
    pre_p.add_argument("--service-root", required=True, type=Path)
    pre_p.add_argument("--generated-script", required=True, type=Path)
    model_p = sub.add_parser("model-files")
    model_p.add_argument("--container", default=os.getenv("VLLM_CONTAINER", "/home/xhan74/containers/vllm-openai-v0.19.1.sif"))
    model_p.add_argument("--model-path", default=os.getenv("LABELCRITIC_MODEL_DIR", "/projects/bodymaps/users/xhan74/medical_agent/models/checkpoints/Qwen/Qwen2-VL-72B-Instruct-AWQ"))
    model_p.add_argument("--python", default=os.getenv("VLLM_PYTHON", "python3"))
    model_p.add_argument("--output-json", type=Path)
    gpu_p = sub.add_parser("gpu-topology")
    gpu_p.add_argument("--container", default=os.getenv("VLLM_CONTAINER", "/home/xhan74/containers/vllm-openai-v0.19.1.sif"))
    gpu_p.add_argument("--python", default=os.getenv("VLLM_PYTHON", "python3"))
    gpu_p.add_argument("--expected-count", default=int(os.getenv("LABELCRITIC_EXPECTED_GPU_COUNT", "2")), type=int)
    gpu_p.add_argument("--expected-type", default=os.getenv("LABELCRITIC_EXPECTED_GPU_TYPE", "H100"))
    gpu_p.add_argument("--tp", default=int(os.getenv("LABELCRITIC_TENSOR_PARALLEL_SIZE", "2")), type=int)
    gpu_p.add_argument("--output-json", type=Path)
    endpoint_p = sub.add_parser("endpoint")
    endpoint_p.add_argument("--base-url", required=True)
    endpoint_p.add_argument("--port", required=True, type=int)
    endpoint_p.add_argument("--expected-model", default=os.getenv("LABELCRITIC_MODEL_ID", "Qwen/Qwen2-VL-72B-Instruct-AWQ"))
    endpoint_p.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.command == "write-spec":
        payload = write_service_spec(args.service_root, generated_script=args.generated_script)
    elif args.command == "preflight":
        payload = static_preflight(args.service_root, generated_script=args.generated_script)
    elif args.command == "model-files":
        payload = validate_local_model(args.container, args.model_path, args.python)
    elif args.command == "gpu-topology":
        payload = validate_gpu_topology(args.container, args.python, expected_count=args.expected_count, expected_type=args.expected_type, tp=args.tp)
    else:
        payload = validate_model_endpoint(args.base_url, args.port, args.expected_model)
    if getattr(args, "output_json", None):
        _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("status") in {"READY", "PASSED", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
