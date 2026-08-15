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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_delivery.delivery_lib import utc_now


LABELCRITIC_LAUNCHER_VERSION = "labelcritic_vllm_python3_spec_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run(command: list[str], *, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
        return {
            "command": command,
            "return_code": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except Exception as exc:
        return {"command": command, "return_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "ok": False}


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
    version = _run(["apptainer", "exec", str(container), vllm_python, "--version"])
    report["checks"]["python_version"] = version
    if not version["ok"]:
        return fail("configured_vllm_python_missing", "python_version", version)
    report["python_version"] = version["stdout"] or version["stderr"]
    probe = _run(
        [
            "apptainer",
            "exec",
            str(container),
            vllm_python,
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
    args = parser.parse_args()
    if args.command == "write-spec":
        payload = write_service_spec(args.service_root, generated_script=args.generated_script)
    else:
        payload = static_preflight(args.service_root, generated_script=args.generated_script)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("status") in {"READY", "PASSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
