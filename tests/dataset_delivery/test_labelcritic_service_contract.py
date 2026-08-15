from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_hpc_bin(tmp_path: Path, *, mode: str = "ok") -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    (bindir / "apptainer").write_text(
        "#!/usr/bin/env bash\n"
        "shift # exec\n"
        "container=$1; shift\n"
        "py=$1; shift\n"
        f"mode={mode!r}\n"
        "if [[ \"$py\" != \"python3\" ]]; then echo missing python >&2; exit 127; fi\n"
        "if [[ \"$1\" == \"--version\" ]]; then echo 'Python 3.12.13'; exit 0; fi\n"
        "if [[ \"$mode\" == \"import_fail\" ]]; then echo 'No module named vllm' >&2; exit 1; fi\n"
        "if [[ \"$mode\" == \"api_missing\" ]]; then echo '{\"vllm_version\":\"0.19.1\",\"api_server_module_found\":false}'; exit 0; fi\n"
        "echo '{\"vllm_version\":\"0.19.1\",\"api_server_module_found\":true}'\n",
        encoding="utf-8",
    )
    (bindir / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {tmp_path / 'sbatch.log'}\n"
        "echo 999999\n",
        encoding="utf-8",
    )
    for path in bindir.iterdir():
        path.chmod(0o755)
    return bindir


def _run_submit(tmp_path: Path, *, mode: str = "ok", vllm_python: str = "python3") -> subprocess.CompletedProcess[str]:
    bindir = _fake_hpc_bin(tmp_path, mode=mode)
    service_root = tmp_path / "service"
    container = tmp_path / "vllm.sif"
    model = tmp_path / "model"
    container.write_text("container", encoding="utf-8")
    model.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bindir}:{env.get('PATH', '')}",
            "CODE_ROOT": str(REPO_ROOT),
            "STATE_ROOT": str(tmp_path / "state"),
            "LABELCRITIC_SERVICE_ROOT": str(service_root),
            "VLLM_CONTAINER": str(container),
            "VLLM_PYTHON": vllm_python,
            "LABELCRITIC_MODEL_DIR": str(model),
            "LABELCRITIC_MODEL_ID": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            "LABELCRITIC_TENSOR_PARALLEL_SIZE": "2",
            "LABELCRITIC_GRES": "gpu:H100:2",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.88",
            "VLLM_MAX_MODEL_LEN": "8192",
        }
    )
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/task2/submit_labelcritic_72b_service.sh")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_labelcritic_submit_container_with_python3_preflight_passes_and_sbatch_uses_python3(tmp_path: Path):
    proc = _run_submit(tmp_path, mode="ok", vllm_python="python3")

    assert proc.returncode == 0, proc.stderr
    service_root = tmp_path / "service"
    preflight = json.loads((service_root / "labelcritic_service_preflight.json").read_text(encoding="utf-8"))
    spec = json.loads((service_root / "labelcritic_service_spec.json").read_text(encoding="utf-8"))
    sbatch = (service_root / "labelcritic_72b_service.sbatch").read_text(encoding="utf-8")
    assert preflight["status"] == "PASSED"
    assert preflight["python_executable"] == "python3"
    assert preflight["vllm_version"] == "0.19.1"
    assert preflight["api_server_module_found"] is True
    assert spec["spec"]["served_model_name"] == "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    assert spec["spec"]["tensor_parallel_size"] == 2
    assert spec["spec"]["gres"] == "gpu:H100:2"
    assert spec["spec"]["gpu_memory_utilization"] == "0.88"
    assert spec["spec"]["max_model_len"] == 8192
    assert spec["spec"]["port"] == 8000
    assert '"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server' in sbatch
    assert " python -m vllm.entrypoints.openai.api_server" not in sbatch
    assert "--comment medical_agent:labelcritic_72b:" in (tmp_path / "sbatch.log").read_text(encoding="utf-8")


def test_labelcritic_configured_vllm_python_missing_fails_before_sbatch(tmp_path: Path):
    proc = _run_submit(tmp_path, mode="ok", vllm_python="python")

    assert proc.returncode != 0
    preflight = json.loads((tmp_path / "service" / "labelcritic_service_preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "FAILED"
    assert preflight["failure_reason"] == "configured_vllm_python_missing"
    assert not (tmp_path / "sbatch.log").exists()


def test_labelcritic_import_vllm_and_api_server_failures_block_submission(tmp_path: Path):
    import_fail = _run_submit(tmp_path / "import_fail", mode="import_fail", vllm_python="python3")
    api_missing = _run_submit(tmp_path / "api_missing", mode="api_missing", vllm_python="python3")

    assert import_fail.returncode != 0
    assert json.loads((tmp_path / "import_fail" / "service" / "labelcritic_service_preflight.json").read_text(encoding="utf-8"))["failure_reason"] == "import_vllm_failed"
    assert not (tmp_path / "import_fail" / "sbatch.log").exists()
    assert api_missing.returncode != 0
    assert json.loads((tmp_path / "api_missing" / "service" / "labelcritic_service_preflight.json").read_text(encoding="utf-8"))["failure_reason"] == "api_server_module_missing"
    assert not (tmp_path / "api_missing" / "sbatch.log").exists()
