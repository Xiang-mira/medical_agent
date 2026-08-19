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
        "bind_seen=0\n"
        "while [[ \"$1\" == --* ]]; do\n"
        "  if [[ \"$1\" == \"--bind\" ]]; then bind_seen=1; shift 2; continue; fi\n"
        "  shift\n"
        "done\n"
        "container=$1; shift\n"
        "py=$1; shift\n"
        f"mode={mode!r}\n"
        "if [[ \"$py\" != \"python3\" ]]; then echo missing python >&2; exit 127; fi\n"
        "if [[ \"$1\" == \"--version\" ]]; then echo 'Python 3.12.13'; exit 0; fi\n"
        "if [[ \"$*\" == *\"LABELCRITIC_MODEL_DIR\"* ]]; then\n"
        "  if [[ \"$bind_seen\" != \"1\" ]]; then echo '{\"status\":\"FAIL\",\"failure_reason\":\"missing_bind\",\"dir\":false,\"config_json\":false}'; exit 0; fi\n"
        "  echo '{\"status\":\"PASS\",\"shard_count\":4,\"total_weight_bytes\":1234,\"config_model_type\":\"qwen2_vl\",\"quantization_method\":\"awq\",\"missing_files\":[],\"processor_ok\":true,\"tokenizer_ok\":true,\"config_ok\":true,\"dir\":true,\"config_json\":true}'; exit 0\n"
        "fi\n"
        "if [[ \"$*\" == *\"EXPECTED_GPU_COUNT\"* ]]; then echo '{\"status\":\"PASS\",\"torch_cuda_available\":true,\"torch_cuda_device_count\":2,\"gpu_names\":[\"NVIDIA H100\",\"NVIDIA H100\"],\"gpu_memory_mb\":[81920,81920],\"tp\":2}'; exit 0; fi\n"
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


def test_validate_local_model_requires_project_bind_inside_container(tmp_path: Path, monkeypatch):
    from tools.dataset_delivery import labelcritic_service_contract as contract

    bindir = _fake_hpc_bin(tmp_path, mode="ok")
    container = tmp_path / "vllm.sif"
    model = tmp_path / "model"
    container.write_text("container", encoding="utf-8")
    model.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("LABELCRITIC_PROJECT_BIND", "")

    missing = contract.validate_local_model(container, model, "python3")
    monkeypatch.setenv("LABELCRITIC_PROJECT_BIND", "/projects/bodymaps/users/xhan74/medical_agent:/projects/bodymaps/users/xhan74/medical_agent")
    visible = contract.validate_local_model(container, model, "python3")

    assert missing["status"] == "FAIL"
    assert missing["failure_reason"] == "missing_bind"
    assert visible["status"] == "PASS"
    assert visible["config_model_type"] == "qwen2_vl"


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
    assert spec["spec"]["project_bind"] == "/projects/bodymaps/users/xhan74/medical_agent:/projects/bodymaps/users/xhan74/medical_agent"
    assert spec["spec"]["tensor_parallel_size"] == 2
    assert spec["spec"]["gres"] == "gpu:H100:2"
    assert spec["spec"]["gpu_memory_utilization"] == "0.88"
    assert spec["spec"]["max_model_len"] == 8192
    assert spec["spec"]["port"] == 8000
    assert '"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server' in sbatch
    assert "--bind" in sbatch
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


def test_labelcritic_acceptance_sbatch_is_isolated_and_uses_h100_bind(tmp_path: Path):
    from argparse import Namespace
    from tools.dataset_delivery import labelcritic_72b_acceptance as acceptance

    args = Namespace(
        partition="gpuh100",
        gres="gpu:H100:2",
        cpus=16,
        mem="192G",
        time_limit="04:00:00",
        container=tmp_path / "vllm.sif",
        model_dir=tmp_path / "model",
        model_id="Qwen/Qwen2-VL-72B-Instruct-AWQ",
        vllm_python="python3",
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization="0.88",
        startup_timeout_sec=1800,
        stability_sec=60,
    )
    root = tmp_path / "acceptance" / "20260819_000000"
    root.mkdir(parents=True)

    sbatch = acceptance.render_sbatch(args, root)
    text = sbatch.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpuh100" in text
    assert "#SBATCH --gres=gpu:H100:2" in text
    assert "--validation-root" in text
    assert str(root) in text
    assert "--max-model-len 8192" in text
    assert "--gpu-memory-utilization 0.88" in text
    assert "LABELCRITIC_PORT=8000" not in text
    assert "round1_orchestrated/full_373_multiteacher_round1" not in text
