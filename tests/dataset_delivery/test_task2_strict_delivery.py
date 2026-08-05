from __future__ import annotations

import importlib.util
import json
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from click.testing import CliRunner


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _load_nnunet_wrapper_module():
    path = Path("scripts/nnunetv2_predict_and_split.py").resolve()
    spec = importlib.util.spec_from_file_location("nnunetv2_predict_and_split_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wait_for_text(path: Path, needle: str, timeout_sec: float = 3.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return True
        time.sleep(0.05)
    return False


def _pid_alive(pid: int) -> bool:
    proc = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    state = proc.stdout.strip()
    return proc.returncode == 0 and bool(state) and "Z" not in state


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_names(output: Path) -> list[str]:
    stage_log = output / "nnunet_stage_events.jsonl"
    return [
        json.loads(line)["stage"]
        for line in stage_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_fake_nnunet_predict(path: Path, mode: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "import nibabel as nib\n"
        "import numpy as np\n"
        f"MODE = {mode!r}\n"
        "args = sys.argv\n"
        "input_dir = Path(args[args.index('-i') + 1])\n"
        "output_dir = Path(args[args.index('-o') + 1])\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n"
        "src = next(input_dir.glob('*.nii.gz'))\n"
        "out = output_dir / src.name.replace('_0000.nii.gz', '.nii.gz')\n"
        "def save_label(label=1):\n"
        "    img = nib.load(str(src))\n"
        "    arr = np.zeros(img.shape[:3], dtype=np.uint8)\n"
        "    if label:\n"
        "        arr[1:3, 1:3, 1:3] = label\n"
        "    nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(out))\n"
        "if MODE == 'normal':\n"
        "    save_label(1)\n"
        "    print('fake predictor completed', flush=True)\n"
        "elif MODE == 'hang_valid':\n"
        "    save_label(1)\n"
        "    print('sending off prediction to background worker for resampling and export', flush=True)\n"
        "    print('done with fake_case', flush=True)\n"
        "    time.sleep(60)\n"
        "elif MODE == 'corrupt_hang':\n"
        "    out.write_bytes(b'not a nifti')\n"
        "    print('wrote corrupt output', flush=True)\n"
        "    time.sleep(60)\n"
        "elif MODE == 'no_output_hang':\n"
        "    print('no output yet', flush=True)\n"
        "    time.sleep(60)\n"
        "elif MODE == 'changing_hang':\n"
        "    for _ in range(30):\n"
        "        save_label(1)\n"
        "        time.sleep(0.05)\n"
        "    time.sleep(60)\n"
        "elif MODE == 'grandchild_hang':\n"
        "    save_label(1)\n"
        "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "    pid_file = os.environ.get('GRANDCHILD_PID_FILE')\n"
        "    if pid_file:\n"
        "        Path(pid_file).write_text(str(p.pid), encoding='utf-8')\n"
        "    time.sleep(60)\n"
        "elif MODE == 'empty_normal':\n"
        "    save_label(0)\n"
        "    print('empty output completed', flush=True)\n"
        "elif MODE == 'nonzero':\n"
        "    print('predictor failed intentionally', file=sys.stderr, flush=True)\n"
        "    sys.exit(7)\n"
        "else:\n"
        "    raise SystemExit(f'unknown mode: {MODE}')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_fake_nnunet_wrapper(
    tmp_path: Path,
    mode: str,
    *,
    organs: str = "airway_tree",
    stable_sec: float = 0.1,
    shutdown_grace_sec: float = 0.2,
    timeout_sec: float = 2.0,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps({"labels": {"background": 0, "airway_tree": 1}}), encoding="utf-8")
    nnunet_results = tmp_path / "nnunet_results"
    nnunet_results.mkdir()
    fake_predict = _write_fake_nnunet_predict(tmp_path / f"fake_{mode}.py", mode)
    output = tmp_path / "out"
    per_model = output / "per_model" / "atm"
    cmd = [
        sys.executable,
        "scripts/nnunetv2_predict_and_split.py",
        "--image", str(image),
        "--output", str(output),
        "--dataset-id", "1370",
        "--nnunet-results", str(nnunet_results),
        "--dataset-json", str(dataset_json),
        "--trainer", "nnUNetTrainer",
        "--plans", "nnUNetPlans",
        "--predict-executable", str(fake_predict),
        "--organs", organs,
        "--per-model-dir", str(per_model),
        "--combined-output-stable-sec", str(stable_sec),
        "--post-export-shutdown-grace-sec", str(shutdown_grace_sec),
        "--process-termination-grace-sec", "0.2",
        "--post-export-poll-interval-sec", "0.05",
        "--prediction-timeout-sec", str(timeout_sec),
    ]
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=max(5.0, timeout_sec + 3.0),
        env=run_env,
    )
    return proc, output, per_model, image


def test_nnunet_streaming_subprocess_writes_stdout_and_stderr_before_exit(tmp_path: Path):
    wrapper = _load_nnunet_wrapper_module()
    child = tmp_path / "child.py"
    child.write_text(
        "import sys, time\n"
        "print('stdout-first', flush=True)\n"
        "print('stderr-first', file=sys.stderr, flush=True)\n"
        "time.sleep(1.0)\n"
        "print('stdout-done', flush=True)\n",
        encoding="utf-8",
    )
    stdout_log = tmp_path / "nnunet_stdout.log"
    stderr_log = tmp_path / "nnunet_stderr.log"
    result_box: dict[str, dict] = {}
    metadata = {
        "start_time": "test",
        "python_executable": sys.executable,
        "predict_executable": sys.executable,
        "nnUNet_results": str(tmp_path),
        "workdir": None,
        "CUDA_VISIBLE_DEVICES": "",
        "input_dir": str(tmp_path / "input"),
        "combined_output_dir": str(tmp_path / "combined"),
        "per_model_output_dir": str(tmp_path / "per_model"),
    }

    def run_child() -> None:
        result_box["result"] = wrapper._run_streaming_subprocess(
            [sys.executable, "-u", str(child)],
            env=os.environ.copy(),
            cwd=None,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            metadata=metadata,
        )

    thread = threading.Thread(target=run_child)
    thread.start()
    assert _wait_for_text(stdout_log, "stdout-first")
    assert _wait_for_text(stderr_log, "stderr-first")
    assert thread.is_alive()
    thread.join(timeout=5)
    assert result_box["result"]["return_code"] == 0
    assert "stdout-done" in stdout_log.read_text(encoding="utf-8")


def test_registered_streaming_command_returns_zero_and_logs(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    stdout_log = tmp_path / "registered_stdout.log"
    stderr_log = tmp_path / "registered_stderr.log"
    result = ri._run_shell_command_streaming(
        shlex.quote(sys.executable) + " -c " + shlex.quote('print("ok")'),
        env=os.environ.copy(),
        timeout_sec=10,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        kill_grace_sec=1,
    )

    assert result["return_code"] == 0
    assert result["timed_out"] is False
    assert "ok" in stdout_log.read_text(encoding="utf-8")


def test_registered_streaming_command_propagates_nonzero_return_code(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    result = ri._run_shell_command_streaming(
        shlex.quote(sys.executable) + " -c " + shlex.quote('import sys; print("bad"); sys.exit(7)'),
        env=os.environ.copy(),
        timeout_sec=10,
        stdout_log=tmp_path / "registered_stdout.log",
        stderr_log=tmp_path / "registered_stderr.log",
        kill_grace_sec=1,
    )

    assert result["return_code"] == 7
    assert result["timed_out"] is False


def test_registered_streaming_timeout_returns_124_and_kills_process_group(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    child_pid_file = tmp_path / "grandchild.pid"
    payload = (
        "import pathlib, subprocess, sys, time\n"
        f"pid_file = pathlib.Path({str(child_pid_file)!r})\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pid_file.write_text(str(p.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    result = ri._run_shell_command_streaming(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        env=os.environ.copy(),
        timeout_sec=1,
        stdout_log=tmp_path / "registered_stdout.log",
        stderr_log=tmp_path / "registered_stderr.log",
        kill_grace_sec=0.2,
    )

    assert result["return_code"] == 124
    assert result["timed_out"] is True
    assert result["process_group_id"]
    assert child_pid_file.exists()
    grandchild_pid = int(child_pid_file.read_text(encoding="utf-8"))
    deadline = time.time() + 5
    while time.time() < deadline and _pid_alive(grandchild_pid):
        time.sleep(0.1)
    assert not _pid_alive(grandchild_pid)


def test_nnunet_diagnostic_workdir_is_retained(tmp_path: Path):
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps({"labels": {"background": 0, "airway_tree": 1}}), encoding="utf-8")
    nnunet_results = tmp_path / "nnunet_results"
    nnunet_results.mkdir()
    fake_predict = tmp_path / "fake_nnunet_predict.py"
    fake_predict.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import nibabel as nib\n"
        "import numpy as np\n"
        "args = sys.argv\n"
        "input_dir = Path(args[args.index('-i') + 1])\n"
        "output_dir = Path(args[args.index('-o') + 1])\n"
        "output_dir.mkdir(parents=True, exist_ok=True)\n"
        "src = next(input_dir.glob('*.nii.gz'))\n"
        "img = nib.load(str(src))\n"
        "arr = np.ones(img.shape[:3], dtype=np.uint8)\n"
        "nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(output_dir / src.name.replace('_0000.nii.gz', '.nii.gz')))\n"
        "print('fake predictor completed', flush=True)\n",
        encoding="utf-8",
    )
    fake_predict.chmod(0o755)
    output = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/nnunetv2_predict_and_split.py",
            "--image", str(image),
            "--output", str(output),
            "--dataset-id", "1370",
            "--nnunet-results", str(nnunet_results),
            "--dataset-json", str(dataset_json),
            "--trainer", "nnUNetTrainer",
            "--plans", "nnUNetPlans",
            "--predict-executable", str(fake_predict),
            "--organs", "airway_tree",
            "--diagnostic",
            "--keep-workdir",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = json.loads((output / "nnunet_run_metadata.json").read_text(encoding="utf-8"))
    retained = Path(metadata["retained_workdir"])
    assert retained.exists()
    assert (retained / "input").exists()
    assert (retained / "combined").exists()
    assert (output / "segmentations" / "airway_tree.nii.gz").exists()
    assert "-npp 1" in metadata["command_text"]
    assert "-nps 1" in metadata["command_text"]


def test_nnunet_wrapper_normal_exit_valid_output_splits_and_validates(tmp_path: Path):
    proc, output, per_model, image = _run_fake_nnunet_wrapper(tmp_path, "normal")

    assert proc.returncode == 0, proc.stderr
    mask = per_model / "segmentations" / "airway_tree.nii.gz"
    assert mask.exists()
    mask_img = nib.load(str(mask))
    ct_img = nib.load(str(image))
    mask_arr = np.asanyarray(mask_img.dataobj)
    assert set(np.unique(mask_arr).tolist()).issubset({0, 1})
    assert int(mask_arr.sum()) > 0
    assert mask_img.shape[:3] == ct_img.shape[:3]
    assert np.allclose(mask_img.affine, ct_img.affine)

    summary = _read_json(per_model / "inference_summary.json")
    assert summary["raw_child_return_code"] == 0
    assert summary["effective_inference_status"] == "completed"
    assert summary["forced_process_cleanup"] is False
    assert summary["combined_output_valid"] is True
    assert summary["target_mask_validation"]["valid"] is True
    stages = _stage_names(output)
    for stage in [
        "predictor_started",
        "combined_output_detected",
        "combined_output_validated",
        "predictor_completed",
        "split_started",
        "split_completed",
        "validation_completed",
    ]:
        assert stage in stages
    assert "post_export_shutdown_hang_detected" not in stages


def test_nnunet_wrapper_recovers_valid_output_after_predictor_sleep(tmp_path: Path):
    proc, output, per_model, _image = _run_fake_nnunet_wrapper(tmp_path, "hang_valid")

    assert proc.returncode == 0, proc.stderr
    assert (per_model / "segmentations" / "airway_tree.nii.gz").exists()
    summary = _read_json(per_model / "inference_summary.json")
    assert summary["forced_process_cleanup"] is True
    assert summary["forced_cleanup_reason"] == "post_prediction_process_shutdown_hang"
    assert summary["effective_inference_status"] == "valid_output_completed_after_forced_process_cleanup"
    assert summary["raw_child_return_code"] is not None
    assert summary["termination_signal"] in (signal.SIGTERM, signal.SIGKILL)
    assert summary["combined_output_valid"] is True
    assert summary["target_mask_validation"]["valid"] is True
    stages = _stage_names(output)
    assert "post_prediction_process_shutdown_hang" in stages
    assert "post_export_shutdown_hang_detected" in stages
    assert "process_group_termination_started" in stages
    assert "process_group_terminated" in stages
    assert stages[-1] == "validation_completed"


def test_nnunet_wrapper_does_not_recover_corrupt_nifti_sleep(tmp_path: Path):
    proc, output, per_model, _image = _run_fake_nnunet_wrapper(
        tmp_path,
        "corrupt_hang",
        stable_sec=0.05,
        shutdown_grace_sec=0.1,
        timeout_sec=0.7,
    )

    assert proc.returncode == 124
    summary = _read_json(per_model / "inference_summary.json")
    assert summary["status"] == "failed"
    assert summary["forced_process_cleanup"] is False
    assert summary["combined_output_valid"] is False
    assert any("nifti_load_failed" in reason for reason in summary["combined_output_validation"]["reasons"])
    stages = _stage_names(output)
    assert "combined_output_detected" in stages
    assert "combined_output_validated" not in stages
    assert "post_export_shutdown_hang_detected" not in stages
    assert not (per_model / "segmentations" / "airway_tree.nii.gz").exists()


def test_nnunet_wrapper_times_out_when_no_output_is_written(tmp_path: Path):
    proc, output, per_model, _image = _run_fake_nnunet_wrapper(
        tmp_path,
        "no_output_hang",
        stable_sec=0.05,
        shutdown_grace_sec=0.1,
        timeout_sec=0.6,
    )

    assert proc.returncode == 124
    summary = _read_json(per_model / "inference_summary.json")
    assert summary["status"] == "failed"
    assert summary["effective_inference_status"] == "timed_out"
    assert summary["combined_output_valid"] is False
    assert "combined_output_detected" not in _stage_names(output)


def test_nnunet_wrapper_does_not_validate_output_while_it_is_changing(tmp_path: Path):
    proc, output, per_model, _image = _run_fake_nnunet_wrapper(
        tmp_path,
        "changing_hang",
        stable_sec=0.5,
        shutdown_grace_sec=0.1,
        timeout_sec=0.8,
    )

    assert proc.returncode == 124
    summary = _read_json(per_model / "inference_summary.json")
    assert summary["status"] == "failed"
    assert summary["combined_output_valid"] is False
    stages = _stage_names(output)
    assert "combined_output_detected" in stages
    assert "combined_output_validated" not in stages
    assert "post_export_shutdown_hang_detected" not in stages
    assert not (per_model / "segmentations" / "airway_tree.nii.gz").exists()


def test_nnunet_wrapper_forced_cleanup_kills_grandchild_process(tmp_path: Path):
    grandchild_pid_file = tmp_path / "grandchild.pid"
    proc, _output, per_model, _image = _run_fake_nnunet_wrapper(
        tmp_path,
        "grandchild_hang",
        env={"GRANDCHILD_PID_FILE": str(grandchild_pid_file)},
    )

    assert proc.returncode == 0, proc.stderr
    assert (per_model / "segmentations" / "airway_tree.nii.gz").exists()
    assert grandchild_pid_file.exists()
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
    deadline = time.time() + 5
    while time.time() < deadline and _pid_alive(grandchild_pid):
        time.sleep(0.1)
    assert not _pid_alive(grandchild_pid)


def test_nnunet_wrapper_nonzero_exit_preserves_raw_return_code(tmp_path: Path):
    proc, _output, per_model, _image = _run_fake_nnunet_wrapper(tmp_path, "nonzero")

    assert proc.returncode == 7
    summary = _read_json(per_model / "inference_summary.json")
    assert summary["status"] == "failed"
    assert summary["return_code"] == 7
    assert summary["raw_child_return_code"] == 7
    assert summary["forced_process_cleanup"] is False
    assert summary["effective_inference_status"] == "failed"


def test_registered_infer_preserves_expected_mask_missing_for_empty_nnunet_target(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps({"labels": {"background": 0, "airway_tree": 1}}), encoding="utf-8")
    nnunet_results = tmp_path / "nnunet_results"
    nnunet_results.mkdir()
    fake_predict = _write_fake_nnunet_predict(tmp_path / "fake_empty.py", "empty_normal")
    registry_path = tmp_path / "configs" / "model_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        f"""
models:
  atm:
    name: ATM
    enabled: true
    runner: command_template
    covered_organs: [airway_tree]
    checkpoint_path: {nnunet_results}
    dataset_json_path: {dataset_json}
    dataset_id: 1370
    trainer: nnUNetTrainer
    plans: nnUNetPlans
    predict_executable: {fake_predict}
    command_template: python scripts/nnunetv2_predict_and_split.py --image {{image}} --output {{case_output}} --dataset-id {{dataset_id}} --nnunet-results {{checkpoint_path}} --dataset-json {{dataset_json_path}} --trainer {{trainer}} --plans {{plans}} --predict-executable {{predict_executable}} --organs airway_tree --combined-output-stable-sec 0.05 --post-export-shutdown-grace-sec 0.1 --post-export-poll-interval-sec 0.05 --process-termination-grace-sec 0.2
""",
        encoding="utf-8",
    )

    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "atm",
        registry_path=registry_path,
        case_id="case",
        timeout_sec=3,
        extra_context={"requested_organs": ["airway_tree"]},
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "expected_mask_missing"
    assert result["missing_expected_outputs"] == ["airway_tree.nii.gz"]
    assert result["raw_child_return_code"] == 0
    assert result["combined_output_valid"] is False


def test_strict_delivery_cli_failure_exits_nonzero(tmp_path: Path, monkeypatch):
    from cli_anything.medai import medai_cli

    def fake_loop(*args, **kwargs):
        return {
            "stage": "run_loop",
            "status": "failed",
            "strict_delivery_failure_count": 1,
            "strict_delivery_failures": [{"status": "failed", "reason": "expected_mask_missing"}],
        }

    monkeypatch.setattr(medai_cli, "run_multimodel_annotation_loop", fake_loop)
    result = CliRunner().invoke(
        medai_cli.cli,
        [
            "--json", "run-loop",
            "--case-list", str(tmp_path / "cases.csv"),
            "--models", "atm",
            "--organs", "airway_tree",
            "--output", str(tmp_path / "out"),
            "--dry-run",
            "--strict-delivery-targets",
        ],
    )

    assert result.exit_code == 2
    assert '"strict_delivery_failure_count": 1' in result.output


def test_strict_delivery_cli_success_exits_zero_when_expected_mask_succeeds(tmp_path: Path, monkeypatch):
    from cli_anything.medai import medai_cli

    def fake_loop(*args, **kwargs):
        return {
            "stage": "run_loop",
            "status": "success",
            "strict_delivery_failure_count": 0,
            "strict_delivery_failures": [],
            "inference_results": [{"status": "success", "expected_present": ["airway_tree.nii.gz"], "missing_expected_outputs": []}],
        }

    monkeypatch.setattr(medai_cli, "run_multimodel_annotation_loop", fake_loop)
    result = CliRunner().invoke(
        medai_cli.cli,
        [
            "--json", "run-loop",
            "--case-list", str(tmp_path / "cases.csv"),
            "--models", "atm",
            "--organs", "airway_tree",
            "--output", str(tmp_path / "out"),
            "--dry-run",
            "--strict-delivery-targets",
        ],
    )

    assert result.exit_code == 0
    assert '"strict_delivery_failure_count": 0' in result.output


def test_registered_infer_fails_when_only_auxiliary_masks_exist(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    registry_path = tmp_path / "configs" / "model_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
models:
  atm:
    name: ATM
    enabled: true
    runner: command_template
    covered_organs: [airway_tree]
    command_template: python fake.py
""",
        encoding="utf-8",
    )
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")

    def fake_run(command, **kwargs):
        seg = tmp_path / "out" / "case" / "segmentations"
        _save(np.zeros((4, 4, 4), dtype=np.uint8), seg / "image.nii.gz")
        _save(np.zeros((4, 4, 4), dtype=np.uint8), seg / "zero_mask.nii.gz")
        stdout_log = Path(kwargs["stdout_log"])
        stderr_log = Path(kwargs["stderr_log"])
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return {"return_code": 0, "timed_out": False, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}

    monkeypatch.setattr(ri, "_run_shell_command_streaming", fake_run)
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "atm",
        registry_path=registry_path,
        extra_context={"requested_organs": ["airway_tree"]},
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "expected_mask_missing"
    assert result["formal_mask_count"] == 0
    assert result["auxiliary_outputs"] == ["image.nii.gz", "zero_mask.nii.gz"]


def test_registered_infer_resolves_task2_nnunet_command(tmp_path: Path):
    from cli_anything.medai.core import registered_infer as ri

    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "atm",
        registry_path=Path("configs/model_registry.yaml"),
        case_id="case",
        dry_run=True,
        extra_context={"requested_organs": ["airway_tree"]},
    )

    assert result["status"] == "dry_run"
    assert result["resolved_python"] == "/home/xhan74/envs/medical_agent/bin/python"
    assert "/home/xhan74/nnunet_torch22_wrapper/bin/nnUNetv2_predict" in result["command"]
    assert "--sitecustomize-path" in result["command"]
    assert "/home/xhan74/nnunet_torch22_compat/sitecustomize.py" in result["command"]


def test_registered_infer_unest_python_env_overrides_registry(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    monkeypatch.setenv("MEDAI_UNEST_PYTHON", "/opt/unest/bin/python")
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "unest",
        registry_path=Path("configs/model_registry.yaml"),
        case_id="case",
        dry_run=True,
        extra_context={"requested_organs": ["kidney_cortex"]},
    )

    assert result["status"] == "dry_run"
    assert '--python-executable "/opt/unest/bin/python"' in result["command"]


def test_registered_infer_fails_when_airrc_expected_output_is_missing(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import registered_infer as ri

    registry_path = tmp_path / "configs" / "model_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
models:
  airrc:
    name: AirRC
    enabled: true
    runner: command_template
    covered_organs: [airway_wall, lung_pulmonary_arteries, lung_pulmonary_veins]
    command_template: python fake.py
""",
        encoding="utf-8",
    )
    image = _save(np.zeros((4, 4, 4), dtype=np.int16), tmp_path / "case" / "ct.nii.gz")

    def fake_run(command, **kwargs):
        seg = tmp_path / "out" / "case" / "segmentations"
        _save(np.ones((4, 4, 4), dtype=np.uint8), seg / "airway_wall.nii.gz")
        stdout_log = Path(kwargs["stdout_log"])
        stderr_log = Path(kwargs["stderr_log"])
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return {"return_code": 0, "timed_out": False, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}

    monkeypatch.setattr(ri, "_run_shell_command_streaming", fake_run)
    result = ri.run_registered_model(
        image,
        tmp_path / "out",
        "airrc",
        registry_path=registry_path,
        extra_context={"requested_organs": ["airway_wall", "lung_pulmonary_arteries", "lung_pulmonary_veins"]},
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "expected_mask_missing"
    assert result["missing_expected_outputs"] == [
        "lung_pulmonary_arteries.nii.gz",
        "lung_pulmonary_veins.nii.gz",
    ]


def test_external_kidney_parent_roi_builds_union_from_left_right(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    ref = tmp_path / "segmentations"
    left = np.zeros((8, 8, 8), dtype=np.uint8)
    right = np.zeros((8, 8, 8), dtype=np.uint8)
    left[1:3, 1:3, 1:3] = 1
    right[5:7, 5:7, 5:7] = 1
    _save(left, ref / "kidney_left.nii.gz")
    _save(right, ref / "kidney_right.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))

    parent_masks, records = loop._resolve_external_parent_masks(
        ref_dir=ref,
        requested_organs=["kidney_cortex", "kidney_medulla"],
        taxonomy=taxonomy,
        ct=ct,
        case_out=tmp_path / "case",
    )

    assert set(parent_masks) == {"kidney"}
    assert records[0]["status"] == "built_union_parent"
    merged = np.asanyarray(nib.load(str(parent_masks["kidney"])).dataobj)
    assert int(merged.sum()) == int(left.sum() + right.sum())


def test_unest_child_reports_missing_kidney_parent(tmp_path: Path):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "raw",
        case_out=tmp_path / "case",
        registry_path=Path("configs/model_registry.yaml"),
        execution_plan={
            "per_organ": {
                "kidney_cortex": {"primary_teacher": "unest", "backup_teachers": [], "competition_teachers": []},
                "kidney": {"primary_teacher": None, "backup_teachers": [], "competition_teachers": []},
            }
        },
        requested_organs=["kidney_cortex"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )

    assert any(
        item.get("organ") == "kidney_cortex"
        and item.get("status") == "blocked_by_parent"
        and item.get("missing_parents") == ["kidney"]
        for item in result["blocked"]
    )


def test_atm_child_runs_full_volume_without_airway_parent(tmp_path: Path, monkeypatch):
    from cli_anything.medai.core import multimodel_loop as loop

    ct = _save(np.zeros((8, 8, 8), dtype=np.int16), tmp_path / "ct.nii.gz")
    taxonomy = json.loads(Path("configs/organ_taxonomy.json").read_text(encoding="utf-8"))
    calls = []

    def fake_registered(image, output, model, **kwargs):
        requested = list((kwargs.get("extra_context") or {}).get("requested_organs", []))
        calls.append({"model": model, "requested": requested})
        seg = Path(output) / kwargs["case_id"] / "segmentations"
        arr = np.zeros((8, 8, 8), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        for organ in requested:
            _save(arr, seg / f"{organ}.nii.gz")
        return {"status": "success", "model_key": model, "segmentation_output": str(seg), "num_masks": len(requested)}

    monkeypatch.setattr(loop, "run_registered_model", fake_registered)
    result = loop._run_hierarchical_case_inference(
        ct=ct,
        case_id="case_001",
        case_raw=tmp_path / "raw",
        case_out=tmp_path / "case",
        registry_path=Path("configs/model_registry.yaml"),
        execution_plan={
            "per_organ": {
                "airway_tree": {"primary_teacher": "atm", "backup_teachers": [], "competition_teachers": []},
                "airway": {"primary_teacher": None, "backup_teachers": [], "competition_teachers": []},
            }
        },
        requested_organs=["airway_tree"],
        taxonomy=taxonomy,
        alias_config={"models": {}},
        timeout_sec=30,
        device="cpu",
        dry_run=False,
        margin_mm=2,
    )

    assert calls == [{"model": "atm", "requested": ["airway_tree"]}]
    assert (result["model_seg_dirs"]["atm"] / "airway_tree.nii.gz").exists()
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert not any(item.get("organ") == "airway_tree" and item.get("status") == "blocked_by_parent" for item in manifest["blocked"])


def test_strict_run_loop_fails_when_requested_teacher_pruned_to_empty_plan(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import run_multimodel_annotation_loop

    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,ct_path,annotation_folder,body_region\n"
        f"case_001,{tmp_path / 'missing_ct.nii.gz'},{tmp_path / 'ref'},abdomen\n",
        encoding="utf-8",
    )
    result = run_multimodel_annotation_loop(
        cases,
        tmp_path / "out",
        models=["atm"],
        organs=["airway_tree"],
        registry_path=Path("configs/model_registry.yaml"),
        enable_shapekit=False,
        enable_critic=False,
        dry_run=True,
        strict_delivery_targets=True,
    )

    assert result["status"] == "failed"
    assert result["strict_delivery_failure_count"] >= 1
    assert {row["reason"] for row in result["strict_delivery_failures"]} >= {
        "requested_organs_pruned_by_fov",
        "requested_teacher_not_scheduled",
    }


def test_task2_scope_contains_blocked_totalsegmentator(tmp_path: Path):
    from tools.dataset_delivery.task2_audit import build_scope

    rows = build_scope(tmp_path)
    assert len(rows) == 23
    total = [row for row in rows if row["model_group"] == "TotalSegmentator"]
    assert len(total) == 1
    assert total[0]["execution_status"] == "blocked"
