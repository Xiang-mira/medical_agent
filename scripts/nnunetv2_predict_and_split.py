#!/usr/bin/env python3
"""Run an nnUNet v2 checkpoint on one CT and split the combined label map into organ-wise masks.

This wrapper is designed for the teacher-provided checkpoint folders exported from Google Drive:
CADS_series, MOOSE_series, nnUNet_private, and VSmTrans.

It never bundles private weights. It assumes the checkpoint folder already exists locally or on the
server and only standardizes I/O for the medai CLI.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


ORGAN_ALIASES = {
    "celiac_aa_celiac_artery": "celiac_aa",
    "inferior_vena_cava": "postcava",
    "small_intestine": "intestine",
}


def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _worker_count(explicit: int | None, *, diagnostic: bool, default: int | None) -> int | None:
    if explicit is not None:
        return max(1, int(explicit))
    if diagnostic:
        return 1
    return default


def _tail_file(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_stage(stage_log: Path, status_path: Path, stage: str, **payload) -> None:
    event = {"timestamp": _utc_now(), "stage": stage, **payload}
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    with stage_log.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
    status = dict(event)
    status["last_stage"] = stage
    _write_json(status_path, status)


def _write_log_header(path: Path, stream_name: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[medai nnunet] stream={stream_name}",
        f"[medai nnunet] start_time={metadata.get('start_time')}",
        f"[medai nnunet] command={metadata.get('command_text')}",
        f"[medai nnunet] python_executable={metadata.get('python_executable')}",
        f"[medai nnunet] predict_executable={metadata.get('predict_executable')}",
        f"[medai nnunet] nnUNet_results={metadata.get('nnUNet_results')}",
        f"[medai nnunet] workdir={metadata.get('workdir')}",
        f"[medai nnunet] CUDA_VISIBLE_DEVICES={metadata.get('CUDA_VISIBLE_DEVICES')}",
        f"[medai nnunet] input_dir={metadata.get('input_dir')}",
        f"[medai nnunet] combined_output_dir={metadata.get('combined_output_dir')}",
        f"[medai nnunet] per_model_output_dir={metadata.get('per_model_output_dir')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_log_footer(path: Path, *, end_time: str, return_code: int) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[medai nnunet] end_time={end_time}\n")
        handle.write(f"[medai nnunet] return_code={return_code}\n")
        handle.flush()


def _stream_pipe_to_log(pipe, log_path: Path) -> None:
    try:
        with log_path.open("a", encoding="utf-8", buffering=1, errors="replace") as handle:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                handle.write(line)
                handle.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _run_streaming_subprocess(cmd: list[str], *, env: dict[str, str], cwd: Path | None, stdout_log: Path, stderr_log: Path, metadata: dict) -> dict:
    """Run a subprocess while streaming stdout/stderr to durable logs."""
    command_text = shlex.join([str(part) for part in cmd])
    run_meta = {**metadata, "command": [str(part) for part in cmd], "command_text": command_text}
    _write_log_header(stdout_log, "stdout", run_meta)
    _write_log_header(stderr_log, "stderr", run_meta)
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    threads: list[threading.Thread] = []
    assert proc.stdout is not None
    assert proc.stderr is not None
    for pipe, path in ((proc.stdout, stdout_log), (proc.stderr, stderr_log)):
        thread = threading.Thread(target=_stream_pipe_to_log, args=(pipe, path), daemon=True)
        thread.start()
        threads.append(thread)
    return_code = proc.wait()
    for thread in threads:
        thread.join(timeout=5)
    end_time = _utc_now()
    _append_log_footer(stdout_log, end_time=end_time, return_code=return_code)
    _append_log_footer(stderr_log, end_time=end_time, return_code=return_code)
    return {"return_code": return_code, "child_pid": proc.pid, "end_time": end_time, "command_text": command_text}


def _start_gpu_monitor(path: Path, *, interval_sec: float, enabled: bool) -> tuple[threading.Event | None, threading.Thread | None]:
    if not enabled:
        return None, None
    stop_event = threading.Event()

    def monitor() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "timestamp,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib\n",
                encoding="utf-8",
            )
        while not stop_event.is_set():
            timestamp = _utc_now()
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=20,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    with path.open("a", encoding="utf-8", buffering=1) as handle:
                        for line in proc.stdout.strip().splitlines():
                            handle.write(f"{timestamp},{line}\n")
                            handle.flush()
            except Exception:
                pass
            stop_event.wait(max(1.0, float(interval_sec)))

    thread = threading.Thread(target=monitor, name="medai-nnunet-gpu-monitor", daemon=True)
    thread.start()
    return stop_event, thread


def _stop_gpu_monitor(stop_event: threading.Event | None, thread: threading.Thread | None) -> None:
    if stop_event is None:
        return
    stop_event.set()
    if thread is not None:
        thread.join(timeout=5)


def _coerce_label_id(value, label_arr=None) -> int | None:
    """Coerce a dataset.json label value to a single representative int id.

    nnUNet dataset.json labels are usually ``{name: int}``. For region-based
    datasets the value can be a list/tuple of ids (e.g. ``[1, 2]``). In that case
    we pick a representative id: if a combined label array is available we choose
    the first id that actually occurs in the array, otherwise the first id.
    """
    # Scalar int / float / numeric-string
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().lstrip("-").isdigit()):
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return None
    # Region-based: list/tuple of ids
    if isinstance(value, (list, tuple)) and value:
        ids: list[int] = []
        for v in value:
            try:
                ids.append(int(v))
            except Exception:
                try:
                    ids.append(int(float(v)))
                except Exception:
                    continue
        if not ids:
            return None
        if label_arr is not None:
            present = [i for i in ids if i != 0]
            for i in present:
                try:
                    import numpy as _np
                    if bool((_np.asarray(label_arr) == i).any()):
                        return i
                except Exception:
                    break
            for i in present:
                return i
        for i in ids:
            if i != 0:
                return i
        return ids[0]
    return None


def load_labels(dataset_json: Path, label_arr=None) -> dict[str, int]:
    """Read ``labels`` from a nnUNet dataset.json into ``{name: int_id}``.

    Tolerates both ``{name: id}`` and region-based ``{name: [ids]}`` formats.
    """
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    raw = data.get("labels") or {}
    labels: dict[str, int] = {}
    for name, value in raw.items():
        key = str(name).strip()
        if key.lower() == "background":
            continue
        ivalue = _coerce_label_id(value, label_arr=label_arr)
        if ivalue is None or ivalue == 0:
            continue
        labels[key] = ivalue
    return labels


def dump_local_labels(dataset_json: Path, out_path: Path, label_arr=None) -> dict[str, int]:
    """Write ``{local_label_name: int_id}`` to ``out_path`` and return the dict."""
    labels = load_labels(dataset_json, label_arr=label_arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    return labels


def split_labelmap(label_map: Path, dataset_json: Path, seg_dir: Path, requested_organs: list[str] | None = None) -> dict:
    try:
        import numpy as np
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nibabel and numpy are required to split nnUNet label maps. Install requirements.txt first.") from exc

    labels = load_labels(dataset_json)
    if requested_organs:
        requested = set()
        for organ in requested_organs:
            requested.add(organ)
            requested.add(ORGAN_ALIASES.get(organ, organ))
        labels = {k: v for k, v in labels.items() if k in requested}
    img = nib.load(str(label_map))
    arr = np.asanyarray(img.dataobj)
    seg_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for organ, value in sorted(labels.items()):
        mask = (arr == value).astype("uint8")
        if int(mask.sum()) == 0:
            continue
        out = seg_dir / f"{organ}.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out))
        written.append({"organ": organ, "label_value": value, "path": str(out), "voxels": int(mask.sum())})
    return {"num_written": len(written), "written_masks": written, "labels_available": labels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input CT .nii.gz")
    ap.add_argument("--output", required=True, help="Case output folder. segmentations/*.nii.gz will be written here.")
    ap.add_argument("--dataset-id", required=True, help="nnUNet dataset id, e.g. 551 or 1339")
    ap.add_argument("--nnunet-results", required=True, help="Folder used as nnUNet_results")
    ap.add_argument("--dataset-json", required=True, help="dataset.json containing label map")
    ap.add_argument("--model-folder", default=None, help="Optional trained model folder. Use this for ePAI qchen76_2025_0421, which is launched with nnUNetv2_predict_from_modelfolder -m.")
    ap.add_argument("--workdir", default=None, help="Optional working directory for model-specific entrypoints. ePAI expects csv_header.csv in its train/binary folder.")
    ap.add_argument("--use-python-api", action="store_true", help="Use Python API directly instead of subprocess (for models trained with ePAI nnunetv2 but needing standard predict).")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    ap.add_argument("--predict-executable", default=os.getenv("MEDAI_NNUNETV2_PREDICT", "nnUNetv2_predict"), help="Stable nnUNetv2_predict executable path/name.")
    ap.add_argument("--predict-from-modelfolder-executable", default=os.getenv("MEDAI_NNUNETV2_PREDICT_FROM_MODELFOLDER", "nnUNetv2_predict_from_modelfolder"), help="Stable nnUNetv2_predict_from_modelfolder executable path/name.")
    ap.add_argument("--sitecustomize-path", default=os.getenv("MEDAI_NNUNET_COMPAT_SITECUSTOMIZE"), help="Optional compatibility sitecustomize.py injected only into the nnUNet subprocess.")
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--device", default=None, help="Optional CUDA_VISIBLE_DEVICES value or cpu")
    ap.add_argument("--organs", default=None, help="Optional comma-separated organ names to split")
    ap.add_argument("--output-label-mode", choices=["all_organs", "pancreas_only"], default="all_organs", help="For ePAI native code compatibility; this wrapper keeps all combined labels unless --organs restricts splitting.")
    ap.add_argument("--per-model-dir", default=None, help="Optional directory where the unified per-model contract artifacts (combined_labels.nii.gz, local_labels.json) are written. Defaults to --output.")
    ap.add_argument("--diagnostic", action="store_true", help="Enable smoke/debug diagnostics: retain stage metadata and use conservative nnUNet worker counts.")
    ap.add_argument("--keep-workdir", action="store_true", help="Keep the prepared input and combined nnUNet output workdir under --output for timeout inspection.")
    ap.add_argument("--workdir-root", default=None, help="Optional root for retained diagnostic workdirs. Defaults to <output>/diagnostic_workdirs.")
    ap.add_argument("--preprocess-workers", type=int, default=None, help="Optional nnUNet preprocessing worker count; diagnostic mode defaults this to 1.")
    ap.add_argument("--export-workers", type=int, default=None, help="Optional nnUNet segmentation export worker count; diagnostic mode defaults this to 1.")
    ap.add_argument("--gpu-monitor", action="store_true", help="Diagnostic only: sample nvidia-smi to gpu_monitor.csv while prediction runs.")
    ap.add_argument("--gpu-monitor-interval-sec", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / "segmentations"
    dataset_json = Path(args.dataset_json).resolve()
    nnunet_results = Path(args.nnunet_results).resolve()
    model_folder = Path(args.model_folder).resolve() if args.model_folder else None
    workdir = Path(args.workdir).resolve() if args.workdir else None
    per_model_dir = Path(args.per_model_dir).resolve() if args.per_model_dir else output
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)
    per_model_seg_dir = per_model_dir / "segmentations"

    case_id = image.parent.name if image.name == "ct.nii.gz" else image.name.replace(".nii.gz", "").replace("_0000", "")
    requested_organs = [x.strip() for x in args.organs.replace(";", ",").split(",") if x.strip()] if args.organs else None

    if args.dry_run:
        if model_folder:
            command = [
                args.predict_from_modelfolder_executable, "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", "<input_csv>", "--output_csv", "<output_csv>",
                "--continue_prediction", "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
        else:
            command = [
                args.predict_executable, "-d", str(args.dataset_id), "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
                "-tr", args.trainer, "-c", args.configuration, "-f", str(args.folds), "-p", args.plans, "-chk", args.checkpoint_name, "--continue_prediction",
            ]
        print(json.dumps({"status": "dry_run", "command": command, "output": str(output), "seg_dir": str(seg_dir), "output_label_mode": args.output_label_mode}, indent=2))
        return 0

    if not image.exists():
        raise FileNotFoundError(image)
    if not dataset_json.exists():
        raise FileNotFoundError(dataset_json)
    if not nnunet_results.exists():
        raise FileNotFoundError(nnunet_results)
    if model_folder and not model_folder.exists():
        raise FileNotFoundError(model_folder)
    if workdir and not workdir.exists():
        raise FileNotFoundError(workdir)

    stdout_log = output / "nnunet_stdout.log"
    stderr_log = output / "nnunet_stderr.log"
    stage_log = output / "nnunet_stage_events.jsonl"
    status_path = output / "nnunet_status.json"
    metadata_path = output / "nnunet_run_metadata.json"
    gpu_monitor_path = output / "gpu_monitor.csv"
    stage_log.write_text("", encoding="utf-8")

    retain_workdir = bool(args.keep_workdir or args.diagnostic)
    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    if retain_workdir:
        retained_root = Path(args.workdir_root).resolve() if args.workdir_root else output / "diagnostic_workdirs"
        retained_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="medai_nnunet_", dir=str(retained_root))).resolve()
    else:
        tmp_holder = tempfile.TemporaryDirectory(prefix="medai_nnunet_")
        tmp = Path(tmp_holder.name).resolve()
    retained_workdir = str(tmp) if retain_workdir else None

    try:
        input_dir = tmp / "input"
        combined_dir = tmp / "combined"
        input_dir.mkdir(); combined_dir.mkdir()
        prepared = input_dir / f"{case_id}_0000.nii.gz"
        _append_stage(
            stage_log,
            status_path,
            "prepare_input_started",
            image=str(image),
            prepared_input=str(prepared),
            retained_workdir=retained_workdir,
        )
        shutil.copy2(image, prepared)
        _append_stage(stage_log, status_path, "prepare_input_completed", input_dir=str(input_dir), prepared_input=str(prepared))

        env = os.environ.copy()
        env["EPAI_OUTPUT_LABEL_MODE"] = args.output_label_mode
        env["nnUNet_results"] = str(nnunet_results)
        env.setdefault("nnUNet_raw", str(tmp / "nnUNet_raw"))
        env.setdefault("nnUNet_preprocessed", str(tmp / "nnUNet_preprocessed"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        if args.device:
            if args.device.lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env["CUDA_VISIBLE_DEVICES"] = args.device
        # If workdir contains its own nnunetv2 package, prepend it to PYTHONPATH so
        # the subprocess uses the correct architecture (e.g. VSmTrans vs ePAI).
        if workdir:
            vsm_nnunet = workdir / "nnunetv2"
            if not vsm_nnunet.exists():
                # workdir may be the nnUNet root (contains nnunetv2/ subdir)
                vsm_nnunet = workdir
            if (vsm_nnunet / "__init__.py").exists() or (vsm_nnunet / "nnunetv2").exists():
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(workdir) + (":" + existing if existing else "")
        if args.sitecustomize_path:
            sitecustomize = Path(args.sitecustomize_path).resolve()
            if not sitecustomize.exists():
                raise FileNotFoundError(sitecustomize)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(sitecustomize.parent) + (":" + existing if existing else "")

        metadata_doc = {
            "stage": "nnunet_wrapper",
            "status": "running",
            "start_time": _utc_now(),
            "python_executable": sys.executable,
            "predict_executable": args.predict_from_modelfolder_executable if model_folder else args.predict_executable,
            "nnUNet_results": str(nnunet_results),
            "workdir": str(workdir) if workdir else None,
            "temporary_workdir": str(tmp),
            "retained_workdir": retained_workdir,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "input_dir": str(input_dir),
            "combined_output_dir": str(combined_dir),
            "per_model_output_dir": str(per_model_dir),
            "output": str(output),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stage_log": str(stage_log),
            "gpu_monitor_csv": str(gpu_monitor_path) if (args.gpu_monitor or _truthy(os.getenv("MEDAI_NNUNET_GPU_MONITOR"))) else None,
            "diagnostic": bool(args.diagnostic),
            "keep_workdir": bool(retain_workdir),
            "preprocess_workers": _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=None),
            "export_workers": _worker_count(args.export_workers, diagnostic=args.diagnostic, default=None),
        }
        _write_json(metadata_path, metadata_doc)

        gpu_stop, gpu_thread = _start_gpu_monitor(
            gpu_monitor_path,
            interval_sec=args.gpu_monitor_interval_sec,
            enabled=bool(args.gpu_monitor or _truthy(os.getenv("MEDAI_NNUNET_GPU_MONITOR"))),
        )
        proc_return_code: int
        proc_end_time: str
        proc_end_time = ""
        proc_return_code = 1
        if model_folder:
            input_csv = output / "epai_input.csv"
            output_csv = output / "epai_output.csv"
            input_csv.write_text(f"Original ID,BDMAP ID\n{case_id},{case_id}\n", encoding="utf-8")
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=3)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=3)
            cmd = [
                args.predict_from_modelfolder_executable, "-i", str(input_dir), "-o", str(combined_dir),
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", str(input_csv), "--output_csv", str(output_csv),
                "--continue_prediction", "-npp", str(npp), "-nps", str(nps), "-num_parts", "1", "-part_id", "0",
                "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
            if args.save_probabilities:
                cmd.append("--save_probabilities")
            if args.device and args.device.lower() == "cpu":
                cmd.extend(["-device", "cpu"])
            metadata_doc.update({
                "command": [str(part) for part in cmd],
                "command_text": shlex.join([str(part) for part in cmd]),
                "predict_executable": args.predict_from_modelfolder_executable,
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command=metadata_doc["command_text"], child_backend="subprocess")
            proc_info = _run_streaming_subprocess(
                cmd,
                env=env,
                cwd=workdir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                metadata=metadata_doc,
            )
            proc_return_code = int(proc_info["return_code"])
            proc_end_time = str(proc_info["end_time"])
        elif args.use_python_api:
            # Use Python API directly — needed when the model was trained with ePAI's nnunetv2
            # but the ePAI predict entrypoint requires a CSV (incompatible with standard usage).
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=2)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=2)
            metadata_doc.update({
                "command": ["python_api"],
                "command_text": "python_api",
                "predict_executable": "python_api",
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _write_log_header(stdout_log, "stdout", metadata_doc)
            _write_log_header(stderr_log, "stderr", metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command="python_api", child_backend="python_api")
            with stdout_log.open("a", encoding="utf-8", buffering=1) as stdout_handle, stderr_log.open("a", encoding="utf-8", buffering=1) as stderr_handle:
                with redirect_stdout(stdout_handle), redirect_stderr(stderr_handle):
                    try:
                        import sys as _sys
                        if workdir:
                            _sys.path.insert(0, str(workdir))
                        else:
                            _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "ePAI-main" / "train"))
                        import torch as _torch
                        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
                        from batchgenerators.utilities.file_and_folder_operations import join as _join

                        _device = _torch.device("cpu" if (args.device and args.device.lower() == "cpu") else "cuda")
                        predictor = nnUNetPredictor(
                            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                            perform_everything_on_device=True, device=_device,
                            verbose=False, verbose_preprocessing=False, allow_tqdm=True
                        )
                        # model folder = nnunet_results / DatasetXXX_name / trainer__plans__config
                        _trainer_folder = Path(str(nnunet_results)) / f"Dataset{args.dataset_id:03d}_{args.dataset_id}" if False else None
                        # Find the correct trainer folder under nnunet_results
                        import glob as _glob
                        _pattern = str(nnunet_results / f"Dataset{int(args.dataset_id):03d}_*" / f"{args.trainer}__{args.plans}__{args.configuration}")
                        _matches = _glob.glob(_pattern)
                        if not _matches:
                            raise FileNotFoundError(f"No trainer folder found matching: {_pattern}")
                        _trainer_folder = Path(_matches[0])
                        predictor.initialize_from_trained_model_folder(
                            str(_trainer_folder),
                            use_folds=(args.folds,),
                            checkpoint_name=args.checkpoint_name
                        )
                        _input_files = sorted([str(f) for f in input_dir.glob("*.nii.gz")])
                        _list_of_lists = [[f] for f in _input_files]
                        _output_files = [str(combined_dir / Path(f).name.replace("_0000.nii.gz", ".nii.gz")) for f in _input_files]
                        combined_dir.mkdir(parents=True, exist_ok=True)
                        _data_iter = predictor._internal_get_data_iterator_from_lists_of_filenames(
                            _list_of_lists, None, _output_files, num_processes=npp
                        )
                        predictor.predict_from_data_iterator(_data_iter, save_probabilities=False, num_processes_segmentation_export=nps)
                        print("Python API inference completed.", flush=True)
                        proc_return_code = 0
                    except Exception:
                        import traceback as _tb
                        print(_tb.format_exc(), file=sys.stderr, flush=True)
                        proc_return_code = 1
            proc_end_time = _utc_now()
            _append_log_footer(stdout_log, end_time=proc_end_time, return_code=proc_return_code)
            _append_log_footer(stderr_log, end_time=proc_end_time, return_code=proc_return_code)
        else:
            cmd = [
                args.predict_executable, "-d", str(args.dataset_id), "-i", str(input_dir), "-o", str(combined_dir),
                "-tr", args.trainer, "-c", args.configuration, "-f", str(args.folds), "-p", args.plans, "-chk", args.checkpoint_name, "--continue_prediction",
            ]
            npp = _worker_count(args.preprocess_workers, diagnostic=args.diagnostic, default=None)
            nps = _worker_count(args.export_workers, diagnostic=args.diagnostic, default=None)
            if npp is not None:
                cmd.extend(["-npp", str(npp)])
            if nps is not None:
                cmd.extend(["-nps", str(nps)])
            if args.save_probabilities:
                cmd.append("--save_probabilities")
            metadata_doc.update({
                "command": [str(part) for part in cmd],
                "command_text": shlex.join([str(part) for part in cmd]),
                "predict_executable": args.predict_executable,
                "preprocess_workers": npp,
                "export_workers": nps,
            })
            _write_json(metadata_path, metadata_doc)
            _append_stage(stage_log, status_path, "predictor_started", command=metadata_doc["command_text"], child_backend="subprocess")
            proc_info = _run_streaming_subprocess(
                cmd,
                env=env,
                cwd=workdir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                metadata=metadata_doc,
            )
            proc_return_code = int(proc_info["return_code"])
            proc_end_time = str(proc_info["end_time"])
        _stop_gpu_monitor(gpu_stop, gpu_thread)

        _append_stage(stage_log, status_path, "prediction_completed", return_code=proc_return_code, end_time=proc_end_time)
        metadata_doc.update({"status": "prediction_completed", "end_time": proc_end_time, "return_code": proc_return_code})
        _write_json(metadata_path, metadata_doc)
        if proc_return_code != 0:
            failure = {
                "status": "failed",
                "return_code": proc_return_code,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stderr_tail": _tail_file(stderr_log),
                "stage_log": str(stage_log),
                "run_metadata": str(metadata_path),
                "last_stage": "prediction_completed",
                "retained_workdir": retained_workdir,
            }
            _write_json(per_model_dir / "inference_summary.json", failure)
            if per_model_dir != output:
                _write_json(output / "inference_summary.json", failure)
            print(json.dumps(failure, indent=2))
            return proc_return_code

        _append_stage(stage_log, status_path, "split_started", combined_output_dir=str(combined_dir), per_model_output_dir=str(per_model_dir))
        candidates = sorted(combined_dir.glob("*.nii.gz"))
        if not candidates:
            failure = {
                "status": "failed",
                "return_code": 2,
                "reason": "nnUNet produced no combined label map",
                "combined_dir": str(combined_dir),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stage_log": str(stage_log),
                "run_metadata": str(metadata_path),
                "last_stage": "split_started",
                "retained_workdir": retained_workdir,
            }
            _write_json(per_model_dir / "inference_summary.json", failure)
            if per_model_dir != output:
                _write_json(output / "inference_summary.json", failure)
            print(json.dumps(failure, indent=2))
            return 2

        # Python API outputs a combined label map named after the case (e.g. PanTS_00000002.nii.gz)
        # Standard subprocess outputs a combined label map too. Either way, take the first .nii.gz.
        combined = candidates[0]
        combined_out = per_model_dir / "combined_labels.nii.gz"
        shutil.copy2(combined, combined_out)
        # Also keep a copy at --output for backward compatibility when per-model-dir
        # differs from output.
        if combined_out != output / "combined_labels.nii.gz":
            shutil.copy2(combined, output / "combined_labels.nii.gz")
        # Dump local_labels.json (name -> int id), region-aware via the actual array.
        try:
            import numpy as _np
            import nibabel as _nib
            _label_arr = _np.asanyarray(_nib.load(str(combined)).dataobj)
        except Exception:
            _label_arr = None
        local_labels = dump_local_labels(dataset_json, per_model_dir / "local_labels.json", label_arr=_label_arr)
        split = split_labelmap(combined, dataset_json, per_model_seg_dir, requested_organs=requested_organs)
        if per_model_seg_dir != seg_dir:
            seg_dir.mkdir(parents=True, exist_ok=True)
            for mask_path in per_model_seg_dir.glob("*.nii.gz"):
                shutil.copy2(mask_path, seg_dir / mask_path.name)
        _append_stage(stage_log, status_path, "split_completed", combined_label=str(combined_out), num_written=split.get("num_written"))
        summary = {
            "status": "success", "case_id": case_id,
            "combined_label": str(combined_out),
            "local_labels": str(per_model_dir / "local_labels.json"),
            "num_local_labels": len(local_labels),
            "segmentation_output": str(per_model_seg_dir),
            "legacy_segmentation_output": str(seg_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stage_log": str(stage_log),
            "run_metadata": str(metadata_path),
            "gpu_monitor_csv": str(gpu_monitor_path) if gpu_monitor_path.exists() else None,
            "workdir": str(tmp),
            "retained_workdir": retained_workdir,
            "last_stage": "split_completed",
            "return_code": 0,
            **split,
        }
        (per_model_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if per_model_dir != output:
            (output / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metadata_doc.update({"status": "success", "last_stage": "split_completed", "return_code": 0})
        _write_json(metadata_path, metadata_doc)
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
