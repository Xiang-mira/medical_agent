#!/usr/bin/env python3
"""Run an nnUNet v2 checkpoint on one CT and split the combined label map into organ-wise masks.

This wrapper is designed for the teacher-provided checkpoint folders exported from Google Drive:
CADS_series, MOOSE_series, nnUNet_private, and VSmTrans.

It never bundles private weights. It assumes the checkpoint folder already exists locally or on the
server and only standardizes I/O for the medai CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ORGAN_ALIASES = {
    "celiac_aa_celiac_artery": "celiac_aa",
    "inferior_vena_cava": "postcava",
    "small_intestine": "intestine",
    "portal_splenic_veins": "veins",
    "portal_vein_and_splenic_vein": "veins",
}


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
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--device", default=None, help="Optional CUDA_VISIBLE_DEVICES value or cpu")
    ap.add_argument("--organs", default=None, help="Optional comma-separated organ names to split")
    ap.add_argument("--output-label-mode", choices=["all_organs", "pancreas_only"], default="all_organs", help="For ePAI native code compatibility; this wrapper keeps all combined labels unless --organs restricts splitting.")
    ap.add_argument("--per-model-dir", default=None, help="Optional directory where the unified per-model contract artifacts (combined_labels.nii.gz, local_labels.json) are written. Defaults to --output.")
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
                "nnUNetv2_predict_from_modelfolder", "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", "<input_csv>", "--output_csv", "<output_csv>",
                "--continue_prediction", "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
        else:
            command = [
                "nnUNetv2_predict", "-d", str(args.dataset_id), "-i", "<prepared_input_dir>", "-o", "<combined_output_dir>",
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

    with tempfile.TemporaryDirectory(prefix="medai_nnunet_") as td:
        tmp = Path(td)
        input_dir = tmp / "input"
        combined_dir = tmp / "combined"
        input_dir.mkdir(); combined_dir.mkdir()
        prepared = input_dir / f"{case_id}_0000.nii.gz"
        shutil.copy2(image, prepared)

        env = os.environ.copy()
        env["EPAI_OUTPUT_LABEL_MODE"] = args.output_label_mode
        env["nnUNet_results"] = str(nnunet_results)
        env.setdefault("nnUNet_raw", str(tmp / "nnUNet_raw"))
        env.setdefault("nnUNet_preprocessed", str(tmp / "nnUNet_preprocessed"))
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

        if model_folder:
            input_csv = output / "epai_input.csv"
            output_csv = output / "epai_output.csv"
            input_csv.write_text(f"Original ID,BDMAP ID\n{case_id},{case_id}\n", encoding="utf-8")
            cmd = [
                "nnUNetv2_predict_from_modelfolder", "-i", str(input_dir), "-o", str(combined_dir),
                "-m", str(model_folder), "-f", str(args.folds), "--input_csv", str(input_csv), "--output_csv", str(output_csv),
                "--continue_prediction", "-npp", "3", "-nps", "3", "-num_parts", "1", "-part_id", "0",
                "-chk", args.checkpoint_name, "--output_label_mode", args.output_label_mode,
            ]
            if args.save_probabilities:
                cmd.append("--save_probabilities")
            if args.device and args.device.lower() == "cpu":
                cmd.extend(["-device", "cpu"])
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(workdir) if workdir else None, check=False)
        elif args.use_python_api:
            # Use Python API directly — needed when the model was trained with ePAI's nnunetv2
            # but the ePAI predict entrypoint requires a CSV (incompatible with standard usage).
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
                    _list_of_lists, None, _output_files, num_processes=2
                )
                predictor.predict_from_data_iterator(_data_iter, save_probabilities=False, num_processes_segmentation_export=2)
                _stdout = "Python API inference completed."
                _stderr = ""
                _rc = 0
            except Exception as _e:
                import traceback as _tb
                _stdout = ""
                _stderr = _tb.format_exc()
                _rc = 1
            (output / "nnunet_stdout.log").write_text(_stdout, encoding="utf-8")
            (output / "nnunet_stderr.log").write_text(_stderr, encoding="utf-8")
            if _rc != 0:
                print(json.dumps({"status": "failed", "return_code": _rc, "stderr_tail": _stderr[-4000:]}, indent=2))
                return _rc
        else:
            cmd = [
                "nnUNetv2_predict", "-d", str(args.dataset_id), "-i", str(input_dir), "-o", str(combined_dir),
                "-tr", args.trainer, "-c", args.configuration, "-f", str(args.folds), "-p", args.plans, "-chk", args.checkpoint_name, "--continue_prediction",
            ]
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(workdir) if workdir else None, check=False)
            (output / "nnunet_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (output / "nnunet_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            if proc.returncode != 0:
                print(json.dumps({"status": "failed", "return_code": proc.returncode, "stderr_tail": (proc.stderr or "")[-4000:]}, indent=2))
                return proc.returncode

        candidates = sorted(combined_dir.glob("*.nii.gz"))
        if not candidates:
            print(json.dumps({"status": "failed", "reason": "nnUNet produced no combined label map", "combined_dir": str(combined_dir)}, indent=2))
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
        summary = {
            "status": "success", "case_id": case_id,
            "combined_label": str(combined_out),
            "local_labels": str(per_model_dir / "local_labels.json"),
            "num_local_labels": len(local_labels),
            "segmentation_output": str(per_model_seg_dir),
            "legacy_segmentation_output": str(seg_dir),
            **split,
        }
        (per_model_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if per_model_dir != output:
            (output / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
