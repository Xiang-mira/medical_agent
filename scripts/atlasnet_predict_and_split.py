#!/usr/bin/env python3
"""Run the public ATLAS-Net checkpoint with the medai CLI output contract."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_RELATIVE_PATH = Path(
    "nnUNet_results/Dataset001_ATLASNet/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
DEFAULT_POSTPROCESS_RELATIVE_PATH = Path("utils/atlas_postprocess.py")
PROJECT_FALLBACK_POSTPROCESS = Path("third_party/atlasnet/utils/atlas_postprocess.py")
ATLAS_POSTPROCESS_LABEL_MAP = {
    "aorta": 1,
    "adrenal_l": 2,
    "adrenal_r": 3,
    "common_bile_duct": 4,
    "celiac_artery": 5,
    "colon": 6,
    "duodenum": 7,
    "gallbladder": 8,
    "ivc": 9,
    "kidney_l": 10,
    "kidney_r": 11,
    "liver": 12,
    "pancreas": 13,
    "pancreatic_duct": 14,
    "superior_mesenteric_artery": 15,
    "intestine": 16,
    "spleen": 17,
    "stomach": 18,
    "portal_splenic_vein": 19,
    "renal_vein_l": 20,
    "renal_vein_r": 21,
    "cbd_stent": 22,
}


def _load_labels(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("labels", data)
    labels: dict[str, int] = {}
    for name, value in raw.items():
        try:
            label_id = int(value)
        except (TypeError, ValueError):
            continue
        if label_id != 0 and str(name).lower() != "background":
            labels[str(name)] = label_id
    return labels


def _case_id(image: Path) -> str:
    if image.name == "ct.nii.gz":
        return image.parent.name
    name = image.name.removesuffix(".nii.gz")
    return name.removesuffix("_0000")


def _write_text(path: Path, value: str | None) -> None:
    path.write_text(value or "", encoding="utf-8")


def _postprocess_snippet(atlas_root: Path, pred_path: Path | str, ct_path: Path | str, output_path: Path | str) -> str:
    return "\n".join([
        "import sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(atlas_root)!r})",
        "from utils.atlas_postprocess import run_from_files",
        (
            "run_from_files("
            f"pred_path={str(pred_path)!r}, "
            f"ct_path={str(ct_path)!r}, "
            f"output_path={str(output_path)!r}, "
            f"label_map={ATLAS_POSTPROCESS_LABEL_MAP!r})"
        ),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS-Net wrapper for the medai CLI")
    parser.add_argument("--image", required=True, help="Input 3D CT volume (.nii.gz)")
    parser.add_argument("--output", required=True, help="Case output folder")
    parser.add_argument("--atlas-root", default="checkpoints/ATLAS-Net")
    parser.add_argument("--label-map", default="configs/atlasnet_label_map.json")
    parser.add_argument("--per-model-dir", default=None)
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or a CUDA device index")
    parser.add_argument("--num-processes-preprocessing", type=int, default=2)
    parser.add_argument("--num-processes-segmentation-export", type=int, default=2)
    parser.add_argument(
        "--postprocess-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Run the official ATLAS-Net CT-aware postprocess when available.",
    )
    parser.add_argument(
        "--postprocess-script",
        default=None,
        help="Optional path to utils/atlas_postprocess.py from the cloned ATLAS-Net repo.",
    )
    parser.add_argument("--skip-postprocess", action="store_true", help="Alias for --postprocess-mode never.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    atlas_root = Path(args.atlas_root).resolve()
    label_map = Path(args.label_map).resolve()
    per_model_dir = Path(args.per_model_dir).resolve() if args.per_model_dir else output
    seg_dir = output / "segmentations"
    model_folder = atlas_root / MODEL_RELATIVE_PATH
    postprocess_mode = "never" if args.skip_postprocess else args.postprocess_mode
    if args.postprocess_script:
        postprocess_script = Path(args.postprocess_script).resolve()
        postprocess_import_root = postprocess_script.parent.parent
    else:
        checkpoint_postprocess = atlas_root / DEFAULT_POSTPROCESS_RELATIVE_PATH
        fallback_postprocess = (Path(__file__).resolve().parents[1] / PROJECT_FALLBACK_POSTPROCESS).resolve()
        postprocess_script = checkpoint_postprocess if checkpoint_postprocess.is_file() else fallback_postprocess
        postprocess_import_root = atlas_root if checkpoint_postprocess.is_file() else fallback_postprocess.parent.parent

    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "wrapper": "atlasnet",
        "image": str(image),
        "output": str(output),
        "segmentation_output": str(seg_dir),
        "per_model_dir": str(per_model_dir),
        "atlas_root": str(atlas_root),
        "model_folder": str(model_folder),
        "label_map": str(label_map),
        "postprocess_mode": postprocess_mode,
        "postprocess_script": str(postprocess_script),
        "postprocess_import_root": str(postprocess_import_root),
    }

    command_preview = [
        "nnUNetv2_predict_from_modelfolder",
        "-i", "<prepared_input_dir>",
        "-o", "<prediction_dir>",
        "-m", str(model_folder),
        "-f", args.folds,
        "--continue_prediction",
        "-chk", args.checkpoint_name,
        "-npp", str(args.num_processes_preprocessing),
        "-nps", str(args.num_processes_segmentation_export),
    ]
    if args.device and args.device.lower() in {"cpu", "cuda", "mps"}:
        command_preview.extend(["-device", args.device.lower()])

    postprocess_preview = [
        sys.executable,
        "-c",
        _postprocess_snippet(postprocess_import_root, "<raw_prediction>", image, "<postprocessed_prediction>"),
    ]

    if args.dry_run:
        print(json.dumps({
            **summary,
            "status": "dry_run",
            "command": command_preview,
            "postprocess_command": postprocess_preview if postprocess_mode != "never" else None,
        }, indent=2))
        return 0

    if not image.is_file():
        raise FileNotFoundError(f"ATLAS-Net input image not found: {image}")
    if not label_map.is_file():
        raise FileNotFoundError(f"ATLAS-Net label map not found: {label_map}")
    if not model_folder.is_dir():
        raise FileNotFoundError(
            f"ATLAS-Net model folder not found: {model_folder}. "
            "Clone https://huggingface.co/Koushik45048545309/Atlas-Net "
            f"into {atlas_root}."
        )
    checkpoint = model_folder / f"fold_{args.folds}" / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ATLAS-Net checkpoint not found: {checkpoint}")
    if postprocess_mode == "always" and not postprocess_script.is_file():
        raise FileNotFoundError(f"ATLAS-Net postprocess script not found: {postprocess_script}")

    with tempfile.TemporaryDirectory(prefix="medai_atlasnet_") as temp_root:
        temp = Path(temp_root)
        input_dir = temp / "imagesTs"
        prediction_dir = temp / "predictions"
        postprocess_dir = temp / "postprocessed"
        input_dir.mkdir()
        prediction_dir.mkdir()
        postprocess_dir.mkdir()
        shutil.copy2(image, input_dir / f"{_case_id(image)}_0000.nii.gz")

        command = list(command_preview)
        command[command.index("<prepared_input_dir>")] = str(input_dir)
        command[command.index("<prediction_dir>")] = str(prediction_dir)
        env = os.environ.copy()
        env["nnUNet_results"] = str(atlas_root / "nnUNet_results")
        env.setdefault("nnUNet_raw", str(temp / "nnUNet_raw"))
        env.setdefault("nnUNet_preprocessed", str(temp / "nnUNet_preprocessed"))
        if args.device:
            device = args.device.lower()
            if device.isdigit():
                env["CUDA_VISIBLE_DEVICES"] = device
            elif device.startswith("cuda:"):
                env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
        )
        _write_text(output / "atlasnet_stdout.log", completed.stdout)
        _write_text(output / "atlasnet_stderr.log", completed.stderr)
        if completed.returncode != 0:
            print(json.dumps({
                **summary,
                "status": "failed",
                "return_code": completed.returncode,
                "stderr_tail": (completed.stderr or "")[-4000:],
            }, indent=2))
            return completed.returncode

        predictions = sorted(prediction_dir.glob("*.nii.gz"))
        if not predictions:
            print(json.dumps({**summary, "status": "failed", "reason": "ATLAS-Net produced no prediction"}, indent=2))
            return 2

        raw_prediction = predictions[0]
        final_prediction = raw_prediction
        postprocess_info = {
            "postprocess_status": "skipped",
            "postprocess_used": False,
            "postprocess_reason": "postprocess disabled",
            "raw_prediction": str(raw_prediction),
        }
        if postprocess_mode != "never":
            if postprocess_script.is_file():
                postprocessed_prediction = postprocess_dir / raw_prediction.name
                postprocess_command = [
                    sys.executable,
                    "-c",
                    _postprocess_snippet(postprocess_import_root, raw_prediction, image, postprocessed_prediction),
                ]
                postprocess = subprocess.run(
                    postprocess_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                _write_text(output / "atlasnet_postprocess_stdout.log", postprocess.stdout)
                _write_text(output / "atlasnet_postprocess_stderr.log", postprocess.stderr)
                if postprocess.returncode == 0 and postprocessed_prediction.is_file():
                    final_prediction = postprocessed_prediction
                    postprocess_info = {
                        "postprocess_status": "success",
                        "postprocess_used": True,
                        "postprocess_reason": None,
                        "postprocess_return_code": postprocess.returncode,
                        "postprocess_command": postprocess_command,
                        "raw_prediction": str(raw_prediction),
                        "postprocessed_prediction": str(postprocessed_prediction),
                    }
                else:
                    postprocess_info = {
                        "postprocess_status": "failed",
                        "postprocess_used": False,
                        "postprocess_reason": "official postprocess failed; using raw prediction in auto mode",
                        "postprocess_return_code": postprocess.returncode,
                        "postprocess_command": postprocess_command,
                        "raw_prediction": str(raw_prediction),
                        "postprocessed_prediction": str(postprocessed_prediction),
                        "postprocess_stderr_tail": (postprocess.stderr or "")[-4000:],
                    }
                    if postprocess_mode == "always":
                        print(json.dumps({
                            **summary,
                            **postprocess_info,
                            "status": "failed",
                        }, indent=2))
                        return postprocess.returncode or 4
            else:
                postprocess_info = {
                    "postprocess_status": "unavailable",
                    "postprocess_used": False,
                    "postprocess_reason": f"postprocess script not found: {postprocess_script}",
                    "raw_prediction": str(raw_prediction),
                }
                if postprocess_mode == "always":
                    print(json.dumps({**summary, **postprocess_info, "status": "failed"}, indent=2))
                    return 4

        combined = per_model_dir / "combined_labels.nii.gz"
        shutil.copy2(final_prediction, combined)
        local_labels = _load_labels(label_map)
        (per_model_dir / "local_labels.json").write_text(
            json.dumps(local_labels, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        split_script = Path(__file__).resolve().parent / "split_combined_labelmap.py"
        split = subprocess.run(
            [
                sys.executable,
                str(split_script),
                "--labelmap", str(combined),
                "--labels-json", str(label_map),
                "--output", str(seg_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        _write_text(output / "split_stdout.log", split.stdout)
        _write_text(output / "split_stderr.log", split.stderr)

    masks = sorted(seg_dir.glob("*.nii.gz"))
    status = "success" if split.returncode == 0 and masks else "failed"
    print(json.dumps({
        **summary,
        **postprocess_info,
        "status": status,
        "combined_label": str(combined),
        "num_masks": len(masks),
        "sample_masks": [path.name for path in masks[:80]],
    }, indent=2))
    return 0 if status == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
