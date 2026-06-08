#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="VISTA3D wrapper for medai CLI")
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--vista-root', default='third_party/VISTA3D-Inference-Pipeline-master')
    ap.add_argument('--label-map', default=None)
    ap.add_argument('--per-model-dir', default=None, help='Per-model contract output dir for combined_labels.nii.gz and local_labels.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / 'segmentations'
    vista_root = Path(args.vista_root).resolve()
    per_model_dir = Path(args.per_model_dir).resolve() if args.per_model_dir else output
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    label_map = (
        Path(args.label_map).resolve()
        if args.label_map
        else vista_root / 'label_mappings' / 'label_dict_127_abdomenAtlas3-1.json'
    )

    summary = {
        'wrapper': 'vista3d',
        'image': str(image),
        'output': str(output),
        'vista_root': str(vista_root),
        'label_map': str(label_map),
    }

    if args.dry_run:
        print(json.dumps({**summary, 'status': 'dry_run'}, indent=2))
        return 0

    if not image.exists():
        raise FileNotFoundError(image)
    if not vista_root.exists():
        raise FileNotFoundError(vista_root)
    if not label_map.exists():
        raise FileNotFoundError(label_map)

    with tempfile.TemporaryDirectory(prefix='medai_vista3d_') as td0:
        td = Path(td0)
        eval_dir = td / 'eval'
        eval_dir.mkdir()

        # Pass image path only — omit label_prompt so evaluator uses everything_labels
        # (all 117 abdomen labels). Passing null causes a type error in check_prompts_format.
        input_dict = json.dumps({'image': str(image)})

        # Override postprocessing to remove the batch-only Lambda transform
        # (labels2onehot.seperate_class) which requires VISTA3D_OUTPUT_DIR files.
        infer_override = {
            "postprocessing": {
                "_target_": "Compose",
                "transforms": [
                    {"_target_": "monai.apps.vista3d.transforms.VistaPostTransformd", "keys": "pred"},
                    {"_target_": "Invertd", "keys": "pred",
                     "transform": "$copy.deepcopy(@preprocessing)",
                     "orig_keys": "@image_key", "nearest_interp": True, "to_tensor": True},
                    {"_target_": "Lambdad",
                     "func": "$lambda x: torch.nan_to_num(x, nan=255)", "keys": "pred"},
                    {"_target_": "SaveImaged", "keys": "pred", "resample": False,
                     "data_root_dir": "@input_dir", "output_dir": "@output_dir",
                     "output_ext": "@output_ext", "output_dtype": "@output_dtype",
                     "output_postfix": "@output_postfix",
                     "separate_folder": "@separate_folder"},
                ],
            }
        }
        override_path = td / 'infer_override.json'
        override_path.write_text(json.dumps(infer_override), encoding='utf-8')

        import os
        env = {**os.environ, 'VISTA3D_OUTPUT_DIR': str(eval_dir)}

        command = [
            sys.executable, '-m', 'monai.bundle', 'run',
            '--config_file', f"['{vista_root / 'configs' / 'inference.json'}','{override_path}']",
            '--bundle_root', str(vista_root),
            '--input_dict', input_dict,
            '--input_dir', str(image.parent),
            '--output_dir', str(eval_dir),
            '--output_postfix', 'step1_117',
            '--separate_folder', 'False',
        ]

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
            cwd=str(vista_root),
            env=env,
        )
        (output / 'vista3d_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (output / 'vista3d_stderr.log').write_text(proc.stderr or '', encoding='utf-8')

        if proc.returncode != 0:
            print(json.dumps({
                **summary, 'status': 'failed',
                'return_code': proc.returncode,
                'stderr_tail': (proc.stderr or '')[-4000:],
            }, indent=2))
            return proc.returncode

        candidates = sorted(eval_dir.rglob('*.nii.gz'))
        if not candidates:
            print(json.dumps({
                **summary, 'status': 'failed',
                'reason': 'VISTA3D produced no nii.gz predictions',
                'eval_dir': str(eval_dir),
            }, indent=2))
            return 2

        if len(candidates) > 10:
            # Already split per-organ (e.g. batch mode output)
            for p in candidates:
                shutil.copy2(p, seg_dir / p.name)
        else:
            # Combined label map — split into per-organ masks
            combined = output / 'combined_labels.nii.gz'
            shutil.copy2(candidates[0], combined)
            # Copy to per_model_dir as well
            if per_model_dir != output:
                shutil.copy2(candidates[0], per_model_dir / 'combined_labels.nii.gz')
            split_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / 'split_combined_labelmap.py'),
                '--labelmap', str(combined),
                '--labels-json', str(label_map),
                '--output', str(seg_dir),
            ]
            split = subprocess.run(
                split_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=False,
                cwd=str(vista_root.parent.parent),  # project root
            )
            (output / 'split_stdout.log').write_text(split.stdout or '', encoding='utf-8')
            (output / 'split_stderr.log').write_text(split.stderr or '', encoding='utf-8')

        # Write local_labels.json (label_name -> int_id) from label map file.
        try:
            raw_labels = json.loads(label_map.read_text(encoding='utf-8'))
            # label_map format: {name: int_id} — filter out background (id=0) and non-int values.
            local_labels = {k: v for k, v in raw_labels.items() if isinstance(v, int) and v != 0}
            (per_model_dir / 'local_labels.json').write_text(json.dumps(local_labels, indent=2, sort_keys=True), encoding='utf-8')
        except Exception:
            pass

        masks = list(seg_dir.glob('*.nii.gz'))
        status = 'success' if masks else 'failed'
        print(json.dumps({
            **summary, 'status': status,
            'segmentation_output': str(seg_dir),
            'per_model_dir': str(per_model_dir),
            'num_masks': len(masks),
        }, indent=2))
        return 0 if status == 'success' else 3


if __name__ == '__main__':
    raise SystemExit(main())
