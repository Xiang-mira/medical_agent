#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "agent-harness") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "agent-harness"))

from cli_anything.medai.core.runtime_resolver import resolve_unest_python, scrub_env_for_runner  # noqa: E402

# UNEST outputs a 4-class combined label map (0=bg, 1=cortex, 2=medulla, 3=pelvicalyceal)
LABEL_MAP = {
    1: 'kidney_cortex',
    2: 'kidney_medulla',
    3: 'kidney_pelvicalyceal_system',
}


def _split_combined(combined_path: Path, seg_dir: Path) -> list[Path]:
    """Split UNEST combined label map into per-organ binary masks."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        return []

    img = nib.load(str(combined_path))
    arr = img.get_fdata().astype('int16')
    written = []
    for label_id, organ_name in LABEL_MAP.items():
        mask = (arr == label_id).astype('uint8')
        if mask.sum() == 0:
            continue
        out_path = seg_dir / f'{organ_name}.nii.gz'
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))
        written.append(out_path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="UNEST renal-structure wrapper for medai CLI")
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--unest-root',
                    default='checkpoints/UNEST/UNEST/renalStructures_UNEST_segmentation')
    ap.add_argument('--python-executable', default=None, help='Stable Python executable for MONAI/UNEST subprocess. Defaults to MEDAI_UNEST_PYTHON or current interpreter.')
    ap.add_argument('--per-model-dir', default=None, help='Per-model contract output dir for combined_labels.nii.gz and local_labels.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / 'segmentations'
    unest_root = Path(args.unest_root).resolve()
    python_executable = resolve_unest_python(explicit=args.python_executable, require_exists=not args.dry_run)
    per_model_dir = Path(args.per_model_dir).resolve() if args.per_model_dir else output
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'wrapper': 'unest',
        'image': str(image),
        'output': str(output),
        'unest_root': str(unest_root),
        'python_executable': str(python_executable),
    }

    if args.dry_run:
        print(json.dumps({**summary, 'status': 'dry_run'}, indent=2))
        return 0

    if not image.exists():
        raise FileNotFoundError(image)
    if not unest_root.exists():
        raise FileNotFoundError(unest_root)

    with tempfile.TemporaryDirectory(prefix='medai_unest_') as td0:
        td = Path(td0)
        inp_dir = td / 'dataset'
        out_dir = td / 'eval'
        inp_dir.mkdir()
        out_dir.mkdir()

        # UNEST expects files in dataset/ named *.nii.gz
        shutil.copy2(image, inp_dir / 'ct.nii.gz')

        # Override bundle_root, dataset_dir, output_dir via CLI flags.
        # PYTHONPATH must include the bundle's scripts/ so UNesT network can be imported.
        import os
        base_env = scrub_env_for_runner(os.environ, runner='unest')
        existing_pythonpath = base_env.get('PYTHONPATH', '')
        env_patch = {
            'PYTHONPATH': str(unest_root / 'scripts') + ((os.pathsep + existing_pythonpath) if existing_pythonpath else ''),
        }
        env = {**base_env, **env_patch}

        command = [
            str(python_executable), '-m', 'monai.bundle', 'run', 'evaluating',
            '--meta_file', str(unest_root / 'configs' / 'metadata.json'),
            '--config_file', str(unest_root / 'configs' / 'inference.json'),
            '--logging_file', str(unest_root / 'configs' / 'logging.conf'),
            '--bundle_root', str(unest_root),
            '--dataset_dir', str(inp_dir),
            '--output_dir', str(out_dir),
        ]

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
            cwd=str(unest_root),
            env=env,
        )
        (output / 'unest_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (output / 'unest_stderr.log').write_text(proc.stderr or '', encoding='utf-8')

        if proc.returncode != 0:
            print(json.dumps({
                **summary, 'status': 'failed',
                'return_code': proc.returncode,
                'stderr_tail': (proc.stderr or '')[-4000:],
            }, indent=2))
            return proc.returncode

        # Collect output: MONAI SaveImaged writes to out_dir/<stem>/<stem>_trans.nii.gz
        # or directly to out_dir/*.nii.gz depending on separate_folder setting.
        raw_outputs = sorted(out_dir.rglob('*.nii.gz'))
        if not raw_outputs:
            print(json.dumps({
                **summary, 'status': 'failed',
                'reason': 'UNEST produced no nii.gz output',
            }, indent=2))
            return 2

        # Try to split the combined label map into per-organ masks
        combined = raw_outputs[0]
        written = _split_combined(combined, seg_dir)

        if not written:
            # Fallback: copy raw outputs as-is
            for p in raw_outputs:
                shutil.copy2(p, seg_dir / p.name)

        # Write per_model contract artifacts.
        shutil.copy2(combined, per_model_dir / 'combined_labels.nii.gz')
        local_labels = {v: k for k, v in LABEL_MAP.items()}  # {int_id: name} → {name: int_id}
        (per_model_dir / 'local_labels.json').write_text(
            json.dumps({name: int_id for int_id, name in LABEL_MAP.items()}, indent=2, sort_keys=True),
            encoding='utf-8',
        )

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
