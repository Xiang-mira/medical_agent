#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, subprocess, tempfile
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description="ATLAS-Net wrapper for medai CLI")
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True, help='Case output folder')
    ap.add_argument('--atlas-root', default='checkpoints/ATLAS-Net')
    ap.add_argument('--label-map', default='configs/atlasnet_label_map.json')
    ap.add_argument('--mode', choices=['inference_sh','nnunetv2_predict'], default='inference_sh')
    ap.add_argument('--dataset-id', default='001')
    ap.add_argument('--trainer', default='nnUNetTrainer')
    ap.add_argument('--plans', default='nnUNetPlans')
    ap.add_argument('--configuration', default='3d_fullres')
    ap.add_argument('--folds', default='all')
    ap.add_argument('--device', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / 'segmentations'
    atlas_root = Path(args.atlas_root).resolve()
    label_map = Path(args.label_map).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    case_id = image.parent.name if image.name == 'ct.nii.gz' else image.name.replace('.nii.gz','').replace('_0000','')
    summary = {'wrapper':'atlasnet', 'image':str(image), 'output':str(output), 'atlas_root':str(atlas_root), 'label_map':str(label_map), 'mode':args.mode}
    if args.dry_run:
        summary['status'] = 'dry_run'
        summary['expected_output'] = 'segmentations/*.nii.gz'
        print(json.dumps(summary, indent=2))
        return 0
    if not image.exists():
        raise FileNotFoundError(image)
    if not label_map.exists():
        raise FileNotFoundError(label_map)
    with tempfile.TemporaryDirectory(prefix='medai_atlasnet_') as td0:
        td = Path(td0)
        imagesTs = td / 'nnUNet_eval' / 'Dataset001_ATLASNet' / 'imagesTs'
        pred = td / 'nnUNet_predictions'
        imagesTs.mkdir(parents=True)
        pred.mkdir(parents=True)
        prepared = imagesTs / f'{case_id}_0000.nii.gz'
        shutil.copy2(image, prepared)
        if args.mode == 'inference_sh':
            script = atlas_root / 'inference.sh'
            if not script.exists():
                raise FileNotFoundError(f'ATLAS-Net inference.sh not found: {script}. Download/mount ATLAS-Net first.')
            env = os.environ.copy()
            env['MEDAI_ATLAS_IMAGES_TS'] = str(imagesTs)
            env['MEDAI_ATLAS_OUTPUT'] = str(pred)
            proc = subprocess.run(['bash', str(script)], cwd=str(atlas_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=False)
        else:
            env = os.environ.copy()
            env['nnUNet_results'] = str(atlas_root)
            env.setdefault('nnUNet_raw', str(td / 'nnUNet_raw'))
            env.setdefault('nnUNet_preprocessed', str(td / 'nnUNet_preprocessed'))
            if args.device:
                env['CUDA_VISIBLE_DEVICES'] = '' if args.device.lower() == 'cpu' else args.device
            proc = subprocess.run(['nnUNetv2_predict','-d',str(args.dataset_id),'-i',str(imagesTs),'-o',str(pred),'-tr',args.trainer,'-p',args.plans,'-c',args.configuration,'-f',str(args.folds),'--continue_prediction'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=False)
        (output / 'atlasnet_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (output / 'atlasnet_stderr.log').write_text(proc.stderr or '', encoding='utf-8')
        if proc.returncode != 0:
            print(json.dumps({**summary, 'status':'failed', 'return_code':proc.returncode, 'stderr_tail':(proc.stderr or '')[-4000:]}, indent=2))
            return proc.returncode
        candidates = sorted(pred.rglob('*.nii.gz'))
        if not candidates:
            print(json.dumps({**summary, 'status':'failed', 'reason':'no combined ATLAS-Net prediction found', 'pred_dir':str(pred)}, indent=2))
            return 2
        combined = output / 'combined_labels.nii.gz'
        shutil.copy2(candidates[0], combined)
        split_cmd = ['python','scripts/split_combined_labelmap.py','--labelmap',str(combined),'--labels-json',str(label_map),'--output',str(seg_dir)]
        split = subprocess.run(split_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        (output / 'split_stdout.log').write_text(split.stdout or '', encoding='utf-8')
        (output / 'split_stderr.log').write_text(split.stderr or '', encoding='utf-8')
        status = 'success' if split.returncode == 0 and list(seg_dir.glob('*.nii.gz')) else 'failed'
        print(json.dumps({**summary, 'status':status, 'combined_label':str(combined), 'segmentation_output':str(seg_dir), 'num_masks':len(list(seg_dir.glob('*.nii.gz')))}, indent=2))
        return 0 if status == 'success' else 3

if __name__ == '__main__':
    raise SystemExit(main())
