#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description="VISTA3D wrapper for medai CLI")
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--vista-root', default='third_party/VISTA3D-Inference-Pipeline-master')
    ap.add_argument('--num-gpus', default='1')
    ap.add_argument('--label-map', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / 'segmentations'
    vista_root = Path(args.vista_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    label_map = Path(args.label_map).resolve() if args.label_map else vista_root / 'label_mappings' / 'label_dict_127_abdomenAtlas3-1.json'
    summary = {'wrapper':'vista3d', 'image':str(image), 'output':str(output), 'vista_root':str(vista_root), 'label_map':str(label_map), 'num_gpus':args.num_gpus}
    if args.dry_run:
        summary['status'] = 'dry_run'
        print(json.dumps(summary, indent=2))
        return 0
    if not image.exists():
        raise FileNotFoundError(image)
    if not vista_root.exists():
        raise FileNotFoundError(vista_root)
    if not label_map.exists():
        raise FileNotFoundError(label_map)
    with tempfile.TemporaryDirectory(prefix='medai_vista3d_') as td0:
        td = Path(td0)
        input_dir = td / 'input'
        eval_dir = td / 'eval'
        input_dir.mkdir()
        eval_dir.mkdir()
        shutil.copy2(image, input_dir / 'ct.nii.gz')
        run_script = td / 'run_medai_vista.sh'
        script_lines = [
            '#!/bin/bash',
            'set -e',
            f'cd "{vista_root}"',
            f'input_dir="{input_dir}"',
            f'num_gpus="{args.num_gpus}"',
            'input_suffix="ct.nii.gz"',
            'input_list=\'$labels2onehot.build_input_list(@input_dir,""@input_suffix,""@output_dir)\'',
            f'export VISTA3D_OUTPUT_DIR="{eval_dir}"',
            'torchrun --nnodes=1 --nproc_per_node=$num_gpus -m monai.bundle run \\',
            '  --config_file="[\'configs/inference.json\', \'configs/batch_inference.json\']" \\',
            '  --input_dir=$input_dir \\',
            '  --input_suffix=$input_suffix \\',
            '  --input_list=$input_list \\',
            '  --output_dir=$VISTA3D_OUTPUT_DIR \\',
            '  --output_postfix="step1_117"',
        ]
        run_script.write_text("\n".join(script_lines) + "\n", encoding='utf-8')
        run_script.chmod(0o755)
        proc = subprocess.run(['bash', str(run_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        (output / 'vista3d_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (output / 'vista3d_stderr.log').write_text(proc.stderr or '', encoding='utf-8')
        if proc.returncode != 0:
            print(json.dumps({**summary, 'status':'failed', 'return_code':proc.returncode, 'stderr_tail':(proc.stderr or '')[-4000:]}, indent=2))
            return proc.returncode
        candidates = sorted(eval_dir.rglob('*.nii.gz'))
        if not candidates:
            print(json.dumps({**summary, 'status':'failed', 'reason':'VISTA3D produced no nii.gz predictions', 'eval_dir':str(eval_dir)}, indent=2))
            return 2
        if len(candidates) > 10:
            for p in candidates:
                shutil.copy2(p, seg_dir / p.name)
        else:
            combined = output / 'combined_labels.nii.gz'
            shutil.copy2(candidates[0], combined)
            split_cmd = ['python','scripts/split_combined_labelmap.py','--labelmap',str(combined),'--labels-json',str(label_map),'--output',str(seg_dir)]
            split = subprocess.run(split_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            (output / 'split_stdout.log').write_text(split.stdout or '', encoding='utf-8')
            (output / 'split_stderr.log').write_text(split.stderr or '', encoding='utf-8')
        masks = list(seg_dir.glob('*.nii.gz'))
        status = 'success' if masks else 'failed'
        print(json.dumps({**summary, 'status':status, 'segmentation_output':str(seg_dir), 'num_masks':len(masks)}, indent=2))
        return 0 if status == 'success' else 3

if __name__ == '__main__':
    raise SystemExit(main())
