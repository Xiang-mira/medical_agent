#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description="UNEST renal-structure wrapper for medai CLI")
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--unest-root', default='third_party/UNEST_renalStructures_lightweight')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    seg_dir = output / 'segmentations'
    unest_root = Path(args.unest_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    summary = {'wrapper':'unest', 'image':str(image), 'output':str(output), 'unest_root':str(unest_root), 'script':str(unest_root / 'run_UNEST.sh')}
    if args.dry_run:
        summary['status'] = 'dry_run'
        print(json.dumps(summary, indent=2))
        return 0
    if not image.exists():
        raise FileNotFoundError(image)
    script = unest_root / 'run_UNEST.sh'
    if not script.exists():
        raise FileNotFoundError(script)
    with tempfile.TemporaryDirectory(prefix='medai_unest_') as td0:
        td = Path(td0)
        inp = td / 'input'
        outp = td / 'output'
        inp.mkdir()
        outp.mkdir()
        shutil.copy2(image, inp / 'ct.nii.gz')
        proc = subprocess.run(['bash', str(script), str(inp), str(outp)], cwd=str(unest_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        (output / 'unest_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (output / 'unest_stderr.log').write_text(proc.stderr or '', encoding='utf-8')
        if proc.returncode != 0:
            print(json.dumps({**summary, 'status':'failed', 'return_code':proc.returncode, 'stderr_tail':(proc.stderr or '')[-4000:]}, indent=2))
            return proc.returncode
        for p in outp.rglob('*.nii.gz'):
            shutil.copy2(p, seg_dir / p.name)
        masks = list(seg_dir.glob('*.nii.gz'))
        print(json.dumps({**summary, 'status':'success' if masks else 'failed', 'num_masks':len(masks), 'segmentation_output':str(seg_dir)}, indent=2))
        return 0 if masks else 2

if __name__ == '__main__':
    raise SystemExit(main())
