from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Mock medical segmentation model for medai-cli demos.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", default=None)
    args = parser.parse_args()
    image = Path(args.image).resolve(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    scan_id = args.case_id or image.parent.name
    patient_root = image.parents[2] if len(image.parents) >= 3 else image.parent
    demo_mask_dir = patient_root / "demo_masks" / scan_id
    if not demo_mask_dir.exists():
        print(f"[mock_seg_infer] ERROR: demo mask folder not found: {demo_mask_dir}", file=sys.stderr); sys.exit(2)
    copied = 0
    for src in sorted(demo_mask_dir.glob("*.nii.gz")):
        shutil.copy2(src, out / src.name); copied += 1
    print(f"[mock_seg_infer] image={image}")
    print(f"[mock_seg_infer] output={out}")
    print(f"[mock_seg_infer] copied {copied} masks from {demo_mask_dir}")

if __name__ == "__main__": main()
