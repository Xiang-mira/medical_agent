from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


ORGANS = ["liver", "pancreas", "spleen", "kidney_left", "kidney_right", "aorta", "postcava"]


def _synthetic_box(shape: tuple[int, int, int], idx: int) -> tuple[slice, slice, slice]:
    """Return a deterministic non-empty box inside a 3D image shape."""
    spans = [max(2, min(int(s), max(2, int(s) // 6))) for s in shape]
    starts = []
    for axis, (size, span) in enumerate(zip(shape, spans)):
        room = max(1, int(size) - int(span))
        starts.append((idx * (axis + 2) * 7) % room)
    return tuple(slice(st, min(st + span, int(size))) for st, span, size in zip(starts, spans, shape))  # type: ignore[return-value]


def _write_synthetic_masks(image: Path, out: Path) -> int:
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(str(image))
        shape = tuple(int(x) for x in img.shape[:3])
        copied = 0
        for idx, organ in enumerate(ORGANS, start=1):
            arr = np.zeros(shape, dtype=np.uint8)
            arr[_synthetic_box(shape, idx)] = 1
            nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(out / f"{organ}.nii.gz"))
            copied += 1
        return copied
    except Exception as exc:
        print(f"[mock_seg_infer] ERROR: synthetic mask generation failed: {exc}", file=sys.stderr)
        return 0


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
        copied = _write_synthetic_masks(image, out)
        if copied <= 0:
            print(f"[mock_seg_infer] ERROR: demo mask folder not found and synthetic fallback failed: {demo_mask_dir}", file=sys.stderr); sys.exit(2)
        print(f"[mock_seg_infer] demo mask folder not found; wrote {copied} synthetic masks for smoke testing")
    else:
        copied = 0
        for src in sorted(demo_mask_dir.glob("*.nii.gz")):
            shutil.copy2(src, out / src.name); copied += 1
    print(f"[mock_seg_infer] image={image}")
    print(f"[mock_seg_infer] output={out}")
    print(f"[mock_seg_infer] produced {copied} masks")

if __name__ == "__main__": main()
