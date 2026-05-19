"""
Reference-label inference stub for real PanTS cases.

Instead of running a real AI model, this copies the reference labels
(ground-truth masks) into the output folder. This lets the full pipeline
(ShapeKit, QC, RadThinking trace, agent-loop) be tested on real PanTS data
without requiring a trained model checkpoint.

Usage:
    python reference_label_infer.py --image <ct.nii.gz> --output <seg_output_dir>

The script locates reference labels by walking up from the CT image path to
find the patient folder, then looking for reference_labels/.../segmentations/.
"""
import argparse
import shutil
import sys
from pathlib import Path


def find_reference_labels(image_path: Path) -> Path | None:
    # Walk up from ct.nii.gz to find the patient root
    # Expected: .../patient_id/scans/scan_id/ct.nii.gz
    candidate = image_path.parent  # scan_id dir
    for _ in range(4):
        candidate = candidate.parent
        ref_root = candidate / "reference_labels"
        if ref_root.exists():
            # Find first segmentations folder inside reference_labels
            seg_dirs = list(ref_root.glob("*/segmentations"))
            if seg_dirs:
                return seg_dirs[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image = Path(args.image).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"[reference_label_infer] image={image}")
    print(f"[reference_label_infer] output={output}")

    ref_seg = find_reference_labels(image)
    if ref_seg is None or not ref_seg.exists():
        print(f"[reference_label_infer] ERROR: could not find reference_labels near {image}", file=sys.stderr)
        sys.exit(2)

    masks = list(ref_seg.glob("*.nii.gz"))
    if not masks:
        print(f"[reference_label_infer] ERROR: no masks found in {ref_seg}", file=sys.stderr)
        sys.exit(2)

    for m in masks:
        shutil.copy2(m, output / m.name)

    print(f"[reference_label_infer] copied {len(masks)} masks from {ref_seg}")


if __name__ == "__main__":
    main()
