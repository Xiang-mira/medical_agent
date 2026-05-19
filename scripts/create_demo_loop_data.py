from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import nibabel as nib


def box(shape, start, end):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 1
    return arr


def save_mask(path: Path, arr, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), str(path))


def main():
    root = Path("data/demo_loop").resolve()
    mask_root = Path("data/demo_masks").resolve()
    root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    organs = {
        "pancreas": ((10, 12, 13), (22, 18, 18)),
        "liver": ((5, 5, 12), (22, 17, 25)),
        "spleen": ((21, 18, 12), (29, 27, 21)),
        "aorta": ((15, 15, 3), (18, 18, 29)),
        "postcava": ((19, 15, 3), (22, 18, 29)),
    }
    rows = []
    affine = np.eye(4)
    for i in range(1, 3):
        case_id = f"demo_case_{i:03d}"
        case_dir = root / case_id
        ref_dir = case_dir / "reference" / "segmentations"
        pred_dir = mask_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        ct = rng.normal(40, 12, size=(32, 32, 32)).astype(np.float32)
        nib.save(nib.Nifti1Image(ct, affine), str(case_dir / "ct.nii.gz"))
        for organ, (st, en) in organs.items():
            ref = box((32, 32, 32), st, en)
            shift = i  # creates a deterministic but imperfect prediction
            pred = np.roll(ref, shift=shift, axis=0)
            save_mask(ref_dir / f"{organ}.nii.gz", ref, affine)
            save_mask(pred_dir / f"{organ}.nii.gz", pred, affine)
        (case_dir / "report.txt").write_text("Findings: abdominal CT with pancreas and liver visible. Impression: demo case for workflow validation.\n", encoding="utf-8")
        rows.append({"case_id": case_id, "ct_path": str(case_dir / "ct.nii.gz"), "annotation_folder": str(ref_dir), "report_path": str(case_dir / "report.txt")})
    csv_path = Path("data_manifest/demo_case_list.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "ct_path", "annotation_folder", "report_path"])
        w.writeheader(); w.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
