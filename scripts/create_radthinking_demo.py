from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import nibabel as nib


def save_nii(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))


def sphere(shape=(32,32,32), center=(16,16,16), radius=5):
    zz, yy, xx = np.indices(shape)
    return ((zz-center[0])**2 + (yy-center[1])**2 + (xx-center[2])**2 <= radius**2).astype(np.uint8)


def main():
    root = Path("data/radthinking_demo/patient_001").resolve()
    if root.exists():
        import shutil; shutil.rmtree(root)
    scans = [
        ("2013-07", "Findings: Stable postoperative liver change. Impression: No definite recurrent lesion. Recommendation: routine follow-up.", 4),
        ("2013-10", "Findings: New suspicious hypervascular liver lesion near resection margin. Impression: recurrence cannot be excluded. Recommendation: follow-up imaging.", 6),
        ("2014-08", "Findings: Prior suspicious area resolved. Impression: benign perfusion alteration favored. Recommendation: continue surveillance.", 3),
    ]
    rng = np.random.default_rng(1)
    for scan_id, report, r in scans:
        scan_dir = root / "scans" / scan_id
        ct = rng.normal(40, 20, size=(32,32,32)).astype(np.float32)
        save_nii(scan_dir / "ct.nii.gz", ct)
        (scan_dir / "report.txt").write_text(report, encoding="utf-8")
        (scan_dir / "clinical.json").write_text(json.dumps({"age": 67, "sex": "male", "contrast_phase": "portal venous", "history": "synthetic HCC follow-up demo"}, indent=2), encoding="utf-8")
        mask = sphere(radius=r)
        save_nii(root / "demo_masks" / scan_id / "liver.nii.gz", mask)
    (root / "metadata.json").write_text(json.dumps({"patient_id": "patient_001", "sex": "male", "age": 67, "note": "Synthetic RadThinking-style demo only; not clinical data."}, indent=2), encoding="utf-8")
    (root / "pathology.json").write_text(json.dumps({"diagnosis": "demo_HCC_placeholder", "source": "synthetic_demo", "warning": "Not a real diagnosis. Replace with authorized pathology/follow-up data."}, indent=2), encoding="utf-8")
    print(f"Created RadThinking-style demo patient at: {root}")

if __name__ == "__main__": main()
