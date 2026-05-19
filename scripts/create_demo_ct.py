from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

p = Path("data/single_case/case_001"); p.mkdir(parents=True, exist_ok=True)
arr = np.random.default_rng(0).normal(30, 10, size=(32,32,32)).astype(np.float32)
nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p / "ct.nii.gz"))
print(p / "ct.nii.gz")
