from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def save(path: Path, array: np.ndarray, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), str(path))
    return path


def test_scan_coverage_is_label_free(tmp_path):
    from cli_anything.medai.core.scan_coverage import infer_scan_coverage

    ct = np.zeros((32, 32, 160), dtype=np.int16)
    result = infer_scan_coverage(save(tmp_path / "ct.nii.gz", ct))
    assert result["uses_annotations"] is False
    assert result["source"] == "ct_intensity_geometry_v1"
    assert "abdomen" in result["coverage_regions"]


def test_affine_mismatch_is_hard_candidate_failure(tmp_path):
    from cli_anything.medai.core.multimodel_loop import _compute_candidate_qc

    ct = save(tmp_path / "ct.nii.gz", np.zeros((16, 16, 16), dtype=np.int16))
    affine = np.eye(4); affine[0, 3] = 100
    mask = save(tmp_path / "mask.nii.gz", np.ones((16, 16, 16), dtype=np.uint8), affine)
    result = _compute_candidate_qc(ct=ct, mask=mask, organ="aorta")
    assert result["status"] == "fail"
    assert result["eligible_for_labelcritic"] is False
    assert "affine_mismatch_ct" in result["flags"]


def test_pancreatic_duct_extreme_volume_is_hard_failure(tmp_path):
    from cli_anything.medai.core.multimodel_loop import _compute_candidate_qc

    ct = save(tmp_path / "ct.nii.gz", np.zeros((32, 32, 32), dtype=np.int16))
    pred = np.zeros((32, 32, 32), dtype=np.uint8); pred[5:15, 5:15, 5:15] = 1
    ref = np.zeros_like(pred); ref[8, 8, 8] = 1
    result = _compute_candidate_qc(
        ct=ct,
        mask=save(tmp_path / "segmentations/pancreatic_duct.nii.gz", pred),
        organ="pancreatic_duct",
        reference=save(tmp_path / "reference/pancreatic_duct.nii.gz", ref),
    )
    assert result["status"] == "fail"
    assert "pancreatic_duct_extreme_volume_ratio" in result["flags"]


def test_unknown_fov_is_not_distillation_eligible():
    from cli_anything.medai.core.multimodel_loop import _distillation_gate_for_selected_label

    result = _distillation_gate_for_selected_label({
        "grade": "A", "target_type": "hard", "training_weight": 1.0,
        "expected_presence": "unknown", "selected_candidate_qc_status": "pass",
    })
    assert result == {"eligible": False, "reason": "fov_not_confirmed_present:unknown"}


def test_postcava_collapses_to_inferior_vena_cava():
    from cli_anything.medai.core.target_space import canonical_target_name, validate_formal_373_target_space

    assert canonical_target_name("postcava") == "inferior_vena_cava"
    result = validate_formal_373_target_space("configs/student_3d_prompt_target_organs.json")
    assert result["status"] == "success"
    assert result["counts"]["target_organs"] == 373
    assert result["counts"]["effective_target_organs"] == 372


def test_blind_case_list_has_no_reference_column():
    path = Path("data_manifest/blind10_batch1.csv")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["case_id", "ct_path"]
    assert len(rows) == 10
    assert all("Label" not in row["ct_path"] for row in rows)
