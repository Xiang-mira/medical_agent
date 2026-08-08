from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_HARNESS = REPO_ROOT / "agent-harness"
if str(AGENT_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENT_HARNESS))


def _save(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def test_production_fov_does_not_treat_thorax_landmarks_as_head_or_airway():
    from cli_anything.medai.core.multimodel_loop import _fov_status_for_organ

    context = {
        "has_region_evidence": True,
        "has_abdomen_coverage": False,
        "has_pelvis_coverage": False,
        "has_thorax_coverage": True,
        "has_partial_thorax_coverage": False,
        "has_head_coverage": False,
        "has_extremity_coverage": False,
        "target_fov_landmark_summary": {
            "head_evidence": False,
            "thorax_evidence": True,
            "partial_thorax_evidence": True,
            "central_airway_evidence": False,
        },
    }

    assert _fov_status_for_organ("brain_ventricle", context) == "out_of_fov"
    assert _fov_status_for_organ("gray_matter", context) == "out_of_fov"
    assert _fov_status_for_organ("airway_tree", context) == "out_of_fov"
    assert _fov_status_for_organ("airway_wall", context) == "out_of_fov"
    assert _fov_status_for_organ("lung_pulmonary_arteries", context) == "fully_visible"
    assert _fov_status_for_organ("lung_pulmonary_veins", context) == "fully_visible"


def test_reference_landmarks_make_append_thorax_case_pulmonary_only(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import (
        _augment_presence_from_reference_landmarks,
        _fov_status_for_organ,
    )

    shape = (40, 40, 40)
    ct = _save(np.zeros(shape, dtype=np.int16), tmp_path / "ct.nii.gz")
    ref = tmp_path / "ref"
    large = np.ones(shape, dtype=np.uint8)
    heart = np.zeros(shape, dtype=np.uint8)
    heart[:25, :25, :25] = 1
    aorta = np.zeros(shape, dtype=np.uint8)
    aorta[:11, :11, :11] = 1
    weak_brain = np.zeros(shape, dtype=np.uint8)
    weak_brain[0:5, 0:5, 0:5] = 1
    skull = np.zeros(shape, dtype=np.uint8)
    skull[0:15, 0:15, 0:15] = 1
    _save(large, ref / "lung_left.nii.gz")
    _save(large, ref / "lung_right.nii.gz")
    _save(heart, ref / "heart.nii.gz")
    _save(aorta, ref / "aorta.nii.gz")
    _save(weak_brain, ref / "brain.nii.gz")
    _save(skull, ref / "skull.nii.gz")

    context = _augment_presence_from_reference_landmarks(
        {
            "has_region_evidence": False,
            "coverage_evidence": [],
            "confirmed_absent_organs": [],
        },
        ref,
        ct=ct,
        case_out=tmp_path / "case_out",
    )

    summary = context["target_fov_landmark_summary"]
    assert summary["thorax_evidence"] is True
    assert summary["head_evidence"] is False
    assert summary["central_airway_evidence"] is False
    assert _fov_status_for_organ("brain_ventricle", context) == "out_of_fov"
    assert _fov_status_for_organ("airway_wall", context) == "out_of_fov"
    assert _fov_status_for_organ("lung_pulmonary_arteries", context) == "fully_visible"


def test_brain_ventricle_fov_requires_brain_skull_not_eyeballs(tmp_path: Path):
    from cli_anything.medai.core.multimodel_loop import (
        _augment_presence_from_reference_landmarks,
        _fov_status_for_organ,
    )

    shape = (40, 40, 40)
    ct = _save(np.zeros(shape, dtype=np.int16), tmp_path / "ct.nii.gz")
    ref = tmp_path / "ref"
    brain = np.zeros(shape, dtype=np.uint8)
    brain.flat[:12000] = 1
    skull = np.zeros(shape, dtype=np.uint8)
    skull.flat[:1200] = 1
    _save(brain, ref / "brain.nii.gz")
    _save(skull, ref / "skull.nii.gz")

    context = _augment_presence_from_reference_landmarks(
        {
            "has_region_evidence": False,
            "coverage_evidence": [],
            "confirmed_absent_organs": [],
        },
        ref,
        ct=ct,
        case_out=tmp_path / "case_out",
    )

    assert _fov_status_for_organ("brain_ventricle", context) == "fully_visible"
    assert _fov_status_for_organ("gray_matter", context) == "fully_visible"
    assert _fov_status_for_organ("face", context) == "out_of_fov"
    assert _fov_status_for_organ("eyeball", context) == "out_of_fov"
