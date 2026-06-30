from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_student_masks_can_live_directly_in_case_directory(tmp_path):
    protocol = load_script("run_blind10_protocol.py")
    case = tmp_path / "student_only" / "case_001"
    case.mkdir(parents=True)
    (case / "liver.nii.gz").touch()

    assert protocol.find_student_segmentations(tmp_path / "student_only", "case_001") == case


def test_evaluator_accepts_direct_case_directory(tmp_path):
    evaluator = load_script("evaluate_blind_segmentation.py")
    case = tmp_path / "student_only" / "case_001"
    case.mkdir(parents=True)
    (case / "liver.nii.gz").touch()

    assert evaluator.prediction_dir(tmp_path / "student_only", "case_001") == case
