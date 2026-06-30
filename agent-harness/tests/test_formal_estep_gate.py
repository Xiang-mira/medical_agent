from __future__ import annotations

import importlib.util
from pathlib import Path


def load_training_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_em_training.py"
    spec = importlib.util.spec_from_file_location("run_em_training_for_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cohort_coverage_allows_rejected_minority_and_fov_not_applicable(monkeypatch):
    module = load_training_module()
    monkeypatch.setenv("MEDAI_FORMAL_KEY_ORGAN_MIN_COVERAGE", "0.80")
    rows = [
        {"case_id": f"case_{index}", "organ": "kidney_left", "expected_presence": "expected_present"}
        for index in range(10)
    ]
    rows += [
        {"case_id": f"case_{index}", "organ": "small_bowel", "expected_presence": "expected_present"}
        for index in range(10)
    ]
    selected = {}
    for index in range(9):
        selected[(f"case_{index}", "kidney_left")] = {"grade": "C", "final_mask": "/mask"}
    for index in range(8):
        selected[(f"case_{index}", "small_bowel")] = {"grade": "C", "final_mask": "/mask"}

    coverage, failures, not_applicable = module._cohort_coverage(
        ["kidney_left", "small_bowel", "bladder"], rows, selected
    )

    assert failures == []
    assert coverage["kidney_left"]["coverage_rate"] == 0.9
    assert coverage["small_bowel"]["coverage_rate"] == 0.8
    assert coverage["bladder"]["status"] == "not_applicable_no_expected_present_case"
    assert not_applicable == ["bladder"]


def test_cohort_coverage_blocks_systematic_missing_labels(monkeypatch):
    module = load_training_module()
    monkeypatch.setenv("MEDAI_FORMAL_KEY_ORGAN_MIN_COVERAGE", "0.80")
    rows = [
        {"case_id": f"case_{index}", "organ": "pancreas", "expected_presence": "expected_present"}
        for index in range(10)
    ]
    selected = {
        (f"case_{index}", "pancreas"): {"grade": "C", "final_mask": "/mask"}
        for index in range(7)
    }

    coverage, failures, _ = module._cohort_coverage(["pancreas"], rows, selected)

    assert coverage["pancreas"]["coverage_rate"] == 0.7
    assert failures == [{
        "organ": "pancreas",
        "usable": 7,
        "expected_present": 10,
        "coverage_rate": 0.7,
        "minimum_coverage_rate": 0.8,
    }]
