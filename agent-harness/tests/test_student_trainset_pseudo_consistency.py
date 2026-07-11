from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_student_trainset_pseudo_consistency_test",
        ROOT / "scripts" / "run_student_trainset_pseudo_consistency.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_binary_metrics_perfect_empty_and_oversegmentation() -> None:
    mod = _module()
    empty = np.zeros((3, 3, 3), dtype=bool)
    assert mod.binary_metrics(empty, empty)["pseudo_consistency_dsc"] == 1.0

    ref = np.zeros((4, 4, 4), dtype=bool)
    ref[:2, :2, :2] = True
    pred = ref.copy()
    perfect = mod.binary_metrics(pred, ref)
    assert perfect["pseudo_consistency_dsc"] == 1.0
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["volume_ratio"] == 1.0

    over = np.ones((4, 4, 4), dtype=bool)
    metrics = mod.binary_metrics(over, ref)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] < 0.5
    assert metrics["volume_ratio"] > 1.5


def test_binary_metrics_shape_mismatch_reports_no_scores() -> None:
    mod = _module()
    metrics = mod.binary_metrics(
        np.zeros((2, 2, 2), dtype=bool),
        np.zeros((3, 2, 2), dtype=bool),
    )
    assert metrics["status"] == "shape_mismatch"
    assert metrics["pseudo_consistency_dsc"] is None


def test_round2_gate_blocks_fallback_and_low_consistency() -> None:
    mod = _module()
    gate = mod.build_round2_gate(
        preflight={"status": "passed"},
        consistency_summary={
            "mean_positive_pseudo_consistency_dsc": 0.4,
            "oversegmentation_positive_rate": 0.0,
            "key_organ_median_volume_ratio": 1.0,
            "negative_false_positive_count": 0,
        },
        overseg_audit={"status": "passed"},
        shapekit_summary={"status": "partial_success"},
        postprocess_summary={"status": "success"},
        args=SimpleNamespace(
            positive_min_mean_dsc=0.6,
            max_oversegmentation_rate=0.15,
            max_key_organ_median_volume_ratio=1.5,
            negative_false_positive_voxel_threshold=0,
        ),
    )
    assert gate["status"] == "blocked"
    assert "positive_pseudo_consistency_below_threshold" in gate["block_reasons"]
    assert "student_shapekit_not_fully_successful" in gate["block_reasons"]


def test_round2_gate_blocks_negative_false_positive() -> None:
    mod = _module()
    gate = mod.build_round2_gate(
        preflight={"status": "passed"},
        consistency_summary={
            "mean_positive_pseudo_consistency_dsc": 0.8,
            "oversegmentation_positive_rate": 0.0,
            "key_organ_median_volume_ratio": 1.0,
            "negative_false_positive_count": 1,
        },
        overseg_audit={"status": "passed"},
        shapekit_summary={"status": "success"},
        postprocess_summary={"status": "success"},
        args=SimpleNamespace(
            positive_min_mean_dsc=0.6,
            max_oversegmentation_rate=0.15,
            max_key_organ_median_volume_ratio=1.5,
            negative_false_positive_voxel_threshold=0,
        ),
    )
    assert gate["status"] == "blocked"
    assert "negative_absent_false_positive_detected" in gate["block_reasons"]


def _write_mask(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), str(path))


def test_negative_absent_metrics_include_provenance_and_never_replacement_eligible(tmp_path: Path) -> None:
    mod = _module()
    case_id = "case_001"
    organ = "brain"
    post_root = tmp_path / "post"
    selected_root = tmp_path / "selected"
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    pred[0, 0, 0] = 1
    ref = np.zeros((3, 3, 3), dtype=np.uint8)
    _write_mask(post_root / case_id / f"{organ}.nii.gz", pred)
    _write_mask(selected_root / case_id / "updated" / f"{organ}.nii.gz", ref)
    manifest = {
        (case_id, organ): {
            "case_id": case_id,
            "organ": organ,
            "target_type": "negative_absent",
            "supervision_type": "negative",
            "training_weight": 0.1,
            "fov_status": "out_of_fov",
            "fov_evidence": ["dataset_prior:abdomen_only"],
            "negative_source": "case_373_expected_absent",
            "zero_mask_role": "negative_absent_target_mask",
            "negative_reason": "out_of_scan_by_scan_coverage",
            "absence_confidence": "high",
        }
    }
    rows, summary, _, extra = mod.compute_metrics(
        cases=[{"case_id": case_id, "ct_path": "ct.nii.gz"}],
        targets=[organ],
        manifest=manifest,
        selected_root=selected_root,
        post_root=post_root,
        shapekit_rows=[{"case_id": case_id, "organ": organ, "student_shapekit_status": "success"}],
        suppression_rows=[],
        negative_fp_threshold=0,
    )
    row = rows[0]
    assert row["negative_false_positive"] is True
    assert row["replacement_eligible"] is False
    assert row["fov_status"] == "out_of_fov"
    assert row["negative_source"] == "case_373_expected_absent"
    assert summary["negative_false_positive_count"] == 1
    assert len(extra["negative_false_positive_rows"]) == 1


def test_negative_safe_postprocess_forces_only_confirmed_negative_empty(tmp_path: Path) -> None:
    mod = _module()
    case_id = "case_001"
    post_root = tmp_path / "post"
    targets = ["brain", "liver"]
    nonempty = np.ones((2, 2, 2), dtype=np.uint8)
    _write_mask(post_root / case_id / "brain.nii.gz", nonempty)
    _write_mask(post_root / case_id / "liver.nii.gz", nonempty)
    manifest = {
        (case_id, "brain"): {
            "case_id": case_id,
            "organ": "brain",
            "target_type": "negative_absent",
            "supervision_type": "negative",
            "fov_status": "out_of_fov",
            "negative_source": "case_373_expected_absent",
            "zero_mask_role": "negative_absent_target_mask",
        },
        (case_id, "liver"): {
            "case_id": case_id,
            "organ": "liver",
            "target_type": "positive_hard",
            "supervision_type": "positive",
        },
    }
    rows, summary = mod.apply_negative_safe_postprocess(
        manifest=manifest,
        post_root=post_root,
        cases=[{"case_id": case_id, "ct_path": "ct.nii.gz"}],
        targets=targets,
        threshold=0,
        enabled=True,
    )
    assert summary["suppressed_negative_candidate_count"] == 1
    assert rows[0]["suppression_reason"] == "confirmed_negative_absent_out_of_fov"
    assert mod.mask_volume(post_root / case_id / "brain.nii.gz") == 0
    assert mod.mask_volume(post_root / case_id / "liver.nii.gz") == 8
