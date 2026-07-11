from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _trainer_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_voxtell_sampling_audit_test",
        ROOT / "scripts" / "train_voxtell_prompt_student.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(*organs: str) -> list[dict]:
    return [
        {"organ": organ, "supervision_type": "positive"}
        for organ in organs
    ]


def test_nonfinite_gradient_does_not_zero_all_organ_shares() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver", "spleen"),
        loss_history=[
            {"step": 1, "positive_organs": ["liver", "spleen"], "grad_norm_before_clip": float("nan")},
            {"step": 2, "positive_organs": ["liver", "spleen"], "grad_norm_before_clip": 1.0},
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    organ_audit = audit["organ_gradient_audit"]
    assert organ_audit["finite_gradient_mass_total"] > 0
    assert math.isclose(organ_audit["gradient_shares"]["liver"], 0.5)
    assert math.isclose(organ_audit["gradient_shares"]["spleen"], 0.5)
    assert "gradient_mass_nonfinite_or_insufficient" not in organ_audit["failure_reasons"]


def test_never_sampled_positive_organ_fails_exposure_not_gradient_share() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver", "spleen"),
        loss_history=[
            {"step": 1, "positive_organs": ["liver"], "grad_norm_before_clip": 1.0},
            {"step": 2, "positive_organs": ["liver"], "grad_norm_before_clip": 1.0},
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    exposure = audit["organ_exposure_audit"]
    assert exposure["status"] == "failed"
    assert "spleen" in exposure["underexposed_organs"]
    rows = {row["organ"]: row for row in audit["organ_rows"]}
    assert "organ_never_sampled" in rows["spleen"]["issues"]


def test_negative_organs_do_not_satisfy_positive_coverage() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver"),
        loss_history=[
            {"step": 1, "sample_kind": "negative", "negative_organ": "liver", "grad_norm_before_clip": 1.0},
            {"step": 2, "sample_kind": "negative", "negative_organ": "liver", "grad_norm_before_clip": 1.0},
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    assert audit["organ_exposure_audit"]["status"] == "failed"
    assert audit["organ_exposure_audit"]["negative_organ_sample_counts"]["liver"] == 2
    assert audit["organ_exposure_audit"]["positive_organ_sample_counts"].get("liver", 0) == 0


def test_zero_finite_gradient_mass_has_specific_failure_reason() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver"),
        loss_history=[
            {"step": 1, "positive_organs": ["liver"], "grad_norm_before_clip": float("nan")},
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    reasons = audit["organ_gradient_audit"]["failure_reasons"]
    assert "gradient_mass_nonfinite_or_insufficient" in reasons
    assert audit["organ_gradient_audit"]["gradient_shares"]["liver"] is None


def test_nonfinite_gradient_audit_reports_skipped_optimizer_steps() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver"),
        loss_history=[
            {
                "step": 1,
                "positive_organs": ["liver"],
                "grad_norm_before_clip": float("nan"),
                "optimizer_step_skipped_nonfinite_grad": True,
            },
            {"step": 2, "positive_organs": ["liver"], "grad_norm_before_clip": 1.0},
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    nonfinite = audit["nonfinite_gradient_audit"]
    assert nonfinite["optimizer_steps_skipped_nonfinite_grad"] == 1


def test_legacy_all_zero_gradient_share_bug_is_diagnosed() -> None:
    trainer = _trainer_module()
    audit = trainer.build_sampling_gradient_audits(
        training_rows=_rows("liver", "spleen"),
        loss_history=[
            {"step": 1, "grad_norm_before_clip": float("nan")},
            {"step": 2, "grad_norm_before_clip": 1.0},
        ],
        legacy_organ_sample_counts={"liver": 10, "spleen": 10},
        legacy_gradient_shares={"liver": 0.0, "spleen": 0.0},
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    diagnosis = audit["student_sampling_gradient_diagnosis"]
    assert diagnosis["all_gradient_shares_zero_due_to_nan_total"] is True
    assert diagnosis["paper_aligned_metadata_missing_organ_in_loss_history"] is True


def test_paper_sampler_contract_fails_unsampleable_positive_organs() -> None:
    trainer = _trainer_module()
    training_rows = [
        {"case_id": "case_a", "organ": "liver", "supervision_type": "positive"},
        {"case_id": "case_a", "organ": "spleen", "supervision_type": "positive"},
        {"case_id": "case_a", "organ": "brain", "supervision_type": "negative"},
        {"case_id": "case_b", "organ": "aorta", "supervision_type": "positive"},
    ]
    paper_case_pools = {
        "case_a": {
            "case_id": "case_a",
            "image": "case_a.nii.gz",
            "positive": training_rows[:2],
            "negative": [training_rows[2]],
        }
    }

    audit = trainer.build_paper_sampler_contract_audit(training_rows, paper_case_pools)

    assert audit["status"] == "failed"
    assert "trainable_positive_organs_not_sampleable_by_paper_profile" in audit["failure_reasons"]
    assert audit["unsampleable_positive_organs"] == ["aorta"]


def test_paper_sampler_contract_passes_when_all_positive_organs_sampleable() -> None:
    trainer = _trainer_module()
    training_rows = [
        {"case_id": "case_a", "organ": "liver", "supervision_type": "positive"},
        {"case_id": "case_a", "organ": "spleen", "supervision_type": "positive"},
        {"case_id": "case_a", "organ": "brain", "supervision_type": "negative"},
    ]
    paper_case_pools = {
        "case_a": {
            "case_id": "case_a",
            "image": "case_a.nii.gz",
            "positive": training_rows[:2],
            "negative": [training_rows[2]],
        }
    }

    audit = trainer.build_paper_sampler_contract_audit(training_rows, paper_case_pools)

    assert audit["status"] == "passed"
    assert audit["unsampleable_positive_organs"] == []


def test_negative_loss_with_finite_components_passes_stability() -> None:
    trainer = _trainer_module()
    diagnosis, rows = trainer.build_training_stability_diagnosis(
        [
            {
                "step": 1,
                "loss": -0.2,
                "bce_component": 0.3,
                "dice_component": -0.5,
                "grad_norm_before_clip": 1.0,
                "loss_component_finite": True,
                "optimizer_step_performed": True,
            }
        ]
    )

    assert diagnosis["status"] == "passed"
    assert diagnosis["loss_component_finite_rate"] == 1.0
    assert rows == []


def test_nonfinite_loss_component_fails_stability() -> None:
    trainer = _trainer_module()
    diagnosis, rows = trainer.build_training_stability_diagnosis(
        [
            {
                "step": 1,
                "loss": float("nan"),
                "bce_component": float("nan"),
                "dice_component": -0.5,
                "grad_norm_before_clip": float("nan"),
                "loss_component_finite": False,
                "optimizer_step_skipped_nonfinite_grad": True,
            }
        ],
        nonfinite_gradient_step_rate_threshold=1.0,
    )

    assert diagnosis["status"] == "failed"
    assert "loss_component_nonfinite" in diagnosis["failure_reasons"]
    assert rows and "loss_component_nonfinite" in rows[0]["reasons"]


def test_alignment_audit_catches_positive_empty_and_negative_nonzero() -> None:
    trainer = _trainer_module()
    image_meta = {
        "shape": [4, 4, 4],
        "spacing": [1.0, 1.0, 1.0],
        "orientation": "RAS",
        "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    empty_mask = dict(image_meta, foreground_voxels=0)
    nonzero_mask = dict(image_meta, foreground_voxels=3)

    positive_reasons = trainer._alignment_fail_reasons(
        image_meta,
        empty_mask,
        {"organ": "liver", "prompt": "liver", "supervision_type": "positive", "target_type": "positive_hard"},
    )
    negative_reasons = trainer._alignment_fail_reasons(
        image_meta,
        nonzero_mask,
        {
            "organ": "brain",
            "prompt": "brain",
            "supervision_type": "negative",
            "target_type": "negative_absent",
            "negative_source": "case_373_expected_absent",
            "zero_mask_role": "negative_absent_target_mask",
        },
    )

    assert "positive_mask_empty" in positive_reasons
    assert "negative_absent_mask_nonzero" in negative_reasons


def test_alignment_audit_catches_shape_mismatch() -> None:
    trainer = _trainer_module()
    reasons = trainer._alignment_fail_reasons(
        {"shape": [4, 4, 4], "spacing": [1, 1, 1], "orientation": "RAS"},
        {"shape": [4, 4, 5], "spacing": [1, 1, 1], "orientation": "RAS", "foreground_voxels": 1},
        {"organ": "liver", "prompt": "liver", "supervision_type": "positive", "target_type": "positive_hard"},
    )

    assert "ct_mask_shape_mismatch" in reasons
