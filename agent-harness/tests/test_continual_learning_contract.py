from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cli_anything.medai.core.continual_learning import (
    RunSpec,
    RoundStateMachine,
    canonical_target_type,
    canonicalize_training_record,
    novelty_audit,
    promotion_decision,
    retroactively_block_rounds,
    resolve_canonical_memory,
)


def _mask(path: Path, *, soft: bool = False, empty: bool = False) -> Path:
    array = np.zeros((4, 4, 4), dtype=np.float32)
    if not empty:
        array[1:3, 1:3, 1:3] = 0.6 if soft else 1.0
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return path


def _teacher_row(tmp_path: Path, case: str = "c1", organ: str = "liver") -> dict:
    return {
        "case_id": case,
        "organ": organ,
        "image": str(_mask(tmp_path / f"{case}_ct.nii.gz")),
        "mask": str(_mask(tmp_path / f"{case}_{organ}.nii.gz")),
        "target_type": "hard",
        "grade": "A",
        "training_weight": 1.0,
        "distillation_eligible": True,
        "selected_model": "teacher_a",
        "selected_candidate_qc_status": "pass",
        "identity_status": "valid",
    }


def test_target_type_aliases_are_canonical() -> None:
    assert canonical_target_type("hard") == "positive_hard"
    assert canonical_target_type("positive_soft") == "positive_soft"
    assert canonical_target_type("absent_negative") == "negative_absent"


def test_positive_soft_uses_probability_mask_and_rejects_fake_soft(tmp_path: Path) -> None:
    hard = _mask(tmp_path / "hard.nii.gz")
    probability = _mask(tmp_path / "probability.nii.gz", soft=True)
    row = _teacher_row(tmp_path)
    row.update(
        {
            "mask": str(hard),
            "target_type": "positive_soft",
            "grade": "C",
            "probability_mask_path": str(probability),
        }
    )
    record = canonicalize_training_record(row)
    assert record["training_eligible"] is True
    assert record["mask"] == str(probability)
    assert record["probability_audit"]["has_intermediate_probability"] is True

    row["probability_mask_path"] = str(hard)
    rejected = canonicalize_training_record(row)
    assert rejected["training_eligible"] is False
    assert "soft_target_has_no_intermediate_probabilities" in rejected["contract_failures"]


def test_student_gt_unknown_and_empty_positive_are_blocked(tmp_path: Path) -> None:
    for provider in ("student_prev", "round_prev_selected"):
        row = _teacher_row(tmp_path, case=provider)
        row["selected_model"] = provider
        assert canonicalize_training_record(row)["training_eligible"] is False

    gt = _teacher_row(tmp_path, case="gt")
    gt["dataset_role"] = "ground_truth"
    assert canonicalize_training_record(gt)["training_eligible"] is False

    empty = _teacher_row(tmp_path, case="empty")
    empty["mask"] = str(_mask(tmp_path / "empty.nii.gz", empty=True))
    assert canonicalize_training_record(empty)["training_eligible"] is False


def test_memory_prefers_stronger_stable_teacher(tmp_path: Path) -> None:
    historical = canonicalize_training_record(_teacher_row(tmp_path))
    historical["historical_replay"] = True
    current_row = _teacher_row(tmp_path)
    current_row["mask"] = str(_mask(tmp_path / "current.nii.gz"))
    current_row["grade"] = "B"
    current = canonicalize_training_record(current_row)
    winners, rejected = resolve_canonical_memory([current, historical])
    assert len(winners) == 1
    assert winners[0]["grade"] == "A"
    assert rejected[0]["memory_rejection"] == "lower_quality_than_winner"


def test_novelty_is_cohort_size_independent(tmp_path: Path) -> None:
    for count in (3, 10, 50):
        previous = [
            canonicalize_training_record(_teacher_row(tmp_path, case=f"c{count}_{i}"))
            for i in range(count)
        ]
        current = [dict(row) for row in previous]
        audit = novelty_audit(current, previous)
        assert audit["decision"] == "no_material_update"
        assert audit["max_steps"] == 0


def test_promotion_is_fail_closed_and_supports_no_material_update() -> None:
    regression = {
        "mean_delta": 0.0,
        "paired_median_delta": 0.0,
        "worst_protected_delta": 0.0,
        "pseudo_delta": 0.0,
    }
    passed = promotion_decision(
        requested_status="promoted",
        hard_checks={"contract": True},
        regression=regression,
    )
    assert passed["status"] == "promoted"

    blocked = promotion_decision(
        requested_status="promoted",
        hard_checks={"contract": True},
        regression={**regression, "mean_delta": -0.003},
    )
    assert blocked["status"] == "competition_blocked"

    unchanged = promotion_decision(
        requested_status="promoted",
        hard_checks={"contract": True},
        regression=regression,
        novelty={"decision": "no_material_update"},
    )
    assert unchanged["status"] == "no_material_update"


def test_run_spec_has_no_fixed_cohort_or_target_count() -> None:
    for count in (3, 10, 50):
        spec = RunSpec(
            run_id=f"r{count}",
            round_index=2,
            backend="test",
            case_list_path="/cases.csv",
            case_list_sha256="a",
            case_count=count,
            target_space_path="/targets.json",
            target_space_sha256="b",
            baseline_checkpoint_sha256=None,
            previous_checkpoint_sha256=None,
            evaluation_protocol_sha256="c",
            protected_organs=("liver",),
        )
        assert spec.to_dict()["case_count"] == count


def test_retrospective_block_preserves_history_and_is_idempotent() -> None:
    registry = {
        "rounds": {
            "1": {"round": 1, "status": "promoted"},
            "2": {
                "round": 2,
                "status": "promoted",
                "checkpoint_sha256": "bad",
                "eligible_for_next_round_prompt_student": True,
            },
        },
        "history": [],
    }
    updated, changed = retroactively_block_rounds(
        registry,
        [2],
        evidence={"mean_delta": -0.01},
    )
    assert changed == [2]
    assert updated["rounds"]["1"]["status"] == "promoted"
    assert updated["rounds"]["2"]["status"] == "competition_blocked"
    assert updated["rounds"]["2"]["eligible_for_next_round_prompt_student"] is False
    assert [row["status"] for row in updated["history"]] == [
        "promoted",
        "competition_blocked",
    ]

    second, changed_again = retroactively_block_rounds(updated, [2])
    assert changed_again == []
    assert second["history"] == updated["history"]


def test_same_round_state_machine_accepts_variable_cohorts_and_target_spaces(
    tmp_path: Path,
) -> None:
    stages = (
        "preflight",
        "estep",
        "manifest",
        "novelty_decision",
        "mstep",
        "inference",
        "evaluation",
        "promotion",
    )
    for case_count, target_count in ((3, 7), (10, 373), (50, 19)):
        spec = RunSpec(
            run_id=f"synthetic-{case_count}-{target_count}",
            round_index=2,
            backend="synthetic",
            case_list_path="/cases.csv",
            case_list_sha256=f"cases-{case_count}",
            case_count=case_count,
            target_space_path="/targets.json",
            target_space_sha256=f"targets-{target_count}",
            baseline_checkpoint_sha256="baseline",
            previous_checkpoint_sha256="previous",
            evaluation_protocol_sha256="evaluation",
        ).to_dict()
        machine = RoundStateMachine(
            tmp_path / f"state-{case_count}-{target_count}.json",
            spec,
        )
        for stage in stages:
            machine.advance(
                stage,
                "promoted" if stage == "promotion" else "passed",
            )
        assert machine.document["status"] == "completed"
        assert machine.document["run_spec"]["case_count"] == case_count
        assert len(machine.document["transitions"]) == len(stages)
