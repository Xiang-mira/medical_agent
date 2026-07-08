import importlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_run_em_training_accepts_safe_start_round_args():
    import scripts.run_em_training as em
    from cli_anything.medai.core.backend_capabilities import (
        PROJECT_PROMPT_STUDENT,
        canonical_backend_name,
    )

    args = em.parse_args(
        [
            "--rounds",
            "2",
            "--start-round",
            "2",
            "--baseline-run-root",
            "outputs/em_round_pure_cached_10case_formal_lite_20260703",
        ]
    )

    assert args.rounds == 2
    assert args.start_round == 2
    assert args.baseline_run_root.endswith("em_round_pure_cached_10case_formal_lite_20260703")
    assert canonical_backend_name("project_prompt_student") == PROJECT_PROMPT_STUDENT


def test_run_em_training_rejects_unknown_args():
    import scripts.run_em_training as em

    with pytest.raises(SystemExit):
        em.parse_args(["--start-round", "2", "--baseline-run-root", "x", "--typo-unsafe"])


def test_rounds_two_from_round_one_is_guarded(monkeypatch):
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "START_ROUND", 1)
    monkeypatch.setattr(em, "NUM_ROUNDS", 2)
    monkeypatch.delenv("MEDAI_ALLOW_ROUND1_TO_ROUND2_RESTART", raising=False)

    with pytest.raises(RuntimeError, match="Refusing to run NUM_ROUNDS=2 from Round1"):
        em.enforce_round2_restart_guard()


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.run_repaired_round2_continuation",
        "scripts.run_round3_after_promoted_round2",
        "scripts.run_pure_cached_10case_formal_lite_em",
    ],
)
def test_deprecated_round2_launchers_fail_closed(module_name):
    module = importlib.import_module(module_name)

    with pytest.raises(SystemExit, match="disabled"):
        module.fail_disabled_launcher()


def test_pure_cached_round2_estep_launcher_is_audit_only():
    text = Path("scripts/run_pure_cached_10case_round2_estep.py").read_text(encoding="utf-8")

    assert "audit_only_launcher_disables_round2_mstep" in text
    assert "audit_only_launcher_disables_student_inference" in text
    assert "run_student_mstep(2" not in text
    assert "save_student_predictions(2" not in text


def test_round2_estep_readiness_audit_does_not_train_or_infer():
    text = Path("scripts/audit_round2_estep_mstep_readiness.py").read_text(encoding="utf-8")

    assert "run_student_mstep" not in text
    assert "run_prompt_student_mstep" not in text
    assert "save_student_predictions" not in text
    assert '"mstep_allowed": False' in text


def test_round2_estep_preseeds_cleaned_student_prev_for_em_update():
    text = Path("scripts/run_em_training.py").read_text(encoding="utf-8")

    assert 'preseeded["student_prev"]' in text
    assert "EM_STUDENT_VS_PREVIOUS_MODE" in text
    assert "student 预测不注入 E-step 候选池" not in text
    assert "Round 2+ requires Round 1 teacher cache" not in text
    assert "material_update_audit.json" in text
    assert "no_material_update_converged_no_new_student_candidate" in text


def test_multimodel_loop_has_formal_student_vs_previous_mode():
    text = Path("agent-harness/cli_anything/medai/core/multimodel_loop.py").read_text(encoding="utf-8")

    assert 'EM_STUDENT_VS_PREVIOUS_MODE = "em_student_vs_previous"' in text
    assert '{"round_prev_selected", "student_prev"}' in text
    assert "student_prev cannot auto-overwrite" in text
    assert 'key != "student_prev"' in text  # legacy modes still filter diagnostic student masks.


def test_em_student_gate_requires_decisive_labelcritic() -> None:
    from cli_anything.medai.core.multimodel_loop import _apply_em_student_vs_previous_gate

    previous = {
        "model": "round_prev_selected",
        "candidate_exists": True,
        "prediction": "/tmp/previous.nii.gz",
        "candidate_qc_status": "pass",
    }
    student = {
        "model": "student_prev",
        "candidate_exists": True,
        "prediction": "/tmp/student.nii.gz",
        "candidate_qc_status": "pass",
        "eligible_for_labelcritic": True,
    }
    selection = {
        "selection_method": "label_critic",
        "selection_status": "selected",
        "selected_model": "student_prev",
        "selected_prediction": "/tmp/student.nii.gz",
    }

    selected, gated = _apply_em_student_vs_previous_gate(
        selected=student,
        selection=selection,
        candidates=[previous, student],
    )
    assert selected == previous
    assert gated["selection_method"] == "em_student_vs_previous_carry_forward"

    selected, gated = _apply_em_student_vs_previous_gate(
        selected=student,
        selection={**selection, "labelcritic_decisive": True},
        candidates=[previous, student],
    )
    assert selected == student
    assert gated["verified_student_replacement"] is True


def _save_mask(path: Path, *, empty: bool = False, shift: int = 0) -> Path:
    array = np.zeros((4, 4, 4), dtype=np.float32)
    if not empty:
        start = 1 if shift == 0 else 0
        array[start:start + 2, 1:3, 1:3] = 1.0
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    return path


def test_material_update_audit_carry_forward_previous_is_not_material(tmp_path: Path, monkeypatch):
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    image = _save_mask(tmp_path / "ct.nii.gz")
    old_mask = _save_mask(tmp_path / "old_liver.nii.gz")
    replay_mask = _save_mask(tmp_path / "replay_liver.nii.gz")
    zero_mask = _save_mask(tmp_path / "zero.nii.gz", empty=True)
    previous = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(old_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
            }
        ]
    }
    current = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(replay_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "round_prev_selected",
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
                "historical_replay": True,
            },
            {
                "case_id": "case001",
                "organ": "brain",
                "image": str(image),
                "mask": str(zero_mask),
                "target_type": "negative_absent",
                "grade": "A",
                "training_weight": 0.1,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "fov_status": "full",
                "negative_evidence": {"scan_coverage": "absent"},
            },
        ]
    }
    audit = em.build_material_update_audit(
        2,
        current,
        previous_manifest=previous,
        novelty={"decision": "limited_update", "max_steps": 500},
    )

    assert audit["new_reliable_positive_count"] == 0
    assert audit["verified_student_replacement_count"] == 0
    assert audit["old_replay_positive_count"] == 1
    assert audit["legal_negative_count"] == 1
    assert audit["material_update_decision"] == "no_material_update"
    assert audit["max_steps"] == 0
    assert (tmp_path / "round2" / "estep" / "material_update_audit.json").is_file()


def test_material_update_audit_counts_verified_student_replacement(tmp_path: Path, monkeypatch):
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    image = _save_mask(tmp_path / "ct.nii.gz")
    old_mask = _save_mask(tmp_path / "old_liver.nii.gz")
    student_mask = _save_mask(tmp_path / "student_liver.nii.gz", shift=1)
    previous = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(old_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
            }
        ]
    }
    current = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(student_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "student_prev",
                "source_model": "student_prev",
                "selection_method": "label_critic",
                "selection_status": "selected",
                "labelcritic_decisive": True,
                "labelcritic_records": [{"status": "success", "decision": {"winner": "b"}}],
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
            }
        ]
    }
    audit = em.build_material_update_audit(
        2,
        current,
        previous_manifest=previous,
        novelty={"decision": "limited_update", "max_steps": 500},
    )

    assert audit["verified_student_replacement_count"] == 1
    assert audit["verified_student_replacement_keys"] == [["case001", "liver"]]
    assert audit["material_update_decision"] == "material_update"
    assert audit["max_steps"] == 500


def test_material_update_audit_rejects_nondecisive_student_replacement(tmp_path: Path, monkeypatch):
    import scripts.run_em_training as em

    monkeypatch.setattr(em, "OUTPUT_ROOT", tmp_path)
    image = _save_mask(tmp_path / "ct.nii.gz")
    old_mask = _save_mask(tmp_path / "old_liver.nii.gz")
    student_mask = _save_mask(tmp_path / "student_liver.nii.gz", shift=1)
    previous = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(old_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "teacher_a",
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
            }
        ]
    }
    current = {
        "items": [
            {
                "case_id": "case001",
                "organ": "liver",
                "image": str(image),
                "mask": str(student_mask),
                "target_type": "positive_hard",
                "grade": "A",
                "training_weight": 1.0,
                "distillation_eligible": True,
                "selected_model": "student_prev",
                "source_model": "student_prev",
                "selection_method": "label_critic",
                "selection_status": "selected",
                "labelcritic_records": [{"status": "success", "decision": {"winner": "b"}}],
                "selected_candidate_qc_status": "pass",
                "identity_status": "valid",
            }
        ]
    }

    audit = em.build_material_update_audit(
        2,
        current,
        previous_manifest=previous,
        novelty={"decision": "limited_update", "max_steps": 500},
    )

    assert audit["verified_student_replacement_count"] == 0
    assert audit["material_update_decision"] == "no_material_update"
    assert audit["max_steps"] == 0


def test_round2_plus10_case_plan_is_read_only_contract():
    text = Path("scripts/build_round2_plus10_case_plan.py").read_text(encoding="utf-8")

    assert "annotation_folder_case_selection_only_not_training_label" in text
    assert "case_list_round2_plus10.csv" in text
    assert "cache_manifest.json" in text
    assert "teacher_inference_rerun" in text
