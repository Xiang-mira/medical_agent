import importlib
from pathlib import Path

import pytest


def test_run_em_training_accepts_safe_start_round_args():
    import scripts.run_em_training as em

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
