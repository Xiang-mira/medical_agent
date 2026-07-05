import importlib

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
