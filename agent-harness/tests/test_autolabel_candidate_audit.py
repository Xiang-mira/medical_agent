from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_autolabel_candidate_selection import (
    best_score,
    build_confidence_summary,
    build_grade_calibration,
    discover_teacher_models,
    pearson,
    rank_of_score_against_pool,
    rank_scores,
    rankability_reason,
    reference_kind,
    review_reasons,
    spearman,
)


def test_rank_scores_ties_and_missing_values() -> None:
    ranks = rank_scores({"a": 0.9, "b": 0.5, "c": 0.9, "missing": None})
    assert ranks["a"] == 1
    assert ranks["c"] == 1
    assert ranks["b"] == 3
    assert ranks["missing"] is None


def test_best_score_ignores_missing_values() -> None:
    assert best_score({"a": None, "b": 0.2, "c": 0.7}) == ("c", 0.7)
    assert best_score({"a": None}) == (None, None)


def test_review_reasons_flag_rank_and_low_dice() -> None:
    row = {
        "selected_gt_dice": 0.1,
        "selected_rank_by_gt": 2,
        "selected_reference_quality_bucket": "critical_low_dice",
        "confidence_flag": "high_confidence_low_dice",
        "selected_model": "model_a",
        "full_raw_candidate_models": ["model_b", "model_c"],
    }
    reasons = set(review_reasons(row))
    assert "selected_gt_dice_below_0_5" in reasons
    assert "selected_not_gt_rank1" in reasons
    assert "critical_low_dice" in reasons
    assert "high_confidence_low_dice" in reasons
    assert "selected_model_missing_from_full_raw_pool" in reasons


def test_correlation_helpers_are_cpu_only() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]
    assert pearson(xs, ys) is not None
    assert spearman(xs, ys) == 1.0


def test_selected_artifact_rank_is_independent_of_selected_model() -> None:
    pool = {"teacher_a": 0.9, "teacher_b": 0.7}
    assert rank_of_score_against_pool(pool, 0.95) == 1
    assert rank_of_score_against_pool(pool, 0.8) == 2
    assert rank_of_score_against_pool(pool, None) is None


def test_grade_calibration_marks_missing_gt_as_insufficient() -> None:
    report = build_grade_calibration([
        {"grade": "A", "dice_selected_vs_gt": 0.9},
        {"grade": "B", "dice_selected_vs_gt": None},
        {"grade": "C"},
    ])
    rows = {row["grade"]: row for row in report["rows"]}
    assert rows["A"]["status"] == "success"
    assert rows["B"]["status"] == "insufficient_gt_for_calibration"
    assert rows["D"]["status"] == "insufficient_gt_for_calibration"

def test_reference_kind_treats_pants_as_non_mainline_pseudo_reference() -> None:
    assert reference_kind("/workspace/data/PanTS/LabelTr/case.nii.gz") == "historical_pseudo_reference"
    assert reference_kind("outputs/run/annotation_versions/case/updated/liver.nii.gz") == "historical_pseudo_reference"
    assert reference_kind("/external/manual_ref/liver.nii.gz") == "unknown_reference"


def test_discover_teacher_models_unions_config_and_disk(tmp_path) -> None:
    estep = tmp_path / "estep"
    (estep / "cases" / "case001" / "hierarchical_predictions" / "disk_teacher").mkdir(parents=True)
    (estep / "cases" / "case001" / "raw_predictions" / "raw_teacher").mkdir(parents=True)
    models = discover_teacher_models(estep, ["case001"], {"models_requested": ["configured_teacher"]})
    assert models == ["configured_teacher", "disk_teacher", "raw_teacher"]


def test_confidence_summary_excludes_pseudo_reference_from_gt_correlation() -> None:
    summary = build_confidence_summary(
        selection_rows=[
            {"gt_available": False, "confidence": 0.9, "selected_gt_dice": None, "selected_pseudo_gt_dice": 0.9},
            {"gt_available": True, "confidence": 0.8, "selected_gt_dice": 0.7},
        ],
        calibration_rows=[
            {"gt_available": False, "autolabel_candidate_relative_score": 0.9, "dice_candidate_vs_pseudo_gt": 0.8, "dice_candidate_vs_gt": None},
            {"gt_available": True, "autolabel_candidate_relative_score": 0.6, "dice_candidate_vs_gt": 0.5},
        ],
    )
    assert summary["selected_confidence_gt_dice_n"] == 1
    assert summary["candidate_relative_score_gt_dice_n"] == 1


def test_rankability_reason_is_explicit_for_missing_reference() -> None:
    assert rankability_reason(False, None, False, "mask.nii.gz", None) == "reference_missing"
    assert rankability_reason(True, None, False, None, None) == "dice_skipped"
