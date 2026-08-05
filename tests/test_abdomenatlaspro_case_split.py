from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scheduler.case_selection import CaseSelectionConfig, select_and_split, validate_selection, write_selection_outputs
from scheduler.manifest import assert_strict_no_gt_manifest
from scheduler.utils import SchedulerError


def _cfg(tmp_path: Path) -> CaseSelectionConfig:
    return CaseSelectionConfig(
        path=tmp_path / "cfg.yaml",
        data={
            "paths": {"image_root": str(tmp_path / "images"), "mask_root": str(tmp_path / "masks")},
            "candidate_pool": {"backup_ratio": 0.25},
            "scoring": {"mapped_positive_target_weight": 0.55, "physical_si_extent_weight": 0.35, "quality_weight": 0.10},
        },
    )


def _rows(n: int) -> list[dict[str, object]]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "case_id": f"BDMAP_{i:04d}",
                "ct_path": f"/tmp/BDMAP_{i:04d}/ct.nii.gz",
                "mask_dir": f"/tmp/BDMAP_{i:04d}/segmentations",
                "quality_pass": True,
                "mapped_373_positive_count": i % 9 + 1,
                "physical_si_extent_mm": 100 + i * 3,
                "si_spacing_mm": 1.0,
                "mask_file_count": 20 + i % 4,
                "quality_score": 1.0,
                "ct_file_size": 1000 + i,
                "array_shape_x": 4,
                "array_shape_y": 5,
                "array_shape_z": 6 + i,
                "spacing_x": 1,
                "spacing_y": 1,
                "spacing_z": 1,
                "affine_summary": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        )
    return rows


def test_split_counts_overlap_reproducible_and_unequal_counts(tmp_path):
    cfg = _cfg(tmp_path)
    train, test, _ = select_and_split(_rows(30), 7, 5, 42, cfg)
    train2, test2, _ = select_and_split(_rows(30), 7, 5, 42, cfg)
    train3, test3, _ = select_and_split(_rows(30), 7, 5, 43, cfg)
    assert len(train) == 7
    assert len(test) == 5
    assert {r["case_id"] for r in train}.isdisjoint({r["case_id"] for r in test})
    assert [r["case_id"] for r in train] == [r["case_id"] for r in train2]
    assert [r["case_id"] for r in test] == [r["case_id"] for r in test2]
    assert ({r["case_id"] for r in train}, {r["case_id"] for r in test}) != ({r["case_id"] for r in train3}, {r["case_id"] for r in test3})
    assert validate_selection(train, test, train_cases=7, test_cases=5)["status"] == "success"


def test_duplicate_group_not_cross_split_and_backfill(tmp_path):
    cfg = _cfg(tmp_path)
    rows = _rows(14)
    rows[1]["ct_file_size"] = rows[0]["ct_file_size"]
    rows[1]["array_shape_z"] = rows[0]["array_shape_z"]
    train, test, backups = select_and_split(rows, 5, 5, 1, cfg)
    validation = validate_selection(train, test, train_cases=5, test_cases=5)
    assert validation["DUPLICATE_GROUP_CROSS_SPLIT"] == 0
    assert len(backups) >= 1


def test_feature_distribution_is_reasonably_matched(tmp_path):
    cfg = _cfg(tmp_path)
    train, test, _ = select_and_split(_rows(40), 10, 10, 7, cfg)
    train_mean = sum(float(r["physical_si_extent_mm"]) for r in train) / len(train)
    test_mean = sum(float(r["physical_si_extent_mm"]) for r in test) / len(test)
    assert abs(train_mean - test_mean) < 50


def test_candidate_shortage_fails(tmp_path):
    with pytest.raises(SchedulerError):
        select_and_split(_rows(3), 2, 2, 1, _cfg(tmp_path))


def test_no_gt_manifest_and_stable_outputs_no_success_on_failed_validation(tmp_path):
    cfg = _cfg(tmp_path)
    train, test, backups = select_and_split(_rows(12), 3, 3, 9, cfg)
    out = tmp_path / "selection"
    write_selection_outputs(out, cfg, train, test, backups, [], seed=9)
    manifest = out / "train3_input_strict_no_gt.csv"
    assert assert_strict_no_gt_manifest(manifest)["rows"] == 3
    with manifest.open(newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    assert not (columns & {"annotation_folder", "mask_dir", "gt_path", "reference_path", "mapped_373_positive_count", "selection_score"})
    assert (out / "SUCCESS").exists()

    bad = validate_selection(train[:2], test, train_cases=3, test_cases=3)
    assert bad["status"] == "failed"
